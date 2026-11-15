import copy

import pytest

from data.geometry_codec import codec_for
from data.geometry_interleaved import (InterleavedCodec, RELATIVE_REFERENCE_SCHEMA, RelativeReferenceCodec,
                                       first_use_order, fit_interleaved_vocabulary,
                                       fit_relative_reference_vocabulary, relative_reference_fields,
                                       restore_authoring)
from data.geometry_tokens import BOS, BYTE_OFFSET, EOS, encode_prediction
from data.learned_geometry import AUTHORING_FIELDS, build_target, core_hashes
from generation.learned_geometry import decode_prediction
from tests.geometry_fixtures import two_rooms


def test_dependency_order_preserves_all_records_controls_and_exact_source_byte_roundtrip():
    source = build_target(two_rooms())['supervision']
    # Force a non-identity vertex permutation and retain unused source records.
    source['vertices'].reverse()
    count = len(source['vertices'])
    for line in source['linedefs']:
        line['v1'], line['v2'] = count-1-line['v1'], count-1-line['v2']
    source['vertices'].append({'x': -32768, 'y': 32767})
    source['sectors'][1].update(floor_height=-24, ceil_height=-24, tag=37)
    source['sectors'].append({**source['sectors'][1], 'ceil_height': -32, 'tag': 41})
    source['sidedefs'].append({**source['sidedefs'][0], 'sector': 2})
    source['linedefs'][0].update(special=1, tag=37, flags=source['linedefs'][0]['flags'] | 0x8000)
    before = copy.deepcopy(source)
    world, events, source_ids = first_use_order(source)
    assert source == before
    assert source_ids['vertices'] != list(range(len(source['vertices'])))
    assert len(events) == sum(len(source[name]) for name in AUTHORING_FIELDS)
    restored = restore_authoring(world, source_ids)
    assert restored == source
    assert encode_prediction(restored) == encode_prediction(source)
    assert core_hashes(decode_prediction(restored)) == core_hashes(decode_prediction(source))
    codec = InterleavedCodec(fit_interleaved_vocabulary([source]))
    tokens = codec.encode(source)
    assert codec_for(codec.definition).decode(tokens) == world
    assert world['sectors'][1]['ceil_height'] == -24
    assert world['sectors'][2]['ceil_height'] == -32
    assert world['linedefs'][0]['flags'] & 0x8000
    invalid = copy.deepcopy(source_ids); invalid['vertices'][0] = invalid['vertices'][1]
    with pytest.raises(ValueError, match='permutation'):
        restore_authoring(world, invalid)


def test_headers_are_model_choices_with_only_file_dependency_masks_and_no_fabricated_ending():
    source = build_target(two_rooms())['supervision']
    codec = InterleavedCodec(fit_interleaved_vocabulary([source]))
    state = codec.state(); state.push(BOS)
    vertex, sector, side, line, thing = codec.continuation_tokens
    assert set(state.allowed_tokens()) == {vertex, sector, thing}
    assert EOS not in state.allowed_tokens()
    with pytest.raises(ValueError, match='header'):
        state.push(line)
    # A deliberately degenerate single-vertex/single-line document is legal
    # syntax. This codec neither creates a polygon nor fixes self references.
    tiny = copy.deepcopy(source)
    tiny['vertices'] = tiny['vertices'][:1]; tiny['sectors'] = tiny['sectors'][:1]
    tiny['sidedefs'] = [{**tiny['sidedefs'][0], 'sector': 0}]
    tiny['linedefs'] = [{**tiny['linedefs'][0], 'v1': 0, 'v2': 0, 'right_sidedef': 0, 'left_sidedef': -1}]
    tiny['things'] = []
    tokens = codec.encode(tiny)
    assert codec.decode(tokens) == tiny
    with pytest.raises(ValueError, match='incomplete'):
        codec.decode(tokens[:-1])
    assert tokens.count(EOS) == 1


def test_interleaved_escapes_extremes_and_prefixes_are_lossless_and_causal():
    source = build_target(two_rooms())['supervision']
    codec = InterleavedCodec(fit_interleaved_vocabulary([source], max_integers=0))
    changed = copy.deepcopy(source)
    changed['vertices'][0].update(x=-32768, y=32767)
    changed['vertices'][1].update(x=32767, y=-32768)
    changed['sectors'][0].update(floor_height=-32768, ceil_height=32767, floor_flat='newflat')
    tokens = codec.encode(changed)
    assert codec.decode(tokens) == first_use_order(changed)[0]
    state = codec.state()
    for i, token in enumerate(tokens):
        state.push(token)
        assert state.features() == codec.features(tokens[:i+1])[-1]
    changed['vertices'][0]['x'] = 32768
    with pytest.raises(ValueError):
        codec.encode(changed)


def test_record_type_accuracy_is_separate_from_binary_continuation_recall(monkeypatch):
    torch = pytest.importorskip('torch')
    from models.geometry import GeometryRNN
    source = build_target(two_rooms())['supervision']
    codec = InterleavedCodec(fit_interleaved_vocabulary([source])); tokens = codec.encode(source)
    model = GeometryRNN(hidden=8, embedding=8, layers=1, context_dim=4, codec_config=codec.definition)
    masks = torch.zeros(1, len(tokens)-1, codec.vocab_size, dtype=torch.bool)
    for i, choices in enumerate(codec.masks(tokens)):
        masks[0, i, list(choices)] = True
    logits = torch.zeros(*masks.shape)
    logits[:, :, codec.continuation_tokens[0]] = 10
    monkeypatch.setattr(model, 'forward', lambda *args, **kwargs: (logits, None))
    _, metrics = model.loss(torch.tensor([tokens]), torch.zeros(1, 4), masks)
    assert metrics['list_stop_count'] == 1
    assert metrics['list_stop_recall'] == 0
    assert metrics['list_continue_recall'] == 1
    expected = len(source['vertices']) / sum(len(source[name]) for name in AUTHORING_FIELDS)
    torch.testing.assert_close(metrics['record_type_accuracy'], torch.tensor(expected))


def test_relative_reference_codec_encodes_causal_event_ages_and_preserves_actual_ids_and_bytes():
    from data.geometry_memory import RELATIVE_REFERENCE_MEMORY_SCHEMA, RecordMemory, current_record_anchors, memory_trace, reference_query
    source = build_target(two_rooms())['supervision']
    source['vertices'].reverse()
    count = len(source['vertices'])
    for line in source['linedefs']:
        line['v1'], line['v2'] = count - 1 - line['v1'], count - 1 - line['v2']
    world, events, values = relative_reference_fields(source)
    _, _, source_ids = first_use_order(source)
    # At the first linedef event only its two vertices exist: v1=0 has age 1,
    # v2=1 has age 0, never len(world['vertices']) - 1 - id.
    assert world['linedefs'][0]['v1'] == 0 and world['linedefs'][0]['v2'] == 1
    assert values['linedefs'][0]['v1'] == 1 and values['linedefs'][0]['v2'] == 0
    assert values['linedefs'][0]['left_sidedef'] == -1
    definition = fit_relative_reference_vocabulary([source])
    assert definition['schema'] == RELATIVE_REFERENCE_SCHEMA
    codec = RelativeReferenceCodec(definition)
    tokens = codec.encode(source)
    decoded = codec_for(codec.definition).decode(tokens)
    assert decoded == world
    restored = restore_authoring(decoded, source_ids)
    assert restored == source
    assert encode_prediction(restored) == encode_prediction(source)
    assert core_hashes(decode_prediction(restored)) == core_hashes(decode_prediction(source))
    trace = memory_trace(codec, tokens, anchors=True)
    assert trace['schema'] == RELATIVE_REFERENCE_MEMORY_SCHEMA
    state, bank = codec.state(), RecordMemory()
    for token, query, anchor in zip(tokens, trace['references'], trace['anchors']):
        state.push(token)
        bank.observe(state)
        assert reference_query(state) == query
        assert current_record_anchors(state) == anchor
    assert bank.rows == memory_trace(InterleavedCodec(fit_interleaved_vocabulary([source])),
                                     InterleavedCodec(fit_interleaved_vocabulary([source])).encode(source))['records']


def test_relative_reference_escapes_counts_crossing_255_null_and_65536_nonnull_ids():
    from data.geometry_memory import reference_query
    source = build_target(two_rooms())['supervision']
    many = copy.deepcopy(source)
    for index in range(300):
        vid = len(many['vertices'])
        sid = len(many['sidedefs'])
        many['vertices'].append({'x': index - 150, 'y': index})
        many['sidedefs'].append({**source['sidedefs'][0], 'x_offset': index})
        if index < 260:
            many['linedefs'].append({**source['linedefs'][0], 'v1': vid - 1, 'v2': vid,
                                     'right_sidedef': sid, 'left_sidedef': -1})
    many['linedefs'].append({**source['linedefs'][0], 'v1': 0, 'v2': 260,
                             'right_sidedef': 0, 'left_sidedef': 260})
    many['linedefs'].append({**source['linedefs'][0], 'v1': 260, 'v2': 260,
                             'right_sidedef': 260, 'left_sidedef': -1})
    world, _, source_ids = first_use_order(many)
    for max_integers in (0, 8):
        codec = RelativeReferenceCodec(fit_relative_reference_vocabulary([many], max_integers=max_integers))
        tokens = codec.encode(many)
        assert codec.decode(tokens) == world
        assert restore_authoring(codec.decode(tokens), source_ids) == many
    escaped = RelativeReferenceCodec(fit_relative_reference_vocabulary([many], max_integers=0))
    tokens = escaped.encode(many)
    state = escaped.state()
    low_ages = []
    for token in tokens:
        state.push(token)
        query = reference_query(state)
        if query[0] == 0 and query[1] == 1:
            low_ages.append((len(state.prediction['vertices']), query[2]))
    # With >256 emitted vertices, referencing vertex 0 emits age = count - 1 > 255
    # whose low byte is (count - 1) & 255, not 0.
    assert any(count > 256 and low == ((count - 1) & 255) and low != 0 for count, low in low_ages)
    # Verify non-null vertex/sector ID 65535 vs age 65535 when count is 65536,
    # while optional_side 65535 remains NULL (-1).
    state = escaped.state()
    state.started = True
    state.prediction['vertices'] = [{'x': 0, 'y': 0}] * 65536
    state.prediction['sectors'] = [source['sectors'][0]] * 65536
    state.prediction['sidedefs'] = [source['sidedefs'][0]]
    state.section = 3
    state.record = {}
    assert (BYTE_OFFSET + 255) in state.allowed_tokens()
    state.push(BYTE_OFFSET + 0); state.push(BYTE_OFFSET + 0)    # v1 age 0 -> actual ID 65535
    state.push(BYTE_OFFSET + 255); state.push(BYTE_OFFSET + 255)  # v2 age 65535 -> actual ID 0
    for _ in range(3):
        state.push(BYTE_OFFSET + 0); state.push(BYTE_OFFSET + 0)
    state.push(BYTE_OFFSET + 0); state.push(BYTE_OFFSET + 0)    # right_sidedef age 0 -> ID 0
    state.push(BYTE_OFFSET + 255); state.push(BYTE_OFFSET + 255)  # left_sidedef 65535 -> NULL (-1)
    emitted = state.prediction['linedefs'][-1]
    assert emitted['v1'] == 65535 and emitted['v2'] == 0
    assert emitted['right_sidedef'] == 0 and emitted['left_sidedef'] == -1
    state.section = 2
    state.record = {'x_offset': 0, 'y_offset': 0, 'upper_tex': '-', 'lower_tex': '-', 'mid_tex': '-'}
    state.field_index = 5
    state.push(BYTE_OFFSET + 0); state.push(BYTE_OFFSET + 0)    # sector age 0 -> actual ID 65535
    assert state.prediction['sidedefs'][-1]['sector'] == 65535
    with pytest.raises(ValueError, match='out of causal range'):
        state.section = 3; state.record = {}; state.field_index = 5
        state._finish(2)

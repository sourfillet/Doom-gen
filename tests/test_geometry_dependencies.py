"""Learned messages follow only already emitted authoring dependencies."""
import pytest

torch = pytest.importorskip('torch')

from data.geometry_memory import RecordMemory
from models.geometry import GeometryRNN
from tests.geometry_model_fixtures import anchored_example, one_thread  # noqa: F401 - autouse fixture


def model_for(codec, dependencies=True, record_values=False):
    return GeometryRNN(hidden=12, embedding=8, layers=1, context_dim=4, prefix_features=True,
                       codec_config=codec.definition, record_memory=True, memory_read=True,
                       record_anchors=True, relative_pointers=True, spatial_pointers=True,
                       dependency_memory=dependencies, record_values=record_values)


def test_dependency_messages_start_at_v6_function_preserve_rng_and_receive_gradients():
    codec, tokens, features, bank, queries, anchors = anchored_example('interleaved')
    torch.manual_seed(7); baseline = model_for(codec, False)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(7); candidate = model_for(codec)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, candidate.state_dict()[k]) for k, v in baseline.state_dict().items())
    kwargs = dict(features=features, memory=bank, references=queries, anchors=anchors)
    context = torch.zeros(1, 4)
    a = baseline(torch.tensor([tokens[:-1]]), context, **kwargs)[0]
    b = candidate(torch.tensor([tokens[:-1]]), context, **kwargs)[0]
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    masks = torch.zeros(1, len(tokens)-1, codec.vocab_size, dtype=torch.bool)
    for i, choices in enumerate(codec.masks(tokens)):
        masks[0, i, list(choices)] = True
    loss, _ = candidate.loss(torch.tensor([tokens]), context, masks, **kwargs)
    loss.backward()
    gradients = candidate.memory_readout.dependency_messages.grad
    assert all(value.abs().sum() > 0 for value in gradients)
    assert all(torch.isfinite(p.grad).all() for p in candidate.parameters() if p.grad is not None)


@pytest.mark.parametrize('mode', ['interleaved', 'relative-reference'])
def test_active_dependency_full_bank_equals_incremental_cached_logits(mode):
    codec, tokens, features, bank, queries, anchors = anchored_example(mode)
    model = model_for(codec).eval()
    with torch.no_grad():
        head = model.memory_readout
        head.dependency_messages.normal_(0, .2)
        head.anchor_read.weight.normal_(0, .1)
        head.relative_pointer.normal_(0, .1)
        head.spatial_pointer.normal_(0, .1)
        teacher, _ = model(torch.tensor([tokens[:-1]]), torch.zeros(1, 4), features=features,
                            memory=bank, references=queries, anchors=anchors)
        state, builder, hidden = codec.state(), RecordMemory(), None
        live = bank[:, :0]
        cache = head.prepare_records(live)
        for i, token in enumerate(tokens[:-1]):
            state.push(token); old = len(builder.rows); builder.observe(state)
            if len(builder.rows) > old:
                new = torch.tensor([builder.rows[old:]])
                addition = head.prepare_records(new, history=live, encoded_history=cache)
                live = torch.cat([live, new], 1)
                cache = tuple(torch.cat([a, b], 1) for a, b in zip(cache, addition))
                for a, b in zip(cache, head.prepare_records(live)):
                    torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-6)
            actual, hidden = model(torch.tensor([[token]]), torch.zeros(1, 4), hidden,
                                   features=features[:, i:i+1], memory=live, encoded_memory=cache,
                                   references=queries[:, i:i+1], anchors=anchors[:, i:i+1])
            torch.testing.assert_close(actual[0, 0], teacher[0, i], atol=6e-6, rtol=6e-6)
    with pytest.raises(ValueError, match='cache'):
        head.prepare_records(bank[:, :1], history=bank)


def test_dependency_messages_include_two_hop_sector_to_side_to_wall_without_future_reads():
    codec, *_ = anchored_example('interleaved')
    head = model_for(codec).memory_readout
    # Vertex0, sector0, side0 -> sector0, wall0 -> side0, future sector1.
    bank = torch.tensor([[[0, 0, 24, 32, 0, 0, 0, 0, 0],
                          [1, 0, -16, 128, 0, 0, 160, 0, 0],
                          [2, 0, 0, 0, 0, 0, 0, 0, 0],
                          [3, 0, 0, 0, 0, 0, 0, 0, -1],
                          [1, 1, 100, 200, 0, 0, 128, 0, 0]]], dtype=torch.float)
    with torch.no_grad():
        head.dependency_messages.zero_()
        head.dependency_messages[0, :12] = torch.eye(12)
        head.dependency_messages[3, :12] = torch.eye(12)
    raw = head.encode_records(bank)
    enriched = head.prepare_records(bank)[0]
    torch.testing.assert_close(enriched[:, 2], raw[:, 2] + raw[:, 1])
    torch.testing.assert_close(enriched[:, 3], raw[:, 3] + raw[:, 2] + raw[:, 1])
    differentiable = bank.clone().requires_grad_()
    head.prepare_records(differentiable)[0][:, 3].square().sum().backward()
    assert differentiable.grad[0, 1, 2:4].abs().sum() > 0
    assert differentiable.grad[0, 4].count_nonzero() == 0
    # Even an explicitly forward reference must not read the future sector.
    changed = bank.clone(); changed[0, 2, 7] = 1
    a = head.prepare_records(changed)[0]
    changed[0, 4, 2:] += 1234
    b = head.prepare_records(changed)[0]
    torch.testing.assert_close(a[:, :4], b[:, :4], atol=0, rtol=0)
    # Padding cannot become a parent of any real row.
    padded = torch.cat([bank, bank.new_full((1, 2, 9), -1)], 1)
    torch.testing.assert_close(enriched, head.prepare_records(padded)[0][:, :5], atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize('mode', ['interleaved', 'relative-reference'])
def test_future_record_values_and_references_have_no_effect_or_gradient_on_prefix(mode):
    codec, tokens, features, bank, queries, anchors = anchored_example(mode)
    model = model_for(codec).eval()
    with torch.no_grad():
        model.memory_readout.dependency_messages.normal_(0, .2)
        model.memory_readout.anchor_read.weight.normal_(0, .1)
    # This prefix has a visible side and vertex anchor, plus future rows of
    # several kinds including malformed parent references after perturbation.
    index = int(((queries[0, :, 0] == 2) & (queries[0, :, 3] == 1)).nonzero()[0])
    length = index + 1
    future = bank[0, :, 1] >= features[0, index, 1:][bank[0, :, 0]]
    def run(records):
        return model(torch.tensor([tokens[:length]]), torch.zeros(1, 4), features=features[:, :length],
                     memory=records, references=queries[:, :length], anchors=anchors[:, :length])[0]
    memory = bank.float().requires_grad_()
    a = run(memory)
    changed = bank.clone(); changed[0, future, 2:] += 1234
    torch.testing.assert_close(a, run(changed), atol=0, rtol=0)
    a[0, -1].masked_fill(~torch.isfinite(a[0, -1]), 0).sum().backward()
    assert memory.grad[0, future].count_nonzero() == 0


def test_relative_reference_pointer_head_projects_ages_crossing_255_null_and_65536_vertex_ids():
    from data.geometry_interleaved import RELATIVE_REFERENCE_SCHEMA, RelativeReferenceCodec
    from data.geometry_tokens import BYTE_OFFSET
    from models.geometry_memory import GeometryRecordReadout
    codec = RelativeReferenceCodec({'schema': RELATIVE_REFERENCE_SCHEMA, 'assets': [], 'integers': [0, 65535]})
    head = GeometryRecordReadout(8, codec)
    bank = torch.tensor([[[0, index, index, 0, 0, 0, 0, 0, 0] for index in range(300)]], dtype=torch.float, requires_grad=True)
    features = torch.tensor([[[0, 300, 0, 0, 0, 0]]])
    hidden, logits = torch.ones(1, 1, 8), torch.zeros(1, 1, codec.vocab_size)
    # Low byte 1 encodes age 1 (id 298, high=0) and age 257 (id 42, high=1),
    # never absolute id 1 or 257.
    values = head(hidden, logits, features, bank, torch.tensor([[[0, 1, 1, 0]]]))
    assert set(torch.isfinite(values[0, 0]).nonzero().flatten().tolist()) == {BYTE_OFFSET, BYTE_OFFSET + 1}
    values[0, 0, [BYTE_OFFSET, BYTE_OFFSET + 1]].sum().backward()
    nonzero = bank.grad[0, :, 2].nonzero().flatten().tolist()
    assert nonzero == [42, 298]
    # Optional side with count=300 and low=255 allows age 255 (side 44, high=0) and NULL (65535, high=255).
    side_bank = torch.tensor([[[2, index, index, 0, 0, 0, 0, 0, 0] for index in range(300)]])
    side_features = torch.tensor([[[0, 0, 0, 300, 0, 0]]])
    values = head(hidden, logits, side_features, side_bank, torch.tensor([[[2, 1, 255, 1]]]))
    assert set(torch.isfinite(values[0, 0]).nonzero().flatten().tolist()) == {BYTE_OFFSET, BYTE_OFFSET + 255}
    # Optional side with count=0 allows only NULL (65535).
    values = head(hidden, logits, torch.tensor([[[0, 0, 0, 0, 0, 0]]]), side_bank, torch.tensor([[[2, 0, -1, 1]]]))
    assert torch.isfinite(values[0, 0]).nonzero().flatten().tolist() == [codec.integer_ids[65535]]
    # Non-null vertex query at count=65536 maps id 65535 -> age 0 and id 0 -> age 65535 (not NULL).
    full_bank = torch.tensor([[[0, 0, 10, 20, 0, 0, 0, 0, 0], [0, 65535, 30, 40, 0, 0, 0, 0, 0]]],
                             dtype=torch.float, requires_grad=True)
    head.null_reference.grad = None
    values = head(hidden, logits, torch.tensor([[[0, 65536, 0, 0, 0, 0]]]), full_bank, torch.tensor([[[0, 0, -1, 0]]]))
    assert set(torch.isfinite(values[0, 0]).nonzero().flatten().tolist()) == {codec.integer_ids[0], codec.integer_ids[65535]}
    values[0, 0, codec.integer_ids[65535]].backward()
    assert full_bank.grad[0, 0, 2].abs() > 0 and full_bank.grad[0, 1, 2] == 0
    assert head.null_reference.grad is None or head.null_reference.grad.count_nonzero() == 0


def test_dependency_contract_rejects_unordered_or_nonspatial_configuration():
    codec, *_ = anchored_example('interleaved')
    with pytest.raises(ValueError, match='dependency memory requires'):
        GeometryRNN(codec_config=codec.definition, dependency_memory=True)
    with pytest.raises(ValueError, match='record values require'):
        GeometryRNN(codec_config=codec.definition, record_memory=True, memory_read=True, record_values=True)


def test_partial_record_values_expose_only_completed_scalars_with_exact_world_vs_relative_wraparound():
    import copy
    from data.geometry_compact import CompactCodec
    from data.geometry_interleaved import InterleavedCodec, RelativeReferenceCodec, RELATIVE_REFERENCE_SCHEMA, SCHEMA as INTERLEAVED_SCHEMA
    from data.geometry_memory import (RECORD_VALUES_SCHEMA, RELATIVE_REFERENCE_RECORD_VALUES_SCHEMA,
                                      current_record_values, memory_trace)
    from data.geometry_relative import RelativeCodec, SCHEMA as RELATIVE_SCHEMA, signed16
    from data.geometry_tokens import FIELD_TYPES
    from data.learned_geometry import AUTHORING_FIELDS, RECORD_FIELDS, build_target
    from tests.geometry_fixtures import two_rooms

    prediction = build_target(two_rooms())['supervision']
    changed = copy.deepcopy(prediction)
    changed['vertices'][0].update(x=-32768, y=32767)
    changed['vertices'][1].update(x=32767, y=-32768)
    changed['sectors'][0].update(floor_height=-32768, ceil_height=32767)
    assets = sorted({record[field] for name in ('sectors', 'sidedefs')
                     for record in changed[name]
                     for field, kind in zip(RECORD_FIELDS[name], FIELD_TYPES[name]) if kind == 'name'})
    codecs = [
        CompactCodec({'schema': 'doom_geometry_field_tokens_v1', 'assets': assets, 'integers': []}),
        RelativeCodec({'schema': RELATIVE_SCHEMA, 'assets': assets, 'integers': []}),
        InterleavedCodec({'schema': INTERLEAVED_SCHEMA, 'assets': assets, 'integers': []}),
        RelativeReferenceCodec({'schema': RELATIVE_REFERENCE_SCHEMA, 'assets': assets, 'integers': []}),
    ]
    for codec in codecs:
        tokens = codec.encode(changed)
        trace = memory_trace(codec, tokens, anchors=True, record_values=True)
        assert trace['schema'] == (RELATIVE_REFERENCE_RECORD_VALUES_SCHEMA
                                   if getattr(codec, 'relative_references', False)
                                   else RECORD_VALUES_SCHEMA)
        state = codec.state()
        seen_half_byte = False
        seen_second_vertex_y = False
        seen_sector_after_ceil = False
        for index, token in enumerate(tokens[:-1]):
            state.push(token)
            actual = current_record_values(state)
            assert actual == trace['record_values'][index]
            if state.section >= 5 or state.record is None:
                assert actual == [[0, 0]] * 7
                continue
            name = AUTHORING_FIELDS[state.section]
            for slot, (field, kind) in enumerate(zip(RECORD_FIELDS[name], FIELD_TYPES[name])):
                if slot >= state.field_index or kind not in {'signed', 'unsigned'}:
                    assert actual[slot] == [0, 0]
                else:
                    assert actual[slot][1] == 1
            for slot in range(len(RECORD_FIELDS[name]), 7):
                assert actual[slot] == [0, 0]
            if len(state.body) == 1 and FIELD_TYPES[name][state.field_index] in {'signed', 'unsigned'}:
                assert actual[state.field_index] == [0, 0]
                seen_half_byte = True
            if name == 'vertices' and len(state.prediction['vertices']) == 1 and state.field_index == 1 and not state.body:
                expected_x = 32767 if type(codec) is CompactCodec else signed16(32767 - (-32768))
                assert expected_x == (32767 if type(codec) is CompactCodec else -1)
                assert actual[0] == [expected_x, 1]
                assert actual[1] == [0, 0]
                seen_second_vertex_y = True
            if name == 'sectors' and len(state.prediction['sectors']) == 0 and state.field_index == 2 and not state.body:
                expected_ceil = 32767 if type(codec) is CompactCodec else -1
                assert actual[0] == [-32768, 1]
                assert actual[1] == [expected_ceil, 1]
                seen_sector_after_ceil = True
        assert seen_half_byte and seen_second_vertex_y and seen_sector_after_ceil


@pytest.mark.parametrize('mode', ['interleaved', 'relative-reference'])
def test_record_values_start_at_previous_function_preserve_rng_and_receive_gradients(mode):
    from data.geometry_memory import memory_trace
    codec, tokens, features, bank, queries, anchors = anchored_example(mode)
    trace = memory_trace(codec, tokens, anchors=True, record_values=True)
    record_values = torch.tensor([trace['record_values'][:-1]])
    torch.manual_seed(7)
    baseline = model_for(codec, True, False)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(7)
    candidate = model_for(codec, True, True)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, candidate.state_dict()[k]) for k, v in baseline.state_dict().items())
    context = torch.zeros(1, 4)
    a = baseline(torch.tensor([tokens[:-1]]), context, features=features, memory=bank,
                 references=queries, anchors=anchors)[0]
    b = candidate(torch.tensor([tokens[:-1]]), context, features=features, memory=bank,
                  references=queries, anchors=anchors, record_values=record_values)[0]
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    masks = torch.zeros(1, len(tokens) - 1, codec.vocab_size, dtype=torch.bool)
    for i, choices in enumerate(codec.masks(tokens)):
        masks[0, i, list(choices)] = True
    loss, _ = candidate.loss(torch.tensor([tokens]), context, masks, features=features,
                             memory=bank, references=queries, anchors=anchors, record_values=record_values)
    loss.backward()
    assert candidate.memory_readout.record_value.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in candidate.parameters() if p.grad is not None)


@pytest.mark.parametrize('mode', ['interleaved', 'relative-reference'])
def test_record_values_known_x_affects_y_and_queries_while_excluding_missing_and_future_values(mode):
    from data.geometry_memory import RecordMemory, memory_trace
    codec, tokens, features, bank, queries, anchors = anchored_example(mode)
    trace = memory_trace(codec, tokens, anchors=True, record_values=True)
    record_values = torch.tensor([trace['record_values'][:-1]])
    model = model_for(codec, True, True).eval()
    with torch.no_grad():
        model.memory_readout.record_value.normal_(0, 0.25)
    # Vertex Y prediction step: section 0, field_index 1 (position == 10), slot 0 present, slot 1 absent.
    y_steps = ((features[0, :, 0] // 64 == 0) & (features[0, :, 0] % 64 == 10)).nonzero().flatten()
    y_index = int(y_steps[0])
    assert record_values[0, y_index, 0, 1] == 1 and record_values[0, y_index, 1, 1] == 0
    # Sidedef sector query step: section 2, reference kind 1 (sectors), slots 0 and 1 (offsets) present.
    side_steps = ((features[0, :, 0] // 64 == 2) & (queries[0, :, 0] == 1) & (queries[0, :, 1] == 0)).nonzero().flatten()
    side_index = int(side_steps[0])
    assert record_values[0, side_index, 0, 1] == 1 and record_values[0, side_index, 1, 1] == 1
    length = max(y_index, side_index) + 1

    def run(records, values):
        return model(torch.tensor([tokens[:length]]), torch.zeros(1, 4), features=features[:, :length],
                     memory=records, references=queries[:, :length], anchors=anchors[:, :length],
                     record_values=values[:, :length])[0]

    base_logits = run(bank, record_values)
    changed_x = record_values.clone()
    changed_x[0, y_index, 0, 0] = 64 if changed_x[0, y_index, 0, 0] == 0 else 0
    assert not torch.equal(base_logits[0, y_index], run(bank, changed_x)[0, y_index])

    changed_offset = record_values.clone()
    changed_offset[0, side_index, 0, 0] += 32
    assert not torch.equal(base_logits[0, side_index], run(bank, changed_offset)[0, side_index])

    # Perturbing absent slots (where presence == 0) or future steps (> length) has zero effect and zero gradient.
    diff_values = record_values.float().requires_grad_()
    diff_bank = bank.float().requires_grad_()
    out = run(diff_bank, diff_values)
    absent = record_values[:, :length, :, 1] == 0
    perturbed_absent = record_values.clone()
    perturbed_absent[:, :length, :, 0] = torch.where(absent, 12345, perturbed_absent[:, :length, :, 0])
    torch.testing.assert_close(out, run(bank, perturbed_absent), atol=0, rtol=0)

    future_bank = bank[0, :, 1] >= features[0, length - 1, 1:][bank[0, :, 0]]
    perturbed_bank = bank.clone()
    perturbed_bank[0, future_bank, 2:] += 999
    torch.testing.assert_close(out, run(perturbed_bank, record_values), atol=0, rtol=0)

    out[0, [y_index, side_index]].masked_fill(~torch.isfinite(out[0, [y_index, side_index]]), 0).sum().backward()
    assert diff_values.grad[:, :length, :, 0][absent].count_nonzero() == 0
    assert diff_values.grad[:, length:].count_nonzero() == 0
    assert diff_bank.grad[0, future_bank].count_nonzero() == 0
    assert diff_values.grad[0, y_index, 0, 0].abs() > 0

    # Exposing unknown Y, a reference slot, a header slot, invalid presence bit or out-of-range scalar raises ValueError.
    for mutator, match in [
        (lambda v: v[0, y_index, 1].copy_(torch.tensor([0, 1])), 'incomplete, future or non-scalar'),
        (lambda v: v[0, 0, 0].copy_(torch.tensor([0, 1])), 'incomplete, future or non-scalar'),
        (lambda v: v[0, side_index, 5].copy_(torch.tensor([0, 1])), 'incomplete, future or non-scalar'),
        (lambda v: v[0, y_index, 0].copy_(torch.tensor([0, 2])), 'presence bits'),
        (lambda v: v[0, y_index, 0].copy_(torch.tensor([32768, 1])), '16-bit authoring range'),
    ]:
        bad = record_values.clone()
        mutator(bad)
        with pytest.raises(ValueError, match=match):
            run(bank, bad)

    # Incremental cached decoding matches full teacher bank with active record_values.
    with torch.no_grad():
        head = model.memory_readout
        head.dependency_messages.normal_(0, 0.2)
        head.anchor_read.weight.normal_(0, 0.1)
        head.relative_pointer.normal_(0, 0.1)
        head.spatial_pointer.normal_(0, 0.1)
        teacher, _ = model(torch.tensor([tokens[:-1]]), torch.zeros(1, 4), features=features,
                           memory=bank, references=queries, anchors=anchors, record_values=record_values)
        state, builder, hidden = codec.state(), RecordMemory(), None
        live = bank[:, :0]
        cache = head.prepare_records(live)
        for i, token in enumerate(tokens[:-1]):
            state.push(token)
            old = len(builder.rows)
            builder.observe(state)
            if len(builder.rows) > old:
                new = torch.tensor([builder.rows[old:]])
                addition = head.prepare_records(new, history=live, encoded_history=cache)
                live = torch.cat([live, new], 1)
                cache = tuple(torch.cat([a, b], 1) for a, b in zip(cache, addition))
            actual, hidden = model(torch.tensor([[token]]), torch.zeros(1, 4), hidden,
                                   features=features[:, i:i + 1], memory=live, encoded_memory=cache,
                                   references=queries[:, i:i + 1], anchors=anchors[:, i:i + 1],
                                   record_values=record_values[:, i:i + 1])
            torch.testing.assert_close(actual[0, 0], teacher[0, i], atol=6e-6, rtol=6e-6)

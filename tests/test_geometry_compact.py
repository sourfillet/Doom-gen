import copy

import pytest

from data.geometry_compact import CompactCodec, fit_vocabulary
from data.geometry_tokens import END_LIST, encode_prediction
from data.learned_geometry import build_target, core_hashes
from generation.learned_geometry import decode_prediction
from tests.geometry_fixtures import polygon_document, square, two_rooms


def test_field_compression_is_lossless_and_keeps_record_counts_model_emitted():
    doc = two_rooms()
    doc.sectors[1].floor_height = 24
    prediction = build_target(doc)['supervision']
    codec = CompactCodec(fit_vocabulary([prediction]))
    tokens = codec.encode(prediction)
    assert len(tokens) < len(encode_prediction(prediction)) / 2
    assert tokens.count(END_LIST) == 5
    assert codec.decode(tokens) == prediction
    assert core_hashes(decode_prediction(codec.decode(tokens))) == core_hashes(doc)
    assert len(codec.masks(tokens)) == len(tokens) - 1
    assert codec.features(tokens)[0][1:] == [0] * 5
    assert codec.features(tokens)[-1][1:] == [6, 2, 8, 7, 1]


def test_heldout_names_and_numbers_use_escapes_without_refitting_vocabulary():
    train = build_target(two_rooms())['supervision']
    definition = fit_vocabulary([train])
    before = copy.deepcopy(definition)
    codec = CompactCodec(definition)
    heldout = copy.deepcopy(train)
    heldout['vertices'][0]['x'] = -12345
    heldout['sectors'][0]['floor_flat'] = 'NEW_FLAT'
    tokens = codec.encode(heldout)
    assert codec.decode(tokens) == heldout
    assert definition == before
    assert 'NEW_FLAT' not in definition['assets']
    assert (-12345 & 65535) not in definition['integers']


def test_compact_causal_counts_and_references_do_not_come_from_source_scaffolds():
    small = build_target(polygon_document([[square(0, 0, 128, 128)]], start=(64, 64)))['supervision']
    big = build_target(two_rooms())['supervision']
    codec = CompactCodec(fit_vocabulary([small, big]))
    a, b = codec.encode(small), codec.encode(big)
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    assert codec.features(a[:common]) == codec.features(b[:common])
    broken = copy.deepcopy(big)
    broken['linedefs'][0]['v1'] = 65000
    with pytest.raises(ValueError, match='illegal compact'):
        codec.encode(broken)
    with pytest.raises(ValueError, match='incomplete'):
        codec.decode(b[:-1])


@pytest.mark.parametrize('field,value', [('assets', ['BAD\0']), ('integers', [True]), ('integers', [65536])])
def test_invalid_checkpoint_vocabularies_are_rejected(field, value):
    definition = fit_vocabulary([build_target(two_rooms())['supervision']])
    definition[field] = value
    with pytest.raises(ValueError):
        CompactCodec(definition)

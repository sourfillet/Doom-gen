import copy

import pytest

from data.geometry_codec import codec_for
from data.geometry_memory import memory_trace
from data.geometry_relative import RelativeCodec, fit_relative_vocabulary, relative_fields
from data.learned_geometry import build_target, core_hashes
from generation.learned_geometry import decode_prediction
from tests.geometry_fixtures import two_rooms


def test_relative_values_roundtrip_exactly_without_changing_authored_controls():
    doc = two_rooms()
    doc.sectors[1].floor_height = -24
    doc.sectors[1].ceil_height = -24  # Closed controls are not raised.
    prediction = build_target(doc)['supervision']
    before = copy.deepcopy(prediction)
    codec = RelativeCodec(fit_relative_vocabulary([prediction]))
    tokens = codec.encode(prediction)
    assert codec.decode(tokens) == prediction == before
    assert core_hashes(decode_prediction(codec.decode(tokens))) == core_hashes(doc)
    assert relative_fields(prediction)['sectors'][1]['ceil_height'] == 0
    trace = memory_trace(codec, tokens)
    assert trace['records'][6][2:4] == [0, 128]  # Memory holds world floor/ceiling.
    assert codec_for(codec.definition).decode(tokens) == prediction


def test_extreme_deltas_are_modular_and_unknown_values_escape_losslessly():
    prediction = build_target(two_rooms())['supervision']
    codec = RelativeCodec(fit_relative_vocabulary([prediction]))
    changed = copy.deepcopy(prediction)
    changed['vertices'][0].update(x=-32768, y=32767)
    changed['vertices'][1].update(x=32767, y=-32768)
    changed['sectors'][0].update(floor_height=-32768, ceil_height=32767)
    changed['sectors'][1].update(floor_height=32767, ceil_height=-32768)
    assert codec.decode(codec.encode(changed)) == changed
    assert relative_fields(changed)['vertices'][1] == {'x': -1, 'y': 1}
    assert relative_fields(changed)['sectors'][0]['ceil_height'] == -1
    changed['vertices'][0]['x'] = 32768
    with pytest.raises(ValueError):
        codec.encode(changed)


def test_relative_prefix_memory_never_uses_future_vertices_or_ceilings():
    prediction = build_target(two_rooms())['supervision']
    codec = RelativeCodec(fit_relative_vocabulary([prediction]))
    original = codec.encode(prediction)
    changed = copy.deepcopy(prediction)
    changed['vertices'][-1]['x'] += 17
    other = codec.encode(changed)
    common = next(i for i, (a, b) in enumerate(zip(original, other)) if a != b)
    assert codec.features(original[:common]) == codec.features(other[:common])
    a, b = codec.state(), codec.state()
    for x, y in zip(original[:common], other[:common]):
        a.push(x)
        b.push(y)
    assert a.prediction == b.prediction
    assert a.record == b.record

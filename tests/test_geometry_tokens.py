import copy
import subprocess
import sys

import pytest

from data.geometry_tokens import (BOS, BYTE_OFFSET, END_LIST, EOS, PAD, RECORD,
                                  TokenState, decode_tokens, encode_prediction, prefix_features, token_masks)
from data.learned_geometry import build_target, core_hashes
from generation.learned_geometry import decode_prediction
from tests.geometry_fixtures import polygon_document, square, two_rooms
from tests.progression_fixtures import short_switch_document


@pytest.mark.parametrize('kind', ['one', 'two', 'hole', 'switch'])
def test_variable_structure_and_complete_authoring_roundtrip(kind):
    if kind == 'one':
        doc = polygon_document([[square(-128, -128, 0, 0)]], start=(-64, -64))
    elif kind == 'two':
        doc = two_rooms()
        doc.sectors[1].floor_height, doc.sectors[1].ceil_height = 24, 192
    elif kind == 'hole':
        doc = polygon_document([[square(0, 0, 256, 256), list(reversed(square(96, 96, 160, 160)))]], start=(48, 48))
    else:
        doc = short_switch_document()
    prediction = build_target(doc)['supervision']
    before = copy.deepcopy(prediction)
    tokens = encode_prediction(prediction)
    assert decode_tokens(tokens) == prediction
    assert core_hashes(decode_prediction(decode_tokens(tokens))) == core_hashes(doc)
    assert prediction == before
    assert len(token_masks(tokens)) == len(tokens) - 1
    assert tokens[0] == BOS and tokens[-1] == EOS
    assert tokens.count(END_LIST) == 5


def test_material_strings_are_lossless_including_empty_and_lowercase():
    prediction = build_target(two_rooms())['supervision']
    prediction['sectors'][0]['floor_flat'] = ''
    prediction['sidedefs'][0]['mid_tex'] = 'startan2'
    assert decode_tokens(encode_prediction(prediction)) == prediction


def test_list_count_is_a_model_decision_and_reference_masks_use_only_emitted_records():
    state = TokenState()
    state.push(BOS)
    assert state.allowed_tokens() == (RECORD,)
    state.push(RECORD)
    for _ in range(4):
        state.push(BYTE_OFFSET)
    assert set(state.allowed_tokens()) == {RECORD, END_LIST}
    # No source object/count is held anywhere in the parser.
    assert state.prediction['vertices'] == [{'x': 0, 'y': 0}]
    prediction = build_target(two_rooms())['supervision']
    tokens = encode_prediction(prediction)
    state = TokenState()
    for token in tokens:
        if state.section == 3 and state.body is not None and len(state.body) == 0:
            assert set(state.allowed_tokens()) == {BYTE_OFFSET + i for i in range(6)}
            break
        state.push(token)
    else:
        pytest.fail('never reached a linedef reference')


@pytest.mark.parametrize('damage', ['truncate', 'pad', 'extra', 'reference', 'float', 'asset', 'early_end'])
def test_invalid_or_incomplete_streams_are_never_completed_from_source(damage):
    prediction = build_target(two_rooms())['supervision']
    tokens = encode_prediction(prediction)
    if damage == 'truncate':
        tokens.pop()
    elif damage == 'pad':
        tokens[2] = PAD
    elif damage == 'extra':
        tokens.append(EOS)
    elif damage in {'reference', 'float', 'asset'}:
        if damage == 'reference':
            prediction['linedefs'][0]['v1'] = 60000
        elif damage == 'float':
            prediction['vertices'][0]['x'] = 1.5
        else:
            prediction['sidedefs'][0]['mid_tex'] = 'NUL\0BAD'
        with pytest.raises(ValueError):
            encode_prediction(prediction)
        return
    else:
        tokens[1] = END_LIST
    with pytest.raises(ValueError):
        decode_tokens(tokens)


def test_tokens_and_decoder_import_without_torch():
    code = '''
import importlib.abc
import sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('Torch was imported')
sys.meta_path.insert(0, NoTorch())
from data.geometry_tokens import encode_prediction, decode_tokens
from data.learned_geometry import build_target
from tests.geometry_fixtures import two_rooms
p = build_target(two_rooms())['supervision']
assert decode_tokens(encode_prediction(p)) == p
'''
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)


def test_prefix_features_do_not_leak_source_count_or_future_structure():
    one = polygon_document([[square(0, 0, 128, 128)]], start=(64, 64))
    small = encode_prediction(build_target(one)['supervision'])
    big = encode_prediction(build_target(two_rooms())['supervision'])
    shared = 0
    for first, second in zip(small, big):
        if first != second:
            break
        shared += 1
    assert shared > 10
    assert prefix_features(small[:shared]) == prefix_features(big[:shared])
    features = prefix_features(big)
    assert features[0][1:] == [0] * 5
    assert features[-1][1:] == [6, 2, 8, 7, 1]

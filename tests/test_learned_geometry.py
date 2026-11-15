"""The new geometry contract must not inherit source scaffolds or repair maps."""
import copy
import json
import subprocess
import sys

import pytest

from data.learned_geometry import (AUTHORING_FIELDS, build_target, core_hashes,
                                   digest, validate_target)
from generation.learned_geometry import assess_prediction, decode_prediction
from tests.geometry_fixtures import polygon_document, square, two_rooms
from tests.progression_fixtures import add_exit, add_key, door_document, short_switch_document
from wad_parser.geometry import sector_area_perimeter
from wad_parser.map_extractor import extract_all_maps
from wad_parser.progression import search_exit_history
from wad_parser.types import SectorSlope, Thing, ThingFlags, Vertex
from wad_parser.wad_reader import WadFile
from wad_writer.wad_writer import write_wad


def prediction(doc=None):
    return build_target(two_rooms() if doc is None else doc)['supervision']


def test_target_separates_explicit_context_supervision_and_audit_without_mutation():
    source = two_rooms()
    before = copy.deepcopy(source.to_dict())
    tags = {'irregular': 0.5}
    target = build_target(source, tags=tags, target_game='doom2')
    validate_target(json.loads(json.dumps(target)))
    assert target['conditioning'] == {'tags': tags, 'target_game': 'doom2'}
    assert set(target['supervision']) == {'schema', *AUTHORING_FIELDS}
    assert 'frame' not in target['supervision']
    assert 'meta' not in target['supervision']
    assert target['audit']['geometry_eligibility'] == 'not_assessed'
    tags['irregular'] = 99
    assert target['conditioning']['tags']['irregular'] == 0.5
    assert source.to_dict() == before
    assert build_target(source)['conditioning'] == {'tags': {}, 'target_game': ''}


def test_irregular_holes_interior_heights_and_shared_boundaries_survive_wad_readback(tmp_path):
    ring = [(0, 0), (0, 384), (192, 384), (192, 256), (384, 256), (384, 0)]
    hole = list(reversed(square(64, 64, 128, 128)))
    source = polygon_document([[ring, hole], [square(384, 0, 512, 256)]], start=(48, 192))
    source.sectors[1].floor_height, source.sectors[1].ceil_height = 24, 192
    source.sectors[1].light_level = 112
    source.vertices.append(Vertex(len(source.vertices), -512, 123))
    payload = prediction(source)
    decoded = decode_prediction(json.loads(json.dumps(payload)))
    assert core_hashes(source) == core_hashes(decoded)
    assert sector_area_perimeter(source, 0) == sector_area_perimeter(decoded, 0)
    assert len(decoded.vertices) == len(source.vertices)
    assert decoded.sectors[1].floor_height == 24
    assert decoded.sectors[1].ceil_height == 192
    assert sum(line.left_sidedef >= 0 for line in decoded.linedefs) == 1
    path = tmp_path / 'prediction.wad'
    write_wad([decoded], path)
    reread, = extract_all_maps(WadFile(str(path)))
    assert core_hashes(reread) == core_hashes(source)
    report = assess_prediction(payload, require_exit=False)
    assert report['status'] == 'static_valid'
    assert report['checks']['serialized']['errors'] == []
    assert report['checks']['node_built'] == report['checks']['engine'] == 'not_run'
    assert report['model_execution'] == 'not_verified'


def test_no_target_metadata_cache_or_source_scaffold_is_needed_to_decode():
    target = build_target(two_rooms())
    payload = copy.deepcopy(target['supervision'])
    del target
    decoded = decode_prediction(payload, map_name='MAP03')
    assert decoded.meta.map_name == 'MAP03'
    assert decoded.meta.source_wad == '<prediction>'
    assert decoded.meta.wad_hash == ''
    assert decoded.derived.sector_regions == []
    assert decoded.extra['learned_geometry_prediction']['repairs'] == []
    assert decoded.extra['learned_geometry_prediction']['model_execution'] == 'not_verified'


def test_same_decoder_accepts_different_predicted_structure_and_counts():
    small = polygon_document([[square(0, 0, 128, 128)]], start=(64, 64))
    big = two_rooms()
    for source in (small, big):
        decoded = decode_prediction(prediction(source))
        assert core_hashes(decoded) == core_hashes(source)
    assert len(prediction(small)['vertices']) != len(prediction(big)['vertices'])
    assert len(prediction(small)['sectors']) != len(prediction(big)['sectors'])


def test_loops_and_distinct_openings_are_not_collapsed_to_one_room_pair_edge():
    loop = polygon_document([[square(0, 0, 128, 128)], [square(128, 0, 256, 128)],
                             [square(128, 128, 256, 256)], [square(0, 128, 128, 256)]], start=(64, 64))
    decoded = decode_prediction(prediction(loop))
    assert sum(line.left_sidedef >= 0 for line in decoded.linedefs) == 4
    assert core_hashes(decoded) == core_hashes(loop)
    split = polygon_document([
        [[(0, 0), (0, 128), (128, 128), (128, 96), (128, 32), (128, 0)]],
        [[(128, 0), (128, 32), (128, 96), (128, 128), (256, 128), (256, 0)]],
    ], start=(64, 64))
    middle = next(line for line in split.linedefs if line.left_sidedef >= 0
                  and {split.vertices[line.v1].y, split.vertices[line.v2].y} == {32, 96})
    middle.flags.impassable = True
    decoded = decode_prediction(prediction(split))
    assert sum(line.left_sidedef >= 0 for line in decoded.linedefs) == 3
    assert sum(line.left_sidedef >= 0 and line.flags.impassable for line in decoded.linedefs) == 1
    assert core_hashes(decoded) == core_hashes(split)


def test_predicted_opening_and_height_change_physical_crossing_not_just_metadata():
    payload = prediction()
    before = copy.deepcopy(payload)
    assert assess_prediction(payload, require_exit=False)['status'] == 'static_valid'
    shared = next(line for line in payload['linedefs'] if line['left_sidedef'] >= 0)
    shared['flags'] |= 1  # Physical blocking, not a decorative opening label.
    blocked = assess_prediction(payload, require_exit=False)
    assert blocked['status'] == 'rejected'
    assert blocked['checks']['in_memory']['unreachable_sectors'] == [1]
    payload = copy.deepcopy(before)
    payload['sectors'][1]['floor_height'] = 25
    raised = assess_prediction(payload, require_exit=False)
    assert raised['status'] == 'rejected'
    assert raised['checks']['in_memory']['unreachable_sectors'] == [1]
    payload['sectors'][1]['floor_height'] = 24
    assert assess_prediction(payload, require_exit=False)['status'] == 'static_valid'


def test_predicted_xy_and_reference_changes_cannot_fall_back_to_source_values():
    source = two_rooms()
    payload = prediction(source)
    for row in payload['vertices'] + payload['things']:
        row['x'] += 8
    shifted = decode_prediction(payload)
    assert [v.x for v in shifted.vertices] == [v.x + 8 for v in source.vertices]
    shared = next(row for row in payload['linedefs'] if row['left_sidedef'] >= 0)
    shared['v2'] = shared['v1']
    with pytest.raises(ValueError, match='zero length'):
        decode_prediction(payload)


@pytest.mark.parametrize('kind', ['key', 'remote_switch'])
def test_full_predicted_actions_preserve_and_control_supported_exit_history(kind):
    if kind == 'key':
        source = door_document(32)
        add_key(source)
        add_exit(source, 2)
    else:
        source = short_switch_document()
    payload = prediction(source)
    decoded = decode_prediction(payload)
    assert core_hashes(source) == core_hashes(decoded)
    assert decoded.sectors[1].ceil_height == 0
    assert search_exit_history(decoded)['status'] == 'success'
    if kind == 'key':
        payload['things'] = [row for row in payload['things'] if row['type'] != 5]
    else:
        next(row for row in payload['linedefs'] if row['special'] == 103)['tag'] = 8
    assert search_exit_history(decode_prediction(payload))['status'] == 'failure'
    assert assess_prediction(payload)['status'] == 'rejected'


def test_remote_controls_unknown_flags_offsets_and_skill_flags_are_retained():
    source = polygon_document([[square(0, 0, 128, 128)], [square(512, 512, 544, 544)]], start=(64, 64))
    source.sectors[1].ceil_height = 0
    source.sectors[1].tag = 7
    source.linedefs[0].special, source.linedefs[0].tag = 103, 7
    source.linedefs[0].extra['unrecognized_flag_bits'] = 0x4000
    source.sidedefs[0].x_offset, source.sidedefs[0].y_offset = -16, 24
    source.things.append(Thing(1, 64, 64, 90, 5, 'BlueCard', 'key', ThingFlags(skill_3=True),
                               extra={'unrecognized_flag_bits': 0x1000}))
    decoded = decode_prediction(prediction(source))
    assert core_hashes(decoded) == core_hashes(source)
    assert len(decoded.sectors) == 2
    assert decoded.sectors[1].ceil_height == 0
    assert decoded.things[1].flags.skill_3
    assert not decoded.things[1].flags.skill_4_5
    # Control reachability policy is unresolved; it must not be silently exempted.
    assert assess_prediction(prediction(source))['status'] == 'rejected'


def test_missing_start_and_exit_are_reported_without_insertion_or_input_mutation():
    payload = prediction()
    payload['things'] = []
    before = copy.deepcopy(payload)
    decoded = decode_prediction(payload)
    assert decoded.things == []
    report = assess_prediction(payload)
    assert report['status'] == 'rejected'
    assert any('start' in error for error in report['checks']['in_memory']['errors'])
    assert any('exit' in error for error in report['checks']['in_memory']['errors'])
    assert report['repairs'] == report['transformations'] == []
    assert payload == before


@pytest.mark.parametrize('bad', ['schema', 'extra', 'missing', 'float', 'bool', 'nan',
                               'reference', 'side_reference', 'range', 'flags', 'asset'])
def test_malformed_predictions_fail_explicitly_without_repair(bad):
    payload = prediction()
    if bad == 'schema':
        payload['schema'] = 'doom_spatial_prediction_v1'
    elif bad == 'extra':
        payload['frame'] = {'scale': 128}
    elif bad == 'missing':
        del payload['vertices'][0]['x']
    elif bad in {'float', 'bool', 'nan'}:
        payload['vertices'][0]['x'] = {'float': 1.25, 'bool': True, 'nan': float('nan')}[bad]
    elif bad == 'reference':
        payload['linedefs'][0]['v1'] = 999999
    elif bad == 'side_reference':
        payload['linedefs'][0]['left_sidedef'] = -2
    elif bad == 'range':
        payload['vertices'][0]['x'] = 32768
    elif bad == 'flags':
        payload['linedefs'][0]['flags'] = 65536
    else:
        payload['sectors'][0]['floor_flat'] = 'TOOLONGNAME'
    with pytest.raises(ValueError):
        decode_prediction(payload)
    report = assess_prediction(payload)
    assert report['status'] == 'rejected'
    assert report['checks']['decode'] == 'failed'
    assert report['checks']['engine'] == 'not_run'


@pytest.mark.parametrize('bad', ['udmf', 'hexen', 'slope', 'thing_z', 'args', 'extra', 'flag'])
def test_source_extensions_cannot_be_silently_discarded(bad):
    source = two_rooms()
    if bad in {'udmf', 'hexen'}:
        source.meta.format = bad
    elif bad == 'slope':
        source.sectors[0].floor_slope = SectorSlope(0, 0, 1, 0)
    elif bad == 'thing_z':
        source.things[0].z = 0
    elif bad == 'args':
        source.linedefs[0].args = [0] * 5
    elif bad == 'flag':
        source.linedefs[0].flags.block_everything = True
    else:
        source.sectors[0].extra['floorheight'] = 8
    with pytest.raises(ValueError):
        build_target(source)


def test_poisoned_source_bsp_is_not_in_prediction_or_decoder():
    source = two_rooms()
    source.derived.sector_regions = [[[(0, 0), (1000, 1000), (0, 1000)]]] * 2
    payload = prediction(source)
    assert decode_prediction(payload).derived.sector_regions == []
    assert sector_area_perimeter(decode_prediction(payload), 0)[0] == 128**2


def test_changed_target_and_rehashed_stale_audit_are_rejected():
    target = build_target(two_rooms())
    target['supervision']['sectors'][0]['ceil_height'] += 8
    with pytest.raises(ValueError, match='fingerprint'):
        validate_target(target)
    target['fingerprint'] = digest({key: value for key, value in target.items() if key != 'fingerprint'})
    with pytest.raises(ValueError, match='audit differs'):
        validate_target(target)


def test_torch_is_optional_for_targets_decoding_and_assessment():
    code = '''
import importlib.abc
import sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('Torch was imported')
sys.meta_path.insert(0, NoTorch())
from data.learned_geometry import build_target
from generation.learned_geometry import assess_prediction
from tests.geometry_fixtures import two_rooms
assert assess_prediction(build_target(two_rooms())['supervision'], require_exit=False)['status'] == 'static_valid'
'''
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)

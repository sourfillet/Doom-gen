import copy

from data.learned_geometry import build_target
from generation.geometry_evidence import assess_geometry, layout_statistics
from generation.learned_geometry import assess_prediction, decode_prediction
from tests.geometry_fixtures import polygon_document, square, two_rooms
from wad_parser.types import Thing, ThingFlags


def test_square_failure_descriptor_ignores_collinear_vertices_and_decorative_sector_counts():
    divided = two_rooms()
    assert layout_statistics(divided)['single_flat_rectangle']
    collinear = polygon_document([[[
        (0, 0), (0, 64), (0, 128), (64, 128), (128, 128), (128, 64), (128, 0), (64, 0),
    ]]], start=(64, 64))
    stats = layout_statistics(collinear)
    assert stats['outline_corners'] == 4
    assert stats['single_flat_rectangle']
    divided.sectors[1].floor_height = 24
    assert not layout_statistics(divided)['single_flat_rectangle']


def test_l_shaped_footprint_and_courtyard_are_measured_without_changing_geometry():
    shape = polygon_document([[[(0, 0), (0, 256), (128, 256), (128, 128), (256, 128), (256, 0)]]], start=(64, 64))
    before = copy.deepcopy(shape.to_dict())
    stats = layout_statistics(shape)
    assert stats['outline_corners'] == 6
    assert not stats['single_flat_rectangle']
    assert shape.to_dict() == before
    courtyard = polygon_document([[square(0, 0, 256, 256), list(reversed(square(96, 96, 160, 160)))]], start=(48, 48))
    assert layout_statistics(courtyard)['holes'] == 1
    assert not layout_statistics(courtyard)['single_flat_rectangle']


def test_geometry_admission_does_not_mistake_actor_history_failures_for_bad_wall_labels():
    source = two_rooms()
    source.things.append(Thing(1, 999, 999, 0, 3004, 'Zombieman', 'monster', ThingFlags(skill_3=True)))
    payload = build_target(source)['supervision']
    before = copy.deepcopy(payload)
    evidence = assess_geometry(payload)
    assert evidence['status'] == 'raw_geometry_valid'
    assert evidence['checks']['actors'] == evidence['checks']['progression'] == 'not_assessed'
    assert evidence['checks']['compiled'] == 'not_run'
    assert assess_prediction(payload)['status'] == 'rejected'
    assert payload == before
    assert len(decode_prediction(payload).things) == 2


def test_geometry_policy_still_rejects_crossing_or_missing_boundaries():
    payload = build_target(two_rooms())['supervision']
    payload['linedefs'].pop(0)
    evidence = assess_geometry(payload)
    assert evidence['status'] == 'rejected'
    assert evidence['checks']['authoring']['errors']
    assert evidence['repairs'] == []

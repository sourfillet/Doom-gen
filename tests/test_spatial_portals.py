import copy
import math

import pytest

from data.spatial_portals import audit_portals, EVIDENCE_SCHEMA
from tests.geometry_fixtures import polygon_document, square, two_rooms
from wad_parser.types import Thing, ThingFlags


def evidence(doc, *, door=None):
    """Minimal candidate records, deliberately independent of segmentation."""
    ids = [sector.id for sector in doc.sectors if sector.id != door]
    rooms = [{'id': sid, 'components': [{'source_component': [sid, 0]}]} for sid in ids]

    def span(line):
        return {'linedef': line.id, 'special': line.special, 'tag': line.tag,
                'points': [[doc.vertices[v].x, doc.vertices[v].y] for v in (line.v1, line.v2)]}

    shared = [line for line in doc.linedefs if line.left_sidedef >= 0 and not line.flags.impassable]
    if door is None:
        edges = [{'a': ids[0], 'b': ids[1], 'kind': 'open',
                  'authored_boundaries': [span(line) for line in shared], 'door_faces': []}]
    else:
        faces = {'source_component': [door, 0]}
        for side, sid in zip(('a', 'b'), ids):
            faces[side] = [span(line) for line in shared
                           if {doc.sidedefs[line.right_sidedef].sector,
                               doc.sidedefs[line.left_sidedef].sector} == {sid, door}]
        edges = [{'a': ids[0], 'b': ids[1], 'kind': 'door', 'authored_boundaries': [], 'door_faces': [faces]}]
    return {'schema': EVIDENCE_SCHEMA, 'frame': {'origin': [0, 0], 'scale': 1},
            'rooms': rooms, 'connections': edges}


def windows(contiguous):
    ys = [0, 64, 88, 112 if contiguous else 120, 144, 256]
    left = [(0, 0), (0, 256), *[(128, y) for y in reversed(ys)]]
    right = [*[(128, y) for y in ys], (256, 256), (256, 0)]
    doc = polygon_document([[left], [right]], start=(64, 76))
    for line in doc.linedefs:
        if line.left_sidedef >= 0:
            bounds = sorted([doc.vertices[v].y for v in (line.v1, line.v2)])
            line.flags.impassable = bounds not in ([64, 88], [88, 112]) if contiguous else bounds not in ([64, 88], [120, 144])
    return doc


def test_separate_narrow_windows_are_not_added_together():
    doc = windows(False)
    result = audit_portals(doc, evidence(doc))['connections'][0]
    assert [r['length'] for r in result['contiguous_runs']] == [24, 24]
    assert result['status'] == 'unsupported_or_unusable'


def test_touching_split_windows_merge_and_allow_crossing_without_mutation():
    doc = windows(True)
    target = evidence(doc)
    before = copy.deepcopy((doc.to_dict(), target))
    result = audit_portals(doc, target)['connections'][0]
    assert [r['length'] for r in result['contiguous_runs']] == [48]
    assert result['local_directions']['initial'] == [(0, 1), (1, 0)]
    assert (doc.to_dict(), target) == before
    assert all(0 <= u <= 1 for span in result['authored_boundaries']
               for r in span['crossings']['initial'] for interval in r['entry_intervals'] for u in interval)


@pytest.mark.parametrize('width, expected', [(40, False), (48, True)])
def test_diagonal_player_box_clearance(width, expected):
    def rotate(ring):
        return [((x - y) / math.sqrt(2), (x + y) / math.sqrt(2)) for x, y in ring]
    doc = polygon_document([[rotate(square(0, 0, 128, width))],
                            [rotate(square(128, 0, 256, width))]],
                           start=rotate([(64, width / 2)])[0])
    result = audit_portals(doc, evidence(doc))['connections'][0]
    assert bool(result['local_directions']['initial']) is expected


def test_high_drop_preserves_direction():
    doc = two_rooms()
    doc.sectors[0].floor_height = 64
    result = audit_portals(doc, evidence(doc))['connections'][0]
    assert result['local_directions']['initial'] == [(0, 1)]


@pytest.mark.parametrize('has_key', [False, True])
def test_locked_door_preserves_actions_and_conditional_status(has_key):
    doc = polygon_document([[square(0, 0, 128, 128)], [square(128, 0, 160, 128)],
                            [square(160, 0, 288, 128)]], start=(64, 64))
    doc.sectors[1].ceil_height = 0
    trigger = next(line for line in doc.linedefs if line.left_sidedef >= 0)
    trigger.special = 26
    if has_key:
        doc.things.append(Thing(1, 64, 64, 0, 5, 'BlueCard', 'key', ThingFlags(skill_3=True)))
    before = copy.deepcopy(doc.to_dict())
    result = audit_portals(doc, evidence(doc, door=1))
    assert result['connections'][0]['local_directions']['initial'] == []
    assert result['connections'][0]['status'] == ('conditional_local' if has_key else 'unsupported_or_unusable')
    assert result['door_activators'] == [{'linedef': trigger.id, 'special': 26, 'tag': 0,
                                        'key': 'blue', 'target_sectors': [1], 'front_component': (0, 0)}]
    assert doc.to_dict() == before


def test_teleports_are_remote_and_active_trigger_is_not_an_open_contact():
    doc = polygon_document([[square(0, 0, 128, 128)], [square(128, 0, 256, 128)],
                            [square(512, 0, 640, 128)]], start=(64, 64))
    trigger = next(line for line in doc.linedefs if line.left_sidedef >= 0)
    trigger.special, trigger.tag = 39, 7
    doc.sectors[2].tag = 7
    doc.things.append(Thing(1, 576, 64, 0, 14, 'TeleportDest', 'teleport-dest', ThingFlags(skill_3=True)))
    target = evidence(doc)
    target['connections'].append({'a': 0, 'b': 2, 'kind': 'teleport'})
    result = audit_portals(doc, target)
    assert (0, 1) not in result['connections'][0]['local_directions']['potential']
    assert result['connections'][1]['status'] == 'remote'


def test_stale_boundary_or_wrong_room_is_rejected():
    doc = two_rooms()
    target = evidence(doc)
    target['connections'][0]['authored_boundaries'][0]['points'][0][0] += 1
    with pytest.raises(ValueError, match='stale candidate'):
        audit_portals(doc, target)
    target = evidence(doc)
    target['connections'][0]['b'] = 10
    with pytest.raises(ValueError, match='claimed rooms'):
        audit_portals(doc, target)


def test_door_faces_cannot_join_disconnected_internal_free_regions():
    neck = [(128, 0), (128, 128), (256, 128), (256, 76), (288, 76), (288, 128),
            (416, 128), (416, 0), (288, 0), (288, 52), (256, 52), (256, 0)]
    doc = polygon_document([[square(0, 0, 128, 128)], [neck],
                            [square(416, 0, 544, 128)]], start=(64, 64))
    doc.sectors[1].ceil_height = 0
    next(line for line in doc.linedefs if line.left_sidedef >= 0).special = 1
    edge = audit_portals(doc, evidence(doc, door=1))['connections'][0]
    door = edge['door_faces'][0]
    assert all(any(face['crossings']['potential'] for face in door['faces'][side]) for side in ('a', 'b'))
    assert door['joins']['potential'] == []
    assert edge['status'] == 'unsupported_or_unusable'

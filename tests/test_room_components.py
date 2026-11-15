import copy

import pytest

from tests.geometry_fixtures import polygon_document, square
from roomgraph.segment import segment
from data.roomgraph_targets import graph_to_sequence
from wad_parser.map_extractor import compute_derived
from wad_parser.types import Linedef, LinedefFlags, Sidedef, Thing, ThingFlags


def test_disconnected_islands_of_one_sector_are_separate_rooms_and_objectives():
    doc = polygon_document([[square(0, 0, 128, 128), square(512, 0, 640, 128)]], start=(64, 64))
    exit_line = next(line for line in doc.linedefs if doc.vertices[line.v1].x >= 512)
    exit_line.special = 11
    before = copy.deepcopy(doc.to_dict())
    full = segment(doc)
    assert len(full.rooms) == 2 and all(room.sectors == [0] for room in full.rooms)
    assert len({tuple(room.components) for room in full.rooms}) == 2
    with pytest.raises(ValueError, match='disconnected'):
        graph_to_sequence(full)
    playable = segment(doc, playable_only=True)
    assert len(playable.rooms) == 1
    assert playable.excluded_sectors == []  # sector 0 is partly reachable
    assert playable.excluded_components == [(0, 1)]
    assert 'exit' not in playable.rooms[0].roles
    assert doc.to_dict() == before


def test_shared_sector_islands_do_not_shortcut_a_room_between_them():
    doc = polygon_document([[square(0, 0, 128, 128), square(256, 0, 384, 128)],
                             [square(128, 0, 256, 128)]], start=(64, 64))
    separate = segment(doc, merge_width=200, playable_only=True)
    assert len(separate.rooms) == 3 and len(separate.connections) == 2
    assert [step['parent'] for step in graph_to_sequence(separate)] == [-1, 0, 1]
    joined = segment(doc, playable_only=True)
    assert len(joined.rooms) == 1
    assert joined.rooms[0].area == 384 * 128
    assert joined.rooms[0].sectors == [0, 1]
    assert len(joined.rooms[0].components) == 3


def test_teleport_between_islands_of_the_same_sector_remains_a_connection():
    doc = polygon_document([[square(0, 0, 128, 128), square(512, 0, 640, 128)]], start=(64, 96))
    doc.sectors[0].tag = 7
    right = len(doc.sidedefs)
    doc.sidedefs.extend([Sidedef(right, 0, 0, '-', '-', '-', 0),
                         Sidedef(right + 1, 0, 0, '-', '-', '-', 0)])
    doc.linedefs.append(Linedef(len(doc.linedefs), 0, 2, LinedefFlags(two_sided=True),
                                97, 7, right, right + 1))
    doc.things.append(Thing(1, 576, 64, 0, 14, 'TeleportDest', 'teleport-dest', ThingFlags(skill_3=True)))
    doc.bounds, doc.derived = compute_derived(doc.vertices, doc.sectors, doc.linedefs, doc.sidedefs, doc.things)
    graph = segment(doc, playable_only=True)
    assert len(graph.rooms) == 2
    assert len(graph.connections) == 1 and graph.connections[0].kind == 'teleport'
    assert not graph.excluded_components

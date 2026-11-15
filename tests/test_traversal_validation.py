import copy
import math
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parent)]
from geometry_fixtures import two_rooms, polygon_document, square
from wad_parser.traversal import distances_from_start, movement_neighbors
from wad_parser.types import Thing, ThingFlags
from generation.validator import validate_and_repair, validate_document
from generation.progression import ensure_exit
import pytest


def _shared(doc):
    return next(line for line in doc.linedefs if line.left_sidedef != -1)


def test_blocking_and_shared_clearance_are_not_walkable():
    doc = two_rooms()
    assert set(distances_from_start(doc)) == {0, 1}
    _shared(doc).flags.impassable = True
    assert set(distances_from_start(doc)) == {0}
    _shared(doc).flags.impassable = False
    doc.sectors[0].ceil_height = 56
    doc.sectors[1].floor_height, doc.sectors[1].ceil_height = 24, 152
    assert set(distances_from_start(doc)) == {0}
    report = validate_document(doc, require_exit=False, require_connected=True)
    assert any("unreachable" in error for error in report["errors"])


def test_narrow_opening_is_not_walkable():
    doc = polygon_document([[square(0, 0, 128, 16)], [square(128, 0, 256, 16)]], start=(64, 8))
    assert set(distances_from_start(doc)) == {0}


def test_short_decorative_edges_do_not_trap_a_player_or_split_a_room():
    from tests.geometry_fixtures import decorative_start
    from roomgraph.segment import segment
    from generation.actors import spawn_problem
    doc = decorative_start()
    assert set(distances_from_start(doc)) == {0, 1}
    assert spawn_problem(doc, 1, 256, 256, ignore_id=0) is None
    assert not validate_document(doc, require_exit=False, require_connected=True)['errors']
    assert len(segment(doc, playable_only=True).rooms) == 1


@pytest.mark.parametrize('width,passable', [(40, False), (48, True)])
def test_diagonal_opening_respects_the_axis_aligned_player_box(width, passable):
    def rotate(ring):
        return [((x - y) / math.sqrt(2), (x + y) / math.sqrt(2)) for x, y in ring]
    start = rotate([(64, width / 2)])[0]
    doc = polygon_document([[rotate(square(0, 0, 128, width))],
                             [rotate(square(128, 0, 256, width))]], start=start)
    assert (1 in distances_from_start(doc)) is passable


def test_drops_are_directed():
    doc = two_rooms()
    doc.sectors[0].floor_height, doc.sectors[0].ceil_height = 128, 256
    doc.sectors[1].ceil_height = 256
    neighbors = movement_neighbors(doc.vertices, doc.sectors, doc.linedefs, doc.sidedefs)
    assert neighbors[0] == {1}
    assert neighbors[1] == set()


def test_default_repairs_preserve_darkness_and_floor_steps():
    doc = two_rooms()
    doc.sectors[0].light_level = 0
    doc.sectors[1].floor_height, doc.sectors[1].ceil_height = 128, 256
    repairs = validate_and_repair(doc)
    assert doc.sectors[0].light_level == 0
    assert doc.sectors[1].floor_height == 128
    assert not any("lowered" in action or "teleporter" in action for action in repairs)


def test_closed_unlocked_door_is_potentially_reachable_and_preserved():
    doc = polygon_document([[square(0, 0, 128, 128)], [square(128, 0, 160, 128)],
                             [square(160, 0, 288, 128)]], start=(64, 64))
    doc.sectors[1].ceil_height = 0
    for line in doc.linedefs:
        if line.left_sidedef >= 0 and doc.sidedefs[line.left_sidedef].sector == 1:
            line.special = 1
    assert set(distances_from_start(doc)) == {0, 1, 2}
    assert set(distances_from_start(doc, allow_doors=False)) == {0}
    validate_and_repair(doc)
    assert doc.sectors[1].ceil_height == 0
    for line in doc.linedefs:
        if line.special == 1:
            line.special = 26  # locked doors also retain their authored ceiling
    validate_and_repair(doc)
    assert doc.sectors[1].ceil_height == 0
    assert set(distances_from_start(doc)) == {0}


def test_teleports_are_directed_and_do_not_include_monster_only_or_untagged_actions():
    doc = polygon_document([[square(0, 0, 128, 128)], [square(128, 0, 256, 128)],
                             [square(512, 0, 640, 128)]], start=(64, 64))
    line = _shared(doc)
    line.special, line.tag = 97, 7
    doc.sectors[2].tag = 7
    doc.things.append(Thing(1, 576, 64, 0, 14, "TeleportDest", "teleport-dest", ThingFlags(skill_3=True)))
    assert set(distances_from_start(doc)) == {0, 2}
    from wad_parser.map_extractor import compute_derived
    from roomgraph.segment import segment
    doc.bounds, doc.derived = compute_derived(doc.vertices, doc.sectors, doc.linedefs, doc.sidedefs, doc.things)
    graph = segment(doc, playable_only=True)
    assert graph.excluded_sectors == [1]
    assert {sid for room in graph.rooms for sid in room.sectors} == {0, 2}
    destination = doc.things[-1]
    destination.flags.skill_3 = False
    assert set(distances_from_start(doc)) == {0, 1}
    destination.flags.skill_3 = True
    destination.flags.not_single_player = True
    assert set(distances_from_start(doc)) == {0, 1}
    destination.flags.not_single_player = False
    doc.things[0].x = 576
    assert set(distances_from_start(doc)) == {2}
    doc.things[0].x = 64
    line.special = 125
    assert set(distances_from_start(doc)) == {0, 1}
    line.special, line.tag = 97, 0
    assert set(distances_from_start(doc)) == {0, 1}


def test_final_validation_is_non_mutating_and_requires_a_usable_exit():
    doc = two_rooms()
    before = copy.deepcopy(doc.to_dict())
    report = validate_document(doc)
    assert any("exit" in error for error in report["errors"])
    assert doc.to_dict() == before
    ensure_exit(doc)
    assert not validate_document(doc)["errors"]
    doc.things[0].x = -100
    assert any("start" in error for error in validate_document(doc)["errors"])


def test_disjoint_parts_of_one_sector_are_not_magically_connected():
    doc = polygon_document([[square(0, 0, 128, 128), square(512, 0, 640, 128)]], start=(64, 64))
    from wad_parser.traversal import component_distances
    assert len(component_distances(doc)) == 1
    for line in doc.linedefs:
        if doc.vertices[line.v1].x >= 512:
            line.special = 11
            break
    report = validate_document(doc, require_connected=True)
    assert any('components' in error for error in report['errors'])
    assert any('exit' in error for error in report['errors'])


def _ledge_case(rings, floors, exit_y):
    doc = polygon_document([[square(0, 0, 64, 128)], *rings], start=(32, 96))
    for sector, floor in zip(doc.sectors, floors):
        sector.floor_height = floor
    next(line for line in doc.linedefs
         if doc.vertices[line.v1].y == doc.vertices[line.v2].y == exit_y).special = 11
    return doc


def _thin_step(gap):
    # Native drop controls: P 48 | ledge | thin step K 16 (8 deep, `gap`
    # below the ledge) | room D 0 with the exit on its far wall.
    d = [square(0, -256, 64, -gap - 8)] + ([square(0, -gap, 64, 0)] if gap else [])
    return _ledge_case([d, [square(0, -gap - 8, 64, -gap)]], (48, 0, 16), -256)


@pytest.mark.parametrize('gap', [0, 4, 8])
def test_drop_crosses_thin_landing_components_at_the_ledge_height(gap):
    # Every step control descends natively. A box over the strip or step still
    # overlaps the ledge, so the player stays at its floor until clear of it.
    from data.roomgraph_targets import graph_to_sequence
    from data.source_endings import assess_source_ending
    from roomgraph.segment import segment
    doc = _thin_step(gap)
    graph = segment(doc, playable_only=True)
    assert any(room.role == 'exit' or 'exit' in room.roles for room in graph.rooms)
    assert assess_source_ending(doc, graph)['complete']
    # The strip/step pair is crossed only at the ledge height; the room graph
    # must still connect it, or target encoding rejects the source.
    assert len(graph_to_sequence(graph)) == len(graph.rooms)


@pytest.mark.parametrize('depth', [16, 24, 40, 128])
def test_ledge_support_ends_when_the_box_clears_the_ledge(depth):
    # P 48 | ledge | D 0 | Q 64: once clear of the ledge the player stands at 0
    # and cannot climb 64, although 64 is within a step of the ledge height.
    from data.source_endings import assess_source_ending
    from roomgraph.segment import segment
    from wad_parser.traversal import component_distances
    doc = _ledge_case([[square(0, -depth, 64, 0)], [square(0, -depth - 128, 64, -depth)]],
                      (48, 0, 64), -depth - 128)
    assert {sid for sid, _ in component_distances(doc)} == {0, 1}
    assert not assess_source_ending(doc, segment(doc, playable_only=True))['complete']


def test_ledge_height_step_across_a_thin_gap_is_potentially_reachable():
    # The box touches the 48 ledge and the 64 platform's line together across
    # an 8-unit strip, a 16-unit step under vanilla P_TryMove (not natively
    # observed here). Physical navigation remains conservative for it.
    from wad_parser.traversal import component_distances
    doc = _ledge_case([[square(0, -8, 64, 0)], [square(0, -136, 64, -8)]], (48, 0, 64), -136)
    assert {sid for sid, _ in component_distances(doc)} == {0, 1, 2}

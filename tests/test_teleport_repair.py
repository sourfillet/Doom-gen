"""Repair teleporters need clear player landings, not marker dimensions."""
from copy import deepcopy

import pytest

from generation.actors import actor_profile, spawn_problem
from generation.style import _add_thing
from generation.validator import _teleport_landing_point, validate_and_repair
from tests.geometry_fixtures import polygon_document, square, repair_arena
from wad_parser.clearance import MapClearance
from wad_parser.traversal import distances_from_start


def landing(doc, sid):
    geometry = MapClearance(doc.vertices, doc.sectors, doc.linedefs, doc.sidedefs)
    return _teleport_landing_point(doc, sid, geometry)


def test_repair_connects_both_islands_and_preserves_existing_solid_actors():
    doc = repair_arena()
    _add_thing(doc, 1152, 128, 3004)
    before = deepcopy(doc.things)
    original_lines = deepcopy(doc.linedefs)
    repairs = validate_and_repair(doc, reconnect=True)
    assert any('teleporter pairs' in repair for repair in repairs)
    assert not any('remain disconnected' in repair for repair in repairs)
    assert set(distances_from_start(doc)) == set(range(6))
    # The original passages remain ordinary walk-through lines.
    assert doc.linedefs[:len(original_lines)] == original_lines
    assert doc.things[:len(before)] == before
    assert actor_profile(14) is None  # marker dimensions do not model a landing
    markers = [t for t in doc.things if t.type == 14]
    assert len(markers) == 2
    assert all(spawn_problem(doc, 1, t.x, t.y) is None for t in markers)


@pytest.mark.parametrize('width,headroom', [(24, 128), (256, 55)])
def test_landing_rejects_space_that_only_fits_a_marker(width, headroom):
    doc = polygon_document([[square(0, 0, width, 256)]])
    doc.sectors[0].ceil_height = headroom
    assert landing(doc, 0) is None


def test_landing_avoids_active_solid_actors_but_allows_non_solid_pickups():
    doc = polygon_document([[square(0, 0, 256, 256)]])
    pickup = _add_thing(doc, 128, 128, 2008)
    assert landing(doc, 0) == (128, 128)
    enemy = _add_thing(doc, 128, 128, 3004)
    point = landing(doc, 0)
    assert point is not None and point != (128, 128)
    assert spawn_problem(doc, 1, *point) is None
    enemy.flags.skill_3 = False
    assert landing(doc, 0) == (128, 128)
    assert pickup.type == 2008


def test_fully_blocked_island_remains_unmodified_and_is_reported():
    doc = repair_arena()
    for x in (1152, 1408):
        _add_thing(doc, x, 128, 7)  # radius 128 blocks every player-sized landing
    before = deepcopy(doc)
    repairs = validate_and_repair(doc, reconnect=True)
    assert any('remain disconnected' in repair for repair in repairs)
    assert doc.linedefs == before.linedefs
    assert doc.sectors == before.sectors
    assert doc.things == before.things


def test_landing_does_not_reuse_an_existing_destination_marker():
    doc = polygon_document([[square(0, 0, 256, 256)]])
    _add_thing(doc, 128, 128, 14)
    point = landing(doc, 0)
    assert point is not None
    assert abs(point[0] - 128) >= 32 or abs(point[1] - 128) >= 32

import copy

import pytest

from tests.geometry_fixtures import polygon_document, square
from tests.progression_fixtures import (add_exit, add_key, branch_document, door_document,
                                        shared_key_branches, short_switch_document)
from wad_parser.progression import search_exit_history
from wad_parser.traversal import potential_state


def test_keys_from_an_irreversible_branch_cannot_open_another_branch():
    from generation.validator import validate_document
    doc = branch_document()
    # The optimistic union is a useful upper bound, but not a valid history.
    assert {sid for sid, _ in potential_state(doc)['distances']} == set(range(5))
    before = copy.deepcopy(doc.to_dict())
    report = search_exit_history(doc)
    assert report['status'] == 'failure'
    assert any('door/key progression' in error for error in validate_document(doc)['errors'])
    assert doc.to_dict() == before


def test_a_return_route_allows_the_same_key_and_door_sequence():
    from generation.validator import validate_document
    doc = branch_document(return_from_key=True)
    report = search_exit_history(doc)
    assert report['status'] == 'success'
    assert [event['action'] for event in report['trace']] == ['key', 'door-open']
    assert not validate_document(doc)['errors']


@pytest.mark.parametrize('x', [4, 32, 96])
def test_keys_near_walls_can_be_touched_without_standing_on_their_centres(x):
    doc = door_document(32)
    add_key(doc, x=x)
    add_exit(doc, 2)
    assert search_exit_history(doc)['status'] == 'success'


def test_other_skill_keys_do_not_unlock_a_history():
    doc = door_document(32)
    add_key(doc, active=False)
    add_exit(doc, 2)
    assert search_exit_history(doc)['status'] == 'failure'


def test_budget_exhaustion_is_explicit_and_never_a_success():
    doc = branch_document(return_from_key=True)
    assert search_exit_history(doc, max_states=1)['status'] == 'inconclusive'
    assert search_exit_history(doc, max_geometries=1)['status'] == 'inconclusive'


def test_a_switch_does_not_make_an_impassable_portal_traversable():
    doc = door_document(0)
    doc.sectors[1].tag = 7
    switch = next(line for line in doc.linedefs if line.left_sidedef == -1
                  and doc.sidedefs[line.right_sidedef].sector == 0)
    switch.special, switch.tag = 103, 7
    assert search_exit_history(doc)['status'] == 'failure'  # no exit yet
    add_exit(doc, 2)
    assert search_exit_history(doc)['status'] == 'success'
    for line in doc.linedefs:
        if line.left_sidedef >= 0 and doc.sidedefs[line.right_sidedef].sector == 0:
            line.flags.impassable = True
    # The switch can still open it; the impassable portal cannot be crossed.
    assert search_exit_history(doc)['status'] == 'failure'


def test_switch_use_ray_must_not_pass_through_a_closed_pillar():
    pillar = square(4, 8, 8, 120)
    doc = polygon_document([[square(0, 0, 128, 128), list(reversed(pillar))],
                            [square(128, 0, 160, 128)], [square(160, 0, 288, 128)],
                            [pillar]], start=(64, 64))
    doc.sectors[1].ceil_height = doc.sectors[3].ceil_height = 0
    doc.sectors[1].tag = 7
    switch = next(line for line in doc.linedefs
                  if doc.vertices[line.v1].x == doc.vertices[line.v2].x == 0)
    switch.special, switch.tag = 103, 7
    add_exit(doc, 2)
    assert {sid for sid, _ in potential_state(doc)['distances']} >= {0, 1, 2}
    assert search_exit_history(doc)['status'] == 'failure'


def test_a_key_above_the_players_touch_height_cannot_unlock_a_door():
    # A tiny raised pedestal is too small to stand on. Its key overlaps the
    # player's XY touch box but sits above the player's head.
    pedestal = square(56, 56, 72, 72)
    doc = polygon_document([[square(0, 0, 128, 128), list(reversed(pedestal))],
                            [square(128, 0, 160, 128)], [square(160, 0, 288, 128)],
                            [pedestal]], start=(32, 64))
    doc.sectors[1].ceil_height = 0
    doc.sectors[3].floor_height, doc.sectors[3].ceil_height = 80, 208
    door = next(line for line in doc.linedefs if line.left_sidedef >= 0
                and doc.sidedefs[line.right_sidedef].sector == 0
                and doc.sidedefs[line.left_sidedef].sector == 1)
    door.special = 32
    add_key(doc, x=64)
    doc.things[-1].y = 64
    add_exit(doc, 2)
    assert search_exit_history(doc)['status'] == 'failure'


def test_a_use_switch_can_be_shorter_than_the_players_collision_box():
    doc = short_switch_document()
    assert search_exit_history(doc)['status'] == 'success'


def test_pickup_search_keeps_alternative_approaches_from_irreversible_branches():
    # The key can be touched from either side of a thin wall. Collecting it
    # from the left branch traps the player; the right approach permits exit.
    assert search_exit_history(shared_key_branches())['status'] == 'success'


def test_cached_door_states_keep_independent_physical_heights():
    from wad_parser.clearance import MapClearance
    doc = door_document(31)
    closed = MapClearance(doc.vertices, doc.sectors, doc.linedefs, doc.sidedefs)
    opened = closed.with_heights(potential_state(doc)['sectors'])
    assert opened.problem(1, 144, 64) is None
    assert closed.problem(1, 144, 64) == 'insufficient vertical clearance'
    opened.ceilings[1] = 64
    assert closed.ceilings[1] == 0

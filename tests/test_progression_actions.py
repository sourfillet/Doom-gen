import copy

import pytest

from tests.progression_fixtures import add_key, door_document
from wad_parser.traversal import distances_from_start, potential_state


@pytest.mark.parametrize('kind', [5, 40])
def test_manual_locked_door_accepts_a_reachable_card_or_skull(kind):
    doc = door_document()
    assert set(distances_from_start(doc)) == {0}
    add_key(doc, kind)
    before = copy.deepcopy(doc.to_dict())
    state = potential_state(doc)
    assert {sid for sid, _ in state['distances']} == {0, 1, 2}
    assert state['keys'] == ['blue'] and state['actions'][0]['key'] == 'blue'
    assert doc.to_dict() == before
    assert set(distances_from_start(doc, allow_doors=False)) == {0}


@pytest.mark.parametrize('kind,x,active', [(13, 96, True), (5, 224, True), (5, 96, False)])
def test_wrong_unreachable_and_other_difficulty_keys_do_not_unlock_doors(kind, x, active):
    doc = door_document()
    add_key(doc, kind, x, active=active)
    assert set(distances_from_start(doc)) == {0}


@pytest.mark.parametrize('special', [29, 61, 63, 103, 111, 112, 114, 115])
def test_reachable_remote_switch_opens_only_its_tagged_door(special):
    doc = door_document(0)
    doc.sectors[1].tag = 7
    switch = next(line for line in doc.linedefs if line.left_sidedef == -1
                  and doc.sidedefs[line.right_sidedef].sector == 0)
    switch.special, switch.tag = special, 7
    assert set(distances_from_start(doc)) == {0, 1, 2}
    switch.tag = 8
    assert set(distances_from_start(doc)) == {0}


def test_switch_behind_its_own_door_cannot_open_it():
    doc = door_document(0)
    doc.sectors[1].tag = 7
    switch = next(line for line in doc.linedefs if line.left_sidedef == -1
                  and doc.sidedefs[line.right_sidedef].sector == 2)
    switch.special, switch.tag = 103, 7
    assert set(distances_from_start(doc)) == {0}


def test_an_already_open_locked_door_does_not_physically_block_a_player():
    doc = door_document()
    doc.sectors[1].ceil_height = 128
    assert set(distances_from_start(doc)) == {0, 1, 2}

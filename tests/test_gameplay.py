import copy
from collections import Counter
import math

import pytest

from generation.gameplay import apply_intro, MONSTER_HEALTH
from generation.interiors import add_columns
from generation.style import StyleModel, apply_style
from generation.validator import require_valid
from tests.gameplay_fixtures import arena
from wad_parser.geometry import sector_points, point_in_rings
from wad_parser.navigation import PlayerNavigation


def test_style_learns_lights_without_repeating_palette_or_adding_another_exit():
    doc = arena()
    source = copy.deepcopy(doc)
    source.sectors[0].light_level = 112
    source.sectors[1].light_level = 192
    style = StyleModel()
    style.fit_document(source)
    style.wall_mids = Counter({'STARTAN2': 100000, 'STONE2': 1, 'METAL1': 1})
    style.switches = Counter({'SW1COMP': 1})
    loaded = StyleModel.from_json(style.to_json())
    assert loaded.light_levels == {112: 1, 192: 1, 160: 1}
    exits = [line.id for line in doc.linedefs if line.special == 11]
    report = apply_style(doc, loaded, seed=7, place_things=False, place_dm_starts=False)
    assert len(set(report['wall_palette'])) == 3
    assert {s.light_level for s in doc.sectors} <= set(loaded.light_levels)
    assert [line.id for line in doc.linedefs if line.special == 11] == exits
    assert not any(thing.type == 11 for thing in doc.things)
    assert doc.sidedefs[doc.linedefs[exits[0]].right_sidedef].mid_tex == 'SW1COMP'
    legacy = style.to_json()
    del legacy['light_levels']
    del legacy['door_walls']
    assert not StyleModel.from_json(legacy).light_levels


def test_columns_are_real_holes_and_preserve_navigation_and_objectives():
    doc = arena()
    report = add_columns(doc, seed=1, max_columns=2)
    assert len(report['columns']) == 2
    assert len(doc.sectors) == 3  # no unreachable closed-sector furniture
    for column in report['columns']:
        assert not point_in_rings(*column['center'], sector_points(doc, column['sector']))
    assert not require_valid(doc, require_connected=True)['errors']


def test_intro_budgets_equipment_before_combat_and_distributes_recovery():
    doc = arena()
    old = copy.deepcopy(doc)
    report = apply_intro(doc, seed=4, monster_budget=4)
    assert 1 <= report['monsters'] <= 4
    assert report['enemy_hit_points'] == sum(MONSTER_HEALTH[t.type] for t in doc.things if t.category == 'monster')
    assert report['shells_supplied_before_combat'] >= math.ceil(report['enemy_hit_points'] / 20) + 8
    assert report['shells_supplied_before_combat'] <= 50
    assert 25 <= report['health_supplied'] <= 100
    assert report['health_supplied'] == 25 * report['health_stations_planned']
    start = doc.things[0]
    for thing in doc.things:
        if thing.category == 'monster':
            assert math.dist((thing.x, thing.y), (start.x, start.y)) >= 512
    static = PlayerNavigation(doc, state={'sectors': doc.sectors, 'teleport_edges': []})
    for record in report['things']:
        if record['role'].startswith('opening_'):
            thing = doc.things[record['thing']]
            assert static.reaches_point(thing.x, thing.y)
            assert record['sector'] == 0
    assert report['objective']['sector'] == 2
    assert not require_valid(doc, require_connected=True)['errors']
    duplicate = apply_intro(old, seed=4, monster_budget=4)
    assert duplicate == report and old.to_dict() == doc.to_dict()


def test_intro_rejects_an_incompatible_game_or_existing_population_before_mutation():
    from generation.game_profile import GameProfile
    doc = arena()
    before = doc.to_dict()
    with pytest.raises(ValueError, match='stock shotgun'):
        apply_intro(doc, game=GameProfile(name='empty'))
    assert doc.to_dict() == before
    from generation.style import _add_thing
    _add_thing(doc, 512, 512, 3004)
    before = doc.to_dict()
    with pytest.raises(ValueError, match='undecorated population'):
        apply_intro(doc)
    assert doc.to_dict() == before


def test_intro_size_policy_preserves_requested_topology_floors_and_source():
    from roomgraph.types import Room, RoomGraph, Connection
    from generation.gameplay import prepare_intro_graph
    source = RoomGraph([Room(0, [], 0, 72, 4096**2, (0, 0), (0, 0, 4096, 4096)),
                        Room(1, [], 64, 136, 192**2, (0, 0), (0, 0, 192, 192))],
                       [Connection(0, 1, 'door', 128)])
    before = source.to_dict()
    result, report = prepare_intro_graph(source)
    assert source.to_dict() == before
    assert result.connections == source.connections
    assert [room.floor_h for room in result.rooms] == [0, 64]
    assert result.rooms[0].size == (1024, 1024) and result.rooms[1].size == (384, 384)
    assert all(room.ceil_h - room.floor_h == 128 for room in result.rooms)
    assert report['rooms'][0]['requested_size'] == [4096, 4096]

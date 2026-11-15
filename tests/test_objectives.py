import copy
import json

import pytest

from generation.objectives import select_intro_objective
from tests.gameplay_fixtures import keyed_branch
from tests.geometry_fixtures import polygon_document, square
from wad_parser.navigation import PlayerNavigation


def fixture():
    doc = polygon_document([[square(0, 0, 512, 512)], [square(512, 0, 1024, 512)],
                            [square(1024, 0, 1536, 512)]], start=(64, 256))
    doc.extra.update(room_sectors={0: 0, 1: 1, 2: 2}, room_roles={0: ['start'], 1: [], 2: ['exit']},
                     connection_realization=[{'a': 0, 'b': 2, 'realized': True, 'kind': 'open'},
                                             {'a': 2, 'b': 1, 'realized': True, 'kind': 'open'}])
    nav = PlayerNavigation(doc)
    walls = [line for line in doc.linedefs if line.left_sidedef == -1 and nav.reaches_line(line)]
    return doc, walls


def test_objective_ranks_requested_rooms_instead_of_physical_sector_hops():
    doc, walls = fixture()
    before = copy.deepcopy(doc.to_dict())
    line, report = select_intro_objective(doc, walls, start_sector=0, sector_distances={0: 0, 1: 1, 2: 8})
    assert doc.sidedefs[line.right_sidedef].sector == 1
    assert report['policy'] == 'semantic_room_hops_v2'
    assert report['room'] == 1 and report['distance'] == 2
    assert doc.to_dict() == before


def test_transit_sectors_cannot_win_and_json_keys_are_supported():
    doc, walls = fixture()
    doc.extra['room_sectors'][10] = 2
    doc.extra['room_roles'][10] = ['corridor']
    # The corridor owns an otherwise eligible physical sector, but it was
    # introduced by assembly and is not an endpoint in the requested graph.
    doc.extra['room_sectors'][2] = 99
    doc.extra = json.loads(json.dumps(doc.extra))
    line, report = select_intro_objective(doc, list(reversed(walls)), start_sector=0,
                                         sector_distances={0: 0, 1: 1, 2: 100})
    assert doc.sidedefs[line.right_sidedef].sector == 1
    assert report['room'] == 1


def test_semantic_ties_use_exit_role_then_stable_room_identity():
    doc, walls = fixture()
    doc.extra['connection_realization'][1] = {'a': 0, 'b': 1, 'realized': True, 'kind': 'open'}
    line, report = select_intro_objective(doc, walls, start_sector=0, sector_distances={0: 0, 1: 20, 2: 1})
    assert doc.sidedefs[line.right_sidedef].sector == 2 and report['room'] == 2
    doc.extra['room_roles'][2] = []
    line, report = select_intro_objective(doc, walls, start_sector=0, sector_distances={0: 0, 1: 1, 2: 20})
    assert doc.sidedefs[line.right_sidedef].sector == 1 and report['room'] == 1


def test_legacy_source_without_semantic_metadata_reports_explicit_fallback():
    doc, walls = fixture()
    doc.extra = {}
    line, report = select_intro_objective(doc, walls, start_sector=0, sector_distances={0: 0, 1: 1, 2: 2})
    assert doc.sidedefs[line.right_sidedef].sector == 2
    assert report['policy'] == 'legacy_sector_hops_v1'
    assert report['reason']


def test_unrealized_semantic_edges_are_not_used_as_ranking_evidence():
    doc, walls = fixture()
    doc.extra['connection_realization'][0]['realized'] = False
    _, report = select_intro_objective(doc, walls, start_sector=0, sector_distances={0: 0, 1: 1, 2: 2})
    assert report['policy'] == 'legacy_sector_hops_v1'
    assert report['reason'] == 'incomplete requested-connection realization'


def test_equal_distance_prefers_terminal_room_over_junction_before_id():
    doc = polygon_document([[square(i * 512, 0, (i + 1) * 512, 512)] for i in range(5)],
                           start=(64, 256))
    doc.extra.update(room_sectors={i: i for i in range(5)},
                     connection_realization=[{'a': a, 'b': b, 'realized': True}
                                             for a, b in [(0, 1), (1, 2), (1, 3), (2, 4), (0, 4)]])
    walls = [line for line in doc.linedefs if line.left_sidedef == -1]
    _, report = select_intro_objective(doc, walls, start_sector=0, sector_distances=dict.fromkeys(range(5), 0))
    assert report['room'] == 3 and report['distance'] == 2


@pytest.mark.parametrize('profile', ['compact', 'adjoining'])
def test_generated_branch_goal_and_key_gate_survive_connector_changes(profile):
    """Original nine-room failure, without a dependency on generated artifacts."""
    from generation.baselines import realize
    from generation.gameplay import prepare_intro_graph, apply_intro
    from generation.keyed import add_blue_key
    from roomgraph.types import Room, RoomGraph, Connection
    floors = [0, 0, -24, -24, -24, 0, 24, -24, 24]
    ceilings = [96, 72, 136, 104, 168, 72, 280, 168, 152]
    dimensions = [(332.8, 179.2)] * 3 + [(256, 256), (332.8, 179.2), (256, 256),
                                                   (179.2, 332.8), (256, 256), (179.2, 332.8)]
    rooms = []
    for rid, (w, h) in enumerate(dimensions):
        role = 'start' if rid == 0 else 'exit' if rid == 8 else 'room'
        rooms.append(Room(rid, [], floors[rid], ceilings[rid], w * h, (0, 0), (0, 0, w, h),
                          role=role, roles=[role] if role != 'room' else []))
    edges = [Connection(a, b, kind, width) for a, b, kind, width in
             [(0, 1, 'corridor', 64), (0, 2, 'corridor', 64), (2, 3, 'door', 64),
              (2, 4, 'corridor', 64), (3, 4, 'corridor', 64), (0, 5, 'corridor', 64),
              (5, 6, 'corridor', 128), (3, 7, 'corridor', 64), (0, 7, 'corridor', 128), (6, 8, 'door', 128)]]
    graph, _ = prepare_intro_graph(RoomGraph(rooms, edges))
    doc = realize(graph, geometry_profile=profile, seed=1)
    report = apply_intro(doc, seed=1, monster_budget=2, interior_details=False)
    assert report['objective']['selection']['room'] == 8
    key = add_blue_key(doc, seed=1)
    assert key['exit_reachable_without_key'] is False


@pytest.mark.parametrize('profile', ['compact', 'adjoining'])
def test_requested_key_goal_preflight_finds_gate_beyond_the_default_open_leaf(profile):
    from generation.gameplay import apply_intro
    from generation.keyed import add_blue_key
    ordinary = keyed_branch(profile)
    normal = apply_intro(ordinary, seed=3, monster_budget=2, interior_details=False)
    assert normal['objective']['selection']['room'] == 2
    with pytest.raises(ValueError, match='exit bypasses'):
        add_blue_key(ordinary, seed=3)
    doc = keyed_branch(profile)
    report = apply_intro(doc, seed=3, monster_budget=2, interior_details=False, required_progression='blue-key')
    selection = report['objective']['selection']
    assert selection['room'] == 3 and selection['required_progression'] == 'blue-key'
    assert selection['key_preflight']['goals_attempted'] == 2
    assert selection['key_preflight']['rejected_goals'][0]['sector'] == 2
    key = add_blue_key(doc, seed=3)
    assert not key['exit_reachable_without_key']
    assert key['validation']['status'] == 'success'
    assert report['progression'] is doc.extra['key_progression']


def test_keyed_preflight_is_nonmutating_and_its_budget_is_explicit():
    from generation.objectives import select_keyed_intro_objective
    from wad_parser.traversal import distances_from_start
    doc = keyed_branch()
    nav = PlayerNavigation(doc)
    walls = [line for line in doc.linedefs if line.left_sidedef == -1 and nav.reaches_line(line)]
    before = copy.deepcopy(doc.to_dict())
    with pytest.raises(ValueError, match='budget exhausted'):
        select_keyed_intro_objective(doc, walls, start_sector=0, sector_distances=distances_from_start(doc),
                                     max_goals=1)
    assert doc.to_dict() == before
    _, selection = select_keyed_intro_objective(doc, walls, start_sector=0,
                                                sector_distances=distances_from_start(doc))
    assert selection['room'] == 3
    assert doc.to_dict() == before


def test_no_gate_does_not_invent_connections_or_change_the_input():
    from generation.objectives import select_keyed_intro_objective
    from wad_parser.traversal import distances_from_start
    doc = keyed_branch(door=False)
    walls = [line for line in doc.linedefs if line.left_sidedef == -1]
    before = copy.deepcopy(doc.to_dict())
    with pytest.raises(ValueError, match='no supported goal'):
        select_keyed_intro_objective(doc, walls, start_sector=0, sector_distances=distances_from_start(doc))
    assert doc.to_dict() == before


def test_unknown_required_progression_fails_before_intro_mutation():
    from generation.gameplay import apply_intro
    doc = keyed_branch()
    before = copy.deepcopy(doc.to_dict())
    with pytest.raises(ValueError, match='required_progression'):
        apply_intro(doc, required_progression='unknown')
    assert doc.to_dict() == before

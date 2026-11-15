"""Optional real-engine regressions; run with DOOM_GEN_ENGINE_TESTS=1.

Install the engine extra and provide a display (xvfb-run on headless Linux).
Freedoom II supplied by ViZDoom avoids a dependency on a proprietary IWAD.
"""
import math
import os
from pathlib import Path

import pytest

from tests.engine_fixtures import require_clean_native_shutdown, walk_to, walking_game  # noqa: F401 - autouse fixture

pytestmark = pytest.mark.skipif(os.environ.get('DOOM_GEN_ENGINE_TESTS') != '1',
                                reason='opt-in Doom engine interaction tests')


def _document(kind, difference=0, side=512):
    from roomgraph.types import Room, RoomGraph, Connection
    from roomgraph.assemble import assemble
    rooms = [Room(i, [], difference * i, difference * i + 128,
                  side**2, (0, 0), (0, 0, side, side)) for i in range(2)]
    return assemble(RoomGraph(rooms, [Connection(0, 1, kind, 192)]))


@pytest.mark.parametrize('kind,difference', [('open', 128), ('door', 0), ('door', 128), ('door', 384)])
def test_engine_walks_stairs_and_opens_closed_doors(tmp_path, kind, difference):
    from wad_parser.geometry import interior_point
    from wad_writer.wad_writer import write_wad
    doc = _document(kind, difference)
    path = tmp_path / 'interaction.wad'
    write_wad([doc], path)
    game = walking_game(path)
    try:
        destination = interior_point(doc, doc.extra['room_sectors'][1], clearance=32)
        assert walk_to(game, destination), game.get_state().game_variables
        assert game.get_state().game_variables[2] == pytest.approx(difference)
        # Doors must also be operable from the other side.
        assert walk_to(game, (doc.things[0].x, doc.things[0].y))
    finally:
        game.close()


@pytest.mark.parametrize('side', [192, 256, 512])
def test_engine_teleports_in_both_directions(tmp_path, side):
    from wad_parser.geometry import interior_point
    from wad_writer.wad_writer import write_wad
    doc = _document('teleport', side=side)
    path = tmp_path / 'teleports.wad'
    write_wad([doc], path)
    game = walking_game(path)
    try:
        first, second = [thing for thing in doc.things if thing.type == 14]
        assert walk_to(game, (first.x, first.y), arrival=(second.x, second.y), max_tics=150)
        # Leave the arrival pad and approach it again, causing the return trip.
        departure = interior_point(doc, doc.extra['room_sectors'][1], clearance=16)
        assert walk_to(game, departure, max_tics=150)
        assert walk_to(game, (second.x, second.y), arrival=(first.x, first.y), max_tics=150)
    finally:
        game.close()


def test_engine_teleport_destination_must_spawn_on_the_current_skill(tmp_path):
    from wad_writer.wad_writer import write_wad
    from wad_parser.traversal import distances_from_start
    doc = _document('teleport')
    first, second = [thing for thing in doc.things if thing.type == 14]
    second.flags.skill_3 = False
    assert doc.extra['room_sectors'][1] not in distances_from_start(doc)
    path = tmp_path / 'inactive-teleport.wad'
    write_wad([doc], path)
    game = walking_game(path)
    try:
        assert not walk_to(game, (first.x, first.y), arrival=(second.x, second.y), max_tics=200)
        assert math.dist(game.get_state().game_variables[:2], (second.x, second.y)) > 256
    finally:
        game.close()


def test_engine_walks_both_repaired_teleport_directions(tmp_path):
    from generation.pipeline import documents_to_wad
    from generation.progression import ensure_exit
    from generation.validator import validate_and_repair
    from tests.geometry_fixtures import repair_arena
    from wad_parser.geometry import sector_geometry
    from wad_parser.map_extractor import _sector_containing
    doc = repair_arena()
    repairs = validate_and_repair(doc, reconnect=True)
    assert any('teleporter pairs' in repair for repair in repairs)
    ensure_exit(doc)
    path = tmp_path / 'repaired-teleports.wad'
    documents_to_wad([doc], path, require_playable=True)
    game = walking_game(path)
    try:
        first, second = [t for t in doc.things if t.type == 14]
        assert walk_to(game, (first.x, first.y), arrival=(second.x, second.y), max_tics=1000)
        # The original passage into the far room must still work.
        assert walk_to(game, (1408, 128), max_tics=1000)
        assert walk_to(game, (second.x, second.y), arrival=(first.x, first.y), max_tics=1000)
        assert walk_to(game, (384, 128), max_tics=1000)
        assert _sector_containing(*game.get_state().game_variables[:2], doc.sectors,
                                  sector_geometry(doc), doc.vertices) == 1
        assert not game.is_player_dead()
    finally:
        game.close()


def test_engine_activates_generated_exit(tmp_path):
    from tests.geometry_fixtures import two_rooms
    from generation.progression import ensure_exit
    from wad_writer.wad_writer import write_wad
    doc = two_rooms()
    ensure_exit(doc)
    exit_line = next(line for line in doc.linedefs if line.special == 11)
    a, b = doc.vertices[exit_line.v1], doc.vertices[exit_line.v2]
    target = ((a.x + b.x) / 2, (a.y + b.y) / 2)
    path = tmp_path / 'exit.wad'
    write_wad([doc], path)
    game = walking_game(path)
    try:
        walk_to(game, target, tolerance=0, max_tics=500)
        assert game.is_episode_finished()
        assert not game.is_player_dead()
    finally:
        game.close()


def test_engine_spawns_and_completes_from_an_open_sector_boundary(tmp_path):
    from tests.geometry_fixtures import boundary_start
    from generation.progression import ensure_exit
    from tests.engine_navigation import navigate_exit
    from wad_writer.wad_writer import write_wad
    doc = boundary_start()
    ensure_exit(doc)
    path = tmp_path / 'boundary-start.wad'
    write_wad([doc], path)
    game = walking_game(path)
    try:
        result = navigate_exit(game, doc, walk_to)
        assert result['completed'], result
        assert not game.is_player_dead()
    finally:
        game.close()


@pytest.mark.parametrize('geometry_profile', ['classic', 'varied'])
def test_engine_completes_a_wheel_with_a_split_hub_and_cycle(tmp_path, geometry_profile):
    from tests.geometry_fixtures import wheel_graph
    from tests.engine_navigation import navigate_exit
    from generation.progression import ensure_exit
    from roomgraph.assemble import assemble
    from roomgraph.teleports import require_connections
    from wad_writer.wad_writer import write_wad
    doc = assemble(wheel_graph(), geometry_profile=geometry_profile, seed=3)
    require_connections(doc)
    ensure_exit(doc)
    path = tmp_path / 'wheel.wad'
    write_wad([doc], path)
    game = walking_game(path, timeout=60000)
    try:
        result = navigate_exit(game, doc, walk_to)
        assert result['completed'], result
        assert game.is_episode_finished() and not game.is_player_dead()
    finally:
        game.close()


@pytest.mark.parametrize('kind,difference,side', [
    ('open', 128, 512), ('door', 0, 512), ('door', 384, 512),
    ('door', 128, 192), ('teleport', 128, 192),
])
def test_engine_completes_varied_rooms_with_stairs_doors_and_teleports(tmp_path, kind, difference, side):
    from generation.baselines import realize
    from roomgraph.types import Room, RoomGraph, Connection
    from tests.engine_navigation import navigate_exit
    from wad_writer.wad_writer import write_wad
    rooms = [Room(i, [], i * difference, i * difference + 128, side**2,
                  (0, 0), (0, 0, side, side), roles=['start'] if i == 0 else ['exit'])
             for i in range(2)]
    graph = RoomGraph(rooms, [Connection(0, 1, kind, 160)])
    doc = realize(graph, geometry_profile='varied', seed=19)
    path = tmp_path / 'varied-interactions.wad'
    write_wad([doc], path)
    game = walking_game(path)
    try:
        result = navigate_exit(game, doc, walk_to)
        assert result['completed'], result
        assert game.is_episode_finished() and not game.is_player_dead()
    finally:
        game.close()


def test_engine_player_leaves_a_tiny_decorative_sector(tmp_path):
    from tests.geometry_fixtures import decorative_start
    from wad_writer.wad_writer import write_wad
    path = tmp_path / 'decorative.wad'
    write_wad([decorative_start()], path)
    game = walking_game(path)
    try:
        assert walk_to(game, (400, 256))
        assert not game.is_player_dead()
    finally:
        game.close()


@pytest.mark.parametrize('key_x', [4, 32])
def test_engine_key_is_required_before_a_locked_door_opens(tmp_path, key_x):
    from tests.progression_fixtures import door_document, add_key
    from wad_writer.wad_writer import write_wad
    doc = door_document()
    add_key(doc, x=key_x)
    doc.things[-1].y = 104
    path = tmp_path / 'locked.wad'
    write_wad([doc], path)
    game = walking_game(path)
    try:
        assert not walk_to(game, (224, 64), max_tics=150)
        assert walk_to(game, (max(20, key_x), 104), tolerance=12)
        assert walk_to(game, (224, 64))
    finally:
        game.close()


def test_engine_remote_switch_opens_the_tagged_door(tmp_path):
    from tests.progression_fixtures import door_document
    from wad_writer.wad_writer import write_wad
    doc = door_document(0)
    doc.sectors[1].tag = 7
    switch = next(line for line in doc.linedefs if line.left_sidedef == -1
                  and doc.vertices[line.v1].x == doc.vertices[line.v2].x == 0)
    switch.special, switch.tag = 103, 7
    path = tmp_path / 'remote.wad'
    write_wad([doc], path)
    game = walking_game(path)
    try:
        assert not walk_to(game, (224, 64), max_tics=150)
        assert walk_to(game, (32, 64), tolerance=8)
        game.make_action([0, 0, 1], 1)
        game.make_action([0, 0, 0], 70)
        assert walk_to(game, (224, 64))
    finally:
        game.close()


@pytest.mark.parametrize('special,can_revisit', [(39, False), (97, True)])
def test_engine_one_shot_teleport_cannot_be_reused_to_finish_a_key_history(tmp_path, special, can_revisit):
    from tests.progression_fixtures import one_shot_revisit_puzzle
    from wad_parser.progression import search_exit_history
    from wad_writer.wad_writer import write_wad
    doc = one_shot_revisit_puzzle(special)
    assert search_exit_history(doc)['status'] == ('success' if can_revisit else 'failure')
    path = tmp_path / 'teleport-key-history.wad'
    write_wad([doc], path)
    game = walking_game(path)
    try:
        assert walk_to(game, (128, 64), arrival=(576, 64))
        assert walk_to(game, (576, 128), arrival=(64, 64))
        assert walk_to(game, (64, 224))  # yellow door, then blue key
        assert walk_to(game, (64, 64))
        revisited = walk_to(game, (128, 64), arrival=(576, 64), max_tics=300)
        assert revisited is can_revisit
        if revisited:
            assert walk_to(game, (736, 64))  # blue door, then red key
            assert walk_to(game, (576, 128), arrival=(64, 64))
            walk_to(game, (-160, 64), tolerance=0, max_tics=600)
            assert game.is_episode_finished() and not game.is_player_dead()
        else:
            assert not game.is_episode_finished() and not game.is_player_dead()
    finally:
        game.close()


def test_engine_consumed_teleport_becomes_an_ordinary_passage_after_return(tmp_path):
    from tests.progression_fixtures import consumption_walk_puzzle
    from wad_parser.progression import search_exit_history
    from wad_writer.wad_writer import write_wad
    doc = consumption_walk_puzzle()
    assert search_exit_history(doc)['status'] == 'success'
    path = tmp_path / 'consumed-teleport-walk.wad'
    write_wad([doc], path)
    game = walking_game(path)
    try:
        assert walk_to(game, (128, 64), arrival=(576, 64))
        assert walk_to(game, (576, 128), arrival=(64, 64))
        line = next(line for line in doc.linedefs if line.special == 11)
        a, b = doc.vertices[line.v1], doc.vertices[line.v2]
        walk_to(game, ((a.x + b.x) / 2, (a.y + b.y) / 2), tolerance=0, max_tics=600)
        assert game.is_episode_finished() and not game.is_player_dead()
    finally:
        game.close()


def test_engine_uses_a_switch_shorter_than_the_player_box(tmp_path):
    from tests.progression_fixtures import short_switch_document
    from wad_writer.wad_writer import write_wad
    path = tmp_path / 'short-switch.wad'
    write_wad([short_switch_document()], path)
    game = walking_game(path)
    try:
        assert not walk_to(game, (224, 64), max_tics=150)
        assert walk_to(game, (32, 64), tolerance=8)
        game.make_action([0, 0, 1], 1)
        game.make_action([0, 0, 0], 70)
        assert walk_to(game, (224, 64))
    finally:
        game.close()


@pytest.mark.parametrize('return_from_key', [False, True])
def test_engine_key_branch_requires_a_return_route(tmp_path, return_from_key):
    from tests.progression_fixtures import branch_document
    from wad_parser.progression import search_exit_history
    from wad_writer.wad_writer import write_wad
    doc = branch_document(return_from_key=return_from_key)
    assert (search_exit_history(doc)['status'] == 'success') == return_from_key
    path = tmp_path / 'key-branch.wad'
    write_wad([doc], path)
    game = walking_game(path)
    try:
        assert walk_to(game, (32, 192), tolerance=12)
        if return_from_key:
            assert walk_to(game, (320, 192))
            assert walk_to(game, (480, 192))
            walk_to(game, (544, 192), tolerance=0, max_tics=500)
            assert game.is_episode_finished() and not game.is_player_dead()
        else:
            assert not walk_to(game, (192, 192), max_tics=200)
            assert game.get_state().game_variables[0] < 128
    finally:
        game.close()


def test_engine_touches_the_same_key_from_the_branch_that_reaches_the_exit(tmp_path):
    from tests.progression_fixtures import shared_key_branches
    from wad_writer.wad_writer import write_wad
    path = tmp_path / 'shared-key.wad'
    write_wad([shared_key_branches()], path)
    game = walking_game(path)
    try:
        assert walk_to(game, (208, 192))
        assert walk_to(game, (161, 64), tolerance=2)
        assert walk_to(game, (368, 64))
        walk_to(game, (432, 64), tolerance=0, max_tics=500)
        assert game.is_episode_finished() and not game.is_player_dead()
    finally:
        game.close()


def test_navigation_budget_is_not_mistaken_for_exit_activation(tmp_path):
    from tests.engine_navigation import navigate_exit
    from generation.progression import ensure_exit
    from wad_writer.wad_writer import write_wad
    doc = _document('open')
    ensure_exit(doc)
    path = tmp_path / 'timeout.wad'
    write_wad([doc], path)
    game = walking_game(path, timeout=4)
    try:
        def wait_for_timeout(clock, target, **kwargs):
            clock.make_action([0, 0, 0], 100)
            return False
        result = navigate_exit(game, doc, wait_for_timeout, max_tics=100)
        assert game.is_episode_finished() and not game.is_player_dead()
        assert not result['completed']
        assert result['tics'] >= 100
    finally:
        game.close()


@pytest.mark.skipif(not os.environ.get('DOOM_GEN_ENGINE_PANEL'), reason='no generated panel selected')
def test_generated_panel_can_reach_its_exits():
    import json
    from tests.engine_navigation import navigate_exit
    from wad_parser.wad_reader import WadFile
    from wad_parser.map_extractor import extract_all_maps
    from wad_writer.compatibility import require_target_port
    path = Path(os.environ['DOOM_GEN_ENGINE_PANEL']).resolve()
    wad = WadFile(str(path))
    require_target_port(wad, 'zdoom')
    docs = extract_all_maps(wad)
    results = []
    # Leave headroom above the controller budget so engine timeout cannot be
    # mistaken for exit activation, even if its own post-level clock resets.
    game = walking_game(path, timeout=60000)
    try:
        for doc in docs:
            game.set_doom_map(doc.meta.map_name)
            game.new_episode()
            results.append({'map': doc.meta.map_name, **navigate_exit(game, doc, walk_to)})
    finally:
        game.close()
    if os.environ.get('DOOM_GEN_ENGINE_REPORT'):
        report_path = Path(os.environ['DOOM_GEN_ENGINE_REPORT'])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    assert results and all(result['completed'] for result in results), results


def test_engine_loads_stock_and_custom_assets_and_activates_exit(tmp_path):
    engine = pytest.importorskip('vizdoom')
    from tests.pipeline_fixtures import asset_fixture
    from generation.pipeline import documents_to_wad
    from tests.engine_navigation import navigate_exit
    doc, assets = asset_fixture(Path(engine.__file__).parent / 'freedoom2.wad')
    path = tmp_path / 'assets.wad'
    documents_to_wad([doc], path, assets=assets, require_playable=True)
    game = walking_game(path)
    try:
        result = navigate_exit(game, doc, walk_to)
        assert result['completed'], result
        assert not game.is_player_dead()
    finally:
        game.close()

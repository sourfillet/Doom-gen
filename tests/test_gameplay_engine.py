"""Ordinary pickup/combat/exit checks for the opt-in introductory planner."""
import os

import pytest

from tests.engine_fixtures import require_clean_native_shutdown  # noqa: F401 - autouse fixture

pytestmark = pytest.mark.skipif(os.environ.get('DOOM_GEN_ENGINE_TESTS') != '1',
                                reason='opt-in Doom combat interaction tests')


@pytest.mark.parametrize('profile', ['compact', 'adjoining'])
def test_requested_key_goal_preflight_collects_key_clears_both_branches_and_exits(tmp_path, profile):
    pytest.importorskip('vizdoom')
    from tests.gameplay_fixtures import keyed_branch
    from generation.gameplay import apply_intro
    from generation.keyed import add_blue_key
    from generation.pipeline import documents_to_wad
    from tests.engine_combat import open_game, CombatController
    from wad_parser.map_extractor import extract_all_maps
    from wad_parser.wad_reader import WadFile
    doc = keyed_branch(profile)
    plan = apply_intro(doc, seed=3, monster_budget=4, interior_details=False, required_progression='blue-key')
    assert plan['objective']['selection']['room'] == 3
    add_blue_key(doc, seed=3)
    path = tmp_path / 'joint-key-goal.wad'
    documents_to_wad([doc], path, require_playable=True)
    compiled, = extract_all_maps(WadFile(str(path)))
    game = open_game(path, seed=3)
    try:
        result = CombatController(game, compiled).complete(plan)
    finally:
        game.close()
    assert result['completed'], result
    assert result['kills'] == 4
    assert result['progression']['key_collected'] and result['progression']['gate_crossed']


@pytest.mark.parametrize('connection', ['open', 'door'])
def test_intro_player_collects_equipment_fights_and_exits(tmp_path, connection):
    pytest.importorskip('vizdoom')
    from generation.gameplay import apply_intro
    from generation.progression import ensure_exit
    from generation.validator import require_valid
    from roomgraph.assemble import assemble
    from roomgraph.types import Room, RoomGraph, Connection
    from tests.engine_combat import open_game, CombatController
    from wad_writer.wad_writer import write_wad
    rooms = [Room(i, [], 0, 128, 512**2, (0, 0), (0, 0, 512, 512)) for i in range(2)]
    doc = assemble(RoomGraph(rooms, [Connection(0, 1, connection, 192)]), geometry_profile='varied', seed=2)
    ensure_exit(doc)
    plan = apply_intro(doc, seed=3, monster_budget=2)
    assert plan['monsters'] == 2
    require_valid(doc, require_connected=True)
    path = write_wad([doc], tmp_path / 'intro.wad')
    game = open_game(path, seed=3)
    try:
        result = CombatController(game, doc).complete(plan)
    finally:
        game.close()
    assert result['completed'], result
    assert result['kills'] == 2 and result['attack_tics'] > 0
    assert result['opening']['weapon_owned'] == 1
    assert result['opening']['shells'] == plan['shells_supplied_before_combat']
    assert result['opening']['attack_tics'] == 0
    assert result['opening']['armor'] == 100


@pytest.mark.skipif(not os.environ.get('DOOM_GEN_GAMEPLAY_DEMO'), reason='no populated demo selected')
def test_generated_gameplay_panel_collects_supplies_clears_combat_and_exits(tmp_path):
    from tools.playtest_gameplay import playtest
    report = playtest(os.environ['DOOM_GEN_GAMEPLAY_DEMO'], tmp_path / 'playthrough')
    assert report['status'] == 'complete' and not report['code_changed_during_playtest']
    assert report['records'] and all(row['completed'] and not row['native_crash']
        and row['kills'] == row['planned_monsters'] and row['opening']['attack_tics'] == 0
        for row in report['records']), report['records']


def test_demo_preserves_original_map_for_before_frame_and_snapshots_from_another_cwd(tmp_path, monkeypatch):
    pytest.importorskip('vizdoom')
    from generation.game_profile import GameProfile
    from generation.style import StyleModel
    from tests.gameplay_fixtures import arena
    from wad_writer.wad_writer import write_wad
    from tools.build_gameplay_demo import build
    from tools.playtest_gameplay import playtest
    import tests.engine_combat as combat
    doc = arena()
    doc.meta.map_name = 'MAP17'
    source = write_wad([doc], tmp_path / 'source.wad')
    style = StyleModel()
    style.fit_document(doc)
    style_path = tmp_path / 'style.json'
    style.save(style_path)
    game = GameProfile(name='fixture', textures={'STARTAN2', 'SW1COMP'},
                       flats={'FLOOR4_8', 'CEIL3_5'},
                       thing_types={3004, 9, 3001, 2001, 2008, 2012, 2018})
    game_path = tmp_path / 'game.json'
    game.save(game_path)
    # The native crash guard already moved cwd away from the checkout.
    demo = tmp_path / 'demo'
    result = build(source, style_path, game_path, demo, seed=3)
    assert result['records'][0]['source_map'] == 'MAP17'
    assert result['records'][0]['map'] == 'MAP01'
    assert (demo / 'library/generation/gameplay.py').exists()
    seen = []
    original = combat.open_game
    def track(path, map_name, **kwargs):
        seen.append(map_name)
        return original(path, map_name, **kwargs)
    monkeypatch.setattr(combat, 'open_game', track)
    report = playtest(demo, tmp_path / 'engine', seed=3)
    assert seen == ['MAP17', 'MAP01']
    assert report['records'][0]['completed'], report['records']
    assert not report['code_changed_during_playtest']
    assert set(report['helpers_sha256']) == {'engine_combat.py', 'engine_navigation.py'}
    assert (tmp_path / 'engine/engine_navigation.py').exists()


def test_intro_collects_blue_key_crosses_required_gate_and_exits(tmp_path):
    pytest.importorskip('vizdoom')
    from tests.gameplay_fixtures import keyed_arena
    from generation.gameplay import apply_intro
    from generation.keyed import add_blue_key
    from generation.pipeline import documents_to_wad
    from tests.engine_combat import open_game, CombatController
    from wad_parser.map_extractor import extract_all_maps
    from wad_parser.wad_reader import WadFile
    doc = keyed_arena()
    plan = apply_intro(doc, seed=3, monster_budget=4)
    key = add_blue_key(doc, seed=3)
    path = tmp_path / 'keyed.wad'
    documents_to_wad([doc], path, require_playable=True)
    compiled, = extract_all_maps(WadFile(str(path)))
    game = open_game(path, seed=3)
    try:
        result = CombatController(game, compiled).complete(plan)
    finally:
        game.close()
    assert result['completed'], result
    assert result['progression']['key_collected'] and result['progression']['gate_crossed']
    assert result['kills'] == plan['monsters']
    assert key['validation']['status'] == 'success'


def test_terrace_tour_skips_viewpoint_inside_room_but_too_close_to_solid_wall(tmp_path):
    pytest.importorskip('vizdoom')
    from generation.gameplay import apply_intro
    from generation.pipeline import documents_to_wad
    from generation.progression import ensure_exit
    from tests.engine_combat import CombatController, open_game
    from tests.geometry_fixtures import polygon_document, square
    from wad_parser.map_extractor import extract_all_maps
    from wad_parser.wad_reader import WadFile
    doc = polygon_document([[square(0, 0, 512, 544)],
                            [square(512, 0, 1536, 544)],
                            [square(1536, 0, 1920, 544)]], start=(96, 272))
    ensure_exit(doc)
    plan = apply_intro(doc, seed=4, monster_budget=2, interior_profile='stepped')
    terrace, = plan['interiors']['terraces']
    path = tmp_path / 'viewpoint.wad'
    documents_to_wad([doc], path, require_playable=True)
    compiled, = extract_all_maps(WadFile(str(path)))
    game = open_game(path, seed=7)
    try:
        controller = CombatController(game, compiled)
        cx, cy = terrace['center']
        first_viewpoint = (cx, cy - terrace['steps'][0]['width'] / 2 - 80)
        assert controller.locator.owner(*first_viewpoint) == terrace['base_sector']
        assert controller.clearance.problem(terrace['base_sector'], *first_viewpoint, radius=20)
        result = controller.complete(plan)
    finally:
        game.close()
    assert result['completed'], result
    assert result['kills'] == 2
    assert len(result['interior_walks']) == 1


def test_blue_gate_stays_closed_without_key_under_actual_use_inputs(tmp_path):
    pytest.importorskip('vizdoom')
    from tests.gameplay_fixtures import keyed_arena
    from generation.keyed import add_blue_key
    from generation.pipeline import documents_to_wad
    from tests.engine_combat import open_game, CombatController
    from wad_parser.navigation import PlayerNavigation
    from wad_parser.geometry import sector_points
    from wad_parser.bsp_geometry import component_polygons
    doc = keyed_arena()
    plan = add_blue_key(doc, seed=3)
    doc.things.pop(plan['key_thing'])
    # Deliberately publish an unreachable diagnostic fixture, never a generated
    # playable result: its missing key is the engine counterfactual.
    path = tmp_path / 'without-key.wad'
    documents_to_wad([doc], path, require_playable=False)
    navigation = PlayerNavigation(doc)
    approach = next((navigation.line_approach(doc.linedefs[i]) for i in plan['door_lines']
                     if navigation.line_approach(doc.linedefs[i]) is not None), None)
    assert approach is not None
    polygon = max((p for _, p in component_polygons(sector_points(doc, plan['door_sector']))), key=lambda p: p.area)
    center = polygon.representative_point()
    game = open_game(path, seed=3, sectors_info=True)
    try:
        controller = CombatController(game, doc)
        assert controller.goto(approach, tolerance=4)
        assert not controller.walk(controller.game, (center.x, center.y), tolerance=4, max_tics=300)
        sector = game.get_state().sectors[plan['door_sector']]
        assert sector.ceiling_height == pytest.approx(sector.floor_height)
        assert not game.is_episode_finished() and not game.is_player_dead()
    finally:
        game.close()

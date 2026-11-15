"""Ordinary-input checks on decoded shared-wall spatial geometry."""
import json
import os

import pytest

from tests.engine_fixtures import require_clean_native_shutdown, walking_game, walk_to  # noqa: F401 - autouse fixture

pytestmark = pytest.mark.skipif(os.environ.get('DOOM_GEN_ENGINE_TESTS') != '1',
                                reason='opt-in Doom engine interaction tests')


def _publish(doc, path):
    from data.spatial_targets import build_spatial_target
    from roomgraph.spatial import reconstruct_spatial_target
    from generation.pipeline import documents_to_wad
    from wad_parser.map_extractor import extract_all_maps
    from wad_parser.wad_reader import WadFile
    target = json.loads(json.dumps(build_spatial_target(doc)))
    reconstructed = reconstruct_spatial_target(target)
    result = documents_to_wad([reconstructed], path, require_playable=True)
    if os.environ.get('DOOM_GEN_PORT_TESTS') == '1':
        assert result['nodes_built']
    compiled, = extract_all_maps(WadFile(str(path)))
    assert not compiled.derived.geometry_issues
    return compiled


def test_engine_walks_decoded_irregular_courtyard(tmp_path):
    pytest.importorskip('vizdoom')
    from tests.geometry_fixtures import polygon_document, square
    outer = [(0, 0), (0, 384), (192, 384), (192, 256), (384, 256), (384, 0)]
    doc = polygon_document([[outer, list(reversed(square(64, 64, 128, 128)))]], start=(48, 192))
    next(line for line in doc.linedefs
         if doc.vertices[line.v1].y == doc.vertices[line.v2].y == 256).special = 11
    path = tmp_path / 'courtyard.wad'
    _publish(doc, path)
    game = walking_game(path)
    try:
        for point in ((48, 320), (144, 320), (144, 192), (320, 192), (320, 32), (48, 32), (48, 192)):
            assert walk_to(game, point, tolerance=12), game.get_state()
        assert not game.is_player_dead()
    finally:
        game.close()


def test_engine_collects_key_and_crosses_decoded_shared_door_and_step(tmp_path):
    pytest.importorskip('vizdoom')
    from tests.geometry_fixtures import polygon_document, square
    from wad_parser.types import Thing, ThingFlags
    doc = polygon_document([[square(0, 0, 256, 192)], [square(256, 0, 288, 192)],
                            [square(288, 0, 544, 192)]], start=(64, 96))
    doc.sectors[1].ceil_height = 0
    doc.sectors[2].floor_height = 24
    for line in doc.linedefs:
        if line.left_sidedef >= 0:
            if doc.sidedefs[line.right_sidedef].sector == 1:
                line.v1, line.v2 = line.v2, line.v1
                line.right_sidedef, line.left_sidedef = line.left_sidedef, line.right_sidedef
            line.special = 26
    next(line for line in doc.linedefs if line.left_sidedef == -1
         and doc.sidedefs[line.right_sidedef].sector == 2
         and doc.vertices[line.v1].y == doc.vertices[line.v2].y == 0).special = 11
    doc.things.append(Thing(1, 160, 96, 0, 5, 'BlueCard', 'key', ThingFlags(skill_3=True)))
    path = tmp_path / 'shared-door.wad'
    _publish(doc, path)
    game = walking_game(path)
    try:
        assert walk_to(game, (160, 96))
        assert walk_to(game, (448, 96)), game.get_state()
        assert game.get_state().game_variables[2] == pytest.approx(24)
        game.make_action([0, 0, 0], 320)  # Let the door close before returning.
        assert walk_to(game, (64, 96)), game.get_state()
        assert game.get_state().game_variables[2] == pytest.approx(0)
        assert not game.is_player_dead()
    finally:
        game.close()

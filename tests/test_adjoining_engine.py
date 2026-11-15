"""Actual walking/door use through the short adjoining-room thresholds."""
import os

import pytest

from tests.engine_fixtures import require_clean_native_shutdown, walking_game, walk_to  # noqa: F401 - autouse fixture

pytestmark = pytest.mark.skipif(os.environ.get('DOOM_GEN_ENGINE_TESTS') != '1',
                                reason='opt-in Doom engine interaction tests')


@pytest.mark.parametrize('kind,rise', [('open', 0), ('open', 24), ('corridor', -24), ('door', 0)])
def test_engine_crosses_adjoining_rooms_in_both_directions(tmp_path, kind, rise):
    pytest.importorskip('vizdoom')
    from generation.baselines import realize
    from generation.pipeline import documents_to_wad
    from tests.geometry_fixtures import room_pair
    from wad_parser.geometry import interior_point
    doc = realize(room_pair(kind, rise, 512), geometry_profile='adjoining', seed=3)
    path = tmp_path / 'adjoining.wad'
    documents_to_wad([doc], path, require_playable=True)
    game = walking_game(path)
    try:
        assert walk_to(game, interior_point(doc, 1, clearance=32)), game.get_state().game_variables
        assert game.get_state().game_variables[2] == pytest.approx(rise)
        assert walk_to(game, (doc.things[0].x, doc.things[0].y)), game.get_state().game_variables
        assert game.get_state().game_variables[2] == pytest.approx(0)
        assert not game.is_player_dead()
    finally:
        game.close()

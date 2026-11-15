"""Actual vanilla-style walking on the optional interior terrace sectors."""
import os

import pytest

from tests.engine_fixtures import require_clean_native_shutdown  # noqa: F401 - autouse fixture

pytestmark = pytest.mark.skipif(os.environ.get('DOOM_GEN_ENGINE_TESTS') != '1',
                                reason='opt-in Doom interior interaction tests')


@pytest.mark.parametrize('side,rise', [(512, 32), (1024, 64)])
def test_engine_walks_up_and_down_compiled_terraces(tmp_path, side, rise):
    pytest.importorskip('vizdoom')
    from generation.interiors import add_terraces
    from generation.pipeline import documents_to_wad
    from tests.gameplay_fixtures import terrace_arena
    from tests.engine_fixtures import walking_game, walk_to
    doc = terrace_arena(side)
    result = add_terraces(doc, seed=3, max_terraces=1)
    terrace, = result['terraces']
    path = tmp_path / 'terrace.wad'
    documents_to_wad([doc], path, require_playable=True)
    game = walking_game(path)
    try:
        assert walk_to(game, terrace['center'], tolerance=12, max_tics=5000)
        assert game.get_state().game_variables[2] == pytest.approx(rise)
        start = next(t for t in doc.things if t.type == 1)
        assert walk_to(game, (start.x, start.y), tolerance=12, max_tics=5000)
        assert game.get_state().game_variables[2] == pytest.approx(0)
        assert not game.is_player_dead()
    finally:
        game.close()

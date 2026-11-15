"""ViZDoom launch, steering and native-crash checks shared by opt-in engine tests."""
import math
import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def require_clean_native_shutdown(tmp_path, monkeypatch):
    # The engine can crash on close while its Python client returns success.
    # Isolate native crash logs, and keep relative panel/report paths anchored
    # to the caller's original working directory.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    for name in ('DOOM_GEN_ENGINE_PANEL', 'DOOM_GEN_ENGINE_REPORT', 'DOOM_GEN_GAMEPLAY_DEMO'):
        if os.environ.get(name):
            monkeypatch.setenv(name, str(Path(os.environ[name]).resolve()))
    monkeypatch.chdir(tmp_path)
    yield
    crash = tmp_path / 'vizdoom-crash.log'
    assert not crash.exists(), f'native Doom backend crashed:\n{crash.read_text(errors="replace")}'


def walking_game(path, timeout=10000):
    v = pytest.importorskip('vizdoom')
    game = v.DoomGame()
    game.set_doom_config_path(str(path.with_suffix('.ini')))
    game.set_doom_game_path(str(Path(v.__file__).parent / 'freedoom2.wad'))
    game.set_doom_skill(3)
    game.set_doom_scenario_path(str(path))
    game.set_doom_map('MAP01')
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_episode_timeout(timeout)
    game.set_available_buttons([v.Button.MOVE_FORWARD, v.Button.TURN_LEFT_RIGHT_DELTA, v.Button.USE])
    game.set_available_game_variables([v.GameVariable.POSITION_X, v.GameVariable.POSITION_Y,
                                       v.GameVariable.POSITION_Z, v.GameVariable.ANGLE])
    game.init()
    return game


def walk_to(game, target, *, tolerance=20, max_tics=2000, arrival=None, stop_on_teleport=False):
    previous = None
    for tick in range(max_tics):
        if game.is_episode_finished():
            return False
        x, y, z, angle = game.get_state().game_variables
        if arrival is not None and math.dist((x, y), arrival) <= tolerance:
            return True
        if stop_on_teleport and previous is not None and math.dist((x, y), previous) > 128:
            return False
        previous = x, y
        if arrival is None and math.dist((x, y), target) <= tolerance:
            return True
        desired = math.degrees(math.atan2(target[1] - y, target[0] - x))
        error = (desired - angle + 180) % 360 - 180
        # Re-using a moving Doom door reverses it. Allow a full opening cycle
        # between presses instead of toggling it before the player fits.
        game.make_action([int(abs(error) < 15), max(-10, min(10, -error)), int(tick % 128 == 0)], 1)
    return False

"""A key must be accessible before its door and actually gate the exit."""
from copy import deepcopy

import pytest

from generation.keyed import add_blue_key
from generation.gameplay import apply_intro
from generation.validator import require_valid
from tests.gameplay_fixtures import keyed_arena
from wad_parser.navigation import PlayerNavigation
from wad_parser.map_extractor import compute_derived


def test_blue_key_has_a_real_gate_reachable_earlier_room_and_location_history():
    doc = keyed_arena()
    report = apply_intro(doc, seed=3, monster_budget=4)
    before = deepcopy(doc)
    plan = add_blue_key(doc, seed=3)
    assert plan['profile'] == 'blue-key' and plan['uses_predicted_key_room']
    assert plan['key_reachable_without_key'] and not plan['exit_reachable_without_key']
    assert plan['validation']['status'] == 'success'
    assert any(a['action'] == 'key' for a in plan['validation']['trace'])
    assert report['progression'] is plan
    assert doc.things[plan['key_thing']].type == 5
    assert plan['blue_frame_lines'] and all(doc.linedefs[i].special == 26 for i in plan['door_lines'])
    assert not require_valid(doc, require_connected=True)['errors']
    no_key = deepcopy(doc)
    key = no_key.things.pop(plan['key_thing'])
    no_key.bounds, no_key.derived = compute_derived(no_key.vertices, no_key.sectors,
        no_key.linedefs, no_key.sidedefs, no_key.things)
    navigation = PlayerNavigation(no_key)
    assert navigation.reaches_point(key.x, key.y, sid=plan['key_sector'])
    assert not any(navigation.reaches_line(line) for line in no_key.linedefs if line.special == 11)
    duplicate = add_blue_key(before, seed=3)
    assert duplicate == plan and before.to_dict() == doc.to_dict()


def test_blue_key_rejects_bypassable_or_existing_keys_without_mutation():
    from generation.style import _add_thing
    from generation.game_profile import GameProfile
    doc = keyed_arena(bypass=True)
    before = doc.to_dict()
    with pytest.raises(ValueError, match='mandatory blue-key gate'):
        add_blue_key(doc)
    assert doc.to_dict() == before
    with pytest.raises(ValueError, match='stock blue keycard'):
        add_blue_key(doc, game=GameProfile(name='empty'))
    assert doc.to_dict() == before
    _add_thing(doc, doc.things[0].x + 48, doc.things[0].y, 5)
    before = doc.to_dict()
    with pytest.raises(ValueError, match='existing keys'):
        add_blue_key(doc)
    assert doc.to_dict() == before


def test_blue_key_cli_requires_intro_before_model_loading(tmp_path):
    from click.testing import CliRunner
    from cli.generate_roomgen import main
    checkpoint = tmp_path / 'stub.pt'
    checkpoint.touch()
    result = CliRunner().invoke(main, ['--checkpoint', str(checkpoint), '--progression', 'blue-key'])
    assert result.exit_code != 0 and 'requires --gameplay intro' in result.output

import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parent)]
from geometry_fixtures import polygon_document, square
from generation.actors import spawn_problem
from generation.style import _add_thing
from generation.validator import validate_document


def test_large_monster_wall_height_and_solid_overlap_checks():
    doc = polygon_document([[square(0, 0, 512, 512)]], start=(64, 64))
    assert spawn_problem(doc, 68, 40, 256) == "insufficient wall clearance"
    assert spawn_problem(doc, 16, 256, 256) is None
    doc.sectors[0].ceil_height = 96
    assert spawn_problem(doc, 16, 256, 256) == "insufficient vertical clearance"
    doc.sectors[0].ceil_height = 128
    _add_thing(doc, 256, 256, 3004)
    assert "overlaps" in spawn_problem(doc, 3004, 286, 286)  # axis-aligned Doom collision
    assert spawn_problem(doc, 3004, 320, 320) is None
    assert spawn_problem(doc, 9999, 320, 320) == "unsupported actor dimensions"


def test_final_validation_detects_actor_overlap():
    doc = polygon_document([[square(0, 0, 512, 512)]], start=(64, 64))
    _add_thing(doc, 256, 256, 3004)
    _add_thing(doc, 260, 256, 3004)
    assert any("overlaps" in error for error in validate_document(doc, require_exit=False)["errors"])


def test_actor_box_can_overlap_an_open_border_but_not_a_wall_or_high_step():
    from tests.geometry_fixtures import two_rooms
    doc = two_rooms()
    assert spawn_problem(doc, 3004, 120, 64) is None
    shared = next(line for line in doc.linedefs if line.left_sidedef >= 0)
    shared.flags.impassable = True
    assert spawn_problem(doc, 3004, 120, 64) == 'insufficient wall clearance'
    shared.flags.impassable = False
    doc.sectors[1].floor_height, doc.sectors[1].ceil_height = 64, 192
    assert spawn_problem(doc, 3004, 120, 64) == 'step exceeds actor clearance'


def test_small_monsters_fit_narrow_spaces_and_pickups_can_touch_walls():
    doc = polygon_document([[square(0, 0, 48, 256)]], start=(24, 32))
    assert spawn_problem(doc, 3004, 24, 128) is None  # radius 20, not blanket margin 32
    assert spawn_problem(doc, 68, 24, 128) == 'insufficient wall clearance'
    assert spawn_problem(doc, 2007, 4, 128) is None  # non-solid ammo touch radius
    assert spawn_problem(doc, 5, 4, 128) is None
    assert spawn_problem(doc, 11, 4, 128) == 'insufficient wall clearance'  # player marker


def test_inactive_skill_actors_and_coop_starts_do_not_block_single_player_spawns():
    doc = polygon_document([[square(0, 0, 512, 512)]], start=(64, 64))
    other = _add_thing(doc, 256, 256, 3004)
    other.flags.skill_3 = False
    assert spawn_problem(doc, 3004, 256, 256) is None
    assert not validate_document(doc, require_exit=False)['errors']
    other.flags.skill_3 = True
    assert 'overlaps' in spawn_problem(doc, 3004, 256, 256)
    other.type = 2  # a cooperative player start, not a simultaneous SP actor
    assert spawn_problem(doc, 3004, 256, 256) is None


def test_player_one_start_ignores_thing_skill_and_mode_bits():
    from wad_parser.traversal import distances_from_start
    doc = polygon_document([[square(0, 0, 128, 128)]], start=(64, 64))
    doc.things[0].flags.not_single_player = True
    doc.things[0].flags.skill_3 = False
    assert set(distances_from_start(doc)) == {0}
    assert not validate_document(doc, require_exit=False)['errors']

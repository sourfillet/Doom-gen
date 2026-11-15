import pytest

from tests.geometry_fixtures import polygon_document, decorative_start, square, neck_map
from wad_parser.navigation import PlayerNavigation
from generation.validator import validate_document
from generation.progression import ensure_exit


def test_one_sector_narrow_neck_does_not_make_its_exit_reachable():
    doc, line = neck_map(24)
    nav = PlayerNavigation(doc)
    assert nav.reaches_point(64, 64)
    assert not nav.reaches_point(320, 64)
    assert not nav.reaches_line(line)
    report = validate_document(doc, require_connected=True)
    assert any('configuration' in error for error in report['errors'])


@pytest.mark.parametrize('width', [32, 40, 64])
def test_clear_necks_preserve_player_routes(width):
    doc, line = neck_map(width)
    nav = PlayerNavigation(doc)
    assert nav.reaches_point(320, 64)
    assert nav.reaches_line(line)
    assert not validate_document(doc, require_connected=True)['errors']


def test_open_decorative_borders_are_not_eroded_as_walls():
    doc = decorative_start()
    ensure_exit(doc)
    nav = PlayerNavigation(doc)
    assert nav.reaches_point(64, 64)
    assert not validate_document(doc, require_connected=True)['errors']


def test_a_high_ledge_can_be_dropped_from_but_not_climbed():
    from tests.geometry_fixtures import two_rooms
    doc = two_rooms()
    doc.sectors[0].floor_height = 64
    nav = PlayerNavigation(doc)
    assert nav.reaches_point(192, 64)
    below = nav.reachable_from([(192, 64)])
    assert {sid for sid, _ in below} == {1}


def test_a_short_use_exit_does_not_need_a_player_width_portal():
    ring = [(0, 0), (0, 128), (128, 128), (128, 72), (128, 56), (128, 0)]
    doc = polygon_document([[ring]], start=(64, 64))
    line = next(line for line in doc.linedefs if doc.vertices[line.v1].y == 72)
    line.special = 11
    assert PlayerNavigation(doc).reaches_line(line)
    assert not validate_document(doc)['errors']


def test_exit_fallback_chooses_a_reachable_side_of_a_narrow_neck():
    doc, old = neck_map(24)
    old.special = 0
    ensure_exit(doc)
    nav = PlayerNavigation(doc)
    assert nav.reaches_line(next(line for line in doc.linedefs if line.special == 11))


def test_exit_use_ray_cannot_activate_through_a_closed_pillar():
    from tests.geometry_fixtures import square
    pillar = square(240, 32, 244, 224)
    doc = polygon_document([[square(0, 0, 256, 256), list(reversed(pillar))], [pillar]], start=(64, 128))
    doc.sectors[1].ceil_height = 0
    exit_line = next(line for line in doc.linedefs if doc.vertices[line.v1].x == doc.vertices[line.v2].x == 256)
    exit_line.special = 11
    nav = PlayerNavigation(doc)
    assert not nav.reaches_line(exit_line)
    assert any('configuration space' in error for error in validate_document(doc)['errors'])


@pytest.mark.parametrize('step', [0, 24])
def test_oblique_portal_does_not_become_one_way_from_roundoff(step):
    # Native forward-input controls cross this same authored portal both ways.
    doc = polygon_document([[[(0, 0), (0, 40), (129, 40), (256, 0)]],
                            [[(256, 0), (129, 40), (512, 40), (512, 0)]]], start=(32, 20))
    doc.sectors[1].floor_height = step
    nav = PlayerNavigation(doc)
    assert nav.reaches_point(480, 20)
    assert {sid for sid, _ in nav.reachable_from([(480, 20)])} == {0, 1}


def _shifted(ring, offset):
    return [(x + offset[0], y + offset[1]) for x, y in ring]


@pytest.mark.parametrize('offset', [(0, 0), (22000, -22000)])
@pytest.mark.parametrize('drop', [25, 128])
@pytest.mark.parametrize('width, reachable', [(31, False), (32, True), (40, True)])
def test_diagonal_ledge_drop_follows_native_width_limit(width, reachable, drop, offset):
    # Native forward-input controls drop off 32/40-wide diagonal ledges and
    # stop at a 31-wide neck; nobody climbs back up a >24-unit ledge.
    doc = polygon_document([[_shifted([(0, 0), (0, width), (293, width), (256, 0)], offset)],
                            [_shifted([(256, 0), (293, width), (512, width), (512, 0)], offset)]],
                           start=(480 + offset[0], width / 2 + offset[1]))
    doc.sectors[1].floor_height = drop
    for sector in doc.sectors:
        sector.ceil_height = drop + 128
    nav = PlayerNavigation(doc)
    low = 32 + offset[0], width / 2 + offset[1]
    assert nav.reaches_point(*low) == reachable
    assert not any(sid == 1 for sid, _ in nav.reachable_from([low]))


@pytest.mark.parametrize('width, ceiling, reachable', [(24, 96, False), (24, 119, False),
                                                       (24, 120, True), (40, 96, True)])
def test_drop_keeps_old_floor_headroom_until_the_box_clears_the_ledge(width, ceiling, reachable):
    # Native: past a 24-wide trench, a ceiling < 56 above the ledge stops the
    # player 8 units out on the old floor (PLUTONIA MAP22's start platform).
    doc = polygon_document([[square(0, 0, 64, 128)], [square(0, -width, 64, 0)],
                            [square(0, -width - 192, 64, -width)]], start=(32, 96))
    doc.sectors[0].floor_height, doc.sectors[0].ceil_height = 64, 192
    doc.sectors[1].ceil_height = 128
    doc.sectors[2].ceil_height = ceiling
    assert PlayerNavigation(doc).reaches_point(32, -width - 96) == reachable


def step_below_ledge(gap):
    doc = polygon_document([[square(0, 0, 64, 128)],
                            [square(0, -256, 64, -gap - 8), square(0, -gap, 64, 0)],
                            [square(0, -gap - 8, 64, -gap)]], start=(32, 96))
    doc.sectors[0].floor_height, doc.sectors[2].floor_height = 48, 16
    return doc


@pytest.mark.parametrize('gap', [4, 8])
def test_drop_passes_over_a_lower_step_inside_the_ledge_sweep(gap):
    # Native: the box stays on the ledge while it passes over a step sector
    # that does not touch the ledge (PLUTONIA MAP10's switch alcove).
    nav = PlayerNavigation(step_below_ledge(gap))
    assert nav.reaches_point(32, -128)
    assert not any(sid == 0 for sid, _ in nav.reachable_from([(32, -128)]))


def test_drop_bridge_ignores_compiled_sub_fixed_point_seams():
    # Exact zdbsp-derived outlines of the native step control: the 3e-14
    # seam between neighbours must not split the old-height bridge.
    doc = step_below_ledge(8)
    doc.derived.sector_regions = [
        [[(0.0, 0.0), (0.0, 128.0), (64.0, 128.0), (64.0, 0.0)]],
        [[(0.0, -16.0), (64.0, -16.0), (64.0, -256.0), (0.0, -256.0)],
         [(0.0, -8.0), (0.0, 0.0), (64.0, 0.0), (64.0, -8.0)]],
        [[(0.0, -16.0), (0.0, -8.0), (64.0, -8.000000000000028), (64.0, -16.0)]]]
    assert PlayerNavigation(doc).reaches_point(32, -128)


def test_drop_lands_beyond_a_thin_step_that_touches_the_ledge():
    # Native step-gap0: the 8-unit step across the mouth has no free region of
    # its own, but the ledge-supported box lands in the room beyond it.
    doc = polygon_document([[square(0, 0, 64, 128)], [square(0, -256, 64, -8)], [square(0, -8, 64, 0)]],
                           start=(32, 96))
    doc.sectors[0].floor_height, doc.sectors[2].floor_height = 48, 16
    nav = PlayerNavigation(doc)
    assert nav.reaches_point(32, -128)
    assert not any(sid == 0 for sid, _ in nav.reachable_from([(32, -128)]))

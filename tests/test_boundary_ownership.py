import pytest

from tests.geometry_fixtures import polygon_document, square, two_rooms, boundary_start
from wad_parser.components import ComponentIndex
from wad_parser.geometry import point_in_rings, sector_geometry
from wad_parser.map_extractor import _sector_containing, compute_derived
from wad_parser.navigation import PlayerNavigation


def test_player_start_on_an_open_boundary_bootstraps_rooms_and_a_physical_exit():
    from generation.progression import ensure_exit
    from generation.validator import require_valid
    from roomgraph.segment import segment
    from data.source_endings import assess_source_ending
    doc = boundary_start()
    assert set(doc.derived.reachable_from_start) == {0, 1}
    ensure_exit(doc)
    graph = segment(doc, playable_only=True)
    assert graph.rooms and any('start' in room.roles for room in graph.rooms)
    assert assess_source_ending(doc, graph)['complete']
    require_valid(doc, require_connected=True)


@pytest.mark.parametrize('large', [False, True])
def test_indexed_and_direct_boundary_ownership_agree_without_filling_hole_interiors(large):
    shell = square(0, 0, 256, 256)
    if large:
        shell = [(a[0] + (b[0] - a[0]) * t / 16, a[1] + (b[1] - a[1]) * t / 16)
                 for a, b in zip(shell, shell[1:] + shell[:1]) for t in range(16)]
    doc = polygon_document([[shell, list(reversed(square(64, 64, 192, 192)))]])
    rings = sector_geometry(doc)
    index = ComponentIndex(doc.vertices, rings)
    assert bool(index.prepared[0]) == large
    for point, expected in [((0, 128), 0), ((64, 128), 0), ((128, 128), None),
                            ((-1, 128), None), ((32, 128), 0)]:
        assert _sector_containing(*point, doc.sectors, rings, doc.vertices) == expected
        assert _sector_containing(*point, doc.sectors, rings, doc.vertices, locator=index) == expected
    assert not point_in_rings(0, 128, [shell])
    assert point_in_rings(0, 128, [shell], boundary=True)


def test_geometric_boundary_ownership_does_not_make_a_wall_spawn_physically_valid():
    from generation.validator import validate_document
    doc = two_rooms()
    doc.things[0].x = 0
    doc.bounds, doc.derived = compute_derived(doc.vertices, doc.sectors, doc.linedefs, doc.sidedefs, doc.things)
    assert _sector_containing(0, 64, doc.sectors, sector_geometry(doc), doc.vertices) == 0
    assert not PlayerNavigation(doc).reached
    assert validate_document(doc, require_exit=False)['errors']

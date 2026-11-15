import copy
import sys
from pathlib import Path

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parent)]
from geometry_fixtures import polygon_document, square, two_rooms
from data.rasterizer import rasterize, _grid_transform
from wad_parser.geometry import sector_area_perimeter, sector_clearance, interior_point
from wad_parser.map_extractor import _sector_containing, compute_derived
from wad_parser.types import MapDocument, Linedef, LinedefFlags, Sidedef


def _owner(doc, x, y):
    return _sector_containing(x, y, doc.sectors, doc.derived.sector_rings, doc.vertices)


def test_hole_island_and_nested_sector_ownership_and_raster():
    outer = square(0, 0, 256, 256)
    hole = list(reversed(square(64, 64, 192, 192)))
    island = square(320, 0, 384, 64)
    doc = polygon_document([[outer, hole, island], [square(64, 64, 192, 192)]])
    assert not doc.derived.geometry_issues
    assert len(doc.derived.sector_rings[0]) == 3
    assert _owner(doc, 32, 32) == 0
    assert _owner(doc, 350, 32) == 0
    assert _owner(doc, 128, 128) == 1
    assert _owner(doc, 300, 32) is None
    assert _owner(doc, 1e6, 1e6) is None
    area, perimeter = sector_area_perimeter(doc, 0)
    assert area == 256**2 - 128**2 + 64**2
    assert perimeter == 4 * (256 + 128 + 64)
    doc.sectors[1].floor_height = 32
    raster, _ = rasterize(doc, 128)
    grid, _ = _grid_transform(doc, 128)
    for x, y, floor, occupied in [(128, 128, 32, 1), (32, 32, 0, 1), (300, 32, 0, 0)]:
        c, r = grid(x, y)
        assert raster[9, int(r), int(c)] == occupied
        if occupied:
            assert raster[2, int(r), int(c)] == pytest.approx(floor / 2048 + 0.5)


def test_void_hole_is_not_painted_or_selected_for_spawns():
    doc = polygon_document([[square(0, 0, 256, 256),
                              list(reversed(square(64, 64, 192, 192)))]])
    raster, _ = rasterize(doc, 64)
    assert raster[9, 32, 32] == 0
    assert _owner(doc, 128, 128) is None
    point = interior_point(doc, 0, clearance=16)
    assert point and _owner(doc, *point) == 0
    assert sector_clearance(doc, 0, *point) >= 16


def test_same_sector_internal_line_is_not_a_boundary():
    doc = two_rooms()
    before = copy.deepcopy(doc.derived.sector_rings)
    sd = len(doc.sidedefs)
    doc.sidedefs.extend([Sidedef(sd, 0, 0, "-", "-", "-", 0),
                         Sidedef(sd + 1, 0, 0, "-", "-", "-", 0)])
    doc.linedefs.append(Linedef(len(doc.linedefs), 0, 2, LinedefFlags(two_sided=True),
                                0, 0, sd, sd + 1))
    _, derived = compute_derived(doc.vertices, doc.sectors, doc.linedefs, doc.sidedefs, doc.things)
    assert derived.sector_rings == before
    assert not derived.geometry_issues


def test_open_walk_is_diagnosed_and_never_filled():
    doc = polygon_document([[square(0, 0, 128, 128)]])
    doc.linedefs.pop()
    doc.bounds, doc.derived = compute_derived(doc.vertices, doc.sectors, doc.linedefs, doc.sidedefs, [])
    assert doc.derived.geometry_issues
    assert doc.derived.sector_rings == [[]]
    assert rasterize(doc, 32)[0][9].sum() == 0


def test_legacy_json_upgrades_derived_data_without_reextracting():
    doc = polygon_document([[square(0, 0, 256, 256),
                              list(reversed(square(64, 64, 192, 192)))]])
    raw = doc.to_dict()
    for field in ("sector_rings", "geometry_issues", "derived_version"):
        raw["derived"].pop(field)
    loaded = MapDocument.from_dict(raw)
    from wad_parser.geometry import DERIVED_VERSION
    assert loaded.derived.derived_version == DERIVED_VERSION
    assert loaded.derived.sector_rings == doc.derived.sector_rings
    assert _owner(loaded, 128, 128) is None


def test_pinched_face_walk_retains_touching_hole_and_outer_boundary():
    doc = polygon_document([[square(0, 0, 256, 256), [(0, 0), (64, 32), (32, 64)]]])
    # Start away from the shared vertex so tracing encounters the inner
    # closed cycle midway through the outer walk.
    doc.linedefs = doc.linedefs[1:] + doc.linedefs[:1]
    for index, line in enumerate(doc.linedefs):
        line.id = index
    doc.bounds, doc.derived = compute_derived(doc.vertices, doc.sectors, doc.linedefs, doc.sidedefs, doc.things)
    assert not doc.derived.geometry_issues
    assert len(doc.derived.sector_rings[0]) == 2
    assert sector_area_perimeter(doc, 0)[0] == 256**2 - 1536

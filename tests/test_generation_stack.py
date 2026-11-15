"""Tests for the generation stack: WAD writer round-trip, vectorizer,
validator, model-free pipeline, and (when torch is installed) the U-Net /
diffusion plumbing."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wad_builder import build_hexen_wad, build_two_sector_wad

from data.rasterizer import rasterize
from generation.pipeline import regenerate_from_json
from generation.validator import validate_and_repair
from generation.vectorizer import vectorize
from wad_parser.map_extractor import extract_all_maps
from wad_parser.wad_reader import WadFile
from wad_writer.assembler import assemble_map_lumps
from wad_writer.wad_writer import write_wad


def _extract(tmp_path, builder=build_two_sector_wad, name="t.wad"):
    p = tmp_path / name
    p.write_bytes(builder())
    return extract_all_maps(WadFile(str(p)))[0]


# ---------------------------------------------------------------------------
# WAD writer
# ---------------------------------------------------------------------------

def test_write_parse_round_trip(tmp_path):
    doc = _extract(tmp_path)
    out = write_wad([doc], tmp_path / "out.wad")
    doc2 = extract_all_maps(WadFile(str(out)))[0]

    d1, d2 = doc.to_dict(), doc2.to_dict()
    for section in ("vertices", "sectors", "linedefs", "sidedefs", "things"):
        assert d1[section] == d2[section], section
    assert d2["meta"]["map_name"] == "MAP01"
    # derived data recomputes identically from identical raw data
    assert d1["derived"]["sector_graph"] == d2["derived"]["sector_graph"]


def test_round_trip_preserves_junk_flag_bits(tmp_path):
    doc = _extract(tmp_path)
    doc.linedefs[0].extra["unrecognized_flag_bits"] = 0xC000  # E2M7-style junk
    out = write_wad([doc], tmp_path / "junk.wad")
    doc2 = extract_all_maps(WadFile(str(out)))[0]
    assert doc2.linedefs[0].extra["unrecognized_flag_bits"] == 0xC000


def test_writer_rejects_non_doom_format(tmp_path):
    doc = _extract(tmp_path, build_hexen_wad)
    with pytest.raises(ValueError):
        assemble_map_lumps(doc)


def test_multi_map_wad(tmp_path):
    doc = _extract(tmp_path)
    import copy
    doc2 = copy.deepcopy(doc)
    doc2.meta.map_name = "MAP02"
    out = write_wad([doc, doc2], tmp_path / "two.wad")
    maps = extract_all_maps(WadFile(str(out)))
    assert [m.meta.map_name for m in maps] == ["MAP01", "MAP02"]


# ---------------------------------------------------------------------------
# vectorizer + validator
# ---------------------------------------------------------------------------

def test_vectorize_reconstructs_two_rooms(tmp_path):
    doc = _extract(tmp_path)
    doc.sectors[1].floor_height = 64  # make the rooms distinguishable
    raster, _ = rasterize(doc, resolution=128)
    vec = vectorize(raster, cell_units=64, coarse_factor=4)

    assert len(vec.sectors) >= 2
    floors = sorted({s.floor_height for s in vec.sectors})
    assert floors[0] <= 16 and floors[-1] >= 48  # ~0 and ~64, snapped
    assert all(s.ceil_height > s.floor_height for s in vec.sectors)
    # structural invariants: every linedef has a front side, every sidedef
    # references a real sector
    assert all(ld.right_sidedef != -1 for ld in vec.linedefs)
    assert all(0 <= sd.sector < len(vec.sectors) for sd in vec.sidedefs)
    two_sided = [ld for ld in vec.linedefs if ld.left_sidedef != -1]
    assert two_sided, "adjacent rooms must share a two-sided boundary"
    # boundary rings close: every sector polygon reconstructs
    assert all(len(ring) >= 4 for ring in vec.derived.sector_polygons)


def test_vectorize_empty_raster_raises():
    with pytest.raises(ValueError):
        vectorize(np.zeros((9, 64, 64), dtype=np.float32))


def _disc_raster(res=128):
    """A circular room: the acid test for de-blockifying - rectilinear
    tracing yields a pixel staircase, simplification should yield diagonal
    walls and far fewer vertices."""
    raster = np.zeros((9, res, res), dtype=np.float32)
    yy, xx = np.mgrid[0:res, 0:res]
    disc = (xx - res / 2) ** 2 + (yy - res / 2) ** 2 < (res * 0.4) ** 2
    raster[2][disc] = 0.5           # floor 0
    raster[4][disc] = 0.5625        # ceiling 128
    raster[5][disc] = 160 / 255
    raster[6][disc] = 1.0
    return raster


def test_simplification_produces_diagonals_and_fewer_vertices():
    raster = _disc_raster()
    blocky = vectorize(raster, coarse_factor=2, simplify_tolerance=0.0)
    smooth = vectorize(raster, coarse_factor=2, simplify_tolerance=1.3)
    assert len(smooth.vertices) < len(blocky.vertices) / 3
    vpos = {v.id: (v.x, v.y) for v in smooth.vertices}
    diagonals = [ld for ld in smooth.linedefs
                 if vpos[ld.v1][0] != vpos[ld.v2][0] and vpos[ld.v1][1] != vpos[ld.v2][1]]
    assert diagonals, "a circle must simplify to include non-axis-aligned walls"
    # still watertight: the sector ring closes
    assert all(len(ring) >= 3 for ring in smooth.derived.sector_polygons)


def test_validator_inserts_start_and_fixes_heights(tmp_path):
    doc = _extract(tmp_path)
    raster, _ = rasterize(doc, resolution=128)
    vec = vectorize(raster)
    vec.sectors[0].ceil_height = vec.sectors[0].floor_height  # crushed room
    vec.sectors[0].light_level = 0
    repairs = validate_and_repair(vec, brighten=True)
    assert any("Player1Start" in r for r in repairs)
    assert any(t.type == 1 for t in vec.things)
    assert vec.sectors[0].ceil_height >= vec.sectors[0].floor_height + 56
    assert vec.sectors[0].light_level >= 96
    # the inserted start sits inside a sector, not in the void
    start = next(t for t in vec.things if t.type == 1)
    from wad_parser.map_extractor import _sector_containing
    assert _sector_containing(start.x, start.y, vec.sectors,
                              vec.derived.sector_polygons, vec.vertices) is not None


def test_reachability_repair_lowers_cliff(tmp_path):
    # sector 1 at floor 200 is an unclimbable ledge from sector 0 (step
    # limit is 24); the repair must pull it into reach, moving its ceiling
    # with it so room height is preserved
    doc = _extract(tmp_path)
    doc.sectors[1].floor_height = 200
    doc.sectors[1].ceil_height = 328
    repairs = validate_and_repair(doc, repair_steps=True)
    assert any("traversable" in r for r in repairs)
    assert doc.sectors[1].floor_height - doc.sectors[0].floor_height <= 24
    assert doc.sectors[1].ceil_height - doc.sectors[1].floor_height == 128
    assert not any("disconnected" in r for r in repairs)


def test_reachable_map_needs_no_reachability_repair(tmp_path):
    doc = _extract(tmp_path)
    doc.sectors[1].floor_height = 16  # ordinary step
    repairs = validate_and_repair(doc)
    assert not any("traversable" in r or "disconnected" in r for r in repairs)


# ---------------------------------------------------------------------------
# model-free pipeline (the "regenerate a single map" harness)
# ---------------------------------------------------------------------------

def test_regenerate_from_json_end_to_end(tmp_path):
    doc = _extract(tmp_path)
    json_path = tmp_path / "MAP01.json"
    json_path.write_text(json.dumps(doc.to_dict()))

    result = regenerate_from_json(json_path, tmp_path / "regen.wad")
    assert result["sectors"] >= 1
    # the written WAD parses back with playable structure
    redoc = extract_all_maps(WadFile(result["path"]))[0]
    assert redoc.meta.map_name == "MAP01"
    assert any(t.type == 1 for t in redoc.things)
    assert redoc.derived.reachable_from_start  # start reaches somewhere


# torch-dependent model tests live in tests/test_torch_models.py so this
# module's tests run even before torch is installed.

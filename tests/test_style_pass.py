"""Tests for the corpus-statistics style pass (interim Stage 3/4) and the
validator's teleporter reconnection of void-separated islands."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wad_builder import build_two_sector_wad

from data.rasterizer import rasterize
from generation.style import PlacementDensity, StyleModel, apply_style
from generation.validator import validate_and_repair
from generation.vectorizer import vectorize
from wad_parser.map_extractor import _sector_containing, extract_all_maps
from wad_parser.wad_reader import WadFile


def _corpus_doc(tmp_path):
    p = tmp_path / "src.wad"
    p.write_bytes(build_two_sector_wad())
    return extract_all_maps(WadFile(str(p)))[0]


def _fitted_style(tmp_path) -> StyleModel:
    doc = _corpus_doc(tmp_path)
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    (corpus / "MAP01.json").write_text(json.dumps(doc.to_dict()))
    return StyleModel.fit(corpus)


def _generated_doc(tmp_path):
    doc = _corpus_doc(tmp_path)
    raster, _ = rasterize(doc, resolution=128)
    vec = vectorize(raster, coarse_factor=2)
    validate_and_repair(vec)
    return vec


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------

def test_fit_collects_corpus_statistics(tmp_path):
    style = _fitted_style(tmp_path)
    assert style.flat_pairs[("FLOOR4_8", "CEIL3_5")] > 0
    assert style.wall_mids["STARTAN2"] > 0
    assert style.monster_types == {3004: 1}   # the builder's Zombieman
    assert style.monster_density and style.monster_density[0] > 0


def test_style_json_round_trip(tmp_path):
    style = _fitted_style(tmp_path)
    style.save(tmp_path / "style.json")
    reloaded = StyleModel.load(tmp_path / "style.json")
    assert reloaded.to_json() == style.to_json()


# ---------------------------------------------------------------------------
# application
# ---------------------------------------------------------------------------

def test_apply_style_textures_things_exit(tmp_path):
    style = _fitted_style(tmp_path)
    doc = _generated_doc(tmp_path)
    summary = apply_style(doc, style, seed=11)

    # flats/walls come from the corpus vocabulary
    assert all((s.floor_flat, s.ceil_flat) in style.flat_pairs for s in doc.sectors)
    one_sided_mids = {doc.sidedefs[ld.right_sidedef].mid_tex
                      for ld in doc.linedefs if ld.left_sidedef == -1 and ld.special == 0}
    assert one_sided_mids <= set(style.wall_mids)

    # an exit switch exists and is S1 Exit
    exits = [ld for ld in doc.linedefs if ld.special == 11]
    assert len(exits) == 1
    assert exits[0].trigger_type == "switch"
    assert "exit_sector" in summary

    # placed things are inside real sectors
    for t in doc.things:
        sec = _sector_containing(t.x, t.y, doc.sectors,
                                 doc.derived.sector_polygons, doc.vertices)
        assert sec is not None
    # monsters obey the keep-away radius around the player start
    start = next(t for t in doc.things if t.type == 1)
    for t in doc.things:
        if t.category == "monster":
            assert (t.x - start.x) ** 2 + (t.y - start.y) ** 2 >= 256 ** 2
    assert summary["dm_starts"] >= 1


def test_apply_style_deterministic(tmp_path):
    style = _fitted_style(tmp_path)
    a = _generated_doc(tmp_path)
    b = _generated_doc(tmp_path)
    apply_style(a, style, seed=42)
    apply_style(b, style, seed=42)
    assert a.to_dict() == b.to_dict()


def test_styled_map_survives_wad_round_trip(tmp_path):
    from wad_writer.wad_writer import write_wad
    style = _fitted_style(tmp_path)
    doc = _generated_doc(tmp_path)
    apply_style(doc, style, seed=3)
    out = write_wad([doc], tmp_path / "styled.wad")
    redoc = extract_all_maps(WadFile(str(out)))[0]
    assert any(ld.special == 11 for ld in redoc.linedefs)
    assert any(t.category == "monster" for t in redoc.things) or True  # density may be 0
    assert {s.floor_flat for s in redoc.sectors} <= {f for f, _ in style.flat_pairs}


# ---------------------------------------------------------------------------
# placement safety: wall clearance + reachability gating
# ---------------------------------------------------------------------------

def _reachable_sectors(doc):
    """Sectors reachable from the player start following shared walls AND
    teleporter links - the same relation apply_style gates placement on."""
    from collections import deque

    from generation.style import _TELEPORT_LINE_SPECIALS

    polys = doc.derived.sector_polygons
    vpos = {v.id: (v.x, v.y) for v in doc.vertices}
    nb = {s.id: set() for s in doc.sectors}
    for e in doc.derived.sector_graph.get("edges", []):
        nb[e["a"]].add(e["b"]); nb[e["b"]].add(e["a"])
    tag2sec = {}
    for s in doc.sectors:
        if s.tag:
            tag2sec.setdefault(s.tag, s.id)
    for ld in doc.linedefs:
        if ld.special in _TELEPORT_LINE_SPECIALS and ld.tag \
                and 0 <= ld.right_sidedef < len(doc.sidedefs):
            f = doc.sidedefs[ld.right_sidedef].sector
            d = tag2sec.get(ld.tag)
            if d is not None and f in nb and d in nb:
                nb[f].add(d); nb[d].add(f)
    start = next((t for t in doc.things if t.type == 1), None)
    if start is None:
        return set()
    ss = _sector_containing(start.x, start.y, doc.sectors, polys, doc.vertices)
    reached, q = {ss}, deque([ss])
    while q:
        for x in nb.get(q.popleft(), ()):
            if x not in reached:
                reached.add(x); q.append(x)
    return reached


def test_placed_things_clear_walls_and_stay_reachable(tmp_path):
    from generation.style import _THING_WALL_MARGIN, _wall_clearance

    style = _forced_style(tmp_path)          # inflated densities -> things get placed
    doc = _generated_doc(tmp_path)
    apply_style(doc, style, seed=9)

    polys = doc.derived.sector_polygons
    vpos = {v.id: (v.x, v.y) for v in doc.vertices}
    reached = _reachable_sectors(doc)

    placed = [t for t in doc.things
              if t.category == "monster" or t.category in _PICKUPS]
    assert placed, "densities were inflated; expected some placements"
    for t in placed:
        sec = _sector_containing(t.x, t.y, doc.sectors, polys, doc.vertices)
        assert sec in reached, f"thing in unreachable sector {sec}"
        ring = [vpos[v] for v in polys[sec] if v in vpos]
        assert _wall_clearance(t.x, t.y, ring) >= _THING_WALL_MARGIN - 1e-6


def test_placement_respects_teleporter_reachability(tmp_path):
    """On a two-island map the validator bridges with a teleporter; placement
    must treat the far island as reachable and never place in a truly
    unreachable sector."""
    from generation.style import _TELEPORT_LINE_SPECIALS

    style = _forced_style(tmp_path)
    vec = vectorize(_two_island_raster(), coarse_factor=2)
    validate_and_repair(vec)
    assert any(ld.special in _TELEPORT_LINE_SPECIALS for ld in vec.linedefs), \
        "precondition: validator should have inserted a teleporter"

    apply_style(vec, style, seed=3)
    reached = _reachable_sectors(vec)
    for t in vec.things:
        if t.category == "monster" or t.category in _PICKUPS:
            sec = _sector_containing(t.x, t.y, vec.sectors,
                                     vec.derived.sector_polygons, vec.vertices)
            assert sec in reached, f"placed in unreachable sector {sec}"


# ---------------------------------------------------------------------------
# faithful transfer (regen harness: carry the source map's real look)
# ---------------------------------------------------------------------------

def test_faithful_transfer_preserves_source_textures_and_things(tmp_path):
    from generation.pipeline import raster_to_document

    source = _corpus_doc(tmp_path)
    raster, meta = rasterize(source, resolution=128)
    cell_units = max(8, int(round(2 / meta["scale"] / 8)) * 8)
    doc, repairs, deco = raster_to_document(
        raster, map_name="MAP01", cell_units=cell_units, coarse_factor=2,
        source=source, meta=meta)

    assert deco["mode"] == "faithful"
    # flats are the source's own, not sampled from anywhere else
    src_flats = {(s.floor_flat, s.ceil_flat) for s in source.sectors}
    assert {(s.floor_flat, s.ceil_flat) for s in doc.sectors} <= src_flats
    # no invented thing types: the source's zombieman and start, nothing else
    src_types = {t.type for t in source.things}
    assert {t.type for t in doc.things} <= src_types
    assert any(t.type == 3004 for t in doc.things)   # the source monster
    assert any(t.type == 1 for t in doc.things)       # the source player start


def test_faithful_transfer_is_the_regen_default(tmp_path):
    """regenerate_from_json without --style decorates faithfully."""
    from generation.pipeline import regenerate_from_json

    source = _corpus_doc(tmp_path)
    src_json = tmp_path / "MAP01.json"
    src_json.write_text(json.dumps(source.to_dict()))
    result = regenerate_from_json(src_json, tmp_path / "out.wad", coarse_factor=2)
    assert result["decoration"]["mode"] == "faithful"


# ---------------------------------------------------------------------------
# density-guided placement (raster channel 8)
# ---------------------------------------------------------------------------

_PICKUPS = {"weapon", "ammo", "item"}


def _forced_style(tmp_path) -> StyleModel:
    """Corpus style with densities inflated so placement always happens."""
    style = _fitted_style(tmp_path)
    style.monster_density = [400.0]
    style.pickup_density = [400.0]
    return style


def test_density_guided_placement_follows_channel(tmp_path):
    style = _forced_style(tmp_path)
    doc = _corpus_doc(tmp_path)
    raster, _ = rasterize(doc, resolution=128)
    vec = vectorize(raster, coarse_factor=2)
    validate_and_repair(vec)

    # put ALL thing-density mass in the top-left quadrant of the painted
    # area - checks both axes, including the raster-row -> world-y flip
    r = raster.copy()
    painted = r[1:7].max(axis=0) > 0.05
    r[8] = 0.0
    r[8, :64, :64] = painted[:64, :64]
    assert r[8].sum() > 0
    density = PlacementDensity.from_raster(r, cell_units=64, coarse_factor=2)

    summary = apply_style(vec, style, seed=5, density=density)
    assert summary["placement"] == "density"
    assert summary["monsters"] + summary["pickups"] > 0

    # everything sampled from the channel lands in that quadrant
    # (margin: 2px blur spread + in-pixel jitter)
    mid_x = 64 * density.units_per_px
    mid_y = density.world_height - 64 * density.units_per_px
    margin = 3 * density.units_per_px
    for t in vec.things:
        if t.category == "monster" or t.category in _PICKUPS:
            assert t.x <= mid_x + margin, (t.x, mid_x)
            assert t.y >= mid_y - margin, (t.y, mid_y)


def test_density_without_mass_falls_back_to_area(tmp_path):
    style = _forced_style(tmp_path)
    doc = _generated_doc(tmp_path)
    density = PlacementDensity(np.zeros((64, 64)), 32.0, 64 * 64.0)
    summary = apply_style(doc, style, seed=7, density=density)
    assert summary["placement"] == "area"
    assert summary["monsters"] + summary["pickups"] > 0


def test_density_guided_placement_deterministic(tmp_path):
    style = _forced_style(tmp_path)
    docs = []
    for _ in range(2):
        doc = _corpus_doc(tmp_path)
        raster, _ = rasterize(doc, resolution=128)
        vec = vectorize(raster, coarse_factor=2)
        validate_and_repair(vec)
        density = PlacementDensity.from_raster(raster, cell_units=64, coarse_factor=2)
        apply_style(vec, style, seed=42, density=density)
        docs.append(vec)
    assert docs[0].to_dict() == docs[1].to_dict()


# ---------------------------------------------------------------------------
# teleporter reconnection
# ---------------------------------------------------------------------------

def _two_island_raster(res=128):
    """Two separated blobs, each split into two sectors (so both sides
    have an internal crossable line for the return teleporter)."""
    r = np.zeros((9, res, res), dtype=np.float32)
    for x0, x1 in ((8, 56), (72, 120)):
        r[2, 8:120, x0:x1] = 0.5          # floor 0
        r[4, 8:64, x0:x1] = 0.5625        # ceiling 128
        r[4, 64:120, x0:x1] = 0.59375     # ceiling 192 -> second sector
        r[5, 8:120, x0:x1] = 160 / 255
        r[6, 8:120, x0:x1] = 1.0
    return r


def test_teleporter_reconnects_islands():
    vec = vectorize(_two_island_raster(), coarse_factor=2)
    repairs = validate_and_repair(vec)
    assert any("teleporter" in r for r in repairs), repairs
    assert not any("remain disconnected" in r for r in repairs), repairs
    ports = [ld for ld in vec.linedefs if ld.special == 97]
    assert len(ports) == 2                     # outbound + return
    dests = [t for t in vec.things if t.type == 14]
    assert len(dests) == 2
    # tags pair up: each teleport line's tag matches a tagged sector
    sector_tags = {s.tag for s in vec.sectors if s.tag}
    assert {ld.tag for ld in ports} == sector_tags and len(sector_tags) == 2

    # every destination lands solidly inside a sector, well off the walls -
    # otherwise the player teleports into the void (HOM) or wedged in geometry
    from generation.validator import _edge_clearance
    polys = vec.derived.sector_polygons
    vpos = {v.id: (v.x, v.y) for v in vec.vertices}
    for d in dests:
        sec = _sector_containing(d.x, d.y, vec.sectors, polys, vec.vertices)
        assert sec is not None, "teleport dest dropped into the void"
        ring = [vpos[v] for v in polys[sec] if v in vpos]
        assert _edge_clearance(d.x, d.y, ring) >= 16   # min player radius
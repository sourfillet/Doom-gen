"""Stage 4 placement GNN: vocab/target extraction, a forward/backward step,
and end-to-end learned placement onto a generated map (reachable, clear of
walls)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import torch  # noqa: F401
except Exception:
    pytest.skip("torch not installed", allow_module_level=True)

from wad_builder import build_two_sector_wad

from data.graph_builder import EDGE_FEATURES
from data.placement_targets import (
    build_placement_vocabs, placement_sample, vocab_sizes)
from data.rasterizer import rasterize
from generation.placement import PlacementModel, apply_learned_placement, _reachable_sectors
from generation.style import _THING_WALL_MARGIN, _wall_clearance
from generation.validator import validate_and_repair
from generation.vectorizer import vectorize
from models.conditioning import TagConditioner
from models.placement.gnn import PlacementGNN
from wad_parser.map_extractor import _sector_containing, extract_all_maps
from wad_parser.types import Thing, ThingFlags
from wad_parser.wad_reader import WadFile

_PICKUPS = {"weapon", "ammo", "item"}


def _corpus(tmp_path):
    p = tmp_path / "src.wad"
    p.write_bytes(build_two_sector_wad())
    doc = extract_all_maps(WadFile(str(p)))[0]
    # the fixture has a monster but no pickup; add one so both heads have vocab
    doc.things.append(Thing(id=len(doc.things), x=180, y=64, angle=0, type=2007,
                            type_name="Clip", category="ammo", flags=ThingFlags()))
    md = tmp_path / "maps"
    md.mkdir(exist_ok=True)
    (md / "MAP01.json").write_text(json.dumps(doc.to_dict()))
    return md, doc


def test_vocabs_and_targets(tmp_path):
    md, doc = _corpus(tmp_path)
    vocabs = build_placement_vocabs(md)
    assert vocabs["monster"] == [3004]       # the fixture Zombieman
    assert vocabs["pickup"] == [2007]        # the Clip we added
    sample = placement_sample(doc, vocabs)
    assert sample["monster_count"].sum() == 1
    assert sample["pickup_count"].sum() == 1
    assert sample["monster_type"].shape == (len(doc.sectors), 1)


def test_forward_backward_step(tmp_path):
    md, doc = _corpus(tmp_path)
    vocabs = build_placement_vocabs(md)
    sizes = vocab_sizes(vocabs)
    model = PlacementGNN(edge_features=len(EDGE_FEATURES), vocab=sizes)
    cond = TagConditioner(256)
    s = placement_sample(doc, vocabs)
    c = cond.encode([{"doom2": 1.0}], device="cpu", register_new=True)[0]
    out = model(torch.from_numpy(s["node_features"]),
                torch.from_numpy(s["edge_index"]),
                torch.from_numpy(s["edge_features"]), c)
    assert out["monster_rate"].shape == (len(doc.sectors),)
    assert (out["monster_rate"] >= 0).all()      # softplus keeps it non-negative
    loss = ((out["monster_rate"] - torch.from_numpy(s["monster_count"])) ** 2).mean()
    loss.backward()
    assert torch.isfinite(loss)


def test_apply_learned_placement(tmp_path):
    md, doc = _corpus(tmp_path)
    vocabs = build_placement_vocabs(md)
    model = PlacementGNN(edge_features=len(EDGE_FEATURES), vocab=vocab_sizes(vocabs))
    pm = PlacementModel(model, TagConditioner(256), vocabs, "cpu")

    raster, _ = rasterize(doc, resolution=128)
    gen = vectorize(raster, coarse_factor=2)
    validate_and_repair(gen)
    summary = apply_learned_placement(gen, pm, {}, seed=1)
    assert summary["mode"] == "gnn-placement"

    polys = gen.derived.sector_polygons
    vpos = {v.id: (v.x, v.y) for v in gen.vertices}
    reachable = _reachable_sectors(gen)
    start = next(t for t in gen.things if t.type == 1)
    for t in gen.things:
        if t.category == "monster" or t.category in _PICKUPS:
            sec = _sector_containing(t.x, t.y, gen.sectors, polys, gen.vertices)
            assert sec in reachable
            ring = [vpos[v] for v in polys[sec] if v in vpos]
            assert _wall_clearance(t.x, t.y, ring) >= _THING_WALL_MARGIN - 1e-6
            if t.category == "monster":
                assert (t.x - start.x) ** 2 + (t.y - start.y) ** 2 >= 256 ** 2

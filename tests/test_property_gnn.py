"""Stage 3 property GNN: vocab/target extraction, a forward/backward step,
and end-to-end apply_properties onto a generated map."""
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
from data.property_targets import build_vocabs, graph_sample, vocab_sizes
from data.rasterizer import rasterize
from generation.properties import PropertyModel, apply_properties
from generation.validator import validate_and_repair
from generation.vectorizer import vectorize
from models.conditioning import TagConditioner
from models.properties.gnn import HEADS, INPUT_COLUMNS, PropertyGNN
from wad_parser.map_extractor import extract_all_maps
from wad_parser.wad_reader import WadFile


def _corpus(tmp_path):
    p = tmp_path / "src.wad"
    p.write_bytes(build_two_sector_wad())
    doc = extract_all_maps(WadFile(str(p)))[0]
    md = tmp_path / "maps"
    md.mkdir(exist_ok=True)
    (md / "MAP01.json").write_text(json.dumps(doc.to_dict()))
    return md, doc


def test_vocabs_and_targets(tmp_path):
    md, doc = _corpus(tmp_path)
    vocabs = build_vocabs(md)
    assert "FLOOR4_8" in vocabs["flat"] and "CEIL3_5" in vocabs["flat"]
    assert "STARTAN2" in vocabs["tex"]
    assert vocabs["flat"]["UNK"] == 0

    sample = graph_sample(doc, vocabs)
    assert sample["node_features"].shape == (len(doc.sectors), len(INPUT_COLUMNS))
    for head in HEADS:
        assert sample[head].shape == (len(doc.sectors),)
    # the fixture's floor flat resolves to its real vocab index, not UNK
    assert sample["floor_flat"][0] == vocabs["flat"]["FLOOR4_8"]


def test_forward_backward_step(tmp_path):
    md, doc = _corpus(tmp_path)
    vocabs = build_vocabs(md)
    sizes = vocab_sizes(vocabs)
    model = PropertyGNN(edge_features=len(EDGE_FEATURES), vocab=sizes)
    cond = TagConditioner(256)
    sample = graph_sample(doc, vocabs)
    nf = torch.from_numpy(sample["node_features"])
    ei = torch.from_numpy(sample["edge_index"])
    ef = torch.from_numpy(sample["edge_features"])
    c = cond.encode([{"doom2": 1.0}], device="cpu", register_new=True)[0]
    logits = model(nf, ei, ef, c)
    assert logits["floor_flat"].shape == (len(doc.sectors), sizes["flat"])
    loss = sum(torch.nn.functional.cross_entropy(
        logits[h], torch.from_numpy(sample[h])) for h in HEADS)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.node_enc.weight.grad is not None


def test_apply_properties_writes_vocab_assets(tmp_path):
    md, doc = _corpus(tmp_path)
    vocabs = build_vocabs(md)
    model = PropertyGNN(edge_features=len(EDGE_FEATURES), vocab=vocab_sizes(vocabs))
    pm = PropertyModel(model, TagConditioner(256), vocabs, "cpu")

    raster, _ = rasterize(doc, resolution=128)
    gen = vectorize(raster, coarse_factor=2)
    validate_and_repair(gen)
    summary = apply_properties(gen, pm, {})
    assert summary["mode"] == "gnn"

    flat_names = set(vocabs["flat"])
    for s in gen.sectors:
        # every flat is a real vocab flat (UNK keeps the vectorizer default,
        # which is itself a corpus flat)
        assert s.floor_flat.upper() in flat_names
        assert s.ceil_flat.upper() in flat_names
        assert s.special in set(pm.inv_special.values())

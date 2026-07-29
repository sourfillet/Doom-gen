"""Vocabularies and per-sector targets for the Stage 3 property GNN.

Builds classification vocabularies (flats, wall textures, sector specials)
from the extracted corpus, and turns a MapDocument into the target index
arrays the GNN is trained against. Index 0 is reserved as UNK/none in each
vocabulary, so an unseen flat/texture or an uncommon special degrades to a
safe default instead of crashing.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from data.graph_builder import build_graph
from models.properties.gnn import INPUT_COLUMNS
from wad_parser.types import MapDocument

_NON_ASSET = ("-", "")


def build_vocabs(maps_dir: Path | str, top_flats: int = 192,
                 top_tex: int = 384) -> dict:
    flat_c, tex_c, spec_c = Counter(), Counter(), Counter()
    for p in sorted(Path(maps_dir).glob("*.json")):
        if p.name.endswith(".textures.json"):
            continue
        doc = MapDocument.from_dict(json.loads(p.read_text()))
        for s in doc.sectors:
            for flat in (s.floor_flat, s.ceil_flat):
                if flat not in _NON_ASSET:
                    flat_c[flat.upper()] += 1
            if s.special:
                spec_c[int(s.special)] += 1
        for sd in doc.sidedefs:
            for tex in (sd.upper_tex, sd.lower_tex, sd.mid_tex):
                if tex not in _NON_ASSET:
                    tex_c[tex.upper()] += 1
    flat = {"UNK": 0}
    for name, _ in flat_c.most_common(top_flats):
        flat[name] = len(flat)
    tex = {"UNK": 0}
    for name, _ in tex_c.most_common(top_tex):
        tex[name] = len(tex)
    special = {"0": 0}   # index 0 = "no special"; keys are stringified ints
    for num, _ in spec_c.most_common(48):
        special[str(num)] = len(special)
    return {"flat": flat, "tex": tex, "special": special}


def vocab_sizes(vocabs: dict) -> dict:
    return {"flat": len(vocabs["flat"]), "tex": len(vocabs["tex"]),
            "special": len(vocabs["special"])}


def _sector_wall_texture(doc: MapDocument) -> dict[int, str]:
    """Most common non-empty wall texture per sector."""
    counts: dict[int, Counter] = {s.id: Counter() for s in doc.sectors}
    for sd in doc.sidedefs:
        if sd.sector in counts:
            for tex in (sd.mid_tex, sd.upper_tex, sd.lower_tex):
                if tex not in _NON_ASSET:
                    counts[sd.sector][tex.upper()] += 1
    return {sid: (c.most_common(1)[0][0] if c else "") for sid, c in counts.items()}


def graph_sample(doc: MapDocument, vocabs: dict) -> dict:
    """Returns input node features (geometry subset), edge tensors, and the
    per-sector target indices for each head."""
    g = build_graph(doc, count_things=False)   # geometry only; thing counts unused
    node_in = g["node_features"][:, INPUT_COLUMNS]
    wall = _sector_wall_texture(doc)
    n = len(doc.sectors)
    floor_t = np.zeros(n, dtype=np.int64)
    ceil_t = np.zeros(n, dtype=np.int64)
    tex_t = np.zeros(n, dtype=np.int64)
    spec_t = np.zeros(n, dtype=np.int64)
    fv, tv, sv = vocabs["flat"], vocabs["tex"], vocabs["special"]
    for s in doc.sectors:
        floor_t[s.id] = fv.get(s.floor_flat.upper(), 0)
        ceil_t[s.id] = fv.get(s.ceil_flat.upper(), 0)
        tex_t[s.id] = tv.get(wall.get(s.id, ""), 0)
        spec_t[s.id] = sv.get(str(int(s.special)), 0)
    return {
        "node_features": node_in.astype(np.float32),
        "edge_index": g["edge_index"],
        "edge_features": g["edge_features"],
        "floor_flat": floor_t, "ceil_flat": ceil_t,
        "wall_tex": tex_t, "special": spec_t,
    }


def save_vocabs(vocabs: dict, path: Path | str) -> None:
    Path(path).write_text(json.dumps(vocabs))


def load_vocabs(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())

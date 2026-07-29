"""Vocabularies and per-sector targets for the Stage 4 placement GNN.

For each sector we record how many monsters / pickups it holds and the
distribution of their types - the supervision for the count (Poisson) and
type (categorical) heads. Types outside the top-K vocab are dropped so a
sector's count and its type distribution stay consistent.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from data.graph_builder import build_graph
from models.placement.gnn import INPUT_COLUMNS
from wad_parser.map_extractor import _sector_containing
from wad_parser.types import MapDocument

_PICKUPS = {"weapon", "ammo", "item"}


def build_placement_vocabs(maps_dir: Path | str, top: int = 24) -> dict:
    mc, pc = Counter(), Counter()
    for p in sorted(Path(maps_dir).glob("*.json")):
        if p.name.endswith(".textures.json"):
            continue
        doc = MapDocument.from_dict(json.loads(p.read_text()))
        for t in doc.things:
            if t.category == "monster":
                mc[t.type] += 1
            elif t.category in _PICKUPS:
                pc[t.type] += 1
    return {"monster": [t for t, _ in mc.most_common(top)],
            "pickup": [t for t, _ in pc.most_common(top)]}


def vocab_sizes(vocabs: dict) -> dict:
    return {"monster": len(vocabs["monster"]), "pickup": len(vocabs["pickup"])}


def placement_sample(doc: MapDocument, vocabs: dict) -> dict:
    g = build_graph(doc, count_things=False)
    node_in = g["node_features"][:, INPUT_COLUMNS]
    n = len(doc.sectors)
    mvi = {t: i for i, t in enumerate(vocabs["monster"])}
    pvi = {t: i for i, t in enumerate(vocabs["pickup"])}
    mon_count = np.zeros(n, dtype=np.float32)
    pick_count = np.zeros(n, dtype=np.float32)
    mon_type = np.zeros((n, len(mvi)), dtype=np.float32)
    pick_type = np.zeros((n, len(pvi)), dtype=np.float32)
    for t in doc.things:
        sec = _sector_containing(t.x, t.y, doc.sectors,
                                 doc.derived.sector_polygons, doc.vertices)
        if sec is None or sec >= n:
            continue
        if t.category == "monster" and t.type in mvi:
            mon_count[sec] += 1
            mon_type[sec, mvi[t.type]] += 1
        elif t.category in _PICKUPS and t.type in pvi:
            pick_count[sec] += 1
            pick_type[sec, pvi[t.type]] += 1
    return {
        "node_features": node_in.astype(np.float32),
        "edge_index": g["edge_index"], "edge_features": g["edge_features"],
        "monster_count": mon_count, "pickup_count": pick_count,
        "monster_type": mon_type, "pickup_type": pick_type,
    }


def save_vocabs(vocabs: dict, path: Path | str) -> None:
    Path(path).write_text(json.dumps(vocabs))


def load_vocabs(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())

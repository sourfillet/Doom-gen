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
from wad_parser.geometry import sector_geometry
from wad_parser.map_extractor import _sector_containing
from wad_parser.types import MapDocument, active_single_player

_PICKUPS = {"weapon", "ammo", "item", "key"}


def build_placement_vocabs(maps_dir: Path | str, top: int = 24, *, paths=None) -> dict:
    mc, pc = Counter(), Counter()
    for p in sorted(paths if paths is not None else Path(maps_dir).glob("*.json")):
        p = Path(p)
        if p.name.endswith(".textures.json"):
            continue
        doc = MapDocument.from_dict(json.loads(p.read_text(encoding="utf-8")))
        for t in doc.things:
            if not active_single_player(t):
                continue
            if t.category == "monster":
                mc[t.type] += 1
            elif t.category in _PICKUPS:
                pc[t.type] += 1
    pickups = [t for t, _ in pc.most_common(top)]
    # Rare keys must not disappear merely because ammo dominates the top-K.
    # This preserves observed key targets; a key/lock progression planner is
    # still responsible for deciding when and where a generated key is needed.
    pickups.extend(t for t in (5, 6, 13, 38, 39, 40) if t in pc and t not in pickups)
    return {"monster": [t for t, _ in mc.most_common(top)], "pickup": pickups}


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
    positions = {kind: np.zeros((n, 16 * 16), dtype=np.float32) for kind in ('monster', 'pickup')}
    angles = {kind: np.zeros((n, 8), dtype=np.float32) for kind in ('monster', 'pickup')}
    from wad_parser.components import ComponentIndex
    geom = sector_geometry(doc)
    geometry = ComponentIndex(doc.vertices, geom)
    boxes = geometry.bounds
    for t in doc.things:
        if not active_single_player(t):
            continue
        sec = _sector_containing(t.x, t.y, doc.sectors, geom, doc.vertices, locator=geometry)
        if sec is None or not (0 <= sec < n):
            continue
        if t.category == "monster" and t.type in mvi:
            mon_count[sec] += 1
            mon_type[sec, mvi[t.type]] += 1
            kind = 'monster'
        elif t.category in _PICKUPS and t.type in pvi:
            pick_count[sec] += 1
            pick_type[sec, pvi[t.type]] += 1
            kind = 'pickup'
        else:
            continue
        if sec in boxes:
            x0, y0, x1, y1 = boxes[sec]
            col = int(np.clip((t.x - x0) / max(x1 - x0, 1) * 16, 0, 15))
            row = int(np.clip((t.y - y0) / max(y1 - y0, 1) * 16, 0, 15))
            positions[kind][sec, row * 16 + col] += 1
            angles[kind][sec, round(t.angle / 45) % 8] += 1
    return {
        "node_features": node_in.astype(np.float32),
        "edge_index": g["edge_index"], "edge_features": g["edge_features"],
        "monster_count": mon_count, "pickup_count": pick_count,
        "monster_type": mon_type, "pickup_type": pick_type,
        'monster_position': positions['monster'], 'pickup_position': positions['pickup'],
        'monster_angle': angles['monster'], 'pickup_angle': angles['pickup'],
    }


def save_vocabs(vocabs: dict, path: Path | str) -> None:
    Path(path).write_text(json.dumps(vocabs), encoding="utf-8")


def load_vocabs(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

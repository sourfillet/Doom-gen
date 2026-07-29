"""Stage 4 inference: run the trained placement GNN and populate a generated
map's reachable sectors with the counts and monster/pickup types it predicts.

Division of labour: the GNN owns HOW MANY and WHICH TYPE per sector; the
proven geometric placer here owns WHERE - a point inside the sector polygon,
clear of walls, and (for monsters) off the player start. Only reachable,
big-enough sectors are populated (same rule as the heuristic placer).

torch is imported lazily.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from data.graph_builder import _ring_area_perimeter
from generation.style import (
    _THING_WALL_MARGIN, _TELEPORT_LINE_SPECIALS, _add_thing, _wall_clearance,
)
from models.properties.gnn import INPUT_COLUMNS
from wad_parser.map_extractor import _point_in_polygon, _sector_containing
from wad_parser.types import MapDocument

_MAX_PER_SECTOR = 24


class PlacementModel:
    def __init__(self, model, conditioner, vocabs: dict, device: str):
        self.model = model
        self.conditioner = conditioner
        self.vocabs = vocabs
        self.device = device
        self.monster_types = vocabs["monster"]
        self.pickup_types = vocabs["pickup"]


def load_placement_model(checkpoint_path: Path | str,
                         device: str | None = None) -> PlacementModel:
    import torch

    from models.conditioning import TagConditioner
    from models.placement.gnn import PlacementGNN

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = PlacementGNN(edge_features=cfg["edge_features"], vocab=cfg["vocab_sizes"],
                         hidden=cfg["hidden"], n_layers=cfg["layers"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    conditioner = TagConditioner(embed_dim=256).to(device)
    conditioner.load_state_dict(ck["conditioner"])
    conditioner.tag_to_index = ck["tag_to_index"]
    return PlacementModel(model, conditioner, ck["vocabs"], device)


def _reachable_sectors(doc: MapDocument) -> set[int]:
    """From the player start, over shared walls AND teleporter links."""
    vpos = {v.id: (v.x, v.y) for v in doc.vertices}
    nb: dict[int, set[int]] = {s.id: set() for s in doc.sectors}
    for e in doc.derived.sector_graph.get("edges", []):
        nb[e["a"]].add(e["b"])
        nb[e["b"]].add(e["a"])
    tag2sec: dict[int, int] = {}
    for s in doc.sectors:
        if s.tag:
            tag2sec.setdefault(s.tag, s.id)
    for ld in doc.linedefs:
        if ld.special in _TELEPORT_LINE_SPECIALS and ld.tag \
                and 0 <= ld.right_sidedef < len(doc.sidedefs):
            f = doc.sidedefs[ld.right_sidedef].sector
            d = tag2sec.get(ld.tag)
            if d is not None and f in nb and d in nb:
                nb[f].add(d)
                nb[d].add(f)
    start = next((t for t in doc.things if t.type == 1), None)
    if start is None:
        return set()
    ss = _sector_containing(start.x, start.y, doc.sectors,
                            doc.derived.sector_polygons, doc.vertices)
    if ss is None:                      # nearest-centroid fallback
        best = None
        for s in doc.sectors:
            ring = (doc.derived.sector_polygons[s.id]
                    if s.id < len(doc.derived.sector_polygons) else [])
            pts = [vpos[v] for v in ring if v in vpos]
            if len(pts) < 3:
                continue
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            d = (cx - start.x) ** 2 + (cy - start.y) ** 2
            if best is None or d < best:
                best, ss = d, s.id
    reached, q = {ss}, deque([ss])
    while q:
        for x in nb.get(q.popleft(), ()):
            if x not in reached:
                reached.add(x)
                q.append(x)
    return reached


def apply_learned_placement(doc: MapDocument, pm: PlacementModel,
                            tags: dict[str, float], seed: int | None = None) -> dict:
    import torch

    from data.graph_builder import build_graph

    rng = np.random.default_rng(seed)
    vpos = {v.id: (v.x, v.y) for v in doc.vertices}

    reachable = _reachable_sectors(doc)
    areas, rings = {}, {}
    for s in doc.sectors:
        ring = (doc.derived.sector_polygons[s.id]
                if s.id < len(doc.derived.sector_polygons) else [])
        pts = [vpos[v] for v in ring if v in vpos]
        a, _ = _ring_area_perimeter(ring, vpos)
        if a >= 64 * 64 and (s.ceil_height - s.floor_height) >= 56:
            areas[s.id], rings[s.id] = a, pts
    place_ids = [i for i in areas if i in reachable] or list(areas)
    if not place_ids:
        return {"mode": "gnn-placement", "monsters": 0, "pickups": 0}

    g = build_graph(doc, count_things=False)
    nf = torch.from_numpy(g["node_features"][:, INPUT_COLUMNS].copy()).to(pm.device)
    ei = torch.from_numpy(g["edge_index"]).to(pm.device)
    ef = torch.from_numpy(g["edge_features"]).to(pm.device)
    cond = pm.conditioner.encode([tags], device=pm.device)[0]
    with torch.no_grad():
        out = pm.model(nf, ei, ef, cond)
    m_rate = out["monster_rate"].cpu().numpy()
    p_rate = out["pickup_rate"].cpu().numpy()
    m_probs = torch.softmax(out["monster_type"], dim=-1).cpu().numpy()
    p_probs = torch.softmax(out["pickup_type"], dim=-1).cpu().numpy()
    start = next((t for t in doc.things if t.type == 1), None)

    def place(sid, count, probs, types, min_start):
        if count <= 0 or not types:
            return 0
        pts = rings[sid]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        placed = 0
        for _ in range(count):
            for _try in range(16):
                x = rng.uniform(min(xs), max(xs))
                y = rng.uniform(min(ys), max(ys))
                if not _point_in_polygon(x, y, pts):
                    continue
                if _wall_clearance(x, y, pts) < _THING_WALL_MARGIN:
                    continue
                if start is not None and min_start > 0 and \
                        (x - start.x) ** 2 + (y - start.y) ** 2 < min_start ** 2:
                    continue
                ti = int(rng.choice(len(types), p=probs[sid]))
                _add_thing(doc, x, y, types[ti], angle=int(rng.integers(0, 8)) * 45)
                placed += 1
                break
        return placed

    mons = picks = 0
    for sid in place_ids:
        nm = int(np.clip(round(float(m_rate[sid])), 0, _MAX_PER_SECTOR))
        npk = int(np.clip(round(float(p_rate[sid])), 0, _MAX_PER_SECTOR))
        mons += place(sid, nm, m_probs, pm.monster_types, 256)
        picks += place(sid, npk, p_probs, pm.pickup_types, 0)
    return {"mode": "gnn-placement", "monsters": mons, "pickups": picks}

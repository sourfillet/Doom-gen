"""Map document -> attributed sector graph tensors (plan/generation.md,
Stage 3). Nodes are sectors; edges come from the derived sector graph
(shared two-sided linedefs plus teleport links), duplicated in both
directions for message passing.

Feature columns are listed in NODE_FEATURES / EDGE_FEATURES so heads can
be wired by name, not index arithmetic.
"""
from __future__ import annotations

import math

import numpy as np

from wad_parser.map_extractor import _sector_containing
from wad_parser.types import MapDocument

NODE_FEATURES = [
    "floor_h_abs",      # /1024, clipped to [-1,1] then mapped to [0,1]
    "ceil_h_abs",
    "room_height",      # (ceil-floor)/1024 clipped
    "light",            # /255
    "has_special",
    "has_tag",
    "area",             # polygon area / map bbox area
    "perimeter",        # polygon perimeter / map bbox diagonal
    "degree",           # adjacency degree / 8, clipped
    "n_monsters",       # per-sector thing counts, min(n,8)/8
    "n_pickups",        # weapons+ammo+items
    "n_keys",
    "n_starts",         # player/deathmatch starts
]

EDGE_FEATURES = [
    "is_walk",
    "is_teleport",
    "link_length",      # shared linedef length / map bbox diagonal (0 for teleports)
    "floor_step",       # |floor_a - floor_b| / 128, clipped - stairs vs cliffs
]

_PICKUP_CATEGORIES = {"weapon", "ammo", "item"}
_START_CATEGORIES = {"player-start", "deathmatch-start"}


def _ring_area_perimeter(ring, vpos) -> tuple[float, float]:
    pts = [vpos[v] for v in ring if v in vpos]
    if len(pts) < 3:
        return 0.0, 0.0
    area = 0.0
    perim = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
        perim += math.hypot(x2 - x1, y2 - y1)
    return abs(area) / 2.0, perim


def build_graph(doc: MapDocument) -> dict[str, np.ndarray]:
    """Returns {"node_features": (N, F), "edge_index": (2, 2E), and
    "edge_features": (2E, Fe)} - every undirected link appears once per
    direction."""
    b = doc.bounds
    bbox_area = max((b.max_x - b.min_x) * (b.max_y - b.min_y), 1.0)
    diag = max(math.hypot(b.max_x - b.min_x, b.max_y - b.min_y), 1.0)
    vpos = {v.id: (v.x, v.y) for v in doc.vertices}
    n = len(doc.sectors)

    # per-sector thing counts
    counts = {s.id: {"monster": 0, "pickup": 0, "key": 0, "start": 0} for s in doc.sectors}
    for t in doc.things:
        sec = _sector_containing(t.x, t.y, doc.sectors, doc.derived.sector_polygons, doc.vertices)
        if sec is None or sec not in counts:
            continue
        if t.category == "monster":
            counts[sec]["monster"] += 1
        elif t.category in _PICKUP_CATEGORIES:
            counts[sec]["pickup"] += 1
        elif t.category == "key":
            counts[sec]["key"] += 1
        elif t.category in _START_CATEGORIES:
            counts[sec]["start"] += 1

    degree = {s.id: 0 for s in doc.sectors}
    for e in doc.derived.sector_graph.get("edges", []):
        degree[e["a"]] = degree.get(e["a"], 0) + 1
        degree[e["b"]] = degree.get(e["b"], 0) + 1

    def _clip01(x):
        return float(np.clip(x, 0.0, 1.0))

    node_features = np.zeros((n, len(NODE_FEATURES)), dtype=np.float32)
    for s in doc.sectors:
        ring = (doc.derived.sector_polygons[s.id]
                if s.id < len(doc.derived.sector_polygons) else [])
        area, perim = _ring_area_perimeter(ring, vpos)
        c = counts[s.id]
        node_features[s.id] = [
            float(np.clip(s.floor_height / 1024.0, -1, 1)) * 0.5 + 0.5,
            float(np.clip(s.ceil_height / 1024.0, -1, 1)) * 0.5 + 0.5,
            _clip01((s.ceil_height - s.floor_height) / 1024.0),
            _clip01(s.light_level / 255.0),
            1.0 if s.special else 0.0,
            1.0 if s.tag else 0.0,
            _clip01(area / bbox_area),
            _clip01(perim / diag),
            _clip01(degree[s.id] / 8.0),
            _clip01(c["monster"] / 8.0),
            _clip01(c["pickup"] / 8.0),
            _clip01(c["key"] / 8.0),
            _clip01(c["start"] / 8.0),
        ]

    linedef_by_id = {ld.id: ld for ld in doc.linedefs}
    src, dst, eattr = [], [], []
    for e in doc.derived.sector_graph.get("edges", []):
        a, b_ = e["a"], e["b"]
        if a >= n or b_ >= n:
            continue
        is_walk = 1.0 if e.get("kind") == "walk" else 0.0
        is_tp = 1.0 if e.get("kind") == "teleport" else 0.0
        length = 0.0
        ld = linedef_by_id.get(e.get("linedef"))
        if is_walk and ld and ld.v1 in vpos and ld.v2 in vpos:
            (x1, y1), (x2, y2) = vpos[ld.v1], vpos[ld.v2]
            length = _clip01(math.hypot(x2 - x1, y2 - y1) / diag)
        step = _clip01(abs(doc.sectors[a].floor_height - doc.sectors[b_].floor_height) / 128.0)
        for u, v in ((a, b_), (b_, a)):
            src.append(u)
            dst.append(v)
            eattr.append([is_walk, is_tp, length, step])

    return {
        "node_features": node_features,
        "edge_index": np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), dtype=np.int64),
        "edge_features": np.array(eattr, dtype=np.float32) if eattr else np.zeros((0, len(EDGE_FEATURES)), dtype=np.float32),
    }

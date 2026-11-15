"""Map document -> attributed sector graph tensors (Stage 3). Nodes are sectors; edges come from the derived sector graph
(shared two-sided linedefs plus teleport links), duplicated in both
directions for message passing.

Feature columns are listed in NODE_FEATURES / EDGE_FEATURES so heads can
be wired by name, not index arithmetic.
"""
from __future__ import annotations

import math

import numpy as np

from wad_parser.geometry import sector_geometry
from wad_parser.map_extractor import _sector_containing
from wad_parser.traversal import distances_from_start
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
    "absolute_area_log", # log2(area) / 24
    "absolute_perimeter_log", # log2(perimeter) / 16
    "absolute_width",    # bounding width / 4096
    "absolute_depth",    # bounding depth / 4096
    "start_distance",    # directed graph distance / 64 (1 for unreachable)
]

EDGE_FEATURES = [
    "is_walk",
    "is_teleport",
    "link_length",      # shared linedef length / map bbox diagonal (0 for teleports)
    "floor_step",       # |floor_a - floor_b| / 128, clipped - stairs vs cliffs
]

_PICKUP_CATEGORIES = {"weapon", "ammo", "item"}
_START_CATEGORIES = {"player-start", "deathmatch-start"}


def build_graph(doc: MapDocument, count_things: bool = True) -> dict[str, np.ndarray]:
    """Returns {"node_features": (N, F), "edge_index": (2, 2E), and
    "edge_features": (2E, Fe)} - every undirected link appears once per
    direction.

    count_things=False leaves the per-sector thing-count columns (9-12) at
    zero and skips the point-in-sector test for every thing - a big speedup
    for callers (Stage 3) that only read geometry columns."""
    b = doc.bounds
    bbox_area = max((b.max_x - b.min_x) * (b.max_y - b.min_y), 1.0)
    diag = max(math.hypot(b.max_x - b.min_x, b.max_y - b.min_y), 1.0)
    vpos = {v.id: (v.x, v.y) for v in doc.vertices}
    n = len(doc.sectors)
    from wad_parser.components import ComponentIndex
    geom = sector_geometry(doc)
    geometry = ComponentIndex(vpos, geom)

    # per-sector thing counts
    counts = {s.id: {"monster": 0, "pickup": 0, "key": 0, "start": 0} for s in doc.sectors}
    for t in (doc.things if count_things else ()):
        sec = _sector_containing(t.x, t.y, doc.sectors, geom, doc.vertices, locator=geometry)
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
        a, b_ = e["a"], e["b"]
        if not (0 <= a < n and 0 <= b_ < n):
            continue
        degree[a] = degree.get(a, 0) + 1
        degree[b_] = degree.get(b_, 0) + 1

    def _clip01(x):
        return float(np.clip(x, 0.0, 1.0))

    distances = distances_from_start(doc)
    node_features = np.zeros((n, len(NODE_FEATURES)), dtype=np.float32)
    for s in doc.sectors:
        area, perim = geometry.sector_areas[s.id], geometry.sector_perimeters[s.id]
        c = counts[s.id]
        box = geometry.bounds.get(s.id)
        width = (box[2] - box[0]) if box is not None else 0.0
        depth = (box[3] - box[1]) if box is not None else 0.0
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
            _clip01(math.log2(max(area, 1)) / 24),
            _clip01(math.log2(max(perim, 1)) / 16),
            _clip01(width / 4096), _clip01(depth / 4096),
            _clip01(distances.get(s.id, 64) / 64),
        ]

    linedef_by_id = {ld.id: ld for ld in doc.linedefs}
    src, dst, eattr = [], [], []
    for e in doc.derived.sector_graph.get("edges", []):
        a, b_ = e["a"], e["b"]
        if not (0 <= a < n and 0 <= b_ < n):
            continue
        is_walk = 1.0 if e.get("kind") == "walk" else 0.0
        is_tp = 1.0 if e.get("kind") == "teleport" else 0.0
        length = 0.0
        ld = linedef_by_id.get(e.get("linedef"))
        if is_walk and ld and ld.v1 in vpos and ld.v2 in vpos:
            (x1, y1), (x2, y2) = vpos[ld.v1], vpos[ld.v2]
            length = _clip01(math.hypot(x2 - x1, y2 - y1) / diag)
        step = _clip01(abs(doc.sectors[a].floor_height - doc.sectors[b_].floor_height) / 128.0)
        for u, v in (((a, b_),) if e.get("directed") else ((a, b_), (b_, a))):
            src.append(u)
            dst.append(v)
            eattr.append([is_walk, is_tp, length, step])

    return {
        "node_features": node_features,
        "edge_index": np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), dtype=np.int64),
        "edge_features": np.array(eattr, dtype=np.float32) if eattr else np.zeros((0, len(EDGE_FEATURES)), dtype=np.float32),
    }

"""Training targets for the autoregressive room-graph generator (Phase 9.4).

A RoomGraph is linearized into a DEPTH-FIRST-ordered sequence rooted at the
largest room. We train on the SPANNING TREE (each non-root room has exactly
one parent) so generated graphs are connected by construction; non-tree edges
are dropped for v1. DFS (not BFS) matters: a BFS tree makes the root the
parent of ALL its neighbours, so even a non-star map linearizes to a root-heavy
"bushy" tree and the model learns to generate STARS. A DFS tree is deep and
chain-like - each room's parent is the room you just came from - which is what
organic Doom progression actually looks like (Phase 10.3 de-star fix). Room
attributes are binned into small categorical vocabs (floor step relative to
parent, room height, size class, aspect), far more learnable on ~1370 graphs
than raw regression.
"""
from __future__ import annotations

import math

from roomgraph.types import Connection, Room, RoomGraph

# --- attribute bins (index i in a head <-> value here) ---
FLOOR_STEPS = [-256, -128, -64, -24, 0, 24, 64, 128, 256]   # relative to parent
HEIGHTS = [72, 96, 128, 160, 192, 256, 384]                  # ceil - floor
SIZES = [256, 384, 512, 640]                                 # base room dim
ASPECTS = ["tall", "square", "wide"]
KINDS = ["open", "corridor", "door", "teleport", "none"]     # "none" = root, no edge

N_FLOOR, N_HEIGHT, N_SIZE, N_ASPECT, N_KIND = (
    len(FLOOR_STEPS), len(HEIGHTS), len(SIZES), len(ASPECTS), len(KINDS))
ROOT_FLOOR_IDX = FLOOR_STEPS.index(0)
NONE_KIND_IDX = KINDS.index("none")


def _nearest(value: float, options: list) -> int:
    return min(range(len(options)), key=lambda i: abs(options[i] - value))


def _size_idx(area: float) -> int:
    s = math.sqrt(max(area, 1.0))
    return 0 if s < 400 else 1 if s < 700 else 2 if s < 1100 else 3


def _aspect_idx(w: float, h: float) -> int:
    r = (w / h) if h > 0 else 1.0
    return 0 if r < 0.8 else 1 if r < 1.4 else 2


def graph_to_sequence(rg: RoomGraph) -> list[dict] | None:
    """DFS spanning tree from the largest room -> a list of per-room steps:
    {floor, height, size, aspect, parent, kind}. parent is the index (in
    emission order) of the parent room; root has parent -1, kind 'none'.
    Depth-first keeps parents recent (deep chains, not root-heavy stars)."""
    if not rg.rooms:
        return None
    by_id = {r.id: r for r in rg.rooms}
    adj: dict[int, list[tuple[int, str]]] = {r.id: [] for r in rg.rooms}
    for c in rg.connections:
        if c.a in adj and c.b in adj:
            adj[c.a].append((c.b, c.kind))
            adj[c.b].append((c.a, c.kind))
    root = max(rg.rooms, key=lambda r: r.area).id

    # iterative DFS: emit a room when popped (its parent is already emitted, so
    # parent index < child index as the autoregressive model requires)
    order: list[int] = []
    pos: dict[int, int] = {}
    parent_of = {root: (-1, "none")}
    visited = {root}
    stack = [root]
    while stack:
        cur = stack.pop()
        pos[cur] = len(order)
        order.append(cur)
        # push reverse-sorted so the smallest neighbour is popped/emitted first
        for nb, kind in sorted(adj[cur], reverse=True):
            if nb not in visited:
                visited.add(nb)
                parent_of[nb] = (cur, kind)
                stack.append(nb)

    seq = []
    for rid in order:
        r = by_id[rid]
        pid, kind = parent_of[rid]
        w = r.bbox[2] - r.bbox[0]
        h = r.bbox[3] - r.bbox[1]
        if pid == -1:
            floor_i = ROOT_FLOOR_IDX
            kind_i = NONE_KIND_IDX
            parent = -1
        else:
            floor_i = _nearest(r.floor_h - by_id[pid].floor_h, FLOOR_STEPS)
            kind_i = KINDS.index(kind) if kind in KINDS else 0
            parent = pos[pid]
        seq.append({
            "floor": floor_i,
            "height": _nearest(r.ceil_h - r.floor_h, HEIGHTS),
            "size": _size_idx(r.area),
            "aspect": _aspect_idx(w, h),
            "parent": parent,
            "kind": kind_i,
        })
    return seq


def sequence_to_graph(seq: list[dict], map_name: str = "MAP01") -> RoomGraph:
    """Inverse of graph_to_sequence: rebuild a RoomGraph (rooms with concrete
    bbox/area/floor/ceil + tree connections) a generated sequence describes."""
    rooms: list[Room] = []
    conns: list[Connection] = []
    floors: list[int] = []
    for i, step in enumerate(seq):
        base = SIZES[step["size"]]
        asp = ASPECTS[step["aspect"]]
        w, h = ((base * 0.7, base * 1.3) if asp == "tall"
                else (base * 1.3, base * 0.7) if asp == "wide" else (base, base))
        parent = step["parent"]
        floor = (0 if parent < 0
                 else floors[parent] + FLOOR_STEPS[step["floor"]])
        floors.append(floor)
        ceil = floor + HEIGHTS[step["height"]]
        rooms.append(Room(id=i, sectors=[], floor_h=floor, ceil_h=ceil,
                          area=w * h, centroid=(0.0, 0.0),
                          bbox=(0.0, 0.0, w, h)))
        if parent >= 0:
            kind = KINDS[step["kind"]] if step["kind"] < NONE_KIND_IDX else "open"
            conns.append(Connection(a=parent, b=i, kind=kind, width=192))
    return RoomGraph(rooms=rooms, connections=conns, map_name=map_name)

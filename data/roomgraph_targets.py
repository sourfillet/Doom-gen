"""Versioned DFS room targets, rooted at a start room when available.

A spanning tree supplies parent pointers; version 3 also retains non-tree
connections, widths, and start/exit/key roles. World-unit attribute bins are
shared by encoding and decoding. Older checkpoint size bins remain explicit.
"""
from __future__ import annotations

import math

from roomgraph.types import Connection, Room, RoomGraph

# Version 2 fixes parent assignment on cyclic source graphs. Tensor shapes and
# inference remain compatible; rebuilding targets changes the training labels.
ROOMGRAPH_TARGET_VERSION = 3

# --- attribute bins (index i in a head <-> value here) ---
FLOOR_STEPS = [-256, -128, -64, -24, 0, 24, 64, 128, 256]   # relative to parent
HEIGHTS = [72, 96, 128, 160, 192, 256, 384]                  # ceil - floor
LEGACY_SIZES = [256, 384, 512, 640]
SIZES = [256, 512, 1024, 2048]                             # world-unit room dimension
WIDTHS = [64, 128, 192, 256]
ROLE_BITS = {"start": 1, "exit": 2, "key": 4}
ASPECTS = ["tall", "square", "wide"]
KINDS = ["open", "corridor", "door", "teleport", "none"]     # "none" = root, no edge

N_FLOOR, N_HEIGHT, N_SIZE, N_ASPECT, N_KIND = (
    len(FLOOR_STEPS), len(HEIGHTS), len(SIZES), len(ASPECTS), len(KINDS))
ROOT_FLOOR_IDX = FLOOR_STEPS.index(0)
NONE_KIND_IDX = KINDS.index("none")


def _nearest(value: float, options: list) -> int:
    return min(range(len(options)), key=lambda i: abs(options[i] - value))


def _size_idx(area: float, aspect: int = 1) -> int:
    s = math.sqrt(max(area, 1.0))
    if aspect != 1:
        s /= math.sqrt(0.7 * 1.3)
    return _nearest(s, SIZES)


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
    starts = [r for r in rg.rooms if r.role == "start" or "start" in r.roles]
    root = max(starts or rg.rooms, key=lambda r: r.area).id

    # Stack frames retain the neighbor iterator, like recursive DFS: discover
    # only one child before descending. Reserving every sibling up front makes
    # cycles into root-heavy stars instead of depth-first spanning trees.
    order: list[int] = []
    pos: dict[int, int] = {}
    parent_of = {root: (-1, "none")}
    visited = {root}
    order.append(root)
    pos[root] = 0
    stack = [(root, iter(sorted(adj[root])))]
    while stack:
        cur, neighbors = stack[-1]
        try:
            nb, kind = next(neighbors)
        except StopIteration:
            stack.pop()
            continue
        if nb not in visited:
            visited.add(nb)
            parent_of[nb] = (cur, kind)
            pos[nb] = len(order)
            order.append(nb)
            stack.append((nb, iter(sorted(adj[nb]))))

    if len(order) != len(rg.rooms):
        raise ValueError(f"disconnected room graph: only {len(order)}/{len(rg.rooms)} rooms reached")
    links = {frozenset((c.a, c.b)): c for c in rg.connections}
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
        aspect = _aspect_idx(w, h)
        roles = sum(bit for role, bit in ROLE_BITS.items() if role == r.role or role in r.roles)
        if pid == -1:
            roles |= ROLE_BITS["start"]
        link = links.get(frozenset((rid, pid)))
        extra = []
        for other, other_pos in pos.items():
            edge = links.get(frozenset((rid, other)))
            if other_pos < pos[rid] and other != pid and edge is not None:
                extra.append({"parent": other_pos, "kind": KINDS.index(edge.kind) if edge.kind in KINDS else 0,
                              "width": _nearest(edge.width, WIDTHS)})
        seq.append({
            "floor": floor_i,
            "height": _nearest(r.ceil_h - r.floor_h, HEIGHTS),
            "size": _size_idx(r.area, aspect),
            "aspect": aspect,
            "parent": parent,
            "kind": kind_i,
            "width": _nearest(link.width, WIDTHS) if link else 2,
            "roles": roles,
            "extra_edges": extra,
        })
    return seq


def sequence_to_graph(seq: list[dict], map_name: str = "MAP01",
                      target_version: int = ROOMGRAPH_TARGET_VERSION) -> RoomGraph:
    """Inverse of graph_to_sequence: rebuild a RoomGraph (rooms with concrete
    bbox/area/floor/ceil + tree connections) a generated sequence describes."""
    rooms: list[Room] = []
    conns: list[Connection] = []
    floors: list[int] = []
    for i, step in enumerate(seq):
        base = (SIZES if target_version >= 3 else LEGACY_SIZES)[step["size"]]
        asp = ASPECTS[step["aspect"]]
        w, h = ((base * 0.7, base * 1.3) if asp == "tall"
                else (base * 1.3, base * 0.7) if asp == "wide" else (base, base))
        parent = step["parent"]
        floor = (0 if parent < 0
                 else floors[parent] + FLOOR_STEPS[step["floor"]])
        floors.append(floor)
        ceil = floor + HEIGHTS[step["height"]]
        roles = [role for role, bit in ROLE_BITS.items() if step.get("roles", 0) & bit]
        rooms.append(Room(id=i, sectors=[], floor_h=floor, ceil_h=ceil,
                          area=w * h, centroid=(0.0, 0.0),
                          bbox=(0.0, 0.0, w, h), roles=roles,
                          role=roles[0] if roles else "room"))
        if parent >= 0:
            kind = KINDS[step["kind"]] if step["kind"] < NONE_KIND_IDX else "open"
            conns.append(Connection(a=parent, b=i, kind=kind, width=WIDTHS[step.get("width", 2)]))
        if target_version >= 3:
            for edge in step.get("extra_edges", []):
                if not 0 <= edge["parent"] < i:
                    raise ValueError("extra edge must point to an earlier room")
                conns.append(Connection(edge["parent"], i, KINDS[edge["kind"]], WIDTHS[edge.get("width", 2)]))
    return RoomGraph(rooms=rooms, connections=conns, map_name=map_name)

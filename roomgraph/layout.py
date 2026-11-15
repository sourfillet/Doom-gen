"""Force-directed room layout. Replaces the grid placement: rooms are
bodies, graph connections are springs pulling neighbours together, and
overlap is resolved by repulsion. The result is ORGANIC, non-grid positions - the first half of
killing the "squares on a grid" look (the other half is emitting each room's
real polygon shape instead of a rectangle).

Rooms are treated as circles (bounding radius) for the physics; the actual
polygon is translated to the solved centroid afterwards.
"""
from __future__ import annotations

import math
import random
from collections import deque

from shapely.geometry import Polygon

from roomgraph.types import RoomGraph

GAP = 288.0           # desired clear space between room circles
EDGE_SLACK = 96.0     # desired extra distance between connected rooms (corridor)


def _radius(room) -> float:
    shells = [Polygon(ring) for ring in room.polygons if len(ring) >= 3]
    if shells:
        # Assembly places the largest shell's area centroid at the solved
        # position, which need not be the bounding-box midpoint (or the stored
        # room centroid). Enclose the vertices around that same origin.
        shell = max(shells, key=lambda poly: poly.area)
        center = shell.centroid.coords[0]
        return max(64.0, max(math.dist(center, point) for point in shell.exterior.coords))
    w = room.bbox[2] - room.bbox[0]
    h = room.bbox[3] - room.bbox[1]
    return max(0.5 * math.hypot(w, h), 64.0)


def _tree(graph: RoomGraph):
    """BFS spanning tree of the (usually already-tree) graph, rooted at the
    highest-degree room so a hub sits at the centre. Returns
    (root, order, parent, children, depth)."""
    adj: dict[int, list[int]] = {r.id: [] for r in graph.rooms}
    for c in graph.connections:
        if c.a in adj and c.b in adj and c.a != c.b:
            adj[c.a].append(c.b)
            adj[c.b].append(c.a)
    ids = [r.id for r in graph.rooms]
    root = max(ids, key=lambda i: len(adj[i]))
    parent = {}
    children: dict[int, list[int]] = {i: [] for i in ids}
    depth, order, seen = {}, [], set()
    for origin in [root] + [i for i in ids if i != root]:
        if origin in seen:
            continue
        parent[origin], depth[origin] = None, 0
        order.append(origin)
        seen.add(origin)
        dq = deque([origin])
        while dq:
            u = dq.popleft()
            for v in sorted(adj[u]):
                if v not in seen:
                    seen.add(v)
                    parent[v] = u
                    children[u].append(v)
                    depth[v] = depth[u] + 1
                    order.append(v)
                    dq.append(v)
    return root, order, parent, children, depth


def radial_layout(graph: RoomGraph, seed: int = 0) -> dict[int, tuple[float, float]]:
    """Place a spanning forest in angular wedges with bounded separation.

    All graph nodes get positions, including disconnected components. This is
    a layout heuristic, not a crossing-free embedding of cyclic graphs: the
    assembler must still verify every requested corridor. Returns room centers.
    """
    rng = random.Random(seed)
    ids = [r.id for r in graph.rooms]
    if not ids:
        return {}
    rad = {r.id: _radius(r) for r in graph.rooms}
    root, order, parent, children, depth = _tree(graph)

    # subtree angular demand: a leaf asks for 1, a parent for the sum of its
    # children (so wide subtrees get proportionally more arc)
    demand = {i: 1 for i in ids}
    for u in reversed(order):
        if children[u]:
            demand[u] = max(1, sum(demand[c] for c in children[u]))

    ring = 2.4 * (sum(rad.values()) / len(ids)) + 220.0   # radius per depth level
    room_by_id = {room.id: room for room in graph.rooms}
    edge_gaps: dict[frozenset[int], float] = {}
    for edge in graph.connections:
        if edge.kind != 'teleport' and edge.a in rad and edge.b in rad and edge.a != edge.b:
            stair_run = abs(room_by_id[edge.a].floor_h - room_by_id[edge.b].floor_h) * 3 + 192
            ring = max(ring, rad[edge.a] + rad[edge.b] + stair_run)
            key = frozenset((edge.a, edge.b))
            edge_gaps[key] = max(edge_gaps.get(key, 64.0), float(stair_run))
    pos: dict[int, list] = {}
    roots = [i for i in order if parent[i] is None]
    interval = {i: (0.0, 2 * math.pi) for i in roots}
    separation = 2 * (max(depth.values()) * ring + max(rad.values())) + 64
    origins = {i: (index * separation, 0.0) for index, i in enumerate(roots)}
    for u in order:
        a0, a1 = interval.get(u, (0.0, 2 * math.pi))
        width = a1 - a0
        if parent[u] is None:
            pos[u] = list(origins[u])
        else:
            origins[u] = origins[parent[u]]
            ang = (a0 + a1) / 2 + rng.uniform(-0.16, 0.16) * width
            r = depth[u] * ring * (1.0 + rng.uniform(-0.12, 0.12))
            ox, oy = origins[u]
            pos[u] = [ox + r * math.cos(ang), oy + r * math.sin(ang)]
        tot = sum(demand[c] for c in children[u]) or 1
        cur = a0
        for c in children[u]:                       # split arc among children
            w = width * demand[c] / tot
            interval[c] = (cur, cur + w)
            cur += w
    _separate(pos, rad, min_gaps=edge_gaps)         # guarantee no room overlap
    return {i: (pos[i][0], pos[i][1]) for i in ids}


def _separate(pos: dict, rad: dict, iterations: int = 120,
              min_gaps: dict[frozenset[int], float] | None = None) -> None:
    """Push rooms apart until their bounding circles are disjoint - once the
    circles clear, the real polygons (which sit inside them) cannot overlap.
    Radial layout separates centroids by wedge but real room shapes have extent
    and adjacent-wedge rooms can still cross. In place."""
    ids = list(pos)
    for _ in range(iterations):
        disp = {i: [0.0, 0.0] for i in ids}
        moved = False
        for a in range(len(ids)):
            ia = ids[a]
            for b in range(a + 1, len(ids)):
                ib = ids[b]
                dx = pos[ia][0] - pos[ib][0]
                dy = pos[ia][1] - pos[ib][1]
                d = math.hypot(dx, dy)
                gap = min_gaps.get(frozenset((ia, ib)), 64.0) if min_gaps else 64.0
                want = rad[ia] + rad[ib] + gap      # circles clear + corridor slack
                if d < want:
                    f = (want - d) * 0.5
                    if d < 1e-9:
                        # Coincident centers have no displacement direction.
                        # A deterministic pair-specific normal breaks the tie
                        # without touching the caller's RNG.
                        angle = (a * len(ids) + b) * 2.399963229728653
                        ux, uy = math.cos(angle), math.sin(angle)
                    else:
                        ux, uy = dx / d, dy / d
                    disp[ia][0] += ux * f
                    disp[ia][1] += uy * f
                    disp[ib][0] -= ux * f
                    disp[ib][1] -= uy * f
                    moved = True
        for i in ids:
            pos[i][0] += disp[i][0]
            pos[i][1] += disp[i][1]
        if not moved:
            break
    # Finite relaxation is not a proof of convergence. Reject any remaining
    # overlap, allowing extra space for the assembler's integer XY rounding.
    if any(math.dist(pos[ia], pos[ib]) < rad[ia] + rad[ib] + 2
           for a, ia in enumerate(ids) for ib in ids[a + 1:]):
        raise ValueError('room separation did not converge within the iteration budget')


def force_layout(graph: RoomGraph, iterations: int = 400,
                 seed: int = 0) -> dict[int, tuple[float, float]]:
    """Returns {room_id: (cx, cy)} organic centroids for the rooms."""
    rng = random.Random(seed)
    ids = [r.id for r in graph.rooms]
    if not ids:
        return {}
    rad = {r.id: _radius(r) for r in graph.rooms}
    # seed positions on a loose circle so they spread, not all at the origin
    R = sum(rad.values()) / max(len(ids), 1) * len(ids) ** 0.5
    pos = {i: [R * math.cos(k) + rng.uniform(-20, 20),
               R * math.sin(k) + rng.uniform(-20, 20)]
           for k, i in ((2 * math.pi * n / len(ids), i) for n, i in enumerate(ids))}

    for it in range(iterations):
        cool = 1.0 - it / iterations
        disp = {i: [0.0, 0.0] for i in ids}

        # repulsion: only when circles are closer than they want to be
        for a in range(len(ids)):
            ia = ids[a]
            for b in range(a + 1, len(ids)):
                ib = ids[b]
                dx = pos[ia][0] - pos[ib][0]
                dy = pos[ia][1] - pos[ib][1]
                d = math.hypot(dx, dy) or 0.01
                want = rad[ia] + rad[ib] + GAP
                if d < want:
                    f = (want - d)
                    ux, uy = dx / d, dy / d
                    disp[ia][0] += ux * f
                    disp[ia][1] += uy * f
                    disp[ib][0] -= ux * f
                    disp[ib][1] -= uy * f

        # springs: pull connected rooms toward touching (+ corridor slack)
        for c in graph.connections:
            ia, ib = c.a, c.b
            if ia not in pos or ib not in pos:
                continue
            dx = pos[ia][0] - pos[ib][0]
            dy = pos[ia][1] - pos[ib][1]
            d = math.hypot(dx, dy) or 0.01
            rest = rad[ia] + rad[ib] + EDGE_SLACK
            f = (d - rest) * 0.15
            ux, uy = dx / d, dy / d
            disp[ia][0] -= ux * f
            disp[ia][1] -= uy * f
            disp[ib][0] += ux * f
            disp[ib][1] += uy * f

        max_step = 60.0 * cool + 4.0
        for i in ids:
            dx, dy = disp[i]
            dd = math.hypot(dx, dy) or 0.01
            step = min(dd, max_step)
            pos[i][0] += dx / dd * step
            pos[i][1] += dy / dd * step

    _separate(pos, rad)
    return {i: (pos[i][0], pos[i][1]) for i in ids}

"""Force-directed room layout (plan/graph-native.md, Phase 10 - real
geometry). Replaces the grid placement: rooms are bodies, graph connections
are springs pulling neighbours together, and overlap is resolved by
repulsion. The result is ORGANIC, non-grid positions - the first half of
killing the "squares on a grid" look (the other half is emitting each room's
real polygon shape instead of a rectangle).

Rooms are treated as circles (bounding radius) for the physics; the actual
polygon is translated to the solved centroid afterwards.
"""
from __future__ import annotations

import math
import random
from collections import deque

from roomgraph.types import Connection, RoomGraph

GAP = 288.0           # desired clear space between room circles
EDGE_SLACK = 96.0     # desired extra distance between connected rooms (corridor)


def _radius(room) -> float:
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
    parent = {root: None}
    children: dict[int, list[int]] = {i: [] for i in ids}
    depth = {root: 0}
    order = [root]
    seen = {root}
    dq = deque([root])
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
    # disconnected remnants (shouldn't happen for a tree) get their own roots
    for i in ids:
        if i not in seen:
            parent[i] = None
            depth[i] = 0
            order.append(i)
            seen.add(i)
    return root, order, parent, children, depth


def radial_layout(graph: RoomGraph, seed: int = 0) -> dict[int, tuple[float, float]]:
    """Non-crossing tree layout: every subtree owns a disjoint angular wedge,
    nodes sit on depth rings. Crossing-free BY CONSTRUCTION (so corridors don't
    get dropped -> no inaccessible rooms), with wedge-scaled jitter so it reads
    organic instead of a bullseye. Returns {room_id: (cx, cy)}."""
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
    pos: dict[int, tuple[float, float]] = {}
    interval = {root: (0.0, 2 * math.pi)}
    pos[root] = (0.0, 0.0)
    for u in order:
        a0, a1 = interval.get(u, (0.0, 2 * math.pi))
        width = a1 - a0
        if parent[u] is not None:
            ang = (a0 + a1) / 2 + rng.uniform(-0.16, 0.16) * width
            r = depth[u] * ring * (1.0 + rng.uniform(-0.12, 0.12))
            pos[u] = (r * math.cos(ang), r * math.sin(ang))
        tot = sum(demand[c] for c in children[u]) or 1
        cur = a0
        for c in children[u]:                       # split arc among children
            w = width * demand[c] / tot
            interval[c] = (cur, cur + w)
            cur += w
    return pos


def add_loops(graph: RoomGraph, positions: dict[int, tuple[float, float]],
              seed: int = 0, ratio: float = 0.5) -> int:
    """Turn the radial tree into a graph with CYCLES by connecting angular
    neighbours on the same depth ring - the natural "circle around" routes a
    radial layout affords. These chords are short and tangential, so they carve
    cleanly and any that would cross are dropped later by the assembler.
    Mutates graph.connections; returns the number of loops added."""
    ids = [i for i in positions]
    if len(ids) < 4:
        return 0
    rng = random.Random(seed)
    rad = {r.id: _radius(r) for r in graph.rooms}
    ring = 2.4 * (sum(rad.values()) / len(rad)) + 220.0
    have = {frozenset((c.a, c.b)) for c in graph.connections}

    # group rooms by depth ring (radius from the layout centre), order by angle
    rings: dict[int, list] = {}
    for i in ids:
        x, y = positions[i]
        r = math.hypot(x, y)
        rings.setdefault(round(r / ring), []).append((math.atan2(y, x), i))

    cand = []
    for k, members in rings.items():
        if len(members) < 2:
            continue
        members.sort()
        m = len(members)
        wrap = m if m > 2 else m - 1          # close the full ring only if >2
        for t in range(wrap):
            a = members[t][1]
            b = members[(t + 1) % m][1]
            if a == b or frozenset((a, b)) in have:
                continue
            d = math.hypot(positions[a][0] - positions[b][0],
                           positions[a][1] - positions[b][1])
            if d > 2.4 * ring:                # skip long stringy loop chords
                continue
            cand.append((d, a, b))
    cand.sort()                                # shortest chords first
    budget = max(1, int(len(rings) * ratio) + len(ids) // 6)
    added = 0
    for d, a, b in cand:
        if added >= budget:
            break
        if frozenset((a, b)) in have:
            continue
        graph.connections.append(Connection(a=a, b=b, kind="corridor", width=160))
        have.add(frozenset((a, b)))
        added += 1
    return added


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

    return {i: (pos[i][0], pos[i][1]) for i in ids}

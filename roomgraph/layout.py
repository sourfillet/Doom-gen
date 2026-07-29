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

from roomgraph.types import RoomGraph

GAP = 288.0           # desired clear space between room circles
EDGE_SLACK = 96.0     # desired extra distance between connected rooms (corridor)


def _radius(room) -> float:
    w = room.bbox[2] - room.bbox[0]
    h = room.bbox[3] - room.bbox[1]
    return max(0.5 * math.hypot(w, h), 64.0)


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

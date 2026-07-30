"""Corridor geometry between force-placed real room polygons (Phase 10.2b).

The hard half of "real geometry": once rooms sit at organic positions with
their real shapes, connected rooms are still islands. This module solves, for
one connection, *where* a straight corridor punches through each room's wall:

  1. cast a ray from room A's interior toward room B, find where it EXITS A's
     boundary (an edge + the point on it), and symmetrically where it exits B;
  2. widen each crossing into an OPENING span along that wall edge (corridor
     width, clamped to the edge);
  3. hand back the four opening endpoints so the assembler can carve the walls
     and drop a corridor sector between them.

Pure geometry, no WAD types - the assembler owns emission and winding.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_EPS = 1e-6
CORR_HALF = 72.0        # half corridor width (144u total) - one player wide+
MIN_OPENING = 48.0      # skip a corridor whose carved opening is thinner than this
                        # (player radius 16 -> 32u min pass; 48 keeps a margin)


@dataclass
class Opening:
    edge: int                       # index i: span lies on poly[i] -> poly[i+1]
    d0: float                       # distance from poly[i] to opening start (<= d1)
    d1: float                       # distance from poly[i] to opening end
    p0: tuple[float, float]         # opening start point (nearer poly[i])
    p1: tuple[float, float]         # opening end point   (nearer poly[i+1])


@dataclass
class CorridorPlan:
    a: Opening
    b: Opening


def _ray_exit(origin, target, poly):
    """First boundary crossing of the ray origin->target (going outward).

    Returns (edge_index, cross_point, dist_from_edge_start) or None. Uses the
    smallest positive ray parameter so, for an interior origin, it is the exit.
    """
    ox, oy = origin
    dx, dy = target[0] - ox, target[1] - oy
    dlen = math.hypot(dx, dy)
    if dlen < _EPS:
        return None
    best = None
    n = len(poly)
    for i in range(n):
        px, py = poly[i]
        qx, qy = poly[(i + 1) % n]
        ex, ey = qx - px, qy - py
        denom = dx * ey - dy * ex
        if abs(denom) < _EPS:
            continue                        # ray parallel to edge
        # origin + t*d = p + s*e  ->  solve for t (ray) and s (edge)
        t = ((px - ox) * ey - (py - oy) * ex) / denom
        s = ((px - ox) * dy - (py - oy) * dx) / denom
        if t <= _EPS or s < -_EPS or s > 1 + _EPS:
            continue
        if best is None or t < best[0]:
            elen = math.hypot(ex, ey)
            best = (t, i, (ox + t * dx, oy + t * dy), s * elen)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _opening(poly, edge, dist_from_start, half=CORR_HALF):
    """Widen a crossing at `dist_from_start` along edge into a clamped span.

    Lenient on short edges: if the whole edge is shorter than the ideal opening
    we use as much of it as we can (down to a hard passable floor) rather than
    giving up - a narrow doorway still connects the rooms."""
    n = len(poly)
    px, py = poly[edge]
    qx, qy = poly[(edge + 1) % n]
    ex, ey = qx - px, qy - py
    L = math.hypot(ex, ey)
    floor = min(MIN_OPENING, 40.0)                 # hard "still walkable" width
    if L < floor:
        return None
    ux, uy = ex / L, ey / L
    d0 = max(0.0, dist_from_start - half)
    d1 = min(L, dist_from_start + half)
    if d1 - d0 < MIN_OPENING:                      # slide the window inward
        want = min(2 * half, L)
        d0 = min(max(dist_from_start - want / 2, 0.0), L - want)
        d1 = d0 + want
    if d1 - d0 < floor:
        return None
    p0 = (px + ux * d0, py + uy * d0)
    p1 = (px + ux * d1, py + uy * d1)
    return Opening(edge=edge, d0=d0, d1=d1, p0=p0, p1=p1)


def _closest_on_segment(p, a, b):
    """Closest point on segment a-b to point p, and its distance-from-a param."""
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    L2 = dx * dx + dy * dy
    if L2 < _EPS:
        return a, 0.0
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2
    t = max(0.0, min(1.0, t))
    return (ax + t * dx, ay + t * dy), t


def _closest_boundary_point(poly, target):
    """Boundary point of `poly` closest to `target`, and the edge it lies on.
    Returns (edge_index, point). Fallback for _facing_edge."""
    best = None
    n = len(poly)
    for i in range(n):
        cp, _ = _closest_on_segment(target, poly[i], poly[(i + 1) % n])
        d = math.hypot(target[0] - cp[0], target[1] - cp[1])
        if best is None or d < best[0]:
            best = (d, i, cp)
    return (best[1], best[2]) if best else None


def _facing_edge(poly, target):
    """Pick the LONG wall that best faces `target` and open there (at target's
    projection). Real room outlines have many tiny edges; aiming at the single
    closest boundary point often lands on one too short to carve. Choosing the
    longest outward-facing wall makes the opening fit. Returns (edge, point)."""
    n = len(poly)
    cx = sum(p[0] for p in poly) / n
    cy = sum(p[1] for p in poly) / n
    best = None
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        ex, ey = q[0] - p[0], q[1] - p[1]
        L = math.hypot(ex, ey)
        if L < MIN_OPENING:
            continue
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        nx, ny = -ey, ex                            # a perpendicular
        if nx * (mx - cx) + ny * (my - cy) < 0:     # orient it OUTWARD
            nx, ny = -nx, -ny
        nlen = math.hypot(nx, ny) or 1.0
        tx, ty = target[0] - mx, target[1] - my
        tlen = math.hypot(tx, ty) or 1.0
        facing = (nx * tx + ny * ty) / (nlen * tlen)   # cos(normal, dir-to-target)
        if facing <= 0.05:                          # wall faces away
            continue
        score = facing + 0.0004 * L - 0.0003 * tlen    # face target, long, near
        if best is None or score > best[0]:
            cp, _ = _closest_on_segment(target, p, q)
            best = (score, i, cp)
    if best is not None:
        return best[1], best[2]
    return _closest_boundary_point(poly, target)    # spiky room: best effort


def _dist_along(poly, edge, pt):
    n = len(poly)
    px, py = poly[edge]
    qx, qy = poly[(edge + 1) % n]
    ex, ey = qx - px, qy - py
    L = math.hypot(ex, ey) or 1.0
    return ((pt[0] - px) * ex + (pt[1] - py) * ey) / L


def plan_corridor(origin_a, poly_a, origin_b, poly_b) -> CorridorPlan | None:
    """Plan the opening on each room by carving on the wall each room presents
    toward the OTHER room's interior point (origin). Robust on concave shapes
    and short edges, where centroid ray-casting failed (dropped corridors)."""
    ba = _facing_edge(poly_a, origin_b)
    bb = _facing_edge(poly_b, origin_a)
    if ba is None or bb is None:
        return None
    oa = _opening(poly_a, ba[0], _dist_along(poly_a, ba[0], ba[1]))
    ob = _opening(poly_b, bb[0], _dist_along(poly_b, bb[0], bb[1]))
    if oa is None or ob is None:
        return None
    return CorridorPlan(a=oa, b=ob)


def seg_intersect(p1, p2, p3, p4) -> bool:
    """True if open segments p1p2 and p3p4 properly cross (shared endpoints ok)."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    return ((d1 > _EPS and d2 < -_EPS) or (d1 < -_EPS and d2 > _EPS)) and \
           ((d3 > _EPS and d4 < -_EPS) or (d3 < -_EPS and d4 > _EPS))


def signed_area(pts) -> float:
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return a / 2.0

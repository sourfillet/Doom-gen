"""Real-geometry assembler (Phase 10.2): emit rooms as their ACTUAL polygon
shapes at force-directed positions, with real CORRIDOR sectors carved between
connected rooms. This is the assembler that kills the boxy grid look - organic
placement (layout.force_layout) + real room outlines + carved corridors.

Each room is emitted as its largest source sector's real polygon (CW winding =
interior on the right). For every graph connection we plan a straight corridor
(corridor.plan_corridor): it finds where a ray between the two rooms exits each
wall, widens those crossings into openings, and drops a corridor sector whose
two ends are two-sided lines shared with the rooms. Room walls skip the opening
spans. Corridors that can't be realised cleanly (twisted quad, tiny edge) are
dropped; the validator teleporter-reconnects anything left isolated.
"""
from __future__ import annotations

import math

from roomgraph.corridor import plan_corridor, seg_intersect, signed_area
from roomgraph.layout import radial_layout
from roomgraph.types import RoomGraph
from wad_parser.map_extractor import compute_derived
from wad_parser.types import (
    SCHEMA_VERSION, Linedef, LinedefFlags, MapDocument, MapMeta, Sector,
    Sidedef, Thing, ThingFlags, Vertex, resolve_doomed, resolve_linedef_special,
)

WALL = "STARTAN2"
FLOOR, CEIL = "FLOOR4_8", "CEIL3_5"
DOOR_SPECIAL = 1          # DR door on the corridor mouth for "door" connections
_EPS = 1e-6


def _poly_centroid(pts):
    a = 0.0
    cx = cy = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        cr = x1 * y2 - x2 * y1
        a += cr
        cx += (x1 + x2) * cr
        cy += (y1 + y2) * cr
    if abs(a) < _EPS:
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        return (mx, my)
    a *= 0.5
    return (cx / (6 * a), cy / (6 * a))


def _largest_poly(room):
    best, best_a = None, -1.0
    for poly in room.polygons:
        if len(poly) < 3:
            continue
        a = abs(signed_area(poly))
        if a > best_a:
            best, best_a = poly, a
    return best


def _cw(pts):
    """Return the ring wound clockwise (signed area < 0 = interior on right)."""
    return list(pts) if signed_area(pts) < 0 else list(reversed(pts))


def _point_in_poly(pt, poly) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
                (x < (xj - xi) * (y - yi) / ((yj - yi) or _EPS) + xi):
            inside = not inside
        j = i
    return inside


def _interior_point(poly):
    c = _poly_centroid(poly)
    if _point_in_poly(c, poly):
        return c
    # fall back to the midpoint between the centroid and each vertex
    for vx, vy in poly:
        mid = ((c[0] + vx) / 2, (c[1] + vy) / 2)
        if _point_in_poly(mid, poly):
            return mid
    return c


def _cw_quad(a0, a1, b0, b1):
    """Order the four opening endpoints into a simple CW quad, tagged by which
    room each corner belongs to. Returns [(pt, tag), ...] in CW order, or None
    if the two openings interleave (a genuinely twisted corridor we can't
    realise as a straight quad)."""
    import math as _m
    tagged = [(a0, "A"), (a1, "A"), (b0, "B"), (b1, "B")]
    cx = sum(p[0] for p, _ in tagged) / 4
    cy = sum(p[1] for p, _ in tagged) / 4
    ordered = sorted(tagged, key=lambda t: _m.atan2(t[0][1] - cy, t[0][0] - cx))
    if signed_area([p for p, _ in ordered]) > 0:      # make it CW
        ordered = list(reversed(ordered))
    tags = [t for _, t in ordered]
    # each room's two corners must be adjacent in the ring (an opening = an edge)
    for k in range(4):
        if tags[k] == tags[(k + 1) % 4] and tags[(k + 2) % 4] == tags[(k + 3) % 4]:
            return ordered
    return None


def _complement(length, spans):
    """Sub-segments of [0, length] not covered by (d0, d1) spans."""
    segs, cur = [], 0.0
    for a, b in sorted(spans):
        if a > cur + _EPS:
            segs.append((cur, a))
        cur = max(cur, b)
    if cur < length - _EPS:
        segs.append((cur, length))
    return segs


def assemble_real(graph: RoomGraph, positions=None, seed: int = 0) -> MapDocument:
    positions = positions or radial_layout(graph, seed=seed)

    # --- one representative polygon per room, translated to its solved centroid
    polys: dict[int, list[tuple[float, float]]] = {}
    origins: dict[int, tuple[float, float]] = {}
    room_by_id = {r.id: r for r in graph.rooms}
    for r in graph.rooms:
        base = _largest_poly(r)
        if base is None or r.id not in positions:
            continue
        ring = _cw(base)
        pc = _poly_centroid(ring)
        px, py = positions[r.id]
        dx, dy = px - pc[0], py - pc[1]
        placed_ring = [(x + dx, y + dy) for x, y in ring]
        polys[r.id] = placed_ring
        # ray origin must be strictly INSIDE the room; a concave room's centroid
        # can fall outside, which makes _ray_exit pick the wrong wall (dropped
        # corridors). _interior_point guarantees an inside seed.
        origins[r.id] = _interior_point(placed_ring)

    verts: dict[tuple[int, int], int] = {}
    linedefs: list[Linedef] = []
    sidedefs: list[Sidedef] = []
    sectors: list[Sector] = []

    def vid(x, y):
        key = (int(round(x)), int(round(y)))
        return verts.setdefault(key, len(verts))

    def side(sector, wall):
        i = len(sidedefs)
        sidedefs.append(Sidedef(id=i, x_offset=0, y_offset=0,
                                upper_tex="-" if wall else WALL,
                                lower_tex="-" if wall else WALL,
                                mid_tex=WALL if wall else "-", sector=sector))
        return i

    def add_line(p1, p2, sec_r, sec_l=-1, special=0):
        two = sec_l != -1
        r = side(sec_r, wall=not two)
        li = side(sec_l, wall=False) if two else -1
        name, trig = (resolve_linedef_special(special) if special else (None, None))
        linedefs.append(Linedef(
            id=len(linedefs), v1=vid(*p1), v2=vid(*p2),
            flags=LinedefFlags(two_sided=two, impassable=not two),
            special=special, tag=0, right_sidedef=r, left_sidedef=li,
            special_name=name, trigger_type=trig))

    # room sectors keep their graph id so corridors can reference them
    sec_of: dict[int, int] = {}
    for rid in polys:
        r = room_by_id[rid]
        sid = len(sectors)
        sectors.append(Sector(id=sid, floor_height=int(r.floor_h),
                              ceil_height=int(max(r.ceil_h, r.floor_h + 64)),
                              floor_flat=FLOOR, ceil_flat=CEIL, light_level=160,
                              special=0, tag=0))
        sec_of[rid] = sid

    # --- plan + emit corridors; accumulate wall openings per (room, edge)
    openings: dict[int, dict[int, list[tuple[float, float]]]] = {
        rid: {} for rid in polys}
    laid: list[tuple[tuple, tuple]] = []          # placed corridor centrelines
    uf = {rid: rid for rid in polys}              # union-find over corridor-joined rooms

    def _find(x):
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x

    def _union(a, b):
        uf[_find(a)] = _find(b)

    def _mid(p, q):
        return ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)

    def _blocked(m0, m1, ra, rb):
        # corridor centreline must not tunnel through a third room or cross an
        # already-placed corridor (either = overlapping sectors / HOM in engine)
        for rid, poly in polys.items():
            if rid == ra or rid == rb:
                continue
            if _point_in_poly(m0, poly) or _point_in_poly(m1, poly):
                return True
            n = len(poly)
            for i in range(n):
                if seg_intersect(m0, m1, poly[i], poly[(i + 1) % n]):
                    return True
        for q0, q1 in laid:
            if seg_intersect(m0, m1, q0, q1):
                return True
        return False

    def record(rid, plan_side):
        spans = openings[rid].setdefault(plan_side.edge, [])
        for d0, d1 in spans:                       # reject overlapping carves
            if plan_side.d0 < d1 - _EPS and d0 < plan_side.d1 - _EPS:
                return False
        spans.append((plan_side.d0, plan_side.d1))
        return True

    for c in graph.connections:
        if c.kind == "teleport" or c.a not in polys or c.b not in polys:
            continue
        plan = plan_corridor(origins[c.a], polys[c.a], origins[c.b], polys[c.b])
        if plan is None:
            continue
        a0, a1 = plan.a.p0, plan.a.p1
        b0, b1 = plan.b.p0, plan.b.p1
        ring = _cw_quad(a0, a1, b0, b1)            # simple CW quad or None (twisted)
        if ring is None:
            continue
        mA, mB = _mid(a0, a1), _mid(b0, b1)
        if _blocked(mA, mB, c.a, c.b):             # tunnels a room / crosses a corridor
            continue
        if not record(c.a, plan.a) or not record(c.b, plan.b):
            continue
        laid.append((mA, mB))
        _union(c.a, c.b)
        ra, rb = room_by_id[c.a], room_by_id[c.b]
        floor = min(ra.floor_h, rb.floor_h)
        ceil = max(floor + 64, min(ra.ceil_h, rb.ceil_h))
        cs = len(sectors)
        sectors.append(Sector(id=cs, floor_height=int(floor), ceil_height=int(ceil),
                              floor_flat=FLOOR, ceil_flat=CEIL, light_level=144,
                              special=0, tag=0))
        door = DOOR_SPECIAL if c.kind == "door" else 0
        # walk the CW ring: corridor interior on the right of every edge. An edge
        # between same-room corners is that room's shared mouth (two-sided);
        # a mixed edge is a corridor side wall (one-sided).
        for k in range(4):
            (p, tp), (q, tq) = ring[k], ring[(k + 1) % 4]
            if tp == tq == "A":
                # a door acts on its BACK sector; emit reversed so the small
                # corridor (not the whole room) is the moving door sector
                if door:
                    add_line(q, p, sec_of[c.a], cs, special=door)
                else:
                    add_line(p, q, cs, sec_of[c.a])
            elif tp == tq == "B":
                add_line(p, q, cs, sec_of[c.b])
            else:
                add_line(p, q, cs)

    # --- room walls, skipping the carved opening spans
    for rid, poly in polys.items():
        n = len(poly)
        for i in range(n):
            p, q = poly[i], poly[(i + 1) % n]
            L = math.hypot(q[0] - p[0], q[1] - p[1])
            if L < _EPS:
                continue
            ux, uy = (q[0] - p[0]) / L, (q[1] - p[1]) / L
            for d0, d1 in _complement(L, openings[rid].get(i, [])):
                s = (p[0] + ux * d0, p[1] + uy * d0)
                e = (p[0] + ux * d1, p[1] + uy * d1)
                add_line(s, e, sec_of[rid])

    # --- player start in the LARGEST corridor-connected component, so the
    # player never spawns in an isolated room (the validator can't teleporter-
    # reconnect a lone single-sector island - it has no internal two-sided line)
    if polys:
        comps: dict[int, list[int]] = {}
        for rid in polys:
            comps.setdefault(_find(rid), []).append(rid)
        biggest = max(comps.values(), key=len)
        start_rid = max(biggest, key=lambda r: room_by_id[r].area)
        sx, sy = _interior_point(polys[start_rid])
    else:
        sx, sy = 0, 0
    name, cat, unres = resolve_doomed(1)
    things = [Thing(id=0, x=int(sx), y=int(sy), angle=90, type=1,
                    type_name=name, category=cat,
                    flags=ThingFlags(skill_1_2=True, skill_3=True, skill_4_5=True),
                    unresolved=unres)]

    vertex_objs = [Vertex(id=i, x=float(p[0]), y=float(p[1]))
                   for p, i in sorted(verts.items(), key=lambda kv: kv[1])]
    bounds, derived = compute_derived(vertex_objs, sectors, linedefs, sidedefs, things)
    meta = MapMeta(map_name=graph.map_name, source_wad="<assembled>", wad_hash="",
                   format="doom", is_iwad=False)
    return MapDocument(
        schema_version=SCHEMA_VERSION, meta=meta, bounds=bounds,
        vertices=vertex_objs, sectors=sectors, linedefs=linedefs,
        sidedefs=sidedefs, things=things, derived=derived,
        extra={"assembled": "real"})

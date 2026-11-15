"""Real-geometry assembler (Phase 10.2): emit rooms as their ACTUAL polygon
shapes at force-directed positions, with real CORRIDOR sectors carved between
connected rooms. This is the assembler that kills the boxy grid look - organic
placement (layout.force_layout) + real room outlines + carved corridors.

Each room is emitted as its largest source sector's real polygon (CW winding =
interior on the right). For every graph connection we plan a straight corridor
(corridor.plan_corridor): it finds where a ray between the two rooms exits each
wall, widens those crossings into openings, and drops a corridor sector whose
two ends are two-sided lines shared with the rooms. Room walls skip the opening
spans. Unrealized connections are explicitly recorded and generation rejects
the sample. Holes are retained; interior sector height detail remains reduced.
"""
from __future__ import annotations

import math

from shapely.geometry import Polygon

from roomgraph.corridor import plan_candidates, seg_intersect, signed_area
from roomgraph.layout import radial_layout
from roomgraph.types import RoomGraph
from roomgraph.building import (
    CEIL, FLOOR, WALL, GeometryBuilder,
    complement_spans, emit_corridor, largest_poly, poly_centroid,
)
from roomgraph.teleports import finalize_connections
from wad_parser.geometry import interior_point
from wad_parser.map_extractor import compute_derived
from wad_parser.types import (
    SCHEMA_VERSION, MapDocument, MapMeta, Thing, ThingFlags, Vertex, resolve_doomed,
)

_EPS = 1e-6
_MIN_STUB = 2.0
_poly_centroid = poly_centroid


def _largest_poly(room):
    return largest_poly(room.polygons)


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
                (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _interior_point(poly, room_holes=()):
    if room_holes:
        shape = Polygon(poly, room_holes)
        if not shape.is_empty:
            rp = shape.representative_point()
            return (float(rp.x), float(rp.y))
    c = _poly_centroid(poly)
    if _point_in_poly(c, poly):
        return c
    # fall back to the midpoint between the centroid and each vertex
    for vx, vy in poly:
        mid = ((c[0] + vx) / 2, (c[1] + vy) / 2)
        if _point_in_poly(mid, poly):
            return mid
    shape = Polygon(poly)
    if not shape.is_empty:
        rp = shape.representative_point()
        return (float(rp.x), float(rp.y))
    return c


def _cw_quad(a0, a1, b0, b1):
    """Order the four opening endpoints into a simple CW quad, tagged by which
    room each corner belongs to. Returns [(pt, tag), ...] in CW order, or None
    if the two openings interleave (a genuinely twisted corridor we can't
    realise as a straight quad)."""
    tagged = [(a0, "A"), (a1, "A"), (b0, "B"), (b1, "B")]
    cx = sum(p[0] for p, _ in tagged) / 4
    cy = sum(p[1] for p, _ in tagged) / 4
    ordered = sorted(tagged, key=lambda t: math.atan2(t[0][1] - cy, t[0][0] - cx))
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
    return complement_spans(0.0, length, spans, eps=_EPS)


def assemble_real(graph: RoomGraph, positions=None, seed: int = 0) -> MapDocument:
    if not graph.rooms:
        raise ValueError("cannot assemble an empty room graph")
    if positions is None:
        positions = radial_layout(graph, seed=seed)

    # --- one representative polygon per room, translated to its solved centroid
    polys: dict[int, list[tuple[float, float]]] = {}
    holes: dict[int, list] = {}
    room_shapes: dict[int, Polygon] = {}
    hole_shapes: dict[int, list[Polygon]] = {}
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
        holes[r.id] = [[(x + dx, y + dy) for x, y in hole] for hole in r.holes]
        room_shapes[r.id] = Polygon(placed_ring)
        hole_shapes[r.id] = [Polygon(h) for h in holes[r.id] if len(h) >= 3]
        # ray origin must be strictly INSIDE the room (and outside any holes);
        # a concave room's centroid can fall outside, which makes _ray_exit pick
        # the wrong wall (dropped corridors).
        origins[r.id] = _interior_point(placed_ring, holes[r.id])

    if not polys:
        raise ValueError("cannot assemble a room graph with no valid room outlines")

    builder = GeometryBuilder(wall=WALL, floor=FLOOR, ceil=CEIL)
    verts, linedefs, sidedefs, sectors = (
        builder.verts, builder.linedefs, builder.sidedefs, builder.sectors)
    add_line = builder.add_line

    # room sectors keep their graph id so corridors can reference them
    sec_of: dict[int, int] = {}
    for rid in polys:
        r = room_by_id[rid]
        sec_of[rid] = builder.new_sector(r.floor_h, r.ceil_h)

    # --- plan + emit corridors; accumulate wall openings per (room, edge)
    openings: dict[int, dict[int, list[tuple[float, float]]]] = {
        rid: {} for rid in polys}
    laid_quads: list[Polygon] = []   # placed corridor quads
    uf = {rid: rid for rid in polys}              # union-find over corridor-joined rooms

    def _find(x):
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x

    def _union(a, b):
        uf[_find(a)] = _find(b)

    def _seg_crosses_poly(p, q, poly):
        n = len(poly)
        return any(seg_intersect(p, q, poly[i], poly[(i + 1) % n]) for i in range(n))

    def _quad_blocked(quad, side_walls, ra, rb):
        quad_poly = Polygon(quad)
        if not quad_poly.is_valid or quad_poly.area < 1.0:
            return True
        # Test the whole quad against every other room and already-placed corridor
        # so fanned-out hub corridors and tangential loop chords can't overlap or
        # share rounded integer vertices with unrelated sectors.
        for rid, shape in room_shapes.items():
            if rid == ra or rid == rb:
                continue
            if quad_poly.intersection(shape).area > 1e-6 or quad_poly.distance(shape) < 1.0:
                return True
        for q_shape in laid_quads:
            if quad_poly.intersection(q_shape).area > 1e-6 or quad_poly.distance(q_shape) < 1.0:
                return True
        # Endpoint rooms share only the 1D mouth segment (zero area); any positive
        # overlap area or hole intersection means the corridor folds inward or
        # re-enters a concave wing.
        for endpoint in (ra, rb):
            if quad_poly.intersection(room_shapes[endpoint]).area > 1e-2:
                return True
            if any(quad_poly.intersects(h_shape) for h_shape in hole_shapes[endpoint]):
                return True
        for p, q in side_walls:
            if _seg_crosses_poly(p, q, polys[ra]) or _seg_crosses_poly(p, q, polys[rb]):
                return True
        return False

    def _span_free(rid, plan_side):
        poly = polys[rid]
        p, q = poly[plan_side.edge], poly[(plan_side.edge + 1) % len(poly)]
        edge_len = math.hypot(q[0] - p[0], q[1] - p[1])
        # Avoid tiny <2u wall slivers at corners or between two mouths that could
        # collapse to zero-length linedefs after integer rounding.
        if (0.0 < plan_side.d0 < _MIN_STUB) or (0.0 < edge_len - plan_side.d1 < _MIN_STUB):
            return False
        for d0, d1 in openings[rid].get(plan_side.edge, []):
            if plan_side.d0 < d1 + _MIN_STUB and d0 < plan_side.d1 + _MIN_STUB:
                return False
        return True

    def _commit_span(rid, plan_side):
        openings[rid].setdefault(plan_side.edge, []).append((plan_side.d0, plan_side.d1))

    realized = set()
    for conn_idx, c in enumerate(graph.connections):
        if c.a not in polys or c.b not in polys:
            continue
        if c.kind == "teleport":
            if c.a != c.b:
                _union(c.a, c.b)
            continue
        # try candidate wall pairs (shortest first) until one carves cleanly -
        # lets a corridor route around a concave room instead of being dropped
        for plan in plan_candidates(origins[c.a], polys[c.a], origins[c.b], polys[c.b],
                                     half_width=max(24, c.width / 2) if c.width > 0 else 72):
            ring = _cw_quad(plan.a.p0, plan.a.p1, plan.b.p0, plan.b.p1)
            if ring is None:
                continue
            quad = [p for p, _ in ring]
            side_walls = [(ring[k][0], ring[(k + 1) % 4][0]) for k in range(4)
                          if ring[k][1] != ring[(k + 1) % 4][1]]
            if _quad_blocked(quad, side_walls, c.a, c.b):
                continue
            if not _span_free(c.a, plan.a) or not _span_free(c.b, plan.b):
                continue
            ra, rb = room_by_id[c.a], room_by_id[c.b]
            # Rotate the CW quad so its first edge is the A mouth.
            index = next(k for k in range(4) if ring[k][1] == ring[(k + 1) % 4][1] == "A")
            ordered_quad = [ring[(index + k) % 4][0] for k in range(4)]
            def new_corridor_sector(floor, ceil, closed):
                return builder.new_sector(floor, ceil, closed=closed, light=144)
            try:
                emit_corridor(ordered_quad, sec_of[c.a], sec_of[c.b], ra.floor_h, rb.floor_h,
                              ra.ceil_h, rb.ceil_h, c.kind == "door", new_corridor_sector, add_line)
            except ValueError:
                continue
            _commit_span(c.a, plan.a)
            _commit_span(c.b, plan.b)
            laid_quads.append(Polygon(quad))
            _union(c.a, c.b)
            realized.add(conn_idx)
            break                                       # this connection is placed

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
                s = p if d0 <= 0.0 else (p[0] + ux * d0, p[1] + uy * d0)
                e = q if d1 >= L else (p[0] + ux * d1, p[1] + uy * d1)
                add_line(s, e, sec_of[rid])
        for hole in holes[rid]:
            # The room is outside an inner ring, so its right-facing walls
            # wind counterclockwise around the empty space.
            ring = list(reversed(_cw(hole)))
            for p, q in zip(ring, ring[1:] + ring[:1]):
                add_line(p, q, sec_of[rid])

    # --- player start in the LARGEST connected component, so the player never
    # spawns in an isolated room when a disconnected subset could not be routed
    comps: dict[int, list[int]] = {}
    for rid in polys:
        comps.setdefault(_find(rid), []).append(rid)
    biggest = max(comps.values(), key=len)
    preferred = [rid for rid in biggest
                 if room_by_id[rid].role == 'start' or 'start' in room_by_id[rid].roles]
    start_rid = max(preferred or biggest, key=lambda r: room_by_id[r].area)
    sx, sy = _interior_point(polys[start_rid], holes[start_rid])
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
    doc = MapDocument(
        schema_version=SCHEMA_VERSION, meta=meta, bounds=bounds,
        vertices=vertex_objs, sectors=sectors, linedefs=linedefs,
        sidedefs=sidedefs, things=things, derived=derived,
        extra={"assembled": "real", "room_sectors": sec_of,
               "room_roles": {r.id: r.roles or [r.role] for r in graph.rooms},
               "connection_realization": [
                   {"a": c.a, "b": c.b, "kind": c.kind,
                    "realized": idx in realized,
                    "reason": "" if idx in realized else "no collision-free corridor route"}
                   for idx, c in enumerate(graph.connections)]})

    # A polygon's centroid can be inside a hole or too close to a wall.
    point = interior_point(doc, sec_of[start_rid], clearance=16)
    if point is None:
        raise ValueError('requested start room has no clear player spawn')
    things[0].x, things[0].y = point
    doc.bounds, doc.derived = compute_derived(vertex_objs, sectors, linedefs, sidedefs, things)
    return finalize_connections(doc, sec_of)

"""Shared, framework-free sector geometry, including holes and islands.

All rings of a sector participate in even/odd filling. `sector_polygons` is
retained as a raw preview; consumers use sector_geometry/sector_points so
compiled BSP regions can override incomplete raw boundaries.
"""
from __future__ import annotations

import math
from collections import defaultdict

DERIVED_VERSION = 5


def signed_area(points):
    return sum(a[0] * b[1] - b[0] * a[1]
               for a, b in zip(points, points[1:] + points[:1])) / 2


def point_on_segment(x, y, a, b, tolerance=1e-7):
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    return (abs(dx * (y - a[1]) - dy * (x - a[0])) <= tolerance * max(length, 1)
            and min(a[0], b[0]) - tolerance <= x <= max(a[0], b[0]) + tolerance
            and min(a[1], b[1]) - tolerance <= y <= max(a[1], b[1]) + tolerance)


def point_in_ring(x, y, points, *, boundary=False):
    inside = False
    for a, b in zip(points, points[1:] + points[:1]):
        if point_on_segment(x, y, a, b):
            return boundary
        if (a[1] > y) != (b[1] > y):
            crossing = a[0] + (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1])
            if x < crossing:
                inside = not inside
    return inside


def point_in_rings(x, y, rings, *, boundary=False):
    if boundary and any(point_on_segment(x, y, a, b)
                        for ring in rings for a, b in zip(ring, ring[1:] + ring[:1])):
        return True
    return sum(point_in_ring(x, y, points) for points in rings) % 2 == 1


def ring_depths(rings):
    """Containment depth: shell/island at even depth, hole at odd depth."""
    def contained(ring, other):
        # A hole can touch its outer shell at a vertex. Probe a nonshared
        # boundary point instead of assuming the first vertex is decisive.
        candidates = list(ring) + [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                                    for a, b in zip(ring, ring[1:] + ring[:1])]
        for x, y in candidates:
            if not any(point_on_segment(x, y, a, b) for a, b in zip(other, other[1:] + other[:1])):
                return point_in_ring(x, y, other)
        return False
    return [sum(contained(ring, other) for j, other in enumerate(rings) if i != j)
            for i, ring in enumerate(rings)]


def resolve_rings(parts, positions):
    """Resolve raw vertex-ID rings or derived BSP coordinate rings."""
    if parts and isinstance(parts[0], int):
        parts = [parts]
    result = []
    for ring in parts:
        if len(ring) < 3:
            continue
        if isinstance(ring[0], int):
            if all(vertex in positions for vertex in ring):
                result.append([positions[vertex] for vertex in ring])
        else:
            result.append([tuple(point) for point in ring])
    return result


def sector_geometry(doc):
    """Effective geometry; BSP recovery never edits the authored vertex list."""
    return (doc.derived.sector_regions or doc.derived.sector_rings
            or [[ring] if ring else [] for ring in doc.derived.sector_polygons])


def rings_area_perimeter(rings):
    area = sum((-1 if depth % 2 else 1) * abs(signed_area(ring))
               for ring, depth in zip(rings, ring_depths(rings)))
    perimeter = sum(math.dist(a, b) for ring in rings
                    for a, b in zip(ring, ring[1:] + ring[:1]))
    return max(area, 0.0), perimeter


def sector_points(doc, sector_id):
    positions = {v.id: (v.x, v.y) for v in doc.vertices}
    geometry = sector_geometry(doc)
    return resolve_rings(geometry[sector_id], positions) if 0 <= sector_id < len(geometry) else []


def sector_area_perimeter(doc, sector_id):
    return rings_area_perimeter(sector_points(doc, sector_id))


def edge_clearance(x, y, points):
    best = math.inf
    for a, b in zip(points, points[1:] + points[:1]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length2 = dx * dx + dy * dy
        t = max(0, min(1, ((x - a[0]) * dx + (y - a[1]) * dy) / length2)) if length2 else 0
        best = min(best, math.hypot(x - a[0] - t * dx, y - a[1] - t * dy))
    return best


def sector_clearance(doc, sector_id, x, y):
    rings = sector_points(doc, sector_id)
    if not point_in_rings(x, y, rings):
        return -1.0
    return min((edge_clearance(x, y, ring) for ring in rings), default=-1.0)


def box_fits_sector(doc, sector_id, x, y, radius):
    """Actor radius-box does not intersect any shell/hole boundary."""
    rings = sector_points(doc, sector_id)
    if not point_in_rings(x, y, rings):
        return False
    epsilon = 1e-7
    for ring in rings:
        for a, b in zip(ring, ring[1:] + ring[:1]):
            lo, hi = 0.0, 1.0
            for axis, center in ((0, x), (1, y)):
                lower, upper = center - radius + epsilon, center + radius - epsilon
                delta = b[axis] - a[axis]
                if abs(delta) < epsilon:
                    if not lower < a[axis] < upper:
                        lo, hi = 1, 0
                        break
                else:
                    begin, end = sorted(((lower - a[axis]) / delta, (upper - a[axis]) / delta))
                    lo, hi = max(lo, begin), min(hi, end)
            if lo <= hi:
                return False
    return True


def interior_point(doc, sector_id, clearance=0.0):
    """Find a rounded interior point in any shell, checked against all holes."""
    rings = sector_points(doc, sector_id)
    candidates = []
    seed = doc.extra.get("sector_seed_points", {}).get(str(sector_id))
    if seed:
        candidates.append(tuple(seed))
    for ring, depth in zip(rings, ring_depths(rings)):
        if depth % 2:
            continue
        candidates.append((sum(x for x, _ in ring) / len(ring),
                           sum(y for _, y in ring) / len(ring)))
        xs, ys = zip(*ring)
        for i in range(1, 17):
            for j in range(1, 17):
                candidates.append((min(xs) + (max(xs) - min(xs)) * i / 17,
                                   min(ys) + (max(ys) - min(ys)) * j / 17))
    best, best_clear = None, -1.0
    for x, y in candidates:
        x, y = round(x), round(y)
        if point_in_rings(x, y, rings):
            c = min(edge_clearance(x, y, ring) for ring in rings)
            if c >= clearance and c > best_clear:
                best, best_clear = (x, y), c
    return best


def geometry_conflicts(doc):
    """Detect crossed/overlapping lines and overlapping filled sector areas."""
    positions = {v.id: (v.x, v.y) for v in doc.vertices}
    issues = []
    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    lines = doc.linedefs
    for i, first in enumerate(lines):
        a, b = positions[first.v1], positions[first.v2]
        for second in lines[i + 1:]:
            c, d = positions[second.v1], positions[second.v2]
            if (max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0])
                    or max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1])):
                continue
            ac, ad, ca, cb = cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b)
            if ac * ad < -1e-8 and ca * cb < -1e-8:
                issues.append(f"linedefs {first.id}/{second.id} cross")
            elif all(abs(value) < 1e-7 for value in (ac, ad, ca, cb)):
                axis = 0 if abs(b[0] - a[0]) >= abs(b[1] - a[1]) else 1
                overlap = min(max(a[axis], b[axis]), max(c[axis], d[axis])) - max(min(a[axis], b[axis]), min(c[axis], d[axis]))
                if overlap > 1e-7:
                    issues.append(f"linedefs {first.id}/{second.id} overlap")
            elif any(p not in (a, b) and point_on_segment(*p, a, b) for p in (c, d)) or any(
                    p not in (c, d) and point_on_segment(*p, c, d) for p in (a, b)):
                issues.append(f"linedefs {first.id}/{second.id} form an unsplit junction")
    rings = {s.id: sector_points(doc, s.id) for s in doc.sectors}
    witnesses = {s.id: interior_point(doc, s.id) for s in doc.sectors}
    for sector in doc.sectors:
        point = witnesses[sector.id]
        if point is None:
            continue
        for other in doc.sectors:
            if other.id > sector.id and point_in_rings(*point, rings[other.id]):
                issues.append(f"sectors {sector.id}/{other.id} overlap")
            # For nested sectors the smaller room's point may be the only
            # witness; also test its representative point in the larger one.
            if other.id > sector.id:
                other_point = witnesses[other.id]
                if other_point and point_in_rings(*other_point, rings[sector.id]):
                    issues.append(f"sectors {sector.id}/{other.id} overlap")
    return list(dict.fromkeys(issues))


def reconstruct_rings(vertices, sectors, linedefs, sidedefs):
    """Trace closed directed boundary cycles; diagnose and omit broken walks.

    Same-sector lines carry no sector boundary. At shared vertices, the next
    half-edge is the first counterclockwise from the reversed incoming edge,
    keeping the face on the right. No reliance on a sector's enumeration order.
    """
    vpos = {v.id: (v.x, v.y) for v in vertices}
    sector_ids = {s.id for s in sectors}
    edges = defaultdict(list)
    issues = []
    for ld in linedefs:
        if ld.v1 not in vpos or ld.v2 not in vpos:
            issues.append(f"linedef {ld.id}: invalid vertex reference")
            continue
        if not all(math.isfinite(n) for p in (vpos[ld.v1], vpos[ld.v2]) for n in p):
            issues.append(f"linedef {ld.id}: non-finite coordinate")
            continue
        sides = []
        for side in (ld.right_sidedef, ld.left_sidedef):
            if side == -1:
                sides.append(None)
            elif not 0 <= side < len(sidedefs) or sidedefs[side].sector not in sector_ids:
                issues.append(f"linedef {ld.id}: invalid sidedef/sector reference")
                sides.append(None)
            else:
                sides.append(sidedefs[side].sector)
        if ld.v1 == ld.v2 or vpos[ld.v1] == vpos[ld.v2]:
            issues.append(f"linedef {ld.id}: zero-length boundary")
            continue
        right, left = sides
        if right == left:
            continue
        if right is not None:
            edges[right].append((ld.v1, ld.v2))
        if left is not None:
            edges[left].append((ld.v2, ld.v1))
    rings = [[] for _ in sectors]
    for sid, boundary in edges.items():
        if not 0 <= sid < len(rings):
            issues.append(f"sector {sid}: non-dense sector IDs")
            continue
        outgoing = defaultdict(list)
        for i, (a, b) in enumerate(boundary):
            outgoing[a].append(i)
        unused = set(range(len(boundary)))
        while unused:
            first = min(unused)
            a, b = boundary[first]
            unused.remove(first)
            walk = [a]
            while b != walk[0]:
                if b in walk:
                    # A valid face can revisit a pinch vertex where a hole
                    # touches its shell. Peel off that closed cycle instead
                    # of discarding the entire balanced boundary walk.
                    index = walk.index(b)
                    cycle = walk[index:]
                    if len(cycle) < 3 or abs(signed_area([vpos[v] for v in cycle])) <= 1e-6:
                        break
                    rings[sid].append(cycle)
                    walk = walk[:index]
                walk.append(b)
                candidates = [i for i in outgoing[b] if i in unused]
                if not candidates:
                    break
                reverse = math.atan2(vpos[a][1] - vpos[b][1], vpos[a][0] - vpos[b][0])
                def turn(i):
                    c = boundary[i][1]
                    return (math.atan2(vpos[c][1] - vpos[b][1], vpos[c][0] - vpos[b][0]) - reverse) % (2 * math.pi)
                nxt = min(candidates, key=lambda i: (turn(i), i))
                unused.remove(nxt)
                a, b = boundary[nxt]
            if b == walk[0] and len(walk) >= 3 and abs(signed_area([vpos[v] for v in walk])) > 1e-6:
                rings[sid].append(walk)
            else:
                issues.append(f"sector {sid}: open or degenerate boundary at vertex {b}")
    return rings, issues

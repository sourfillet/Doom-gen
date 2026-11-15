"""Shared geometry builder, polygon utilities, corridor stairs, and door slabs."""
from __future__ import annotations

import math

from roomgraph.corridor import signed_area
from wad_parser.types import (
    Linedef, LinedefFlags, Sector, Sidedef, resolve_linedef_special,
)

WALL = "STARTAN2"
FLOOR, CEIL = "FLOOR4_8", "CEIL3_5"
DOOR_SPECIAL = 1


def complement_spans(lo: float, hi: float, spans, eps: float = 0.0) -> list[tuple[float, float]]:
    """Sub-segments of [lo, hi] not covered by (a, b) spans."""
    segs, cur = [], lo
    for a, b in sorted(spans):
        if a > cur + eps:
            segs.append((cur, a))
        cur = max(cur, b)
    if cur < hi - eps:
        segs.append((cur, hi))
    return segs


def poly_centroid(pts) -> tuple[float, float]:
    a = cx = cy = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        cr = x1 * y2 - x2 * y1
        a += cr
        cx += (x1 + x2) * cr
        cy += (y1 + y2) * cr
    if abs(a) < 1e-6:
        return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)
    a *= 0.5
    return (cx / (6 * a), cy / (6 * a))


def largest_poly(polys):
    best, best_a = None, -1.0
    for p in polys:
        if len(p) < 3:
            continue
        a = abs(signed_area(p))
        if a > best_a:
            best, best_a = p, a
    return best


class GeometryBuilder:
    """Shared vertex/sidedef/linedef/sector accumulator for roomgraph assemblers."""

    def __init__(self, *, wall: str = WALL, floor: str = FLOOR, ceil: str = CEIL):
        self.wall = wall
        self.floor = floor
        self.ceil = ceil
        self.verts: dict[tuple[int, int], int] = {}
        self.linedefs: list[Linedef] = []
        self.sidedefs: list[Sidedef] = []
        self.sectors: list[Sector] = []

    def vid(self, x: float, y: float) -> int:
        key = (int(round(x)), int(round(y)))
        return self.verts.setdefault(key, len(self.verts))

    def side(self, sector: int, wall: bool = True) -> int:
        i = len(self.sidedefs)
        self.sidedefs.append(Sidedef(
            id=i, x_offset=0, y_offset=0,
            upper_tex="-" if wall else self.wall,
            lower_tex="-" if wall else self.wall,
            mid_tex=self.wall if wall else "-",
            sector=sector,
        ))
        return i

    def add_line(self, p1, p2, sec_r: int, sec_l: int = -1, special: int = 0) -> None:
        two = sec_l != -1
        r = self.side(sec_r, wall=not two)
        li = self.side(sec_l, wall=False) if two else -1
        name, trig = (resolve_linedef_special(special) if special else (None, None))
        self.linedefs.append(Linedef(
            id=len(self.linedefs), v1=self.vid(*p1), v2=self.vid(*p2),
            flags=LinedefFlags(two_sided=two, impassable=not two),
            special=special, tag=0, right_sidedef=r, left_sidedef=li,
            special_name=name, trigger_type=trig,
        ))

    def new_sector(self, floor: float, ceil: float, sid: int | None = None,
                   closed: bool = False, light: int = 160) -> int:
        i = sid if sid is not None else len(self.sectors)
        self.sectors.append(Sector(
            id=i, floor_height=int(floor),
            ceil_height=int(floor if closed else max(ceil, floor + 64)),
            floor_flat=self.floor, ceil_flat=self.ceil, light_level=light,
            special=0, tag=0,
        ))
        return i


def emit_corridor(quad, room_a, room_b, floor_a, floor_b, ceil_a, ceil_b,
                  is_door, new_sector, add_line):
    """CW quad starts with its A mouth; opposite edge is the B mouth.

    Interpolate strips along both side walls. Doors occupy a 32-unit slab;
    other strips bridge floor differences in steps no higher than 24 units.
    """
    a0, a1, b0, b1 = quad
    length = min(math.dist(a0, b1), math.dist(a1, b0))
    if length < 1:
        raise ValueError("corridor has no length")
    n = max(1, math.ceil(abs(floor_b - floor_a) / 24))
    cuts = {i / n for i in range(n + 1)}
    door_interval = None
    if is_door:
        if length < 96:
            raise ValueError("corridor is too short for a dedicated door slab")
        half = 16 / length
        door_interval = (0.5 - half, 0.5 + half)
        # The door is a flat landing: reserve its length before distributing
        # stair strips. Removing arbitrary cuts could otherwise create a
        # floor jump greater than 24 on one approach.
        reserved = 2 * half
        cuts = {t * (1 - reserved) + (reserved if t > 0.5 else 0) for t in cuts}
        cuts.discard(door_interval[0])
        cuts.update(door_interval)
    cuts = sorted(cuts)
    def mix(a, b, t):
        if t <= 0.0:
            return a
        if t >= 1.0:
            return b
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
    pieces = []
    for lo, hi in zip(cuts, cuts[1:]):
        door = door_interval is not None and abs(lo - door_interval[0]) < 1e-8 and abs(hi - door_interval[1]) < 1e-8
        midpoint = (lo + hi) / 2
        if door_interval is not None:
            width = door_interval[1] - door_interval[0]
            midpoint = (midpoint if midpoint <= door_interval[0] else
                        midpoint - width if midpoint >= door_interval[1] else door_interval[0]) / (1 - width)
        # Consistent half-up rounding matters: Python's ties-to-even can turn
        # consecutive exact 24-unit rises into alternating 16/32-unit steps.
        floor = math.floor((floor_a + (floor_b - floor_a) * midpoint) / 8 + 0.5 + 1e-9) * 8
        # Equal-floor corridors retain exact authored heights (including
        # heights outside the 8-unit quantization used by the learned model).
        if floor_a == floor_b:
            floor = floor_a
        points = [mix(a0, b1, lo), mix(a1, b0, lo), mix(a1, b0, hi), mix(a0, b1, hi)]
        pieces.append((points, floor, door))
    heights = [floor_a] + [floor for _, floor, _ in pieces] + [floor_b]
    if any(abs(a - b) > 24 for a, b in zip(heights, heights[1:])):
        raise ValueError("corridor cannot bridge heights with legal steps")
    for points, floor, door in pieces:
        if floor_a != floor_b and min(math.dist(points[0], points[3]), math.dist(points[1], points[2])) < 32 - 1e-5:
            raise ValueError("corridor is too short for safe stair treads")
    ids = [new_sector(floor, floor if door else max(floor + 128, min(ceil_a, ceil_b)), door)
           for _, floor, door in pieces]
    for i, ((p0, p1, p2, p3), floor, door) in enumerate(pieces):
        sid = ids[i]
        add_line(p1, p2, sid)
        add_line(p3, p0, sid)
        if i == 0:
            add_line(p0, p1, sid, room_a)
        following = ids[i + 1] if i + 1 < len(ids) else room_b
        following_door = i + 1 < len(ids) and pieces[i + 1][2]
        if door:
            add_line(p3, p2, following, sid, special=1)
        else:
            add_line(p2, p3, sid, following, special=1 if following_door else 0)
    return ids

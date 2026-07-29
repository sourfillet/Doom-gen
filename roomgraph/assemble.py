"""Geometry assembler (plan/graph-native.md, Phase 9.2/9.3, G2+G3): a
RoomGraph -> a playable MapDocument. The graph-native analog of the
vectorizer - it turns abstract room structure into real sectors, walls,
corridors and doorways.

Layout (G2, v2): BFS-place rooms in a square grid of SLOTs; each room is a
rectangle SIZED from its own bbox/area (varied, not uniform), centred in its
slot with a gap all around. A connection between grid-adjacent rooms becomes
a real CORRIDOR sector bridging the gap between their facing walls (a door
connection gets a door special on its entrance); connections whose rooms
didn't land adjacent are left for the validator's teleporter reconnection.
So rooms, walls, corridors and doors all come from the graph by construction.
"""
from __future__ import annotations

import math

from wad_parser.map_extractor import compute_derived
from wad_parser.types import (
    SCHEMA_VERSION, Linedef, LinedefFlags, MapDocument, MapMeta, Sector,
    Sidedef, Thing, ThingFlags, Vertex, resolve_doomed, resolve_linedef_special,
)
from roomgraph.types import RoomGraph

SLOT = 768               # grid cell; rooms sit inside with a gap for corridors
CORR_W = 160             # corridor width
MIN_ROOM, MAX_ROOM = 192, SLOT - 192
WALL = "STARTAN2"
FLOOR, CEIL = "FLOOR4_8", "CEIL3_5"
DOOR_SPECIAL = 1         # DR Door (open-wait-close, repeatable)
_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _layout(graph: RoomGraph) -> dict[int, tuple[int, int]]:
    adj = graph.neighbors()
    placed: dict[int, tuple[int, int]] = {}
    occupied: dict[tuple[int, int], int] = {}

    def nearest_free(gx, gy):
        for radius in range(1, 400):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    c = (gx + dx, gy + dy)
                    if c not in occupied:
                        return c
        return (gx + 1, gy)

    for root in (r.id for r in graph.rooms):
        if root in placed:
            continue
        placed[root] = (0, 0) if not occupied else nearest_free(0, 0)
        occupied[placed[root]] = root
        queue = [root]
        while queue:
            r = queue.pop(0)
            gx, gy = placed[r]
            for nb in sorted(adj[r]):
                if nb in placed:
                    continue
                cell = next(((gx + d[0], gy + d[1]) for d in _DIRS
                             if (gx + d[0], gy + d[1]) not in occupied), None)
                if cell is None:
                    cell = nearest_free(gx, gy)
                placed[nb] = cell
                occupied[cell] = nb
                queue.append(nb)
    return placed


def _room_dims(room) -> tuple[float, float]:
    w0, h0 = room.size
    if w0 <= 0 or h0 <= 0:
        return (384.0, 384.0)
    aspect = min(max(w0 / h0, 0.45), 2.2)
    base = min(max(math.sqrt(max(room.area, 1.0)) * 0.6, MIN_ROOM), MAX_ROOM)
    w = min(max(base * math.sqrt(aspect), MIN_ROOM), MAX_ROOM)
    h = min(max(base / math.sqrt(aspect), MIN_ROOM), MAX_ROOM)
    return (w, h)


def _complement(lo, hi, gaps):
    segs, cur = [], lo
    for a, b in sorted(gaps):
        if a > cur:
            segs.append((cur, a))
        cur = max(cur, b)
    if cur < hi:
        segs.append((cur, hi))
    return segs


def assemble(graph: RoomGraph, map_name: str | None = None) -> MapDocument:
    placed = _layout(graph)
    room_by_id = {r.id: r for r in graph.rooms}
    rects: dict[int, tuple[float, float, float, float]] = {}
    for r in graph.rooms:
        gx, gy = placed[r.id]
        cx, cy = (gx + 0.5) * SLOT, (gy + 0.5) * SLOT
        w, h = _room_dims(r)
        rects[r.id] = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    verts: dict[tuple[int, int], int] = {}
    linedefs: list[Linedef] = []
    sidedefs: list[Sidedef] = []
    sectors: list[Sector] = []

    def vid(x, y):
        key = (int(round(x)), int(round(y)))
        return verts.setdefault(key, len(verts))

    def side(sector, wall=True):
        i = len(sidedefs)
        sidedefs.append(Sidedef(id=i, x_offset=0, y_offset=0,
                                upper_tex=WALL if not wall else "-",
                                lower_tex=WALL if not wall else "-",
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

    def new_sector(floor, ceil, sid=None):
        i = sid if sid is not None else len(sectors)
        sectors.append(Sector(id=i, floor_height=int(floor),
                              ceil_height=int(max(ceil, floor + 64)),
                              floor_flat=FLOOR, ceil_flat=CEIL, light_level=160,
                              special=0, tag=0))
        return i

    # room sectors keep their graph id; corridors get ids after all rooms
    for r in graph.rooms:
        new_sector(r.floor_h, r.ceil_h, sid=r.id)

    gaps: dict[int, dict[str, list]] = {
        r.id: {"E": [], "W": [], "N": [], "S": []} for r in graph.rooms}
    corridors = []   # (rect, orient, roomA, roomB, is_door)
    occupied = set(placed.values())

    for c in graph.connections:
        if c.a not in placed or c.b not in placed:
            continue
        (ax, ay), (bx, by) = placed[c.a], placed[c.b]
        dx, dy = bx - ax, by - ay
        if (dx != 0) == (dy != 0):     # diagonal (or same cell): can't route straight
            continue
        # a straight corridor is only valid if every cell between is empty
        if dx != 0:
            step = 1 if dx > 0 else -1
            if any((ax + i, ay) in occupied for i in range(step, dx, step)):
                continue
        else:
            step = 1 if dy > 0 else -1
            if any((ax, ay + i) in occupied for i in range(step, dy, step)):
                continue
        is_door = c.kind == "door"
        if dx != 0:        # horizontal corridor (may span several empty cells)
            west, east = (c.a, c.b) if dx > 0 else (c.b, c.a)
            wx1 = rects[west][2]
            ex0 = rects[east][0]
            yc = (min(rects[west][3], rects[east][3])
                  + max(rects[west][1], rects[east][1])) / 2
            lo, hi = yc - CORR_W / 2, yc + CORR_W / 2
            gaps[west]["E"].append((lo, hi))
            gaps[east]["W"].append((lo, hi))
            corridors.append(((wx1, lo, ex0, hi), "H", west, east, is_door))
        else:              # vertical corridor (may span several empty cells)
            south, north = (c.a, c.b) if dy > 0 else (c.b, c.a)
            sy1 = rects[south][3]
            ny0 = rects[north][1]
            xc = (min(rects[south][2], rects[north][2])
                  + max(rects[south][0], rects[north][0])) / 2
            lo, hi = xc - CORR_W / 2, xc + CORR_W / 2
            gaps[south]["N"].append((lo, hi))
            gaps[north]["S"].append((lo, hi))
            corridors.append(((lo, sy1, hi, ny0), "V", south, north, is_door))

    # room rectangle walls, skipping corridor openings
    for r in graph.rooms:
        x0, y0, x1, y1 = rects[r.id]
        g = gaps[r.id]
        for a, b in _complement(y0, y1, g["E"]):
            add_line((x1, b), (x1, a), r.id)
        for a, b in _complement(y0, y1, g["W"]):
            add_line((x0, a), (x0, b), r.id)
        for a, b in _complement(x0, x1, g["S"]):
            add_line((b, y0), (a, y0), r.id)
        for a, b in _complement(x0, x1, g["N"]):
            add_line((a, y1), (b, y1), r.id)

    # corridor sectors: floor/ceil bridging the two rooms
    for (cx0, cy0, cx1, cy1), orient, room_a, room_b, is_door in corridors:
        fa, fb = room_by_id[room_a], room_by_id[room_b]
        floor = min(fa.floor_h, fb.floor_h)
        ceil = max(floor + 64, min(fa.ceil_h, fb.ceil_h))
        cs = new_sector(floor, ceil)
        door = DOOR_SPECIAL if is_door else 0
        if orient == "H":            # corridor runs west(room_a)->east(room_b)
            add_line((cx0, cy1), (cx1, cy1), cs)                 # top wall
            add_line((cx1, cy0), (cx0, cy0), cs)                 # bottom wall
            add_line((cx0, cy0), (cx0, cy1), cs, room_a)         # west opening -> room_a
            add_line((cx1, cy0), (cx1, cy1), room_b, cs, special=door)  # east -> room_b
        else:                        # corridor runs south(room_a)->north(room_b)
            add_line((cx0, cy0), (cx0, cy1), cs)                 # left wall
            add_line((cx1, cy1), (cx1, cy0), cs)                 # right wall
            add_line((cx0, cy0), (cx1, cy0), room_a, cs, special=door)  # south -> room_a
            add_line((cx1, cy1), (cx0, cy1), cs, room_b)         # north -> room_b

    r0 = graph.rooms[0]
    x0, y0, x1, y1 = rects[r0.id]
    name, cat, unres = resolve_doomed(1)
    things = [Thing(id=0, x=int((x0 + x1) / 2), y=int((y0 + y1) / 2), angle=90,
                    type=1, type_name=name, category=cat,
                    flags=ThingFlags(skill_1_2=True, skill_3=True, skill_4_5=True),
                    unresolved=unres)]

    vertex_objs = [Vertex(id=i, x=float(p[0]), y=float(p[1]))
                   for p, i in sorted(verts.items(), key=lambda kv: kv[1])]
    bounds, derived = compute_derived(vertex_objs, sectors, linedefs, sidedefs, things)
    meta = MapMeta(map_name=map_name or graph.map_name, source_wad="<assembled>",
                   wad_hash="", format="doom", is_iwad=False)
    return MapDocument(
        schema_version=SCHEMA_VERSION, meta=meta, bounds=bounds,
        vertices=vertex_objs, sectors=sectors, linedefs=linedefs,
        sidedefs=sidedefs, things=things, derived=derived,
        extra={"assembled": True})

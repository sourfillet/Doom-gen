"""Sector -> room segmentation (plan/graph-native.md, Phase 9.1).

A *room* is a set of sectors a player reads as one space: adjacent sectors
joined by a WIDE, OPEN two-sided line (steps, ledges, light changes) are the
same room; a DOOR or a NARROW opening (doorway / corridor mouth) marks a
boundary between rooms. We union-find over the sector-adjacency graph the
extractor already computed, then every surviving cross-room adjacency becomes
a typed Connection.

Imperfect and heuristic by design - it only needs to yield believable
training graphs; it is not on the generation path.
"""
from __future__ import annotations

import math
from collections import defaultdict

from roomgraph.types import Connection, Room, RoomGraph
from wad_parser.types import MapDocument

# An opening at least this wide (summed over the shared boundary) with no door
# reads as "same room"; narrower than this is a doorway/corridor between rooms.
MERGE_WIDTH = 128.0
# Only merge across a step this small - a bigger floor change (ledge, platform,
# pit) marks a distinct space. Without this the whole open area of a map merges
# into ONE giant hub room, which makes every training graph a degenerate star.
MERGE_STEP = 40


def _is_door(ld) -> bool:
    return "door" in (ld.special_name or "").lower()


def _ring_area(pts: list[tuple[float, float]]) -> float:
    a = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def segment(doc: MapDocument, merge_width: float = MERGE_WIDTH) -> RoomGraph:
    vpos = {v.id: (v.x, v.y) for v in doc.vertices}
    linedef_by_id = {ld.id: ld for ld in doc.linedefs}
    sector_ids = [s.id for s in doc.sectors]
    floor_of = {s.id: s.floor_height for s in doc.sectors}

    # --- aggregate each sector-pair's shared boundary from the adjacency graph
    # pair -> {"len": float, "door": bool, "teleport": bool, "step": float}
    pairs: dict[tuple[int, int], dict] = defaultdict(
        lambda: {"len": 0.0, "door": False, "teleport": False, "step": 0.0})
    for e in doc.derived.sector_graph.get("edges", []):
        a, b = e["a"], e["b"]
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        rec = pairs[key]
        rec["step"] = abs(floor_of.get(a, 0) - floor_of.get(b, 0))
        if e.get("kind") == "teleport":
            rec["teleport"] = True
            continue
        ld = linedef_by_id.get(e.get("linedef"))
        if ld is None:
            continue
        if ld.v1 in vpos and ld.v2 in vpos:
            (x1, y1), (x2, y2) = vpos[ld.v1], vpos[ld.v2]
            rec["len"] += math.hypot(x2 - x1, y2 - y1)
        if _is_door(ld):
            rec["door"] = True

    # --- union-find: merge sectors joined by a wide, door-free opening
    parent = {s: s for s in sector_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for (a, b), rec in pairs.items():
        if a in parent and b in parent and not rec["teleport"] \
                and not rec["door"] and rec["len"] >= merge_width \
                and rec["step"] <= MERGE_STEP:
            union(a, b)

    # --- build rooms from components
    comps: dict[int, list[int]] = defaultdict(list)
    for s in sector_ids:
        comps[find(s)].append(s)
    room_of: dict[int, int] = {}
    rooms: list[Room] = []
    sector_by_id = {s.id: s for s in doc.sectors}
    for rid, secs in enumerate(comps.values()):
        polys, areas, cxs, cys = [], [], [], []
        minx = miny = math.inf
        maxx = maxy = -math.inf
        for sid in secs:
            room_of[sid] = rid
            ring = (doc.derived.sector_polygons[sid]
                    if sid < len(doc.derived.sector_polygons) else [])
            pts = [vpos[v] for v in ring if v in vpos]
            if len(pts) >= 3:
                polys.append(pts)
                a = _ring_area(pts)
                areas.append((a, sid))
                cxs.append(sum(p[0] for p in pts) / len(pts) * a)
                cys.append(sum(p[1] for p in pts) / len(pts) * a)
                for x, y in pts:
                    minx, miny = min(minx, x), min(miny, y)
                    maxx, maxy = max(maxx, x), max(maxy, y)
        total_a = sum(a for a, _ in areas) or 1.0
        # representative height = the largest sector in the room
        rep = max(areas, key=lambda t: t[0])[1] if areas else secs[0]
        rs = sector_by_id[rep]
        centroid = ((sum(cxs) / total_a, sum(cys) / total_a)
                    if areas else (0.0, 0.0))
        bbox = ((minx, miny, maxx, maxy) if areas != [] and minx != math.inf
                else (0.0, 0.0, 0.0, 0.0))
        rooms.append(Room(id=rid, sectors=secs, floor_h=rs.floor_height,
                          ceil_h=rs.ceil_height, area=total_a, centroid=centroid,
                          bbox=bbox, polygons=polys))

    # --- connections: surviving cross-room adjacencies, one per room pair
    conn_acc: dict[tuple[int, int], dict] = defaultdict(
        lambda: {"len": 0.0, "door": False, "teleport": False})
    for (a, b), rec in pairs.items():
        if a not in room_of or b not in room_of:
            continue
        ra, rb = room_of[a], room_of[b]
        if ra == rb:
            continue
        key = (min(ra, rb), max(ra, rb))
        acc = conn_acc[key]
        acc["len"] += rec["len"]
        acc["door"] = acc["door"] or rec["door"]
        acc["teleport"] = acc["teleport"] or rec["teleport"]

    connections: list[Connection] = []
    for (ra, rb), acc in conn_acc.items():
        if acc["door"]:
            kind = "door"
        elif acc["teleport"] and acc["len"] == 0.0:
            kind = "teleport"
        elif acc["len"] < merge_width:
            kind = "corridor"
        else:
            kind = "open"
        connections.append(Connection(a=ra, b=rb, kind=kind, width=acc["len"]))

    return RoomGraph(rooms=rooms, connections=connections,
                     map_name=doc.meta.map_name)

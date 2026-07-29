"""Hard-correctness checks and repairs for generated maps
(plan/generation.md, "Validation & post-processing"). Returns the list of
repairs applied so the pipeline can report what was touched; raises only
when a map is unsalvageable (no geometry at all).
"""
from __future__ import annotations

from collections import deque

from wad_parser.map_extractor import _point_in_polygon, _sector_containing
from wad_parser.types import (
    MapDocument, Thing, ThingFlags, resolve_doomed, resolve_linedef_special,
)

MIN_ROOM_HEIGHT = 56   # player height 56: anything lower is unenterable
MIN_LIGHT = 96
MAX_STEP_UP = 24       # vanilla movement: steps above 24 units are unclimbable
TELEPORT_SPECIAL = 97  # WR Teleport (walk-over, repeatable)
MIN_TELEPORT_CLEARANCE = 16  # player radius: a dest closer than this to a wall = stuck


def validate_and_repair(doc: MapDocument) -> list[str]:
    repairs: list[str] = []
    if not doc.sectors or not doc.linedefs:
        raise ValueError("generated map has no geometry")

    ceilings = lights = 0
    for s in doc.sectors:
        if s.ceil_height < s.floor_height + MIN_ROOM_HEIGHT:
            s.ceil_height = s.floor_height + MIN_ROOM_HEIGHT + 72
            ceilings += 1
        if s.light_level < MIN_LIGHT:
            s.light_level = MIN_LIGHT
            lights += 1
    if ceilings:
        repairs.append(f"raised {ceilings} ceiling(s) to enterable height")
    if lights:
        repairs.append(f"raised {lights} light level(s) to {MIN_LIGHT}")

    before = len(doc.linedefs)
    doc.linedefs = [ld for ld in doc.linedefs if ld.v1 != ld.v2]
    if len(doc.linedefs) != before:
        repairs.append(f"removed {before - len(doc.linedefs)} zero-length linedef(s)")

    if not any(t.type == 1 for t in doc.things):
        seed = _largest_sector_seed(doc)
        name, category, _ = resolve_doomed(1)
        doc.things.append(Thing(
            id=len(doc.things), x=seed[0], y=seed[1], angle=90, type=1,
            type_name=name, category=category,
            flags=ThingFlags(skill_1_2=True, skill_3=True, skill_4_5=True),
        ))
        repairs.append("inserted Player1Start")

    reached = _repair_reachability(doc, repairs)
    reached = _teleport_reconnect(doc, reached, repairs)
    disconnected = len(doc.sectors) - len(reached)
    if disconnected:
        repairs.append(f"{disconnected} sector(s) remain disconnected from the start")
    return repairs


def _repair_reachability(doc: MapDocument, repairs: list[str]) -> set[int]:
    """Plan gauntlet row "all sectors reachable from a player start":
    walking the sector graph with the 24-unit step-up rule (drops of any
    height are legal), unreachable-but-adjacent sectors get their floor
    lowered to within MAX_STEP_UP of a reachable neighbor (ceiling moved
    with it, preserving room height). Returns the reached sector set;
    void-separated components are left for _teleport_reconnect."""
    all_ids = {s.id for s in doc.sectors}
    start = next((t for t in doc.things if t.type == 1), None)
    if start is None or not doc.sectors:
        return all_ids
    start_sector = _sector_containing(start.x, start.y, doc.sectors,
                                      doc.derived.sector_polygons, doc.vertices)
    if start_sector is None:
        return all_ids

    neighbors: dict[int, set[int]] = {s.id: set() for s in doc.sectors}
    for e in doc.derived.sector_graph.get("edges", []):
        neighbors[e["a"]].add(e["b"])
        neighbors[e["b"]].add(e["a"])
    floor = {s.id: s for s in doc.sectors}

    def climbable(frm: int, to: int) -> bool:
        return floor[to].floor_height - floor[frm].floor_height <= MAX_STEP_UP

    def reach_from(seed: set[int]) -> set[int]:
        seen, q = set(seed), deque(seed)
        while q:
            cur = q.popleft()
            for nb in neighbors[cur]:
                if nb not in seen and climbable(cur, nb):
                    seen.add(nb)
                    q.append(nb)
        return seen

    lowered = 0
    reached = reach_from({start_sector})
    while True:
        frontier = [(floor[nb].floor_height - floor[r].floor_height, nb, r)
                    for r in reached for nb in neighbors[r] if nb not in reached]
        if not frontier:
            break
        step, target, via = min(frontier)
        s = floor[target]
        delta = s.floor_height - (floor[via].floor_height + MAX_STEP_UP)
        s.floor_height -= delta
        s.ceil_height -= delta
        lowered += 1
        reached = reach_from(reached | {target})

    if lowered:
        repairs.append(f"lowered {lowered} floor(s) to keep the map traversable")
    return reached


def _edge_clearance(x: float, y: float, pts: list[tuple[float, float]]) -> float:
    """Shortest distance from (x, y) to any edge of a polygon ring."""
    best = float("inf")
    n = len(pts)
    for i in range(n):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        if seg2 == 0.0:
            d2 = (x - ax) ** 2 + (y - ay) ** 2
        else:
            t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / seg2))
            px, py = ax + t * dx, ay + t * dy
            d2 = (x - px) ** 2 + (y - py) ** 2
        best = min(best, d2)
    return best ** 0.5


def _sector_interior_point(doc: MapDocument, sector_id: int) -> tuple[float, float] | None:
    """A point solidly INSIDE the sector, as far from its walls as we can
    find. A teleport destination lands the player here, so it must be inside
    the polygon (else they drop into the void -> HOM) and off the walls (else
    they materialize stuck). The recorded seed point and the polygon centroid
    are only candidates now - both can fall on or outside a boundary that
    simplification moved - not trusted blindly."""
    vpos = {v.id: (v.x, v.y) for v in doc.vertices}
    ring = (doc.derived.sector_polygons[sector_id]
            if sector_id < len(doc.derived.sector_polygons) else [])
    pts = [vpos[v] for v in ring if v in vpos]
    seed = doc.extra.get("sector_seed_points", {}).get(str(sector_id))
    if len(pts) < 3:
        return (float(seed[0]), float(seed[1])) if seed else None

    candidates: list[tuple[float, float]] = []
    if seed:
        candidates.append((float(seed[0]), float(seed[1])))
    candidates.append((sum(p[0] for p in pts) / len(pts),
                       sum(p[1] for p in pts) / len(pts)))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    n = 9
    for i in range(1, n):
        for j in range(1, n):
            candidates.append((minx + (maxx - minx) * i / n,
                               miny + (maxy - miny) * j / n))

    best, best_clear = None, -1.0
    for x, y in candidates:
        if _point_in_polygon(x, y, pts):
            c = _edge_clearance(x, y, pts)
            if c > best_clear:
                best, best_clear = (int(round(x)), int(round(y))), c
    # Refuse a landing the player can't fit into: a too-thin sector would
    # teleport them wedged in a wall (HOM/stuck). Caller skips this teleporter.
    if best is None or best_clear < MIN_TELEPORT_CLEARANCE:
        return None
    return best


def _teleport_reconnect(doc: MapDocument, reached: set[int],
                        repairs: list[str]) -> set[int]:
    """Void-separated islands can't be fixed by floor adjustment - there is
    no shared boundary. When both sides have a crossable line to spare,
    connect them with a PAIR of walk-over teleporters (in and back), the
    same normal map-design device the extractor already models as
    connectivity. Islands without an internal crossable line (a lone
    sector) are left alone and reported by the caller."""
    unreached = {s.id for s in doc.sectors} - reached
    if not unreached:
        return reached

    neighbors: dict[int, set[int]] = {s.id: set() for s in doc.sectors}
    for e in doc.derived.sector_graph.get("edges", []):
        neighbors[e["a"]].add(e["b"])
        neighbors[e["b"]].add(e["a"])

    vpos = {v.id: (v.x, v.y) for v in doc.vertices}

    def crossable_lines(sector_set: set[int]):
        """Two-sided, special-free lines inside the set, shortest first: a
        walk-over teleporter hijacks its whole line, and short lines are
        the least likely to be a main path. (Dedicated teleporter pads
        would need geometry insertion - a Stage 3 concern.)"""
        out = []
        for ld in doc.linedefs:
            if ld.special or ld.left_sidedef == -1:
                continue
            if ld.right_sidedef >= len(doc.sidedefs) or ld.left_sidedef >= len(doc.sidedefs):
                continue
            a = doc.sidedefs[ld.right_sidedef].sector
            b = doc.sidedefs[ld.left_sidedef].sector
            if a in sector_set and b in sector_set and sectors_by_id[a].tag == 0:
                (x1, y1), (x2, y2) = vpos[ld.v1], vpos[ld.v2]
                out.append(((x2 - x1) ** 2 + (y2 - y1) ** 2, ld, a))
        out.sort(key=lambda t: t[0])
        return [(ld, a) for _, ld, a in out]

    next_tag = max([s.tag for s in doc.sectors] + [ld.tag for ld in doc.linedefs]) + 1
    sectors_by_id = {s.id: s for s in doc.sectors}
    connected_islands = 0

    while True:
        components = []
        seen: set[int] = set()
        for sid in unreached:
            if sid in seen:
                continue
            comp, q = {sid}, deque([sid])
            seen.add(sid)
            while q:
                cur = q.popleft()
                for nb in neighbors[cur]:
                    if nb in unreached and nb not in seen:
                        seen.add(nb)
                        comp.add(nb)
                        q.append(nb)
            components.append(comp)

        progress = False
        for comp in components:
            island_lines = crossable_lines(comp)
            source_lines = crossable_lines(reached)
            if not island_lines or not source_lines:
                continue
            dest_sector = max(comp, key=lambda sid: len(
                doc.derived.sector_polygons[sid]) if sid < len(doc.derived.sector_polygons) else 0)
            dest_pt = _sector_interior_point(doc, dest_sector)
            return_line, return_front = island_lines[0]
            source_line, source_front = source_lines[0]
            back_pt = _sector_interior_point(doc, source_front)
            back_sector = sectors_by_id[source_front]
            if dest_pt is None or back_pt is None or back_sector.tag:
                continue

            # outbound: cross a line in the reached area -> island
            sectors_by_id[dest_sector].tag = next_tag
            source_line.special = TELEPORT_SPECIAL
            source_line.tag = next_tag
            source_line.special_name, source_line.trigger_type = \
                resolve_linedef_special(TELEPORT_SPECIAL)
            _insert_teleport_dest(doc, dest_pt)
            next_tag += 1
            # return trip: island line -> back where you came from
            back_sector.tag = next_tag
            return_line.special = TELEPORT_SPECIAL
            return_line.tag = next_tag
            return_line.special_name, return_line.trigger_type = \
                resolve_linedef_special(TELEPORT_SPECIAL)
            _insert_teleport_dest(doc, back_pt)
            next_tag += 1

            reached |= comp
            unreached -= comp
            connected_islands += 1
            progress = True
        if not progress or not unreached:
            break

    if connected_islands:
        repairs.append(f"connected {connected_islands} island(s) with teleporter pairs")
    return reached


def _insert_teleport_dest(doc: MapDocument, pt: tuple[float, float]) -> None:
    name, category, _ = resolve_doomed(14)
    doc.things.append(Thing(
        id=len(doc.things), x=pt[0], y=pt[1], angle=90, type=14,
        type_name=name, category=category,
        flags=ThingFlags(skill_1_2=True, skill_3=True, skill_4_5=True),
    ))


def _largest_sector_seed(doc: MapDocument) -> tuple[float, float]:
    """Interior point of the biggest sector: the vectorizer records a known
    interior cell per sector (rectilinear sectors can have exterior
    centroids, so a recorded seed beats a computed centroid)."""
    seeds = doc.extra.get("sector_seed_points", {})
    best_id, best_len = 0, -1
    for s in doc.sectors:
        ring = (doc.derived.sector_polygons[s.id]
                if s.id < len(doc.derived.sector_polygons) else [])
        if len(ring) > best_len:
            best_id, best_len = s.id, len(ring)
    if str(best_id) in seeds:
        x, y = seeds[str(best_id)]
        return float(x), float(y)
    vpos = {v.id: (v.x, v.y) for v in doc.vertices}
    ring = doc.derived.sector_polygons[best_id]
    pts = [vpos[v] for v in ring if v in vpos]
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

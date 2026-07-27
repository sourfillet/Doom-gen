"""Hard-correctness checks and repairs for generated maps
(plan/generation.md, "Validation & post-processing"). Returns the list of
repairs applied so the pipeline can report what was touched; raises only
when a map is unsalvageable (no geometry at all).
"""
from __future__ import annotations

from collections import deque

from wad_parser.map_extractor import _sector_containing
from wad_parser.types import MapDocument, Thing, ThingFlags, resolve_doomed

MIN_ROOM_HEIGHT = 56   # player height 56: anything lower is unenterable
MIN_LIGHT = 96
MAX_STEP_UP = 24       # vanilla movement: steps above 24 units are unclimbable


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

    _repair_reachability(doc, repairs)
    return repairs


def _repair_reachability(doc: MapDocument, repairs: list[str]) -> None:
    """Plan gauntlet row "all sectors reachable from a player start":
    walking the sector graph with the 24-unit step-up rule (drops of any
    height are legal), unreachable-but-adjacent sectors get their floor
    lowered to within MAX_STEP_UP of a reachable neighbor (ceiling moved
    with it, preserving room height). Sectors with no adjacency to the
    reachable component at all are disconnected geometry - reported, not
    invented a corridor for."""
    start = next((t for t in doc.things if t.type == 1), None)
    if start is None or not doc.sectors:
        return
    start_sector = _sector_containing(start.x, start.y, doc.sectors,
                                      doc.derived.sector_polygons, doc.vertices)
    if start_sector is None:
        return

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
    disconnected = len(doc.sectors) - len(reached)
    if disconnected:
        repairs.append(f"{disconnected} sector(s) disconnected from the start (no shared boundary)")


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

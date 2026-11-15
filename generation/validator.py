"""Separate explicit repair policies from non-mutating final validation."""
from __future__ import annotations

import math
from collections import deque

from wad_parser.geometry import geometry_conflicts, interior_point, sector_area_perimeter, sector_geometry
from wad_parser.traversal import distances_from_start, portal_widths, LOCAL_UNLOCKED_DOORS
from wad_parser.map_extractor import _sector_containing, compute_derived
from wad_parser.types import MapDocument, Thing, ThingFlags, resolve_doomed, active_single_player

MIN_ROOM_HEIGHT = 56   # player height 56: anything lower is unenterable
MIN_LIGHT = 96
MAX_STEP_UP = 24       # vanilla movement: steps above 24 units are unclimbable
TELEPORT_SPECIAL = 97  # WR Teleport (walk-over, repeatable)
MIN_TELEPORT_CLEARANCE = 16  # player radius: a dest closer than this to a wall = stuck


def validate_and_repair(doc: MapDocument, *, repair_heights=True, brighten=False,
                        repair_steps=False, reconnect=False) -> list[str]:
    """Apply explicit repair policy; traversal rewrites and brightening opt in.

    This compatibility helper returns actions, not a certificate of validity.
    Use validate_document after every decorating pass for a non-mutating report.
    """
    repairs: list[str] = []
    if not doc.sectors or not doc.linedefs:
        raise ValueError("generated map has no geometry")

    door_actions = LOCAL_UNLOCKED_DOORS | {26, 27, 28, 32, 33, 34}
    closed_doors = {doc.sidedefs[ld.left_sidedef].sector for ld in doc.linedefs
                    if (ld.special in door_actions or 'door' in (ld.special_name or '').lower())
                    and 0 <= ld.left_sidedef < len(doc.sidedefs)}
    tagged_doors = {ld.tag for ld in doc.linedefs if ld.tag
                    and 'door' in (ld.special_name or '').lower()}
    closed_doors.update(s.id for s in doc.sectors if s.tag and s.tag in tagged_doors)
    ceilings = lights = 0
    for s in doc.sectors:
        if repair_heights and s.id not in closed_doors and s.ceil_height < s.floor_height + MIN_ROOM_HEIGHT:
            s.ceil_height = s.floor_height + MIN_ROOM_HEIGHT + 72
            ceilings += 1
        if brighten and s.light_level < MIN_LIGHT:
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
        for index, line in enumerate(doc.linedefs):
            line.id = index

    if not any(t.type == 1 for t in doc.things):
        seed = _largest_sector_seed(doc)
        name, category, _ = resolve_doomed(1)
        doc.things.append(Thing(
            id=len(doc.things), x=seed[0], y=seed[1], angle=90, type=1,
            type_name=name, category=category,
            flags=ThingFlags(skill_1_2=True, skill_3=True, skill_4_5=True),
        ))
        repairs.append("inserted Player1Start")

    doc.bounds, doc.derived = compute_derived(doc.vertices, doc.sectors, doc.linedefs, doc.sidedefs, doc.things)
    reached = _repair_reachability(doc, repairs) if repair_steps else set(distances_from_start(doc))
    if reconnect:
        reached = _teleport_reconnect(doc, reached, repairs)
    doc.bounds, doc.derived = compute_derived(doc.vertices, doc.sectors, doc.linedefs, doc.sidedefs, doc.things)
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
    start = next((t for t in doc.things if t.type == 1), None)
    if start is None or not doc.sectors:
        return set()
    start_sector = _sector_containing(start.x, start.y, doc.sectors,
                                      sector_geometry(doc), doc.vertices)
    if start_sector is None:
        return set()

    neighbors: dict[int, set[int]] = {s.id: set() for s in doc.sectors}
    widths = portal_widths(doc.vertices, doc.linedefs, doc.sidedefs)
    for line in doc.linedefs:
        if widths.get(line.id, 0) < 32 or line.special:
            continue
        a, b = doc.sidedefs[line.right_sidedef].sector, doc.sidedefs[line.left_sidedef].sector
        if a in neighbors and b in neighbors:
            neighbors[a].add(b)
            neighbors[b].add(a)
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



def _sector_interior_point(doc: MapDocument, sector_id: int) -> tuple[float, float] | None:
    """A point solidly INSIDE the sector, as far from its walls as we can
    find. A teleport destination lands the player here, so it must be inside
    the polygon (else they drop into the void -> HOM) and off the walls (else
    they materialize stuck). The recorded seed point and the polygon centroid
    are only candidates now - both can fall on or outside a boundary that
    simplification moved - not trusted blindly."""
    return interior_point(doc, sector_id, clearance=MIN_TELEPORT_CLEARANCE)


def _teleport_landing_point(doc: MapDocument, sector_id: int, geometry) -> tuple[int, int] | None:
    """Find space for the arriving player, rather than the type-14 marker.

    The most central point may already contain the player or a monster. Try
    bounded alternatives without moving existing actors or reserving pickups
    as solid obstacles. Keep separate destination markers from sharing a spot.
    """
    from generation.actors import spawn_problem
    from wad_parser.bsp_geometry import component_polygons
    from wad_parser.geometry import sector_points

    sector = next((s for s in doc.sectors if s.id == sector_id), None)
    if sector is None or sector.ceil_height - sector.floor_height < MIN_ROOM_HEIGHT:
        return None
    candidates = [_sector_interior_point(doc, sector_id)]
    for _, polygon in component_polygons(sector_points(doc, sector_id)):
        x0, y0, x1, y1 = polygon.bounds
        candidates.extend((round(x0 + (x1 - x0) * i / 8), round(y0 + (y1 - y0) * j / 8))
                          for i in range(1, 8) for j in range(1, 8))
    seen = set()
    polygons = sector_geometry(doc)
    for point in candidates:
        if point is None:
            continue
        point = (round(point[0]), round(point[1]))
        if point in seen:
            continue
        seen.add(point)
        if _sector_containing(*point, doc.sectors, polygons, doc.vertices) != sector_id:
            continue
        if spawn_problem(doc, 1, *point, margin=MIN_TELEPORT_CLEARANCE, geometry=geometry):
            continue
        if any(t.type == 14 and active_single_player(t)
               and abs(point[0] - t.x) < 32 and abs(point[1] - t.y) < 32 for t in doc.things):
            continue
        return point
    return None


def _teleport_reconnect(doc: MapDocument, reached: set[int],
                        repairs: list[str]) -> set[int]:
    """Connect islands with dedicated pads, preserving existing passages.

    Trials must retain the old reachable sectors and reach the entire new
    island and both pads. No original line is converted into a teleporter:
    a repeatable teleport on a bottleneck would prevent normal passage.
    """
    from copy import deepcopy
    from roomgraph.teleports import add_teleport_pair
    from wad_parser.clearance import MapClearance

    unreached = {s.id for s in doc.sectors} - reached
    if not unreached or not reached:
        return reached
    original_ids = {s.id for s in doc.sectors}
    neighbors: dict[int, set[int]] = {s.id: set() for s in doc.sectors}
    for e in doc.derived.sector_graph.get("edges", []):
        neighbors[e["a"]].add(e["b"])
        neighbors[e["b"]].add(e["a"])
    connected_islands = 0
    while True:
        components = []
        seen: set[int] = set()
        for sid in sorted(unreached):
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
            clearance = MapClearance(doc.vertices, doc.sectors, doc.linedefs, doc.sidedefs)

            def candidates(ids):
                result = []
                for sid in sorted(ids & original_ids,
                                  key=lambda s: (-sector_area_perimeter(doc, s)[0], s)):
                    sector = doc.sectors[sid]
                    if (sector.tag or sector.special or sector.floor_slope is not None
                            or sector.ceil_slope is not None):
                        continue
                    if _teleport_landing_point(doc, sid, clearance) is not None:
                        result.append(sid)
                    if len(result) == 8:
                        break
                return result

            sources, targets = candidates(reached), candidates(comp)
            connected = False
            for source in sources:
                for target in targets:
                    trial = deepcopy(doc)
                    try:
                        pads = add_teleport_pair(trial, source, target)
                        new_reached = set(distances_from_start(trial))
                    except ValueError:
                        continue
                    if not (reached | comp | set(pads)) <= new_reached:
                        continue
                    doc.__dict__.update(trial.__dict__)
                    reached = new_reached
                    unreached = {s.id for s in doc.sectors} - reached
                    connected_islands += 1
                    progress = connected = True
                    break
                if connected:
                    break
        if not progress or not unreached:
            break

    if connected_islands:
        repairs.append(f"connected {connected_islands} island(s) with teleporter pairs")
    return reached


def validate_document(doc: MapDocument, *, require_exit=True, require_connected=False) -> dict:
    """Non-mutating validation of a finished Doom document.

    Movement is conservative potential traversal. Supported door/key histories
    retain location across irreversible branches. Timing, scripts, one-shot
    teleport history and moving-floor state remain unsupported.
    """
    from copy import copy
    from wad_writer.assembler import assemble_map_lumps
    from generation.progression import EXIT_SPECIALS
    errors, warnings = [], []
    progression = None
    try:
        assemble_map_lumps(doc)
    except (ValueError, TypeError, OverflowError) as exc:
        return {"errors": [str(exc)], "warnings": [], "reachable": []}
    if not doc.sectors or not doc.linedefs:
        return {"errors": ["map has no geometry"], "warnings": [], "reachable": []}
    fresh = copy(doc)
    fresh.bounds, fresh.derived = compute_derived(doc.vertices, doc.sectors, doc.linedefs,
                                                doc.sidedefs, doc.things)
    errors.extend(fresh.derived.geometry_issues)
    errors.extend(geometry_conflicts(fresh))
    for sector in fresh.sectors:
        if not sector_geometry(fresh)[sector.id]:
            errors.append(f"sector {sector.id} has no closed boundary")
    starts = [t for t in fresh.things if t.type == 1]
    if len(starts) != 1:
        errors.append(f"expected one single-player start, found {len(starts)}")
    for start in starts:
        sid = _sector_containing(start.x, start.y, fresh.sectors, sector_geometry(fresh), fresh.vertices)
        if sid is None:
            errors.append("player start is outside playable space")
        elif fresh.sectors[sid].ceil_height - fresh.sectors[sid].floor_height < 56:
            errors.append("player start lacks vertical clearance")
    from wad_parser.traversal import potential_state, line_component
    from wad_parser.geometry import ring_depths, sector_points
    state = potential_state(fresh)
    components = state['distances']
    distances = {}
    for (sid, _), distance in components.items():
        distances[sid] = min(distance, distances.get(sid, distance))
    from generation.actors import spawn_problem, actor_profile
    from wad_parser.clearance import MapClearance
    geometry = MapClearance(fresh.vertices, fresh.sectors, fresh.linedefs, fresh.sidedefs)
    for thing in fresh.things:
        if not active_single_player(thing) or thing.category not in {"monster", "player-start", "obstacle"}:
            continue
        if thing.category == "obstacle" and actor_profile(thing.type) is None:
            warnings.append(f"thing {thing.id}: custom obstacle dimensions not validated")
            continue
        problem = spawn_problem(fresh, thing.type, thing.x, thing.y, ignore_id=thing.id,
                                reachable=components, geometry=geometry)
        if problem:
            errors.append(f"thing {thing.id}: {problem}")
    if require_connected:
        missing = {s.id for s in fresh.sectors} - distances.keys()
        if missing:
            errors.append(f"unreachable playable sectors: {_id_summary(missing)}")
        all_components = {(s.id, index) for s in fresh.sectors
                          for index, depth in enumerate(ring_depths(sector_points(fresh, s.id))) if depth % 2 == 0}
        missing_components = all_components - components.keys()
        if missing_components:
            errors.append(f"unreachable playable components: {_id_summary(missing_components)}")
    if require_exit:
        positions = {v.id: (v.x, v.y) for v in fresh.vertices}
        exits = [line for line in fresh.linedefs if line.special in EXIT_SPECIALS]
        usable = False
        for line in exits:
            sid = fresh.sidedefs[line.right_sidedef].sector
            node = line_component(line, sid, positions, fresh.sectors, fresh.sidedefs, sector_geometry(fresh))
            if node in components and math.dist(positions[line.v1], positions[line.v2]) > 0:
                if line.special not in {52, 124} or line.left_sidedef != -1:
                    usable = True
        if not usable:
            errors.append("no reachable usable exit")
    if not errors:
        from wad_parser.navigation import PlayerNavigation
        navigation = PlayerNavigation(fresh, state=state)
        if not navigation.reached:
            errors.append('player start has no navigable configuration space')
        if require_connected:
            missing = navigation.regions.keys() - navigation.reached
            if missing:
                errors.append(f'unreachable player configuration regions: {_id_summary(missing)}')
        if require_exit and not any(navigation.reaches_line(line) for line in exits):
            errors.append('no reachable usable exit in player configuration space')
        if require_exit and not errors:
            from wad_parser.progression import requires_action_history, search_exit_history
            if requires_action_history(fresh):
                progression = search_exit_history(fresh)
                if progression['status'] in {'failure', 'inconclusive'}:
                    errors.append('door/key progression: ' + progression['reason'])
                elif progression['status'] == 'unsupported':
                    (errors if fresh.meta.format == 'doom' else warnings).append(progression['reason'])
    return {"errors": errors, "warnings": warnings, "reachable": sorted(distances),
            'progression': progression,
            'unreachable_sectors': sorted({s.id for s in fresh.sectors} - distances.keys()),
            'unreachable_components': sorted(all_components - components.keys()) if require_connected else []}


def _id_summary(values, limit=20):
    values = sorted(values)
    return str(values) if len(values) <= limit else f'{values[:limit]} … ({len(values)} total)'


def require_valid(doc: MapDocument, *, require_exit=True, require_connected=False) -> dict:
    report = validate_document(doc, require_exit=require_exit, require_connected=require_connected)
    if report["errors"]:
        detail = '; '.join(report['errors'][:20])
        if len(report['errors']) > 20:
            detail += f"; … ({len(report['errors'])} errors total; inspect validate_document report)"
        raise ValueError('invalid map: ' + detail)
    return report


def _largest_sector_seed(doc: MapDocument) -> tuple[float, float]:
    """Interior point of the biggest sector: the vectorizer records a known
    interior cell per sector (rectilinear sectors can have exterior
    centroids, so a recorded seed beats a computed centroid)."""
    for sector in sorted(doc.sectors, key=lambda s: sector_area_perimeter(doc, s.id)[0], reverse=True):
        point = interior_point(doc, sector.id, clearance=16)
        if point is not None:
            return point
    raise ValueError("no sector contains a valid player spawn")

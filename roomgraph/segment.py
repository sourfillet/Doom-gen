"""Sector -> room segmentation.

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
from wad_parser.types import MapDocument, active_single_player
from wad_parser.geometry import (
    point_in_ring, resolve_rings, ring_depths, rings_area_perimeter, sector_geometry, signed_area,
)
from wad_parser.traversal import portal_widths, movement_neighbors, potential_state
from wad_parser.map_extractor import _sector_containing

# An opening at least this wide (summed over the shared boundary) with no door
# reads as "same room"; narrower than this is a doorway/corridor between rooms.
MERGE_WIDTH = 128.0
# Only merge across a step this small - a bigger floor change (ledge, platform,
# pit) marks a distinct space. Without this the whole open area of a map merges
# into ONE giant hub room, which makes every training graph a degenerate star.
MERGE_STEP = 40
SEGMENTATION_VERSION = 2


def _is_door(ld) -> bool:
    return "door" in (ld.special_name or "").lower()


def segment(doc: MapDocument, merge_width: float = MERGE_WIDTH, *, playable_only=False) -> RoomGraph:
    from roomgraph.components import split_components
    original = doc
    doc, state, sources = split_components(doc, potential_state(doc))
    graph = _segment(doc, merge_width, playable_only=playable_only, state=state)
    retained = {sid for room in graph.rooms for sid in room.sectors} | set(graph.collapsed_door_sectors)
    graph.excluded_components = sorted(sources[sid] for sid in graph.excluded_sectors)
    graph.excluded_sectors = sorted({sector.id for sector in original.sectors}
                                     - {sources[sid][0] for sid in retained})
    graph.collapsed_door_components = sorted(sources[sid] for sid in graph.collapsed_door_sectors)
    graph.collapsed_door_sectors = sorted({sources[sid][0] for sid in graph.collapsed_door_sectors})
    for room in graph.rooms:
        room.components = sorted(sources[sid] for sid in room.sectors)
        room.sectors = sorted({sources[sid][0] for sid in room.sectors})
    return graph


def _segment(doc: MapDocument, merge_width, *, playable_only, state) -> RoomGraph:
    vpos = {v.id: (v.x, v.y) for v in doc.vertices}
    linedef_by_id = {ld.id: ld for ld in doc.linedefs}
    sector_ids = [s.id for s in doc.sectors if sector_geometry(doc)[s.id]]
    widths = portal_widths(doc.vertices, doc.linedefs, doc.sidedefs)
    teleports = [edge for edge in doc.derived.sector_graph.get('edges', []) if edge.get('kind') == 'teleport']
    from wad_parser.actions import LOCAL_DOORS, USE_DOORS, CROSS_DOORS
    door_sectors = {doc.sidedefs[line.left_sidedef].sector for line in doc.linedefs
                    if line.special in LOCAL_DOORS and 0 <= line.left_sidedef < len(doc.sidedefs)}
    door_tags = {line.tag for line in doc.linedefs if line.tag and line.special in USE_DOORS | CROSS_DOORS}
    door_sectors.update(sector.id for sector in doc.sectors if sector.tag and sector.tag in door_tags)
    sources = doc.extra.get('_roomgraph_component_sources', {})
    original_doors = {sources[sid][0] for sid in door_sectors if sid in sources}
    door_sectors.update(sid for sid, source in sources.items() if source[0] in original_doors)
    walk = movement_neighbors(doc.vertices, state['sectors'], doc.linedefs, doc.sidedefs, teleports,
                              allow_doors=False)
    # Crossings made only while still supported by a ledge (thin landings).
    supported_pairs = {frozenset((a, b)) for a, b, _ in state.get('supported_crossings', [])}
    floor_of = {s.id: s.floor_height for s in doc.sectors}
    border_neighbors = {sid: set() for sid in sector_ids}
    solid_borders = set()
    for line in doc.linedefs:
        a, b = [doc.sidedefs[index].sector if 0 <= index < len(doc.sidedefs) else None
                for index in (line.right_sidedef, line.left_sidedef)]
        if a == b:
            continue
        for own, other in ((a, b), (b, a)):
            if own not in border_neighbors:
                continue
            if other is None or line.flags.impassable or line.flags.block_players or line.flags.block_everything:
                solid_borders.add(own)
            else:
                border_neighbors[own].add(other)
    enclosed = {frozenset((sid, next(iter(neighbors)))) for sid, neighbors in border_neighbors.items()
                if sid not in solid_borders and len(neighbors) == 1}

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
            rec["len"] = max(rec["len"], widths.get(ld.id, 0.0))
        if _is_door(ld) or a in door_sectors or b in door_sectors:
            rec["door"] = True
        elif (b not in walk.get(a, ()) and a not in walk.get(b, ())
              and frozenset((a, b)) not in supported_pairs):
            rec['len'] = 0.0

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
                and not rec["door"] and (rec["len"] >= merge_width or frozenset((a, b)) in enclosed) \
                and rec["step"] <= MERGE_STEP \
                and b in walk.get(a, ()) and a in walk.get(b, ()):
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
            rings = resolve_rings(sector_geometry(doc)[sid], vpos)
            a, _ = rings_area_perimeter(rings)
            shells = [poly for poly, depth in zip(rings, ring_depths(rings)) if depth % 2 == 0]
            polys.extend(shells)
            if a > 0:
                areas.append((a, sid))
                points = [point for poly in shells for point in poly]
                cxs.append(sum(p[0] for p in points) / len(points) * a)
                cys.append(sum(p[1] for p in points) / len(points) * a)
                for x, y in points:
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
        open_ceiling = state['sectors'][rep].ceil_height if rep in door_sectors else rs.ceil_height
        rooms.append(Room(id=rid, sectors=secs, floor_h=rs.floor_height,
                          ceil_h=open_ceiling, area=total_a, centroid=centroid,
                          bbox=bbox, polygons=polys))

    # Trace the room union rather than choosing one constituent sector's
    # outline. Internal ledge boundaries cancel; actual holes remain holes.
    from wad_parser.bsp_geometry import union_sector_regions
    coordinate_rings = [union_sector_regions([resolve_rings(sector_geometry(doc)[sid], vpos)
                                              for sid in room.sectors]) for room in rooms]
    for room in rooms:
        rings = coordinate_rings[room.id]
        depths = ring_depths(rings)
        room.polygons = [ring for ring, depth in zip(rings, depths) if depth % 2 == 0]
        if room.polygons:
            primary = max(room.polygons, key=lambda ring: abs(signed_area(ring)))
            room.holes = []
            for ring, depth in zip(rings, depths):
                if depth % 2:
                    containers = [shell for shell in room.polygons if point_in_ring(*ring[0], shell)]
                    if containers and min(containers, key=lambda shell: abs(signed_area(shell))) is primary:
                        room.holes.append(ring)

    # --- connections: surviving cross-room adjacencies, one per room pair
    conn_acc: dict[tuple[int, int], dict] = defaultdict(
        lambda: {"len": 0.0, "door": False, "teleport": False})
    for (a, b), rec in pairs.items():
        if a not in room_of or b not in room_of:
            continue
        if not rec['door'] and not rec['teleport'] and rec['len'] == 0:
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

    by_room = {room.id: room for room in rooms}
    from wad_parser.components import ComponentIndex
    owner_index = ComponentIndex(doc.vertices, sector_geometry(doc))
    for thing in doc.things:
        role = 'start' if thing.type == 1 else 'key' if thing.category == 'key' else None
        if role and not active_single_player(thing):
            continue
        if role:
            sid = _sector_containing(thing.x, thing.y, doc.sectors, sector_geometry(doc), doc.vertices,
                                     locator=owner_index)
            if sid in room_of:
                room = by_room[room_of[sid]]
                if role not in room.roles:
                    room.roles.append(role)
    for line in doc.linedefs:
        if line.special in {11, 51, 52, 124, 197, 198} and 0 <= line.right_sidedef < len(doc.sidedefs):
            sid = doc.sidedefs[line.right_sidedef].sector
            if sid not in room_of:
                continue
            room = by_room[room_of[sid]]
            if 'exit' not in room.roles:
                room.roles.append('exit')
    for room in rooms:
        room.role = next((role for role in ('start', 'exit', 'key') if role in room.roles), 'room')
    excluded = []
    if playable_only:
        reachable = {sid for sid, _ in state['distances']}
        selected = [room for room in rooms if any(sid in reachable for sid in room.sectors)]
        selected_ids = {room.id for room in selected}
        excluded = sorted(sid for room in rooms if room.id not in selected_ids for sid in room.sectors)
        remap = {room.id: i for i, room in enumerate(selected)}
        connections = [Connection(remap[edge.a], remap[edge.b], edge.kind, edge.width)
                       for edge in connections if edge.a in remap and edge.b in remap]
        for room in selected:
            room.id = remap[room.id]
        rooms = selected
    # A thin moving door slab is a connection, not an extra low-ceiling room.
    # Keeping it as a node taught the decoder to invent a minimum-size room
    # around every door and quantized a closed ceiling into a 72-unit room.
    collapsed = []
    for room in list(rooms):
        if (room.roles or not room.sectors or not set(room.sectors) <= door_sectors
                or min(room.size) > 64):
            continue
        incident = [edge for edge in connections if room.id in (edge.a, edge.b)]
        if len(incident) != 2 or any(edge.kind != 'door' for edge in incident):
            continue
        a, b = [edge.b if edge.a == room.id else edge.a for edge in incident]
        if a == b or any({edge.a, edge.b} == {a, b} for edge in connections):
            continue  # retain parallel routes rather than losing topology
        widths = [edge.width for edge in incident if edge.width > 0]
        connections = [edge for edge in connections if edge not in incident]
        connections.append(Connection(a, b, 'door', min(widths) if widths else 64))
        rooms.remove(room)
        collapsed.extend(room.sectors)
    remap = {room.id: index for index, room in enumerate(rooms)}
    connections = [Connection(remap[edge.a], remap[edge.b], edge.kind, edge.width) for edge in connections]
    for room in rooms:
        room.id = remap[room.id]
    return RoomGraph(rooms=rooms, connections=connections, excluded_sectors=excluded,
                     collapsed_door_sectors=sorted(collapsed), map_name=doc.meta.map_name)

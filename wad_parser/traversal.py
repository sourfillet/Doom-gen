"""Shared conservative player traversal for Doom geometry.

Physical adjacency is separate from movement. Directed drops and teleports are
preserved; unsupported locks/scripts are never assumed to be traversable.
Unlocked local doors can be evaluated in their potential open state.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque

from wad_parser.geometry import sector_geometry

PLAYER_HEIGHT = 56
PLAYER_RADIUS = 16
MAX_STEP_UP = 24
TRAVERSAL_VERSION = 5
LOCAL_UNLOCKED_DOORS = frozenset({1, 31, 117, 118})


def portal_widths(vertices, linedefs, sidedefs):
    """Combine only touching, collinear opening segments of the same pair.

    A width is never the sum of disconnected windows. Splitting one opening
    into several linedefs does not make it artificially too narrow.
    """
    positions = {v.id: (v.x, v.y) for v in vertices}
    groups = defaultdict(list)
    for line in linedefs:
        if line.flags.impassable or line.flags.block_players or line.flags.block_everything:
            continue
        if not (0 <= line.right_sidedef < len(sidedefs) and 0 <= line.left_sidedef < len(sidedefs)):
            continue
        if line.v1 not in positions or line.v2 not in positions:
            continue
        a, b = positions[line.v1], positions[line.v2]
        length = math.dist(a, b)
        if length <= 0:
            continue
        dx, dy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
        if dx < -1e-8 or abs(dx) <= 1e-8 and dy < 0:
            dx, dy = -dx, -dy
        pair = tuple(sorted((sidedefs[line.right_sidedef].sector, sidedefs[line.left_sidedef].sector)))
        key = pair, round(dx, 6), round(dy, 6), round(a[0] * -dy + a[1] * dx, 5), line.special
        lo, hi = sorted((a[0] * dx + a[1] * dy, b[0] * dx + b[1] * dy))
        groups[key].append((lo, hi, line.id))
    widths = {}
    for intervals in groups.values():
        current = []
        lo = hi = None
        for begin, end, line_id in sorted(intervals):
            if hi is None or begin > hi + 1e-5:
                if current:
                    widths.update({idx: hi - lo for idx in current})
                lo, hi, current = begin, end, [line_id]
            else:
                hi = max(hi, end)
                current.append(line_id)
        if current:
            widths.update({idx: hi - lo for idx in current})
    return widths


def _potential_ceilings(sectors, linedefs, sidedefs, *, allow_doors, hexen_style):
    by_id = {s.id: s for s in sectors}
    ceilings = {s.id: s.ceil_height for s in sectors}
    if allow_doors and not hexen_style:
        door_ids = {sidedefs[line.left_sidedef].sector for line in linedefs
                    if line.special in LOCAL_UNLOCKED_DOORS and 0 <= line.left_sidedef < len(sidedefs)}
        for sid in door_ids:
            if sid not in by_id:
                continue
            adjacent = []
            for line in linedefs:
                if not (0 <= line.right_sidedef < len(sidedefs) and 0 <= line.left_sidedef < len(sidedefs)):
                    continue
                a, b = sidedefs[line.right_sidedef].sector, sidedefs[line.left_sidedef].sector
                if a == sid and b != sid and b in by_id:
                    adjacent.append(by_id[b].ceil_height)
                if b == sid and a != sid and a in by_id:
                    adjacent.append(by_id[a].ceil_height)
            if adjacent:
                ceilings[sid] = max(ceilings[sid], min(adjacent) - 4)
    return ceilings


def movement_neighbors(vertices, sectors, linedefs, sidedefs, teleport_edges=(), *,
                       allow_doors=True, hexen_style=False, edge_records=None, clearance=None,
                       potential_teleport_consumption=False):
    by_id = {s.id: s for s in sectors}
    neighbors = {s.id: set() for s in sectors}
    teleport_edges = list(teleport_edges)
    # Crossing an active vanilla walk-teleport from the front transports the
    # player; it does not also grant access to the sector behind the line.
    teleport_lines = {line.id for line in linedefs if line.special in {39, 97}}
    forced_teleports = {edge['linedef'] for edge in teleport_edges
                        if edge['linedef'] in teleport_lines} if not hexen_style else set()
    ceilings = _potential_ceilings(sectors, linedefs, sidedefs, allow_doors=allow_doors, hexen_style=hexen_style)
    from wad_parser.clearance import MapClearance
    if clearance is None:
        clearance = MapClearance(vertices, sectors, linedefs, sidedefs, ceilings=ceilings)
    else:
        # Reuse the XY index across mover states, refreshing vertical values.
        # The supplied index must belong to these same vertices/lines/sides.
        clearance.floors = {sector.id: sector.floor_height for sector in sectors}
        clearance.ceilings = ceilings
    for line in linedefs:
        if not (0 <= line.right_sidedef < len(sidedefs) and 0 <= line.left_sidedef < len(sidedefs)):
            continue
        if line.flags.impassable or line.flags.block_players or line.flags.block_everything:
            continue
        a, b = sidedefs[line.right_sidedef].sector, sidedefs[line.left_sidedef].sector
        if a == b or a not in by_id or b not in by_id:
            continue
        # Locks govern activation, not physical passage through an already
        # open door. The component solver opens them only after finding a key.
        opening = min(ceilings[a], ceilings[b]) - max(by_id[a].floor_height, by_id[b].floor_height)
        if opening < PLAYER_HEIGHT:
            continue
        for src, dst in ((a, b), (b, a)):
            if (src == a and line.id in forced_teleports
                    and not (potential_teleport_consumption and line.special == 39)):
                continue
            if by_id[dst].floor_height - by_id[src].floor_height <= MAX_STEP_UP and clearance.crossing(line, src):
                neighbors[src].add(dst)
                if edge_records is not None:
                    edge_records.append((src, dst, line.id, None))
    lines = {line.id: line for line in linedefs}
    for edge in teleport_edges:
        line = lines.get(edge.get("linedef"))
        if line is None:
            continue
        supported = line.special in ({70, 71} if hexen_style else {39, 97, 174, 195, 207, 208, 209, 210, 243, 244, 262, 263})
        if not supported or line.flags.block_everything or line.flags.block_players or line.flags.impassable:
            continue
        if line.left_sidedef < 0:  # no crossable activation line
            continue
        a, b = edge["a"], edge["b"]
        if (a in neighbors and b in neighbors and ceilings[b] - by_id[b].floor_height >= PLAYER_HEIGHT
                and clearance.crossing(line, a)):
            neighbors[a].add(b)
            if edge_records is not None:
                edge_records.append((a, b, line.id, edge))
    return neighbors



def component_for_point(sid, x, y, rings, positions, component_index=None):
    if component_index is not None:
        return component_index.point(sid, x, y)
    from wad_parser.geometry import point_in_rings, point_in_ring, signed_area, ring_depths, resolve_rings
    if not 0 <= sid < len(rings):
        return None
    points = resolve_rings(rings[sid], positions)
    if not point_in_rings(x, y, points, boundary=True):
        return None
    shells = [(abs(signed_area(ring)), index) for index, (ring, depth) in enumerate(zip(points, ring_depths(points)))
              if depth % 2 == 0 and point_in_ring(x, y, ring, boundary=True)]
    return (sid, min(shells)[1]) if shells else None


def line_component(line, sid, vertices, sectors, sidedefs, rings, component_index=None):
    if component_index is not None:
        node = component_index.edges.get((sid, line.v1, line.v2))
        if node is None and line.v1 in component_index.positions and line.v2 in component_index.positions:
            node = component_index.coordinate_edges.get((sid, component_index.positions[line.v1],
                                                          component_index.positions[line.v2]))
        if node is not None:
            return node
    positions = vertices if isinstance(vertices, dict) else {v.id: (v.x, v.y) for v in vertices}
    a, b = positions[line.v1], positions[line.v2]
    length = max(math.dist(a, b), 1e-6)
    sign = 1 if 0 <= line.right_sidedef < len(sidedefs) and sidedefs[line.right_sidedef].sector == sid else -1
    dx, dy = (b[1] - a[1]) / length * 0.01, -(b[0] - a[0]) / length * 0.01
    midpoint = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    for direction in (sign, -sign):
        node = component_for_point(sid, midpoint[0] + direction * dx, midpoint[1] + direction * dy,
                                   rings, positions, component_index)
        if node is not None:
            return node
    return None


def _component_distances_static(vertices, sectors, linedefs, sidedefs, things, rings,
                             teleport_edges=(), *, allow_doors=True, hexen_style=False, component_index=None,
                             clearance=None, potential_teleport_consumption=False, supported_records=None):
    from wad_parser.map_extractor import _sector_containing
    positions = {v.id: (v.x, v.y) for v in vertices}
    if component_index is None:
        from wad_parser.components import ComponentIndex
        component_index = ComponentIndex(positions, rings)
    if clearance is None:
        from wad_parser.clearance import MapClearance
        clearance = MapClearance(vertices, sectors, linedefs, sidedefs,
                                 ceilings=_potential_ceilings(sectors, linedefs, sidedefs, allow_doors=allow_doors,
                                                              hexen_style=hexen_style))
    records = []
    movement_neighbors(vertices, sectors, linedefs, sidedefs, teleport_edges,
                       allow_doors=allow_doors, hexen_style=hexen_style, edge_records=records,
                       clearance=clearance, potential_teleport_consumption=potential_teleport_consumption)
    lines = {line.id: line for line in linedefs}
    floors = {sector.id: sector.floor_height for sector in sectors}
    adjacency = defaultdict(set)
    ledges = defaultdict(set)
    for src, dst, line_id, teleport in records:
        line = lines[line_id]
        a = line_component(line, src, positions, sectors, sidedefs, rings, component_index)
        if teleport is None:
            b = line_component(line, dst, positions, sectors, sidedefs, rings, component_index)
        elif 'destination_point' in teleport:
            b = component_for_point(dst, *teleport['destination_point'], rings, positions, component_index)
        elif teleport.get('destination_line') in lines:
            b = line_component(lines[teleport['destination_line']], dst, positions, sectors, sidedefs, rings, component_index)
        else:
            b = None
        if a is not None and b is not None:
            adjacency[a].add(b)
            if teleport is None and floors[src] > floors[dst]:
                ledges[a].add((b, line_id))
    # Teleport lines are never crossed by the ledge-supported search.
    teleport_lines = ({edge['linedef'] for edge in teleport_edges}
                      | {line.id for line in linedefs if line.special in {39, 97}})
    supported_cache, supported_links = {}, set()

    def supported_moves(node, ledge):
        # After descending over ``ledge`` the player is still at its height
        # while the box overlaps it, even above a component too thin to stand
        # in. Cross nearby boundaries of that component only at that height.
        key = node, ledge
        if key in supported_cache:
            return supported_cache[key]
        moves = set()
        height = max(floors.get(sidedefs[index].sector, -math.inf) for index in
                     (lines[ledge].right_sidedef, lines[ledge].left_sidedef))
        for other in sorted(clearance.lines_near(ledge, PLAYER_RADIUS)):
            line = lines.get(other)
            if (line is None or other == ledge or other in teleport_lines or line.flags.impassable
                    or line.flags.block_players or line.flags.block_everything
                    or not (0 <= line.right_sidedef < len(sidedefs) and 0 <= line.left_sidedef < len(sidedefs))):
                continue
            a, b = sidedefs[line.right_sidedef].sector, sidedefs[line.left_sidedef].sector
            for src, dst in ((a, b), (b, a)):
                if (src != node[0] or src == dst or dst not in floors
                        or line_component(line, src, positions, sectors, sidedefs, rings, component_index) != node
                        or not clearance.supported_crossing(line, src, ledge, height)):
                    continue
                target = line_component(line, dst, positions, sectors, sidedefs, rings, component_index)
                if target is not None:
                    moves.add(target)
                    supported_links.add((node, target, other))
        supported_cache[key] = sorted(moves)
        return supported_cache[key]

    distances = {}
    for start in things:
        if start.type != 1:
            continue
        sid = _sector_containing(start.x, start.y, sectors, rings, vertices, locator=component_index)
        if sid is not None:
            node = component_for_point(sid, start.x, start.y, rings, positions, component_index)
            if node is not None:
                distances[node] = 0
    # Breadth-first over plain components and (component, ledge) states.
    supported = {}
    queue = deque((node, None) for node in distances)
    while queue:
        current, ledge = queue.popleft()
        if ledge is None:
            step = distances[current] + 1
            moves = ([(other, None) for other in sorted(adjacency[current])]
                     + sorted(ledges[current]))
        else:
            step = supported[current, ledge] + 1
            reached = supported_moves(current, ledge)
            moves = [(other, None) for other in reached] + [(other, ledge) for other in reached]
        for other, via in moves:
            if via is None:
                if other not in distances:
                    distances[other] = step
                    queue.append((other, None))
            elif (other, via) not in supported:
                supported[other, via] = step
                queue.append((other, via))
    if supported_records is not None:
        # Crossings made only at a ledge height, for callers that rebuild
        # connectivity from ordinary walk records (room-graph connections).
        supported_records.extend(sorted(supported_links))
    return distances


def potential_state_raw(vertices, sectors, linedefs, sidedefs, things, rings,
                        teleport_edges=(), *, allow_doors=True, hexen_style=False):
    teleport_edges = list(teleport_edges)
    if allow_doors and not hexen_style:
        from wad_parser.actions import door_state
        return door_state(vertices, sectors, linedefs, sidedefs, things, rings, teleport_edges)
    supported = []
    return {'distances': _component_distances_static(vertices, sectors, linedefs, sidedefs,
             things, rings, teleport_edges, allow_doors=False, hexen_style=hexen_style, supported_records=supported),
            'sectors': sectors, 'keys': [], 'actions': [], 'supported_crossings': supported}


def component_distances_raw(*args, **kwargs):
    return potential_state_raw(*args, **kwargs)['distances']


def distances_from_raw(*args, **kwargs):
    distances = {}
    for (sid, component), distance in component_distances_raw(*args, **kwargs).items():
        distances[sid] = min(distance, distances.get(sid, distance))
    return distances


def potential_state(doc, *, allow_doors=True):
    from wad_parser.map_extractor import _teleport_edges
    from wad_parser.types import HEXEN_UDMF_NAMESPACES
    hexen_style = doc.meta.format == 'hexen' or str(doc.extra.get('udmf_namespace', '')).lower() in HEXEN_UDMF_NAMESPACES
    edges = _teleport_edges(doc.sectors, doc.linedefs, doc.sidedefs, doc.things,
                            sector_geometry(doc), doc.vertices, hexen_style)
    state = potential_state_raw(doc.vertices, doc.sectors, doc.linedefs, doc.sidedefs,
                                doc.things, sector_geometry(doc), edges,
                                allow_doors=allow_doors, hexen_style=hexen_style)
    state['teleport_edges'] = edges
    return state


def component_distances(doc, *, allow_doors=True):
    return potential_state(doc, allow_doors=allow_doors)['distances']


def distances_from_start(doc, *, allow_doors=True):
    distances = {}
    for (sid, component), distance in component_distances(doc, allow_doors=allow_doors).items():
        distances[sid] = min(distance, distances.get(sid, distance))
    return distances

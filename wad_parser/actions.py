"""Potential vanilla door/key progression, without mutating source geometry.

Rules follow id Software's p_switch.c, p_spec.c and p_doors.c. This computes
supported potential access, not a solution to timed/one-shot puzzles or scripts.
Only reachable activators and keys can unlock a new route.
"""
from __future__ import annotations

from dataclasses import replace

LOCAL_DOORS = frozenset({1, 26, 27, 28, 31, 32, 33, 34, 117, 118})
USE_DOORS = frozenset({29, 61, 63, 103, 111, 112, 114, 115, 99, 133, 134, 135, 136, 137})
CROSS_DOORS = frozenset({2, 4, 86, 90, 105, 106, 108, 109})
LOCK_COLORS = {26: 'blue', 32: 'blue', 27: 'yellow', 34: 'yellow', 28: 'red', 33: 'red',
               99: 'blue', 133: 'blue', 134: 'red', 135: 'red', 136: 'yellow', 137: 'yellow'}
KEY_COLORS = {5: 'blue', 40: 'blue', 6: 'yellow', 39: 'yellow', 13: 'red', 38: 'red'}


def door_state(vertices, sectors, linedefs, sidedefs, things, rings, teleport_edges):
    from wad_parser.clearance import MapClearance
    from wad_parser.map_extractor import _sector_containing
    from wad_parser.traversal import _component_distances_static, component_for_point, line_component
    from wad_parser.components import ComponentIndex

    virtual = [replace(sector) for sector in sectors]
    by_id = {sector.id: sector for sector in virtual}
    positions = {v.id: (v.x, v.y) for v in vertices}
    component_index = ComponentIndex(positions, rings)
    adjacent = {sector.id: set() for sector in sectors}
    for line in linedefs:
        if not (0 <= line.right_sidedef < len(sidedefs) and 0 <= line.left_sidedef < len(sidedefs)):
            continue
        a, b = sidedefs[line.right_sidedef].sector, sidedefs[line.left_sidedef].sector
        if a != b and a in adjacent and b in adjacent:
            adjacent[a].add(b)
            adjacent[b].add(a)
    nodes = {}
    def node(line, sid):
        key = line.id, sid
        if key not in nodes:
            nodes[key] = line_component(line, sid, positions, sectors, sidedefs, rings, component_index)
        return nodes[key]
    key_nodes = []
    for thing in things:
        if thing.type not in KEY_COLORS or thing.flags.not_single_player or not thing.flags.skill_3:
            continue
        sid = _sector_containing(thing.x, thing.y, sectors, rings, vertices, locator=component_index)
        if sid is not None:
            key_nodes.append((KEY_COLORS[thing.type], component_for_point(sid, thing.x, thing.y,
                                                                         rings, positions, component_index)))
    keys, trace = set(), []
    candidates = [line for line in linedefs if line.special in LOCAL_DOORS | USE_DOORS | CROSS_DOORS]
    geometry = MapClearance(vertices, virtual, linedefs, sidedefs)
    for _ in range(len(candidates) + len(key_nodes) + 2):
        supported = []
        distances = _component_distances_static(vertices, virtual, linedefs, sidedefs, things, rings,
                                                teleport_edges, allow_doors=False, component_index=component_index,
                                                clearance=geometry, potential_teleport_consumption=True,
                                                supported_records=supported)
        keys.update(color for color, owner in key_nodes if owner in distances)
        changed = False
        for line in candidates:
            color = LOCK_COLORS.get(line.special)
            if color and color not in keys:
                continue
            if not 0 <= line.right_sidedef < len(sidedefs):
                continue
            front = sidedefs[line.right_sidedef].sector
            if line.special in LOCAL_DOORS | USE_DOORS:
                active = node(line, front) in distances
            else:
                active = False
                if 0 <= line.left_sidedef < len(sidedefs):
                    geometry.ceilings = {sector.id: sector.ceil_height for sector in virtual}
                    back = sidedefs[line.left_sidedef].sector
                    for sid in (front, back):
                        if node(line, sid) in distances and geometry.crossing(line, sid):
                            active = True
                            break
            if not active:
                continue
            if line.special in LOCAL_DOORS:
                targets = ([sidedefs[line.left_sidedef].sector]
                           if 0 <= line.left_sidedef < len(sidedefs) else [])
            else:
                # Tag-zero remote actions have port-dependent compatibility
                # behavior and are deliberately not used to infer progression.
                targets = [sector.id for sector in virtual if line.tag and sector.tag == line.tag]
            for sid in targets:
                if not adjacent.get(sid):
                    continue
                height = min(by_id[other].ceil_height for other in adjacent[sid]) - 4
                if height > by_id[sid].ceil_height:
                    by_id[sid].ceil_height = height
                    trace.append({'linedef': line.id, 'sector': sid, 'ceiling': height,
                                  'key': color, 'action': 'door-open'})
                    changed = True
        if not changed:
            return {'distances': distances, 'sectors': virtual, 'keys': sorted(keys), 'actions': trace,
                    'potential_teleport_consumption': True, 'supported_crossings': supported}
    raise ValueError('door progression did not converge within the supported action bound')

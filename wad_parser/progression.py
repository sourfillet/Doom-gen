"""Bounded, location-aware potential histories for vanilla doors and keys.

An optimistic union of reachable sectors can mix events from incompatible
one-way branches. Here each state retains its location, keys and actual door
ceilings. Geometry and use approaches are checked again after every action.
Doors remain open in this abstraction. Vanilla type-39 teleports are consumed
after a supported front-side activation. Backside/failed activation consumption,
same-sector trigger planes, extension teleports, one-shot doors, moving floors,
scripts, combat and resource balance remain outside this potential abstraction.
"""
from __future__ import annotations

from dataclasses import replace

from wad_parser.actions import LOCAL_DOORS, USE_DOORS, CROSS_DOORS, LOCK_COLORS, KEY_COLORS

PLAYER_TELEPORT_SPECIALS = frozenset({39, 97, 174, 195, 207, 208, 209, 210, 243, 244, 262, 263})


def door_targets(doc):
    """Supported activators and their targets, independent of reachability."""
    adjacent = {sector.id: set() for sector in doc.sectors}
    for line in doc.linedefs:
        if min(line.right_sidedef, line.left_sidedef) < 0:
            continue
        a, b = (doc.sidedefs[side].sector for side in (line.right_sidedef, line.left_sidedef))
        if a != b:
            adjacent[a].add(b)
            adjacent[b].add(a)
    actions = []
    for line in doc.linedefs:
        if line.special in LOCAL_DOORS:
            targets = ([doc.sidedefs[line.left_sidedef].sector] if line.left_sidedef >= 0 else [])
        elif line.special in USE_DOORS | CROSS_DOORS:
            targets = [sector.id for sector in doc.sectors if line.tag and sector.tag == line.tag]
        else:
            continue
        targets = [sid for sid in targets if adjacent[sid]]
        if targets:
            actions.append((line, targets))
    return actions, adjacent


def requires_action_history(doc):
    return bool(door_targets(doc)[0]) or any(line.special in PLAYER_TELEPORT_SPECIALS - {97}
                                           for line in doc.linedefs)


def _pickup_points(thing, sid, nav, reached, sectors):
    from shapely import box
    # P_TouchSpecialThing permits -8 <= pickup.z - player.z <= player.height.
    # Pickups are touched by overlapping actor boxes, including near a wall;
    # the player need not stand on the pickup's centre.
    touch = box(thing.x - 36 + 1e-6, thing.y - 36 + 1e-6,
                thing.x + 36 - 1e-6, thing.y + 36 - 1e-6)
    for region in sorted(reached):
        if not -8 <= sectors[sid].floor_height - sectors[region[0]].floor_height <= 56:
            continue
        intersection = nav.regions[region].intersection(touch)
        if not intersection.is_empty and intersection.area > 1e-9:
            point = intersection.representative_point()
            yield point.x, point.y


def search_exit_history(doc, *, max_states=4096, max_geometries=256, consume_one_shot_teleports=True):
    """Return success, failure or an explicit budget-exhausted result.

    Results describe the supported potential abstraction, never a full engine
    completion certificate. Input geometry and things are not modified.
    """
    from generation.progression import EXIT_SPECIALS
    from wad_parser.geometry import sector_geometry
    from wad_parser.map_extractor import _teleport_edges, _sector_containing
    from wad_parser.navigation import PlayerNavigation
    from wad_parser.types import active_single_player, HEXEN_UDMF_NAMESPACES

    def result(status, reason, trace=()):
        return {'status': status, 'reason': reason, 'states': len(seen),
                'geometries': len(geometries), 'trace': list(trace),
                'abstraction': ('location_keys_potential_doors_once_teleports_v2'
                                if consume_one_shot_teleports else 'location_keys_potential_doors_v1')}

    seen, geometries = set(), {}
    if doc.meta.format == 'hexen' or str(doc.extra.get('udmf_namespace', '')).lower() in HEXEN_UDMF_NAMESPACES:
        return result('unsupported', 'Hexen/script action histories are not implemented')
    starts = [thing for thing in doc.things if thing.type == 1]
    if len(starts) != 1:
        return result('failure', 'expected exactly one player-one start')
    exits = [line for line in doc.linedefs if line.special in EXIT_SPECIALS]
    actions, adjacent = door_targets(doc)
    rings = sector_geometry(doc)
    edges = _teleport_edges(doc.sectors, doc.linedefs, doc.sidedefs, doc.things, rings, doc.vertices, False)
    one_shot = {line.id: line for line in doc.linedefs if line.special == 39} if consume_one_shot_teleports else {}
    bits = {line_id: 1 << index for index, line_id in enumerate(sorted(one_shot))}
    if consume_one_shot_teleports:
        specials = {line.id: line.special for line in doc.linedefs}
        if any(specials[edge['linedef']] in PLAYER_TELEPORT_SPECIALS - {39, 97} for edge in edges):
            return result('unsupported', 'extension teleport action histories are not implemented')
        from wad_parser.geometry import point_on_segment
        positions = {v.id: (v.x, v.y) for v in doc.vertices}
        for line in one_shot.values():
            if line.left_sidedef < 0 or doc.sidedefs[line.left_sidedef].sector == doc.sidedefs[line.right_sidedef].sector:
                return result('unsupported', 'same-sector/one-sided one-shot teleport trigger planes are not implemented')
            if point_on_segment(starts[0].x, starts[0].y, positions[line.v1], positions[line.v2]):
                return result('unsupported', 'player starts on one-shot teleport trigger planes are not implemented')
    keys = []
    for thing in doc.things:
        if thing.type in KEY_COLORS and active_single_player(thing):
            sid = _sector_containing(thing.x, thing.y, doc.sectors, rings, doc.vertices)
            if sid is not None:
                keys.append((thing, sid, KEY_COLORS[thing.type]))
    colors = {'blue': 1, 'yellow': 2, 'red': 4}
    ceilings = tuple(sector.ceil_height for sector in doc.sectors)
    layouts, views = {}, {}
    stack = [(ceilings, 0, 0, (starts[0].x, starts[0].y), ())]
    while stack:
        heights, key_mask, consumed, point, trace = stack.pop()
        geometry_key = heights, consumed
        if geometry_key not in geometries:
            if len(geometries) >= max_geometries:
                return result('inconclusive', ('action geometry budget exhausted' if consume_one_shot_teleports
                                              else 'door geometry budget exhausted'))
            sectors = [replace(sector, ceil_height=heights[sector.id]) for sector in doc.sectors]
            if consumed not in views:
                lines = [replace(line, special=0, special_name=None, trigger_type=None)
                         if consumed & bits.get(line.id, 0) else line for line in doc.linedefs]
                view = replace(doc, linedefs=lines) if consumed else doc
                view_edges = (_teleport_edges(view.sectors, lines, view.sidedefs, view.things,
                                              rings, view.vertices, False) if consumed else edges)
                active = set(one_shot) - {line_id for line_id, bit in bits.items() if consumed & bit}
                views[consumed] = view, view_edges, active
            view, view_edges, active = views[consumed]
            nav = PlayerNavigation(view, state={'sectors': sectors, 'teleport_edges': view_edges,
                                                'teleport_events': active},
                                   clearance_layout=layouts.get(consumed))
            layouts[consumed] = nav.clearance
            geometries[geometry_key] = nav, sectors
        nav, sectors = geometries[geometry_key]
        reached = frozenset(nav.reachable_from([point]))
        # Positions with the same forward-reachable set are equivalent for
        # subsequent events; importantly, different irreversible branches are not.
        state_key = heights, key_mask, consumed, reached
        if state_key in seen:
            continue
        if len(seen) >= max_states:
            return result('inconclusive', 'action history budget exhausted')
        seen.add(state_key)
        if any(nav.line_approach(line, reached=reached) is not None for line in exits):
            return result('success', 'reachable exit in a supported potential history', trace)
        following = []
        for thing, sid, color in keys:
            bit = colors[color]
            if key_mask & bit:
                continue
            for pickup in _pickup_points(thing, sid, nav, reached, sectors):
                following.append((heights, key_mask | bit, consumed, pickup,
                                  trace + ({'action': 'key', 'thing': thing.id, 'color': color},)))
        for line, targets in actions:
            color = LOCK_COLORS.get(line.special)
            if color and not key_mask & colors[color]:
                continue
            changed = list(heights)
            for sid in targets:
                changed[sid] = max(changed[sid], min(heights[other] for other in adjacent[sid]) - 4)
            changed = tuple(changed)
            if changed == heights:
                continue
            for approach in nav.line_approaches(line, reached=reached, crossing=line.special in CROSS_DOORS):
                following.append((changed, key_mask, consumed, approach,
                                  trace + ({'action': 'door-open', 'linedef': line.id,
                                            'sectors': list(targets), 'key': color},)))
        for line_id, crossings in nav.teleport_events.items():
            for source, destination, landing in crossings:
                if source not in reached:
                    continue
                new_consumed = consumed | bits[line_id]
                # An arrival on another active trigger has ambiguous initial
                # side ownership; leave that extension unsupported.
                if any(not new_consumed & bits[other] and point_on_segment(
                        *landing, positions[line.v1], positions[line.v2]) for other, line in one_shot.items()):
                    continue
                following.append((heights, key_mask, new_consumed, landing,
                    trace + ({'action': 'teleport-once', 'linedef': line_id,
                              'destination': list(landing)},)))
        # Depth-first search reaches useful cumulative states without enumerating
        # every subset of independent doors first. All alternatives remain bounded.
        stack.extend(reversed(following))
    return result('failure', 'no location-consistent supported door/key history reaches an exit')

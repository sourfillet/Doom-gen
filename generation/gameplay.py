"""Introductory single-player encounters with explicit resource budgets.

This opt-in procedural baseline is intentionally small: ordinary zombies,
shotgunners and imps; a directly reachable shotgun/shell cache; armor and
room-local health. It does not claim solved encounter design or arbitrary
combat completion. Counts use vanilla skill-3, pistol-start rules.

Actor health: id-Software/DOOM/linuxdoom-1.10/info.c.
Pickup amounts/capacity: id-Software/DOOM/linuxdoom-1.10/p_inter.c.
https://github.com/id-Software/DOOM/tree/master/linuxdoom-1.10
"""
from __future__ import annotations

import math
import random

MONSTER_HEALTH = {3004: 20, 9: 30, 3001: 60}
PROFILES = ('none', 'intro')


def prepare_intro_graph(graph):
    """Keep topology/floors, explicitly constrain sampled combat-room sizes.

    The raw sampler often combines very small rooms and huge rooms; one huge
    room sets a global routing spacing and produces long empty transit. This
    optional policy is procedural and records every changed prediction.
    """
    from copy import deepcopy
    from roomgraph.assemble import _room_dims
    result = deepcopy(graph)
    changes = []
    for room in result.rooms:
        if room.polygons:
            raise ValueError('intro sizing expects sampled attributes before shape retrieval')
        width, height = _room_dims(room)
        scale = max(384 / min(width, height), min(1.0, 1024 / max(width, height)))
        w, h = [max(384, min(1024, round(value * scale / 8) * 8)) for value in (width, height)]
        clearance = max(128, min(256, room.ceil_h - room.floor_h))
        changes.append({'room': room.id, 'requested_size': [width, height], 'realized_size': [w, h],
                        'requested_clearance': room.ceil_h - room.floor_h, 'realized_clearance': clearance})
        room.bbox, room.area = (0, 0, w, h), w * h
        room.ceil_h = room.floor_h + clearance
    return result, {'version': 1, 'rooms': changes, 'size_range': [384, 1024],
                    'clearance_range': [128, 256],
                    'scope': 'Explicit intro gameplay size/headroom policy; original topology/floors retained, no learned-shape claim.'}


def apply_intro(doc, *, seed=0, game=None, monster_budget=8, interior_details=True,
                accent_texture=None, interior_profile='columns', required_progression='none'):
    from shapely import Point, LineString
    from wad_parser.geometry import sector_points, sector_geometry
    from wad_parser.bsp_geometry import component_polygons
    from wad_parser.map_extractor import _sector_containing
    from wad_parser.navigation import PlayerNavigation
    from wad_parser.traversal import distances_from_start, component_distances
    from wad_parser.clearance import MapClearance
    from generation.actors import spawn_problem
    from generation.style import _add_thing
    from generation.validator import require_valid
    from generation.interiors import add_columns, add_terraces, INTERIOR_PROFILES

    if required_progression not in {'none', 'blue-key'}:
        raise ValueError('required_progression must be none or blue-key')
    if interior_profile not in INTERIOR_PROFILES:
        raise ValueError('interior_profile must be columns or stepped')
    if not 1 <= monster_budget <= 12:
        raise ValueError('intro monster_budget must be between 1 and 12')
    if any(thing.category in {'monster', 'weapon', 'ammo', 'item'} for thing in doc.things):
        raise ValueError('intro gameplay requires an undecorated population; omit other thing placers')
    required = {2001, 2008, 2012, 2018}
    if game and any(game.sub_thing(number) != number for number in required):
        raise ValueError('intro gameplay needs stock shotgun, shells, medikit and green armor')
    if required_progression == 'blue-key':
        from wad_parser.actions import KEY_COLORS
        if any(thing.type in KEY_COLORS for thing in doc.things):
            raise ValueError('blue-key progression requires a map without existing keys')
        if game and game.sub_thing(5) != 5:
            raise ValueError('blue-key progression requires the stock blue keycard')
    allowed = [number for number in MONSTER_HEALTH if game is None or game.sub_thing(number) == number]
    if not allowed:
        raise ValueError('game profile has no supported introductory monsters')
    require_valid(doc, require_connected=True)
    terraces = None
    if interior_details and interior_profile == 'stepped':
        flats = sorted({sector.floor_flat for sector in doc.sectors
                        if sector.floor_flat != 'F_SKY1'
                        and not sector.floor_flat.startswith(('FWATER', 'NUKAGE', 'BLOOD', 'LAVA', 'SLIME'))})
        terraces = add_terraces(doc, seed=seed, wall_texture=accent_texture,
                               floor_flat=random.Random(seed).choice(flats) if flats else None)
    details = (add_columns(doc, seed=seed, wall_texture=accent_texture) if interior_details else
               {'columns': [], 'scope': 'disabled'})
    if terraces is not None:
        details = {**details, 'version': 2, 'profile': interior_profile,
                   'terraces': terraces['terraces'], 'terrace_candidates_rejected': terraces['rejected_candidates']}
        doc.extra['interior_details'] = details
    rng = random.Random(seed)
    start = next(thing for thing in doc.things if thing.type == 1)
    start_sid = _sector_containing(start.x, start.y, doc.sectors, sector_geometry(doc), doc.vertices)
    for sector in doc.sectors:
        sector.light_level = max(112, min(208, sector.light_level))
    doc.sectors[start_sid].light_level = max(192, doc.sectors[start_sid].light_level)
    # The opening equipment needs no door opening, teleport or key pickup.
    static = PlayerNavigation(doc, state={'sectors': doc.sectors, 'teleport_edges': []})
    start_regions = [region for region in static.by_sector[start_sid]
                     if static.regions[region].covers(Point(start.x, start.y))]
    if not start_regions:
        raise ValueError('intro player start has no initial free region')
    geometry = MapClearance(doc.vertices, doc.sectors, doc.linedefs, doc.sidedefs)
    components = component_distances(doc)
    navigation = PlayerNavigation(doc)
    distances = distances_from_start(doc)
    records, used_pickups = [], []
    triggers = [LineString([(doc.vertices[line.v1].x, doc.vertices[line.v1].y),
                            (doc.vertices[line.v2].x, doc.vertices[line.v2].y)])
                for line in doc.linedefs if line.special]

    def add(number, point, role, sid):
        thing = _add_thing(doc, *point, number)
        records.append({'thing': thing.id, 'type': number, 'role': role, 'sector': sid})
        if role != 'enemy':
            used_pickups.append(point)
        return thing

    def opening_point():
        for radius in (48, 88, 128, 176, 224):
            for offset in (0, 45, -45, 90, -90, 135, -135, 180):
                angle = math.radians(start.angle + offset)
                point = (round(start.x + radius * math.cos(angle)), round(start.y + radius * math.sin(angle)))
                path = LineString([(start.x, start.y), point])
                if (not any(static.regions[region].covers(path) for region in start_regions)
                        or any(trigger.intersects(path) for trigger in triggers)
                        or any(math.dist(point, previous) < 40 for previous in used_pickups)):
                    continue
                if spawn_problem(doc, 2001, *point, margin=4, reachable=components, geometry=geometry) is None:
                    return point
        raise ValueError('not enough unobstructed start space for the opening equipment')

    add(2001, opening_point(), 'opening_weapon', start_sid)
    add(2018, opening_point(), 'opening_armor', start_sid)

    sectors = []
    for sector in doc.sectors:
        if sector.id not in distances or sector.ceil_height - sector.floor_height < 64:
            continue
        for _, polygon in component_polygons(sector_points(doc, sector.id)):
            x0, y0, x1, y1 = polygon.bounds
            small = min(x1 - x0, y1 - y0)
            if polygon.area >= 192**2 and small >= 192 and max(x1 - x0, y1 - y0) / small <= 3:
                sectors.append((sector.id, polygon))
    if any(sid != start_sid for sid, _ in sectors):
        sectors = [(sid, poly) for sid, poly in sectors if sid != start_sid]
    sectors.sort(key=lambda item: (distances[item[0]], item[0]))
    # The structural model can request an exit beside the start. For this
    # explicit gameplay profile, choose a distant usable objective instead;
    # keep that procedural change visible in the report.
    from generation.progression import EXIT_SPECIALS
    from wad_parser.types import resolve_linedef_special
    if any(line.special in EXIT_SPECIALS - {11} for line in doc.linedefs):
        raise ValueError('intro gameplay supports generated normal exits only')
    old_exits = [line.id for line in doc.linedefs if line.special == 11]
    exit_texture = next((doc.sidedefs[line.right_sidedef].mid_tex for line in doc.linedefs
                         if line.id in old_exits and doc.sidedefs[line.right_sidedef].mid_tex.startswith(('SW1', 'SW2'))), None)
    if exit_texture is None:
        if game is None or 'SW1COMP' in game.textures:
            exit_texture = 'SW1COMP'
        else:
            exit_texture = next((name for name in sorted(game.textures) if name.startswith(('SW1', 'SW2'))), None)
    walls = [line for line in doc.linedefs if line.left_sidedef == -1 and line.special in {0, 11}
             and navigation.reaches_line(line)]
    eligible = {sid for sid, _ in sectors}
    walls = [line for line in walls if doc.sidedefs[line.right_sidedef].sector in eligible]
    if not walls:
        raise ValueError('intro layout has no usable objective wall in an encounter room')
    from generation.objectives import select_intro_objective, select_keyed_intro_objective
    if required_progression == 'blue-key':
        objective, objective_policy = select_keyed_intro_objective(doc, walls, start_sector=start_sid,
            sector_distances=distances, seed=seed, game=game)
    else:
        objective, objective_policy = select_intro_objective(doc, walls, start_sector=start_sid,
                                                            sector_distances=distances)
    for line in doc.linedefs:
        if line.id in old_exits:
            line.special, line.special_name, line.trigger_type = 0, None, None
            doc.sidedefs[line.right_sidedef].mid_tex = accent_texture or (game.default_wall if game else 'STARTAN2')
    objective.special, objective.tag = 11, 0
    objective.special_name, objective.trigger_type = resolve_linedef_special(11)
    if exit_texture:
        doc.sidedefs[objective.right_sidedef].mid_tex = exit_texture
    goal_sid = doc.sidedefs[objective.right_sidedef].sector
    from collections import deque
    previous = {key: None for key, polygon in navigation.regions.items() if polygon.covers(Point(start.x, start.y))}
    queue, goal = deque(previous), None
    while queue:
        key = queue.popleft()
        if key[0] == goal_sid:
            goal = key
            break
        for neighbor in sorted(navigation.adjacency[key]):
            if neighbor not in previous:
                previous[neighbor] = key
                queue.append(neighbor)
    if goal is None:
        raise ValueError('intro objective has no supported region route from start')
    route = []
    while goal is not None:
        route.append(goal[0])
        goal = previous[goal]
    route.reverse()
    triggers = [LineString([(doc.vertices[line.v1].x, doc.vertices[line.v1].y),
                            (doc.vertices[line.v2].x, doc.vertices[line.v2].y)])
                for line in doc.linedefs if line.special]
    ranks = {sid: index for index, sid in enumerate(route)}
    sectors.sort(key=lambda item: (item[0] not in ranks, ranks.get(item[0], distances[item[0]]), item[0]))
    encounters = []
    for sid, polygon in sectors:
        if sum(len(row['enemies']) for row in encounters) >= monster_budget:
            break
        enemies = []
        x0, y0, x1, y1 = polygon.bounds
        for slot in range(min(2, monster_budget - sum(len(row['enemies']) for row in encounters))):
            number = allowed[(len(encounters) + slot) % len(allowed)]
            for _ in range(80):
                point = (round(rng.uniform(x0, x1)), round(rng.uniform(y0, y1)))
                if (not polygon.covers(Point(*point)) or math.dist(point, (start.x, start.y)) < 512
                        or not navigation.reaches_point(*point, sid=sid)
                        or any(trigger.distance(Point(*point)) < 80 for trigger in triggers)
                        or any(math.dist(point, (thing.x, thing.y)) < 192 for thing in doc.things
                               if thing.category == 'monster')):
                    continue
                if spawn_problem(doc, number, *point, margin=48, reachable=components, geometry=geometry):
                    continue
                thing = add(number, point, 'enemy', sid)
                thing.flags.deaf = True
                # A forgiving opening: ambush actors initially face away from
                # the start, so collecting equipment does not begin a gunfight.
                thing.angle = round(math.degrees(math.atan2(point[1] - start.y, point[0] - start.x)) / 45) * 45 % 360
                enemies.append(thing.id)
                break
        if enemies:
            encounters.append({'sector': sid, 'enemies': enemies, 'order': len(encounters)})
    enemy_records = [row for row in records if row['role'] == 'enemy']
    if not enemy_records:
        raise ValueError('intro layout has no safe encounter positions away from the start')
    total_hp = sum(MONSTER_HEALTH[row['type']] for row in enemy_records)
    # A deliberately conservative planning allowance, not a damage guarantee:
    # 20 useful damage/shell plus 8 spare shells; drops/pistol ammo are uncredited.
    requested_shells = math.ceil(total_hp / 20) + 8
    packs = max(0, math.ceil((requested_shells - 8) / 4))
    for _ in range(packs):
        add(2008, opening_point(), 'opening_ammo', start_sid)
    supplied_shells = 8 + packs * 4
    if supplied_shells > 50:
        raise ValueError('intro starting ammo budget exceeds vanilla shell capacity')
    health = 0
    # Give a recognizable recovery supply in each encounter space, outside
    # the monster boxes, rather than scattering unrelated random pickup types.
    recovery = encounters[1::2]
    if len(encounters) % 2:
        recovery.append(encounters[-1])
    for encounter in recovery:
        sid = encounter['sector']
        regions = [navigation.regions[key] for key in navigation.reached if key[0] == sid]
        for region in regions:
            x0, y0, x1, y1 = region.bounds
            for _ in range(80):
                point = (round(rng.uniform(x0, x1)), round(rng.uniform(y0, y1)))
                if (not region.covers(Point(*point))
                        or any(math.dist(point, previous) < 40 for previous in used_pickups)
                        or any(math.dist(point, (thing.x, thing.y)) < 96 for thing in doc.things
                               if thing.category == 'monster')
                        or any(trigger.distance(Point(*point)) < 96 for trigger in triggers)):
                    continue
                if spawn_problem(doc, 2012, *point, margin=4, reachable=components, geometry=geometry):
                    continue
                add(2012, point, 'recovery_health', sid)
                health += 25
                break
            else:
                continue
            break
    if health < 25 * len(recovery):
        raise ValueError('intro layout cannot supply the planned encounter recovery health')
    report = {'version': 1, 'profile': 'intro', 'seed': seed, 'encounters': encounters,
              'objective': {'old_exit_lines': old_exits, 'exit_line': objective.id,
                            'selection': objective_policy,
                            'sector': goal_sid, 'primary_route_sectors': route},
              'things': records, 'monsters': len(enemy_records), 'monster_budget': monster_budget,
              'enemy_hit_points': total_hp, 'shells_required_by_plan': requested_shells,
              'shells_supplied_before_combat': supplied_shells, 'shell_capacity': 50,
              'health_supplied': health, 'starting_armor': 100, 'interiors': details,
              'health_stations_planned': len(recovery),
              'limits': ['Procedural skill-3 pistol-start baseline, not learned encounter design.',
                         'Damage allowance is a planning assumption; aim, exposure and combat still need engine/human checks.',
                         'Monster drops and initial pistol bullets are not credited to the budget.']}
    require_valid(doc, require_connected=True)
    doc.extra['gameplay'] = report
    return report

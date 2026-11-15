"""Bounded procedural blue-card objective on an existing mandatory local door.

The current exit/topology stay intact. Layouts without a genuine gate or a
reachable earlier key room are rejected. General multi-key/script solving is
outside this small gameplay profile.
"""
from copy import deepcopy
import math
import random


def add_blue_key(doc, *, seed=0, game=None, max_candidates=8):
    from shapely import Point, LineString
    from generation.actors import spawn_problem
    from generation.style import _add_thing
    from generation.validator import require_valid
    from wad_parser.actions import KEY_COLORS
    from wad_parser.geometry import sector_geometry, sector_points
    from wad_parser.bsp_geometry import component_polygons
    from wad_parser.map_extractor import compute_derived, _sector_containing
    from wad_parser.navigation import PlayerNavigation
    from wad_parser.types import resolve_linedef_special

    if max_candidates < 1:
        raise ValueError('max_candidates must be positive')
    if game and game.sub_thing(5) != 5:
        raise ValueError('blue-key progression requires the stock blue keycard')
    if any(thing.type in KEY_COLORS for thing in doc.things):
        raise ValueError('blue-key progression requires a map without existing keys')
    require_valid(doc, require_connected=True)
    exits = [line for line in doc.linedefs if line.special in {11, 51, 52, 124}]
    if not exits or any(line.special != 11 for line in exits):
        raise ValueError('blue-key progression requires generated normal use exits')
    start = next(thing for thing in doc.things if thing.type == 1)
    start_sid = _sector_containing(start.x, start.y, doc.sectors, sector_geometry(doc), doc.vertices)
    room_sids = set()
    for sector in doc.sectors:
        for _, polygon in component_polygons(sector_points(doc, sector.id)):
            x, y, X, Y = polygon.bounds
            if min(X - x, Y - y) >= 384:
                room_sids.add(sector.id)
    hints = set()
    roles = doc.extra.get('room_roles', {})
    for room, sid in doc.extra.get('room_sectors', {}).items():
        if 'key' in roles.get(room, roles.get(str(room), [])):
            hints.add(sid)
    from wad_parser.clearance import MapClearance
    from wad_parser.traversal import component_distances, distances_from_start

    parent = {}
    terraces = (doc.extra.get('gameplay', {}).get('interiors', {}).get('terraces')
                or doc.extra.get('terrace_details', {}).get('terraces', []))
    for terrace in terraces:
        parent.update({step['sector']: terrace['base_sector'] for step in terrace['steps']})
    unlocked = {1, 31, 117, 118}
    distances = distances_from_start(doc)
    doors = sorted(
        {doc.sidedefs[line.left_sidedef].sector for line in doc.linedefs
         if line.special in unlocked and 0 <= line.left_sidedef < len(doc.sidedefs)},
        key=lambda sid: (sid not in distances, -distances.get(sid, 0), sid),
    )
    rejected = []
    rng = random.Random(seed)
    for door_sid in doors[:max_candidates]:
        trial = deepcopy(doc)
        lines = []
        for line in trial.linedefs:
            if (line.special in unlocked and 0 <= line.left_sidedef < len(trial.sidedefs)
                    and trial.sidedefs[line.left_sidedef].sector == door_sid):
                line.special = 26
                line.special_name, line.trigger_type = resolve_linedef_special(26)
                lines.append(line.id)
        trial.bounds, trial.derived = compute_derived(trial.vertices, trial.sectors,
            trial.linedefs, trial.sidedefs, trial.things)
        before = PlayerNavigation(trial)
        if any(before.reaches_line(trial.linedefs[line.id]) for line in exits):
            rejected.append({'door_sector': door_sid, 'reason': 'exit bypasses this door'})
            continue
        candidates = []
        for region in before.reached:
            sid = region[0]
            room = parent.get(sid, sid)
            if room not in room_sids or room == start_sid:
                continue
            polygon = before.regions[region]
            point = polygon.representative_point()
            candidates.append((room not in hints, -math.dist((start.x, start.y), (point.x, point.y)), region))
        triggers = [LineString([(trial.vertices[line.v1].x, trial.vertices[line.v1].y),
                                (trial.vertices[line.v2].x, trial.vertices[line.v2].y)])
                    for line in trial.linedefs if line.special]
        trial_clearance = MapClearance(trial.vertices, trial.sectors, trial.linedefs, trial.sidedefs)
        trial_components = component_distances(trial)
        point, key_sid = None, None
        for _, _, region in sorted(candidates):
            polygon = before.regions[region]
            representative = polygon.representative_point()
            x, y, X, Y = polygon.bounds
            proposals = [(round(representative.x), round(representative.y))]
            proposals += [(round(rng.uniform(x, X)), round(rng.uniform(y, Y))) for _ in range(64)]
            for proposal in proposals:
                p = Point(*proposal)
                if (not polygon.covers(p) or math.dist(proposal, (start.x, start.y)) < 512
                        or any(trigger.distance(p) < 80 for trigger in triggers)
                        or any(math.dist(proposal, (thing.x, thing.y)) <
                               (96 if thing.category == 'monster' else 40) for thing in trial.things)
                        or spawn_problem(trial, 5, *proposal, margin=4,
                                         reachable=trial_components, geometry=trial_clearance)):
                    continue
                point, key_sid = proposal, region[0]
                break
            if point is not None:
                break
        if point is None:
            rejected.append({'door_sector': door_sid, 'reason': 'no reachable earlier key-room position'})
            continue
        key = _add_thing(trial, *point, 5)
        frame_lines = _blue_frames(trial, lines, game)
        trial.bounds, trial.derived = compute_derived(trial.vertices, trial.sectors,
            trial.linedefs, trial.sidedefs, trial.things)
        try:
            validation = require_valid(trial, require_connected=True)
        except ValueError as exc:
            rejected.append({'door_sector': door_sid, 'reason': str(exc)})
            continue
        plan = {'version': 1, 'profile': 'blue-key', 'key_type': 5, 'key_thing': key.id,
                'key_sector': key_sid, 'key_room_sector': parent.get(key_sid, key_sid),
                'uses_predicted_key_room': parent.get(key_sid, key_sid) in hints,
                'door_sector': door_sid, 'door_lines': lines + frame_lines,
                'blue_frame_lines': frame_lines, 'exit_reachable_without_key': False,
                'key_reachable_without_key': before.reaches_point(*point, sid=key_sid),
                'validation': validation['progression'], 'rejected_candidates': rejected,
                'scope': 'One procedural blue card/gate on existing topology/objective; no multi-key/script claim.'}
        # Preserve report aliases while replacing only the changed raw fields.
        doc.vertices, doc.linedefs, doc.sidedefs = trial.vertices, trial.linedefs, trial.sidedefs
        doc.things, doc.bounds, doc.derived = trial.things, trial.bounds, trial.derived
        doc.extra['key_progression'] = plan
        if 'gameplay' in doc.extra:
            report = doc.extra['gameplay']
            report['progression'] = plan
            report['things'].append({'thing': key.id, 'type': 5, 'role': 'progression_key', 'sector': key_sid})
        return plan
    raise ValueError('no supported mandatory blue-key gate with an earlier key room: '
                     + '; '.join(row['reason'] for row in rejected))


def _blue_frames(doc, lines, game):
    """Add narrow stock blue strips without replacing the main door artwork."""
    from wad_parser.types import Vertex
    texture = next((name for name in ('DOORBLU', 'DOORBLU2')
                    if game is None or name in game.textures), None)
    if texture is None:
        return []
    result = []
    for line_id in lines:
        line = doc.linedefs[line_id]
        a, b = doc.vertices[line.v1], doc.vertices[line.v2]
        length = math.dist((a.x, a.y), (b.x, b.y))
        if length < 48 or a.x != b.x and a.y != b.y:
            continue  # Keep arbitrary source outlines unchanged.
        dx, dy = (b.x - a.x) / length, (b.y - a.y) / length
        p, q = len(doc.vertices), len(doc.vertices) + 1
        doc.vertices += [Vertex(p, round(a.x + 8 * dx), round(a.y + 8 * dy)),
                         Vertex(q, round(b.x - 8 * dx), round(b.y - 8 * dy))]
        outer_a, outer_b = line.v1, line.v2
        line.v1, line.v2 = p, q
        for left, right in ((outer_a, p), (q, outer_b)):
            edge = deepcopy(line)
            edge.id, edge.v1, edge.v2 = len(doc.linedefs), left, right
            sides = []
            for side_id in (line.right_sidedef, line.left_sidedef):
                side = deepcopy(doc.sidedefs[side_id])
                side.id = len(doc.sidedefs)
                side.upper_tex = side.lower_tex = texture
                side.x_offset = 0
                doc.sidedefs.append(side)
                sides.append(side.id)
            edge.right_sidedef, edge.left_sidedef = sides
            edge.extra['progression_detail'] = 'blue-key-frame'
            doc.linedefs.append(edge)
            result.append(edge.id)
    return result

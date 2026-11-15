"""Independent waypoint controller for optional generated-panel engine tests.

The controller uses the exported geometry, ordinary movement/use inputs, and
no warp/noclip commands. A budget failure is a controller result, not proof
that an arbitrary authored map is impossible to complete.
"""
from collections import deque
import heapq
import math


class _EpisodeClock:
    """Keep elapsed tics when ViZDoom clears its clock after a level ends."""
    def __init__(self, game):
        self.game = game
        self.elapsed = game.get_episode_time()

    def __getattr__(self, name):
        return getattr(self.game, name)

    def make_action(self, action, tics=1):
        self.elapsed = max(self.elapsed, self.game.get_episode_time()) + tics
        return self.game.make_action(action, tics)

    def get_episode_time(self):
        return max(self.elapsed, self.game.get_episode_time())


def _inside_path(doc, sid, start, target):
    from shapely import LineString, Point
    from shapely.ops import nearest_points
    from wad_parser.bsp_geometry import component_polygons
    from wad_parser.geometry import sector_points
    polygons = [polygon for _, polygon in component_polygons(sector_points(doc, sid))]
    polygon = next((part for part in polygons if part.covers(Point(*start))), None)
    if polygon is None:
        return [target]
    if not polygon.interiors and polygon.equals(polygon.convex_hull):
        return [target]
    # An inset avoids solid corners and unintended entries into teleport pads.
    safe = polygon.buffer(-16 * math.sqrt(2), join_style='mitre')
    if safe.is_empty:
        return [target]
    first = nearest_points(safe, Point(*start))[0]
    last = nearest_points(safe, Point(*target))[0]
    points = [(first.x, first.y), (last.x, last.y)]
    parts = [safe] if safe.geom_type == 'Polygon' else list(safe.geoms)
    for part in parts:
        for ring in [part.exterior, *part.interiors]:
            points.extend(list(ring.coords)[:-1])
    distances, previous, queue = {0: 0}, {}, [(0, 0)]
    while queue:
        distance, current = heapq.heappop(queue)
        if current == 1:
            path = [1]
            while path[-1] != 0:
                path.append(previous[path[-1]])
            return [points[index] for index in reversed(path)] + [target]
        if distance != distances[current]:
            continue
        for other, point in enumerate(points):
            if other == current or not safe.covers(LineString([points[current], point])):
                continue
            candidate = distance + math.dist(points[current], point)
            if candidate < distances.get(other, float('inf')):
                distances[other], previous[other] = candidate, current
                heapq.heappush(queue, (candidate, other))
    return [target]


def navigate_exit(game, doc, walk, *, max_tics=30000):
    from wad_parser.geometry import sector_geometry
    from wad_parser.components import ComponentIndex
    from wad_parser.traversal import potential_state, movement_neighbors
    from generation.progression import EXIT_SPECIALS
    game = _EpisodeClock(game)
    state = potential_state(doc)
    records = []
    movement_neighbors(doc.vertices, state['sectors'], doc.linedefs, doc.sidedefs,
                       [edge for edge in doc.derived.sector_graph['edges'] if edge.get('kind') == 'teleport'],
                       allow_doors=False, edge_records=records)
    adjacency = {}
    for source, destination, line, teleport in records:
        adjacency.setdefault(source, []).append((destination, line, teleport))
    lines = {line.id: line for line in doc.linedefs}
    positions = {v.id: (v.x, v.y) for v in doc.vertices}
    locator = ComponentIndex(positions, sector_geometry(doc))
    exits = {}
    for line in doc.linedefs:
        if line.special in EXIT_SPECIALS:
            exits.setdefault(doc.sidedefs[line.right_sidedef].sector, []).append(line)
    replans = 0
    while not game.is_episode_finished() and game.get_episode_time() < max_tics and replans < 64:
        replans += 1
        x, y = game.get_state().game_variables[:2]
        current = locator.owner(x, y)
        if current is None:
            return {'completed': False, 'reason': 'controller position has no sector', 'replans': replans}
        previous, queue, exit_sid = {current: None}, deque([current]), None
        while queue:
            sid = queue.popleft()
            if sid in exits:
                exit_sid = sid
                break
            for edge in adjacency.get(sid, []):
                if edge[0] not in previous:
                    previous[edge[0]] = (sid, edge)
                    queue.append(edge[0])
        if exit_sid is None:
            return {'completed': False, 'reason': 'controller found no supported exit route', 'replans': replans}
        route, sid = [], exit_sid
        while previous[sid] is not None:
            source, edge = previous[sid]
            route.append((source, edge))
            sid = source
        interrupted = False
        for source, (destination, line_id, teleport) in reversed(route):
            line = lines[line_id]
            a, b = positions[line.v1], positions[line.v2]
            length = math.dist(a, b)
            sign = 1 if doc.sidedefs[line.right_sidedef].sector == source else -1
            dx, dy = -(b[1] - a[1]) / length * sign, (b[0] - a[0]) / length * sign
            midpoint = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            approach = midpoint[0] - 20 * dx, midpoint[1] - 20 * dy
            crossing = midpoint[0] + 20 * dx, midpoint[1] + 20 * dy
            start = tuple(game.get_state().game_variables[:2])
            for target in _inside_path(doc, source, start, approach):
                if not walk(game, target, tolerance=4, max_tics=2000, stop_on_teleport=True):
                    interrupted = True
                    break
            if interrupted:
                break
            arrival = tuple(teleport['destination_point']) if teleport and 'destination_point' in teleport else None
            if not walk(game, crossing, arrival=arrival, tolerance=4, max_tics=2000, stop_on_teleport=True):
                interrupted = True
                break
        if interrupted:
            continue
        exit_line = exits[exit_sid][0]
        a, b = positions[exit_line.v1], positions[exit_line.v2]
        midpoint = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        start = tuple(game.get_state().game_variables[:2])
        for target in _inside_path(doc, exit_sid, start, midpoint)[:-1]:
            if not walk(game, target, tolerance=4, stop_on_teleport=True):
                interrupted = True
                break
        if not interrupted:
            walk(game, midpoint, tolerance=0, max_tics=500, stop_on_teleport=True)
    completed = game.is_episode_finished() and not game.is_player_dead() and game.get_episode_time() < max_tics
    return {'completed': completed, 'tics': game.get_episode_time(), 'replans': replans,
            'reason': None if completed else 'controller exhausted its navigation budget'}

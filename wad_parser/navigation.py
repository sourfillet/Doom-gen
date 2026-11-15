"""Static player configuration space inside potentially open map geometry.

Subtract axis-aligned player-box sweeps of blocking lines, then connect the
remaining regions only through supported directed openings or teleports.
This detects narrow necks within a single authored sector. Door/key action
history, moving floors, monsters, and timed puzzles remain separate problems.
"""
from __future__ import annotations

from collections import deque


class PlayerNavigation:
    def __init__(self, doc, *, state=None, radius=16, height=56, clearance_layout=None):
        from shapely import LineString, Point, STRtree, convex_hull, multipoints, set_precision, union_all
        from wad_parser.bsp_geometry import GRID, component_polygons
        from wad_parser.geometry import sector_points
        from wad_parser.traversal import potential_state, movement_neighbors
        from wad_parser.clearance import MapClearance
        from wad_parser.portal_contacts import portal_intervals, overlap_length
        self.doc, self.radius = doc, radius
        state = state or potential_state(doc)
        sectors = {sector.id: sector for sector in state['sectors']}
        self.positions = {vertex.id: (vertex.x, vertex.y) for vertex in doc.vertices}
        self.clearance = (MapClearance(doc.vertices, state['sectors'], doc.linedefs, doc.sidedefs)
                          if clearance_layout is None else clearance_layout.with_heights(state['sectors']))
        sweeps, openings, use_segments, use_ids = [], [], [], []
        # A tiny tolerance permits exact outside-edge contact, matching Doom's
        # box collision test and retaining exact 32-unit parallel passages.
        r = radius - 1e-7
        for line, a, b, owners in self.clearance.lines.values():
            one_sided = any(owner not in sectors for owner in owners)
            if (one_sided or line.special or
                    min(sectors[owner].ceil_height for owner in owners)
                    <= max(sectors[owner].floor_height for owner in owners)):
                use_segments.append(LineString([a, b]))
                use_ids.append(line.id)
            solid = (any(owner not in sectors for owner in owners) or line.flags.impassable
                     or line.flags.block_players or line.flags.block_everything)
            if not solid:
                floor = max(sectors[owner].floor_height for owner in owners)
                ceiling = min(sectors[owner].ceil_height for owner in owners)
                solid = ceiling - floor < height
            openings.append(None if solid else (floor, ceiling))
            sweeps.append(convex_hull(multipoints([(x + dx, y + dy) for x, y in (a, b)
                                                   for dx in (-r, r) for dy in (-r, r)])))
        wall_index = STRtree(sweeps)
        clearance_lines = list(self.clearance.lines.values())
        step_up_neighbors = {s.id: set() for s in doc.sectors}
        for (line, a, b, owners), opening in zip(clearance_lines, openings):
            if opening is not None and len(owners) >= 2 and owners[0] in sectors and owners[1] in sectors:
                o0, o1 = owners[0], owners[1]
                if 0 <= sectors[o1].floor_height - sectors[o0].floor_height <= 24:
                    step_up_neighbors[o0].add(o1)
                if 0 <= sectors[o0].floor_height - sectors[o1].floor_height <= 24:
                    step_up_neighbors[o1].add(o0)
        self.use_segments, self.use_ids, self.use_index = use_segments, use_ids, STRtree(use_segments)
        self.regions, by_sector = {}, {}
        sector_polygons = []
        for sector in doc.sectors:
            sid = sector.id
            parts = []
            if sectors[sid].ceil_height - sectors[sid].floor_height >= height:
                for _, polygon in component_polygons(sector_points(doc, sid)):
                    sector_polygons.append((sid, polygon))
                    obstacles = []
                    for index in wall_index.query(polygon):
                        idx = int(index)
                        opening = openings[idx]
                        if opening is None:
                            obstacles.append(sweeps[idx])
                        elif (min(opening[1], sectors[sid].ceil_height)
                              - max(opening[0], sectors[sid].floor_height) < height):
                            obstacles.append(sweeps[idx])
                        elif opening[0] - sectors[sid].floor_height > 24:
                            # Model collision-floor changes before sector-center crossings for compact stairs
                            owners = clearance_lines[idx][3]
                            can_climb = any(
                                owner is not None and owner in sectors and owner in step_up_neighbors.get(sid, set())
                                and 0 <= opening[0] - sectors[owner].floor_height <= 24
                                for owner in owners
                            )
                            if not can_climb:
                                obstacles.append(sweeps[idx])
                    free = polygon.difference(union_all(obstacles)) if obstacles else polygon
                    components = [free] if free.geom_type == 'Polygon' else list(getattr(free, 'geoms', []))
                    parts.extend(part for part in components if part.geom_type == 'Polygon' and part.area > 1e-9)
            by_sector[sid] = []
            for index, polygon in enumerate(parts):
                key = sid, index
                self.regions[key] = polygon
                by_sector[sid].append(key)
        self.by_sector = by_sector
        records = []
        movement_neighbors(doc.vertices, state['sectors'], doc.linedefs, doc.sidedefs,
                           state.get('teleport_edges', [edge for edge in doc.derived.sector_graph['edges']
                                                        if edge.get('kind') == 'teleport']),
                           allow_doors=False, edge_records=records, clearance=self.clearance,
                           potential_teleport_consumption=state.get('potential_teleport_consumption', False))
        self.adjacency = {key: set() for key in self.regions}
        self.crossings = {}
        self.teleport_events = {}
        event_lines = state.get('teleport_events', set())
        lines = {line.id: line for line in doc.linedefs}
        contacts = {}

        def intervals(node, line_id, mouth):
            key = node, line_id
            if key not in contacts:
                contacts[key] = portal_intervals(self.regions[node], mouth)
            return contacts[key]

        bridge_cache, polygon_index = {}, []

        def descending_bridges(source, line_id, mouth):
            # Only the swept ledge can support the player at its old height.
            # Any non-source sector under that sweep (the destination, or a
            # lower step beside it) can hold the box meanwhile; keep each
            # one's solid/ceiling obstacles at the old height and do not
            # infer a flight across the whole destination sector.
            source_floor = sectors[source].floor_height
            cache_key = source, line_id
            if cache_key in bridge_cache:
                return bridge_cache[cache_key]
            if not polygon_index:
                polygon_index.append(STRtree([polygon for _, polygon in sector_polygons]))
            tolerance_radius = radius + 1e-5
            # Same point set and GEOS hull as MultiPoint(...).convex_hull,
            # without building a Python Point per corner (history searches
            # rebuild navigation hundreds of times).
            ledge = convex_hull(multipoints([(x + dx, y + dy) for x, y in mouth.coords
                                             for dx in (-tolerance_radius, tolerance_radius)
                                             for dy in (-tolerance_radius, tolerance_radius)]))
            parts, under = [], set()
            for index in polygon_index[0].query(ledge):
                sid, polygon = sector_polygons[int(index)]
                if (sid == source or sectors[sid].floor_height > source_floor
                        or sectors[sid].ceil_height - source_floor < height):
                    continue
                # Compiled sector outlines can carry sub-fixed-point seams
                # between neighbours. Snap the outlines (never the clearance
                # sweeps) so adjacent pieces share an edge and fuse. GEOS keeps
                # a set grid for later overlays, so drop it before clipping.
                local = set_precision(set_precision(polygon, GRID), 0).intersection(ledge)
                if local.is_empty:
                    continue
                under.add(sid)
                obstacles = []
                for wall in wall_index.query(local):
                    idx = int(wall)
                    opening = openings[idx]
                    if (opening is None or
                        min(opening[1], sectors[sid].ceil_height)
                        - max(opening[0], source_floor) < height or
                        opening[0] - source_floor > 24):
                        obstacles.append(sweeps[idx])
                parts.append(local.difference(union_all(obstacles)) if obstacles else local)
            bridge = union_all(parts)
            pieces = [bridge] if bridge.geom_type == 'Polygon' else list(getattr(bridge, 'geoms', []))
            bridge_cache[cache_key] = under, [(part, portal_intervals(part, mouth)) for part in pieces
                                              if part.geom_type == 'Polygon' and part.area > 1e-9]
            return bridge_cache[cache_key]

        for source, destination, line_id, teleport in records:
            if line_id in event_lines and teleport is None:
                # Active one-shot crossings are explicit history events.
                # Backside/failed activation consumption is not inferred.
                continue
            line = lines[line_id]
            mouth = LineString([self.positions[line.v1], self.positions[line.v2]])
            descending = sectors[source].floor_height - sectors[destination].floor_height > 24
            for a in by_sector.get(source, []):
                if teleport:
                    entry = self.regions[a].intersection(mouth)
                    if entry.length <= 1e-8:
                        continue
                else:
                    source_intervals = intervals(a, line_id, mouth)
                    if not source_intervals:
                        continue
                if teleport:
                    point = teleport.get('destination_point')
                    if point is None:
                        continue
                    destinations = [key for key in by_sector.get(destination, [])
                                    if self.regions[key].covers(Point(*point))]
                else:
                    if descending:
                        # The lower sector's static free space is inset from
                        # the high ledge (it cannot be climbed from below).
                        # A descending player remains at the upper floor until
                        # its box clears the ledge, so land only where a bridge
                        # piece touching this mouth reaches that free space.
                        # A sector across the mouth can be too thin to stand
                        # in; any lower sector under the bridge may hold it.
                        under, pieces = descending_bridges(source, line_id, mouth)
                        contacts_here = [part for part, bridge_intervals in pieces
                                         if overlap_length(source_intervals, bridge_intervals,
                                                           mouth.length) > 1e-8]
                        destinations = [key for sid in sorted(under | {destination})
                                        for key in by_sector.get(sid, [])
                                        if any(self.regions[key].intersection(part).length > 1e-8
                                               for part in contacts_here)]
                    else:
                        destinations = [key for key in by_sector.get(destination, [])
                                        if overlap_length(source_intervals, intervals(key, line_id, mouth),
                                                          mouth.length) > 1e-8]
                for b in destinations:
                    if line_id in event_lines:
                        self.teleport_events.setdefault(line_id, []).append((a, b, tuple(teleport['destination_point'])))
                    else:
                        self.adjacency[a].add(b)
                    if teleport is None:
                        self.crossings.setdefault(line.id, []).append((a, b))
        starts = [thing for thing in doc.things if thing.type == 1]
        self.reached = self.reachable_from((start.x, start.y) for start in starts)

    def reachable_from(self, points):
        """Reachable free regions from actual positions in this geometry state."""
        from shapely import Point
        reached = set()
        queue = deque()
        for x, y in points:
            for key, polygon in self.regions.items():
                if polygon.covers(Point(x, y)):
                    queue.append(key)
                    reached.add(key)
        while queue:
            for other in self.adjacency[queue.popleft()]:
                if other not in reached:
                    reached.add(other)
                    queue.append(other)
        return reached

    def reaches_point(self, x, y, sid=None):
        from shapely import Point
        return any((sid is None or key[0] == sid) and self.regions[key].covers(Point(x, y))
                   for key in self.reached)

    def reaches_line(self, line):
        return self.line_approach(line) is not None

    def line_approach(self, line, *, reached=None, crossing=False):
        return next(self.line_approaches(line, reached=reached, crossing=crossing), None)

    def line_approaches(self, line, *, reached=None, crossing=False):
        """A reachable use position, or a position after a directed crossing.

        Returning a point permits action-history searches to keep the player's
        location instead of pooling events from incompatible one-way branches.
        """
        from shapely import Point
        import math
        reached = self.reached if reached is None else reached
        visited = set()
        if crossing or line.special in {52, 124}:
            for source, destination in self.crossings.get(line.id, []):
                if source in reached and destination not in visited:
                    visited.add(destination)
                    p = self.regions[destination].representative_point()
                    yield p.x, p.y
            return
        sid = self.doc.sidedefs[line.right_sidedef].sector
        a, b = self.positions[line.v1], self.positions[line.v2]
        length = math.dist(a, b)
        if length <= 1e-8:
            return
        dx, dy = (b[1] - a[1]) / length, -(b[0] - a[0]) / length
        # Stand on the front side, within vanilla's 64-unit use range.
        for distance in (32, 24, self.radius + 1e-6):
            for t in (0.5, 0.25, 0.75):
                point = Point(a[0] + (b[0] - a[0]) * t + dx * distance,
                              a[1] + (b[1] - a[1]) * t + dy * distance)
                owners = [key for key in sorted(reached) if key[0] == sid and key not in visited
                          and self.regions[key].covers(point)]
                if (owners and self._clear_use_ray(point, Point(a[0] + (b[0] - a[0]) * t,
                                                               a[1] + (b[1] - a[1]) * t), line.id)):
                    visited.update(owners)
                    yield point.x, point.y

    def _clear_use_ray(self, start, target, line_id):
        from shapely import LineString
        ray = LineString([start, target])
        for index in self.use_index.query(ray, predicate='intersects'):
            index = int(index)
            if (self.use_ids[index] != line_id
                    and self.use_segments[index].intersection(ray).distance(start) < ray.length - 1e-6):
                return False
        return True

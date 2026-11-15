"""Reusable ring/component lookup for many linedefs in the same geometry."""
from __future__ import annotations

import math

from wad_parser.geometry import point_in_ring, ring_depths, signed_area, resolve_rings


class ComponentIndex:
    def __init__(self, vertices, rings):
        from shapely import Polygon, contains_xy, intersects_xy, prepare
        self.positions = vertices if isinstance(vertices, dict) else {v.id: (v.x, v.y) for v in vertices}
        self.rings, self.depths, self.areas, self.boxes, self.edges = {}, {}, {}, {}, {}
        self.coordinate_edges = {}
        self.prepared, self.bounds, self.sector_areas, self.sector_perimeters = {}, {}, {}, {}
        self.contains_xy = contains_xy
        self.intersects_xy = intersects_xy
        for sid, parts in enumerate(rings):
            points = resolve_rings(parts, self.positions)
            self.rings[sid] = points
            self.depths[sid] = ring_depths(points)
            self.areas[sid] = [abs(signed_area(ring)) for ring in points]
            self.boxes[sid] = [(min(x for x, _ in ring), min(y for _, y in ring),
                                max(x for x, _ in ring), max(y for _, y in ring)) for ring in points]
            self.sector_areas[sid] = max(0, sum(area * (-1 if depth % 2 else 1)
                                              for area, depth in zip(self.areas[sid], self.depths[sid])))
            self.sector_perimeters[sid] = sum(math.dist(a, b) for ring in points
                                              for a, b in zip(ring, ring[1:] + ring[:1]))
            if points:
                boxes = self.boxes[sid]
                self.bounds[sid] = (min(box[0] for box in boxes), min(box[1] for box in boxes),
                                    max(box[2] for box in boxes), max(box[3] for box in boxes))
            prepared = {}
            for index, ring in enumerate(points):
                if len(ring) >= 64:
                    polygon = Polygon(ring)
                    if polygon.is_valid:
                        prepare(polygon)
                        prepared[index] = polygon
            self.prepared[sid] = prepared
            shells = [index for index, depth in enumerate(self.depths[sid]) if depth % 2 == 0]
            for index, ring in enumerate(points):
                owner = index
                if self.depths[sid][index] % 2:
                    candidates = [shell for shell in shells if point_in_ring(*points[index][0], points[shell])]
                    if not candidates:
                        continue
                    owner = min(candidates, key=lambda shell: self.areas[sid][shell])
                for a, b in zip(ring, ring[1:] + ring[:1]):
                    self.coordinate_edges[(sid, a, b)] = self.coordinate_edges[(sid, b, a)] = (sid, owner)
                if parts[index] and isinstance(parts[index][0], int):
                    ids = parts[index]
                    for a, b in zip(ids, ids[1:] + ids[:1]):
                        self.edges[(sid, a, b)] = self.edges[(sid, b, a)] = (sid, owner)

    def point(self, sid, x, y):
        inside, boundary = [], []
        for index, (ring, box) in enumerate(zip(self.rings.get(sid, ()), self.boxes.get(sid, ()))):
            if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                prepared = self.prepared[sid].get(index)
                if self.contains_xy(prepared, x, y) if prepared is not None else point_in_ring(x, y, ring):
                    inside.append(index)
                elif (self.intersects_xy(prepared, x, y) if prepared is not None
                      else point_in_ring(x, y, ring, boundary=True)):
                    boundary.append(index)
        if boundary:
            # A hole boundary belongs to the surrounding filled shell, while
            # its strict interior is still void. Keep the same owner for small
            # rings and prepared large rings, including nested islands.
            shells = [index for index in inside + boundary if self.depths[sid][index] % 2 == 0]
            return (sid, min(shells, key=lambda index: self.areas[sid][index])) if shells else None
        if len(inside) % 2 == 0:
            return None
        shells = [index for index in inside if self.depths[sid][index] % 2 == 0]
        return (sid, min(shells, key=lambda index: self.areas[sid][index])) if shells else None

    def owner(self, x, y):
        candidates = [sid for sid, box in self.bounds.items()
                      if box[0] <= x <= box[2] and box[1] <= y <= box[3] and self.point(sid, x, y) is not None]
        return min(candidates, key=lambda sid: (self.sector_areas[sid], sid)) if candidates else None

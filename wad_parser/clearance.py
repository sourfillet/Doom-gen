"""Player/actor radius-box clearance across open sector boundaries.

Doom collides with blocking linedefs and the floor/ceiling opening of every
line touched by the actor box. An individual decorative boundary is not a
wall, and its length does not determine whether the actor fits.
"""
from __future__ import annotations

import math
from collections import defaultdict


def segment_intersects_box(a, b, x, y, radius):
    """Intersect the open box; touching its outside edge is permitted."""
    lo, hi, epsilon = 0.0, 1.0, 1e-7
    for axis, center in ((0, x), (1, y)):
        lower, upper = center - radius + epsilon, center + radius - epsilon
        delta = b[axis] - a[axis]
        if abs(delta) < epsilon:
            if not lower < a[axis] < upper:
                return False
        else:
            begin, end = sorted(((lower - a[axis]) / delta, (upper - a[axis]) / delta))
            lo, hi = max(lo, begin), min(hi, end)
        if lo > hi:
            return False
    return True


class MapClearance:
    """Spatially indexed raw geometry with optional potential mover heights."""
    CELL = 128

    def __init__(self, vertices, sectors, linedefs, sidedefs, *, floors=None, ceilings=None):
        positions = {v.id: (v.x, v.y) for v in vertices}
        self.floors = {s.id: s.floor_height for s in sectors}
        self.ceilings = {s.id: s.ceil_height for s in sectors}
        self.floors.update(floors or {})
        self.ceilings.update(ceilings or {})
        self.lines = {}
        self.cells = defaultdict(set)
        for line in linedefs:
            if line.v1 not in positions or line.v2 not in positions:
                continue
            a, b = positions[line.v1], positions[line.v2]
            if not all(math.isfinite(value) for point in (a, b) for value in point):
                continue
            owners = [sidedefs[index].sector if 0 <= index < len(sidedefs) else None
                      for index in (line.right_sidedef, line.left_sidedef)]
            self.lines[line.id] = (line, a, b, owners)
            # Supercover by samples plus neighboring cells, avoiding the huge
            # bounding rectangle of a long diagonal line.
            count = max(1, math.ceil(max(abs(b[0] - a[0]), abs(b[1] - a[1])) / self.CELL))
            for index in range(count + 1):
                gx = math.floor((a[0] + (b[0] - a[0]) * index / count) / self.CELL)
                gy = math.floor((a[1] + (b[1] - a[1]) * index / count) / self.CELL)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        self.cells[(gx + dx, gy + dy)].add(line.id)

    def with_heights(self, sectors):
        """Share immutable XY indexing, with independent mover-state heights.

        Use only while the authored vertices, lines and sides are unchanged.
        Cached navigation states must never share mutable height dictionaries.
        """
        from copy import copy
        result = copy(self)
        result.floors = {sector.id: sector.floor_height for sector in sectors}
        result.ceilings = {sector.id: sector.ceil_height for sector in sectors}
        return result

    def lines_near(self, line_id, radius):
        """Indexed lines that may touch a box centred within ``radius`` of a line."""
        _, a, b, _ = self.lines[line_id]
        found = set()
        for gx in range(math.floor((min(a[0], b[0]) - radius) / self.CELL),
                        math.floor((max(a[0], b[0]) + radius) / self.CELL) + 1):
            for gy in range(math.floor((min(a[1], b[1]) - radius) / self.CELL),
                            math.floor((max(a[1], b[1]) + radius) / self.CELL) + 1):
                found.update(self.cells.get((gx, gy), ()))
        return found

    def problem(self, sid, x, y, *, radius=16, height=56, max_step=24, monster=False, support=None):
        # ``support`` is (line id, height): a player who left that ledge's
        # high side is still at least that high while its box overlaps it.
        if sid not in self.floors:
            return 'outside playable space'
        candidates = set()
        for gx in range(math.floor((x - radius) / self.CELL), math.floor((x + radius) / self.CELL) + 1):
            for gy in range(math.floor((y - radius) / self.CELL), math.floor((y + radius) / self.CELL) + 1):
                candidates.update(self.cells.get((gx, gy), ()))
        floor, ceiling = self.floors[sid], self.ceilings[sid]
        touched_lines = []
        bases = {self.floors[sid]}
        for index in candidates:
            line, a, b, owners = self.lines[index]
            if not segment_intersects_box(a, b, x, y, radius):
                continue
            if (any(owner not in self.floors for owner in owners) or line.flags.impassable
                    or line.flags.block_players or line.flags.block_everything
                    or monster and (line.flags.block_monsters or line.flags.block_land_monsters)):
                return 'insufficient wall clearance'
            touched_lines.append(owners)
            if support is not None and index == support[0]:
                bases.add(support[1])
            floor = max(floor, *(self.floors[owner] for owner in owners))
            ceiling = min(ceiling, *(self.ceilings[owner] for owner in owners))
        if floor - max(bases) > max_step:
            # Model collision-floor changes before sector-center crossings for compact stairs:
            # verify that floor can be reached from self.floors[sid] via step-up transitions <= max_step
            # across open boundaries touched by the actor's bounding box.
            reachable_floors = set(bases)
            changed = True
            while changed:
                changed = False
                for owners in touched_lines:
                    if len(owners) >= 2 and owners[0] in self.floors and owners[1] in self.floors:
                        f0, f1 = self.floors[owners[0]], self.floors[owners[1]]
                        if f0 in reachable_floors and 0 <= f1 - f0 <= max_step and f1 not in reachable_floors:
                            reachable_floors.add(f1)
                            changed = True
                        if f1 in reachable_floors and 0 <= f0 - f1 <= max_step and f0 not in reachable_floors:
                            reachable_floors.add(f0)
                            changed = True
            if floor not in reachable_floors:
                return 'step exceeds actor clearance'
        if ceiling - floor < height:
            return 'insufficient vertical clearance'
        return None

    def crossing(self, line, sid, *, radius=16, height=56):
        _, a, b, _ = self.lines[line.id]
        if a == b:
            return False
        # A clear crossing is evidence of an opening; failure remains
        # conservative. Long mouths get more probes without quadratic scans.
        count = min(64, max(4, math.ceil(math.dist(a, b) / max(radius, 1))))
        samples = [0.5] + [index / count for index in range(count + 1)]
        return any(self.problem(sid, a[0] + (b[0] - a[0]) * t,
                                a[1] + (b[1] - a[1]) * t,
                                radius=radius, height=height) is None for t in samples)

    def supported_crossing(self, line, sid, ledge, ledge_floor, *, radius=16, height=56):
        """Cross ``line`` out of ``sid`` while the box still overlaps ``ledge``.

        Vanilla keeps the floor at the ledge's opening bottom until the box
        clears it, so a thin landing strip is crossed at the old height. Only
        positions whose open box intersects the ledge are probed; once it is
        cleared the player falls and the ordinary crossing applies.
        """
        _, a, b, _ = self.lines[line.id]
        _, p, q, _ = self.lines[ledge]
        if a == b or p == q:
            return False
        lo, hi = 0.0, 1.0
        for axis in (0, 1):
            low, high = min(p[axis], q[axis]) - radius, max(p[axis], q[axis]) + radius
            delta = b[axis] - a[axis]
            if abs(delta) < 1e-9:
                if not low <= a[axis] <= high:
                    return False
                continue
            begin, end = sorted(((low - a[axis]) / delta, (high - a[axis]) / delta))
            lo, hi = max(lo, begin), min(hi, end)
        if lo > hi:
            return False
        count = min(64, max(4, math.ceil(math.dist(a, b) * (hi - lo) / max(radius / 2, 1))))
        for t in [(lo + hi) / 2] + [lo + (hi - lo) * index / count for index in range(count + 1)]:
            x, y = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
            if (segment_intersects_box(p, q, x, y, radius)
                    and self.problem(sid, x, y, radius=radius, height=height,
                                     support=(ledge, ledge_floor)) is None):
                return True
        return False

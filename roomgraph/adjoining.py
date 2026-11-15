"""Bounded room growth toward neighboring rooms, with real wall thickness.

This realizes captured graph connections procedurally; it does not predict
architecture. Larger height changes and routed bends keep their stair space.
"""
from __future__ import annotations

import math
import random


VERSION = 1
WALL_BANDS = {'open': 16, 'door': 32, 'corridor': 64}


def _overlap(a, b, margin=0):
    return (min(a[2], b[2]) + margin > max(a[0], b[0])
            and min(a[3], b[3]) + margin > max(a[1], b[1]))


def _envelope(edge, placed, rects):
    if edge.kind == 'teleport':
        return None
    a, b = placed[edge.a], placed[edge.b]
    if a[1] == b[1] and a[0] != b[0]:
        west, east = (edge.a, edge.b) if a[0] < b[0] else (edge.b, edge.a)
        left, right = rects[west], rects[east]
        return (left[2], max(left[1], right[1]), right[0], min(left[3], right[3]))
    if a[0] == b[0] and a[1] != b[1]:
        south, north = (edge.a, edge.b) if a[1] < b[1] else (edge.b, edge.a)
        low, high = rects[south], rects[north]
        return (max(low[0], high[0]), low[3], min(low[2], high[2]), high[1])
    return None


def adjoining_rects(graph, placed, rects, *, seed=0):
    """Grow eligible main rooms without overlapping any room/transit envelope.

    Grid order and every requested edge are preserved. Growth on each facing
    side is bounded by 75% of that room's original dimension and 384 units.
    Reserve the complete possible mouth span of other routes, so later seeded
    entrance placement cannot turn an accepted growth into a corridor crossing.
    """
    original = {rid: tuple(round(v) for v in bounds) for rid, bounds in rects.items()}
    rects = dict(original)
    rooms = {room.id: room for room in graph.rooms}
    selected, accepted, rejected = set(), [], []
    order = list(range(len(graph.connections)))
    random.Random(seed).shuffle(order)
    physical_counts = {}
    for edge in graph.connections:
        if edge.kind != 'teleport':
            key = frozenset((edge.a, edge.b))
            physical_counts[key] = physical_counts.get(key, 0) + 1
    for index in order:
        edge = graph.connections[index]
        if edge.kind not in WALL_BANDS:
            continue
        a, b = rooms[edge.a], rooms[edge.b]
        if a.role == 'corridor' or b.role == 'corridor':
            continue
        reason = None
        difference = abs(a.floor_h - b.floor_h)
        if (difference > 24 or edge.kind == 'door' and difference
                or min(a.ceil_h, b.ceil_h) - max(a.floor_h, b.floor_h) < 64):
            reason = 'retain stair run or headroom'
        elif physical_counts[frozenset((edge.a, edge.b))] != 1:
            reason = 'parallel physical connections'
        envelope = _envelope(edge, placed, rects)
        if envelope is None:
            reason = 'not a straight routed neighbor'
        if reason:
            rejected.append({'a': edge.a, 'b': edge.b, 'kind': edge.kind, 'reason': reason})
            continue
        axis = 0 if placed[edge.a][1] == placed[edge.b][1] else 1
        left, right = sorted((edge.a, edge.b), key=lambda rid: placed[rid][axis])
        gap = rects[right][axis] - rects[left][axis + 2]
        band = WALL_BANDS[edge.kind]
        cap_left = min(384, .75 * (original[left][axis + 2] - original[left][axis]))
        cap_right = min(384, .75 * (original[right][axis + 2] - original[right][axis]))
        low = max(rects[left][axis + 2], original[right][axis] - cap_right - band)
        high = min(rects[right][axis] - band, original[left][axis + 2] + cap_left)
        low, high = math.ceil(low / 8) * 8, math.floor(high / 8) * 8
        if gap <= band or low > high:
            rejected.append({'a': edge.a, 'b': edge.b, 'kind': edge.kind, 'reason': 'bounded growth cannot close gap'})
            continue
        plane = min(high, max(low, round((rects[left][axis + 2] + rects[right][axis] - band) / 16) * 8))
        trial = dict(rects)
        first, second = list(rects[left]), list(rects[right])
        first[axis + 2], second[axis] = plane, plane + band
        trial[left], trial[right] = tuple(first), tuple(second)
        blocked = any(_overlap(trial[x], trial[y], margin=8)
                      for x in trial for y in trial if x < y)
        if not blocked:
            for other in graph.connections:
                transit = _envelope(other, placed, trial)
                if transit is None:
                    continue
                if any(_overlap(bounds, transit, margin=8)
                       for rid, bounds in trial.items() if rid not in {other.a, other.b}):
                    blocked = True
                    break
        if blocked:
            rejected.append({'a': edge.a, 'b': edge.b, 'kind': edge.kind, 'reason': 'room or transit clearance'})
            continue
        rects = trial
        selected.add(index)
        accepted.append({'a': edge.a, 'b': edge.b, 'kind': edge.kind, 'axis': axis,
                         'previous_gap': gap, 'wall_band': band})
    return rects, selected, {
        'version': VERSION, 'connections': accepted, 'rejected': rejected,
        'requested_bounds': {str(k): list(v) for k, v in original.items()},
        'realized_bounds': {str(k): list(v) for k, v in rects.items()},
        'scope': 'Procedural bounded room growth and thin wall bands; graph/floors retained, no learned architecture claim.'}


def emit_threshold(quad, room_a, room_b, floor_a, floor_b, ceil_a, ceil_b,
                   is_door, new_sector, add_line):
    """Emit a short opening, or a flat 32-unit door usable from both rooms."""
    p0, p1, p2, p3 = quad
    floor = max(floor_a, floor_b)
    sid = new_sector(floor, min(ceil_a, ceil_b), is_door)
    add_line(p1, p2, sid)
    add_line(p3, p0, sid)
    if is_door:
        add_line(p1, p0, room_a, sid, special=1)
        add_line(p3, p2, room_b, sid, special=1)
    else:
        add_line(p0, p1, sid, room_a)
        add_line(p2, p3, sid, room_b)

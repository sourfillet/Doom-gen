"""Bounded procedural variation for the routed room assembler.

Rows and columns remain ordered, preserving the router's crossing structure.
Openings move only within the shared room wall span. Corner cuts retain a
clear center and stay away from the actual openings; they never enlarge a room.
"""
from __future__ import annotations

import math
import random


PROFILES = ('classic', 'varied', 'compact', 'adjoining')
VERSION = 1
WALL_MARGIN = 32.0
COMPACT_GAP = 192.0


def compact_axis_positions(half_extents, minimum_spans, rng: random.Random):
    """Compact ordered grid coordinates without changing route crossings.

    Adjacent occupied lanes reserve their largest room half extents and a
    corridor gap. Additional (left, right, distance) constraints reserve stair
    runs for routed edges, even across several lanes. All constraints point
    forward, so one pass gives the minimum positions before bounded jitter.
    Two extra units protect the clear gaps when vertices are rounded to ints.
    Empty grid lanes consume no world space; routed bend rooms still do.
    """
    cells = sorted(half_extents)
    constraints = {cell: [] for cell in cells}
    for left, right, distance in minimum_spans:
        if left not in constraints or right not in constraints or left >= right:
            raise ValueError('compact spacing constraints must connect ordered occupied lanes')
        constraints[right].append((left, distance))
    jitters = [8 * rng.randrange(5) for _ in range(max(len(cells) - 1, 0))]
    def _solve(scale: float) -> dict:
        pos = {cells[0]: 0}
        for (previous, cell), jitter in zip(zip(cells, cells[1:]), jitters):
            value = pos[previous] + half_extents[previous] + half_extents[cell] + COMPACT_GAP + 2
            for left, distance in constraints[cell]:
                value = max(value, pos[left] + distance)
            pos[cell] = math.ceil(value / 8) * 8 + int(jitter * scale // 8) * 8
        return pos
    positions = _solve(1.0)
    envelope = (positions[cells[-1]] + half_extents[cells[-1]]) - (positions[cells[0]] - half_extents[cells[0]])
    if envelope > 65534.0 and any(jitters):
        base_pos = _solve(0.0)
        base_env = (base_pos[cells[-1]] + half_extents[cells[-1]]) - (base_pos[cells[0]] - half_extents[cells[0]])
        budget = max(0.0, 65534.0 - base_env)
        positions = _solve(min(1.0, budget / sum(jitters)))
        if (positions[cells[-1]] + half_extents[cells[-1]]) - (positions[cells[0]] - half_extents[cells[0]]) > 65534.0:
            positions = base_pos
    return positions


def axis_positions(cells, slot, extent, rng: random.Random):
    """Stretch grid intervals without reducing stair space or exceeding int16.

    Coordinate translation is handled by the publisher. Budget the complete
    envelope here so variation does not make an otherwise fitting map too big.
    """
    cells = sorted(set(cells))
    low, high = cells[0], cells[-1]
    count = high - low
    room_span = count * slot + extent
    budget = max(0.0, 65534.0 - room_span)
    steps = [rng.uniform(0.02, 0.16) * slot for _ in range(count)]
    factor = min(1.0, budget / sum(steps)) if steps else 1.0
    positions = {low: (low + 0.5) * slot}
    for index, step in enumerate(steps, low + 1):
        positions[index] = positions[index - 1] + slot + step * factor
    return {cell: positions[cell] for cell in cells}


def opening_center(low, high, width, rng: random.Random):
    """Move a shared mouth, keeping jamb space in both endpoint rooms."""
    center = (low + high) / 2
    freedom = max(0.0, (high - low - width) / 2 - WALL_MARGIN)
    offset = rng.uniform(-1, 1) * min(freedom, (high - low) * 0.22)
    return center + offset


def room_outline(rect, gaps, rng: random.Random):
    """Clockwise convex ring with independent, opening-aware corner bevels."""
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0

    def from_low(side, low, length):
        return min((a for a, b in gaps[side]), default=low + length) - low - WALL_MARGIN

    def from_high(side, high, length):
        return high - max((b for a, b in gaps[side]), default=high - length) - WALL_MARGIN

    def corner(x_limit, y_limit):
        # Some square corners make the profile less uniform than an octagon.
        if rng.random() < 0.25:
            return 0.0, 0.0
        a = min(w * rng.uniform(0.08, 0.22), 128.0, x_limit)
        b = min(h * rng.uniform(0.08, 0.22), 128.0, y_limit)
        return (a, b) if min(a, b) >= 8 else (0.0, 0.0)

    swx, swy = corner(from_low('S', x0, w), from_low('W', y0, h))
    nwx, nwy = corner(from_low('N', x0, w), from_high('W', y1, h))
    nex, ney = corner(from_high('N', x1, w), from_high('E', y1, h))
    sex, sey = corner(from_high('S', x1, w), from_low('E', y0, h))
    points = [(x0, y0 + swy), (x0, y1 - nwy), (x0 + nwx, y1),
              (x1 - nex, y1), (x1, y1 - ney), (x1, y0 + sey),
              (x1 - sex, y0), (x0 + swx, y0)]
    return [p for i, p in enumerate(points) if p != points[i - 1]]

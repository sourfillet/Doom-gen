"""Vanilla Doom actor clearances used by placement and final validation.

Dimensions: id Software, linuxdoom-1.10/info.c, mobjinfo (DoomEd numbers).
https://github.com/id-Software/DOOM/blob/master/linuxdoom-1.10/info.c
Custom monsters require an explicit profile; guesses are not safe spawn rules.
"""
from __future__ import annotations

from dataclasses import dataclass

from wad_parser.clearance import MapClearance
from wad_parser.geometry import sector_geometry
from wad_parser.map_extractor import _sector_containing
from wad_parser.types import resolve_doomed, active_single_player


@dataclass(frozen=True)
class ActorProfile:
    radius: float
    height: float
    solid: bool


_MONSTERS = {
    3004: (20, 56), 9: (20, 56), 65: (20, 56), 3001: (20, 56),
    3002: (30, 56), 58: (30, 56), 3003: (24, 64), 69: (24, 64),
    3005: (31, 56), 3006: (16, 56), 64: (20, 56), 66: (20, 56),
    67: (48, 64), 68: (64, 64), 71: (31, 56), 7: (128, 100),
    16: (40, 110), 84: (20, 56),
}


def actor_profile(type_num):
    if type_num in _MONSTERS:
        return ActorProfile(*_MONSTERS[type_num], True)
    if type_num in {1, 2, 3, 4}:
        return ActorProfile(16, 56, True)
    if type_num == 11:
        return ActorProfile(16, 56, False)  # spawn marker, not a simultaneous player
    if type_num == 2035:
        return ActorProfile(10, 42, True)
    if resolve_doomed(type_num)[1] in {"weapon", "ammo", "item", "key"}:
        return ActorProfile(20, 16, False)
    return None


def spawn_problem(doc, type_num, x, y, *, ignore_id=None, margin=0, reachable=None, geometry=None):
    profile = actor_profile(type_num)
    if profile is None:
        return "unsupported actor dimensions"
    sid = _sector_containing(x, y, doc.sectors, sector_geometry(doc), doc.vertices)
    if sid is None:
        return "outside playable space"
    if reachable is not None:
        from wad_parser.traversal import component_for_point
        positions = {v.id: (v.x, v.y) for v in doc.vertices}
        if component_for_point(sid, x, y, sector_geometry(doc), positions) not in reachable:
            return 'unreachable component'
    geometry = geometry or MapClearance(doc.vertices, doc.sectors, doc.linedefs, doc.sidedefs)
    # Pickups are non-solid: their radius is a touch range, not a wall
    # collision box. Player spawn markers still need a full player box.
    pickup = resolve_doomed(type_num)[1] in {'weapon', 'ammo', 'item', 'key'}
    radius = margin if pickup else max(margin, profile.radius)
    problem = geometry.problem(sid, x, y, radius=radius, height=profile.height,
                               monster=type_num in _MONSTERS)
    if problem:
        return problem
    if profile.solid:
        for other in doc.things:
            if other.id == ignore_id or not active_single_player(other):
                continue
            other_profile = actor_profile(other.type)
            if other_profile is None or not other_profile.solid:
                continue
            radius = profile.radius + other_profile.radius
            if abs(other.x - x) < radius and abs(other.y - y) < radius:
                return f"overlaps solid thing {other.id}"
    return None


def placement_margin(type_num):
    """Keep non-solid pickup centres inside walls; actors use their radii."""
    return 1 if resolve_doomed(type_num)[1] in {'weapon', 'ammo', 'item', 'key'} else 0

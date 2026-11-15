"""Mandatory minimal progression for generated single-player Doom maps.

This guarantees exit presence; full door/lock/teleport traversal validation is
still required before claiming a level is completable (REPO_REVIEW.md #3/#7).
"""
from __future__ import annotations

import math

from wad_parser.geometry import sector_geometry
from wad_parser.types import MapDocument, resolve_linedef_special

EXIT_SPECIALS = frozenset({11, 51, 52, 124, 197, 198})


def ensure_exit(doc: MapDocument) -> str | None:
    """Keep existing exits, or add a use-activated exit on a reachable wall.

    Prefer a requested exit room, then distant walls with a player use approach.
    Preserve all existing nonzero specials. Fail if no suitable wall exists.
    """
    if doc.meta.format != "doom":
        raise ValueError("exit generation currently supports Doom-format maps only")
    if any(ld.special in EXIT_SPECIALS for ld in doc.linedefs):
        return None
    start = next((t for t in doc.things if t.type == 1), None)
    if start is None:
        raise ValueError("cannot add exit without a single-player start")
    vpos = {v.id: (v.x, v.y) for v in doc.vertices}
    from wad_parser.map_extractor import _sector_containing
    sector_id = _sector_containing(start.x, start.y, doc.sectors, sector_geometry(doc), doc.vertices)
    from wad_parser.traversal import distances_from_start
    distances = distances_from_start(doc)
    from wad_parser.traversal import component_distances, line_component
    reachable_components = component_distances(doc)
    containing = set(distances) if sector_id is not None else set()
    room_sectors = doc.extra.get('room_sectors', {})
    exit_sectors = {room_sectors.get(rid, room_sectors.get(str(rid)))
                    for rid, roles in doc.extra.get('room_roles', {}).items() if 'exit' in roles}
    from wad_parser.navigation import PlayerNavigation
    navigation = PlayerNavigation(doc)
    candidates = []
    for ld in doc.linedefs:
        if ld.special or ld.left_sidedef != -1 or not 0 <= ld.right_sidedef < len(doc.sidedefs):
            continue
        if doc.sidedefs[ld.right_sidedef].sector not in containing:
            continue
        sid = doc.sidedefs[ld.right_sidedef].sector
        if line_component(ld, sid, vpos, doc.sectors, doc.sidedefs, sector_geometry(doc)) not in reachable_components:
            continue
        if ld.v1 not in vpos or ld.v2 not in vpos:
            continue
        a, b = vpos[ld.v1], vpos[ld.v2]
        length = math.hypot(a[0] - b[0], a[1] - b[1])
        if length >= 32:
            if not navigation.reaches_line(ld):
                continue
            sid = doc.sidedefs[ld.right_sidedef].sector
            candidates.append(((sid in exit_sectors, distances.get(sid, 0), length), ld))
    if not candidates:
        raise ValueError("cannot add exit: no unused wall in reachable playable space")
    line = max(candidates, key=lambda pair: pair[0])[1]
    line.special = 11
    line.tag = 0
    line.special_name, line.trigger_type = resolve_linedef_special(line.special)
    return f"inserted normal exit on linedef {line.id}"

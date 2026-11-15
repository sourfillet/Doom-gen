"""Dedicated teleport pads for explicitly requested room connections."""
from __future__ import annotations

from wad_parser.geometry import sector_points, box_fits_sector, sector_clearance
from wad_parser.map_extractor import compute_derived
from wad_parser.types import Sector, Vertex, Sidedef, Linedef, LinedefFlags, Thing, ThingFlags, resolve_doomed, resolve_linedef_special


def add_teleport_pair(doc, sector_a, sector_b):
    from generation.actors import actor_profile, spawn_problem
    from wad_parser.clearance import MapClearance

    if sector_a == sector_b:
        raise ValueError("teleport endpoints must be different rooms")
    half, margin = 32, 8
    clearance = MapClearance(doc.vertices, doc.sectors, doc.linedefs, doc.sidedefs)

    def avoids_things(x, y):
        for thing in doc.things:
            # Destinations reserve a whole existing pad, whereas actors only
            # need their own radius plus a gap from the activation boundary.
            profile = actor_profile(thing.type)
            separation = (2 * half + 16 if thing.type == 14
                          else half + (profile.radius if profile else 16) + margin)
            if abs(x - thing.x) < separation and abs(y - thing.y) < separation:
                return False
        return True

    def center(sid):
        rings = sector_points(doc, sid)
        candidates = []
        for ring in rings:
            xs, ys = zip(*ring)
            x0, x1 = min(xs) + half + margin, max(xs) - half - margin
            y0, y1 = min(ys) + half + margin, max(ys) - half - margin
            if x0 > x1 or y0 > y1:
                continue
            # Include the usable extrema: sampling only interior fractions of
            # a small room can miss every pad that clears its central player.
            for i in range(9):
                for j in range(9):
                    x, y = round(x0 + (x1 - x0) * i / 8), round(y0 + (y1 - y0) * j / 8)
                    if (avoids_things(x, y) and box_fits_sector(doc, sid, x, y, half + margin)
                            and spawn_problem(doc, 1, x, y, geometry=clearance) is None):
                        candidates.append((sector_clearance(doc, sid, x, y), x, y))
        if not candidates:
            raise ValueError(f"room sector {sid} has no space for a dedicated teleport pad")
        _, x, y = max(candidates)
        return x, y
    centers = [center(sector_a), center(sector_b)]
    next_tag = max([0] + [s.tag for s in doc.sectors] + [line.tag for line in doc.linedefs]) + 1
    if next_tag + 1 > 32767:
        raise ValueError("no Doom sector tags available for teleport pads")
    pad_ids = [len(doc.sectors), len(doc.sectors) + 1]
    for index, (parent, (x, y)) in enumerate(zip((sector_a, sector_b), centers)):
        source = doc.sectors[parent]
        pad = pad_ids[index]
        doc.sectors.append(Sector(pad, source.floor_height, source.ceil_height,
                                  source.floor_flat, source.ceil_flat, source.light_level, 0, next_tag + index))
        corners = [(x - 32, y - 32), (x - 32, y + 32), (x + 32, y + 32), (x + 32, y - 32)]
        ids = []
        for px, py in corners:
            ids.append(len(doc.vertices))
            doc.vertices.append(Vertex(ids[-1], px, py))
        for a, b in zip(ids, ids[1:] + ids[:1]):
            right = len(doc.sidedefs)
            for sid in (parent, pad):
                doc.sidedefs.append(Sidedef(len(doc.sidedefs), 0, 0, "-", "-", "-", sid))
            name, trigger = resolve_linedef_special(97)
            doc.linedefs.append(Linedef(len(doc.linedefs), b, a, LinedefFlags(two_sided=True),
                                        97, next_tag + (1 - index), right, right + 1,
                                        special_name=name, trigger_type=trigger))
        name, category, unresolved = resolve_doomed(14)
        doc.things.append(Thing(len(doc.things), x, y, 90, 14, name, category,
                                ThingFlags(skill_1_2=True, skill_3=True, skill_4_5=True), unresolved=unresolved))
    doc.bounds, doc.derived = compute_derived(doc.vertices, doc.sectors, doc.linedefs, doc.sidedefs, doc.things)
    return pad_ids


def finalize_connections(doc, room_sectors):
    for connection in doc.extra.get("connection_realization", []):
        if connection["kind"] == "teleport":
            a, b = room_sectors.get(connection["a"]), room_sectors.get(connection["b"])
            if a is None or b is None:
                connection["realized"] = False
                connection["reason"] = "missing endpoint geometry"
                continue
            try:
                connection["pad_sectors"] = add_teleport_pair(doc, a, b)
                connection["realized"], connection["reason"] = True, ""
            except ValueError as exc:
                connection["realized"], connection["reason"] = False, str(exc)
    return doc


def require_connections(doc):
    failures = [edge for edge in doc.extra.get("connection_realization", []) if not edge["realized"]]
    if failures:
        raise ValueError("unrealized room connections: " + "; ".join(
            f"{edge['a']}->{edge['b']} ({edge['kind']}): {edge['reason']}" for edge in failures))

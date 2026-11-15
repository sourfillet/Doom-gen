"""Spatial geometry assembler for topology-conditioned spatial targets.

Converts a doom_spatial_geometry_v1 target (room placements, interior sector
components, boundary openings, and elevation profiles) into a valid MapDocument.
Shared boundaries between adjacent components are split at all collinear vertex
junctions and paired into two-sided linedefs without T-junctions or gaps.
"""
from __future__ import annotations

import math
from typing import Any

from wad_parser.map_extractor import compute_derived
from wad_parser.types import (
    Linedef, LinedefFlags, MapDocument, MapMeta,
    Sector, Sidedef, Thing, ThingFlags, Vertex, resolve_doomed,
)

DEFAULT_WALL = "STARTAN2"
DEFAULT_FLOOR = "FLOOR4_8"
DEFAULT_CEIL = "CEIL3_5"


def _split_edge(
    p1: tuple[int, int],
    p2: tuple[int, int],
    candidates: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Split segment (p1, p2) at any collinear points in candidates strictly between them."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    dist_sq = dx * dx + dy * dy
    if dist_sq == 0:
        return []

    min_x, max_x = min(x1, x2), max(x1, x2)
    min_y, max_y = min(y1, y2), max(y1, y2)

    inter_pts: list[tuple[float, tuple[int, int]]] = []
    for p in candidates:
        if p == p1 or p == p2:
            continue
        px, py = p
        if not (min_x <= px <= max_x and min_y <= py <= max_y):
            continue
        # Collinear cross product == 0
        if (px - x1) * dy - (py - y1) * dx == 0:
            t = ((px - x1) * dx + (py - y1) * dy) / dist_sq
            if 0 < t < 1:
                inter_pts.append((t, p))

    if not inter_pts:
        return [(p1, p2)]

    inter_pts.sort(key=lambda item: item[0])
    chain = [p1] + [p for _, p in inter_pts] + [p2]
    return [(chain[i], chain[i + 1]) for i in range(len(chain) - 1)]


def assemble_spatial_geometry(target: dict[str, Any]) -> MapDocument:
    """Assemble a doom_spatial_geometry_v1 target into a MapDocument.

    Parameters:
        target: Dictionary conforming to doom_spatial_geometry_v1 schema.

    Returns:
        A fully formed, valid MapDocument with computed derived geometry.
    """
    if target.get("schema") != "doom_spatial_geometry_v1":
        raise ValueError(
            f"unsupported spatial geometry schema: {target.get('schema')!r}, "
            f"expected 'doom_spatial_geometry_v1'"
        )

    frame = target["frame"]
    origin = frame["origin"]
    scale = float(frame["scale"])
    floor_origin = float(frame["floor_origin"])

    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("invalid spatial scale in coordinate frame")

    # Map room placements by id for relative ring reconstruction
    placements_by_id = {
        p["room_id"]: p for p in target.get("geometry", {}).get("placements", [])
    }

    sectors: list[Sector] = []
    sector_id_map: dict[tuple[int, ...], int] = {}
    rings_with_sector: list[tuple[int, list[tuple[int, int]]]] = []
    all_vertices: set[tuple[int, int]] = set()

    interiors = target.get("geometry", {}).get("interiors", [])
    for room in interiors:
        room_id = room["room_id"]
        placement = placements_by_id.get(room_id, {})
        norm_center = placement.get("center", [0.0, 0.0])

        for comp in room.get("components", []):
            cid_raw = comp.get("component_id")
            if cid_raw is not None:
                # Group by original sector ID if present
                cid_key = (cid_raw[0],)
            else:
                cid_key = (len(sectors),)

            if cid_key not in sector_id_map:
                new_sid = len(sectors)
                sector_id_map[cid_key] = new_sid
                sec = Sector(
                    id=new_sid,
                    floor_height=int(round(floor_origin + comp.get("floor", 0.0))),
                    ceil_height=int(round(floor_origin + comp.get("ceiling", 128.0))),
                    floor_flat=comp.get("floor_texture", DEFAULT_FLOOR),
                    ceil_flat=comp.get("ceiling_texture", DEFAULT_CEIL),
                    light_level=int(comp.get("light_level", 160)),
                    special=0,
                    tag=0,
                )
                sectors.append(sec)
            sid = sector_id_map[cid_key]

            # Reconstruct rings in integer world coordinates
            # Prefer absolute exterior_ring, fall back to relative_exterior_ring + norm_center
            if "exterior_ring" in comp and comp["exterior_ring"]:
                ext_raw = comp["exterior_ring"]
                ext_pts = [
                    (
                        int(round(origin[0] + pt[0] * scale)),
                        int(round(origin[1] + pt[1] * scale)),
                    )
                    for pt in ext_raw
                ]
            elif "relative_exterior_ring" in comp and comp["relative_exterior_ring"]:
                ext_raw = comp["relative_exterior_ring"]
                ext_pts = [
                    (
                        int(round(origin[0] + (norm_center[0] + pt[0]) * scale)),
                        int(round(origin[1] + (norm_center[1] + pt[1]) * scale)),
                    )
                    for pt in ext_raw
                ]
            else:
                ext_pts = []

            if len(ext_pts) >= 3:
                rings_with_sector.append((sid, ext_pts))
                all_vertices.update(ext_pts)

            # Interior rings (holes)
            int_rings_raw = comp.get("interior_rings", [])
            if not int_rings_raw and "relative_interior_rings" in comp:
                int_rings_raw = [
                    [
                        [norm_center[0] + pt[0], norm_center[1] + pt[1]]
                        for pt in hole
                    ]
                    for hole in comp["relative_interior_rings"]
                ]

            for hole_raw in int_rings_raw:
                hole_pts = [
                    (
                        int(round(origin[0] + pt[0] * scale)),
                        int(round(origin[1] + pt[1] * scale)),
                    )
                    for pt in hole_raw
                ]
                if len(hole_pts) >= 3:
                    rings_with_sector.append((sid, hole_pts))
                    all_vertices.update(hole_pts)

    if not sectors or not rings_with_sector:
        raise ValueError("spatial geometry target contains no interior sectors")

    # Split all ring segments at collinear junctions to prevent T-junctions
    atomic_segments: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for sid, ring in rings_with_sector:
        n = len(ring)
        for i in range(n):
            p1, p2 = ring[i], ring[(i + 1) % n]
            for sub_p1, sub_p2 in _split_edge(p1, p2, all_vertices):
                atomic_segments.append((sid, sub_p1, sub_p2))

    # Assemble vertices, sidedefs, linedefs
    verts: list[Vertex] = []
    vert_map: dict[tuple[int, int], int] = {}

    def get_vid(p: tuple[int, int]) -> int:
        if p not in vert_map:
            vert_map[p] = len(verts)
            verts.append(Vertex(len(verts), p[0], p[1]))
        return vert_map[p]

    linedefs: list[Linedef] = []
    sidedefs: list[Sidedef] = []
    edge_to_line: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}

    for sid, p1, p2 in atomic_segments:
        rev = (p2, p1)
        if rev in edge_to_line:
            # Shared boundary: two-sided line
            line_idx = edge_to_line[rev]
            line = linedefs[line_idx]
            sec_l = sectors[sid]
            sec_r = sectors[sidedefs[line.right_sidedef].sector]

            sd_id = len(sidedefs)
            # Textures for left side
            lower_l = DEFAULT_WALL if sec_r.floor_height > sec_l.floor_height else "-"
            upper_l = DEFAULT_WALL if sec_r.ceil_height < sec_l.ceil_height else "-"
            sidedefs.append(
                Sidedef(
                    id=sd_id,
                    x_offset=0,
                    y_offset=0,
                    upper_tex=upper_l,
                    lower_tex=lower_l,
                    mid_tex="-",
                    sector=sid,
                )
            )
            line.left_sidedef = sd_id
            line.flags.two_sided = True
            line.flags.impassable = False

            # Update right side textures for step / drop
            r_sd = sidedefs[line.right_sidedef]
            r_sd.mid_tex = "-"
            r_sd.lower_tex = (
                DEFAULT_WALL if sec_l.floor_height > sec_r.floor_height else "-"
            )
            r_sd.upper_tex = (
                DEFAULT_WALL if sec_l.ceil_height < sec_r.ceil_height else "-"
            )
        else:
            # Exterior boundary: one-sided line
            sd_id = len(sidedefs)
            sidedefs.append(
                Sidedef(
                    id=sd_id,
                    x_offset=0,
                    y_offset=0,
                    upper_tex="-",
                    lower_tex="-",
                    mid_tex=DEFAULT_WALL,
                    sector=sid,
                )
            )
            line_idx = len(linedefs)
            line = Linedef(
                id=line_idx,
                v1=get_vid(p1),
                v2=get_vid(p2),
                flags=LinedefFlags(impassable=True),
                special=0,
                tag=0,
                right_sidedef=sd_id,
                left_sidedef=-1,
            )
            linedefs.append(line)
            edge_to_line[(p1, p2)] = line_idx

    # Things: place player-one start
    things: list[Thing] = []
    if "things" in target and target["things"]:
        for t in target["things"]:
            name, cat, _ = resolve_doomed(t["type"])
            things.append(
                Thing(
                    id=len(things),
                    x=int(round(origin[0] + t["x"] * scale)),
                    y=int(round(origin[1] + t["y"] * scale)),
                    angle=int(t.get("angle", 0)),
                    type=int(t["type"]),
                    type_name=name,
                    category=cat,
                    flags=ThingFlags(
                        skill_1_2=True, skill_3=True, skill_4_5=True
                    ),
                )
            )
    else:
        name, cat, _ = resolve_doomed(1)
        things.append(
            Thing(
                id=0,
                x=int(round(origin[0])),
                y=int(round(origin[1])),
                angle=0,
                type=1,
                type_name=name,
                category=cat,
                flags=ThingFlags(skill_1_2=True, skill_3=True, skill_4_5=True),
            )
        )

    # Compute derived spatial geometry
    bounds, derived = compute_derived(verts, sectors, linedefs, sidedefs, things)
    meta = MapMeta("MAP01", "spatial_geometry_assembler", "assembled", "doom")
    return MapDocument(
        schema_version="1.0",
        meta=meta,
        bounds=bounds,
        vertices=verts,
        sectors=sectors,
        linedefs=linedefs,
        sidedefs=sidedefs,
        things=things,
        derived=derived,
    )

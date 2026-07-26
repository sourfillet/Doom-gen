"""Assembles a MapDocument from parsed lumps (binary or UDMF) and computes
the derived layer described in plan/data-model.md: sector polygons,
adjacency graph, trigger map, reachability from player starts, a texture
alignment heuristic, and symmetry detection.

The derived layer is always a pure function of the raw layer - nothing here
is authoritative, all of it is meant to be recomputable after a bugfix
without re-reading the source WAD.
"""
from __future__ import annotations

import math
import struct
from collections import defaultdict, deque

from wad_parser import lump_parser, udmf_parser
from wad_parser.types import (
    BoundingBox, DerivedData, MapDocument, MapMeta, SCHEMA_VERSION,
)
from wad_parser.wad_reader import MapLumpGroup, WadFile


def _bounds(vertices) -> BoundingBox:
    if not vertices:
        return BoundingBox(0, 0, 0, 0)
    xs = [v.x for v in vertices]
    ys = [v.y for v in vertices]
    return BoundingBox(min(xs), max(xs), min(ys), max(ys))


def _sector_polygons(sectors, linedefs, sidedefs) -> list[list[int]]:
    """One boundary ring per sector (the largest reconstructed loop, in the
    rare case a sector has disjoint islands - a simplification documented
    here rather than silently made).
    """
    sector_edges: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for ld in linedefs:
        if ld.right_sidedef != -1 and ld.right_sidedef < len(sidedefs):
            sector_edges[sidedefs[ld.right_sidedef].sector].append((ld.v1, ld.v2))
        if ld.left_sidedef != -1 and ld.left_sidedef < len(sidedefs):
            sector_edges[sidedefs[ld.left_sidedef].sector].append((ld.v2, ld.v1))

    polygons: list[list[int]] = [[] for _ in sectors]
    for sector_id, edges in sector_edges.items():
        if sector_id >= len(polygons):
            continue
        edge_map: dict[int, list[int]] = defaultdict(list)
        for a, b in edges:
            edge_map[a].append(b)
        visited: set[tuple[int, int]] = set()
        loops: list[list[int]] = []
        for a, b in edges:
            if (a, b) in visited:
                continue
            loop = [a]
            visited.add((a, b))
            cur, nxt = a, b
            steps = 0
            while nxt != a and steps < len(edges) + 1:
                loop.append(nxt)
                found = None
                for c in edge_map.get(nxt, []):
                    if (nxt, c) not in visited:
                        found = c
                        break
                if found is None:
                    break
                visited.add((nxt, found))
                cur, nxt = nxt, found
                steps += 1
            loops.append(loop)
        polygons[sector_id] = max(loops, key=len) if loops else []
    return polygons


def _sector_adjacency(sectors, linedefs, sidedefs) -> dict:
    edges = []
    for ld in linedefs:
        if ld.right_sidedef == -1 or ld.left_sidedef == -1:
            continue
        if ld.right_sidedef >= len(sidedefs) or ld.left_sidedef >= len(sidedefs):
            continue
        a = sidedefs[ld.right_sidedef].sector
        b = sidedefs[ld.left_sidedef].sector
        if a != b:
            edges.append({"a": a, "b": b, "linedef": ld.id, "kind": "walk"})
    return {"nodes": [s.id for s in sectors], "edges": edges}


# Linedef specials that teleport the player: vanilla W1/WR (any-thing and
# monster-only) plus the Boom W1/WR silent-teleport variants. Without these,
# reachability analysis treats every teleporter-linked area as disconnected
# geometry, which is wrong - teleporters are normal, common map design, not
# a defect - so they get their own edge kind rather than being folded into
# (or omitted from) the walk-adjacency graph.
_TELEPORT_SPECIALS = {39, 97, 125, 126, 207, 208, 209, 210, 243, 244}


def _teleport_edges(sectors, linedefs, sidedefs, things, polygons, vertices) -> list[dict]:
    if not any(ld.special in _TELEPORT_SPECIALS for ld in linedefs):
        return []
    dest_by_sector: list[tuple[int, int]] = []  # (sector_id, sector_tag)
    tag_by_sector = {s.id: s.tag for s in sectors}
    for t in things:
        if t.category != "teleport-dest":
            continue
        sec = _sector_containing(t.x, t.y, sectors, polygons, vertices)
        if sec is not None:
            dest_by_sector.append((sec, tag_by_sector.get(sec, 0)))
    if not dest_by_sector:
        return []

    edges = []
    for ld in linedefs:
        if ld.special not in _TELEPORT_SPECIALS:
            continue
        if ld.right_sidedef == -1 or ld.right_sidedef >= len(sidedefs):
            continue
        src_sector = sidedefs[ld.right_sidedef].sector
        # Tagged teleporters (Boom/Hexen) target only destinations in a
        # same-tagged sector; untagged (vanilla) teleporters aren't
        # disambiguated by the game either, so connect to every destination.
        candidates = [s for s, tag in dest_by_sector if ld.tag and tag == ld.tag] if ld.tag else \
            [s for s, _ in dest_by_sector]
        for dest_sector in candidates:
            if dest_sector != src_sector:
                edges.append({"a": src_sector, "b": dest_sector, "linedef": ld.id, "kind": "teleport"})
    return edges


def _trigger_map(sectors, linedefs) -> dict[str, list[int]]:
    tag_to_sectors: dict[int, list[int]] = defaultdict(list)
    for s in sectors:
        if s.tag:
            tag_to_sectors[s.tag].append(s.id)
    triggers: dict[str, list[int]] = defaultdict(list)
    for ld in linedefs:
        if ld.special and ld.tag and ld.tag in tag_to_sectors:
            for sector_id in tag_to_sectors[ld.tag]:
                triggers[str(sector_id)].append(ld.id)
    return dict(triggers)


def _point_in_polygon(x, y, poly_pts) -> bool:
    inside = False
    n = len(poly_pts)
    for i in range(n):
        x1, y1 = poly_pts[i]
        x2, y2 = poly_pts[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
    return inside


def _sector_containing(x, y, sectors, polygons, vertices) -> int | None:
    vpos = {v.id: (v.x, v.y) for v in vertices}
    best_fallback, best_dist = None, math.inf
    for s in sectors:
        ring = polygons[s.id] if s.id < len(polygons) else []
        pts = [vpos[v] for v in ring if v in vpos]
        if len(pts) >= 3 and _point_in_polygon(x, y, pts):
            return s.id
        if pts:
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            d = (cx - x) ** 2 + (cy - y) ** 2
            if d < best_dist:
                best_dist, best_fallback = d, s.id
    return best_fallback


def _reachable_from_start(sectors, things, polygons, vertices, adjacency) -> list[int]:
    starts = [t for t in things if t.category == "player-start"]
    if not starts:
        return []
    start_sectors = set()
    for t in starts:
        sec = _sector_containing(t.x, t.y, sectors, polygons, vertices)
        if sec is not None:
            start_sectors.add(sec)

    neighbors: dict[int, set[int]] = defaultdict(set)
    for e in adjacency["edges"]:
        neighbors[e["a"]].add(e["b"])
        neighbors[e["b"]].add(e["a"])

    seen = set(start_sectors)
    q = deque(start_sectors)
    while q:
        cur = q.popleft()
        for nb in neighbors.get(cur, ()):
            if nb not in seen:
                seen.add(nb)
                q.append(nb)
    return sorted(seen)


def _texture_alignment_score(sectors, linedefs, sidedefs, polygons) -> float | None:
    """Fraction of consecutive wall segments (walking each sector's
    boundary loop) whose facing sidedef shares the same x/y offset as the
    previous segment - a rough proxy for the "offsets kept consistent
    around a sector" habit common in hand-built maps.
    """
    sidedef_for_edge: dict[tuple[int, int], int] = {}
    for ld in linedefs:
        if ld.right_sidedef != -1:
            sidedef_for_edge[(ld.v1, ld.v2)] = ld.right_sidedef
        if ld.left_sidedef != -1:
            sidedef_for_edge[(ld.v2, ld.v1)] = ld.left_sidedef

    matches, total = 0, 0
    for ring in polygons:
        if len(ring) < 2:
            continue
        prev_off = None
        for i in range(len(ring)):
            a, b = ring[i], ring[(i + 1) % len(ring)]
            sd_id = sidedef_for_edge.get((a, b))
            if sd_id is None or sd_id >= len(sidedefs):
                continue
            off = (sidedefs[sd_id].x_offset, sidedefs[sd_id].y_offset)
            if prev_off is not None:
                total += 1
                if off == prev_off:
                    matches += 1
            prev_off = off
    return (matches / total) if total else None


def _normalized_space(vertices, bounds: BoundingBox) -> dict | None:
    """Re-centered, unit-scaled vertex coordinates, kept alongside (not
    instead of) the raw vertex list - see plan/data-model.md.
    """
    if not vertices:
        return None
    center_x = (bounds.min_x + bounds.max_x) / 2
    center_y = (bounds.min_y + bounds.max_y) / 2
    half_w = (bounds.max_x - bounds.min_x) / 2
    half_h = (bounds.max_y - bounds.min_y) / 2
    scale = 1.0 / max(half_w, half_h, 1e-6)
    return {
        "center_x": center_x,
        "center_y": center_y,
        "scale": scale,
        "vertices": [[(v.x - center_x) * scale, (v.y - center_y) * scale] for v in vertices],
    }


def _bsp_sanity_check(group: MapLumpGroup, linedefs, sidedefs, sectors, polygons) -> dict | None:
    """Opportunistic cross-check against the map's own compiled BSP, per
    plan/data-model.md: SEGS/SSECTORS/NODES/REJECT aren't stored as training
    data (they're artifacts of a node-builder's internal choices, not
    authored map content), but reading them lets us flag a sector-polygon
    reconstruction bug rather than silently shipping bad derived data.
    Only meaningful for classic binary nodes; UDMF maps typically ship
    ZNODES instead, so this returns None for those.
    """
    segs_data = group.lumps.get("SEGS")
    ssectors_data = group.lumps.get("SSECTORS")
    reject_data = group.lumps.get("REJECT")
    if not segs_data or not ssectors_data:
        return None

    n_segs = len(segs_data) // 12
    bsp_sectors: set[int] = set()
    for i in range(n_segs):
        off = i * 12
        _v1, _v2, _angle, linedef_idx, direction, _offset = struct.unpack_from("<hhhHhh", segs_data, off)
        if linedef_idx == 0xFFFF or linedef_idx >= len(linedefs):
            continue  # miniseg: BSP partition artifact, not a real wall
        ld = linedefs[linedef_idx]
        sd_id = ld.left_sidedef if direction == 1 else ld.right_sidedef
        if sd_id is not None and sd_id != -1 and sd_id < len(sidedefs):
            bsp_sectors.add(sidedefs[sd_id].sector)

    n_ssectors = len(ssectors_data) // 4
    covered, prev_end, contiguous = 0, 0, True
    for i in range(n_ssectors):
        off = i * 4
        numsegs, firstseg = struct.unpack_from("<HH", ssectors_data, off)
        if firstseg != prev_end:
            contiguous = False
        prev_end = firstseg + numsegs
        covered += numsegs

    sectors_with_geometry = {i for i, ring in enumerate(polygons) if ring}
    result = {
        "subsector_count": n_ssectors,
        "segs_total": n_segs,
        "segs_covered_by_subsectors": covered,
        "subsectors_contiguous": contiguous,
        "sectors_seen_in_bsp": len(bsp_sectors),
        "sectors_missing_from_reconstruction": sorted(bsp_sectors - sectors_with_geometry),
    }
    if reject_data is not None:
        expected = (len(sectors) * len(sectors) + 7) // 8
        result["reject_size_bytes"] = len(reject_data)
        result["reject_size_matches_expected"] = len(reject_data) == expected
    return result


def _detect_symmetry(vertices, bounds: BoundingBox) -> dict | None:
    if not vertices:
        return None
    coords = {(round(v.x, 1), round(v.y, 1)) for v in vertices}
    mid_x = (bounds.min_x + bounds.max_x) / 2
    mid_y = (bounds.min_y + bounds.max_y) / 2

    def score(mirror):
        hits = sum(1 for v in vertices if mirror(v) in coords)
        return hits / len(vertices)

    x_score = score(lambda v: (round(2 * mid_x - v.x, 1), round(v.y, 1)))
    y_score = score(lambda v: (round(v.x, 1), round(2 * mid_y - v.y, 1)))
    best_axis = "x" if x_score >= y_score else "y"
    return {"x_mirror_score": x_score, "y_mirror_score": y_score, "best_axis": best_axis}


def _parse_binary_group(group: MapLumpGroup):
    is_hexen = group.format == "hexen"
    vertices = lump_parser.parse_vertexes(group.lumps.get("VERTEXES", b""))
    sectors = lump_parser.parse_sectors(group.lumps.get("SECTORS", b""))
    sidedefs = lump_parser.parse_sidedefs(group.lumps.get("SIDEDEFS", b""))
    if is_hexen:
        linedefs = lump_parser.parse_linedefs_hexen(group.lumps.get("LINEDEFS", b""))
        things = lump_parser.parse_things_hexen(group.lumps.get("THINGS", b""))
    else:
        linedefs = lump_parser.parse_linedefs_doom(group.lumps.get("LINEDEFS", b""))
        things = lump_parser.parse_things_doom(group.lumps.get("THINGS", b""))
    return vertices, sectors, linedefs, sidedefs, things


def _parse_udmf_group(group: MapLumpGroup):
    doc = udmf_parser.parse_textmap(group.lumps.get("TEXTMAP", b""))
    vertices = udmf_parser.build_vertices(doc)
    sectors = udmf_parser.build_sectors(doc)
    linedefs = udmf_parser.build_linedefs(doc)
    sidedefs = udmf_parser.build_sidedefs(doc)
    things = udmf_parser.build_things(doc)
    return vertices, sectors, linedefs, sidedefs, things


def extract_map(wad: WadFile, group: MapLumpGroup, mapinfo: dict | None = None) -> MapDocument:
    if group.format == "udmf":
        vertices, sectors, linedefs, sidedefs, things = _parse_udmf_group(group)
    else:
        vertices, sectors, linedefs, sidedefs, things = _parse_binary_group(group)

    polygons = _sector_polygons(sectors, linedefs, sidedefs)
    adjacency = _sector_adjacency(sectors, linedefs, sidedefs)
    adjacency["edges"] += _teleport_edges(sectors, linedefs, sidedefs, things, polygons, vertices)
    triggers = _trigger_map(sectors, linedefs)
    reachable = _reachable_from_start(sectors, things, polygons, vertices, adjacency)
    alignment = _texture_alignment_score(sectors, linedefs, sidedefs, polygons)
    bounds = _bounds(vertices)
    symmetry = _detect_symmetry(vertices, bounds)
    normalized = _normalized_space(vertices, bounds)
    bsp_check = _bsp_sanity_check(group, linedefs, sidedefs, sectors, polygons)

    entry = (mapinfo or {}).get(group.name)

    meta = MapMeta(
        map_name=group.name,
        source_wad=wad.path,
        wad_hash=wad.hash,
        format=group.format,
        is_iwad=(wad.identification == "IWAD"),
        mapinfo=entry,
    )
    derived = DerivedData(
        sector_graph=adjacency,
        sector_polygons=polygons,
        trigger_map=triggers,
        reachable_from_start=reachable,
        texture_alignment_score=alignment,
        symmetry=symmetry,
        normalized_space=normalized,
        bsp_sanity_check=bsp_check,
    )
    return MapDocument(
        schema_version=SCHEMA_VERSION,
        meta=meta,
        bounds=bounds,
        vertices=vertices,
        sectors=sectors,
        linedefs=linedefs,
        sidedefs=sidedefs,
        things=things,
        derived=derived,
    )


def extract_all_maps(wad: WadFile) -> list[MapDocument]:
    from wad_parser.mapinfo_parser import parse_mapinfo

    mapinfo = {}
    for lump_name in ("MAPINFO", "ZMAPINFO", "EMAPINFO"):
        data = wad.get_lump(lump_name)
        if data:
            mapinfo.update(parse_mapinfo(data))

    return [extract_map(wad, group, mapinfo) for group in wad.find_map_groups()]

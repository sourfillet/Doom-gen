"""Assembles a MapDocument from parsed lumps (binary or UDMF) and computes
the derived layer: sector polygons,
adjacency graph, trigger map, reachability from player starts, a texture
alignment heuristic, and symmetry detection.

The derived layer is always a pure function of the raw layer - nothing here
is authoritative, all of it is meant to be recomputable after a bugfix
without re-reading the source WAD.
"""
from __future__ import annotations

import struct
from collections import defaultdict

from wad_parser import lump_parser, udmf_parser
from wad_parser.types import (
    BoundingBox, DerivedData, HEXEN_UDMF_NAMESPACES, MapDocument, MapMeta,
    SCHEMA_VERSION,
)
from wad_parser.wad_reader import MapLumpGroup, WadFile


def _bounds(vertices) -> BoundingBox:
    if not vertices:
        return BoundingBox(0, 0, 0, 0)
    xs = [v.x for v in vertices]
    ys = [v.y for v in vertices]
    return BoundingBox(min(xs), max(xs), min(ys), max(ys))


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


# Doom-namespace linedef specials that teleport to a teleport-dest THING in
# the tagged sector: vanilla W1/WR (39/97) and monster-only (125/126), Boom
# switch-activated (174/195), Boom silent (207-210), Boom monster-only
# silent (268/269). Without these, reachability analysis treats every
# teleporter-linked area as disconnected geometry, which is wrong -
# teleporters are normal, common map design, not a defect - so they get
# their own edge kind rather than being folded into (or omitted from) the
# walk-adjacency graph.
_THING_TELEPORT_SPECIALS = {39, 97, 125, 126, 174, 195, 207, 208, 209, 210, 268, 269}
# Boom line-to-line teleports: the destination is the OTHER linedef with the
# same tag, not a thing (243/244 plus reversed/monster-only variants).
_LINE_TELEPORT_SPECIALS = {243, 244, 262, 263, 264, 265, 266, 267}
# Hexen action specials that teleport to a TeleportDest/MapSpot thing whose
# TID is the special's first argument.
_HEXEN_TELEPORT_SPECIALS = {70, 71}


def _front_sector(ld, sidedefs) -> int | None:
    if not 0 <= ld.right_sidedef < len(sidedefs):
        return None
    return sidedefs[ld.right_sidedef].sector


def _teleport_edges(sectors, linedefs, sidedefs, things, polygons, vertices,
                    hexen_style: bool) -> list[dict]:
    edges: list[dict] = []
    thing_sector_cache: dict[int, int | None] = {}

    def sector_of_thing(t):
        if t.id not in thing_sector_cache:
            thing_sector_cache[t.id] = _sector_containing(t.x, t.y, sectors, polygons, vertices)
        return thing_sector_cache[t.id]

    if hexen_style:
        for ld in linedefs:
            if ld.special not in _HEXEN_TELEPORT_SPECIALS or not ld.args:
                continue
            src_sector = _front_sector(ld, sidedefs)
            if src_sector is None:
                continue
            dest_tid = ld.args[0]
            for t in things:
                # A nonzero arg0 addresses a destination by TID (any thing
                # type - ZDoom MapSpots resolve as "unknown" here); arg0 == 0
                # means "any TeleportDest".
                if dest_tid:
                    if t.tid != dest_tid:
                        continue
                elif t.category != "teleport-dest":
                    continue
                dest = sector_of_thing(t)
                if dest is not None:
                    edges.append({"a": src_sector, "b": dest, "linedef": ld.id, "kind": "teleport",
                                  "destination_point": [t.x, t.y]})
        return edges

    # --- Doom/Boom namespace ---
    if any(ld.special in _THING_TELEPORT_SPECIALS for ld in linedefs):
        dest_by_sector = []  # (sector_id, sector_tag, thing)
        tag_by_sector = {s.id: s.tag for s in sectors}
        for t in things:
            if t.category != "teleport-dest":
                continue
            from wad_parser.types import active_single_player
            if not active_single_player(t):
                continue
            sec = sector_of_thing(t)
            if sec is not None:
                dest_by_sector.append((sec, tag_by_sector.get(sec, 0), t))
        for ld in linedefs:
            if ld.special not in _THING_TELEPORT_SPECIALS:
                continue
            src_sector = _front_sector(ld, sidedefs)
            if src_sector is None:
                continue
            # Vanilla searches sectors in index order, then the first
            # teleport destination in that sector, including matching tag 0.
            candidates = sorted((record for record in dest_by_sector if record[1] == ld.tag),
                                key=lambda record: (record[0], record[2].id))
            if ld.special in {39, 97, 125, 126}:
                candidates = candidates[:1]
            for dest_sector, _, destination in candidates:
                edges.append({"a": src_sector, "b": dest_sector, "linedef": ld.id,
                              "kind": "teleport", "destination_point": [destination.x, destination.y]})

    line_teleports = [ld for ld in linedefs if ld.special in _LINE_TELEPORT_SPECIALS and ld.tag]
    if line_teleports:
        lines_by_tag: dict[int, list] = defaultdict(list)
        for ld in linedefs:
            if ld.tag:
                lines_by_tag[ld.tag].append(ld)
        for ld in line_teleports:
            src_sector = _front_sector(ld, sidedefs)
            if src_sector is None:
                continue
            for other in lines_by_tag[ld.tag]:
                if other.id == ld.id:
                    continue
                dest_sector = _front_sector(other, sidedefs)
                if dest_sector is not None and dest_sector != src_sector:
                    edges.append({"a": src_sector, "b": dest_sector, "linedef": ld.id, "kind": "teleport", "destination_line": other.id})
    return edges


# Hexen action-special families whose first argument is a sector tag; used
# to feed the trigger map for Hexen-format / Hexen-namespace-UDMF maps,
# where linedefs have args instead of a tag field. (Specials outside these
# families point their arg0 at lines, things, scripts, or nothing.)
_HEXEN_SECTOR_TAG_FAMILIES = (
    "Door_", "Floor_", "Ceiling_", "Plat_", "Stairs_", "Pillar_",
    "FloorAndCeiling_", "Elevator_", "Light_", "Generic", "Sector_",
)


def _effective_sector_tag(ld, hexen_style: bool) -> int:
    if not hexen_style:
        return ld.tag
    if ld.args and ld.special_name and ld.special_name.startswith(_HEXEN_SECTOR_TAG_FAMILIES):
        return ld.args[0]
    return 0


def _trigger_map(sectors, linedefs, hexen_style: bool = False) -> dict[str, list[int]]:
    tag_to_sectors: dict[int, list[int]] = defaultdict(list)
    for s in sectors:
        if s.tag:
            tag_to_sectors[s.tag].append(s.id)
    triggers: dict[str, list[int]] = defaultdict(list)
    for ld in linedefs:
        tag = _effective_sector_tag(ld, hexen_style)
        if ld.special and tag and tag in tag_to_sectors:
            for sector_id in tag_to_sectors[tag]:
                triggers[str(sector_id)].append(ld.id)
    return dict(triggers)


def _sector_containing(x, y, sectors, polygons, vertices, locator=None) -> int | None:
    """Deterministic ownership of filled regions and their covered boundaries.

    A center on an open sector boundary still needs an owner. This geometric
    lookup does not certify actor clearance; points in void interiors stay None.
    """
    if locator is not None:
        return locator.owner(x, y)
    from wad_parser.geometry import point_in_rings, rings_area_perimeter, resolve_rings
    vpos = {v.id: (v.x, v.y) for v in vertices}
    candidates = []
    for sector in sectors:
        parts = polygons[sector.id] if 0 <= sector.id < len(polygons) else []
        points = resolve_rings(parts, vpos)
        if point_in_rings(x, y, points, boundary=True):
            candidates.append((rings_area_perimeter(points)[0], sector.id))
    return min(candidates)[1] if candidates else None


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
    for ring in (ring for parts in polygons for ring in parts):
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
    instead of) the raw vertex list.
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
    """Opportunistic cross-check against the map's own compiled BSP.
    SEGS/SSECTORS/NODES/REJECT aren't stored as training
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
    return vertices, sectors, linedefs, sidedefs, things, doc


def _recoverable_boundary_issue(issue):
    return ((issue.startswith('sector ') and 'open or degenerate' in issue)
            or (issue.startswith('linedef ') and issue.endswith(': zero-length boundary')))


def compute_derived(vertices, sectors, linedefs, sidedefs, things,
                    hexen_style: bool = False,
                    group: MapLumpGroup | None = None, *, region_geometry=None,
                    region_info=None) -> tuple[BoundingBox, DerivedData]:
    """The full derived-layer pass, callable on any raw element set - used
    by extract_map and by data/augmentation.py to recompute derived data
    after a geometric transform (derived data is never transformed in
    place; it is always a pure function of the raw layer). The BSP
    sanity-check needs the original lump group and is skipped without one.
    """
    from wad_parser.geometry import DERIVED_VERSION, reconstruct_rings, signed_area
    rings, geometry_issues = reconstruct_rings(vertices, sectors, linedefs, sidedefs)
    raw_geometry_issues = list(geometry_issues)
    sector_regions, bsp_geometry = [], None
    if region_geometry is not None and all(_recoverable_boundary_issue(issue) for issue in geometry_issues):
        from wad_parser.bsp_geometry import component_polygons, _union
        if len(region_geometry) != len(sectors):
            raise ValueError('compiled region count differs from sectors')
        polygons = [polygon for parts in region_geometry for _, polygon in component_polygons(parts)]
        area = sum(polygon.area for polygon in polygons)
        if area - _union(polygons).area > max(1 / 65536, area * 1e-9):
            raise ValueError('preserved compiled regions overlap')
        sector_regions = region_geometry
        bsp_geometry = dict(region_info or {'method': 'preserved-compiled-regions-v1'})
        geometry_issues = []
    if group is not None and all(_recoverable_boundary_issue(issue) for issue in geometry_issues):
        from wad_parser.bsp_geometry import reconstruct_bsp
        try:
            recovered = reconstruct_bsp(group, vertices, linedefs, sidedefs, len(sectors))
            if recovered is not None:
                sector_regions, bsp_geometry = recovered
                geometry_issues = []
        except ValueError as exc:
            bsp_geometry = {'failure': str(exc)}
    effective = sector_regions or rings
    positions = {v.id: (v.x, v.y) for v in vertices}
    polygons = [max(parts, key=lambda ring: abs(signed_area([positions[v] for v in ring])), default=[])
                for parts in rings]
    adjacency = _sector_adjacency(sectors, linedefs, sidedefs)
    adjacency["edges"] += _teleport_edges(sectors, linedefs, sidedefs, things, effective, vertices,
                                          hexen_style=hexen_style)
    triggers = _trigger_map(sectors, linedefs, hexen_style=hexen_style)
    from wad_parser.traversal import distances_from_raw, TRAVERSAL_VERSION
    teleport_edges = [edge for edge in adjacency['edges'] if edge.get('kind') == 'teleport']
    for edge in teleport_edges:
        edge['directed'] = True
    reachable = sorted(distances_from_raw(vertices, sectors, linedefs, sidedefs, things,
                                          effective, teleport_edges, hexen_style=hexen_style))
    alignment = _texture_alignment_score(sectors, linedefs, sidedefs, rings)
    bounds = _bounds(vertices)
    derived = DerivedData(
        sector_graph=adjacency,
        sector_polygons=polygons,
        sector_rings=rings,
        sector_regions=sector_regions,
        raw_geometry_issues=raw_geometry_issues,
        bsp_geometry=bsp_geometry,
        geometry_issues=geometry_issues,
        derived_version=DERIVED_VERSION,
        traversal_version=TRAVERSAL_VERSION,
        trigger_map=triggers,
        reachable_from_start=reachable,
        texture_alignment_score=alignment,
        symmetry=_detect_symmetry(vertices, bounds),
        normalized_space=_normalized_space(vertices, bounds),
        bsp_sanity_check=(
            _bsp_sanity_check(group, linedefs, sidedefs, sectors, polygons) if group else None
        ),
    )
    return bounds, derived


def extract_map(wad: WadFile, group: MapLumpGroup, mapinfo: dict | None = None) -> MapDocument:
    doc_extra: dict = {}
    if group.format == "udmf":
        vertices, sectors, linedefs, sidedefs, things, udmf_doc = _parse_udmf_group(group)
        doc_extra["udmf_namespace"] = udmf_doc.namespace
        doc_extra["udmf_globals"] = {k: v for k, v in udmf_doc.globals.items() if k != "namespace"}
        hexen_style = (udmf_doc.namespace or "").lower() in HEXEN_UDMF_NAMESPACES
    else:
        vertices, sectors, linedefs, sidedefs, things = _parse_binary_group(group)
        hexen_style = group.format == "hexen"

    bounds, derived = compute_derived(vertices, sectors, linedefs, sidedefs, things,
                                      hexen_style=hexen_style, group=group)

    entry = (mapinfo or {}).get(group.name)

    meta = MapMeta(
        map_name=group.name,
        source_wad=wad.path,
        wad_hash=wad.hash,
        format=group.format,
        is_iwad=(wad.identification == "IWAD"),
        mapinfo=entry,
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
        extra=doc_extra,
    )


def extract_all_maps(wad: WadFile, map_name: str | None = None) -> list[MapDocument]:
    from wad_parser.mapinfo_parser import parse_mapinfo

    # UMAPINFO (the Boom/MBF21-lineage standard) uses the same
    # `map MAPxx { key = value }` shape the tolerant parser handles; listed
    # first so the more specific port dialects override it where both exist.
    mapinfo = {}
    for lump_name in ("UMAPINFO", "MAPINFO", "ZMAPINFO", "EMAPINFO"):
        data = wad.get_lump(lump_name)
        if data:
            mapinfo.update(parse_mapinfo(data))

    return [extract_map(wad, group, mapinfo) for group in wad.find_map_groups()
            if map_name is None or group.name.upper() == map_name.upper()]

"""
High-level map extraction.

extract_map() assembles a fully typed DoomMap from raw WAD lumps and
computes all derived data:
  • sector adjacency graph
  • per-sector boundary polygon rings (ordered vertex-ID lists)
  • trigger map  (sector_id → list of activating linedef IDs)
  • connectivity  (reachable / isolated sectors from player starts)
"""
from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path

from .lump_parser import (
    parse_linedefs,
    parse_sectors,
    parse_sidedefs,
    parse_things,
    parse_vertexes,
)
from .types import (
    BoundingBox,
    Linedef,
    LindefFlags,
    DoomMap,
    MapMeta,
    LINEDEF_SPECIALS,
    SECTOR_SPECIALS,
    Sector,
    Sidedef,
    THING_TYPES,
    Thing,
    ThingFlags,
    TextureSet,
    Vertex,
)
from .wad_reader import WadReader


# ─── Public entry point ───────────────────────────────────────────────────────

def extract_map(
    wad: WadReader,
    map_name: str,
    texture_set: TextureSet | None = None,
) -> DoomMap:
    """
    Parse all map lumps for *map_name* from *wad* and return a populated DoomMap.

    *texture_set* is optional; if provided it is stored for reference but not
    embedded in the returned map (textures are referenced by name).
    """
    lumps = wad.get_map_lumps(map_name)
    _require(lumps, "THINGS",   map_name)
    _require(lumps, "LINEDEFS", map_name)
    _require(lumps, "SIDEDEFS", map_name)
    _require(lumps, "VERTEXES", map_name)
    _require(lumps, "SECTORS",  map_name)

    # ── Raw parse ────────────────────────────────────────────────────────────
    raw_vertices = parse_vertexes(lumps["VERTEXES"])
    raw_sectors  = parse_sectors(lumps["SECTORS"])
    raw_sidedefs = parse_sidedefs(lumps["SIDEDEFS"])
    raw_linedefs = parse_linedefs(lumps["LINEDEFS"])
    raw_things   = parse_things(lumps["THINGS"])

    # ── Build typed objects ──────────────────────────────────────────────────
    vertices = [Vertex(id=i, x=r["x"], y=r["y"]) for i, r in enumerate(raw_vertices)]
    sectors  = _build_sectors(raw_sectors)
    sidedefs = _build_sidedefs(raw_sidedefs)
    linedefs = _build_linedefs(raw_linedefs, len(sidedefs))
    things   = _build_things(raw_things)

    bounds = _compute_bounds(vertices)
    meta   = MapMeta(
        map_name=map_name,
        source_wad=str(Path(wad.path).name),
        wad_type=wad.wad_type,
    )

    doom_map = DoomMap(
        meta=meta,
        bounds=bounds,
        vertices=vertices,
        sectors=sectors,
        linedefs=linedefs,
        sidedefs=sidedefs,
        things=things,
    )

    # ── Derived data (order matters) ─────────────────────────────────────────
    _compute_sector_graph(doom_map)
    _compute_sector_polygons(doom_map)
    _compute_trigger_map(doom_map)
    _compute_connectivity(doom_map)

    return doom_map


# ─── Builder helpers ──────────────────────────────────────────────────────────

def _build_sectors(raw: list) -> list[Sector]:
    sectors = []
    for i, r in enumerate(raw):
        special_name = SECTOR_SPECIALS.get(r["special"], "")
        sectors.append(Sector(
            id=i,
            floor_height=r["floor_height"],
            ceil_height=r["ceil_height"],
            floor_flat=r["floor_flat"],
            ceil_flat=r["ceil_flat"],
            light_level=r["light_level"],
            special=r["special"],
            tag=r["tag"],
            special_name=special_name,
        ))
    return sectors


def _build_sidedefs(raw: list) -> list[Sidedef]:
    return [
        Sidedef(
            id=i,
            x_offset=r["x_offset"],
            y_offset=r["y_offset"],
            upper_tex=r["upper_tex"],
            lower_tex=r["lower_tex"],
            mid_tex=r["mid_tex"],
            sector=r["sector"],
        )
        for i, r in enumerate(raw)
    ]


def _build_linedefs(raw: list, num_sidedefs: int) -> list[Linedef]:
    linedefs = []
    for i, r in enumerate(raw):
        # Validate sidedef references
        right = r["right_sidedef"]
        left  = r["left_sidedef"]
        if right >= num_sidedefs:
            warnings.warn(f"Linedef {i}: right_sidedef {right} out of range; clamping to 0")
            right = 0
        if left != -1 and left >= num_sidedefs:
            warnings.warn(f"Linedef {i}: left_sidedef {left} out of range; treating as one-sided")
            left = -1

        special = r["special"]
        sp_info = LINEDEF_SPECIALS.get(special, (f"Unknown ({special})", None, None))
        linedefs.append(Linedef(
            id=i,
            v1=r["v1"],
            v2=r["v2"],
            flags=LindefFlags.from_int(r["flags"]),
            special=special,
            tag=r["tag"],
            right_sidedef=right,
            left_sidedef=left,
            special_name=sp_info[0] if special else "",
            trigger_type=sp_info[1] or "",
            action_type=sp_info[2] or "",
        ))
    return linedefs


def _build_things(raw: list) -> list[Thing]:
    things = []
    for i, r in enumerate(raw):
        ttype = r["type"]
        info  = THING_TYPES.get(ttype)
        things.append(Thing(
            id=i,
            x=r["x"],
            y=r["y"],
            angle=r["angle"],
            type=ttype,
            flags=ThingFlags.from_int(r["flags"]),
            type_name=info[0] if info else "",
            category=info[1] if info else "unknown",
        ))
    return things


# ─── Bounds ───────────────────────────────────────────────────────────────────

def _compute_bounds(vertices: list[Vertex]) -> BoundingBox:
    if not vertices:
        return BoundingBox(0, 0, 0, 0)
    xs = [v.x for v in vertices]
    ys = [v.y for v in vertices]
    return BoundingBox(min_x=min(xs), max_x=max(xs), min_y=min(ys), max_y=max(ys))


# ─── Sector adjacency graph ───────────────────────────────────────────────────

def _compute_sector_graph(m: DoomMap) -> None:
    """
    Build a sector adjacency graph from two-sided linedefs.

    Result stored in m.sector_graph:
      nodes: list of {id, floor_h, ceil_h, light}
      edges: list of {sector_a, sector_b, linedef}
      adjacency: {str(sector_id): [adjacent_sector_id, ...]}
    """
    edges: list[dict] = []
    adjacency: dict[int, list[int]] = defaultdict(list)

    seen_pairs: set[tuple[int, int, int]] = set()

    for ld in m.linedefs:
        if ld.left_sidedef == -1:
            continue  # one-sided: no adjacency
        rs = m.sidedefs[ld.right_sidedef].sector
        ls = m.sidedefs[ld.left_sidedef].sector
        if rs == ls:
            continue  # self-referencing sector (special effect trick)

        key = (min(rs, ls), max(rs, ls), ld.id)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)

        edges.append({"sector_a": rs, "sector_b": ls, "linedef": ld.id})
        adjacency[rs].append(ls)
        adjacency[ls].append(rs)

    nodes = [
        {"id": s.id, "floor_h": s.floor_height, "ceil_h": s.ceil_height, "light": s.light_level}
        for s in m.sectors
    ]

    m.sector_graph = {
        "nodes": nodes,
        "edges": edges,
        "adjacency": {str(k): v for k, v in adjacency.items()},
    }


# ─── Sector polygon tracing ───────────────────────────────────────────────────

def _compute_sector_polygons(m: DoomMap) -> None:
    """
    For each sector, trace the ordered boundary ring(s) of vertex IDs.

    A sector is a face in the planar subdivision: its boundary consists of
    one or more oriented rings of directed half-edges.  Each half-edge
    (v_from → v_to) "faces" the sector — i.e. the sector lies on the right
    side of the directed edge.

    Conventions:
      right sidedef → linedef goes v1 → v2  (sector on the right)
      left  sidedef → linedef goes v2 → v1  (sector on the right of the reversed edge)

    Stores m.sector_polygons as a list indexed by sector ID, where each entry
    is a list of rings, and each ring is an ordered list of vertex IDs.
    """
    num_sectors = len(m.sectors)

    # Collect directed half-edges per sector: sector_id → list[(v_from, v_to)]
    half_edges: list[list[tuple[int, int]]] = [[] for _ in range(num_sectors)]

    for ld in m.linedefs:
        rs_idx = ld.right_sidedef
        ls_idx = ld.left_sidedef

        if rs_idx >= 0:
            sec_r = m.sidedefs[rs_idx].sector
            if 0 <= sec_r < num_sectors:
                half_edges[sec_r].append((ld.v1, ld.v2))

        if ls_idx >= 0:
            sec_l = m.sidedefs[ls_idx].sector
            if 0 <= sec_l < num_sectors:
                half_edges[sec_l].append((ld.v2, ld.v1))

    polygons: list[list[list[int]]] = []

    for sector_id in range(num_sectors):
        edges = half_edges[sector_id]
        if not edges:
            polygons.append([])
            continue

        # Build next-vertex map.  If a vertex has multiple outgoing half-edges
        # within this sector's boundary the map is non-manifold; keep the first
        # and warn (degenerate map data).
        next_v: dict[int, int] = {}
        for v_from, v_to in edges:
            if v_from in next_v and next_v[v_from] != v_to:
                warnings.warn(
                    f"Sector {sector_id}: vertex {v_from} has multiple outgoing "
                    f"half-edges — map geometry may be degenerate"
                )
            next_v[v_from] = v_to

        # Trace rings by following the next-vertex chain
        visited: set[int] = set()
        rings: list[list[int]] = []

        for start_v in list(next_v.keys()):
            if start_v in visited:
                continue
            ring: list[int] = []
            v = start_v
            # Safety limit: a ring can be at most len(next_v) long
            for _ in range(len(next_v) + 1):
                if v in visited or v not in next_v:
                    break
                visited.add(v)
                ring.append(v)
                v = next_v[v]
            if ring:
                rings.append(ring)

        polygons.append(rings)

    m.sector_polygons = polygons


# ─── Trigger map ─────────────────────────────────────────────────────────────

def _compute_trigger_map(m: DoomMap) -> None:
    """
    For every sector with tag > 0, collect the IDs of linedefs whose
    special + tag would activate that sector.

    Stored in m.trigger_map as {str(sector_id): [linedef_id, ...]}.
    """
    sectors_by_tag: dict[int, list[int]] = defaultdict(list)
    for s in m.sectors:
        if s.tag > 0:
            sectors_by_tag[s.tag].append(s.id)

    trigger_map: dict[str, list[int]] = {}
    for ld in m.linedefs:
        if ld.special > 0 and ld.tag > 0:
            for sector_id in sectors_by_tag.get(ld.tag, []):
                trigger_map.setdefault(str(sector_id), []).append(ld.id)

    m.trigger_map = trigger_map


# ─── Connectivity ─────────────────────────────────────────────────────────────

def _compute_connectivity(m: DoomMap) -> None:
    """
    BFS from every player-start sector through two-sided linedefs to find
    reachable / isolated sectors.

    Player starts are things with type 1–4 (any skill flag set).
    Their containing sector is found via point-in-polygon against the
    pre-computed sector_polygons.
    """
    # Build adjacency from two-sided linedefs
    adj: dict[int, set[int]] = defaultdict(set)
    for ld in m.linedefs:
        if ld.left_sidedef == -1:
            continue
        rs = m.sidedefs[ld.right_sidedef].sector
        ls = m.sidedefs[ld.left_sidedef].sector
        if rs != ls:
            adj[rs].add(ls)
            adj[ls].add(rs)

    # Locate player start sectors
    start_sectors: set[int] = set()
    for thing in m.things:
        if thing.type in (1, 2, 3, 4):
            sid = _find_sector_for_point(thing.x, thing.y, m)
            if sid >= 0:
                start_sectors.add(sid)

    if not start_sectors:
        all_ids = list(range(len(m.sectors)))
        m.connectivity = {
            "player_start_sectors": [],
            "reachable_sectors": [],
            "isolated_sectors": all_ids,
        }
        return

    # BFS
    reachable: set[int] = set(start_sectors)
    queue = list(start_sectors)
    while queue:
        current = queue.pop()
        for neighbor in adj.get(current, set()):
            if neighbor not in reachable:
                reachable.add(neighbor)
                queue.append(neighbor)

    all_sectors = set(range(len(m.sectors)))
    isolated = sorted(all_sectors - reachable)

    m.connectivity = {
        "player_start_sectors": sorted(start_sectors),
        "reachable_sectors": sorted(reachable),
        "isolated_sectors": isolated,
    }


# ─── Point-in-polygon ────────────────────────────────────────────────────────

def _point_in_polygon(px: int, py: int, ring: list[int], vertices: list[Vertex]) -> bool:
    """Ray-casting even-odd rule for a ring given as vertex-ID list."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = vertices[ring[i]].x, vertices[ring[i]].y
        xj, yj = vertices[ring[j]].x, vertices[ring[j]].y
        if (yi > py) != (yj > py):
            x_intersect = (xj - xi) * (py - yi) / (yj - yi) + xi
            if px < x_intersect:
                inside = not inside
        j = i
    return inside


def _find_sector_for_point(px: int, py: int, m: DoomMap) -> int:
    """Return the sector ID containing (px, py), or -1 if none found."""
    for sector_id, rings in enumerate(m.sector_polygons):
        for ring in rings:
            if ring and _point_in_polygon(px, py, ring, m.vertices):
                return sector_id
    return -1


# ─── Utility ─────────────────────────────────────────────────────────────────

def _require(lumps: dict[str, bytes], name: str, map_name: str) -> None:
    if name not in lumps:
        raise ValueError(f"Map '{map_name}' is missing required lump '{name}'")

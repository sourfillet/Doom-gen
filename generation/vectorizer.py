"""Stage 2: sampled raster -> MapDocument skeleton (plan/generation.md).

Pipeline: downsample to cells -> cluster cells into sectors by quantized
(floor, ceiling, light) + 4-connectivity -> merge speck regions -> trace
boundary chains -> simplify each chain (Douglas-Peucker) -> emit linedefs.

Two properties are load-bearing:

1. **Watertightness by shared chains.** A boundary CHAIN is the maximal
   polyline separating one pair of regions (or a region and the void)
   between junction points. Each chain is simplified once and both
   neighbors reference the same resulting vertices - adjacent sectors can
   never disagree about their shared border, so simplification cannot
   open leaks. Junction points (where boundary ownership changes) are
   chain endpoints by construction and are never moved.

2. **Right-side-front.** Unit boundary edges are generated with the
   owning region on the right of v1->v2; a chain inherits a consistent
   orientation from its first edge, so emitted linedefs keep Doom's
   front-side invariant.

simplify_tolerance is in cell units; 0 disables simplification and gives
the old fully rectilinear output.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from wad_parser.map_extractor import compute_derived
from wad_parser.types import (
    Linedef, LinedefFlags, MapDocument, MapMeta, SCHEMA_VERSION, Sector,
    Sidedef, Vertex,
)
from data.rasterizer import ABS_HEIGHT_SCALE

DEFAULT_WALL_TEX = "STARTAN2"
DEFAULT_FLOOR_FLAT = "FLOOR4_8"
DEFAULT_CEIL_FLAT = "CEIL3_5"

VOID = -1


def _decode_abs_height(v: float) -> int:
    """Inverse of the rasterizer's absolute encoding, snapped to 8 units."""
    h = (v * 2.0 - 1.0) * ABS_HEIGHT_SCALE
    return int(round(h / 8.0)) * 8


def _downsample(chan: np.ndarray, factor: int) -> np.ndarray:
    h, w = chan.shape
    h2, w2 = h // factor, w // factor
    return chan[: h2 * factor, : w2 * factor].reshape(h2, factor, w2, factor).mean(axis=(1, 3))


def _label_regions(interior: np.ndarray, bins: np.ndarray) -> np.ndarray:
    h, w = interior.shape
    labels = np.full((h, w), VOID, dtype=np.int64)
    next_label = 0
    for r in range(h):
        for c in range(w):
            if not interior[r, c] or labels[r, c] != VOID:
                continue
            q = deque([(r, c)])
            labels[r, c] = next_label
            while q:
                rr, cc = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and interior[nr, nc] \
                            and labels[nr, nc] == VOID \
                            and (bins[nr, nc] == bins[rr, cc]).all():
                        labels[nr, nc] = next_label
                        q.append((nr, nc))
            next_label += 1
    return labels


def _merge_small_regions(labels: np.ndarray, min_cells: int) -> np.ndarray:
    """Specks below min_cells merge into the neighbor they share the most
    boundary with (model samples produce single-cell confetti otherwise).
    Speck clusters entirely surrounded by void become void."""
    h, w = labels.shape
    for _ in range(4):  # merging can create new merge targets; cap passes
        counts = {}
        for v in labels.flat:
            if v != VOID:
                counts[v] = counts.get(v, 0) + 1
        small = {lab for lab, n in counts.items() if n < min_cells}
        if not small:
            break
        changed = False
        for lab in small:
            shared: dict[int, int] = {}
            cells = np.argwhere(labels == lab)
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    nb = labels[nr, nc] if 0 <= nr < h and 0 <= nc < w else VOID
                    if nb != lab:
                        shared[nb] = shared.get(nb, 0) + 1
            if shared:
                target = max(shared, key=shared.get)
                labels[labels == lab] = target
                changed = True
        if not changed:
            break
    # compact label ids
    remap = {old: new for new, old in enumerate(sorted(set(labels.flat) - {VOID}))}
    out = np.full_like(labels, VOID)
    for old, new in remap.items():
        out[labels == old] = new
    return out


def _douglas_peucker(points: list[tuple[int, int]], tolerance: float) -> list[tuple[int, int]]:
    if tolerance <= 0 or len(points) <= 2:
        return points
    (x1, y1), (x2, y2) = points[0], points[-1]
    dx, dy = x2 - x1, y2 - y1
    norm = max((dx * dx + dy * dy) ** 0.5, 1e-12)
    max_d, max_i = -1.0, 0
    for i in range(1, len(points) - 1):
        px, py = points[i]
        d = abs(dx * (y1 - py) - dy * (x1 - px)) / norm
        if d > max_d:
            max_d, max_i = d, i
    if max_d <= tolerance:
        return [points[0], points[-1]]
    left = _douglas_peucker(points[: max_i + 1], tolerance)
    right = _douglas_peucker(points[max_i:], tolerance)
    return left[:-1] + right


def _simplify_ring(points: list[tuple[int, int]], tolerance: float) -> list[tuple[int, int]]:
    """Closed loop (first == last): anchor two opposite points so DP has
    stable endpoints, simplify both halves."""
    if tolerance <= 0 or len(points) <= 4:
        return points
    mid = len(points) // 2
    first = _douglas_peucker(points[: mid + 1], tolerance)
    second = _douglas_peucker(points[mid:], tolerance)
    return first[:-1] + second


def vectorize(raster: np.ndarray, cell_units: int = 64, coarse_factor: int = 4,
              interior_threshold: float = 0.05, map_name: str = "MAP01",
              simplify_tolerance: float = 1.3, min_region_cells: int = 3) -> MapDocument:
    """Raster (9, R, R) -> doom-format MapDocument. Raises ValueError when
    the raster contains no usable interior."""
    painted = raster[1:7].max(axis=0) > interior_threshold
    coverage = _downsample(painted.astype(np.float64), coarse_factor)
    safe_cov = np.maximum(coverage, 1e-9)

    def masked_mean(chan: np.ndarray) -> np.ndarray:
        return _downsample(chan * painted, coarse_factor) / safe_cov

    floor_ch = masked_mean(raster[2])
    ceil_ch = masked_mean(raster[4])
    light_ch = masked_mean(raster[5])

    interior = coverage > 0.5
    if not interior.any():
        raise ValueError("raster has no interior cells above threshold")

    bins = np.stack([
        np.round(floor_ch * 32), np.round(ceil_ch * 32), np.round(light_ch * 8),
    ], axis=-1)
    labels = _merge_small_regions(_label_regions(interior, bins), min_region_cells)
    if labels.max() < 0:
        raise ValueError("raster interior collapsed to nothing after speck removal")
    n_regions = labels.max() + 1
    grid_h, grid_w = labels.shape

    # --- sectors ---
    sectors: list[Sector] = []
    seed_points: dict[int, list[float]] = {}
    for region in range(n_regions):
        mask = labels == region
        floor = _decode_abs_height(float(floor_ch[mask].mean()))
        ceil = _decode_abs_height(float(ceil_ch[mask].mean()))
        light = int(np.clip(round(float(light_ch[mask].mean()) * 255), 0, 255))
        sectors.append(Sector(
            id=region, floor_height=floor, ceil_height=ceil,
            floor_flat=DEFAULT_FLOOR_FLAT, ceil_flat=DEFAULT_CEIL_FLAT,
            light_level=light, special=0, tag=0,
        ))
        rr, cc = np.nonzero(mask)
        # a cell center, in lattice coords * cell_units
        seed_points[region] = [(int(cc[0]) + 0.5) * cell_units,
                               (grid_h - 1 - int(rr[0]) + 0.5) * cell_units]

    # --- unit boundary edges in corner-lattice coords ---
    # segment key: (lo_pt, hi_pt); owners: {region: oriented (v1, v2)}
    segments: dict[tuple, dict[int, tuple]] = {}
    for r in range(grid_h):
        for c in range(grid_w):
            region = labels[r, c]
            if region == VOID:
                continue
            x, y = c, grid_h - 1 - r  # SW corner
            sides = [  # (neighbor cell, v1, v2): region on the RIGHT of v1->v2
                ((r - 1, c), (x, y + 1), (x + 1, y + 1)),
                ((r + 1, c), (x + 1, y), (x, y)),
                ((r, c - 1), (x, y), (x, y + 1)),
                ((r, c + 1), (x + 1, y + 1), (x + 1, y)),
            ]
            for (nr, nc), v1, v2 in sides:
                nb = labels[nr, nc] if 0 <= nr < grid_h and 0 <= nc < grid_w else VOID
                if nb == region:
                    continue
                key = (min(v1, v2), max(v1, v2))
                segments.setdefault(key, {})[region] = (v1, v2)

    # --- group segments into per-region-pair graphs ---
    groups: dict[frozenset, dict] = {}
    for key, owners in segments.items():
        pair = frozenset(owners) if len(owners) == 2 else frozenset([next(iter(owners)), VOID])
        g = groups.setdefault(pair, {"adj": {}, "owners": {}})
        a, b = key
        g["adj"].setdefault(a, []).append(b)
        g["adj"].setdefault(b, []).append(a)
        g["owners"][key] = owners

    vertices: dict[tuple[int, int], int] = {}
    linedefs: list[Linedef] = []
    sidedefs: list[Sidedef] = []

    def vid(pt) -> int:
        pt = (int(pt[0] * cell_units), int(pt[1] * cell_units))
        if pt not in vertices:
            vertices[pt] = len(vertices)
        return vertices[pt]

    def emit_chain(points: list[tuple[int, int]], owners_of_first: dict[int, tuple],
                   pair: frozenset, closed: bool) -> None:
        pts = (_simplify_ring if closed else _douglas_peucker)(points, simplify_tolerance)
        # orientation: which region has the chain's first ORIGINAL edge as
        # its oriented (front-on-right) direction?
        first_dir = (points[0], points[1])
        front = back = None
        for region, oriented in owners_of_first.items():
            if oriented == first_dir:
                front = region
            else:
                back = region
        if front is None:  # walked against the stored orientation
            front, back = back, None
            pts = list(reversed(pts))
            if len(owners_of_first) == 2:
                back = next(r for r in owners_of_first if r != front)
        if back is None and len(pair - {VOID}) == 2:
            back = next(r for r in pair if r != front)
        two_sided = back is not None and back != VOID

        for i in range(len(pts) - 1):
            right_id = len(sidedefs)
            sidedefs.append(Sidedef(
                id=right_id, x_offset=0, y_offset=0,
                upper_tex=DEFAULT_WALL_TEX if two_sided else "-",
                lower_tex=DEFAULT_WALL_TEX if two_sided else "-",
                mid_tex="-" if two_sided else DEFAULT_WALL_TEX,
                sector=front,
            ))
            left_id = -1
            if two_sided:
                left_id = len(sidedefs)
                sidedefs.append(Sidedef(
                    id=left_id, x_offset=0, y_offset=0,
                    upper_tex=DEFAULT_WALL_TEX, lower_tex=DEFAULT_WALL_TEX,
                    mid_tex="-", sector=back,
                ))
            flags = LinedefFlags(two_sided=two_sided, impassable=not two_sided)
            linedefs.append(Linedef(
                id=len(linedefs), v1=vid(pts[i]), v2=vid(pts[i + 1]), flags=flags,
                special=0, tag=0, right_sidedef=right_id, left_sidedef=left_id,
                special_name="None", trigger_type="none",
            ))

    for pair, g in groups.items():
        adj = g["adj"]
        owners = g["owners"]
        unused = set(owners)

        def seg_key(a, b):
            return (min(a, b), max(a, b))

        def walk(start, first_next):
            """Consume a path from `start` through `first_next` until a
            junction (degree != 2) or the loop closes."""
            path = [start, first_next]
            unused.discard(seg_key(start, first_next))
            while True:
                cur, prev = path[-1], path[-2]
                nbrs = adj[cur]
                if len(nbrs) != 2 or cur == start:
                    return path
                nxt = nbrs[0] if nbrs[1] == prev else nbrs[1]
                if seg_key(cur, nxt) not in unused:
                    return path
                unused.discard(seg_key(cur, nxt))
                path.append(nxt)

        endpoints = [p for p, nbrs in adj.items() if len(nbrs) != 2]
        for p in endpoints:
            for nb in list(adj[p]):
                if seg_key(p, nb) in unused:
                    path = walk(p, nb)
                    owners_first = owners[seg_key(path[0], path[1])]
                    emit_chain(path, owners_first, pair, closed=False)
        while unused:  # leftover pure loops (isolated islands/holes)
            a, b = next(iter(unused))
            path = walk(a, b)
            owners_first = owners[seg_key(path[0], path[1])]
            emit_chain(path, owners_first, pair, closed=path[0] == path[-1])

    vertex_objs = [Vertex(id=i, x=float(pt[0]), y=float(pt[1]))
                   for pt, i in sorted(vertices.items(), key=lambda kv: kv[1])]

    bounds, derived = compute_derived(vertex_objs, sectors, linedefs, sidedefs, [])
    meta = MapMeta(map_name=map_name, source_wad="<generated>", wad_hash="",
                   format="doom", is_iwad=False)
    return MapDocument(
        schema_version=SCHEMA_VERSION, meta=meta, bounds=bounds,
        vertices=vertex_objs, sectors=sectors, linedefs=linedefs,
        sidedefs=sidedefs, things=[], derived=derived,
        extra={"generated": True,
               "sector_seed_points": {str(k): v for k, v in seed_points.items()}},
    )

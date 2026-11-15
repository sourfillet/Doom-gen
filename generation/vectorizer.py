"""Stage 2: sampled raster -> MapDocument skeleton.

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
from data.rasterizer import ABS_HEIGHT_SCALE, SEMANTIC_SPECIALS, SEMANTIC_END, N_CHANNELS, RASTER_VERSION

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


def _components(mask):
    h, w = mask.shape
    labels = np.full(h * w, -1, dtype=np.int64)
    values, count = mask.ravel(), 0
    for seed in np.flatnonzero(values):
        if labels[seed] >= 0:
            continue
        stack, labels[seed] = [int(seed)], count
        while stack:
            index = stack.pop()
            row, col = divmod(index, w)
            neighbors = ([] if col == 0 else [index - 1]) + ([] if col + 1 == w else [index + 1])
            if row:
                neighbors.append(index - w)
            if row + 1 < h:
                neighbors.append(index + w)
            for other in neighbors:
                if values[other] and labels[other] < 0:
                    labels[other] = count
                    stack.append(other)
        count += 1
    return labels.reshape(mask.shape), count


def _changes_occupancy_topology(painted, coarse, factor):
    # Match connected components in both occupancy and void. This detects
    # erased/split rooms and passages as well as filled holes/merged islands.
    for value in (True, False):
        source, n_source = _components(painted == value)
        target, n_target = _components(coarse == value)
        if not n_source and not n_target:
            continue
        if n_source != n_target:
            return True
        rr, cc = np.nonzero(source >= 0)
        rows, cols = rr // factor, cc // factor
        valid = (rows < target.shape[0]) & (cols < target.shape[1])
        rr, cc, rows, cols = rr[valid], cc[valid], rows[valid], cols[valid]
        matched = target[rows, cols] >= 0
        pairs = np.unique(source[rr[matched], cc[matched]] * n_target + target[rows[matched], cols[matched]])
        if (not np.all(np.bincount(pairs // n_target, minlength=n_source) == 1)
                or not np.all(np.bincount(pairs % n_target, minlength=n_target) == 1)):
            return True
    return False


def _label_regions(interior: np.ndarray, bins: np.ndarray, barriers=frozenset()) -> np.ndarray:
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
                            and frozenset(((rr, cc), (nr, nc))) not in barriers \
                            and (bins[nr, nc] == bins[rr, cc]).all():
                        labels[nr, nc] = next_label
                        q.append((nr, nc))
            next_label += 1
    return labels


def _merge_small_regions(labels: np.ndarray, min_cells: int, barriers=frozenset()) -> np.ndarray:
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
                        if frozenset(((r, c), (nr, nc))) in barriers:
                            continue
                        shared[nb] = shared.get(nb, 0) + 1
            if shared:
                non_void = {nb: n for nb, n in shared.items() if nb != VOID}
                pool = non_void or shared
                target = max(pool, key=pool.get)
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
    simplified = first[:-1] + second
    return simplified if len(simplified) >= 4 else points


def vectorize(raster: np.ndarray, cell_units: float = 64, coarse_factor: int = 2,
              interior_threshold: float = 0.05, map_name: str = "MAP01",
              simplify_tolerance: float = 0, min_region_cells: int = 1,
              raster_version: int = RASTER_VERSION, preserve_topology: bool = True) -> MapDocument:
    """Raster (10, R, R) -> doom-format MapDocument. Raises ValueError when
    the raster contains no usable interior."""
    if coarse_factor < 1 or cell_units <= 0 or min_region_cells < 1:
        raise ValueError("coarse_factor/cell_units/min_region_cells must be positive")
    if raster.ndim != 3 or raster.shape[0] < 9 or not np.isfinite(raster).all():
        raise ValueError("raster must have at least nine finite channels")
    if min(raster.shape[1:]) < coarse_factor:
        raise ValueError("coarse factor exceeds raster dimensions")
    faces = raster_version >= 5 and raster.shape[0] >= N_CHANNELS
    if faces and coarse_factor > 1:
        doc = vectorize(raster, cell_units=cell_units / coarse_factor, coarse_factor=1,
                        interior_threshold=interior_threshold, map_name=map_name,
                        simplify_tolerance=simplify_tolerance, min_region_cells=min_region_cells,
                        raster_version=raster_version, preserve_topology=preserve_topology)
        doc.extra['raster_decoding'] = {'requested_factor': coarse_factor, 'effective_factor': 1,
                                      'reason': 'grid-face targets decode at native pixel resolution'}
        return doc
    # With an occupancy channel (index 9) present, void is explicit: threshold
    # it at 0.5. Otherwise fall back to "any structural channel is nonzero".
    if raster.shape[0] >= 10:
        painted = raster[9] > 0.5
    else:
        painted = raster[1:7].max(axis=0) > interior_threshold
    coverage = _downsample(painted.astype(np.float64), coarse_factor)
    if preserve_topology and coarse_factor > 1 and _changes_occupancy_topology(painted, coverage > 0.5, coarse_factor):
        doc = vectorize(raster, cell_units=cell_units / coarse_factor, coarse_factor=1,
                        interior_threshold=interior_threshold, map_name=map_name,
                        simplify_tolerance=simplify_tolerance, min_region_cells=min_region_cells,
                        raster_version=raster_version, preserve_topology=False)
        doc.extra['raster_decoding'] = {'requested_factor': coarse_factor, 'effective_factor': 1,
                                      'reason': 'coarse occupancy changed connected components or holes'}
        return doc
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
    def boundary_barriers(channel):
        barriers = set()
        h, w = interior.shape
        for r in range(h):
            for c in range(w):
                for dr, dc in ((0, 1), (1, 0)):
                    nr, nc = r + dr, c + dc
                    if nr >= h or nc >= w or not (interior[r, c] and interior[nr, nc]):
                        continue
                    t = np.linspace(0.06, 0.94, max(4, coarse_factor * 4))
                    rr = np.clip(np.round((r + 0.5 + t * dr) * coarse_factor - 0.5).astype(int), 0, channel.shape[0] - 1)
                    cc = np.clip(np.round((c + 0.5 + t * dc) * coarse_factor - 0.5).astype(int), 0, channel.shape[1] - 1)
                    values = channel[rr, cc]
                    # A one-pixel wall parallel to the centre-to-centre
                    # segment can paint the whole sampling run. It is not a
                    # crossed barrier. Retain a saturated run only when the
                    # wall also extends in a perpendicular direction (one
                    # side is sufficient at an outer-wall junction).
                    before_r = np.clip(rr - dc, 0, channel.shape[0] - 1)
                    after_r = np.clip(rr + dc, 0, channel.shape[0] - 1)
                    before_c = np.clip(cc - dr, 0, channel.shape[1] - 1)
                    after_c = np.clip(cc + dr, 0, channel.shape[1] - 1)
                    transverse = ((channel[before_r, before_c] > 0.5)
                                  | (channel[after_r, after_c] > 0.5)).any()
                    if values.max() > 0.5 and (values.min() <= 0.5 or transverse):
                        barriers.add(frozenset(((r, c), (nr, nc))))
        return barriers
    solid_barriers = boundary_barriers(raster[0]) if not faces else set()
    # Nine-channel fixtures and old checkpoints used numeric sector coloring.
    boundaries = np.maximum(raster[0], raster[6]) if raster_version >= 2 and raster.shape[0] >= 10 else raster[0]
    barriers = boundary_barriers(boundaries) if not faces else set()
    if faces:
        for r, c in np.argwhere(interior):
            for axis, (dr, dc) in enumerate(((0, 1), (1, 0))):
                nr, nc = r + dr, c + dc
                if nr < interior.shape[0] and nc < interior.shape[1] and interior[nr, nc]:
                    key = frozenset(((int(r), int(c)), (int(nr), int(nc))))
                    if raster[16 + axis, r, c] > 0.5 or raster[18 + axis, r, c] > 0.5:
                        barriers.add(key)
                    if raster[18 + axis, r, c] > 0.5:
                        solid_barriers.add(key)
    labels = _merge_small_regions(_label_regions(interior, bins, barriers), min_region_cells, barriers)
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
    segment_solids = {}
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
                segment_solids[key] = nb == VOID or frozenset(((r, c), (nr, nc))) in solid_barriers

    # --- group segments into per-region-pair graphs ---
    groups: dict[frozenset, dict] = {}
    for key, owners in segments.items():
        pair = frozenset(owners) if len(owners) == 2 else frozenset([next(iter(owners)), VOID])
        g = groups.setdefault((pair, segment_solids[key]), {"adj": {}, "owners": {}})
        a, b = key
        g["adj"].setdefault(a, []).append(b)
        g["adj"].setdefault(b, []).append(a)
        g["owners"][key] = owners

    vertices: dict[tuple[int, int], int] = {}
    linedefs: list[Linedef] = []
    sidedefs: list[Sidedef] = []

    def semantic_code(a, b):
        if raster_version < 4 or raster.shape[0] < SEMANTIC_END:
            return 0
        scores = np.zeros(len(SEMANTIC_SPECIALS))
        for t in (0.25, 0.5, 0.75):
            col = (a[0] + (b[0] - a[0]) * t) * coarse_factor
            row = (grid_h - a[1] - (b[1] - a[1]) * t) * coarse_factor
            cols = np.clip([int(np.floor(col)), int(np.ceil(col))], 0, raster.shape[2] - 1)
            rows = np.clip([int(np.floor(row)), int(np.ceil(row))], 0, raster.shape[1] - 1)
            scores += raster[10:SEMANTIC_END][:, rows[:, None], cols].max(axis=(1, 2)) / 3
        return SEMANTIC_SPECIALS[int(scores.argmax())] if scores.max() >= 0.6 else 0

    def vid(pt) -> int:
        pt = (int(round(pt[0] * cell_units)), int(round(pt[1] * cell_units)))
        if pt not in vertices:
            vertices[pt] = len(vertices)
        return vertices[pt]

    def emit_chain(points: list[tuple[int, int]], owners_of_first: dict[int, tuple],
                   pair: frozenset, closed: bool, solid: bool) -> None:
        # Remove redundant collinear lattice vertices without moving the
        # contour. Aggressive DP smoothing can cross nearby boundaries or
        # flatten a narrow corridor, so it is an explicit opt-in operation.
        codes = [semantic_code(a, b) for a, b in zip(points, points[1:])]
        compact = [points[0]]
        for i in range(1, len(points) - 1):
            a, b, c = points[i - 1], points[i], points[i + 1]
            if ((b[0] - a[0]) * (c[1] - b[1]) != (b[1] - a[1]) * (c[0] - b[0])
                    or codes[i - 1] != codes[i]):
                compact.append(b)
        compact.append(points[-1])
        pts = (compact if any(codes) else
               (_simplify_ring if closed else _douglas_peucker)(compact, simplify_tolerance))
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
        two_sided = bool(back is not None and back != VOID)

        for i in range(len(pts) - 1):
            right_id = len(sidedefs)
            sidedefs.append(Sidedef(
                id=right_id, x_offset=0, y_offset=0,
                upper_tex=DEFAULT_WALL_TEX if two_sided else "-",
                lower_tex=DEFAULT_WALL_TEX if two_sided else "-",
                mid_tex="-" if two_sided and not solid else DEFAULT_WALL_TEX,
                sector=int(front),
            ))
            left_id = -1
            if two_sided:
                left_id = len(sidedefs)
                sidedefs.append(Sidedef(
                    id=left_id, x_offset=0, y_offset=0,
                    upper_tex=DEFAULT_WALL_TEX, lower_tex=DEFAULT_WALL_TEX,
                    mid_tex=DEFAULT_WALL_TEX if solid else "-", sector=int(back),
                ))
            flags = LinedefFlags(two_sided=two_sided, impassable=not two_sided or solid)
            linedefs.append(Linedef(
                id=len(linedefs), v1=vid(pts[i]), v2=vid(pts[i + 1]), flags=flags,
                special=0, tag=0, right_sidedef=right_id, left_sidedef=left_id,
                special_name="None", trigger_type="none",
            ))
            line = linedefs[-1]
            special = semantic_code(pts[i], pts[i + 1])
            if special in {11, 51} and not two_sided:
                line.special = special
            elif special in {52, 124} and two_sided and not solid:
                line.special = special
            elif special in {1, 31} and two_sided and not solid:
                closed = [sid for sid in (front, back) if sectors[sid].ceil_height <= sectors[sid].floor_height + 8]
                if len(closed) == 1:
                    line.special = special
                    if closed[0] == front:
                        line.v1, line.v2 = line.v2, line.v1
                        line.right_sidedef, line.left_sidedef = line.left_sidedef, line.right_sidedef
            if line.special:
                from wad_parser.types import resolve_linedef_special
                line.special_name, line.trigger_type = resolve_linedef_special(line.special)

    for (pair, solid), g in groups.items():
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
                    emit_chain(path, owners_first, pair, closed=False, solid=solid)
        while unused:  # leftover pure loops (isolated islands/holes)
            a, b = next(iter(unused))
            path = walk(a, b)
            owners_first = owners[seg_key(path[0], path[1])]
            emit_chain(path, owners_first, pair, closed=path[0] == path[-1], solid=solid)

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
               'raster_decoding': {'requested_factor': coarse_factor, 'effective_factor': coarse_factor},
               "sector_seed_points": {str(k): v for k, v in seed_points.items()}},
    )

"""Map document -> multi-channel top-down raster (plan/generation.md,
Stage 1).

Channel layout (all float32 in [0, 1]):
    0  wall mask            one-sided linedefs
    1  floor height         map-relative (min..max -> 0..1)
    2  floor height         absolute, world units / 1024, clipped
    3  ceiling height       map-relative
    4  ceiling height       absolute / 1024, clipped
    5  light level          /255
    6  sector-ID coloring   (id+1)/n_sectors - topology hint, not a value
    7  linedef special density   line-rasterized specials, max-normalized
    8  thing density             cell counts, max-normalized
    9  occupancy                 binary 1=sector here, 0=void - the explicit
                                 negative-space target so the model learns
                                 rooms-in-void instead of filling the canvas

Two height encodings are kept because both matter: the model needs the
map's internal shape (relative) AND real-world scale (absolute - an 8-unit
step is a stair, a 64-unit step is a wall; see plan/data-model.md on why
normalized coordinates never replace raw ones).

Sectors are painted largest-bounding-box-first so nested sectors overwrite
their containers - a consequence of sector polygons being single boundary
rings (holes are not subtracted; documented simplification in the derived
layer).
"""
from __future__ import annotations

import numpy as np

from wad_parser.types import MapDocument

N_CHANNELS = 10
OCCUPANCY_CHANNEL = 9
ABS_HEIGHT_SCALE = 1024.0


def _grid_transform(doc: MapDocument, resolution: int, pad_frac: float = 0.03):
    b = doc.bounds
    span = max(b.max_x - b.min_x, b.max_y - b.min_y, 1.0)
    pad = span * pad_frac
    scale = resolution / (span + 2 * pad)
    # center the map in the square tile
    cx = (b.min_x + b.max_x) / 2
    cy = (b.min_y + b.max_y) / 2
    off = resolution / 2

    def to_grid(x, y):
        """world -> (col, row); row 0 is north (y-up flipped)."""
        col = (np.asarray(x, dtype=np.float64) - cx) * scale + off
        row = off - (np.asarray(y, dtype=np.float64) - cy) * scale
        return col, row

    meta = {"scale": scale, "center_x": cx, "center_y": cy, "resolution": resolution}
    return to_grid, meta


def _fill_polygon(mask_shape, cols, rows):
    """Even-odd point-in-polygon test, vectorized over the polygon's
    bounding box. Returns (row_idx, col_idx) of interior cells."""
    res_y, res_x = mask_shape
    min_c = max(int(np.floor(cols.min())), 0)
    max_c = min(int(np.ceil(cols.max())), res_x - 1)
    min_r = max(int(np.floor(rows.min())), 0)
    max_r = min(int(np.ceil(rows.max())), res_y - 1)
    if min_c > max_c or min_r > max_r:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    gc, gr = np.meshgrid(np.arange(min_c, max_c + 1) + 0.5,
                         np.arange(min_r, max_r + 1) + 0.5)
    inside = np.zeros(gc.shape, dtype=bool)
    n = len(cols)
    for i in range(n):
        x1, y1 = cols[i], rows[i]
        x2, y2 = cols[(i + 1) % n], rows[(i + 1) % n]
        crosses = (y1 > gr) != (y2 > gr)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = (x2 - x1) * (gr - y1) / (y2 - y1 + 1e-12) + x1
        inside ^= crosses & (gc < x_at)
    rr, cc = np.nonzero(inside)
    return rr + min_r, cc + min_c


def _draw_line(channel: np.ndarray, c1, r1, c2, r2, value: float = 1.0, accumulate=False):
    length = max(abs(c2 - c1), abs(r2 - r1))
    n = max(int(np.ceil(length * 2)), 1)
    t = np.linspace(0.0, 1.0, n + 1)
    cols = np.clip(np.round(c1 + (c2 - c1) * t).astype(int), 0, channel.shape[1] - 1)
    rows = np.clip(np.round(r1 + (r2 - r1) * t).astype(int), 0, channel.shape[0] - 1)
    if accumulate:
        # dedupe cells within this one line so its density contribution is 1
        cells = np.unique(rows * channel.shape[1] + cols)
        channel.flat[cells] += value
    else:
        channel[rows, cols] = value


def rasterize(doc: MapDocument, resolution: int = 128) -> tuple[np.ndarray, dict]:
    """Returns (raster (9, R, R) float32, meta dict with the world<->grid
    transform and height-normalization constants for invertibility)."""
    to_grid, meta = _grid_transform(doc, resolution)
    raster = np.zeros((N_CHANNELS, resolution, resolution), dtype=np.float32)

    vpos = {v.id: (v.x, v.y) for v in doc.vertices}

    floors = [s.floor_height for s in doc.sectors] or [0]
    ceils = [s.ceil_height for s in doc.sectors] or [0]
    f_min, f_max = min(floors), max(floors)
    c_min, c_max = min(ceils), max(ceils)
    meta.update({"floor_min": f_min, "floor_max": f_max,
                 "ceil_min": c_min, "ceil_max": c_max})
    f_span = max(f_max - f_min, 1)
    c_span = max(c_max - c_min, 1)
    n_sectors = max(len(doc.sectors), 1)

    # --- sector-area channels (1-6), nested sectors painted last ---
    order = []
    for s in doc.sectors:
        ring = (doc.derived.sector_polygons[s.id]
                if s.id < len(doc.derived.sector_polygons) else [])
        pts = np.array([vpos[v] for v in ring if v in vpos], dtype=np.float64)
        if len(pts) < 3:
            continue
        bbox_area = float(np.ptp(pts[:, 0]) * np.ptp(pts[:, 1]))
        order.append((bbox_area, s, pts))
    order.sort(key=lambda t: -t[0])

    for _, s, pts in order:
        cols, rows = to_grid(pts[:, 0], pts[:, 1])
        rr, cc = _fill_polygon((resolution, resolution), cols, rows)
        if len(rr) == 0:
            continue
        raster[1, rr, cc] = (s.floor_height - f_min) / f_span
        raster[2, rr, cc] = np.clip(s.floor_height / ABS_HEIGHT_SCALE, -1, 1) * 0.5 + 0.5
        raster[3, rr, cc] = (s.ceil_height - c_min) / c_span
        raster[4, rr, cc] = np.clip(s.ceil_height / ABS_HEIGHT_SCALE, -1, 1) * 0.5 + 0.5
        raster[5, rr, cc] = np.clip(s.light_level, 0, 255) / 255.0
        raster[6, rr, cc] = (s.id + 1) / n_sectors
        raster[OCCUPANCY_CHANNEL, rr, cc] = 1.0   # negative-space target

    # --- line channels (0, 7) ---
    for ld in doc.linedefs:
        if ld.v1 not in vpos or ld.v2 not in vpos:
            continue
        (x1, y1), (x2, y2) = vpos[ld.v1], vpos[ld.v2]
        c1, r1 = to_grid(x1, y1)
        c2, r2 = to_grid(x2, y2)
        if ld.left_sidedef == -1:
            _draw_line(raster[0], c1, r1, c2, r2)
        if ld.special:
            _draw_line(raster[7], c1, r1, c2, r2, accumulate=True)
    if raster[7].max() > 0:
        raster[7] /= raster[7].max()

    # --- thing density (8) ---
    for t in doc.things:
        c, r = to_grid(t.x, t.y)
        ci, ri = int(np.clip(c, 0, resolution - 1)), int(np.clip(r, 0, resolution - 1))
        raster[8, ri, ci] += 1.0
    if raster[8].max() > 0:
        raster[8] /= raster[8].max()

    return raster, meta

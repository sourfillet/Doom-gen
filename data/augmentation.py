"""Geometric/photometric augmentation on MAP DOCUMENTS, not tensors -
mirror/rotate transform the raw layer (with thing-angle correction,
linedef v1/v2 reversal, and slope-plane rotation), then the
derived layer is recomputed via compute_derived. Compiled BSP coordinate
regions are geometric input unavailable from raw authoring lines alone;
their affine transform is validated and preserved during recomputation.

Mirroring flips handedness: the sector that was on the right of v1->v2
lands on the LEFT of the mirrored line. Swapping v1/v2 (and ONLY that -
sidedefs stay put) reverses the direction so the front sector is back on
the right, preserving the "right sidedef faces the front" invariant the
format depends on. Swapping the sidedefs too would double-correct and
leave one-sided walls with no right side.

Doom thing angles are degrees CCW, 0 = east:
    mirror_x: a -> 180 - a      mirror_y: a -> -a      rot90 CCW: a -> a + 90
"""
from __future__ import annotations

import copy

import numpy as np

from wad_parser.map_extractor import compute_derived
from wad_parser.types import HEXEN_UDMF_NAMESPACES, MapDocument

AUGMENT_OPS = ("mirror_x", "mirror_y", "rot90", "rot180", "rot270")

# op -> (vertex/point transform, angle transform, flips_handedness,
#        slope (a,b) transform)
_OPS = {
    "mirror_x": (lambda x, y: (-x, y), lambda a: (180 - a) % 360, True,
                 lambda pa, pb: (-pa, pb)),
    "mirror_y": (lambda x, y: (x, -y), lambda a: (-a) % 360, True,
                 lambda pa, pb: (pa, -pb)),
    "rot90":    (lambda x, y: (-y, x), lambda a: (a + 90) % 360, False,
                 lambda pa, pb: (-pb, pa)),
    "rot180":   (lambda x, y: (-x, -y), lambda a: (a + 180) % 360, False,
                 lambda pa, pb: (-pa, -pb)),
    "rot270":   (lambda x, y: (y, -x), lambda a: (a + 270) % 360, False,
                 lambda pa, pb: (pb, -pa)),
}


def _hexen_style(doc: MapDocument) -> bool:
    if doc.meta.format == "hexen":
        return True
    ns = (doc.extra or {}).get("udmf_namespace") or ""
    return ns.lower() in HEXEN_UDMF_NAMESPACES


def _recompute(doc: MapDocument) -> None:
    bounds, derived = compute_derived(
        doc.vertices, doc.sectors, doc.linedefs, doc.sidedefs, doc.things,
        hexen_style=_hexen_style(doc), region_geometry=doc.derived.sector_regions or None,
        region_info=doc.derived.bsp_geometry)
    doc.bounds = bounds
    doc.derived = derived


def _mark_augmented(doc: MapDocument, label: str) -> None:
    doc.extra = dict(doc.extra or {})
    doc.extra["augmented"] = doc.extra.get("augmented", []) + [label]


def transform_document(doc: MapDocument, op: str) -> MapDocument:
    """Returns a NEW MapDocument; the input is untouched."""
    if op not in _OPS:
        raise ValueError(f"unknown augmentation op {op!r}; one of {AUGMENT_OPS}")
    point_fn, angle_fn, flips, slope_fn = _OPS[op]
    out = copy.deepcopy(doc)

    for v in out.vertices:
        v.x, v.y = point_fn(v.x, v.y)
    for t in out.things:
        t.x, t.y = point_fn(t.x, t.y)
        t.angle = angle_fn(t.angle)
    if out.derived.sector_regions:
        out.derived.sector_regions = [[[point_fn(x, y) for x, y in ring] for ring in rings]
                                      for rings in out.derived.sector_regions]
    extra = out.extra or {}
    for key, point in extra.get('sector_seed_points', {}).items():
        extra['sector_seed_points'][key] = list(point_fn(*point))
    if flips:
        for ld in out.linedefs:
            ld.v1, ld.v2 = ld.v2, ld.v1
    for s in out.sectors:
        for slope in (s.floor_slope, s.ceil_slope):
            if slope is not None:
                slope.a, slope.b = slope_fn(slope.a, slope.b)

    _mark_augmented(out, op)
    _recompute(out)
    return out


def jitter_lights(doc: MapDocument, amount: int = 16, seed: int | None = None) -> MapDocument:
    """Per-sector light-level jitter, +/- amount, clamped to 0..255."""
    if amount < 0:
        raise ValueError("amount must be non-negative")
    rng = np.random.default_rng(seed)
    out = copy.deepcopy(doc)
    for s in out.sectors:
        s.light_level = int(np.clip(s.light_level + rng.integers(-amount, amount + 1), 0, 255))
    _mark_augmented(out, "jitter_lights")
    return out


def jitter_heights(doc: MapDocument, amount: int = 16, seed: int | None = None) -> MapDocument:
    """Per-sector vertical shift in 8-unit quanta, floor and ceiling moved
    TOGETHER so room heights, door states (ceil==floor), and relative step
    feel are preserved; only inter-sector steps jitter."""
    if amount < 0:
        raise ValueError("amount must be non-negative")
    rng = np.random.default_rng(seed)
    out = copy.deepcopy(doc)
    for s in out.sectors:
        delta = int(rng.integers(-amount // 8, amount // 8 + 1)) * 8
        lo = -32768 - min(s.floor_height, s.ceil_height)
        hi = 32767 - max(s.floor_height, s.ceil_height)
        if lo <= hi:
            delta = max(lo, min(hi, delta))
        else:
            delta = 0
        s.floor_height += delta
        s.ceil_height += delta
    _mark_augmented(out, "jitter_heights")
    _recompute(out)  # reachability heuristics may care about steps
    return out

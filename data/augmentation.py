"""Geometric/photometric augmentation on MAP DOCUMENTS, not tensors -
mirror/rotate transform the raw layer (with thing-angle correction,
linedef v1/v2 + sidedef swapping, and slope-plane rotation), then the
derived layer is recomputed from scratch via compute_derived, never
transformed in place (plan/generation.md + the derived-layer purity rule).

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
        hexen_style=_hexen_style(doc))
    doc.bounds = bounds
    doc.derived = derived


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
    if flips:
        for ld in out.linedefs:
            ld.v1, ld.v2 = ld.v2, ld.v1
    for s in out.sectors:
        for slope in (s.floor_slope, s.ceil_slope):
            if slope is not None:
                slope.a, slope.b = slope_fn(slope.a, slope.b)

    out.extra = dict(out.extra)
    out.extra["augmented"] = out.extra.get("augmented", []) + [op]
    _recompute(out)
    return out


def jitter_lights(doc: MapDocument, amount: int = 16, seed: int | None = None) -> MapDocument:
    """Per-sector light-level jitter, +/- amount, clamped to 0..255."""
    rng = np.random.default_rng(seed)
    out = copy.deepcopy(doc)
    for s in out.sectors:
        s.light_level = int(np.clip(s.light_level + rng.integers(-amount, amount + 1), 0, 255))
    out.extra = dict(out.extra)
    out.extra["augmented"] = out.extra.get("augmented", []) + ["jitter_lights"]
    return out


def jitter_heights(doc: MapDocument, amount: int = 16, seed: int | None = None) -> MapDocument:
    """Per-sector vertical shift in 8-unit quanta, floor and ceiling moved
    TOGETHER so room heights, door states (ceil==floor), and relative step
    feel are preserved; only inter-sector steps jitter."""
    rng = np.random.default_rng(seed)
    out = copy.deepcopy(doc)
    for s in out.sectors:
        delta = int(rng.integers(-amount // 8, amount // 8 + 1)) * 8
        s.floor_height += delta
        s.ceil_height += delta
    out.extra = dict(out.extra)
    out.extra["augmented"] = out.extra.get("augmented", []) + ["jitter_heights"]
    _recompute(out)  # reachability heuristics may care about steps
    return out

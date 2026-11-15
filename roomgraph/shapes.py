"""Room-shape library (Phase 10.3): real room outlines harvested from real
maps so a GENERATED graph node - which only carries a size class + aspect, no
geometry - can be dressed in a real organic polygon instead of a rectangle.

build_library() segments a set of real maps and banks each room's largest
sector polygon, normalised to the origin and bucketed by aspect (tall / square
/ wide). apply_shapes() then, for each generated room, retrieves a real polygon
of the matching aspect and uniformly scales it to the node's target area - the
real shape's character is preserved while the generator still controls size.
This is what lets assemble_real turn a *generated* graph into real geometry.
"""
from __future__ import annotations

import json
import math
import random

from roomgraph.building import largest_poly, poly_centroid
from roomgraph.corridor import signed_area
from roomgraph.segment import segment

MIN_VERTS = 4            # skip degenerate rings
MAX_VERTS = 48           # skip ultra-detailed rings (they read as noise scaled down)
MIN_AREA = 2.5e4         # skip slivers


def _bbox(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


_centroid = poly_centroid
_largest = largest_poly


def _aspect(w: float, h: float) -> str:
    if h <= 0 or w <= 0:
        return "square"
    r = w / h
    return "wide" if r >= 1.4 else "tall" if r < 0.8 else "square"


def build_library(docs, *, playable_only: bool = False) -> dict:
    """Segment maps and bank normalised room polygons by aspect bucket."""
    lib: dict[str, list] = {"tall": [], "square": [], "wide": []}
    for doc in docs:
        rg = segment(doc, playable_only=playable_only)
        for r in rg.rooms:
            poly = _largest(r.polygons)
            if poly is None or not (MIN_VERTS <= len(poly) <= MAX_VERTS):
                continue
            area = abs(signed_area(poly)) - sum(abs(signed_area(hole)) for hole in r.holes)
            if area < MIN_AREA:
                continue
            cx, cy = _centroid(poly)
            norm = [(x - cx, y - cy) for x, y in poly]
            x0, y0, x1, y1 = _bbox(norm)
            lib[_aspect(x1 - x0, y1 - y0)].append(
                {"poly": norm, "area": area,
                 "holes": [[(x - cx, y - cy) for x, y in hole] for hole in r.holes]})
    return lib


def save_library(lib: dict, path) -> None:
    with open(path, "w") as f:
        json.dump(lib, f)


def load_library(path) -> dict:
    with open(path) as f:
        raw = json.load(f)
    return {k: [{"poly": [tuple(p) for p in s["poly"]], "area": s["area"],
                 "holes": [[tuple(p) for p in hole] for hole in s.get('holes', [])]}
                for s in v] for k, v in raw.items()}


def _bucket(lib: dict, aspect: str):
    if lib.get(aspect):
        return lib[aspect]
    for v in lib.values():                 # fall back to any non-empty bucket
        if v:
            return v
    return None


def retrieve(lib: dict, aspect: str, target_area: float, rng: random.Random, *, with_holes=False):
    """A real polygon of the given aspect, uniformly scaled to target_area."""
    bucket = _bucket(lib, aspect)
    if not bucket:
        return None
    shape = rng.choice(bucket)
    s = math.sqrt(max(target_area, 1.0) / max(shape["area"], 1.0))
    poly = [(x * s, y * s) for x, y in shape["poly"]]
    holes = [[(x * s, y * s) for x, y in hole] for hole in shape.get('holes', [])]
    return (poly, holes) if with_holes else poly


def apply_shapes(graph, lib: dict, seed: int = 0):
    """Give every room in a (shape-less, generated) graph a real polygon,
    updating its bbox/centroid so layout + assemble_real can run. In place."""
    rng = random.Random(seed)
    for r in graph.rooms:
        w, h = r.size
        aspect = _aspect(w, h) if (w > 0 and h > 0) else "square"
        shape = retrieve(lib, aspect, max(r.area, MIN_AREA), rng, with_holes=True)
        if shape is None:
            continue
        poly, holes = shape
        x0, y0, x1, y1 = _bbox(poly)
        r.polygons = [poly]
        r.holes = holes
        r.bbox = (x0, y0, x1, y1)
        r.centroid = _centroid(poly)
    return graph

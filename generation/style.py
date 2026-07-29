"""Corpus-statistics style model: the interim Stage 3/4 pass
(plan/generation.md) until the property GNN and thing-placement model
exist. Everything here is fitted from the extracted corpus - no
hand-picked texture lists - so generated maps only ever reference
textures/flats/things that actually occur in the training WADs.

Applied per map with spatial coherence: flats/wall textures propagate
across the sector adjacency graph with a keep-probability, so areas read
as themed regions rather than per-sector confetti.

    python -m generation.style dataset/maps/ style.json     # fit + save
"""
from __future__ import annotations

import json
from collections import Counter, deque
from copy import deepcopy
from pathlib import Path

import numpy as np

from data.graph_builder import _ring_area_perimeter
from generation.vectorizer import DEFAULT_WALL_TEX
from wad_parser.map_extractor import _point_in_polygon, _sector_containing
from wad_parser.types import (
    MapDocument, Thing, ThingFlags, resolve_doomed, resolve_linedef_special,
)

# Context-free sector light effects (blink/flicker/glow); damage floors and
# door/close types need matching flats or geometry, so they stay out.
_LIGHT_SPECIALS = {1, 2, 3, 8, 17}
_PICKUP_CATEGORIES = {"weapon", "ammo", "item"}
_KEEP_NEIGHBOR_P = 0.55
_EXIT_SPECIAL = 11  # S1 Exit (Normal)
# Keep placed things this far off any sector wall so they aren't wedged
# against it (covers most monster radii; the big ones - Mancubus 48,
# Arachnotron 64 - can still clip, but those are rare).
_THING_WALL_MARGIN = 32.0
# Classic line teleport types (incl. WR-97, which the validator uses to
# reconnect void-separated islands). Walking one lands you in the sector
# tagged with the line's tag, so reachability must follow it.
_TELEPORT_LINE_SPECIALS = {39, 97, 125, 126}
# Thing-count controls. A corpus mixing normal maps with slaughter maps
# (Alien Vendetta, Hell Revealed) has a fat high-density tail; a raw draw
# grabs a slaughter density and applies it to a large generated map. Clamp
# the drawn density to a percentile and cap the per-map totals.
_DENSITY_PERCENTILE = 65
_MAX_MONSTERS = 100
_MAX_PICKUPS = 80


def _draw_density(dist: list[float], rng: np.random.Generator) -> float:
    """A random per-map density, clamped to the corpus percentile so a
    slaughter-map outlier can't blow up the count."""
    if not dist:
        return 0.0
    cap = float(np.percentile(dist, _DENSITY_PERCENTILE))
    return min(float(rng.choice(dist)), cap)


def _wall_clearance(x: float, y: float, ring_pts: list[tuple[float, float]]) -> float:
    """Shortest distance from (x, y) to any edge of a sector's boundary ring.
    inf for a degenerate ring (nothing to clip against)."""
    n = len(ring_pts)
    if n < 2:
        return float("inf")
    best = float("inf")
    for i in range(n):
        ax, ay = ring_pts[i]
        bx, by = ring_pts[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        if seg2 == 0.0:
            d2 = (x - ax) ** 2 + (y - ay) ** 2
        else:
            t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / seg2))
            px, py = ax + t * dx, ay + t * dy
            d2 = (x - px) ** 2 + (y - py) ** 2
        best = min(best, d2)
    return best ** 0.5


class StyleModel:
    def __init__(self):
        self.flat_pairs: Counter = Counter()        # (floor, ceil) -> count
        self.wall_mids: Counter = Counter()          # one-sided mid textures
        self.switches: Counter = Counter()           # SW1*/SW2* textures
        self.monster_types: Counter = Counter()      # DoomEd -> count
        self.pickup_types: Counter = Counter()
        self.monster_density: list[float] = []       # per map, per 1e6 units^2
        self.pickup_density: list[float] = []
        self.light_special_rate: float = 0.0         # fraction of dark sectors
        self.light_specials: Counter = Counter()

    # -- fitting -------------------------------------------------------------

    def fit_document(self, doc: MapDocument) -> None:
        vpos = {v.id: (v.x, v.y) for v in doc.vertices}
        area = 0.0
        dark = with_light_special = 0
        for s in doc.sectors:
            ring = (doc.derived.sector_polygons[s.id]
                    if s.id < len(doc.derived.sector_polygons) else [])
            a, _ = _ring_area_perimeter(ring, vpos)
            area += a
            if s.floor_flat != "-" and s.ceil_flat != "-":
                self.flat_pairs[(s.floor_flat, s.ceil_flat)] += 1
            if s.light_level < 144:
                dark += 1
                if s.special in _LIGHT_SPECIALS:
                    with_light_special += 1
            if s.special in _LIGHT_SPECIALS:
                self.light_specials[s.special] += 1
        for sd in doc.sidedefs:
            if sd.mid_tex not in ("-", ""):
                self.wall_mids[sd.mid_tex] += 1
            for tex in (sd.upper_tex, sd.lower_tex, sd.mid_tex):
                if tex.startswith(("SW1", "SW2")):
                    self.switches[tex] += 1
        monsters = pickups = 0
        for t in doc.things:
            if t.category == "monster":
                self.monster_types[t.type] += 1
                monsters += 1
            elif t.category in _PICKUP_CATEGORIES:
                self.pickup_types[t.type] += 1
                pickups += 1
        if area > 0:
            self.monster_density.append(monsters / area * 1e6)
            self.pickup_density.append(pickups / area * 1e6)
        if dark:
            # running average of per-map rates
            rate = with_light_special / dark
            n = len(self.monster_density)
            self.light_special_rate += (rate - self.light_special_rate) / max(n, 1)

    @classmethod
    def fit(cls, maps_dir: Path | str) -> "StyleModel":
        model = cls()
        for p in sorted(Path(maps_dir).glob("*.json")):
            if p.name.endswith(".textures.json"):
                continue
            model.fit_document(MapDocument.from_dict(json.loads(p.read_text())))
        return model

    # -- persistence -----------------------------------------------------------

    def to_json(self) -> dict:
        return {
            "flat_pairs": [[f, c, n] for (f, c), n in self.flat_pairs.items()],
            "wall_mids": dict(self.wall_mids),
            "switches": dict(self.switches),
            "monster_types": {str(k): v for k, v in self.monster_types.items()},
            "pickup_types": {str(k): v for k, v in self.pickup_types.items()},
            "monster_density": self.monster_density,
            "pickup_density": self.pickup_density,
            "light_special_rate": self.light_special_rate,
            "light_specials": {str(k): v for k, v in self.light_specials.items()},
        }

    @classmethod
    def from_json(cls, data: dict) -> "StyleModel":
        m = cls()
        m.flat_pairs = Counter({(f, c): n for f, c, n in data["flat_pairs"]})
        m.wall_mids = Counter(data["wall_mids"])
        m.switches = Counter(data["switches"])
        m.monster_types = Counter({int(k): v for k, v in data["monster_types"].items()})
        m.pickup_types = Counter({int(k): v for k, v in data["pickup_types"].items()})
        m.monster_density = data["monster_density"]
        m.pickup_density = data["pickup_density"]
        m.light_special_rate = data["light_special_rate"]
        m.light_specials = Counter({int(k): v for k, v in data["light_specials"].items()})
        return m

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "StyleModel":
        return cls.from_json(json.loads(Path(path).read_text()))


_BLUR_KERNEL = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0


def _blur(grid: np.ndarray) -> np.ndarray:
    """Separable binomial blur (sigma ~1px): real maps store things as
    single-pixel spikes in channel 8, and sampling should mean 'near
    here', not 'this exact pixel'."""
    out = grid
    for axis in (0, 1):
        out = np.apply_along_axis(
            lambda m: np.convolve(m, _BLUR_KERNEL, mode="same"), axis, out)
    return out


class PlacementDensity:
    """Raster channel 8 (thing density) plus the raster->world transform
    the vectorizer used, so placement can importance-sample WHERE the
    layout model (or the source map, in the regen harness) put things
    instead of scattering them area-weighted per sector."""

    def __init__(self, grid: np.ndarray, units_per_px: float, world_height: float):
        self.grid = _blur(np.maximum(np.asarray(grid, dtype=np.float64), 0.0))
        self.units_per_px = units_per_px
        self.world_height = world_height
        total = self.grid.sum()
        self._flat_p = self.grid.ravel() / total if total > 1e-9 else None

    @classmethod
    def from_raster(cls, raster: np.ndarray, cell_units: int,
                    coarse_factor: int) -> "PlacementDensity":
        """Must be called with the same cell_units/coarse_factor the raster
        was vectorized with - that pair defines the world coordinates of
        the output document."""
        res_h, res_w = raster.shape[1], raster.shape[2]
        grid_h, grid_w = res_h // coarse_factor, res_w // coarse_factor
        # crop exactly like the vectorizer's _downsample does
        grid = raster[8][: grid_h * coarse_factor, : grid_w * coarse_factor]
        units_per_px = cell_units / coarse_factor
        return cls(grid, units_per_px, grid_h * cell_units)

    @property
    def has_mass(self) -> bool:
        return self._flat_p is not None

    def sample_point(self, rng: np.random.Generator) -> tuple[float, float]:
        idx = int(rng.choice(self.grid.size, p=self._flat_p))
        pr, pc = divmod(idx, self.grid.shape[1])
        x = (pc + rng.random()) * self.units_per_px
        y = self.world_height - (pr + rng.random()) * self.units_per_px
        return x, y


def _sample(counter: Counter, rng: np.random.Generator):
    items = list(counter)
    weights = np.array([counter[i] for i in items], dtype=np.float64)
    return items[int(rng.choice(len(items), p=weights / weights.sum()))]


def _random_point_in_sector(doc: MapDocument, sector_id: int,
                            rng: np.random.Generator) -> tuple[float, float] | None:
    vpos = {v.id: (v.x, v.y) for v in doc.vertices}
    ring = (doc.derived.sector_polygons[sector_id]
            if sector_id < len(doc.derived.sector_polygons) else [])
    pts = [vpos[v] for v in ring if v in vpos]
    if len(pts) < 3:
        seed = doc.extra.get("sector_seed_points", {}).get(str(sector_id))
        return tuple(seed) if seed else None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    for _ in range(40):
        x = rng.uniform(min(xs), max(xs))
        y = rng.uniform(min(ys), max(ys))
        if _point_in_polygon(x, y, pts):
            return x, y
    seed = doc.extra.get("sector_seed_points", {}).get(str(sector_id))
    return tuple(seed) if seed else None


def _add_thing(doc: MapDocument, x: float, y: float, type_num: int,
               angle: int = 0) -> Thing:
    name, category, unresolved = resolve_doomed(type_num)
    t = Thing(id=len(doc.things), x=x, y=y, angle=angle, type=type_num,
              type_name=name, category=category,
              flags=ThingFlags(skill_1_2=True, skill_3=True, skill_4_5=True),
              unresolved=unresolved)
    doc.things.append(t)
    return t


def _make_transforms(meta: dict, cell_units: int, coarse_factor: int):
    """Forward/inverse maps between SOURCE world coords and the REGENERATED
    map's world coords. The regen map is rebuilt on a fresh origin-anchored
    lattice, so the two frames differ by the rasterizer's center/scale plus
    the vectorizer's cell size and row flip - this composes both."""
    scale, cx, cy = meta["scale"], meta["center_x"], meta["center_y"]
    res = meta["resolution"]
    upp = cell_units / coarse_factor          # regen world units per raster pixel

    def fwd(x, y):                            # source world -> regen world
        col = (x - cx) * scale + res / 2
        row = res / 2 - (y - cy) * scale
        return col * upp, (res - row) * upp

    def inv(rx, ry):                          # regen world -> source world
        col, row = rx / upp, res - ry / upp
        return cx + (col - res / 2) / scale, cy + (res / 2 - row) / scale

    return fwd, inv


def transfer_source_attributes(doc: MapDocument, source: MapDocument, meta: dict,
                               cell_units: int, coarse_factor: int) -> dict:
    """Regen-harness decoration: carry the SOURCE map's real flats, wall
    textures, sector lighting/specials and things onto the regenerated
    skeleton, instead of sampling a corpus.

    The raster round-trip discards textures and thing *types* entirely (the
    raster holds only geometry, a density channel, and a light channel), but
    we still hold the source document - so a faithful transfer beats invented
    randomness. Any remaining deviation is then attributable to the geometry
    vectorization, which is exactly what the harness is meant to measure.

    Must be called with the same meta / cell_units / coarse_factor used to
    rasterize and vectorize, and BEFORE validate_and_repair, so reachability
    is computed from the real player start."""
    fwd, inv = _make_transforms(meta, cell_units, coarse_factor)
    summary: dict = {"mode": "faithful"}

    src_polys = source.derived.sector_polygons
    src_vpos = {v.id: (v.x, v.y) for v in source.vertices}

    # representative (most common) wall texture per source sector
    tex_counts: dict[int, Counter] = {s.id: Counter() for s in source.sectors}
    for sd in source.sidedefs:
        if sd.sector in tex_counts:
            for tex in (sd.mid_tex, sd.upper_tex, sd.lower_tex):
                if tex not in ("-", ""):
                    tex_counts[sd.sector][tex] += 1
    src_wall = {sid: (c.most_common(1)[0][0] if c else DEFAULT_WALL_TEX)
                for sid, c in tex_counts.items()}

    src_centroids: dict[int, tuple[float, float]] = {}
    for s in source.sectors:
        ring = src_polys[s.id] if s.id < len(src_polys) else []
        pts = [src_vpos[v] for v in ring if v in src_vpos]
        if pts:
            src_centroids[s.id] = (sum(p[0] for p in pts) / len(pts),
                                   sum(p[1] for p in pts) / len(pts))

    def nearest_src_sector(sx: float, sy: float):
        sec = _sector_containing(sx, sy, source.sectors, src_polys, source.vertices)
        if sec is not None:
            return sec
        best, best_d = None, None
        for sid, (ccx, ccy) in src_centroids.items():
            d = (ccx - sx) ** 2 + (ccy - sy) ** 2
            if best_d is None or d < best_d:
                best, best_d = sid, d
        return best

    # --- per regen sector: find the source sector beneath it, copy its look ---
    regen_vpos = {v.id: (v.x, v.y) for v in doc.vertices}
    regen_to_src: dict[int, int] = {}
    for s in doc.sectors:
        ring = (doc.derived.sector_polygons[s.id]
                if s.id < len(doc.derived.sector_polygons) else [])
        pts = [regen_vpos[v] for v in ring if v in regen_vpos]
        if not pts:
            continue
        rc = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
        src_id = nearest_src_sector(*inv(*rc))
        if src_id is None:
            continue
        regen_to_src[s.id] = src_id
        src_sec = source.sectors[src_id]
        s.floor_flat, s.ceil_flat = src_sec.floor_flat, src_sec.ceil_flat
        s.light_level = src_sec.light_level
        s.special = src_sec.special      # sector specials are standalone effects
    for sd in doc.sidedefs:
        wall = src_wall.get(regen_to_src.get(sd.sector), DEFAULT_WALL_TEX)
        if sd.mid_tex not in ("-", ""):
            sd.mid_tex = wall
        if sd.upper_tex not in ("-", ""):
            sd.upper_tex = wall
        if sd.lower_tex not in ("-", ""):
            sd.lower_tex = wall
    summary["sectors_matched"] = len(regen_to_src)

    # --- things: the source's real things, in the source's positions ---
    doc.things = []
    kept = dropped = 0
    for t in source.things:
        rx, ry = fwd(t.x, t.y)
        if _sector_containing(rx, ry, doc.sectors,
                              doc.derived.sector_polygons, doc.vertices) is None:
            dropped += 1
            continue
        nt = deepcopy(t)
        nt.id = len(doc.things)
        nt.x, nt.y = int(round(rx)), int(round(ry))
        doc.things.append(nt)
        kept += 1
    summary["things_kept"] = kept
    summary["things_dropped"] = dropped
    return summary


def apply_style(doc: MapDocument, style: StyleModel,
                seed: int | None = None,
                density: PlacementDensity | None = None,
                place_things: bool = True) -> dict:
    """Textures, flats, light specials, monsters/pickups, DM starts, and an
    exit switch - mutates doc, returns a summary dict.

    When a PlacementDensity is given (and has mass), monster/pickup
    positions are importance-sampled from it; otherwise sectors are chosen
    area-weighted and positions uniformly within them.

    place_things=False skips the heuristic monster/pickup placement (the
    Stage 4 learned placer takes over that job) but still does flats, light,
    the exit switch and DM starts."""
    rng = np.random.default_rng(seed)
    summary: dict = {}
    vpos = {v.id: (v.x, v.y) for v in doc.vertices}

    neighbors: dict[int, set[int]] = {s.id: set() for s in doc.sectors}
    for e in doc.derived.sector_graph.get("edges", []):
        neighbors[e["a"]].add(e["b"])
        neighbors[e["b"]].add(e["a"])
    # Teleporter links: a teleport line connects its front sector to the
    # sector tagged with its tag (where the destination sits). Without this,
    # islands the validator reconnected by teleporter look unreachable.
    tag_to_sector: dict[int, int] = {}
    for s in doc.sectors:
        if s.tag:
            tag_to_sector.setdefault(s.tag, s.id)
    for ld in doc.linedefs:
        if ld.special in _TELEPORT_LINE_SPECIALS and ld.tag \
                and 0 <= ld.right_sidedef < len(doc.sidedefs):
            front = doc.sidedefs[ld.right_sidedef].sector
            dest = tag_to_sector.get(ld.tag)
            if dest is not None and front in neighbors and dest in neighbors:
                neighbors[front].add(dest)
                neighbors[dest].add(front)

    # --- flats + wall textures with adjacency coherence ---
    if style.flat_pairs:
        assigned_flats: dict[int, tuple[str, str]] = {}
        assigned_walls: dict[int, str] = {}
        palette = [
            _sample(style.wall_mids, rng) for _ in range(min(3, len(style.wall_mids)))
        ] or ["STARTAN2"]
        for s in doc.sectors:
            if s.id in assigned_flats:
                continue
            assigned_flats[s.id] = _sample(style.flat_pairs, rng)
            assigned_walls[s.id] = palette[int(rng.integers(len(palette)))]
            q = deque([s.id])
            while q:
                cur = q.popleft()
                for nb in neighbors[cur]:
                    if nb in assigned_flats:
                        continue
                    keep = rng.random() < _KEEP_NEIGHBOR_P
                    assigned_flats[nb] = (assigned_flats[cur] if keep
                                          else _sample(style.flat_pairs, rng))
                    assigned_walls[nb] = (assigned_walls[cur] if keep
                                          else palette[int(rng.integers(len(palette)))])
                    q.append(nb)
        for s in doc.sectors:
            s.floor_flat, s.ceil_flat = assigned_flats[s.id]
        for sd in doc.sidedefs:
            wall = assigned_walls.get(sd.sector, palette[0])
            if sd.mid_tex not in ("-", ""):
                sd.mid_tex = wall
            if sd.upper_tex not in ("-", ""):
                sd.upper_tex = wall
            if sd.lower_tex not in ("-", ""):
                sd.lower_tex = wall
        summary["wall_palette"] = palette

    # --- light-effect sector specials ---
    flicker = 0
    for s in doc.sectors:
        if s.special == 0 and s.light_level < 144 and style.light_specials \
                and rng.random() < style.light_special_rate:
            s.special = _sample(style.light_specials, rng)
            flicker += 1
    summary["light_specials"] = flicker

    # --- start sector + graph distances ---
    start = next((t for t in doc.things if t.type == 1), None)
    start_sector = None
    if start is not None:
        start_sector = _sector_containing(start.x, start.y, doc.sectors,
                                          doc.derived.sector_polygons, doc.vertices)
        if start_sector is None:
            # The inserted start can sit a hair outside its polygon (seed-point
            # rounding), which would leave `dist` empty and silently disable the
            # reachability gate. Anchor to the nearest sector centroid instead.
            best_d = None
            for s in doc.sectors:
                ring = (doc.derived.sector_polygons[s.id]
                        if s.id < len(doc.derived.sector_polygons) else [])
                pts = [vpos[v] for v in ring if v in vpos]
                if len(pts) < 3:
                    continue
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                d = (cx - start.x) ** 2 + (cy - start.y) ** 2
                if best_d is None or d < best_d:
                    best_d, start_sector = d, s.id
    dist = {start_sector: 0} if start_sector is not None else {}
    q = deque(dist)
    while q:
        cur = q.popleft()
        for nb in neighbors[cur]:
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                q.append(nb)

    # --- exit switch in the farthest reachable sector ---
    if dist:
        exit_sector = max(dist, key=dist.get)
        candidates = [ld for ld in doc.linedefs
                      if ld.special == 0 and ld.left_sidedef == -1
                      and ld.right_sidedef < len(doc.sidedefs)
                      and doc.sidedefs[ld.right_sidedef].sector == exit_sector]
        if candidates:
            line = candidates[int(rng.integers(len(candidates)))]
            line.special = _EXIT_SPECIAL
            line.special_name, line.trigger_type = resolve_linedef_special(_EXIT_SPECIAL)
            if style.switches:
                doc.sidedefs[line.right_sidedef].mid_tex = _sample(style.switches, rng)
            summary["exit_sector"] = int(exit_sector)

    # --- thing placement ---
    # Big enough to fight in AND tall enough to enter.
    areas: dict[int, float] = {}
    rings: dict[int, list[tuple[float, float]]] = {}
    for s in doc.sectors:
        ring = (doc.derived.sector_polygons[s.id]
                if s.id < len(doc.derived.sector_polygons) else [])
        pts = [vpos[v] for v in ring if v in vpos]
        a, _ = _ring_area_perimeter(ring, vpos)
        if a >= 64 * 64 and (s.ceil_height - s.floor_height) >= 56:
            areas[s.id] = a
            rings[s.id] = pts

    # Only populate sectors reachable from the player start (dist, computed
    # above, following walls AND teleporters). Things in disconnected islands
    # would be unfightable / uncollectable. Fall back to all big sectors ONLY
    # when there is no start at all - never place into known-unreachable ones.
    place_ids = [i for i in areas if i in dist] if dist else list(areas)
    place_set = set(place_ids)
    place_area = sum(areas[i] for i in place_ids)
    use_density = density is not None and density.has_mass

    def _place_point(min_start_dist: float, weights):
        """A point inside a reachable sector, clear of walls and the start."""
        for _attempt in range(12):
            if use_density:
                cand = density.sample_point(rng)
                sec = _sector_containing(cand[0], cand[1], doc.sectors,
                                         doc.derived.sector_polygons, doc.vertices)
                if sec not in place_set:
                    continue
            else:
                sec = place_ids[int(rng.choice(len(place_ids), p=weights))]
                cand = _random_point_in_sector(doc, sec, rng)
                if cand is None:
                    continue
            if _wall_clearance(cand[0], cand[1], rings.get(sec, [])) < _THING_WALL_MARGIN:
                continue
            if start is not None and (cand[0] - start.x) ** 2 + (cand[1] - start.y) ** 2 \
                    < min_start_dist ** 2:
                continue
            return cand
        return None

    def place(count: int, types: Counter, min_start_dist: float) -> int:
        if not place_ids or not types:
            return 0
        weights = np.array([areas[i] for i in place_ids], dtype=float)
        weights /= weights.sum()
        placed = 0
        for _ in range(count):
            pt = _place_point(min_start_dist, weights)
            if pt is None:
                continue
            _add_thing(doc, pt[0], pt[1], _sample(types, rng),
                       angle=int(rng.integers(0, 8)) * 45)
            placed += 1
        return placed

    summary["placement"] = "density" if use_density else "area"
    # Budget on the reachable area only, so cutting the islands doesn't
    # cram their share of monsters into the reachable part.
    if place_things and place_area > 0:
        m_density = _draw_density(style.monster_density, rng)
        p_density = _draw_density(style.pickup_density, rng)
        n_monsters = int(np.clip(round(m_density * place_area / 1e6), 0, _MAX_MONSTERS))
        n_pickups = int(np.clip(round(p_density * place_area / 1e6), 0, _MAX_PICKUPS))
        summary["monsters"] = place(n_monsters, style.monster_types, min_start_dist=256)
        summary["pickups"] = place(n_pickups, style.pickup_types, min_start_dist=0)

    # --- deathmatch starts in the four largest sectors (reachability not
    # required - DM spawns are teleport-in - but still keep them off walls) ---
    dm = 0
    for sector_id in sorted(areas, key=areas.get, reverse=True)[:4]:
        for _attempt in range(12):
            pt = _random_point_in_sector(doc, sector_id, rng)
            if pt and _wall_clearance(pt[0], pt[1], rings.get(sector_id, [])) >= _THING_WALL_MARGIN:
                _add_thing(doc, pt[0], pt[1], 11, angle=int(rng.integers(0, 8)) * 45)
                dm += 1
                break
    summary["dm_starts"] = dm
    return summary


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Fit a style model from an extracted corpus.")
    ap.add_argument("maps_dir")
    ap.add_argument("out_json")
    args = ap.parse_args()
    model = StyleModel.fit(args.maps_dir)
    model.save(args.out_json)
    print(f"fitted style from {args.maps_dir}: "
          f"{len(model.flat_pairs)} flat pairs, {len(model.wall_mids)} wall textures, "
          f"{len(model.monster_types)} monster types, {len(model.pickup_types)} pickup types")


if __name__ == "__main__":
    main()

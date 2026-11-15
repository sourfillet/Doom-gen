"""Corpus-statistics style model: the Stage 3/4 pass used when no property
or placement GNN checkpoint is supplied. Everything here is fitted from the extracted corpus - no
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

from generation.actors import actor_profile, placement_margin, spawn_problem
from generation.vectorizer import DEFAULT_WALL_TEX
from wad_parser.geometry import (
    edge_clearance, interior_point, point_in_rings, sector_area_perimeter, sector_geometry, sector_points,
)
from wad_parser.map_extractor import _sector_containing
from wad_parser.traversal import distances_from_start, movement_neighbors
from wad_parser.types import (
    MapDocument, Thing, ThingFlags, active_single_player, resolve_doomed,
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
    if ring_pts and isinstance(ring_pts[0][0], (list, tuple)):
        return min((_wall_clearance(x, y, ring) for ring in ring_pts), default=0.0)
    return edge_clearance(x, y, list(ring_pts)) if len(ring_pts) >= 2 else float("inf")


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
        self._dark_maps: int = 0
        self.light_specials: Counter = Counter()
        self.light_levels: Counter = Counter()
        self.door_walls: Counter = Counter()

    # -- fitting -------------------------------------------------------------

    def fit_document(self, doc: MapDocument) -> None:
        area = 0.0
        dark = with_light_special = 0
        for s in doc.sectors:
            self.light_levels[s.light_level] += 1
            a, _ = sector_area_perimeter(doc, s.id)
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
            for tex in (sd.upper_tex, sd.lower_tex, sd.mid_tex):
                if tex.startswith(("SW1", "SW2")):
                    self.switches[tex] += 1
        from wad_parser.actions import LOCAL_DOORS
        for line in doc.linedefs:
            if line.left_sidedef == -1 and 0 <= line.right_sidedef < len(doc.sidedefs):
                texture = doc.sidedefs[line.right_sidedef].mid_tex
                if texture not in {'-', ''} and not texture.startswith(('SW1', 'SW2')):
                    self.wall_mids[texture] += 1
            if line.special in LOCAL_DOORS and 0 <= line.right_sidedef < len(doc.sidedefs):
                texture = doc.sidedefs[line.right_sidedef].upper_tex
                if texture not in {'-', ''}:
                    self.door_walls[texture] += 1
        monsters = pickups = 0
        for t in doc.things:
            if not active_single_player(t):
                continue
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
            # running average of per-map rates over maps that actually have dark sectors
            rate = with_light_special / dark
            self._dark_maps += 1
            self.light_special_rate += (rate - self.light_special_rate) / self._dark_maps

    @classmethod
    def fit(cls, maps_dir: Path | str) -> "StyleModel":
        model = cls()
        for p in sorted(Path(maps_dir).glob("*.json")):
            if p.name.endswith(".textures.json"):
                continue
            model.fit_document(MapDocument.from_dict(json.loads(p.read_text(encoding="utf-8"))))
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
            "dark_maps": self._dark_maps,
            "light_specials": {str(k): v for k, v in self.light_specials.items()},
            "light_levels": {str(k): v for k, v in self.light_levels.items()},
            "door_walls": dict(self.door_walls),
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
        m._dark_maps = int(data.get("dark_maps", len(m.monster_density)))
        m.light_specials = Counter({int(k): v for k, v in data["light_specials"].items()})
        m.light_levels = Counter({int(k): v for k, v in data.get('light_levels', {}).items()})
        m.door_walls = Counter(data.get('door_walls', {}))
        return m

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "StyleModel":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


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
    items = [item for item, count in counter.items() if count > 0]
    if not items:
        raise ValueError("cannot sample from an empty or non-positive Counter")
    weights = np.array([counter[i] for i in items], dtype=np.float64)
    return items[int(rng.choice(len(items), p=weights / weights.sum()))]


def _random_point_in_sector(doc: MapDocument, sector_id: int,
                            rng: np.random.Generator) -> tuple[float, float] | None:
    rings = sector_points(doc, sector_id)
    points = [point for ring in rings for point in ring]
    if not points:
        return None
    xs, ys = zip(*points)
    for _ in range(80):
        x, y = round(rng.uniform(min(xs), max(xs))), round(rng.uniform(min(ys), max(ys)))
        if point_in_rings(x, y, rings):
            return x, y
    return interior_point(doc, sector_id)


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

    src_polys = sector_geometry(source)

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
        pts = [point for ring in sector_points(source, s.id) for point in ring]
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
        rc = interior_point(doc, s.id)
        if rc is None:
            continue
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
    regen_polys = sector_geometry(doc)
    kept = dropped = 0
    for t in source.things:
        rx, ry = fwd(t.x, t.y)
        if _sector_containing(rx, ry, doc.sectors, regen_polys, doc.vertices) is None:
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
                place_things: bool = True, place_dm_starts: bool = True) -> dict:
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

    neighbors = movement_neighbors(doc.vertices, doc.sectors, doc.linedefs, doc.sidedefs)
    light_values = {level: count for level, count in style.light_levels.items()
                    if isinstance(level, int) and 0 <= level <= 255 and count > 0}
    assigned_lights = {}
    for sector in doc.sectors:
        if not light_values or sector.id in assigned_lights:
            continue
        assigned_lights[sector.id] = int(_sample(light_values, rng))
        queue = deque([sector.id])
        while queue:
            current = queue.popleft()
            for neighbor in sorted(neighbors[current]):
                if neighbor not in assigned_lights:
                    assigned_lights[neighbor] = (assigned_lights[current] if rng.random() < _KEEP_NEIGHBOR_P
                                                 else int(_sample(light_values, rng)))
                    queue.append(neighbor)
    for sector in doc.sectors:
        if sector.id in assigned_lights:
            sector.light_level = assigned_lights[sector.id]
    summary['light_levels'] = dict(Counter(sector.light_level for sector in doc.sectors))

    # --- flats + wall textures with adjacency coherence ---
    pairs = [pair for pair, count in style.flat_pairs.items() if count > 0]
    if pairs:
        assigned_flats: dict[int, tuple[str, str]] = {}
        assigned_walls: dict[int, str] = {}
        flat_weights = np.array([style.flat_pairs[pair] for pair in pairs], dtype=float)
        selected = rng.choice(len(pairs), size=min(3, len(pairs)), replace=False,
                              p=flat_weights / flat_weights.sum())
        flat_palette = Counter({pairs[int(index)]: style.flat_pairs[pairs[int(index)]] for index in selected})
        materials = [name for name, count in style.wall_mids.items() if count > 0]
        weights = np.array([style.wall_mids[name] for name in materials], dtype=float)
        palette = (rng.choice(materials, size=min(3, len(materials)), replace=False,
                              p=weights / weights.sum()).tolist() if materials else ['STARTAN2'])
        for s in doc.sectors:
            if s.id in assigned_flats:
                continue
            assigned_flats[s.id] = _sample(flat_palette, rng)
            assigned_walls[s.id] = palette[int(rng.integers(len(palette)))]
            q = deque([s.id])
            while q:
                cur = q.popleft()
                for nb in sorted(neighbors[cur]):
                    if nb in assigned_flats:
                        continue
                    keep = rng.random() < _KEEP_NEIGHBOR_P
                    assigned_flats[nb] = (assigned_flats[cur] if keep
                                          else _sample(flat_palette, rng))
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
        summary['flat_palette'] = list(flat_palette)
    positive_door_walls = Counter({k: v for k, v in style.door_walls.items() if v > 0})
    if positive_door_walls:
        from wad_parser.actions import LOCAL_DOORS
        door_texture = _sample(positive_door_walls, rng)
        for line in doc.linedefs:
            if line.special in LOCAL_DOORS and 0 <= line.right_sidedef < len(doc.sidedefs):
                doc.sidedefs[line.right_sidedef].upper_tex = door_texture
        summary['door_texture'] = door_texture

    # --- light-effect sector specials ---
    positive_light_specials = Counter({k: v for k, v in style.light_specials.items() if v > 0})
    flicker = 0
    for s in doc.sectors:
        if s.special == 0 and s.light_level < 144 and positive_light_specials \
                and rng.random() < style.light_special_rate:
            s.special = _sample(positive_light_specials, rng)
            flicker += 1
    summary["light_specials"] = flicker

    # Use the same directed movement model as validation and learned placement.
    start = next((t for t in doc.things if t.type == 1), None)
    dist = distances_from_start(doc)
    from wad_parser.traversal import component_distances
    reachable_components = component_distances(doc)

    # --- exit switch on a reachable usable wall ---
    from generation.progression import EXIT_SPECIALS, ensure_exit
    if dist and not any(line.special in EXIT_SPECIALS for line in doc.linedefs):
        try:
            if ensure_exit(doc):
                exit_line = next(ld for ld in doc.linedefs if ld.special == _EXIT_SPECIAL)
                summary["exit_sector"] = int(doc.sidedefs[exit_line.right_sidedef].sector)
        except ValueError:
            pass
    # Keep existing objectives visible without adding an easier extra exit.
    positive_switches = Counter({k: v for k, v in style.switches.items() if v > 0})
    for line in doc.linedefs:
        if line.special in {11, 51} and positive_switches and 0 <= line.right_sidedef < len(doc.sidedefs):
            doc.sidedefs[line.right_sidedef].mid_tex = _sample(positive_switches, rng)

    # --- thing placement ---
    # Big enough to fight in AND tall enough to enter.
    areas: dict[int, float] = {}
    rings: dict[int, list[tuple[float, float]]] = {}
    for s in doc.sectors:
        pts = sector_points(doc, s.id)
        a, _ = sector_area_perimeter(doc, s.id)
        if a >= 64 * 64 and (s.ceil_height - s.floor_height) >= 56:
            areas[s.id] = a
            rings[s.id] = pts

    # Only populate sectors reachable from the player start (dist, computed
    # above, following walls AND teleporters). Things in disconnected islands
    # would be unfightable / uncollectable. Fall back to all big sectors ONLY
    # when there is no start at all - never place into known-unreachable ones.
    place_ids = [i for i in areas if i in dist]
    place_set = set(place_ids)
    place_area = sum(areas[i] for i in place_ids)
    use_density = density is not None and density.has_mass
    from wad_parser.clearance import MapClearance
    geometry = MapClearance(doc.vertices, doc.sectors, doc.linedefs, doc.sidedefs)
    doc_polys = sector_geometry(doc)

    def _place_point(min_start_dist: float, weights, type_num):
        """A point inside a reachable sector, clear of walls and the start."""
        for _attempt in range(12):
            if use_density:
                cand = density.sample_point(rng)
                sec = _sector_containing(cand[0], cand[1], doc.sectors, doc_polys, doc.vertices)
                if sec not in place_set:
                    continue
            else:
                sec = place_ids[int(rng.choice(len(place_ids), p=weights))]
                cand = _random_point_in_sector(doc, sec, rng)
                if cand is None:
                    continue
            cand = (round(cand[0]), round(cand[1]))
            if spawn_problem(doc, type_num, *cand, margin=placement_margin(type_num),
                             reachable=reachable_components, geometry=geometry):
                continue
            if start is not None and (cand[0] - start.x) ** 2 + (cand[1] - start.y) ** 2 \
                    < min_start_dist ** 2:
                continue
            return cand
        return None

    def place(count: int, types: Counter, min_start_dist: float) -> int:
        valid_types = Counter({t: c for t, c in types.items() if c > 0 and actor_profile(t) is not None})
        if not place_ids or not valid_types:
            return 0
        weights = np.array([areas[i] for i in place_ids], dtype=float)
        weights /= weights.sum()
        placed = 0
        for _ in range(count):
            type_num = _sample(valid_types, rng)
            pt = _place_point(min_start_dist, weights, type_num)
            if pt is None:
                continue
            _add_thing(doc, pt[0], pt[1], type_num,
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
        summary["requested_monsters"] = n_monsters
        summary["requested_pickups"] = n_pickups
        summary["monsters"] = place(n_monsters, style.monster_types, min_start_dist=256)
        summary["pickups"] = place(n_pickups, style.pickup_types, min_start_dist=0)

    # --- deathmatch starts in the four largest sectors (reachability not
    # required - DM spawns are teleport-in - but still keep them off walls/actors) ---
    dm = 0
    for sector_id in (sorted(areas, key=areas.get, reverse=True)[:4] if place_dm_starts else []):
        for _attempt in range(12):
            pt = _random_point_in_sector(doc, sector_id, rng)
            if pt is None:
                continue
            pt = (round(pt[0]), round(pt[1]))
            if (_wall_clearance(pt[0], pt[1], rings.get(sector_id, [])) >= _THING_WALL_MARGIN
                    and spawn_problem(doc, 11, pt[0], pt[1], margin=16, geometry=geometry) is None):
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

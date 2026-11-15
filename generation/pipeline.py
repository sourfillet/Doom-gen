"""End-to-end generation: sample rasters -> vectorize -> validate ->
write a playable PWAD (Stage 1 -> 2 -> validate -> write). Stage 3/4
decoration uses the property/placement GNNs when checkpoints are given,
otherwise the corpus-statistics style pass.

Also provides regenerate_from_json(): re-runs the raster->vector->WAD
back half on an EXISTING extracted map. That path is the pipeline's test
harness, not generation - it isolates vectorizer/writer bugs from model
bugs, because its input raster is known-good.
"""
from __future__ import annotations

import copy
import json
import math
import shutil
import tempfile
from pathlib import Path

from data.rasterizer import rasterize
from generation.game_profile import GameProfile
from generation.placement import PlacementModel, apply_learned_placement
from generation.properties import PropertyModel, apply_properties
from generation.progression import ensure_exit
from generation.style import (
    PlacementDensity, StyleModel, apply_style, transfer_source_attributes,
)
from generation.validator import validate_and_repair, require_valid
from generation.vectorizer import vectorize
from wad_parser.map_extractor import compute_derived
from wad_parser.types import MapDocument
from wad_writer.assets import AssetBundle
from wad_writer.wad_writer import build_nodes, write_wad, write_lumps
from wad_writer.assembler import fit_coordinates


def load_style(path_or_dir: Path | str | None) -> StyleModel | None:
    """A fitted style .json, or a maps directory to fit from on the fly."""
    if path_or_dir is None:
        return None
    p = Path(path_or_dir)
    return StyleModel.fit(p) if p.is_dir() else StyleModel.load(p)


def raster_to_document(raster, map_name: str = "MAP01",
                       cell_units: float = 64, coarse_factor: int = 2,
                       style: StyleModel | None = None,
                       style_seed: int | None = None,
                       source: MapDocument | None = None,
                       meta: dict | None = None,
                       game: GameProfile | None = None,
                       properties: PropertyModel | None = None,
                       placement: PlacementModel | None = None,
                       tags: dict | None = None, raster_version: int = 5,
                       asset_resolution=None) -> tuple[MapDocument, list[str], dict]:
    doc = vectorize(raster, cell_units=cell_units, coarse_factor=coarse_factor,
                    map_name=map_name, raster_version=raster_version)
    eff_factor = doc.extra.get("raster_decoding", {}).get("effective_factor", coarse_factor)
    eff_cell_units = cell_units * eff_factor / coarse_factor
    if source is not None:
        # Regen harness: we still have the real map, so carry its actual
        # textures/things across rather than inventing them. Runs BEFORE
        # validation so reachability starts from the real player start.
        decoration = transfer_source_attributes(doc, source, meta, eff_cell_units,
                                                 eff_factor)
        repairs = validate_and_repair(doc, repair_heights=False)
    else:
        repairs = validate_and_repair(doc)
        decoration = {}
        if style:
            # channel 8 tells us WHERE things go; the style model decides how
            # many and which kinds (unless Stage 4 owns thing placement)
            density = PlacementDensity.from_raster(raster, cell_units=eff_cell_units,
                                                   coarse_factor=eff_factor)
            decoration = apply_style(doc, style, seed=style_seed, density=density,
                                     place_things=placement is None)
    if placement is not None:
        # Stage 4: learned per-sector counts + types, placed by the geometric
        # placer. Runs after style (which set flats/exit/DM) so the map has a
        # player start to anchor reachability + keep-away on.
        decoration["placement"] = apply_learned_placement(
            doc, placement, tags or {}, seed=style_seed, game=game)
    if properties is not None:
        # Stage 3: overwrite the heuristic flats/textures/specials with the
        # GNN's context-aware, tag-conditioned predictions.
        decoration["properties"] = apply_properties(
            doc, properties, tags or {}, game=game, seed=style_seed)
    if game is not None:
        if asset_resolution is not None:
            asset_resolution.resolve(doc)
        # Last: guarantee every asset the map references exists in the target
        # game (or a bundled PWAD asset registered on the profile).
        decoration["game_enforcement"] = game.enforce(doc)
    # Gameplay essentials must not depend on optional texture/placement models.
    # Source reconstruction retains authored progression for diagnosis.
    if source is None:
        exit_repair = ensure_exit(doc)
        if exit_repair:
            repairs.append(exit_repair)
    doc.bounds, doc.derived = compute_derived(
        doc.vertices, doc.sectors, doc.linedefs, doc.sidedefs, doc.things)
    translation = fit_coordinates(doc)
    if translation:
        repairs.append(translation)
    if source is None:
        decoration["validation"] = require_valid(doc, require_connected=True)
    return doc, repairs, decoration


def documents_to_wad(docs: list[MapDocument], out_path: Path | str,
                     try_nodes: bool = True,
                     assets: AssetBundle | None = None,
                     require_playable: bool = False, target_port: str = 'zdoom',
                     companion_textures: Path | str | None = None) -> dict:
    if companion_textures is not None:
        return _publish_with_resources(out_path, companion_textures, assets, target_port,
            lambda path, bundle: documents_to_wad(docs, path, try_nodes=try_nodes,
                assets=bundle, require_playable=require_playable, target_port=target_port))
    from wad_writer.compatibility import TARGET_PORTS
    if target_port not in TARGET_PORTS:
        raise ValueError('target_port must be zdoom or vanilla')
    if assets is not None:
        assets.require_target(target_port)
    if require_playable:
        for doc in docs:
            require_valid(doc, require_connected=True)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent, prefix='.generation-') as directory:
        candidate = write_wad(docs, Path(directory) / 'candidate.wad', assets=assets)
        nodes_built = build_nodes(candidate) if try_nodes else False
        names = [doc.meta.map_name.upper() for doc in docs]
        compatibility = _check_wad(candidate, names, target_port, require_playable,
                                   names if nodes_built else [])
        candidate.replace(path)
    return {"path": str(path), "maps": [d.meta.map_name for d in docs],
            "nodes_built": nodes_built, 'target_port': target_port,
            'compatibility': compatibility}


def _publish_with_resources(out_path, companion_path, assets, target_port, publish):
    """Validate both outputs before replacing either destination.

    Each rename is atomic. If replacing the map fails, restore the companion;
    a process/power failure between two renames cannot be made atomic across
    independent files, so this is not a filesystem transaction.
    """
    path, companion = Path(out_path), Path(companion_path)
    if path.resolve() == companion.resolve():
        raise ValueError('companion textures and map WAD output must use different paths')
    if assets is None or assets.is_empty:
        return publish(path, assets)
    assets.require_target(target_port)
    path.parent.mkdir(parents=True, exist_ok=True)
    companion.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent, prefix='.map-output-') as map_dir, \
            tempfile.TemporaryDirectory(dir=companion.parent, prefix='.resource-output-') as resource_dir:
        candidate = Path(map_dir) / 'map.wad'
        resource = write_lumps(assets.build_lumps(), Path(resource_dir) / 'resources.wad')
        result = publish(candidate, None)
        backup = Path(resource_dir) / 'previous.wad'
        existed = companion.exists()
        if existed:
            shutil.copy2(companion, backup)
        resource.replace(companion)
        try:
            candidate.replace(path)
        except OSError:
            if existed:
                backup.replace(companion)
            else:
                companion.unlink()
            raise
    return {**result, 'path': str(path), 'companion_textures': str(companion)}


def _check_compiled_output(doc: MapDocument, require_playable: bool = False):
    bsp = doc.derived.bsp_geometry
    if not bsp or bsp.get('failure'):
        reason = (bsp or {}).get('failure', 'missing compiled BSP')
        raise ValueError(f'node builder produced invalid BSP for {doc.meta.map_name}: {reason}')
    if not bsp.get('sectors_with_geometry') or bsp.get('area', 0) <= 0:
        raise ValueError(f'compiled BSP has no physical sector regions for {doc.meta.map_name}')
    if bsp.get('subsectors', 0) > 0 and bsp.get('collapsed_subsectors', 0) == bsp.get('subsectors', 0):
        raise ValueError(f'compiled BSP collapsed all subsectors for {doc.meta.map_name}')
    missing_sectors = [s.id for s in doc.sectors
                       if s.id < len(doc.derived.sector_rings) and doc.derived.sector_rings[s.id]
                       and (s.id >= len(doc.derived.sector_regions) or not doc.derived.sector_regions[s.id])]
    if missing_sectors:
        raise ValueError(f'compiled BSP is missing physical regions for sector(s) {missing_sectors} in {doc.meta.map_name}')

    if require_playable:
        from wad_parser.types import active_single_player
        from wad_parser.map_extractor import _sector_containing
        from generation.actors import spawn_problem, actor_profile
        from wad_parser.navigation import PlayerNavigation
        from generation.progression import EXIT_SPECIALS

        starts = [t for t in doc.things if t.type == 1 and active_single_player(t)]
        if not starts:
            starts = [t for t in doc.things if t.type == 1]
        for start in starts:
            sid = _sector_containing(start.x, start.y, doc.sectors, doc.derived.sector_regions, doc.vertices)
            if sid is None:
                raise ValueError(f'player start is outside compiled physical space in {doc.meta.map_name}')
            problem = spawn_problem(doc, start.type, start.x, start.y, ignore_id=start.id)
            if problem:
                raise ValueError(f'player start in {doc.meta.map_name} lacks clearance in compiled geometry: {problem}')

        for thing in doc.things:
            if (thing.type == 1 or not active_single_player(thing)
                    or thing.category not in {'monster', 'player-start', 'obstacle'}
                    or actor_profile(thing.type) is None):
                continue
            problem = spawn_problem(doc, thing.type, thing.x, thing.y, ignore_id=thing.id)
            if problem:
                raise ValueError(f'thing {thing.id} in compiled {doc.meta.map_name}: {problem}')

        navigation = PlayerNavigation(doc)
        if not navigation.reached:
            raise ValueError(f'player start has no navigable configuration space in compiled {doc.meta.map_name}')
        missing = navigation.regions.keys() - navigation.reached
        if missing:
            raise ValueError(f'unreachable compiled player configuration regions for {doc.meta.map_name}: {sorted(missing)}')
        exits = [line for line in doc.linedefs if line.special in EXIT_SPECIALS]
        if exits and not any(navigation.reaches_line(line) for line in exits):
            raise ValueError(f'no reachable usable exit in compiled player configuration space for {doc.meta.map_name}')
        # The optimistic door union can pool keys from incompatible one-way
        # branches. Search histories directly on the freshly decoded regions;
        # raw validation deliberately rebuilds its own authoring geometry.
        from wad_parser.progression import requires_action_history, search_exit_history
        if requires_action_history(doc):
            history = search_exit_history(doc)
            if history['status'] != 'success':
                raise ValueError(f'compiled door/key progression in {doc.meta.map_name}: '
                                 + history['reason'])


def _check_wad(path, names, target_port, require_playable, compiled_maps):
    from wad_writer.compatibility import require_target_port
    from wad_parser.wad_reader import WadFile, MAP_MARKER_RE
    from wad_parser.map_extractor import extract_all_maps
    wad = WadFile(str(path))
    compatibility = require_target_port(wad, target_port)
    if [lump.name for lump in wad.lumps if MAP_MARKER_RE.fullmatch(lump.name)] != names:
        raise ValueError('serialized map markers differ from the requested maps (duplicate or orphan marker)')
    if [group.name for group in wad.find_map_groups()] != names:
        raise ValueError('serialized map markers differ from the requested maps')
    if require_playable or compiled_maps:
        for doc in extract_all_maps(wad):
            if doc.meta.map_name in compiled_maps or (require_playable and doc.derived.bsp_geometry is not None):
                _check_compiled_output(doc, require_playable=require_playable)
            if require_playable:
                require_valid(doc, require_connected=True)
    return compatibility


def map_wads_to_wad(paths, out_path, *, assets=None, target_port='zdoom'):
    """Combine individually checked Doom maps without running a node builder.

    Inputs each contain one map and its compiled/GL lumps, with no bundled
    graphics. Supply shared custom assets once through ``assets``. Each input
    and the combined candidate receive full target/BSP/playability validation;
    any failure preserves the previous destination.
    """
    from wad_parser.wad_reader import WadFile, BINARY_MAP_LUMPS, GL_MAP_LUMPS
    from wad_writer.compatibility import TARGET_PORTS
    if target_port not in TARGET_PORTS:
        raise ValueError('target_port must be zdoom or vanilla')
    if assets is not None:
        assets.require_target(target_port)
    names, lumps, publications = [], [], []
    for source in paths:
        wad = WadFile(str(source))
        groups = wad.find_map_groups()
        if len(groups) != 1 or groups[0].format != 'doom':
            raise ValueError('each publication input must contain exactly one Doom-format map')
        group, = groups
        if group.name in names:
            raise ValueError(f'duplicate map marker in publication inputs: {group.name}')
        allowed = BINARY_MAP_LUMPS | GL_MAP_LUMPS | {group.name}
        gl_markers = {lump.name for index, lump in enumerate(wad.lumps[:-1])
                      if lump.name.startswith('GL_') and wad.lumps[index + 1].name == 'GL_VERT'}
        if any(lump.name not in allowed | gl_markers for lump in wad.lumps):
            raise ValueError('publication inputs must not contain global assets; supply shared assets separately')
        compiled = (any(group.lumps.get(name) for name in ('NODES', 'SSECTORS', 'ZNODES'))
                    or bool(group.gl_metadata))
        compatibility = _check_wad(source, [group.name], target_port, True,
                                   [group.name] if compiled else [])
        names.append(group.name)
        publications.append({'map': group.name, 'nodes_built': compiled, 'compatibility': compatibility})
        lumps.extend((lump.name, lump.data) for lump in wad.lumps)
    if not names:
        raise ValueError('cannot publish a map batch with no accepted maps')
    if assets is not None and not assets.is_empty:
        lumps.extend(assets.build_lumps())
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent, prefix='.map-panel-') as directory:
        candidate = write_lumps(lumps, Path(directory) / 'panel.wad')
        compiled_maps = [row['map'] for row in publications if row['nodes_built']]
        compatibility = _check_wad(candidate, names, target_port, True, compiled_maps)
        candidate.replace(path)
    return {'path': str(path), 'maps': names, 'nodes_built': len(compiled_maps) == len(names),
            'target_port': target_port, 'compatibility': compatibility, 'map_publication': publications}


class NoUsableMapsError(RuntimeError):
    """All candidates failed, retaining per-attempt evidence for callers."""

    def __init__(self, failures, requested_maps):
        self.report = {'maps': [], 'summaries': [], 'failed_samples': failures,
                       'requested_maps': requested_maps, 'status': 'failed'}
        super().__init__('no usable map samples: ' + '; '.join(row['reason'] for row in failures))


class MapBatch:
    """Compile/check candidates separately; publish accepted bytes atomically.

    A failed add leaves earlier candidates available and the destination intact.
    Generators own sampling/retry budgets and failure records. The final combine
    revalidates the full WAD without rebuilding accepted maps a second time.
    """

    def __init__(self, out_path, *, assets=None, target_port='zdoom', try_nodes=True):
        from wad_writer.compatibility import TARGET_PORTS
        if target_port not in TARGET_PORTS:
            raise ValueError('target_port must be zdoom or vanilla')
        self.path = Path(out_path)
        self.assets, self.target_port, self.try_nodes = assets, target_port, try_nodes
        self.accepted = []
        self._directory = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._directory = tempfile.TemporaryDirectory(dir=self.path.parent, prefix='.map-batch-')
        return self

    def __exit__(self, *exc):
        self._directory.cleanup()
        self._directory = None

    def add(self, doc):
        if self._directory is None:
            raise RuntimeError('MapBatch must be used as a context manager')
        name = doc.meta.map_name.upper()
        if any(row['map'] == name for row in self.accepted):
            raise ValueError(f'duplicate accepted map marker: {name}')
        path = Path(self._directory.name) / f'{len(self.accepted):03d}.wad'
        result = documents_to_wad([doc], path, try_nodes=self.try_nodes,
                                  require_playable=True, target_port=self.target_port)
        row = {'map': name, 'nodes_built': result['nodes_built'],
               'compatibility': result['compatibility']}
        self.accepted.append(row)
        return row

    def publish(self, companion_textures=None):
        if self._directory is None:
            raise RuntimeError('MapBatch must be used as a context manager')
        if not self.accepted:
            raise ValueError('cannot publish a map batch with no accepted maps')
        paths = [Path(self._directory.name) / f'{index:03d}.wad' for index in range(len(self.accepted))]
        if companion_textures is not None:
            return _publish_with_resources(self.path, companion_textures, self.assets, self.target_port,
                lambda path, bundle: map_wads_to_wad(paths, path, assets=bundle, target_port=self.target_port))
        return map_wads_to_wad(paths, self.path, assets=self.assets, target_port=self.target_port)


def _register_assets(game: GameProfile | None, assets: AssetBundle | None, target_port='zdoom') -> GameProfile | None:
    """Bundled custom assets must count as valid so enforcement keeps them."""
    if assets is not None:
        assets.require_target(target_port)
    if game is not None and assets is not None and not assets.is_empty:
        game = copy.deepcopy(game)
        game.extend_assets(assets.texture_names(), assets.flat_names())
    return game


def _source_assets(game, assets, sources, target_port):
    if not sources:
        return None
    from generation.asset_extractor import AssetResolution
    return AssetResolution(game, assets, sources, target_port)


def regenerate_from_json(json_path: Path | str, out_path: Path | str,
                         resolution: int = 128, coarse_factor: int = 2,
                         style: StyleModel | None = None,
                         seed: int | None = None,
                         game: GameProfile | None = None,
                         assets: AssetBundle | None = None,
                         properties: PropertyModel | None = None,
                         placement: PlacementModel | None = None, target_port: str = 'zdoom',
                         sources: list[Path | str] | None = None,
                         companion_textures: Path | str | None = None) -> dict:
    """Extracted-map JSON -> raster -> vectorize -> validate -> WAD.
    Model-free harness for the back half of the pipeline.

    Unlike model samples (which have no ground-truth scale and use the
    default cell size), the raster meta here carries the exact world->grid
    scale, so the regenerated map comes back at the ORIGINAL map's true
    world scale.

    By default it decorates faithfully: the source map's own textures and
    things are carried across, since randomness only obscures how well the
    geometry survived. Passing an explicit `style` opts into the corpus
    decorator instead (useful for exercising that path)."""
    from wad_writer.compatibility import TARGET_PORTS
    if target_port not in TARGET_PORTS:
        raise ValueError("target_port must be zdoom or vanilla")
    game = _register_assets(game, assets, target_port)
    resource_context = _source_assets(game, assets, sources, target_port)
    if resource_context is not None:
        game = resource_context.game
    doc = MapDocument.from_dict(json.loads(Path(json_path).read_text(encoding="utf-8")))
    raster, meta = rasterize(doc, resolution=resolution)
    cell_units = coarse_factor / meta["scale"]
    new_doc, repairs, decoration = raster_to_document(
        raster, map_name=doc.meta.map_name, cell_units=cell_units,
        coarse_factor=coarse_factor, style=style, style_seed=seed,
        source=None if style else doc, meta=meta, game=game,
        properties=properties, placement=placement, tags={}, asset_resolution=resource_context)
    if resource_context is not None:
        assets = resource_context.bundle
    result = documents_to_wad([new_doc], out_path, assets=assets, target_port=target_port,
                              companion_textures=companion_textures)
    if resource_context is not None:
        result['asset_resolution'] = resource_context.report()
    result["repairs"] = repairs
    result["decoration"] = decoration
    result["sectors"] = len(new_doc.sectors)
    result["linedefs"] = len(new_doc.linedefs)
    result["cell_units"] = cell_units
    return result


def generate(checkpoint_path: Path | str, tag_weights: dict[str, float],
             n_maps: int = 1, out_path: Path | str = "generated.wad",
             steps: int = 50, guidance: float = 2.0,
             seed: int | None = None, style: StyleModel | None = None,
             game: GameProfile | None = None,
             assets: AssetBundle | None = None,
             coarse_factor: int = 2,
             properties: PropertyModel | None = None,
             placement: PlacementModel | None = None,
             world_span: float | None = None, target_port: str = 'zdoom',
             sources: list[Path | str] | None = None,
             companion_textures: Path | str | None = None) -> dict:
    """Model-driven generation. Imports torch lazily so the rest of the
    package works without it installed.

    coarse_factor is the cell size in raster pixels, defaulting to 2.
    Smaller cells retain finer contours at greater geometry/validation cost.
    """
    import torch
    from wad_writer.compatibility import TARGET_PORTS

    if target_port not in TARGET_PORTS:
        raise ValueError("target_port must be zdoom or vanilla")
    if not 1 <= n_maps <= 99:
        raise ValueError("n_maps must be between 1 and 99")
    if steps < 1 or not math.isfinite(guidance):
        raise ValueError("steps must be positive and guidance must be finite")
    if world_span is not None and (not math.isfinite(world_span) or world_span <= 0):
        raise ValueError("n_maps and world_span must be positive")

    game = _register_assets(game, assets, target_port)
    resource_context = _source_assets(game, assets, sources, target_port)
    if resource_context is not None:
        game = resource_context.game

    from models.checkpoint import load_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, diffusion, conditioner, config = load_checkpoint(checkpoint_path, device)

    if seed is not None:
        torch.manual_seed(seed)
    import numpy as np
    spans = None
    if config.get('numeric_dim', 0):
        rng = np.random.default_rng(seed)
        spans = [world_span or float(rng.choice(config['world_spans'])) for _ in range(n_maps)]
    numeric_vals = [[math.log(span / 2048)] for span in spans] if spans else None
    cond = conditioner.encode([tag_weights] * n_maps, device=device, numeric_values=numeric_vals)
    uncond = conditioner.encode([{}] * n_maps, device=device, numeric_values=numeric_vals)
    with torch.no_grad():
        x = diffusion.ddim_sample(
            (n_maps, config["in_channels"], config["resolution"], config["resolution"]),
            cond, uncond, steps=steps, guidance=guidance, device=device,
        )
    rasters = ((x.clamp(-1, 1) + 1) / 2).cpu().numpy()  # model space -> [0,1]

    all_repairs, all_decoration, failed = {}, {}, []
    with MapBatch(out_path, assets=assets, target_port=target_port) as batch:
        for i in range(n_maps):
            name = f"MAP{len(batch.accepted) + 1:02d}"
            stage = 'validation'
            try:
                doc, repairs, decoration = raster_to_document(
                    rasters[i], map_name=name, coarse_factor=coarse_factor, style=style,
                    cell_units=(spans[i] * coarse_factor / config['resolution']) if spans else
                               (world_span * coarse_factor / config['resolution'] if world_span else 64),
                    style_seed=None if seed is None else seed + i, game=game,
                    properties=properties, placement=placement, tags=tag_weights,
                    raster_version=config.get('raster_version', 1), asset_resolution=resource_context)
                if resource_context is not None:
                    batch.assets = resource_context.bundle
                stage = 'publication'
                batch.add(doc)
            except ValueError as exc:
                failed.append({'sample': i, 'seed': seed,
                               'style_seed': None if seed is None else seed + i,
                               'reason': str(exc), 'failure_stage': stage,
                               'validated': stage == 'publication'})
                continue
            all_repairs[name] = repairs
            all_decoration[name] = decoration
        if not batch.accepted:
            raise NoUsableMapsError(failed, n_maps)
        result = batch.publish(companion_textures=companion_textures)
    if resource_context is not None:
        result['asset_resolution'] = resource_context.report()
    result["repairs"] = all_repairs
    result["decoration"] = all_decoration
    result["failed_samples"] = failed
    result["requested_maps"] = n_maps
    result["tags"] = tag_weights
    return result

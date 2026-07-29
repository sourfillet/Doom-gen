"""End-to-end generation: sample rasters -> vectorize -> validate ->
write a playable PWAD (plan/generation.md, Stage 1 -> 2 -> validate ->
write; Stages 3/4 are heuristic placeholders at this point - default
textures and a deterministic player start).

Also provides regenerate_from_json(): re-runs the raster->vector->WAD
back half on an EXISTING extracted map. That path is the pipeline's test
harness, not generation - it isolates vectorizer/writer bugs from model
bugs, because its input raster is known-good.
"""
from __future__ import annotations

import json
from pathlib import Path

from data.rasterizer import rasterize
from generation.game_profile import GameProfile
from generation.placement import PlacementModel, apply_learned_placement
from generation.properties import PropertyModel, apply_properties
from generation.style import (
    PlacementDensity, StyleModel, apply_style, transfer_source_attributes,
)
from generation.validator import validate_and_repair
from generation.vectorizer import vectorize
from wad_parser.types import MapDocument
from wad_writer.assets import AssetBundle
from wad_writer.wad_writer import build_nodes, write_wad


def load_style(path_or_dir: Path | str | None) -> StyleModel | None:
    """A fitted style .json, or a maps directory to fit from on the fly."""
    if path_or_dir is None:
        return None
    p = Path(path_or_dir)
    return StyleModel.fit(p) if p.is_dir() else StyleModel.load(p)


def raster_to_document(raster, map_name: str = "MAP01",
                       cell_units: int = 64, coarse_factor: int = 4,
                       style: StyleModel | None = None,
                       style_seed: int | None = None,
                       source: MapDocument | None = None,
                       meta: dict | None = None,
                       game: GameProfile | None = None,
                       properties: PropertyModel | None = None,
                       placement: PlacementModel | None = None,
                       tags: dict | None = None) -> tuple[MapDocument, list[str], dict]:
    doc = vectorize(raster, cell_units=cell_units, coarse_factor=coarse_factor,
                    map_name=map_name)
    if source is not None:
        # Regen harness: we still have the real map, so carry its actual
        # textures/things across rather than inventing them. Runs BEFORE
        # validation so reachability starts from the real player start.
        decoration = transfer_source_attributes(doc, source, meta, cell_units,
                                                 coarse_factor)
        repairs = validate_and_repair(doc)
    else:
        repairs = validate_and_repair(doc)
        decoration = {}
        if style:
            # channel 8 tells us WHERE things go; the style model decides how
            # many and which kinds (unless Stage 4 owns thing placement)
            density = PlacementDensity.from_raster(raster, cell_units=cell_units,
                                                   coarse_factor=coarse_factor)
            decoration = apply_style(doc, style, seed=style_seed, density=density,
                                     place_things=placement is None)
    if placement is not None:
        # Stage 4: learned per-sector counts + types, placed by the geometric
        # placer. Runs after style (which set flats/exit/DM) so the map has a
        # player start to anchor reachability + keep-away on.
        decoration["placement"] = apply_learned_placement(
            doc, placement, tags or {}, seed=style_seed)
    if properties is not None:
        # Stage 3: overwrite the heuristic flats/textures/specials with the
        # GNN's context-aware, tag-conditioned predictions.
        decoration["properties"] = apply_properties(doc, properties, tags or {})
    if game is not None:
        # Last: guarantee every asset the map references exists in the target
        # game (or a bundled PWAD asset registered on the profile).
        decoration["game_enforcement"] = game.enforce(doc)
    return doc, repairs, decoration


def documents_to_wad(docs: list[MapDocument], out_path: Path | str,
                     try_nodes: bool = True,
                     assets: AssetBundle | None = None) -> dict:
    path = write_wad(docs, out_path, assets=assets)
    nodes_built = build_nodes(path) if try_nodes else False
    return {"path": str(path), "maps": [d.meta.map_name for d in docs],
            "nodes_built": nodes_built}


def _register_assets(game: GameProfile | None, assets: AssetBundle | None) -> None:
    """Bundled custom assets must count as valid so enforcement keeps them."""
    if game is not None and assets is not None and not assets.is_empty:
        game.extend_assets(assets.texture_names(), assets.flat_names())


def regenerate_from_json(json_path: Path | str, out_path: Path | str,
                         resolution: int = 128, coarse_factor: int = 4,
                         style: StyleModel | None = None,
                         seed: int | None = None,
                         game: GameProfile | None = None,
                         assets: AssetBundle | None = None,
                         properties: PropertyModel | None = None,
                         placement: PlacementModel | None = None) -> dict:
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
    _register_assets(game, assets)
    doc = MapDocument.from_dict(json.loads(Path(json_path).read_text()))
    raster, meta = rasterize(doc, resolution=resolution)
    cell_units = max(8, int(round(coarse_factor / meta["scale"] / 8)) * 8)
    new_doc, repairs, decoration = raster_to_document(
        raster, map_name=doc.meta.map_name, cell_units=cell_units,
        coarse_factor=coarse_factor, style=style, style_seed=seed,
        source=None if style else doc, meta=meta, game=game,
        properties=properties, placement=placement, tags={})
    result = documents_to_wad([new_doc], out_path, assets=assets)
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
             coarse_factor: int = 4,
             properties: PropertyModel | None = None,
             placement: PlacementModel | None = None) -> dict:
    """Model-driven generation. Imports torch lazily so the rest of the
    package works without it installed.

    coarse_factor is the vectorizer cell size in raster pixels: 2 keeps far
    more (and narrower) sectors alive than the default 4, at the cost of
    noisier geometry - the single cheapest knob for less-blocky output from a
    fixed-resolution checkpoint."""
    import torch

    _register_assets(game, assets)

    from models.checkpoint import load_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, diffusion, conditioner, config = load_checkpoint(checkpoint_path, device)

    if seed is not None:
        torch.manual_seed(seed)
    cond = conditioner.encode([tag_weights] * n_maps, device=device)
    uncond = torch.zeros_like(cond)
    with torch.no_grad():
        x = diffusion.ddim_sample(
            (n_maps, config["in_channels"], config["resolution"], config["resolution"]),
            cond, uncond, steps=steps, guidance=guidance, device=device,
        )
    rasters = ((x.clamp(-1, 1) + 1) / 2).cpu().numpy()  # model space -> [0,1]

    docs, all_repairs, all_decoration, failed = [], {}, {}, []
    for i in range(n_maps):
        name = f"MAP{len(docs) + 1:02d}"
        try:
            doc, repairs, decoration = raster_to_document(
                rasters[i], map_name=name, coarse_factor=coarse_factor, style=style,
                style_seed=None if seed is None else seed + i, game=game,
                properties=properties, placement=placement, tags=tag_weights)
        except ValueError as exc:
            failed.append(f"sample {i}: {exc}")
            continue
        docs.append(doc)
        all_repairs[name] = repairs
        all_decoration[name] = decoration
    if not docs:
        raise RuntimeError(
            "no sample vectorized into a usable map: " + "; ".join(failed))
    result = documents_to_wad(docs, out_path, assets=assets)
    result["repairs"] = all_repairs
    result["decoration"] = all_decoration
    result["failed_samples"] = failed
    result["tags"] = tag_weights
    return result

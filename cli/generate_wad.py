"""Generation CLI: sample tag-conditioned layouts and write a playable
PWAD, or exercise the model-free back half of the pipeline.

    generate-wad --checkpoint checkpoints/layout_020000.pt \\
        --tags "single-player,secrets:0.7" --n 3 --out generated.wad

    generate-wad --from-json dataset/maps/E1M1.json --out roundtrip.wad

--from-json regenerates an EXISTING map through raster -> vectorize ->
validate -> write. That is a test harness, not generation: its input
raster is known-good, so any breakage it shows lives in the vectorizer or
writer, never the model.
"""
from __future__ import annotations

import click
from pathlib import Path

from generation.game_profile import load_profile
from generation.pipeline import generate, load_style, regenerate_from_json
from generation.placement import load_placement_model
from generation.properties import load_property_model
from wad_writer.assets import AssetBundle
from cli.options import parse_tags as _parse_tags, generation_errors


@click.command()
@click.option("--checkpoint", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--tags", "tag_spec", default=None, help='e.g. "techbase,boss:0.5"')
@click.option("--n", "n_maps", default=1, type=click.IntRange(min=1), show_default=True)
@click.option("--steps", default=50, type=click.IntRange(min=1), show_default=True, help="DDIM steps.")
@click.option("--guidance", default=2.0, type=click.FloatRange(min=0), show_default=True,
              help="CFG scale: 0 unconditional, 1 conditional, >1 stronger tags.")
@click.option("--world-span", default=None, type=click.FloatRange(min=0, min_open=True),
              help="Generated XY canvas span in Doom units; otherwise sample the trained scale prior.")
@click.option("--seed", default=None, type=int)
@click.option("--out", "out_path", default="generated.wad", show_default=True)
@click.option('--target-port', type=click.Choice(['zdoom', 'vanilla']), default='zdoom', show_default=True,
              help='Vanilla requires compiled classic nodes and static index/BLOCKMAP limits.')
@click.option("--from-json", "from_json", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="Model-free: regenerate one extracted map (pipeline test).")
@click.option("--resolution", default=128, type=click.IntRange(min=8), show_default=True,
              help="Raster resolution for --from-json (finer = smaller blocks).")
@click.option("--coarse-factor", default=2, type=click.IntRange(min=1), show_default=True,
              help="Cell size in raster pixels; 2 keeps narrow corridors alive.")
@click.option("--style", "style_path", default=None, type=click.Path(exists=True),
              help="Corpus style: a fitted style .json, or a maps dir to fit from. "
                   "Enables textures and heuristic things.")
@click.option("--game", "game_path", default=None, type=click.Path(exists=True),
              help="Constrain output to one game's assets: a profile .json or an "
                   "IWAD to derive one from; restricts asset/type choices.")
@click.option("--assets", "assets_dir", default=None, type=click.Path(exists=True, file_okay=False),
              help="Bundle custom flats/patches/textures into the PWAD (dir with "
                   "flats/, patches/, textures.json). Registered on --game so they "
                   "survive enforcement.")
@click.option("--sources", "sources_spec", default=None,
              help="Comma-separated WAD files or directories to automatically extract missing textures/flats from.")
@click.option("--companion-textures", default=None, type=click.Path(dir_okay=False),
              help="Write extracted assets to a separate companion resource WAD instead of embedding.")
@click.option("--properties", "properties_path", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="Stage 3 property GNN checkpoint: context-aware learned "
                   "flats/textures/specials instead of corpus-frequency ones.")
@click.option("--placement", "placement_path", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="Stage 4 placement GNN checkpoint: learned per-sector "
                   "monster/pickup counts + types instead of the density heuristic.")
@generation_errors
def main(checkpoint, tag_spec, n_maps, steps, guidance, seed, out_path, from_json,
         resolution, coarse_factor, style_path, game_path, assets_dir,
         properties_path, placement_path, world_span, target_port,
         sources_spec, companion_textures) -> None:
    tags = _parse_tags(tag_spec)
    style = load_style(style_path)
    game = load_profile(game_path)
    assets = AssetBundle.from_dir(assets_dir) if assets_dir else None
    if assets and game_path and Path(game_path).suffix.lower() == '.wad':
        assets = assets.with_base_wad(game_path)
    sources = [s.strip() for s in sources_spec.split(',') if s.strip()] if sources_spec else None
    if companion_textures and Path(companion_textures).resolve() == Path(out_path).resolve():
        raise click.UsageError('companion textures and map WAD output must use different paths')
    properties = load_property_model(properties_path) if properties_path else None
    placement = load_placement_model(placement_path) if placement_path else None
    if from_json:
        result = regenerate_from_json(from_json, out_path, resolution=resolution,
                                      coarse_factor=coarse_factor, style=style,
                                      seed=seed, game=game, assets=assets,
                                      properties=properties, placement=placement, target_port=target_port,
                                      sources=sources, companion_textures=companion_textures)
    elif checkpoint:
        result = generate(checkpoint, tags, n_maps=n_maps,
                          out_path=out_path, steps=steps, guidance=guidance,
                          seed=seed, style=style, game=game, assets=assets,
                          coarse_factor=coarse_factor, world_span=world_span, properties=properties,
                          placement=placement, target_port=target_port,
                          sources=sources, companion_textures=companion_textures)
    else:
        raise click.ClickException("need --checkpoint (generation) or --from-json (pipeline test)")

    click.echo(f"wrote {result['path']}  maps: {', '.join(result['maps'])}")
    if result.get("companion_textures"):
        click.echo(f"wrote companion texture resource: {result['companion_textures']}")
    if not result.get("nodes_built"):
        click.echo("nodes left empty (no zdbsp found); the selected port must rebuild them")
    if result.get("repairs"):
        click.echo(f"repairs: {result['repairs']}")
    if result.get("decoration"):
        click.echo(f"decoration: {result['decoration']}")
    if result.get('failed_samples'):
        click.echo(f"failed samples: {result['failed_samples']}")


if __name__ == "__main__":
    main()

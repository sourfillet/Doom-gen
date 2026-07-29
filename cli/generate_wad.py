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

from generation.game_profile import load_profile
from generation.pipeline import generate, load_style, regenerate_from_json
from generation.placement import load_placement_model
from generation.properties import load_property_model
from wad_writer.assets import AssetBundle


def _parse_tags(spec: str | None) -> dict[str, float]:
    tags: dict[str, float] = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            name, _, w = part.rpartition(":")
            tags[name.strip()] = float(w)
        else:
            tags[part] = 1.0
    return tags


@click.command()
@click.option("--checkpoint", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--tags", "tag_spec", default=None, help='e.g. "techbase,boss:0.5"')
@click.option("--n", "n_maps", default=1, show_default=True)
@click.option("--steps", default=50, show_default=True, help="DDIM steps.")
@click.option("--guidance", default=2.0, show_default=True, help="CFG scale.")
@click.option("--seed", default=None, type=int)
@click.option("--out", "out_path", default="generated.wad", show_default=True)
@click.option("--from-json", "from_json", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="Model-free: regenerate one extracted map (pipeline test).")
@click.option("--resolution", default=128, show_default=True,
              help="Raster resolution for --from-json (finer = smaller blocks).")
@click.option("--coarse-factor", default=4, show_default=True,
              help="Cell size in raster pixels; 2 keeps narrow corridors alive.")
@click.option("--style", "style_path", default=None, type=click.Path(exists=True),
              help="Corpus style: a fitted style .json, or a maps dir to fit from. "
                   "Enables textures/things/exit placement.")
@click.option("--game", "game_path", default=None, type=click.Path(exists=True),
              help="Constrain output to one game's assets: a profile .json or an "
                   "IWAD to derive one from. Guarantees the map loads cleanly "
                   "against that game (no missing textures / unspawnable things).")
@click.option("--assets", "assets_dir", default=None, type=click.Path(exists=True, file_okay=False),
              help="Bundle custom flats/patches/textures into the PWAD (dir with "
                   "flats/, patches/, textures.json). Registered on --game so they "
                   "survive enforcement.")
@click.option("--properties", "properties_path", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="Stage 3 property GNN checkpoint: context-aware learned "
                   "flats/textures/specials instead of corpus-frequency ones.")
@click.option("--placement", "placement_path", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="Stage 4 placement GNN checkpoint: learned per-sector "
                   "monster/pickup counts + types instead of the density heuristic.")
def main(checkpoint, tag_spec, n_maps, steps, guidance, seed, out_path, from_json,
         resolution, coarse_factor, style_path, game_path, assets_dir,
         properties_path, placement_path) -> None:
    style = load_style(style_path)
    game = load_profile(game_path)
    assets = AssetBundle.from_dir(assets_dir) if assets_dir else None
    properties = load_property_model(properties_path) if properties_path else None
    placement = load_placement_model(placement_path) if placement_path else None
    if from_json:
        result = regenerate_from_json(from_json, out_path, resolution=resolution,
                                      coarse_factor=coarse_factor, style=style,
                                      seed=seed, game=game, assets=assets,
                                      properties=properties, placement=placement)
    elif checkpoint:
        result = generate(checkpoint, _parse_tags(tag_spec), n_maps=n_maps,
                          out_path=out_path, steps=steps, guidance=guidance,
                          seed=seed, style=style, game=game, assets=assets,
                          coarse_factor=coarse_factor, properties=properties,
                          placement=placement)
    else:
        raise click.ClickException("need --checkpoint (generation) or --from-json (pipeline test)")

    click.echo(f"wrote {result['path']}  maps: {', '.join(result['maps'])}")
    if not result.get("nodes_built"):
        click.echo("nodes left empty (no zdbsp found) - fine for GZDoom/ZDoom-family ports")
    if result.get("repairs"):
        click.echo(f"repairs: {result['repairs']}")
    if result.get("decoration"):
        click.echo(f"decoration: {result['decoration']}")


if __name__ == "__main__":
    main()

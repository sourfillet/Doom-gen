"""Graph-native generation CLI (Phase 9.4): sample room-graphs from the
trained generator and write a playable PWAD with real rooms/corridors/doors.

    generate-roomgen --checkpoint checkpoints/roomgen_015000.pt \\
        --tags "doom2,single-player" --n 3 --game wads/DOOM2.WAD \\
        --style style.json --out roomgen.wad
"""
from __future__ import annotations

import click

from generation.game_profile import load_profile
from generation.pipeline import load_style
from generation.roomgen import generate


def _parse_tags(spec: str | None) -> dict[str, float]:
    tags: dict[str, float] = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        name, _, w = part.rpartition(":") if ":" in part else (part, "", "1")
        tags[name.strip()] = float(w or 1.0)
    return tags


@click.command()
@click.option("--checkpoint", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--tags", "tag_spec", default=None, help='e.g. "doom2,single-player"')
@click.option("--n", "n_maps", default=1, show_default=True)
@click.option("--seed", default=None, type=int)
@click.option("--temperature", default=1.0, show_default=True,
              help="Sampling temperature; <1 tighter, >1 more varied.")
@click.option("--out", "out_path", default="generated_roomgen.wad", show_default=True)
@click.option("--style", "style_path", default=None, type=click.Path(exists=True),
              help="Corpus style .json or maps dir for textures/things.")
@click.option("--game", "game_path", default=None, type=click.Path(exists=True),
              help="Constrain to a game's assets (profile .json or IWAD).")
@click.option("--properties", "properties_path", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="Stage 3 property GNN checkpoint (learned textures).")
@click.option("--placement", "placement_path", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="Stage 4 placement GNN checkpoint (learned thing placement).")
def main(checkpoint, tag_spec, n_maps, seed, temperature, out_path,
         style_path, game_path, properties_path, placement_path) -> None:
    from generation.placement import load_placement_model
    from generation.properties import load_property_model

    result = generate(
        checkpoint, _parse_tags(tag_spec), n_maps=n_maps, out_path=out_path,
        seed=seed, temperature=temperature,
        style=load_style(style_path),
        game=load_profile(game_path),
        properties=load_property_model(properties_path) if properties_path else None,
        placement=load_placement_model(placement_path) if placement_path else None)

    click.echo(f"wrote {result['path']}  maps: {', '.join(result['maps'])}")
    for s in result["summaries"]:
        click.echo(f"  {s['map']}: {s['rooms']} rooms -> {s['sectors']} sectors")
    if not result["nodes_built"]:
        click.echo("nodes left empty (no zdbsp) - fine for GZDoom/ZDoom ports")


if __name__ == "__main__":
    main()

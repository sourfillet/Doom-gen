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

from generation.pipeline import generate, regenerate_from_json


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
def main(checkpoint, tag_spec, n_maps, steps, guidance, seed, out_path, from_json,
         resolution, coarse_factor) -> None:
    if from_json:
        result = regenerate_from_json(from_json, out_path, resolution=resolution,
                                      coarse_factor=coarse_factor)
    elif checkpoint:
        result = generate(checkpoint, _parse_tags(tag_spec), n_maps=n_maps,
                          out_path=out_path, steps=steps, guidance=guidance, seed=seed)
    else:
        raise click.ClickException("need --checkpoint (generation) or --from-json (pipeline test)")

    click.echo(f"wrote {result['path']}  maps: {', '.join(result['maps'])}")
    if not result.get("nodes_built"):
        click.echo("nodes left empty (no zdbsp found) - fine for GZDoom/ZDoom-family ports")
    repairs = result.get("repairs")
    if repairs:
        click.echo(f"repairs: {repairs}")


if __name__ == "__main__":
    main()

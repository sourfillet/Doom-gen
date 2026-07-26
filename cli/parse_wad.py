"""CLI: parse a WAD into per-map JSON documents (plus one WAD-wide texture
set document), per plan/data-model.md.

    parse-wad DOOM2.WAD --output ./dataset/maps/
    parse-wad mymap.wad --map MAP03 --stdout
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import click

from wad_parser.map_extractor import extract_all_maps
from wad_parser.texture_parser import build_texture_set
from wad_parser.wad_reader import WadFile


def _to_json(obj) -> str:
    return json.dumps(dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj,
                       indent=2, sort_keys=False)


@click.command()
@click.argument("wad_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "output_dir", type=click.Path(file_okay=False), default=None,
              help="Directory to write one JSON file per map, plus a *.textures.json file.")
@click.option("--map", "map_name", default=None, help="Only parse this one map (e.g. MAP03).")
@click.option("--stdout", "to_stdout", is_flag=True, help="Print JSON to stdout instead of writing files.")
def main(wad_path: str, output_dir: str | None, map_name: str | None, to_stdout: bool) -> None:
    wad = WadFile(wad_path)
    maps = extract_all_maps(wad)
    if map_name:
        maps = [m for m in maps if m.meta.map_name.upper() == map_name.upper()]
        if not maps:
            raise click.ClickException(f"Map {map_name!r} not found in {wad_path}")

    if to_stdout:
        for doc in maps:
            click.echo(_to_json(doc))
        return

    if not output_dir:
        raise click.ClickException("Specify --output <dir> or --stdout")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for doc in maps:
        dest = out / f"{doc.meta.map_name}.json"
        dest.write_text(_to_json(doc))
        click.echo(f"wrote {dest}")

    flat_bytes = wad.find_flats()
    playpal = wad.get_lump("PLAYPAL")
    colormap = wad.get_lump("COLORMAP")

    texture_set = build_texture_set(
        texture1=wad.get_lump("TEXTURE1"),
        texture2=wad.get_lump("TEXTURE2"),
        pnames_data=wad.get_lump("PNAMES"),
        flat_bytes=flat_bytes,
        playpal=playpal,
        colormap=colormap,
        animated=wad.get_lump("ANIMATED"),
        switches=wad.get_lump("SWITCHES"),
    )
    tex_dest = out / f"{wad.hash}.textures.json"
    tex_dest.write_text(_to_json(texture_set))
    click.echo(f"wrote {tex_dest}")

    # Raw pixel/palette/colormap bytes are kept out of the JSON (they're
    # binary blobs, not structured fields) but still captured to disk,
    # content-addressed by sha256 so repeated assets across WADs dedupe.
    assets_dir = out / f"{wad.hash}_assets"
    if flat_bytes:
        flats_dir = assets_dir / "flats"
        flats_dir.mkdir(parents=True, exist_ok=True)
        for asset in texture_set.flats:
            (flats_dir / f"{asset.sha256}.flat").write_bytes(flat_bytes[asset.name])
    if playpal:
        assets_dir.mkdir(parents=True, exist_ok=True)
        (assets_dir / f"{texture_set.playpal_sha256}.playpal").write_bytes(playpal)
    if colormap:
        assets_dir.mkdir(parents=True, exist_ok=True)
        (assets_dir / f"{texture_set.colormap_sha256}.colormap").write_bytes(colormap)
    if flat_bytes or playpal or colormap:
        click.echo(f"wrote raw assets under {assets_dir}")


if __name__ == "__main__":
    main()

"""
parse-wad CLI

Usage examples:
  parse-wad DOOM2.WAD --output ./dataset/maps/
  parse-wad DOOM2.WAD --map MAP01 --stdout
  parse-wad DOOM2.WAD --maps MAP01,MAP03 --output ./maps/
  parse-wad DOOM2.WAD --output ./maps/ --textures
  parse-wad DOOM2.WAD --list
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from wad_parser.map_extractor import extract_map
from wad_parser.texture_parser import build_texture_set
from wad_parser.wad_reader import WadReader


@click.command()
@click.argument("wad_path", metavar="WAD", type=click.Path(exists=True, dir_okay=False))
@click.option("-m", "--map",  "map_name",  default=None,
              help="Parse a single map by name (e.g. MAP01, E1M1).")
@click.option("--maps",       "map_names", default=None,
              help="Comma-separated list of maps to parse.")
@click.option("-o", "--output", default=None, metavar="DIR",
              type=click.Path(file_okay=False),
              help="Output directory.  One JSON file is written per map.")
@click.option("--stdout",     is_flag=True,
              help="Print JSON to stdout (single-map mode only).")
@click.option("--textures",   is_flag=True,
              help="Include texture metadata in each map JSON.")
@click.option("--list",       "list_maps", is_flag=True,
              help="List all map names in the WAD and exit.")
@click.option("--pretty/--compact", default=True,
              help="Pretty-print (default) or compact JSON output.")
def main(
    wad_path: str,
    map_name: str | None,
    map_names: str | None,
    output: str | None,
    stdout: bool,
    textures: bool,
    list_maps: bool,
    pretty: bool,
) -> None:
    """Parse Doom WAD files and emit structured JSON map data."""

    wad = WadReader(wad_path)

    if list_maps:
        names = wad.get_map_names()
        if not names:
            click.echo("No maps found.", err=True)
        else:
            click.echo("\n".join(names))
        return

    # Resolve which maps to parse
    if map_name:
        targets = [map_name.upper()]
    elif map_names:
        targets = [n.strip().upper() for n in map_names.split(",") if n.strip()]
    else:
        targets = wad.get_map_names()

    if not targets:
        click.echo("No maps found in WAD.", err=True)
        sys.exit(1)

    if stdout and len(targets) > 1:
        click.echo(
            "Error: --stdout requires a single map (use --map or --maps with one name).",
            err=True,
        )
        sys.exit(1)

    if not stdout and not output:
        click.echo("Error: specify --output DIR or --stdout.", err=True)
        sys.exit(1)

    # Build texture set once (if requested)
    tex_set = build_texture_set(wad) if textures else None

    # Set up output directory
    out_dir = Path(output) if output else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    indent = 2 if pretty else None
    errors = 0

    for name in targets:
        try:
            click.echo(f"Parsing {name}...", err=True)
            doom_map = extract_map(wad, name, texture_set=tex_set)

            if textures and tex_set:
                # Embed texture metadata for textures actually referenced by this map
                used_tex_names = _collect_used_textures(doom_map)
                tex_dict = {
                    n: t.to_dict()
                    for n, t in tex_set.textures.items()
                    if n in used_tex_names
                }
                data = doom_map.to_dict()
                data["textures"] = tex_dict
            else:
                data = doom_map.to_dict()

            json_str = json.dumps(data, indent=indent)

            if stdout:
                click.echo(json_str)
            else:
                out_path = out_dir / f"{name}.json"   # type: ignore[operator]
                out_path.write_text(json_str, encoding="utf-8")
                click.echo(f"  → {out_path}", err=True)

        except Exception as exc:
            click.echo(f"  ERROR parsing {name}: {exc}", err=True)
            errors += 1

    if errors:
        sys.exit(1)


def _collect_used_textures(doom_map) -> set[str]:
    """Return the set of texture/flat names actually referenced by this map."""
    used: set[str] = set()
    for sd in doom_map.sidedefs:
        for name in (sd.upper_tex, sd.lower_tex, sd.mid_tex):
            if name and name not in ("-", ""):
                used.add(name)
    for sec in doom_map.sectors:
        used.add(sec.floor_flat)
        used.add(sec.ceil_flat)
    return used


if __name__ == "__main__":
    main()

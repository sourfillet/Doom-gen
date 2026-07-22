"""
tag-maps CLI

Annotate parsed map JSON files with tags from the taxonomy.

Usage examples:
  tag-maps ./dataset/maps/                      # interactive, all maps
  tag-maps ./dataset/maps/ --auto               # auto-infer, no prompts
  tag-maps ./dataset/maps/MAP01.json --auto     # auto-infer single file
  tag-maps ./dataset/maps/ --map MAP01          # interactive, one map
  tag-maps ./dataset/maps/ --show               # print current tags, no editing
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from data.tags import TAG_TAXONOMY, infer_tags, validate_tags
from wad_parser.types import DoomMap, MapMeta, BoundingBox, Vertex, Sector, Linedef, Sidedef, Thing


def _load_map_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_map_json(path: Path, data: dict, pretty: bool = True) -> None:
    path.write_text(json.dumps(data, indent=2 if pretty else None), encoding="utf-8")


def _infer_from_dict(data: dict) -> list[str]:
    """Run auto-inference on a map dict without a full DoomMap object."""
    # Reconstruct a lightweight DoomMap-like object sufficient for tag inference
    doom_map = _dict_to_doom_map(data)
    return infer_tags(doom_map)


def _dict_to_doom_map(data: dict) -> DoomMap:
    """Reconstruct a DoomMap from its serialised dict (for inference only)."""
    meta = data.get("meta", {})
    bounds_d = data.get("bounds", {})

    vertices = [
        Vertex(id=v["id"], x=v["x"], y=v["y"])
        for v in data.get("vertices", [])
    ]
    sectors = [
        Sector(
            id=s["id"],
            floor_height=s["floor_h"],
            ceil_height=s["ceil_h"],
            floor_flat=s["floor_flat"],
            ceil_flat=s["ceil_flat"],
            light_level=s["light"],
            special=s["special"],
            tag=s["tag"],
            special_name=s.get("special_name", ""),
        )
        for s in data.get("sectors", [])
    ]

    from wad_parser.types import LindefFlags, ThingFlags
    linedefs_raw = data.get("linedefs", [])
    linedefs = [
        Linedef(
            id=l["id"], v1=l["v1"], v2=l["v2"],
            flags=LindefFlags(),  # flags not needed for tag inference
            special=l["special"], tag=l["tag"],
            right_sidedef=l["right"], left_sidedef=l["left"],
        )
        for l in linedefs_raw
    ]
    sidedefs = [
        Sidedef(
            id=s["id"], x_offset=s["x_off"], y_offset=s["y_off"],
            upper_tex=s["upper"], lower_tex=s["lower"], mid_tex=s["mid"],
            sector=s["sector"],
        )
        for s in data.get("sidedefs", [])
    ]
    things = [
        Thing(
            id=t["id"], x=t["x"], y=t["y"], angle=t["angle"],
            type=t["type"], flags=ThingFlags(),
            type_name=t.get("type_name", ""),
            category=t.get("category", "unknown"),
        )
        for t in data.get("things", [])
    ]

    derived = data.get("derived", {})

    dm = DoomMap(
        meta=MapMeta(
            map_name=meta.get("map_name", ""),
            source_wad=meta.get("source_wad", ""),
            wad_type=meta.get("wad_type", "PWAD"),
            format=meta.get("format", "doom"),
            tags=meta.get("tags", []),
        ),
        bounds=BoundingBox(
            min_x=bounds_d.get("min_x", 0), max_x=bounds_d.get("max_x", 0),
            min_y=bounds_d.get("min_y", 0), max_y=bounds_d.get("max_y", 0),
        ),
        vertices=vertices,
        sectors=sectors,
        linedefs=linedefs,
        sidedefs=sidedefs,
        things=things,
    )
    dm.sector_graph    = derived.get("sector_graph", {})
    dm.sector_polygons = derived.get("sector_polygons", [])
    dm.trigger_map     = derived.get("trigger_map", {})
    dm.connectivity    = derived.get("connectivity", {})
    return dm


# ─── Interactive tagging ─────────────────────────────────────────────────────

def _interactive_tag(map_name: str, current_tags: list[str], inferred: list[str]) -> list[str]:
    """
    Prompt the user to confirm or edit tags for one map.

    Returns the final tag list.
    """
    click.echo()
    click.echo(f"{'─' * 60}")
    click.echo(f"  Map: {map_name}")
    click.echo(f"  Current tags : {', '.join(current_tags) if current_tags else '(none)'}")
    click.echo(f"  Auto-inferred: {', '.join(inferred) if inferred else '(none)'}")
    click.echo()
    click.echo("  Tag categories:")
    for category, values in TAG_TAXONOMY.items():
        click.echo(f"    {category:12s}: {', '.join(values)}")
    click.echo()

    default_str = ",".join(inferred) if inferred else ",".join(current_tags)
    prompt_text = (
        "  Enter tags (comma-separated)\n"
        "  Press Enter to accept auto-inferred, or 's' to skip, or 'q' to quit"
    )
    click.echo(prompt_text)
    if default_str:
        click.echo(f"  [default: {default_str}]")

    raw = click.prompt("  Tags", default=default_str, show_default=False).strip()

    if raw.lower() == "q":
        raise SystemExit(0)
    if raw.lower() == "s":
        return current_tags

    raw_tags = [t.strip().lower() for t in raw.split(",") if t.strip()]
    valid, unknown = validate_tags(raw_tags)
    if unknown:
        click.echo(f"  Warning: unknown tags ignored: {', '.join(unknown)}", err=True)
    return sorted(set(valid))


# ─── CLI ─────────────────────────────────────────────────────────────────────

@click.command()
@click.argument("path", metavar="PATH",
                type=click.Path(exists=True))
@click.option("--map",  "map_name", default=None,
              help="Process only this map name (e.g. MAP01).")
@click.option("--auto", is_flag=True,
              help="Auto-infer tags without interactive prompts.")
@click.option("--show", is_flag=True,
              help="Print current tags for each map without editing.")
@click.option("--merge/--replace", default=True,
              help="Merge inferred tags with existing (default) or replace them.")
def main(
    path: str,
    map_name: str | None,
    auto: bool,
    show: bool,
    merge: bool,
) -> None:
    """
    Annotate parsed map JSON files with taxonomy tags.

    PATH may be a directory of .json files or a single .json file.
    """
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.json"))
        if not files:
            click.echo("No JSON files found in directory.", err=True)
            sys.exit(1)
    else:
        files = [p]

    if map_name:
        target = map_name.upper()
        files = [f for f in files if f.stem.upper() == target]
        if not files:
            click.echo(f"No JSON file found for map '{map_name}'.", err=True)
            sys.exit(1)

    changed = 0
    for fpath in files:
        try:
            data = _load_map_json(fpath)
        except Exception as exc:
            click.echo(f"Error reading {fpath}: {exc}", err=True)
            continue

        meta = data.get("meta", {})
        current_tags: list[str] = meta.get("tags", [])
        fname = fpath.stem

        if show:
            click.echo(f"{fname:10s}  {', '.join(current_tags) if current_tags else '(untagged)'}")
            continue

        # Auto-infer
        inferred = _infer_from_dict(data)

        if auto:
            new_tags = sorted(set(inferred) | set(current_tags)) if merge else sorted(inferred)
        else:
            new_tags = _interactive_tag(fname, current_tags, inferred)
            if merge and current_tags:
                new_tags = sorted(set(new_tags) | set(current_tags))

        if set(new_tags) != set(current_tags):
            data.setdefault("meta", {})["tags"] = new_tags
            _save_map_json(fpath, data)
            changed += 1
            click.echo(f"  {fname}: {' '.join(new_tags)}", err=True)

    if not show:
        click.echo(f"\n{changed} file(s) updated.", err=True)


if __name__ == "__main__":
    main()

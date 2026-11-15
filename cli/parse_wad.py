"""CLI: parse a WAD into per-map JSON documents (plus one WAD-wide texture
set document).

    parse-wad DOOM2.WAD --output ./dataset/maps/
    parse-wad mymap.wad --map MAP03 --stdout
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.parse import quote

import click

from wad_parser.map_extractor import extract_all_maps
from wad_parser.texture_parser import build_texture_set
from wad_parser.wad_reader import WadFile


def _to_json(obj) -> str:
    return json.dumps(dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj,
                       indent=2, sort_keys=False)


def _write_if_changed(path: Path, text: str) -> None:
    """Repeated extraction of the same source does not rewrite its files."""
    if path.exists() and path.read_text() == text:
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@click.command()
@click.argument("wad_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "output_dir", type=click.Path(file_okay=False), default=None,
              help="Directory to write one JSON file per map, plus a *.textures.json file.")
@click.option("--map", "map_name", default=None, help="Only parse this one map (e.g. MAP03).")
@click.option("--stdout", "to_stdout", is_flag=True, help="Print JSON to stdout instead of writing files.")
def main(wad_path: str, output_dir: str | None, map_name: str | None, to_stdout: bool) -> None:
    wad = WadFile(wad_path)
    maps = extract_all_maps(wad, map_name=map_name)
    if map_name:
        maps = [m for m in maps if m.meta.map_name.upper() == map_name.upper()]
        if not maps:
            raise click.ClickException(f"Map {map_name!r} not found in {wad_path}")

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

    if to_stdout:
        for doc in maps:
            click.echo(_to_json(doc))
        if not map_name:  # WAD-wide document; skip when filtering to one map
            click.echo(_to_json(texture_set))
        return

    if not output_dir:
        raise click.ClickException("Specify --output <dir> or --stdout")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    destinations = set()
    for doc in maps:
        # A flat directory remains compatible with all existing dataset loaders.
        # Escape free-form UDMF marker names so they cannot become paths.
        dest = out / f"{wad.hash}_{quote(doc.meta.map_name, safe='')}.json"
        if dest in destinations:
            raise click.ClickException(f"duplicate map name in {wad_path}: {doc.meta.map_name!r}")
        destinations.add(dest)
        if dest.exists():
            try:
                old_meta = json.loads(dest.read_text())["meta"]
                same_map = (old_meta["wad_hash"] == wad.hash
                            and old_meta["map_name"] == doc.meta.map_name)
            except (ValueError, KeyError, TypeError):
                same_map = False
            if not same_map:
                raise click.ClickException(f"map filename collision: refusing to overwrite {dest}")
        # Exclude provenance and derived data: repackaged identical authored
        # maps should have the same content fingerprint. This records duplicate
        # candidates without silently removing either source's maps or tags.
        raw = doc.to_dict()
        content = {key: raw[key] for key in
                   ("vertices", "sectors", "linedefs", "sidedefs", "things")}
        content["format"] = doc.meta.format
        content["udmf_namespace"] = doc.extra.get("udmf_namespace")
        content["udmf_globals"] = doc.extra.get("udmf_globals")
        content_hash = hashlib.sha256(json.dumps(
            content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        entries.append({"map_name": doc.meta.map_name, "path": dest.name,
                        "format": doc.meta.format, "content_hash": content_hash})
    manifest_dir = out / ".manifests"
    manifest_path = manifest_dir / f"{wad.hash}.json"
    # --map is an incremental extraction, so keep previous manifest entries.
    by_name = {}
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text())
            if previous["wad_hash"] != wad.hash or previous["version"] != 1:
                raise ValueError("manifest identity/version mismatch")
            by_name = {entry["map_name"]: entry for entry in previous["maps"]}
        except (ValueError, KeyError, TypeError) as exc:
            raise click.ClickException(f"invalid extraction manifest {manifest_path}: {exc}") from exc
    by_name.update({entry["map_name"]: entry for entry in entries})
    manifest = {"version": 1, "wad_hash": wad.hash, "source_wad": str(wad_path),
                "maps": sorted(by_name.values(), key=lambda entry: entry["map_name"])}
    # Preflight the complete batch before writing any map documents.
    for doc, entry in zip(maps, entries):
        dest = out / entry["path"]
        _write_if_changed(dest, _to_json(doc))
        click.echo(f"wrote {dest}")
    manifest_dir.mkdir(exist_ok=True)
    _write_if_changed(manifest_path, _to_json(manifest))
    tex_dest = out / f"{wad.hash}.textures.json"
    _write_if_changed(tex_dest, _to_json(texture_set))
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

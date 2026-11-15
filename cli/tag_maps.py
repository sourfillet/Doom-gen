"""Tagging CLI: assign user-defined tags to WADs and maps.

    tag-maps tag mywad.wad techbase "slaughter:0.6"          # WAD-level
    tag-maps tag mywad.wad --map MAP03 city                  # map-level
    tag-maps show mywad.wad --map MAP03                      # effective tags
    tag-maps exclude mywad.wad --map MAP07 multiplayer       # drop inherited
    tag-maps preview mywad.wad --out previews/               # SVG line art
    tag-maps interactive mywad.wad                           # tag map-by-map
    tag-maps registry / rename / merge / describe            # vocabulary ops

Tags are created on first use; the registry powers autocomplete so
near-duplicates get reused instead of forking the vocabulary.
"""
from __future__ import annotations

from pathlib import Path

import click

from tagging.preview import map_to_svg
from tagging.registry import canonicalize
from tagging.store import TagStore
from wad_parser.map_extractor import extract_all_maps
from wad_parser.wad_reader import WadFile


def _parse_assignment(spec: str) -> tuple[str, float]:
    """"techbase" -> (techbase, 1.0); "hell:0.3" -> (hell, 0.3)."""
    if ":" in spec:
        name, _, w = spec.rpartition(":")
        try:
            return name, float(w)
        except ValueError:
            return spec, 1.0  # a colon inside the tag name itself
    return spec, 1.0


def _load_wad(wad_path: str) -> WadFile:
    return WadFile(wad_path)


def _format_tags(tags: dict[str, float]) -> str:
    if not tags:
        return "(none)"
    return ", ".join(t if w == 1.0 else f"{t}:{w:g}" for t, w in sorted(tags.items()))


def _format_detailed(tags: dict[str, tuple[float, str]]) -> str:
    """Auto-derived tags are marked with a trailing asterisk."""
    if not tags:
        return "(none)"
    parts = []
    for t, (w, source) in sorted(tags.items()):
        label = t if w == 1.0 else f"{t}:{w:g}"
        parts.append(label + ("*" if source == "auto" else ""))
    return ", ".join(parts)


@click.group()
@click.option("--tags-dir", default="tags", show_default=True,
              help="Directory holding the tag store (registry + assignments).")
@click.pass_context
def main(ctx, tags_dir: str) -> None:
    ctx.obj = TagStore(Path(tags_dir))


@main.command()
@click.argument("wad_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("tags", nargs=-1, required=True)
@click.option("--map", "map_name", default=None, help="Tag one map instead of the whole WAD.")
@click.pass_obj
def tag(store: TagStore, wad_path: str, tags: tuple[str, ...], map_name: str | None) -> None:
    """Assign TAGS (optionally NAME:WEIGHT) to a WAD or one of its maps."""
    wad = _load_wad(wad_path)
    for spec in tags:
        name, weight = _parse_assignment(spec)
        if map_name:
            canon = store.tag_map(wad.hash, map_name, name, weight)
            click.echo(f"{Path(wad_path).name} {map_name.upper()}: +{canon} ({weight:g})")
        else:
            canon = store.tag_wad(wad.hash, name, weight, wad_name=Path(wad_path).name)
            click.echo(f"{Path(wad_path).name}: +{canon} ({weight:g})")


@main.command()
@click.argument("wad_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("tags", nargs=-1, required=True)
@click.option("--map", "map_name", default=None)
@click.pass_obj
def untag(store: TagStore, wad_path: str, tags: tuple[str, ...], map_name: str | None) -> None:
    """Remove TAGS from a WAD or one of its maps.

    Removing an auto-derived tag records a veto instead (the auto layer is
    regenerated wholesale, so a plain delete would not survive re-runs).
    """
    wad = _load_wad(wad_path)
    for spec in tags:
        name, _ = _parse_assignment(spec)
        canon = canonicalize(name)
        removed = (store.untag_map(wad.hash, map_name, name) if map_name
                   else store.untag_wad(wad.hash, name))
        if removed:
            click.echo(f"-{canon}")
            continue
        auto_here = (store.map_auto_tags(wad.hash, map_name) if map_name
                     else store.wad_auto_tags(wad.hash))
        if canon in auto_here:
            store.deny_auto(wad.hash, canon, map_name)
            click.echo(f"-{canon} (auto tag - recorded as veto, survives regeneration)")
        elif map_name and canon in store.effective_tags(wad.hash, map_name):
            store.exclude_inherited(wad.hash, map_name, canon)
            click.echo(f"-{canon} (inherited from WAD - excluded for {map_name.upper()})")
        else:
            click.echo(f"{name}: not present")


@main.command()
@click.argument("wad_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--map", "map_name", default=None)
@click.pass_obj
def show(store: TagStore, wad_path: str, map_name: str | None) -> None:
    """Show tags: WAD-level, or one map's own/inherited/effective set."""
    wad = _load_wad(wad_path)
    if map_name is None:
        wad_detail = {t: (w, "user") for t, w in store.wad_tags(wad.hash).items()}
        for t, w in store.wad_auto_tags(wad.hash).items():
            wad_detail.setdefault(t, (w, "auto"))
        click.echo(f"WAD tags: {_format_detailed(wad_detail)}   (* = auto-derived)")
        for group in wad.find_map_groups():
            eff = store.effective_tags_detailed(wad.hash, group.name)
            click.echo(f"  {group.name}: {_format_detailed(eff)}")
        return
    rec = store.map_record(wad.hash, map_name)
    click.echo(f"own:       {_format_tags(store.map_own_tags(wad.hash, map_name))}")
    auto_own = store.map_auto_tags(wad.hash, map_name)
    if auto_own:
        click.echo(f"auto:      {_format_tags(auto_own)}")
    click.echo(f"inherited: {_format_tags({**store.wad_auto_tags(wad.hash), **store.wad_tags(wad.hash)})}")
    if rec.get("excluded_inherited"):
        click.echo(f"excluded:  {', '.join(rec['excluded_inherited'])}")
    if rec.get("denied_auto"):
        click.echo(f"vetoed:    {', '.join(rec['denied_auto'])}")
    click.echo(f"effective: {_format_detailed(store.effective_tags_detailed(wad.hash, map_name))}"
               "   (* = auto-derived)")


@main.command()
@click.argument("wad_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("tag_name")
@click.option("--map", "map_name", required=True)
@click.pass_obj
def exclude(store: TagStore, wad_path: str, tag_name: str, map_name: str) -> None:
    """Exclude an inherited WAD-level tag from one map."""
    wad = _load_wad(wad_path)
    store.exclude_inherited(wad.hash, map_name, tag_name)
    click.echo(f"{map_name.upper()}: excluded inherited {canonicalize(tag_name)!r}")


@main.command()
@click.argument("wad_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("tag_name")
@click.option("--map", "map_name", required=True)
@click.pass_obj
def include(store: TagStore, wad_path: str, tag_name: str, map_name: str) -> None:
    """Undo a previous exclude."""
    wad = _load_wad(wad_path)
    store.include_inherited(wad.hash, map_name, tag_name)
    click.echo(f"{map_name.upper()}: re-included {canonicalize(tag_name)!r}")


@main.command()
@click.pass_obj
def registry(store: TagStore) -> None:
    """List every known tag with usage count and description."""
    usage = store.tag_usage()
    if not store.registry.tags:
        click.echo("(no tags yet)")
        return
    for canon, entry in sorted(store.registry.tags.items()):
        desc = f"  - {entry.description}" if entry.description else ""
        click.echo(f"{canon}  (x{usage.get(canon, 0)}){desc}")


@main.command()
@click.argument("tag_name")
@click.argument("description")
@click.pass_obj
def describe(store: TagStore, tag_name: str, description: str) -> None:
    """Set a tag's free-text description (also the cold-start embedding text)."""
    store.registry.ensure(tag_name)
    store.registry.describe(tag_name, description)
    click.echo(f"{canonicalize(tag_name)}: description set")


@main.command()
@click.argument("old")
@click.argument("new")
@click.pass_obj
def rename(store: TagStore, old: str, new: str) -> None:
    """Rename a tag across the registry and every assignment file."""
    n = store.rename_tag(old, new)
    click.echo(f"renamed {canonicalize(old)!r} -> {canonicalize(new)!r} in {n} file(s)")


@main.command()
@click.argument("src")
@click.argument("dst")
@click.pass_obj
def merge(store: TagStore, src: str, dst: str) -> None:
    """Merge tag SRC into DST across the whole corpus."""
    n = store.merge_tags(src, dst)
    click.echo(f"merged {canonicalize(src)!r} into {canonicalize(dst)!r} in {n} file(s)")


@main.command()
@click.argument("wad_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--dry-run", is_flag=True, help="Show what would be derived without writing.")
@click.pass_obj
def auto(store: TagStore, wad_path: str, dry_run: bool) -> None:
    """Derive objective fact tags (deathmatch/coop/secrets/format tier/...).

    Facts shared by every map are hoisted to the WAD level. Auto tags are a
    regenerable layer: rerunning replaces them, user tags always win, and
    `untag` on an auto tag records a veto that survives regeneration.
    """
    from tagging.auto import AUTO_TAG_DESCRIPTIONS, derive_wad_facts

    wad = _load_wad(wad_path)
    docs = extract_all_maps(wad)
    wad_tags, per_map = derive_wad_facts(wad, docs)

    click.echo(f"WAD: {_format_tags(wad_tags)}")
    for map_name, tags in per_map.items():
        if tags:
            click.echo(f"  {map_name}: {_format_tags(tags)}")
    if dry_run:
        click.echo("(dry run - nothing written)")
        return
    store.write_auto(wad.hash, wad_tags, per_map, AUTO_TAG_DESCRIPTIONS)
    n_map_tags = sum(len(t) for t in per_map.values())
    click.echo(f"wrote {len(wad_tags)} WAD-level + {n_map_tags} map-level auto tags")


@main.command()
@click.argument("wad_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--map", "map_name", default=None, help="Only this map.")
@click.option("--out", "out_dir", default="previews", show_default=True)
@click.pass_obj
def preview(store: TagStore, wad_path: str, map_name: str | None, out_dir: str) -> None:
    """Write top-down SVG previews for a WAD's maps."""
    wad = _load_wad(wad_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for doc in extract_all_maps(wad):
        if map_name and doc.meta.map_name != map_name.upper():
            continue
        dest = out / f"{Path(wad_path).stem}_{doc.meta.map_name}.svg"
        dest.write_text(map_to_svg(doc))
        click.echo(f"wrote {dest}")


@main.command()
@click.argument("wad_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--out", "preview_dir", default="previews", show_default=True,
              help="Where per-map preview SVGs are written as you go.")
@click.pass_obj
def interactive(store: TagStore, wad_path: str, preview_dir: str) -> None:
    """Walk a WAD map by map: preview, current tags, prompt for new ones.

    At the prompt: comma-separated tags (NAME:WEIGHT allowed), empty to
    keep as-is, `?` to list known tags, `-NAME` to remove, `!NAME` to
    exclude an inherited tag, `q` to stop.
    """
    try:
        import readline

        def completer(text, state):
            matches = store.registry.complete(text)
            return matches[state] if state < len(matches) else None

        readline.set_completer(completer)
        readline.set_completer_delims(", ")
        readline.parse_and_bind("tab: complete")
    except ImportError:
        pass  # readline unavailable (non-POSIX) - prompts still work

    wad = _load_wad(wad_path)
    docs = extract_all_maps(wad)
    out = Path(preview_dir)
    out.mkdir(parents=True, exist_ok=True)
    wad_stem = Path(wad_path).stem

    click.echo(f"{len(docs)} map(s) in {Path(wad_path).name}; WAD tags: "
               f"{_format_tags(store.wad_tags(wad.hash))}")
    for doc in docs:
        name = doc.meta.map_name
        svg_path = out / f"{wad_stem}_{name}.svg"
        svg_path.write_text(map_to_svg(doc))
        click.echo(f"\n=== {name}  ({len(doc.sectors)} sectors, "
                   f"{len(doc.things)} things)  preview: {svg_path}")
        if doc.meta.mapinfo and doc.meta.mapinfo.level_name:
            click.echo(f"    level name: {doc.meta.mapinfo.level_name}")
        click.echo(f"    effective: {_format_detailed(store.effective_tags_detailed(wad.hash, name))}")
        while True:
            line = click.prompt("tags", default="", show_default=False).strip()
            if not line:
                break
            if line == "q":
                return
            if line == "?":
                click.echo("known: " + (", ".join(sorted(store.registry.tags)) or "(none)"))
                continue
            for spec in (s.strip() for s in line.split(",") if s.strip()):
                if spec.startswith("!"):
                    store.exclude_inherited(wad.hash, name, spec[1:])
                    click.echo(f"  excluded inherited {canonicalize(spec[1:])!r}")
                elif spec.startswith("-"):
                    removed = store.untag_map(wad.hash, name, spec[1:])
                    click.echo(f"  removed {canonicalize(spec[1:])!r}" if removed
                               else f"  {spec[1:]}: not present")
                else:
                    tag_name, weight = _parse_assignment(spec)
                    canon = store.tag_map(wad.hash, name, tag_name, weight)
                    click.echo(f"  +{canon} ({weight:g})")
            click.echo(f"    effective: {_format_detailed(store.effective_tags_detailed(wad.hash, name))}")


if __name__ == "__main__":
    main()

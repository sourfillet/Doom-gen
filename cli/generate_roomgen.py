"""Graph-native generation CLI (Phase 9.4): sample room-graphs from the
trained generator and write a playable PWAD with real rooms/corridors/doors.

    generate-roomgen --checkpoint checkpoints/roomgen_015000.pt \\
        --tags "doom2,single-player" --n 3 --game wads/DOOM2.WAD \\
        --style style.json --out roomgen.wad
"""
from __future__ import annotations

import json
from pathlib import Path

import click

from generation.game_profile import load_profile
from generation.pipeline import load_style, NoUsableMapsError
from generation.roomgen import generate
from cli.options import parse_tags as _parse_tags, generation_errors


@click.command()
@click.option("--checkpoint", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--tags", "tag_spec", default=None, help='e.g. "doom2,single-player"')
@click.option("--n", "n_maps", default=1, type=click.IntRange(min=1), show_default=True)
@click.option("--seed", default=None, type=int)
@click.option("--temperature", default=1.0, type=click.FloatRange(min=0, min_open=True), show_default=True,
              help="Sampling temperature; <1 tighter, >1 more varied.")
@click.option("--out", "out_path", default="generated_roomgen.wad", show_default=True)
@click.option('--report', 'report_path', default=None, type=click.Path(dir_okay=False),
              help='Save generation, encounter, sizing and rejection details to a new JSON file.')
@click.option('--target-port', type=click.Choice(['zdoom', 'vanilla']), default='zdoom', show_default=True,
              help='Vanilla requires compiled classic nodes and static index/BLOCKMAP limits.')
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
@click.option("--shape-library", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="Room polygon library for the real-shape assembler.")
@click.option('--geometry-profile', type=click.Choice(['classic', 'varied', 'compact', 'adjoining']), default='classic', show_default=True,
              help='Routed rooms: varied changes shapes/openings, compact reduces transit, adjoining grows eligible neighbors toward a shared wall band. Incompatible with --shape-library.')
@click.option('--gameplay', type=click.Choice(['none', 'intro']), default='none', show_default=True,
              help='Intro adds procedural cover, encounters and supply budgets; requires --style and excludes --placement.')
@click.option('--interior-profile', type=click.Choice(['columns', 'stepped']), default='columns', show_default=True,
              help='Stepped adds walkable raised terraces inside rooms; requires --gameplay intro.')
@click.option('--progression', type=click.Choice(['none', 'blue-key']), default='none', show_default=True,
              help='Blue-key requires an earlier key room and an existing door that gates the exit; requires intro gameplay.')
@click.option("--max-rooms", default=40, type=click.IntRange(min=1), show_default=True)
@click.option("--max-attempts", default=3, type=click.IntRange(min=1), show_default=True,
              help="Bounded attempts per requested map, including compiled-map failures; all rejections are reported.")
@click.option("--assets", "assets_dir", default=None, type=click.Path(exists=True, file_okay=False),
              help="Bundle custom flats/patches/textures into the PWAD (dir with "
                   "flats/, patches/, textures.json). Registered on --game so they "
                   "survive enforcement.")
@click.option("--sources", "sources_spec", default=None,
              help="Comma-separated WAD files or directories to automatically extract missing textures/flats from.")
@click.option("--companion-textures", default=None, type=click.Path(dir_okay=False),
              help="Write extracted assets to a separate companion resource WAD instead of embedding.")
@generation_errors
def main(checkpoint, tag_spec, n_maps, seed, temperature, out_path, report_path,
         style_path, game_path, properties_path, placement_path,
         shape_library, geometry_profile, gameplay, interior_profile, progression, max_rooms, max_attempts, target_port,
         assets_dir, sources_spec, companion_textures) -> None:
    from generation.placement import load_placement_model
    from generation.properties import load_property_model
    from wad_writer.assets import AssetBundle

    if gameplay == 'intro' and (not style_path or placement_path):
        raise click.UsageError('intro gameplay requires --style and excludes --placement')
    if interior_profile != 'columns' and gameplay != 'intro':
        raise click.UsageError('stepped interiors require --gameplay intro')
    if progression != 'none' and gameplay != 'intro':
        raise click.UsageError('blue-key progression requires --gameplay intro')
    if report_path and Path(report_path).exists():
        raise click.UsageError('report already exists; choose a new report path')
    if report_path and Path(report_path).resolve() == Path(out_path).resolve():
        raise click.UsageError('report and WAD output must use different paths')
    if companion_textures and Path(companion_textures).resolve() == Path(out_path).resolve():
        raise click.UsageError('companion textures and map WAD output must use different paths')
    assets = AssetBundle.from_dir(assets_dir) if assets_dir else None
    if assets and game_path and Path(game_path).suffix.lower() == '.wad':
        assets = assets.with_base_wad(game_path)
    sources = [s.strip() for s in sources_spec.split(',') if s.strip()] if sources_spec else None
    def save_report(result):
        if not report_path:
            return
        report = Path(report_path)
        report.parent.mkdir(parents=True, exist_ok=True)
        temporary = report.with_suffix(report.suffix + '.tmp')
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(report)
    try:
        result = generate(
            checkpoint, _parse_tags(tag_spec), n_maps=n_maps, out_path=out_path,
            seed=seed, temperature=temperature,
            style=load_style(style_path),
            game=load_profile(game_path),
            properties=load_property_model(properties_path) if properties_path else None,
            placement=load_placement_model(placement_path) if placement_path else None,
            shape_library=shape_library, geometry_profile=geometry_profile, gameplay=gameplay,
            interior_profile=interior_profile, progression=progression,
            max_rooms=max_rooms, max_attempts=max_attempts, target_port=target_port,
            assets=assets, sources=sources, companion_textures=companion_textures)
    except NoUsableMapsError as exc:
        save_report(exc.report)
        raise
    save_report(result)

    click.echo(f"wrote {result['path']}  maps: {', '.join(result['maps'])}")
    if result.get("companion_textures"):
        click.echo(f"wrote companion texture resource: {result['companion_textures']}")
    for s in result["summaries"]:
        click.echo(f"  {s['map']}: {s['rooms']} rooms -> {s['sectors']} sectors")
        plan = s.get('decoration', {}).get('gameplay')
        if plan:
            click.echo(f"    intro: {plan['monsters']} enemies, {plan['shells_supplied_before_combat']} opening shells, "
                       f"{plan['health_supplied']} recovery health, {len(plan['interiors']['columns'])} cover columns")
            if plan['interiors'].get('terraces'):
                click.echo(f"    interiors: {len(plan['interiors']['terraces'])} walkable terraces")
            if plan.get('progression', {}).get('profile') == 'blue-key':
                click.echo('    progression: collect the blue key to reach the exit')
    if result.get('failed_samples'):
        click.echo(f"failed attempts: {result['failed_samples']}")
    if not result["nodes_built"]:
        click.echo("nodes left empty (no zdbsp); the selected port must rebuild them")


if __name__ == "__main__":
    main()

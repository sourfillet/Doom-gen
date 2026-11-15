"""Style and populate validated generated layouts without resampling a model.

    python -m tools.build_gameplay_demo --input generated/layout.wad \
        --style style.json --game doom2-profile.json --out generated/gameplay

The original layout remains an exact comparison input. Population, supplies,
lighting and cover are procedural; the underlying room graph is not retrained.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import shutil
from pathlib import Path


def build(source, style_path, game_path, out, *, seed=7, interior_profile='columns', progression='none'):
    from generation.game_profile import load_profile
    from generation.gameplay import apply_intro
    from generation.pipeline import documents_to_wad, map_wads_to_wad
    from generation.style import StyleModel, apply_style
    from generation.validator import require_valid
    from models.training import code_fingerprint
    from wad_parser.map_extractor import extract_all_maps
    from wad_parser.wad_reader import WadFile
    from wad_writer.compatibility import require_target_port
    from generation.interiors import INTERIOR_PROFILES

    if interior_profile not in INTERIOR_PROFILES:
        raise ValueError('interior_profile must be columns or stepped')
    if progression not in {'none', 'blue-key'}:
        raise ValueError('progression must be none or blue-key')
    source, style_path, game_path, out = map(Path, (source, style_path, game_path, out))
    if out.exists() and any(out.iterdir()):
        raise ValueError('gameplay demo output directory must be empty')
    source_bytes, style_bytes, game_bytes = source.read_bytes(), style_path.read_bytes(), game_path.read_bytes()
    wad = WadFile(str(source))
    require_target_port(wad, 'zdoom')
    originals = extract_all_maps(wad)
    if not originals:
        raise ValueError('no maps in input WAD')
    style, game = StyleModel.load(style_path), load_profile(game_path)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'input.wad').write_bytes(source_bytes)
    (out / 'style.json').write_bytes(style_bytes)
    (out / ('game' + game_path.suffix)).write_bytes(game_bytes)
    tool_bytes = Path(__file__).read_bytes()
    (out / 'build_gameplay_demo.py').write_bytes(tool_bytes)
    signature = code_fingerprint()
    repository = Path(__file__).resolve().parents[1]
    for directory in ('data', 'models', 'cli', 'wad_parser', 'wad_writer', 'generation', 'roomgraph', 'tagging'):
        for path in (repository / directory).rglob('*.py'):
            target = out / 'library' / path.relative_to(repository)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    if code_fingerprint(root=out / 'library') != signature:
        raise ValueError('source changed while snapshotting the gameplay build')
    report = {'status': 'running', 'input_sha256': hashlib.sha256(source_bytes).hexdigest(),
              'style_sha256': hashlib.sha256(style_bytes).hexdigest(),
              'game_sha256': hashlib.sha256(game_bytes).hexdigest(),
              'tool_sha256': hashlib.sha256(tool_bytes).hexdigest(), 'code_fingerprint': signature,
              'seed': seed, 'interior_profile': interior_profile, 'progression': progression, 'records': [],
              'limits': ['Same supplied layouts, procedural style/interiors/encounters; no retraining claim.',
                         'Resource budgets are explicit planning assumptions, not a proof of combat balance.',
                         'Static publication is separate from pickup/combat/exit engine verification.']}
    def save():
        report['code_changed_during_build'] = (code_fingerprint() != signature
                                              or Path(__file__).read_bytes() != tool_bytes)
        temporary = out / 'report.json.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(out / 'report.json')
    save()
    accepted = []
    for index, original in enumerate(originals):
        row = {'source_map': original.meta.map_name, 'written': False, 'seed': seed + index}
        report['records'].append(row)
        doc = deepcopy(original)
        try:
            require_valid(doc, require_connected=True)
            decoration = apply_style(doc, style, seed=seed + index, place_things=False, place_dm_starts=False)
            decoration['game'] = game.enforce(doc)
            palette = decoration.get('wall_palette', [])
            accent = game.sub_texture(palette[-1]) if palette else game.default_wall
            decoration['gameplay'] = apply_intro(doc, seed=seed + index, game=game, accent_texture=accent,
                                                interior_profile=interior_profile, required_progression=progression)
            if progression == 'blue-key':
                from generation.keyed import add_blue_key
                add_blue_key(doc, seed=seed + index, game=game)
            require_valid(doc, require_connected=True)
            doc.meta.map_name = f'MAP{len(accepted) + 1:02d}'
            artifact = out / f'{doc.meta.map_name}.wad'
            result = documents_to_wad([doc], artifact, require_playable=True)
            row.update(written=True, map=doc.meta.map_name, path=artifact.name,
                       nodes_built=result['nodes_built'], sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                       decoration=decoration,
                       appearance={'wall_textures': sorted({s.mid_tex for s in doc.sidedefs if s.mid_tex not in {'-', ''}}),
                                   'floor_flats': sorted({s.floor_flat for s in doc.sectors}),
                                   'light_levels': sorted({s.light_level for s in doc.sectors})})
            (out / f'{doc.meta.map_name}.json').write_text(json.dumps(doc.to_dict(), indent=2) + '\n')
            accepted.append(artifact)
        except (ValueError, RuntimeError) as exc:
            row['reason'] = str(exc)
        save()
        print(f"{row['source_map']}: " + ('published' if row['written'] else row['reason']), flush=True)
    if accepted:
        try:
            result = map_wads_to_wad(accepted, out / 'gameplay-panel.wad')
            report['panel'] = {'path': 'gameplay-panel.wad', 'maps': len(accepted),
                               'nodes_built': result['nodes_built'],
                               'sha256': hashlib.sha256((out / 'gameplay-panel.wad').read_bytes()).hexdigest()}
        except (ValueError, RuntimeError) as exc:
            report['panel'] = {'maps': 0, 'publication_error': str(exc)}
    report['status'] = 'complete'
    save()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--style', required=True, type=Path)
    parser.add_argument('--game', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--interior-profile', choices=['columns', 'stepped'], default='columns')
    parser.add_argument('--progression', choices=['none', 'blue-key'], default='none')
    args = parser.parse_args()
    build(args.input, args.style, args.game, args.out, seed=args.seed, interior_profile=args.interior_profile,
          progression=args.progression)

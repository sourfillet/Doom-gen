"""Verify opening pickups, ordinary-input combat and exits in a saved demo.

    xvfb-run -a python -m tools.playtest_gameplay --demo generated/gameplay \
        --out generated/gameplay-engine --iwad wads/DOOM2.WAD

Requires the optional engine dependency and the checkout's test controllers.
No console cheats, invulnerability, grants, kill commands or noclip are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import zlib


def save_frame(path, state):
    pixels = state.screen_buffer
    if pixels.shape[0] == 3:
        pixels = pixels.transpose(1, 2, 0)
    height, width = pixels.shape[:2]
    raw = b''.join(b'\x00' + row.tobytes() for row in pixels)
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    Path(path).write_bytes(b'\x89PNG\r\n\x1a\n'
        + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
        + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


def playtest(demo, out, *, iwad=None, seed=7):
    repository = Path(__file__).resolve().parents[1]
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    from tests.engine_combat import open_game, CombatController
    from models.training import code_fingerprint
    from wad_parser.wad_reader import WadFile
    from wad_parser.map_extractor import extract_all_maps
    from wad_writer.compatibility import require_target_port
    import vizdoom as v
    demo, out = Path(demo).resolve(), Path(out).resolve()
    iwad = Path(iwad).resolve() if iwad else Path(v.__file__).parent / 'freedoom2.wad'
    if out.exists() and any(out.iterdir()):
        raise ValueError('engine playtest output directory must be empty')
    raw = (demo / 'report.json').read_bytes()
    build = json.loads(raw)
    if build['status'] != 'complete' or build['code_changed_during_build']:
        raise ValueError('playtest needs a completed stable demo')
    panel = demo / build['panel']['path']
    if hashlib.sha256(panel.read_bytes()).hexdigest() != build['panel']['sha256']:
        raise ValueError('demo panel changed since publication')
    wad = WadFile(str(panel))
    require_target_port(wad, 'zdoom')
    docs = extract_all_maps(wad)
    plans = {r['map']: r['decoration']['gameplay'] for r in build['records'] if r['written']}
    source_maps = {r['map']: r['source_map'] for r in build['records'] if r['written']}
    out.mkdir(parents=True, exist_ok=True)
    signature = code_fingerprint()
    tool_bytes = Path(__file__).read_bytes()
    (out / 'playtest_gameplay.py').write_bytes(tool_bytes)
    helpers = {repository / 'tests' / name: (repository / 'tests' / name).read_bytes()
               for name in ('engine_combat.py', 'engine_navigation.py')}
    for helper, helper_bytes in helpers.items():
        (out / helper.name).write_bytes(helper_bytes)
    report = {'status': 'running', 'input_report_sha256': hashlib.sha256(raw).hexdigest(),
              'wad_sha256': wad.hash, 'iwad': str(iwad), 'iwad_sha256': hashlib.sha256(iwad.read_bytes()).hexdigest(),
              'seed': seed, 'skill': 3, 'records': [], 'code_fingerprint': signature,
              'controller_sha256': hashlib.sha256(helpers[repository / 'tests/engine_combat.py']).hexdigest(),
              'helpers_sha256': {p.name: hashlib.sha256(data).hexdigest() for p, data in helpers.items()},
              'limits': ['Controller uses actor positions and geometry with ordinary movement/aim/fire/use inputs.',
                         'Fixed-seed bot completion is not a human balance/fun assessment.']}
    def save():
        report['code_changed_during_playtest'] = (code_fingerprint() != signature
            or Path(__file__).read_bytes() != tool_bytes
            or any(path.read_bytes() != data for path, data in helpers.items()))
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    previous_cwd = Path.cwd()
    try:
        os.chdir(out)
        for doc in docs:
            before = open_game(demo / 'input.wad', source_maps[doc.meta.map_name], iwad=iwad, seed=seed,
                               config_path=out / 'vizdoom-before.ini')
            try:
                save_frame(out / f'{doc.meta.map_name}-before.png', before.get_state())
            finally:
                before.close()
            game = open_game(panel, doc.meta.map_name, iwad=iwad, seed=seed,
                             config_path=out / 'vizdoom.ini',
                             sectors_info=bool(plans[doc.meta.map_name].get('progression')))
            try:
                save_frame(out / f'{doc.meta.map_name}-start.png', game.get_state())
                controller = CombatController(game, doc)
                controller.capture = lambda stage, state: save_frame(out / f'{doc.meta.map_name}-{stage}.png', state)
                result = controller.complete(plans[doc.meta.map_name])
                result['map'] = doc.meta.map_name
            except Exception as exc:
                result = {'completed': False, 'map': doc.meta.map_name,
                          'reason': f'{type(exc).__name__}: {exc}'}
            finally:
                game.close()
            result['native_crash'] = (out / 'vizdoom-crash.log').exists()
            if result['native_crash']:
                result.update(completed=False, reason='native backend crash')
            report['records'].append(result)
            save()
            print(result, flush=True)
        report['status'] = 'complete'
        save()
        cards = []
        for row in report['records']:
            name = row['map']
            cards.append(f'<h2>{name}</h2><p>Exit completed: {row["completed"]}; kills: {row.get("kills", "unknown")}</p>'
                + ''.join(f'<figure><img src="{name}-{stage}.png"><figcaption>{label}</figcaption></figure>'
                          for stage, label in [('before', 'Original structural control'),
                                               ('start', 'Styled and populated start'), ('combat', 'Ordinary-input combat')]
                             + [(f'terrace-{index + 1}', f'Walkable terrace {index + 1}')
                                for index in range(len(plans[name].get('interiors', {}).get('terraces', [])))]
                             + [('key', 'Blue key collected'), ('gate', 'Required blue door')]
                          if (out / f'{name}-{stage}.png').exists()))
        (out / 'comparison.html').write_text('<!doctype html><meta charset="utf-8"><title>Gameplay comparison</title>'
            '<style>body{background:#171719;color:#eee;font:16px sans-serif;padding:24px}figure{display:inline-block;'
            'margin:8px}img{width:320px;image-rendering:pixelated}figcaption{max-width:320px}</style>'
            '<h1>Matched-layout gameplay comparison</h1><p>Materials, lighting, cover, encounters and resources '
            'are procedural additions. Fixed-seed bot completion is not a human balance judgment.</p>' + ''.join(cards))
    finally:
        os.chdir(previous_cwd)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--iwad', type=Path)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    result = playtest(args.demo, args.out, iwad=args.iwad, seed=args.seed)
    if not result['records'] or not all(row['completed'] for row in result['records']):
        raise SystemExit(1)

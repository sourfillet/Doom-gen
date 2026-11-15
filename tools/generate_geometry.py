"""Publish checkpoint-generated geometry using target assets and compiled checks.

    python -m tools.generate_geometry --checkpoint training/geometry_001000.pt \
        --game /path/to/DOOM2.WAD --out generated/learned.wad \
        --report-dir generated/learned-report --maps 4 --max-attempts 1

Requires zdbsp on PATH. Every bounded attempt is retained. No procedural layout,
geometry/exit repair, population or asset substitutions are applied.
"""
from __future__ import annotations

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--game', required=True, help='Target IWAD or saved GameProfile JSON.')
    parser.add_argument('--out', dest='out_path', required=True)
    parser.add_argument('--report-dir', required=True)
    parser.add_argument('--maps', dest='n_maps', type=int, default=1)
    parser.add_argument('--max-attempts', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--tags-json', default='{}')
    parser.add_argument('--target-game', default='')
    parser.add_argument('--target-port', choices=['zdoom', 'vanilla'], default='zdoom')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--reject-flat-rectangles', action='store_true',
                        help='Reject single-flat-rectangle outputs instead of treating exit validity as layout quality.')
    args = vars(parser.parse_args())
    from generation.game_profile import load_profile
    from generation.geometry_model import generate_checkpoint
    context = {'tags': json.loads(args.pop('tags_json')), 'target_game': args.pop('target_game')}
    args['game'] = load_profile(args['game'])
    try:
        report = generate_checkpoint(args.pop('checkpoint'), context, **args)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps({'publication': report['publication'], 'accepted_maps': report['accepted_maps'],
                      'failed_attempts': report['failed_attempts'], 'engine': report['engine']}))


if __name__ == '__main__':
    main()

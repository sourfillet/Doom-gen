"""Retain a fixed panel of checkpoint-generated geometry diagnostics.

    python -m tools.sample_geometry --checkpoint generated/training/geometry_000500.pt \
        --out generated/geometry-samples --samples 8 --max-tokens 4096

Every requested sample is kept, including invalid/incomplete token streams.
Candidate WADs are diagnostic, uncompiled files; this is not playable publication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sample_panel(checkpoint, out, *, samples=8, seed=0, max_tokens=4096, temperature=1.0,
                 tags=None, target_game='', device='cpu', syntax_masks=True, threads=1, raw_wads=False):
    import torch
    from data.learned_geometry import validate_conditioning
    from generation.geometry_model import infer, load_model
    from generation.learned_geometry import decode_prediction
    from wad_writer.wad_writer import write_wad
    if samples < 1 or threads < 1:
        raise ValueError('sample count and threads must be positive')
    context = {'tags': {} if tags is None else tags, 'target_game': target_game}
    validate_conditioning(context)
    out, checkpoint = Path(out), Path(checkpoint)
    if out.exists() and any(out.iterdir()):
        raise ValueError('sample output must be empty; choose a fresh directory')
    torch.set_num_threads(threads)
    raw = checkpoint.read_bytes()
    model, conditioner, config = load_model(checkpoint, device=device)
    out.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'doom_geometry_panel_v1', 'checkpoint_sha256': hashlib.sha256(raw).hexdigest(),
              'config': config, 'context': context, 'attempts_per_sample': 1,
              'node_building': 'not_run', 'engine': 'not_run', 'records': []}
    if raw_wads:
        report['raw_diagnostic_wads'] = True
    (out / 'sample_geometry.py').write_bytes(Path(__file__).read_bytes())
    for index in range(samples):
        result = infer(model, conditioner, context, seed=seed + index, max_tokens=max_tokens,
                       temperature=temperature, syntax_masks=syntax_masks)
        if result['prediction'] is not None:
            try:
                doc = decode_prediction(result['prediction'])
                path = out / f'{index:04d}-candidate.wad'
                write_wad([doc], path)
                result['diagnostic_wad'] = path.name
                result['diagnostic_wad_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            except Exception as exc:
                import traceback
                result['candidate_serialization_error'] = f'{type(exc).__name__}: {exc}'
                result['candidate_serialization_traceback'] = traceback.format_exc()
        if raw_wads:
            from tools.dump_geometry import dump_result
            result['raw_wad'] = dump_result(result, out / f'{index:04d}-raw.wad',
                                            provenance={'checkpoint_sha256': report['checkpoint_sha256'],
                                                        'context': context})
        path = out / f'{index:04d}.json'
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        report['records'].append({'seed': seed + index, 'status': result['status'], 'report': path.name,
                                  'tokens': len(result['sample']['tokens'])})
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    if checkpoint.read_bytes() != raw:
        raise ValueError('checkpoint changed during sampling; retain the diagnostic panel')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--tags-json', default='{}')
    parser.add_argument('--target-game', default='')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--no-syntax-masks', dest='syntax_masks', action='store_false')
    parser.add_argument('--raw-wads', action='store_true',
                        help='Also preserve rejected authoring bytes or incomplete trace-only WADs; not playable publication.')
    args = vars(parser.parse_args())
    args['tags'] = json.loads(args.pop('tags_json'))
    try:
        report = sample_panel(**args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    from collections import Counter
    print(dict(Counter(row['status'] for row in report['records'])))


if __name__ == '__main__':
    main()

"""Run the existing evaluator with verified prepared source targets.

    python -m tools.evaluate_prepared_roomgen --corpus generated/complete-corpus \
        --checkpoint generated/training/best.pt --out generated/evaluation

Only source segmentation/ending assessment is reused. Sampling, baselines,
copy checks, geometry realization, validation and publication use the ordinary
evaluator. Source documents are still verified and used for raw copy probes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile


def evaluate(directory, checkpoint, out, *, samples=16, seed=0, max_rooms=64,
             temperature=1.0, write_maps=8):
    import torch
    import roomgraph.segment as segmentation
    import data.source_endings as endings
    from cli.evaluate_roomgen import main
    from models.training import code_fingerprint
    from roomgraph.types import RoomGraph
    from tools.train_prepared_roomgen import prepared_pools

    directory, out = Path(directory), Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('evaluation output directory must be empty; choose a new directory')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    helper_path = Path(__file__).with_name('train_prepared_roomgen.py')
    helper_bytes = helper_path.read_bytes()
    helper_sha256 = hashlib.sha256(helper_bytes).hexdigest()
    saved_helper = payload.get('settings', {}).get('training_tool_sha256')
    if saved_helper is not None and saved_helper != helper_sha256:
        raise ValueError('prepared training/cache-loader tool changed; use the archived tool')
    with tempfile.TemporaryDirectory(prefix='doom-gen-prepared-eval-') as temporary:
        path = Path(temporary) / 'manifest.json'
        path.write_text(json.dumps(payload['corpus']))
        _, _, signatures = prepared_pools(directory, manifest_path=path,
            include_incomplete=payload.get('settings', {}).get('include_incomplete', False))
    for name, value in signatures.items():
        saved = payload.get('settings', {}).get(name)
        if saved is not None and saved != value:
            raise ValueError('prepared evaluation inputs differ from training')
    prepared = json.loads((directory / 'report.json').read_text())
    rows = {}
    for meta in prepared['records']:
        key = meta['wad_hash'], meta['map']
        if key in rows:
            raise ValueError('duplicate source map identity in prepared corpus')
        raw = (directory / 'records' / meta['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != meta['target_snapshot_sha256']:
            raise ValueError('prepared evaluation snapshot changed')
        rows[key] = json.loads(raw)
    original_segment, original_ending = segmentation.segment, endings.assess_source_ending
    def cached_segment(doc, *args, **kwargs):
        if args or not kwargs.get('playable_only', False):
            return original_segment(doc, *args, **kwargs)
        return RoomGraph.from_dict(rows[doc.meta.wad_hash, doc.meta.map_name]['graph_snapshot'])
    def cached_ending(doc, graph, *, version=endings.SOURCE_ENDINGS_VERSION):
        if version != prepared['source_endings']:
            raise ValueError('prepared source-ending version differs from evaluation')
        return rows[doc.meta.wad_hash, doc.meta.map_name]['ending']
    fingerprint = code_fingerprint()
    tool_bytes = Path(__file__).read_bytes()
    segmentation.segment, endings.assess_source_ending = cached_segment, cached_ending
    try:
        main.main(['--maps-dir', str(directory / 'maps'), '--checkpoint', str(checkpoint),
                   '--out', str(out), '--samples', str(samples), '--seed', str(seed),
                   '--max-rooms', str(max_rooms), '--temperature', str(temperature),
                   '--write-maps', str(write_maps)], standalone_mode=False)
    finally:
        segmentation.segment, endings.assess_source_ending = original_segment, original_ending
    report = json.loads((out / 'report.json').read_text())
    report['prepared_targets'] = {**signatures, 'source_segmentation_and_endings_reused': True,
        'evaluation_tool_sha256': hashlib.sha256(tool_bytes).hexdigest(),
        'prepared_loader_sha256': helper_sha256,
        'source_edits_during_evaluation': code_fingerprint() != fingerprint
            or Path(__file__).read_bytes() != tool_bytes or helper_path.read_bytes() != helper_bytes}
    (out / 'evaluate_prepared_roomgen.py').write_bytes(tool_bytes)
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-rooms', type=int, default=64)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--write-maps', type=int, default=8)
    args = parser.parse_args()
    evaluate(args.corpus, args.checkpoint, args.out, samples=args.samples, seed=args.seed,
             max_rooms=args.max_rooms, temperature=args.temperature, write_maps=args.write_maps)

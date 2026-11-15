"""Train an unconditional room model using verified prepared target snapshots.

    python -m tools.train_prepared_roomgen --corpus generated/complete-corpus \
        --out generated/training --include-incomplete --steps 1000

Uses the existing model, censored-ending loss, validation and exact-resume loop.
Cached inputs must match their document/target hashes and audited code/runtime.
Opt-in `--stop-targets` chooses which sequences train the stop hazard (see
models.roomgen.stopping), and `--complete-repeat N` lists each complete training
map N times in the uniformly sampled pool. Defaults keep the original objective.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

from models.roomgen.stopping import STOP_TARGETS


def prepared_pools(directory, *, manifest_path=None, include_incomplete=False):
    from data.corpus import training_split
    from data.roomgraph_targets import graph_to_sequence
    from data.source_endings import SOURCE_ENDINGS_VERSION
    from models.training import source_preprocessing_fingerprint
    from roomgraph.types import RoomGraph

    directory = Path(directory)
    report_bytes = (directory / 'report.json').read_bytes()
    report = json.loads(report_bytes)
    runtime = {'python': platform.python_version()}
    for name in ('numpy', 'shapely', 'networkx'):
        runtime[name] = importlib.metadata.version(name)
    if (report.get('status') != 'complete' or report.get('code_changed_during_materialization')
            or report.get('code_fingerprint') != source_preprocessing_fingerprint(
                report.get('code_fingerprint_scope', 'repository_v1'))
            or report.get('source_endings') != SOURCE_ENDINGS_VERSION
            or report.get('runtime') != runtime):
        raise ValueError('prepared corpus assessment/code/runtime does not match current training inputs')
    splits, manifest = training_split(directory / 'maps', manifest_path or directory / 'corpus.json')
    metadata = {row['path']: row for row in report['records']}
    document_hashes = {row['path']: row['document_hash'] for row in manifest['entries']}
    pools, digest = {}, hashlib.sha256()
    for split, paths in splits.items():
        pool = []
        for path in paths:
            meta = metadata[path.name]
            if meta['document_sha256'] != document_hashes[path.name]:
                raise ValueError(f'prepared document metadata mismatch: {path.name}')
            raw = (directory / 'records' / path.name).read_bytes()
            signature = hashlib.sha256(raw).hexdigest()
            if signature != meta['target_snapshot_sha256']:
                raise ValueError(f'prepared target snapshot changed: {path.name}')
            row = json.loads(raw)
            if (row['status'] != 'assessed' or row.get('graph_snapshot_version') != 1
                    or row.get('ending') != meta['ending']
                    or row.get('document_sha256') != meta['document_sha256']
                    or row.get('complete') != meta['ending']['complete']):
                raise ValueError(f'prepared target metadata mismatch: {path.name}')
            if graph_to_sequence(RoomGraph.from_dict(row['graph_snapshot'])) != row['sequence']:
                raise ValueError(f'prepared graph/sequence mismatch: {path.name}')
            digest.update(path.name.encode())
            digest.update(signature.encode())
            complete = bool(row['ending']['complete'])
            if complete or (split == 'train' and include_incomplete):
                pool.append((row['sequence'], {}, complete))
        pools[split] = pool
    if not any(record[2] for record in pools['train']):
        raise ValueError('prepared training pool has no complete source endings')
    signature = {'prepared_report_sha256': hashlib.sha256(report_bytes).hexdigest(),
                 'prepared_targets_sha256': digest.hexdigest()}
    return pools, manifest, signature


def train(directory, out, *, manifest_path=None, include_incomplete=False, seed=0,
          hidden=64, steps=250, max_train_rooms=128, validate_every=50, resume=None,
          stop_targets='all', complete_repeat=1):
    if hidden < 1 or steps < 1 or max_train_rooms < 0 or validate_every < 1 or complete_repeat < 1:
        raise ValueError('hidden/steps/validate_every/complete_repeat must be positive; '
                         'max_train_rooms must be non-negative')
    if stop_targets not in STOP_TARGETS:
        raise ValueError(f'stop_targets must be one of {STOP_TARGETS}')
    import torch
    from data.roomgraph_targets import ROOMGRAPH_TARGET_VERSION
    from data.source_endings import SOURCE_ENDINGS_VERSION
    from models.conditioning import TagConditioner
    from models.roomgen.model import RoomGraphRNN
    from models.training import fit, load_resume, seed_all
    from models.training import code_fingerprint
    from roomgraph.segment import SEGMENTATION_VERSION
    from wad_parser.geometry import DERIVED_VERSION
    from wad_parser.traversal import TRAVERSAL_VERSION

    checkpoint = load_resume(resume, 'cpu')
    if checkpoint:
        include_incomplete = checkpoint['settings']['include_incomplete']
    pools, manifest, signatures = prepared_pools(directory, manifest_path=manifest_path,
                                                include_incomplete=include_incomplete)
    tool_bytes = Path(__file__).read_bytes()
    fingerprint = code_fingerprint('roomgen')
    signatures['training_tool_sha256'] = hashlib.sha256(tool_bytes).hexdigest()
    settings = (checkpoint['settings'] if checkpoint else
                {'seed': seed, 'lr': 0.001, 'cfg_dropout': 0.15, 'max_train_rooms': max_train_rooms,
                 'include_incomplete': include_incomplete, **signatures})
    if not checkpoint and (stop_targets, complete_repeat) != ('all', 1):
        # Recorded only when used, so default checkpoints keep their settings.
        settings.update(stop_targets=stop_targets, complete_repeat=complete_repeat)
    elif checkpoint and (stop_targets, complete_repeat) != ('all', 1) and (stop_targets, complete_repeat) != (
            settings.get('stop_targets', 'all'), settings.get('complete_repeat', 1)):
        raise ValueError('resume keeps the checkpoint stop settings; omit --stop-targets/--complete-repeat')
    if any(settings.get(key) != value for key, value in signatures.items()):
        raise ValueError('prepared inputs or training tool changed; exact resume is unavailable')
    seed_all(settings['seed'])
    config = (checkpoint['config'] if checkpoint else {'kind': 'roomgen', 'hidden': hidden,
        'target_version': ROOMGRAPH_TARGET_VERSION, 'derived_version': DERIVED_VERSION,
        'stop_loss': 'sequence_nll_v1', 'source_endings': SOURCE_ENDINGS_VERSION,
        'traversal_version': TRAVERSAL_VERSION, 'segmentation_version': SEGMENTATION_VERSION,
        'physical_planarity': True})
    model = RoomGraphRNN(hidden=config['hidden'], target_version=config['target_version'])
    conditioner = TagConditioner(256)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(conditioner.parameters()), lr=settings['lr'])
    from models.roomgen.stopping import stop_labels
    def loss_fn(record, training):
        sequence, tags, complete = record
        limit = settings['max_train_rooms']
        terminal, stop = stop_labels(complete, len(sequence), limit, settings.get('stop_targets', 'all'))
        prefix = sequence[:limit] if limit else sequence
        condition = conditioner.encode([tags], 'cpu', cfg_dropout=settings['cfg_dropout'] if training else 0,
                                       register_new=training)[0]
        return model.loss(prefix, condition, terminal=terminal, stop=stop)
    counts = {split: {'graphs': len(pool), 'complete': sum(r[2] for r in pool)} for split, pool in pools.items()}
    repeat = settings.get('complete_repeat', 1)
    train_pool = [record for record in pools['train'] for _ in range(repeat if record[2] else 1)]
    if repeat > 1:
        counts['train']['sampled_pool'] = len(train_pool)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # fit performs the fresh/resume checks before writing metrics/checkpoints.
    result = fit(model, conditioner, optimizer, train_pool, pools['validation'], loss_fn,
                 config=config, settings=settings, manifest=manifest, out_dir=out, steps=steps,
                 save_every=steps, validate_every=validate_every, resume=checkpoint)
    (out / 'train_prepared_roomgen.py').write_bytes(tool_bytes)
    summary = {**result, 'pools': counts, **signatures, 'unconditional': True,
               'code_changed_during_training': code_fingerprint('roomgen') != fingerprint
                   or Path(__file__).read_bytes() != tool_bytes,
               'limits': ['Filtered source coverage and small holdouts limit quality claims.',
                          'Geometry realization is procedural and must be evaluated separately.']}
    (out / 'prepared-training.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--include-incomplete', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--steps', type=int, default=250)
    parser.add_argument('--max-train-rooms', type=int, default=128)
    parser.add_argument('--validate-every', type=int, default=50)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-targets', choices=STOP_TARGETS, default='all',
                        help='complete: only complete maps train the stop hazard (fragments train structure only); '
                             'reachable-ends: every map ends after its last reachable room')
    parser.add_argument('--complete-repeat', type=int, default=1,
                        help='list each complete training map this many times in the sampled pool')
    args = parser.parse_args()
    print(json.dumps(train(args.corpus, args.out, manifest_path=args.manifest,
        include_incomplete=args.include_incomplete, seed=args.seed, hidden=args.hidden,
        steps=args.steps, max_train_rooms=args.max_train_rooms,
        validate_every=args.validate_every, resume=args.resume, stop_targets=args.stop_targets,
        complete_repeat=args.complete_repeat), indent=2))

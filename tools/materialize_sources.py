"""Prepare complete source examples from a frozen audit, with family holdouts.

    python -m tools.materialize_sources --audit generated/source-audit/report.json \
        --out generated/complete-corpus

By default only complete maps are selected; --include-fragments also exports
assessed fragments with their ending evidence retained. Verify and reuse audited
document/target snapshots when available; otherwise re-extract and re-assess.
Changed sources, code, runtime and eligibility are rejected.
This does not certify combat, resource balance, or unsupported mechanics.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

from tools.audit_sources import _save, _run_job


def materialize(source, out, *, seed=0, families=None, include_fragments=False, workers=2, timeout=120,
                validation_fraction=0.1, test_fraction=0.1):
    from data.corpus import build_manifest
    from data.source_endings import SOURCE_ENDINGS_VERSION
    from models.training import source_preprocessing_fingerprint
    if workers < 1 or timeout <= 0:
        raise ValueError('workers and timeout must be positive')
    if not (0 <= validation_fraction < 1 and 0 <= test_fraction < 1
            and validation_fraction + test_fraction < 1):
        raise ValueError('holdout fractions must be non-negative and sum to less than one')

    source, out = Path(source), Path(out).resolve()
    input_bytes = source.read_bytes()
    audit = json.loads(input_bytes)
    scope = audit.get('code_fingerprint_scope', 'repository_v1')
    fingerprint = source_preprocessing_fingerprint(scope)
    if audit.get('status') != 'complete' or audit.get('code_changed_during_audit'):
        raise ValueError('materialization requires a completed audit with stable source code')
    if audit.get('code_fingerprint') != fingerprint or audit.get('source_endings') != SOURCE_ENDINGS_VERSION:
        raise ValueError('assessment code changed since the audit; run a new audit first')
    runtime = {'python': platform.python_version()}
    for name in ('numpy', 'shapely', 'networkx'):
        runtime[name] = importlib.metadata.version(name)
    if audit.get('runtime') != runtime:
        raise ValueError('assessment runtime changed since the audit; use the audited Python/library versions')
    selected = [r for r in audit['records'] if r.get('status') == 'assessed'
                and (r.get('complete') or include_fragments)]
    if not selected:
        raise ValueError('audit has no supported complete maps')
    if out.exists() and any(out.iterdir()):
        raise ValueError('materialization output directory must be empty; choose a new directory')
    reuse = bool(audit.get('export_documents'))
    if reuse:
        for wad, signature in sorted({(row['wad'], row['wad_hash']) for row in selected}):
            if hashlib.sha256(Path(wad).read_bytes()).hexdigest() != signature:
                raise ValueError(f'source WAD changed: {wad}')
    maps = out / 'maps'
    maps.mkdir(parents=True, exist_ok=True)
    (out / 'records').mkdir()
    (out / 'source-audit.json').write_bytes(input_bytes)
    tool_bytes = Path(__file__).read_bytes()
    helper_path = Path(__file__).with_name('audit_sources.py')
    helper_bytes = helper_path.read_bytes()
    (out / 'materialize_sources.py').write_bytes(tool_bytes)
    (out / 'audit_sources.py').write_bytes(helper_bytes)
    report = {'status': 'running', 'code_fingerprint': fingerprint, 'code_fingerprint_scope': scope, 'runtime': runtime,
              'source_endings': SOURCE_ENDINGS_VERSION, 'audit': str(source),
              'audit_sha256': hashlib.sha256(input_bytes).hexdigest(),
              'tool_sha256': hashlib.sha256(tool_bytes).hexdigest(),
              'audit_tool_sha256': hashlib.sha256(helper_bytes).hexdigest(),
              'include_fragments': include_fragments, 'timeout': timeout,
              'reused_audit_snapshots': 0, 'records': []}
    def save():
        report['code_changed_during_materialization'] = (source_preprocessing_fingerprint(scope) != fingerprint
                                                        or Path(__file__).read_bytes() != tool_bytes
                                                        or helper_path.read_bytes() != helper_bytes)
        _save(out / 'report.json', report)
    save()
    def reuse_snapshot(row):
        filename = row['id'] + '.json'
        raw = (source.parent / 'records' / filename).read_bytes()
        document = (source.parent / 'maps' / filename).read_bytes()
        if (hashlib.sha256(raw).hexdigest() != row.get('record_sha256')
                or hashlib.sha256(document).hexdigest() != row.get('document_sha256')):
            raise ValueError(f'audited document/target snapshot changed: {filename}')
        record = json.loads(raw)
        if (record.get('graph_snapshot_version') != 1 or record.get('ending') != row.get('ending')
                or record.get('document_sha256') != row.get('document_sha256')
                or any(record.get(key) != row.get(key) for key in ('id', 'wad_hash', 'map_index', 'map'))):
            raise ValueError('audited target snapshot metadata mismatch')
        # The audit already assessed these exact inputs with the checked code
        # and runtime. Publish independent files, preserving their exact bytes.
        for path, data in ((maps / filename, document), (out / 'records' / filename, raw)):
            temporary = path.with_suffix('.json.tmp')
            temporary.write_bytes(data)
            temporary.replace(path)
        return record
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for row in selected:
                job = {key: row[key] for key in ('id', 'wad', 'wad_hash', 'map_index', 'map', 'format', 'lump_sizes', 'specials')
                       if key in row}
                filename = row['id'] + '.json'
                future = (pool.submit(reuse_snapshot, row) if reuse else
                          pool.submit(_run_job, job, out / 'records' / filename, timeout,
                                      document_output=maps / filename))
                futures[future] = row
            try:
                for future in as_completed(futures):
                    old, row = futures[future], future.result()
                    if row['status'] != 'assessed' or row.get('complete') != old['complete']:
                        raise ValueError(f"source eligibility changed or inconclusive: {old['id']}: {row.get('reason')}")
                    report['reused_audit_snapshots'] += int(reuse)
                    report['records'].append({'path': row['id'] + '.json', 'source_id': row['id'],
                        'wad_hash': row['wad_hash'], 'map': row['map'], 'rooms': row['graph']['rooms'],
                        'ending': row['ending'], 'document_sha256': row['document_sha256'],
                        'target_snapshot_sha256': hashlib.sha256(
                            (out / 'records' / (row['id'] + '.json')).read_bytes()).hexdigest()})
                    if len(report['records']) % 16 == 0:
                        save()
                    print(f"{len(report['records'])}/{len(selected)} {Path(row['wad']).name} {row['map']}: "
                          f"{row['graph']['rooms']} rooms", flush=True)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        report['records'].sort(key=lambda row: row['path'])
        manifest = build_manifest(maps, seed=seed, families=families,
                                  validation_fraction=validation_fraction, test_fraction=test_fraction)
        report.update(splits=dict(Counter(r['split'] for r in manifest['entries'])),
                      duplicate_maps=len(manifest['duplicates']),
                      families=len({r['group'] for r in manifest['entries']}),
                      complete_endings=sum(row['ending']['complete'] for row in report['records']))
        if (source_preprocessing_fingerprint(scope) != fingerprint or Path(__file__).read_bytes() != tool_bytes
                or helper_path.read_bytes() != helper_bytes):
            raise ValueError('assessment code changed while preparing the corpus')
        _save(out / 'corpus.json', manifest)
        report['status'] = 'complete'
        save()
        return report
    except Exception as exc:
        report.update(status='error', reason=str(exc))
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--test-fraction', type=float, default=0.1)
    parser.add_argument('--families', type=Path)
    parser.add_argument('--include-fragments', action='store_true',
                        help='Also export assessed fragments; train with --include-incomplete to censor their endings.')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    materialize(args.audit, args.out, seed=args.seed,
                families=json.loads(args.families.read_text()) if args.families else None,
                include_fragments=args.include_fragments, workers=args.workers, timeout=args.timeout,
                validation_fraction=args.validation_fraction, test_fraction=args.test_fraction)

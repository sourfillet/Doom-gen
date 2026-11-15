"""Retry inconclusive maps in an independent branch of an exported source audit.

    python -m tools.retry_source_audit --audit generated/audit/report.json \
        --wads wads --out generated/audit-retry --timeout 120

Verified terminal records and documents are copied independently. Only timed-out
maps are reassessed. Original audit bytes remain untouched, and the new report
retains the parent's identity, original budget and source provenance.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil

from tools.audit_sources import inventory, _run_job, _save, _summary_record


def retry(source, wads, out, *, timeout=120, workers=2):
    from data.source_endings import SOURCE_ENDINGS_VERSION
    from models.training import source_preprocessing_fingerprint, code_fingerprint
    if not math.isfinite(timeout) or timeout <= 0 or workers < 1:
        raise ValueError('timeout and workers must be positive')
    source, out = Path(source), Path(out).resolve()
    raw = source.read_bytes()
    parent = json.loads(raw)
    scope = parent.get('code_fingerprint_scope', 'repository_v1')
    fingerprint = source_preprocessing_fingerprint(scope)
    runtime = {'python': platform.python_version()}
    for name in ('numpy', 'shapely', 'networkx'):
        runtime[name] = importlib.metadata.version(name)
    if (parent.get('status') != 'complete' or parent.get('code_changed_during_audit')
            or not parent.get('export_documents')):
        raise ValueError('retry needs a completed stable audit with exported snapshots')
    if (parent.get('code_fingerprint') != fingerprint or parent.get('source_endings') != SOURCE_ENDINGS_VERSION
            or parent.get('runtime') != runtime):
        raise ValueError('retry assessment code/runtime differs; run a new audit')
    sources = inventory(wads)
    signature = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    if signature != parent['inventory_sha256']:
        raise ValueError('source WAD inventory changed since the audit')
    records = {row['id']: row for row in parent['records']}
    if len(records) != len(parent['records']) or set(records) != {job['id'] for job in sources['jobs']}:
        raise ValueError('parent records do not cover the verified source inventory')
    if any(any(records[job['id']].get(key) != value for key, value in job.items()) for job in sources['jobs']):
        raise ValueError('parent record identities differ from the source inventory')
    selected = [job for job in sources['jobs'] if records[job['id']]['status'] == 'timeout']
    if not selected:
        raise ValueError('parent audit has no timeouts to retry')
    if out.exists() and any(out.iterdir()):
        raise ValueError('retry output directory must be empty; choose a new directory')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'records').mkdir()
    (out / 'maps').mkdir()
    tool_bytes = Path(__file__).read_bytes()
    helper = Path(__file__).with_name('audit_sources.py')
    helper_bytes = helper.read_bytes()
    (out / 'retry_source_audit.py').write_bytes(tool_bytes)
    (out / 'audit_sources.py').write_bytes(helper_bytes)
    (out / 'parent-audit.json').write_bytes(raw)
    _save(out / 'inventory.json', sources)
    report = {key: value for key, value in parent.items() if key not in {'records', 'aggregate_correction'}}
    report.update(status='running', timeout=timeout, records=[],
                  repository_fingerprint=code_fingerprint(),
                  tool_sha256=hashlib.sha256(helper_bytes).hexdigest(),
                  retry_tool_sha256=hashlib.sha256(tool_bytes).hexdigest(),
                  retry_parent={'path': str(source), 'sha256': hashlib.sha256(raw).hexdigest(),
                                'timeout': parent['timeout'], 'tool_sha256': parent['tool_sha256']},
                  retried_maps=len(selected), completed_retries=0)
    report['limits'] = [*parent.get('limits', []),
        'Non-timeout records were copied from the verified parent; their original runtime/budget provenance is retained.']
    def save():
        rows = [records[job['id']] for job in sources['jobs']]
        report.update(records=rows, counts=dict(Counter(row['status'] for row in rows)),
                      complete_endings=sum(row['status'] == 'assessed' and row.get('complete', False) for row in rows),
                      code_changed_during_audit=source_preprocessing_fingerprint(scope) != fingerprint
                          or Path(__file__).read_bytes() != tool_bytes or helper.read_bytes() != helper_bytes)
        _save(out / 'report.json', report)
    save()
    try:
        for row in records.values():
            filename = row['id'] + '.json'
            record = source.parent / 'records' / filename
            if hashlib.sha256(record.read_bytes()).hexdigest() != row['record_sha256']:
                raise ValueError(f'parent target record changed: {filename}')
            shutil.copyfile(record, out / 'records' / filename)
            if row['status'] == 'assessed':
                document = source.parent / 'maps' / filename
                if hashlib.sha256(document.read_bytes()).hexdigest() != row['document_sha256']:
                    raise ValueError(f'parent document changed: {filename}')
                shutil.copyfile(document, out / 'maps' / filename)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_job, job, out / 'records' / (job['id'] + '.json'), timeout,
                                   document_output=out / 'maps' / (job['id'] + '.json')): job for job in selected}
            for future in as_completed(futures):
                job, row = futures[future], future.result()
                records[job['id']] = _summary_record(row, out / 'records' / (job['id'] + '.json'))
                report['completed_retries'] += 1
                save()
                print(f"{report['completed_retries']}/{len(selected)} {Path(job['wad']).name} {job['map']}: "
                      f"{row['status']} — {row.get('reason', '')}", flush=True)
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
    parser.add_argument('--wads', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    retry(args.audit, args.wads, args.out, timeout=args.timeout, workers=args.workers)

"""Incremental, bounded audit of source WADs and room-training eligibility.

    python -m tools.audit_sources --wads wads --out generated/source-audit --workers 4
    python -m tools.audit_sources --wads wads --out generated/source-audit --resume

Every map runs in an isolated subprocess with a wall-clock budget. Resume
requires identical source WADs, audit code, dependencies, and assessment code.
Timeouts and unsupported mechanics are evidence gaps, not unplayable-map claims.
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
import struct
import subprocess
import sys
import time


def _save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def inventory(directory):
    from wad_parser.wad_reader import WadFile
    files, jobs = [], {}
    for path in sorted(Path(directory).resolve().rglob('*')):
        if path.suffix.lower() != '.wad' or not path.is_file():
            continue
        raw = path.read_bytes()
        row = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        try:
            wad = WadFile(str(path))
            if wad.hash != row['sha256']:
                raise ValueError('source WAD changed during inventory')
            groups = wad.find_map_groups()
            row.update(maps=len(groups), formats=dict(Counter(g.format for g in groups)),
                       mapless=not groups)
            for index, group in enumerate(groups):
                identity = f'{wad.hash}-{index:04d}'
                record = {'id': identity, 'wad': str(path), 'wad_hash': wad.hash,
                          'map_index': index, 'map': group.name, 'format': group.format,
                          'lump_sizes': {k: len(v) for k, v in group.lumps.items()}}
                if group.format == 'doom':
                    data = group.lumps.get('LINEDEFS', b'')
                    if len(data) % 14 == 0:
                        record['specials'] = dict(Counter(str(v[3]) for v in struct.iter_unpack('<7H', data) if v[3]))
                jobs.setdefault(identity, record)
        except (ValueError, OSError) as exc:
            row.update(error=str(exc), maps=0, mapless=False)
        files.append(row)
    if not files:
        raise ValueError('no WAD files found')
    # Interleave map indices across families so bounded batches cover the
    # collection instead of spending their entire budget on its first WAD.
    jobs = sorted(jobs.values(), key=lambda r: (r['map_index'], r['wad'], r['id']))
    return {'files': files, 'jobs': jobs}


def inspect_map(job, output, *, document_output=None):
    from data.source_endings import assess_source_ending
    from data.roomgraph_targets import graph_to_sequence
    from roomgraph.segment import segment
    from wad_parser.map_extractor import extract_map
    from wad_parser.wad_reader import WadFile

    started = time.perf_counter()
    row = {**job, 'status': 'running', 'stage': 'extraction'}
    def save():
        row['seconds'] = time.perf_counter() - started
        _save(output, row)
    save()
    try:
        wad = WadFile(job['wad'])
        if wad.hash != job['wad_hash']:
            raise ValueError('source WAD changed after inventory')
        group = wad.find_map_groups()[job['map_index']]
        if group.name != job['map']:
            raise ValueError('map group changed after inventory')
        doc = extract_map(wad, group)
        row['geometry'] = {'sectors': len(doc.sectors), 'linedefs': len(doc.linedefs),
                           'things': len(doc.things), 'issues': doc.derived.geometry_issues,
                           'raw_issues': doc.derived.raw_geometry_issues,
                           'bsp': doc.derived.bsp_geometry,
                           'reachable_sectors': len(doc.derived.reachable_from_start)}
        if doc.meta.format != 'doom' or doc.extra.get('udmf_namespace'):
            row.update(status='unsupported', reason='room training requires Doom binary maps')
        elif doc.derived.geometry_issues:
            row.update(status='excluded', reason='invalid sector boundaries')
        else:
            row['stage'] = 'segmentation'
            save()
            graph = segment(doc, playable_only=True)
            row['graph'] = {**graph.summary(), 'excluded_sectors': graph.excluded_sectors}
            row['stage'] = 'target_encoding'
            save()
            sequence = graph_to_sequence(graph)
            if not sequence:
                raise ValueError('no rooms')
            row['historical_ending'] = assess_source_ending(doc, graph, version='reachable_exit_v1')
            row['stage'] = 'physical_ending'
            save()
            row['ending'] = assess_source_ending(doc, graph)
            row.update(status='assessed', complete=bool(row['ending']['complete']),
                       reason=row['ending']['reason'])
            if document_output is not None:
                row['graph_snapshot_version'] = 1
                row['graph_snapshot'] = graph.to_dict()
                row['sequence'] = sequence
                row['stage'] = 'document_output'
                save()
                _save(document_output, doc.to_dict())
                row['document_sha256'] = hashlib.sha256(Path(document_output).read_bytes()).hexdigest()
    except Exception as exc:
        # A bad map must leave an explicit record and not discard the rest of
        # the collection. Preserve exception types to distinguish code faults.
        row.update(status='error', reason=str(exc), exception_type=type(exc).__name__)
    save()
    return row


def _run_job(job, output, timeout, *, document_output=None):
    started = time.perf_counter()
    command = [sys.executable, '-m', 'tools.audit_sources', '--worker-job', json.dumps(job),
               '--worker-out', str(output)]
    if document_output is not None:
        command += ['--worker-document', str(document_output)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                                cwd=Path(__file__).resolve().parents[1])
        if result.returncode == 0 and output.exists():
            return json.loads(output.read_text())
        status, reason = 'error', f'audit worker exited with code {result.returncode}: {result.stderr[-2000:]}'
    except subprocess.TimeoutExpired:
        status, reason = 'timeout', f'per-map wall-clock budget exhausted ({timeout}s)'
    row = json.loads(output.read_text()) if output.exists() else {**job, 'stage': 'worker_start'}
    row.update(status=status, reason=reason, seconds=time.perf_counter() - started)
    _save(output, row)
    return row


def _summary_record(row, path):
    # Full geometry/targets live once in the per-map record, rather than being
    # copied into the growing aggregate after every progress update.
    return {**{key: value for key, value in row.items() if key not in {'graph_snapshot', 'sequence'}},
            'record_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def audit(directory, out, *, workers=2, timeout=60, limit=0, resume=False, export_documents=False):
    from data.source_endings import SOURCE_ENDINGS_VERSION
    from models.training import code_fingerprint, source_preprocessing_fingerprint
    if workers < 1 or timeout <= 0 or limit < 0:
        raise ValueError('workers/timeout must be positive and limit non-negative')
    out = Path(out).resolve()
    sources = inventory(directory)
    runtime = {'python': platform.python_version()}
    for name in ('numpy', 'shapely', 'networkx'):
        runtime[name] = importlib.metadata.version(name)
    tool_bytes = Path(__file__).read_bytes()
    signature = {'version': 1, 'source_endings': SOURCE_ENDINGS_VERSION,
                 'code_fingerprint_scope': 'roomgraph_preprocessing_v1',
                 'code_fingerprint': source_preprocessing_fingerprint(),
                 'tool_sha256': hashlib.sha256(tool_bytes).hexdigest(),
                 'runtime': runtime, 'timeout': timeout,
                 'export_documents': export_documents,
                 'inventory_sha256': hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()}
    if resume:
        if not (out / 'config.json').exists() or json.loads((out / 'config.json').read_text()) != signature:
            raise ValueError('audit resume inputs/code/runtime/budget changed; use a new output directory')
    elif out.exists() and any(out.iterdir()):
        raise ValueError('audit output directory must be empty; use --resume or choose a new directory')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'records').mkdir(exist_ok=True)
    if export_documents:
        (out / 'maps').mkdir(exist_ok=True)
    if not resume:
        _save(out / 'config.json', signature)
        _save(out / 'provenance.json', {'repository_fingerprint': code_fingerprint()})
        _save(out / 'inventory.json', sources)
        (out / 'audit_sources.py').write_bytes(tool_bytes)
    records, pending = {}, []
    for job in sources['jobs']:
        path = out / 'records' / (job['id'] + '.json')
        row = json.loads(path.read_text()) if path.exists() else None
        if row and any(row.get(key) != value for key, value in job.items()):
            raise ValueError(f'audit record identity mismatch: {path.name}')
        if row and row.get('status') != 'running':
            if export_documents and row.get('status') == 'assessed':
                document = out / 'maps' / (job['id'] + '.json')
                if (not document.exists() or hashlib.sha256(document.read_bytes()).hexdigest()
                        != row.get('document_sha256')):
                    raise ValueError(f'exported audit document changed or missing: {document.name}')
            records[job['id']] = _summary_record(row, path)
        else:
            pending.append(job)
    selected = pending[:limit] if limit else pending
    def save():
        rows = [records[job['id']] for job in sources['jobs'] if job['id'] in records]
        report = {**signature, 'status': 'complete' if len(rows) == len(sources['jobs']) else 'partial',
                  'code_changed_during_audit': source_preprocessing_fingerprint() != signature['code_fingerprint']
                      or Path(__file__).read_bytes() != tool_bytes,
                  'repository_fingerprint': json.loads((out / 'provenance.json').read_text())['repository_fingerprint'],
                  'files': len(sources['files']), 'mapless_files': sum(r.get('mapless', False) for r in sources['files']),
                  'inventory_errors': [r for r in sources['files'] if r.get('error')],
                  'total_maps': len(sources['jobs']), 'processed': len(rows),
                  'counts': dict(Counter(r['status'] for r in rows)),
                  'complete_endings': sum(r.get('status') == 'assessed' and r.get('complete', False) for r in rows),
                  'reasons': dict(Counter(r.get('reason', '') for r in rows)),
                  'records': rows,
                  'limits': ['Supported source-ending evidence, not engine/combat completion.',
                             'Timeout/error records are inconclusive and never counted as complete.',
                             'No source-family holdout or train/validation assignment is inferred.']}
        _save(out / 'report.json', report)
        return report
    save()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_job, job, out / 'records' / (job['id'] + '.json'), timeout,
                    document_output=(out / 'maps' / (job['id'] + '.json')) if export_documents else None): job
                   for job in selected}
        for future in as_completed(futures):
            job = futures[future]
            row = future.result()
            records[job['id']] = _summary_record(row, out / 'records' / (job['id'] + '.json'))
            # Workers persist every completed map atomically. Avoid rewriting
            # the growing aggregate and rehashing the code tree after each one.
            if len(records) % 16 == 0:
                save()
            print(f"{len(records)}/{len(sources['jobs'])} {Path(job['wad']).name} {job['map']}: "
                  f"{records[job['id']]['status']} — {records[job['id']].get('reason', '')}", flush=True)
    return save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wads', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--export-documents', action='store_true',
                        help='Save assessed source JSON and exact targets for reuse during materialization.')
    parser.add_argument('--worker-job', help=argparse.SUPPRESS)
    parser.add_argument('--worker-out', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-document', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_job:
        inspect_map(json.loads(args.worker_job), args.worker_out, document_output=args.worker_document)
    elif args.wads is None or args.out is None:
        parser.error('--wads and --out are required')
    else:
        audit(args.wads, args.out, workers=args.workers, timeout=args.timeout,
              limit=args.limit, resume=args.resume, export_documents=args.export_documents)

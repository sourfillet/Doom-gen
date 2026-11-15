"""Replay exact room graphs through the geometry back half, without a model.

    python -m tools.evaluate_routing --input generated/evaluation/report.json \
        --out generated/routing-check --samples 16

Accepts evaluate-roomgen reports with graph snapshots, or a capture containing
records with method/seed/graph fields. Failed seeds remain in the report.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time


def evaluate(source, out, *, samples=16, geometry_profile='classic'):
    from generation.baselines import realize
    from models.training import code_fingerprint
    from roomgraph.types import RoomGraph
    from generation.pipeline import MapBatch
    from roomgraph.variation import PROFILES

    if geometry_profile not in PROFILES:
        raise ValueError(f'unknown geometry profile: {geometry_profile}')

    source, out = Path(source), Path(out)
    input_bytes = source.read_bytes()
    payload = json.loads(input_bytes)
    if payload.get('graph_snapshot_version', 1) != 1:
        raise ValueError('unsupported graph snapshot version')
    records = payload.get('records')
    if records is None:
        records = []
        panels = [('neural', payload['panel'], payload.get('geometry', {}))]
        panels += [(name, baseline['panel'], baseline['geometry'])
                   for name, baseline in payload.get('baselines', {}).items()]
        for method, panel, geometry in panels:
            if 'outcomes' in geometry:
                original = {row['seed']: row['written'] for row in geometry['outcomes']}
                records += [{'method': method, **row, 'original_written': original.get(row['seed'])}
                            for row in panel]
                continue
            failed = {row['seed'] for row in geometry.get('failed_samples', [])}
            if method == 'neural':
                attempted = failed | {row['seed'] for row in geometry.get('summaries', [])}
            else:
                count = geometry.get('written_maps', 0) + len(failed)
                attempted = {row['seed'] for row in panel[:count]}
            records += [{'method': method, **row,
                         'original_written': row['seed'] not in failed if row['seed'] in attempted else None}
                        for row in panel]
    if any('graph' not in row for row in records):
        raise ValueError('input has no exact graph snapshots; rerun evaluate-roomgen with current code')
    if out.exists() and any(out.iterdir()):
        raise ValueError('routing evaluation output directory must be empty; choose a new directory')
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = code_fingerprint()
    tool_bytes = Path(__file__).read_bytes()
    (out / 'input-graphs.json').write_bytes(input_bytes)
    (out / 'evaluate_routing.py').write_bytes(tool_bytes)
    runtime = {'python': platform.python_version()}
    for name in ('numpy', 'shapely', 'networkx'):
        runtime[name] = importlib.metadata.version(name)
    report = {'status': 'running', 'input': str(source), 'input_sha256': hashlib.sha256(input_bytes).hexdigest(),
              'input_snapshot': 'input-graphs.json', 'tool_sha256': hashlib.sha256(tool_bytes).hexdigest(),
              'runtime': runtime, 'graph_snapshot_version': 1, 'geometry_profile': geometry_profile,
              'code_fingerprint': fingerprint, 'samples_per_method': samples, 'records': [],
              'limits': ['Exact graph replay; no retraining or learned-structure improvement is inferred.',
                         'Written maps pass static validation; engine completion needs a separate run.']}
    counts, panels = {}, {}
    def save():
        report['code_changed_during_evaluation'] = (code_fingerprint() != fingerprint
                                                   or Path(__file__).read_bytes() != tool_bytes)
        (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for row in records:
        method = row['method']
        index = counts.get(method, 0)
        if index >= samples:
            continue
        counts[method] = index + 1
        docs = panels.setdefault(method, [])
        graph = RoomGraph.from_dict(row['graph'])
        graph.map_name = f'MAP{len(docs) + 1:02d}'
        result = {'method': method, 'seed': row['seed'], 'written': False, 'validated': False,
                  'rooms': len(graph.rooms), 'edges': len(graph.connections),
                  'graph': row['graph'], 'original_written': row.get('original_written')}
        start = time.perf_counter()
        try:
            doc = realize(graph, geometry_profile=geometry_profile, seed=row['seed'] or 0)
            docs.append(doc)
            result.update(validated=True, map=doc.meta.map_name, routing=doc.extra.get('routing'),
                          geometry_profile=doc.extra.get('geometry_profile'),
                          geometry_seed=doc.extra.get('geometry_seed'),
                          geometry_variation=doc.extra.get('geometry_variation'),
                          connection_realization=doc.extra.get('connection_realization'))
        except ValueError as exc:
            result['reason'] = str(exc)
            result['failure_stage'] = 'validation'
        result['seconds'] = time.perf_counter() - start
        report['records'].append(result)
        save()
        print(f'{method} seed {row["seed"]}: ' + ('validated' if result['validated'] else result['reason']), flush=True)
    report['panels'] = {}
    for method, docs in panels.items():
        selected = [row for row in report['records'] if row['method'] == method]
        summary = {'attempted': len(selected), 'validated': len(docs), 'written': 0}
        report['panels'][method] = summary
        if docs:
            with MapBatch(out / f'{method}-panel.wad') as batch:
                for doc, record in zip(docs, [row for row in selected if row['validated']]):
                    doc.meta.map_name = f'MAP{len(batch.accepted) + 1:02d}'
                    try:
                        publication = batch.add(doc)
                        record.update(map=doc.meta.map_name, publication=publication)
                    except (ValueError, RuntimeError) as exc:
                        record.update(reason=str(exc), failure_stage='publication')
                    save()
                if batch.accepted:
                    try:
                        result = batch.publish()
                        summary.update(path=Path(result['path']).name, written=len(result['maps']),
                                       map_publication=result['map_publication'])
                        for record in selected:
                            record['written'] = record['validated'] and 'failure_stage' not in record
                    except (ValueError, RuntimeError) as exc:
                        summary['publication_error'] = str(exc)
                        for record in selected:
                            if record['validated'] and 'failure_stage' not in record:
                                record.update(reason='panel publication failed: ' + str(exc), failure_stage='publication')
                else:
                    summary['publication_error'] = '; '.join(row['reason'] for row in selected if row['validated'])
        summary['failures'] = len(selected) - summary['written']
        save()
    report['status'] = 'complete'
    save()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--geometry-profile', choices=['classic', 'varied', 'compact', 'adjoining'], default='classic')
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    evaluate(args.input, args.out, samples=args.samples, geometry_profile=args.geometry_profile)

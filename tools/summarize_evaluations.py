"""Pool disjoint fixed-checkpoint seed panels without dropping failed samples.

    python -m tools.summarize_evaluations --out generated/seed-summary \
        generated/evaluation-a/report.json generated/evaluation-b/report.json

Intervals describe sampling seeds for one fixed model and realization budget.
They do not measure training-run, dataset, engine-completion or human uncertainty.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


def rate(successes, count):
    """95% Wilson score interval; an empty experiment has no estimated rate."""
    if not 0 <= successes <= count:
        raise ValueError('invalid rate counts')
    if not count:
        return {'successes': 0, 'count': 0, 'rate': None, 'wilson_95': None}
    z = statistics.NormalDist().inv_cdf(0.975)
    p, d = successes / count, 1 + z * z / count
    center = (p + z * z / (2 * count)) / d
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / d
    return {'successes': successes, 'count': count, 'rate': p,
            'wilson_95': [max(0, center - half), min(1, center + half)]}


def outcomes(panel, geometry, *, neural):
    seeds = [row['seed'] for row in panel]
    if len(seeds) != len(set(seeds)):
        raise ValueError('duplicate seeds within a panel')
    if 'outcomes' in geometry:
        rows = geometry['outcomes']
        requested = geometry['requested_seeds']
        if (len(requested) != len(set(requested)) or len(rows) != len(requested)
                or {r['seed'] for r in rows} != set(requested)):
            raise ValueError('explicit geometry outcomes do not cover requested seeds exactly once')
        if any(type(r.get('written')) is not bool for r in rows):
            raise ValueError('geometry outcomes need explicit boolean publication results')
        if any(r['written'] and r.get('validated') is not True for r in rows):
            raise ValueError('published geometry lacks successful validation evidence')
    else:
        if geometry.get('failure') or geometry.get('publication_error'):
            raise ValueError('legacy batch failure has no per-seed outcomes; rerun evaluation')
        failed = [row['seed'] for row in geometry.get('failed_samples', [])]
        if len(failed) != len(set(failed)):
            raise ValueError('duplicate geometry failure seeds')
        if neural:
            written = [row['seed'] for row in geometry.get('summaries', [])]
            if len(written) != len(set(written)) or set(written) & set(failed):
                raise ValueError('duplicate or contradictory geometry outcomes')
            requested = written + failed
        else:
            # Historical baseline reports attempted a prefix and retained
            # individual failures, but omitted successful seed identities.
            count = geometry.get('written_maps', 0) + len(failed)
            if count > len(seeds):
                raise ValueError('baseline attempt count exceeds panel size')
            requested = seeds[:count]
            if not set(failed) <= set(requested):
                raise ValueError('baseline failures are outside the attempted prefix')
        rows = [{'seed': seed, 'written': seed not in failed} for seed in requested]
    if not {r['seed'] for r in rows} <= set(seeds):
        raise ValueError('geometry outcomes refer to seeds outside the sampled panel')
    if sum(row['written'] for row in rows) != geometry.get('written_maps', 0):
        raise ValueError('geometry publication count disagrees with per-seed results')
    return {row['seed']: row['written'] for row in rows}


def describe(rows, published, *, max_rooms=None):
    rooms = [row['rooms'] for row in rows]
    matches = [row.get('matches_binned_training_graph') for row in rows]
    return {'samples': len(rows), 'room_counts': dict(sorted(Counter(rooms).items())),
            'mean_rooms': statistics.mean(rooms), 'median_rooms': statistics.median(rooms),
            'min_rooms': min(rooms), 'max_rooms': max(rooms),
            'one_room_samples': rate(sum(n == 1 for n in rooms), len(rooms)),
            'room_budget_hits': rate(sum(n == max_rooms for n in rooms), len(rooms)) if max_rooms else None,
            'exact_binned_training_matches': rate(sum(matches), len(matches))
                if all(type(value) is bool for value in matches) else None,
            'publication': rate(sum(published.values()), len(published))}


def summarize(paths, out):
    paths, out = [Path(p) for p in paths], Path(out)
    if not paths:
        raise ValueError('at least one evaluation report is required')
    inputs, reports = [p.read_bytes() for p in paths], []
    for raw in inputs:
        report = json.loads(raw)
        if (report.get('code_changed_during_evaluation') is not False
                or report.get('checkpoint_changed_during_evaluation')
                or report.get('prepared_targets', {}).get('source_edits_during_evaluation')):
            raise ValueError('evaluation source/checkpoint changed or lacks stable-source evidence')
        reports.append(report)
    identity_keys = ('training_step', 'corpus_fingerprint', 'temperature', 'max_rooms',
                     'source_endings_version', 'training_code_fingerprint', 'evaluation_code_fingerprint')
    identity = {key: reports[0][key] for key in identity_keys}
    identity['geometry_profile'] = reports[0].get('geometry_profile', 'classic')
    if any(any(report.get(key) != value for key, value in identity.items() if key != 'geometry_profile')
           or report.get('geometry_profile', 'classic') != identity['geometry_profile'] for report in reports):
        raise ValueError('evaluation model/corpus/settings/source identities differ')
    hashes = {report['checkpoint_sha256'] for report in reports if report.get('checkpoint_sha256')}
    legacy = any(not report.get('checkpoint_sha256') for report in reports)
    if len(hashes) > 1 or (legacy and len({report['checkpoint'] for report in reports}) > 1):
        raise ValueError('evaluation checkpoints differ')
    identity['checkpoint_sha256'] = next(iter(hashes), None)
    identity['checkpoint_identity'] = 'path_and_step_only' if legacy else 'content_hash'
    methods = ['neural', *sorted(reports[0].get('baselines', {}))]
    rows = {name: [] for name in methods}
    published = {name: {} for name in methods}
    panels, seen = [], set()
    for report in reports:
        if sorted(report.get('baselines', {})) != methods[1:]:
            raise ValueError('baseline methods differ across panels')
        seeds = [row['seed'] for row in report['panel']]
        if not seeds or len(seeds) != len(set(seeds)) or seen & set(seeds):
            raise ValueError('empty panel or overlapping/duplicate sample seeds')
        seen.update(seeds)
        panel_summary = {'sample_seeds': seeds, 'methods': {}}
        attempted = None
        for method in methods:
            source = report if method == 'neural' else report['baselines'][method]
            if [row['seed'] for row in source['panel']] != seeds:
                raise ValueError('baseline sample seeds do not match the neural panel')
            results = outcomes(source['panel'], source.get('geometry', {}), neural=method == 'neural')
            if attempted is not None and set(results) != attempted:
                raise ValueError('methods attempted different geometry seed sets')
            attempted = set(results)
            rows[method].extend(source['panel'])
            published[method].update(results)
            panel_summary['methods'][method] = describe(source['panel'], results,
                max_rooms=identity['max_rooms'] if method == 'neural' else None)
        panels.append(panel_summary)
    comparisons = {}
    neural = published['neural']
    for method in methods[1:]:
        other = published[method]
        wins = sum(neural[s] and not other[s] for s in neural)
        losses = sum(other[s] and not neural[s] for s in neural)
        discordant = wins + losses
        # Exact two-sided sign/binomial test on paired discordant seeds.
        p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1))
                / 2 ** discordant) if discordant else 1.0
        comparisons[method] = {'paired_attempts': len(neural), 'neural_only_written': wins,
                               'baseline_only_written': losses,
                               'both_written': sum(neural[s] and other[s] for s in neural),
                               'neither_written': sum(not neural[s] and not other[s] for s in neural),
                               'write_rate_difference': (wins - losses) / len(neural) if neural else None,
                               'exact_two_sided_discordant_p': p}
    result = {'identity': identity, 'panels': panels,
              'methods': {name: describe(rows[name], published[name],
                           max_rooms=identity['max_rooms'] if name == 'neural' else None) for name in methods},
              'paired_neural_vs_baseline': comparisons,
              'inputs': [{'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}
                         for path, raw in zip(paths, inputs)],
              'limits': ['95% Wilson intervals concern sampler seeds for one fixed checkpoint and budget.',
                         'Paired exploratory p-values are unadjusted for multiple comparisons.',
                         'No training-run, dataset, engine-completion or human-quality uncertainty is measured.',
                         'Room-budget hits need not imply the model would have continued.',
                         'Nearest-size retrieval can have different room counts; only attempt seeds are paired.']}
    if legacy:
        result['limits'].append('Historical reports lack checkpoint hashes: path/step identity is weaker than content identity.')
        result['limits'].append('Historical baseline successful seeds are reconstructed from the documented attempted prefix.')
    if out.exists() and any(out.iterdir()):
        raise ValueError('summary output directory must be empty; choose a new directory')
    out.mkdir(parents=True, exist_ok=True)
    for index, raw in enumerate(inputs):
        (out / f'input-{index:02d}.json').write_bytes(raw)
    tool = Path(__file__).read_bytes()
    (out / 'summarize_evaluations.py').write_bytes(tool)
    result['tool_sha256'] = hashlib.sha256(tool).hexdigest()
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    summarize(args.reports, args.out)

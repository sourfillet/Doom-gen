"""Select a bounded, family-balanced geometry experiment from a frozen corpus.

Uses raw record counts to budget the exact byte codec before expensive geometry
work. Original family splits and documents are unchanged. Vertex count is only
a coarse inventory filter; actual outline/geometry checks belong to preparation
and training. This tool does not assign positive geometry or gameplay labels.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from data.geometry_tokens import STRUCTS
from data.learned_geometry import AUTHORING_FIELDS


def _manifest_digest(payload):
    # Match data.corpus exactly; its manifest contract includes JSON spacing.
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def select(maps_dir, manifest_path, out, *, max_tokens=4096, min_vertices=12,
           max_maps=24, per_family=2, seed=7):
    if min(max_tokens, min_vertices, max_maps, per_family) < 1:
        raise ValueError('selection budgets must be positive')
    maps_dir, manifest_path, out = Path(maps_dir), Path(manifest_path), Path(out)
    raw_manifest = manifest_path.read_bytes()
    parent = json.loads(raw_manifest)
    if (parent.get('version') != 1 or _manifest_digest({k: v for k, v in parent.items() if k != 'fingerprint'})
            != parent.get('fingerprint')):
        raise ValueError('invalid parent corpus manifest')
    if out.exists() and any(out.iterdir()):
        raise ValueError('selection output must be empty; choose a fresh directory')
    records, candidates = [], defaultdict(list)
    for entry in parent['entries']:
        raw = (maps_dir / entry['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry['document_hash']:
            raise ValueError(f'source document changed: {entry["path"]}')
        source = json.loads(raw)
        counts = {name: len(source[name]) for name in AUTHORING_FIELDS}
        tokens = 7 + sum(counts[name] * (STRUCTS[name].size + 1) for name in AUTHORING_FIELDS)
        row = {'path': entry['path'], 'family': entry['family'], 'split': entry['split'],
               'document_hash': entry['document_hash'], 'counts': counts, 'tokens': tokens,
               'status': 'candidate'}
        if counts['vertices'] < min_vertices:
            row.update(status='excluded', reason='below minimum authored vertex count')
        elif tokens > max_tokens:
            row.update(status='excluded', reason='above exact codec token budget')
        else:
            candidates[entry['family']].append(entry)
        records.append(row)
    rank = lambda value: hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest()
    families = sorted(candidates, key=rank)
    for values in candidates.values():
        values.sort(key=lambda entry: rank(entry['document_hash']))
    selected = []
    for index in range(per_family):
        for family in families:
            if index < len(candidates[family]) and len(selected) < max_maps:
                selected.append(candidates[family][index])
    selected_paths = {entry['path'] for entry in selected}
    for row in records:
        if row['status'] == 'candidate':
            row['status'] = 'selected' if row['path'] in selected_paths else 'budget_not_selected'
    child = {key: value for key, value in parent.items() if key not in {'entries', 'duplicates', 'fingerprint'}}
    child.update(entries=selected,
                 duplicates=[entry for entry in parent.get('duplicates', []) if entry['canonical_path'] in selected_paths],
                 selection_parent_fingerprint=parent['fingerprint'])
    child['fingerprint'] = _manifest_digest(child)
    report = {'schema': 'doom_geometry_source_selection_v1', 'parent_sha256': hashlib.sha256(raw_manifest).hexdigest(),
              'parent_fingerprint': parent['fingerprint'], 'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'settings': {'max_tokens': max_tokens, 'min_vertices': min_vertices, 'max_maps': max_maps,
                           'per_family': per_family, 'seed': seed}, 'selected': len(selected),
              'geometry_eligibility': 'not_assessed', 'records': records}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'parent-manifest.json').write_bytes(raw_manifest)
    (out / 'select_geometry_sources.py').write_bytes(Path(__file__).read_bytes())
    (out / 'corpus.json').write_text(json.dumps(child, indent=2) + '\n')
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps-dir', required=True)
    parser.add_argument('--manifest', dest='manifest_path', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--min-vertices', type=int, default=12)
    parser.add_argument('--max-maps', type=int, default=24)
    parser.add_argument('--per-family', type=int, default=2)
    parser.add_argument('--seed', type=int, default=7)
    args = vars(parser.parse_args())
    try:
        report = select(**args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"{report['selected']} selected; original family splits retained; geometry not yet assessed")


if __name__ == '__main__':
    main()

"""Export experimental full-map targets without changing existing corpora.

    python -m tools.prepare_learned_geometry --maps-dir dataset/maps \
        --manifest dataset/corpus.json --out generated/learned-geometry --limit 8

Reuses the frozen family split and retains every rejection/source snapshot.
Exported does NOT mean physically eligible, complete, or ready for training.
This is a source-tree tool; it registers no new installed console command.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform

from data.corpus import training_split
from data.learned_geometry import build_target, digest, validate_target
from wad_parser.types import MapDocument

SCHEMA = 'doom_learned_map_preparation_v1'
SOURCE_SCOPE = 'learned_map_preprocessing_v1'
# Conservative scope: deterministic geometry/data code, excluding unrelated ML
# model/trainer edits. All scoped bytes are archived; old corpora are untouched.
SOURCE_DIRECTORIES = ('data', 'generation', 'roomgraph', 'tagging', 'wad_parser', 'wad_writer')
ROOT = Path(__file__).resolve().parents[1]


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _source_bytes():
    return {str(path.relative_to(ROOT)): path.read_bytes()
            for directory in SOURCE_DIRECTORIES for path in sorted((ROOT / directory).rglob('*.py'))}


def _source_fingerprint(sources):
    return digest({name: _sha(raw) for name, raw in sources.items()})


def _runtime():
    import shapely
    return {'python': platform.python_version(), 'geos': shapely.geos_version_string,
            **{name: importlib.metadata.version(name) for name in ('numpy', 'shapely', 'networkx')}}


def _save(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def prepare(maps_dir, manifest, out, *, limit=0, tags=None, target_game=''):
    """Prepare signed targets from a verified frozen manifest into a fresh dir.

    A bounded subset keeps its original split labels. Every source is copied
    before decoding, including sources that are rejected. No source ending or
    geometry eligibility is inferred, and existing trainers cannot use this
    preparation as their older room/raster corpus contract.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError('limit must be a nonnegative integer')
    maps_dir, manifest, out = Path(maps_dir), Path(manifest), Path(out)
    from data.learned_geometry import validate_conditioning
    validate_conditioning({'tags': {} if tags is None else tags, 'target_game': target_game})
    manifest_raw = manifest.read_bytes()
    frozen = json.loads(manifest_raw)
    training_split(maps_dir, manifest)  # Verify document hashes, splits and manifest identity.
    if manifest.read_bytes() != manifest_raw:
        raise ValueError('source manifest changed during preparation preflight')
    if out.exists() and any(out.iterdir()):
        raise ValueError('preparation output must be empty; choose a fresh directory')
    sources = _source_bytes()
    tool_raw = Path(__file__).read_bytes()
    runtime = _runtime()
    entries = frozen['entries'][:limit or None]
    out.mkdir(parents=True, exist_ok=True)
    for name in ('source-documents', 'targets', 'source-code'):
        (out / name).mkdir()
    (out / 'source-manifest.json').write_bytes(manifest_raw)
    (out / 'prepare_learned_geometry.py').write_bytes(tool_raw)
    for name, raw in sources.items():
        path = out / 'source-code' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    report = {'schema': SCHEMA, 'status': 'running', 'limit': limit,
              'source_manifest_sha256': _sha(manifest_raw),
              'source_corpus_fingerprint': frozen['fingerprint'],
              'source_fingerprint_scope': SOURCE_SCOPE,
              'source_fingerprint': _source_fingerprint(sources),
              'tool_sha256': _sha(tool_raw), 'runtime': runtime,
              'conditioning': {'tags': {} if tags is None else tags, 'target_game': target_game},
              'geometry_eligibility': 'not_assessed', 'progression_eligibility': 'not_assessed',
              'source_changed_during_preparation': False, 'records': []}
    _save(out / 'report.json', report)
    try:
        for index, entry in enumerate(entries):
            record = {'path': entry['path'], 'family': entry['family'], 'group': entry['group'],
                      'split': entry['split'], 'source_document_sha256': entry['document_hash'],
                      'status': 'rejected'}
            raw = (maps_dir / entry['path']).read_bytes()
            record['source_snapshot'] = f'source-documents/{index:06d}.json'
            (out / record['source_snapshot']).write_bytes(raw)
            record['source_snapshot_sha256'] = _sha(raw)
            report['records'].append(record)
            if record['source_snapshot_sha256'] != entry['document_hash']:
                raise ValueError(f'source document changed during preparation: {entry["path"]}')
            try:
                doc = MapDocument.from_dict(json.loads(raw))
                target = build_target(doc, tags=tags, target_game=target_game)
                validate_target(target)
                target_path = f'targets/{index:06d}.json'
                target_raw = (json.dumps(target, indent=2, allow_nan=False) + '\n').encode()
                (out / target_path).write_bytes(target_raw)
                record.update(status='exported', target=target_path, target_sha256=_sha(target_raw),
                              target_fingerprint=target['fingerprint'], counts=target['audit']['counts'])
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                record['reason'] = f'{type(exc).__name__}: {exc}'
            _save(out / 'report.json', report)
        report['source_changed_during_preparation'] = (
            _source_fingerprint(_source_bytes()) != report['source_fingerprint']
            or Path(__file__).read_bytes() != tool_raw or manifest.read_bytes() != manifest_raw
            or any(_sha((maps_dir / row['path']).read_bytes()) != row['source_document_sha256']
                   for row in report['records']))
        if report['source_changed_during_preparation']:
            raise ValueError('source/code changed during preparation; retain this run and use a fresh directory')
        report.update(status='complete', exported=sum(row['status'] == 'exported' for row in report['records']),
                      rejected=sum(row['status'] == 'rejected' for row in report['records']))
    except Exception as exc:
        report.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
        _save(out / 'report.json', report)
        raise
    _save(out / 'report.json', report)
    return report


def _artifact(root, name):
    if not isinstance(name, str) or not (root.resolve() / name).resolve().is_relative_to(root.resolve()):
        raise ValueError('prepared artifact must stay within the preparation directory')
    return root / name


def verify_prepared(out):
    """Verify inputs, helper/code/runtime and every saved target before reuse.

    Returns exported records only after verifying rejected source snapshots too.
    This does not upgrade not-assessed eligibility into positive training labels.
    """
    out = Path(out)
    report = json.loads((out / 'report.json').read_bytes())
    if report.get('schema') != SCHEMA or report.get('status') != 'complete':
        raise ValueError('preparation must have completed under the current schema')
    if report.get('source_changed_during_preparation') or report.get('source_fingerprint_scope') != SOURCE_SCOPE:
        raise ValueError('preparation source identity is invalid')
    if _runtime() != report['runtime']:
        raise ValueError('preparation runtime changed; rebuild in a fresh directory')
    sources = _source_bytes()
    if _source_fingerprint(sources) != report['source_fingerprint']:
        raise ValueError('preprocessing source changed; rebuild in a fresh directory')
    if any((out / 'source-code' / name).read_bytes() != raw for name, raw in sources.items()):
        raise ValueError('preparation source snapshot changed')
    if (Path(__file__).read_bytes() != (out / 'prepare_learned_geometry.py').read_bytes()
            or _sha(Path(__file__).read_bytes()) != report['tool_sha256']):
        raise ValueError('preparation helper changed')
    manifest_raw = (out / 'source-manifest.json').read_bytes()
    if _sha(manifest_raw) != report['source_manifest_sha256']:
        raise ValueError('source manifest snapshot changed')
    frozen = json.loads(manifest_raw)
    expected_entries = frozen['entries'][:report['limit'] or None]
    if len(expected_entries) != len(report['records']):
        raise ValueError('preparation omitted source records')
    exported = []
    for entry, row in zip(expected_entries, report['records']):
        if any(row[name] != entry[name] for name in ('path', 'family', 'group', 'split')):
            raise ValueError('prepared source identity or split changed')
        source_raw = _artifact(out, row['source_snapshot']).read_bytes()
        if (_sha(source_raw) != entry['document_hash']
                or _sha(source_raw) != row['source_snapshot_sha256']
                or row['source_document_sha256'] != entry['document_hash']):
            raise ValueError('prepared source snapshot changed')
        if row['status'] == 'rejected':
            continue
        if row['status'] != 'exported':
            raise ValueError('incomplete prepared record')
        raw = _artifact(out, row['target']).read_bytes()
        if _sha(raw) != row['target_sha256']:
            raise ValueError('prepared target changed')
        target = json.loads(raw)
        validate_target(target)
        source_doc = MapDocument.from_dict(json.loads(source_raw))
        expected_target = build_target(source_doc, tags=report['conditioning']['tags'],
                                       target_game=report['conditioning']['target_game'])
        if (target['fingerprint'] != row['target_fingerprint']
                or target['conditioning'] != report['conditioning']
                or target['fingerprint'] != expected_target['fingerprint']):
            raise ValueError('prepared target/source binding changed')
        exported.append(row)
    return exported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps-dir', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--target-game', default='')
    args = parser.parse_args()
    try:
        report = prepare(args.maps_dir, args.manifest, args.out, limit=args.limit, target_game=args.target_game)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"{report['exported']} exported, {report['rejected']} rejected; geometry/progression eligibility not assessed")


if __name__ == '__main__':
    main()

"""Frozen-family preparation is separate from old corpora and training labels."""
import json

import pytest

from data.corpus import build_manifest
from data.learned_geometry import digest
from tests.geometry_fixtures import polygon_document, square, two_rooms
from tools.prepare_learned_geometry import prepare, verify_prepared


def sources(tmp_path):
    root = tmp_path / 'maps'
    root.mkdir()
    for index in range(3):
        source = (polygon_document([[square(0, 0, 128, 128)]], start=(64, 64))
                  if index == 0 else two_rooms())
        source.meta.wad_hash = f'family-{index}'
        source.vertices[-1].x += 8 * index
        if index == 2:
            source.linedefs[0].args = [0] * 5  # Preserved explicit rejection.
        (root / f'{index}.json').write_text(json.dumps(source.to_dict()))
    manifest = build_manifest(root)
    path = tmp_path / 'corpus.json'
    path.write_text(json.dumps(manifest))
    return root, path, manifest


def test_preparation_preserves_family_assignments_rejections_and_not_assessed_scope(tmp_path):
    root, manifest, frozen = sources(tmp_path)
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    out = tmp_path / 'prepared'
    report = prepare(root, manifest, out)
    assert report['status'] == 'complete'
    assert (report['exported'], report['rejected']) == (2, 1)
    assert report['geometry_eligibility'] == report['progression_eligibility'] == 'not_assessed'
    assert {row['split'] for row in report['records']} == {'train', 'validation', 'test'}
    assert [row['split'] for row in report['records']] == [entry['split'] for entry in frozen['entries']]
    rejected = next(row for row in report['records'] if row['status'] == 'rejected')
    assert 'args' in rejected['reason']
    assert (out / rejected['source_snapshot']).exists()
    assert len(verify_prepared(out)) == 2
    assert before == {path.name: path.read_bytes() for path in root.iterdir()}
    assert not (out / 'corpus.json').exists()  # Cannot masquerade as old trainer inputs.
    with pytest.raises(ValueError, match='fresh directory'):
        prepare(root, manifest, out)


@pytest.mark.parametrize('change', ['target', 'source', 'code', 'tool', 'split', 'omission', 'escape'])
def test_prepared_identity_checks_detect_changed_inputs_or_snapshots(tmp_path, change):
    root, manifest, _ = sources(tmp_path)
    out = tmp_path / 'prepared'
    report = prepare(root, manifest, out)
    row = next(row for row in report['records'] if row['status'] == 'exported')
    if change in {'target', 'source'}:
        path = out / row['target' if change == 'target' else 'source_snapshot']
        path.write_bytes(path.read_bytes() + b' ')
    elif change == 'code':
        path = out / 'source-code/data/learned_geometry.py'
        path.write_bytes(path.read_bytes() + b'\n')
    elif change == 'tool':
        path = out / 'prepare_learned_geometry.py'
        path.write_bytes(path.read_bytes() + b'\n')
    else:
        if change == 'split':
            row['split'] = 'different'
        elif change == 'omission':
            report['records'].pop()
        else:
            row['source_snapshot'] = '../outside.json'
        (out / 'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):
        verify_prepared(out)


def test_rehashed_target_cannot_claim_unchanged_source_geometry(tmp_path):
    from tools.prepare_learned_geometry import _sha
    root, manifest, _ = sources(tmp_path)
    out = tmp_path / 'prepared'
    report = prepare(root, manifest, out)
    row = next(row for row in report['records'] if row['status'] == 'exported')
    path = out / row['target']
    target = json.loads(path.read_bytes())
    target['audit']['source_meta']['source_wad'] = '<changed-source>'
    target['fingerprint'] = digest({k: v for k, v in target.items() if k != 'fingerprint'})
    path.write_text(json.dumps(target))
    row.update(target_sha256=_sha(path.read_bytes()), target_fingerprint=target['fingerprint'])
    (out / 'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='binding changed'):
        verify_prepared(out)


def test_bounded_export_keeps_original_splits_and_excludes_unrequested_maps(tmp_path):
    root, manifest, frozen = sources(tmp_path)
    out = tmp_path / 'prepared'
    report = prepare(root, manifest, out, limit=1)
    assert len(report['records']) == 1
    assert report['records'][0]['split'] == frozen['entries'][0]['split']
    assert len(list((out / 'source-documents').iterdir())) == 1
    assert len(verify_prepared(out)) == 1


def test_changed_source_is_rejected_before_creating_preparation(tmp_path):
    root, manifest, _ = sources(tmp_path)
    path = root / '0.json'
    path.write_bytes(path.read_bytes() + b' ')
    out = tmp_path / 'prepared'
    with pytest.raises(ValueError, match='document changed'):
        prepare(root, manifest, out)
    assert not out.exists()

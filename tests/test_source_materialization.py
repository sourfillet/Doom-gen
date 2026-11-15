import json

import pytest

from data.corpus import training_split
from tests.pipeline_fixtures import source_wads
from tools.audit_sources import audit
from tools.materialize_sources import materialize


def test_materialized_complete_maps_have_verified_documents_and_a_loadable_manifest(tmp_path):
    sources = source_wads(tmp_path)
    audit(sources, tmp_path / 'audit', workers=2)
    out = tmp_path / 'corpus'
    report = materialize(tmp_path / 'audit/report.json', out, validation_fraction=.2, test_fraction=.3)
    splits, manifest = training_split(out / 'maps', out / 'corpus.json')
    assert report['status'] == 'complete' and not report['code_changed_during_materialization']
    assert len(splits['train']) == len(report['records']) == 1
    assert not splits['validation'] and not splits['test']
    assert report['records'][0]['ending']['complete']
    assert report['splits'] == {'train': 1}
    assert manifest['validation_fraction'] == .2 and manifest['test_fraction'] == .3
    assert manifest['entries'][0]['wad_hash'] == report['records'][0]['wad_hash']
    with pytest.raises(ValueError, match='must be empty'):
        materialize(tmp_path / 'audit/report.json', out)
    path = tmp_path / 'audit/report.json'
    changed = json.loads(path.read_text())
    changed['runtime']['shapely'] = 'different'
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='runtime changed'):
        materialize(path, tmp_path / 'other-corpus')
    assert not (tmp_path / 'other-corpus').exists()
    for fractions in [(.5, .5), (float('nan'), .1), (.1, -1)]:
        with pytest.raises(ValueError, match='holdout fractions'):
            materialize(path, tmp_path / 'bad-split', validation_fraction=fractions[0], test_fraction=fractions[1])
        assert not (tmp_path / 'bad-split').exists()


def test_partial_audits_or_changed_assessment_code_cannot_publish_training_corpora(tmp_path):
    source = source_wads(tmp_path)
    audit(source, tmp_path / 'audit', limit=1, workers=1)
    path = tmp_path / 'audit/report.json'
    with pytest.raises(ValueError, match='completed audit'):
        materialize(path, tmp_path / 'corpus')
    report = json.loads(path.read_text())
    report.update(status='complete', code_fingerprint='changed')
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='code changed'):
        materialize(path, tmp_path / 'corpus')
    assert not (tmp_path / 'corpus').exists()


def test_fragment_materialization_retains_incomplete_labels_and_checks_changed_wads(tmp_path):
    from data.roomgraph_targets import graph_to_sequence
    from roomgraph.types import RoomGraph
    source = source_wads(tmp_path)
    audit(source, tmp_path / 'audit', workers=2)
    path = tmp_path / 'audit/report.json'
    report = materialize(path, tmp_path / 'fragments', include_fragments=True)
    assert report['complete_endings'] == 1 and len(report['records']) == 2
    assert sum(row['ending']['complete'] for row in report['records']) == 1
    for row in report['records']:
        snapshot = json.loads((tmp_path / 'fragments/records' / row['path']).read_text())
        assert snapshot['graph_snapshot_version'] == 1
        assert graph_to_sequence(RoomGraph.from_dict(snapshot['graph_snapshot'])) == snapshot['sequence']
    splits, _ = training_split(tmp_path / 'fragments/maps', tmp_path / 'fragments/corpus.json')
    assert len(splits['train']) == 2
    (source / 'complete.WAD').write_bytes((source / 'complete.WAD').read_bytes() + b'x')
    with pytest.raises(ValueError, match='eligibility changed'):
        materialize(path, tmp_path / 'changed')
    assert not (tmp_path / 'changed/corpus.json').exists()
    failure = json.loads((tmp_path / 'changed/report.json').read_text())
    assert failure['status'] == 'error'


def test_exported_audit_snapshots_are_reused_and_corruption_is_rejected(tmp_path, monkeypatch):
    import tools.materialize_sources
    source = source_wads(tmp_path)
    report = audit(source, tmp_path / 'audit', workers=2, export_documents=True)
    assert all('graph_snapshot' not in row for row in report['records'])
    assert all(row.get('record_sha256') and row.get('document_sha256') for row in report['records'])
    def cannot_reextract(*args, **kwargs):
        raise AssertionError('exported audit targets should be reused')
    monkeypatch.setattr(tools.materialize_sources, '_run_job', cannot_reextract)
    prepared = materialize(tmp_path / 'audit/report.json', tmp_path / 'corpus', include_fragments=True)
    assert prepared['reused_audit_snapshots'] == 2
    for row in report['records']:
        filename = row['id'] + '.json'
        assert (tmp_path / 'audit/maps' / filename).read_bytes() == (tmp_path / 'corpus/maps' / filename).read_bytes()
    changed = tmp_path / 'audit/maps' / (report['records'][0]['id'] + '.json')
    changed.write_bytes(changed.read_bytes() + b' ')
    with pytest.raises(ValueError, match='snapshot changed'):
        materialize(tmp_path / 'audit/report.json', tmp_path / 'changed', include_fragments=True)
    with pytest.raises(ValueError, match='document changed'):
        audit(source, tmp_path / 'audit', resume=True, export_documents=True)

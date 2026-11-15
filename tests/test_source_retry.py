import hashlib
import json

import pytest

from tests.pipeline_fixtures import source_wads
from tools.audit_sources import audit
from tools.materialize_sources import materialize
from tools.retry_source_audit import retry


def interrupted_export(tmp_path):
    sources = source_wads(tmp_path)
    root = tmp_path / 'audit'
    report = audit(sources, root, workers=2, export_documents=True)
    row = next(row for row in report['records'] if not row['complete'])
    record = root / 'records' / (row['id'] + '.json')
    raw = json.loads(record.read_text())
    raw.update(status='timeout', stage='document_output', reason='simulated export timeout')
    record.write_text(json.dumps(raw))
    row.update(status=raw['status'], stage=raw['stage'], reason=raw['reason'],
               record_sha256=hashlib.sha256(record.read_bytes()).hexdigest())
    report['counts'] = {'assessed': 1, 'timeout': 1}
    path = root / 'report.json'
    path.write_text(json.dumps(report))
    return sources, path


def test_retry_branches_preserve_evidence_and_publish_materializable_snapshots(tmp_path):
    sources, path = interrupted_export(tmp_path)
    parent_bytes = {p.relative_to(path.parent): p.read_bytes() for p in path.parent.rglob('*') if p.is_file()}
    branch = tmp_path / 'retry'
    report = retry(path, sources, branch, workers=1)
    assert report['status'] == 'complete' and not report['code_changed_during_audit']
    assert report['counts'] == {'assessed': 2} and report['complete_endings'] == 1
    assert report['completed_retries'] == report['retried_maps'] == 1
    assert report['retry_parent']['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    for name, raw in parent_bytes.items():
        assert (path.parent / name).read_bytes() == raw
    prepared = materialize(branch / 'report.json', tmp_path / 'corpus', include_fragments=True)
    assert prepared['reused_audit_snapshots'] == 2 and prepared['complete_endings'] == 1
    # Copies must be independent: modifying a branch cannot corrupt the parent.
    original = next((path.parent / 'maps').glob('*.json'))
    (branch / 'maps' / original.name).write_bytes(b'changed')
    assert original.read_bytes() == parent_bytes[original.relative_to(path.parent)]
    with pytest.raises(ValueError, match='must be empty'):
        retry(path, sources, branch)


def test_retry_rejects_changed_sources_and_corrupt_parent_snapshots(tmp_path):
    sources, path = interrupted_export(tmp_path)
    row = next(row for row in json.loads(path.read_text())['records'] if row['status'] == 'assessed')
    doc = path.parent / 'maps' / (row['id'] + '.json')
    doc.write_bytes(doc.read_bytes() + b' ')
    with pytest.raises(ValueError, match='parent document changed'):
        retry(path, sources, tmp_path / 'bad-copy')
    assert json.loads((tmp_path / 'bad-copy/report.json').read_text())['status'] == 'error'
    (sources / 'complete.WAD').write_bytes((sources / 'complete.WAD').read_bytes() + b'x')
    with pytest.raises(ValueError, match='inventory changed'):
        retry(path, sources, tmp_path / 'bad-source')
    assert not (tmp_path / 'bad-source').exists()

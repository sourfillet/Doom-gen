import json

import pytest

from tests.pipeline_fixtures import source_wads
from tests.wad_builder import build_two_sector_wad
from tools.audit_sources import audit, inventory


def test_bounded_audit_resumes_without_repeating_completed_maps_and_keeps_fragments(tmp_path):
    source = source_wads(tmp_path)
    out = tmp_path / 'audit'
    report = audit(source, out, workers=1, limit=1)
    assert report['status'] == 'partial' and report['processed'] == 1 and report['total_maps'] == 2
    record = out / 'records' / (report['records'][0]['id'] + '.json')
    before, timestamp = record.read_bytes(), record.stat().st_mtime_ns
    report = audit(source, out, workers=2, resume=True)
    assert report['status'] == 'complete' and report['processed'] == 2
    assert record.read_bytes() == before and record.stat().st_mtime_ns == timestamp
    assert report['counts'] == {'assessed': 2} and report['complete_endings'] == 1
    assert report['mapless_files'] == 1
    assert not report['code_changed_during_audit']
    assert (out / 'audit_sources.py').exists()
    assert not any(path.suffix == '.tmp' for path in out.rglob('*'))
    with pytest.raises(ValueError, match='must be empty'):
        audit(source, out)
    with pytest.raises(ValueError, match='changed'):
        audit(source, out, timeout=61, resume=True)
    (source / 'fragment.wad').write_bytes(build_two_sector_wad() + b'x')
    with pytest.raises(ValueError, match='changed'):
        audit(source, out, resume=True)


def test_audit_timeout_is_inconclusive_and_preserves_raw_inventory(tmp_path):
    source = source_wads(tmp_path)
    (source / 'broken.wad').write_bytes(b'not a WAD')
    out = tmp_path / 'audit'
    report = audit(source, out, workers=1, timeout=0.001)
    assert report['counts'] == {'timeout': 2} and report['complete_endings'] == 0
    assert report['inventory_errors'][0]['path'].endswith('broken.wad')
    assert report['mapless_files'] == 1  # broken WAD is a separate failure
    assert all(row['stage'] == 'worker_start' for row in report['records'])
    saved = json.loads((out / 'inventory.json').read_text())
    assert any(job['specials'].get('11') for job in saved['jobs'])
    assert all('lump_sizes' in job for job in saved['jobs'])


def test_source_audit_deduplicates_identical_wads_without_losing_file_inventory(tmp_path):
    source = source_wads(tmp_path)
    (source / 'copy.WAD').write_bytes((source / 'fragment.wad').read_bytes())
    result = inventory(source)
    assert len(result['files']) == 4 and len(result['jobs']) == 2


def test_positive_assessment_followed_by_export_timeout_is_not_eligible(tmp_path, monkeypatch):
    def export_timeout(job, output, timeout, *, document_output=None):
        row = {**job, 'status': 'timeout', 'stage': 'document_output', 'complete': True,
               'ending': {'complete': True}, 'reason': 'export budget exhausted'}
        output.write_text(json.dumps(row))
        return row
    monkeypatch.setattr('tools.audit_sources._run_job', export_timeout)
    report = audit(source_wads(tmp_path), tmp_path / 'audit', export_documents=True)
    assert report['status'] == 'complete' and report['counts'] == {'timeout': 2}
    assert report['complete_endings'] == 0
    assert all(row['complete'] for row in report['records'])  # retain partial evidence

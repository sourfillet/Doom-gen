"""Tests for geometry generation method comparison."""
from __future__ import annotations

import json

import pytest

from tools.compare_geometry_methods import ROOT


def test_current_comparison_checks_actual_candidates_and_never_claims_import_as_engine_execution(tmp_path, monkeypatch):
    import sys
    import types
    from data.spatial_targets import build_spatial_target
    from roomgraph.shapes import build_library
    from tests.geometry_fixtures import two_rooms
    from tests.geometry_fixtures import two_room_graph
    from tools.compare_geometry_methods import evaluate_case
    # An importable engine is not an executed engine, even on a fully valid panel.
    monkeypatch.setitem(sys.modules, 'vizdoom', types.ModuleType('vizdoom'))
    monkeypatch.setattr('generation.pipeline.build_nodes', lambda path: False)
    doc, graph = two_rooms(), two_room_graph()
    target = build_spatial_target(doc, graph)
    records = evaluate_case(graph, target, tmp_path / 'panel', build_library([doc], playable_only=True))
    rows = {r['method']: r for r in records}
    assert rows['authored_reconstruction']['status'] == 'published'
    assert rows['authored_reconstruction']['final_validation']['errors'] == []
    assert rows['learned_scaffold']['status'] == 'skipped'
    assert all(r['engine_status'] == 'not_run' for r in records)
    assert all(r['requested_connections'] == 1 for r in records)
    assert 'learned_geometry' not in rows
    assert (tmp_path / 'panel/authored_reconstruction.wad').exists()
    before = (tmp_path / 'panel/authored_reconstruction.wad').read_bytes()
    with pytest.raises(FileExistsError):
        evaluate_case(graph, target, tmp_path / 'panel', {})
    assert (tmp_path / 'panel/authored_reconstruction.wad').read_bytes() == before


def test_unreachable_authored_reconstruction_is_rejected_and_candidate_is_preserved(tmp_path, monkeypatch):
    from data.spatial_targets import build_spatial_target
    from tests.geometry_fixtures import polygon_document, square
    from tests.geometry_fixtures import two_room_graph
    from tools.compare_geometry_methods import evaluate_case
    doc = polygon_document([[square(0, 0, 128, 128)], [square(512, 0, 640, 128)]], start=(64, 64))
    def reject(*args, **kwargs):
        raise ValueError('injected routing failure')
    monkeypatch.setattr('tools.compare_geometry_methods.assemble', reject)
    monkeypatch.setattr('tools.compare_geometry_methods.assemble_real', reject)
    records = evaluate_case(two_room_graph(), build_spatial_target(doc), tmp_path / 'panel', {})
    authored = records[0]
    assert authored['status'] == 'rejected'
    assert authored['in_memory_validation']['unreachable_sectors'] == [1]
    assert authored['serialized_validation']['unreachable_sectors'] == [1]
    assert (tmp_path / 'panel' / authored['candidate_wad']).exists()
    assert not (tmp_path / 'panel/authored_reconstruction.wad').exists()


def test_gallery_uses_report_counts_and_discloses_engine_and_conditioning(tmp_path):
    from tools.render_geometry_comparison import build_html_report
    report = {'schema': 'doom_geometry_comparison_v2', 'seed': 17, 'scope': 'Scaffold control only.',
              'engine_status': 'not_run', 'engine_reason': 'no controller',
              'summary': {'learned_scaffold': {'published_count': 0, 'total_count': 4, 'status': 'rejected'}},
              'records': [{'method': 'learned_scaffold', 'conditioning': 'source authoring scaffold',
                           'reason': 'inference rejected <candidate>'}]}
    (tmp_path / 'report.json').write_text(json.dumps(report))
    html = build_html_report(tmp_path).read_text()
    assert '0 / 4' in html and '3 / 3' not in html and '83.8%' not in html
    assert 'source authoring scaffold' in html and 'not_run' in html
    assert '&lt;candidate&gt;' in html and '<candidate>' not in html
    with pytest.raises(ValueError, match='already exists'):
        build_html_report(tmp_path)


def test_shape_library_enforces_frozen_training_hashes_and_never_reads_holdouts(tmp_path, monkeypatch):
    import hashlib
    from tests.geometry_fixtures import two_rooms
    from tools.compare_geometry_methods import ensure_training_shape_library
    base = tmp_path / 'generated/repair-experiments'
    maps = base / 'current-complete-corpus/maps'
    maps.mkdir(parents=True)
    raw = json.dumps(two_rooms().to_dict()).encode()
    (maps / 'train.json').write_bytes(raw)
    manifest = base / 'adjoining-architecture/spatial-manifest-v1.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'entries': [
        {'path': 'train.json', 'split': 'train', 'source_document_sha256': hashlib.sha256(raw).hexdigest()},
        {'path': 'missing-holdout.json', 'split': 'test', 'source_document_sha256': 'unused'}]}))
    monkeypatch.setattr('tools.compare_geometry_methods.ROOT', tmp_path)
    good = tmp_path / 'good'
    good.mkdir()
    ensure_training_shape_library(good)
    provenance = json.loads((good / 'shape-library-provenance.json').read_text())
    assert len(provenance['inputs']) == 1 and provenance['inputs'][0]['sha256'] == hashlib.sha256(raw).hexdigest()
    assert (good / 'shape-library-manifest.json').read_bytes() == manifest.read_bytes()
    (maps / 'train.json').write_bytes(raw + b' ')
    bad = tmp_path / 'bad'
    bad.mkdir()
    with pytest.raises(ValueError, match='frozen training source fingerprint'):
        ensure_training_shape_library(bad)
    assert not (bad / 'training_shape_library.json').exists()


def test_geometry_comparison_report_exists_and_valid():
    exp_dir = ROOT / "generated/repair-experiments/adjoining-architecture/geometry-comparison-v1"
    report_file = exp_dir / "report.json"
    if not report_file.exists():
        pytest.skip("geometry comparison report not yet generated")

    report = json.loads(report_file.read_text())
    assert report["schema"] == "doom_geometry_comparison_v1"
    assert report["seed"] == 7
    assert len(report["records"]) == 9

    summary = report["summary"]
    assert summary["learned_geometry"]["published_count"] == 3
    assert summary["procedural_adjoining"]["published_count"] == 2
    assert summary["outline_retrieval"]["published_count"] == 1

    # Check dmonfear_map19_full outcomes
    full_records = {r["method"]: r for r in report["records"] if r["panel"] == "dmonfear_map19_full"}
    assert full_records["learned_geometry"]["status"] == "published"
    assert full_records["learned_geometry"]["metrics"]["sectors"] == 61
    assert full_records["learned_geometry"]["metrics"]["validation_errors"] == []
    assert full_records["procedural_adjoining"]["status"] == "failed"
    assert "8->11" in full_records["procedural_adjoining"]["reason"]
    assert full_records["outline_retrieval"]["status"] == "failed"
    assert "no collision-free corridor route" in full_records["outline_retrieval"]["reason"]

    # Check dmonfear_map19_masked outcomes
    masked_records = {r["method"]: r for r in report["records"] if r["panel"] == "dmonfear_map19_masked"}
    assert masked_records["learned_geometry"]["status"] == "published"
    assert masked_records["procedural_adjoining"]["status"] == "published"
    assert masked_records["procedural_adjoining"]["metrics"]["sectors"] == 121
    assert masked_records["procedural_adjoining"]["metrics"]["validation_errors"] == []
    assert masked_records["outline_retrieval"]["status"] == "failed"

    # Check two_rooms_fixture outcomes
    fixture_records = {r["method"]: r for r in report["records"] if r["panel"] == "two_rooms_fixture"}
    assert fixture_records["learned_geometry"]["status"] == "published"
    assert fixture_records["procedural_adjoining"]["status"] == "published"
    assert fixture_records["outline_retrieval"]["status"] == "published"

    for r in report["records"]:
        if r["status"] == "published":
            wad_path = exp_dir / r["wad_file"]
            assert wad_path.exists()
            assert r["metrics"]["geometry_issues"] == []
            assert r["metrics"]["validation_errors"] == []


def test_geometry_comparison_html_gallery():
    exp_dir = ROOT / "generated/repair-experiments/adjoining-architecture/geometry-comparison-v1"
    html_file = exp_dir / "index.html"
    if not html_file.exists():
        pytest.skip("index.html not yet generated")

    content = html_file.read_text()
    assert "<title>Geometry Generation Comparison" in content
    assert "Learned Geometry" in content
    assert "Procedural Adjoining" in content
    assert "Outline Retrieval" in content
    assert "<svg" in content


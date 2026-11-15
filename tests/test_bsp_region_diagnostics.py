import copy
import hashlib
import json
from pathlib import Path

import pytest
from shapely import Polygon, box

from tests.geometry_fixtures import two_rooms
from tests.wad_builder import build_two_sector_wad_with_nodes
from tools.diagnose_bsp_regions import compare_regions, main, summarize, trace_leaves


def test_empty_mismatches_preserve_lost_area_and_cannot_disappear_from_summary():
    empty, filled = Polygon(), box(0, 0, 128, 128)
    rows = [compare_regions(a, b) for a, b in [(empty, filled), (filled, empty), (empty, empty)]]
    assert rows[0]['rebuilt_area'] == rows[1]['source_area'] == 16384
    assert rows[0]['symmetric_difference_area'] == rows[1]['symmetric_difference_area'] == 16384
    assert rows[0]['hausdorff'] is None
    result = summarize(rows)
    assert result['beyond_one_unit'] == 0
    assert result['empty_mismatches'] == result['unresolved'] == 2
    assert result['topology_changes'] == 2


def test_thin_interior_slit_is_kept_even_when_its_area_is_tiny():
    original = box(0, 0, 256, 256)
    # A tiny area delta can introduce a boundary deep inside playable space.
    slit = box(127.999999, 0, 128.000001, 128)
    rebuilt = original.difference(slit)
    before = rebuilt.wkb
    result = compare_regions(original, rebuilt)
    assert result['symmetric_difference_area'] < 0.001
    assert result['hausdorff'] > 100
    assert result['difference_outside_source_boundary_band'] > 0
    assert rebuilt.wkb == before
    assert summarize([result])['beyond_one_unit'] == 1


def test_invalid_regions_remain_unresolved_without_repair():
    bowtie = Polygon([(0, 0), (2, 2), (0, 2), (2, 0)])
    result = compare_regions(box(0, 0, 2, 2), bowtie)
    assert result['status'] == 'invalid_geometry'
    assert summarize([result])['unresolved'] == 1


def test_leaf_evidence_retains_self_referencing_owners_and_actions_without_mutation():
    doc = two_rooms()
    line = next(line for line in doc.linedefs if line.left_sidedef >= 0)
    right = doc.sidedefs[line.right_sidedef].sector
    doc.sidedefs[line.left_sidedef].sector = right
    line.special, line.tag = 123, 7
    before = copy.deepcopy(doc.to_dict())
    result = trace_leaves([(3, right, box(0, 0, 128, 128), [(0, 1, 0, line.id, 0, 0)])],
                          doc, box(16, 16, 32, 32))
    assert len(result) == 1 and result[0]['difference_overlap_area'] == 256
    assert result[0]['all_boundaries_self_referencing']
    assert result[0]['boundaries'][0]['special'] == 123
    assert result[0]['boundaries'][0]['tag'] == 7
    assert doc.to_dict() == before


def test_panel_run_preserves_inputs_snapshots_and_rejects_overwriting(tmp_path):
    decoder = Path(__file__).resolve().parents[1] / 'wad_parser/bsp_geometry.py'
    prior = tmp_path / 'prior.json'
    prior.write_text(json.dumps({'schema': 'exact_authored_line_bsp_probe_v2', 'status': 'complete',
                                'candidate_decoder_sha256': hashlib.sha256(decoder.read_bytes()).hexdigest(),
                                'records': [{'path': 'fixture.json', 'map': 'MAP01', 'split': 'train'}]}))
    artifacts = tmp_path / 'reconstruction/fixture'
    artifacts.mkdir(parents=True)
    wad = build_two_sector_wad_with_nodes()
    for name in ('raw.wad', 'compiled.wad'):
        (artifacts / name).write_bytes(wad)
    out = tmp_path / 'diagnostic'
    args = ['--prior-report', str(prior), '--candidate-decoder', str(decoder),
            '--reconstruction-root', str(artifacts.parent), '--out', str(out)]
    assert main(args) == 0
    report = json.loads((out / 'report.json').read_text())
    assert report['status'] == 'complete' and not report['source_changed']
    assert len(report['records']) == 1
    assert (out / 'source/wad_parser/bsp_geometry.py').read_bytes() == decoder.read_bytes()
    assert (out / 'inputs/fixture/compiled.wad').read_bytes() == wad
    assert (artifacts / 'compiled.wad').read_bytes() == wad
    assert json.loads((out / 'fixture.json').read_text())['summary']['unresolved'] == 0
    with pytest.raises(FileExistsError):
        main(args)

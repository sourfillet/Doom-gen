"""Tests for single-source spatial geometry overfit probe."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

torch = pytest.importorskip("torch")

from cli.overfit_spatial import main
from data.spatial_targets import build_spatial_target, authoring_digest
from data.spatial_prediction import decode_prediction
from models.spatial import predict_checkpoint
from tests.geometry_fixtures import two_rooms
from roomgraph.types import Room, RoomGraph, Connection


def test_overfit_spatial_directory_must_be_empty(tmp_path):
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("must be preserved")

    result = CliRunner().invoke(main, ["--out", str(tmp_path)])
    assert result.exit_code != 0
    assert "must be empty" in result.output
    assert sentinel.read_text() == "must be preserved"


def test_overfit_spatial_synthetic_target(tmp_path):
    doc = two_rooms()
    rooms = [
        Room(0, [0], 0, 128, 512**2, (0, 0), (0, 0, 512, 512), roles=["start"]),
        Room(1, [1], 0, 128, 512**2, (512, 0), (512, 0, 1024, 512), roles=["exit"]),
    ]
    graph = RoomGraph(rooms, [Connection(0, 1, "door", 64)])
    target = build_spatial_target(doc, graph)

    target_file = tmp_path / "target.json"
    target_file.write_text(json.dumps(target))

    out_dir = tmp_path / "out"
    result = CliRunner().invoke(
        main,
        [
            "--target-json",
            str(target_file),
            "--out",
            str(out_dir),
            "--steps",
            "600",
            "--hidden",
            "32",
            "--lr",
            "0.01",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert (out_dir / "report.json").exists()
    assert (out_dir / "target.json").exists()
    assert (out_dir / "model.pt").exists()
    assert (out_dir / "candidate.wad").exists()

    report = json.loads((out_dir / "report.json").read_text())
    assert report["schema"] == "doom_spatial_overfit_v2"
    assert report["assembled_geometry"]["sectors"] == 2
    assert report["assembled_geometry"]["connected_validation"]["errors"] == []
    assert report["assembled_geometry"]["serialization_preserves_authoring"]
    assert report["convergence"]["loss_reduction_pct"] > 0.0
    assert report["convergence"]["subunit_reconstruction"]
    assert report["engine_status"] == "not_run"
    assert report["publication_status"] == "diagnostic_only"
    # Loading the saved model without the source target must reproduce inference.
    target_file.unlink()
    (out_dir / 'target.json').unlink()
    scaffold, prediction = predict_checkpoint(out_dir / 'model.pt')
    assert authoring_digest(decode_prediction(scaffold, prediction)) == authoring_digest(doc)
    prediction['values'][0][0] += 8 / scaffold['frame']['scale']
    assert authoring_digest(decode_prediction(scaffold, prediction)) != authoring_digest(doc)


def test_spatial_checkpoint_rejects_wrong_scaffold_and_legacy_heads(tmp_path):
    from models.spatial import CHECKPOINT_SCHEMA, SpatialScaffoldModel
    from data.spatial_prediction import build_prediction_scaffold, digest
    scaffold, _ = build_prediction_scaffold(build_spatial_target(two_rooms()))
    model = SpatialScaffoldModel(scaffold, hidden=8)
    path = tmp_path / 'model.pt'
    torch.save({'schema': CHECKPOINT_SCHEMA, 'scaffold': scaffold,
                'scaffold_sha256': digest(scaffold), 'model': model.state_dict(), 'hidden': 8}, path)
    import copy
    changed = copy.deepcopy(scaffold)
    changed['sidedefs'][0]['mid_tex'] = 'STONE2'
    with pytest.raises(ValueError, match='another source scaffold'):
        predict_checkpoint(path, expected_scaffold=changed)
    torch.save(model.state_dict(), path)
    with pytest.raises(ValueError, match='unsupported spatial checkpoint'):
        predict_checkpoint(path)


@pytest.mark.parametrize('kind', ['truncated', 'corrupt', 'root_type', 'incomplete'])
def test_spatial_checkpoint_reports_invalid_weights_as_explicit_input_errors(tmp_path, kind):
    from models.spatial import CHECKPOINT_SCHEMA
    path = tmp_path / 'bad.pt'
    if kind == 'truncated':
        path.write_bytes(b'')
    elif kind == 'corrupt':
        path.write_bytes(b'corrupt spatial checkpoint')
    elif kind == 'root_type':
        torch.save('wrong root type', path)
    else:
        torch.save({'schema': CHECKPOINT_SCHEMA}, path)
    with pytest.raises(ValueError, match='spatial checkpoint'):
        predict_checkpoint(path)


def test_overfit_spatial_dmonfear_map19_convergence():
    report_file = Path(
        "generated/repair-experiments/adjoining-architecture/spatial-overfit-v1/report.json"
    )
    if not report_file.exists():
        pytest.skip("overfit experiment report not yet generated")

    report = json.loads(report_file.read_text())
    assert report["schema"] == "doom_spatial_overfit_v1"
    assert report["stem"] == "6ce8b4b0e602140261a202180dbe7c317453b8f1991985258e503643caf2ee26-0018"
    assert report["convergence"]["converged"] is True
    assert report["convergence"]["loss_reduction_pct"] > 99.0
    assert report["assembled_geometry"]["sectors"] == 61
    assert report["assembled_geometry"]["geometry_issues"] == []
    assert report["assembled_geometry"]["validation_errors"] == []
    assert report["exclusions_applied"]["unsupported_edges"] == [[8, 11]]
    assert report["exclusions_applied"]["outside_audit_count"] == 2


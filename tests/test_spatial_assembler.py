"""Unit tests for topology-conditioned spatial geometry assembler."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from data.spatial_geometry import build_spatial_geometry_target
from generation.validator import validate_document
from roomgraph.assemble_spatial import assemble_spatial_geometry
from roomgraph.types import Connection, Room, RoomGraph
from tests.geometry_fixtures import decorative_start, two_room_graph, two_rooms
from wad_parser.types import MapDocument


def test_assemble_spatial_geometry_two_rooms():
    doc = two_rooms()
    graph = two_room_graph()
    target = build_spatial_geometry_target(doc, graph)

    assembled = assemble_spatial_geometry(target)

    assert assembled.schema_version == "1.0"
    assert len(assembled.sectors) == 2
    assert len(assembled.things) == 1
    assert assembled.things[0].type == 1

    two_sided = [ld for ld in assembled.linedefs if ld.flags.two_sided]
    one_sided = [ld for ld in assembled.linedefs if not ld.flags.two_sided]
    assert len(two_sided) == 1
    assert len(one_sided) == 6
    assert len(assembled.linedefs) == 7

    assert assembled.derived.geometry_issues == []
    val = validate_document(assembled, require_exit=False)
    assert val["errors"] == []


def test_assemble_spatial_relative_rings():
    doc = two_rooms()
    graph = two_room_graph()
    target = build_spatial_geometry_target(doc, graph)

    # Strip absolute exterior_ring to force use of relative_exterior_ring + center
    stripped_target = copy.deepcopy(target)
    for room in stripped_target["geometry"]["interiors"]:
        for comp in room["components"]:
            del comp["exterior_ring"]

    assembled = assemble_spatial_geometry(stripped_target)

    assert len(assembled.sectors) == 2
    assert len(assembled.linedefs) == 7
    assert assembled.derived.geometry_issues == []
    val = validate_document(assembled, require_exit=False)
    assert val["errors"] == []


def test_assemble_spatial_multiheight_room():
    doc = two_rooms()
    doc.sectors[1].floor_height = 24
    doc.sectors[1].ceil_height = 160
    graph = two_room_graph()
    target = build_spatial_geometry_target(doc, graph)

    assembled = assemble_spatial_geometry(target)

    assert len(assembled.sectors) == 2
    shared_line = next(ld for ld in assembled.linedefs if ld.flags.two_sided)
    assert shared_line is not None

    # Verify step textures exist on two-sided line
    r_sd = assembled.sidedefs[shared_line.right_sidedef]
    l_sd = assembled.sidedefs[shared_line.left_sidedef]
    assert r_sd.lower_tex != "-" or l_sd.lower_tex != "-"

    assert assembled.derived.geometry_issues == []
    val = validate_document(assembled, require_exit=False)
    assert val["errors"] == []


def test_assemble_spatial_courtyard_with_hole():
    doc = decorative_start()
    rooms = [
        Room(0, [0], 0, 128, 512**2, (0, 0), (0, 0, 512, 512), roles=['start']),
        Room(1, [1], 0, 128, 64**2, (256, 256), (240, 240, 272, 272), roles=['exit']),
    ]
    graph = RoomGraph(rooms, [Connection(0, 1, 'door', 64)])
    target = build_spatial_geometry_target(doc, graph)

    assembled = assemble_spatial_geometry(target)

    assert len(assembled.sectors) == 2
    assert assembled.derived.geometry_issues == []
    val = validate_document(assembled, require_exit=False)
    assert val["errors"] == []


def test_assemble_spatial_invalid_schema_rejected():
    with pytest.raises(ValueError, match="unsupported spatial geometry schema"):
        assemble_spatial_geometry({"schema": "invalid_schema"})

    target = {
        "schema": "doom_spatial_geometry_v1",
        "frame": {"origin": [0, 0], "scale": 1.0, "floor_origin": 0},
        "geometry": {"interiors": []},
    }
    with pytest.raises(ValueError, match="contains no interior sectors"):
        assemble_spatial_geometry(target)


def test_assemble_spatial_real_dmonfear_map19(tmp_path):
    stem = "6ce8b4b0e602140261a202180dbe7c317453b8f1991985258e503643caf2ee26-0018"
    corpus = Path("generated/repair-experiments/current-complete-corpus")
    base = Path("generated/repair-experiments/adjoining-architecture")
    doc_path = corpus / "maps" / f"{stem}.json"
    if not doc_path.exists():
        pytest.skip("corpus map not available")

    doc = MapDocument.from_dict(json.loads(doc_path.read_text()))
    graph_snapshot = json.loads(
        (corpus / "records" / f"{stem}.json").read_text()
    )["graph_snapshot"]
    graph = RoomGraph.from_dict(graph_snapshot)
    audit = json.loads((base / "spatial-audit-v1" / f"{stem}.json").read_text())
    index = json.loads((base / "spatial-index-v2" / f"{stem}.json").read_text())

    target = build_spatial_geometry_target(
        doc, graph, audit_data=audit, index_data=index
    )

    assembled = assemble_spatial_geometry(target)

    assert len(assembled.sectors) == 61
    assert len(assembled.linedefs) == 370
    assert len(assembled.sidedefs) == 487
    assert len(assembled.vertices) == 310
    assert assembled.derived.geometry_issues == []

    val = validate_document(assembled, require_exit=False)
    assert val["errors"] == []

    # Write WAD and build nodes with zdbsp if available
    import shutil
    from wad_writer.wad_writer import write_wad, build_nodes
    wad_file = tmp_path / "map19_assembled.wad"
    write_wad([assembled], wad_file)
    assert wad_file.exists()
    if shutil.which("zdbsp"):
        assert build_nodes(wad_file) is True

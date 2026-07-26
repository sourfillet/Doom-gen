import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wad_builder import build_two_sector_wad, build_two_sector_wad_with_nodes

from wad_parser.map_extractor import extract_all_maps
from wad_parser.wad_reader import WadFile


def _write_wad(tmp_path) -> str:
    p = tmp_path / "synthetic.wad"
    p.write_bytes(build_two_sector_wad())
    return str(p)


def test_wad_header_and_lumps(tmp_path):
    wad = WadFile(_write_wad(tmp_path))
    assert wad.identification == "PWAD"
    assert wad.num_lumps == 6
    assert wad.lump_names() == ["MAP01", "THINGS", "LINEDEFS", "SIDEDEFS", "VERTEXES", "SECTORS"]


def test_find_map_groups(tmp_path):
    wad = WadFile(_write_wad(tmp_path))
    groups = wad.find_map_groups()
    assert len(groups) == 1
    assert groups[0].name == "MAP01"
    assert groups[0].format == "doom"


def test_extract_map_core_counts(tmp_path):
    wad = WadFile(_write_wad(tmp_path))
    docs = extract_all_maps(wad)
    assert len(docs) == 1
    doc = docs[0]
    assert len(doc.vertices) == 6
    assert len(doc.sectors) == 2
    assert len(doc.linedefs) == 7
    assert len(doc.sidedefs) == 8
    assert len(doc.things) == 2
    assert doc.meta.map_name == "MAP01"
    assert doc.meta.format == "doom"
    assert doc.meta.wad_hash == wad.hash


def test_thing_resolution(tmp_path):
    wad = WadFile(_write_wad(tmp_path))
    doc = extract_all_maps(wad)[0]
    player_start = doc.things[0]
    assert player_start.category == "player-start"
    assert player_start.type_name == "Player1Start"
    assert player_start.flags.skill_1_2 and player_start.flags.skill_3 and player_start.flags.skill_4_5
    assert not player_start.unresolved

    monster = doc.things[1]
    assert monster.type_name == "Zombieman"
    assert monster.category == "monster"


def test_linedef_flags_decoded(tmp_path):
    wad = WadFile(_write_wad(tmp_path))
    doc = extract_all_maps(wad)[0]
    shared_wall = doc.linedefs[1]
    assert shared_wall.flags.two_sided
    assert shared_wall.right_sidedef == 0
    assert shared_wall.left_sidedef == 1

    south_wall = doc.linedefs[0]
    assert south_wall.flags.impassable
    assert south_wall.left_sidedef == -1


def test_derived_sector_adjacency(tmp_path):
    wad = WadFile(_write_wad(tmp_path))
    doc = extract_all_maps(wad)[0]
    edges = doc.derived.sector_graph["edges"]
    assert len(edges) == 1
    assert {edges[0]["a"], edges[0]["b"]} == {0, 1}


def test_derived_sector_polygons_are_closed_rings(tmp_path):
    wad = WadFile(_write_wad(tmp_path))
    doc = extract_all_maps(wad)[0]
    for ring in doc.derived.sector_polygons:
        assert len(ring) == 4


def test_derived_reachability_covers_both_sectors(tmp_path):
    wad = WadFile(_write_wad(tmp_path))
    doc = extract_all_maps(wad)[0]
    assert doc.derived.reachable_from_start == [0, 1]


def test_derived_texture_alignment_score_present(tmp_path):
    wad = WadFile(_write_wad(tmp_path))
    doc = extract_all_maps(wad)[0]
    assert doc.derived.texture_alignment_score is not None
    assert 0.0 <= doc.derived.texture_alignment_score <= 1.0


def test_bounds(tmp_path):
    wad = WadFile(_write_wad(tmp_path))
    doc = extract_all_maps(wad)[0]
    assert doc.bounds.min_x == 0
    assert doc.bounds.max_x == 256
    assert doc.bounds.min_y == 0
    assert doc.bounds.max_y == 128


def test_linedef_special_name_resolved(tmp_path):
    wad = WadFile(_write_wad(tmp_path))
    doc = extract_all_maps(wad)[0]
    # our synthetic map's linedefs are all special=0
    assert doc.linedefs[0].special_name == "None"
    assert doc.linedefs[0].trigger_type == "none"


def test_meta_is_iwad_false_for_pwad(tmp_path):
    wad = WadFile(_write_wad(tmp_path))
    doc = extract_all_maps(wad)[0]
    assert doc.meta.is_iwad is False


def test_derived_normalized_space(tmp_path):
    wad = WadFile(_write_wad(tmp_path))
    doc = extract_all_maps(wad)[0]
    norm = doc.derived.normalized_space
    assert norm is not None
    assert norm["center_x"] == 128
    assert norm["center_y"] == 64
    assert len(norm["vertices"]) == len(doc.vertices)
    # every normalized coordinate should land within [-1, 1]
    for x, y in norm["vertices"]:
        assert -1.0 <= x <= 1.0
        assert -1.0 <= y <= 1.0


def test_derived_bsp_sanity_check_absent_without_nodes(tmp_path):
    wad = WadFile(_write_wad(tmp_path))
    doc = extract_all_maps(wad)[0]
    assert doc.derived.bsp_sanity_check is None


def test_derived_bsp_sanity_check_with_nodes(tmp_path):
    p = tmp_path / "with_nodes.wad"
    p.write_bytes(build_two_sector_wad_with_nodes())
    wad = WadFile(str(p))
    doc = extract_all_maps(wad)[0]
    check = doc.derived.bsp_sanity_check
    assert check is not None
    assert check["subsector_count"] == 2
    assert check["segs_total"] == 8
    assert check["segs_covered_by_subsectors"] == 8
    assert check["subsectors_contiguous"] is True
    assert check["sectors_seen_in_bsp"] == 2
    assert check["sectors_missing_from_reconstruction"] == []
    assert check["reject_size_matches_expected"] is True


def test_document_serializes_to_json(tmp_path):
    import json
    wad = WadFile(_write_wad(tmp_path))
    doc = extract_all_maps(wad)[0]
    payload = json.dumps(doc.to_dict())
    reloaded = json.loads(payload)
    assert reloaded["meta"]["map_name"] == "MAP01"
    assert reloaded["schema_version"] == "1.0"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wad_builder import build_two_sector_wad

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


def test_document_serializes_to_json(tmp_path):
    import json
    wad = WadFile(_write_wad(tmp_path))
    doc = extract_all_maps(wad)[0]
    payload = json.dumps(doc.to_dict())
    reloaded = json.loads(payload)
    assert reloaded["meta"]["map_name"] == "MAP01"
    assert reloaded["schema_version"] == "1.0"

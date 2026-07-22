import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wad_builder import build_hexen_wad, build_udmf_wad

from wad_parser.map_extractor import extract_all_maps
from wad_parser.udmf_parser import parse_textmap, build_linedefs, build_sectors, build_things, build_vertices
from wad_parser.wad_reader import WadFile


def test_hexen_format_detected(tmp_path):
    p = tmp_path / "hexen.wad"
    p.write_bytes(build_hexen_wad())
    wad = WadFile(str(p))
    groups = wad.find_map_groups()
    assert groups[0].format == "hexen"


def test_hexen_thing_and_linedef_fields(tmp_path):
    p = tmp_path / "hexen.wad"
    p.write_bytes(build_hexen_wad())
    wad = WadFile(str(p))
    doc = extract_all_maps(wad)[0]

    player = doc.things[0]
    assert player.tid == 1
    assert player.category == "player-start"

    teleport_dest = doc.things[1]
    assert teleport_dest.tid == 5
    assert teleport_dest.type_name == "TeleportDest"

    door_line = doc.linedefs[3]
    assert door_line.special == 12
    assert door_line.args == [16, 35, 0, 0, 0]


def test_udmf_parses_to_same_field_shapes(tmp_path):
    p = tmp_path / "udmf.wad"
    p.write_bytes(build_udmf_wad())
    wad = WadFile(str(p))
    groups = wad.find_map_groups()
    assert groups[0].format == "udmf"

    doc = extract_all_maps(wad)[0]
    assert len(doc.vertices) == 4
    assert len(doc.sectors) == 1
    assert len(doc.linedefs) == 4
    assert len(doc.things) == 1

    thing = doc.things[0]
    assert thing.category == "player-start"
    assert thing.flags.skill_1_2 and thing.flags.skill_3 and thing.flags.skill_4_5

    triggered_line = doc.linedefs[2]
    assert triggered_line.special == 1
    assert triggered_line.tag == 5
    assert triggered_line.args == [16, 0, 0, 0, 0]

    sector = doc.sectors[0]
    assert sector.floor_slope is not None
    assert sector.floor_slope.b == 0.1


def test_udmf_vertex_zfloor_captured():
    doc = parse_textmap(__import__("wad_builder").UDMF_TEXTMAP)
    vertices = build_vertices(doc)
    assert vertices[2].z == 16.0


def test_udmf_unknown_field_goes_to_extra():
    from wad_parser.udmf_parser import parse_textmap, build_sectors
    text = b'sector { heightfloor = 0; heightceiling = 64; texturefloor = "FLAT1"; textureceiling = "FLAT1"; lightlevel = 100; xpanningfloor = 32.0; }'
    doc = parse_textmap(text)
    sectors = build_sectors(doc)
    assert sectors[0].extra.get("xpanningfloor") == 32.0

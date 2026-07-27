"""Tests for the regenerated lookup tables (sourced from doomwiki.org and
ZDoom's zspecial.acs), Hexen-specific flag decoding, Boom generalized
specials, and the UDMF capture-completeness fixes.
"""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wad_builder import _assemble_wad, _pad8, build_two_sector_wad

from wad_parser.lump_parser import parse_linedefs_hexen, parse_things_hexen
from wad_parser.map_extractor import extract_all_maps
from wad_parser.mapinfo_parser import parse_mapinfo
from wad_parser.types import (
    DOOMED_TABLE, HEXEN_SPECIAL_TABLE, LINEDEF_SPECIAL_TABLE,
    resolve_linedef_special,
)
from wad_parser.udmf_parser import (
    build_linedefs, build_sidedefs, build_things, build_vertices, parse_textmap,
)
from wad_parser.wad_reader import WadFile


# ---------------------------------------------------------------------------
# Lookup tables (values verified against doomwiki.org / zspecial.acs)
# ---------------------------------------------------------------------------

def test_doom_linedef_special_triggers():
    # (special, trigger): entries the old hand-written table got wrong.
    expected = {
        7: "switch", 9: "switch", 21: "switch", 23: "switch",   # S1, not walk
        31: "push", 32: "push", 33: "push", 34: "push",         # D1 manual doors
        52: "walk",                                             # W1 NORMAL exit
        109: "walk", 110: "walk",                               # W1, not switch
        111: "switch",
        136: "switch",                                          # SR, not walk
        39: "walk",                                             # was missing entirely
        123: "switch", 88: "walk",                              # were missing
    }
    for special, trigger in expected.items():
        assert LINEDEF_SPECIAL_TABLE[special][1] == trigger, special


def test_vanilla_linedef_specials_all_present():
    missing = [n for n in range(1, 142) if n not in LINEDEF_SPECIAL_TABLE]
    assert missing == []


def test_boom_generalized_special_decoded():
    name, trigger = resolve_linedef_special(0x3C00)  # generalized door, W1
    assert name == "GenDoor (W1)" and trigger == "walk"
    name, trigger = resolve_linedef_special(0x6000 | 0x0003)  # gen floor, SR
    assert name == "GenFloor (SR)" and trigger == "switch"
    name, trigger = resolve_linedef_special(0x2F80 | 0x0005)  # gen crusher, GR
    assert name == "GenCrusher (GR)" and trigger == "gun"


def test_hexen_action_special_numbering():
    assert HEXEN_SPECIAL_TABLE[62][0] == "Plat_DownWaitUpStay"
    assert HEXEN_SPECIAL_TABLE[70][0] == "Teleport"
    assert HEXEN_SPECIAL_TABLE[80][0] == "ACS_Execute"
    assert HEXEN_SPECIAL_TABLE[120][0] == "Radius_Quake"
    assert HEXEN_SPECIAL_TABLE[121][0] == "Line_SetIdentification"
    assert HEXEN_SPECIAL_TABLE[138][0] == "Floor_Waggle"


def test_doomed_table_corrections():
    assert DOOMED_TABLE[36][0] == "Short green pillar with beating heart"
    assert DOOMED_TABLE[37][0] == "Short red pillar with skull"
    assert DOOMED_TABLE[55][0] == "Short blue firestick"
    assert DOOMED_TABLE[73] == ("Hanging victim, guts removed", "obstacle")
    # 59-63 are the NON-blocking hanging decorations
    assert DOOMED_TABLE[63] == ("Hanging victim, twitching", "decoration")
    # blocking-ness now tracks the doomwiki obstacle tables
    assert DOOMED_TABLE[44][1] == "obstacle"   # tall blue firestick blocks
    assert DOOMED_TABLE[41][1] == "obstacle"   # evil eye blocks
    assert DOOMED_TABLE[34][1] == "decoration"  # candle does not
    assert 2057 not in DOOMED_TABLE  # "Chair" never existed


# ---------------------------------------------------------------------------
# Hexen flag decoding
# ---------------------------------------------------------------------------

def _hexen_linedef(flags: int, special: int = 12) -> bytes:
    return struct.pack("<HHH", 0, 1, flags) + bytes([special, 0, 0, 0, 0, 0]) + struct.pack("<HH", 0, 0xFFFF)


def test_hexen_spac_gives_trigger_type():
    lines = parse_linedefs_hexen(
        _hexen_linedef(0x0400 | 0x0200)      # SPAC_USE + repeatable
        + _hexen_linedef(0x0000)             # SPAC_CROSS
        + _hexen_linedef(0x1000)             # SPAC_PUSH (player bumps)
        + _hexen_linedef(0x0C00)             # SPAC_IMPACT
        + _hexen_linedef(0x0400, special=0)  # no special -> no trigger
    )
    assert lines[0].trigger_type == "switch" and lines[0].flags.repeat_special
    assert lines[1].trigger_type == "walk" and not lines[1].flags.repeat_special
    assert lines[2].trigger_type == "push"
    assert lines[3].trigger_type == "gun"
    assert lines[4].trigger_type == "none"


def test_hexen_linedef_0x200_is_not_boom_passthrough():
    (line,) = parse_linedefs_hexen(_hexen_linedef(0x0200))
    assert line.flags.repeat_special and not line.flags.pass_through


def _hexen_thing(flags: int) -> bytes:
    return struct.pack("<hhhhhhH", 0, 0, 0, 0, 0, 1, flags) + bytes(6)


def test_hexen_thing_flags():
    things = parse_things_hexen(
        _hexen_thing(0x0007 | 0x0010 | 0x0100)        # all skills, dormant, single-only
        + _hexen_thing(0x0020 | 0x0100 | 0x0200 | 0x0400)  # fighter, all modes
    )
    t = things[0]
    assert t.flags.dormant
    assert not t.flags.not_single_player       # 0x0100 set -> in single player
    assert t.flags.not_coop and t.flags.not_deathmatch
    t = things[1]
    assert t.flags.class_fighter and not t.flags.class_cleric
    assert not (t.flags.not_single_player or t.flags.not_coop or t.flags.not_deathmatch)
    assert not t.flags.dormant                 # 0x0010 is NOT "not in single player"


# ---------------------------------------------------------------------------
# UDMF capture-completeness
# ---------------------------------------------------------------------------

def test_udmf_sidedef_extra_preserved():
    doc = parse_textmap(b'sidedef { sector = 0; texturemiddle = "STARTAN2"; scalex_mid = 2.0; light = 16; }')
    (sd,) = build_sidedefs(doc)
    assert sd.extra == {"scalex_mid": 2.0, "light": 16}


def test_udmf_thing_skills_stay_in_extra():
    doc = parse_textmap(b'thing { x = 0.0; y = 0.0; type = 1; skill1 = true; skill3 = true; single = true; }')
    (t,) = build_things(doc)
    assert t.flags.skill_1_2 and t.flags.skill_3 and not t.flags.skill_4_5
    assert t.extra.get("skill1") is True and t.extra.get("skill3") is True
    assert not t.flags.not_single_player
    assert t.flags.not_deathmatch and t.flags.not_coop  # absent = not present, per spec


def test_udmf_zceiling_only_vertex_not_mislabeled():
    doc = parse_textmap(b'vertex { x = 0.0; y = 0.0; zceiling = 32.0; }')
    (v,) = build_vertices(doc)
    assert v.z is None
    assert v.extra["zceiling"] == 32.0


def test_udmf_hexen_ns_activation_fields_give_trigger():
    text = b'''
namespace = "zdoom";
linedef { v1 = 0; v2 = 1; sidefront = 0; special = 80; arg0 = 1; playeruse = true; repeatspecial = true; }
linedef { v1 = 1; v2 = 2; sidefront = 0; special = 70; arg0 = 2; playercross = true; monsteractivate = true; }
'''
    doc = parse_textmap(text)
    lines = build_linedefs(doc)
    assert lines[0].special_name == "ACS_Execute"
    assert lines[0].trigger_type == "switch" and lines[0].flags.repeat_special
    assert lines[1].special_name == "Teleport"
    assert lines[1].trigger_type == "walk" and lines[1].flags.monsters_can_activate


# ---------------------------------------------------------------------------
# UDMF map grouping
# ---------------------------------------------------------------------------

def test_udmf_group_collects_everything_to_endmap(tmp_path):
    textmap = b'namespace = "zdoom";\nvertex { x = 0.0; y = 0.0; }\n'
    lumps = [
        ("SPACEMAP", b""),          # non-standard marker name, legal in UDMF
        ("TEXTMAP", textmap),
        ("MYCUSTOM", b"custom-data"),  # unknown lump inside the block
        ("ZNODES", b"znode-data"),
        ("ENDMAP", b""),
    ]
    p = tmp_path / "u.wad"
    p.write_bytes(_assemble_wad(lumps))
    (group,) = WadFile(str(p)).find_map_groups()
    assert group.name == "SPACEMAP" and group.format == "udmf"
    assert group.lumps["MYCUSTOM"] == b"custom-data"
    assert group.lumps["ZNODES"] == b"znode-data"


# ---------------------------------------------------------------------------
# Teleport edges
# ---------------------------------------------------------------------------

def test_boom_line_teleport_edge(tmp_path):
    # Reuse the two-sector map but give linedef 0 (sector 0) a 243 line-
    # teleport special tagged to linedef 5 (sector 1).
    import wad_builder
    vertices, linedefs, sidedefs, sectors, things = wad_builder._two_sector_core()
    linedefs = [list(l) for l in linedefs]
    linedefs[0][3], linedefs[0][4] = 243, 9   # special, tag on a sector-0 line
    linedefs[5][4] = 9                        # same tag on a sector-1 line
    linedefs = [tuple(l) for l in linedefs]

    things_data = b"".join(struct.pack("<hhhhH", *t) for t in things)
    linedefs_data = b"".join(struct.pack("<HHHHHHH", *l) for l in linedefs)
    sidedefs_data = b"".join(
        struct.pack("<hh", sd[0], sd[1]) + _pad8(sd[2]) + _pad8(sd[3]) + _pad8(sd[4]) + struct.pack("<H", sd[5])
        for sd in sidedefs
    )
    vertexes_data = b"".join(struct.pack("<hh", *v) for v in vertices)
    sectors_data = b"".join(
        struct.pack("<hh", s[0], s[1]) + _pad8(s[2]) + _pad8(s[3]) + struct.pack("<hhh", s[4], s[5], s[6])
        for s in sectors
    )
    p = tmp_path / "t.wad"
    p.write_bytes(_assemble_wad([
        ("MAP01", b""), ("THINGS", things_data), ("LINEDEFS", linedefs_data),
        ("SIDEDEFS", sidedefs_data), ("VERTEXES", vertexes_data), ("SECTORS", sectors_data),
    ]))
    doc = extract_all_maps(WadFile(str(p)))[0]
    teleport_edges = [e for e in doc.derived.sector_graph["edges"] if e["kind"] == "teleport"]
    assert {"a": 0, "b": 1, "linedef": 0, "kind": "teleport"} in teleport_edges


# ---------------------------------------------------------------------------
# UMAPINFO
# ---------------------------------------------------------------------------

def test_umapinfo_parsed():
    text = b'''
MAP MAP01
{
    levelname = "The Gantlet"
    music = "D_RUNNIN"
    skytexture = "SKY1"
    next = "MAP02"
}
'''
    entries = parse_mapinfo(text)
    e = entries["MAP01"]
    assert e.level_name == "The Gantlet"
    assert e.music == "D_RUNNIN"
    assert e.sky == "SKY1"
    assert e.next_map == "MAP02"


# ---------------------------------------------------------------------------
# WAD reader robustness
# ---------------------------------------------------------------------------

def test_corrupt_directory_rejected(tmp_path):
    wad = bytearray(build_two_sector_wad())
    # point the first real lump's filepos outside the file
    numlumps, infotableofs = struct.unpack_from("<ii", wad, 4)
    struct.pack_into("<ii", wad, infotableofs + 16, -50, 100)
    p = tmp_path / "bad.wad"
    p.write_bytes(bytes(wad))
    try:
        WadFile(str(p))
        assert False, "expected ValueError for corrupt directory"
    except ValueError:
        pass


def test_multiple_flat_spans_all_collected(tmp_path):
    lumps = [
        ("F_START", b""), ("FLATA", b"\x01" * 64), ("F_END", b""),
        ("OTHER", b"x"),
        ("F_START", b""), ("FLATB", b"\x02" * 64), ("F_END", b""),
    ]
    p = tmp_path / "flats.wad"
    p.write_bytes(_assemble_wad(lumps))
    flats = WadFile(str(p)).find_flats()
    assert set(flats) == {"FLATA", "FLATB"}

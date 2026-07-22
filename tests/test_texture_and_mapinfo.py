import struct

from wad_parser.mapinfo_parser import parse_mapinfo
from wad_parser.texture_parser import (
    parse_animated, parse_pnames, parse_switches, parse_texture_lump,
)


def _pad8(name: str) -> bytes:
    return name.encode("ascii").ljust(8, b"\x00")[:8]


def _build_pnames(names: list[str]) -> bytes:
    return struct.pack("<i", len(names)) + b"".join(_pad8(n) for n in names)


def _build_texture1(name: str, width: int, height: int, patch_indices: list[int]) -> bytes:
    header = struct.pack("<i", 1) + struct.pack("<i", 8)  # 1 texture, offset=8
    record = _pad8(name) + struct.pack("<i", 0) + struct.pack("<hh", width, height) + struct.pack("<i", 0)
    record += struct.pack("<h", len(patch_indices))
    for idx in patch_indices:
        record += struct.pack("<hhhhh", 0, 0, idx, 1, 0)
    return header + record


def test_pnames_and_texture1_roundtrip():
    pnames_data = _build_pnames(["STARTAN2", "STARTAN3"])
    pnames = parse_pnames(pnames_data)
    assert pnames == ["STARTAN2", "STARTAN3"]

    tex1_data = _build_texture1("WALL1", 64, 128, [0])
    textures = parse_texture_lump(tex1_data, pnames)
    assert "WALL1" in textures
    tex = textures["WALL1"]
    assert (tex.width, tex.height) == (64, 128)
    assert len(tex.patches) == 1
    assert tex.patches[0].name == "STARTAN2"


def test_animated_lump():
    entries = [
        (1, "BLODGR4", "BLODGR1", 8),  # texture cycle
        (0, "NUKAGE3", "NUKAGE1", 8),  # flat cycle
    ]
    data = b""
    for kind, last, first, speed in entries:
        data += struct.pack("<B", kind) + last.encode().ljust(9, b"\x00") + first.encode().ljust(9, b"\x00") + struct.pack("<i", speed)
    data += struct.pack("<B", 0xFF) + b"\x00" * 22

    parsed = parse_animated(data)
    assert len(parsed) == 2
    assert parsed[0].kind == "texture"
    assert parsed[0].first == "BLODGR1"
    assert parsed[0].last == "BLODGR4"
    assert parsed[1].kind == "flat"


def test_switches_lump():
    data = b""
    data += "SW1BRIK".encode().ljust(9, b"\x00") + "SW2BRIK".encode().ljust(9, b"\x00") + struct.pack("<H", 3)
    data += b"\x00" * 18 + struct.pack("<H", 0)  # terminator (game==0)
    parsed = parse_switches(data)
    assert len(parsed) == 1
    assert parsed[0].off_texture == "SW1BRIK"
    assert parsed[0].on_texture == "SW2BRIK"


MAPINFO_SAMPLE = b"""
map MAP01 "Entryway"
{
    music = "D_RUNNIN"
    sky1 = "SKY1"
    cluster = 1
    next = "MAP02"
}

map MAP02 "Underhalls"
{
    music = "D_STALKS"
    next = "MAP03"
}
"""


def test_mapinfo_brace_style():
    entries = parse_mapinfo(MAPINFO_SAMPLE)
    assert set(entries.keys()) == {"MAP01", "MAP02"}
    m1 = entries["MAP01"]
    assert m1.level_name == "Entryway"
    assert m1.music == "D_RUNNIN"
    assert m1.sky == "SKY1"
    assert m1.next_map == "MAP02"
    assert m1.cluster == "1"


EMAPINFO_SAMPLE = b"""
[MAP01]
levelname = Entryway
music = D_RUNNIN
next = MAP02
"""


def test_emapinfo_bracket_style():
    entries = parse_mapinfo(EMAPINFO_SAMPLE)
    assert "MAP01" in entries
    assert entries["MAP01"].level_name == "Entryway"
    assert entries["MAP01"].music == "D_RUNNIN"

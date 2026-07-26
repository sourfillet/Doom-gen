"""Builds minimal synthetic WAD bytes for tests, since real IWADs are
copyrighted and not available in this environment. Constructs just enough
of a valid binary-format PWAD (map marker + THINGS/LINEDEFS/SIDEDEFS/
VERTEXES/SECTORS) to exercise the parser end to end.
"""
from __future__ import annotations

import struct


def _pad8(name: str) -> bytes:
    return name.encode("ascii").ljust(8, b"\x00")[:8]


def _two_sector_core():
    """Two adjacent square sectors sharing one two-sided wall, one player
    start in sector 0, one monster in sector 1.

    Sector 0: (0,0)-(128,0)-(128,128)-(0,128)
    Sector 1: (128,0)-(256,0)-(256,128)-(128,128)
    Shared wall: (128,0)-(128,128)
    """
    vertices = [
        (0, 0), (128, 0), (128, 128), (0, 128),   # 0-3: sector 0 ring
        (256, 0), (256, 128),                        # 4-5: sector 1 outer corners
    ]

    # linedefs: (v1, v2, flags, special, tag, right_sd, left_sd)
    linedefs = [
        (0, 1, 0x0001, 0, 0, 0, 0xFFFF),   # sector0 south wall (one-sided)
        (1, 2, 0x0004, 0, 0, 1, 2),        # shared wall (two-sided): right=sector0 side, left=sector1 side
        (2, 3, 0x0001, 0, 0, 3, 0xFFFF),   # sector0 north wall
        (3, 0, 0x0001, 0, 0, 4, 0xFFFF),   # sector0 west wall
        (1, 4, 0x0001, 0, 0, 5, 0xFFFF),   # sector1 south wall
        (4, 5, 0x0001, 0, 0, 6, 0xFFFF),   # sector1 east wall
        (5, 2, 0x0001, 0, 0, 7, 0xFFFF),   # sector1 north wall
    ]

    # sidedefs: (x_off, y_off, upper, lower, mid, sector)
    sidedefs = [
        (0, 0, "-", "-", "-", 0),          # 0: shared wall, sector0 side (no mid tex, two-sided)
        (0, 0, "-", "-", "STARTAN2", 0),   # duplicate placeholder unused (kept for clarity, not referenced)
    ]
    # Rebuild sidedefs precisely matching linedef references above:
    sidedefs = [
        (0, 0, "-", "-", "-", 0),          # id0: shared wall sector0 side (right of linedef1)
        (0, 0, "-", "-", "-", 1),          # id1: shared wall sector1 side (left of linedef1)
        (0, 0, "-", "-", "STARTAN2", 0),   # id2: sector0 north wall (right of linedef2)
        (0, 0, "-", "-", "STARTAN2", 0),   # id3: sector0 west wall (right of linedef3)
        (0, 0, "-", "-", "STARTAN2", 1),   # id4: sector1 south wall (right of linedef4)
        (0, 0, "-", "-", "STARTAN2", 1),   # id5: sector1 east wall (right of linedef5)
        (0, 0, "-", "-", "STARTAN2", 1),   # id6: sector1 north wall (right of linedef6)
        (0, 0, "-", "-", "STARTAN2", 0),   # id7: sector0 south wall (right of linedef0)
    ]
    # Fix linedef sidedef indices to match the table above.
    linedefs = [
        (0, 1, 0x0001, 0, 0, 7, 0xFFFF),   # sector0 south wall -> sidedef 7 (sector0)
        (1, 2, 0x0004, 0, 0, 0, 1),        # shared wall -> sidedef0 (sector0 right), sidedef1 (sector1 left)
        (2, 3, 0x0001, 0, 0, 2, 0xFFFF),   # sector0 north wall -> sidedef2 (sector0)
        (3, 0, 0x0001, 0, 0, 3, 0xFFFF),   # sector0 west wall -> sidedef3 (sector0)
        (1, 4, 0x0001, 0, 0, 4, 0xFFFF),   # sector1 south wall -> sidedef4 (sector1)
        (4, 5, 0x0001, 0, 0, 5, 0xFFFF),   # sector1 east wall -> sidedef5 (sector1)
        (5, 2, 0x0001, 0, 0, 6, 0xFFFF),   # sector1 north wall -> sidedef6 (sector1)
    ]

    # sectors: (floor_h, ceil_h, floor_flat, ceil_flat, light, special, tag)
    sectors = [
        (0, 128, "FLOOR4_8", "CEIL3_5", 160, 0, 0),
        (0, 128, "FLOOR4_8", "CEIL3_5", 160, 0, 0),
    ]

    # things: (x, y, angle, type, flags)
    things = [
        (64, 64, 90, 1, 0x0007),     # Player1Start, all skills
        (192, 64, 0, 3004, 0x0007),  # Zombieman
    ]
    return vertices, linedefs, sidedefs, sectors, things


def build_two_sector_wad() -> bytes:
    vertices, linedefs, sidedefs, sectors, things = _two_sector_core()

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

    lumps = [
        ("MAP01", b""),
        ("THINGS", things_data),
        ("LINEDEFS", linedefs_data),
        ("SIDEDEFS", sidedefs_data),
        ("VERTEXES", vertexes_data),
        ("SECTORS", sectors_data),
    ]
    return _assemble_wad(lumps)


def build_two_sector_wad_with_nodes() -> bytes:
    """Same map as build_two_sector_wad(), plus hand-built SEGS/SSECTORS/
    REJECT lumps consistent with the linedef/sidedef data, so the BSP
    sanity-check derived-data pass has something real to cross-check against.
    """
    vertices, linedefs, sidedefs, sectors, things = _two_sector_core()

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

    # (v1, v2, angle, linedef_idx, direction, offset) - sector0's segs first,
    # then sector1's, so SSECTORS below can reference contiguous ranges.
    segs = [
        (0, 1, 0, 0, 0, 0),   # linedef0 front (sector0)
        (2, 3, 0, 2, 0, 0),   # linedef2 front (sector0)
        (3, 0, 0, 3, 0, 0),   # linedef3 front (sector0)
        (1, 2, 0, 1, 0, 0),   # linedef1 front = sector0 side
        (1, 4, 0, 4, 0, 0),   # linedef4 front (sector1)
        (4, 5, 0, 5, 0, 0),   # linedef5 front (sector1)
        (5, 2, 0, 6, 0, 0),   # linedef6 front (sector1)
        (2, 1, 0, 1, 1, 0),   # linedef1 back = sector1 side
    ]
    segs_data = b"".join(struct.pack("<hhhHhh", *s) for s in segs)

    # (numsegs, firstseg)
    ssectors = [(4, 0), (4, 4)]
    ssectors_data = b"".join(struct.pack("<HH", *s) for s in ssectors)

    reject_data = bytes([0])  # ceil(2*2/8) = 1 byte

    lumps = [
        ("MAP01", b""),
        ("THINGS", things_data),
        ("LINEDEFS", linedefs_data),
        ("SIDEDEFS", sidedefs_data),
        ("VERTEXES", vertexes_data),
        ("SEGS", segs_data),
        ("SSECTORS", ssectors_data),
        ("NODES", b""),
        ("SECTORS", sectors_data),
        ("REJECT", reject_data),
    ]
    return _assemble_wad(lumps)


def build_hexen_wad() -> bytes:
    """One square sector, one Hexen-format thing (with tid/special/args) and
    one Hexen-format linedef (with special/args), plus a BEHAVIOR lump so
    the format is detected as Hexen rather than Doom.
    """
    vertices = [(0, 0), (128, 0), (128, 128), (0, 128)]

    # Hexen linedef: (v1, v2, flags, special, arg1..5, right_sd, left_sd)
    linedefs = [
        (0, 1, 0x0001, 0, 0, 0, 0, 0, 0, 0, 0xFFFF),
        (1, 2, 0x0001, 0, 0, 0, 0, 0, 0, 1, 0xFFFF),
        (2, 3, 0x0001, 0, 0, 0, 0, 0, 0, 2, 0xFFFF),
        # door special (Door_Raise generalized special 12 in hexen numbering, args = speed/delay/lock)
        (3, 0, 0x0004, 12, 16, 35, 0, 0, 0, 3, 4),
    ]
    sidedefs = [
        (0, 0, "-", "-", "STARTAN2", 0),
        (0, 0, "-", "-", "STARTAN2", 0),
        (0, 0, "-", "-", "STARTAN2", 0),
        (0, 0, "-", "-", "-", 0),
        (0, 0, "-", "-", "-", 0),
    ]
    sectors = [(0, 128, "FLOOR4_8", "CEIL3_5", 160, 0, 0)]

    # Hexen thing: (tid, x, y, z, angle, type, flags, special, arg1..5)
    things = [
        (1, 64, 64, 0, 90, 1, 0x0007, 0, 0, 0, 0, 0, 0),
        (5, 96, 96, 0, 0, 14, 0x0007, 0, 0, 0, 0, 0, 0),  # TeleportDest w/ tid
    ]

    things_data = b"".join(
        struct.pack("<hhhhhhH", t[0], t[1], t[2], t[3], t[4], t[5], t[6]) + bytes(t[7:13])
        for t in things
    )
    linedefs_data = b"".join(
        struct.pack("<HHH", l[0], l[1], l[2]) + bytes(l[3:9]) + struct.pack("<HH", l[9], l[10])
        for l in linedefs
    )
    sidedefs_data = b"".join(
        struct.pack("<hh", sd[0], sd[1]) + _pad8(sd[2]) + _pad8(sd[3]) + _pad8(sd[4]) + struct.pack("<H", sd[5])
        for sd in sidedefs
    )
    vertexes_data = b"".join(struct.pack("<hh", *v) for v in vertices)
    sectors_data = b"".join(
        struct.pack("<hh", s[0], s[1]) + _pad8(s[2]) + _pad8(s[3]) + struct.pack("<hhh", s[4], s[5], s[6])
        for s in sectors
    )

    lumps = [
        ("MAP01", b""),
        ("THINGS", things_data),
        ("LINEDEFS", linedefs_data),
        ("SIDEDEFS", sidedefs_data),
        ("VERTEXES", vertexes_data),
        ("SECTORS", sectors_data),
        ("BEHAVIOR", b"\x00" * 4),  # presence alone signals Hexen format
    ]
    return _assemble_wad(lumps)


UDMF_TEXTMAP = b"""
namespace = "zdoom";

thing // 0
{
x = 64.0;
y = 64.0;
angle = 90;
type = 1;
skill1 = true;
skill2 = true;
skill3 = true;
skill4 = true;
skill5 = true;
single = true;
};

vertex { x = 0.0; y = 0.0; }
vertex { x = 128.0; y = 0.0; }
vertex { x = 128.0; y = 128.0; zfloor = 16.0; }
vertex { x = 0.0; y = 128.0; }

linedef
{
v1 = 0;
v2 = 1;
sidefront = 0;
blocking = true;
}
linedef
{
v1 = 1;
v2 = 2;
sidefront = 1;
blocking = true;
}
linedef
{
v1 = 2;
v2 = 3;
sidefront = 2;
blocking = true;
special = 1;
id = 5;
arg0 = 16;
}
linedef
{
v1 = 3;
v2 = 0;
sidefront = 3;
blocking = true;
}

sidedef { sector = 0; texturemiddle = "STARTAN2"; }
sidedef { sector = 0; texturemiddle = "STARTAN2"; }
sidedef { sector = 0; texturemiddle = "STARTAN2"; }
sidedef { sector = 0; texturemiddle = "STARTAN2"; }

sector
{
heightfloor = 0;
heightceiling = 128;
texturefloor = "FLOOR4_8";
textureceiling = "CEIL3_5";
lightlevel = 160;
id = 5;
floorplane_a = 0.0;
floorplane_b = 0.1;
floorplane_c = 1.0;
floorplane_d = 0.0;
}
"""


def build_udmf_wad() -> bytes:
    lumps = [
        ("MAP01", b""),
        ("TEXTMAP", UDMF_TEXTMAP),
        ("ENDMAP", b""),
    ]
    return _assemble_wad(lumps)


def _assemble_wad(lumps: list[tuple[str, bytes]]) -> bytes:
    header = b"PWAD" + struct.pack("<ii", len(lumps), 0)  # infotableofs patched below
    body = b""
    directory = b""
    offset = 12
    for name, data in lumps:
        body += data
        directory += struct.pack("<ii", offset, len(data)) + _pad8(name)
        offset += len(data)
    infotableofs = 12 + len(body)
    header = b"PWAD" + struct.pack("<ii", len(lumps), infotableofs)
    return header + body + directory

"""MapDocument -> binary map lump bytes (Doom format), the inverse of
wad_parser.lump_parser.

Flag words are rebuilt from the typed flag fields PLUS each element's
preserved `unrecognized_flag_bits`, so parse -> assemble round-trips
byte-exactly even for maps carrying junk bits (e.g. the IWAD's E2M7).

BSP lumps (SEGS/SSECTORS/NODES/REJECT/BLOCKMAP) are written EMPTY: they
are deterministic artifacts of the geometry, ZDoom-family ports rebuild
them on load, and a node builder (zdbsp) can fill them in later for
vanilla-port compatibility (plan/generation.md, WAD Writer).
"""
from __future__ import annotations

import struct

from wad_parser.lump_parser import _LINEDEF_FLAG_BITS, _THING_FLAG_BITS
from wad_parser.types import MapDocument

# Vanilla lump order after the map marker.
MAP_LUMP_ORDER = [
    "THINGS", "LINEDEFS", "SIDEDEFS", "VERTEXES", "SEGS",
    "SSECTORS", "NODES", "SECTORS", "REJECT", "BLOCKMAP",
]


def _name8(name: str) -> bytes:
    return name.upper().encode("ascii", errors="replace").ljust(8, b"\x00")[:8]


def _clamp16(v: float) -> int:
    return max(-32768, min(32767, int(round(v))))


def _encode_flags(flag_obj, bit_table, extra: dict) -> int:
    raw = extra.get("unrecognized_flag_bits", 0)
    for bit, attr in bit_table:
        if getattr(flag_obj, attr):
            raw |= bit
    return raw


def assemble_map_lumps(doc: MapDocument) -> dict[str, bytes]:
    """Returns {lump_name: bytes} for one map. Doom binary format only -
    Hexen/UDMF output is a later phase (plan roadmap Phase 8)."""
    if doc.meta.format != "doom":
        raise ValueError(f"only doom-format output is supported, got {doc.meta.format!r}")

    things = b"".join(
        struct.pack("<hhhhH", _clamp16(t.x), _clamp16(t.y), _clamp16(t.angle),
                    t.type, _encode_flags(t.flags, _THING_FLAG_BITS, t.extra) & 0xFFFF)
        for t in doc.things
    )
    linedefs = b"".join(
        struct.pack(
            "<HHHHHHH", ld.v1, ld.v2,
            _encode_flags(ld.flags, _LINEDEF_FLAG_BITS, ld.extra) & 0xFFFF,
            ld.special & 0xFFFF, ld.tag & 0xFFFF,
            ld.right_sidedef & 0xFFFF if ld.right_sidedef != -1 else 0xFFFF,
            ld.left_sidedef & 0xFFFF if ld.left_sidedef != -1 else 0xFFFF,
        )
        for ld in doc.linedefs
    )
    sidedefs = b"".join(
        struct.pack("<hh", _clamp16(sd.x_offset), _clamp16(sd.y_offset))
        + _name8(sd.upper_tex) + _name8(sd.lower_tex) + _name8(sd.mid_tex)
        + struct.pack("<H", sd.sector)
        for sd in doc.sidedefs
    )
    vertexes = b"".join(
        struct.pack("<hh", _clamp16(v.x), _clamp16(v.y)) for v in doc.vertices
    )
    sectors = b"".join(
        struct.pack("<hh", _clamp16(s.floor_height), _clamp16(s.ceil_height))
        + _name8(s.floor_flat) + _name8(s.ceil_flat)
        + struct.pack("<hhh", _clamp16(s.light_level), s.special, s.tag)
        for s in doc.sectors
    )

    return {
        "THINGS": things,
        "LINEDEFS": linedefs,
        "SIDEDEFS": sidedefs,
        "VERTEXES": vertexes,
        "SEGS": b"",
        "SSECTORS": b"",
        "NODES": b"",
        "SECTORS": sectors,
        "REJECT": b"",
        "BLOCKMAP": b"",
    }

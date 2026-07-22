"""
Low-level struct unpacking for each Doom map lump type.

All functions return lists of plain dicts with integer/string fields
exactly as they appear in the binary.  The richer typed objects are
assembled in map_extractor.py.

Byte-order is little-endian throughout (Doom WAD format).

Lump sizes:
  THINGS   10 bytes each:  x y angle type flags
  LINEDEFS 14 bytes each:  v1 v2 flags special tag right_sidedef left_sidedef
  SIDEDEFS 30 bytes each:  x_off y_off upper(8) lower(8) mid(8) sector
  VERTEXES  4 bytes each:  x y
  SECTORS  26 bytes each:  floor_h ceil_h floor_flat(8) ceil_flat(8) light special tag
"""
from __future__ import annotations

import struct
from typing import TypedDict


# ─── Record types (for IDE / type-checker clarity) ────────────────────────────

class ThingRecord(TypedDict):
    x: int; y: int; angle: int; type: int; flags: int


class LinedefRecord(TypedDict):
    v1: int; v2: int; flags: int; special: int; tag: int
    right_sidedef: int; left_sidedef: int


class SidedefRecord(TypedDict):
    x_offset: int; y_offset: int
    upper_tex: str; lower_tex: str; mid_tex: str
    sector: int


class VertexRecord(TypedDict):
    x: int; y: int


class SectorRecord(TypedDict):
    floor_height: int; ceil_height: int
    floor_flat: str; ceil_flat: str
    light_level: int; special: int; tag: int


# ─── Struct formats ──────────────────────────────────────────────────────────

_THING_FMT   = struct.Struct("<hhHHH")    # 10 bytes
_LINEDEF_FMT = struct.Struct("<HHHHHHH")  # 14 bytes
_SIDEDEF_FMT = struct.Struct("<hh8s8s8sH")# 30 bytes
_VERTEX_FMT  = struct.Struct("<hh")       # 4 bytes
_SECTOR_FMT  = struct.Struct("<hh8s8sHHH")# 26 bytes

_NO_SIDEDEF = 0xFFFF  # unsigned sentinel for "no sidedef"


def _decode_name(raw: bytes) -> str:
    """Decode a null-padded 8-byte texture/flat name to an uppercase string."""
    return raw.rstrip(b"\x00").decode("ascii", errors="replace").upper()


# ─── Public parsers ──────────────────────────────────────────────────────────

def parse_things(data: bytes) -> list[ThingRecord]:
    size = _THING_FMT.size
    if len(data) % size:
        raise ValueError(f"THINGS lump size {len(data)} is not a multiple of {size}")
    records: list[ThingRecord] = []
    for off in range(0, len(data), size):
        x, y, angle, ttype, flags = _THING_FMT.unpack_from(data, off)
        records.append(ThingRecord(x=x, y=y, angle=angle, type=ttype, flags=flags))
    return records


def parse_linedefs(data: bytes) -> list[LinedefRecord]:
    size = _LINEDEF_FMT.size
    if len(data) % size:
        raise ValueError(f"LINEDEFS lump size {len(data)} is not a multiple of {size}")
    records: list[LinedefRecord] = []
    for off in range(0, len(data), size):
        v1, v2, flags, special, tag, right, left = _LINEDEF_FMT.unpack_from(data, off)
        records.append(LinedefRecord(
            v1=v1, v2=v2, flags=flags, special=special, tag=tag,
            right_sidedef=right,
            left_sidedef=-1 if left == _NO_SIDEDEF else left,
        ))
    return records


def parse_sidedefs(data: bytes) -> list[SidedefRecord]:
    size = _SIDEDEF_FMT.size
    if len(data) % size:
        raise ValueError(f"SIDEDEFS lump size {len(data)} is not a multiple of {size}")
    records: list[SidedefRecord] = []
    for off in range(0, len(data), size):
        xoff, yoff, upper, lower, mid, sector = _SIDEDEF_FMT.unpack_from(data, off)
        records.append(SidedefRecord(
            x_offset=xoff,
            y_offset=yoff,
            upper_tex=_decode_name(upper),
            lower_tex=_decode_name(lower),
            mid_tex=_decode_name(mid),
            sector=sector,
        ))
    return records


def parse_vertexes(data: bytes) -> list[VertexRecord]:
    size = _VERTEX_FMT.size
    if len(data) % size:
        raise ValueError(f"VERTEXES lump size {len(data)} is not a multiple of {size}")
    records: list[VertexRecord] = []
    for off in range(0, len(data), size):
        x, y = _VERTEX_FMT.unpack_from(data, off)
        records.append(VertexRecord(x=x, y=y))
    return records


def parse_sectors(data: bytes) -> list[SectorRecord]:
    size = _SECTOR_FMT.size
    if len(data) % size:
        raise ValueError(f"SECTORS lump size {len(data)} is not a multiple of {size}")
    records: list[SectorRecord] = []
    for off in range(0, len(data), size):
        floor_h, ceil_h, floor_flat, ceil_flat, light, special, tag = _SECTOR_FMT.unpack_from(data, off)
        records.append(SectorRecord(
            floor_height=floor_h,
            ceil_height=ceil_h,
            floor_flat=_decode_name(floor_flat),
            ceil_flat=_decode_name(ceil_flat),
            light_level=light,
            special=special,
            tag=tag,
        ))
    return records

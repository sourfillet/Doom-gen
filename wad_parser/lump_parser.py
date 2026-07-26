"""Struct-unpack parsers for binary-format (Doom/Hexen) map lumps."""
from __future__ import annotations

import struct

from wad_parser.types import (
    LinedefFlags, Sector, Sidedef, Thing, ThingFlags, Vertex,
    resolve_doomed, resolve_linedef_special, resolve_hexen_special,
)


def _decode_name(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("ascii", errors="replace").upper() or "-"


def parse_vertexes(data: bytes) -> list[Vertex]:
    count = len(data) // 4
    out = []
    for i in range(count):
        x, y = struct.unpack_from("<hh", data, i * 4)
        out.append(Vertex(id=i, x=x, y=y))
    return out


def parse_sectors(data: bytes) -> list[Sector]:
    count = len(data) // 26
    out = []
    for i in range(count):
        off = i * 26
        floor_h, ceil_h = struct.unpack_from("<hh", data, off)
        floor_flat = _decode_name(data[off + 4: off + 12])
        ceil_flat = _decode_name(data[off + 12: off + 20])
        light, special, tag = struct.unpack_from("<hhh", data, off + 20)
        out.append(Sector(
            id=i, floor_height=floor_h, ceil_height=ceil_h,
            floor_flat=floor_flat, ceil_flat=ceil_flat,
            light_level=light, special=special, tag=tag,
        ))
    return out


def parse_sidedefs(data: bytes) -> list[Sidedef]:
    count = len(data) // 30
    out = []
    for i in range(count):
        off = i * 30
        x_off, y_off = struct.unpack_from("<hh", data, off)
        upper = _decode_name(data[off + 4: off + 12])
        lower = _decode_name(data[off + 12: off + 20])
        mid = _decode_name(data[off + 20: off + 28])
        (sector,) = struct.unpack_from("<H", data, off + 28)
        out.append(Sidedef(
            id=i, x_offset=x_off, y_offset=y_off,
            upper_tex=upper, lower_tex=lower, mid_tex=mid, sector=sector,
        ))
    return out


# Bit -> LinedefFlags attribute, for the vanilla+Boom flag word.
_LINEDEF_FLAG_BITS: list[tuple[int, str]] = [
    (0x0001, "impassable"),
    (0x0002, "block_monsters"),
    (0x0004, "two_sided"),
    (0x0008, "upper_unpegged"),
    (0x0010, "lower_unpegged"),
    (0x0020, "secret"),
    (0x0040, "blocks_sound"),
    (0x0080, "not_on_map"),
    (0x0100, "already_on_map"),
    (0x0200, "pass_through"),        # Boom (ML_PASSUSE)
    (0x1000, "block_land_monsters"),  # MBF21
    (0x2000, "block_players"),        # MBF21
    # Bits 0x0400/0x0800 exist in some ports' extended flag words but don't
    # have a single settled cross-port meaning worth asserting here; they
    # fall through to `unrecognized_flag_bits` below rather than being
    # mislabeled.
]
_KNOWN_FLAG_MASK = 0
for _bit, _ in _LINEDEF_FLAG_BITS:
    _KNOWN_FLAG_MASK |= _bit


def _decode_linedef_flags(raw: int) -> tuple[LinedefFlags, dict]:
    flags = LinedefFlags()
    for bit, attr in _LINEDEF_FLAG_BITS:
        if raw & bit:
            setattr(flags, attr, True)
    leftover = raw & ~_KNOWN_FLAG_MASK
    extra = {"unrecognized_flag_bits": leftover} if leftover else {}
    return flags, extra


def parse_linedefs_doom(data: bytes) -> list:
    from wad_parser.types import Linedef
    count = len(data) // 14
    out = []
    for i in range(count):
        off = i * 14
        v1, v2, raw_flags, special, tag, right, left = struct.unpack_from("<HHHHHHH", data, off)
        flags, extra = _decode_linedef_flags(raw_flags)
        special_name, trigger_type = resolve_linedef_special(special)
        out.append(Linedef(
            id=i, v1=v1, v2=v2, flags=flags, special=special, tag=tag,
            right_sidedef=right if right != 0xFFFF else -1,
            left_sidedef=left if left != 0xFFFF else -1,
            special_name=special_name, trigger_type=trigger_type,
            extra=extra,
        ))
    return out


def parse_linedefs_hexen(data: bytes) -> list:
    from wad_parser.types import Linedef
    count = len(data) // 16
    out = []
    for i in range(count):
        off = i * 16
        v1, v2, raw_flags = struct.unpack_from("<HHH", data, off)
        special = data[off + 6]
        args = list(data[off + 7: off + 12])
        right, left = struct.unpack_from("<HH", data, off + 12)
        flags, extra = _decode_linedef_flags(raw_flags)
        special_name, trigger_type = resolve_hexen_special(special)
        out.append(Linedef(
            id=i, v1=v1, v2=v2, flags=flags, special=special, tag=0,
            right_sidedef=right if right != 0xFFFF else -1,
            left_sidedef=left if left != 0xFFFF else -1,
            args=args, special_name=special_name, trigger_type=trigger_type,
            extra=extra,
        ))
    return out


# Bit -> ThingFlags attribute (vanilla + Boom + MBF).
_THING_FLAG_BITS: list[tuple[int, str]] = [
    (0x0001, "skill_1_2"),
    (0x0002, "skill_3"),
    (0x0004, "skill_4_5"),
    (0x0008, "deaf"),
    (0x0010, "not_single_player"),
    (0x0020, "not_deathmatch"),   # Boom
    (0x0040, "not_coop"),         # Boom
    (0x0080, "friendly"),         # MBF
]
_KNOWN_THING_FLAG_MASK = 0
for _bit, _ in _THING_FLAG_BITS:
    _KNOWN_THING_FLAG_MASK |= _bit


def _decode_thing_flags(raw: int) -> tuple[ThingFlags, dict]:
    flags = ThingFlags()
    for bit, attr in _THING_FLAG_BITS:
        if raw & bit:
            setattr(flags, attr, True)
    leftover = raw & ~_KNOWN_THING_FLAG_MASK
    extra = {"unrecognized_flag_bits": leftover} if leftover else {}
    return flags, extra


def parse_things_doom(data: bytes) -> list[Thing]:
    count = len(data) // 10
    out = []
    for i in range(count):
        off = i * 10
        x, y, angle, type_num, raw_flags = struct.unpack_from("<hhhhH", data, off)
        flags, extra = _decode_thing_flags(raw_flags)
        name, category, unresolved = resolve_doomed(type_num)
        out.append(Thing(
            id=i, x=x, y=y, angle=angle, type=type_num,
            type_name=name, category=category, flags=flags,
            unresolved=unresolved, extra=extra,
        ))
    return out


def parse_things_hexen(data: bytes) -> list[Thing]:
    count = len(data) // 20
    out = []
    for i in range(count):
        off = i * 20
        tid, x, y, z, angle, type_num, raw_flags = struct.unpack_from("<hhhhhhH", data, off)
        special = data[off + 14]
        args = list(data[off + 15: off + 20])
        flags, extra = _decode_thing_flags(raw_flags)
        name, category, unresolved = resolve_doomed(type_num)
        out.append(Thing(
            id=i, x=x, y=y, z=z, angle=angle, type=type_num,
            type_name=name, category=category, flags=flags,
            tid=tid, special=special, args=args,
            unresolved=unresolved, extra=extra,
        ))
    return out

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


# Hexen redefines the linedef flag word above bit 8 (doomwiki.org "Linedef",
# Hexen level format): 0x0200 is ML_REPEAT_SPECIAL (not Boom's pass-through),
# bits 10-12 (0x1C00) are the SPAC activation field, and 0x2000/0x8000 are
# ZDoom-in-Hexen extensions. Bits 0-8 match Doom.
_HEXEN_EXTRA_FLAG_BITS: list[tuple[int, str]] = [
    (0x0200, "repeat_special"),
    (0x2000, "monsters_can_activate"),  # ZDoom
    (0x8000, "block_everything"),        # ZDoom
]
_HEXEN_SPAC_MASK = 0x1C00
_HEXEN_SPAC_SHIFT = 10
# SPAC value -> trigger_type. 6 (use w/ pass-through) is ZDoom-only.
_SPAC_TRIGGER = {
    0: "walk",             # crossed by player
    1: "switch",           # used by player
    2: "monster-walk",     # crossed by monster
    3: "gun",              # projectile impact
    4: "push",             # player bumps
    5: "projectile-walk",  # crossed by projectile
    6: "switch",           # used by player, pass-through (ZDoom)
}
_KNOWN_HEXEN_LINEDEF_MASK = 0x01FF | _HEXEN_SPAC_MASK
for _bit, _ in _HEXEN_EXTRA_FLAG_BITS:
    _KNOWN_HEXEN_LINEDEF_MASK |= _bit


def _decode_linedef_flags_hexen(raw: int) -> tuple[LinedefFlags, dict, str]:
    flags = LinedefFlags()
    for bit, attr in _LINEDEF_FLAG_BITS:
        if bit <= 0x0100 and raw & bit:  # Doom-shared low bits only
            setattr(flags, attr, True)
    for bit, attr in _HEXEN_EXTRA_FLAG_BITS:
        if raw & bit:
            setattr(flags, attr, True)
    spac = (raw & _HEXEN_SPAC_MASK) >> _HEXEN_SPAC_SHIFT
    trigger_type = _SPAC_TRIGGER.get(spac, "unknown")
    if spac == 6:
        flags.pass_through = True
    leftover = raw & ~_KNOWN_HEXEN_LINEDEF_MASK
    extra = {"unrecognized_flag_bits": leftover} if leftover else {}
    return flags, extra, trigger_type


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
        flags, extra, trigger_type = _decode_linedef_flags_hexen(raw_flags)
        special_name, _ = resolve_hexen_special(special)
        if special == 0:
            trigger_type = "none"
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


# Hexen thing flags (doomwiki.org "Thing", Hexen format): skills and ambush
# match Doom, but 0x0010 is dormant (not "not in single player"), 0x0020-
# 0x0080 are class-presence bits, and 0x0100-0x0400 are POSITIVE mode-
# presence bits (set = appears in that mode) - the reverse sense of Doom's
# not_* flags, so they are inverted here to keep ThingFlags semantics
# uniform across formats.
_HEXEN_THING_FLAG_BITS: list[tuple[int, str]] = [
    (0x0001, "skill_1_2"),
    (0x0002, "skill_3"),
    (0x0004, "skill_4_5"),
    (0x0008, "deaf"),
    (0x0010, "dormant"),
    (0x0020, "class_fighter"),
    (0x0040, "class_cleric"),
    (0x0080, "class_mage"),
]
_HEXEN_MODE_BITS = [(0x0100, "not_single_player"), (0x0200, "not_coop"), (0x0400, "not_deathmatch")]
_KNOWN_HEXEN_THING_MASK = 0x07FF


def _decode_thing_flags_hexen(raw: int) -> tuple[ThingFlags, dict]:
    flags = ThingFlags()
    for bit, attr in _HEXEN_THING_FLAG_BITS:
        if raw & bit:
            setattr(flags, attr, True)
    for bit, attr in _HEXEN_MODE_BITS:
        if not raw & bit:
            setattr(flags, attr, True)
    leftover = raw & ~_KNOWN_HEXEN_THING_MASK
    extra = {"unrecognized_flag_bits": leftover} if leftover else {}
    return flags, extra


def parse_things_hexen(data: bytes) -> list[Thing]:
    count = len(data) // 20
    out = []
    for i in range(count):
        off = i * 20
        tid, x, y, z, angle, type_num, raw_flags = struct.unpack_from("<hhhhhhH", data, off)
        special = data[off + 14]
        args = list(data[off + 15: off + 20])
        flags, extra = _decode_thing_flags_hexen(raw_flags)
        name, category, unresolved = resolve_doomed(type_num)
        out.append(Thing(
            id=i, x=x, y=y, z=z, angle=angle, type=type_num,
            type_name=name, category=category, flags=flags,
            tid=tid, special=special, args=args,
            unresolved=unresolved, extra=extra,
        ))
    return out

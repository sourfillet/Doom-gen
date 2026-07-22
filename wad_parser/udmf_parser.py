"""Parser for the UDMF text map format (TEXTMAP lump), used by Boom-derived
ports (Eternity) and ZDoom/GZDoom instead of the binary Doom/Hexen lumps.

UDMF's defining feature is that any block can carry arbitrary named fields —
that's exactly the kind of richer data (vertex-based sector slopes, 3D floors
via control-sector specials, port-specific extensions) plan/data-model.md
calls out as needing to be represented rather than dropped. Every field is
parsed into a raw dict first; a fixed set of well-known field names then maps
onto the same typed schema the binary parser produces, and anything left over
goes into each object's `extra` dict.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from wad_parser.types import (
    Linedef, LinedefFlags, Sector, SectorSlope, Sidedef, Thing, ThingFlags,
    Vertex, resolve_doomed,
)

_TOKEN_RE = re.compile(r"""
    (?P<ws>\s+)
  | (?P<line_comment>//[^\n]*)
  | (?P<block_comment>/\*.*?\*/)
  | (?P<string>"(?:[^"\\]|\\.)*")
  | (?P<number>-?\d+\.\d+(?:[eE][-+]?\d+)?|-?0x[0-9a-fA-F]+|-?\d+)
  | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<lbrace>\{)
  | (?P<rbrace>\})
  | (?P<eq>=)
  | (?P<semi>;)
""", re.VERBOSE | re.DOTALL)


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens = []
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m:
            pos += 1
            continue
        kind = m.lastgroup
        if kind not in ("ws", "line_comment", "block_comment"):
            tokens.append((kind, m.group()))
        pos = m.end()
    return tokens


def _parse_value(raw: str):
    if raw == "true":
        return True
    if raw == "false":
        return False
    if raw.startswith('"'):
        return raw[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    if raw.startswith("0x") or raw.startswith("-0x"):
        return int(raw, 16)
    if "." in raw or "e" in raw.lower():
        return float(raw)
    return int(raw)


@dataclass
class UdmfDocument:
    namespace: str | None
    globals: dict
    blocks: dict[str, list[dict]]  # block type (lowercase) -> list of field dicts


def parse_textmap(data: bytes) -> UdmfDocument:
    text = data.decode("utf-8", errors="replace")
    tokens = _tokenize(text)
    i = 0
    n = len(tokens)
    globals_: dict = {}
    blocks: dict[str, list[dict]] = {}

    while i < n:
        kind, val = tokens[i]
        if kind != "ident":
            i += 1
            continue
        name = val.lower()
        # lookahead for '=' (assignment) or '{' (block)
        if i + 1 < n and tokens[i + 1][0] == "eq":
            j = i + 2
            value_tokens = []
            while j < n and tokens[j][0] != "semi":
                value_tokens.append(tokens[j])
                j += 1
            if value_tokens:
                globals_[name] = _parse_value(value_tokens[0][1])
            i = j + 1
        elif i + 1 < n and tokens[i + 1][0] == "lbrace":
            j = i + 2
            fields: dict = {}
            while j < n and tokens[j][0] != "rbrace":
                if tokens[j][0] == "ident" and j + 1 < n and tokens[j + 1][0] == "eq":
                    field_name = tokens[j][1].lower()
                    k = j + 2
                    value_tokens = []
                    while k < n and tokens[k][0] != "semi":
                        value_tokens.append(tokens[k])
                        k += 1
                    if value_tokens:
                        fields[field_name] = _parse_value(value_tokens[0][1])
                    j = k + 1
                else:
                    j += 1
            blocks.setdefault(name, []).append(fields)
            i = j + 1
        else:
            i += 1

    return UdmfDocument(namespace=globals_.get("namespace"), globals=globals_, blocks=blocks)


# ---------------------------------------------------------------------------
# UDMF field name -> typed schema translation
# ---------------------------------------------------------------------------

_LINEDEF_FLAG_FIELDS = {
    "blocking": "impassable",
    "blockmonsters": "block_monsters",
    "twosided": "two_sided",
    "dontpegtop": "upper_unpegged",
    "dontpegbottom": "lower_unpegged",
    "secret": "secret",
    "blocksound": "blocks_sound",
    "dontdraw": "not_on_map",
    "mapped": "already_on_map",
    "passuse": "pass_through",
    "blockplayers": "block_players",
}

_KNOWN_VERTEX_FIELDS = {"x", "y", "zfloor", "zceiling"}
_KNOWN_LINEDEF_FIELDS = {
    "v1", "v2", "sidefront", "sideback", "special", "id",
    *_LINEDEF_FLAG_FIELDS.keys(),
    *(f"arg{i}" for i in range(5)),
}
_KNOWN_SIDEDEF_FIELDS = {"offsetx", "offsety", "texturetop", "texturebottom", "texturemiddle", "sector"}
_KNOWN_SECTOR_FIELDS = {
    "heightfloor", "heightceiling", "texturefloor", "textureceiling",
    "lightlevel", "special", "id",
    "floorplane_a", "floorplane_b", "floorplane_c", "floorplane_d",
    "ceilingplane_a", "ceilingplane_b", "ceilingplane_c", "ceilingplane_d",
}
_KNOWN_THING_FIELDS = {
    "x", "y", "height", "angle", "type", "id", "special",
    "skill1", "skill2", "skill3", "skill4", "skill5",
    "ambush", "single", "dm", "coop", "friend",
    *(f"arg{i}" for i in range(5)),
}


def _split_extra(fields: dict, known: set[str]) -> dict:
    return {k: v for k, v in fields.items() if k not in known}


def build_vertices(doc: UdmfDocument) -> list[Vertex]:
    out = []
    for i, f in enumerate(doc.blocks.get("vertex", [])):
        zfloor = f.get("zfloor")
        zceiling = f.get("zceiling")
        z = zfloor if zfloor is not None else zceiling
        extra = _split_extra(f, _KNOWN_VERTEX_FIELDS)
        if zfloor is not None and zceiling is not None:
            extra["zceiling"] = zceiling
        out.append(Vertex(id=i, x=f.get("x", 0.0), y=f.get("y", 0.0), z=z, extra=extra))
    return out


def _slope_from(fields: dict, prefix: str) -> SectorSlope | None:
    keys = [f"{prefix}_a", f"{prefix}_b", f"{prefix}_c", f"{prefix}_d"]
    if all(k in fields for k in keys):
        return SectorSlope(*(fields[k] for k in keys))
    return None


def build_sectors(doc: UdmfDocument) -> list[Sector]:
    out = []
    for i, f in enumerate(doc.blocks.get("sector", [])):
        out.append(Sector(
            id=i,
            floor_height=f.get("heightfloor", 0),
            ceil_height=f.get("heightceiling", 0),
            floor_flat=f.get("texturefloor", "-"),
            ceil_flat=f.get("textureceiling", "-"),
            light_level=f.get("lightlevel", 160),
            special=f.get("special", 0),
            tag=f.get("id", 0),
            floor_slope=_slope_from(f, "floorplane"),
            ceil_slope=_slope_from(f, "ceilingplane"),
            extra=_split_extra(f, _KNOWN_SECTOR_FIELDS),
        ))
    return out


def build_sidedefs(doc: UdmfDocument) -> list[Sidedef]:
    out = []
    for i, f in enumerate(doc.blocks.get("sidedef", [])):
        out.append(Sidedef(
            id=i,
            x_offset=f.get("offsetx", 0),
            y_offset=f.get("offsety", 0),
            upper_tex=f.get("texturetop", "-"),
            lower_tex=f.get("texturebottom", "-"),
            mid_tex=f.get("texturemiddle", "-"),
            sector=f.get("sector", 0),
        ))
    return out


def build_linedefs(doc: UdmfDocument) -> list[Linedef]:
    out = []
    for i, f in enumerate(doc.blocks.get("linedef", [])):
        flags = LinedefFlags()
        for udmf_key, attr in _LINEDEF_FLAG_FIELDS.items():
            if f.get(udmf_key):
                setattr(flags, attr, True)
        args = [f.get(f"arg{n}", 0) for n in range(5)] if any(f"arg{n}" in f for n in range(5)) else None
        out.append(Linedef(
            id=i,
            v1=f.get("v1", 0), v2=f.get("v2", 0),
            flags=flags,
            special=f.get("special", 0),
            tag=f.get("id", 0),
            right_sidedef=f.get("sidefront", -1) if f.get("sidefront") is not None else -1,
            left_sidedef=f.get("sideback", -1) if f.get("sideback") is not None else -1,
            args=args,
            extra=_split_extra(f, _KNOWN_LINEDEF_FIELDS),
        ))
    return out


def build_things(doc: UdmfDocument) -> list[Thing]:
    out = []
    for i, f in enumerate(doc.blocks.get("thing", [])):
        flags = ThingFlags(
            skill_1_2=bool(f.get("skill1") or f.get("skill2")),
            skill_3=bool(f.get("skill3")),
            skill_4_5=bool(f.get("skill4") or f.get("skill5")),
            deaf=bool(f.get("ambush")),
            not_single_player=("single" in f and not f["single"]),
            not_deathmatch=("dm" in f and not f["dm"]),
            not_coop=("coop" in f and not f["coop"]),
            friendly=bool(f.get("friend")),
        )
        type_num = f.get("type", 0)
        name, category, unresolved = resolve_doomed(type_num)
        args = [f.get(f"arg{n}", 0) for n in range(5)] if any(f"arg{n}" in f for n in range(5)) else None
        out.append(Thing(
            id=i,
            x=f.get("x", 0.0), y=f.get("y", 0.0), z=f.get("height"),
            angle=f.get("angle", 0),
            type=type_num, type_name=name, category=category,
            flags=flags,
            tid=f.get("id"), special=f.get("special"), args=args,
            unresolved=unresolved,
            extra=_split_extra(f, _KNOWN_THING_FIELDS),
        ))
    return out

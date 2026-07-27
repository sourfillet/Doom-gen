"""Dataclasses for the extracted-map schema, plus lookup tables for DoomEd
thing numbers and linedef specials.

See plan/data-model.md for the design rationale: raw fields are never
dropped, unrecognized data goes into an `extra` dict rather than being
discarded, and `schema_version` lets the format grow without invalidating
already-parsed documents.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
from typing import Any, Optional

SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass
class Vertex:
    id: int
    x: float
    y: float
    # UDMF sector-slope support and Hexen-format vertex height hints land here
    z: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SectorSlope:
    # UDMF plane-equation slope: z = -(a*x + b*y + d) / c, or omitted (flat)
    a: float
    b: float
    c: float
    d: float


@dataclass
class Sector:
    id: int
    floor_height: int
    ceil_height: int
    floor_flat: str
    ceil_flat: str
    light_level: int
    special: int
    tag: int
    floor_slope: Optional[SectorSlope] = None
    ceil_slope: Optional[SectorSlope] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LinedefFlags:
    impassable: bool = False
    block_monsters: bool = False
    two_sided: bool = False
    upper_unpegged: bool = False
    lower_unpegged: bool = False
    secret: bool = False
    blocks_sound: bool = False
    not_on_map: bool = False
    already_on_map: bool = False
    # Boom/MBF21 extensions
    pass_through: bool = False
    block_land_monsters: bool = False
    block_players: bool = False
    # Hexen-format / UDMF extensions
    repeat_special: bool = False        # Hexen 0x0200 / UDMF repeatspecial
    monsters_can_activate: bool = False  # ZDoom-in-Hexen 0x2000 / UDMF monsteractivate
    block_everything: bool = False       # ZDoom-in-Hexen 0x8000 / UDMF blockeverything


@dataclass
class Linedef:
    id: int
    v1: int
    v2: int
    flags: LinedefFlags
    special: int
    tag: int
    right_sidedef: int
    left_sidedef: int  # -1 = one-sided
    args: Optional[list[int]] = None  # Hexen/UDMF action args
    special_name: Optional[str] = None   # from LINEDEF_SPECIAL_TABLE, best-effort
    trigger_type: Optional[str] = None   # walk/switch/push/gun/unknown/none
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Sidedef:
    id: int
    x_offset: int
    y_offset: int
    upper_tex: str
    lower_tex: str
    mid_tex: str
    sector: int
    # UDMF sidedefs can carry per-texture offsets/scales, light overrides,
    # etc. - preserved here per the capture-completeness principle. Always
    # empty for binary-format sidedefs (they have no other fields).
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ThingFlags:
    skill_1_2: bool = False
    skill_3: bool = False
    skill_4_5: bool = False
    deaf: bool = False
    not_single_player: bool = False
    not_deathmatch: bool = False
    not_coop: bool = False
    friendly: bool = False  # MBF
    # Hexen-format extensions (mode bits are stored positive in the WAD and
    # inverted into the not_* fields above so semantics match across formats)
    dormant: bool = False
    class_fighter: bool = False
    class_cleric: bool = False
    class_mage: bool = False


@dataclass
class Thing:
    id: int
    x: float
    y: float
    angle: int
    type: int
    type_name: str
    category: str
    flags: ThingFlags
    z: Optional[int] = None  # Hexen/UDMF height offset above floor
    tid: Optional[int] = None
    special: Optional[int] = None
    args: Optional[list[int]] = None
    unresolved: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Texture system (WAD-wide)
# ---------------------------------------------------------------------------

@dataclass
class Patch:
    name: str
    x_origin: int
    y_origin: int


@dataclass
class Texture:
    name: str
    width: int
    height: int
    patches: list[Patch]


@dataclass
class FlatAsset:
    name: str
    width: Optional[int]   # None if not inferable (non-square, non-4096-byte)
    height: Optional[int]
    byte_length: int
    sha256: str  # raw indexed-pixel bytes are written to a sidecar file named by this hash


@dataclass
class AnimatedEntry:
    kind: str  # "flat" or "texture"
    first: str
    last: str
    speed: int  # tics per frame


@dataclass
class SwitchEntry:
    off_texture: str
    on_texture: str
    game: int  # bitmask: which games this switch pair is valid for


@dataclass
class TextureSet:
    textures: dict[str, Texture] = field(default_factory=dict)
    flats: list[FlatAsset] = field(default_factory=list)
    palette_count: int = 0
    colormap_count: int = 0
    playpal_sha256: Optional[str] = None      # raw PLAYPAL bytes written to a sidecar file
    colormap_sha256: Optional[str] = None     # raw COLORMAP bytes written to a sidecar file
    animated: list[AnimatedEntry] = field(default_factory=list)
    switches: list[SwitchEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-map metadata
# ---------------------------------------------------------------------------

@dataclass
class MapInfoEntry:
    level_name: Optional[str] = None
    music: Optional[str] = None
    sky: Optional[str] = None
    cluster: Optional[str] = None
    next_map: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class MapMeta:
    map_name: str
    source_wad: str
    wad_hash: str
    format: str  # "doom" | "hexen" | "udmf"
    is_iwad: bool = False
    mapinfo: Optional[MapInfoEntry] = None


@dataclass
class BoundingBox:
    min_x: float
    max_x: float
    min_y: float
    max_y: float


@dataclass
class DerivedData:
    sector_graph: dict[str, Any] = field(default_factory=dict)
    sector_polygons: list[list[int]] = field(default_factory=list)
    trigger_map: dict[str, list[int]] = field(default_factory=dict)
    reachable_from_start: list[int] = field(default_factory=list)
    texture_alignment_score: Optional[float] = None
    symmetry: Optional[dict[str, Any]] = None
    # Re-centered/unit-scaled vertex coordinates, kept ALONGSIDE raw vertices
    # (never instead of - absolute scale still matters, e.g. an 8/16/24 unit
    # step is a stair, a 64-unit step is not climbable).
    normalized_space: Optional[dict[str, Any]] = None
    # Opportunistic cross-check of sector-polygon reconstruction against the
    # map's own compiled BSP (SEGS/SSECTORS/REJECT), when present. Diagnostic
    # only - never authoritative, and absent entirely for UDMF maps that ship
    # ZNODES instead of classic nodes.
    bsp_sanity_check: Optional[dict[str, Any]] = None


@dataclass
class MapDocument:
    schema_version: str
    meta: MapMeta
    bounds: BoundingBox
    vertices: list[Vertex]
    sectors: list[Sector]
    linedefs: list[Linedef]
    sidedefs: list[Sidedef]
    things: list[Thing]
    derived: DerivedData
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MapDocument":
        """Rebuilds a MapDocument from its to_dict()/JSON form. Unknown keys
        are dropped rather than erroring, so documents written by a NEWER
        schema still load (the reverse of the extra-bag guarantee)."""
        meta_d = dict(data["meta"])
        mapinfo = meta_d.get("mapinfo")
        meta_d["mapinfo"] = _build(MapInfoEntry, mapinfo) if mapinfo else None
        return cls(
            schema_version=data["schema_version"],
            meta=_build(MapMeta, meta_d),
            bounds=_build(BoundingBox, data["bounds"]),
            vertices=[_build(Vertex, v) for v in data["vertices"]],
            sectors=[_build_sector(s) for s in data["sectors"]],
            linedefs=[_build_linedef(l) for l in data["linedefs"]],
            sidedefs=[_build(Sidedef, s) for s in data["sidedefs"]],
            things=[_build_thing(t) for t in data["things"]],
            derived=_build(DerivedData, data["derived"]),
            extra=data.get("extra", {}),
        )


def _build(cls, d: dict):
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in known})


def _build_sector(d: dict) -> Sector:
    d = dict(d)
    for key in ("floor_slope", "ceil_slope"):
        if d.get(key):
            d[key] = _build(SectorSlope, d[key])
    return _build(Sector, d)


def _build_linedef(d: dict) -> Linedef:
    d = dict(d)
    d["flags"] = _build(LinedefFlags, d["flags"])
    return _build(Linedef, d)


def _build_thing(d: dict) -> Thing:
    d = dict(d)
    d["flags"] = _build(ThingFlags, d["flags"])
    return _build(Thing, d)


# ---------------------------------------------------------------------------
# Lookup tables + resolvers
# ---------------------------------------------------------------------------
# The tables live in wad_parser/lookup_tables.py, GENERATED from doomwiki.org
# (thing types, linedef types) and the ZDoom acc compiler's zspecial.acs
# (Hexen/ZDoom action-special numbering) by tools/gen_lookup_tables.py.
# Re-exported here so existing imports keep working.

from wad_parser.lookup_tables import (  # noqa: E402
    DOOMED_TABLE, LINEDEF_SPECIAL_TABLE, HEXEN_SPECIAL_TABLE,
)

# Thing categories: player-start, deathmatch-start, teleport-dest, monster,
# weapon, ammo, item, key, obstacle, decoration, special, unknown.
#
# Linedef trigger types: walk, switch, gun, push, passive (permanent line
# effect with no activation, e.g. scrollers/transfers), none, unknown, plus
# the Hexen SPAC-derived monster-walk and projectile ones produced by
# lump_parser/udmf_parser.


def resolve_doomed(type_num: int) -> tuple[str, str, bool]:
    """Returns (name, category, unresolved)."""
    entry = DOOMED_TABLE.get(type_num)
    if entry is None:
        return (f"Unknown_{type_num}", "unknown", True)
    return (entry[0], entry[1], False)


# Boom generalized linedef specials (0x2F80-0x7FFF): the special is a packed
# bitfield, not a table index. Ranges and the 3-bit trigger encoding are from
# doomwiki.org "Linedef type", "Detailed generalized linedef specification".
_GENERALIZED_RANGES = [
    (0x2F80, 0x2FFF, "GenCrusher"),
    (0x3000, 0x33FF, "GenStairs"),
    (0x3400, 0x37FF, "GenLift"),
    (0x3800, 0x3BFF, "GenLockedDoor"),
    (0x3C00, 0x3FFF, "GenDoor"),
    (0x4000, 0x5FFF, "GenCeiling"),
    (0x6000, 0x7FFF, "GenFloor"),
]
_GENERALIZED_TRIGGERS = ["W1", "WR", "S1", "SR", "G1", "GR", "P1", "PR"]
_GEN_TRIG_TO_TYPE = {"W": "walk", "S": "switch", "G": "gun", "P": "push"}


def resolve_linedef_special(special_num: int) -> tuple[str, str]:
    for lo, hi, family in _GENERALIZED_RANGES:
        if lo <= special_num <= hi:
            trig = _GENERALIZED_TRIGGERS[special_num & 0x0007]
            return (f"{family} ({trig})", _GEN_TRIG_TO_TYPE[trig[0]])
    entry = LINEDEF_SPECIAL_TABLE.get(special_num)
    if entry is None:
        return (f"Special_{special_num}", "unknown")
    return entry


HEXEN_UDMF_NAMESPACES = {"hexen", "zdoom", "zdoomtranslated", "vavoom"}


def resolve_hexen_special(special_num: int) -> tuple[str, str]:
    """Hexen/ZDoom action-special numbering is unrelated to the Doom scheme;
    see HEXEN_SPECIAL_TABLE. The returned trigger type is always "unknown"
    (or "none") because in Hexen-format maps activation lives in the linedef
    SPAC flag bits, and in UDMF in per-line activation fields - callers
    derive trigger_type from those instead.
    """
    entry = HEXEN_SPECIAL_TABLE.get(special_num)
    if entry is None:
        return (f"HexenAction_{special_num}", "unknown")
    return entry

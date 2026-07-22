"""Dataclasses for the extracted-map schema, plus lookup tables for DoomEd
thing numbers and linedef specials.

See plan/data-model.md for the design rationale: raw fields are never
dropped, unrecognized data goes into an `extra` dict rather than being
discarded, and `schema_version` lets the format grow without invalidating
already-parsed documents.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
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
    translucent_25: bool = False
    translucent_75: bool = False
    block_land_monsters: bool = False
    block_players: bool = False


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
    flats: list[str] = field(default_factory=list)  # names only; raw pixels kept out of JSON
    palette_count: int = 0
    colormap_count: int = 0
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


# ---------------------------------------------------------------------------
# DoomEd thing-number lookup
# ---------------------------------------------------------------------------
# (name, category) -- category is one of: player-start, deathmatch-start,
# teleport-dest, monster, weapon, ammo, item, key, decoration, obstacle,
# light, sound, special

DOOMED_TABLE: dict[int, tuple[str, str]] = {
    1: ("Player1Start", "player-start"),
    2: ("Player2Start", "player-start"),
    3: ("Player3Start", "player-start"),
    4: ("Player4Start", "player-start"),
    11: ("DeathmatchStart", "deathmatch-start"),
    14: ("TeleportDest", "teleport-dest"),
    # Keys
    5: ("BlueKeycard", "key"),
    6: ("YellowKeycard", "key"),
    13: ("RedKeycard", "key"),
    38: ("RedSkullKey", "key"),
    39: ("YellowSkullKey", "key"),
    40: ("BlueSkullKey", "key"),
    # Weapons
    2001: ("Shotgun", "weapon"),
    2002: ("Chaingun", "weapon"),
    2003: ("RocketLauncher", "weapon"),
    2004: ("PlasmaRifle", "weapon"),
    2005: ("Chainsaw", "weapon"),
    2006: ("BFG9000", "weapon"),
    82: ("SuperShotgun", "weapon"),
    # Ammo
    2007: ("ClipAmmo", "ammo"),
    2008: ("ShellAmmo", "ammo"),
    2010: ("RocketAmmo", "ammo"),
    2047: ("CellAmmo", "ammo"),
    2048: ("ClipBox", "ammo"),
    2049: ("ShellBox", "ammo"),
    17: ("CellPack", "ammo"),
    2046: ("RocketBox", "ammo"),
    # Health/armor items
    2011: ("Stimpack", "item"),
    2012: ("Medikit", "item"),
    2013: ("SoulSphere", "item"),
    2014: ("HealthBonus", "item"),
    2015: ("ArmorBonus", "item"),
    2018: ("GreenArmor", "item"),
    2019: ("BlueArmor", "item"),
    2022: ("Invulnerability", "item"),
    2023: ("Berserk", "item"),
    2024: ("Invisibility", "item"),
    2025: ("RadSuit", "item"),
    2026: ("Computer Map", "item"),
    2045: ("LightAmp", "item"),
    83: ("MegaSphere", "item"),
    # Monsters
    3004: ("Zombieman", "monster"),
    9: ("Shotgunguy", "monster"),
    3001: ("Imp", "monster"),
    3002: ("Demon", "monster"),
    58: ("Spectre", "monster"),
    3003: ("BaronOfHell", "monster"),
    3005: ("Cacodemon", "monster"),
    3006: ("LostSoul", "monster"),
    69: ("HellKnight", "monster"),
    16: ("Cyberdemon", "monster"),
    7: ("SpiderMastermind", "monster"),
    68: ("Arachnotron", "monster"),
    64: ("ArchVile", "monster"),
    65: ("HeavyWeaponDude", "monster"),
    66: ("Revenant", "monster"),
    67: ("Mancubus", "monster"),
    71: ("PainElemental", "monster"),
    84: ("Wolfenstein SS", "monster"),
    72: ("Commander Keen", "monster"),
    88: ("Romero head", "monster"),
    89: ("Monsters Spawner", "monster"),
    # Decorations / obstacles (subset)
    2028: ("Floor lamp", "decoration"),
    2035: ("Barrel", "obstacle"),
    2057: ("Chair", "decoration"),
    25: ("Impaled human", "decoration"),
    26: ("Twitching impaled human", "decoration"),
    27: ("Skull on pole", "decoration"),
    28: ("5 skulls shishkebab", "decoration"),
    29: ("Pile of skulls", "decoration"),
    30: ("Tall green pillar", "obstacle"),
    31: ("Short green pillar", "obstacle"),
    32: ("Tall red pillar", "obstacle"),
    33: ("Short red pillar", "obstacle"),
    35: ("Candelabra", "decoration"),
    36: ("Short red pillar w/skull", "obstacle"),
    41: ("Evil eye", "decoration"),
    42: ("Floating skull", "decoration"),
    43: ("Torch tree", "decoration"),
    44: ("Blue torch", "decoration"),
    45: ("Green torch", "decoration"),
    46: ("Red torch", "decoration"),
    47: ("Stalagmite", "obstacle"),
    48: ("Tall techno pillar", "obstacle"),
    49: ("Hanging victim, twitching", "decoration"),
    50: ("Hanging victim, arms out", "decoration"),
    51: ("Hanging victim, 1-legged", "decoration"),
    52: ("Hanging pair of legs", "decoration"),
    53: ("Hanging leg", "decoration"),
    55: ("Tall techno floor lamp", "decoration"),
    56: ("Short techno floor lamp", "decoration"),
    57: ("Hanging victim, guts removed", "decoration"),
    70: ("Burning barrel", "obstacle"),
    73: ("Hanging victim, guts and brain", "decoration"),
    74: ("Hanging torso, looking down", "decoration"),
    75: ("Hanging victim, meat hooks", "decoration"),
    76: ("Hanging victim, 1-legged (variant)", "decoration"),
    77: ("Hanging pair of legs (variant)", "decoration"),
    78: ("Hanging leg (variant)", "decoration"),
    85: ("Tall techno lamp", "decoration"),
    86: ("Short techno lamp", "decoration"),
    87: ("Spawn spot", "special"),
    24: ("Pool of blood and guts", "decoration"),
    79: ("Pool of blood", "decoration"),
    80: ("Pool of blood (variant)", "decoration"),
    81: ("Pool of brains", "decoration"),
    10: ("Bloody mess", "decoration"),
    12: ("Bloody mess (variant)", "decoration"),
    34: ("Candle", "decoration"),
    37: ("Skull on pole (variant)", "decoration"),
    54: ("Large brown tree", "obstacle"),
    59: ("Big hanging pair of legs", "decoration"),
    60: ("Big hanging victim, twitching", "decoration"),
    61: ("Big hanging victim, arms out", "decoration"),
    62: ("Big hanging torso", "decoration"),
    63: ("Big hanging leg", "decoration"),
    15: ("Dead player", "decoration"),
    18: ("Dead former human", "decoration"),
    19: ("Dead former sergeant", "decoration"),
    20: ("Dead imp", "decoration"),
    21: ("Dead demon", "decoration"),
    22: ("Dead cacodemon", "decoration"),
    23: ("Dead lost soul, invisible", "decoration"),
}


def resolve_doomed(type_num: int) -> tuple[str, str, bool]:
    """Returns (name, category, unresolved)."""
    entry = DOOMED_TABLE.get(type_num)
    if entry is None:
        return (f"Unknown_{type_num}", "unknown", True)
    return (entry[0], entry[1], False)


# ---------------------------------------------------------------------------
# Linedef special lookup
# ---------------------------------------------------------------------------
# (name, trigger_type) -- trigger_type is one of: walk, switch, push, shoot,
# gun, cross (walk-over), manual, unknown

LINEDEF_SPECIAL_TABLE: dict[int, tuple[str, str]] = {
    0: ("None", "none"),
    1: ("Door_Raise (manual, repeatable)", "push"),
    2: ("Door_Open (walk, once)", "walk"),
    3: ("Door_Close (walk, once)", "walk"),
    4: ("Door_Raise (walk, once)", "walk"),
    5: ("Floor_RaiseToLowestCeiling (walk, once)", "walk"),
    6: ("Ceiling_CrushAndRaiseFast (walk, once)", "walk"),
    7: ("Stairs_BuildBy8 (walk, once)", "walk"),
    8: ("Stairs_BuildBy8 (walk, repeatable)", "walk"),
    9: ("Floor_Donut (walk, once)", "walk"),
    10: ("Plat_DownWaitUpStay (walk, once)", "walk"),
    11: ("Exit_Normal (switch)", "switch"),
    12: ("Light_MatchBrightest (walk, once)", "walk"),
    13: ("Light_ChangeTo255 (walk, once)", "walk"),
    14: ("Plat_RaiseAndChange (walk, once)", "walk"),
    16: ("Door_CloseThenOpen (walk, once)", "walk"),
    17: ("Light_Strobe (walk, once)", "walk"),
    19: ("Floor_LowerToHighestFloor (walk, once)", "walk"),
    20: ("Plat_RaiseToNearestAndChange (walk, once)", "walk"),
    21: ("Plat_DownWaitUpStay (walk, repeatable)", "walk"),
    23: ("Floor_LowerToLowest (walk, once)", "walk"),
    26: ("Door_Locked_Raise (blue key, manual, repeatable)", "push"),
    27: ("Door_Locked_Raise (yellow key, manual, repeatable)", "push"),
    28: ("Door_Locked_Raise (red key, manual, repeatable)", "push"),
    29: ("Door_Raise (switch, once)", "switch"),
    31: ("Door_Open (switch, once)", "switch"),
    32: ("Door_Locked_Open (blue key, switch, once)", "switch"),
    33: ("Door_Locked_Open (red key, switch, once)", "switch"),
    34: ("Door_Locked_Open (yellow key, switch, once)", "switch"),
    36: ("Floor_LowerToHighestPlus8 (walk, once)", "walk"),
    38: ("Floor_LowerToLowest (walk, once, alt)", "walk"),
    40: ("Ceiling_RaiseToHighest (walk, once)", "walk"),
    42: ("Door_Close (switch, repeatable)", "switch"),
    46: ("Door_Open (gunfire, repeatable)", "gun"),
    52: ("Exit_Secret (walk, once)", "walk"),
    62: ("Plat_DownWaitUpStay (switch, repeatable)", "switch"),
    63: ("Door_Raise (switch, repeatable)", "switch"),
    64: ("Floor_RaiseToLowestCeiling (switch, repeatable)", "switch"),
    72: ("Ceiling_LowerAndCrush (walk, repeatable)", "walk"),
    73: ("Ceiling_CrushAndRaise (walk, repeatable)", "walk"),
    75: ("Door_Close (walk, repeatable)", "walk"),
    76: ("Door_CloseThenOpen (walk, repeatable)", "walk"),
    77: ("Door_CloseThenOpen (walk, repeatable, alt)", "walk"),
    86: ("Door_Open (walk, repeatable)", "walk"),
    90: ("Door_Raise (walk, repeatable)", "walk"),
    97: ("Teleport (walk, repeatable)", "walk"),
    99: ("Door_Locked_Raise (blue skull, switch, repeatable)", "switch"),
    103: ("Door_Open (switch, once, alt)", "switch"),
    105: ("Door_Raise (walk, fast, repeatable)", "walk"),
    106: ("Door_Open (walk, fast, once)", "walk"),
    107: ("Door_Close (walk, fast, once)", "walk"),
    108: ("Door_Raise (walk, fast, once)", "walk"),
    109: ("Door_Raise (switch, fast, once)", "switch"),
    110: ("Door_Open (switch, fast, once)", "switch"),
    111: ("Door_Close (switch, fast, once)", "switch"),
    117: ("Door_Raise (push, fast, repeatable)", "push"),
    118: ("Door_Open (push, fast, once)", "push"),
    134: ("Door_Locked_Raise (red skull, switch, repeatable)", "switch"),
    135: ("Door_Locked_Raise (red skull, switch, once)", "switch"),
    136: ("Door_Locked_Raise (yellow skull, walk, repeatable)", "walk"),
    137: ("Door_Locked_Raise (yellow skull, switch, once)", "switch"),
}


def resolve_linedef_special(special_num: int) -> tuple[str, str]:
    entry = LINEDEF_SPECIAL_TABLE.get(special_num)
    if entry is None:
        return (f"Special_{special_num}", "unknown")
    return entry

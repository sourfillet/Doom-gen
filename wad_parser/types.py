"""All data classes and lookup tables for Doom map elements."""
from __future__ import annotations

from dataclasses import dataclass, field


# ─── Flag types ──────────────────────────────────────────────────────────────

@dataclass
class LindefFlags:
    impassable:      bool = False
    block_monsters:  bool = False
    two_sided:       bool = False
    upper_unpegged:  bool = False
    lower_unpegged:  bool = False
    secret:          bool = False
    blocks_sound:    bool = False
    not_on_map:      bool = False
    already_on_map:  bool = False
    pass_through:    bool = False   # Boom

    @classmethod
    def from_int(cls, v: int) -> LindefFlags:
        return cls(
            impassable=bool(v & 0x0001),
            block_monsters=bool(v & 0x0002),
            two_sided=bool(v & 0x0004),
            upper_unpegged=bool(v & 0x0008),
            lower_unpegged=bool(v & 0x0010),
            secret=bool(v & 0x0020),
            blocks_sound=bool(v & 0x0040),
            not_on_map=bool(v & 0x0080),
            already_on_map=bool(v & 0x0100),
            pass_through=bool(v & 0x0200),
        )

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v}


@dataclass
class ThingFlags:
    skill_1_2:        bool = False
    skill_3:          bool = False
    skill_4_5:        bool = False
    deaf:             bool = False
    multiplayer_only: bool = False   # not in single-player
    not_in_dm:        bool = False   # Boom
    not_in_coop:      bool = False   # Boom
    friendly:         bool = False   # MBF

    @classmethod
    def from_int(cls, v: int) -> ThingFlags:
        return cls(
            skill_1_2=bool(v & 0x0001),
            skill_3=bool(v & 0x0002),
            skill_4_5=bool(v & 0x0004),
            deaf=bool(v & 0x0008),
            multiplayer_only=bool(v & 0x0010),
            not_in_dm=bool(v & 0x0020),
            not_in_coop=bool(v & 0x0040),
            friendly=bool(v & 0x0080),
        )

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v}


# ─── Map elements ─────────────────────────────────────────────────────────────

@dataclass
class Vertex:
    id: int
    x: int
    y: int

    def to_dict(self) -> dict:
        return {"id": self.id, "x": self.x, "y": self.y}


@dataclass
class Sector:
    id: int
    floor_height: int
    ceil_height:  int
    floor_flat:   str
    ceil_flat:    str
    light_level:  int
    special:      int
    tag:          int
    special_name: str = ""

    def to_dict(self) -> dict:
        d: dict = {
            "id":         self.id,
            "floor_h":    self.floor_height,
            "ceil_h":     self.ceil_height,
            "floor_flat": self.floor_flat,
            "ceil_flat":  self.ceil_flat,
            "light":      self.light_level,
            "special":    self.special,
            "tag":        self.tag,
        }
        if self.special_name:
            d["special_name"] = self.special_name
        return d


@dataclass
class Sidedef:
    id:        int
    x_offset:  int
    y_offset:  int
    upper_tex: str
    lower_tex: str
    mid_tex:   str
    sector:    int

    def to_dict(self) -> dict:
        return {
            "id":     self.id,
            "x_off":  self.x_offset,
            "y_off":  self.y_offset,
            "upper":  self.upper_tex,
            "lower":  self.lower_tex,
            "mid":    self.mid_tex,
            "sector": self.sector,
        }


@dataclass
class Linedef:
    id:            int
    v1:            int
    v2:            int
    flags:         LindefFlags
    special:       int
    tag:           int
    right_sidedef: int
    left_sidedef:  int          # -1 = one-sided
    special_name:  str = ""
    trigger_type:  str = ""     # W1 WR S1 SR G1 GR DR D1
    action_type:   str = ""     # door floor ceiling platform etc.

    def to_dict(self) -> dict:
        d: dict = {
            "id":      self.id,
            "v1":      self.v1,
            "v2":      self.v2,
            "flags":   self.flags.to_dict(),
            "special": self.special,
            "tag":     self.tag,
            "right":   self.right_sidedef,
            "left":    self.left_sidedef,
        }
        if self.special_name:
            d["special_name"] = self.special_name
        if self.trigger_type:
            d["trigger_type"] = self.trigger_type
        if self.action_type:
            d["action_type"] = self.action_type
        return d


@dataclass
class Thing:
    id:        int
    x:         int
    y:         int
    angle:     int
    type:      int
    flags:     ThingFlags
    type_name: str = ""
    category:  str = ""

    def to_dict(self) -> dict:
        d: dict = {
            "id":    self.id,
            "x":     self.x,
            "y":     self.y,
            "angle": self.angle,
            "type":  self.type,
            "flags": self.flags.to_dict(),
        }
        if self.type_name:
            d["type_name"] = self.type_name
        if self.category:
            d["category"] = self.category
        return d


# ─── Texture types ────────────────────────────────────────────────────────────

@dataclass
class TexturePatch:
    name:        str
    x_origin:    int
    y_origin:    int
    patch_index: int

    def to_dict(self) -> dict:
        return {"name": self.name, "x": self.x_origin, "y": self.y_origin}


@dataclass
class Texture:
    name:    str
    width:   int
    height:  int
    patches: list[TexturePatch]

    def to_dict(self) -> dict:
        return {
            "name":    self.name,
            "width":   self.width,
            "height":  self.height,
            "patches": [p.to_dict() for p in self.patches],
        }


@dataclass
class TextureSet:
    textures:    dict[str, Texture] = field(default_factory=dict)
    flats:       dict[str, bytes]   = field(default_factory=dict)
    palettes:    list[bytes]        = field(default_factory=list)  # 14 × 768 bytes
    colormaps:   list[bytes]        = field(default_factory=list)  # 34 × 256 bytes
    patch_names: list[str]          = field(default_factory=list)


# ─── Map container ────────────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    min_x: int
    max_x: int
    min_y: int
    max_y: int

    @property
    def width(self) -> int:
        return self.max_x - self.min_x

    @property
    def height(self) -> int:
        return self.max_y - self.min_y

    def to_dict(self) -> dict:
        return {
            "min_x": self.min_x, "max_x": self.max_x,
            "min_y": self.min_y, "max_y": self.max_y,
        }


@dataclass
class MapMeta:
    map_name:   str
    source_wad: str
    wad_type:   str              # IWAD or PWAD
    format:     str = "doom"    # doom | hexen | udmf
    tags:       list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "map_name":   self.map_name,
            "source_wad": self.source_wad,
            "wad_type":   self.wad_type,
            "format":     self.format,
            "tags":       self.tags,
        }


@dataclass
class DoomMap:
    meta:     MapMeta
    bounds:   BoundingBox
    vertices: list[Vertex]
    sectors:  list[Sector]
    linedefs: list[Linedef]
    sidedefs: list[Sidedef]
    things:   list[Thing]
    # Derived data — populated by map_extractor after raw parse
    sector_graph:    dict                  = field(default_factory=dict)
    sector_polygons: list[list[list[int]]] = field(default_factory=list)
    trigger_map:     dict[str, list[int]]  = field(default_factory=dict)
    connectivity:    dict                  = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "meta":     self.meta.to_dict(),
            "bounds":   self.bounds.to_dict(),
            "vertices": [v.to_dict() for v in self.vertices],
            "sectors":  [s.to_dict() for s in self.sectors],
            "linedefs": [l.to_dict() for l in self.linedefs],
            "sidedefs": [s.to_dict() for s in self.sidedefs],
            "things":   [t.to_dict() for t in self.things],
            "derived": {
                "sector_graph":    self.sector_graph,
                "sector_polygons": self.sector_polygons,
                "trigger_map":     self.trigger_map,
                "connectivity":    self.connectivity,
            },
        }


# ─── Lookup tables ────────────────────────────────────────────────────────────

# (display_name, category, frozenset_of_game_keys)
# game keys: "d1"=Doom 1, "d2"=Doom 2, "fd"=Freedoom
THING_TYPES: dict[int, tuple[str, str, frozenset]] = {
    # Player starts / spawns
    1:    ("Player 1 Start",            "player-start",  frozenset(["d1", "d2", "fd"])),
    2:    ("Player 2 Start",            "player-start",  frozenset(["d1", "d2", "fd"])),
    3:    ("Player 3 Start",            "player-start",  frozenset(["d1", "d2", "fd"])),
    4:    ("Player 4 Start",            "player-start",  frozenset(["d1", "d2", "fd"])),
    11:   ("Deathmatch Start",          "player-start",  frozenset(["d1", "d2", "fd"])),
    # Teleport destinations
    14:   ("Teleport Destination",      "teleport-dest", frozenset(["d1", "d2", "fd"])),
    # Keys
    5:    ("Blue Keycard",              "key",           frozenset(["d1", "d2", "fd"])),
    6:    ("Yellow Keycard",            "key",           frozenset(["d1", "d2", "fd"])),
    13:   ("Red Keycard",               "key",           frozenset(["d1", "d2", "fd"])),
    38:   ("Red Skull Key",             "key",           frozenset(["d1", "d2", "fd"])),
    39:   ("Yellow Skull Key",          "key",           frozenset(["d1", "d2", "fd"])),
    40:   ("Blue Skull Key",            "key",           frozenset(["d1", "d2", "fd"])),
    # Monsters
    3004: ("Zombieman",                 "monster",       frozenset(["d1", "d2", "fd"])),
    9:    ("Shotgun Guy",               "monster",       frozenset(["d1", "d2", "fd"])),
    65:   ("Chaingunner",               "monster",       frozenset(["d2", "fd"])),
    3001: ("Imp",                       "monster",       frozenset(["d1", "d2", "fd"])),
    3002: ("Demon",                     "monster",       frozenset(["d1", "d2", "fd"])),
    58:   ("Spectre",                   "monster",       frozenset(["d1", "d2", "fd"])),
    3006: ("Lost Soul",                 "monster",       frozenset(["d1", "d2", "fd"])),
    3005: ("Cacodemon",                 "monster",       frozenset(["d1", "d2", "fd"])),
    69:   ("Hell Knight",               "monster",       frozenset(["d2", "fd"])),
    3003: ("Baron of Hell",             "monster",       frozenset(["d1", "d2", "fd"])),
    68:   ("Arachnotron",               "monster",       frozenset(["d2", "fd"])),
    71:   ("Pain Elemental",            "monster",       frozenset(["d2", "fd"])),
    66:   ("Revenant",                  "monster",       frozenset(["d2", "fd"])),
    67:   ("Mancubus",                  "monster",       frozenset(["d2", "fd"])),
    64:   ("Archvile",                  "monster",       frozenset(["d2", "fd"])),
    72:   ("Commander Keen",            "decoration",    frozenset(["d2"])),
    16:   ("Cyberdemon",                "monster",       frozenset(["d1", "d2", "fd"])),
    7:    ("Spider Mastermind",         "monster",       frozenset(["d1", "d2", "fd"])),
    84:   ("Wolfenstein SS",            "monster",       frozenset(["d2"])),
    88:   ("Romero's Head",             "decoration",    frozenset(["d2"])),
    # Weapons
    2005: ("Chainsaw",                  "weapon",        frozenset(["d1", "d2", "fd"])),
    2001: ("Shotgun",                   "weapon",        frozenset(["d1", "d2", "fd"])),
    82:   ("Super Shotgun",             "weapon",        frozenset(["d2", "fd"])),
    2002: ("Chaingun",                  "weapon",        frozenset(["d1", "d2", "fd"])),
    2003: ("Rocket Launcher",           "weapon",        frozenset(["d1", "d2", "fd"])),
    2004: ("Plasma Rifle",              "weapon",        frozenset(["d1", "d2", "fd"])),
    2006: ("BFG 9000",                  "weapon",        frozenset(["d1", "d2", "fd"])),
    # Ammo
    2007: ("Clip",                      "ammo",          frozenset(["d1", "d2", "fd"])),
    2048: ("Box of Bullets",            "ammo",          frozenset(["d1", "d2", "fd"])),
    2008: ("Shotgun Shells",            "ammo",          frozenset(["d1", "d2", "fd"])),
    2049: ("Box of Shells",             "ammo",          frozenset(["d1", "d2", "fd"])),
    2010: ("Rocket",                    "ammo",          frozenset(["d1", "d2", "fd"])),
    2046: ("Box of Rockets",            "ammo",          frozenset(["d1", "d2", "fd"])),
    2047: ("Energy Cell",               "ammo",          frozenset(["d1", "d2", "fd"])),
    17:   ("Cell Pack",                 "ammo",          frozenset(["d1", "d2", "fd"])),
    8:    ("Backpack",                  "ammo",          frozenset(["d1", "d2", "fd"])),
    # Health
    2014: ("Health Bonus",              "health",        frozenset(["d1", "d2", "fd"])),
    2011: ("Stimpack",                  "health",        frozenset(["d1", "d2", "fd"])),
    2012: ("Medikit",                   "health",        frozenset(["d1", "d2", "fd"])),
    2013: ("Soulsphere",                "health",        frozenset(["d1", "d2", "fd"])),
    # Armor
    2015: ("Armor Bonus",               "armor",         frozenset(["d1", "d2", "fd"])),
    2018: ("Green Armor",               "armor",         frozenset(["d1", "d2", "fd"])),
    2019: ("Blue Armor",                "armor",         frozenset(["d1", "d2", "fd"])),
    # Powerups
    2022: ("Invulnerability Sphere",    "powerup",       frozenset(["d1", "d2", "fd"])),
    2023: ("Berserk",                   "powerup",       frozenset(["d1", "d2", "fd"])),
    2024: ("Partial Invisibility",      "powerup",       frozenset(["d1", "d2", "fd"])),
    2025: ("Radiation Shielding Suit",  "powerup",       frozenset(["d1", "d2", "fd"])),
    2026: ("Computer Area Map",         "powerup",       frozenset(["d1", "d2", "fd"])),
    2045: ("Light Amplification Visor", "powerup",       frozenset(["d1", "d2", "fd"])),
    83:   ("Megasphere",                "powerup",       frozenset(["d2", "fd"])),
    # Decorations — blocking
    30:   ("Tall Green Column",         "decoration",    frozenset(["d1", "d2", "fd"])),
    31:   ("Short Green Column",        "decoration",    frozenset(["d1", "d2", "fd"])),
    32:   ("Tall Red Column",           "decoration",    frozenset(["d1", "d2", "fd"])),
    33:   ("Short Red Column",          "decoration",    frozenset(["d1", "d2", "fd"])),
    37:   ("Short Skull Column",        "decoration",    frozenset(["d2", "fd"])),
    36:   ("Tall Skull Column",         "decoration",    frozenset(["d2", "fd"])),
    41:   ("Evil Eye",                  "decoration",    frozenset(["d1", "d2", "fd"])),
    42:   ("Floating Skull Rock",       "decoration",    frozenset(["d1", "d2", "fd"])),
    43:   ("Gray Tree",                 "decoration",    frozenset(["d1", "d2", "fd"])),
    54:   ("Large Brown Tree",          "decoration",    frozenset(["d1", "d2", "fd"])),
    35:   ("Candelabra",               "decoration",    frozenset(["d1", "d2", "fd"])),
    48:   ("Technical Column",          "decoration",    frozenset(["d1", "d2", "fd"])),
    85:   ("Tall Techno Pillar",        "decoration",    frozenset(["d2", "fd"])),
    70:   ("Burning Barrel",            "decoration",    frozenset(["d2", "fd"])),
    2035: ("Barrel",                    "decoration",    frozenset(["d1", "d2", "fd"])),
    2028: ("Floor Lamp",                "decoration",    frozenset(["d1", "d2", "fd"])),
    # Decorations — non-blocking / ambient
    34:   ("Candle",                    "decoration",    frozenset(["d1", "d2", "fd"])),
    44:   ("Tall Blue Firestick",       "decoration",    frozenset(["d1", "d2", "fd"])),
    45:   ("Tall Green Firestick",      "decoration",    frozenset(["d1", "d2", "fd"])),
    46:   ("Tall Red Firestick",        "decoration",    frozenset(["d1", "d2", "fd"])),
    47:   ("Short Blue Firestick",      "decoration",    frozenset(["d1", "d2", "fd"])),
    55:   ("Short Green Firestick",     "decoration",    frozenset(["d1", "d2", "fd"])),
    56:   ("Short Red Firestick",       "decoration",    frozenset(["d1", "d2", "fd"])),
    # Gore / dead things
    10:   ("Bloody Mess",               "gore",          frozenset(["d1", "d2", "fd"])),
    12:   ("Bloody Mess (alt)",         "gore",          frozenset(["d1", "d2", "fd"])),
    24:   ("Gibs",                      "gore",          frozenset(["d1", "d2", "fd"])),
    15:   ("Dead Player",               "gore",          frozenset(["d1", "d2", "fd"])),
    18:   ("Dead Zombieman",            "gore",          frozenset(["d1", "d2", "fd"])),
    19:   ("Dead Shotgun Guy",          "gore",          frozenset(["d1", "d2", "fd"])),
    20:   ("Dead Imp",                  "gore",          frozenset(["d1", "d2", "fd"])),
    21:   ("Dead Demon",                "gore",          frozenset(["d1", "d2", "fd"])),
    22:   ("Dead Cacodemon",            "gore",          frozenset(["d1", "d2", "fd"])),
    23:   ("Dead Lost Soul (invis)",    "gore",          frozenset(["d1", "d2", "fd"])),
    79:   ("Pool of Blood",             "gore",          frozenset(["d2", "fd"])),
    80:   ("Pool of Blood (alt)",       "gore",          frozenset(["d2", "fd"])),
    81:   ("Pool of Brains",            "gore",          frozenset(["d2", "fd"])),
    # Impaled / hanging decorations (blocking)
    25:   ("Impaled Human",                          "decoration", frozenset(["d1", "d2", "fd"])),
    26:   ("Twitching Impaled Human",                "decoration", frozenset(["d1", "d2", "fd"])),
    27:   ("Skull on Pole",                          "decoration", frozenset(["d1", "d2", "fd"])),
    28:   ("5 Skulls Shish Kebab",                   "decoration", frozenset(["d1", "d2", "fd"])),
    29:   ("Pile of Skulls and Candles",             "decoration", frozenset(["d1", "d2", "fd"])),
    49:   ("Hanging Victim, Twitching (blocking)",   "decoration", frozenset(["d1", "d2", "fd"])),
    50:   ("Hanging Victim, Arms Out (blocking)",    "decoration", frozenset(["d1", "d2", "fd"])),
    51:   ("Hanging Victim, 1-Legged (blocking)",    "decoration", frozenset(["d1", "d2", "fd"])),
    52:   ("Hanging Pair of Legs (blocking)",        "decoration", frozenset(["d1", "d2", "fd"])),
    53:   ("Hanging Leg (blocking)",                 "decoration", frozenset(["d1", "d2", "fd"])),
    73:   ("Hanging Victim, Bleeding Brain (bl.)",   "decoration", frozenset(["d2", "fd"])),
    75:   ("Hanging Torso, Looking Down (bl.)",      "decoration", frozenset(["d2", "fd"])),
    76:   ("Hanging Torso, Open Skull (bl.)",        "decoration", frozenset(["d2", "fd"])),
    77:   ("Hanging Torso, Looking Up (bl.)",        "decoration", frozenset(["d2", "fd"])),
    # Hanging decorations (non-blocking)
    59:   ("Hanging Victim, Arms Out (non-bl.)",     "decoration", frozenset(["d1", "d2", "fd"])),
    60:   ("Hanging Pair of Legs (non-bl.)",         "decoration", frozenset(["d1", "d2", "fd"])),
    61:   ("Hanging Victim, 1-Legged (non-bl.)",     "decoration", frozenset(["d1", "d2", "fd"])),
    62:   ("Hanging Leg (non-bl.)",                  "decoration", frozenset(["d1", "d2", "fd"])),
    63:   ("Hanging Victim, Twitching (non-bl.)",    "decoration", frozenset(["d1", "d2", "fd"])),
    74:   ("Hanging Victim, Bleeding Brain (non-bl.)","decoration",frozenset(["d2", "fd"])),
    78:   ("Hanging Torso, Mouth Open (non-bl.)",    "decoration", frozenset(["d2", "fd"])),
}

# (display_name, trigger_type, action_type)
# trigger_type: W1 WR S1 SR G1 GR DR D1 None
# action_type: door floor ceiling platform stairs light teleport exit scroll
LINEDEF_SPECIALS: dict[int, tuple[str, str | None, str | None]] = {
    0:   ("None",                                     None,  None),
    1:   ("DR Door (open/close)",                     "DR",  "door"),
    2:   ("W1 Open Door (stay)",                      "W1",  "door"),
    3:   ("W1 Close Door",                            "W1",  "door"),
    4:   ("W1 Raise Door",                            "W1",  "door"),
    5:   ("W1 Raise Floor",                           "W1",  "floor"),
    6:   ("W1 Fast Ceiling Crush & Raise",            "W1",  "ceiling"),
    7:   ("S1 Build Stairs (8)",                      "S1",  "stairs"),
    8:   ("W1 Build Stairs (8)",                      "W1",  "stairs"),
    9:   ("S1 Donut",                                 "S1",  "floor"),
    10:  ("W1 Platform Down-Wait-Up-Stay",            "W1",  "platform"),
    11:  ("S1 Exit Level",                            "S1",  "exit"),
    12:  ("W1 Light to Brightest",                    "W1",  "light"),
    13:  ("W1 Light to 255",                          "W1",  "light"),
    14:  ("S1 Platform Raise 32 (tex chg)",           "S1",  "platform"),
    15:  ("S1 Platform Raise 24 (tex chg)",           "S1",  "platform"),
    16:  ("W1 Close Door (open after 30 sec)",        "W1",  "door"),
    17:  ("W1 Start Light Strobing",                  "W1",  "light"),
    18:  ("S1 Raise Floor to Next",                   "S1",  "floor"),
    19:  ("W1 Lower Floor",                           "W1",  "floor"),
    20:  ("S1 Platform Raise 24 (tex chg)",           "S1",  "platform"),
    21:  ("S1 Platform Down-Wait-Up-Stay",            "S1",  "platform"),
    22:  ("W1 Platform Raise 24 (tex chg)",           "W1",  "platform"),
    23:  ("S1 Lower Floor to Lowest",                 "S1",  "floor"),
    24:  ("G1 Raise Floor",                           "G1",  "floor"),
    25:  ("W1 Ceiling Crush & Raise",                 "W1",  "ceiling"),
    26:  ("DR Blue Door (open/close)",                "DR",  "door"),
    27:  ("DR Yellow Door (open/close)",              "DR",  "door"),
    28:  ("DR Red Door (open/close)",                 "DR",  "door"),
    29:  ("S1 Raise Door",                            "S1",  "door"),
    30:  ("W1 Raise Floor by Shortest Lower",         "W1",  "floor"),
    31:  ("D1 Open Door (stay)",                      "D1",  "door"),
    32:  ("D1 Blue Door (stay)",                      "D1",  "door"),
    33:  ("D1 Red Door (stay)",                       "D1",  "door"),
    34:  ("D1 Yellow Door (stay)",                    "D1",  "door"),
    35:  ("W1 Light to 35",                           "W1",  "light"),
    36:  ("W1 Lower Floor Fast",                      "W1",  "floor"),
    37:  ("W1 Lower Floor (tex chg)",                 "W1",  "floor"),
    38:  ("W1 Lower Floor to Lowest",                 "W1",  "floor"),
    39:  ("W1 Teleport",                              "W1",  "teleport"),
    40:  ("W1 Raise Ceiling / Lower Floor",           "W1",  "ceiling"),
    41:  ("S1 Lower Ceiling",                         "S1",  "ceiling"),
    42:  ("SR Close Door",                            "SR",  "door"),
    43:  ("SR Lower Ceiling to Floor",                "SR",  "ceiling"),
    44:  ("W1 Ceiling Crush (stay)",                  "W1",  "ceiling"),
    45:  ("SR Lower Floor",                           "SR",  "floor"),
    46:  ("GR Open Door (stay)",                      "GR",  "door"),
    47:  ("G1 Platform Raise 16 (tex chg)",           "G1",  "platform"),
    48:  ("Scroll Texture Left",                      None,  "scroll"),
    49:  ("S1 Ceiling Crush & Raise",                 "S1",  "ceiling"),
    50:  ("S1 Close Door",                            "S1",  "door"),
    51:  ("S1 Exit Secret Level",                     "S1",  "exit"),
    52:  ("W1 Exit Level",                            "W1",  "exit"),
    53:  ("W1 Platform Perpetual",                    "W1",  "platform"),
    54:  ("W1 Platform Stop",                         "W1",  "platform"),
    55:  ("S1 Floor Crush & Raise",                   "S1",  "floor"),
    56:  ("W1 Floor Crush & Raise",                   "W1",  "floor"),
    57:  ("W1 Ceiling Crush Stop",                    "W1",  "ceiling"),
    58:  ("W1 Raise Floor 24",                        "W1",  "floor"),
    59:  ("W1 Raise Floor 24 (tex chg)",              "W1",  "floor"),
    60:  ("SR Lower Floor to Lowest",                 "SR",  "floor"),
    61:  ("SR Open Door (stay)",                      "SR",  "door"),
    62:  ("SR Platform Down-Wait-Up-Stay",            "SR",  "platform"),
    63:  ("SR Raise Door",                            "SR",  "door"),
    64:  ("SR Raise Floor to Ceiling",                "SR",  "floor"),
    65:  ("SR Floor Crush & Raise",                   "SR",  "floor"),
    66:  ("SR Platform Raise 24 (tex chg)",           "SR",  "platform"),
    67:  ("SR Platform Raise 32 (tex chg)",           "SR",  "platform"),
    68:  ("SR Platform Raise to Next (tex chg)",      "SR",  "platform"),
    69:  ("SR Raise Floor to Next",                   "SR",  "floor"),
    70:  ("SR Lower Floor Fast",                      "SR",  "floor"),
    71:  ("S1 Lower Floor Fast",                      "S1",  "floor"),
    72:  ("WR Ceiling Crush (stay)",                  "WR",  "ceiling"),
    73:  ("WR Ceiling Crush & Raise",                 "WR",  "ceiling"),
    74:  ("WR Ceiling Crush Stop",                    "WR",  "ceiling"),
    75:  ("WR Close Door",                            "WR",  "door"),
    76:  ("WR Close Door (open after 30 sec)",        "WR",  "door"),
    77:  ("WR Fast Ceiling Crush & Raise",            "WR",  "ceiling"),
    79:  ("WR Light to 35",                           "WR",  "light"),
    80:  ("WR Light to Brightest",                    "WR",  "light"),
    81:  ("WR Light to 255",                          "WR",  "light"),
    82:  ("WR Lower Floor to Lowest",                 "WR",  "floor"),
    83:  ("WR Lower Floor",                           "WR",  "floor"),
    84:  ("WR Lower Floor (tex chg)",                 "WR",  "floor"),
    86:  ("WR Open Door (stay)",                      "WR",  "door"),
    87:  ("WR Platform Perpetual",                    "WR",  "platform"),
    88:  ("WR Platform Down-Wait-Up-Stay",            "WR",  "platform"),
    89:  ("WR Platform Stop",                         "WR",  "platform"),
    90:  ("WR Raise Door",                            "WR",  "door"),
    91:  ("WR Raise Floor",                           "WR",  "floor"),
    92:  ("WR Raise Floor 24",                        "WR",  "floor"),
    93:  ("WR Raise Floor 24 (tex chg)",              "WR",  "floor"),
    94:  ("WR Floor Crush & Raise",                   "WR",  "floor"),
    95:  ("WR Platform Raise to Next (tex chg)",      "WR",  "platform"),
    96:  ("WR Raise Floor by Shortest Lower",         "WR",  "floor"),
    97:  ("WR Teleport",                              "WR",  "teleport"),
    98:  ("WR Lower Floor Fast",                      "WR",  "floor"),
    99:  ("SR Blue Door Fast",                        "SR",  "door"),
    100: ("W1 Build Stairs (16)",                     "W1",  "stairs"),
    101: ("S1 Raise Floor",                           "S1",  "floor"),
    102: ("S1 Lower Floor to Highest",                "S1",  "floor"),
    103: ("S1 Open Door (stay)",                      "S1",  "door"),
    104: ("W1 Light to Darkest",                      "W1",  "light"),
    105: ("WR Raise Door Fast",                       "WR",  "door"),
    106: ("WR Open Door Fast (stay)",                 "WR",  "door"),
    107: ("WR Close Door Fast",                       "WR",  "door"),
    108: ("W1 Raise Door Fast",                       "W1",  "door"),
    109: ("W1 Open Door Fast (stay)",                 "W1",  "door"),
    110: ("W1 Close Door Fast",                       "W1",  "door"),
    111: ("S1 Raise Door Fast",                       "S1",  "door"),
    112: ("S1 Open Door Fast (stay)",                 "S1",  "door"),
    113: ("S1 Close Door Fast",                       "S1",  "door"),
    114: ("SR Raise Door Fast",                       "SR",  "door"),
    115: ("SR Open Door Fast (stay)",                 "SR",  "door"),
    116: ("SR Close Door Fast",                       "SR",  "door"),
    117: ("DR Raise Door Fast",                       "DR",  "door"),
    118: ("D1 Open Door Fast (stay)",                 "D1",  "door"),
    119: ("W1 Raise Floor to Next",                   "W1",  "floor"),
    120: ("WR Platform Fast Down-Wait-Up-Stay",       "WR",  "platform"),
    121: ("W1 Platform Fast Down-Wait-Up-Stay",       "W1",  "platform"),
    122: ("S1 Platform Fast Down-Wait-Up-Stay",       "S1",  "platform"),
    123: ("SR Platform Fast Down-Wait-Up-Stay",       "SR",  "platform"),
    124: ("W1 Exit Secret Level",                     "W1",  "exit"),
    125: ("W1 Teleport (monsters only)",              "W1",  "teleport"),
    126: ("WR Teleport (monsters only)",              "WR",  "teleport"),
    127: ("S1 Build Stairs (16)",                     "S1",  "stairs"),
    128: ("WR Raise Floor to Next",                   "WR",  "floor"),
    129: ("WR Raise Floor Fast",                      "WR",  "floor"),
    130: ("W1 Raise Floor Fast",                      "W1",  "floor"),
    131: ("S1 Raise Floor Fast",                      "S1",  "floor"),
    132: ("SR Raise Floor Fast",                      "SR",  "floor"),
    133: ("S1 Blue Door Fast",                        "S1",  "door"),
    134: ("SR Red Door Fast",                         "SR",  "door"),
    135: ("S1 Red Door Fast",                         "S1",  "door"),
    136: ("SR Yellow Door Fast",                      "SR",  "door"),
    137: ("S1 Yellow Door Fast",                      "S1",  "door"),
    138: ("SR Light to 255",                          "SR",  "light"),
    139: ("SR Light to 35",                           "SR",  "light"),
    140: ("S1 Raise Floor 512",                       "S1",  "floor"),
    141: ("W1 Ceiling Crush (silent, stay)",          "W1",  "ceiling"),
}

SECTOR_SPECIALS: dict[int, str] = {
    0:  "Normal",
    1:  "Light Blink (random)",
    2:  "Light Blink (0.5 sec)",
    3:  "Light Blink (1.0 sec)",
    4:  "Damage -10/20% + Light Blink (0.5 sec)",
    5:  "Damage -5/10%",
    7:  "Damage -2/5%",
    8:  "Light Oscillate",
    9:  "Secret",
    10: "Door Close in 30 sec",
    11: "Damage -10/20%; End Level on Death",
    12: "Light Blink (0.5 sec, synced)",
    13: "Light Blink (1.0 sec, synced)",
    14: "Door Open after 300 sec",
    16: "Damage -10/20%",
    17: "Light Flicker (random)",
}

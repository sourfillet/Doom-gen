"""Doom WAD parser — public API."""
from .wad_reader import WadReader
from .map_extractor import extract_map
from .texture_parser import build_texture_set
from .types import (
    DoomMap,
    MapMeta,
    BoundingBox,
    Vertex,
    Sector,
    Sidedef,
    Linedef,
    LindefFlags,
    Thing,
    ThingFlags,
    Texture,
    TexturePatch,
    TextureSet,
    THING_TYPES,
    LINEDEF_SPECIALS,
    SECTOR_SPECIALS,
)

__all__ = [
    "WadReader",
    "extract_map",
    "build_texture_set",
    "DoomMap",
    "MapMeta",
    "BoundingBox",
    "Vertex",
    "Sector",
    "Sidedef",
    "Linedef",
    "LindefFlags",
    "Thing",
    "ThingFlags",
    "Texture",
    "TexturePatch",
    "TextureSet",
    "THING_TYPES",
    "LINEDEF_SPECIALS",
    "SECTOR_SPECIALS",
]

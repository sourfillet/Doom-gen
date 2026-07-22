"""
Parse the Doom texture system: TEXTURE1/TEXTURE2, PNAMES, flats, PLAYPAL, COLORMAP.

Binary formats
──────────────
PNAMES
  int32  count
  char[8][count]  patch names (null-padded, uppercase)

TEXTURE1 / TEXTURE2
  int32  numtextures
  int32[numtextures]  offsets  (from start of lump)
  At each offset:
    char[8]  name
    uint32   masked   (always 0, ignored)
    int16    width
    int16    height
    uint32   columndirectory  (unused, ignored)
    int16    patchcount
    patchcount × mappatch_t:
      int16  originx
      int16  originy
      int16  patch    (index into PNAMES)
      int16  stepdir  (unused)
      int16  colormap (unused)

PLAYPAL   14 palettes × 768 bytes  (256 × RGB)
COLORMAP  34 colormaps × 256 bytes
"""
from __future__ import annotations

import struct
import warnings

from .types import Texture, TexturePatch, TextureSet
from .wad_reader import WadReader

_PATCH_FMT    = struct.Struct("<hhhhh")   # 10 bytes per mappatch_t
_TEX_HDR_FMT  = struct.Struct("<8sIhhIh") # 22 bytes per texture header
_PALETTE_SIZE = 14 * 768
_COLORMAP_SIZE = 34 * 256


def _decode_name(raw: bytes) -> str:
    return raw.rstrip(b"\x00").decode("ascii", errors="replace").upper()


def parse_pnames(data: bytes) -> list[str]:
    """Return an ordered list of patch names from a PNAMES lump."""
    if len(data) < 4:
        return []
    (count,) = struct.unpack_from("<I", data, 0)
    names: list[str] = []
    offset = 4
    for _ in range(count):
        if offset + 8 > len(data):
            break
        names.append(_decode_name(data[offset : offset + 8]))
        offset += 8
    return names


def parse_textures(data: bytes, patch_names: list[str]) -> dict[str, Texture]:
    """
    Parse a TEXTURE1 or TEXTURE2 lump.

    Returns a dict of texture_name → Texture.
    """
    if len(data) < 4:
        return {}

    (num_textures,) = struct.unpack_from("<I", data, 0)
    textures: dict[str, Texture] = {}

    for i in range(num_textures):
        offset_pos = 4 + i * 4
        if offset_pos + 4 > len(data):
            break
        (tex_offset,) = struct.unpack_from("<I", data, offset_pos)

        if tex_offset + _TEX_HDR_FMT.size > len(data):
            warnings.warn(f"Texture {i} header extends beyond lump end")
            continue

        raw_name, _masked, width, height, _coldir, patchcount = \
            _TEX_HDR_FMT.unpack_from(data, tex_offset)
        name = _decode_name(raw_name)

        patches: list[TexturePatch] = []
        patch_base = tex_offset + _TEX_HDR_FMT.size
        for j in range(patchcount):
            p_off = patch_base + j * _PATCH_FMT.size
            if p_off + _PATCH_FMT.size > len(data):
                break
            originx, originy, patch_idx, _stepdir, _cmap = \
                _PATCH_FMT.unpack_from(data, p_off)
            patch_name = (
                patch_names[patch_idx]
                if 0 <= patch_idx < len(patch_names)
                else f"PATCH{patch_idx}"
            )
            patches.append(TexturePatch(
                name=patch_name,
                x_origin=originx,
                y_origin=originy,
                patch_index=patch_idx,
            ))

        textures[name] = Texture(name=name, width=width, height=height, patches=patches)

    return textures


def parse_playpal(data: bytes) -> list[bytes]:
    """
    Return up to 14 palettes, each 768 bytes (256 × RGB triples).
    Doom always has exactly 14.
    """
    palettes: list[bytes] = []
    for i in range(14):
        start = i * 768
        end = start + 768
        if end > len(data):
            break
        palettes.append(data[start:end])
    return palettes


def parse_colormap(data: bytes) -> list[bytes]:
    """
    Return up to 34 colormaps, each 256 bytes (index remapping table).
    Doom always has exactly 34.
    """
    colormaps: list[bytes] = []
    for i in range(34):
        start = i * 256
        end = start + 256
        if end > len(data):
            break
        colormaps.append(data[start:end])
    return colormaps


def parse_flats(wad: WadReader) -> dict[str, bytes]:
    """
    Collect all flats (64×64 = 4096 raw bytes) between FF_START/FF_END
    (or F_START/F_END) markers.

    Flat names are taken from the lump directory; data is the raw 4096 bytes.
    Flats of unexpected sizes are retained with a warning.
    """
    flats: dict[str, bytes] = {}
    in_flats = False

    for entry in wad.lump_list:
        if entry.name in ("F_START", "FF_START"):
            in_flats = True
            continue
        if entry.name in ("F_END", "FF_END"):
            in_flats = False
            continue
        if not in_flats:
            continue
        if entry.name.startswith("F") and entry.name.endswith(("_START", "_END")):
            # Sub-namespace markers (F1_START etc.) — skip
            continue
        if entry.size == 0:
            continue
        if entry.size != 4096:
            warnings.warn(
                f"Flat '{entry.name}' has unexpected size {entry.size} "
                f"(expected 4096); keeping raw bytes"
            )
        data = wad.get_lump(entry.name)
        if data is not None:
            flats[entry.name] = data

    return flats


def build_texture_set(wad: WadReader) -> TextureSet:
    """
    Convenience: parse every texture-related lump from a WAD and return
    a fully populated TextureSet.
    """
    ts = TextureSet()

    # Patch name table
    pnames_data = wad.get_lump("PNAMES")
    if pnames_data:
        ts.patch_names = parse_pnames(pnames_data)

    # Composite textures
    for lump_name in ("TEXTURE1", "TEXTURE2"):
        tex_data = wad.get_lump(lump_name)
        if tex_data:
            ts.textures.update(parse_textures(tex_data, ts.patch_names))

    # Flats
    ts.flats = parse_flats(wad)

    # Palettes and colormaps
    playpal_data = wad.get_lump("PLAYPAL")
    if playpal_data:
        ts.palettes = parse_playpal(playpal_data)

    colormap_data = wad.get_lump("COLORMAP")
    if colormap_data:
        ts.colormaps = parse_colormap(colormap_data)

    return ts

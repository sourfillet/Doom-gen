"""Parsers for WAD-wide texture-system lumps: TEXTURE1/2 + PNAMES, flats,
PLAYPAL, COLORMAP, and the Boom ANIMATED/SWITCHES lumps that record which
textures/flats are animation-cycle frames of each other and which texture
pairs are switches.
"""
from __future__ import annotations

import hashlib
import math
import struct

from wad_parser.types import AnimatedEntry, FlatAsset, Patch, SwitchEntry, Texture, TextureSet


def _decode_name(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("ascii", errors="replace").upper() or "-"


def parse_pnames(data: bytes) -> list[str]:
    if len(data) < 4:
        return []
    (count,) = struct.unpack_from("<i", data, 0)
    names = []
    for i in range(count):
        off = 4 + i * 8
        if off + 8 > len(data):
            break
        names.append(_decode_name(data[off: off + 8]))
    return names


def parse_texture_lump(data: bytes, pnames: list[str]) -> dict[str, Texture]:
    """Parses a TEXTURE1 or TEXTURE2 lump into name -> Texture."""
    if len(data) < 4:
        return {}
    (count,) = struct.unpack_from("<i", data, 0)
    textures: dict[str, Texture] = {}
    for i in range(count):
        (offset,) = struct.unpack_from("<i", data, 4 + i * 4)
        if offset < 0 or offset + 22 > len(data):
            continue
        name = _decode_name(data[offset: offset + 8])
        width, height = struct.unpack_from("<hh", data, offset + 12)
        (patch_count,) = struct.unpack_from("<h", data, offset + 20)
        patches = []
        for p in range(patch_count):
            p_off = offset + 22 + p * 10
            if p_off + 10 > len(data):
                break
            origin_x, origin_y, patch_idx = struct.unpack_from("<hhh", data, p_off)
            patch_name = pnames[patch_idx] if 0 <= patch_idx < len(pnames) else f"UNKNOWN_{patch_idx}"
            patches.append(Patch(name=patch_name, x_origin=origin_x, y_origin=origin_y))
        textures[name] = Texture(name=name, width=width, height=height, patches=patches)
    return textures


def parse_animated(data: bytes) -> list[AnimatedEntry]:
    entries = []
    count = len(data) // 23
    for i in range(count):
        off = i * 23
        (kind_byte,) = struct.unpack_from("<B", data, off)
        if kind_byte == 0xFF:
            break
        last = _decode_name(data[off + 1: off + 10])
        first = _decode_name(data[off + 10: off + 19])
        (speed,) = struct.unpack_from("<i", data, off + 19)
        kind = "texture" if kind_byte == 1 else "flat"
        entries.append(AnimatedEntry(kind=kind, first=first, last=last, speed=speed))
    return entries


def parse_switches(data: bytes) -> list[SwitchEntry]:
    entries = []
    count = len(data) // 20
    for i in range(count):
        off = i * 20
        off_tex = _decode_name(data[off: off + 9])
        on_tex = _decode_name(data[off + 9: off + 18])
        (game,) = struct.unpack_from("<H", data, off + 18)
        if game == 0:
            break
        entries.append(SwitchEntry(off_texture=off_tex, on_texture=on_tex, game=game))
    return entries


def _infer_flat_dims(size: int) -> tuple[int | None, int | None]:
    if size == 4096:
        return 64, 64  # the standard vanilla flat size
    root = math.isqrt(size)
    if root * root == size:
        return root, root
    return None, None


def build_flat_assets(flat_bytes: dict[str, bytes]) -> list[FlatAsset]:
    """One FlatAsset per flat, referencing (by sha256) the raw indexed-pixel
    bytes a caller is expected to persist to a sidecar file - see
    plan/data-model.md: flats are captured with their raw pixels, not just
    their name.
    """
    assets = []
    for name, data in flat_bytes.items():
        width, height = _infer_flat_dims(len(data))
        assets.append(FlatAsset(
            name=name, width=width, height=height,
            byte_length=len(data), sha256=hashlib.sha256(data).hexdigest(),
        ))
    return assets


def build_texture_set(
    texture1: bytes | None,
    texture2: bytes | None,
    pnames_data: bytes | None,
    flat_bytes: dict[str, bytes],
    playpal: bytes | None,
    colormap: bytes | None,
    animated: bytes | None,
    switches: bytes | None,
) -> TextureSet:
    pnames = parse_pnames(pnames_data) if pnames_data else []
    textures: dict[str, Texture] = {}
    if texture1:
        textures.update(parse_texture_lump(texture1, pnames))
    if texture2:
        textures.update(parse_texture_lump(texture2, pnames))
    return TextureSet(
        textures=textures,
        flats=build_flat_assets(flat_bytes),
        palette_count=(len(playpal) // 768) if playpal else 0,
        colormap_count=(len(colormap) // 256) if colormap else 0,
        playpal_sha256=hashlib.sha256(playpal).hexdigest() if playpal else None,
        colormap_sha256=hashlib.sha256(colormap).hexdigest() if colormap else None,
        animated=parse_animated(animated) if animated else [],
        switches=parse_switches(switches) if switches else [],
    )

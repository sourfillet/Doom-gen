"""Bundle custom flats / patches / composed textures into a generated PWAD.

A generated map is normally constrained to a target game's built-in assets
(generation/game_profile.py). This module is the escape hatch the user asked
for: drop custom graphics into the output WAD so a map can legitimately use
textures/flats the IWAD doesn't ship, and register those names on the game
profile so enforcement keeps them.

Binary layouts (PNAMES, TEXTURE1's maptexture_t/mappatch_t) mirror exactly
what wad_parser/texture_parser.py reads, so a bundle written here round-trips
back through the parser. Raw flat/patch pixel lumps are passed through as-is -
this module bundles graphics, it does not synthesize them.

A bundle can be assembled in code or loaded from a directory:

    <dir>/flats/NAME.lump          raw 64x64 (or NxN) indexed-pixel flat
    <dir>/patches/NAME.lump        raw Doom picture-format patch
    <dir>/textures.json            [{"name","width","height",
                                     "patches":[["PATCH",x,y], ...]}, ...]
"""
from __future__ import annotations

import json
import operator
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

from wad_parser.wad_reader import MAP_MARKER_RE, BINARY_MAP_LUMPS, GL_MAP_LUMPS
from wad_writer.assembler import _name8

_SYSTEM_LUMPS = {'PNAMES', 'TEXTURE1', 'TEXTURE2', 'PLAYPAL', 'COLORMAP', 'ANIMATED',
    'SWITCHES', 'GENMIDI', 'DMXGUS', 'DEHACKED', 'MAPINFO', 'ZMAPINFO', 'UMAPINFO',
    'EMAPINFO', 'GAMEINFO', 'SNDINFO', 'DECORATE', 'ZSCRIPT', 'KEYCONF', 'LOADACS',
    'GLDEFS', 'ANIMDEFS', 'TEXTMAP', 'ENDMAP', 'DIALOGUE', 'GL_LEVEL'}
_MARKER = re.compile(r'^(?:F{1,2}|P{1,2}|S{1,2}|[FPS][0-9]|TX|HI|C|VX|A|AC)_(?:START|END)$')


def _asset_name(name, *, lump=False):
    if not isinstance(name, str) or not name or name == '-':
        raise ValueError('asset name must be a nonempty Doom name')
    _name8(name)  # Strict ASCII, eight bytes, no embedded NUL.
    up = name.upper()
    if lump and (up in _SYSTEM_LUMPS | BINARY_MAP_LUMPS | GL_MAP_LUMPS
                 or MAP_MARKER_RE.fullmatch(up) or _MARKER.fullmatch(up)
                 or (up.startswith(('GL_', 'GX_')) and MAP_MARKER_RE.fullmatch(up[3:]))):
        raise ValueError(f'raw asset name {name!r} is reserved for engine/map/namespace data')
    return up


def _short(value, label, *, positive=False):
    if isinstance(value, bool):
        raise ValueError(f'{label} must be an integer')
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f'{label} must be an integer') from exc
    if not (1 if positive else -32768) <= value <= 32767:
        raise ValueError(f'{label} is outside the Doom signed 16-bit range')
    return value


@dataclass
class EmbeddedTexture:
    name: str
    width: int
    height: int
    patches: list[tuple[str, int, int]]  # (patch_name, x_origin, y_origin)


@dataclass
class AssetBundle:
    flats: dict[str, bytes] = field(default_factory=dict)      # name -> flat lump
    patches: dict[str, bytes] = field(default_factory=dict)    # name -> patch lump
    textures: list[EmbeddedTexture] = field(default_factory=list)
    base_tables: dict[str, bytes] = field(default_factory=dict, repr=False)
    base_flat_names: set[str] = field(default_factory=set, repr=False)

    def with_base_wad(self, path: Path | str) -> "AssetBundle":
        """Bind complete target texture tables, without copying base images."""
        from copy import deepcopy
        from wad_parser.wad_reader import WadFile
        wad = WadFile(str(path))
        result = deepcopy(self)
        result.base_tables = {name: wad.get_all_lumps(name)[-1] for name in
                              ('PNAMES', 'TEXTURE1', 'TEXTURE2') if wad.get_all_lumps(name)}
        result.base_flat_names = set(wad.find_flat_names())
        if not {'PNAMES', 'TEXTURE1'} <= result.base_tables.keys():
            raise ValueError('base WAD needs PNAMES and TEXTURE1 tables')
        result.validate()
        return result

    def require_target(self, target_port):
        self.validate()
        if target_port == 'vanilla' and not self.is_empty:
            if not self.base_tables:
                raise ValueError('vanilla bundled assets need base texture tables; use --game IWAD or with_base_wad()')
            base_patches = set(self._base_patch_names())
            if any(name.upper() in base_patches | self.base_flat_names for name in self.flats):
                raise ValueError('vanilla custom flats must use new names, preserving base patches and animation ranges')
            if any(name.upper() in self.base_flat_names for name in self.patches):
                raise ValueError('vanilla patch lumps must not replace base flats')
            if any(len(data) != 4096 for data in self.flats.values()):
                raise ValueError('vanilla flats must contain exactly 4096 indexed pixels')

    def texture_names(self) -> set[str]:
        return {t.name.upper() for t in self.textures}

    def flat_names(self) -> set[str]:
        return {n.upper() for n in self.flats}

    @property
    def is_empty(self) -> bool:
        return not (self.flats or self.patches or self.textures)

    def validate(self):
        base_patches: set[str] | None = None
        if self.base_tables:
            if not {'PNAMES', 'TEXTURE1'} <= self.base_tables.keys():
                raise ValueError('base texture tables need PNAMES and TEXTURE1')
            self._base_records()
            base_patches = set(self._base_patch_names())
        seen = set()
        for name in [*self.flats, *self.patches]:
            up = _asset_name(name, lump=True)
            if up in seen:
                raise ValueError(f'duplicate encoded raw asset name: {up}')
            seen.add(up)
        available_patches = ({p.upper() for p in self.patches} | base_patches) if base_patches is not None else None
        textures = set()
        for texture in self.textures:
            up = _asset_name(texture.name)
            if up in textures:
                raise ValueError(f'duplicate encoded texture name: {up}')
            textures.add(up)
            _short(texture.width, f'texture {up} width', positive=True)
            _short(texture.height, f'texture {up} height', positive=True)
            if not 1 <= len(texture.patches) <= 32767:
                raise ValueError(f'texture {up} needs 1..32767 patches')
            for name, x, y in texture.patches:
                patch_up = _asset_name(name, lump=True)
                if available_patches is not None and patch_up not in available_patches:
                    raise ValueError(f'texture {up} references unknown patch {patch_up!r}')
                _short(x, f'texture {up} patch x offset')
                _short(y, f'texture {up} patch y offset')

    def _base_patch_names(self):
        data = self.base_tables.get('PNAMES')
        if data is None:
            return []
        if len(data) < 4:
            raise ValueError('truncated base PNAMES')
        count, = struct.unpack_from('<i', data)
        if count < 0 or 4 + 8 * count > len(data):
            raise ValueError('invalid base PNAMES count')
        try:
            return [data[4 + 8*i:12 + 8*i].split(b'\x00', 1)[0].decode('ascii').upper() for i in range(count)]
        except UnicodeDecodeError as exc:
            raise ValueError('base patch names must be ASCII') from exc

    def _base_records(self):
        names, records = self._base_patch_names(), []
        for name in ('TEXTURE1', 'TEXTURE2'):
            data = self.base_tables.get(name)
            if data is None:
                continue
            if len(data) < 4:
                raise ValueError(f'truncated base {name}')
            count, = struct.unpack_from('<i', data)
            if count < 0 or 4 + 4 * count > len(data):
                raise ValueError(f'invalid base {name} directory')
            for index in range(count):
                offset, = struct.unpack_from('<i', data, 4 + 4 * index)
                if offset < 4 + 4 * count or offset + 22 > len(data):
                    raise ValueError(f'invalid base {name} texture offset')
                patches, = struct.unpack_from('<h', data, offset + 20)
                end = offset + 22 + 10 * patches
                if patches < 0 or end > len(data):
                    raise ValueError(f'truncated base {name} patch list')
                for p in range(patches):
                    patch, = struct.unpack_from('<h', data, offset + 26 + 10 * p)
                    if not 0 <= patch < len(names):
                        raise ValueError(f'invalid base {name} patch index')
                records.append(data[offset:end])
        return records

    # -- PNAMES / TEXTURE1 encoders (inverse of texture_parser) --------------

    def _pnames_order(self) -> list[str]:
        """Every patch name TEXTURE1 will reference, plus any bundled patch
        lump - deduped, order stable."""
        # Preserve the complete prefix, including duplicate slots: existing
        # base texture records refer to these original numeric indices.
        order: list[str] = self._base_patch_names()
        seen: set[str] = set(order)
        for name in list(self.patches) + [p for t in self.textures for p, _, _ in t.patches]:
            up = name.upper()
            if up not in seen:
                seen.add(up)
                order.append(up)
        return order

    def _build_pnames(self, order: list[str]) -> bytes:
        return struct.pack("<i", len(order)) + b"".join(_name8(n) for n in order)

    def _build_texture1(self, order: list[str]) -> bytes:
        patch_index = {n: i for i, n in enumerate(order)}
        records = self._base_records()
        for t in self.textures:
            body = (_name8(t.name)
                    + struct.pack("<i", 0)            # masked (unused; 0 = opaque)
                    + struct.pack("<hh", t.width, t.height)
                    + struct.pack("<i", 0)            # columndirectory (obsolete)
                    + struct.pack("<h", len(t.patches)))
            for pname, ox, oy in t.patches:
                body += struct.pack("<hhhhh", ox, oy,
                                    patch_index[pname.upper()], 1, 0)  # stepdir=1, colormap=0
            target_name = t.name.encode("ascii").upper()
            matches = [
                i for i, record in enumerate(records)
                if record[:8].split(b"\x00", 1)[0].upper() == target_name
            ]
            if matches:
                for index in matches:
                    records[index] = body
            else:
                records.append(body)
        n = len(records)
        header_len = 4 + n * 4
        offsets = []
        cursor = header_len
        for body in records:
            offsets.append(cursor)
            cursor += len(body)
        return (struct.pack("<i", n)
                + b"".join(struct.pack("<i", o) for o in offsets)
                + b"".join(records))

    def build_lumps(self) -> list[tuple[str, bytes]]:
        """Directory entries to append to a PWAD. Global asset lumps are
        found by name anywhere in the directory, so order among them and
        relative to maps does not matter to ZDoom-family ports."""
        self.validate()
        lumps: list[tuple[str, bytes]] = []
        if self.textures:
            order = self._pnames_order()
            if len(order) > 32768:
                raise ValueError('patch indices exceed the Doom signed 16-bit range')
            lumps.append(("PNAMES", self._build_pnames(order)))
            lumps.append(("TEXTURE1", self._build_texture1(order)))
            if self.base_tables:
                # Base TEXTURE2 is merged into TEXTURE1 in original order.
                # Shadow the IWAD table so it cannot be loaded a second time.
                lumps.append(('TEXTURE2', struct.pack('<i', 0)))
        if self.patches:
            lumps.append(("P_START", b""))
            lumps += [(name, data) for name, data in self.patches.items()]
            lumps.append(("P_END", b""))
        if self.flats:
            # Modern namespace pair plus the classic end marker: vanilla
            # retains the IWAD's F_START and extends its flat range. A new
            # F_START would hide the IWAD's original flat indices.
            lumps.append(("FF_START", b""))
            lumps += [(name, data) for name, data in self.flats.items()]
            lumps.append(("FF_END", b""))
            lumps.append(("F_END", b""))
        return lumps

    @classmethod
    def from_dir(cls, path: Path | str) -> "AssetBundle":
        root = Path(path)
        bundle = cls()
        flats_dir = root / "flats"
        if flats_dir.is_dir():
            for f in sorted(flats_dir.glob("*.lump")):
                name = _asset_name(f.stem, lump=True)
                if name in bundle.flats:
                    raise ValueError(f'duplicate encoded flat name: {name}')
                bundle.flats[name] = f.read_bytes()
        patches_dir = root / "patches"
        if patches_dir.is_dir():
            for f in sorted(patches_dir.glob("*.lump")):
                name = _asset_name(f.stem, lump=True)
                if name in bundle.patches:
                    raise ValueError(f'duplicate encoded patch name: {name}')
                bundle.patches[name] = f.read_bytes()
        tex_json = root / "textures.json"
        if tex_json.is_file():
            for t in json.loads(tex_json.read_text(encoding="utf-8")):
                bundle.textures.append(EmbeddedTexture(
                    name=t["name"], width=t["width"], height=t["height"],
                    patches=[tuple(p) for p in t.get("patches", [])]))
        bundle.validate()
        return bundle

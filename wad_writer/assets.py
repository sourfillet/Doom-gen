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
import struct
from dataclasses import dataclass, field
from pathlib import Path


def _name8(name: str) -> bytes:
    return name.upper().encode("ascii", errors="replace").ljust(8, b"\x00")[:8]


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

    def texture_names(self) -> set[str]:
        return {t.name.upper() for t in self.textures}

    def flat_names(self) -> set[str]:
        return {n.upper() for n in self.flats}

    @property
    def is_empty(self) -> bool:
        return not (self.flats or self.patches or self.textures)

    # -- PNAMES / TEXTURE1 encoders (inverse of texture_parser) --------------

    def _pnames_order(self) -> list[str]:
        """Every patch name TEXTURE1 will reference, plus any bundled patch
        lump - deduped, order stable."""
        order: list[str] = []
        seen: set[str] = set()
        for name in list(self.patches) + [p for t in self.textures for p, _, _ in t.patches]:
            up = name.upper()
            if up not in seen:
                seen.add(up)
                order.append(up)
        return order

    def _build_pnames(self, order: list[str]) -> bytes:
        return struct.pack("<i", len(order)) + b"".join(_name8(n) for n in order)

    def _build_texture1(self, order: list[str]) -> bytes:
        index = {n: i for i, n in enumerate(order)}
        n = len(self.textures)
        header_len = 4 + n * 4
        bodies, offsets = b"", []
        for t in self.textures:
            offsets.append(header_len + len(bodies))
            body = (_name8(t.name)
                    + struct.pack("<i", 0)            # masked (unused; 0 = opaque)
                    + struct.pack("<hh", t.width, t.height)
                    + struct.pack("<i", 0)            # columndirectory (obsolete)
                    + struct.pack("<h", len(t.patches)))
            for pname, ox, oy in t.patches:
                body += struct.pack("<hhhhh", ox, oy,
                                    index.get(pname.upper(), 0), 1, 0)  # stepdir=1, colormap=0
            bodies += body
        return (struct.pack("<i", n)
                + b"".join(struct.pack("<i", o) for o in offsets)
                + bodies)

    def build_lumps(self) -> list[tuple[str, bytes]]:
        """Directory entries to append to a PWAD. Global asset lumps are
        found by name anywhere in the directory, so order among them and
        relative to maps does not matter to ZDoom-family ports."""
        lumps: list[tuple[str, bytes]] = []
        if self.patches or self.textures:
            order = self._pnames_order()
            lumps.append(("PNAMES", self._build_pnames(order)))
            lumps.append(("TEXTURE1", self._build_texture1(order)))
            if self.patches:
                lumps.append(("P_START", b""))
                lumps += [(name, data) for name, data in self.patches.items()]
                lumps.append(("P_END", b""))
        if self.flats:
            lumps.append(("F_START", b""))
            lumps += [(name, data) for name, data in self.flats.items()]
            lumps.append(("F_END", b""))
        return lumps

    @classmethod
    def from_dir(cls, path: Path | str) -> "AssetBundle":
        root = Path(path)
        bundle = cls()
        flats_dir = root / "flats"
        if flats_dir.is_dir():
            for f in sorted(flats_dir.glob("*.lump")):
                bundle.flats[f.stem.upper()] = f.read_bytes()
        patches_dir = root / "patches"
        if patches_dir.is_dir():
            for f in sorted(patches_dir.glob("*.lump")):
                bundle.patches[f.stem.upper()] = f.read_bytes()
        tex_json = root / "textures.json"
        if tex_json.is_file():
            for t in json.loads(tex_json.read_text()):
                bundle.textures.append(EmbeddedTexture(
                    name=t["name"], width=t["width"], height=t["height"],
                    patches=[tuple(p) for p in t.get("patches", [])]))
        return bundle

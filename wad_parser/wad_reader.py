"""Binary WAD I/O: header, lump directory, and grouping of lumps into maps
vs. WAD-wide global lumps.
"""
from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass, field

MAP_MARKER_RE = re.compile(r"^(MAP\d{2,3}|E\d[MT]\d\d?)$")

# Lumps that appear directly after a map marker, in the binary (Doom/Hexen) format.
BINARY_MAP_LUMPS = {
    "THINGS", "LINEDEFS", "SIDEDEFS", "VERTEXES", "SEGS",
    "SSECTORS", "NODES", "SECTORS", "REJECT", "BLOCKMAP",
    "BEHAVIOR", "SCRIPTS", "ZNODES",
}
GL_MAP_LUMPS = {"GL_VERT", "GL_SEGS", "GL_SSECT", "GL_NODES", "GL_PVS"}
# UDMF needs no lump whitelist: everything between a TEXTMAP and its ENDMAP
# belongs to the map regardless of name (see find_map_groups).

FLAT_MARKERS = [("F_START", "F_END"), ("FF_START", "FF_END")]
PRIMARY_FLAT_STARTS = frozenset({"F_START", "FF_START"})
PRIMARY_FLAT_ENDS = frozenset({"F_END", "FF_END"})
SUB_FLAT_STARTS = frozenset({"F1_START", "F2_START", "F3_START"})
SUB_FLAT_ENDS = frozenset({"F1_END", "F2_END", "F3_END"})
ALL_FLAT_MARKERS = PRIMARY_FLAT_STARTS | PRIMARY_FLAT_ENDS | SUB_FLAT_STARTS | SUB_FLAT_ENDS


@dataclass
class LumpEntry:
    name: str
    data: bytes


@dataclass
class MapLumpGroup:
    name: str
    format: str  # "doom" | "hexen" | "udmf"
    lumps: dict[str, bytes]
    gl_metadata: dict[str, str] = field(default_factory=dict)


class WadFile:
    """Parsed WAD: header info, the ordered lump list, and lumps grouped
    into per-map groups vs. everything else (global/texture-system lumps).
    """

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            data = f.read()
        self.raw = data
        self.hash = hashlib.sha256(data).hexdigest()
        self._parse(data)

    def _parse(self, data: bytes) -> None:
        if len(data) < 12:
            raise ValueError(f"{self.path}: too small to be a WAD")
        ident = data[0:4].decode("ascii", errors="replace")
        if ident not in ("IWAD", "PWAD"):
            raise ValueError(f"{self.path}: not a WAD (identification={ident!r})")
        numlumps, infotableofs = struct.unpack_from("<ii", data, 4)
        if numlumps < 0 or infotableofs < 0 or infotableofs + numlumps * 16 > len(data):
            raise ValueError(f"{self.path}: invalid lump directory bounds")
        self.identification = ident
        self.num_lumps = numlumps

        self.lumps: list[LumpEntry] = []
        for i in range(numlumps):
            entry_off = infotableofs + i * 16
            if entry_off + 16 > len(data):
                raise ValueError(f"{self.path}: lump directory extends past end of file")
            filepos, size = struct.unpack_from("<ii", data, entry_off)
            raw_name = data[entry_off + 8: entry_off + 16]
            name = raw_name.split(b"\x00", 1)[0].decode("ascii", errors="replace").upper()
            # A negative filepos would silently slice from the END of the
            # buffer in Python; treat any out-of-bounds entry as corrupt.
            if size < 0 or size > 0 and (filepos < 0 or filepos + size > len(data)):
                raise ValueError(
                    f"{self.path}: lump {name!r} points outside the file "
                    f"(filepos={filepos}, size={size})"
                )
            lump_data = data[filepos: filepos + size] if size > 0 else b""
            self.lumps.append(LumpEntry(name=name, data=lump_data))

    def lump_names(self) -> list[str]:
        return [l.name for l in self.lumps]

    def find_map_groups(self) -> list[MapLumpGroup]:
        """Walk the lump directory and group consecutive lumps following a
        map marker into one MapLumpGroup, detecting binary vs. UDMF format
        and Doom vs. Hexen binary sub-format (BEHAVIOR lump present).

        UDMF's rule differs from the binary formats: ANY lump immediately
        followed by TEXTMAP is a map marker (the marker name is free-form),
        and EVERY lump up to the closing ENDMAP belongs to the map, whatever
        its name - custom lumps in the block are preserved, not dropped.
        """
        groups: list[MapLumpGroup] = []
        i = 0
        n = len(self.lumps)
        while i < n:
            name = self.lumps[i].name
            if i + 1 < n and self.lumps[i + 1].name == "TEXTMAP":
                j = i + 1
                collected: dict[str, bytes] = {}
                while j < n and self.lumps[j].name != "ENDMAP":
                    collected[self.lumps[j].name] = self.lumps[j].data
                    j += 1
                groups.append(MapLumpGroup(name=name, format="udmf", lumps=collected))
                i = j + 1
            elif MAP_MARKER_RE.match(name) and i + 1 < n and self.lumps[i + 1].name in BINARY_MAP_LUMPS:
                j = i + 1
                collected = {}
                while j < n and self.lumps[j].name in BINARY_MAP_LUMPS:
                    collected[self.lumps[j].name] = self.lumps[j].data
                    j += 1
                fmt = "hexen" if "BEHAVIOR" in collected else "doom"
                groups.append(MapLumpGroup(name=name, format=fmt, lumps=collected))
                i = j
            elif name.startswith('GL_') and i + 1 < n and self.lumps[i + 1].name == 'GL_VERT':
                # Standalone GL blocks have their own marker, and can appear
                # after all the normal maps. Never treat them as extra maps.
                # https://glbsp.sourceforge.net/specs.php (Level Marker)
                metadata = dict(line.split('=', 1) for line in
                                self.lumps[i].data.decode('ascii', errors='replace').splitlines()
                                if '=' in line)
                level = metadata.get('LEVEL') if name == 'GL_LEVEL' else name[3:]
                j, collected = i + 1, {}
                while j < n and (self.lumps[j].name in GL_MAP_LUMPS
                                 or self.lumps[j].name.startswith('GX_')):
                    collected[self.lumps[j].name] = self.lumps[j].data
                    j += 1
                target = next((group for group in reversed(groups) if group.name == level), None)
                if target is not None:
                    # A later block replaces the whole family. Mixing a new
                    # partial tree with old GL leaves/nodes can hide corruption.
                    target.lumps = {key: value for key, value in target.lumps.items()
                                    if key not in GL_MAP_LUMPS and not key.startswith('GX_')}
                    target.lumps.update(collected)
                    target.gl_metadata = metadata
                i = j
            else:
                i += 1
        return groups

    def find_flats(self) -> dict[str, bytes]:
        """Raw indexed-pixel bytes for every flat between F_START/F_END (or
        FF_START/FF_END, mixed markers, or sub-markers), collected in WAD
        directory order so later definitions override earlier ones.
        """
        flats: dict[str, bytes] = {}
        in_primary = False
        in_sub = False
        n = len(self.lumps)
        for i, l in enumerate(self.lumps):
            name = l.name
            if not in_primary and not in_sub:
                if name in PRIMARY_FLAT_STARTS:
                    # Require a matching primary end marker somewhere in the remaining directory
                    if any(self.lumps[j].name in PRIMARY_FLAT_ENDS for j in range(i + 1, n)):
                        in_primary = True
                elif name in SUB_FLAT_STARTS:
                    # Require a matching sub or primary end marker later in the directory
                    if any(self.lumps[j].name in (SUB_FLAT_ENDS | PRIMARY_FLAT_ENDS) for j in range(i + 1, n)):
                        in_sub = True
            elif in_primary:
                if name in PRIMARY_FLAT_ENDS:
                    in_primary = False
                elif name not in ALL_FLAT_MARKERS and l.data:
                    flats[name] = l.data
            elif in_sub:
                if name in (PRIMARY_FLAT_ENDS | SUB_FLAT_ENDS):
                    in_sub = False
                elif name not in ALL_FLAT_MARKERS and l.data:
                    flats[name] = l.data
        return flats

    def find_flat_names(self) -> list[str]:
        return list(self.find_flats().keys())

    def get_lump(self, name: str) -> bytes | None:
        """Find the last lump matching name in the WAD directory, matching
        Doom's W_GetNumForName backward-scan search order where later lumps
        override earlier ones.
        """
        for l in reversed(self.lumps):
            if l.name == name:
                return l.data
        return None

    def get_all_lumps(self, name: str) -> list[bytes]:
        return [l.data for l in self.lumps if l.name == name]

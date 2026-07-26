"""Binary WAD I/O: header, lump directory, and grouping of lumps into maps
vs. WAD-wide global lumps.
"""
from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass

MAP_MARKER_RE = re.compile(r"^(MAP\d{2}|E\d[MT]\d)$")

# Lumps that appear directly after a map marker, in the binary (Doom/Hexen) format.
BINARY_MAP_LUMPS = {
    "THINGS", "LINEDEFS", "SIDEDEFS", "VERTEXES", "SEGS",
    "SSECTORS", "NODES", "SECTORS", "REJECT", "BLOCKMAP",
    "BEHAVIOR", "SCRIPTS",
}
# Lumps that appear directly after a map marker, in UDMF format.
UDMF_MAP_LUMPS = {"TEXTMAP", "ZNODES", "DIALOGUE", "ENDMAP", "BLOCKMAP", "REJECT"}

ALL_MAP_LUMPS = BINARY_MAP_LUMPS | UDMF_MAP_LUMPS

FLAT_MARKERS = [("F_START", "F_END"), ("FF_START", "FF_END")]


@dataclass
class LumpEntry:
    name: str
    data: bytes


@dataclass
class MapLumpGroup:
    name: str
    format: str  # "doom" | "hexen" | "udmf"
    lumps: dict[str, bytes]


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
        self.identification = ident
        self.num_lumps = numlumps

        self.lumps: list[LumpEntry] = []
        for i in range(numlumps):
            entry_off = infotableofs + i * 16
            filepos, size = struct.unpack_from("<ii", data, entry_off)
            raw_name = data[entry_off + 8: entry_off + 16]
            name = raw_name.split(b"\x00", 1)[0].decode("ascii", errors="replace").upper()
            lump_data = data[filepos: filepos + size] if size > 0 else b""
            self.lumps.append(LumpEntry(name=name, data=lump_data))

    def lump_names(self) -> list[str]:
        return [l.name for l in self.lumps]

    def find_map_groups(self) -> list[MapLumpGroup]:
        """Walk the lump directory and group consecutive lumps following a
        map marker into one MapLumpGroup, detecting binary vs. UDMF format
        and Doom vs. Hexen binary sub-format (BEHAVIOR lump present).
        """
        groups: list[MapLumpGroup] = []
        i = 0
        n = len(self.lumps)
        while i < n:
            name = self.lumps[i].name
            if MAP_MARKER_RE.match(name) and i + 1 < n and self.lumps[i + 1].name in ALL_MAP_LUMPS:
                j = i + 1
                collected: dict[str, bytes] = {}
                while j < n and self.lumps[j].name in ALL_MAP_LUMPS:
                    collected[self.lumps[j].name] = self.lumps[j].data
                    j += 1
                if "TEXTMAP" in collected:
                    fmt = "udmf"
                elif "BEHAVIOR" in collected:
                    fmt = "hexen"
                else:
                    fmt = "doom"
                groups.append(MapLumpGroup(name=name, format=fmt, lumps=collected))
                i = j
            else:
                i += 1
        return groups

    def find_flats(self) -> dict[str, bytes]:
        """Raw indexed-pixel bytes for every flat between F_START/F_END (or
        FF_START/FF_END) - see plan/data-model.md's requirement that flat
        pixel data be captured, not just the flat's name.
        """
        flats: dict[str, bytes] = {}
        lump_names = self.lump_names()
        for start_marker, end_marker in FLAT_MARKERS:
            try:
                start = lump_names.index(start_marker)
                end = lump_names.index(end_marker, start)
            except ValueError:
                continue
            for l in self.lumps[start + 1: end]:
                if l.name and l.data:  # skip sub-markers like F1_START
                    flats[l.name] = l.data
        return flats

    def find_flat_names(self) -> list[str]:
        return list(self.find_flats().keys())

    def get_lump(self, name: str) -> bytes | None:
        for l in self.lumps:
            if l.name == name:
                return l.data
        return None

    def get_all_lumps(self, name: str) -> list[bytes]:
        return [l.data for l in self.lumps if l.name == name]

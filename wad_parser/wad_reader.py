"""Binary WAD reader: parses the header and lump directory, exposes raw lump bytes."""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path

_MAP_MARKER = re.compile(r"^(E\dM\d|MAP\d{2})$")

_MAP_LUMP_NAMES: frozenset[str] = frozenset({
    "THINGS", "LINEDEFS", "SIDEDEFS", "VERTEXES",
    "SEGS", "SSECTORS", "NODES", "SECTORS",
    "REJECT", "BLOCKMAP",
    # Extended formats
    "BEHAVIOR", "SCRIPTS", "TEXTMAP", "ZNODES", "DIALOGUE",
})


@dataclass(frozen=True)
class LumpEntry:
    name:   str
    offset: int
    size:   int


class WadReader:
    """Reads a WAD file and exposes raw lump data by name."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._data = self.path.read_bytes()
        self._parse_header()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def wad_type(self) -> str:
        """'IWAD' or 'PWAD'."""
        return self._wad_type

    @property
    def lump_list(self) -> list[LumpEntry]:
        return list(self._lump_list)

    def has_lump(self, name: str) -> bool:
        return name.upper() in self._lumps

    def get_lump(self, name: str) -> bytes | None:
        """Return the first matching lump's raw bytes, or None."""
        entries = self._lumps.get(name.upper())
        if not entries:
            return None
        e = entries[0]
        if e.size == 0:
            return b""
        return self._data[e.offset : e.offset + e.size]

    def get_all_lumps(self, name: str) -> list[bytes]:
        """Return raw bytes for every lump with this name."""
        return [
            self._data[e.offset : e.offset + e.size]
            for e in self._lumps.get(name.upper(), [])
        ]

    def get_map_names(self) -> list[str]:
        """Return all map-marker lump names in WAD order."""
        return [e.name for e in self._lump_list if _MAP_MARKER.match(e.name) and e.size == 0]

    def get_map_lumps(self, map_name: str) -> dict[str, bytes]:
        """
        Return every lump belonging to *map_name*.

        Lumps are collected from the entry immediately after the map marker
        until we hit something that is not a known map-lump name (or the end
        of the directory).
        """
        name = map_name.upper()
        result: dict[str, bytes] = {}
        collecting = False

        for entry in self._lump_list:
            if entry.name == name:
                collecting = True
                continue
            if collecting:
                if entry.name in _MAP_LUMP_NAMES:
                    data = self._data[entry.offset : entry.offset + entry.size]
                    result[entry.name] = data
                else:
                    break  # hit next map marker or unrelated lump

        return result

    # ── Internal ──────────────────────────────────────────────────────────────

    def _parse_header(self) -> None:
        if len(self._data) < 12:
            raise ValueError(f"File too small to be a WAD: {self.path}")

        ident = self._data[:4].decode("ascii", errors="replace")
        if ident not in ("IWAD", "PWAD"):
            raise ValueError(f"Not a valid WAD file (bad magic '{ident}'): {self.path}")

        self._wad_type: str = ident
        num_lumps, dir_offset = struct.unpack_from("<ii", self._data, 4)
        self._parse_directory(num_lumps, dir_offset)

    def _parse_directory(self, num_lumps: int, dir_offset: int) -> None:
        self._lump_list: list[LumpEntry] = []
        self._lumps: dict[str, list[LumpEntry]] = {}

        required = dir_offset + num_lumps * 16
        if required > len(self._data):
            raise ValueError(
                f"WAD directory extends beyond file end "
                f"(need {required} bytes, have {len(self._data)})"
            )

        for i in range(num_lumps):
            base = dir_offset + i * 16
            filepos, size = struct.unpack_from("<ii", self._data, base)
            raw_name = self._data[base + 8 : base + 16]
            name = raw_name.rstrip(b"\x00").decode("ascii", errors="replace").upper()

            entry = LumpEntry(name=name, offset=filepos, size=size)
            self._lump_list.append(entry)
            self._lumps.setdefault(name, []).append(entry)

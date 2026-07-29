"""PWAD file assembly: map documents -> a playable .wad on disk.

Nodes are left empty (see assembler.py); ZDoom-family ports rebuild them
on load. If a `zdbsp` binary is available it can be run afterwards for
vanilla-port compatibility.
"""
from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path

from wad_parser.types import MapDocument
from wad_writer.assembler import MAP_LUMP_ORDER, assemble_map_lumps
from wad_writer.assets import AssetBundle


def write_wad(docs: list[MapDocument], path: Path | str,
              assets: AssetBundle | None = None) -> Path:
    """Writes one PWAD containing every given map, in order. If an
    AssetBundle is given, its custom flats/patches/textures (plus the
    PNAMES/TEXTURE1 lumps describing them) are appended so the map can use
    graphics beyond the target IWAD."""
    path = Path(path)
    lumps: list[tuple[str, bytes]] = []
    for doc in docs:
        assembled = assemble_map_lumps(doc)
        lumps.append((doc.meta.map_name, b""))
        for name in MAP_LUMP_ORDER:
            lumps.append((name, assembled[name]))
    if assets is not None and not assets.is_empty:
        lumps += assets.build_lumps()

    body = b""
    directory = b""
    offset = 12
    for name, data in lumps:
        body += data
        directory += struct.pack("<ii", offset, len(data))
        directory += name.upper().encode("ascii", errors="replace").ljust(8, b"\x00")[:8]
        offset += len(data)
    header = b"PWAD" + struct.pack("<ii", len(lumps), 12 + len(body))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + body + directory)
    return path


def build_nodes(wad_path: Path | str) -> bool:
    """Runs zdbsp in place if available; returns False (wad still playable
    in ZDoom-family ports) when the binary is missing."""
    zdbsp = shutil.which("zdbsp")
    if not zdbsp:
        return False
    wad_path = str(wad_path)
    subprocess.run([zdbsp, wad_path, "-o", wad_path], check=True, capture_output=True)
    return True

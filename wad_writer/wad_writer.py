"""PWAD file assembly: map documents -> a playable .wad on disk.

Nodes are left empty (see assembler.py); ZDoom-family ports rebuild them
on load. If a `zdbsp` binary is available it can be run afterwards for
vanilla-port compatibility.
"""
from __future__ import annotations

import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

from wad_parser.types import MapDocument
from wad_parser.wad_reader import MAP_MARKER_RE
from wad_writer.assembler import MAP_LUMP_ORDER, assemble_map_lumps, _name8
from wad_writer.assets import AssetBundle


def write_wad(docs: list[MapDocument], path: Path | str,
              assets: AssetBundle | None = None) -> Path:
    """Writes one PWAD containing every given map, in order. If an
    AssetBundle is given, its custom flats/patches/textures (plus the
    PNAMES/TEXTURE1 lumps describing them) are appended so the map can use
    graphics beyond the target IWAD."""
    path = Path(path)
    if not docs:
        raise ValueError('cannot write a map WAD with no maps')
    names = [doc.meta.map_name.upper() for doc in docs]
    if any(not MAP_MARKER_RE.fullmatch(name) for name in names):
        raise ValueError('binary Doom output requires a recognized MAPxx/EyMx map marker')
    if len(names) != len(set(names)):
        raise ValueError('duplicate map markers in output WAD')
    lumps: list[tuple[str, bytes]] = []
    for doc in docs:
        assembled = assemble_map_lumps(doc)
        lumps.append((doc.meta.map_name, b""))
        for name in MAP_LUMP_ORDER:
            lumps.append((name, assembled[name]))
    if assets is not None and not assets.is_empty:
        lumps += assets.build_lumps()

    return write_lumps(lumps, path)


def write_lumps(lumps: list[tuple[str, bytes]], path: Path | str) -> Path:
    """Atomically write ordered lumps, preserving compiled map bytes verbatim.

    Callers assembling maps from existing lumps must validate their structure
    and target-port compatibility before publishing the resulting WAD.
    """
    path = Path(path)

    dir_parts: list[bytes] = []
    offset = 12
    for name, data in lumps:
        size = len(data)
        if size > 0x7FFFFFFF or offset > 0x7FFFFFFF - size:
            raise ValueError("WAD exceeds signed 32-bit lump offset/size range")
        dir_parts.append(struct.pack("<ii", offset, size) + _name8(name))
        offset += size
    if len(lumps) > 0x7FFFFFFF:
        raise ValueError("too many lumps for WAD directory")
    header = b"PWAD" + struct.pack("<ii", len(lumps), offset)
    directory = b"".join(dir_parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.wad', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(header)
            for _, data in lumps:
                stream.write(data)
            stream.write(directory)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def build_nodes(wad_path: Path | str) -> bool:
    """Run zdbsp in place if available; return False when it is missing.

    Some ports can build missing nodes at runtime. That fallback is not an
    engine compatibility certificate; externally compiled nodes are preferred.
    """
    zdbsp = shutil.which("zdbsp")
    if not zdbsp:
        return False
    wad_path = Path(wad_path).resolve()
    with tempfile.TemporaryDirectory(dir=wad_path.parent, prefix='.nodes-') as directory:
        built = Path(directory) / 'built.wad'
        try:
            # Pruning can change the sector count. Build a correctly sized,
            # all-visible REJECT matrix for the resulting sectors as well.
            subprocess.run([zdbsp, str(wad_path), '-R', '-o', str(built)], check=True,
                           capture_output=True, timeout=60)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            detail = (getattr(exc, 'stderr', b'') or b'').decode(errors='replace')
            raise ValueError(f"node builder failed: {detail or str(exc)}") from exc
        if not built.is_file():
            raise ValueError('node builder produced no output WAD')
        built.replace(wad_path)
    return True

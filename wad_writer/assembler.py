"""MapDocument -> binary map lump bytes (Doom format), the inverse of
wad_parser.lump_parser.

Flag words are rebuilt from the typed flag fields PLUS each element's
preserved `unrecognized_flag_bits`, so parse -> assemble round-trips
byte-exactly even for maps carrying junk bits (e.g. the IWAD's E2M7).

BSP lumps (SEGS/SSECTORS/NODES/BLOCKMAP) are written EMPTY: they
are deterministic artifacts of the geometry, ZDoom-family ports rebuild
them on load, and a node builder (zdbsp) can fill them in later for
vanilla-port compatibility.
REJECT is a correctly sized all-visible matrix, avoiding unsafe reads by
engines that expect one sector-to-sector visibility bit per pair.
"""
from __future__ import annotations

import math
import operator
import struct

from wad_parser.lump_parser import _LINEDEF_FLAG_BITS, _THING_FLAG_BITS
from wad_parser.types import HEXEN_UDMF_NAMESPACES, MapDocument


def fit_coordinates(doc: MapDocument) -> str | None:
    """Translate XY into binary range, preserving all distances and heights."""
    points = doc.vertices + doc.things
    offsets = []
    snap_axes: set[str] = set()
    for axis in ('x', 'y'):
        values = [getattr(point, axis) for point in points]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
            raise ValueError(f"non-finite {axis} coordinate")
        rounded = [round(value) for value in values] or [0]
        low, high = min(rounded), max(rounded)
        if high - low > 65535:
            raise ValueError(f"map {axis} span exceeds Doom coordinate range")
        if low >= -32768 and high <= 32767:
            offsets.append(0)
            continue
        shift = round(-0.5 - (low + high) / 2)
        has_half = any((value - math.floor(value)) == 0.5 for value in values)
        if has_half and shift % 2 != 0:
            for candidate in (shift + 1, shift - 1):
                if low + candidate >= -32768 and high + candidate <= 32767:
                    shift = candidate
                    break
            else:
                snap_axes.add(axis)
        offsets.append(shift)
    dx, dy = offsets
    if not dx and not dy:
        return None
    for point in points:
        point.x = float(round(point.x) + dx) if 'x' in snap_axes else point.x + dx
        point.y = float(round(point.y) + dy) if 'y' in snap_axes else point.y + dy
    for sector in doc.sectors:
        for slope in (sector.floor_slope, sector.ceil_slope):
            if slope is not None:
                slope.d -= slope.a * dx + slope.b * dy
    # Reports can share the same point/list through gameplay, decoration and
    # interior metadata. Translate each object once, including tuple footprints.
    translated = {}
    originals = []
    def translate(point):
        identity = id(point)
        if identity not in translated:
            originals.append(point)  # Keep tuple identities alive until the pass ends.
            if isinstance(point, list):
                point[0] += dx
                point[1] += dy
                translated[identity] = point
            else:
                translated[identity] = (point[0] + dx, point[1] + dy)
                translated[id(translated[identity])] = translated[identity]
        return translated[identity]
    extra = doc.extra or {}
    seeds = extra.get('sector_seed_points', {})
    for sid, point in seeds.items():
        seeds[sid] = translate(point)
    gameplay = extra.get('gameplay', {})
    decoration = extra.get('decoration', {}).get('gameplay', {})
    for details in (extra.get('interior_details', {}), extra.get('terrace_details', {}),
                    gameplay.get('interiors', {}), decoration.get('interiors', {})):
        for column in details.get('columns', []):
            column['center'] = translate(column['center'])
        for terrace in details.get('terraces', []):
            terrace['center'] = translate(terrace['center'])
            footprint = terrace['footprint']
            for index, point in enumerate(footprint):
                footprint[index] = translate(point)
    region_geometry = None
    region_info = None
    if doc.derived is not None and doc.derived.sector_regions:
        region_geometry = [
            [[(x + dx, y + dy) for x, y in ring] for ring in rings]
            for rings in doc.derived.sector_regions
        ]
        region_info = doc.derived.bsp_geometry
    hexen_style = (
        doc.meta.format == "hexen"
        or str(extra.get("udmf_namespace") or "").lower() in HEXEN_UDMF_NAMESPACES
    )
    from wad_parser.map_extractor import compute_derived
    doc.bounds, doc.derived = compute_derived(
        doc.vertices, doc.sectors, doc.linedefs, doc.sidedefs, doc.things,
        hexen_style=hexen_style, region_geometry=region_geometry, region_info=region_info,
    )
    if doc.extra is None:
        doc.extra = {}
    previous = doc.extra.get('coordinate_translation', [0, 0])
    doc.extra['coordinate_translation'] = [previous[0] + dx, previous[1] + dy]
    return f"translated XY coordinates by ({dx}, {dy}) for Doom binary range"

# Vanilla lump order after the map marker.
MAP_LUMP_ORDER = [
    "THINGS", "LINEDEFS", "SIDEDEFS", "VERTEXES", "SEGS",
    "SSECTORS", "NODES", "SECTORS", "REJECT", "BLOCKMAP",
]


def _name8(name: str) -> bytes:
    try:
        raw = name.encode("ascii").upper()
    except UnicodeEncodeError as exc:
        raise ValueError(f"Doom asset name must be ASCII: {name!r}") from exc
    if len(raw) > 8 or b"\x00" in raw:
        raise ValueError(f"Doom asset name must fit in 8 bytes: {name!r}")
    return raw.ljust(8, b"\x00")


def _integer(value, label: str, low: int, high: int, *, snap=False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    if not isinstance(value, (int, float)):
        try:
            value = operator.index(value)
        except TypeError as exc:
            raise ValueError(f"{label} must be an integer, got {value!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite, got {value!r}")
    try:
        number = int(round(value)) if snap else operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be an integer, got {value!r}") from exc
    if not low <= number <= high:
        raise ValueError(f"{label} must be an integer in [{low}, {high}], got {value!r}")
    return number


def _signed16(value, label: str, *, snap=False) -> int:
    return _integer(value, label, -32768, 32767, snap=snap)


def _unsigned16(value, label: str) -> int:
    return _integer(value, label, 0, 65535)


def _encode_flags(flag_obj, bit_table, extra: dict) -> int:
    raw = extra.get("unrecognized_flag_bits", 0)
    for bit, attr in bit_table:
        if getattr(flag_obj, attr):
            raw |= bit
    return raw


def assemble_map_lumps(doc: MapDocument) -> dict[str, bytes]:
    """Returns {lump_name: bytes} for one map. Doom binary format only -
    Hexen/UDMF output is a later phase (plan roadmap Phase 8)."""
    if doc.meta.format != "doom":
        raise ValueError(f"only doom-format output is supported, got {doc.meta.format!r}")

    # Indices refer to positions in these lists. Refuse a document whose IDs
    # would point to different elements once serialized, or use the reserved
    # 0xffff sidedef sentinel as an actual side index.
    for name, elements, maximum in (("vertex", doc.vertices, 65536),
                                     ("linedef", doc.linedefs, 65535),
                                     ("sector", doc.sectors, 65536),
                                     ("sidedef", doc.sidedefs, 65535)):
        if len(elements) > maximum:
            raise ValueError(f"too many {name} elements for Doom format: {len(elements)}")
        for index, element in enumerate(elements):
            if element.id != index:
                raise ValueError(f"{name} {element.id} must have dense ID {index} for Doom output")

    snapped = [(_signed16(v.x, f"vertex {v.id} x", snap=True),
                _signed16(v.y, f"vertex {v.id} y", snap=True)) for v in doc.vertices]
    for ld in doc.linedefs:
        for field in ("v1", "v2"):
            _integer(getattr(ld, field), f"linedef {ld.id} {field}", 0, len(snapped) - 1)
        _integer(ld.right_sidedef, f"linedef {ld.id} right_sidedef", 0, len(doc.sidedefs) - 1)
        if ld.left_sidedef != -1:
            _integer(ld.left_sidedef, f"linedef {ld.id} left_sidedef", 0, len(doc.sidedefs) - 1)
        if snapped[ld.v1] == snapped[ld.v2]:
            raise ValueError(f"linedef {ld.id} collapses to zero length after integer rounding")
    for sd in doc.sidedefs:
        _integer(sd.sector, f"sidedef {sd.id} sector", 0, len(doc.sectors) - 1)

    things = b"".join(
        struct.pack("<hhhhH", _signed16(t.x, f"thing {t.id} x", snap=True),
                    _signed16(t.y, f"thing {t.id} y", snap=True),
                    _signed16(t.angle, f"thing {t.id} angle"),
                    _signed16(t.type, f"thing {t.id} type"),
                    _unsigned16(_encode_flags(t.flags, _THING_FLAG_BITS, t.extra), f"thing {t.id} flags"))
        for t in doc.things
    )
    linedefs = b"".join(
        struct.pack(
            "<HHHHHHH", ld.v1, ld.v2,
            _unsigned16(_encode_flags(ld.flags, _LINEDEF_FLAG_BITS, ld.extra), f"linedef {ld.id} flags"),
            _unsigned16(ld.special, f"linedef {ld.id} special"),
            _unsigned16(ld.tag, f"linedef {ld.id} tag"),
            ld.right_sidedef,
            ld.left_sidedef if ld.left_sidedef != -1 else 0xFFFF,
        )
        for ld in doc.linedefs
    )
    sidedefs = b"".join(
        struct.pack("<hh", _signed16(sd.x_offset, f"sidedef {sd.id} x_offset"),
                    _signed16(sd.y_offset, f"sidedef {sd.id} y_offset"))
        + _name8(sd.upper_tex) + _name8(sd.lower_tex) + _name8(sd.mid_tex)
        + struct.pack("<H", sd.sector)
        for sd in doc.sidedefs
    )
    vertexes = b"".join(
        struct.pack("<hh", x, y) for x, y in snapped
    )
    sectors = b"".join(
        struct.pack("<hh", _signed16(s.floor_height, f"sector {s.id} floor_height"),
                    _signed16(s.ceil_height, f"sector {s.id} ceil_height"))
        + _name8(s.floor_flat) + _name8(s.ceil_flat)
        + struct.pack("<hhh", _signed16(s.light_level, f"sector {s.id} light_level"),
                      _signed16(s.special, f"sector {s.id} special"),
                      _signed16(s.tag, f"sector {s.id} tag"))
        for s in doc.sectors
    )

    return {
        "THINGS": things,
        "LINEDEFS": linedefs,
        "SIDEDEFS": sidedefs,
        "VERTEXES": vertexes,
        "SEGS": b"",
        "SSECTORS": b"",
        "NODES": b"",
        "SECTORS": sectors,
        "REJECT": bytes((len(doc.sectors) ** 2 + 7) // 8),
        "BLOCKMAP": b"",
    }

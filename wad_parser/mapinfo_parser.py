"""Best-effort parser for MAPINFO/ZMAPINFO (brace, ZDoom/GZDoom-style) and
EMAPINFO (bracket-section, Eternity-style) lumps.

These formats vary a fair amount between ports and haven't had a single
formal grammar the way UDMF does, so this is a tolerant line-based parser
rather than a strict one: it recognizes the common `map MAPxx "Level Name"`
/ `[MAPxx]` headers and `key = value` (or bare `key value`) statement lines,
and keeps anything it doesn't have a typed field for in `MapInfoEntry.extra`
rather than dropping it.
"""
from __future__ import annotations

import re

from wad_parser.types import MapInfoEntry

_MAP_HEADER_RE = re.compile(r'^map\s+([A-Za-z0-9]+)\s*(?:"([^"]*)")?', re.IGNORECASE)
_SECTION_HEADER_RE = re.compile(r'^\[([A-Za-z0-9]+)\]$')
_KEY_VALUE_RE = re.compile(r'^([A-Za-z0-9_]+)\s*=?\s*(.*)$')

_FIELD_MAP = {
    "levelname": "level_name",
    "music": "music",
    "sky1": "sky",
    "cluster": "cluster",
    "next": "next_map",
}


def _parse_scalar(raw: str):
    raw = raw.strip().rstrip(";").strip()
    if raw.startswith('"'):
        end = raw.find('"', 1)
        return raw[1:end] if end != -1 else raw.strip('"')
    first = raw.split(",")[0].strip()
    if not first:
        return raw
    try:
        return int(first)
    except ValueError:
        try:
            return float(first)
        except ValueError:
            return first


def _build_entry(fields: dict) -> MapInfoEntry:
    entry = MapInfoEntry()
    for key, value in fields.items():
        attr = _FIELD_MAP.get(key)
        if attr:
            setattr(entry, attr, str(value))
        else:
            entry.extra[key] = value
    return entry


def parse_mapinfo(data: bytes) -> dict[str, MapInfoEntry]:
    """Returns {map_name (uppercased): MapInfoEntry}."""
    text = data.decode("utf-8", errors="replace")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    entries: dict[str, MapInfoEntry] = {}
    current_map: str | None = None
    current_fields: dict = {}

    def flush():
        nonlocal current_map, current_fields
        if current_map:
            entries[current_map] = _build_entry(current_fields)
        current_map, current_fields = None, {}

    for raw_line in text.splitlines():
        line = raw_line.split("//", 1)[0].strip()
        if not line or line in ("{", "}"):
            continue

        m = _SECTION_HEADER_RE.match(line)
        if m:
            flush()
            current_map = m.group(1).upper()
            continue

        m = _MAP_HEADER_RE.match(line)
        if m:
            flush()
            current_map = m.group(1).upper()
            if m.group(2):
                current_fields["levelname"] = m.group(2)
            continue

        if current_map is None:
            continue  # global directives (gamedefaults, clearepisodes, ...) - not map-scoped

        m = _KEY_VALUE_RE.match(line)
        if m:
            key = m.group(1).lower()
            current_fields[key] = _parse_scalar(m.group(2))

    flush()
    return entries

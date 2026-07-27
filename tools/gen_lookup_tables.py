"""Regenerates wad_parser/lookup_tables.py from authoritative online sources.

Sources (fetch the raw text of each into --cache-dir first, or let this
script download them):

  thing_types.wiki     https://doomwiki.org/w/index.php?title=Thing_types&action=raw
  linedef_type.wiki    https://doomwiki.org/w/index.php?title=Linedef_type&action=raw
  zspecial.acs         https://raw.githubusercontent.com/ZDoom/acc/master/zspecial.acs

doomwiki.org is the reference for vanilla/Boom linedef types and DoomEd
thing numbers; zspecial.acs is the ZDoom team's own machine-readable
action-special list (the ACS compiler's definition file), which covers the
Hexen numbering plus ZDoom extensions.

Usage:
    python tools/gen_lookup_tables.py --cache-dir /path/to/downloads \
        --out wad_parser/lookup_tables.py
"""
from __future__ import annotations

import argparse
import datetime
import re
import urllib.request
from pathlib import Path

SOURCES = {
    "thing_types.wiki": "https://doomwiki.org/w/index.php?title=Thing_types&action=raw",
    "linedef_type.wiki": "https://doomwiki.org/w/index.php?title=Linedef_type&action=raw",
    "zspecial.acs": "https://raw.githubusercontent.com/ZDoom/acc/master/zspecial.acs",
}

# ---------------------------------------------------------------------------
# wikitext helpers
# ---------------------------------------------------------------------------

_LINK_RE = re.compile(r"\[\[(?:[^]|]*\|)?([^]|]*)\]\]")
_SUP_RE = re.compile(r"<sup>.*?</sup>")
_TEMPLATE_RE = re.compile(r"\{\{[^}]*\}\}")


def _clean_cell(raw: str) -> str:
    """Strips wikitext markup from one table cell, keeping display text."""
    # cell may carry attributes: | align="right" | 68
    if "|" in raw and re.match(r'\s*[a-z]+="', raw):
        raw = raw.split("|", 1)[1]
    text = _SUP_RE.sub("", raw)
    text = _LINK_RE.sub(r"\1", text)
    text = _TEMPLATE_RE.sub("", text)
    text = text.replace("''", "").replace("&amp;", "&").replace("&nbsp;", " ")
    text = text.replace("<nowiki>", "").replace("</nowiki>", "")
    return text.strip()


def _iter_tables(wikitext: str):
    """Yields (preceding_h2_heading, header_cells, rows) per wikitable."""
    heading = ""
    lines = wikitext.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^==([^=].*?)==\s*$", line)
        if m:
            heading = m.group(1).strip()
            i += 1
            continue
        if line.startswith("{|"):
            headers: list[str] = []
            rows: list[list[str]] = []
            row: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith("|}"):
                ln = lines[i]
                if ln.startswith("!"):
                    # header line; may pack several with !!
                    for part in ln.lstrip("!").split("!!"):
                        headers.append(_clean_cell(part))
                elif ln.startswith("|-"):
                    if row:
                        rows.append(row)
                    row = []
                elif ln.startswith("|"):
                    for part in ln[1:].split("||"):
                        row.append(_clean_cell(part))
                i += 1
            if row:
                rows.append(row)
            yield heading, headers, rows
        i += 1


# ---------------------------------------------------------------------------
# DOOMED_TABLE from doomwiki Thing_types
# ---------------------------------------------------------------------------

_SECTION_CATEGORY = {
    "Monsters": "monster",
    "Weapons": "weapon",
    "Ammunition": "ammo",
    "Artifact items": "item",
    "Powerups": "item",
    "Keys": "key",
    "Obstacles": "obstacle",
    "Decorations": "decoration",
    "Other": "special",
}

# Structural things keep their established schema names/categories.
_THING_OVERRIDES = {
    1: ("Player1Start", "player-start"),
    2: ("Player2Start", "player-start"),
    3: ("Player3Start", "player-start"),
    4: ("Player4Start", "player-start"),
    11: ("DeathmatchStart", "deathmatch-start"),
    14: ("TeleportDest", "teleport-dest"),
}


def build_doomed_table(wikitext: str) -> dict[int, tuple[str, str]]:
    doom_section = wikitext.split("== Heretic ==", 1)[0]
    # h3 subsections carry the category; rewrite them to h2 so _iter_tables
    # (which tracks h2) sees them.
    doom_section = re.sub(r"^===(.+?)===\s*$", r"==\1==", doom_section, flags=re.M)
    table: dict[int, tuple[str, str]] = {}
    for heading, headers, rows in _iter_tables(doom_section):
        category = _SECTION_CATEGORY.get(_clean_cell(heading))
        if category is None or "Decimal" not in headers:
            continue
        dec_i = headers.index("Decimal")
        desc_i = headers.index("Description")
        for row in rows:
            if len(row) <= max(dec_i, desc_i):
                continue
            try:
                num = int(row[dec_i])
            except ValueError:
                continue
            desc = row[desc_i]
            if not desc:
                continue
            table[num] = _THING_OVERRIDES.get(num, (desc, category))
    return dict(sorted(table.items()))


# ---------------------------------------------------------------------------
# LINEDEF_SPECIAL_TABLE from doomwiki Linedef_type
# ---------------------------------------------------------------------------

_TRIG_TO_TYPE = {
    "W": "walk", "S": "switch", "G": "gun", "P": "push", "D": "push",
}

_SECTION_PREFIX = {
    "Doors": "Door",
    "Floors": "Floor",
    "Ceilings": "Ceiling",
    "Platforms (lifts)": "Lift",
    "Crusher ceilings": "Crusher",
    "Stair builders": "Stairs",
    "Lighting": "Light",
    "Exits": "Exit",
    "Teleporters": "Teleport",
    "Donuts": "Donut",
    "Elevators": "Elevator",
}

# The one descriptive column each category table uses, in preference order.
_DESCRIPTOR_COLUMNS = ["Function", "Target", "Dest", "Type"]

# Only these Origin values share the vanilla-Doom special numbering; the
# page's Heretic/Hexen/Strife and per-port (SMMU/Legacy/ZDoom/Eternity/EDGE)
# tables reuse the same numbers with conflicting meanings and must not be
# merged into this table.
_DOOM_NAMESPACE_ORIGINS = {
    "Doom", "Doom II", "Ultimate Doom", "Final Doom",
    "Boom", "MBF", "MBF21", "ID24",
}

_DESC_TRIG_RE = re.compile(r"^(W1|WR|S1|SR|G1|GR|P1|PR|D1|DR)\s+")


def _trigger_type(trig: str) -> str:
    trig = trig.strip()
    if not trig or trig in ("-", "--", "—", "–", "N/A"):
        return "passive"
    return _TRIG_TO_TYPE.get(trig[0].upper(), "unknown")


def _cell(row: list[str], idx: int | None) -> str:
    return row[idx].strip() if idx is not None and len(row) > idx else ""


def build_linedef_table(wikitext: str) -> dict[int, tuple[str, str]]:
    table: dict[int, tuple[str, str]] = {0: ("None", "none")}
    # The per-category reference tables (with a Trig column) live from
    # ==Doors== onward.
    _, _, categorized = wikitext.partition("==Doors==")
    categorized = "==Doors==\n" + categorized

    for heading, headers, rows in _iter_tables(categorized):
        if "#" not in headers or "Trig" not in headers:
            continue
        prefix = _SECTION_PREFIX.get(heading, heading.rstrip("s"))
        num_i = headers.index("#")
        trig_i = headers.index("Trig")
        desc_i = next((headers.index(c) for c in _DESCRIPTOR_COLUMNS if c in headers), None)
        dir_i = headers.index("Dir") if "Dir" in headers else None
        lock_i = headers.index("Lock") if "Lock" in headers else None
        spd_i = next((headers.index(c) for c in ("Spd", "Speed") if c in headers), None)
        silent_i = headers.index("Silent") if "Silent" in headers else None
        mon_i = headers.index("Mon") if "Mon" in headers else None
        for row in rows:
            try:
                num = int(_cell(row, num_i))
            except ValueError:
                continue  # generalized ranges ("6000H-7FFFH") and blanks
            trig = _cell(row, trig_i)
            direction = _cell(row, dir_i)
            descriptor = _cell(row, desc_i)
            body = " ".join(p for p in (direction, descriptor) if p and p not in ("--", "—"))
            qualifiers = [trig] if trig and trig not in ("-", "--", "—") else []
            lock = _cell(row, lock_i)
            if lock and lock.lower() not in ("no", "-", "--", "—", ""):
                lock = {"Yell": "Yellow"}.get(lock, lock)
                qualifiers.append(f"{lock} key")
            speed = _cell(row, spd_i)
            if speed.lower() in ("fast", "turbo"):
                qualifiers.append(speed.lower())
            if _cell(row, silent_i).lower() == "yes":
                qualifiers.append("silent")
            if _cell(row, mon_i).lower() == "yes":
                qualifiers.append("also monsters")
            suffix = f" ({', '.join(qualifiers)})" if qualifiers else ""
            name = f"{prefix}: {body}{suffix}" if body else f"{prefix}{suffix}"
            table.setdefault(num, (name, _trigger_type(trig)))

    # The Doom/Boom/MBF/MBF21/ID24 summary tables (Type/Origin/Description)
    # cover static specials (scrollers, transfers, friction/wind) that have
    # no Trig-column table; an embedded leading trigger code in the
    # description marks the few active ones not caught above.
    for heading, headers, rows in _iter_tables(wikitext):
        if headers[:2] != ["Type", "Origin"] or "Description" not in headers:
            continue
        desc_i = headers.index("Description")
        for row in rows:
            if len(row) <= desc_i or row[1] not in _DOOM_NAMESPACE_ORIGINS:
                continue
            try:
                num = int(row[0])
            except ValueError:
                continue
            desc = row[desc_i]
            m = _DESC_TRIG_RE.match(desc)
            ttype = _trigger_type(m.group(1)) if m else "passive"
            name = f"{row[1]}: {desc}"
            table.setdefault(num, (name, ttype))
    return dict(sorted(table.items()))


# ---------------------------------------------------------------------------
# HEXEN_SPECIAL_TABLE from zspecial.acs
# ---------------------------------------------------------------------------

_ZSPECIAL_RE = re.compile(r"^\s*(?://)?\s*(\d+):([A-Za-z0-9_]+)")


def build_hexen_table(acs_text: str) -> dict[int, tuple[str, str]]:
    table: dict[int, tuple[str, str]] = {0: ("None", "none")}
    for line in acs_text.splitlines():
        m = _ZSPECIAL_RE.match(line)
        if m:
            table.setdefault(int(m.group(1)), (m.group(2), "unknown"))
    return dict(sorted(table.items()))


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

HEADER = '''"""Lookup tables for DoomEd thing numbers and linedef/action specials.

GENERATED FILE - do not edit by hand; regenerate with
tools/gen_lookup_tables.py (see that file for the source URLs).

Generated {date} from:
  - doomwiki.org "Thing types" (DoomEd numbers, names, categories)
  - doomwiki.org "Linedef type" (vanilla + Boom/MBF linedef specials)
  - ZDoom acc zspecial.acs (Hexen/ZDoom action-special numbering)
"""

'''


def _emit(name: str, table: dict[int, tuple[str, str]]) -> str:
    lines = [f"{name}: dict[int, tuple[str, str]] = {{"]
    for num, (label, kind) in table.items():
        lines.append(f"    {num}: ({label!r}, {kind!r}),")
    lines.append("}\n")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    texts = {}
    for fname, url in SOURCES.items():
        path = args.cache_dir / fname
        if not path.exists():
            print(f"downloading {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "doom-gen table generator"})
            path.write_bytes(urllib.request.urlopen(req).read())
        texts[fname] = path.read_text(encoding="utf-8", errors="replace")

    doomed = build_doomed_table(texts["thing_types.wiki"])
    linedef = build_linedef_table(texts["linedef_type.wiki"])
    hexen = build_hexen_table(texts["zspecial.acs"])

    out = HEADER.format(date=datetime.date.today().isoformat())
    out += _emit("DOOMED_TABLE", doomed) + "\n\n"
    out += _emit("LINEDEF_SPECIAL_TABLE", linedef) + "\n\n"
    out += _emit("HEXEN_SPECIAL_TABLE", hexen)
    args.out.write_text(out)
    print(f"wrote {args.out}: {len(doomed)} things, {len(linedef)} linedef specials, "
          f"{len(hexen)} hexen/zdoom action specials")


if __name__ == "__main__":
    main()

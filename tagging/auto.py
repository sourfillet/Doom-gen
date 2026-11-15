"""Auto-derived FACT tags: objective, mechanically-checkable properties of
a map/WAD, derived from parsed data.

These are deliberately different from heuristic *suggestions* (style/theme
guesses, which stay human-confirmed): every rule here is a verifiable fact
with a documented basis, not a judgment call. They form a derived layer,
like the extracted map's derived data:

  - stored in separate .auto.json files, regenerated wholesale, never
    hand-edited;
  - user assignments always win on collision;
  - a user can veto any auto tag (a per-scope denial list) and the veto
    survives regeneration.

Rule bases:
  - deathmatch: >= 4 deathmatch starts - vanilla exits with an error below
    four (doomwiki "Deathmatch start").
  - secrets: sector special 9 (vanilla) or generalized bit 0x0080 (Boom,
    doomwiki "Sector"); the bit test only applies to Doom-format maps
    because Heretic/Hexen reuse overlapping special ranges.
  - feature tiers (vanilla/boom/mbf/mbf21): detected from the linedef
    specials and flag bits this parser already decodes from authoritative
    tables (see tools/gen_lookup_tables.py).
  - megawad: >= 15 maps; community convention for "large level collection",
    our threshold choice, documented here rather than claimed as a spec.
"""
from __future__ import annotations

from wad_parser.types import MapDocument
from wad_parser.wad_reader import WadFile

MEGAWAD_MIN_MAPS = 15
_BOSS_DOOMED_NUMS = {7, 16, 88}  # Spiderdemon, Cyberdemon, Romero's head
_MBF21_SPECIALS = {1024, 1025, 1026}
_MBF_SPECIALS = {271, 272}

# Registry descriptions written when an auto tag is first created; the
# "[auto]" prefix marks provenance in `tag-maps registry` output and gives
# the future text-encoder cold-start something accurate to embed.
AUTO_TAG_DESCRIPTIONS = {
    "single-player": "[auto] has a Player 1 start",
    "coop": "[auto] has Player 2-4 starts",
    "deathmatch": "[auto] has the >=4 deathmatch starts vanilla requires",
    "secrets": "[auto] has at least one secret sector",
    "teleporters": "[auto] has teleporter connectivity",
    "keys": "[auto] has key pickups (keyed progression)",
    "boss": "[auto] has a boss-class monster (Cyberdemon/Spiderdemon/boss brain)",
    "slopes": "[auto] uses sloped floors/ceilings or vertex heights",
    "vanilla": "[auto] uses only vanilla Doom linedef/sector specials",
    "boom": "[auto] uses Boom-extended specials",
    "mbf": "[auto] uses MBF features (friendly things / sky transfers)",
    "mbf21": "[auto] uses MBF21 features",
    "hexen-format": "[auto] Hexen binary map format",
    "udmf": "[auto] UDMF text map format",
    "iwad": "[auto] map came from an IWAD",
    "megawad": f"[auto] WAD with >= {MEGAWAD_MIN_MAPS} maps",
}


def _feature_tier(doc: MapDocument) -> str | None:
    """vanilla < boom < mbf < mbf21, for Doom-format maps only.

    Deliberately special-number-based, never flag-bit-based: old vanilla
    maps carry garbage in the high linedef/thing flag bits (the IWAD's own
    E2M7 has special-0 linedefs with 0xFC00 junk), so port-extension flag
    bits (pass_through, block_land_monsters, friendly, ...) are not
    trustworthy evidence of the map targeting that port.
    """
    if doc.meta.format != "doom":
        return None
    tier = "vanilla"
    has_unrecognized = False
    for ld in doc.linedefs:
        if ld.special in _MBF21_SPECIALS:
            return "mbf21"
        if ld.special in _MBF_SPECIALS:
            tier = "mbf"
        elif 141 < ld.special <= 269 or 0x2F80 <= ld.special <= 0x7FFF:
            if tier == "vanilla":
                tier = "boom"
        elif ld.special < 0 or ld.special > 141:
            has_unrecognized = True
    if any(s.special < 0 or s.special >= 32 for s in doc.sectors):
        if any(s.special < 0 or s.special > 0x0FFF for s in doc.sectors):
            has_unrecognized = True
        elif tier == "vanilla":
            tier = "boom"  # generalized sector-special bits (0x0020..0x0FFF) in use
    if has_unrecognized and tier == "vanilla":
        return None
    return tier


def derive_map_facts(doc: MapDocument) -> dict[str, float]:
    tags: dict[str, float] = {}
    thing_types = [t.type for t in doc.things]

    if 1 in thing_types:
        tags["single-player"] = 1.0
    if any(t in thing_types for t in (2, 3, 4)):
        tags["coop"] = 1.0
    if sum(1 for t in thing_types if t == 11) >= 4:
        tags["deathmatch"] = 1.0

    is_doom_format = doc.meta.format == "doom"
    udmf_ns = str((doc.extra or {}).get("udmf_namespace") or "").lower()
    is_doom_actors = is_doom_format or (doc.meta.format == "udmf" and udmf_ns not in {"hexen", "heretic", "strife"})
    if any(s.special == 9 or (is_doom_format and s.special >= 32 and (s.special & 0x0080)) for s in doc.sectors):
        tags["secrets"] = 1.0
    if any(e.get("kind") == "teleport" for e in doc.derived.sector_graph.get("edges", [])):
        tags["teleporters"] = 1.0
    if any(t.category == "key" for t in doc.things):
        tags["keys"] = 1.0
    if is_doom_actors and any(t.type in _BOSS_DOOMED_NUMS for t in doc.things):
        tags["boss"] = 1.0
    if any(s.floor_slope or s.ceil_slope for s in doc.sectors) or \
            any(v.z is not None for v in doc.vertices):
        tags["slopes"] = 1.0

    tier = _feature_tier(doc)
    if tier:
        tags[tier] = 1.0
    elif doc.meta.format == "hexen":
        tags["hexen-format"] = 1.0
    elif doc.meta.format == "udmf":
        tags["udmf"] = 1.0
    return tags


def derive_wad_facts(wad: WadFile, docs: list[MapDocument]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Returns (wad_tags, {map_name: map_tags}). A fact shared by every map
    is hoisted to the WAD level (matching the plan's inheritance model:
    'a WAD that's entirely deathmatch maps gets the tag once')."""
    per_map = {doc.meta.map_name: derive_map_facts(doc) for doc in docs}

    wad_tags: dict[str, float] = {}
    if wad.identification == "IWAD":
        wad_tags["iwad"] = 1.0
    if len(docs) >= MEGAWAD_MIN_MAPS:
        wad_tags["megawad"] = 1.0

    if per_map:
        shared = set.intersection(*(set(t) for t in per_map.values()))
        for tag in shared:
            wad_tags[tag] = 1.0
            for tags in per_map.values():
                del tags[tag]
    return wad_tags, per_map

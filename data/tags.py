"""
Tag taxonomy, encoding, and auto-inference for Doom maps.

Tags are multi-label strings drawn from a fixed vocabulary organized
into named categories.  They are stored in DoomMap.meta.tags as a flat
list of strings (e.g. ["doom2", "tech", "medium", "single-player"]).

Auto-inference is heuristic and intentionally conservative — it only adds
tags it is reasonably confident about.  Human review via tag-maps CLI is
the authoritative source.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wad_parser.types import DoomMap


# ─── Taxonomy ─────────────────────────────────────────────────────────────────

TAG_TAXONOMY: dict[str, list[str]] = {
    "game":       ["doom1", "doom2", "tnt", "plutonia", "heretic", "hexen", "freedoom"],
    "mode":       ["single-player", "cooperative", "deathmatch", "ctf"],
    "theme":      ["tech", "hell", "gothic", "wood", "outdoor", "cave", "city", "abstract", "snow", "hell-tech"],
    "size":       ["tiny", "small", "medium", "large", "huge"],
    "style":      ["linear", "open", "hub", "arena", "labyrinthine", "progression"],
    "difficulty": ["easy", "medium", "hard", "slaughter"],
    "era":        ["vanilla", "limit-removing", "boom", "mbf", "zdoom", "gzdoom"],
}

ALL_TAGS: list[str] = [tag for tags in TAG_TAXONOMY.values() for tag in tags]
TAG_TO_IDX: dict[str, int] = {tag: i for i, tag in enumerate(ALL_TAGS)}

# Reverse: tag → category
TAG_CATEGORY: dict[str, str] = {
    tag: category
    for category, tags in TAG_TAXONOMY.items()
    for tag in tags
}


def validate_tags(tags: list[str]) -> tuple[list[str], list[str]]:
    """Return (valid_tags, unknown_tags)."""
    valid   = [t for t in tags if t in TAG_TO_IDX]
    unknown = [t for t in tags if t not in TAG_TO_IDX]
    return valid, unknown


def encode_tags(tags: list[str]) -> list[int]:
    """Multi-hot encode a tag list as a binary vector over ALL_TAGS."""
    vec = [0] * len(ALL_TAGS)
    for tag in tags:
        idx = TAG_TO_IDX.get(tag)
        if idx is not None:
            vec[idx] = 1
    return vec


def decode_tags(vec: list[int]) -> list[str]:
    """Decode a multi-hot vector back to a list of tag strings."""
    return [ALL_TAGS[i] for i, v in enumerate(vec) if v]


def tags_by_category(tags: list[str]) -> dict[str, list[str]]:
    """Group a flat tag list by category for display."""
    result: dict[str, list[str]] = {cat: [] for cat in TAG_TAXONOMY}
    for tag in tags:
        cat = TAG_CATEGORY.get(tag)
        if cat:
            result[cat].append(tag)
    return result


# ─── Theme texture sets ───────────────────────────────────────────────────────
# Sampled from common IWAD textures; not exhaustive.

_TECH_TEXTURES: frozenset[str] = frozenset({
    "STARTAN1", "STARTAN2", "STARTAN3", "STARGR1", "STARGR2",
    "COMPSPAN", "COMPSTA1", "COMPSTA2", "COMPTALL",
    "COMPUTE1", "COMPUTE2", "COMPUTE3",
    "METAL", "METAL2", "METAL3", "METAL4", "METAL5", "METAL6", "METAL7",
    "SILVER1", "SILVER2", "SILVER3",
    "TEKWALL1", "TEKWALL2", "TEKWALL3", "TEKWALL4", "TEKWALL6",
    "PIPE1", "PIPE2", "PIPE4",
    "ZZWOLF1", "ZZWOLF2", "ZZWOLF3", "ZZWOLF4", "ZZWOLF5",
    "ZZWOLF6", "ZZWOLF7", "ZZWOLF9", "ZZWOLF11", "ZZWOLF12",
    "GRAY1", "GRAY2", "GRAY4", "GRAY5", "GRAY7",
    "BROWN1", "BROWN96", "BROWNGRN", "BRNBIGR", "BRNBIGL",
    "SUPPORT2", "SUPPORT3",
})

_HELL_TEXTURES: frozenset[str] = frozenset({
    "SP_HOT1", "SP_ROCK1", "SP_ROCK2",
    "MARBLE1", "MARBLE2", "MARBLE3",
    "SKIN2", "SKIN3", "SKIN4", "SKIN5",
    "SKINFACE", "SKINSYMB", "SKINTEK1", "SKINTEK2",
    "FIREBLU1", "FIREBLU2", "FIRELAV3", "FIRELAVA", "FIREWALL",
    "REDWALL1",
    "BFALL1", "BFALL2", "BFALL3", "BFALL4",
    "SFALL1", "SFALL2", "SFALL3", "SFALL4",
    "BLOOD1", "BLOOD2", "BLOOD3",
    "NUKE24", "NUKAGE1", "NUKAGE2", "NUKAGE3",
    "SKULWAL3", "SKULWALL",
    "GSTONE1", "GSTONE2",
    "TANROCK1", "TANROCK2", "TANROCK3", "TANROCK4", "TANROCK5",
    "ROCKRED1", "ROCKRED2", "ROCKRED3",
})

_GOTHIC_TEXTURES: frozenset[str] = frozenset({
    "MARBLE1", "MARBLE2", "MARBLE3",
    "GSTONE1", "GSTONE2",
    "STONE", "STONE2", "STONE3", "STONE4", "STONE5", "STONE6",
    "STONGARG", "STONPOIS",
    "MARBFACE", "MARBGRAY",
    "MIDSPACE", "MIDVINE1", "MIDVINE2",
    "WOODMET1", "WOODMET2", "WOODMET3", "WOODMET4",
})

_WOOD_TEXTURES: frozenset[str] = frozenset({
    "WOOD1", "WOOD3", "WOOD4", "WOOD5",
    "WOODMET1", "WOODMET2", "WOODMET3", "WOODMET4",
    "DOORSTOP",
})

_OUTDOOR_FLATS: frozenset[str] = frozenset({
    "GRASS1", "GRASS2", "RROCK01", "RROCK02", "RROCK03", "RROCK04",
    "RROCK05", "RROCK06", "RROCK07", "RROCK08",
    "RROCK09", "RROCK10", "RROCK11", "RROCK12",
    "RROCK13", "RROCK14", "RROCK15", "RROCK16",
    "RROCK17", "RROCK18", "RROCK19", "RROCK20",
    "FLAT10", "FLAT5_1", "FLAT5_2",
    "F_SKY1",  # sky flat — definitive outdoor marker
})

_CITY_TEXTURES: frozenset[str] = frozenset({
    "BRICK1", "BRICK2", "BRICK3", "BRICK4", "BRICK5", "BRICK6",
    "BRICK7", "BRICK8", "BRICK9", "BRICK10", "BRICK11", "BRICK12",
    "BRICKLIT",
    "CEMENT1", "CEMENT2", "CEMENT3", "CEMENT4", "CEMENT5", "CEMENT6",
    "CEMENT7", "CEMENT8", "CEMENT9",
    "CRACKLE2", "CRACKLE4",
})

_SNOW_TEXTURES: frozenset[str] = frozenset({
    "SNOW1", "SNOW2", "SNOW3", "SNOW4", "SNOW5", "SNOW6",
    "ICE1", "ICE2",
    "SKSNOW1",
})


# ─── Auto-inference ──────────────────────────────────────────────────────────

def infer_tags(doom_map: DoomMap) -> list[str]:
    """
    Heuristically infer tags from parsed map data.

    Returns only tags that can be determined with reasonable confidence.
    Does not infer 'game' (requires knowledge of source IWAD) or 'era'
    (requires knowledge of features used).
    """
    tags: list[str] = []

    things = doom_map.things
    thing_types  = {t.type for t in things}
    thing_cats   = [t.category for t in things]
    monster_count = thing_cats.count("monster")

    # ── Mode ─────────────────────────────────────────────────────────────────
    has_dm_start    = 11 in thing_types
    player_starts   = sum(1 for t in things if t.type in (1, 2, 3, 4))
    has_coop_starts = player_starts > 1

    if has_dm_start:
        tags.append("deathmatch")
    if not has_dm_start or monster_count > 0:
        tags.append("single-player")
    if has_coop_starts:
        tags.append("cooperative")

    # ── Size (by sector count as a proxy for map footprint) ──────────────────
    n_sectors = len(doom_map.sectors)
    if n_sectors < 15:
        tags.append("tiny")
    elif n_sectors < 40:
        tags.append("small")
    elif n_sectors < 120:
        tags.append("medium")
    elif n_sectors < 350:
        tags.append("large")
    else:
        tags.append("huge")

    # ── Difficulty ───────────────────────────────────────────────────────────
    if not has_dm_start:  # skip difficulty for pure DM maps
        if monster_count == 0:
            pass  # no judgement
        elif monster_count < 25:
            tags.append("easy")
        elif monster_count < 100:
            tags.append("medium")
        elif monster_count < 300:
            tags.append("hard")
        else:
            tags.append("slaughter")

    # ── Theme (texture frequency voting) ─────────────────────────────────────
    all_tex: set[str] = set()
    for sd in doom_map.sidedefs:
        for name in (sd.upper_tex, sd.lower_tex, sd.mid_tex):
            if name and name not in ("-", ""):
                all_tex.add(name)
    all_flats: set[str] = set()
    for sec in doom_map.sectors:
        all_flats.add(sec.floor_flat)
        all_flats.add(sec.ceil_flat)

    scores: dict[str, int] = {
        "tech":    len(all_tex & _TECH_TEXTURES),
        "hell":    len(all_tex & _HELL_TEXTURES),
        "gothic":  len(all_tex & _GOTHIC_TEXTURES),
        "wood":    len(all_tex & _WOOD_TEXTURES),
        "outdoor": len(all_flats & _OUTDOOR_FLATS),
        "city":    len(all_tex & _CITY_TEXTURES),
        "snow":    len((all_tex | all_flats) & _SNOW_TEXTURES),
    }

    # "outdoor" triggers on even a single sky flat
    if "F_SKY1" in all_flats:
        scores["outdoor"] += 5

    best_theme, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score >= 2:
        tags.append(best_theme)
        # hell-tech: strong presence of both
        if best_theme in ("tech", "hell") and scores.get("tech", 0) >= 2 and scores.get("hell", 0) >= 2:
            tags = [t for t in tags if t not in ("tech", "hell")]
            tags.append("hell-tech")

    # ── Style (structural heuristics) ────────────────────────────────────────
    if n_sectors > 0:
        # Arena: one large sector with many things
        max_sector_things = _max_things_in_sector(doom_map)
        if max_sector_things > max(15, monster_count * 0.4):
            tags.append("arena")

        # Open: few linedefs per sector ratio
        ld_per_sector = len(doom_map.linedefs) / n_sectors
        if ld_per_sector < 5:
            tags.append("open")

        # Labyrinthine: high linedef density, many sectors
        if n_sectors > 60 and ld_per_sector > 12:
            tags.append("labyrinthine")

        # Hub: any sector has 4+ two-sided linedef connections
        adj = doom_map.sector_graph.get("adjacency", {})
        if any(len(v) >= 4 for v in adj.values()):
            tags.append("hub")

    return sorted(set(tags))


def _max_things_in_sector(doom_map: DoomMap) -> int:
    """Approximate count of things in the most-populated sector."""
    from wad_parser.map_extractor import _find_sector_for_point

    counts: dict[int, int] = {}
    for thing in doom_map.things:
        if thing.category in ("monster", "weapon", "ammo", "health", "armor", "powerup"):
            sid = _find_sector_for_point(thing.x, thing.y, doom_map)
            if sid >= 0:
                counts[sid] = counts.get(sid, 0) + 1
    return max(counts.values(), default=0)

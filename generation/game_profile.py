"""A game's asset vocabulary: the wall textures, flats, and thing types a
particular IWAD can actually render, derived authoritatively FROM that
IWAD. Generation uses it to guarantee an output map only references assets
the target game provides (or ones bundled into the PWAD) - so a map built
from a mixed training corpus still loads cleanly against, say, DOOM2.WAD,
with no missing-texture errors and no unspawnable things.

    python -m generation.game_profile wads/DOOM2.WAD doom2_profile.json

Textures/flats come from the IWAD's TEXTURE1/2 + PNAMES + F_*_START spans
(the set the engine can display); the valid thing set is the DoomEd numbers
the IWAD's own maps actually use (there is no lump that lists a game's
monster roster, so its maps are the authoritative sample). Usage counts are
kept so substitution can prefer common assets and match by category.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path

from wad_parser.map_extractor import extract_all_maps
from wad_parser.texture_parser import build_texture_set
from wad_parser.types import MapDocument, resolve_doomed
from wad_parser.wad_reader import WadFile

# Thing types that are structural, not decorative - always valid, never
# substituted or dropped, because generation/validation places them by rule.
_ESSENTIAL_THINGS = {1, 2, 3, 4, 11, 14}  # player 1-4 starts, DM start, teleport dest
_NON_ASSET_FLATS = {"-", ""}


@dataclass
class GameProfile:
    name: str
    textures: set[str] = field(default_factory=set)      # renderable wall textures
    flats: set[str] = field(default_factory=set)
    thing_types: set[int] = field(default_factory=set)
    texture_use: Counter = field(default_factory=Counter)
    flat_use: Counter = field(default_factory=Counter)
    thing_use: Counter = field(default_factory=Counter)

    # -- construction --------------------------------------------------------

    @classmethod
    def from_wad(cls, wad_path: Path | str, name: str | None = None) -> "GameProfile":
        wad = WadFile(str(wad_path))
        tex_set = build_texture_set(
            texture1=wad.get_lump("TEXTURE1"),
            texture2=wad.get_lump("TEXTURE2"),
            pnames_data=wad.get_lump("PNAMES"),
            flat_bytes=wad.find_flats(),
            playpal=wad.get_lump("PLAYPAL"),
            colormap=wad.get_lump("COLORMAP"),
            animated=wad.get_lump("ANIMATED"),
            switches=wad.get_lump("SWITCHES"),
        )
        prof = cls(name=name or Path(wad_path).stem.upper())
        prof.textures = {t.upper() for t in tex_set.textures}
        prof.flats = {a.name.upper() for a in tex_set.flats}
        for doc in extract_all_maps(wad):
            prof._accumulate_usage(doc)
        return prof

    def _accumulate_usage(self, doc: MapDocument) -> None:
        for s in doc.sectors:
            for flat in (s.floor_flat, s.ceil_flat):
                if flat.upper() not in _NON_ASSET_FLATS:
                    self.flat_use[flat.upper()] += 1
        for sd in doc.sidedefs:
            for tex in (sd.upper_tex, sd.lower_tex, sd.mid_tex):
                if tex not in ("-", ""):
                    self.texture_use[tex.upper()] += 1
        for t in doc.things:
            self.thing_types.add(t.type)
            self.thing_use[t.type] += 1

    def extend_assets(self, textures: set[str] = frozenset(),
                      flats: set[str] = frozenset()) -> None:
        """Register PWAD-bundled custom assets so they pass enforcement."""
        self.textures |= {t.upper() for t in textures}
        self.flats |= {f.upper() for f in flats}

    # -- substitution --------------------------------------------------------

    @property
    def default_wall(self) -> str:
        if self.texture_use:
            return self.texture_use.most_common(1)[0][0]
        return next(iter(sorted(self.textures)), "STARTAN2")

    @property
    def default_flat(self) -> str:
        if self.flat_use:
            return self.flat_use.most_common(1)[0][0]
        return next(iter(sorted(self.flats)), "FLOOR4_8")

    def _closest(self, name: str, pool: set[str], default: str) -> str:
        match = get_close_matches(name, pool, n=1, cutoff=0.6)
        return match[0] if match else default

    def sub_texture(self, name: str) -> str:
        if name in ("-", ""):
            return name
        up = name.upper()
        if up in self.textures:
            return up
        return self._closest(up, self.textures, self.default_wall)

    def sub_flat(self, name: str) -> str:
        if name in _NON_ASSET_FLATS:
            return name
        up = name.upper()
        if up in self.flats:
            return up
        return self._closest(up, self.flats, self.default_flat)

    def sub_thing(self, num: int) -> int | None:
        """Nearest valid thing of the same category, or None to drop it."""
        if num in self.thing_types or num in _ESSENTIAL_THINGS:
            return num
        category = resolve_doomed(num)[1]
        same = [t for t in self.thing_use if resolve_doomed(t)[1] == category]
        if not same:
            return None
        return max(same, key=self.thing_use.get)

    # -- enforcement ---------------------------------------------------------

    def enforce(self, doc: MapDocument) -> dict:
        """Rewrite every texture/flat/thing on `doc` to something this game
        can render. Mutates doc; returns a summary of what changed."""
        tex_fixed = flat_fixed = thing_fixed = 0
        for s in doc.sectors:
            new_floor, new_ceil = self.sub_flat(s.floor_flat), self.sub_flat(s.ceil_flat)
            flat_fixed += (new_floor != s.floor_flat) + (new_ceil != s.ceil_flat)
            s.floor_flat, s.ceil_flat = new_floor, new_ceil
        for sd in doc.sidedefs:
            for attr in ("upper_tex", "lower_tex", "mid_tex"):
                old = getattr(sd, attr)
                new = self.sub_texture(old)
                if new != old:
                    tex_fixed += 1
                    setattr(sd, attr, new)
        kept = []
        dropped = 0
        for t in doc.things:
            new_type = self.sub_thing(t.type)
            if new_type is None:
                dropped += 1
                continue
            if new_type != t.type:
                thing_fixed += 1
                t.type = new_type
                t.type_name, t.category, t.unresolved = resolve_doomed(new_type)
            kept.append(t)
        for i, t in enumerate(kept):
            t.id = i
        doc.things = kept
        return {"game": self.name, "textures_remapped": tex_fixed,
                "flats_remapped": flat_fixed, "things_remapped": thing_fixed,
                "things_dropped": dropped}

    # -- persistence ---------------------------------------------------------

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "textures": sorted(self.textures),
            "flats": sorted(self.flats),
            "thing_types": sorted(self.thing_types),
            "texture_use": dict(self.texture_use),
            "flat_use": dict(self.flat_use),
            "thing_use": {str(k): v for k, v in self.thing_use.items()},
        }

    @classmethod
    def from_json(cls, data: dict) -> "GameProfile":
        return cls(
            name=data["name"],
            textures=set(data["textures"]),
            flats=set(data["flats"]),
            thing_types=set(data["thing_types"]),
            texture_use=Counter(data.get("texture_use", {})),
            flat_use=Counter(data.get("flat_use", {})),
            thing_use=Counter({int(k): v for k, v in data.get("thing_use", {}).items()}),
        )

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "GameProfile":
        return cls.from_json(json.loads(Path(path).read_text()))


def load_profile(path_or_wad: Path | str | None) -> GameProfile | None:
    """A saved profile .json, or an IWAD to derive one from on the fly."""
    if path_or_wad is None:
        return None
    p = Path(path_or_wad)
    if p.suffix.lower() == ".wad":
        return GameProfile.from_wad(p)
    return GameProfile.load(p)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Derive a game asset profile from an IWAD.")
    ap.add_argument("wad")
    ap.add_argument("out_json")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    prof = GameProfile.from_wad(args.wad, name=args.name)
    prof.save(args.out_json)
    print(f"{prof.name}: {len(prof.textures)} textures, {len(prof.flats)} flats, "
          f"{len(prof.thing_types)} thing types "
          f"(from {sum(prof.thing_use.values())} placed things)")


if __name__ == "__main__":
    main()

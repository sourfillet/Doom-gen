"""WAD-level and per-map tag assignment storage with inheritance.

Layout (keyed by WAD hash so re-tagging never requires re-parsing and
re-parsing never touches tags):

    <tags_dir>/registry.json          (owned by tagging.registry)
    <tags_dir>/<wad_hash>.wad.json
    <tags_dir>/<wad_hash>/<MAPNAME>.json

Effective tags for a map = its own tags ∪ its WAD's tags, minus the map's
excluded-inherited list. A map-level assignment of the same tag overrides
the inherited weight. Resolution happens at read time; files only ever
store what the user asserted.

Auto-derived fact tags (tagging.auto) live in parallel .auto.json files:

    <tags_dir>/<wad_hash>.wad.auto.json
    <tags_dir>/<wad_hash>/<MAPNAME>.auto.json

They are a regenerable derived layer: rewritten wholesale by write_auto(),
never touched by rename/merge (regeneration would just recreate the rule's
name), always outranked by user assignments, and vetoable via a per-scope
"denied_auto" list in the user's own file so a veto survives regeneration.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from tagging.registry import Registry, _write_json, canonicalize

SCHEMA_VERSION = "1.0"


def _validate_weight(weight: float) -> float:
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
        raise ValueError(f"tag weight must be a finite non-negative number, got {weight!r}")
    return float(weight)


class TagStore:
    def __init__(self, tags_dir: Path | str):
        self.root = Path(tags_dir)
        self.registry = Registry(self.root / "registry.json")

    # -- file paths --------------------------------------------------------

    def _wad_path(self, wad_hash: str) -> Path:
        return self.root / f"{wad_hash}.wad.json"

    def _map_path(self, wad_hash: str, map_name: str) -> Path:
        return self.root / wad_hash / f"{map_name.upper()}.json"

    def _wad_auto_path(self, wad_hash: str) -> Path:
        return self.root / f"{wad_hash}.wad.auto.json"

    def _map_auto_path(self, wad_hash: str, map_name: str) -> Path:
        return self.root / wad_hash / f"{map_name.upper()}.auto.json"

    # -- raw reads ---------------------------------------------------------

    def _load(self, path: Path, default: dict) -> dict:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return dict(default)

    def wad_record(self, wad_hash: str) -> dict:
        rec = self._load(self._wad_path(wad_hash),
                         {"schema_version": SCHEMA_VERSION, "wad_hash": wad_hash,
                          "wad_name": None, "tags": []})
        rec.setdefault("denied_auto", [])
        return rec

    def map_record(self, wad_hash: str, map_name: str) -> dict:
        rec = self._load(self._map_path(wad_hash, map_name),
                         {"schema_version": SCHEMA_VERSION, "tags": [],
                          "excluded_inherited": []})
        rec.setdefault("denied_auto", [])
        return rec

    @staticmethod
    def _tags_as_dict(record: dict) -> dict[str, float]:
        return {a["tag"]: a.get("weight", 1.0) for a in record.get("tags", [])}

    # -- assignment --------------------------------------------------------

    def tag_wad(self, wad_hash: str, tag: str, weight: float = 1.0,
                wad_name: str | None = None) -> str:
        weight = _validate_weight(weight)
        canon = self.registry.ensure(tag)
        rec = self.wad_record(wad_hash)
        if wad_name:
            rec["wad_name"] = wad_name
        assignments = {a["tag"]: a for a in rec["tags"]}
        assignments[canon] = {"tag": canon, "weight": weight}
        rec["tags"] = sorted(assignments.values(), key=lambda a: a["tag"])
        _write_json(self._wad_path(wad_hash), rec)
        return canon

    def untag_wad(self, wad_hash: str, tag: str) -> bool:
        canon = canonicalize(tag)
        path = self._wad_path(wad_hash)
        if not path.exists():
            return False
        rec = self.wad_record(wad_hash)
        before = len(rec["tags"])
        rec["tags"] = [a for a in rec["tags"] if a["tag"] != canon]
        if len(rec["tags"]) < before:
            _write_json(path, rec)
            return True
        return False

    def tag_map(self, wad_hash: str, map_name: str, tag: str, weight: float = 1.0) -> str:
        weight = _validate_weight(weight)
        canon = self.registry.ensure(tag)
        rec = self.map_record(wad_hash, map_name)
        assignments = {a["tag"]: a for a in rec["tags"]}
        assignments[canon] = {"tag": canon, "weight": weight}
        rec["tags"] = sorted(assignments.values(), key=lambda a: a["tag"])
        _write_json(self._map_path(wad_hash, map_name), rec)
        return canon

    def untag_map(self, wad_hash: str, map_name: str, tag: str) -> bool:
        canon = canonicalize(tag)
        path = self._map_path(wad_hash, map_name)
        if not path.exists():
            return False
        rec = self.map_record(wad_hash, map_name)
        before = len(rec["tags"])
        rec["tags"] = [a for a in rec["tags"] if a["tag"] != canon]
        if len(rec["tags"]) < before:
            _write_json(path, rec)
            return True
        return False

    # -- inheritance exclusion ----------------------------------------------

    def exclude_inherited(self, wad_hash: str, map_name: str, tag: str) -> None:
        canon = canonicalize(tag)
        rec = self.map_record(wad_hash, map_name)
        if canon not in rec["excluded_inherited"]:
            rec["excluded_inherited"].append(canon)
            rec["excluded_inherited"].sort()
            _write_json(self._map_path(wad_hash, map_name), rec)

    def include_inherited(self, wad_hash: str, map_name: str, tag: str) -> None:
        canon = canonicalize(tag)
        path = self._map_path(wad_hash, map_name)
        if not path.exists():
            return
        rec = self.map_record(wad_hash, map_name)
        updated = [t for t in rec["excluded_inherited"] if t != canon]
        if len(updated) != len(rec["excluded_inherited"]):
            rec["excluded_inherited"] = updated
            _write_json(path, rec)

    # -- auto-derived layer ----------------------------------------------------

    def write_auto(self, wad_hash: str, wad_tags: dict[str, float],
                   per_map: dict[str, dict[str, float]],
                   descriptions: dict[str, str] | None = None) -> None:
        """Replaces the WAD's entire auto layer (stale auto files for maps
        no longer present are removed - it's derived data)."""
        canon_wad = {canonicalize(t): _validate_weight(w) for t, w in wad_tags.items()}
        canon_maps = {
            map_name: {canonicalize(t): _validate_weight(w) for t, w in tags.items()}
            for map_name, tags in per_map.items()
        }
        canon_desc = {canonicalize(k): v for k, v in (descriptions or {}).items()}
        added_any = False
        for tag in sorted(set(canon_wad) | {t for m in canon_maps.values() for t in m}):
            if tag not in self.registry:
                self.registry.ensure(tag, canon_desc.get(tag, "[auto]"), save=False)
                added_any = True
        if added_any:
            self.registry.save()
        _write_json(self._wad_auto_path(wad_hash), {
            "schema_version": SCHEMA_VERSION, "wad_hash": wad_hash,
            "tags": [{"tag": t, "weight": w} for t, w in sorted(canon_wad.items())],
        })
        wad_dir = self.root / wad_hash
        for stale in wad_dir.glob("*.auto.json") if wad_dir.is_dir() else []:
            stale.unlink()
        for map_name, tags in canon_maps.items():
            if tags:
                _write_json(self._map_auto_path(wad_hash, map_name), {
                    "schema_version": SCHEMA_VERSION,
                    "tags": [{"tag": t, "weight": w} for t, w in sorted(tags.items())],
                })

    def wad_auto_tags(self, wad_hash: str) -> dict[str, float]:
        rec = self._load(self._wad_auto_path(wad_hash), {"tags": []})
        denied = set(self.wad_record(wad_hash)["denied_auto"])
        return {t: w for t, w in self._tags_as_dict(rec).items() if t not in denied}

    def map_auto_tags(self, wad_hash: str, map_name: str) -> dict[str, float]:
        rec = self._load(self._map_auto_path(wad_hash, map_name), {"tags": []})
        denied = set(self.map_record(wad_hash, map_name)["denied_auto"])
        return {t: w for t, w in self._tags_as_dict(rec).items() if t not in denied}

    def deny_auto(self, wad_hash: str, tag: str, map_name: str | None = None) -> None:
        """User veto of an auto tag; survives regeneration because it lives
        in the user's own file."""
        canon = canonicalize(tag)
        if map_name is None:
            rec = self.wad_record(wad_hash)
            path = self._wad_path(wad_hash)
        else:
            rec = self.map_record(wad_hash, map_name)
            path = self._map_path(wad_hash, map_name)
        if canon not in rec["denied_auto"]:
            rec["denied_auto"] = sorted(rec["denied_auto"] + [canon])
            _write_json(path, rec)

    def allow_auto(self, wad_hash: str, tag: str, map_name: str | None = None) -> bool:
        """Clear a prior deny_auto veto."""
        canon = canonicalize(tag)
        path = self._wad_path(wad_hash) if map_name is None else self._map_path(wad_hash, map_name)
        if not path.exists():
            return False
        rec = self.wad_record(wad_hash) if map_name is None else self.map_record(wad_hash, map_name)
        updated = [t for t in rec["denied_auto"] if t != canon]
        if len(updated) != len(rec["denied_auto"]):
            rec["denied_auto"] = updated
            _write_json(path, rec)
            return True
        return False

    # -- resolution ----------------------------------------------------------

    def wad_tags(self, wad_hash: str) -> dict[str, float]:
        return self._tags_as_dict(self.wad_record(wad_hash))

    def map_own_tags(self, wad_hash: str, map_name: str) -> dict[str, float]:
        return self._tags_as_dict(self.map_record(wad_hash, map_name))

    def effective_tags_detailed(self, wad_hash: str, map_name: str) -> dict[str, tuple[float, str]]:
        """{tag: (weight, source)} where source is "user" or "auto".
        Precedence (low to high): WAD auto < WAD user < map auto < map user;
        excluded_inherited drops WAD-level tags of either source; denials
        filter the auto layer at their own scope."""
        rec = self.map_record(wad_hash, map_name)
        excluded = set(rec.get("excluded_inherited", []))

        effective: dict[str, tuple[float, str]] = {}
        for t, w in self.wad_auto_tags(wad_hash).items():
            effective[t] = (w, "auto")
        for t, w in self.wad_tags(wad_hash).items():
            effective[t] = (w, "user")
        for t in excluded:
            effective.pop(t, None)
        for t, w in self.map_auto_tags(wad_hash, map_name).items():
            effective[t] = (w, "auto")
        for t, w in self._tags_as_dict(rec).items():
            effective[t] = (w, "user")
        return effective

    def effective_tags(self, wad_hash: str, map_name: str) -> dict[str, float]:
        """own ∪ inherited (user and auto layers), user wins on collision,
        exclusions drop inherited tags entirely."""
        return {t: w for t, (w, _) in self.effective_tags_detailed(wad_hash, map_name).items()}

    # -- corpus-wide rename / merge -------------------------------------------

    def _iter_assignment_files(self):
        """User assignment files only - the .auto.json layer is derived
        data, deliberately untouched by rename/merge (regenerating it would
        just recreate the rule's canonical name anyway)."""
        if not self.root.exists():
            return
        yield from self.root.glob("*.wad.json")
        for sub in self.root.iterdir():
            if sub.is_dir():
                for p in sub.glob("*.json"):
                    if not p.name.endswith(".auto.json"):
                        yield p

    def _rewrite_tag(self, old_canon: str, new_canon: str) -> int:
        """Rewrites every assignment file, folding old into new. On
        collision the existing new-tag assignment wins (its weight is what
        the user chose knowing both tags existed)."""
        if old_canon == new_canon:
            return 0
        touched = 0
        for path in self._iter_assignment_files():
            rec = json.loads(path.read_text(encoding="utf-8"))
            changed = False
            assignments = {a["tag"]: a for a in rec.get("tags", [])}
            if old_canon in assignments:
                moved = assignments.pop(old_canon)
                if new_canon not in assignments:
                    moved["tag"] = new_canon
                    assignments[new_canon] = moved
                rec["tags"] = sorted(assignments.values(), key=lambda a: a["tag"])
                changed = True
            excl = rec.get("excluded_inherited")
            if excl and old_canon in excl:
                rec["excluded_inherited"] = sorted(
                    {new_canon if t == old_canon else t for t in excl})
                changed = True
            if changed:
                _write_json(path, rec)
                touched += 1
        return touched

    def rename_tag(self, old: str, new: str) -> int:
        old_c, new_c = self.registry.rename(old, new)
        return self._rewrite_tag(old_c, new_c)

    def merge_tags(self, src: str, dst: str) -> int:
        src_c, dst_c = self.registry.merge(src, dst)
        return self._rewrite_tag(src_c, dst_c)

    # -- corpus stats (registry list, suggestion ranking) ---------------------

    def tag_usage(self, *, include_auto: bool = False) -> dict[str, int]:
        counts: dict[str, int] = {}
        for path in self._iter_assignment_files():
            rec = json.loads(path.read_text(encoding="utf-8"))
            for a in rec.get("tags", []):
                counts[a["tag"]] = counts.get(a["tag"], 0) + 1
        if include_auto and self.root.exists():
            for auto_path in self.root.glob("*.wad.auto.json"):
                wad_hash = auto_path.name[:-len(".wad.auto.json")]
                for tag in self.wad_auto_tags(wad_hash):
                    counts[tag] = counts.get(tag, 0) + 1
            for sub in self.root.iterdir():
                if sub.is_dir():
                    for auto_path in sub.glob("*.auto.json"):
                        map_name = auto_path.name[:-len(".auto.json")]
                        for tag in self.map_auto_tags(sub.name, map_name):
                            counts[tag] = counts.get(tag, 0) + 1
        return counts

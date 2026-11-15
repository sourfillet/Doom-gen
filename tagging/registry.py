"""Tag registry: every tag name the user has ever created, with their own
free-text description (later also the text-encoder cold-start input for
model conditioning).

The registry is organizational, not restrictive: any string can become a
tag the first time it's used, nothing is pre-registered, and the only
normalization is canonicalization (lowercase, trimmed, whitespace
collapsed). The user's preferred display spelling is kept separately.
"""
from __future__ import annotations

import datetime
import json
import re
from dataclasses import dataclass, field, fields
from pathlib import Path

SCHEMA_VERSION = "1.0"

_WS_RE = re.compile(r"\s+")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def canonicalize(name: str) -> str:
    """lowercase / trim / collapse whitespace - the ONLY normalization
    applied to tag names anywhere."""
    canon = _WS_RE.sub(" ", name.strip()).lower()
    if not canon:
        raise ValueError("tag name is empty after normalization")
    return canon


@dataclass
class TagEntry:
    display: str
    description: str = ""
    color: str | None = None
    group: str | None = None
    created: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())


class Registry:
    """Backed by <tags_dir>/registry.json; loads lazily, saves explicitly."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.tags: dict[str, TagEntry] = {}
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            known = {f.name for f in fields(TagEntry)}
            for canon, entry in raw.get("tags", {}).items():
                self.tags[canon] = TagEntry(**{k: v for k, v in entry.items() if k in known})

    def save(self) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "tags": {c: vars(e) for c, e in sorted(self.tags.items())},
        }
        _write_json(self.path, payload)

    # -- creation / lookup ------------------------------------------------

    def ensure(self, name: str, description: str = "", *, save: bool = True) -> str:
        """Creates the tag on first use (single-command creation, per the
        plan); returns the canonical name either way."""
        canon = canonicalize(name)
        if canon not in self.tags:
            self.tags[canon] = TagEntry(display=name.strip(), description=description)
            if save:
                self.save()
        return canon

    def __contains__(self, name: str) -> bool:
        return canonicalize(name) in self.tags

    def get(self, name: str) -> TagEntry | None:
        return self.tags.get(canonicalize(name))

    def describe(self, name: str, description: str) -> None:
        canon = canonicalize(name)
        if canon not in self.tags:
            raise KeyError(f"unknown tag {canon!r}")
        self.tags[canon].description = description
        self.save()

    # -- autocomplete ------------------------------------------------------

    def complete(self, partial: str) -> list[str]:
        """Prefix matches first, then substring matches - this is what makes
        near-duplicates ("techbase" vs "tech base") visible while typing."""
        try:
            p = canonicalize(partial)
        except ValueError:
            return sorted(self.tags)
        prefix = [t for t in sorted(self.tags) if t.startswith(p)]
        substr = [t for t in sorted(self.tags) if p in t and not t.startswith(p)]
        return prefix + substr

    # -- rename / merge ----------------------------------------------------
    # These only touch the registry; TagStore.rename_tag/merge_tags drive
    # the corpus-wide rewrite and call into these.

    def rename(self, old: str, new: str) -> tuple[str, str]:
        old_c, new_c = canonicalize(old), canonicalize(new)
        if old_c not in self.tags:
            raise KeyError(f"unknown tag {old_c!r}")
        if new_c in self.tags and new_c != old_c:
            raise ValueError(f"tag {new_c!r} already exists - use merge instead")
        entry = self.tags.pop(old_c)
        entry.display = new.strip()
        self.tags[new_c] = entry
        self.save()
        return old_c, new_c

    def merge(self, src: str, dst: str) -> tuple[str, str]:
        src_c, dst_c = canonicalize(src), canonicalize(dst)
        if src_c not in self.tags:
            raise KeyError(f"unknown tag {src_c!r}")
        if dst_c not in self.tags:
            raise KeyError(f"unknown tag {dst_c!r}")
        if src_c == dst_c:
            raise ValueError("cannot merge a tag into itself")
        src_entry = self.tags.pop(src_c)
        dst_entry = self.tags[dst_c]
        if src_entry.description and src_entry.description not in dst_entry.description:
            joined = f"{dst_entry.description}\n[merged from {src_c!r}]: {src_entry.description}"
            dst_entry.description = joined.strip()
        self.save()
        return src_c, dst_c

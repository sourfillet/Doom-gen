"""Reproducible source-family splits and duplicate-aware corpus manifests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from wad_parser.types import MapDocument


def content_hash(doc):
    raw = doc.to_dict()
    content = {key: raw[key] for key in ("vertices", "sectors", "linedefs", "sidedefs", "things")}
    content.update(format=doc.meta.format, udmf_namespace=doc.extra.get("udmf_namespace"),
                   udmf_globals=doc.extra.get("udmf_globals"))
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_manifest(maps_dir, *, seed=0, validation_fraction=0.1, test_fraction=0.1,
                   families=None, deduplicate=True):
    if not (0 <= validation_fraction < 1 and 0 <= test_fraction < 1
            and validation_fraction + test_fraction < 1):
        raise ValueError("holdout fractions must be non-negative and sum to less than one")
    root = Path(maps_dir).resolve()
    entries = []
    for path in sorted(root.glob("*.json")):
        if path.name.endswith(".textures.json"):
            continue
        data = path.read_bytes()
        doc = MapDocument.from_dict(json.loads(data))
        if doc.meta.format != "doom" or doc.extra.get("udmf_namespace"):
            raise ValueError(f"training corpus currently requires Doom binary maps: {path.name}")
        entries.append({"path": path.name, "wad_hash": doc.meta.wad_hash,
                        "family": (families or {}).get(doc.meta.wad_hash, doc.meta.wad_hash),
                        "map_name": doc.meta.map_name, "content_hash": content_hash(doc),
                        "document_hash": hashlib.sha256(data).hexdigest()})
    if not entries:
        raise ValueError(f"no map documents in {root}")
    parents = {}
    def find(key):
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = find(parents[key])
        return parents[key]
    def union(a, b):
        a, b = find(a), find(b)
        parents[max(a, b)] = min(a, b)
    seen_content = {}
    for entry in entries:
        family = entry["family"]
        if entry["content_hash"] in seen_content:
            union(family, seen_content[entry["content_hash"]])
        else:
            seen_content[entry["content_hash"]] = family
        find(family)
    groups = sorted({find(entry["family"]) for entry in entries}, key=lambda value:
                    hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
    # Never exhaust the training groups. Tiny corpora explicitly report that
    # an independent holdout cannot be formed; one map must not leak into it.
    n_val = max(1, int(len(groups) * validation_fraction)) if validation_fraction and len(groups) >= 3 else 0
    n_test = max(1, int(len(groups) * test_fraction)) if test_fraction and len(groups) >= 3 else 0
    n_test = min(n_test, len(groups) - n_val - 1)
    assignment = {group: "validation" if i < n_val else "test" if i < n_val + n_test else "train"
                  for i, group in enumerate(groups)}
    canonical, duplicates = {}, []
    for entry in entries:
        entry["group"] = find(entry["family"])
        entry["split"] = assignment[entry["group"]]
        if deduplicate and entry["content_hash"] in canonical:
            duplicates.append({**entry, "canonical_path": canonical[entry["content_hash"]]})
        else:
            canonical[entry["content_hash"]] = entry["path"]
    selected = [entry for entry in entries if not deduplicate or canonical[entry["content_hash"]] == entry["path"]]
    manifest = {"version": 1, "seed": seed, "entries": selected, "duplicates": duplicates,
                "validation_fraction": validation_fraction, "test_fraction": test_fraction,
                "families_supplied": bool(families)}
    manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return manifest


def training_split(maps_dir, manifest_path=None, *, seed=0):
    root = Path(maps_dir).resolve()
    manifest = json.loads(Path(manifest_path).read_text()) if manifest_path else build_manifest(root, seed=seed)
    if manifest.get("version") != 1:
        raise ValueError("unsupported corpus manifest version")
    payload = {key: value for key, value in manifest.items() if key != "fingerprint"}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if manifest.get("fingerprint") != fingerprint:
        raise ValueError("corpus manifest fingerprint mismatch; regenerate it after editing")
    splits = {"train": [], "validation": [], "test": []}
    for entry in manifest["entries"] + manifest.get('duplicates', []):
        path = root / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["document_hash"]:
            raise ValueError(f"corpus document changed: {entry['path']}")
        if 'canonical_path' not in entry:
            splits[entry["split"]].append(path)
    if not splits["train"]:
        raise ValueError("corpus manifest has no training maps")
    return splits, manifest

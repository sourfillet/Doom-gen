"""Dataset over an extracted-JSON corpus: yields (raster, graph,
tag weights) triples for the same underlying map document, per
plan/generation.md - every model stage trains from the same extracted
data, and tags come from the TagStore (user + auto layers resolved,
user winning) rather than anything baked into the parse.

Framework-free by design (plain numpy); the torch Dataset wrapper belongs
with the training loop, not here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.augmentation import transform_document
from data.graph_builder import build_graph
from data.rasterizer import rasterize
from wad_parser.types import MapDocument


class MapDataset:
    def __init__(self, maps_dir: Path | str, tags_dir: Path | str | None = None,
                 resolution: int = 128):
        self.maps_dir = Path(maps_dir)
        self.resolution = resolution
        self.paths = sorted(
            p for p in self.maps_dir.glob("*.json")
            if not p.name.endswith(".textures.json")
        )
        self.store = None
        if tags_dir is not None:
            from tagging.store import TagStore
            self.store = TagStore(tags_dir)

    def __len__(self) -> int:
        return len(self.paths)

    def load_document(self, index: int) -> MapDocument:
        return MapDocument.from_dict(json.loads(self.paths[index].read_text()))

    def tags_for(self, doc: MapDocument) -> dict[str, float]:
        if self.store is None:
            return {}
        return self.store.effective_tags(doc.meta.wad_hash, doc.meta.map_name)

    def get(self, index: int, augment_op: str | None = None) -> dict:
        doc = self.load_document(index)
        if augment_op:
            doc = transform_document(doc, augment_op)
        raster, raster_meta = rasterize(doc, self.resolution)
        return {
            "raster": raster,
            "raster_meta": raster_meta,
            "graph": build_graph(doc),
            "tags": self.tags_for(doc),
            "map_name": doc.meta.map_name,
            "wad_hash": doc.meta.wad_hash,
        }

    def __getitem__(self, index: int) -> dict:
        return self.get(index)

    # -- tag conditioning helpers -------------------------------------------

    def tag_vocab(self) -> list[str]:
        """Sorted union of effective tags across the corpus. Callers should
        expect this to GROW between runs (open vocabulary) - the embedding
        table keys by tag string, not by index (plan/tagging.md)."""
        vocab: set[str] = set()
        for i in range(len(self)):
            doc = self.load_document(i)
            vocab.update(self.tags_for(doc))
        return sorted(vocab)

    @staticmethod
    def tag_vector(tags: dict[str, float], vocab: list[str]) -> np.ndarray:
        """Weighted multi-hot over a given vocab ordering."""
        vec = np.zeros(len(vocab), dtype=np.float32)
        index = {t: i for i, t in enumerate(vocab)}
        for tag, weight in tags.items():
            if tag in index:
                vec[index[tag]] = weight
        return vec

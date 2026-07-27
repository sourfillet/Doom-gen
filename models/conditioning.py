"""Tag conditioning (plan/tagging.md, "From open-vocabulary tags to model
conditioning"): a learned per-tag embedding table keyed by canonical tag
STRING, growable as new tags appear - never an index space frozen at
training time. Conditioning vector = weight-scaled sum of tag embeddings;
the CFG null condition is the zero vector (dropout during training gives
the unconditional prior).

The text-encoder cold-start fallback for zero-example tags is a
documented later step; unknown tags at sample time currently warn and are
skipped.
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn


class TagConditioner(nn.Module):
    def __init__(self, embed_dim: int = 256, max_tags: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_tags = max_tags
        self.table = nn.Embedding(max_tags, embed_dim)
        nn.init.normal_(self.table.weight, std=0.02)
        self.tag_to_index: dict[str, int] = {}

    def _index_for(self, tag: str, register: bool) -> int | None:
        if tag in self.tag_to_index:
            return self.tag_to_index[tag]
        if not register:
            warnings.warn(f"tag {tag!r} unknown to this checkpoint; skipped "
                          "(text-encoder cold start not implemented yet)")
            return None
        if len(self.tag_to_index) >= self.max_tags:
            raise RuntimeError(f"tag table full ({self.max_tags}); raise max_tags")
        self.tag_to_index[tag] = len(self.tag_to_index)
        return self.tag_to_index[tag]

    def encode(self, batch_tags: list[dict[str, float]], device,
               cfg_dropout: float = 0.0, register_new: bool = False) -> torch.Tensor:
        """(B, D) conditioning vectors. With cfg_dropout > 0 (training),
        each sample's tags are dropped wholesale with that probability -
        standard classifier-free-guidance conditioning dropout."""
        out = torch.zeros(len(batch_tags), self.embed_dim, device=device)
        for i, tags in enumerate(batch_tags):
            if cfg_dropout > 0 and torch.rand(()).item() < cfg_dropout:
                continue  # null condition
            for tag, weight in tags.items():
                idx = self._index_for(tag, register=register_new)
                if idx is not None:
                    out[i] += weight * self.table(torch.tensor(idx, device=device))
        return out

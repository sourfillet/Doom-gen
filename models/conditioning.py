"""Tag conditioning: a learned per-tag embedding table keyed by canonical tag
STRING, growable as new tags appear - never an index space frozen at
training time. Conditioning vector = weight-scaled sum of tag embeddings;
the CFG null condition is the zero vector (dropout during training gives
the unconditional prior).

The text-encoder cold-start fallback for zero-example tags is a
documented later step; unknown tags at sample time currently warn and are
skipped.
"""
from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn

from tagging.registry import canonicalize


class TagConditioner(nn.Module):
    def __init__(self, embed_dim: int = 256, max_tags: int = 512, numeric_dim: int = 0):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_tags = max_tags
        self.table = nn.Embedding(max_tags, embed_dim)
        nn.init.normal_(self.table.weight, std=0.02)
        self.tag_to_index: dict[str, int] = {}
        self.numeric_dim = numeric_dim
        if numeric_dim:
            self.numeric_projection = nn.Linear(numeric_dim, embed_dim, bias=False)

    def load_checkpoint_state(self, ckpt: dict) -> "TagConditioner":
        self.load_state_dict(ckpt["conditioner"])
        self.tag_to_index = dict(ckpt["tag_to_index"])
        self.eval()
        return self

    def _index_for(self, tag: str, register: bool) -> int | None:
        tag = canonicalize(tag)
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
               cfg_dropout: float = 0.0, register_new: bool = False,
               numeric_values=None) -> torch.Tensor:
        """(B, D) conditioning vectors. With cfg_dropout > 0 (training),
        each sample's tags are dropped wholesale with that probability -
        standard classifier-free-guidance conditioning dropout."""
        values = None
        if self.numeric_dim > 0 and numeric_values is None:
            raise ValueError(f"conditioner expects numeric_values of dim {self.numeric_dim}")
        if self.numeric_dim == 0 and numeric_values is not None:
            raise ValueError("conditioner has numeric_dim=0 but numeric_values were supplied")
        if numeric_values is not None:
            values = torch.as_tensor(numeric_values, dtype=self.table.weight.dtype, device=device)
            if values.shape != (len(batch_tags), self.numeric_dim) or not torch.isfinite(values).all():
                raise ValueError("numeric conditioning has invalid shape or non-finite values")
        out = torch.zeros(len(batch_tags), self.embed_dim, dtype=self.table.weight.dtype, device=device)
        for i, tags in enumerate(batch_tags):
            if values is not None:
                out[i] += self.numeric_projection(values[i])
            resolved: list[tuple[int, float]] = []
            for tag, weight in tags.items():
                if not math.isfinite(weight):
                    raise ValueError(f"non-finite weight {weight!r} for tag {tag!r}")
                if register_new:
                    idx = self._index_for(tag, register=True)
                    if idx is not None:
                        resolved.append((idx, weight))
            if cfg_dropout > 0 and torch.rand(()).item() < cfg_dropout:
                continue  # null tag condition (numeric scale preserved)
            if register_new:
                for idx, weight in resolved:
                    out[i] += weight * self.table(torch.tensor(idx, device=device))
            else:
                for tag, weight in tags.items():
                    idx = self._index_for(tag, register=False)
                    if idx is not None:
                        out[i] += weight * self.table(torch.tensor(idx, device=device))
        return out

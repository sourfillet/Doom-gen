"""Stage 3 property GNN (plan/generation.md): a tag-conditioned message-
passing network over the sector-adjacency graph that predicts the
per-element properties a pixel raster can't hold - floor/ceiling flats,
wall textures, and sector specials - conditioned on geometry + tags.

The whole point over the corpus-statistics stopgap (generation/style.py) is
CONTEXT: a sector's look depends on its neighbours, size and height, which a
GNN can learn but a global frequency table cannot. Predicts categorical
properties only (flats/textures/specials); it does not move geometry.

Framework note: no torch_geometric dependency - message passing is plain
scatter/index_add over an edge list, one graph per forward pass.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Columns of data.graph_builder.NODE_FEATURES the GNN may read: geometry and
# light only. has_special / has_tag / thing-counts are targets or unknown at
# generation time, so they are deliberately excluded to avoid leakage.
INPUT_COLUMNS = [0, 1, 2, 3, 6, 7, 8]   # floor,ceil,room_h,light,area,perim,degree
IN_FEATURES = len(INPUT_COLUMNS)
HEADS = ("floor_flat", "ceil_flat", "wall_tex", "special")


class _GNNLayer(nn.Module):
    def __init__(self, dim: int, edge_dim: int):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(dim + edge_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.upd = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, edge_index, edge_feat, gamma, beta):
        src, dst = edge_index[0], edge_index[1]
        if src.numel() > 0:
            m = self.msg(torch.cat([h[src], edge_feat], dim=-1))     # (E, dim)
            agg = torch.zeros_like(h).index_add_(0, dst, m)
            deg = torch.zeros(h.size(0), 1, device=h.device, dtype=h.dtype)
            deg.index_add_(0, dst, torch.ones(dst.size(0), 1, device=h.device, dtype=h.dtype))
            agg = agg / deg.clamp(min=1.0)
        else:
            agg = torch.zeros_like(h)
        out = self.upd(torch.cat([h, agg], dim=-1))
        out = out * (1.0 + gamma) + beta                              # FiLM(tag)
        return self.norm(h + out)                                     # residual


class PropertyGNN(nn.Module):
    def __init__(self, edge_features: int, vocab: dict[str, int],
                 hidden: int = 128, n_layers: int = 4, tag_embed_dim: int = 256):
        super().__init__()
        self.vocab = dict(vocab)                       # {"flat","tex","special"} sizes
        self.node_enc = nn.Linear(IN_FEATURES, hidden)
        self.films = nn.ModuleList(
            [nn.Linear(tag_embed_dim, hidden * 2) for _ in range(n_layers)])
        self.layers = nn.ModuleList(
            [_GNNLayer(hidden, edge_features) for _ in range(n_layers)])
        self.heads = nn.ModuleDict({
            "floor_flat": nn.Linear(hidden, vocab["flat"]),
            "ceil_flat": nn.Linear(hidden, vocab["flat"]),
            "wall_tex": nn.Linear(hidden, vocab["tex"]),
            "special": nn.Linear(hidden, vocab["special"]),
        })

    def forward(self, node_feat, edge_index, edge_feat, tag_embed) -> dict:
        """node_feat (N, IN_FEATURES), edge_index (2, E), edge_feat (E, Fe),
        tag_embed (tag_embed_dim,). Returns per-head logits (N, vocab)."""
        h = self.node_enc(node_feat)
        for film, layer in zip(self.films, self.layers):
            gamma, beta = film(tag_embed).chunk(2, dim=-1)
            h = layer(h, edge_index, edge_feat, gamma, beta)
        return {name: head(h) for name, head in self.heads.items()}

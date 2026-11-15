"""Stage 3 property GNN: a tag-conditioned message-
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
LEGACY_INPUT_COLUMNS = [0, 1, 2, 3, 6, 7, 8]
INPUT_COLUMNS = LEGACY_INPUT_COLUMNS + [13, 14, 15, 16, 17]   # floor,ceil,room_h,light,area,perim,degree
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

    def forward(self, h, edge_index, edge_feat, gamma, beta, deg_inv=None):
        src, dst = edge_index[0], edge_index[1]
        if src.numel() > 0:
            m = self.msg(torch.cat([h[src], edge_feat], dim=-1))     # (E, dim)
            agg = torch.zeros_like(h).index_add_(0, dst, m)
            if deg_inv is None:
                deg = torch.zeros(h.size(0), 1, device=h.device, dtype=h.dtype)
                deg.index_add_(0, dst, torch.ones(dst.size(0), 1, device=h.device, dtype=h.dtype))
                deg_inv = 1.0 / deg.clamp(min=1.0)
            agg = agg * deg_inv
        else:
            agg = torch.zeros_like(h)
        out = self.upd(torch.cat([h, agg], dim=-1))
        out = out * (1.0 + gamma) + beta                              # FiLM(tag)
        return self.norm(h + out)                                     # residual


class _SectorGraphBackbone(nn.Module):
    """Shared FiLM message-passing backbone for Stage 3/4 sector GNNs."""

    def __init__(self, edge_features: int, hidden: int = 128, n_layers: int = 4,
                 tag_embed_dim: int = 256, input_features: int = IN_FEATURES):
        super().__init__()
        self.tag_embed_dim = tag_embed_dim
        self.node_enc = nn.Linear(input_features, hidden)
        self.films = nn.ModuleList(
            [nn.Linear(tag_embed_dim, hidden * 2) for _ in range(n_layers)])
        self.layers = nn.ModuleList(
            [_GNNLayer(hidden, edge_features) for _ in range(n_layers)])

    def _encode_nodes(self, node_feat, edge_index, edge_feat, tag_embed):
        if tag_embed.ndim != 1 or tag_embed.shape[0] != self.tag_embed_dim:
            raise ValueError(
                f"tag_embed must have shape ({self.tag_embed_dim},), got {tuple(tag_embed.shape)}"
            )
        h = self.node_enc(node_feat)
        dst = edge_index[1]
        if dst.numel() > 0:
            deg = torch.zeros(h.size(0), 1, device=h.device, dtype=h.dtype)
            deg.index_add_(0, dst, torch.ones(dst.size(0), 1, device=h.device, dtype=h.dtype))
            deg_inv = 1.0 / deg.clamp(min=1.0)
        else:
            deg_inv = None
        for film, layer in zip(self.films, self.layers):
            gamma, beta = film(tag_embed).chunk(2, dim=-1)
            h = layer(h, edge_index, edge_feat, gamma, beta, deg_inv=deg_inv)
        return h


class PropertyGNN(_SectorGraphBackbone):
    def __init__(self, edge_features: int, vocab: dict[str, int],
                 hidden: int = 128, n_layers: int = 4, tag_embed_dim: int = 256,
                 input_features: int = IN_FEATURES, side_heads: bool = True):
        super().__init__(edge_features, hidden=hidden, n_layers=n_layers,
                         tag_embed_dim=tag_embed_dim, input_features=input_features)
        self.vocab = dict(vocab)                       # {"flat","tex","special"} sizes
        self.heads = nn.ModuleDict({
            "floor_flat": nn.Linear(hidden, vocab["flat"]),
            "ceil_flat": nn.Linear(hidden, vocab["flat"]),
            "wall_tex": nn.Linear(hidden, vocab["tex"]),
            "special": nn.Linear(hidden, vocab["special"]),
        })
        self.side_heads = side_heads
        if side_heads:
            self.side_enc = nn.Sequential(nn.Linear(2 * hidden + 6, hidden), nn.SiLU())
            self.side_textures = nn.ModuleDict({field: nn.Linear(hidden, vocab['tex'])
                                               for field in ('mid_tex', 'upper_tex', 'lower_tex')})
            self.side_offsets = nn.Linear(hidden, 2)

    def forward(self, node_feat, edge_index, edge_feat, tag_embed) -> dict:
        """node_feat (N, IN_FEATURES), edge_index (2, E), edge_feat (E, Fe),
        tag_embed (tag_embed_dim,). Returns per-head logits (N, vocab)."""
        h = self._encode_nodes(node_feat, edge_index, edge_feat, tag_embed)
        result = {name: head(h) for name, head in self.heads.items()}
        if self.side_heads:
            result['sector_embedding'] = h
        return result

    def predict_sides(self, embeddings, owner, neighbor, features):
        if not self.side_heads:
            raise ValueError("PropertyGNN was built with side_heads=False")
        hidden = self.side_enc(torch.cat([embeddings[owner], embeddings[neighbor], features], dim=-1))
        result = {f'side_{name}': head(hidden) for name, head in self.side_textures.items()}
        result['side_offsets'] = torch.tanh(self.side_offsets(hidden)) * 32
        return result

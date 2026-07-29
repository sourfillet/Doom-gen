"""Stage 4 placement GNN (plan/generation.md, Stage 4): a tag-conditioned
sector-graph network that predicts, per sector, HOW MANY monsters/pickups
belong there and WHICH types - conditioned on sector geometry + neighbours +
tags. Replaces the corpus-density heuristic (one global density x area, one
global type frequency) with context: a big open room gets a horde, a tight
dark closet a lurker, ammo shows up where the fights are.

Reuses the Stage 3 message-passing backbone (_GNNLayer, INPUT_COLUMNS). The
exact POSITION of each thing is still chosen by the geometric placer
(in-polygon, wall clearance, reachability) - the GNN owns count + type, not
pixel coordinates.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.properties.gnn import IN_FEATURES, INPUT_COLUMNS, _GNNLayer  # noqa: F401

RATE_HEADS = ("monster_rate", "pickup_rate")
TYPE_HEADS = ("monster_type", "pickup_type")


class PlacementGNN(nn.Module):
    def __init__(self, edge_features: int, vocab: dict[str, int],
                 hidden: int = 128, n_layers: int = 4, tag_embed_dim: int = 256):
        super().__init__()
        self.vocab = dict(vocab)                       # {"monster","pickup"} sizes
        self.node_enc = nn.Linear(IN_FEATURES, hidden)
        self.films = nn.ModuleList(
            [nn.Linear(tag_embed_dim, hidden * 2) for _ in range(n_layers)])
        self.layers = nn.ModuleList(
            [_GNNLayer(hidden, edge_features) for _ in range(n_layers)])
        self.monster_rate = nn.Linear(hidden, 1)
        self.pickup_rate = nn.Linear(hidden, 1)
        self.monster_type = nn.Linear(hidden, vocab["monster"])
        self.pickup_type = nn.Linear(hidden, vocab["pickup"])

    def forward(self, node_feat, edge_index, edge_feat, tag_embed) -> dict:
        h = self.node_enc(node_feat)
        for film, layer in zip(self.films, self.layers):
            gamma, beta = film(tag_embed).chunk(2, dim=-1)
            h = layer(h, edge_index, edge_feat, gamma, beta)
        # softplus keeps the expected count non-negative (Poisson rate)
        return {
            "monster_rate": nn.functional.softplus(self.monster_rate(h)).squeeze(-1),
            "pickup_rate": nn.functional.softplus(self.pickup_rate(h)).squeeze(-1),
            "monster_type": self.monster_type(h),
            "pickup_type": self.pickup_type(h),
        }

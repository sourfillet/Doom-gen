"""Stage 4 placement GNN: a tag-conditioned
sector-graph network that predicts, per sector, HOW MANY monsters/pickups
belong there and WHICH types - conditioned on sector geometry + neighbours +
tags. Replaces the corpus-density heuristic (one global density x area, one
global type frequency) with context: a big open room gets a horde, a tight
dark closet a lurker, ammo shows up where the fights are.

Reuses the Stage 3 message-passing backbone (_GNNLayer, INPUT_COLUMNS). The
new models also predict local position and facing distributions. The geometric
placer rejects unsafe samples using containment, actor dimensions, reachability,
and solid collisions. Older checkpoints retain count/type-only heads.
"""
from __future__ import annotations

import torch.nn as nn

from models.properties.gnn import IN_FEATURES, INPUT_COLUMNS, _GNNLayer, _SectorGraphBackbone  # noqa: F401

RATE_HEADS = ("monster_rate", "pickup_rate")
TYPE_HEADS = ("monster_type", "pickup_type")


class PlacementGNN(_SectorGraphBackbone):
    def __init__(self, edge_features: int, vocab: dict[str, int],
                 hidden: int = 128, n_layers: int = 4, tag_embed_dim: int = 256,
                 input_features: int = IN_FEATURES, spatial: bool = True):
        super().__init__(edge_features, hidden=hidden, n_layers=n_layers,
                         tag_embed_dim=tag_embed_dim, input_features=input_features)
        self.vocab = dict(vocab)                       # {"monster","pickup"} sizes
        self.monster_rate = nn.Linear(hidden, 1)
        self.pickup_rate = nn.Linear(hidden, 1)
        self.monster_type = nn.Linear(hidden, vocab["monster"])
        self.pickup_type = nn.Linear(hidden, vocab["pickup"])
        self.spatial = spatial
        if spatial:
            self.monster_position = nn.Linear(hidden, 16 * 16)
            self.pickup_position = nn.Linear(hidden, 16 * 16)
            self.monster_angle = nn.Linear(hidden, 8)
            self.pickup_angle = nn.Linear(hidden, 8)

    def forward(self, node_feat, edge_index, edge_feat, tag_embed) -> dict:
        h = self._encode_nodes(node_feat, edge_index, edge_feat, tag_embed)
        # softplus keeps the expected count non-negative (Poisson rate)
        output = {
            "monster_rate": nn.functional.softplus(self.monster_rate(h)).squeeze(-1),
            "pickup_rate": nn.functional.softplus(self.pickup_rate(h)).squeeze(-1),
            "monster_type": self.monster_type(h),
            "pickup_type": self.pickup_type(h),
        }
        if self.spatial:
            output.update(monster_position=self.monster_position(h), pickup_position=self.pickup_position(h),
                          monster_angle=self.monster_angle(h), pickup_angle=self.pickup_angle(h))
        return output

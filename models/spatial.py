"""Fixed-scaffold GNN memorization control, not a free-running spatial sampler.

Per-slot learned coordinates/elevations make source memorization possible even
for automorphic vertices. Neither target coordinates nor heights enter forward.
"""
from __future__ import annotations

import pickle

import torch
from torch import nn

from data.spatial_prediction import scaffold_edges

CHECKPOINT_SCHEMA = 'doom_spatial_memorization_v2'


class SpatialScaffoldModel(nn.Module):
    def __init__(self, scaffold, hidden=32):
        super().__init__()
        self.slots = len(scaffold['slot_kinds'])
        nodes = self.slots + len(scaffold['rooms'])
        self.embedding = nn.Embedding(nodes, hidden)
        self.message = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(2))
        self.output = nn.Linear(hidden, 2)
        # Learned output slots are initialized independently of supervision.
        # They are explicitly a memorization control, not a geometry prior.
        self.numeric = nn.Parameter(torch.zeros(self.slots, 2))
        edges = scaffold_edges(scaffold)
        index = torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous()
        self.register_buffer('edges', index)
        degree = torch.zeros(nodes, 1)
        if index.numel():
            degree.index_add_(0, index[1], torch.ones(index.shape[1], 1))
        self.register_buffer('degree', degree.clamp_min(1))

    def forward(self):
        h = self.embedding.weight
        for layer in self.message:
            aggregation = torch.zeros_like(h)
            if self.edges.numel():
                aggregation.index_add_(0, self.edges[1], h[self.edges[0]])
            h = torch.tanh(layer(h + aggregation / self.degree))
        return self.numeric + 0.05 * self.output(h[:self.slots])


def predict_checkpoint(path, *, expected_scaffold=None):
    """Load versioned inference weights; never accept old unversioned v1 heads."""
    from data.spatial_prediction import PREDICTION_SCHEMA, SCHEMA, digest
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    except (pickle.UnpicklingError, EOFError, RuntimeError) as exc:
        raise ValueError('unreadable spatial checkpoint; expected a versioned v2 weights file') from exc
    if not isinstance(checkpoint, dict) or checkpoint.get('schema') != CHECKPOINT_SCHEMA:
        raise ValueError('unsupported spatial checkpoint; expected versioned scaffold memorization v2')
    if not {'scaffold', 'scaffold_sha256', 'hidden', 'model'} <= checkpoint.keys():
        raise ValueError('incomplete spatial checkpoint')
    scaffold = checkpoint['scaffold']
    if (not isinstance(scaffold, dict) or scaffold.get('schema') != SCHEMA
            or digest(scaffold) != checkpoint['scaffold_sha256']):
        raise ValueError('spatial checkpoint scaffold fingerprint mismatch')
    if expected_scaffold is not None and digest(expected_scaffold) != checkpoint['scaffold_sha256']:
        raise ValueError('spatial checkpoint belongs to another source scaffold')
    model = SpatialScaffoldModel(scaffold, hidden=checkpoint['hidden'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    with torch.no_grad():
        values = model().tolist()
    return scaffold, {'schema': PREDICTION_SCHEMA, 'values': values}

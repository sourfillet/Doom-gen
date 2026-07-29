"""Autoregressive room-graph generator (Phase 9.4, G1).

A GRU walks the BFS-ordered room sequence. Before emitting room i it predicts,
from the running context + the tag embedding: whether to STOP, the room's
binned attributes (floor step / height / size / aspect), which earlier room is
its PARENT (a pointer via dot-product attention over stored room states), and
the connection KIND. Trees only in v1, so every sampled graph is connected.

Plain torch (GRUCell + MLP heads + a dot-product pointer) - no torch_geometric.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from data.roomgraph_targets import (
    N_ASPECT, N_FLOOR, N_HEIGHT, N_KIND, N_SIZE, NONE_KIND_IDX)

_EMB = 32


class RoomGraphRNN(nn.Module):
    def __init__(self, hidden: int = 128, tag_dim: int = 256):
        super().__init__()
        self.hidden = hidden
        self.e_floor = nn.Embedding(N_FLOOR, _EMB)
        self.e_height = nn.Embedding(N_HEIGHT, _EMB)
        self.e_size = nn.Embedding(N_SIZE, _EMB)
        self.e_aspect = nn.Embedding(N_ASPECT, _EMB)
        self.e_kind = nn.Embedding(N_KIND, _EMB)
        self.gru = nn.GRUCell(_EMB * 5, hidden)
        self.init_h = nn.Linear(tag_dim, hidden)

        c = hidden + tag_dim                       # context = state + tag
        self.h_stop = nn.Linear(c, 1)
        self.h_floor = nn.Linear(c, N_FLOOR)
        self.h_height = nn.Linear(c, N_HEIGHT)
        self.h_size = nn.Linear(c, N_SIZE)
        self.h_aspect = nn.Linear(c, N_ASPECT)
        self.h_query = nn.Linear(c, hidden)        # pointer query for parent
        self.h_kind = nn.Linear(c + hidden, N_KIND)  # + chosen parent state

    def room_embed(self, floor, height, size, aspect, kind) -> torch.Tensor:
        idx = lambda v: torch.tensor(v, device=self.init_h.weight.device)
        return torch.cat([self.e_floor(idx(floor)), self.e_height(idx(height)),
                          self.e_size(idx(size)), self.e_aspect(idx(aspect)),
                          self.e_kind(idx(kind))], dim=-1)

    def heads(self, h, tag):
        c = torch.cat([h, tag], dim=-1)
        return {"stop": self.h_stop(c), "floor": self.h_floor(c),
                "height": self.h_height(c), "size": self.h_size(c),
                "aspect": self.h_aspect(c), "query": self.h_query(c), "c": c}

    def kind_logits(self, c, parent_state):
        return self.h_kind(torch.cat([c, parent_state], dim=-1))

    # -- training: teacher-forced negative log-likelihood over one sequence --
    def loss(self, seq: list[dict], tag: torch.Tensor,
             stop_pos_weight: float = 8.0) -> torch.Tensor:
        ce = nn.functional.cross_entropy
        # each graph has ~N 'continue' labels vs a single 'stop'; upweight the
        # positive (stop) class so the model actually learns to terminate
        pw = tag.new_tensor([stop_pos_weight])
        bce = lambda logit, tgt: nn.functional.binary_cross_entropy_with_logits(
            logit, tgt, pos_weight=pw)
        h = torch.tanh(self.init_h(tag))
        states: list[torch.Tensor] = []
        total = tag.new_zeros(())
        for i, step in enumerate(seq):
            out = self.heads(h, tag)
            if i > 0:                              # STOP=continue before each non-root
                total = total + bce(out["stop"], tag.new_zeros(1))
                total = total + ce(out["floor"][None], _t(step["floor"], tag))
                # parent pointer over states[0..i-1]
                q = out["query"]
                scores = torch.stack([q @ s for s in states])
                total = total + ce(scores[None], _t(step["parent"], tag))
                pstate = states[step["parent"]]
                total = total + ce(self.kind_logits(out["c"], pstate)[None],
                                   _t(step["kind"], tag))
            total = total + ce(out["height"][None], _t(step["height"], tag))
            total = total + ce(out["size"][None], _t(step["size"], tag))
            total = total + ce(out["aspect"][None], _t(step["aspect"], tag))
            emb = self.room_embed(step["floor"], step["height"], step["size"],
                                  step["aspect"], step["kind"])
            h = self.gru(emb[None], h[None])[0]
            states.append(h)
        # final STOP=stop
        total = total + bce(self.heads(h, tag)["stop"], tag.new_ones(1))
        return total / max(len(seq), 1)

    # -- sampling: one room at a time until STOP --
    @torch.no_grad()
    def sample(self, tag: torch.Tensor, max_rooms: int = 40,
               temperature: float = 1.0) -> list[dict]:
        def cat(logits):
            p = torch.softmax(logits / temperature, dim=-1)
            return int(torch.multinomial(p, 1))

        h = torch.tanh(self.init_h(tag))
        states: list[torch.Tensor] = []
        seq: list[dict] = []
        for i in range(max_rooms):
            out = self.heads(h, tag)
            if i > 0 and torch.sigmoid(out["stop"]).item() > 0.5:
                break
            step = {"height": cat(out["height"]), "size": cat(out["size"]),
                    "aspect": cat(out["aspect"])}
            if i == 0:
                from data.roomgraph_targets import ROOT_FLOOR_IDX
                step.update(floor=ROOT_FLOOR_IDX, parent=-1, kind=NONE_KIND_IDX)
            else:
                step["floor"] = cat(out["floor"])
                q = out["query"]
                scores = torch.stack([q @ s for s in states])
                parent = cat(scores)
                step["parent"] = parent
                kl = self.kind_logits(out["c"], states[parent])
                kl = kl.clone(); kl[NONE_KIND_IDX] = -1e9   # non-root can't be 'none'
                step["kind"] = cat(kl)
            seq.append(step)
            emb = self.room_embed(step["floor"], step["height"], step["size"],
                                  step["aspect"], step["kind"])
            h = self.gru(emb[None], h[None])[0]
            states.append(h)
        return seq


def _t(v: int, ref: torch.Tensor) -> torch.Tensor:
    return torch.tensor([v], device=ref.device)

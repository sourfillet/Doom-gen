"""Autoregressive room-graph generator (Phase 9.4, G1).

A GRU walks a DFS-ordered room sequence. Before emitting room i it predicts,
from the running context + the tag embedding: whether to STOP, the room's
binned attributes (floor step / height / size / aspect), which earlier room is
its PARENT (a pointer via dot-product attention over stored room states), and
the connection KIND. Version 3 adds loop edges, widths, and progression roles.

Plain torch (GRUCell + MLP heads + a dot-product pointer) - no torch_geometric.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from data.roomgraph_targets import (
    N_ASPECT, N_FLOOR, N_HEIGHT, N_KIND, N_SIZE, NONE_KIND_IDX, ROOMGRAPH_TARGET_VERSION, WIDTHS)

_EMB = 32


class RoomGraphRNN(nn.Module):
    def __init__(self, hidden: int = 128, tag_dim: int = 256, target_version: int = ROOMGRAPH_TARGET_VERSION):
        super().__init__()
        self.hidden = hidden
        self.target_version = target_version
        self.e_floor = nn.Embedding(N_FLOOR, _EMB)
        self.e_height = nn.Embedding(N_HEIGHT, _EMB)
        self.e_size = nn.Embedding(N_SIZE, _EMB)
        self.e_aspect = nn.Embedding(N_ASPECT, _EMB)
        self.e_kind = nn.Embedding(N_KIND, _EMB)
        if target_version >= 3:
            self.e_width = nn.Embedding(len(WIDTHS), _EMB)
            self.e_roles = nn.Embedding(8, _EMB)
        self.gru = nn.GRUCell(_EMB * (7 if target_version >= 3 else 5), hidden)
        self.init_h = nn.Linear(tag_dim, hidden)

        c = hidden + tag_dim                       # context = state + tag
        self.h_stop = nn.Linear(c, 1)
        self.h_floor = nn.Linear(c, N_FLOOR)
        self.h_height = nn.Linear(c, N_HEIGHT)
        self.h_size = nn.Linear(c, N_SIZE)
        self.h_aspect = nn.Linear(c, N_ASPECT)
        self.h_query = nn.Linear(c, hidden)        # pointer query for parent
        self.h_kind = nn.Linear(c + hidden, N_KIND)  # + chosen parent state
        if target_version >= 3:
            self.h_width = nn.Linear(c, len(WIDTHS))
            self.h_roles = nn.Linear(c, 8)
            self.h_extra = nn.Linear(c + hidden, N_KIND)
            self.h_extra_width = nn.Linear(c + hidden, len(WIDTHS))

    def room_embed(self, floor, height, size, aspect, kind, width=2, roles=0) -> torch.Tensor:
        idx = lambda v: torch.tensor(v, device=self.init_h.weight.device)
        values = [self.e_floor(idx(floor)), self.e_height(idx(height)),
                  self.e_size(idx(size)), self.e_aspect(idx(aspect)), self.e_kind(idx(kind))]
        if self.target_version >= 3:
            values.extend([self.e_width(idx(width)), self.e_roles(idx(roles))])
        return torch.cat(values, dim=-1)

    def heads(self, h, tag):
        c = torch.cat([h, tag], dim=-1)
        return {"stop": self.h_stop(c), "floor": self.h_floor(c),
                "height": self.h_height(c), "size": self.h_size(c),
                "aspect": self.h_aspect(c), "query": self.h_query(c), "c": c}

    def kind_logits(self, c, parent_state):
        return self.h_kind(torch.cat([c, parent_state], dim=-1))

    # -- training: teacher-forced negative log-likelihood over one sequence --
    def loss(self, seq: list[dict], tag: torch.Tensor,
             stop_pos_weight: float = 1.0, *, terminal: bool = True, stop: bool = True) -> torch.Tensor:
        """stop=False drops every STOP term (continue and end) and keeps structure terms.

        Trainers use it for source fragments when only complete maps define the
        stop hazard; a terminal label without stop terms is contradictory.
        """
        if not seq:
            raise ValueError("seq must contain at least one room")
        if terminal and not stop:
            raise ValueError("a terminal sequence needs its stop terms")
        ce = nn.functional.cross_entropy
        # Default unit weight preserves the Bernoulli sequence likelihood.
        pw = tag.new_tensor([stop_pos_weight])
        bce = lambda logit, tgt: nn.functional.binary_cross_entropy_with_logits(
            logit, tgt, pos_weight=pw)
        h = torch.tanh(self.init_h(tag))
        states: list[torch.Tensor] = []
        total = tag.new_zeros(())
        stop_total = tag.new_zeros(())
        for i, step in enumerate(seq):
            out = self.heads(h, tag)
            if i > 0:                              # STOP=continue before each non-root
                if stop:
                    stop_total = stop_total + bce(out["stop"], tag.new_zeros(1))
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
            if self.target_version >= 3:
                total = total + ce(self.h_width(out['c'])[None], _t(step.get('width', 2), tag))
                total = total + ce(self.h_roles(out['c'])[None], _t(step.get('roles', 0), tag))
                targets = {edge['parent']: edge for edge in step.get('extra_edges', [])}
                edge_losses = []
                for j, state in enumerate(states):
                    if j == step['parent']:
                        continue
                    context = torch.cat([out['c'], state])
                    edge = targets.get(j)
                    edge_loss = ce(self.h_extra(context)[None], _t(edge['kind'] if edge else NONE_KIND_IDX, tag))
                    if edge:
                        edge_loss = edge_loss + ce(self.h_extra_width(context)[None], _t(edge.get('width', 2), tag))
                    edge_losses.append(edge_loss)
                if edge_losses:
                    total = total + torch.stack(edge_losses).mean()
            emb = self.room_embed(step["floor"], step["height"], step["size"],
                                  step["aspect"], step["kind"], step.get('width', 2), step.get('roles', 0))
            h = self.gru(emb[None], h[None])[0]
            states.append(h)
        # final STOP=stop
        if terminal:
            stop_total = stop_total + bce(self.heads(h, tag)["stop"], tag.new_ones(1))
        # The stop hazard is a sequence likelihood. Dividing it by this
        # particular map's room count overweights short-map endings relative
        # to long-map continuations and biases free-running samples to stop.
        return total / max(len(seq), 1) + stop_total

    # -- sampling: one room at a time until STOP --
    @torch.no_grad()
    def sample(self, tag: torch.Tensor, max_rooms: int = 40,
               temperature: float = 1.0) -> list[dict]:
        if max_rooms < 1 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('max_rooms must be positive and temperature finite and positive')
        def cat(logits):
            if temperature < 1e-4:
                shifted = logits - logits.max()
                p = torch.softmax(shifted / max(temperature, 1e-12), dim=-1)
            else:
                p = torch.softmax(logits / temperature, dim=-1)
            return int(torch.multinomial(p, 1))

        h = torch.tanh(self.init_h(tag))
        states: list[torch.Tensor] = []
        seq: list[dict] = []
        if self.target_version >= 3:
            import networkx as nx
            from data.roomgraph_targets import KINDS
            physical = nx.Graph()
            teleport_kind = KINDS.index('teleport')
        for i in range(max_rooms):
            out = self.heads(h, tag)
            stop_prob = (
                (out["stop"] > 0).to(out["stop"].dtype)
                if temperature < 1e-4
                else torch.sigmoid(out["stop"] / temperature)
            )
            if i > 0 and (torch.rand((), device=tag.device) < stop_prob).item():
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
                kl = kl.clone()
                kl[NONE_KIND_IDX] = -torch.inf  # non-root can't be 'none'
                step["kind"] = cat(kl)
            if self.target_version >= 3:
                physical.add_node(i)
                if i > 0 and step['kind'] != teleport_kind:
                    physical.add_edge(i, step['parent'])
                roles = cat(self.h_roles(out['c']))
                step['roles'] = (roles | 1) if i == 0 else (roles & ~1)
                step['width'] = cat(self.h_width(out['c']))
                step['extra_edges'] = []
                for j, state in enumerate(states):
                    if j == step['parent']:
                        continue
                    context = torch.cat([out['c'], state])
                    logits = self.h_extra(context)
                    # Doom's ordinary room/corridor adjacency must fit in a
                    # plane. Teleports can connect otherwise crossing graphs.
                    # Mask impossible physical loop types before sampling.
                    physical.add_edge(i, j)
                    planar = nx.check_planarity(physical)[0]
                    physical.remove_edge(i, j)
                    if not planar:
                        logits = logits.clone()
                        for kind_index in range(N_KIND):
                            if kind_index not in {teleport_kind, NONE_KIND_IDX}:
                                logits[kind_index] = -torch.inf
                    kind = cat(logits)
                    if kind != NONE_KIND_IDX:
                        if kind != teleport_kind:
                            physical.add_edge(i, j)
                        step['extra_edges'].append({'parent': j, 'kind': kind,
                                                    'width': cat(self.h_extra_width(context))})
            seq.append(step)
            emb = self.room_embed(step["floor"], step["height"], step["size"],
                                  step["aspect"], step["kind"], step.get('width', 2), step.get('roles', 0))
            h = self.gru(emb[None], h[None])[0]
            states.append(h)
        return seq


def _t(v: int, ref: torch.Tensor) -> torch.Tensor:
    return torch.tensor([v], device=ref.device)

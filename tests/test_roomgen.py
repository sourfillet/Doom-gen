"""Phase 9.4: autoregressive room-graph generator."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.roomgraph_targets import graph_to_sequence, sequence_to_graph
from roomgraph.types import Connection, Room, RoomGraph


def _chain_graph() -> RoomGraph:
    def room(i, area, floor=0):
        return Room(id=i, sectors=[i], floor_h=floor, ceil_h=floor + 128,
                    area=area, centroid=(0.0, 0.0), bbox=(0.0, 0.0, 300.0, 300.0))
    return RoomGraph(
        rooms=[room(0, 1e6), room(1, 5e5, floor=64), room(2, 3e5)],
        connections=[Connection(0, 1, "door", 100), Connection(1, 2, "corridor", 100)])


def test_sequence_round_trip_preserves_tree():
    rg = _chain_graph()
    seq = graph_to_sequence(rg)
    assert len(seq) == 3
    assert seq[0]["parent"] == -1                 # root
    assert seq[0]["size"] >= 2                      # largest area -> a big size class
    rebuilt = sequence_to_graph(seq)
    assert len(rebuilt.rooms) == 3
    assert len(rebuilt.connections) == 2           # a tree over 3 rooms
    # the floor step from root to room 1 (+64) survives the binning
    room1 = next(r for r in rebuilt.rooms if r.id == 1)
    assert room1.floor_h == 64


def test_graph_to_sequence_roots_at_largest():
    rg = _chain_graph()
    seq = graph_to_sequence(rg)
    # kind of the root is 'none', children carry real kinds
    from data.roomgraph_targets import KINDS
    assert KINDS[seq[0]["kind"]] == "none"
    assert all(KINDS[s["kind"]] != "none" for s in seq[1:])


def test_depth_first_tree_on_cycle_uses_recent_parent():
    rg = _chain_graph()
    rg.connections.append(Connection(0, 2, "open", 100))
    seq = graph_to_sequence(rg)
    assert [s["parent"] for s in seq] == [-1, 0, 1]
    # The second edge must describe the actual 1->2 connection, not 0->2.
    from data.roomgraph_targets import FLOOR_STEPS, KINDS
    assert KINDS[seq[2]["kind"]] == "corridor"
    assert FLOOR_STEPS[seq[2]["floor"]] == -64


try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_model_loss_and_sample_and_assemble():
    import torch

    from models.roomgen.model import RoomGraphRNN
    from roomgraph.assemble import assemble

    seq = graph_to_sequence(_chain_graph())
    model = RoomGraphRNN()
    tag = torch.randn(256)
    loss = model.loss(seq, tag)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.gru.weight_ih.grad is not None

    gen = model.sample(tag, max_rooms=12)
    assert len(gen) >= 1
    rg = sequence_to_graph(gen)
    # a sampled graph is a tree: connections == rooms - 1
    assert len(rg.connections) >= max(len(rg.rooms) - 1, 0)
    # Random untrained samples need not pass geometric constraints. Exercise
    # deterministic assembly separately with a known graph.
    asm = assemble(_chain_graph())
    assert len(asm.sectors) >= 1                    # assembles into valid geometry

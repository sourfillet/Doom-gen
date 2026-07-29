"""Phase 9.4 generation: sample room-graphs from the trained generator,
assemble them into geometry, decorate, and write a playable PWAD.

This is the graph-native end-to-end path - the counterpart of
generation.pipeline.generate for the raster stack, but it produces coherent
rooms/corridors by construction. Decoration reuses the existing stages
(style textures/things, Stage 3 property GNN, Stage 4 placement, game
enforcement) since they all operate on a plain MapDocument.
"""
from __future__ import annotations

from pathlib import Path

from data.roomgraph_targets import sequence_to_graph
from generation.game_profile import GameProfile
from generation.placement import PlacementModel, apply_learned_placement
from generation.properties import PropertyModel, apply_properties
from generation.style import StyleModel, apply_style
from generation.validator import validate_and_repair
from roomgraph.assemble import assemble
from roomgraph.assemble_real import assemble_real
from roomgraph.shapes import apply_shapes, load_library
from roomgraph.types import RoomGraph
from wad_writer.wad_writer import build_nodes, write_wad


def load_roomgen_model(checkpoint_path: Path | str, device: str = "cpu"):
    import torch

    from models.conditioning import TagConditioner
    from models.roomgen.model import RoomGraphRNN

    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = RoomGraphRNN(hidden=ck["config"]["hidden"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    cond = TagConditioner(embed_dim=256).to(device)
    cond.load_state_dict(ck["conditioner"])
    cond.tag_to_index = ck["tag_to_index"]
    return model, cond


def sample_room_graph(model, conditioner, tags: dict[str, float],
                      seed: int | None = None, temperature: float = 1.0,
                      max_rooms: int = 40, map_name: str = "MAP01") -> RoomGraph:
    import torch

    if seed is not None:
        torch.manual_seed(seed)
    tag = conditioner.encode([tags], device="cpu")[0]
    seq = model.sample(tag, max_rooms=max_rooms, temperature=temperature)
    return sequence_to_graph(seq, map_name=map_name)


def generate(checkpoint_path: Path | str, tags: dict[str, float], n_maps: int = 1,
             out_path: Path | str = "generated_roomgen.wad", seed: int | None = None,
             temperature: float = 1.0, style: StyleModel | None = None,
             game: GameProfile | None = None, properties: PropertyModel | None = None,
             placement: PlacementModel | None = None,
             shape_library: dict | str | None = None,
             max_rooms: int = 40) -> dict:
    model, conditioner = load_roomgen_model(checkpoint_path)
    # real-geometry path: dress each generated node in a real room polygon
    # (retrieved by aspect/size) and assemble organic rooms + carved corridors
    if isinstance(shape_library, str):
        shape_library = load_library(shape_library)
    docs, summaries = [], []
    for i in range(n_maps):
        name = f"MAP{i + 1:02d}"
        s = None if seed is None else seed + i
        rg = sample_room_graph(model, conditioner, tags, seed=s,
                               temperature=temperature, max_rooms=max_rooms,
                               map_name=name)
        if shape_library is not None:
            apply_shapes(rg, shape_library, seed=s or 0)
            doc = assemble_real(rg, seed=s or 0)
        else:
            doc = assemble(rg, map_name=name)
        repairs = validate_and_repair(doc)
        if style:
            apply_style(doc, style, seed=s, place_things=placement is None)
        if placement:
            apply_learned_placement(doc, placement, tags, seed=s)
        if properties:
            apply_properties(doc, properties, tags)
        if game:
            game.enforce(doc)
        docs.append(doc)
        summaries.append({"map": name, "rooms": len(rg.rooms),
                          "sectors": len(doc.sectors), "repairs": repairs})
    path = write_wad(docs, out_path)
    return {"path": str(path), "maps": [d.meta.map_name for d in docs],
            "nodes_built": build_nodes(path), "summaries": summaries}

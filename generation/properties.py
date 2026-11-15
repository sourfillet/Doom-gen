"""Stage 3 inference: run the trained property GNN over a generated map's
sector graph and write its predicted flats / wall textures / sector specials
onto the document. Replaces the corpus-frequency texture assignment in the
style pass with context-aware, learned choices.

torch is imported lazily so the rest of the package still works without it.
"""
from __future__ import annotations

from pathlib import Path

from wad_parser.types import MapDocument

_NON_ASSET = ("-", "")


class PropertyModel:
    """Loaded GNN + tag conditioner + vocabularies, ready to decorate."""

    def __init__(self, model, conditioner, vocabs: dict, device: str):
        self.model = model
        self.conditioner = conditioner
        self.vocabs = vocabs
        self.device = device
        self.inv_flat = {i: n for n, i in vocabs["flat"].items()}
        self.inv_tex = {i: n for n, i in vocabs["tex"].items()}
        self.inv_special = {i: int(k) for k, i in vocabs["special"].items()}


def load_property_model(checkpoint_path: Path | str,
                        device: str | None = None) -> PropertyModel:
    import torch

    from models.conditioning import TagConditioner
    from models.properties.gnn import PropertyGNN

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = PropertyGNN(edge_features=cfg["edge_features"], vocab=cfg["vocab_sizes"],
                        hidden=cfg["hidden"], n_layers=cfg["layers"], input_features=cfg.get("input_features", 7),
                        side_heads=cfg.get('side_heads', False)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    conditioner = TagConditioner(embed_dim=256).to(device)
    conditioner.load_checkpoint_state(ck)
    return PropertyModel(model, conditioner, ck["vocabs"], device)


def forward_sector_graph(pm, graph: dict, tags: dict[str, float]) -> dict:
    """Run a loaded property/placement GNN over a data.graph_builder graph."""
    import torch

    from models.properties.gnn import INPUT_COLUMNS, LEGACY_INPUT_COLUMNS

    columns = INPUT_COLUMNS if pm.model.node_enc.in_features == len(INPUT_COLUMNS) else LEGACY_INPUT_COLUMNS
    nf = torch.from_numpy(graph["node_features"][:, columns].copy()).to(pm.device)
    ei = torch.from_numpy(graph["edge_index"]).to(pm.device)
    ef = torch.from_numpy(graph["edge_features"]).to(pm.device)
    cond = pm.conditioner.encode([tags], device=pm.device)[0]
    with torch.no_grad():
        return pm.model(nf, ei, ef, cond)


def apply_properties(doc: MapDocument, pm: PropertyModel,
                     tags: dict[str, float], temperature: float = 1.0,
                     special_temperature: float = 0.5, game=None,
                     seed: int | None = None) -> dict:
    """Predict per-sector flats/wall-texture/special and write them onto doc.
    UNK predictions (index 0) leave the existing value in place, so this only
    ever replaces with a real learned choice.

    Flats/textures are SAMPLED from the predicted distribution (not argmax):
    argmax collapses a peaky texture head onto one dominant class, making
    output more monotonous than the corpus. Sampling keeps the learned
    context while restoring variety. Specials use a low temperature (near
    argmax) since a wrong special mislabels a sector."""
    import math
    import torch

    from data.graph_builder import build_graph

    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError(f"temperature must be finite and non-negative, got {temperature!r}")
    if not math.isfinite(special_temperature) or special_temperature < 0:
        raise ValueError(f"special_temperature must be finite and non-negative, got {special_temperature!r}")

    g = build_graph(doc, count_things=False)
    if g["node_features"].shape[0] == 0:
        return {"mode": "gnn", "sectors": 0}
    logits = forward_sector_graph(pm, g, tags)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=pm.device)
        generator.manual_seed(int(seed))

    def sample(head: str, temp: float, supplied=None) -> list[int]:
        values = (logits[head] if supplied is None else supplied).clone()
        allowed = set(range(values.shape[-1]))
        if head == 'special':
            # Damage, movers, and port specials require a progression plan.
            allowed = {index for index, number in pm.inv_special.items() if number in {0, 1, 2, 3, 8, 9, 17}}
        elif game is not None:
            texture_head = head == 'wall_tex' or head.startswith('side_')
            vocabulary = pm.inv_tex if texture_head else pm.inv_flat
            valid_assets = game.textures | {'-'} if texture_head else game.flats | {'F_SKY1'}
            allowed = {index for index, name in vocabulary.items() if name in valid_assets}
        allowed.add(0)
        invalid = [index for index in range(values.shape[-1]) if index not in allowed]
        values[:, invalid] = -torch.inf
        probs = torch.softmax(values / max(temp, 1e-3), dim=-1)
        return torch.multinomial(probs, 1, generator=generator).squeeze(-1).tolist()

    floor = sample("floor_flat", temperature)
    ceil = sample("ceil_flat", temperature)
    tex = sample("wall_tex", temperature)
    spec = sample("special", special_temperature)

    wall_by_sector: dict[int, str] = {}
    for s in doc.sectors:
        f = pm.inv_flat.get(floor[s.id], "UNK")
        c = pm.inv_flat.get(ceil[s.id], "UNK")
        if f != "UNK":
            s.floor_flat = f
        if c != "UNK":
            s.ceil_flat = c
        s.special = pm.inv_special.get(spec[s.id], 0)
        w = pm.inv_tex.get(tex[s.id], "UNK")
        wall_by_sector[s.id] = "" if w in {'UNK', '-'} else w

    switch_sides = {
        ld.right_sidedef
        for ld in doc.linedefs
        if ld.special and ld.trigger_type == "switch" and 0 <= ld.right_sidedef < len(doc.sidedefs)
    }

    if pm.model.side_heads and doc.sidedefs:
        from data.property_targets import side_sample
        sample_data = side_sample(doc)
        with torch.no_grad():
            side_output = pm.model.predict_sides(logits['sector_embedding'],
                torch.from_numpy(sample_data['side_owner']).to(pm.device),
                torch.from_numpy(sample_data['side_neighbor']).to(pm.device),
                torch.from_numpy(sample_data['side_features']).to(pm.device))
        for field in ('mid_tex', 'upper_tex', 'lower_tex'):
            choices = sample(f'side_{field}', temperature, supplied=side_output[f'side_{field}'])
            for side, choice in zip(doc.sidedefs, choices):
                if field == 'mid_tex' and side.id in switch_sides and side.mid_tex.startswith(('SW1', 'SW2')):
                    continue
                value = pm.inv_tex.get(choice, 'UNK')
                # Preserve absence/presence required by the assembled geometry.
                if value not in {'UNK', '-'} and getattr(side, field) not in _NON_ASSET:
                    setattr(side, field, value)
        offsets = (side_output['side_offsets'] * 1024).round().clamp(-32768, 32767).to(torch.int64).tolist()
        for side, (x, y) in zip(doc.sidedefs, offsets):
            side.x_offset, side.y_offset = x, y
        return {'mode': 'gnn', 'sectors': len(doc.sectors), 'sidedefs': len(doc.sidedefs)}

    for sd in doc.sidedefs:
        w = wall_by_sector.get(sd.sector, "")
        if not w:
            continue
        if sd.mid_tex not in _NON_ASSET and not (sd.id in switch_sides and sd.mid_tex.startswith(('SW1', 'SW2'))):
            sd.mid_tex = w
        if sd.upper_tex not in _NON_ASSET:
            sd.upper_tex = w
        if sd.lower_tex not in _NON_ASSET:
            sd.lower_tex = w
    return {"mode": "gnn", "sectors": len(doc.sectors)}

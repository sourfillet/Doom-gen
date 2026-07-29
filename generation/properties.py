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
                        hidden=cfg["hidden"], n_layers=cfg["layers"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    conditioner = TagConditioner(embed_dim=256).to(device)
    conditioner.load_state_dict(ck["conditioner"])
    conditioner.tag_to_index = ck["tag_to_index"]
    return PropertyModel(model, conditioner, ck["vocabs"], device)


def apply_properties(doc: MapDocument, pm: PropertyModel,
                     tags: dict[str, float], temperature: float = 1.0,
                     special_temperature: float = 0.5) -> dict:
    """Predict per-sector flats/wall-texture/special and write them onto doc.
    UNK predictions (index 0) leave the existing value in place, so this only
    ever replaces with a real learned choice.

    Flats/textures are SAMPLED from the predicted distribution (not argmax):
    argmax collapses a peaky texture head onto one dominant class, making
    output more monotonous than the corpus. Sampling keeps the learned
    context while restoring variety. Specials use a low temperature (near
    argmax) since a wrong special mislabels a sector."""
    import torch

    from data.graph_builder import build_graph
    from models.properties.gnn import INPUT_COLUMNS

    g = build_graph(doc, count_things=False)
    if g["node_features"].shape[0] == 0:
        return {"mode": "gnn", "sectors": 0}
    nf = torch.from_numpy(g["node_features"][:, INPUT_COLUMNS].copy()).to(pm.device)
    ei = torch.from_numpy(g["edge_index"]).to(pm.device)
    ef = torch.from_numpy(g["edge_features"]).to(pm.device)
    cond = pm.conditioner.encode([tags], device=pm.device)[0]
    with torch.no_grad():
        logits = pm.model(nf, ei, ef, cond)

    def sample(head: str, temp: float) -> list[int]:
        probs = torch.softmax(logits[head] / max(temp, 1e-3), dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1).tolist()

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
        wall_by_sector[s.id] = "" if w == "UNK" else w

    for sd in doc.sidedefs:
        w = wall_by_sector.get(sd.sector, "")
        if not w:
            continue
        if sd.mid_tex not in _NON_ASSET:
            sd.mid_tex = w
        if sd.upper_tex not in _NON_ASSET:
            sd.upper_tex = w
        if sd.lower_tex not in _NON_ASSET:
            sd.lower_tex = w
    return {"mode": "gnn", "sectors": len(doc.sectors)}

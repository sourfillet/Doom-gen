"""Stage 4 inference: run the trained placement GNN and populate a generated
map's reachable sectors with the counts and monster/pickup types it predicts.

Division of labour: the GNN owns HOW MANY and WHICH TYPE per sector; the
proven geometric placer here owns WHERE - a point inside the sector polygon,
clear of walls, and (for monsters) off the player start. Only reachable,
big-enough sectors are populated (same rule as the heuristic placer).

torch is imported lazily.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from generation.properties import forward_sector_graph
from generation.style import _add_thing
from wad_parser.types import MapDocument
from wad_parser.geometry import sector_points, sector_area_perimeter, point_in_rings
from generation.actors import spawn_problem, placement_margin

_MAX_PER_SECTOR = 24


class PlacementModel:
    def __init__(self, model, conditioner, vocabs: dict, device: str):
        self.model = model
        self.conditioner = conditioner
        self.vocabs = vocabs
        self.device = device
        self.monster_types = vocabs["monster"]
        self.pickup_types = vocabs["pickup"]


def load_placement_model(checkpoint_path: Path | str,
                         device: str | None = None) -> PlacementModel:
    import torch

    from models.conditioning import TagConditioner
    from models.placement.gnn import PlacementGNN

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = PlacementGNN(edge_features=cfg["edge_features"], vocab=cfg["vocab_sizes"],
                         hidden=cfg["hidden"], n_layers=cfg["layers"], input_features=cfg.get("input_features", 7),
                         spatial=cfg.get('spatial', False)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    conditioner = TagConditioner(embed_dim=256).to(device)
    conditioner.load_checkpoint_state(ck)
    return PlacementModel(model, conditioner, ck["vocabs"], device)


def _reachable_sectors(doc: MapDocument) -> set[int]:
    from wad_parser.traversal import distances_from_start
    return set(distances_from_start(doc))


def apply_learned_placement(doc: MapDocument, pm: PlacementModel,
                            tags: dict[str, float], seed: int | None = None, game=None) -> dict:
    import torch

    from data.graph_builder import build_graph

    rng = np.random.default_rng(seed)
    reachable = _reachable_sectors(doc)
    from wad_parser.traversal import component_distances
    reachable_components = component_distances(doc)
    areas, rings = {}, {}
    for s in doc.sectors:
        pts = sector_points(doc, s.id)
        a, _ = sector_area_perimeter(doc, s.id)
        if a >= 64 * 64 and (s.ceil_height - s.floor_height) >= 56:
            areas[s.id], rings[s.id] = a, pts
    place_ids = [i for i in areas if i in reachable]
    if not place_ids:
        return {"mode": "gnn-placement", "monsters": 0, "pickups": 0,
                "requested_monsters": 0, "requested_pickups": 0}

    g = build_graph(doc, count_things=False)
    out = forward_sector_graph(pm, g, tags)
    m_rate = out["monster_rate"].cpu().numpy()
    p_rate = out["pickup_rate"].cpu().numpy()
    m_probs = torch.softmax(out["monster_type"], dim=-1).cpu().numpy()
    p_probs = torch.softmax(out["pickup_type"], dim=-1).cpu().numpy()
    start = next((t for t in doc.things if t.type == 1), None)
    from generation.actors import actor_profile
    from wad_parser.actions import KEY_COLORS
    from wad_parser.clearance import MapClearance
    geometry = MapClearance(doc.vertices, doc.sectors, doc.linedefs, doc.sidedefs)
    spatial_pos = (
        {k: torch.softmax(out[f"{k}_position"], dim=-1).cpu().numpy() for k in ("monster", "pickup")}
        if pm.model.spatial else {}
    )
    spatial_ang = (
        {k: torch.softmax(out[f"{k}_angle"], dim=-1).cpu().numpy() for k in ("monster", "pickup")}
        if pm.model.spatial else {}
    )

    def place(sid, count, probs, types, min_start, kind):
        if count <= 0 or not types:
            return 0
        pts = rings[sid]
        allowed = np.array([
            actor_profile(number) is not None
            and (kind != "pickup" or number not in KEY_COLORS)
            and (game is None or game.sub_thing(number) == number)
            for number in types
        ])
        probabilities = probs[sid] * allowed
        if probabilities.sum() <= 0:
            return 0
        probabilities /= probabilities.sum()
        points = [point for ring in pts for point in ring]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        placed = 0
        position_probs = spatial_pos.get(kind)
        angle_probs = spatial_ang.get(kind)
        for _ in range(count):
            ti = int(rng.choice(len(types), p=probabilities))
            type_num = types[ti]
            for _try in range(16):
                if position_probs is None:
                    x, y = round(rng.uniform(min(xs), max(xs))), round(rng.uniform(min(ys), max(ys)))
                else:
                    row, col = divmod(int(rng.choice(16 * 16, p=position_probs[sid])), 16)
                    x = round(min(xs) + (max(xs) - min(xs)) * (col + rng.random()) / 16)
                    y = round(min(ys) + (max(ys) - min(ys)) * (row + rng.random()) / 16)
                if not point_in_rings(x, y, pts):
                    continue
                if spawn_problem(doc, type_num, x, y, margin=placement_margin(type_num),
                                 reachable=reachable_components, geometry=geometry):
                    continue
                if start is not None and min_start > 0 and \
                        (x - start.x) ** 2 + (y - start.y) ** 2 < min_start ** 2:
                    continue
                angle = int(rng.choice(8, p=angle_probs[sid])) * 45 if angle_probs is not None else int(rng.integers(0, 8)) * 45
                _add_thing(doc, x, y, type_num, angle=angle)
                placed += 1
                break
        return placed

    mons = picks = requested_mons = requested_picks = 0
    for sid in place_ids:
        nm = int(np.clip(round(float(m_rate[sid])), 0, _MAX_PER_SECTOR))
        npk = int(np.clip(round(float(p_rate[sid])), 0, _MAX_PER_SECTOR))
        requested_mons += nm
        requested_picks += npk
        mons += place(sid, nm, m_probs, pm.monster_types, 256, 'monster')
        picks += place(sid, npk, p_probs, pm.pickup_types, 0, 'pickup')
    return {"mode": "gnn-placement", "monsters": mons, "pickups": picks,
            "requested_monsters": requested_mons, "requested_pickups": requested_picks}

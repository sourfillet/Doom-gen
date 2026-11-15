"""Phase 9.4 generation: sample room-graphs from the trained generator,
assemble them into geometry, decorate, and write a playable PWAD.

This is the graph-native end-to-end path. Required connections must be realized
and final geometry must pass validation; failed samples are reported with a
bounded retry budget. Decoration reuses the existing stages
(style textures/things, Stage 3 property GNN, Stage 4 placement, game
enforcement) since they all operate on a plain MapDocument.
"""
from __future__ import annotations

import math
from pathlib import Path

from data.roomgraph_targets import sequence_to_graph
from generation.game_profile import GameProfile
from generation.placement import PlacementModel, apply_learned_placement
from generation.properties import PropertyModel, apply_properties
from generation.progression import ensure_exit
from generation.style import StyleModel, apply_style
from generation.validator import validate_and_repair, require_valid
from roomgraph.assemble import assemble
from roomgraph.assemble_real import assemble_real
from roomgraph.layout import radial_layout
from roomgraph.shapes import apply_shapes, load_library
from roomgraph.types import RoomGraph
from roomgraph.variation import PROFILES
from roomgraph.teleports import require_connections
from wad_parser.map_extractor import compute_derived
from wad_writer.assembler import fit_coordinates
from generation.pipeline import MapBatch, NoUsableMapsError


def load_roomgen_model(checkpoint_path: Path | str, device: str | None = None):
    import torch

    from models.conditioning import TagConditioner
    from models.roomgen.model import RoomGraphRNN

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = RoomGraphRNN(hidden=ck["config"]["hidden"],
                         target_version=ck["config"].get("target_version", 1)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    cond = TagConditioner(embed_dim=256).to(device)
    cond.load_checkpoint_state(ck)
    return model, cond


def sample_room_graph(model, conditioner, tags: dict[str, float],
                      seed: int | None = None, temperature: float = 1.0,
                      max_rooms: int = 40, map_name: str = "MAP01") -> RoomGraph:
    import torch

    if seed is not None:
        torch.manual_seed(seed)
    tag = conditioner.encode([tags], device=next(model.parameters()).device)[0]
    seq = model.sample(tag, max_rooms=max_rooms, temperature=temperature)
    return sequence_to_graph(seq, map_name=map_name, target_version=model.target_version)


def generate(checkpoint_path: Path | str, tags: dict[str, float], n_maps: int = 1,
             out_path: Path | str = "generated_roomgen.wad", seed: int | None = None,
             temperature: float = 1.0, style: StyleModel | None = None,
             game: GameProfile | None = None, properties: PropertyModel | None = None,
             placement: PlacementModel | None = None,
             shape_library: dict | str | None = None,
             max_rooms: int = 40, max_attempts: int = 3, target_port: str = 'zdoom',
             geometry_profile: str = 'classic', gameplay: str = 'none',
             interior_profile: str = 'columns', progression: str = 'none',
             assets: AssetBundle | None = None, sources: list[Path | str] | None = None,
             companion_textures: Path | str | None = None) -> dict:
    if progression not in {'none', 'blue-key'}:
        raise ValueError('progression must be none or blue-key')
    if progression != 'none' and gameplay != 'intro':
        raise ValueError('blue-key progression requires intro gameplay')
    from generation.interiors import INTERIOR_PROFILES
    if interior_profile not in INTERIOR_PROFILES:
        raise ValueError('interior_profile must be columns or stepped')
    if interior_profile != 'columns' and gameplay != 'intro':
        raise ValueError('stepped interiors require intro gameplay')
    if gameplay not in {'none', 'intro'}:
        raise ValueError('gameplay must be none or intro')
    if gameplay == 'intro' and (style is None or placement is not None):
        raise ValueError('intro gameplay requires a corpus style and cannot be combined with learned placement')
    if geometry_profile not in PROFILES:
        raise ValueError(f'unknown geometry profile: {geometry_profile}')
    if shape_library is not None and geometry_profile != 'classic':
        raise ValueError('geometry_profile applies to the routed assembler; omit shape_library to use it')
    from wad_writer.compatibility import TARGET_PORTS
    if target_port not in TARGET_PORTS:
        raise ValueError('target_port must be zdoom or vanilla')
    if not 1 <= n_maps <= 99 or max_rooms < 1:
        raise ValueError("n_maps must be between 1 and 99 and max_rooms must be positive")
    if max_attempts < 1:
        raise ValueError('max_attempts must be positive')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    from generation.pipeline import _register_assets, _source_assets
    game = _register_assets(game, assets, target_port)
    resource_context = _source_assets(game, assets, sources, target_port)
    if resource_context is not None:
        game = resource_context.game
    model, conditioner = load_roomgen_model(checkpoint_path)
    # real-geometry path: dress each generated node in a real room polygon
    # (retrieved by aspect/size) and assemble organic rooms + carved corridors
    if isinstance(shape_library, str):
        shape_library = load_library(shape_library)
    def build_one(name, s):
        rg = sample_room_graph(model, conditioner, tags, seed=s,
                               temperature=temperature, max_rooms=max_rooms,
                               map_name=name)
        geometry_policy = None
        if gameplay == 'intro':
            from generation.gameplay import prepare_intro_graph
            rg, geometry_policy = prepare_intro_graph(rg)
        if shape_library is not None:
            apply_shapes(rg, shape_library, seed=s or 0)
            # Layout realizes the model's requested edges; no extra heuristic
            # loops are added behind the model's back.
            pos = radial_layout(rg, seed=s or 0)
            doc = assemble_real(rg, positions=pos, seed=s or 0)
        else:
            doc = assemble(rg, map_name=name, geometry_profile=geometry_profile, seed=s or 0)
        require_connections(doc)
        translation = fit_coordinates(doc)
        repairs = validate_and_repair(doc)
        if translation:
            repairs.insert(0, translation)
        decoration = {}
        if style:
            decoration = apply_style(doc, style, seed=s, place_things=placement is None and gameplay == 'none',
                                     place_dm_starts=gameplay == 'none')
        if placement:
            decoration['placement'] = apply_learned_placement(doc, placement, tags, seed=s, game=game)
        if properties:
            decoration['properties'] = apply_properties(doc, properties, tags, game=game, seed=s)
        if resource_context is not None:
            resource_context.resolve(doc)
            batch.assets = resource_context.bundle
        if game:
            decoration['game_enforcement'] = game.enforce(doc)
        exit_repair = ensure_exit(doc)
        if exit_repair:
            repairs.append(exit_repair)
        if gameplay == 'intro':
            from generation.gameplay import apply_intro
            palette = decoration.get('wall_palette', [])
            accent = palette[-1] if palette else None
            if game and accent:
                accent = game.sub_texture(accent)
            decoration['gameplay'] = apply_intro(doc, seed=s, game=game, accent_texture=accent,
                                                interior_profile=interior_profile, required_progression=progression)
            if progression == 'blue-key':
                from generation.keyed import add_blue_key
                add_blue_key(doc, seed=s, game=game)
        doc.extra['decoration'] = decoration
        if geometry_policy:
            doc.extra['gameplay_geometry'] = geometry_policy
        doc.bounds, doc.derived = compute_derived(
            doc.vertices, doc.sectors, doc.linedefs, doc.sidedefs, doc.things)
        translation = fit_coordinates(doc)
        if translation:
            repairs.append(translation)
        require_valid(doc, require_connected=True)
        return rg, doc, repairs

    summaries, failures = [], []
    if max_attempts < 1:
        raise ValueError('max_attempts must be positive')
    with MapBatch(out_path, assets=assets, target_port=target_port) as batch:
        for i in range(n_maps):
            name = f'MAP{len(summaries) + 1:02d}'
            for attempt in range(max_attempts):
                s = None if seed is None else seed + i * max_attempts + attempt
                stage = 'validation'
                try:
                    rg, doc, repairs = build_one(name, s)
                    stage = 'publication'
                    publication = batch.add(doc)
                except ValueError as exc:
                    failures.append({'sample': i, 'attempt': attempt + 1, 'seed': s,
                                     'reason': str(exc), 'failure_stage': stage,
                                     'validated': stage == 'publication'})
                    continue
                summaries.append({'map': name, 'rooms': len(rg.rooms), 'sectors': len(doc.sectors),
                                  'seed': s, 'repairs': repairs, 'publication': publication,
                                  'geometry_profile': doc.extra.get('geometry_profile', 'source-outlines'),
                                  'geometry_variation': doc.extra.get('geometry_variation'),
                                  'decoration': doc.extra.get('decoration', {}),
                                  'gameplay_geometry': doc.extra.get('gameplay_geometry'),
                                  'routing': doc.extra.get('routing'),
                                  'connection_realization': doc.extra.get('connection_realization', [])})
                break
        if not summaries:
            raise NoUsableMapsError(failures, n_maps)
        result = batch.publish(companion_textures=companion_textures)
    if resource_context is not None:
        result['asset_resolution'] = resource_context.report()
    return {**result, "summaries": summaries, "failed_samples": failures, "requested_maps": n_maps}

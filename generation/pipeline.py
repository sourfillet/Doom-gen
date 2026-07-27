"""End-to-end generation: sample rasters -> vectorize -> validate ->
write a playable PWAD (plan/generation.md, Stage 1 -> 2 -> validate ->
write; Stages 3/4 are heuristic placeholders at this point - default
textures and a deterministic player start).

Also provides regenerate_from_json(): re-runs the raster->vector->WAD
back half on an EXISTING extracted map. That path is the pipeline's test
harness, not generation - it isolates vectorizer/writer bugs from model
bugs, because its input raster is known-good.
"""
from __future__ import annotations

import json
from pathlib import Path

from data.rasterizer import rasterize
from generation.validator import validate_and_repair
from generation.vectorizer import vectorize
from wad_parser.types import MapDocument
from wad_writer.wad_writer import build_nodes, write_wad


def raster_to_document(raster, map_name: str = "MAP01",
                       cell_units: int = 64, coarse_factor: int = 4) -> tuple[MapDocument, list[str]]:
    doc = vectorize(raster, cell_units=cell_units, coarse_factor=coarse_factor,
                    map_name=map_name)
    repairs = validate_and_repair(doc)
    return doc, repairs


def documents_to_wad(docs: list[MapDocument], out_path: Path | str,
                     try_nodes: bool = True) -> dict:
    path = write_wad(docs, out_path)
    nodes_built = build_nodes(path) if try_nodes else False
    return {"path": str(path), "maps": [d.meta.map_name for d in docs],
            "nodes_built": nodes_built}


def regenerate_from_json(json_path: Path | str, out_path: Path | str,
                         resolution: int = 128, coarse_factor: int = 4) -> dict:
    """Extracted-map JSON -> raster -> vectorize -> validate -> WAD.
    Model-free harness for the back half of the pipeline.

    Unlike model samples (which have no ground-truth scale and use the
    default cell size), the raster meta here carries the exact world->grid
    scale, so the regenerated map comes back at the ORIGINAL map's true
    world scale."""
    doc = MapDocument.from_dict(json.loads(Path(json_path).read_text()))
    raster, meta = rasterize(doc, resolution=resolution)
    cell_units = max(8, int(round(coarse_factor / meta["scale"] / 8)) * 8)
    new_doc, repairs = raster_to_document(raster, map_name=doc.meta.map_name,
                                          cell_units=cell_units,
                                          coarse_factor=coarse_factor)
    result = documents_to_wad([new_doc], out_path)
    result["repairs"] = repairs
    result["sectors"] = len(new_doc.sectors)
    result["linedefs"] = len(new_doc.linedefs)
    result["cell_units"] = cell_units
    return result


def generate(checkpoint_path: Path | str, tag_weights: dict[str, float],
             n_maps: int = 1, out_path: Path | str = "generated.wad",
             steps: int = 50, guidance: float = 2.0,
             seed: int | None = None) -> dict:
    """Model-driven generation. Imports torch lazily so the rest of the
    package works without it installed."""
    import torch

    from models.checkpoint import load_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, diffusion, conditioner, config = load_checkpoint(checkpoint_path, device)

    if seed is not None:
        torch.manual_seed(seed)
    cond = conditioner.encode([tag_weights] * n_maps, device=device)
    uncond = torch.zeros_like(cond)
    with torch.no_grad():
        x = diffusion.ddim_sample(
            (n_maps, config["in_channels"], config["resolution"], config["resolution"]),
            cond, uncond, steps=steps, guidance=guidance, device=device,
        )
    rasters = ((x.clamp(-1, 1) + 1) / 2).cpu().numpy()  # model space -> [0,1]

    docs, all_repairs, failed = [], {}, []
    for i in range(n_maps):
        name = f"MAP{len(docs) + 1:02d}"
        try:
            doc, repairs = raster_to_document(rasters[i], map_name=name)
        except ValueError as exc:
            failed.append(f"sample {i}: {exc}")
            continue
        docs.append(doc)
        all_repairs[name] = repairs
    if not docs:
        raise RuntimeError(
            "no sample vectorized into a usable map: " + "; ".join(failed))
    result = documents_to_wad(docs, out_path)
    result["repairs"] = all_repairs
    result["failed_samples"] = failed
    result["tags"] = tag_weights
    return result

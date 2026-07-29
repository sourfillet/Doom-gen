"""Stage 4 training CLI: the tag-conditioned placement GNN (plan/
generation.md, Stage 4).

    train-placement --maps-dir dataset/maps --tags-dir tags --out checkpoints/

Count heads use Poisson NLL; type heads use a masked soft cross-entropy over
each sector's real type distribution (sectors with no monsters/pickups don't
supervise the corresponding type head). CPU by default - small model/graphs.
"""
from __future__ import annotations

import time
from pathlib import Path

import click
import numpy as np


def _poisson_nll(rate, target):
    import torch
    return (rate - target * torch.log(rate + 1e-6)).mean()


def _soft_ce(logits, target_counts):
    """Masked soft cross-entropy: only sectors with >=1 thing contribute,
    each against its normalized type distribution."""
    import torch
    totals = target_counts.sum(dim=-1)
    mask = totals > 0
    if mask.sum() == 0:
        return logits.sum() * 0.0
    dist = target_counts[mask] / totals[mask].unsqueeze(-1)
    logp = torch.log_softmax(logits[mask], dim=-1)
    return -(dist * logp).sum(dim=-1).mean()


@click.command()
@click.option("--maps-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--tags-dir", default=None, type=click.Path(file_okay=False))
@click.option("--out", "out_dir", default="checkpoints", show_default=True)
@click.option("--steps", default=20000, show_default=True)
@click.option("--lr", default=1e-3, show_default=True)
@click.option("--hidden", default=128, show_default=True)
@click.option("--layers", default=4, show_default=True)
@click.option("--cfg-dropout", default=0.15, show_default=True)
@click.option("--save-every", default=5000, show_default=True)
@click.option("--device", default="cpu", show_default=True)
def main(maps_dir, tags_dir, out_dir, steps, lr, hidden, layers, cfg_dropout,
         save_every, device) -> None:
    import json

    import torch

    from data.graph_builder import EDGE_FEATURES
    from data.placement_targets import (
        build_placement_vocabs, placement_sample, vocab_sizes)
    from models.conditioning import TagConditioner
    from models.placement.gnn import PlacementGNN
    from tagging.store import TagStore
    from wad_parser.types import MapDocument

    maps = sorted(p for p in Path(maps_dir).glob("*.json")
                  if not p.name.endswith(".textures.json"))
    if not maps:
        raise click.ClickException(f"no map JSON in {maps_dir}")
    store = TagStore(tags_dir) if tags_dir else None

    click.echo(f"building placement vocabularies from {len(maps)} maps...")
    vocabs = build_placement_vocabs(maps_dir)
    sizes = vocab_sizes(vocabs)
    click.echo(f"vocab sizes: {sizes}")

    click.echo("precomputing placement samples...")
    t0 = time.time()
    pool = []
    for p in maps:
        doc = MapDocument.from_dict(json.loads(p.read_text()))
        if not doc.sectors:
            continue
        sample = placement_sample(doc, vocabs)
        tags = (store.effective_tags(doc.meta.wad_hash, doc.meta.map_name)
                if store else {})
        pool.append((sample, tags))
    click.echo(f"pool: {len(pool)} graphs in {time.time() - t0:.0f}s")

    model = PlacementGNN(edge_features=len(EDGE_FEATURES), vocab=sizes,
                         hidden=hidden, n_layers=layers).to(device)
    conditioner = TagConditioner(embed_dim=256).to(device)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(conditioner.parameters()), lr=lr)
    click.echo(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    out = Path(out_dir)
    config = {"edge_features": len(EDGE_FEATURES), "hidden": hidden,
              "layers": layers, "vocab_sizes": sizes}

    def save(step):
        out.mkdir(parents=True, exist_ok=True)
        ckpt = out / f"placement_{step:06d}.pt"
        torch.save({"model": model.state_dict(),
                    "conditioner": conditioner.state_dict(),
                    "tag_to_index": conditioner.tag_to_index,
                    "vocabs": vocabs, "config": config, "step": step}, ckpt)
        click.echo(f"saved {ckpt}")

    losses = []
    t0 = time.time()
    for step in range(1, steps + 1):
        sample, tags = pool[np.random.randint(len(pool))]
        nf = torch.from_numpy(sample["node_features"]).to(device)
        ei = torch.from_numpy(sample["edge_index"]).to(device)
        ef = torch.from_numpy(sample["edge_features"]).to(device)
        cond = conditioner.encode([tags], device=device, cfg_dropout=cfg_dropout,
                                  register_new=True)[0]
        out_ = model(nf, ei, ef, cond)
        mc = torch.from_numpy(sample["monster_count"]).to(device)
        pc = torch.from_numpy(sample["pickup_count"]).to(device)
        mt = torch.from_numpy(sample["monster_type"]).to(device)
        pt = torch.from_numpy(sample["pickup_type"]).to(device)
        loss = (_poisson_nll(out_["monster_rate"], mc)
                + _poisson_nll(out_["pickup_rate"], pc)
                + _soft_ce(out_["monster_type"], mt)
                + _soft_ce(out_["pickup_type"], pt))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 200 == 0:
            click.echo(f"step {step}/{steps}  loss {np.mean(losses[-200:]):.3f}  "
                       f"{step / (time.time() - t0):.0f} it/s")
        if step % save_every == 0 or step == steps:
            save(step)
    click.echo(f"known tags: {sorted(conditioner.tag_to_index)}")


if __name__ == "__main__":
    main()

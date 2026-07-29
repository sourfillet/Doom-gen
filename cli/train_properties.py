"""Stage 3 training CLI: the tag-conditioned property GNN over the sector
graph (plan/generation.md, Stage 3).

    train-properties --maps-dir dataset/maps --tags-dir tags --out checkpoints/

The model and graphs are small, so this trains fine on CPU (the default) -
handy when the GPU is busy with a layout run. One graph per step; targets
are per-sector flat/texture/special indices from data.property_targets.
"""
from __future__ import annotations

import time
from pathlib import Path

import click
import numpy as np


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
@click.option("--device", default="cpu", show_default=True, help="cpu or cuda")
def main(maps_dir, tags_dir, out_dir, steps, lr, hidden, layers, cfg_dropout,
         save_every, device) -> None:
    import json

    import torch

    from data.graph_builder import EDGE_FEATURES
    from data.property_targets import build_vocabs, graph_sample, vocab_sizes
    from models.conditioning import TagConditioner
    from models.properties.gnn import HEADS, PropertyGNN
    from tagging.store import TagStore
    from wad_parser.types import MapDocument

    maps = sorted(p for p in Path(maps_dir).glob("*.json")
                  if not p.name.endswith(".textures.json"))
    if not maps:
        raise click.ClickException(f"no map JSON in {maps_dir}")
    store = TagStore(tags_dir) if tags_dir else None

    click.echo(f"building vocabularies from {len(maps)} maps...")
    vocabs = build_vocabs(maps_dir)
    sizes = vocab_sizes(vocabs)
    click.echo(f"vocab sizes: {sizes}")

    click.echo("precomputing graph samples...")
    t0 = time.time()
    pool = []
    for p in maps:
        doc = MapDocument.from_dict(json.loads(p.read_text()))
        if not doc.sectors:
            continue
        sample = graph_sample(doc, vocabs)
        tags = (store.effective_tags(doc.meta.wad_hash, doc.meta.map_name)
                if store else {})
        pool.append((sample, tags))
    click.echo(f"pool: {len(pool)} graphs in {time.time() - t0:.0f}s")

    model = PropertyGNN(edge_features=len(EDGE_FEATURES), vocab=sizes,
                        hidden=hidden, n_layers=layers).to(device)
    conditioner = TagConditioner(embed_dim=256).to(device)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(conditioner.parameters()), lr=lr)
    click.echo(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    out = Path(out_dir)
    config = {"edge_features": len(EDGE_FEATURES), "hidden": hidden,
              "layers": layers, "vocab_sizes": sizes}

    def save(step):
        ckpt = out / f"properties_{step:06d}.pt"
        out.mkdir(parents=True, exist_ok=True)
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
        logits = model(nf, ei, ef, cond)
        loss = sum(torch.nn.functional.cross_entropy(
            logits[h], torch.from_numpy(sample[h]).to(device)) for h in HEADS)
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

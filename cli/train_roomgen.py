"""Phase 9.4 training CLI: the autoregressive room-graph generator.

    train-roomgen --maps-dir dataset/maps --tags-dir tags --out checkpoints/

Segments every corpus map into a room-graph, linearizes it to a DFS-tree
sequence, and trains the GRU generator by teacher-forced NLL. CPU by default
(small model, one graph per step). Each step trains on a random-length PREFIX
of the DFS sequence (a valid connected subtree) so the stop head learns to
terminate at moderate, varied map sizes instead of maxing out - the corpus is
dominated by huge over-segmented maps (avg ~136 rooms).
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
@click.option("--steps", default=15000, show_default=True)
@click.option("--lr", default=1e-3, show_default=True)
@click.option("--hidden", default=128, show_default=True)
@click.option("--cfg-dropout", default=0.15, show_default=True)
@click.option("--save-every", default=5000, show_default=True)
@click.option("--device", default="cpu", show_default=True)
@click.option("--max-train-rooms", default=40, show_default=True,
              help="cap on the random DFS-prefix length trained per step")
def main(maps_dir, tags_dir, out_dir, steps, lr, hidden, cfg_dropout,
         save_every, device, max_train_rooms) -> None:
    import json

    import torch

    from data.roomgraph_targets import graph_to_sequence
    from models.conditioning import TagConditioner
    from models.roomgen.model import RoomGraphRNN
    from roomgraph.segment import segment
    from tagging.store import TagStore
    from wad_parser.types import MapDocument

    maps = sorted(p for p in Path(maps_dir).glob("*.json")
                  if not p.name.endswith(".textures.json"))
    if not maps:
        raise click.ClickException(f"no map JSON in {maps_dir}")
    store = TagStore(tags_dir) if tags_dir else None

    click.echo(f"segmenting {len(maps)} maps into room-graph sequences...")
    t0 = time.time()
    pool = []
    for p in maps:
        doc = MapDocument.from_dict(json.loads(p.read_text()))
        seq = graph_to_sequence(segment(doc))
        if not seq or len(seq) < 2:               # need at least root + 1 room
            continue
        tags = (store.effective_tags(doc.meta.wad_hash, doc.meta.map_name)
                if store else {})
        pool.append((seq, tags))
    click.echo(f"pool: {len(pool)} graphs (avg {np.mean([len(s) for s, _ in pool]):.1f} "
               f"rooms) in {time.time() - t0:.0f}s")

    model = RoomGraphRNN(hidden=hidden).to(device)
    conditioner = TagConditioner(embed_dim=256).to(device)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(conditioner.parameters()), lr=lr)
    click.echo(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    out = Path(out_dir)

    def save(step):
        out.mkdir(parents=True, exist_ok=True)
        ckpt = out / f"roomgen_{step:06d}.pt"
        torch.save({"model": model.state_dict(),
                    "conditioner": conditioner.state_dict(),
                    "tag_to_index": conditioner.tag_to_index,
                    "config": {"hidden": hidden}, "step": step}, ckpt)
        click.echo(f"saved {ckpt}")

    losses = []
    t0 = time.time()
    for step in range(1, steps + 1):
        seq, tags = pool[np.random.randint(len(pool))]
        # random-length DFS prefix = a smaller valid connected subtree; teaches
        # the stop head varied, moderate map sizes (parents stay index-valid)
        L = len(seq)
        if L > 4:
            seq = seq[:np.random.randint(4, min(L, max_train_rooms) + 1)]
        tag = conditioner.encode([tags], device=device, cfg_dropout=cfg_dropout,
                                 register_new=True)[0]
        loss = model.loss(seq, tag)
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

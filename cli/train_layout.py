"""Stage 1 training CLI: layout diffusion over an extracted corpus.

    train-layout --maps-dir dataset/maps --tags-dir tags --out checkpoints/

Rasters (plus mirror/rotate variants when --augment) are precomputed into
one in-memory tensor pool - corpus sizes here are hundreds of maps, not
millions of images. Tag conditioning uses CFG dropout so the model learns
an unconditional structural prior from every map plus a conditional
delta per tag set (plan/generation.md).
"""
from __future__ import annotations

import time
from pathlib import Path

import click
import numpy as np


@click.command()
@click.option("--maps-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--tags-dir", default=None, type=click.Path(file_okay=False),
              help="Tag store; omitting it trains fully unconditional.")
@click.option("--out", "out_dir", default="checkpoints", show_default=True)
@click.option("--resolution", default=128, show_default=True)
@click.option("--steps", default=20000, show_default=True)
@click.option("--batch-size", default=8, show_default=True)
@click.option("--lr", default=2e-4, show_default=True)
@click.option("--cfg-dropout", default=0.15, show_default=True)
@click.option("--base", default=64, show_default=True, help="U-Net base channel count.")
@click.option("--timesteps", default=1000, show_default=True)
@click.option("--augment/--no-augment", default=True, show_default=True)
@click.option("--save-every", default=1000, show_default=True)
@click.option("--resume", default=None, type=click.Path(exists=True, dir_okay=False))
def main(maps_dir, tags_dir, out_dir, resolution, steps, batch_size, lr,
         cfg_dropout, base, timesteps, augment, save_every, resume) -> None:
    import torch

    from data.augmentation import AUGMENT_OPS
    from data.dataset import MapDataset
    from models.checkpoint import load_checkpoint, save_checkpoint
    from models.conditioning import TagConditioner
    from models.layout.diffusion import GaussianDiffusion
    from models.layout.unet import UNet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    click.echo(f"device: {device}")

    ds = MapDataset(maps_dir, tags_dir=tags_dir, resolution=resolution)
    if len(ds) == 0:
        raise click.ClickException(f"no map JSON found in {maps_dir}")

    click.echo(f"precomputing rasters for {len(ds)} map(s)"
               f"{' x ' + str(1 + len(AUGMENT_OPS)) + ' augmented variants' if augment else ''}...")
    pool_x, pool_tags = [], []
    t0 = time.time()
    for i in range(len(ds)):
        ops = [None] + (list(AUGMENT_OPS) if augment else [])
        for op in ops:
            sample = ds.get(i, augment_op=op)
            pool_x.append(sample["raster"])
            pool_tags.append(sample["tags"])
    pool = torch.from_numpy(np.stack(pool_x)).float() * 2 - 1  # [0,1] -> [-1,1]
    click.echo(f"pool: {tuple(pool.shape)} in {time.time() - t0:.0f}s")

    config = {"in_channels": pool.shape[1], "base": base, "mults": [1, 2, 4],
              "emb_dim": 256, "timesteps": timesteps, "max_tags": 512,
              "resolution": resolution}

    if resume:
        _, diffusion, conditioner, config = load_checkpoint(resume, device)
        diffusion.train()
        click.echo(f"resumed from {resume}")
    else:
        model = UNet(in_channels=config["in_channels"], base=base,
                     mults=(1, 2, 4), emb_dim=config["emb_dim"])
        diffusion = GaussianDiffusion(model, timesteps=timesteps).to(device)
        conditioner = TagConditioner(embed_dim=config["emb_dim"],
                                     max_tags=config["max_tags"]).to(device)

    params = list(diffusion.parameters()) + list(conditioner.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    n_params = sum(p.numel() for p in diffusion.parameters())
    click.echo(f"U-Net parameters: {n_params / 1e6:.1f}M")

    out = Path(out_dir)
    losses = []
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(pool), (batch_size,))
        x0 = pool[idx].to(device)
        cond = conditioner.encode([pool_tags[i] for i in idx.tolist()], device=device,
                                  cfg_dropout=cfg_dropout, register_new=True)
        loss = diffusion.loss(x0, cond)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 50 == 0:
            rate = step / (time.time() - t0)
            click.echo(f"step {step}/{steps}  loss {np.mean(losses[-50:]):.4f}  "
                       f"{rate:.1f} it/s")
        if step % save_every == 0 or step == steps:
            ckpt = out / f"layout_{step:06d}.pt"
            save_checkpoint(ckpt, diffusion, conditioner, config, step)
            click.echo(f"saved {ckpt}")

    click.echo(f"known tags: {sorted(conditioner.tag_to_index)}")


if __name__ == "__main__":
    main()

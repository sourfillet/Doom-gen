"""Stream cached map rasters into a reproducible, validated diffusion run."""
from __future__ import annotations

import math
from pathlib import Path

import click
import numpy as np


@click.command()
@click.option('--maps-dir', required=True, type=click.Path(exists=True, file_okay=False))
@click.option('--tags-dir', default=None, type=click.Path(file_okay=False))
@click.option('--out', 'out_dir', default='checkpoints')
@click.option('--manifest', default=None, type=click.Path(exists=True, dir_okay=False))
@click.option('--seed', default=0, type=int)
@click.option('--device', default=None)
@click.option('--resolution', default=128, type=click.IntRange(min=8))
@click.option('--steps', default=20000, type=click.IntRange(min=1), help='Total optimizer steps, including resume.')
@click.option('--batch-size', default=8, type=click.IntRange(min=1))
@click.option('--lr', default=2e-4, type=click.FloatRange(min=0, min_open=True))
@click.option('--cfg-dropout', default=0.15, type=click.FloatRange(0, 1))
@click.option('--base', default=64, type=click.IntRange(min=8))
@click.option('--timesteps', default=1000, type=click.IntRange(min=2))
@click.option('--augment/--no-augment', default=True)
@click.option('--save-every', default=1000, type=click.IntRange(min=1))
@click.option('--validate-every', default=1000, type=click.IntRange(min=1))
@click.option('--resume', default=None, type=click.Path(exists=True, dir_okay=False))
@click.option('--occupancy-weight', default=8.0, type=click.FloatRange(min=0, min_open=True))
def main(maps_dir, tags_dir, out_dir, manifest, seed, device, resolution, steps,
         batch_size, lr, cfg_dropout, base, timesteps, augment, save_every,
         validate_every, resume, occupancy_weight):
    import torch
    from data.augmentation import AUGMENT_OPS
    from data.corpus import training_split
    from data.dataset import MapDataset
    from data.rasterizer import N_CHANNELS, OCCUPANCY_CHANNEL, RASTER_VERSION
    from data.raster_cache import RasterCache
    from models.conditioning import TagConditioner
    from models.layout.diffusion import GaussianDiffusion
    from models.layout.unet import UNet
    from models.training import fit, load_resume, seed_all

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        checkpoint = load_resume(resume, device)
        settings = checkpoint['settings'] if checkpoint else {
            'seed': seed, 'batch_size': batch_size, 'lr': lr, 'cfg_dropout': cfg_dropout,
            'augment': augment, 'occupancy_weight': occupancy_weight}
        seed_all(settings['seed'])
        splits, corpus = training_split(maps_dir, manifest, seed=settings['seed'])
        config = checkpoint['config'] if checkpoint else {
            'kind': 'layout', 'in_channels': N_CHANNELS, 'base': base, 'mults': [1, 2, 4],
            'emb_dim': 256, 'timesteps': timesteps, 'max_tags': 512,
            'resolution': resolution, 'raster_version': RASTER_VERSION, 'numeric_dim': 1}
        if config.get('kind') != 'layout' or config.get('raster_version') != RASTER_VERSION:
            raise ValueError('checkpoint preprocessing is incompatible with current rasters')
        if config['resolution'] % 4 or config['base'] % 8:
            raise ValueError('resolution must be divisible by 4 and base channels by 8')
        datasets = {split: MapDataset(maps_dir, tags_dir, config['resolution'], paths=paths)
                    for split, paths in splits.items()}
        caches = {split: RasterCache(ds, Path(out_dir) / 'rasters' / split,
                                     corpus['fingerprint'], tags_dir) for split, ds in datasets.items()}
        # Fit the world-scale distribution on training maps only. The same
        # numeric condition is supplied at training and sampling.
        if not checkpoint:
            spans = []
            for i in range(len(datasets['train'])):
                metadata = caches['train'].get(i)['raster_meta']
                spans.append(metadata['resolution'] / metadata['scale'])
            config['world_spans'] = spans
        diffusion = GaussianDiffusion(UNet(config['in_channels'], config['base'],
            tuple(config['mults']), config['emb_dim']), timesteps=config['timesteps']).to(device)
        conditioner = TagConditioner(config['emb_dim'], config['max_tags'], numeric_dim=1).to(device)
        optimizer = torch.optim.AdamW(list(diffusion.parameters()) + list(conditioner.parameters()), lr=settings['lr'])
        weights = torch.ones(config['in_channels'], device=device)
        weights[OCCUPANCY_CHANNEL] = settings['occupancy_weight']
        ops = [None] + (list(AUGMENT_OPS) if settings['augment'] else [])
        def loss_fn(item, training):
            split, index = item
            indices = [index]
            if training:
                indices += np.random.randint(len(datasets[split]), size=settings['batch_size'] - 1).tolist()
            samples = [caches[split].get(i, ops[np.random.randint(len(ops))] if training else None) for i in indices]
            arrays = np.stack([sample['raster'] for sample in samples])
            x0 = torch.from_numpy(arrays).to(device).mul(2).sub(1)
            values = [[math.log(sample['raster_meta']['resolution'] / sample['raster_meta']['scale'] / 2048)] for sample in samples]
            cond = conditioner.encode([sample['tags'] for sample in samples], device,
                cfg_dropout=settings['cfg_dropout'] if training else 0, register_new=training, numeric_values=values)
            return diffusion.loss(x0, cond, channel_weights=weights)
        train_pool = [('train', i) for i in range(len(datasets['train']))]
        validation_pool = [('validation', i) for i in range(len(datasets['validation']))]
        click.echo(f"device {device}; train {len(train_pool)}, validation {len(validation_pool)}, test {len(splits['test'])}; rasters streamed from disk")
        if not validation_pool:
            click.echo('No independent validation families: overfit/smoke experiment only.')
        fit(diffusion, conditioner, optimizer, train_pool, validation_pool, loss_fn,
            config=config, settings=settings, manifest=corpus, out_dir=out_dir,
            steps=steps, save_every=save_every, validate_every=validate_every,
            resume=checkpoint, tags_dir=tags_dir)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == '__main__':
    main()

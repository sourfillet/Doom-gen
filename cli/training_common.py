"""Shared property/placement GNN training orchestration."""
from __future__ import annotations

import json

import click

_OPTIONS = (
    click.option('--maps-dir', required=True, type=click.Path(exists=True, file_okay=False)),
    click.option('--tags-dir', default=None, type=click.Path(file_okay=False)),
    click.option('--out', 'out_dir', default='checkpoints'),
    click.option('--manifest', default=None, type=click.Path(exists=True, dir_okay=False)),
    click.option('--seed', default=0, type=int),
    click.option('--resume', default=None, type=click.Path(exists=True, dir_okay=False)),
    click.option('--steps', default=20000, type=click.IntRange(min=1)),
    click.option('--lr', default=1e-3, type=click.FloatRange(min=0, min_open=True)),
    click.option('--hidden', default=128, type=click.IntRange(min=1)),
    click.option('--layers', default=4, type=click.IntRange(min=1)),
    click.option('--cfg-dropout', default=0.15, type=click.FloatRange(0, 1)),
    click.option('--save-every', default=5000, type=click.IntRange(min=1)),
    click.option('--validate-every', default=1000, type=click.IntRange(min=1)),
    click.option('--device', default='cpu'),
)


def graph_training_options(function):
    """Apply the command-line options shared by both graph-model trainers."""
    for option in reversed(_OPTIONS):
        function = option(function)
    return function


def run_graph_training(kind, options):
    try:
        train_graph_model(kind, **options)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


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


def train_graph_model(kind, *, maps_dir, tags_dir, out_dir, steps, lr, hidden, layers,
                      cfg_dropout, save_every, device, manifest, seed, resume, validate_every):
    import torch
    from data.corpus import training_split
    from data.graph_builder import EDGE_FEATURES
    from models.conditioning import TagConditioner
    from models.properties.gnn import IN_FEATURES
    from models.training import fit, load_resume, seed_all
    from tagging.store import TagStore
    from wad_parser.types import MapDocument
    from wad_parser.geometry import DERIVED_VERSION
    from wad_parser.traversal import TRAVERSAL_VERSION
    if kind == 'properties':
        from data.property_targets import build_vocabs as build_vocab, graph_sample as make_sample, vocab_sizes
        from models.properties.gnn import PropertyGNN as Model, HEADS
    else:
        from data.placement_targets import build_placement_vocabs as build_vocab, placement_sample as make_sample, vocab_sizes
        from models.placement.gnn import PlacementGNN as Model
    checkpoint = load_resume(resume, device)
    if checkpoint and checkpoint['config'].get('kind') != kind:
        raise ValueError('checkpoint belongs to a different model kind')
    settings = checkpoint['settings'] if checkpoint else {'seed': seed, 'lr': lr, 'cfg_dropout': cfg_dropout}
    seed_all(settings['seed'])
    splits, corpus = training_split(maps_dir, manifest, seed=settings['seed'])
    vocabs = checkpoint['config']['vocabs'] if checkpoint else build_vocab(maps_dir, paths=splits['train'])
    config = checkpoint['config'] if checkpoint else {
        'kind': kind, 'edge_features': len(EDGE_FEATURES), 'hidden': hidden,
        'layers': layers, 'vocab_sizes': vocab_sizes(vocabs), 'vocabs': vocabs,
        'input_features': IN_FEATURES, 'derived_version': DERIVED_VERSION, 'traversal_version': TRAVERSAL_VERSION,
        'placement_mode': 'single-player-skill3', 'spatial': kind == 'placement',
        'side_heads': kind == 'properties'}
    if config.get('kind') != kind or config.get('derived_version') != DERIVED_VERSION:
        raise ValueError('checkpoint kind or preprocessing version is incompatible')
    store = TagStore(tags_dir) if tags_dir else None
    def pool(paths):
        records = []
        for path in paths:
            doc = MapDocument.from_dict(json.loads(path.read_text()))
            if not doc.sectors or doc.derived.geometry_issues:
                click.echo(f'skipping {path.name}: empty or invalid sector boundaries')
                continue
            tags = store.effective_tags(doc.meta.wad_hash, doc.meta.map_name) if store else {}
            records.append((make_sample(doc, vocabs), tags))
        return records
    train_pool, validation_pool = pool(splits['train']), pool(splits['validation'])
    arguments = {'edge_features': config['edge_features'], 'vocab': config['vocab_sizes'],
                 'hidden': config['hidden'], 'n_layers': config['layers'], 'input_features': config['input_features']}
    if kind == 'placement':
        arguments['spatial'] = config['spatial']
    else:
        arguments['side_heads'] = config['side_heads']
    model = Model(**arguments).to(device)
    conditioner = TagConditioner(256).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(conditioner.parameters()), lr=settings['lr'])
    def loss_fn(record, training):
        sample, tags = record
        arrays = {key: torch.from_numpy(value).to(device) for key, value in sample.items()}
        condition = conditioner.encode([tags], device,
            cfg_dropout=settings['cfg_dropout'] if training else 0, register_new=training)[0]
        output = model(arrays['node_features'], arrays['edge_index'], arrays['edge_features'], condition)
        if kind == 'properties':
            parts = {head: torch.nn.functional.cross_entropy(output[head], arrays[head]) for head in HEADS}
            side_output = model.predict_sides(output['sector_embedding'], arrays['side_owner'],
                                              arrays['side_neighbor'], arrays['side_features'])
            if arrays['side_owner'].numel():
                for field in ('side_mid_tex', 'side_upper_tex', 'side_lower_tex'):
                    parts[field] = torch.nn.functional.cross_entropy(side_output[field], arrays[field])
                parts['side_offsets'] = torch.nn.functional.smooth_l1_loss(side_output['side_offsets'], arrays['side_offsets'])
        else:
            parts = {'monster_count': _poisson_nll(output['monster_rate'], arrays['monster_count']),
                     'pickup_count': _poisson_nll(output['pickup_rate'], arrays['pickup_count']),
                     'monster_type': _soft_ce(output['monster_type'], arrays['monster_type']),
                     'pickup_type': _soft_ce(output['pickup_type'], arrays['pickup_type'])}
            for head in ('monster_position', 'pickup_position', 'monster_angle', 'pickup_angle'):
                parts[head] = _soft_ce(output[head], arrays[head])
        return sum(parts.values()), {key: value.detach() for key, value in parts.items()}
    click.echo(f"{kind}: train {len(train_pool)}, validation {len(validation_pool)}, test {len(splits['test'])}; vocab from train split only")
    if not validation_pool:
        click.echo('No independent validation families: overfit/smoke experiment only.')
    return fit(model, conditioner, optimizer, train_pool, validation_pool, loss_fn,
               config=config, settings=settings, manifest=corpus, out_dir=out_dir,
               steps=steps, save_every=save_every, validate_every=validate_every,
               resume=checkpoint, tags_dir=tags_dir)

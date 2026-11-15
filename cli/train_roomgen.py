"""Train versioned room graphs with real endings, loop targets, and validation."""
from __future__ import annotations

import json

import click

from models.roomgen.stopping import STOP_TARGETS


@click.command()
@click.option('--maps-dir', required=True, type=click.Path(exists=True, file_okay=False))
@click.option('--tags-dir', default=None, type=click.Path(file_okay=False))
@click.option('--out', 'out_dir', default='checkpoints')
@click.option('--manifest', default=None, type=click.Path(exists=True, dir_okay=False))
@click.option('--seed', default=0, type=int)
@click.option('--resume', default=None, type=click.Path(exists=True, dir_okay=False))
@click.option('--steps', default=15000, type=click.IntRange(min=1))
@click.option('--lr', default=1e-3, type=click.FloatRange(min=0, min_open=True))
@click.option('--hidden', default=128, type=click.IntRange(min=1))
@click.option('--cfg-dropout', default=0.15, type=click.FloatRange(0, 1))
@click.option('--save-every', default=5000, type=click.IntRange(min=1))
@click.option('--validate-every', default=1000, type=click.IntRange(min=1))
@click.option('--device', default='cpu')
@click.option('--max-train-rooms', default=128, type=click.IntRange(min=0),
              help='Bound sequence memory; truncated prefixes receive no stop label. 0 uses complete graphs.')
@click.option('--include-incomplete/--complete-only', default=False,
              help='Train on complete reachable-exit graphs by default; optionally include censored source fragments.')
@click.option('--stop-targets', type=click.Choice(STOP_TARGETS), default='all',
              help='complete: only complete maps train the stop hazard (fragments train structure only); '
                   'reachable-ends: every map ends after its last reachable room.')
@click.option('--complete-repeat', default=1, type=click.IntRange(min=1),
              help='List each complete training map this many times in the sampled pool.')
def main(maps_dir, tags_dir, out_dir, manifest, seed, resume, steps, lr, hidden,
         cfg_dropout, save_every, validate_every, device, max_train_rooms, include_incomplete,
         stop_targets, complete_repeat):
    import torch
    from data.corpus import training_split
    from data.roomgraph_targets import graph_to_sequence, ROOMGRAPH_TARGET_VERSION
    from models.conditioning import TagConditioner
    from models.roomgen.model import RoomGraphRNN
    from models.roomgen.stopping import stop_labels
    from models.training import fit, load_resume, seed_all
    from roomgraph.segment import segment, SEGMENTATION_VERSION
    from tagging.store import TagStore
    from wad_parser.types import MapDocument
    from wad_parser.geometry import DERIVED_VERSION
    from wad_parser.traversal import TRAVERSAL_VERSION
    from data.source_endings import SOURCE_ENDINGS_VERSION, assess_source_ending
    try:
        checkpoint = load_resume(resume, device)
        settings = checkpoint['settings'] if checkpoint else {'seed': seed, 'lr': lr,
            'cfg_dropout': cfg_dropout, 'max_train_rooms': max_train_rooms,
            'include_incomplete': include_incomplete}
        if not checkpoint and (stop_targets, complete_repeat) != ('all', 1):
            # Recorded only when used, so default checkpoints keep their settings.
            settings.update(stop_targets=stop_targets, complete_repeat=complete_repeat)
        elif checkpoint and (stop_targets, complete_repeat) != ('all', 1) and (stop_targets, complete_repeat) != (
                settings.get('stop_targets', 'all'), settings.get('complete_repeat', 1)):
            raise ValueError('resume keeps the checkpoint stop settings; omit --stop-targets/--complete-repeat')
        seed_all(settings['seed'])
        splits, corpus = training_split(maps_dir, manifest, seed=settings['seed'])
        config = checkpoint['config'] if checkpoint else {'kind': 'roomgen', 'hidden': hidden,
            'target_version': ROOMGRAPH_TARGET_VERSION, 'derived_version': DERIVED_VERSION,
            'stop_loss': 'sequence_nll_v1', 'source_endings': SOURCE_ENDINGS_VERSION,
            'traversal_version': TRAVERSAL_VERSION, 'segmentation_version': SEGMENTATION_VERSION,
            'physical_planarity': True}
        if config.get('kind') != 'roomgen' or config.get('target_version') != ROOMGRAPH_TARGET_VERSION:
            raise ValueError('checkpoint room-target version is incompatible with current training')
        store = TagStore(tags_dir) if tags_dir else None
        def pool(paths, *, validation=False):
            records = []
            for path in paths:
                doc = MapDocument.from_dict(json.loads(path.read_text()))
                try:
                    if doc.derived.geometry_issues:
                        raise ValueError('invalid sector boundaries')
                    graph = segment(doc, playable_only=True)
                    if graph.excluded_sectors:
                        click.echo(f'{path.name}: excluded {len(graph.excluded_sectors)} sectors outside supported traversal')
                    sequence = graph_to_sequence(graph)
                    if not sequence:
                        raise ValueError('no rooms')
                except ValueError as exc:
                    click.echo(f'skipping {path.name}: {exc}')
                    continue
                tags = store.effective_tags(doc.meta.wad_hash, doc.meta.map_name) if store else {}
                ending = assess_source_ending(doc, graph, version=config.get('source_endings', 'reachable_exit_v1'))
                complete = ending['complete']
                if not complete:
                    click.echo(f'{path.name}: no supported reachable exit; source fragment ({ending["reason"]})')
                    if validation or not settings.get('include_incomplete', False):
                        continue  # incomplete prefixes must not select the best full-map checkpoint
                    click.echo('  including with its ending censored')
                records.append((sequence, tags, complete))
            return records
        train_pool, validation_pool = pool(splits['train']), pool(splits['validation'], validation=True)
        if not any(complete for _, _, complete in train_pool):
            raise ValueError('no training graph has a supported reachable exit; supply complete maps before learning endings')
        model = RoomGraphRNN(hidden=config['hidden'], target_version=config['target_version']).to(device)
        conditioner = TagConditioner(256).to(device)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(conditioner.parameters()), lr=settings['lr'])
        def loss_fn(record, training):
            sequence, tags, complete = record
            limit = settings['max_train_rooms']
            terminal, stop = stop_labels(complete, len(sequence), limit, settings.get('stop_targets', 'all'))
            prefix = sequence[:limit] if limit else sequence
            condition = conditioner.encode([tags], device,
                cfg_dropout=settings['cfg_dropout'] if training else 0, register_new=training)[0]
            return model.loss(prefix, condition, terminal=terminal, stop=stop)
        click.echo(f"roomgen: train {len(train_pool)} ({sum(r[2] for r in train_pool)} complete endings), "
                   f"validation {len(validation_pool)}, test {len(splits['test'])}")
        repeat = settings.get('complete_repeat', 1)
        if repeat > 1:
            train_pool = [record for record in train_pool for _ in range(repeat if record[2] else 1)]
            click.echo(f'complete maps repeated {repeat}x: sampled pool {len(train_pool)}')
        if not validation_pool:
            click.echo('No complete usable validation graphs: no best checkpoint or generalization claim.')
        fit(model, conditioner, optimizer, train_pool, validation_pool, loss_fn,
            config=config, settings=settings, manifest=corpus, out_dir=out_dir,
            steps=steps, save_every=save_every, validate_every=validate_every,
            resume=checkpoint, tags_dir=tags_dir)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == '__main__':
    main()

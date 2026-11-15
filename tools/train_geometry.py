"""Train a small full-sequence geometry model on freshly verified preparations.

    python -m tools.train_geometry --prepared generated/geometry/prepared \
        --out generated/geometry/training --steps 500 --threads 1

Default admission requires raw and serialized static start/exit/connectivity
checks. --source-policy geometry checks authored/serialized geometry separately
from actors/progression; reconstruction is a diagnostic authoring control.
Neither policy certifies playability or compiled-geometry eligibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from data.geometry_tokens import PAD
from data.learned_geometry import TARGET_SCHEMA, digest
from tools.prepare_learned_geometry import verify_prepared


class FamilyPool:
    """Uniform family choice, then uniform admitted map within that family.

    The shared fit loop chooses a uniform index; all extra draws use the saved
    NumPy RNG. Validation still visits every admitted map and reports map means.
    Family names affect sampling only and never enter model conditioning.
    """

    def __init__(self, items):
        groups = {}
        for item in items:
            groups.setdefault(item['family'], []).append(item)
        self.groups = [groups[name] for name in sorted(groups)]

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        import numpy as np
        group = self.groups[index]
        return group[int(np.random.randint(len(group)))]


class CoverageFamilyPool(FamilyPool):
    """One shuffled complete training pass, then ordinary family sampling.

    The permutation uses a local RNG. Its position is reconstructed from the
    saved optimizer step and fixed batch size; no sampler state is guessed from
    the latest source file or from validation data.
    """

    def __init__(self, items, *, seed, offset=0):
        import numpy as np
        super().__init__(items)
        order = np.random.RandomState(seed).permutation(len(items))
        self.warmup = [items[int(index)] for index in order]
        self.offset = offset

    def __getitem__(self, index):
        position = self.offset
        self.offset += 1
        return self.warmup[position] if position < len(self.warmup) else super().__getitem__(index)


def prepared_pools(directory, *, max_tokens=4096, source_policy='static-playable', min_outline_corners=0,
                   token_codec='bytes'):
    if source_policy not in {'static-playable', 'geometry', 'reconstruction'}:
        raise ValueError('unknown geometry source policy')
    if max_tokens < 2 or min_outline_corners < 0:
        raise ValueError('max_tokens must be at least two and min_outline_corners nonnegative')
    from generation.learned_geometry import assess_prediction
    from generation.geometry_evidence import assess_geometry, layout_statistics
    from generation.learned_geometry import decode_prediction
    directory = Path(directory)
    raw_report = (directory / 'report.json').read_bytes()
    records = verify_prepared(directory)
    from data.geometry_codec import codec_for
    if token_codec not in {'bytes', 'compact', 'relative', 'interleaved', 'relative-reference'}:
        raise ValueError('token_codec must be bytes, compact, relative, interleaved or relative-reference')
    definition = None
    if token_codec in {'compact', 'relative', 'interleaved', 'relative-reference'}:
        from data.geometry_compact import fit_vocabulary
        if token_codec == 'relative':
            from data.geometry_relative import fit_relative_vocabulary
            fit_vocabulary = fit_relative_vocabulary
        elif token_codec == 'interleaved':
            from data.geometry_interleaved import fit_interleaved_vocabulary
            fit_vocabulary = fit_interleaved_vocabulary
        elif token_codec == 'relative-reference':
            from data.geometry_interleaved import fit_relative_reference_vocabulary
            fit_vocabulary = fit_relative_reference_vocabulary
        definition = fit_vocabulary(json.loads((directory / row['target']).read_bytes())['supervision']
                                    for row in records if row['split'] == 'train')
    codec = codec_for(definition)
    pools = {name: [] for name in ('train', 'validation', 'test')}
    manifest = {'schema': 'doom_geometry_training_corpus_v1',
                'preparation_report_sha256': hashlib.sha256(raw_report).hexdigest(),
                'source_policy': source_policy, 'max_tokens': max_tokens,
                'entries': [], 'excluded': [],
                'compiled_eligibility': 'not_assessed'}
    if min_outline_corners:
        manifest['min_outline_corners'] = min_outline_corners
    if definition is not None:
        manifest['codec_config'] = definition
    for record in records:
        target = json.loads((directory / record['target']).read_bytes())
        row = {key: record[key] for key in ('path', 'split', 'family', 'target_sha256', 'target_fingerprint')}
        try:
            tokens = codec.encode(target['supervision'])
            row['tokens'] = len(tokens)
            if len(tokens) > max_tokens:
                raise ValueError(f'sequence has {len(tokens)} tokens above budget {max_tokens}; not truncated')
            if token_codec in {'interleaved', 'relative-reference'}:
                from data.geometry_interleaved import ORDER_SCHEMA, first_use_order, restore_authoring
                canonical, _, source_ids = first_use_order(target['supervision'])
                assert codec.decode(tokens) == canonical
                if restore_authoring(canonical, source_ids) != target['supervision']:
                    raise ValueError('reindexing changes source authoring')
                row['encoding'] = {'order_schema': ORDER_SCHEMA, 'source_ids': source_ids,
                                   'source_authoring_restored': True}
                if source_policy == 'geometry':
                    row['encoding']['geometry'] = assess_geometry(canonical)
                    if row['encoding']['geometry']['status'] != 'raw_geometry_valid':
                        raise ValueError('reindexed source fails geometry checks')
                elif source_policy == 'static-playable':
                    row['encoding']['static_assessment'] = assess_prediction(canonical)
                    if row['encoding']['static_assessment']['status'] != 'static_valid':
                        raise ValueError('reindexed source fails static requirements')
            if source_policy == 'static-playable':
                assessment = assess_prediction(target['supervision'])
                if assessment['status'] != 'static_valid':
                    row['assessment'] = assessment
                    raise ValueError('source fails raw/serialized static requirements')
            elif source_policy == 'geometry':
                row['assessment'] = assess_geometry(target['supervision'])
                if row['assessment']['status'] != 'raw_geometry_valid':
                    raise ValueError('source fails authored/serialized geometry checks')
            else:
                row['assessment'] = 'diagnostic reconstruction; physical eligibility not required'
            row['layout'] = layout_statistics(decode_prediction(target['supervision']))
            if row['layout']['outline_corners'] < min_outline_corners:
                raise ValueError(f'outline has {row["layout"]["outline_corners"]} genuine corners below requested {min_outline_corners}')
            masks = codec.masks(tokens)
            pools[row['split']].append({'tokens': tokens, 'allowed': masks,
                                        'conditioning': target['conditioning'], 'path': record['path'],
                                        'family': record['family'],
                                        'prefix_values': codec.features(tokens)[:-1]})
            manifest['entries'].append(row)
        except ValueError as exc:
            row['reason'] = str(exc)
            manifest['excluded'].append(row)
    manifest['fingerprint'] = digest(manifest)
    return pools, manifest


def train(prepared, out, *, steps=500, seed=7, hidden=128, embedding=64,
          layers=2, context_dim=64, batch_size=2, lr=0.002, max_tokens=4096,
          source_policy='static-playable', device='cpu', threads=1,
          validate_every=100, save_every=100, resume=None, min_outline_corners=0, token_codec='bytes',
          stop_weight=1.0, record_memory=False, sampling='map', memory_read=False, record_anchors=False,
          checkpoint_wads=None, loss_balance='token', relative_pointers=False, spatial_pointers=False,
          dependency_memory=False, record_values=False, ending_loss_weight=0.0):
    import numpy as np
    import torch
    from models.conditioning import TagConditioner
    from models.geometry import GeometryRNN, condition_tags
    from models.training import fit, load_resume, restore_rng, rng_state, seed_all

    if min(steps, threads, batch_size, hidden, embedding, layers, context_dim, validate_every, save_every) < 1:
        raise ValueError('training dimensions, budgets and intervals must be positive')
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError('learning rate must be positive and finite')
    if isinstance(stop_weight, bool) or not np.isfinite(stop_weight) or stop_weight <= 0:
        raise ValueError('stop weight must be positive and finite')
    if isinstance(ending_loss_weight, bool) or not np.isfinite(ending_loss_weight) or ending_loss_weight < 0:
        raise ValueError('ending loss weight must be finite and nonnegative')
    if sampling not in {'map', 'family', 'coverage-family'}:
        raise ValueError('sampling must be map, family or coverage-family')
    if loss_balance not in {'token', 'field'}:
        raise ValueError('loss balance must be token or field')
    checkpoint = load_resume(resume, device)
    tool_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    settings = (checkpoint['settings'] if checkpoint else {
        'seed': seed, 'lr': lr, 'batch_size': batch_size, 'max_tokens': max_tokens,
        'source_policy': source_policy, 'threads': threads, 'training_tool_sha256': tool_hash})
    if not checkpoint and min_outline_corners:
        settings['min_outline_corners'] = min_outline_corners
    if not checkpoint and token_codec != 'bytes':
        settings['token_codec'] = token_codec
    if not checkpoint and stop_weight != 1.0:
        settings['stop_weight'] = stop_weight
    if not checkpoint and record_memory:
        settings['record_memory'] = True
    if not checkpoint and memory_read:
        settings['memory_read'] = True
    if not checkpoint and record_anchors:
        settings['record_anchors'] = True
    if not checkpoint and sampling != 'map':
        settings['sampling'] = sampling
    if not checkpoint and loss_balance != 'token':
        settings['loss_balance'] = loss_balance
    if not checkpoint and relative_pointers:
        settings['relative_pointers'] = True
    if not checkpoint and spatial_pointers:
        settings['spatial_pointers'] = True
    if not checkpoint and dependency_memory:
        settings['dependency_memory'] = True
    if not checkpoint and record_values:
        settings['record_values'] = True
    if not checkpoint and ending_loss_weight:
        settings['ending_loss_weight'] = ending_loss_weight
    if settings.get('training_tool_sha256') != tool_hash:
        raise ValueError('geometry training helper changed; exact resume is unavailable')
    interleaved_codecs = {'interleaved', 'relative-reference'}
    if settings.get('record_memory') and settings.get('token_codec', 'bytes') not in {'compact', 'relative', *interleaved_codecs}:
        raise ValueError('record memory requires --token-codec compact, relative, interleaved or relative-reference')
    if settings.get('memory_read') and not settings.get('record_memory'):
        raise ValueError('memory read requires --record-memory')
    if settings.get('record_anchors') and not settings.get('memory_read'):
        raise ValueError('record anchors require --memory-read and --record-memory')
    if (settings.get('token_codec') in interleaved_codecs and settings.get('record_memory')
            and not settings.get('record_anchors')):
        raise ValueError('interleaved memory requires --record-anchors and --memory-read')
    if settings.get('relative_pointers') and (settings.get('token_codec') not in interleaved_codecs or not settings.get('record_anchors')):
        raise ValueError('relative pointers require interleaved tokens and record anchors')
    if settings.get('spatial_pointers') and not settings.get('relative_pointers'):
        raise ValueError('spatial pointers require --relative-pointers')
    if settings.get('dependency_memory') and not settings.get('spatial_pointers'):
        raise ValueError('dependency memory requires --spatial-pointers')
    if settings.get('record_values') and not settings.get('dependency_memory'):
        raise ValueError('record values require --dependency-memory')
    if settings.get('token_codec') == 'relative-reference' and settings.get('record_memory') and not settings.get('dependency_memory'):
        raise ValueError('relative-reference memory requires --dependency-memory')
    if settings.get('ending_loss_weight', 0.0) and settings.get('token_codec') not in interleaved_codecs:
        raise ValueError('ending auxiliary loss requires --token-codec interleaved or relative-reference')
    if settings.get('loss_balance') == 'field':
        if settings.get('token_codec', 'bytes') not in {'compact', 'relative'}:
            raise ValueError('field-balanced loss requires compact or relative tokens')
        if settings.get('stop_weight', 1.0) != 1.0:
            raise ValueError('field-balanced loss requires stop weight 1')
    torch.set_num_threads(settings['threads'])
    pools, manifest = prepared_pools(prepared, max_tokens=settings['max_tokens'],
                                      source_policy=settings['source_policy'],
                                      min_outline_corners=settings.get('min_outline_corners', 0),
                                      token_codec=settings.get('token_codec', 'bytes'))
    from data.geometry_codec import codec_for
    codec = codec_for(manifest.get('codec_config'))
    directory = Path(out)
    if not checkpoint and directory.exists() and any(directory.iterdir()):
        raise ValueError('geometry training output must be empty; use resume or a fresh directory')
    directory.mkdir(parents=True, exist_ok=True)
    if not checkpoint:
        (directory / 'train_geometry.py').write_bytes(Path(__file__).read_bytes())
        (directory / 'input-assessment.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if not pools['train']:
        raise ValueError('no eligible geometry training sequences; retained input-assessment.json explains exclusions')
    seed_all(settings['seed'])
    config = (checkpoint['config'] if checkpoint else {
        'kind': 'geometry', 'target_schema': TARGET_SCHEMA, 'token_schema': codec.definition['schema'],
        'hidden': hidden, 'embedding': embedding, 'layers': layers, 'context_dim': context_dim,
        'vocab_size': codec.vocab_size, 'max_tags': 512, 'loss': 'decision_token_map_mean_v1',
        'prefix_features': True, 'prefix_schema': codec.prefix_schema})
    if not checkpoint and manifest.get('codec_config'):
        config['codec_config'] = manifest['codec_config']
    if not checkpoint and settings.get('stop_weight', 1.0) != 1.0:
        config.update(loss='weighted_stop_decision_map_mean_v1', stop_weight=settings['stop_weight'])
    if not checkpoint and settings.get('ending_loss_weight', 0.0):
        config['ending_auxiliary_schema'] = 'doom_geometry_global_ending_class_mean_v1'
        config['ending_loss_weight'] = settings['ending_loss_weight']
    if (config.get('ending_auxiliary_schema') != ('doom_geometry_global_ending_class_mean_v1' if settings.get('ending_loss_weight', 0.0) else None)
            or config.get('ending_loss_weight', 0.0) != settings.get('ending_loss_weight', 0.0)):
        raise ValueError('geometry ending auxiliary loss contract is incompatible')
    from data.geometry_objective import SCHEMA as FIELD_LOSS
    if not checkpoint and settings.get('loss_balance') == 'field':
        config['loss'] = FIELD_LOSS
    if (config.get('loss') == FIELD_LOSS) != (settings.get('loss_balance') == 'field'):
        raise ValueError('geometry field-balanced loss contract is incompatible')
    from data.geometry_memory import (ANCHOR_SCHEMA, DEPENDENCY_MEMORY_SCHEMA, INTERLEAVED_MEMORY_SCHEMA,
                                      READ_SCHEMA, RECORD_VALUES_SCHEMA, RELATIVE_POINTER_SCHEMA,
                                      RELATIVE_REFERENCE_MEMORY_SCHEMA, RELATIVE_REFERENCE_RECORD_VALUES_SCHEMA,
                                      SCHEMA as MEMORY_SCHEMA, SPATIAL_POINTER_SCHEMA, model_memory_options)
    if not checkpoint and settings.get('record_memory'):
        config['memory_schema'] = (ANCHOR_SCHEMA if settings.get('record_anchors') else
                                   READ_SCHEMA if settings.get('memory_read') else MEMORY_SCHEMA)
    expected_memory = (READ_SCHEMA if settings.get('memory_read') else MEMORY_SCHEMA) if settings.get('record_memory') else None
    if settings.get('record_anchors'):
        expected_memory = ANCHOR_SCHEMA
        if settings.get('token_codec') in interleaved_codecs:
            expected_memory = (
                (RELATIVE_REFERENCE_RECORD_VALUES_SCHEMA if settings.get('token_codec') == 'relative-reference'
                 else RECORD_VALUES_SCHEMA) if settings.get('record_values') else
                RELATIVE_REFERENCE_MEMORY_SCHEMA if settings.get('token_codec') == 'relative-reference' else
                DEPENDENCY_MEMORY_SCHEMA if settings.get('dependency_memory') else
                SPATIAL_POINTER_SCHEMA if settings.get('spatial_pointers') else
                RELATIVE_POINTER_SCHEMA if settings.get('relative_pointers') else INTERLEAVED_MEMORY_SCHEMA)
            if not checkpoint:
                config['memory_schema'] = expected_memory
    if config.get('memory_schema') != expected_memory:
        raise ValueError('geometry record-memory contract is incompatible')
    if config.get('stop_weight', 1.0) != settings.get('stop_weight', 1.0):
        raise ValueError('geometry stop-weight contract is incompatible')
    if (config.get('kind') != 'geometry' or config.get('target_schema') != TARGET_SCHEMA
            or config.get('token_schema') != codec.definition['schema'] or config.get('vocab_size') != codec.vocab_size):
        raise ValueError('incompatible geometry model/target/token contract')
    if config.get('prefix_features') and config.get('prefix_schema') != codec.prefix_schema:
        raise ValueError('geometry prefix feature contract is incompatible')
    model = GeometryRNN(**{key: config[key] for key in ('hidden', 'embedding', 'layers', 'context_dim')},
                        prefix_features=config.get('prefix_features', False), codec_config=config.get('codec_config'),
                        **model_memory_options(config.get('memory_schema'))).to(device)
    conditioner = TagConditioner(config['context_dim'], config['max_tags']).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(conditioner.parameters()), lr=settings['lr'])
    # Fit tag IDs on training context only, in deterministic source order. No
    # source metadata/counts enter the conditioner or inference model.
    for item in pools['train']:
        conditioner.encode([condition_tags(item['conditioning'])], device, register_new=True)
    for pool in pools.values():
        for item in pool:
            item['tensor'] = torch.tensor(item['tokens'], dtype=torch.long)
            mask = torch.zeros(len(item['tokens']) - 1, codec.vocab_size, dtype=torch.bool)
            for index, allowed in enumerate(item['allowed']):
                mask[index, list(allowed)] = True
            item['mask'] = mask
            if config.get('prefix_features'):
                item['prefix'] = torch.tensor(item['prefix_values'], dtype=torch.long)
            if config.get('loss') == FIELD_LOSS:
                from data.geometry_objective import loss_groups
                item['loss_groups'] = torch.tensor(loss_groups(codec, item['tokens']), dtype=torch.long)
            if config.get('memory_schema'):
                from data.geometry_memory import memory_trace
                trace = memory_trace(codec, item['tokens'], anchors=model.record_anchors,
                                     record_values=model.record_values)
                item['memory'] = torch.tensor(trace['records'], dtype=torch.long)
                item['references'] = torch.tensor(trace['references'][:-1], dtype=torch.long)
                if model.record_anchors:
                    item['anchors'] = torch.tensor(trace['anchors'][:-1], dtype=torch.long)
                if model.record_values:
                    item['record_values'] = torch.tensor(trace['record_values'][:-1], dtype=torch.long)

    if settings.get('sampling') == 'coverage-family':
        sampling_pool = CoverageFamilyPool(pools['train'], seed=settings['seed'],
            offset=(checkpoint['step'] if checkpoint else 0) * settings['batch_size'])
    else:
        sampling_pool = FamilyPool(pools['train']) if settings.get('sampling') == 'family' else pools['train']
    draw_step = checkpoint['step'] if checkpoint else 0
    trace_path = directory / f'sampling-trace-{draw_step + 1:06d}.jsonl'
    trace_version = 1
    while trace_path.exists():
        trace_path = directory / f'sampling-trace-{draw_step + 1:06d}-{trace_version:03d}.jsonl'
        trace_version += 1

    def loss_fn(item, training):
        nonlocal draw_step
        batch = [item]
        if training:
            batch.extend(sampling_pool[int(np.random.randint(len(sampling_pool)))]
                         for _ in range(settings['batch_size'] - 1))
            draw_step += 1
            with trace_path.open('a') as trace_file:
                trace_file.write(json.dumps({'step': draw_step, 'sampling': settings.get('sampling', 'map'),
                    'sources': [{'path': row['path'], 'family': row['family'], 'tokens': len(row['tokens'])}
                                for row in batch]}) + '\n')
        length = max(len(row['tokens']) for row in batch)
        tokens = torch.full((len(batch), length), PAD, dtype=torch.long, device=device)
        allowed = torch.zeros(len(batch), length - 1, codec.vocab_size, dtype=torch.bool, device=device)
        features = (torch.zeros(len(batch), length - 1, 6, dtype=torch.long, device=device)
                    if config.get('prefix_features') else None)
        memory_kwargs = {}
        if config.get('loss') == FIELD_LOSS:
            groups = torch.full((len(batch), length - 1, 2), -1, dtype=torch.long, device=device)
            memory_kwargs['loss_groups'] = groups
        if config.get('memory_schema'):
            memory = torch.zeros(len(batch), max(len(row['memory']) for row in batch), 9,
                                 dtype=torch.long, device=device)
            memory[:, :, :2] = -1
            references = torch.zeros(len(batch), length - 1, 4, dtype=torch.long, device=device)
            references[:, :, (0, 2)] = -1
            memory_kwargs.update(memory=memory, references=references)
            if model.record_anchors:
                anchors = torch.full((len(batch), length - 1, 7, 2), -1, dtype=torch.long, device=device)
                memory_kwargs['anchors'] = anchors
            if model.record_values:
                record_values = torch.zeros(len(batch), length - 1, 7, 2, dtype=torch.long, device=device)
                memory_kwargs['record_values'] = record_values
        for index, row in enumerate(batch):
            size = len(row['tokens'])
            tokens[index, :size] = row['tensor'].to(device)
            allowed[index, :size - 1] = row['mask'].to(device)
            if features is not None:
                features[index, :size - 1] = row['prefix'].to(device)
            if config.get('loss') == FIELD_LOSS:
                groups[index, :size - 1] = row['loss_groups'].to(device)
            if config.get('memory_schema'):
                memory[index, :len(row['memory'])] = row['memory'].to(device)
                references[index, :size - 1] = row['references'].to(device)
                if model.record_anchors:
                    anchors[index, :size - 1] = row['anchors'].to(device)
                if model.record_values:
                    record_values[index, :size - 1] = row['record_values'].to(device)
        context = conditioner.encode([condition_tags(row['conditioning']) for row in batch], device)
        return model.loss(tokens, context, allowed, features=features,
                          stop_weight=settings.get('stop_weight', 1.0),
                          ending_loss_weight=settings.get('ending_loss_weight', 0.0), **memory_kwargs)

    print('geometry pools:', {key: len(value) for key, value in pools.items()},
          'excluded:', len(manifest['excluded']), 'policy:', settings['source_policy'])
    if not pools['validation']:
        print('No independent validation pool: overfit/control experiment only; no best checkpoint is selected.')
    handle, export_errors = None, []
    if checkpoint_wads is not None:
        from tools.sample_geometry import sample_panel
        export_root = Path(checkpoint_wads)
        if export_root.exists() and any(export_root.iterdir()):
            raise ValueError('checkpoint WAD output must be empty; choose a fresh directory')
        export_root.mkdir(parents=True, exist_ok=True)
        (export_root / 'policy.json').write_text(json.dumps({
            'schema': 'doom_training_checkpoint_wads_v1', 'seed': 0, 'samples': 1,
            'temperature': 0.65, 'max_tokens': settings['max_tokens'],
            'playable_publication': False, 'training_tool_sha256': tool_hash,
            'intervals': 'initial, save, validation and final optimizer steps',
            'rng': 'restored after each export; no training-objective or data-sampling change'}, indent=2) + '\n')

        def emit(step):
            state, previous_threads = rng_state(), torch.get_num_threads()
            try:
                folder = export_root / f'step-{step:06d}'
                folder.mkdir()
                weights = folder / 'weights.pt'
                torch.save({'config': config, 'model': model.state_dict(),
                            'conditioner': conditioner.state_dict(), 'tag_to_index': conditioner.tag_to_index,
                            'step': step}, weights)
                sample_panel(weights, folder / 'sample', samples=1, seed=0, temperature=0.65,
                             max_tokens=settings['max_tokens'], device=device,
                             threads=previous_threads, raw_wads=True)
            except Exception as exc:
                import traceback
                failure = {'step': step, 'error': f'{type(exc).__name__}: {exc}',
                           'traceback': traceback.format_exc()}
                export_errors.append(failure)
                # Preserve each export failure even if a later training check
                # fails. A full filesystem must not prevent RNG restoration.
                try:
                    (export_root / f'export-error-{step:06d}.json').write_text(json.dumps(failure, indent=2) + '\n')
                except OSError:
                    pass
            finally:
                restore_rng(state)
                torch.set_num_threads(previous_threads)

        current_step = checkpoint['step'] if checkpoint else 0
        if not checkpoint:
            emit(0)

        def after_step(*_):
            nonlocal current_step
            current_step += 1
            if current_step % save_every == 0 or current_step % validate_every == 0 or current_step == steps:
                emit(current_step)

        handle = optimizer.register_step_post_hook(after_step)
    try:
        result = fit(model, conditioner, optimizer, sampling_pool, pools['validation'], loss_fn,
                     config=config, settings=settings, manifest=manifest, out_dir=directory,
                     steps=steps, save_every=save_every, validate_every=validate_every, resume=checkpoint)
    finally:
        if handle is not None:
            handle.remove()
    if export_errors:
        (export_root / 'export-errors.json').write_text(json.dumps(export_errors, indent=2) + '\n')
        raise ValueError('training checkpoints retained, but checkpoint WAD export failed; see export-errors.json')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--embedding', type=int, default=64)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--context-dim', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--source-policy', choices=['static-playable', 'geometry', 'reconstruction'], default='static-playable')
    parser.add_argument('--min-outline-corners', type=int, default=0,
                        help='Explicit training-set complexity filter; collinear wall splits are not corners.')
    parser.add_argument('--token-codec', choices=['bytes', 'compact', 'relative', 'interleaved', 'relative-reference'], default='bytes')
    parser.add_argument('--loss-balance', choices=['token', 'field'], default='token',
                        help='Optional section/field/stop-class balanced objective; compact/relative, stop weight 1.')
    parser.add_argument('--ending-loss-weight', type=float, default=0.0,
                        help='Optional class-balanced binary ending loss alongside token NLL; interleaved only, inference unchanged.')
    parser.add_argument('--stop-weight', type=float, default=1.0,
                        help='Optional loss weight for rare model-emitted list endings; no inference count rules.')
    parser.add_argument('--record-memory', action='store_true',
                        help='Learn causal reads/reference pointers over emitted field records.')
    parser.add_argument('--memory-read', action='store_true',
                        help='Also learn reference queries from the memory read; requires --record-memory (v2).')
    parser.add_argument('--record-anchors', action='store_true',
                        help='Learn reference queries from already completed references in the current record (v3).')
    parser.add_argument('--relative-pointers', action='store_true',
                        help='Learn relative emitted-index scores; requires interleaved tokens and record anchors.')
    parser.add_argument('--dependency-memory', action='store_true',
                        help='Learn messages along emitted sector/side/vertex/wall references; requires --spatial-pointers.')
    parser.add_argument('--record-values', action='store_true',
                        help='Learn next-token and query reads from completed scalar fields in the current record; requires --dependency-memory.')
    parser.add_argument('--spatial-pointers', action='store_true',
                        help='Learn candidate XY relations to known vertices; requires --relative-pointers.')
    parser.add_argument('--checkpoint-wads',
                        help='Fresh output directory for a diagnostic WAD at each initial/save/validation/final checkpoint.')
    parser.add_argument('--sampling', choices=['map', 'family', 'coverage-family'], default='map',
                        help='Training sampling unit; validation remains a complete map-mean assessment.')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--validate-every', type=int, default=100)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--resume')
    args = vars(parser.parse_args())
    try:
        train(**args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()

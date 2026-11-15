"""Load a shared geometry checkpoint and execute source-free token inference."""
from __future__ import annotations

import hashlib
from pathlib import Path

from data.learned_geometry import TARGET_SCHEMA, validate_conditioning


def load_model(checkpoint_path, *, device='cpu'):
    import torch
    from models.conditioning import TagConditioner
    from models.geometry import GeometryRNN
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    from data.geometry_codec import codec_for
    codec = codec_for(config.get('codec_config'))
    if (config.get('kind') != 'geometry' or config.get('token_schema') != codec.definition['schema']
            or config.get('target_schema') != TARGET_SCHEMA or config.get('vocab_size') != codec.vocab_size):
        raise ValueError('checkpoint is not a compatible shared geometry model')
    if config.get('prefix_features') and config.get('prefix_schema') != codec.prefix_schema:
        raise ValueError('incompatible geometry prefix feature contract')
    from data.geometry_memory import (INTERLEAVED_SCHEMAS, MEMORY_SCHEMAS, RELATIVE_REFERENCE_SCHEMAS,
                                      model_memory_options)
    memory_schema = config.get('memory_schema')
    if memory_schema is not None and memory_schema not in MEMORY_SCHEMAS:
        raise ValueError('incompatible geometry record-memory contract')
    unordered = getattr(codec, 'unordered_memory', False)
    if memory_schema is not None and (unordered != (memory_schema in INTERLEAVED_SCHEMAS)):
        raise ValueError('incompatible geometry memory ordering contract')
    relative_references = getattr(codec, 'relative_references', False)
    if memory_schema is not None and (relative_references != (memory_schema in RELATIVE_REFERENCE_SCHEMAS)):
        raise ValueError('incompatible geometry relative-reference memory contract')
    model = GeometryRNN(**{key: config[key] for key in ('hidden', 'embedding', 'layers', 'context_dim')},
                        prefix_features=config.get('prefix_features', False), codec_config=config.get('codec_config'),
                        **model_memory_options(memory_schema)).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    conditioner = TagConditioner(config['context_dim'], config['max_tags']).to(device)
    conditioner.load_checkpoint_state(checkpoint)
    # Deliberately return no source/corpus/scaffold records to the sampler.
    return model, conditioner, config


def infer(model, conditioner, context, *, seed=0, max_tokens=4096, temperature=1.0, syntax_masks=True):
    from models.geometry import condition_tags
    from generation.learned_geometry import assess_prediction
    validate_conditioning(context)
    device = next(model.parameters()).device
    vector = conditioner.encode([condition_tags(context)], device)
    sample = model.sample(vector, seed=seed, max_tokens=max_tokens,
                          temperature=temperature, syntax_masks=syntax_masks)
    result = {'schema': 'doom_geometry_inference_v1', 'context': context, 'sample': sample,
              'model_execution': 'executed', 'status': sample['status'],
              'prediction': None, 'assessment': None, 'node_building': 'not_run', 'engine': 'not_run'}
    if sample['status'] == 'complete':
        try:
            result['prediction'] = model.codec.decode(sample['tokens'])
            result['assessment'] = assess_prediction(result['prediction'])
            result['assessment']['model_execution'] = 'executed'
            result['status'] = result['assessment']['status']
            from generation.geometry_evidence import layout_statistics
            from generation.learned_geometry import decode_prediction
            result['layout'] = layout_statistics(decode_prediction(result['prediction']))
        except ValueError as exc:
            result.update(status='rejected', reason=str(exc))
        except Exception as exc:
            # Diagnostic inference must retain the raw stream even if an
            # evaluator crashes on adversarial geometry. Separate verifier
            # faults from ordinary rejected model predictions; never count
            # either as valid or replace the candidate.
            import traceback
            result.update(status='evaluation_error', reason=f'{type(exc).__name__}: {exc}',
                          evaluation_traceback=traceback.format_exc())
    return result


def sample_checkpoint(checkpoint_path, context, *, device='cpu', **kwargs):
    path = Path(checkpoint_path)
    raw = path.read_bytes()
    model, conditioner, _ = load_model(path, device=device)
    result = infer(model, conditioner, context, **kwargs)
    if path.read_bytes() != raw:
        raise ValueError('checkpoint changed during inference')
    result['checkpoint_sha256'] = hashlib.sha256(raw).hexdigest()
    return result


def require_game_assets(doc, game):
    """Check target assets without substituting/dropping any learned decision."""
    missing = []
    for sector in doc.sectors:
        for field in ('floor_flat', 'ceil_flat'):
            value = getattr(sector, field)
            if value.upper() not in game.flats:
                missing.append(f'sector {sector.id} {field}: {value}')
    for side in doc.sidedefs:
        for field in ('upper_tex', 'lower_tex', 'mid_tex'):
            value = getattr(side, field)
            if value not in {'-', ''} and value.upper() not in game.textures:
                missing.append(f'sidedef {side.id} {field}: {value}')
    for thing in doc.things:
        if game.sub_thing(thing.type) != thing.type:
            missing.append(f'thing {thing.id} type: {thing.type}')
    if missing:
        raise ValueError('predicted assets unsupported by target game: ' + '; '.join(missing[:20]))


def generate_checkpoint(checkpoint_path, context, *, out_path, report_dir, game,
                        n_maps=1, max_attempts=1, seed=0, max_tokens=4096,
                        temperature=1.0, target_port='zdoom', device='cpu', threads=1,
                        reject_flat_rectangles=False):
    """Opt-in learned generation through existing compiled/atomic publication.

    All attempts/raw streams are retained. No layout/shape/population/exit repair
    or asset substitution is called. Failure preserves an existing destination;
    successes retain per-map compiled bytes when combining the batch. Static
    acceptance still does not certify combat, engine completion or human quality.
    """
    import json
    import math
    import shutil
    import torch
    from generation.learned_geometry import decode_prediction
    from generation.pipeline import MapBatch, NoUsableMapsError
    if not 1 <= n_maps <= 99 or max_attempts < 1 or threads < 1 or max_tokens < 2:
        raise ValueError('map/attempt/thread/token budgets must be positive and n_maps at most 99')
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError('temperature must be finite and nonnegative')
    validate_conditioning(context)
    if shutil.which('zdbsp') is None:
        raise ValueError('learned-map publication requires zdbsp on PATH; diagnostic sampling remains available')
    checkpoint_path, report_dir = Path(checkpoint_path), Path(report_dir)
    if report_dir.exists() and any(report_dir.iterdir()):
        raise ValueError('generation report directory must be empty; choose a fresh directory')
    torch.set_num_threads(threads)
    raw = checkpoint_path.read_bytes()
    model, conditioner, config = load_model(checkpoint_path, device=device)
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'doom_learned_geometry_generation_v1', 'status': 'running',
              'checkpoint_sha256': hashlib.sha256(raw).hexdigest(), 'config': config,
              'context': context, 'game': game.name, 'target_port': target_port,
              'requested_maps': n_maps, 'max_attempts': max_attempts,
              'reject_flat_rectangles': reject_flat_rectangles,
              'engine': 'not_run', 'repairs': [], 'records': []}

    def save():
        temporary = report_dir / 'report.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(report_dir / 'report.json')

    save()
    failures = []
    original_hashes = {}
    with MapBatch(out_path, target_port=target_port) as batch:
        for index in range(n_maps):
            for attempt in range(max_attempts):
                sample_seed = seed + index * max_attempts + attempt
                result = infer(model, conditioner, context, seed=sample_seed,
                               max_tokens=max_tokens, temperature=temperature)
                result.update(sample_index=index, attempt=attempt + 1)
                stage = 'inference'
                try:
                    if result['status'] != 'static_valid':
                        raise ValueError(f'raw model sample: {result["status"]}')
                    stage = 'layout'
                    if reject_flat_rectangles and result['layout']['single_flat_rectangle']:
                        raise ValueError('model produced a single flat rectangle; requested varied geometry')
                    stage = 'assets'
                    doc = decode_prediction(result['prediction'], map_name=f'MAP{len(batch.accepted) + 1:02d}')
                    require_game_assets(doc, game)
                    from data.learned_geometry import core_hashes
                    original_hashes[doc.meta.map_name] = core_hashes(doc)
                    stage = 'compiled_publication'
                    result['publication'] = batch.add(doc)
                    if not result['publication']['nodes_built']:
                        batch.accepted.pop()
                        raise ValueError('node builder did not execute')
                    result['status'] = 'compiled_accepted'
                except Exception as exc:
                    import traceback
                    result.update(status='rejected', failure_stage=stage,
                                  reason=f'{type(exc).__name__}: {exc}', failure_traceback=traceback.format_exc())
                    failures.append({'sample': index, 'attempt': attempt + 1, 'seed': sample_seed,
                                     'failure_stage': stage, 'reason': result['reason']})
                filename = f'{index:03d}-{attempt:03d}.json'
                (report_dir / filename).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
                report['records'].append({'sample': index, 'attempt': attempt + 1, 'seed': sample_seed,
                                          'status': result['status'], 'record': filename})
                save()
                if result['status'] == 'compiled_accepted':
                    break
        if checkpoint_path.read_bytes() != raw:
            report.update(status='failed', reason='checkpoint changed during generation')
            save()
            raise ValueError(report['reason'])
        if not batch.accepted:
            report['status'] = 'failed'
            save()
            raise NoUsableMapsError(failures, n_maps)
        try:
            publication = batch.publish()
        except Exception as exc:
            report.update(status='failed', reason=f'final publication: {type(exc).__name__}: {exc}')
            save()
            raise
    from wad_parser.map_extractor import extract_all_maps
    from wad_parser.wad_reader import WadFile
    compiled = extract_all_maps(WadFile(str(out_path)))
    report['compiler_authoring_changes'] = {
        doc.meta.map_name: [name for name, value in core_hashes(doc).items()
                            if value != original_hashes[doc.meta.map_name][name]] for doc in compiled}
    report.update(status='published', publication=publication,
                  accepted_maps=len(publication['maps']), failed_attempts=len(failures))
    save()
    return report

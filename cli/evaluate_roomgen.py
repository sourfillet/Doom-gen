"""Fixed-seed room sampling, held-out losses, and binned-graph copy probes."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import click

from cli.options import generation_errors


def _signature(sequence):
    return hashlib.sha256(json.dumps(sequence, sort_keys=True).encode()).hexdigest()


def _statistics(graph):
    import numpy as np
    count = len(graph.rooms)
    degrees = [sum(room.id in (edge.a, edge.b) for edge in graph.connections) for room in graph.rooms]
    return {'rooms': count, 'edges': len(graph.connections),
            'cycles': max(0, len(graph.connections) - count + 1),
            'mean_degree': float(np.mean(degrees)) if degrees else 0,
            'max_degree': max(degrees, default=0),
            'mean_area': float(np.mean([room.area for room in graph.rooms])) if count else 0,
            'floor_range': max((room.floor_h for room in graph.rooms), default=0)
                           - min((room.floor_h for room in graph.rooms), default=0),
            'door_fraction': sum(edge.kind == 'door' for edge in graph.connections) / max(1, len(graph.connections)),
            'teleport_fraction': sum(edge.kind == 'teleport' for edge in graph.connections) / max(1, len(graph.connections)),
            'exit_roles': sum('exit' in room.roles or room.role == 'exit' for room in graph.rooms)}


def _descriptor(stats):
    import math
    return [math.log1p(stats['rooms']), stats['mean_degree'],
            stats['cycles'] / max(1, stats['rooms']), math.log1p(stats['mean_area']),
            stats['floor_range'] / 256, stats['door_fraction'], stats['teleport_fraction'],
            stats['exit_roles'] / max(1, stats['rooms'])]


@click.command()
@click.option('--checkpoint', required=True, type=click.Path(exists=True, dir_okay=False))
@click.option('--maps-dir', required=True, type=click.Path(exists=True, file_okay=False))
@click.option('--out', required=True, type=click.Path(file_okay=False))
@click.option('--samples', default=32, type=click.IntRange(min=1))
@click.option('--seed', default=0, type=int)
@click.option('--max-rooms', default=40, type=click.IntRange(min=1))
@click.option('--temperature', default=1.0, type=click.FloatRange(min=0, min_open=True))
@click.option('--write-maps', default=8, type=click.IntRange(min=0),
              help='Evaluate geometry for this many panel seeds, without retrying failures.')
@click.option('--baselines/--no-baselines', default=True,
              help='Compare matched-count procedural and nearest-size training-graph retrieval baselines.')
@generation_errors
def main(checkpoint, maps_dir, out, samples, seed, max_rooms, temperature, write_maps, baselines):
    import numpy as np
    import torch
    from data.corpus import training_split
    from data.roomgraph_targets import graph_to_sequence, sequence_to_graph
    from generation.baselines import procedural_graph, retrieval_graph, geometry_fingerprint, realize
    from generation.roomgen import generate, load_roomgen_model, sample_room_graph
    from models.training import code_fingerprint
    from roomgraph.segment import segment
    from wad_parser.types import MapDocument
    from data.source_endings import assess_source_ending, SUPPORTED_ENDINGS_VERSIONS

    if not math.isfinite(temperature):
        raise ValueError('temperature must be finite')
    evaluation_fingerprint = code_fingerprint()
    directory = Path(out)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('evaluation output directory must be empty; choose a new output directory')
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_sha256 = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    from wad_parser.geometry import DERIVED_VERSION
    from wad_parser.traversal import TRAVERSAL_VERSION
    from roomgraph.segment import SEGMENTATION_VERSION
    expected = {'derived_version': DERIVED_VERSION, 'traversal_version': TRAVERSAL_VERSION,
                'segmentation_version': SEGMENTATION_VERSION}
    mismatch = [name for name, value in expected.items() if payload.get('config', {}).get(name) != value]
    if mismatch:
        raise ValueError('checkpoint source preprocessing differs (' + ', '.join(mismatch)
                         + '); evaluate with its archived source to preserve training references. '
                           'Inference sampling remains available.')
    corpus = payload.get('corpus', {})
    if 'entries' not in corpus:
        raise ValueError('evaluation needs a checkpoint with a complete frozen corpus manifest')
    manifest_path = directory / 'corpus-manifest.json'
    manifest_path.write_text(json.dumps(corpus, indent=2))
    splits, _ = training_split(maps_dir, manifest_path)
    model, conditioner = load_roomgen_model(checkpoint)
    evaluation_device = next(model.parameters()).device
    if model.target_version < 3:
        raise ValueError('this evaluator requires version-3 room targets')
    source_stats, signatures, skipped, test_losses = {}, set(), [], []
    reference_graphs, geometry_signatures, thing_signatures = [], set(), set()
    complete_losses, fragment_losses = [], []
    ending_version = payload.get('config', {}).get('source_endings')
    complete_only = (ending_version in SUPPORTED_ENDINGS_VERSIONS
                     and not payload.get('settings', {}).get('include_incomplete', False))
    for split, paths in splits.items():
        records = []
        for path in paths:
            try:
                doc = MapDocument.from_dict(json.loads(path.read_text()))
                if doc.derived.geometry_issues:
                    raise ValueError('invalid sector boundaries')
                graph = segment(doc, playable_only=True)
                sequence = graph_to_sequence(graph)
                if not sequence:
                    raise ValueError('no playable rooms')
            except ValueError as exc:
                skipped.append({'split': split, 'path': path.name, 'reason': str(exc)})
                continue
            stats = _statistics(graph)
            ending = assess_source_ending(doc, graph, version=ending_version or 'reachable_exit_v1')
            complete = ending['complete']
            reference = split == 'train' and (complete or not complete_only)
            record = {'path': path.name, 'excluded_sectors': graph.excluded_sectors,
                      'complete_source': complete, 'training_reference': reference,
                      'ending_assessment': ending, **stats}
            records.append(record)
            if reference:
                signatures.add(_signature(sequence))
                reference_graphs.append(sequence_to_graph(sequence))
                fingerprints = geometry_fingerprint(doc)
                geometry_signatures.add(fingerprints['geometry'])
                thing_signatures.add((fingerprints['geometry'], fingerprints['things']))
            if split == 'test':
                limit = payload.get('settings', {}).get('max_train_rooms', 0)
                prefix = sequence[:limit] if limit else sequence
                with torch.no_grad():
                    condition = conditioner.encode([{}], evaluation_device)[0]
                    terminal = complete if ending_version in SUPPORTED_ENDINGS_VERSIONS else True
                    loss = float(model.loss(prefix, condition,
                                            terminal=terminal and (not limit or len(sequence) <= limit)))
                    test_losses.append(loss)
                    (complete_losses if complete else fragment_losses).append(loss)
                    record['unconditional_prefix_loss'] = loss
                    record['prefix_rooms'] = len(prefix)
                    record['prefix_censored'] = not (terminal and (not limit or len(sequence) <= limit))
        source_stats[split] = records
    references = [record for record in source_stats['train'] if record['training_reference']]
    training = np.array([_descriptor(record) for record in references])
    if not len(training):
        raise ValueError('no usable training reference graphs')
    scale = np.maximum(training.std(axis=0), 0.1)
    panel, exact = [], 0
    for sample_seed in range(seed, seed + samples):
        graph = sample_room_graph(model, conditioner, {}, seed=sample_seed,
                                   temperature=temperature, max_rooms=max_rooms)
        sequence = graph_to_sequence(graph)
        stats = _statistics(graph)
        distances = np.linalg.norm((training - np.array(_descriptor(stats))) / scale, axis=1)
        nearest = int(distances.argmin())
        match = _signature(sequence) in signatures
        exact += match
        panel.append({'seed': sample_seed, **stats, 'graph': graph.to_dict(),
                      'matches_binned_training_graph': match,
                      'nearest_training_descriptor': references[nearest]['path'],
                      'nearest_descriptor_distance': float(distances[nearest])})
    report = {'checkpoint': str(checkpoint), 'training_step': payload.get('step'),
              'evaluation_device': str(evaluation_device),
              'checkpoint_sha256': checkpoint_sha256,
              'geometry_profile': 'classic',
              'graph_snapshot_version': 1,
              'corpus_fingerprint': corpus['fingerprint'], 'temperature': temperature,
              'max_rooms': max_rooms, 'panel': panel, 'sources': source_stats, 'skipped': skipped,
              'exact_binned_training_matches': exact,
              'training_reference_graphs': len(references),
              'source_endings_version': ending_version,
              'unconditional_test_prefix_loss': float(np.mean(test_losses)) if test_losses else None,
              'complete_test_graphs': len(complete_losses),
              'unconditional_complete_test_prefix_loss': float(np.mean(complete_losses)) if complete_losses else None,
              'fragment_test_graphs': len(fragment_losses),
              'unconditional_fragment_test_prefix_loss': float(np.mean(fragment_losses)) if fragment_losses else None,
              'training_code_fingerprint': payload.get('code_fingerprint'),
              'evaluation_code_fingerprint': evaluation_fingerprint,
              'limits': ['Unconditional panel; tag ablations and tag-conditioned test losses require separate runs.',
                         'Binned graph equality and descriptor distance do not measure authored geometry copying.',
                         'Source sectors outside supported traversal are reported, not proven to be control sectors.',
                         'No engine completion, encounter balance, or human quality judgment is inferred.']}
    def copy_checks(path):
        from wad_parser.map_extractor import extract_all_maps
        from wad_parser.wad_reader import WadFile
        records = []
        for doc in extract_all_maps(WadFile(str(path))):
            fingerprints = geometry_fingerprint(doc)
            records.append({'map': doc.meta.map_name,
                            'exact_training_geometry': fingerprints['geometry'] in geometry_signatures,
                            'exact_training_geometry_and_things':
                                (fingerprints['geometry'], fingerprints['things']) in thing_signatures})
        return {'scope': 'Exact translation/ID-independent raw wall/height equality; excludes texture changes. '
                         'Rotated/partial/approximate/outline retrieval copying is not measured.', 'maps': records}
    from generation.pipeline import MapBatch, NoUsableMapsError
    requested_seeds = [row['seed'] for row in panel[:write_maps]]
    if write_maps:
        try:
            try:
                result = generate(checkpoint, {}, n_maps=min(write_maps, samples), seed=seed,
                                  temperature=temperature, max_rooms=max_rooms, max_attempts=1,
                                  out_path=directory / 'sample-panel.wad')
            except NoUsableMapsError as exc:
                result = exc.report
            report['geometry'] = {'written_maps': len(result['maps']), 'failed_samples': result['failed_samples'],
                                  'summaries': result['summaries'],
                                  'requested_seeds': requested_seeds,
                                  'outcomes': ([{'seed': row['seed'], 'validated': True, 'written': True}
                                                for row in result['summaries']]
                                               + [{'seed': row['seed'], 'validated': row.get('validated', False),
                                                   'written': False, 'reason': row['reason'],
                                                   'failure_stage': row.get('failure_stage', 'validation')}
                                                  for row in result['failed_samples']])}
            if result['maps']:
                report['geometry']['copy_checks'] = copy_checks(result['path'])
                report['geometry']['map_publication'] = result['map_publication']
        except (ValueError, RuntimeError) as exc:
            report['geometry'] = {'written_maps': 0, 'failure': str(exc),
                                  'requested_seeds': requested_seeds,
                                  'outcomes': [{'seed': s, 'validated': None, 'written': False,
                                                'reason': str(exc), 'failure_stage': 'batch'}
                                               for s in requested_seeds]}
    if baselines:
        report['baselines'] = {}
        for name in ('procedural', 'retrieval'):
            stats_panel, failures, outcomes = [], [], []
            path = directory / f'{name}-panel.wad'
            with MapBatch(path) as batch:
                for index, sampled in enumerate(panel):
                    sample_seed = sampled['seed']
                    graph = (procedural_graph(reference_graphs, rooms=sampled['rooms'],
                                               edges=sampled['edges'], seed=sample_seed)
                             if name == 'procedural' else
                             retrieval_graph(reference_graphs, rooms=sampled['rooms'], seed=sample_seed))
                    stats_panel.append({'seed': sample_seed, **_statistics(graph), 'graph': graph.to_dict(),
                                        'matches_binned_training_graph': _signature(graph_to_sequence(graph)) in signatures})
                    if index < write_maps:
                        stage = 'validation'
                        try:
                            graph.map_name = f'MAP{len(batch.accepted) + 1:02d}'
                            doc = realize(graph, seed=sample_seed)
                            stage = 'publication'
                            batch.add(doc)
                            outcomes.append({'seed': sample_seed, 'validated': True, 'written': False})
                        except ValueError as exc:
                            failure = {'seed': sample_seed, 'reason': str(exc), 'failure_stage': stage,
                                       'validated': stage == 'publication'}
                            failures.append(failure)
                            outcomes.append({**failure, 'written': False})
                record = {'panel': stats_panel, 'geometry': {'written_maps': 0, 'failed_samples': failures},
                          'policy': ('Matched neural room/edge counts; empirical training attributes, random tree, '
                                     'planar physical loops, forced start/exit.' if name == 'procedural' else
                                     'Uniform choice among nearest-room-count binned training graphs; full graph replay.'),
                          'shared_constraints': 'Same assembler, route budget, per-map compiled checks and one attempt per seed; no decoration.'}
                if batch.accepted:
                    try:
                        publication = batch.publish()
                        record['geometry'].update(written_maps=len(publication['maps']),
                                                  map_publication=publication['map_publication'],
                                                  copy_checks=copy_checks(path))
                        for outcome in outcomes:
                            if 'failure_stage' not in outcome:
                                outcome['written'] = True
                    except (ValueError, RuntimeError) as exc:
                        record['geometry'].update(publication_error=str(exc))
                        for outcome in outcomes:
                            if 'failure_stage' not in outcome:
                                outcome.update(reason=str(exc), failure_stage='publication')
                record['geometry'].update(requested_seeds=requested_seeds, outcomes=outcomes)
                report['baselines'][name] = record
    report['checkpoint_changed_during_evaluation'] = (
        hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != checkpoint_sha256)
    if report['checkpoint_changed_during_evaluation']:
        report['limits'].append('Checkpoint changed during evaluation; repeat with an immutable checkpoint.')
    report['code_changed_during_evaluation'] = code_fingerprint() != evaluation_fingerprint
    if report['code_changed_during_evaluation']:
        report['limits'].append('Source changed during evaluation; repeat under a frozen checkout before comparing runs.')
    (directory / 'report.json').write_text(json.dumps(report, indent=2))
    click.echo(f"{samples} samples; {exact} exact binned training matches; "
               f"{len(complete_losses)} complete and {len(fragment_losses)} fragment held-out graphs; "
               f"report: {directory / 'report.json'}")


if __name__ == '__main__':
    main()

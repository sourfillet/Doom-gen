"""Reproducible single-graph memorization and free-running sampling probe."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import click

from cli.options import generation_errors


@click.command()
@click.option('--source-json', default=None, type=click.Path(exists=True, dir_okay=False),
              help='An extracted map; omitted runs a synthetic three-room stairs/door graph.')
@click.option('--out', required=True, type=click.Path(file_okay=False))
@click.option('--steps', default=1200, type=click.IntRange(min=1))
@click.option('--hidden', default=32, type=click.IntRange(min=1))
@click.option('--seed', default=7, type=int)
@click.option('--samples', default=32, type=click.IntRange(min=1))
@click.option('--temperature', default=0.4, type=click.FloatRange(min=0, min_open=True))
@click.option('--force-terminal', is_flag=True,
              help='Force an artificial END target for an incomplete source graph; representation memorization only.')
@generation_errors
def main(source_json, out, steps, hidden, seed, samples, temperature, force_terminal):
    import numpy as np
    import torch
    from data.roomgraph_targets import graph_to_sequence, ROOMGRAPH_TARGET_VERSION
    from generation.roomgen import generate
    from models.conditioning import TagConditioner
    from models.roomgen.model import RoomGraphRNN
    from models.training import fit, seed_all, code_fingerprint
    from roomgraph.segment import segment
    from roomgraph.types import Room, RoomGraph, Connection
    from wad_parser.types import MapDocument
    from wad_parser.geometry import DERIVED_VERSION
    from wad_parser.traversal import TRAVERSAL_VERSION
    from roomgraph.segment import SEGMENTATION_VERSION
    from data.source_endings import assess_source_ending, SOURCE_ENDINGS_VERSION

    if not math.isfinite(temperature):
        raise ValueError('temperature must be finite')
    directory = Path(out)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('overfit output directory must be empty; choose a new output directory')
    seed_all(seed)
    source_bytes = None
    if source_json:
        source_bytes = Path(source_json).read_bytes()
        doc = MapDocument.from_dict(json.loads(source_bytes))
        if doc.meta.format != 'doom' or doc.derived.geometry_issues:
            raise ValueError('overfit probe needs a Doom map with valid sector boundaries')
        graph = segment(doc, playable_only=True)
        ending = assess_source_ending(doc, graph)
    else:
        rooms = [Room(i, [], i * 24, i * 24 + 128, 512**2, (0, 0), (0, 0, 512, 512),
                      roles=['start'] if i == 0 else ['exit'] if i == 2 else []) for i in range(3)]
        graph = RoomGraph(rooms, [Connection(0, 1, 'corridor', 192), Connection(1, 2, 'door', 192)])
        ending = {'complete': True, 'version': SOURCE_ENDINGS_VERSION,
                  'reason': 'synthetic declared complete graph', 'history': None}
    sequence = graph_to_sequence(graph)
    if not sequence:
        raise ValueError('no usable rooms')
    terminal = bool(ending['complete'] or force_terminal)
    artificial = bool(force_terminal and not ending['complete'])
    tool_bytes = Path(__file__).read_bytes()
    code_signature = code_fingerprint('roomgen')
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'target-sequence.json').write_text(json.dumps(sequence, indent=2))
    (directory / 'overfit_roomgen.py').write_bytes(tool_bytes)
    if source_bytes is not None:
        (directory / 'source.json').write_bytes(source_bytes)
    fingerprint = hashlib.sha256(json.dumps(sequence, sort_keys=True).encode()).hexdigest()
    model, conditioner = RoomGraphRNN(hidden=hidden), TagConditioner()
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(conditioner.parameters()), lr=0.005)
    def loss_fn(item, training):
        return model.loss(item, conditioner.encode([{}], 'cpu')[0], terminal=terminal)
    fit(model, conditioner, optimizer, [sequence], [], loss_fn,
        config={'kind': 'roomgen', 'hidden': hidden, 'target_version': ROOMGRAPH_TARGET_VERSION,
                'derived_version': DERIVED_VERSION, 'traversal_version': TRAVERSAL_VERSION,
                'segmentation_version': SEGMENTATION_VERSION, 'source_endings': SOURCE_ENDINGS_VERSION,
                'stop_loss': 'sequence_nll_v1', 'physical_planarity': True},
        settings={'seed': seed, 'lr': 0.005, 'experiment': 'single-example-overfit',
                  'terminal_target': terminal, 'forced_terminal': artificial,
                  'overfit_tool_sha256': hashlib.sha256(tool_bytes).hexdigest(),
                  'source_sha256': hashlib.sha256(source_bytes).hexdigest() if source_bytes is not None else None},
        manifest={'fingerprint': fingerprint, 'scope': 'single-example; no holdout'},
        out_dir=directory, steps=steps, save_every=steps, validate_every=steps)
    model.eval()
    expected = sequence
    counts, matches, prefixes = [], 0, 0
    for sample_seed in range(samples):
        torch.manual_seed(sample_seed)
        sampled = model.sample(torch.zeros(256), max_rooms=len(sequence) + 3, temperature=temperature)
        counts.append(len(sampled))
        matches += sampled == expected
        prefixes += sampled[:len(expected)] == expected
    scope = ('single-example encoded-target memorization; artificial ending, no source completion claim' if artificial
             else 'single-example memorization; no generalization claim' if terminal
             else 'source-fragment prefix memorization; ending censored, no completion/generalization claim')
    report = {'scope': scope, 'source': source_json, 'source_ending': ending,
              'terminal_target': terminal, 'forced_terminal': artificial,
              'target_rooms': len(sequence), 'excluded_source_sectors': graph.excluded_sectors,
              'steps': steps, 'seed': seed, 'temperature': temperature,
              'sample_seeds': list(range(samples)), 'room_counts': counts,
              'exact_binned_graph_matches': matches if terminal else None,
              'matching_binned_prefixes': prefixes, 'sample_room_budget': len(sequence) + 3,
              'room_budget_hits': sum(count == len(sequence) + 3 for count in counts)}
    try:
        result = generate(directory / f'roomgen_{steps:06d}.pt', {}, n_maps=min(samples, 8),
                          seed=0, temperature=temperature, max_rooms=len(sequence) + 3,
                          max_attempts=1, out_path=directory / 'sample-panel.wad')
        report.update(written_maps=len(result['maps']), failed_samples=result['failed_samples'])
    except RuntimeError as exc:
        report['generation_failure'] = str(exc)
    metrics = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
    report['final_mean_loss'] = float(np.mean([record['train_loss'] for record in metrics[-50:]]))
    report['code_changed_during_probe'] = (code_fingerprint('roomgen') != code_signature
                                         or Path(__file__).read_bytes() != tool_bytes)
    (directory / 'report.json').write_text(json.dumps(report, indent=2))
    click.echo(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

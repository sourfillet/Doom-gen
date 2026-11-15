"""Train a single-source spatial memorization control with actual prediction decoding.

All shared vertex/thing coordinates and every interior/control-sector elevation
are predicted. Source combinatorics, materials and action metadata are retained
explicitly; this is not generation from a new RoomGraph or a gameplay certificate.
"""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

import click

from cli.options import generation_errors

DEFAULT_STEM = '6ce8b4b0e602140261a202180dbe7c317453b8f1991985258e503643caf2ee26-0018'


@click.command()
@click.option('--source-json', default=None, type=click.Path(exists=True, dir_okay=False),
              help='Extracted classic map; retains all authored controls/actions.')
@click.option('--target-json', default=None, type=click.Path(exists=True, dir_okay=False),
              help='Lossless doom_spatial_reconstruction_v1 target (not geometry_v1).')
@click.option('--stem', default=DEFAULT_STEM, help='Frozen corpus stem when no explicit input is supplied.')
@click.option('--out', required=True, type=click.Path(file_okay=False))
@click.option('--steps', default=1200, type=click.IntRange(min=1), help='Fixed training attempt budget.')
@click.option('--hidden', default=32, type=click.IntRange(min=1))
@click.option('--lr', default=0.01, type=click.FloatRange(min=0, min_open=True))
@click.option('--seed', default=7, type=int)
@generation_errors
def main(source_json, target_json, stem, out, steps, hidden, lr, seed):
    """Memorize geometry on a retained authoring scaffold and decode the checkpoint."""
    import shutil
    import torch
    from torch.nn import functional as F
    from data.spatial_prediction import (build_prediction_scaffold, decode_prediction,
                                         digest, graph_from_source_record, HEIGHT_SCALE, PREDICTION_SCHEMA)
    from data.spatial_targets import build_spatial_target
    from generation.validator import validate_document
    from models.spatial import SpatialScaffoldModel, CHECKPOINT_SCHEMA, predict_checkpoint
    from models.training import seed_all, code_fingerprint, source_preprocessing_fingerprint, dependency_versions
    from wad_parser.types import MapDocument
    from wad_writer.wad_writer import write_wad
    from wad_parser.wad_reader import WadFile
    from wad_parser.map_extractor import extract_all_maps
    from wad_writer.assembler import assemble_map_lumps

    directory = Path(out)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('output directory must be empty; choose a fresh experiment directory')
    if source_json and target_json:
        raise ValueError('choose source-json or target-json, not both')
    if not math.isfinite(lr):
        raise ValueError('learning rate must be finite')
    graph = None
    source_record = None
    if target_json:
        input_bytes = Path(target_json).read_bytes()
        target = json.loads(input_bytes)
    else:
        if source_json:
            path = Path(source_json)
        else:
            corpus = Path('generated/repair-experiments/current-complete-corpus')
            path = corpus / 'maps' / f'{stem}.json'
            source_record = (corpus / 'records' / f'{stem}.json').read_bytes()
        input_bytes = path.read_bytes()
        if source_record is not None:
            graph = graph_from_source_record(input_bytes, json.loads(source_record))
        source = MapDocument.from_dict(json.loads(input_bytes))
        target = build_spatial_target(source, graph)
    scaffold, labels = build_prediction_scaffold(target)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'input.json').write_bytes(input_bytes)
    if source_record is not None:
        (directory / 'source-record.json').write_bytes(source_record)
    for name, data in [('target.json', target), ('scaffold.json', scaffold), ('supervision.json', labels)]:
        (directory / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    root = Path(__file__).resolve().parents[1]
    for package in ('data', 'models', 'cli', 'wad_parser', 'wad_writer', 'generation', 'roomgraph', 'tagging'):
        for path in (root / package).rglob('*.py'):
            destination = directory / 'source' / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    code_hash = code_fingerprint()
    preprocess_hash = source_preprocessing_fingerprint()
    seed_all(seed)
    # Tiny graphs are slower and less reproducible with many CPU worker threads.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = SpatialScaffoldModel(scaffold, hidden=hidden)
        y = torch.tensor(labels['values'], dtype=torch.float32)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.01)
        model.train()
        with torch.no_grad():
            initial_loss = float(F.mse_loss(model(), y))
        history = []
        for step in range(steps):
            optimizer.zero_grad()
            loss = F.mse_loss(model(), y)
            if not torch.isfinite(loss):
                raise ValueError(f'non-finite spatial loss at step {step}')
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step == 0 or (step + 1) % 50 == 0 or step + 1 == steps:
                history.append({'step': step + 1, 'loss_before_update': float(loss.detach())})
        model.eval()
        with torch.no_grad():
            final_values = model()
            final_loss = float(F.mse_loss(final_values, y))
            errors = (final_values - y).abs()
        xy = [i for i, kind in enumerate(scaffold['slot_kinds']) if kind != 'sector']
        sectors = [i for i, kind in enumerate(scaffold['slot_kinds']) if kind == 'sector']
        max_xy_error = float(errors[xy].max()) * scaffold['frame']['scale'] if xy else 0
        max_height_error = float(errors[sectors].max()) * HEIGHT_SCALE if sectors else 0
        checkpoint_path = directory / 'model.pt'
        torch.save({'schema': CHECKPOINT_SCHEMA, 'hidden': hidden, 'scaffold': scaffold,
                    'scaffold_sha256': digest(scaffold), 'model': model.state_dict(),
                    'settings': {'steps': steps, 'seed': seed, 'lr': lr},
                    'code_fingerprint': code_hash, 'scope': scaffold['scope']}, checkpoint_path)
        # Exercise the real inference loader rather than reusing training tensors.
        loaded_scaffold, prediction = predict_checkpoint(checkpoint_path, expected_scaffold=scaffold)
        assert prediction == {'schema': PREDICTION_SCHEMA, 'values': final_values.tolist()}
    finally:
        torch.set_num_threads(previous_threads)
    (directory / 'prediction.json').write_text(json.dumps(prediction, indent=2, allow_nan=False) + '\n')
    (directory / 'loss-history.json').write_text(json.dumps(history, indent=2) + '\n')
    report = {'schema': 'doom_spatial_overfit_v2', 'stem': stem,
              'scope': scaffold['scope'], 'target_sha256': digest(target),
              'scaffold_sha256': digest(scaffold), 'prediction_sha256': digest(prediction),
              'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
              'code_fingerprint': code_hash, 'source_preprocessing_fingerprint': preprocess_hash,
              'versions': dependency_versions('cpu'),
              'settings': {'steps': steps, 'hidden': hidden, 'lr': lr, 'seed': seed,
                           'attempts': 1, 'training_scope': 'single source; no holdout or resume claim'},
              'numeric_outputs': {'vertices': len(scaffold['vertices']), 'sectors': len(scaffold['sectors']),
                                  'things': len(scaffold['things']), 'all_source_controls_retained': True},
              'convergence': {'initial_loss': initial_loss, 'final_loss': final_loss,
                              'loss_reduction_pct': 100 * (1 - final_loss / initial_loss) if initial_loss else 0,
                              'max_xy_error_world_units': max_xy_error,
                              'max_height_error_world_units': max_height_error,
                              'subunit_reconstruction': max(max_xy_error, max_height_error) < 0.5},
              'engine_status': 'not_run', 'publication_status': 'diagnostic_only',
              'decode_status': 'pending'}
    try:
        doc = decode_prediction(loaded_scaffold, prediction)
        doc.meta.map_name = 'MAP01'
        basic = validate_document(doc, require_exit=False)
        connected = validate_document(doc, require_exit=False, require_connected=True)
        gameplay = validate_document(doc, require_exit=True, require_connected=True)
        (directory / 'predicted-document.json').write_text(json.dumps(doc.to_dict(), indent=2) + '\n')
        path = directory / 'candidate.wad'
        write_wad([doc], path)
        restored, = extract_all_maps(WadFile(str(path)))
        report.update(decode_status='decoded', assembled_geometry={
            'sectors': len(doc.sectors), 'linedefs': len(doc.linedefs), 'vertices': len(doc.vertices),
            'geometry_issues': doc.derived.geometry_issues, 'basic_validation': basic,
            'connected_validation': connected, 'gameplay_validation': gameplay,
            'serialized_validation': validate_document(restored, require_exit=False),
            'serialization_preserves_authoring': assemble_map_lumps(restored) == assemble_map_lumps(doc),
            'nodes_built': False})
    except (ValueError, TypeError, OverflowError) as exc:
        report.update(decode_status='rejected', decode_reason=f'{type(exc).__name__}: {exc}')
    report['source_changed'] = code_fingerprint() != code_hash or source_preprocessing_fingerprint() != preprocess_hash
    (directory / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    click.echo(f"Spatial memorization: loss {initial_loss:.7g} -> {final_loss:.7g}; "
               f"XY error {max_xy_error:.4g}, height error {max_height_error:.4g}; {report['decode_status']}. "
               'Diagnostic only; engine not run.')


if __name__ == '__main__':
    main()

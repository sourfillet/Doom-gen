import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parent)]
from geometry_fixtures import two_rooms
from data.corpus import build_manifest, training_split
from wad_parser.types import Thing, ThingFlags, resolve_doomed


def _corpus(directory, count=6):
    directory.mkdir()
    for index in range(count):
        doc = two_rooms()
        from generation.progression import ensure_exit
        ensure_exit(doc)
        doc.meta.wad_hash = f'wad-{index}'
        doc.sectors[0].light_level = 100 + index
        for kind, x, y in [(3004, 192, 64), (2007, 48, 96)]:
            name, category, _ = resolve_doomed(kind)
            doc.things.append(Thing(len(doc.things), x, y, 90, kind, name, category,
                                     ThingFlags(skill_3=True)))
        (directory / f'{index}.json').write_text(json.dumps(doc.to_dict()))
    return directory


def test_corpus_splits_keep_related_wads_and_repackaged_maps_together(tmp_path):
    directory = _corpus(tmp_path / 'maps')
    raw = json.loads((directory / '0.json').read_text())
    raw['meta']['wad_hash'] = 'repackaged'
    (directory / 'duplicate.json').write_text(json.dumps(raw))
    manifest = build_manifest(directory, families={'wad-0': 'same', 'wad-1': 'same'}, seed=42)
    assert len(manifest['duplicates']) == 1
    entries = {entry['wad_hash']: entry for entry in manifest['entries']}
    assert entries['wad-0']['split'] == entries['wad-1']['split']
    assert manifest['duplicates'][0]['split'] == entries['wad-0']['split']
    assert set(entry['split'] for entry in manifest['entries']) == {'train', 'validation', 'test'}
    assert manifest == build_manifest(directory, families={'wad-0': 'same', 'wad-1': 'same'}, seed=42)
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    training_split(directory, path)
    (directory / '0.json').write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        training_split(directory, path)


def test_exact_resume_restores_optimizer_conditioner_and_rng(tmp_path):
    torch = pytest.importorskip('torch')
    from models.training import fit, load_resume, seed_all
    from models.conditioning import TagConditioner
    settings, config, manifest = {'seed': 7}, {'kind': 'tiny'}, {'fingerprint': 'fixed'}
    def run(directory, steps, checkpoint=None):
        seed_all(7)
        model = torch.nn.Linear(1, 1)
        conditioner = TagConditioner(1, 8)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(conditioner.parameters()), lr=0.01)
        def loss_fn(item, training):
            cond = conditioner.encode([{'example': 1.0}], 'cpu', cfg_dropout=0.2 if training else 0,
                                       register_new=training)
            return (model(torch.tensor([[float(item)]])) + cond + torch.randn(1, 1) * 0.1).square().mean()
        fit(model, conditioner, optimizer, [1, 2, 3], [1], loss_fn, config=config,
            settings=settings, manifest=manifest, out_dir=directory, steps=steps,
            save_every=1, validate_every=1, resume=checkpoint)
        return model, conditioner
    uninterrupted = run(tmp_path / 'full', 3)
    run(tmp_path / 'split', 1)
    checkpoint = load_resume(tmp_path / 'split' / 'tiny_000001.pt')
    resumed = run(tmp_path / 'split', 3, checkpoint)
    for expected, actual in zip(uninterrupted, resumed):
        assert all(torch.equal(value, actual.state_dict()[key]) for key, value in expected.state_dict().items())
    assert uninterrupted[1].tag_to_index == resumed[1].tag_to_index
    incompatible = copy.deepcopy(checkpoint)
    incompatible['versions']['numpy'] = 'changed-version'
    with pytest.raises(ValueError, match='dependencies changed'):
        run(tmp_path / 'wrong-environment', 3, incompatible)
    with pytest.raises(ValueError, match='already contains a training run'):
        run(tmp_path / 'full', 4)


def test_training_fingerprint_tracks_preprocessing_but_not_unrelated_generation(tmp_path):
    from models.training import code_fingerprint
    for name in ('cli', 'data', 'generation'):
        (tmp_path / name).mkdir()
    (tmp_path / 'cli/train_layout.py').write_text('from data.preprocessing import prepare\n')
    data = tmp_path / 'data/preprocessing.py'
    data.write_text('def prepare(): return 1\n')
    unrelated = tmp_path / 'generation/validator.py'
    unrelated.write_text('value = 1\n')
    signature = code_fingerprint('layout', root=tmp_path)
    unrelated.write_text('value = 2\n')
    assert code_fingerprint('layout', root=tmp_path) == signature
    data.write_text('def prepare(): return 2\n')
    assert code_fingerprint('layout', root=tmp_path) != signature


def test_source_cache_fingerprint_tracks_assessment_but_not_unrelated_diagnostics(tmp_path):
    from models.training import source_preprocessing_fingerprint
    for directory in ('data', 'wad_parser', 'cli'):
        (tmp_path / directory).mkdir()
    (tmp_path / 'data/source_endings.py').write_text('from wad_parser.geometry import DERIVED_VERSION\n')
    geometry = tmp_path / 'wad_parser/geometry.py'
    geometry.write_text('DERIVED_VERSION = 5\n')
    diagnostic = tmp_path / 'cli/overfit_roomgen.py'
    diagnostic.write_text('diagnostic_only = 1\n')
    signature = source_preprocessing_fingerprint(root=tmp_path)
    diagnostic.write_text('diagnostic_only = 2\n')
    assert source_preprocessing_fingerprint(root=tmp_path) == signature
    geometry.write_text('DERIVED_VERSION = 6\n')
    assert source_preprocessing_fingerprint(root=tmp_path) != signature
    with pytest.raises(ValueError, match='unsupported source'):
        source_preprocessing_fingerprint('unknown', root=tmp_path)


@pytest.mark.parametrize('kind', ['layout', 'roomgen', 'properties', 'placement'])
def test_all_training_clis_validate_and_resume(tmp_path, kind):
    pytest.importorskip('torch')
    directory = _corpus(tmp_path / 'maps', count=3)
    module = __import__(f'cli.train_{kind}', fromlist=['main'])
    output = tmp_path / 'run'
    args = ['--maps-dir', str(directory), '--out', str(output), '--steps', '1',
            '--save-every', '1', '--validate-every', '1', '--device', 'cpu']
    args += ['--base', '8', '--resolution', '16', '--batch-size', '1', '--timesteps', '10'] if kind == 'layout' else ['--hidden', '8']
    if kind in {'properties', 'placement'}:
        args += ['--layers', '1']
    result = CliRunner().invoke(module.main, args)
    assert result.exit_code == 0, (result.output, result.exception)
    checkpoint = output / f'{kind}_000001.pt'
    assert checkpoint.exists() and (output / 'best.pt').exists()
    resumed_args = ['--maps-dir', str(directory), '--out', str(output), '--steps', '2',
                    '--resume', str(checkpoint), '--device', 'cpu']
    result = CliRunner().invoke(module.main, resumed_args)
    assert result.exit_code == 0, (result.output, result.exception)
    assert (output / f'{kind}_000002.pt').exists()
    metrics = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    assert [entry['step'] for entry in metrics] == [1, 2]
    assert all(np.isfinite(entry['validation_loss']) for entry in metrics)
    if kind == 'placement':
        for head in ('monster_position', 'pickup_position', 'monster_angle', 'pickup_angle'):
            assert metrics[0][f'validation_{head}'] > 0
    if kind == 'properties':
        assert 'validation_side_mid_tex' in metrics[0]
        assert 'validation_side_offsets' in metrics[0]


def test_roomgen_does_not_learn_false_endings_from_maps_without_reachable_exits(tmp_path):
    pytest.importorskip('torch')
    from cli.train_roomgen import main
    directory = _corpus(tmp_path / 'maps', count=3)
    for path in directory.glob('*.json'):
        raw = json.loads(path.read_text())
        for line in raw['linedefs']:
            line['special'] = 0
        path.write_text(json.dumps(raw))
    result = CliRunner().invoke(main, ['--maps-dir', str(directory), '--out', str(tmp_path / 'run'),
                                     '--steps', '1', '--hidden', '8'])
    assert result.exit_code != 0
    assert 'no training graph has a supported reachable exit' in result.output


def test_roomgen_rejects_exit_roles_without_a_physical_player_route(tmp_path):
    pytest.importorskip('torch')
    from cli.train_roomgen import main
    from tests.geometry_fixtures import neck_map
    directory = tmp_path / 'maps'
    directory.mkdir()
    for index in range(3):
        doc, _ = neck_map(24)
        doc.meta.wad_hash = f'physical-ending-{index}'
        doc.sectors[0].light_level += index
        (directory / f'{index}.json').write_text(json.dumps(doc.to_dict()))
    result = CliRunner().invoke(main, ['--maps-dir', str(directory), '--out', str(tmp_path / 'run'),
                                     '--steps', '1', '--hidden', '8'])
    assert result.exit_code != 0
    assert 'no physical player approach' in result.output
    assert 'no training graph has a supported reachable exit' in result.output


def test_incomplete_validation_graphs_do_not_select_a_best_checkpoint(tmp_path):
    pytest.importorskip('torch')
    from cli.train_roomgen import main
    directory = _corpus(tmp_path / 'maps', count=3)
    manifest = build_manifest(directory, seed=0)
    for entry in manifest['entries']:
        if entry['split'] == 'validation':
            path = directory / entry['path']
            raw = json.loads(path.read_text())
            for line in raw['linedefs']:
                line['special'] = 0
            path.write_text(json.dumps(raw))
    output = tmp_path / 'run'
    result = CliRunner().invoke(main, ['--maps-dir', str(directory), '--out', str(output),
                                     '--steps', '1', '--hidden', '8'])
    assert result.exit_code == 0, (result.output, result.exception)
    assert (output / 'roomgen_000001.pt').exists()
    assert not (output / 'best.pt').exists()
    assert 'No complete usable validation graphs' in result.output


@pytest.mark.parametrize('evaluation_device', ['cpu', 'cuda'])
def test_evaluation_uses_trained_sources_and_separates_fragment_losses(tmp_path, monkeypatch, evaluation_device):
    torch = pytest.importorskip('torch')
    if evaluation_device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA device unavailable')
    import generation.roomgen as roomgen
    original_load = roomgen.load_roomgen_model
    monkeypatch.setattr(roomgen, 'load_roomgen_model',
                        lambda checkpoint: original_load(checkpoint, device=evaluation_device))
    from cli.train_roomgen import main as train
    from cli.evaluate_roomgen import main as evaluate
    directory = _corpus(tmp_path / 'maps')
    manifest = build_manifest(directory, seed=0, test_fraction=0.34)
    for split in ('train', 'test'):
        entry = next(entry for entry in manifest['entries'] if entry['split'] == split)
        path = directory / entry['path']
        raw = json.loads(path.read_text())
        for line in raw['linedefs']:
            line['special'] = 0
        path.write_text(json.dumps(raw))
    manifest = build_manifest(directory, seed=0, test_fraction=0.34)
    manifest_path = tmp_path / 'corpus.json'
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / 'run'
    runner = CliRunner()
    result = runner.invoke(train, ['--maps-dir', str(directory), '--manifest', str(manifest_path),
                                  '--out', str(output), '--steps', '1', '--hidden', '8', '--device', 'cpu'])
    assert result.exit_code == 0, (result.output, result.exception)
    report_dir = tmp_path / 'evaluation'
    result = runner.invoke(evaluate, ['--maps-dir', str(directory),
                                     '--checkpoint', str(output / 'roomgen_000001.pt'),
                                     '--out', str(report_dir), '--samples', '1', '--max-rooms', '2',
                                     '--write-maps', '0'])
    assert result.exit_code == 0, (result.output, result.exception)
    report = json.loads((report_dir / 'report.json').read_text())
    assert torch.device(report['evaluation_device']).type == evaluation_device
    from data.source_endings import SOURCE_ENDINGS_VERSION
    assert report['source_endings_version'] == SOURCE_ENDINGS_VERSION
    assert report['graph_snapshot_version'] == 1
    assert report['panel'][0]['graph']['rooms']
    assert all(record['graph']['rooms'] for baseline in report['baselines'].values() for record in baseline['panel'])
    assert report['training_reference_graphs'] == 2
    assert sum(r['training_reference'] for r in report['sources']['train']) == 2
    assert all(r['complete_source'] == r['training_reference'] for r in report['sources']['train'])
    assert report['complete_test_graphs'] == report['fragment_test_graphs'] == 1
    complete = next(r for r in report['sources']['test'] if r['complete_source'])
    fragment = next(r for r in report['sources']['test'] if not r['complete_source'])
    assert report['unconditional_complete_test_prefix_loss'] == complete['unconditional_prefix_loss']
    assert report['unconditional_fragment_test_prefix_loss'] == fragment['unconditional_prefix_loss']
    assert not complete['prefix_censored'] and fragment['prefix_censored']
    preserved = {path.name: path.read_bytes() for path in report_dir.iterdir() if path.is_file()}
    rejected = runner.invoke(evaluate, ['--maps-dir', str(directory),
                                       '--checkpoint', str(output / 'roomgen_000001.pt'),
                                       '--out', str(report_dir), '--seed', '123', '--write-maps', '0'])
    assert rejected.exit_code != 0 and 'must be empty' in rejected.output
    assert {path.name: path.read_bytes() for path in report_dir.iterdir() if path.is_file()} == preserved
    # Rebuilding source references under a different traversal contract would
    # silently change the checkpoint's actual training/reference population.
    import torch
    checkpoint = torch.load(output / 'roomgen_000001.pt', weights_only=False)
    checkpoint['config']['traversal_version'] -= 1
    changed = output / 'old-preprocessing.pt'
    torch.save(checkpoint, changed)
    rejected = runner.invoke(evaluate, ['--maps-dir', str(directory), '--checkpoint', str(changed),
                                      '--out', str(tmp_path / 'incompatible'), '--write-maps', '0'])
    assert rejected.exit_code != 0 and 'archived source' in rejected.output
    # A publication fault must leave complete evidence, including the
    # distinction between a valid graph and an unpublished WAD.
    def publication_failure(*args, **kwargs):
        raise ValueError('simulated publication failure')
    realized_seeds = []
    def fake_realize(graph, *, seed):
        realized_seeds.append(seed)
        from tests.geometry_fixtures import two_rooms
        doc = two_rooms()
        doc.meta.map_name = graph.map_name
        return doc
    monkeypatch.setattr('generation.roomgen.generate', publication_failure)
    monkeypatch.setattr('generation.baselines.realize', fake_realize)
    monkeypatch.setattr('generation.pipeline.documents_to_wad', publication_failure)
    failed_dir = tmp_path / 'publication-failed'
    result = runner.invoke(evaluate, ['--maps-dir', str(directory),
                                      '--checkpoint', str(output / 'roomgen_000001.pt'),
                                      '--out', str(failed_dir), '--seed', '11', '--samples', '1',
                                      '--write-maps', '1', '--max-rooms', '2'])
    assert result.exit_code == 0, (result.output, result.exception)
    failed = json.loads((failed_dir / 'report.json').read_text())
    import hashlib
    assert failed['checkpoint_sha256'] == hashlib.sha256((output / 'roomgen_000001.pt').read_bytes()).hexdigest()
    assert not failed['checkpoint_changed_during_evaluation']
    assert realized_seeds == [11, 11]
    assert failed['geometry']['requested_seeds'] == [11]
    assert failed['geometry']['outcomes'][0]['validated'] is None
    assert failed['geometry']['outcomes'][0]['failure_stage'] == 'batch'
    for baseline in failed['baselines'].values():
        assert baseline['geometry']['written_maps'] == 0
        assert baseline['geometry']['outcomes'] == [
            {'seed': 11, 'validated': True, 'written': False,
             'reason': 'simulated publication failure', 'failure_stage': 'publication'}]

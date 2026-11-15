import json
import subprocess
import sys

import pytest

torch = pytest.importorskip('torch')

from data.corpus import build_manifest
from generation.geometry_model import sample_checkpoint
from tests.geometry_fixtures import polygon_document, square, two_rooms
from tests.progression_fixtures import add_exit
from tools.prepare_learned_geometry import prepare
from tools.train_geometry import prepared_pools, train


def prepared_sources(tmp_path):
    root = tmp_path / 'maps'
    root.mkdir()
    for index in range(2):
        doc = polygon_document([[square(0, 0, 128, 128)]], start=(64, 64)) if index == 0 else two_rooms()
        doc.meta.wad_hash = 'same-training-family'
        if index == 1:
            doc.sectors[1].floor_height = 24
        add_exit(doc, index)
        (root / f'{index}.json').write_text(json.dumps(doc.to_dict()))
    manifest = build_manifest(root, validation_fraction=0, test_fraction=0)
    path = tmp_path / 'corpus.json'
    path.write_text(json.dumps(manifest))
    directory = tmp_path / 'prepared'
    prepare(root, path, directory)
    return directory


def tiny_train(prepared, out, steps, **kwargs):
    return train(prepared, out, steps=steps, hidden=8, embedding=8, layers=1,
                 context_dim=4, max_tokens=2048, save_every=1, validate_every=1, **kwargs)


def test_full_sequence_training_resume_is_exact_and_sampling_needs_no_source_targets(tmp_path):
    old_threads = torch.get_num_threads()
    try:
        prepared = prepared_sources(tmp_path)
        tiny_train(prepared, tmp_path / 'full', 3)
        tiny_train(prepared, tmp_path / 'split', 1)
        checkpoint = tmp_path / 'split/geometry_000001.pt'
        tiny_train(prepared, tmp_path / 'split', 3, resume=checkpoint)
        first = torch.load(tmp_path / 'full/geometry_000003.pt', weights_only=False)
        second = torch.load(tmp_path / 'split/geometry_000003.pt', weights_only=False)
        assert all(torch.equal(value, second['model'][name]) for name, value in first['model'].items())
        assert all(torch.equal(value, second['conditioner'][name]) for name, value in first['conditioner'].items())
        assert first['corpus']['fingerprint'] == second['corpus']['fingerprint']
        assert len(first['corpus']['entries']) == 2
        assert not (tmp_path / 'full/best.pt').exists()
        # Remove the entire source/preparation path from its expected location.
        prepared.rename(tmp_path / 'preparation-not-available-to-inference')
        context = {'tags': {}, 'target_game': ''}
        result = sample_checkpoint(tmp_path / 'full/geometry_000003.pt', context, seed=7, max_tokens=12)
        assert result['model_execution'] == 'executed'
        assert result['sample']['tokens']
        assert result['status'] == 'budget_exhausted'
        assert result['prediction'] is None
    finally:
        torch.set_num_threads(old_threads)


def test_sequence_budget_rejects_whole_maps_without_false_ending_or_lost_exclusions(tmp_path):
    prepared = prepared_sources(tmp_path)
    pools, manifest = prepared_pools(prepared, max_tokens=8)
    assert not pools['train']
    assert len(manifest['excluded']) == 2
    assert all('not truncated' in row['reason'] for row in manifest['excluded'])
    old_threads = torch.get_num_threads()
    try:
        with pytest.raises(ValueError, match='input-assessment'):
            train(prepared, tmp_path / 'failed-training', steps=1, max_tokens=8)
        assert (tmp_path / 'failed-training/input-assessment.json').exists()
    finally:
        torch.set_num_threads(old_threads)


def test_training_and_sampling_help_work_without_torch():
    code = '''
import importlib.abc
import sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('Torch was imported')
sys.meta_path.insert(0, NoTorch())
import runpy
sys.argv = ['tool', '--help']
try:
    runpy.run_module(MODULE, run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
'''
    for module in ('tools.train_geometry', 'tools.sample_geometry', 'tools.generate_geometry'):
        subprocess.run([sys.executable, '-c', code.replace('MODULE', repr(module))],
                       check=True, capture_output=True, text=True)


def test_game_asset_checks_reject_without_rewriting_predictions():
    import copy
    from generation.game_profile import GameProfile
    from generation.geometry_model import require_game_assets
    source = two_rooms()
    before = copy.deepcopy(source.to_dict())
    game = GameProfile('restricted', textures={'STARTAN2'}, flats={'FLOOR4_8'})
    with pytest.raises(ValueError, match='unsupported by target game'):
        require_game_assets(source, game)
    assert source.to_dict() == before
    game.flats.add('CEIL3_5')
    require_game_assets(source, game)
    assert source.to_dict() == before


def test_all_rejected_generation_retains_each_attempt_and_existing_destination(tmp_path, monkeypatch):
    import shutil
    import generation.geometry_model as module
    from generation.game_profile import GameProfile
    from generation.pipeline import NoUsableMapsError
    old_threads = torch.get_num_threads()
    monkeypatch.setattr(shutil, 'which', lambda name: '/unused/zdbsp')
    monkeypatch.setattr(module, 'load_model', lambda *args, **kwargs: (None, None, {}))
    monkeypatch.setattr(module, 'infer', lambda *args, **kwargs: {
        'status': 'budget_exhausted', 'sample': {'tokens': [1], 'seed': kwargs['seed']}})
    checkpoint = tmp_path / 'weights.pt'
    checkpoint.write_bytes(b'unit-test checkpoint identity')
    destination = tmp_path / 'existing.wad'
    destination.write_bytes(b'preserve original destination')
    try:
        with pytest.raises(NoUsableMapsError):
            module.generate_checkpoint(checkpoint, {'tags': {}, 'target_game': ''},
                                       out_path=destination, report_dir=tmp_path / 'reports',
                                       game=GameProfile('fixture'), n_maps=2, max_attempts=2, seed=7)
        assert destination.read_bytes() == b'preserve original destination'
        report = json.loads((tmp_path / 'reports/report.json').read_text())
        assert report['status'] == 'failed'
        assert [row['seed'] for row in report['records']] == [7, 8, 9, 10]
        assert len(list((tmp_path / 'reports').glob('0*-*.json'))) == 4
        for row in report['records']:
            detail = json.loads((tmp_path / 'reports' / row['record']).read_text())
            assert detail['sample']['tokens'] == [1]
            assert detail['sample']['seed'] == row['seed']
    finally:
        torch.set_num_threads(old_threads)


def test_compact_training_resume_and_layout_filter_preserve_contracts(tmp_path):
    prepared = prepared_sources(tmp_path)
    pools, manifest = prepared_pools(prepared, token_codec='compact', min_outline_corners=6)
    assert not pools['train']
    assert len(manifest['excluded']) == 2
    assert all('genuine corners' in row['reason'] for row in manifest['excluded'])
    old_threads = torch.get_num_threads()
    try:
        tiny_train(prepared, tmp_path / 'full-compact', 2, token_codec='compact', stop_weight=20)
        tiny_train(prepared, tmp_path / 'split-compact', 1, token_codec='compact', stop_weight=20)
        ck = tmp_path / 'split-compact/geometry_000001.pt'
        tiny_train(prepared, tmp_path / 'split-compact', 2, token_codec='compact', resume=ck)
        a = torch.load(tmp_path / 'full-compact/geometry_000002.pt', weights_only=False)
        b = torch.load(tmp_path / 'split-compact/geometry_000002.pt', weights_only=False)
        assert all(torch.equal(v, b['model'][key]) for key, v in a['model'].items())
        assert a['config']['token_schema'] == 'doom_geometry_field_tokens_v1'
        assert a['config']['stop_weight'] == a['settings']['stop_weight'] == 20
        report = sample_checkpoint(tmp_path / 'full-compact/geometry_000002.pt',
                                   {'tags': {}, 'target_game': ''}, max_tokens=12)
        assert report['sample']['status'] == 'budget_exhausted'
    finally:
        torch.set_num_threads(old_threads)


def test_requested_varied_geometry_rejects_a_valid_square_without_repair(tmp_path, monkeypatch):
    import shutil
    import generation.geometry_model as module
    from generation.game_profile import GameProfile
    from generation.pipeline import NoUsableMapsError
    monkeypatch.setattr(shutil, 'which', lambda name: '/unused/zdbsp')
    monkeypatch.setattr(module, 'load_model', lambda *args, **kwargs: (None, None, {}))
    monkeypatch.setattr(module, 'infer', lambda *args, **kwargs: {
        'status': 'static_valid', 'sample': {'tokens': [1]},
        'layout': {'single_flat_rectangle': True}})
    checkpoint = tmp_path / 'weights.pt'
    checkpoint.write_bytes(b'unit-test checkpoint identity')
    destination = tmp_path / 'old.wad'
    destination.write_bytes(b'old')
    old_threads = torch.get_num_threads()
    try:
        with pytest.raises(NoUsableMapsError):
            module.generate_checkpoint(checkpoint, {'tags': {}, 'target_game': ''},
                                       out_path=destination, report_dir=tmp_path / 'report',
                                       game=GameProfile('fixture'), reject_flat_rectangles=True)
        detail = json.loads((tmp_path / 'report/000-000.json').read_text())
        assert detail['failure_stage'] == 'layout'
        assert 'single flat rectangle' in detail['reason']
        assert destination.read_bytes() == b'old'
    finally:
        torch.set_num_threads(old_threads)


def test_family_sampling_chooses_families_before_maps_and_is_rng_reproducible():
    import numpy as np
    from tools.train_geometry import FamilyPool
    items = [{'family': 'a', 'path': str(i)} for i in range(20)] + [{'family': 'b', 'path': 'only-b'}]
    sampler = FamilyPool(items)
    assert len(sampler) == 2
    assert [len(group) for group in sampler.groups] == [20, 1]
    state = np.random.get_state()
    try:
        np.random.seed(7)
        first = [sampler[i % 2]['path'] for i in range(20)]
        np.random.seed(7)
        assert first == [sampler[i % 2]['path'] for i in range(20)]
        assert first[1::2] == ['only-b'] * 10
    finally:
        np.random.set_state(state)


@pytest.mark.parametrize('loss_balance', ['token', 'field'])
def test_relative_memory_training_resume_and_checkpoint_wad_export(tmp_path, loss_balance):
    from data.geometry_memory import ANCHOR_SCHEMA
    from data.geometry_relative import SCHEMA
    from tools.sample_geometry import sample_panel
    from wad_parser.wad_reader import WadFile
    prepared = prepared_sources(tmp_path)
    old_threads = torch.get_num_threads()
    kwargs = {'token_codec': 'relative', 'record_memory': True, 'memory_read': True,
              'record_anchors': True, 'sampling': 'family', 'batch_size': 2, 'loss_balance': loss_balance}
    try:
        tiny_train(prepared, tmp_path / 'relative-full', 2, **kwargs)
        tiny_train(prepared, tmp_path / 'relative-split', 1, **kwargs)
        tiny_train(prepared, tmp_path / 'relative-split', 2,
                   resume=tmp_path / 'relative-split/geometry_000001.pt')
        a = torch.load(tmp_path / 'relative-full/geometry_000002.pt', weights_only=False)
        b = torch.load(tmp_path / 'relative-split/geometry_000002.pt', weights_only=False)
        assert all(torch.equal(value, b['model'][key]) for key, value in a['model'].items())
        assert a['config']['token_schema'] == SCHEMA
        assert a['config']['memory_schema'] == ANCHOR_SCHEMA
        if loss_balance == 'field':
            assert a['config']['loss'] == 'section_field_stop_balance_v1'
        prepared.rename(tmp_path / 'removed-relative-preparation')
        report = sample_panel(tmp_path / 'relative-full/geometry_000002.pt', tmp_path / 'samples',
                              samples=1, seed=0, max_tokens=12, raw_wads=True)
        detail = json.loads((tmp_path / 'samples/0000.json').read_text())
        assert report['raw_diagnostic_wads']
        assert detail['raw_wad']['kind'] == 'incomplete_trace_only'
        assert WadFile(detail['raw_wad']['path']).find_map_groups() == []
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize('token_codec,relative_pointers,spatial_pointers,sampling,dependency_memory,record_values,ending_loss_weight', [('interleaved', False, False, 'family', False, False, 0), ('interleaved', True, False, 'family', False, False, 0), ('interleaved', True, True, 'family', False, False, 0), ('interleaved', True, True, 'coverage-family', False, False, 0), ('interleaved', True, True, 'coverage-family', True, False, 0), ('interleaved', True, True, 'coverage-family', True, False, .1), ('relative-reference', True, True, 'coverage-family', True, False, 0), ('interleaved', True, True, 'coverage-family', True, True, 0), ('relative-reference', True, True, 'coverage-family', True, True, 0)])
def test_interleaved_memory_exact_resume_and_source_free_export(tmp_path, token_codec, relative_pointers, spatial_pointers, sampling, dependency_memory, record_values, ending_loss_weight):
    from tools.sample_geometry import sample_panel
    prepared = prepared_sources(tmp_path)
    old_threads = torch.get_num_threads()
    kwargs = {'token_codec': token_codec, 'record_memory': True, 'memory_read': True,
              'record_anchors': True, 'sampling': sampling, 'batch_size': 2, 'stop_weight': 5,
              'relative_pointers': relative_pointers, 'spatial_pointers': spatial_pointers,
              'dependency_memory': dependency_memory, 'record_values': record_values,
              'ending_loss_weight': ending_loss_weight}
    try:
        tiny_train(prepared, tmp_path/'interleaved-full', 2, **kwargs)
        tiny_train(prepared, tmp_path/'interleaved-split', 1, **kwargs)
        tiny_train(prepared, tmp_path/'interleaved-split', 2,
                   resume=tmp_path/'interleaved-split/geometry_000001.pt')
        a = torch.load(tmp_path/'interleaved-full/geometry_000002.pt', weights_only=False)
        b = torch.load(tmp_path/'interleaved-split/geometry_000002.pt', weights_only=False)
        assert all(torch.equal(v, b['model'][k]) for k, v in a['model'].items())
        assert all(torch.equal(v, b['conditioner'][k]) for k, v in a['conditioner'].items())
        traces = [json.loads(line) for path in sorted((tmp_path/'interleaved-full').glob('sampling-trace-*.jsonl'))
                  for line in path.read_text().splitlines()]
        assert [r['step'] for r in traces] == [1, 2]
        if sampling == 'coverage-family':
            assert len({r['path'] for r in traces[0]['sources']}) == 2
        assert a['config']['memory_schema'] == (
            ('doom_geometry_interleaved_relative_reference_record_values_v9' if token_codec == 'relative-reference'
             else 'doom_geometry_record_values_v9') if record_values else
            'doom_geometry_interleaved_relative_reference_memory_v8' if token_codec == 'relative-reference' else
            'doom_geometry_interleaved_dependency_memory_v7' if dependency_memory else
            'doom_geometry_interleaved_spatial_pointer_v6' if spatial_pointers else
            'doom_geometry_interleaved_relative_pointer_v5' if relative_pointers
            else 'doom_geometry_interleaved_memory_v4')
        assert a['config'].get('ending_loss_weight', 0) == ending_loss_weight
        if ending_loss_weight:
            assert a['config']['ending_auxiliary_schema'] == 'doom_geometry_global_ending_class_mean_v1'
        assert all(r['encoding']['source_authoring_restored'] for r in a['corpus']['entries'])
        prepared.rename(tmp_path/'removed-interleaved-prepared')
        panel = sample_panel(tmp_path/'interleaved-full/geometry_000002.pt',tmp_path/'interleaved-panel',
                             samples=1, seed=0, max_tokens=12, raw_wads=True)
        assert panel['records'][0]['status'] == 'budget_exhausted'
        assert (tmp_path/'interleaved-panel/0000-raw.wad').exists()
        from generation.geometry_model import load_model
        a['config']['memory_schema'] = 'doom_geometry_record_anchor_v3'
        torch.save(a, tmp_path/'invalid-memory-order.pt')
        with pytest.raises(ValueError, match='ordering contract'):
            load_model(tmp_path/'invalid-memory-order.pt')
        a['config']['memory_schema'] = ('doom_geometry_interleaved_dependency_memory_v7'
                                        if token_codec == 'relative-reference'
                                        else 'doom_geometry_interleaved_relative_reference_memory_v8')
        torch.save(a, tmp_path/'invalid-relative-reference-contract.pt')
        with pytest.raises(ValueError, match='relative-reference memory contract'):
            load_model(tmp_path/'invalid-relative-reference-contract.pt')
    finally:
        torch.set_num_threads(old_threads)


def test_complete_coverage_then_family_sampling_resumes_without_consuming_constructor_rng():
    import numpy as np
    from tools.train_geometry import CoverageFamilyPool
    state = np.random.get_state()
    items = [{'path': str(i), 'family': 'large' if i < 20 else str(i)} for i in range(28)]
    try:
        np.random.seed(7);before = np.random.get_state()
        pool = CoverageFamilyPool(items, seed=7)
        after = np.random.get_state()
        assert (before[1] == after[1]).all() and before[2:] == after[2:]
        prefix = [pool[int(np.random.randint(len(pool)))]['path'] for _ in range(25)]
        saved = np.random.get_state()
        tail = [pool[int(np.random.randint(len(pool)))]['path'] for _ in range(20)]
        assert set((prefix+tail)[:28]) == {str(i) for i in range(28)}
        np.random.set_state(saved)
        resumed = CoverageFamilyPool(items, seed=7, offset=25)
        assert [resumed[int(np.random.randint(len(resumed)))]['path'] for _ in range(20)] == tail
    finally:
        np.random.set_state(state)


def test_automatic_checkpoint_wads_do_not_perturb_training_rng_or_weights(tmp_path):
    from wad_parser.wad_reader import WadFile
    prepared = prepared_sources(tmp_path)
    old_threads = torch.get_num_threads()
    try:
        tiny_train(prepared, tmp_path / 'without-wads', 2, token_codec='relative')
        tiny_train(prepared, tmp_path / 'with-wads', 2, token_codec='relative',
                   checkpoint_wads=tmp_path / 'wad-checkpoints')
        first = torch.load(tmp_path / 'without-wads/geometry_000002.pt', weights_only=False)
        other = torch.load(tmp_path / 'with-wads/geometry_000002.pt', weights_only=False)
        assert all(torch.equal(v, other['model'][k]) for k, v in first['model'].items())
        assert all(torch.equal(v, other['conditioner'][k]) for k, v in first['conditioner'].items())
        assert torch.equal(first['rng']['torch'], other['rng']['torch'])
        assert (first['rng']['numpy'][1] == other['rng']['numpy'][1]).all()
        for step in (0, 1, 2):
            folder = tmp_path / f'wad-checkpoints/step-{step:06d}'
            detail = json.loads((folder / 'sample/0000.json').read_text())
            assert WadFile(detail['raw_wad']['path']).identification == 'PWAD'
            snapshot = torch.load(folder / 'weights.pt', weights_only=False)
            if step:
                saved = torch.load(tmp_path / f'with-wads/geometry_{step:06d}.pt', weights_only=False)
                assert all(torch.equal(v, saved['model'][k]) for k, v in snapshot['model'].items())
    finally:
        torch.set_num_threads(old_threads)


def test_checkpoint_export_failure_preserves_training_checkpoint_and_error(tmp_path, monkeypatch):
    import tools.sample_geometry as sampling
    prepared = prepared_sources(tmp_path)
    original = sampling.sample_panel
    def fail_one(checkpoint, out, **kwargs):
        if 'step-000001' in str(out):
            raise RuntimeError('retained diagnostic export failure')
        return original(checkpoint, out, **kwargs)
    monkeypatch.setattr(sampling, 'sample_panel', fail_one)
    old_threads = torch.get_num_threads()
    try:
        with pytest.raises(ValueError, match='training checkpoints retained'):
            tiny_train(prepared, tmp_path / 'training', 1, token_codec='relative',
                       checkpoint_wads=tmp_path / 'wads')
        assert (tmp_path / 'training/geometry_000001.pt').exists()
        error = json.loads((tmp_path / 'wads/export-error-000001.json').read_text())
        assert 'retained diagnostic export failure' in error['error']
        assert error['step'] == 1
    finally:
        torch.set_num_threads(old_threads)


def test_record_memory_family_balanced_ragged_batches_resume_exactly_and_load_without_sources(tmp_path):
    from data.geometry_memory import READ_SCHEMA
    maps = tmp_path / 'maps'
    maps.mkdir()
    for index in range(3):
        doc = (two_rooms() if index == 2 else
               polygon_document([[square(0, 0, 128 + index*32, 128)]], start=(64, 64)))
        doc.meta.wad_hash = 'family-a' if index < 2 else 'family-b'
        add_exit(doc, 1 if index == 2 else 0)
        (maps / f'{index}.json').write_text(json.dumps(doc.to_dict()))
    manifest = tmp_path / 'corpus.json'
    manifest.write_text(json.dumps(build_manifest(maps, validation_fraction=0, test_fraction=0)))
    prepared = tmp_path / 'prepared'
    prepare(maps, manifest, prepared)
    old_threads = torch.get_num_threads()
    kwargs = {'token_codec': 'compact', 'record_memory': True, 'sampling': 'family',
              'batch_size': 2, 'stop_weight': 5, 'memory_read': True}
    try:
        tiny_train(prepared, tmp_path / 'full-memory', 3, **kwargs)
        tiny_train(prepared, tmp_path / 'split-memory', 1, **kwargs)
        checkpoint = tmp_path / 'split-memory/geometry_000001.pt'
        tiny_train(prepared, tmp_path / 'split-memory', 3, resume=checkpoint)
        a = torch.load(tmp_path / 'full-memory/geometry_000003.pt', weights_only=False)
        b = torch.load(tmp_path / 'split-memory/geometry_000003.pt', weights_only=False)
        assert all(torch.equal(value, b['model'][name]) for name, value in a['model'].items())
        assert all(torch.equal(value, b['conditioner'][name]) for name, value in a['conditioner'].items())
        assert a['config']['memory_schema'] == READ_SCHEMA
        assert a['settings']['sampling'] == 'family'
        assert a['corpus']['fingerprint'] == b['corpus']['fingerprint']
        maps.rename(tmp_path / 'removed-maps')
        prepared.rename(tmp_path / 'removed-preparation')
        report = sample_checkpoint(tmp_path / 'full-memory/geometry_000003.pt',
                                   {'tags': {}, 'target_game': ''}, max_tokens=24, seed=7)
        assert report['model_execution'] == 'executed'
        assert report['sample']['tokens']
    finally:
        torch.set_num_threads(old_threads)

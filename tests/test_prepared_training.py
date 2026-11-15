import copy
import hashlib
import json

import pytest

torch = pytest.importorskip('torch')

from tests.pipeline_fixtures import source_wads
from tools.audit_sources import audit
from tools.materialize_sources import materialize
from tools.train_prepared_roomgen import prepared_pools, train


def test_prepared_training_censors_fragments_and_resumes_exactly_with_verified_targets(tmp_path):
    from models.training import rng_state, restore_rng
    saved = rng_state()
    try:
        source = source_wads(tmp_path)
        audit(source, tmp_path / 'audit', workers=2)
        corpus = tmp_path / 'corpus'
        materialize(tmp_path / 'audit/report.json', corpus, include_fragments=True)
        result = train(corpus, tmp_path / 'whole', include_incomplete=True, hidden=8, steps=4,
                       max_train_rooms=16, seed=3, validate_every=2)
        assert result['pools']['train'] == {'graphs': 2, 'complete': 1}
        assert result['pools']['validation']['graphs'] == 0
        assert not result['code_changed_during_training']
        train(corpus, tmp_path / 'resumed', include_incomplete=True, hidden=8, steps=2,
              max_train_rooms=16, seed=3, validate_every=2)
        train(corpus, tmp_path / 'resumed', steps=4, resume=tmp_path / 'resumed/roomgen_000002.pt',
              validate_every=2)
        whole = torch.load(tmp_path / 'whole/roomgen_000004.pt', weights_only=False)
        resumed = torch.load(tmp_path / 'resumed/roomgen_000004.pt', weights_only=False)
        for key in ('model', 'conditioner'):
            assert whole[key].keys() == resumed[key].keys()
            assert all(torch.equal(whole[key][name], resumed[key][name]) for name in whole[key])
        assert torch.equal(whole['rng']['torch'], resumed['rng']['torch'])
        assert 'stop_targets' not in whole['settings'] and 'complete_repeat' not in whole['settings']
        # Opt-in complete-map stop hazard: settings are recorded, the complete
        # map is repeated in the sampled pool, and resume keeps the objective.
        options = dict(include_incomplete=True, hidden=8, max_train_rooms=16, seed=3, validate_every=2,
                       stop_targets='complete', complete_repeat=3)
        stopped = train(corpus, tmp_path / 'complete-stop', steps=4, **options)
        assert stopped['pools']['train'] == {'graphs': 2, 'complete': 1, 'sampled_pool': 4}
        train(corpus, tmp_path / 'complete-stop-resumed', steps=2, **options)
        train(corpus, tmp_path / 'complete-stop-resumed', steps=4, validate_every=2,
              resume=tmp_path / 'complete-stop-resumed/roomgen_000002.pt')
        first = torch.load(tmp_path / 'complete-stop/roomgen_000004.pt', weights_only=False)
        second = torch.load(tmp_path / 'complete-stop-resumed/roomgen_000004.pt', weights_only=False)
        assert first['settings']['stop_targets'] == 'complete' and first['settings']['complete_repeat'] == 3
        assert all(torch.equal(first['model'][name], second['model'][name]) for name in first['model'])
        assert not all(torch.equal(first['model'][name], whole['model'][name]) for name in first['model'])
        with pytest.raises(ValueError, match='stop settings'):
            train(corpus, tmp_path / 'complete-stop-resumed', steps=6, validate_every=2, stop_targets='all',
                  complete_repeat=2, resume=tmp_path / 'complete-stop-resumed/roomgen_000004.pt')
        ends = train(corpus, tmp_path / 'reachable-ends', steps=2, include_incomplete=True, hidden=8,
                     max_train_rooms=16, seed=3, validate_every=2, stop_targets='reachable-ends')
        assert ends['pools']['train'] == {'graphs': 2, 'complete': 1}
        ended = torch.load(tmp_path / 'reachable-ends/roomgen_000002.pt', weights_only=False)
        assert ended['settings']['stop_targets'] == 'reachable-ends' and ended['settings']['complete_repeat'] == 1
        from click.testing import CliRunner
        from cli.evaluate_roomgen import main as normal_evaluate
        from tools.evaluate_prepared_roomgen import evaluate
        normal_out = tmp_path / 'normal-eval'
        ordinary = CliRunner().invoke(normal_evaluate, [
            '--maps-dir', str(corpus / 'maps'), '--checkpoint', str(tmp_path / 'whole/roomgen_000004.pt'),
            '--out', str(normal_out), '--samples', '2', '--max-rooms', '3', '--write-maps', '0'])
        assert ordinary.exit_code == 0, ordinary.output
        cached = evaluate(corpus, tmp_path / 'whole/roomgen_000004.pt', tmp_path / 'cached-eval',
                          samples=2, max_rooms=3, write_maps=0)
        normal = json.loads((normal_out / 'report.json').read_text())
        for key in ('panel', 'sources', 'baselines', 'exact_binned_training_matches'):
            assert cached[key] == normal[key]
        assert not cached['prepared_targets']['source_edits_during_evaluation']
        # An incomplete example must never become a best-checkpoint validation
        # target, even if the source-family manifest places it in that holdout.
        report = json.loads((corpus / 'report.json').read_text())
        fragment = next(row['path'] for row in report['records'] if not row['ending']['complete'])
        manifest = json.loads((corpus / 'corpus.json').read_text())
        changed = copy.deepcopy(manifest)
        next(row for row in changed['entries'] if row['path'] == fragment)['split'] = 'validation'
        changed.pop('fingerprint')
        changed['fingerprint'] = hashlib.sha256(json.dumps(changed, sort_keys=True).encode()).hexdigest()
        custom = tmp_path / 'fragment-holdout.json'
        custom.write_text(json.dumps(changed))
        pools, _, _ = prepared_pools(corpus, manifest_path=custom, include_incomplete=True)
        assert not pools['validation'] and len(pools['train']) == 1
        path = corpus / 'records' / report['records'][0]['path']
        path.write_bytes(path.read_bytes() + b' ')
        with pytest.raises(ValueError, match='snapshot changed'):
            prepared_pools(corpus, include_incomplete=True)
    finally:
        restore_rng(saved)

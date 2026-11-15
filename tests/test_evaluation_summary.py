import copy
import json
import math

import pytest

from tools.summarize_evaluations import outcomes, rate, summarize


def panel(seeds, neural, baseline, *, legacy=False):
    def geometry(written, *, neural_method):
        result = {'written_maps': sum(written),
                  'failed_samples': [{'seed': s, 'reason': 'required edge missing'}
                                     for s, value in zip(seeds, written) if not value]}
        if legacy:
            if neural_method:
                result['summaries'] = [{'seed': s} for s, value in zip(seeds, written) if value]
        else:
            result.update(requested_seeds=seeds,
                outcomes=[{'seed': s, 'written': value, 'validated': value} for s, value in zip(seeds, written)])
        return result
    samples = [{'seed': s, 'rooms': 1 if i == 0 else 8, 'matches_binned_training_graph': False}
               for i, s in enumerate(seeds)]
    result = {'training_step': 50, 'checkpoint': '/model/best.pt', 'corpus_fingerprint': 'corpus',
              'temperature': 1., 'max_rooms': 8, 'source_endings_version': 'physical_exit_v3',
              'training_code_fingerprint': 'train', 'evaluation_code_fingerprint': 'eval',
              'code_changed_during_evaluation': False, 'checkpoint_changed_during_evaluation': False,
              'panel': samples, 'geometry': geometry(neural, neural_method=True),
              'baselines': {'procedural': {'panel': copy.deepcopy(samples),
                                          'geometry': geometry(baseline, neural_method=False)}}}
    if not legacy:
        result['checkpoint_sha256'] = 'checkpoint-content'
    return result


def save(tmp_path, name, value):
    path = tmp_path / name
    path.write_text(json.dumps(value))
    return path


def test_wilson_interval_edges_and_empty_experiment():
    assert rate(0, 0)['rate'] is None and rate(0, 0)['wilson_95'] is None
    assert rate(0, 10)['wilson_95'] == pytest.approx([0, .2775327998628892])
    assert rate(10, 10)['wilson_95'] == pytest.approx([.7224672001371108, 1])
    assert rate(5, 10)['wilson_95'] == pytest.approx([.236593090512564, .763406909487436])


def test_pool_disjoint_panels_and_keep_paired_failures(tmp_path):
    a = save(tmp_path, 'a.json', panel([0, 1], [True, False], [False, False]))
    b = save(tmp_path, 'b.json', panel([2, 3], [True, False], [False, True]))
    result = summarize([a, b], tmp_path / 'summary')
    assert result['identity']['checkpoint_identity'] == 'content_hash'
    neural = result['methods']['neural']
    assert neural['publication']['rate'] == .5
    assert neural['room_counts'] == {1: 2, 8: 2}
    assert neural['room_budget_hits']['successes'] == 2
    paired = result['paired_neural_vs_baseline']['procedural']
    assert paired == {'paired_attempts': 4, 'neural_only_written': 2, 'baseline_only_written': 1,
                      'both_written': 0, 'neither_written': 1, 'write_rate_difference': .25,
                      'exact_two_sided_discordant_p': 1.}
    assert (tmp_path / 'summary/input-00.json').read_bytes() == a.read_bytes()
    with pytest.raises(ValueError, match='must be empty'):
        summarize([a, b], tmp_path / 'summary')


def test_legacy_prefix_reconstruction_has_explicit_identity_limit(tmp_path):
    p = save(tmp_path, 'legacy.json', panel([6, 7], [False, True], [True, False], legacy=True))
    result = summarize([p], tmp_path / 'summary')
    assert result['identity']['checkpoint_identity'] == 'path_and_step_only'
    assert any('lack checkpoint hashes' in limit for limit in result['limits'])
    assert result['methods']['procedural']['publication']['rate'] == .5


@pytest.mark.parametrize('fault', ['duplicate_seeds', 'changed_code', 'changed_checkpoint', 'mismatched_settings',
                                   'mismatched_hash', 'mismatched_attempts', 'wrong_written_count'])
def test_reject_incomparable_or_corrupt_evidence_before_writing(tmp_path, fault):
    a = panel([0, 1], [True, False], [False, True])
    b = panel([2, 3], [True, False], [False, True])
    if fault == 'duplicate_seeds':
        b = copy.deepcopy(a)
    elif fault == 'changed_code':
        b['code_changed_during_evaluation'] = True
    elif fault == 'changed_checkpoint':
        b['checkpoint_changed_during_evaluation'] = True
    elif fault == 'mismatched_settings':
        b['temperature'] = .4
    elif fault == 'mismatched_hash':
        b['checkpoint_sha256'] = 'different-content'
    elif fault == 'mismatched_attempts':
        b['baselines']['procedural']['geometry'] = {'written_maps': 0, 'requested_seeds': [], 'outcomes': []}
    else:
        b['geometry']['written_maps'] = 2
    paths = [save(tmp_path, 'a.json', a), save(tmp_path, 'b.json', b)]
    out = tmp_path / 'summary'
    with pytest.raises(ValueError):
        summarize(paths, out)
    assert not out.exists()


def test_legacy_batch_errors_remain_inconclusive():
    with pytest.raises(ValueError, match='no per-seed outcomes'):
        outcomes([{'seed': 7}], {'written_maps': 0, 'failure': 'publisher failed'}, neural=True)


def test_exact_discordant_test_uses_all_and_only_paired_attempts(tmp_path):
    p = save(tmp_path, 'panel.json', panel(list(range(8)), [True] * 8, [False] * 8))
    result = summarize([p], tmp_path / 'summary')
    assert math.isclose(result['paired_neural_vs_baseline']['procedural']['exact_two_sided_discordant_p'], 2 / 256)

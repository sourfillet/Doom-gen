import pytest

torch = pytest.importorskip('torch')

from data.geometry_tokens import END_LIST, RECORD, PAD, VOCAB_SIZE, encode_prediction, prefix_features, token_masks
from data.learned_geometry import build_target
from models.geometry import GeometryRNN, condition_tags
from tests.geometry_fixtures import two_rooms


def sample_data():
    values = encode_prediction(build_target(two_rooms())['supervision'])
    tokens = torch.tensor([values])
    allowed = torch.zeros(1, len(values) - 1, VOCAB_SIZE, dtype=torch.bool)
    for index, row in enumerate(token_masks(values)):
        allowed[0, index, list(row)] = True
    return tokens, allowed


def test_shared_model_learns_causal_predictions_and_ignores_forced_syntax_in_metrics():
    torch.manual_seed(7)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = GeometryRNN(hidden=24, embedding=16, layers=1, context_dim=8)
        tokens, allowed = sample_data()
        context = torch.zeros(1, 8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        initial, metrics = model.loss(tokens, context, allowed)
        before = float(initial.detach())
        assert 0 < metrics['forced_fraction'] < 1
        for _ in range(12):
            optimizer.zero_grad()
            loss, _ = model.loss(tokens, context, allowed)
            loss.backward()
            optimizer.step()
        after, _ = model.loss(tokens, context, allowed)
        assert float(after.detach()) < before * 0.75
    finally:
        torch.set_num_threads(old_threads)


def test_prefix_logits_do_not_read_future_teacher_tokens():
    torch.manual_seed(7)
    model = GeometryRNN(hidden=16, embedding=8, layers=1, context_dim=4)
    tokens, _ = sample_data()
    changed = tokens.clone()
    changed[:, 20:] = PAD
    context = torch.zeros(1, 4)
    first, _ = model(tokens, context)
    other, _ = model(changed, context)
    assert torch.equal(first[:, :20], other[:, :20])


def test_sampling_has_no_source_scaffold_and_keeps_bounded_failures_and_weight_causality():
    model = GeometryRNN(hidden=8, embedding=8, layers=1, context_dim=4)
    context = torch.zeros(1, 4)
    first = model.sample(context, seed=7, max_tokens=12, temperature=0)
    assert first['status'] == 'budget_exhausted'
    assert len(first['tokens']) == 12
    assert first['tokens'] == model.sample(context, seed=999, max_tokens=12, temperature=0)['tokens']
    # Seed is provenance; compare greedy output, not its seed label.


def test_seed_reproducibility_and_changed_weights_change_free_running_values():
    torch.manual_seed(7)
    model = GeometryRNN(hidden=8, embedding=8, layers=1, context_dim=4)
    context = torch.zeros(1, 4)
    before = model.sample(context, seed=7, max_tokens=12)
    assert before == model.sample(context, seed=7, max_tokens=12)
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.fill_(-100)
        model.output.bias[5 + 127] = 100
    after = model.sample(context, seed=7, max_tokens=12, temperature=0)
    assert before['tokens'] != after['tokens']


def test_conditioning_is_explicit_and_source_structure_cannot_be_smuggled_in():
    assert condition_tags({'tags': {'large': 0.5}, 'target_game': 'doom2'}) == {'tag:large': 0.5, 'game:doom2': 1.0}
    with pytest.raises(ValueError):
        condition_tags({'tags': {}, 'target_game': '', 'source_counts': [4, 1, 4, 4, 1]})


def test_format_features_are_required_by_new_model_and_are_causal_in_sampling():
    model = GeometryRNN(hidden=8, embedding=8, layers=1, context_dim=4, prefix_features=True)
    tokens, allowed = sample_data()
    features = torch.tensor([prefix_features(tokens[0].tolist())[:-1]])
    with pytest.raises(ValueError, match='causal token prefix'):
        model.loss(tokens, torch.zeros(1, 4), allowed)
    loss, _ = model.loss(tokens, torch.zeros(1, 4), allowed, features)
    assert torch.isfinite(loss)
    report = model.sample(torch.zeros(1, 4), seed=7, max_tokens=16)
    assert len(report['tokens']) == 16
    assert report['status'] == 'budget_exhausted'


def test_evaluator_crash_retains_raw_model_tokens_as_a_separate_failure(monkeypatch):
    from generation.geometry_model import infer
    from models.conditioning import TagConditioner
    import generation.learned_geometry
    model = GeometryRNN(hidden=8, embedding=8, layers=1, context_dim=4)
    tokens, _ = sample_data()
    raw = {'status': 'complete', 'tokens': tokens[0].tolist()}
    monkeypatch.setattr(model, 'sample', lambda *args, **kwargs: raw)
    def crash(*args, **kwargs):
        raise RuntimeError('retained verifier counterexample')
    monkeypatch.setattr(generation.learned_geometry, 'assess_prediction', crash)
    report = infer(model, TagConditioner(4), {'tags': {}, 'target_game': ''})
    assert report['status'] == 'evaluation_error'
    assert report['sample']['tokens'] == raw['tokens']
    assert report['prediction'] is not None
    assert 'retained verifier counterexample' in report['evaluation_traceback']


def test_boundary_accuracy_cannot_hide_zero_stop_recall_and_weighting_changes_gradients(monkeypatch):
    model = GeometryRNN(hidden=8, embedding=8, layers=1, context_dim=4)
    tokens, allowed = sample_data()
    logits = torch.nn.Parameter(torch.zeros(*allowed.shape))
    with torch.no_grad():
        logits[:, :, RECORD] = 10
    monkeypatch.setattr(model, 'forward', lambda *args, **kwargs: (logits, None))
    context = torch.zeros(1, 4)
    ordinary, metrics = model.loss(tokens, context, allowed)
    assert metrics['ending_accuracy'] > 0.5
    assert metrics['list_stop_recall'] == 0
    assert metrics['list_continue_recall'] == 1
    assert metrics['list_stop_count'] == 5
    ordinary.backward()
    unweighted_gradients = logits.grad.clone()
    logits.grad.zero_()
    weighted, _ = model.loss(tokens, context, allowed, stop_weight=20)
    weighted.backward()
    endings = (tokens[:, 1:] == END_LIST) & allowed[:, :, RECORD]
    assert weighted.detach() > ordinary.detach()
    assert torch.all(logits.grad[:, :, END_LIST][endings].abs()
                     > unweighted_gradients[:, :, END_LIST][endings].abs())
    with pytest.raises(ValueError, match='stop weight'):
        model.loss(tokens, context, allowed, stop_weight=float('nan'))

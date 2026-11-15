import copy

import pytest

torch = pytest.importorskip('torch')

from data.geometry_interleaved import InterleavedCodec, fit_interleaved_vocabulary
from data.geometry_tokens import PAD
from data.learned_geometry import build_target
from models.geometry import GeometryRNN
from tests.geometry_fixtures import two_rooms


def test_ending_auxiliary_is_map_and_class_balanced_without_padding_or_forced_rows(monkeypatch):
    first = build_target(two_rooms())['supervision']
    second = copy.deepcopy(first)
    second['things'] *= 9  # Additional legal continuation headers, no geometric repair.
    codec = InterleavedCodec(fit_interleaved_vocabulary([first, second]))
    sequences = [codec.encode(p) for p in (first, second)]
    length = max(map(len, sequences))
    tokens = torch.full((2, length), PAD, dtype=torch.long)
    allowed = torch.zeros(2, length-1, codec.vocab_size, dtype=torch.bool)
    for batch, sequence in enumerate(sequences):
        tokens[batch, :len(sequence)] = torch.tensor(sequence)
        for index, choices in enumerate(codec.masks(sequence)):
            allowed[batch, index, list(choices)] = True
    model = GeometryRNN(hidden=8, embedding=8, layers=1, context_dim=4, codec_config=codec.definition)
    logits = torch.randn(*allowed.shape, requires_grad=True)
    monkeypatch.setattr(model, 'forward', lambda *args, **kwargs: (logits, None))
    context = torch.zeros(2, 4)
    baseline, before = model.loss(tokens, context, allowed, stop_weight=5)
    actual, metrics = model.loss(tokens, context, allowed, stop_weight=5, ending_loss_weight=.1)
    gold = tokens[:, 1:]
    expected = []
    for batch in range(2):
        eligible = allowed[batch, :, codec.stop_token] & allowed[batch, :, list(codec.continuation_tokens)].any(-1)
        values = logits[batch, eligible].masked_fill(~allowed[batch, eligible], -torch.inf)
        probability = values.softmax(-1)[:, codec.stop_token]
        positive = gold[batch, eligible] == codec.stop_token
        expected.append(((-probability[positive].log()).mean() + (-(1-probability[~positive]).log()).mean()) / 2)
    expected = torch.stack(expected).mean()
    torch.testing.assert_close(actual - baseline, .1*expected)
    torch.testing.assert_close(metrics['ending_auxiliary_loss'], expected.detach())
    assert all(torch.equal(before[name], metrics[name]) for name in before)
    # Only eligible headers receive the *additional* loss gradient. Padding,
    # body values and unavailable continuation types are excluded exactly.
    gradient = torch.autograd.grad(actual-baseline, logits)[0]
    eligible = allowed[:, :, codec.stop_token] & allowed[:, :, list(codec.continuation_tokens)].any(-1)
    assert gradient[~eligible].count_nonzero() == 0
    assert torch.isfinite(gradient).all()
    assert gradient[:, :, codec.stop_token][eligible].abs().sum() > 0
    assert gradient[~allowed].count_nonzero() == 0


@pytest.mark.parametrize('weight', [-1, float('nan'), float('inf'), True])
def test_ending_auxiliary_rejects_invalid_weights_before_model_execution(weight):
    model = GeometryRNN(hidden=8, embedding=8, layers=1, context_dim=4)
    with pytest.raises(ValueError, match='ending loss weight'):
        model.loss(None, None, None, ending_loss_weight=weight)


def test_ending_auxiliary_rejects_legacy_header_contract_before_execution():
    model = GeometryRNN(hidden=8, embedding=8, layers=1, context_dim=4)
    with pytest.raises(ValueError, match='interleaved headers'):
        model.loss(None, None, None, ending_loss_weight=.1)


def test_ending_auxiliary_requires_complete_map_ending():
    prediction = build_target(two_rooms())['supervision']
    codec = InterleavedCodec(fit_interleaved_vocabulary([prediction]))
    tokens = codec.encode(prediction)[:-1]
    model = GeometryRNN(hidden=8, embedding=8, layers=1, context_dim=4, codec_config=codec.definition)
    masks = torch.zeros(1, len(tokens)-1, codec.vocab_size, dtype=torch.bool)
    for index, choices in enumerate(codec.masks(codec.encode(prediction))[:len(tokens)-1]):
        masks[0, index, list(choices)] = True
    with pytest.raises(ValueError, match='complete global ending'):
        model.loss(torch.tensor([tokens]), torch.zeros(1, 4), masks, ending_loss_weight=.1)

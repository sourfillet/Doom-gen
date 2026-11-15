import pytest

torch = pytest.importorskip('torch')

from data.geometry_compact import CompactCodec, fit_vocabulary
from data.geometry_memory import RecordMemory, memory_trace, reference_query
from data.geometry_tokens import BYTE_OFFSET
from data.learned_geometry import build_target
from models.geometry import GeometryRNN
from models.geometry_memory import GeometryRecordReadout, grouped_logsumexp
from tests.geometry_fixtures import two_rooms
from tests.geometry_model_fixtures import one_thread  # noqa: F401 - autouse fixture


def example(*, escapes=False):
    prediction = build_target(two_rooms())['supervision']
    definition = fit_vocabulary([prediction])
    if escapes:
        definition['integers'] = []
    codec = CompactCodec(definition)
    tokens = codec.encode(prediction)
    trace = memory_trace(codec, tokens)
    return codec, tokens, torch.tensor([codec.features(tokens)[:-1]]), torch.tensor([trace['records']]), torch.tensor([trace['references'][:-1]])


def model_for(codec, **kwargs):
    return GeometryRNN(hidden=12, embedding=8, layers=1, context_dim=4,
                       prefix_features=True, codec_config=codec.definition, record_memory=True, **kwargs)


def test_trace_reveals_completed_records_only_and_preserves_numeric_values():
    codec, tokens, features, bank, queries = example(escapes=True)
    state, memory = codec.state(), RecordMemory()
    for index, token in enumerate(tokens[:-1]):
        state.push(token)
        memory.observe(state)
        assert reference_query(state) == queries[0, index].tolist()
        assert sum(features[0, index, 1:]) == len(memory.rows)
        assert memory.rows == bank[0, :len(memory.rows)].tolist()
    # Coordinates and heights are exact authoring values; no source frame.
    assert bank[0, 0, :4].tolist() == [0, 0, 0, 0]
    assert any(row[1] == 1 for row in memory.rows if row[0] == 1)


def test_full_teacher_bank_matches_incremental_generated_prefix_without_future_reads():
    torch.manual_seed(7)
    codec, tokens, features, bank, queries = example(escapes=True)
    model = model_for(codec).eval()
    context = torch.zeros(1, 4)
    state, memory, hidden = codec.state(), RecordMemory(), None
    with torch.no_grad():
        teacher, _ = model(torch.tensor([tokens[:-1]]), context, features=features,
                           memory=bank, references=queries)
        for index, token in enumerate(tokens[:-1]):
            state.push(token)
            memory.observe(state)
            current = torch.tensor([memory.rows], dtype=torch.long).reshape(1, -1, 9)
            actual, hidden = model(torch.tensor([[token]]), context, hidden, features=features[:, index:index+1],
                                   memory=current, references=queries[:, index:index+1])
            torch.testing.assert_close(actual[0, 0], teacher[0, index], atol=2e-6, rtol=2e-6)


def test_future_geometry_cannot_change_prefix_logits_or_receive_gradients():
    torch.manual_seed(7)
    codec, tokens, features, bank, queries = example()
    model = model_for(codec).eval()
    # Stop after exactly the first emitted vertex, well before the last vertex.
    length = int((features[0, :, 1] < 2).sum())
    past = bank.float().requires_grad_()
    before, _ = model(torch.tensor([tokens[:length]]), torch.zeros(1, 4), features=features[:, :length],
                      memory=past, references=queries[:, :length])
    changed = bank.clone()
    future = (changed[0, :, 0] != 0) | (changed[0, :, 1] >= 2)
    changed[0, future, 2:] += 1234
    after, _ = model(torch.tensor([tokens[:length]]), torch.zeros(1, 4), features=features[:, :length],
                     memory=changed, references=queries[:, :length])
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    before.sum().backward()
    assert torch.isfinite(past.grad).all()
    assert torch.count_nonzero(past.grad[0, future]) == 0
    # Actual emitted coordinates do influence the learned readout.
    changed = bank.clone()
    changed[0, 0, 2] += 1000
    other, _ = model(torch.tensor([tokens[:length]]), torch.zeros(1, 4), features=features[:, :length],
                     memory=changed, references=queries[:, :length])
    assert not torch.equal(before, other)


def test_reference_head_projects_escaped_bytes_and_learned_null_without_geometry_rules():
    codec = CompactCodec({'schema': 'doom_geometry_field_tokens_v1', 'assets': [], 'integers': [0, 65535]})
    head = GeometryRecordReadout(8, codec)
    bank = torch.tensor([[[0, index, index, 0, 0, 0, 0, 0, 0] for index in range(300)]])
    features = torch.tensor([[[0, 300, 0, 0, 0, 0]]])
    hidden, logits = torch.zeros(1, 1, 8), torch.zeros(1, 1, codec.vocab_size)
    # Low byte 1 can complete ID 1 or 257; both are valid learned candidates.
    values = head(hidden, logits, features, bank, torch.tensor([[[0, 1, 1, 0]]]))
    assert set(torch.isfinite(values[0, 0]).nonzero().flatten().tolist()) == {BYTE_OFFSET, BYTE_OFFSET+1}
    # Optional side with no emitted sides has exactly the learned null ID.
    values = head(hidden, logits, features, bank, torch.tensor([[[2, 0, -1, 1]]]))
    assert torch.isfinite(values[0, 0]).nonzero().flatten().tolist() == [codec.integer_ids[65535]]
    with pytest.raises(ValueError, match='no visible emitted record'):
        head(hidden, logits, features, bank, torch.tensor([[[2, 0, -1, 0]]]))


def test_masked_scatter_groups_have_finite_gradients_even_with_very_negative_logits():
    scores = torch.tensor([[[-10000., -torch.inf, -10001., -torch.inf]]], requires_grad=True)
    result = grouped_logsumexp(scores, torch.tensor([[[0, 0, 0, 1]]]), 3)
    torch.testing.assert_close(result[0, 0, 0], torch.logsumexp(scores[0, 0, [0, 2]], 0))
    assert torch.isneginf(result[0, 0, 1:]).all()
    result[0, 0, 0].backward()
    assert torch.isfinite(scores.grad).all()
    assert scores.grad[0, 0, 1] == scores.grad[0, 0, 3] == 0


def test_memory_loss_backpropagates_and_sampling_needs_no_teacher_bank():
    codec, tokens, features, bank, queries = example(escapes=True)
    model = model_for(codec)
    allowed = torch.zeros(1, len(tokens)-1, codec.vocab_size, dtype=torch.bool)
    for index, choices in enumerate(codec.masks(tokens)):
        allowed[0, index, list(choices)] = True
    with pytest.raises(ValueError, match='emitted-record bank'):
        model(torch.tensor([tokens[:-1]]), torch.zeros(1, 4), features=features)
    loss, _ = model.loss(torch.tensor([tokens]), torch.zeros(1, 4), allowed, features,
                         memory=bank, references=queries)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.memory_readout.pointer_key.weight.grad.abs().sum() > 0
    first = model.sample(torch.zeros(1, 4), seed=7, max_tokens=80)
    assert first == model.sample(torch.zeros(1, 4), seed=7, max_tokens=80)
    assert first['status'] in {'complete', 'budget_exhausted'}


def test_cached_record_embeddings_and_keys_match_uncached_readout():
    codec, tokens, features, bank, queries = example()
    model = model_for(codec).eval()
    with torch.no_grad():
        first, _ = model(torch.tensor([tokens[:-1]]), torch.zeros(1, 4), features=features,
                         memory=bank, references=queries)
        encoded = model.memory_readout.prepare_records(bank)
        other, _ = model(torch.tensor([tokens[:-1]]), torch.zeros(1, 4), features=features,
                         memory=bank, references=queries, encoded_memory=encoded)
    torch.testing.assert_close(first, other, atol=0, rtol=0)


def test_checkpoint_rejects_unknown_memory_contract_and_byte_memory_is_not_silently_enabled(tmp_path):
    from generation.geometry_model import load_model
    codec, *_ = example()
    with pytest.raises(ValueError, match='compact field codec'):
        GeometryRNN(prefix_features=True, record_memory=True)
    config = {'kind': 'geometry', 'token_schema': codec.definition['schema'],
              'target_schema': 'doom_learned_map_v1', 'vocab_size': codec.vocab_size,
              'codec_config': codec.definition, 'memory_schema': 'unknown-future-contract'}
    path = tmp_path / 'unsupported.pt'
    torch.save({'config': config}, path)
    with pytest.raises(ValueError, match='record-memory contract'):
        load_model(path)


def test_read_pointer_starts_at_v1_function_and_learns_a_new_memory_query_route():
    codec, tokens, features, bank, queries = example()
    torch.manual_seed(7)
    v1 = model_for(codec)
    torch.manual_seed(7)
    v2 = model_for(codec, memory_read=True)
    assert all(torch.equal(value, v2.state_dict()[key]) for key, value in v1.state_dict().items())
    context = torch.zeros(1, 4)
    first, _ = v1(torch.tensor([tokens[:-1]]), context, features=features, memory=bank, references=queries)
    other, _ = v2(torch.tensor([tokens[:-1]]), context, features=features, memory=bank, references=queries)
    torch.testing.assert_close(first, other, atol=0, rtol=0)
    allowed = torch.zeros(1, len(tokens)-1, codec.vocab_size, dtype=torch.bool)
    for index, choices in enumerate(codec.masks(tokens)):
        allowed[0, index, list(choices)] = True
    loss, _ = v2.loss(torch.tensor([tokens]), context, allowed, features, memory=bank, references=queries)
    loss.backward()
    assert v2.memory_readout.pointer_read.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in v2.parameters() if p.grad is not None)


def test_read_pointer_uses_visible_geometry_for_queries_but_not_future_records():
    codec, tokens, features, bank, queries = example()
    torch.manual_seed(7)
    v1 = model_for(codec).eval()
    torch.manual_seed(7)
    v2 = model_for(codec, memory_read=True).eval()
    with torch.no_grad():
        v2.memory_readout.pointer_read.weight.copy_(torch.eye(12))
    # At the first sidedef's sector-reference field, vertices/sectors are past
    # and every sidedef/line/thing is still future. No selected source IDs added.
    index = int(((queries[0, :, 0] == 1) & (queries[0, :, 1] == 0)).nonzero()[0])
    length = index + 1
    context = torch.zeros(1, 4)
    def run(model, records):
        return model(torch.tensor([tokens[:length]]), context, features=features[:, :length],
                     memory=records, references=queries[:, :length])[0][0, -1]
    memory = bank.float().requires_grad_()
    before = run(v2, memory)
    future = bank[0, :, 0] >= 2
    changed = bank.clone()
    changed[0, future, 2:] += 1234
    torch.testing.assert_close(before, run(v2, changed), atol=0, rtol=0)
    before.masked_fill(~torch.isfinite(before), 0).sum().backward()
    assert torch.count_nonzero(memory.grad[0, future]) == 0
    changed = bank.clone()
    changed[0, 0, 2] += 1000
    assert not torch.equal(before, run(v2, changed))
    # The v1 reference query cannot read that vertex: its candidate sector
    # keys and recurrent teacher history are unchanged by this intervention.
    torch.testing.assert_close(run(v1, bank), run(v1, changed), atol=0, rtol=0)

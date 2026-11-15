import pytest

torch = pytest.importorskip('torch')

from data.geometry_memory import RecordMemory, current_record_anchors
from models.geometry import GeometryRNN
from tests.geometry_model_fixtures import anchored_example, one_thread  # noqa: F401 - autouse fixture


def model_for(codec, anchored, relative_pointers=False, spatial_pointers=False):
    return GeometryRNN(hidden=12, embedding=8, layers=1, context_dim=4, prefix_features=True,
                       codec_config=codec.definition, record_memory=True, memory_read=True,
                       record_anchors=anchored, relative_pointers=relative_pointers, spatial_pointers=spatial_pointers)


def test_partial_record_exposes_only_already_chosen_references():
    codec, tokens, features, bank, queries, anchors = anchored_example()
    state = codec.state()
    for index, token in enumerate(tokens[:-1]):
        state.push(token)
        assert current_record_anchors(state) == anchors[0, index].tolist()
        for kind, record_id in anchors[0, index].tolist():
            if kind >= 0:
                assert record_id < features[0, index, kind + 1]
        if state.section == 3 and state.record is not None and state.field_index == 1:
            assert anchors[0, index, 0].tolist() == [0, state.record['v1']]
            assert anchors[0, index, 1].tolist() == [-1, -1]


def test_anchor_path_starts_at_v2_function_and_receives_learning_gradients():
    codec, tokens, features, bank, queries, anchors = anchored_example()
    torch.manual_seed(7)
    baseline = model_for(codec, False)
    torch.manual_seed(7)
    candidate = model_for(codec, True)
    assert all(torch.equal(v, candidate.state_dict()[k]) for k, v in baseline.state_dict().items())
    context = torch.zeros(1, 4)
    before, _ = baseline(torch.tensor([tokens[:-1]]), context, features=features, memory=bank, references=queries)
    after, _ = candidate(torch.tensor([tokens[:-1]]), context, features=features, memory=bank,
                         references=queries, anchors=anchors)
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    allowed = torch.zeros(1, len(tokens)-1, codec.vocab_size, dtype=torch.bool)
    for index, choices in enumerate(codec.masks(tokens)):
        allowed[0, index, list(choices)] = True
    loss, _ = candidate.loss(torch.tensor([tokens]), context, allowed, features,
                              memory=bank, references=queries, anchors=anchors)
    loss.backward()
    assert candidate.memory_readout.anchor_read.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in candidate.parameters() if p.grad is not None)
    with pytest.raises(ValueError, match='partial-record'):
        candidate(torch.tensor([tokens[:-1]]), context, features=features, memory=bank, references=queries)


@pytest.mark.parametrize('relative,relative_pointers,spatial_pointers', [(False, False, False), (True, False, False), ('interleaved', False, False), ('interleaved', True, False), ('interleaved', True, True)])
def test_active_anchor_full_bank_matches_incremental_prefix_with_world_geometry(relative, relative_pointers, spatial_pointers):
    codec, tokens, features, bank, queries, anchors = anchored_example(relative)
    model = model_for(codec, True, relative_pointers, spatial_pointers).eval()
    with torch.no_grad():
        model.memory_readout.anchor_read.weight[:, :12] = torch.eye(12)
        if relative_pointers:
            model.memory_readout.relative_pointer.normal_()
        if spatial_pointers:
            model.memory_readout.spatial_pointer.normal_()
        teacher, _ = model(torch.tensor([tokens[:-1]]), torch.zeros(1, 4), features=features,
                            memory=bank, references=queries, anchors=anchors)
        state, builder, hidden = codec.state(), RecordMemory(), None
        for index, token in enumerate(tokens[:-1]):
            state.push(token)
            builder.observe(state)
            actual, hidden = model(torch.tensor([[token]]), torch.zeros(1, 4), hidden,
                                   features=features[:, index:index+1],
                                   memory=torch.tensor([builder.rows]).reshape(1, -1, 9),
                                   references=queries[:, index:index+1], anchors=anchors[:, index:index+1])
            torch.testing.assert_close(actual[0, 0], teacher[0, index], atol=3e-6, rtol=3e-6)


def test_known_geometry_anchor_changes_query_but_future_anchor_is_rejected():
    codec, tokens, features, bank, queries, anchors = anchored_example()
    model = model_for(codec, True).eval()
    with torch.no_grad():
        model.memory_readout.pointer_read.weight.zero_()
        model.memory_readout.anchor_read.weight[:, :12] = torch.eye(12)
    # Left-side selection has known endpoint slots but only candidate side keys.
    index = int(((queries[0, :, 0] == 2) & (queries[0, :, 3] == 1)).nonzero()[0])
    length = index + 1
    selected_vertex = int(anchors[0, index, 0, 1])
    def run(records, anchor_values):
        return model(torch.tensor([tokens[:length]]), torch.zeros(1, 4), features=features[:, :length],
                     memory=records, references=queries[:, :length], anchors=anchor_values[:, :length])[0]
    memory = bank.float().requires_grad_()
    before = run(memory, anchors)
    future = bank[0, :, 0] >= 3
    changed = bank.clone()
    changed[0, future, 2:] += 1234
    torch.testing.assert_close(before, run(changed, anchors), atol=0, rtol=0)
    before[0, -1].masked_fill(~torch.isfinite(before[0, -1]), 0).sum().backward()
    assert torch.count_nonzero(memory.grad[0, future]) == 0
    changed = bank.clone()
    changed[0, selected_vertex, 2] += 1000
    assert not torch.equal(before[0, -1], run(changed, anchors)[0, -1])
    bad = anchors.clone()
    bad[0, 0, 0] = torch.tensor([0, 0])  # Source vertex exists in bank but is future at BOS.
    with pytest.raises(ValueError, match='not yet emitted'):
        run(bank, bad)


@pytest.mark.parametrize('relative_pointers,spatial_pointers', [(False, False), (True, False), (True, True)])
def test_interleaved_future_records_of_every_type_have_no_effect_or_gradient(relative_pointers, spatial_pointers):
    codec, tokens, features, bank, queries, anchors = anchored_example('interleaved')
    model = model_for(codec, True, relative_pointers, spatial_pointers).eval()
    with torch.no_grad():
        model.memory_readout.anchor_read.weight.normal_()
        if relative_pointers:
            model.memory_readout.relative_pointer.normal_()
        if spatial_pointers:
            model.memory_readout.spatial_pointer.normal_()
    selection = ((queries[0, :, 0] == 0) & (anchors[0, :, 0, 0] == 0) if spatial_pointers else
                 (queries[0, :, 0] == 2) & (queries[0, :, 3] == 1))
    index = int(selection.nonzero()[0])
    length = index + 1
    future = bank[0, :, 1] >= features[0, index, 1:][bank[0, :, 0]]
    assert future.any() and (bank[0, future, 0] == 0).any()
    def run(records):
        return model(torch.tensor([tokens[:length]]), torch.zeros(1, 4), features=features[:, :length],
                     memory=records, references=queries[:, :length], anchors=anchors[:, :length])[0]
    memory = bank.float().requires_grad_()
    before = run(memory)
    changed = bank.clone(); changed[0, future, 2:] += 1234
    torch.testing.assert_close(before, run(changed), atol=0, rtol=0)
    before[0, -1].masked_fill(~torch.isfinite(before[0, -1]), 0).sum().backward()
    assert torch.count_nonzero(memory.grad[0, future]) == 0


def test_relative_pointer_route_starts_at_exact_v4_function_preserves_rng_and_learns():
    codec, tokens, features, bank, queries, anchors = anchored_example('interleaved')
    torch.manual_seed(7); before = model_for(codec, True)
    state_before = torch.get_rng_state().clone()
    torch.manual_seed(7); after = model_for(codec, True, True)
    assert torch.equal(state_before, torch.get_rng_state())
    assert all(torch.equal(v, after.state_dict()[k]) for k, v in before.state_dict().items())
    kwargs = dict(features=features, memory=bank, references=queries, anchors=anchors)
    a = before(torch.tensor([tokens[:-1]]), torch.zeros(1, 4), **kwargs)[0]
    b = after(torch.tensor([tokens[:-1]]), torch.zeros(1, 4), **kwargs)[0]
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    masks = torch.zeros(1, len(tokens)-1, codec.vocab_size, dtype=torch.bool)
    for i, choices in enumerate(codec.masks(tokens)):
        masks[0, i, list(choices)] = True
    loss, _ = after.loss(torch.tensor([tokens]), torch.zeros(1, 4), masks, **kwargs)
    loss.backward()
    assert after.memory_readout.relative_pointer.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in after.parameters() if p.grad is not None)


def test_spatial_pointer_starts_at_v5_function_and_learns_without_rng_shift():
    codec, tokens, features, bank, queries, anchors = anchored_example('interleaved')
    torch.manual_seed(7); before = model_for(codec, True, True)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(7); after = model_for(codec, True, True, True)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, after.state_dict()[k]) for k, v in before.state_dict().items())
    kwargs = dict(features=features, memory=bank, references=queries, anchors=anchors)
    a = before(torch.tensor([tokens[:-1]]), torch.zeros(1, 4), **kwargs)[0]
    b = after(torch.tensor([tokens[:-1]]), torch.zeros(1, 4), **kwargs)[0]
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    masks = torch.zeros(1, len(tokens)-1, codec.vocab_size, dtype=torch.bool)
    for i, choices in enumerate(codec.masks(tokens)):masks[0, i, list(choices)] = True
    loss, _ = after.loss(torch.tensor([tokens]), torch.zeros(1, 4), masks, **kwargs)
    loss.backward()
    assert after.memory_readout.spatial_pointer.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in after.parameters() if p.grad is not None)


def test_spatial_basis_uses_relative_coordinates_and_keeps_self_reference_legal():
    codec, tokens, features, bank, queries, anchors = anchored_example('interleaved')
    model = model_for(codec, True, True, True).eval()
    with torch.no_grad():model.memory_readout.spatial_pointer.normal_()
    query = torch.randn(1, len(tokens)-1, 12)
    scores = model.memory_readout.spatial_scores(query, bank, features, anchors, queries)
    moved = bank.clone();moved[0, bank[0, :, 0] == 0, 2:4] += torch.tensor([256, -128])
    torch.testing.assert_close(scores, model.memory_readout.spatial_scores(query, moved, features, anchors, queries), atol=0, rtol=0)
    changed = bank.clone();changed[0, 0, 2] += 31
    assert not torch.equal(scores, model.memory_readout.spatial_scores(query, changed, features, anchors, queries))
    logits = model(torch.tensor([tokens[:-1]]), torch.zeros(1, 4), features=features,
                   memory=bank, references=queries, anchors=anchors)[0]
    index = int(((queries[0, :, 0] == 0) & (anchors[0, :, 0, 0] == 0)).nonzero()[0])
    assert torch.isfinite(logits[0, index, codec.integer_ids[0]])

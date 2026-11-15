import pytest

from data.geometry_compact import CompactCodec, fit_vocabulary
from data.geometry_objective import loss_groups
from data.geometry_relative import RelativeCodec, fit_relative_vocabulary
from data.geometry_tokens import END_LIST, RECORD
from data.learned_geometry import AUTHORING_FIELDS, RECORD_FIELDS, build_target
from tests.geometry_fixtures import two_rooms


@pytest.mark.parametrize('codec_type,fit', [(CompactCodec, fit_vocabulary), (RelativeCodec, fit_relative_vocabulary)])
def test_field_groups_cover_actual_fields_and_keep_class_labels_out_of_prefix_inputs(codec_type, fit):
    prediction = build_target(two_rooms())['supervision']
    codec = codec_type(fit([prediction])); tokens = codec.encode(prediction)
    groups = loss_groups(codec, tokens)
    assert len(groups) == len(tokens)-1
    for section, name in enumerate(AUTHORING_FIELDS):
        fields = len(RECORD_FIELDS[name])
        observed = {g for s, g in groups if s == section}
        expected = set(range(fields)) | {fields, fields+1} if prediction[name] else {fields+1}
        assert observed == expected
        assert sum(g == [section, fields+1] for g in groups) == 1
    for i, (section, field) in enumerate(groups):
        if section >= 0 and field >= len(RECORD_FIELDS[AUTHORING_FIELDS[section]]):
            assert tokens[i+1] == (END_LIST if field == len(RECORD_FIELDS[AUTHORING_FIELDS[section]])+1 else RECORD)
    assert groups[-1] == [-1, -1]
    # Supervised boundary labels never alter causal parser features or tokens.
    assert codec.decode(tokens) == prediction


def test_balanced_objective_is_invariant_to_common_token_duplication_and_padding():
    torch = pytest.importorskip('torch')
    from models.geometry_objective import balanced_loss
    # Same field, another field, continuation and rare true stop.
    values = torch.tensor([[2., 4., 1., 8.]], requires_grad=True)
    groups = torch.tensor([[[0, 0], [0, 1], [0, 2], [0, 3]]])
    active = torch.ones_like(values, dtype=torch.bool)
    loss = balanced_loss(values, active, groups)
    torch.testing.assert_close(loss, torch.tensor(3.75))
    repeated = torch.cat([values[:, :3], values[:, 2:3].expand(-1, 100), values[:, 3:], values.new_tensor([[999.]])], 1)
    repeated_groups = torch.cat([groups[:, :3], groups[:, 2:3].expand(-1, 100, -1), groups[:, 3:], groups.new_tensor([[[-1, -1]]])], 1)
    repeated_active = torch.ones_like(repeated, dtype=torch.bool); repeated_active[:, -1] = False
    torch.testing.assert_close(balanced_loss(repeated, repeated_active, repeated_groups), loss)
    loss.backward()
    torch.testing.assert_close(values.grad, torch.full_like(values, .25))
    # Unequal group counts in the next section still give sections equal mass.
    two_section = torch.cat([values.detach(), values.new_tensor([[20.]])], 1)
    two_groups = torch.cat([groups, groups.new_tensor([[[1, 0]]])], 1)
    torch.testing.assert_close(balanced_loss(two_section, torch.ones_like(two_section, dtype=torch.bool), two_groups), torch.tensor((3.75+20)/2))
    bad = groups.clone(); bad[:, 0] = -1
    with pytest.raises(ValueError, match='valid loss group'):
        balanced_loss(values, active, bad)


def test_balanced_full_model_loss_has_finite_gradients_and_rejects_double_stop_weight():
    torch = pytest.importorskip('torch')
    from models.geometry import GeometryRNN
    prediction = build_target(two_rooms())['supervision']
    codec = RelativeCodec(fit_relative_vocabulary([prediction])); tokens = codec.encode(prediction)
    model = GeometryRNN(hidden=8, embedding=8, layers=1, context_dim=4, prefix_features=True,
                        codec_config=codec.definition)
    masks = torch.zeros(1, len(tokens)-1, codec.vocab_size, dtype=torch.bool)
    for index, allowed in enumerate(codec.masks(tokens)):
        masks[0, index, list(allowed)] = True
    kwargs = dict(features=torch.tensor([codec.features(tokens)[:-1]]), loss_groups=torch.tensor([loss_groups(codec, tokens)]))
    loss, metrics = model.loss(torch.tensor([tokens]), torch.zeros(1, 4), masks, **kwargs)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert metrics['list_stop_count'] == 5
    with pytest.raises(ValueError, match='already balanced'):
        model.loss(torch.tensor([tokens]), torch.zeros(1, 4), masks, stop_weight=5, **kwargs)

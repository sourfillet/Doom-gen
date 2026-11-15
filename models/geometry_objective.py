"""Decision-token NLL averaged by field/class, then section, then map."""
import torch


def balanced_loss(losses, decisions, groups):
    if groups.shape != (*losses.shape, 2) or groups.dtype != torch.long:
        raise ValueError('loss groups must be integer section/field pairs matching the targets')
    sections, fields = groups.unbind(-1)
    if torch.any(decisions & ((sections < 0) | (sections >= 5) | (fields < 0) | (fields >= 16))):
        raise ValueError('every learned decision requires a valid loss group')
    # Fixed-width bookkeeping, not geometry. Padded and forced tokens contribute
    # neither a loss nor a denominator; absent sections/groups are excluded.
    indices = (sections.clamp(0, 4) * 16 + fields.clamp(0, 15))
    totals = losses.new_zeros((len(losses), 80)).scatter_add(1, indices, losses * decisions)
    counts = losses.new_zeros((len(losses), 80)).scatter_add(1, indices, decisions.to(losses.dtype))
    present = (counts > 0).reshape(-1, 5, 16)
    field_means = (totals / counts.clamp_min(1)).reshape(-1, 5, 16)
    section_means = (field_means * present).sum(-1) / present.sum(-1).clamp_min(1)
    sections_present = present.any(-1)
    return ((section_means * sections_present).sum(-1) / sections_present.sum(-1).clamp_min(1)).mean()

"""Which room-sequence terms train the stop hazard (Torch-free, shared by both trainers)."""
from __future__ import annotations

STOP_TARGETS = ('all', 'complete', 'reachable-ends')


def stop_labels(complete: bool, length: int, limit: int, stop_targets: str = 'all') -> tuple[bool, bool]:
    """Return (terminal, stop) for RoomGraphRNN.loss.

    all: complete maps end; fragments only continue (censored endings).
    complete: only complete maps carry stop terms; fragments train structure.
    reachable-ends: every map ends after its last reachable room. Fragments are
    the reachable part of maps without a certified exit, so this learns the
    size of reachable regions rather than of certified-complete maps.
    A prefix cut by the room limit is never an ending.
    """
    if stop_targets not in STOP_TARGETS:
        raise ValueError(f'stop_targets must be one of {STOP_TARGETS}')
    whole = not limit or length <= limit
    if stop_targets == 'complete':
        return complete and whole, complete
    if stop_targets == 'reachable-ends':
        return whole, True
    return complete and whole, True

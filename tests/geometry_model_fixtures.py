"""Encoded two-room examples for geometry record-memory tests; skips without torch."""
import pytest

torch = pytest.importorskip('torch')

from data.geometry_compact import CompactCodec, fit_vocabulary
from data.geometry_interleaved import (InterleavedCodec, RelativeReferenceCodec,
                                       fit_interleaved_vocabulary, fit_relative_reference_vocabulary)
from data.geometry_memory import ANCHOR_SCHEMA, RELATIVE_REFERENCE_MEMORY_SCHEMA, memory_trace
from data.geometry_relative import RelativeCodec, fit_relative_vocabulary
from data.learned_geometry import build_target
from tests.geometry_fixtures import two_rooms


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def anchored_example(relative=False):
    prediction = build_target(two_rooms())['supervision']
    codec = (RelativeReferenceCodec(fit_relative_reference_vocabulary([prediction])) if relative == 'relative-reference' else
             InterleavedCodec(fit_interleaved_vocabulary([prediction])) if relative == 'interleaved' else
             RelativeCodec(fit_relative_vocabulary([prediction])) if relative else
             CompactCodec(fit_vocabulary([prediction])))
    tokens = codec.encode(prediction)
    trace = memory_trace(codec, tokens, anchors=True)
    assert trace['schema'] == (RELATIVE_REFERENCE_MEMORY_SCHEMA if relative == 'relative-reference' else
                               'doom_geometry_interleaved_memory_v4' if relative == 'interleaved' else ANCHOR_SCHEMA)
    return (codec, tokens, torch.tensor([codec.features(tokens)[:-1]]), torch.tensor([trace['records']]),
            torch.tensor([trace['references'][:-1]]), torch.tensor([trace['anchors'][:-1]]))

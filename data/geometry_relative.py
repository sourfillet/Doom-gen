"""Lossless causal scalar coding, not a geometry construction algorithm.

The first vertex is absolute. Later vertex coordinates encode differences from
the preceding completed vertex; sector ceilings encode differences from the
already emitted floor. Differences use reversible 16-bit modular arithmetic,
including extreme ranges and zero/negative clearance. No source frame, rounding,
minimum height, generated offset or shape rule is supplied to inference.
"""
from __future__ import annotations

import copy

from data.geometry_compact import CompactCodec, CompactState, SCHEMA as FIELD_SCHEMA, fit_vocabulary
from data.geometry_tokens import _record_bytes
from data.learned_geometry import AUTHORING_FIELDS, RECORD_FIELDS, validate_prediction

SCHEMA = 'doom_geometry_relative_field_tokens_v1'
PREFIX_SCHEMA = 'doom_geometry_relative_field_prefix_v1'


def signed16(value):
    value &= 65535
    return value - 65536 if value >= 32768 else value


def relative_fields(prediction):
    validate_prediction(prediction)
    for name in AUTHORING_FIELDS:
        for record in prediction[name]:
            _record_bytes(name, record)  # Reject original overflow; do not wrap invalid authoring.
    result = copy.deepcopy(prediction)
    previous = {'x': 0, 'y': 0}
    for original, encoded in zip(prediction['vertices'], result['vertices']):
        for field in ('x', 'y'):
            encoded[field] = signed16(original[field] - previous[field])
        previous = original
    for original, encoded in zip(prediction['sectors'], result['sectors']):
        encoded['ceil_height'] = signed16(original['ceil_height'] - original['floor_height'])
    return result


def fit_relative_vocabulary(predictions, *, max_integers=128):
    definition = fit_vocabulary((relative_fields(p) for p in predictions), max_integers=max_integers)
    definition['schema'] = SCHEMA
    return definition


class RelativeCodec(CompactCodec):
    prefix_schema = PREFIX_SCHEMA

    def __init__(self, definition):
        if not isinstance(definition, dict) or definition.get('schema') != SCHEMA:
            raise ValueError('invalid relative geometry vocabulary')
        super().__init__({**definition, 'schema': FIELD_SCHEMA})
        self.definition['schema'] = SCHEMA

    def state(self):
        return RelativeState(self)

    def encode(self, prediction):
        return super().encode(relative_fields(prediction))


class RelativeState(CompactState):
    def _finish(self, value):
        name = AUTHORING_FIELDS[self.section]
        field = RECORD_FIELDS[name][self.field_index]
        if name == 'vertices' and self.prediction[name]:
            value = (value + self.prediction[name][-1][field]) & 65535
        elif name == 'sectors' and field == 'ceil_height':
            value = (value + self.record['floor_height']) & 65535
        super()._finish(value)

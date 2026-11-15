"""Dependency-interleaved authoring, with model-selected record types at inference.

Training visits original linedefs in their original order and emits referenced
records on first use. Unreferenced records are retained at the end. Table IDs
are reindexed; an audit-only permutation restores original authoring byte order.
The decoder has no source permutation, outline, traversal or assembly algorithm.
"""
from __future__ import annotations

import copy

from data import geometry_tokens as base
from data.geometry_compact import CompactCodec, LITERAL_NAME, SCHEMA as FIELD_SCHEMA, fit_vocabulary
from data.geometry_relative import RelativeState, relative_fields
from data.learned_geometry import AUTHORING_FIELDS, PREDICTION_SCHEMA, RECORD_FIELDS

SCHEMA = 'doom_geometry_interleaved_tokens_v1'
PREFIX_SCHEMA = 'doom_geometry_interleaved_prefix_v1'
RELATIVE_REFERENCE_SCHEMA = 'doom_geometry_interleaved_relative_reference_tokens_v2'
RELATIVE_REFERENCE_PREFIX_SCHEMA = 'doom_geometry_interleaved_relative_reference_prefix_v1'
ORDER_SCHEMA = 'doom_geometry_dependency_first_use_v1'


def first_use_order(prediction):
    """Return reindexed records, event schedule and reversible source-ID audit."""
    base.encode_prediction(prediction)  # Full ranges/references, never geometry repair.
    result = {'schema': PREDICTION_SCHEMA, **{name: [] for name in AUTHORING_FIELDS}}
    source_ids = {name: [] for name in AUTHORING_FIELDS}
    remaps = {name: {} for name in AUTHORING_FIELDS}
    events = []

    def emit(name, original_id):
        if original_id in remaps[name]:
            return remaps[name][original_id]
        record = copy.deepcopy(prediction[name][original_id])
        for field, kind in zip(RECORD_FIELDS[name], base.FIELD_TYPES[name]):
            if kind in base._REFERENCE_LIST and record[field] >= 0:
                record[field] = emit(base._REFERENCE_LIST[kind], record[field])
        index = len(result[name])
        result[name].append(record)
        source_ids[name].append(original_id)
        remaps[name][original_id] = index
        events.append((AUTHORING_FIELDS.index(name), index))
        return index

    for index in range(len(prediction['linedefs'])):
        emit('linedefs', index)
    for name in AUTHORING_FIELDS:
        for index in range(len(prediction[name])):
            emit(name, index)
    return result, events, source_ids


def restore_authoring(prediction, source_ids):
    """Audit only: invert a complete table permutation; never an inference input."""
    base.encode_prediction(prediction)
    if not isinstance(source_ids, dict) or set(source_ids) != set(AUTHORING_FIELDS):
        raise ValueError('source-ID audit must contain all authoring lists')
    for name in AUTHORING_FIELDS:
        ids = source_ids[name]
        if (not isinstance(ids, list) or any(isinstance(i, bool) or not isinstance(i, int) for i in ids)
                or sorted(ids) != list(range(len(prediction[name])))):
            raise ValueError('source-ID audit must be a complete permutation')
    result = {'schema': PREDICTION_SCHEMA, **{name: [None] * len(prediction[name]) for name in AUTHORING_FIELDS}}
    for name in AUTHORING_FIELDS:
        for index, original_id in enumerate(source_ids[name]):
            record = copy.deepcopy(prediction[name][index])
            for field, kind in zip(RECORD_FIELDS[name], base.FIELD_TYPES[name]):
                if kind in base._REFERENCE_LIST and record[field] >= 0:
                    record[field] = source_ids[base._REFERENCE_LIST[kind]][record[field]]
            result[name][original_id] = record
    base.encode_prediction(result)
    return result


def relative_reference_fields(prediction):
    """Encode causal reference ages (count - 1 - ID) at each emission event.

    Source first-use reindexing and audit permutations keep actual table IDs.
    Non-null references encode their age relative to the causal count emitted
    before that record; optional-side -1 remains NULL (65535).
    """
    world, events, _ = first_use_order(prediction)
    values = relative_fields(world)
    counts = {name: 0 for name in AUTHORING_FIELDS}
    for section, index in events:
        name = AUTHORING_FIELDS[section]
        record = values[name][index]
        for field, kind in zip(RECORD_FIELDS[name], base.FIELD_TYPES[name]):
            if kind in base._REFERENCE_LIST and record[field] >= 0:
                target = base._REFERENCE_LIST[kind]
                count = counts[target]
                if not 0 <= record[field] < count:
                    raise ValueError(f'{name}[{index}].{field} references {target}[{record[field]}] at count {count}')
                record[field] = count - 1 - record[field]
        counts[name] += 1
    return world, events, values


def fit_interleaved_vocabulary(predictions, *, max_integers=128):
    definition = fit_vocabulary((relative_fields(first_use_order(p)[0]) for p in predictions),
                                max_integers=max_integers)
    definition['schema'] = SCHEMA
    return definition


def fit_relative_reference_vocabulary(predictions, *, max_integers=128):
    definition = fit_vocabulary((relative_reference_fields(p)[2] for p in predictions),
                                max_integers=max_integers)
    definition['schema'] = RELATIVE_REFERENCE_SCHEMA
    return definition


class InterleavedCodec(CompactCodec):
    prefix_schema = PREFIX_SCHEMA
    unordered_memory = True
    stop_token = base.EOS

    def __init__(self, definition):
        if not isinstance(definition, dict) or definition.get('schema') != SCHEMA:
            raise ValueError('invalid interleaved geometry vocabulary')
        super().__init__({**definition, 'schema': FIELD_SCHEMA})
        self.definition['schema'] = SCHEMA
        self.continuation_tokens = tuple(range(self.vocab_size, self.vocab_size + 5))
        self.record_types = dict(zip(self.continuation_tokens, range(5)))
        self.vocab_size += 5

    def state(self):
        return InterleavedState(self)

    def _encode_events(self, events, values):
        tokens = [base.BOS]
        for section, index in events:
            name = AUTHORING_FIELDS[section]
            tokens.append(self.continuation_tokens[section])
            for field, kind in zip(RECORD_FIELDS[name], base.FIELD_TYPES[name]):
                value = values[name][index][field]
                if kind == 'name':
                    if value in self.asset_ids:
                        tokens.append(self.asset_ids[value])
                    else:
                        tokens.append(LITERAL_NAME)
                        tokens.extend(base.BYTE_OFFSET + b for b in value.encode('ascii').ljust(8, b'\0'))
                else:
                    raw = value & 65535
                    if raw in self.integer_ids:
                        tokens.append(self.integer_ids[raw])
                    else:
                        tokens.extend((base.BYTE_OFFSET + (raw & 255), base.BYTE_OFFSET + (raw >> 8)))
        tokens.append(base.EOS)
        return tokens

    def encode(self, prediction):
        world, events, _ = first_use_order(prediction)
        tokens = self._encode_events(events, relative_fields(world))
        if self.decode(tokens) != world:
            raise ValueError('interleaved encoding changes reindexed authoring')
        return tokens


class RelativeReferenceCodec(InterleavedCodec):
    prefix_schema = RELATIVE_REFERENCE_PREFIX_SCHEMA
    relative_references = True

    def __init__(self, definition):
        if not isinstance(definition, dict) or definition.get('schema') != RELATIVE_REFERENCE_SCHEMA:
            raise ValueError('invalid relative-reference geometry vocabulary')
        super().__init__({**definition, 'schema': SCHEMA})
        self.definition['schema'] = RELATIVE_REFERENCE_SCHEMA

    def state(self):
        return RelativeReferenceState(self)

    def encode(self, prediction):
        world, events, values = relative_reference_fields(prediction)
        tokens = self._encode_events(events, values)
        if self.decode(tokens) != world:
            raise ValueError('relative-reference encoding changes reindexed authoring')
        return tokens


class InterleavedState(RelativeState):
    def __init__(self, codec):
        super().__init__(codec)
        self.section = 5  # Generic next-record header, not a source list schedule.

    def allowed_tokens(self):
        if not self.started or self.done or self.record is not None:
            return super().allowed_tokens()
        choices = []
        for token, section in self.codec.record_types.items():
            name = AUTHORING_FIELDS[section]
            if len(self.prediction[name]) >= base.MAX_RECORDS.get(name, float('inf')):
                continue
            dependencies = {base._REFERENCE_LIST[kind] for kind in base.FIELD_TYPES[name]
                            if kind in base._REFERENCE_LIST and kind != 'optional_side'}
            if all(self.prediction[dependency] for dependency in dependencies):
                choices.append(token)
        if all(self.prediction[name] for name in AUTHORING_FIELDS if name != 'things'):
            choices.append(base.EOS)
        return tuple(choices)

    def push(self, token):
        if self.record is not None:
            return super().push(token)
        if isinstance(token, bool) or not isinstance(token, int) or token not in self.allowed_tokens():
            raise ValueError(f'illegal interleaved geometry header {token!r}')
        if not self.started:
            self.started = True
        elif token == base.EOS:
            self.done = True
        else:
            self.section = self.codec.record_types[token]
            self.record = {}

    def _finish(self, value):
        super()._finish(value)
        if self.record is None:
            self.section = 5


class RelativeReferenceState(InterleavedState):
    def _finish(self, value):
        name = AUTHORING_FIELDS[self.section]
        kind = base.FIELD_TYPES[name][self.field_index]
        if kind in base._REFERENCE_LIST and not (kind == 'optional_side' and value == 65535):
            count = len(self.prediction[base._REFERENCE_LIST[kind]])
            if not 0 <= value < count:
                raise ValueError(f'relative {kind} age {value} out of causal range {count}')
            value = count - 1 - value
        super()._finish(value)

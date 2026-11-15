"""Causal numeric record memory for the compact authoring-token contract.

Banks may contain the full teacher sequence during training, but each prefix's
emitted counts are its visibility limits. Inference builds the identical bank
incrementally from its own completed records. Names stay in the token stream;
numeric slots in the memory preserve exact authoring values, without a frame,
source identity, normalization fitted on a map, or geometric reconstruction.
"""
from __future__ import annotations

from data.geometry_compact import SCHEMA as COMPACT_SCHEMA
from data.geometry_relative import RelativeState, SCHEMA as RELATIVE_SCHEMA, signed16
from data.geometry_interleaved import RELATIVE_REFERENCE_SCHEMA, SCHEMA as INTERLEAVED_SCHEMA
from data.geometry_tokens import FIELD_TYPES, _REFERENCE_LIST
from data.learned_geometry import AUTHORING_FIELDS, RECORD_FIELDS

SCHEMA = 'doom_geometry_record_memory_v1'
READ_SCHEMA = 'doom_geometry_record_memory_read_v2'
ANCHOR_SCHEMA = 'doom_geometry_record_anchor_v3'
INTERLEAVED_MEMORY_SCHEMA = 'doom_geometry_interleaved_memory_v4'
RELATIVE_POINTER_SCHEMA = 'doom_geometry_interleaved_relative_pointer_v5'
SPATIAL_POINTER_SCHEMA = 'doom_geometry_interleaved_spatial_pointer_v6'
DEPENDENCY_MEMORY_SCHEMA = 'doom_geometry_interleaved_dependency_memory_v7'
RELATIVE_REFERENCE_MEMORY_SCHEMA = 'doom_geometry_interleaved_relative_reference_memory_v8'
RECORD_VALUES_SCHEMA = 'doom_geometry_record_values_v9'
RELATIVE_REFERENCE_RECORD_VALUES_SCHEMA = 'doom_geometry_interleaved_relative_reference_record_values_v9'
VALUE_SCHEMAS = frozenset({RECORD_VALUES_SCHEMA, RELATIVE_REFERENCE_RECORD_VALUES_SCHEMA})
RELATIVE_REFERENCE_SCHEMAS = frozenset({RELATIVE_REFERENCE_MEMORY_SCHEMA, RELATIVE_REFERENCE_RECORD_VALUES_SCHEMA})
DEPENDENCY_SCHEMAS = frozenset({DEPENDENCY_MEMORY_SCHEMA, *RELATIVE_REFERENCE_SCHEMAS, *VALUE_SCHEMAS})
INTERLEAVED_SCHEMAS = frozenset({INTERLEAVED_MEMORY_SCHEMA, RELATIVE_POINTER_SCHEMA, SPATIAL_POINTER_SCHEMA,
                                 *DEPENDENCY_SCHEMAS})
MEMORY_SCHEMAS = frozenset({SCHEMA, READ_SCHEMA, ANCHOR_SCHEMA, *INTERLEAVED_SCHEMAS})
RECORD_WIDTH = 9  # list kind, emitted local ID, seven numeric field slots
REFERENCE_WIDTH = 4  # referenced list kind, byte position, low byte, nullable
ANCHOR_SLOTS = 7
VALUE_SLOTS = 7


def model_memory_options(memory_schema):
    """GeometryRNN memory flags implied by a checkpoint's memory schema (or None)."""
    return {
        'record_memory': memory_schema is not None,
        'memory_read': memory_schema in {READ_SCHEMA, ANCHOR_SCHEMA, *INTERLEAVED_SCHEMAS},
        'record_anchors': memory_schema in {ANCHOR_SCHEMA, *INTERLEAVED_SCHEMAS},
        'relative_pointers': memory_schema in {RELATIVE_POINTER_SCHEMA, SPATIAL_POINTER_SCHEMA, *DEPENDENCY_SCHEMAS},
        'spatial_pointers': memory_schema in {SPATIAL_POINTER_SCHEMA, *DEPENDENCY_SCHEMAS},
        'dependency_memory': memory_schema in DEPENDENCY_SCHEMAS,
        'record_values': memory_schema in VALUE_SCHEMAS,
    }


def require_memory_codec(codec):
    if codec.definition['schema'] not in {COMPACT_SCHEMA, RELATIVE_SCHEMA, INTERLEAVED_SCHEMA, RELATIVE_REFERENCE_SCHEMA}:
        raise ValueError('record memory requires the compact field codec or a supported field-codec variant')


def reference_query(state):
    """Describe the next field solely from an already consumed token prefix."""
    if state.section >= len(AUTHORING_FIELDS) or state.record is None:
        return [-1, 0, -1, 0]
    kind = FIELD_TYPES[AUTHORING_FIELDS[state.section]][state.field_index]
    if kind not in _REFERENCE_LIST:
        return [-1, 0, -1, 0]
    return [AUTHORING_FIELDS.index(_REFERENCE_LIST[kind]), len(state.body),
            state.body[-1] if state.body else -1, int(kind == 'optional_side')]


def current_record_anchors(state):
    """Typed references already completed within the current partial record.

    This dereferences known authoring fields, not a proposed next endpoint.
    Every unknown/future field remains absent. Numeric coordinates are obtained
    from the emitted bank by the model, never from an outside source scaffold.
    """
    result = [[-1, -1] for _ in range(ANCHOR_SLOTS)]
    if state.section >= len(AUTHORING_FIELDS) or state.record is None:
        return result
    name = AUTHORING_FIELDS[state.section]
    for index, (field, kind) in enumerate(zip(RECORD_FIELDS[name], FIELD_TYPES[name])):
        value = state.record.get(field)
        if kind in _REFERENCE_LIST and value is not None and value >= 0:
            result[index] = [AUTHORING_FIELDS.index(_REFERENCE_LIST[kind]), value]
    return result


def current_record_values(state):
    """Completed scalar fields of the current partial record as [value, present].

    A slot becomes present only after its complete scalar field is consumed.
    Half-read byte escapes, unknown future fields, asset names and reference IDs
    remain [0, 0]. Relative codecs expose the causal 16-bit signed relative
    delta for vertices after vertex 0 and for sector ceiling heights; all other
    scalar fields expose their exact world integer value.
    """
    result = [[0, 0] for _ in range(VALUE_SLOTS)]
    if state.section >= len(AUTHORING_FIELDS) or state.record is None:
        return result
    name = AUTHORING_FIELDS[state.section]
    relative = isinstance(state, RelativeState)
    for index, (field, kind) in enumerate(zip(RECORD_FIELDS[name], FIELD_TYPES[name])):
        if kind not in {'signed', 'unsigned'} or field not in state.record:
            continue
        value = state.record[field]
        if relative and name == 'vertices' and state.prediction['vertices']:
            value = signed16(value - state.prediction['vertices'][-1][field])
        elif relative and name == 'sectors' and field == 'ceil_height':
            value = signed16(value - state.record['floor_height'])
        result[index] = [value, 1]
    return result


class RecordMemory:
    """Append one exact numeric row when a parsed record actually completes."""

    def __init__(self):
        self.rows = []
        self.counts = [0] * len(AUTHORING_FIELDS)

    def observe(self, state):
        for kind, name in enumerate(AUTHORING_FIELDS):
            records = state.prediction[name]
            if len(records) < self.counts[kind]:
                raise ValueError('record memory cannot rewind an emitted prefix')
            for index in range(self.counts[kind], len(records)):
                record = records[index]
                values = [record[field] if isinstance(record[field], int) else 0
                          for field in RECORD_FIELDS[name]]
                self.rows.append([kind, index, *values, *([0] * (7 - len(values)))])
            self.counts[kind] = len(records)


def memory_trace(codec, tokens, *, anchors=False, record_values=False):
    """Teacher bank + causal query rows; never expose future records unmasked.

    Each query aligns with the next-token logits after its supplied token.
    No target token/count is included in that query. Padding is the trainer's
    responsibility; absent bank rows use kind/index -1 and are always hidden.
    Optional anchors expose only completed reference fields of that prefix's
    current record, under the separate v3 contract. Optional record_values
    expose only completed scalar fields of that prefix's current record, under
    the separate v9 contract.
    """
    require_memory_codec(codec)
    state, bank, references = codec.state(), RecordMemory(), []
    anchor_rows, value_rows = [], []
    for token in tokens:
        state.push(token)
        bank.observe(state)
        references.append(reference_query(state))
        if anchors:
            anchor_rows.append(current_record_anchors(state))
        if record_values:
            value_rows.append(current_record_values(state))
    state.result()
    schema = (RELATIVE_REFERENCE_RECORD_VALUES_SCHEMA if (record_values and getattr(codec, 'relative_references', False)) else
              RECORD_VALUES_SCHEMA if record_values else
              RELATIVE_REFERENCE_MEMORY_SCHEMA if getattr(codec, 'relative_references', False) else
              INTERLEAVED_MEMORY_SCHEMA if getattr(codec, 'unordered_memory', False) else
              (ANCHOR_SCHEMA if anchors else SCHEMA))
    result = {'schema': schema, 'records': bank.rows, 'references': references}
    if anchors:
        result['anchors'] = anchor_rows
    if record_values:
        result['record_values'] = value_rows
    return result

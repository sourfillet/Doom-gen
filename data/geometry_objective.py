"""Supervised loss groups, never model inputs or inference record counts."""
from data.geometry_compact import SCHEMA as COMPACT_SCHEMA
from data.geometry_relative import SCHEMA as RELATIVE_SCHEMA
from data.geometry_tokens import END_LIST, RECORD
from data.learned_geometry import AUTHORING_FIELDS, RECORD_FIELDS

SCHEMA = 'section_field_stop_balance_v1'


def loss_groups(codec, tokens):
    """Align section/field labels with next-token training targets.

    Each section has one group per field and separate stop/continue classes.
    Fields come from the causal parser. Only boundary class labels use the
    supervised next token, as in class-balanced classification. These labels
    must never enter the recurrent model, its memory, or standalone sampling.
    """
    if codec.definition['schema'] not in {COMPACT_SCHEMA, RELATIVE_SCHEMA}:
        raise ValueError('field-balanced loss requires compact or relative tokens')
    state, result = codec.state(), []
    for index, token in enumerate(tokens[:-1]):
        state.push(token)
        if state.section == len(AUTHORING_FIELDS):
            result.append([-1, -1])  # forced final EOS
        elif state.record is not None:
            result.append([state.section, state.field_index])
        else:
            target = tokens[index + 1]
            if target not in (RECORD, END_LIST):
                raise ValueError('boundary target must be a record or list ending')
            field_count = len(RECORD_FIELDS[AUTHORING_FIELDS[state.section]])
            result.append([state.section, field_count + int(target == END_LIST)])
    state.push(tokens[-1])
    state.result()
    return result

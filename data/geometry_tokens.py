"""Reversible variable-length binary-record codec for learned-map predictions.

The grammar is a file format, not a layout grammar: the model emits every record,
coordinate, reference, action and list ending. Only binary representation,
padding and backward reference ranges can be masked. No source counts, frame,
outline, room graph or authoring scaffold is needed to parse or sample a stream.
"""
from __future__ import annotations

import struct
from functools import lru_cache

from data.learned_geometry import (AUTHORING_FIELDS, PREDICTION_SCHEMA, RECORD_FIELDS,
                                   validate_prediction)

SCHEMA = 'doom_geometry_byte_tokens_v1'
PREFIX_SCHEMA = 'doom_geometry_causal_prefix_v1'
PAD, BOS, EOS, END_LIST, RECORD = range(5)
BYTE_OFFSET = 5
VOCAB_SIZE = BYTE_OFFSET + 256
FIELD_TYPES = {
    'vertices': ('signed', 'signed'),
    'sectors': ('signed', 'signed', 'name', 'name', 'signed', 'signed', 'signed'),
    'sidedefs': ('signed', 'signed', 'name', 'name', 'name', 'sector'),
    'linedefs': ('vertex', 'vertex', 'unsigned', 'unsigned', 'unsigned', 'side', 'optional_side'),
    'things': ('signed', 'signed', 'signed', 'signed', 'unsigned'),
}
STRUCTS = {name: struct.Struct('<' + ''.join('8s' if kind == 'name' else
                                            'h' if kind == 'signed' else 'H' for kind in kinds))
           for name, kinds in FIELD_TYPES.items()}
MAX_RECORDS = {'vertices': 65536, 'sectors': 65536, 'sidedefs': 65535, 'linedefs': 65535}
_REFERENCE_LIST = {'vertex': 'vertices', 'sector': 'sectors', 'side': 'sidedefs', 'optional_side': 'sidedefs'}
_BYTES = tuple(range(BYTE_OFFSET, VOCAB_SIZE))
_NAME_BYTES = tuple(range(BYTE_OFFSET, BYTE_OFFSET + 128))
_ZERO_BYTE = (BYTE_OFFSET,)


@lru_cache(maxsize=4096)
def _reference_tokens(count, optional, byte, low):
    if byte == 0:
        values = set(range(min(count, 256)))
        if optional:
            values.add(255)
    else:
        values = set(range((count - 1 - low) // 256 + 1))
        if optional and low == 255:
            values.add(255)
    return tuple(BYTE_OFFSET + value for value in sorted(values))


def _record_bytes(name, record):
    values = []
    for field, kind in zip(RECORD_FIELDS[name], FIELD_TYPES[name]):
        value = record[field]
        if kind == 'name':
            try:
                value = value.encode('ascii')
            except UnicodeEncodeError as exc:
                raise ValueError(f'{name}.{field} must be ASCII') from exc
            if len(value) > 8 or b'\0' in value:
                raise ValueError(f'{name}.{field} must fit an eight-byte asset name')
        elif kind == 'optional_side' and value == -1:
            value = 65535
        values.append(value)
    try:
        return STRUCTS[name].pack(*values)
    except struct.error as exc:
        raise ValueError(f'{name} record cannot fit the token contract: {exc}') from exc


def _record(name, raw):
    values = STRUCTS[name].unpack(raw)
    return {field: (value.split(b'\0', 1)[0].decode('ascii') if kind == 'name' else
                    -1 if kind == 'optional_side' and value == 65535 else value)
            for field, kind, value in zip(RECORD_FIELDS[name], FIELD_TYPES[name], values)}


class TokenState:
    """Incremental syntax/reference state derived exclusively from emitted tokens.

    ``allowed_tokens`` never imposes a room shape, count, height, action history
    or geometry repair. Invalid physical maps still need independent assessment.
    Required nonempty lists and representable reference/count limits follow the
    prediction/file contract. Sampling budgets must reject incomplete streams.
    """

    def __init__(self):
        self.started = False
        self.done = False
        self.section = 0
        self.body = None
        self.prediction = {'schema': PREDICTION_SCHEMA, **{name: [] for name in AUTHORING_FIELDS}}

    def _field(self):
        name = AUTHORING_FIELDS[self.section]
        offset = 0
        for field, kind in zip(RECORD_FIELDS[name], FIELD_TYPES[name]):
            width = 8 if kind == 'name' else 2
            if len(self.body) < offset + width:
                return field, kind, offset, len(self.body) - offset
            offset += width
        raise ValueError('record already complete')

    def allowed_tokens(self):
        if not self.started:
            return (BOS,)
        if self.done:
            return ()
        if self.section == len(AUTHORING_FIELDS):
            return (EOS,)
        name = AUTHORING_FIELDS[self.section]
        if self.body is None:
            choices = []
            if len(self.prediction[name]) < MAX_RECORDS.get(name, float('inf')):
                choices.append(RECORD)
            if self.prediction[name] or name == 'things':
                choices.append(END_LIST)
            return tuple(choices)
        _, kind, offset, byte = self._field()
        if kind == 'name':
            return _ZERO_BYTE if 0 in self.body[offset:] else _NAME_BYTES
        elif kind in _REFERENCE_LIST:
            count = len(self.prediction[_REFERENCE_LIST[kind]])
            return _reference_tokens(count, kind == 'optional_side', byte, self.body[-1] if byte else -1)
        return _BYTES

    def push(self, token):
        allowed = self.allowed_tokens()
        legal = BYTE_OFFSET <= token < VOCAB_SIZE if isinstance(token, int) and allowed is _BYTES else token in allowed
        if isinstance(token, bool) or not isinstance(token, int) or not legal:
            raise ValueError(f'illegal geometry token {token!r} at section {self.section}')
        if not self.started:
            self.started = True
        elif self.section == len(AUTHORING_FIELDS):
            self.done = True
        elif self.body is None:
            if token == END_LIST:
                self.section += 1
            else:
                self.body = bytearray()
        else:
            self.body.append(token - BYTE_OFFSET)
            name = AUTHORING_FIELDS[self.section]
            if len(self.body) == STRUCTS[name].size:
                self.prediction[name].append(_record(name, self.body))
                self.body = None

    def result(self):
        if not self.done:
            raise ValueError('incomplete geometry token stream; never infer missing list endings')
        validate_prediction(self.prediction)
        return self.prediction

    def features(self):
        """Format position plus counts already emitted, never future counts.

        After a complete record, its count is visible to subsequent decisions;
        before/during it, it is not. Shared positional features are not source
        identities or a geometry scaffold. Coordinates/actions remain outputs.
        """
        position = 0 if self.body is None else len(self.body) + 1
        return [self.section * 32 + position,
                *[len(self.prediction[name]) for name in AUTHORING_FIELDS]]


def encode_prediction(prediction):
    """Stable list/field order, exact integer/name values; no geometry changes.

    Original record IDs/order are retained as implicit list positions. This is
    deterministic for one authoring order, not invariant to arbitrary ID changes.
    Fixed byte vocabulary does not fit assets or geometry on held-out examples.
    """
    validate_prediction(prediction)
    tokens = [BOS]
    for name in AUTHORING_FIELDS:
        for record in prediction[name]:
            tokens.append(RECORD)
            tokens.extend(BYTE_OFFSET + value for value in _record_bytes(name, record))
        tokens.append(END_LIST)
    tokens.append(EOS)
    # Validate syntax/references as well as scalar ranges before training.
    state = TokenState()
    for token in tokens:
        state.push(token)
    return tokens


def decode_tokens(tokens):
    state = TokenState()
    for token in tokens:
        state.push(token)
    return state.result()


def token_masks(tokens):
    """Legal next-token IDs for each prefix, including record/list decisions.

    Returned rows align with targets tokens[1:] for a causal next-token loss.
    Ranges depend only on preceding tokens, never on a future source count.
    """
    state, masks = TokenState(), []
    for token in tokens[:-1]:
        state.push(token)
        masks.append(state.allowed_tokens())
    state.push(tokens[-1])
    state.result()
    return masks


def prefix_features(tokens):
    """One feature row after each supplied token; incomplete prefixes are valid."""
    state, features = TokenState(), []
    for token in tokens:
        state.push(token)
        features.append(state.features())
    return features

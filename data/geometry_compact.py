"""Field-value compression fitted on training families, never room templates.

Frequent whole asset names and 16-bit scalar values get single symbols. Unknown
values retain a lossless byte escape. Tokens never span fields or records: no
source outline, room, connection or record-count template is in the vocabulary.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache

from data import geometry_tokens as base
from data.learned_geometry import AUTHORING_FIELDS, PREDICTION_SCHEMA, RECORD_FIELDS, validate_prediction

SCHEMA = 'doom_geometry_field_tokens_v1'
PREFIX_SCHEMA = 'doom_geometry_field_prefix_v1'
LITERAL_NAME = 261
VALUE_OFFSET = 262


def fit_vocabulary(predictions, *, max_integers=128):
    assets, integers = set(), Counter()
    for prediction in predictions:
        validate_prediction(prediction)
        for name in AUTHORING_FIELDS:
            for record in prediction[name]:
                base._record_bytes(name, record)  # Range/ASCII checks, no quantization.
                for field, kind in zip(RECORD_FIELDS[name], base.FIELD_TYPES[name]):
                    value = record[field]
                    if kind == 'name':
                        assets.add(value)
                    else:
                        integers[value & 65535] += 1
    frequent = sorted(integers, key=lambda value: (-integers[value], value))[:max_integers]
    return {'schema': SCHEMA, 'assets': sorted(assets), 'integers': frequent}


class CompactCodec:
    prefix_schema = PREFIX_SCHEMA
    prefix_states = 384

    def __init__(self, definition):
        if (not isinstance(definition, dict) or set(definition) != {'schema', 'assets', 'integers'}
                or definition['schema'] != SCHEMA):
            raise ValueError('invalid compact geometry vocabulary')
        assets, integers = definition['assets'], definition['integers']
        if (not isinstance(assets, list) or any(not isinstance(a, str) or not a.isascii() or len(a) > 8 or '\0' in a for a in assets)
                or len(set(assets)) != len(assets) or not isinstance(integers, list)
                or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 65535 for v in integers)
                or len(set(integers)) != len(integers)):
            raise ValueError('invalid compact geometry field values')
        self.definition = {'schema': SCHEMA, 'assets': assets[:], 'integers': integers[:]}
        self.assets = {VALUE_OFFSET + i: value for i, value in enumerate(assets)}
        self.integers = {VALUE_OFFSET + len(assets) + i: value for i, value in enumerate(integers)}
        self.asset_ids = {value: token for token, value in self.assets.items()}
        self.integer_ids = {value: token for token, value in self.integers.items()}
        self.vocab_size = VALUE_OFFSET + len(assets) + len(integers)

    def state(self):
        return CompactState(self)

    @lru_cache(maxsize=4096)
    def scalar_tokens(self, kind, count, byte, low):
        if kind in base._REFERENCE_LIST:
            literals = base._reference_tokens(count, kind == 'optional_side', byte, low)
            choices = (token for token, value in self.integers.items()
                       if value < count or kind == 'optional_side' and value == 65535)
        else:
            literals = base._BYTES
            choices = iter(self.integers)
        return literals if byte else tuple(literals) + tuple(choices)

    def encode(self, prediction):
        validate_prediction(prediction)
        tokens = [base.BOS]
        for name in AUTHORING_FIELDS:
            for record in prediction[name]:
                base._record_bytes(name, record)
                tokens.append(base.RECORD)
                for field, kind in zip(RECORD_FIELDS[name], base.FIELD_TYPES[name]):
                    value = record[field]
                    if kind == 'name':
                        if value in self.asset_ids:
                            tokens.append(self.asset_ids[value])
                        else:
                            tokens.append(LITERAL_NAME)
                            tokens.extend(base.BYTE_OFFSET + v for v in value.encode('ascii').ljust(8, b'\0'))
                    else:
                        raw = value & 65535
                        if raw in self.integer_ids:
                            tokens.append(self.integer_ids[raw])
                        else:
                            tokens.extend((base.BYTE_OFFSET + (raw & 255), base.BYTE_OFFSET + (raw >> 8)))
            tokens.append(base.END_LIST)
        tokens.append(base.EOS)
        self.decode(tokens)
        return tokens

    def decode(self, tokens):
        state = self.state()
        for token in tokens:
            state.push(token)
        return state.result()

    def masks(self, tokens):
        state, result = self.state(), []
        for token in tokens[:-1]:
            state.push(token)
            result.append(state.allowed_tokens())
        state.push(tokens[-1])
        state.result()
        return result

    def features(self, tokens):
        state, result = self.state(), []
        for token in tokens:
            state.push(token)
            result.append(state.features())
        return result


class CompactState:
    def __init__(self, codec):
        self.codec = codec
        self.started = self.done = self.literal = False
        self.section = self.field_index = 0
        self.record = None
        self.body = bytearray()
        self.prediction = {'schema': PREDICTION_SCHEMA, **{name: [] for name in AUTHORING_FIELDS}}

    def allowed_tokens(self):
        if not self.started:
            return (base.BOS,)
        if self.done:
            return ()
        if self.section == 5:
            return (base.EOS,)
        name = AUTHORING_FIELDS[self.section]
        if self.record is None:
            return tuple(([base.RECORD] if len(self.prediction[name]) < base.MAX_RECORDS.get(name, float('inf')) else [])
                         + ([base.END_LIST] if self.prediction[name] or name == 'things' else []))
        kind = base.FIELD_TYPES[name][self.field_index]
        if kind == 'name':
            if not self.literal:
                return (LITERAL_NAME, *self.codec.assets)
            return base._ZERO_BYTE if 0 in self.body else base._NAME_BYTES
        count = len(self.prediction[base._REFERENCE_LIST[kind]]) if kind in base._REFERENCE_LIST else 65536
        return self.codec.scalar_tokens(kind, count, len(self.body), self.body[-1] if self.body else -1)

    def _finish(self, value):
        name = AUTHORING_FIELDS[self.section]
        field, kind = RECORD_FIELDS[name][self.field_index], base.FIELD_TYPES[name][self.field_index]
        if kind == 'signed' and value >= 32768:
            value -= 65536
        elif kind == 'optional_side' and value == 65535:
            value = -1
        self.record[field] = value
        self.body = bytearray()
        self.literal = False
        self.field_index += 1
        if self.field_index == len(RECORD_FIELDS[name]):
            self.prediction[name].append(self.record)
            self.record, self.field_index = None, 0

    def push(self, token):
        if isinstance(token, bool) or not isinstance(token, int) or token not in self.allowed_tokens():
            raise ValueError(f'illegal compact geometry token {token!r} at section {self.section}')
        if not self.started:
            self.started = True
        elif self.section == 5:
            self.done = True
        elif self.record is None:
            if token == base.END_LIST:
                self.section += 1
            else:
                self.record = {}
        else:
            kind = base.FIELD_TYPES[AUTHORING_FIELDS[self.section]][self.field_index]
            if kind == 'name':
                if not self.literal:
                    if token == LITERAL_NAME:
                        self.literal = True
                    else:
                        self._finish(self.codec.assets[token])
                else:
                    self.body.append(token - base.BYTE_OFFSET)
                    if len(self.body) == 8:
                        self._finish(bytes(self.body).split(b'\0', 1)[0].decode('ascii'))
            elif token in self.codec.integers:
                self._finish(self.codec.integers[token])
            else:
                self.body.append(token - base.BYTE_OFFSET)
                if len(self.body) == 2:
                    self._finish(self.body[0] | self.body[1] << 8)

    def features(self):
        position = 0 if self.record is None else self.field_index * 9 + len(self.body) + 1
        return [self.section * 64 + position, *[len(self.prediction[name]) for name in AUTHORING_FIELDS]]

    def result(self):
        if not self.done:
            raise ValueError('incomplete compact geometry stream')
        validate_prediction(self.prediction)
        return self.prediction

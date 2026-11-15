"""Version dispatch; legacy byte checkpoints retain their original grammar."""
from types import SimpleNamespace

from data import geometry_tokens as byte


def codec_for(definition=None):
    if definition is None or definition == {'schema': byte.SCHEMA}:
        return SimpleNamespace(definition={'schema': byte.SCHEMA}, vocab_size=byte.VOCAB_SIZE,
                               prefix_schema=byte.PREFIX_SCHEMA, prefix_states=192,
                               encode=byte.encode_prediction, decode=byte.decode_tokens,
                               masks=byte.token_masks, features=byte.prefix_features, state=byte.TokenState)
    from data.geometry_relative import RelativeCodec, SCHEMA as RELATIVE_SCHEMA
    if isinstance(definition, dict) and definition.get('schema') == RELATIVE_SCHEMA:
        return RelativeCodec(definition)
    from data.geometry_interleaved import (InterleavedCodec, RELATIVE_REFERENCE_SCHEMA,
                                           RelativeReferenceCodec, SCHEMA as INTERLEAVED_SCHEMA)
    if isinstance(definition, dict) and definition.get('schema') == INTERLEAVED_SCHEMA:
        return InterleavedCodec(definition)
    if isinstance(definition, dict) and definition.get('schema') == RELATIVE_REFERENCE_SCHEMA:
        return RelativeReferenceCodec(definition)
    from data.geometry_compact import CompactCodec
    return CompactCodec(definition)

"""Experimental full-map supervision for a future variable-structure learner.

Every binary authoring record, including incidence and counts, is supervision.
There is no source scaffold or inverse coordinate frame at inference. Coordinates
are integral world units in v1; learning a normalized representation is separate
work. This module and the prediction decoder do not import Torch.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, fields

from wad_parser.lump_parser import _LINEDEF_FLAG_BITS, _THING_FLAG_BITS
from wad_writer.assembler import _encode_flags, assemble_map_lumps

TARGET_SCHEMA = 'doom_learned_map_v1'
PREDICTION_SCHEMA = 'doom_learned_map_prediction_v1'
AUTHORING_FIELDS = ('vertices', 'sectors', 'sidedefs', 'linedefs', 'things')
CORE_LUMPS = ('VERTEXES', 'SECTORS', 'SIDEDEFS', 'LINEDEFS', 'THINGS')

# Order is part of the versioned contract, available to a future sequence codec.
RECORD_FIELDS = {
    'vertices': ('x', 'y'),
    'sectors': ('floor_height', 'ceil_height', 'floor_flat', 'ceil_flat',
                'light_level', 'special', 'tag'),
    'sidedefs': ('x_offset', 'y_offset', 'upper_tex', 'lower_tex', 'mid_tex', 'sector'),
    'linedefs': ('v1', 'v2', 'flags', 'special', 'tag', 'right_sidedef', 'left_sidedef'),
    'things': ('x', 'y', 'angle', 'type', 'flags'),
}
ASSET_FIELDS = frozenset(('floor_flat', 'ceil_flat', 'upper_tex', 'lower_tex', 'mid_tex'))


def digest(value):
    """Canonical JSON identity; non-finite numbers are never accepted."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def core_hashes(doc):
    lumps = assemble_map_lumps(doc)
    return {name: hashlib.sha256(lumps[name]).hexdigest() for name in CORE_LUMPS}


def validate_conditioning(conditioning):
    """Only explicitly supplied tag/game context; no source topology or scale.

    Further controls need a versioned extension. Asset masks and skill/equipment
    conditioning are deliberately not claimed by this first contract.
    """
    if not isinstance(conditioning, dict) or set(conditioning) != {'tags', 'target_game'}:
        raise ValueError('conditioning must contain only tags and target_game')
    if not isinstance(conditioning['target_game'], str):
        raise ValueError('conditioning target_game must be a string')
    tags = conditioning['tags']
    if not isinstance(tags, dict) or any(
        not isinstance(tag, str) or not tag or isinstance(weight, bool)
        or not isinstance(weight, (int, float)) or not math.isfinite(weight)
        for tag, weight in tags.items()
    ):
        raise ValueError('conditioning tags must map nonempty strings to finite weights')


def validate_prediction(prediction):
    """Check a strict JSON contract before references are walked by the decoder."""
    if not isinstance(prediction, dict) or prediction.get('schema') != PREDICTION_SCHEMA:
        raise ValueError('unsupported learned-map prediction schema')
    if set(prediction) != {'schema', *AUTHORING_FIELDS}:
        raise ValueError('prediction must contain exactly schema and the five authoring lists')
    for name, columns in RECORD_FIELDS.items():
        records = prediction[name]
        if not isinstance(records, list):
            raise ValueError(f'prediction {name} must be a list')
        if name != 'things' and not records:
            raise ValueError(f'prediction {name} must not be empty')
        for index, record in enumerate(records):
            if not isinstance(record, dict) or set(record) != set(columns):
                raise ValueError(f'{name}[{index}] must contain exactly {columns}')
            for field, value in record.items():
                if field in ASSET_FIELDS:
                    if not isinstance(value, str):
                        raise ValueError(f'{name}[{index}].{field} must be an asset string')
                elif isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f'{name}[{index}].{field} must be an integer; no implicit rounding')
            if 'flags' in record and not 0 <= record['flags'] <= 65535:
                raise ValueError(f'{name}[{index}].flags must fit an unsigned flag word')
    # Reference/range/asset validation is completed by assemble_map_lumps on a
    # fresh document, before derived geometry is computed (see decoder).


def _check_source(doc):
    from data.spatial_targets import _classic
    _classic(doc)
    if doc.extra.get('udmf_namespace'):
        raise ValueError('learned-map v1 does not support UDMF namespaces')
    # The existing writer intentionally covers only binary fields. Refuse source
    # extensions it would otherwise omit, rather than claiming lossless capture.
    for name in AUTHORING_FIELDS:
        for record in getattr(doc, name):
            allowed = {'unrecognized_flag_bits'} if name in {'linedefs', 'things'} else set()
            if set(record.extra) - allowed:
                raise ValueError(f'unsupported {name} {record.id} extension fields')
    for line in doc.linedefs:
        if line.args is not None:
            raise ValueError('learned-map v1 does not support linedef args')
    for thing in doc.things:
        if any(getattr(thing, name) is not None for name in ('z', 'tid', 'special', 'args')):
            raise ValueError('learned-map v1 does not support extended things')
    for name, table in (('linedefs', _LINEDEF_FLAG_BITS), ('things', _THING_FLAG_BITS)):
        represented = {attr for _, attr in table}
        for record in getattr(doc, name):
            if any(getattr(record.flags, f.name) for f in fields(record.flags) if f.name not in represented):
                raise ValueError(f'unsupported {name} {record.id} flag fields')


def build_target(doc, *, tags=None, target_game=''):
    """Export binary-faithful supervision, with all source identity audit-only.

    No geometry/traversal eligibility is inferred: malformed boundaries, source
    controls, missing endings and unsupported mechanics need separate audits.
    Caller context is never inferred from source names or source geometry.
    """
    from data.spatial_targets import authoring_digest
    _check_source(doc)
    conditioning = {'tags': copy.deepcopy({} if tags is None else tags), 'target_game': target_game}
    validate_conditioning(conditioning)
    prediction = {'schema': PREDICTION_SCHEMA}
    for name, columns in RECORD_FIELDS.items():
        prediction[name] = []
        for record in getattr(doc, name):
            row = {}
            for field in columns:
                value = getattr(record, field)
                if field == 'flags':
                    table = _LINEDEF_FLAG_BITS if name == 'linedefs' else _THING_FLAG_BITS
                    value = _encode_flags(value, table, record.extra)
                elif field in {'x', 'y'}:
                    value = int(value)  # _classic already checked exact integrality.
                row[field] = value
            prediction[name].append(row)
    validate_prediction(prediction)
    target = {'schema': TARGET_SCHEMA, 'conditioning': conditioning, 'supervision': prediction,
              'audit': {'source_meta': asdict(doc.meta),
                        'source_document_canonical_sha256': digest(doc.to_dict()),
                        'source_authoring_sha256': authoring_digest(doc),
                        'core_lump_sha256': core_hashes(doc),
                        'counts': {name: len(prediction[name]) for name in AUTHORING_FIELDS},
                        'geometry_eligibility': 'not_assessed',
                        'progression_eligibility': 'not_assessed',
                        'coordinate_contract': 'integral world XY/Z; no source frame at inference'}}
    target['fingerprint'] = digest(target)
    return target


def validate_target(target):
    """Verify signed target contents without consulting any source/scaffold."""
    if not isinstance(target, dict) or target.get('schema') != TARGET_SCHEMA:
        raise ValueError('unsupported learned-map target schema')
    if set(target) != {'schema', 'conditioning', 'supervision', 'audit', 'fingerprint'}:
        raise ValueError('invalid learned-map target fields')
    if not isinstance(target['audit'], dict):
        raise ValueError('invalid learned-map target audit')
    if digest({key: value for key, value in target.items() if key != 'fingerprint'}) != target['fingerprint']:
        raise ValueError('learned-map target fingerprint mismatch')
    validate_conditioning(target['conditioning'])
    validate_prediction(target['supervision'])
    from generation.learned_geometry import decode_prediction
    doc = decode_prediction(target['supervision'])
    if (core_hashes(doc) != target['audit'].get('core_lump_sha256')
            or {name: len(getattr(doc, name)) for name in AUTHORING_FIELDS} != target['audit'].get('counts')):
        raise ValueError('learned-map target audit differs from supervision')


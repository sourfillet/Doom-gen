"""Prediction-only decoder and static assessment for experimental learned maps.

This consumes *all* authoring records, including topology, from its one payload.
It adds no rooms, routes, things, exits, controls or fallback source geometry.
An input payload is not evidence that a neural model produced it. No existing
generator/default or publication policy is changed by this experimental module.
"""
from __future__ import annotations

from data.learned_geometry import (AUTHORING_FIELDS, CORE_LUMPS, PREDICTION_SCHEMA,
                                   core_hashes, digest, validate_prediction)
from wad_parser.lump_parser import (_decode_linedef_flags, _decode_thing_flags,
                                    parse_linedefs_doom, parse_sectors, parse_sidedefs,
                                    parse_things_doom, parse_vertexes)
from wad_parser.types import (BoundingBox, DerivedData, Linedef, MapDocument, MapMeta,
                              Sector, Sidedef, Thing, Vertex, resolve_doomed,
                              resolve_linedef_special)
from wad_writer.assembler import assemble_map_lumps


def decode_prediction(prediction, *, map_name='MAP01'):
    """Decode variable lists of predicted integer binary records, without repair.

    IDs are implicit list positions; shared references stay shared. Ranges,
    indices and collapsed linedefs are checked *before* geometry computation.
    Boundary/player/action validity is a separate assessment, not a decoder
    promise. Missing starts/exits remain missing and control sectors remain.
    """
    validate_prediction(prediction)
    vertices = [Vertex(i, **row) for i, row in enumerate(prediction['vertices'])]
    sectors = [Sector(i, **row) for i, row in enumerate(prediction['sectors'])]
    sides = [Sidedef(i, **row) for i, row in enumerate(prediction['sidedefs'])]
    lines, things = [], []
    for index, row in enumerate(prediction['linedefs']):
        flags, extra = _decode_linedef_flags(row['flags'])
        name, trigger = resolve_linedef_special(row['special'])
        lines.append(Linedef(index, **{key: value for key, value in row.items() if key != 'flags'},
                             flags=flags, extra=extra, special_name=name, trigger_type=trigger))
    for index, row in enumerate(prediction['things']):
        flags, extra = _decode_thing_flags(row['flags'])
        name, category, unresolved = resolve_doomed(row['type'])
        things.append(Thing(index, **{key: value for key, value in row.items() if key != 'flags'},
                            flags=flags, extra=extra, type_name=name, category=category, unresolved=unresolved))
    doc = MapDocument('1.0', MapMeta(map_name, '<prediction>', '', 'doom'), BoundingBox(0, 0, 0, 0),
                      vertices, sectors, lines, sides, things, DerivedData())
    assemble_map_lumps(doc)
    from wad_parser.map_extractor import compute_derived
    doc.bounds, doc.derived = compute_derived(vertices, sectors, lines, sides, things)
    doc.extra['learned_geometry_prediction'] = {
        'schema': PREDICTION_SCHEMA, 'prediction_sha256': digest(prediction),
        'ownership': {name: 'supplied_prediction' for name in AUTHORING_FIELDS},
        'model_execution': 'not_verified', 'repairs': [], 'transformations': [],
    }
    return doc


def assess_prediction(prediction, *, map_name='MAP01', require_exit=True, require_connected=True):
    """Return a stage report, retaining raw failures and making no gameplay claim.

    Checks in-memory and binary-lump readback geometry independently. Node
    building, engine completion and publishing are deliberately not performed.
    The caller must archive its prediction/checkpoint inputs separately.
    """
    from generation.validator import validate_document
    from wad_parser.map_extractor import compute_derived

    report = {'schema': 'doom_learned_map_assessment_v1', 'status': 'rejected',
              'prediction_sha256': None, 'model_execution': 'not_verified',
              'ownership': {name: 'supplied_prediction' for name in AUTHORING_FIELDS},
              'repairs': [], 'transformations': [],
              'requirements': {'exit': require_exit, 'connected': require_connected},
              'checks': {'decode': 'not_run', 'in_memory': 'not_run', 'serialized': 'not_run',
                         'node_built': 'not_run', 'engine': 'not_run'}}
    try:
        # Validate before hashing so malformed non-JSON values fail diagnostically.
        validate_prediction(prediction)
        report['prediction_sha256'] = digest(prediction)
        doc = decode_prediction(prediction, map_name=map_name)
        report['checks']['decode'] = 'passed'
        report['counts'] = {name: len(getattr(doc, name)) for name in AUTHORING_FIELDS}
        report['core_lump_sha256'] = core_hashes(doc)
        report['checks']['in_memory'] = validate_document(doc, require_exit=require_exit,
                                                         require_connected=require_connected)
        lumps = assemble_map_lumps(doc)
        vertices, sectors, sides, lines, things = (
            parser(lumps[name]) for name, parser in zip(CORE_LUMPS,
                (parse_vertexes, parse_sectors, parse_sidedefs, parse_linedefs_doom, parse_things_doom)))
        bounds, derived = compute_derived(vertices, sectors, lines, sides, things)
        reread = MapDocument('1.0', MapMeta(map_name, '<serialized-prediction>', '', 'doom'),
                            bounds, vertices, sectors, lines, sides, things, derived)
        report['checks']['serialized'] = validate_document(reread, require_exit=require_exit,
                                                          require_connected=require_connected)
        if core_hashes(reread) != report['core_lump_sha256']:
            raise ValueError('serialized authoring differs from prediction')
        if not any(report['checks'][key]['errors'] for key in ('in_memory', 'serialized')):
            report['status'] = 'static_valid'
        return report
    except (ValueError, TypeError, OverflowError) as exc:
        report['reason'] = str(exc)
        if report['checks']['decode'] == 'not_run':
            report['checks']['decode'] = 'failed'
        return report

"""Coordinate-free authoring scaffolds for an explicit memorization control.

The source supplies vertex/line/sector/thing identities, actions and assets.
All XY coordinates and every sector elevation are supervision, never fallback
geometry in the inference scaffold. This is not RoomGraph-only generation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

SCHEMA = 'doom_spatial_prediction_scaffold_v1'
PREDICTION_SCHEMA = 'doom_spatial_prediction_v1'
HEIGHT_SCALE = 128.0


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def graph_from_source_record(document_bytes, record):
    """Bind frozen graph evidence to the exact source bytes and map identity."""
    from roomgraph.types import RoomGraph
    if record.get('graph_snapshot_version') != 1:
        raise ValueError('unsupported source graph snapshot version')
    if hashlib.sha256(document_bytes).hexdigest() != record.get('document_sha256'):
        raise ValueError('source document fingerprint differs from graph record')
    meta = json.loads(document_bytes)['meta']
    graph = RoomGraph.from_dict(record['graph_snapshot'])
    if (meta['wad_hash'] != record.get('wad_hash') or meta['map_name'] != record.get('map')
            or graph.map_name != meta['map_name']):
        raise ValueError('source/graph map identity mismatch')
    return graph


def build_prediction_scaffold(target):
    """Split a lossless reconstruction target into conditioning and numeric labels.

    A single shared vertex slot is retained wherever linedefs reference it.
    Unreferenced controls and all things survive. No ring, centroid, width,
    coordinate, height or original BSP cache is retained as fallback geometry.
    """
    from data.spatial_targets import SCHEMA as TARGET_SCHEMA, authoring_digest
    from roomgraph.spatial import reconstruct_spatial_target
    if target.get('schema') != TARGET_SCHEMA:
        raise ValueError('prediction requires a lossless spatial reconstruction target; '
                         'spatial_geometry_v1 omits source controls and is not supported')
    # Verify the incoming contract before removing supervision.
    restored = reconstruct_spatial_target(target)
    if authoring_digest(restored) != target.get('source_authoring_sha256'):
        raise ValueError('source authoring fingerprint mismatch')
    scaffold = {'schema': SCHEMA, 'frame': copy.deepcopy(target['frame']),
                'meta': copy.deepcopy(target['meta']),
                'source_authoring_sha256': target['source_authoring_sha256'],
                'linedefs': copy.deepcopy(target['linedefs']),
                'sidedefs': copy.deepcopy(target['sidedefs'])}
    labels, kinds = [], []
    for name, fields, factor, kind in [('vertices', ('x', 'y'), 1, 'vertex'),
                                       ('sectors', ('floor_height', 'ceil_height'), HEIGHT_SCALE, 'sector'),
                                       ('things', ('x', 'y'), 1, 'thing')]:
        scaffold[name] = []
        for item in target[name]:
            record = copy.deepcopy(item)
            labels.append([record.pop(field) / factor for field in fields])
            kinds.append(kind)
            scaffold[name].append(record)
    # Geometry-bearing auxiliary hierarchy is deliberately omitted. Retain only
    # semantic room/component incidence and typed requested room connections.
    hierarchy = target.get('hierarchy', {})
    rooms = [{'id': r['id'], 'roles': list(r.get('roles', [])),
              'sectors': sorted({c['source_component'][0] for c in r['components']})}
             for r in hierarchy.get('rooms', [])]
    connections = [{k: c[k] for k in ('a', 'b', 'kind')}
                   for c in hierarchy.get('connections', [])]
    scaffold['rooms'], scaffold['connections'] = rooms, connections
    scaffold['slot_kinds'] = kinds
    scaffold['scope'] = ('Single-source memorization with retained authoring topology, actions/assets/thing metadata '
                         'and frame. No novel topology, RoomGraph-only generation or gameplay certificate.')
    return scaffold, {'schema': PREDICTION_SCHEMA, 'values': labels}


def scaffold_edges(scaffold):
    """Coordinate-free incidence graph, with room nodes after supervised slots."""
    nv, ns = len(scaffold['vertices']), len(scaffold['sectors'])
    slots = len(scaffold['slot_kinds'])
    edges = set()

    def connect(a, b):
        if a != b:
            edges.update(((a, b), (b, a)))

    for line in scaffold['linedefs']:
        connect(line['v1'], line['v2'])
        for side in (line['right_sidedef'], line['left_sidedef']):
            if side >= 0:
                sector = nv + scaffold['sidedefs'][side]['sector']
                connect(line['v1'], sector)
                connect(line['v2'], sector)
    room_indices = {r['id']: slots + i for i, r in enumerate(scaffold['rooms'])}
    if len(room_indices) != len(scaffold['rooms']):
        raise ValueError('duplicate scaffold room ID')
    for room in scaffold['rooms']:
        for sid in room['sectors']:
            if not 0 <= sid < ns:
                raise ValueError('scaffold room references missing sector')
            connect(nv + sid, room_indices[room['id']])
    for conn in scaffold['connections']:
        if conn['a'] not in room_indices or conn['b'] not in room_indices:
            raise ValueError('scaffold connection references missing room')
        connect(room_indices[conn['a']], room_indices[conn['b']])
    return sorted(edges)


def decode_prediction(scaffold, prediction):
    """Decode only predicted numeric values, quantized once to classic integers.

    Quantization is explicit prediction policy. The lossless source decoder's
    strict inverse-transform contract is left unchanged. Invalid/collapsed
    geometry stays rejected; no source-coordinate substitution or repair occurs.
    """
    from data.spatial_targets import SCHEMA as TARGET_SCHEMA
    from roomgraph.spatial import reconstruct_spatial_target
    if scaffold.get('schema') != SCHEMA or prediction.get('schema') != PREDICTION_SCHEMA:
        raise ValueError('unsupported spatial prediction contract')
    count = sum(len(scaffold[name]) for name in ('vertices', 'sectors', 'things'))
    values = prediction.get('values', [])
    if len(values) != count or len(scaffold['slot_kinds']) != count:
        raise ValueError('prediction slot count differs from scaffold')
    if any(len(row) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                               or not math.isfinite(v) for v in row) for row in values):
        raise ValueError('prediction values must be finite numeric pairs')
    frame = scaffold['frame']
    origin, scale, floor_origin = frame['origin'], frame['scale'], frame['floor_origin']
    if (len(origin) != 2 or any(not math.isfinite(v) for v in [*origin, scale, floor_origin])
            or scale <= 0):
        raise ValueError('invalid prediction frame')
    target = {key: copy.deepcopy(scaffold[key]) for key in
              ('frame', 'meta', 'source_authoring_sha256', 'linedefs', 'sidedefs')}
    target['schema'] = TARGET_SCHEMA
    cursor = 0
    for name, fields in [('vertices', ('x', 'y')), ('sectors', ('floor_height', 'ceil_height')),
                         ('things', ('x', 'y'))]:
        target[name] = copy.deepcopy(scaffold[name])
        for item in target[name]:
            row = values[cursor]
            cursor += 1
            for axis, field in enumerate(fields):
                if name == 'sectors':
                    # Round in world space, then preserve the strict decoder's
                    # own normalization/inverse transform.
                    item[field] = round(floor_origin + row[axis] * HEIGHT_SCALE) - floor_origin
                else:
                    item[field] = (round(origin[axis] + row[axis] * scale) - origin[axis]) / scale
    doc = reconstruct_spatial_target(target)
    doc.extra['spatial_prediction'] = {'schema': PREDICTION_SCHEMA,
                                     'scaffold_sha256': digest(scaffold),
                                     'prediction_sha256': digest(prediction),
                                     'quantization': 'nearest integer world XY and sector height',
                                     'scope': scaffold['scope']}
    return doc

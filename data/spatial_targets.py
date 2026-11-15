"""Experimental, lossless authored-geometry reconstruction targets.

This is a source reconstruction gate, not the v3 graph target, a vocabulary or
a neural architecture. Shared boundaries keep their original identity; optional
room hierarchy retains every component and height instead of one room height.
No compiled BSP regions or cached traversal results are fed to reconstruction.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict

SCHEMA = 'doom_spatial_reconstruction_v1'
AUTHORING_FIELDS = ('vertices', 'sectors', 'linedefs', 'sidedefs', 'things')


def _authoring(doc):
    return {name: [asdict(item) for item in getattr(doc, name)] for name in AUTHORING_FIELDS}


def authoring_digest(doc):
    """Authored fields, including actions/assets/unknown flags; excludes caches.

    Classic coordinates are integers. Canonicalizing equivalent floats lets a
    serialized normalization round trip be compared without float spelling noise.
    """
    def canonical(value):
        if isinstance(value, float) and math.isfinite(value) and value.is_integer():
            return int(value)
        if isinstance(value, dict):
            return {key: canonical(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [canonical(item) for item in value]
        return value
    raw = json.dumps(canonical(_authoring(doc)), sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _classic(doc):
    if doc.meta.format != 'doom':
        raise ValueError('spatial reconstruction currently supports Doom binary maps only')
    if any(s.floor_slope is not None or s.ceil_slope is not None for s in doc.sectors):
        raise ValueError('spatial reconstruction does not support slopes')
    if any(v.z is not None for v in doc.vertices):
        raise ValueError('spatial reconstruction does not support vertex heights')
    for point in [*doc.vertices, *doc.things]:
        for value in (point.x, point.y):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value != round(value)):
                raise ValueError('spatial reconstruction requires integral classic XY coordinates')
    # Binary validation checks IDs, references, flags, ranges and unsupported
    # per-element extension fields without mutating or repairing the document.
    from wad_writer.assembler import assemble_map_lumps
    assemble_map_lumps(doc)


def build_spatial_target(doc, graph=None):
    """Normalize a source map and optionally attach exact room/component groups.

    All authoring elements survive, including unreferenced vertices, remote
    controls and excluded regions. Component rings are auxiliary evidence; they
    are not trusted as replacement compiled geometry by the decoder.
    """
    from wad_parser.bsp_geometry import component_polygons
    from wad_parser.components import ComponentIndex
    from wad_parser.geometry import sector_geometry, sector_points

    _classic(doc)
    starts = [thing for thing in doc.things if thing.type == 1]
    if len(starts) != 1:
        raise ValueError('spatial reconstruction requires exactly one player-one start')
    start = starts[0]
    origin = [start.x, start.y]
    index = ComponentIndex(doc.vertices, sector_geometry(doc))
    sid = index.owner(*origin)
    if sid is None:
        raise ValueError('player-one start has no source sector')
    floor_origin = next(sector.floor_height for sector in doc.sectors if sector.id == sid)
    xs = [v.x for v in doc.vertices]
    ys = [v.y for v in doc.vertices]
    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1)
    payload = _authoring(doc)
    for record in [*payload['vertices'], *payload['things']]:
        record['x'] = (record['x'] - origin[0]) / scale
        record['y'] = (record['y'] - origin[1]) / scale
    for sector in payload['sectors']:
        sector['floor_height'] -= floor_origin
        sector['ceil_height'] -= floor_origin
    polygons = {(sector.id, shell): polygon for sector in doc.sectors
                for shell, polygon in component_polygons(sector_points(doc, sector.id))}
    sectors = {sector.id: sector for sector in doc.sectors}

    def part(key):
        polygon, sector = polygons[key], sectors[key[0]]
        return {'source_component': list(key),
                'rings': [[[(x - origin[0]) / scale, (y - origin[1]) / scale]
                           for x, y in list(ring.coords)[:-1]]
                          for ring in [polygon.exterior, *polygon.interiors]],
                'floor': sector.floor_height - floor_origin,
                'ceiling': sector.ceil_height - floor_origin}

    hierarchy, assigned = {'rooms': [], 'ungrouped_components': []}, set()
    if graph is not None:
        room_ids = set()
        for room in graph.rooms:
            if room.id in room_ids:
                raise ValueError('duplicate room ID')
            room_ids.add(room.id)
            keys = [tuple(key) for key in room.components]
            if not keys:
                for sid in room.sectors:
                    candidates = [key for key in polygons if key[0] == sid]
                    if len(candidates) != 1:
                        raise ValueError('room requires explicit component ownership')
                    keys.extend(candidates)
            parts = []
            for key in keys:
                if key not in polygons:
                    raise ValueError(f'missing component {key}')
                if key in assigned:
                    raise ValueError(f'duplicate component ownership {key}')
                assigned.add(key)
                parts.append(part(key))
            hierarchy['rooms'].append({'id': room.id, 'roles': list(room.roles), 'components': parts})
        if any(edge.a not in room_ids or edge.b not in room_ids for edge in graph.connections):
            raise ValueError('connection references missing room')
        hierarchy.update(connections=[asdict(edge) for edge in graph.connections],
                         excluded_components=[list(key) for key in graph.excluded_components],
                         collapsed_door_components=[list(key) for key in graph.collapsed_door_components])
    hierarchy['ungrouped_components'] = [part(key) for key in sorted(polygons) if key not in assigned]
    return {'schema': SCHEMA, 'frame': {'origin': origin, 'scale': scale, 'floor_origin': floor_origin},
            'meta': asdict(doc.meta), 'source_authoring_sha256': authoring_digest(doc),
            'hierarchy': hierarchy, **payload,
            'limits': 'Source reconstruction contract; no neural decoder, BSP cache or global history certificate.'}

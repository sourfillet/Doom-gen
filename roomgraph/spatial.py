"""Reconstruct authored spatial targets without graph routing or room templates."""
from __future__ import annotations

import copy
import math


def reconstruct_spatial_target(target):
    """Invert one experimental classic target and recompute all derived data.

    Reject fractional authoring coordinates instead of silently quantizing a
    prediction. Source BSP recovery is deliberately absent: malformed raw
    authoring remains visible and publication must still pass its usual checks.
    """
    from data.spatial_targets import SCHEMA, AUTHORING_FIELDS, _classic
    from wad_parser.types import MapDocument

    if target.get('schema') != SCHEMA:
        raise ValueError('unsupported spatial reconstruction schema')
    frame = target['frame']
    origin, scale, floor = frame['origin'], frame['scale'], frame['floor_origin']
    values = [*origin, scale, floor]
    if (len(origin) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                              or not math.isfinite(v) for v in values) or scale <= 0):
        raise ValueError('invalid spatial coordinate frame')
    payload = {name: copy.deepcopy(target[name]) for name in AUTHORING_FIELDS}

    def coordinate(value):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or abs(value - round(value)) > 1e-6):
            raise ValueError('spatial coordinate cannot reconstruct an integral classic value')
        return round(value)

    for point in [*payload['vertices'], *payload['things']]:
        point['x'] = coordinate(origin[0] + point['x'] * scale)
        point['y'] = coordinate(origin[1] + point['y'] * scale)
    for sector in payload['sectors']:
        sector['floor_height'] = coordinate(sector['floor_height'] + floor)
        sector['ceil_height'] = coordinate(sector['ceil_height'] + floor)
    # Empty derived data forces recomputation from the decoded authoring. Do
    # not carry the original BSP's sector_regions across this boundary.
    payload.update(schema_version='1.0', meta=copy.deepcopy(target['meta']),
                   bounds={'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0}, derived={},
                   extra={'spatial_reconstruction': {'schema': SCHEMA,
                          'source_authoring_sha256': target.get('source_authoring_sha256')}})
    # Validate structural references before geometry walks them.
    from wad_parser.types import (MapMeta, BoundingBox, DerivedData, Vertex, Sector,
                                  Sidedef, Linedef, LinedefFlags, Thing, ThingFlags)
    try:
        probe = MapDocument('1.0', MapMeta(**{**payload['meta'], 'mapinfo': None}), BoundingBox(0, 0, 0, 0),
                            [Vertex(**v) for v in payload['vertices']], [Sector(**s) for s in payload['sectors']],
                            [Linedef(**{**line, 'flags': LinedefFlags(**line['flags'])}) for line in payload['linedefs']],
                            [Sidedef(**side) for side in payload['sidedefs']],
                            [Thing(**{**thing, 'flags': ThingFlags(**thing['flags'])}) for thing in payload['things']],
                            DerivedData())
        _classic(probe)
    except (TypeError, IndexError, KeyError) as exc:
        raise ValueError(f'invalid spatial authoring fields: {exc}') from exc
    return MapDocument.from_dict(payload)

"""Geometry-only source evidence and descriptors, separate from gameplay checks.

These checks do not exempt controls/actors from publication validation and do
not certify progression. They let authored walls/heights teach a geometry model
without incorrectly requiring an entire source encounter to pass its validator.
"""
from __future__ import annotations

import math

from data.learned_geometry import CORE_LUMPS, core_hashes
from generation.learned_geometry import decode_prediction
from wad_parser.lump_parser import (parse_linedefs_doom, parse_sectors, parse_sidedefs,
                                    parse_things_doom, parse_vertexes)
from wad_parser.types import MapDocument, MapMeta
from wad_writer.assembler import assemble_map_lumps


def _geometry_errors(doc):
    from wad_parser.geometry import geometry_conflicts, sector_geometry
    errors = list(doc.derived.geometry_issues)
    errors.extend(geometry_conflicts(doc))
    errors.extend(f'sector {sector.id} has no closed boundary' for sector in doc.sectors
                  if not sector_geometry(doc)[sector.id])
    return errors


def assess_geometry(prediction):
    """Independent authored/serialized boundary checks; no actor/history verdict.

    All source actions, objects and controls remain in the prediction. Compiled
    geometry, local motion, global progression and population need other checks.
    """
    from wad_parser.map_extractor import compute_derived
    report = {'schema': 'doom_geometry_source_evidence_v1', 'status': 'rejected',
              'checks': {'authoring': 'not_run', 'serialized': 'not_run',
                         'compiled': 'not_run', 'actors': 'not_assessed',
                         'progression': 'not_assessed', 'engine': 'not_run'},
              'repairs': []}
    try:
        doc = decode_prediction(prediction)
        report['checks']['authoring'] = {'errors': _geometry_errors(doc)}
        lumps = assemble_map_lumps(doc)
        vertices, sectors, sides, lines, things = (
            parser(lumps[name]) for name, parser in zip(CORE_LUMPS,
                (parse_vertexes, parse_sectors, parse_sidedefs, parse_linedefs_doom, parse_things_doom)))
        bounds, derived = compute_derived(vertices, sectors, lines, sides, things)
        reread = MapDocument('1.0', MapMeta('MAP01', '<geometry-evidence>', '', 'doom'),
                            bounds, vertices, sectors, lines, sides, things, derived)
        report['checks']['serialized'] = {'errors': _geometry_errors(reread)}
        if core_hashes(doc) != core_hashes(reread):
            raise ValueError('serialized authoring differs from prediction')
        if not any(report['checks'][name]['errors'] for name in ('authoring', 'serialized')):
            report['status'] = 'raw_geometry_valid'
        return report
    except (ValueError, TypeError, OverflowError) as exc:
        report['reason'] = str(exc)
        return report


def layout_statistics(doc):
    """Describe visible floor geometry; sector count alone is not a room count.

    Collinear linedef splits do not create corners. A flat rectangle with many
    decorative sector IDs still gets single_flat_rectangle=True. These are
    measurements, not a recipe for adding geometry or a gameplay certificate.
    """
    from shapely import union_all
    from wad_parser.bsp_geometry import component_polygons
    from wad_parser.geometry import sector_points
    polygons, levels = [], set()
    for sector in doc.sectors:
        if sector.ceil_height <= sector.floor_height:
            continue
        parts = [poly for _, poly in component_polygons(sector_points(doc, sector.id))]
        polygons.extend(parts)
        if sum(poly.area for poly in parts) >= 32 * 32:
            levels.add(sector.floor_height)
    if not polygons:
        raise ValueError('no positive-height floor footprint')
    footprint = union_all(polygons)
    parts = [footprint] if footprint.geom_type == 'Polygon' else list(footprint.geoms)

    def corners(ring):
        points = list(ring.coords)[:-1]
        return sum((b[0] - a[0]) * (c[1] - b[1]) != (b[1] - a[1]) * (c[0] - b[0])
                   for a, b, c in zip(points[-1:] + points[:-1], points, points[1:] + points[:1]))

    outer_corners = sum(corners(part.exterior) for part in parts)
    holes = sum(len(part.interiors) for part in parts)
    rectangle = (len(parts) == 1 and not holes and
                 abs(footprint.minimum_rotated_rectangle.area - footprint.area) <= max(1, footprint.area) * 1e-9)
    blocking = 0.0
    for line in doc.linedefs:
        if line.left_sidedef < 0:
            continue
        a, b = [doc.sidedefs[index].sector for index in (line.right_sidedef, line.left_sidedef)]
        if a != b and (line.flags.impassable or line.flags.block_players or line.flags.block_everything):
            first, second = doc.vertices[line.v1], doc.vertices[line.v2]
            blocking += math.hypot(first.x - second.x, first.y - second.y)
    return {'schema': 'doom_layout_statistics_v1', 'sectors': len(doc.sectors),
            'footprint_components': len(parts), 'outline_corners': outer_corners,
            'holes': holes, 'floor_area': footprint.area, 'floor_levels': sorted(levels),
            'internal_blocking_length': blocking,
            'single_flat_rectangle': bool(rectangle and len(levels) <= 1 and not blocking),
            'scope': 'geometry descriptors; not semantic room counts, novel-layout or gameplay evidence'}

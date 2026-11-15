"""Target contract for spatial geometry conditioned on room-graph topology.

Preserves room placement, multi-height interior sectors, shared boundaries,
typed openings, and explicit exclusion masks for unsupported motion/actions.
This contract separates topological graph structure from spatial layout targets.
"""
from __future__ import annotations

from typing import Any

from shapely import Polygon
from wad_parser.types import MapDocument
from roomgraph.types import RoomGraph

SCHEMA = 'doom_spatial_geometry_v1'


def build_spatial_geometry_target(
    doc: MapDocument,
    graph: RoomGraph,
    audit_data: dict[str, Any] | None = None,
    index_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract a topology-conditioned spatial geometry target from an authored map.

    Parameters:
        doc: The authored MapDocument.
        graph: The segmented RoomGraph with rooms, components, and connections.
        audit_data: Optional portal audit record (from spatial-audit-v1).
        index_data: Optional spatial index record (from spatial-index-v2).

    Returns:
        A dictionary matching doom_spatial_geometry_v1 schema.
    """
    from wad_parser.bsp_geometry import component_polygons
    from wad_parser.components import ComponentIndex
    from wad_parser.geometry import sector_geometry, sector_points

    starts = [thing for thing in doc.things if thing.type == 1]
    if len(starts) != 1:
        raise ValueError('spatial geometry target requires exactly one player-one start')
    start = starts[0]
    origin = [float(start.x), float(start.y)]

    index = ComponentIndex(doc.vertices, sector_geometry(doc))
    start_sid = index.owner(*origin)
    if start_sid is None:
        raise ValueError('player-one start has no source sector')
    start_sector = next(s for s in doc.sectors if s.id == start_sid)
    floor_origin = float(start_sector.floor_height)

    xs = [float(v.x) for v in doc.vertices]
    ys = [float(v.y) for v in doc.vertices]
    scale = float(max(max(xs) - min(xs), max(ys) - min(ys), 1.0))

    # Pre-extract all component polygons and sector properties
    polygons: dict[tuple[int, int], Polygon] = {}
    sectors_map = {s.id: s for s in doc.sectors}
    for sector in doc.sectors:
        for shell, poly in component_polygons(sector_points(doc, sector.id)):
            polygons[(sector.id, shell)] = poly

    # 1. Topology conditioning
    topology_rooms = []
    room_nodes_map = {}
    for room in graph.rooms:
        topology_rooms.append({
            'id': room.id,
            'roles': list(room.roles),
            'target_area': float(room.area),
            'source_sectors': list(room.sectors),
        })
        room_nodes_map[room.id] = room

    topology_connections = []
    for conn in graph.connections:
        topology_connections.append({
            'a': conn.a,
            'b': conn.b,
            'kind': conn.kind,
            'length': float(conn.length) if hasattr(conn, 'length') else 0.0,
        })

    # 2. Spatial placements & interior architecture
    room_placements = []
    room_interiors = []
    assigned_components = set()

    for room in graph.rooms:
        keys = [tuple(k) for k in room.components]
        if not keys:
            for sid in room.sectors:
                keys.extend([k for k in polygons if k[0] == sid])

        room_polys = []
        components_data = []
        room_floors = []
        room_ceils = []

        for key in keys:
            if key not in polygons:
                continue
            poly = polygons[key]
            assigned_components.add(key)
            room_polys.append(poly)
            sec = sectors_map[key[0]]
            floor_rel = float(sec.floor_height) - floor_origin
            ceil_rel = float(sec.ceil_height) - floor_origin
            room_floors.append(floor_rel)
            room_ceils.append(ceil_rel)

            ext_ring = [[(x - origin[0]) / scale, (y - origin[1]) / scale]
                        for x, y in list(poly.exterior.coords)[:-1]]
            int_rings = [[[(x - origin[0]) / scale, (y - origin[1]) / scale]
                          for x, y in list(hole.coords)[:-1]]
                         for hole in poly.interiors]

            components_data.append({
                'component_id': list(key),
                'floor': floor_rel,
                'ceiling': ceil_rel,
                'floor_texture': sec.floor_flat,
                'ceiling_texture': sec.ceil_flat,
                'light_level': sec.light_level,
                'exterior_ring': ext_ring,
                'interior_rings': int_rings,
            })

        if room_polys:
            from shapely import union_all
            room_union = union_all(room_polys)
            bounds = room_union.bounds  # (minx, miny, maxx, maxy)
            center_x = (bounds[0] + bounds[2]) / 2.0
            center_y = (bounds[1] + bounds[3]) / 2.0
            norm_bounds = [
                (bounds[0] - origin[0]) / scale,
                (bounds[1] - origin[1]) / scale,
                (bounds[2] - origin[0]) / scale,
                (bounds[3] - origin[1]) / scale,
            ]
            norm_center = [
                (center_x - origin[0]) / scale,
                (center_y - origin[1]) / scale,
            ]
        else:
            norm_bounds = [0.0, 0.0, 0.0, 0.0]
            norm_center = [0.0, 0.0]

        # Calculate relative rings centered at room center for outline learning
        for comp in components_data:
            comp['relative_exterior_ring'] = [
                [p[0] - norm_center[0], p[1] - norm_center[1]]
                for p in comp['exterior_ring']
            ]
            comp['relative_interior_rings'] = [
                [[p[0] - norm_center[0], p[1] - norm_center[1]] for p in hole]
                for hole in comp['interior_rings']
            ]

        room_placements.append({
            'room_id': room.id,
            'center': norm_center,
            'bounds': norm_bounds,
            'floor_base': min(room_floors) if room_floors else 0.0,
            'ceiling_base': max(room_ceils) if room_ceils else 128.0,
            'is_multi_height': len(set(room_floors)) > 1 or len(set(room_ceils)) > 1,
            'distinct_floors': sorted(list(set(room_floors))),
            'distinct_ceilings': sorted(list(set(room_ceils))),
        })

        room_interiors.append({
            'room_id': room.id,
            'component_count': len(components_data),
            'components': components_data,
        })

    # 3. Connection portals & shared boundary openings
    audit_conns_map = {}
    if audit_data and 'connections' in audit_data:
        for c in audit_data['connections']:
            audit_conns_map[(c['a'], c['b'])] = c
            audit_conns_map[(c['b'], c['a'])] = c

    openings_data = []
    unsupported_edges = []
    one_way_edges = []

    for conn in graph.connections:
        audit_conn = audit_conns_map.get((conn.a, conn.b))
        status = audit_conn['status'] if audit_conn else 'unknown'
        init_dirs = audit_conn.get('local_directions', {}).get('initial', []) if audit_conn else []
        pot_dirs = audit_conn.get('local_directions', {}).get('potential', []) if audit_conn else []

        # Check motion support
        is_supported = status in ('initial_local', 'conditional_local') and len(pot_dirs) > 0
        if not is_supported:
            unsupported_edges.append([conn.a, conn.b])
        elif len(pot_dirs) == 1:
            one_way_edges.append(pot_dirs[0])

        contiguous_runs = audit_conn.get('contiguous_runs', []) if audit_conn else []
        total_span_length = sum(float(r.get('length', 0.0)) for r in contiguous_runs)

        openings_data.append({
            'a': conn.a,
            'b': conn.b,
            'kind': conn.kind,
            'status': status,
            'initial_directions': init_dirs,
            'potential_directions': pot_dirs,
            'is_motion_supported': is_supported,
            'total_opening_width': total_span_length,
            'normalized_opening_width': total_span_length / scale,
        })

    # 4. Action scope & exclusions
    outside_audit = []
    if index_data and 'outside_local_audit' in index_data:
        outside_audit = index_data['outside_local_audit']

    is_clean = bool(
        index_data
        and index_data.get('strict_compiled_area_match')
        and index_data.get('raw_validation_errors') == 0
        and index_data.get('compiled_validation_errors') == 0
    )

    return {
        'schema': SCHEMA,
        'frame': {
            'origin': origin,
            'scale': scale,
            'floor_origin': floor_origin,
        },
        'topology': {
            'rooms': topology_rooms,
            'connections': topology_connections,
        },
        'geometry': {
            'placements': room_placements,
            'interiors': room_interiors,
            'openings': openings_data,
        },
        'exclusions': {
            'is_clean_geometry': is_clean,
            'unsupported_edges': unsupported_edges,
            'one_way_edges': one_way_edges,
            'outside_audit_actions': outside_audit,
            'outside_audit_count': len(outside_audit),
        },
        'limits': (
            'Topology-conditioned spatial target; explicit masks isolate '
            'unsupported local edges, one-way drops, and unhandled mechanics. '
            'Does not certify global action history.'
        ),
    }

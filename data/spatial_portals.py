"""Opt-in audits of experimental spatial evidence; never v3 training labels.

Results describe local crossings in two static geometries. The potential door
geometry pools optimistic events and is NOT an executable global action history.
Active teleport triggers remain active in both views.
"""
from __future__ import annotations

import math

from wad_parser.components import ComponentIndex
from wad_parser.geometry import sector_geometry
from wad_parser.traversal import line_component, potential_state

AUDIT_SCHEMA = 'spatial_portal_audit_v1'
EVIDENCE_SCHEMA = 'experimental_spatial_evidence_v1'


def contiguous_runs(doc, line_ids, *, _index=None):
    """Lengths of touching collinear runs, never sums of separate windows.

    Component pairs and action identity must agree before spans are merged.
    Width is authored span length, not a player clearance certificate.
    """
    index = _index or ComponentIndex(doc.vertices, sector_geometry(doc))
    lines = {line.id: line for line in doc.linedefs}
    groups = []
    for lid in sorted(set(line_ids)):
        line = lines[lid]
        a, b = (index.positions[v] for v in (line.v1, line.v2))
        length = math.dist(a, b)
        if length <= 1e-8:
            continue
        dx, dy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
        if dx < -1e-8 or (abs(dx) <= 1e-8 and dy < 0):
            dx, dy = -dx, -dy
        owners = tuple(sorted(_components(doc, line, index), key=str))
        identity = owners, line.special, line.tag
        group = next((g for g in groups if g['identity'] == identity
                      and abs(g['axis'][0] * dy - g['axis'][1] * dx) < 1e-8
                      and abs((a[0] - g['origin'][0]) * dy
                              - (a[1] - g['origin'][1]) * dx) < 1e-6), None)
        if group is None:
            group = {'identity': identity, 'axis': (dx, dy), 'origin': a, 'spans': []}
            groups.append(group)
        axis, origin = group['axis'], group['origin']
        interval = sorted(sum((p[k] - origin[k]) * axis[k] for k in (0, 1)) for p in (a, b))
        group['spans'].append((*interval, lid))
    result = []
    for group in groups:
        runs = []
        for lo, hi, lid in sorted(group['spans']):
            if runs and lo <= runs[-1][1] + 1e-6:
                runs[-1][1] = max(hi, runs[-1][1])
                runs[-1][2].append(lid)
            else:
                runs.append([lo, hi, [lid]])
        for lo, hi, lids in runs:
            result.append({'linedefs': sorted(lids), 'length': hi - lo})
    return result


def _components(doc, line, index):
    return tuple(line_component(line, doc.sidedefs[side].sector, index.positions,
                               doc.sectors, doc.sidedefs, sector_geometry(doc), index)
                 if 0 <= side < len(doc.sidedefs) else None
                 for side in (line.right_sidedef, line.left_sidedef))


def _intervals(geometry, mouth):
    from shapely import Point
    if geometry.geom_type in {'LineString', 'LinearRing'}:
        values = [mouth.project(Point(p), normalized=True)
                  for p in geometry.coords]
        return [[min(values), max(values)]] if geometry.length > 1e-8 else []
    return [span for part in getattr(geometry, 'geoms', []) for span in _intervals(part, mouth)]


def audit_portals(doc, target):
    """Check candidate identity, directed clearance, door interiors and actions.

    Unsupported mechanics and absent crossings remain visible. A failed local
    check does not establish impossibility under motion, alternate histories or
    engine behavior. Inputs are not modified. Only classic Doom is supported.
    """
    from shapely import LineString
    from wad_parser.actions import LOCK_COLORS
    from wad_parser.navigation import PlayerNavigation
    from wad_parser.progression import door_targets

    if target.get('schema') != EVIDENCE_SCHEMA:
        raise ValueError('unsupported spatial evidence schema')
    if doc.meta.format != 'doom':
        raise ValueError('spatial portal audit currently supports Doom format only')
    index = ComponentIndex(doc.vertices, sector_geometry(doc))
    lines = {line.id: line for line in doc.linedefs}
    owners = {}
    for room in target['rooms']:
        for part in room['components']:
            key = tuple(part['source_component'])
            if key in owners:
                raise ValueError(f'duplicate component ownership: {key}')
            owners[key] = room['id']
    origin, scale = target['frame']['origin'], target['frame']['scale']
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('invalid coordinate frame')
    initial = potential_state(doc, allow_doors=False)
    potential = potential_state(doc)
    states = {'initial': initial, 'potential': potential}
    navs, free_components = {}, {}
    for name, state in states.items():
        # Do not allow ordinary passage through a consumed teleport in the
        # same view as its active event. No action-history certificate implied.
        state = {**state, 'potential_teleport_consumption': False}
        nav = navs[name] = PlayerNavigation(doc, state=state)
        free_components[name] = {key: index.point(key[0], *polygon.representative_point().coords[0])
                                 for key, polygon in nav.regions.items()}

    def span_audit(span, expected):
        line = lines.get(span['linedef'])
        if line is None:
            raise ValueError('candidate references missing linedef')
        points = [[origin[k] + p[k] * scale for k in (0, 1)] for p in span['points']]
        actual = [index.positions[v] for v in (line.v1, line.v2)]
        if (len(points) != 2 or any(math.dist(a, b) > 1e-6 for a, b in zip(points, actual))
                or span['special'] != line.special or span['tag'] != line.tag):
            raise ValueError(f'stale candidate coordinates/action on linedef {line.id}')
        components = _components(doc, line, index)
        if set(components) != set(expected):
            raise ValueError(f'candidate component attribution mismatch on linedef {line.id}')
        mouth = LineString(actual)
        output = {'linedef': line.id, 'components': components, 'special': line.special,
                  'tag': line.tag, 'length': mouth.length, 'crossings': {}}
        for name, nav in navs.items():
            records = []
            for source, destination in nav.crossings.get(line.id, []):
                # Entry intervals give source-side centers. A descending drop
                # may land away from the mouth in the destination free region.
                entry = nav.regions[source].intersection(mouth)
                records.append({'source': free_components[name][source],
                                'destination': free_components[name][destination],
                                'source_free_region': source, 'destination_free_region': destination,
                                'entry_intervals': _intervals(entry, mouth)})
            output['crossings'][name] = records
        return output

    edges = []
    for edge in target['connections']:
        entry = {'a': edge['a'], 'b': edge['b'], 'kind': edge['kind']}
        edges.append(entry)
        if edge['kind'] == 'teleport':
            entry.update(status='remote', relation='not a shared boundary')
            continue
        directions = {name: set() for name in navs}
        direct = []
        for span in edge.get('authored_boundaries', []):
            components = _components(doc, lines[span['linedef']], index)
            if {owners.get(key) for key in components} != {edge['a'], edge['b']}:
                raise ValueError('candidate does not connect claimed rooms')
            checked = span_audit(span, components)
            direct.append(checked)
            for name, records in checked['crossings'].items():
                directions[name].update((owners[r['source']], owners[r['destination']]) for r in records)
        doors = []
        for door in edge.get('door_faces', []):
            component = tuple(door['source_component'])
            faces = {}
            for side in ('a', 'b'):
                faces[side] = []
                for span in door[side]:
                    components = _components(doc, lines[span['linedef']], index)
                    other = next((key for key in components if key != component), None)
                    if component not in components or owners.get(other) != edge[side]:
                        raise ValueError('door face does not connect claimed room/component')
                    faces[side].append(span_audit(span, (component, other)))
            joins = {}
            for name in navs:
                joins[name] = []
                for src, dst in (('a', 'b'), ('b', 'a')):
                    entering = [r for face in faces[src] for r in face['crossings'][name]
                                if r['destination'] == component and owners.get(r['source']) == edge[src]]
                    leaving = [r for face in faces[dst] for r in face['crossings'][name]
                               if r['source'] == component and owners.get(r['destination']) == edge[dst]]
                    common = sorted({r['destination_free_region'] for r in entering}
                                    & {r['source_free_region'] for r in leaving})
                    if common:
                        directions[name].add((edge[src], edge[dst]))
                        joins[name].append({'from': edge[src], 'to': edge[dst], 'door_free_regions': common})
            doors.append({'source_component': component, 'faces': faces, 'joins': joins})
        entry.update(authored_boundaries=direct, door_faces=doors,
                     contiguous_runs=contiguous_runs(doc, [r['linedef'] for r in direct], _index=index),
                     local_directions={name: sorted(value) for name, value in directions.items()})
        entry['status'] = ('initial_local' if directions['initial'] else
                           'conditional_local' if directions['potential'] else 'unsupported_or_unusable')
    actions, _ = door_targets(doc)
    return {'schema': AUDIT_SCHEMA, 'source_id': target.get('source_id'),
            'scope': 'local player-box crossings; potential door union is not a global action history',
            'player': {'radius': 16, 'height': 56}, 'connections': edges,
            'potential_keys': potential['keys'], 'potential_actions': potential['actions'],
            'door_activators': [{'linedef': line.id, 'special': line.special, 'tag': line.tag,
                                 'key': LOCK_COLORS.get(line.special), 'target_sectors': targets,
                                 'front_component': _components(doc, line, index)[0]}
                                for line, targets in actions],
            'authored_actions': [{'linedef': line.id, 'special': line.special, 'tag': line.tag}
                                 for line in doc.linedefs if line.special]}

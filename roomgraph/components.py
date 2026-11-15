"""Create a temporary component graph while retaining original sector IDs."""
from __future__ import annotations

from dataclasses import replace

from wad_parser.bsp_geometry import component_polygons, geometry_rings
from wad_parser.components import ComponentIndex
from wad_parser.geometry import sector_geometry
from wad_parser.traversal import line_component, TRAVERSAL_VERSION
from wad_parser.types import DerivedData


def split_components(doc, state):
    index = ComponentIndex(doc.vertices, sector_geometry(doc))
    sectors, regions, sources, ids, open_sectors = [], [], {}, {}, []
    for sector in doc.sectors:
        for shell, polygon in component_polygons(index.rings[sector.id]):
            new_id = len(sectors)
            key = (sector.id, shell)
            ids[key], sources[new_id] = new_id, key
            sectors.append(replace(sector, id=new_id))
            open_sectors.append(replace(state['sectors'][sector.id], id=new_id))
            regions.append(geometry_rings(polygon))
    sidedefs, lines, line_nodes, ghosts = [], [], {}, {}
    for line in doc.linedefs:
        side_ids, owners = [], []
        for side_id in (line.right_sidedef, line.left_sidedef):
            if not 0 <= side_id < len(doc.sidedefs):
                side_ids.append(-1)
                owners.append(None)
                continue
            source = doc.sidedefs[side_id]
            node = line_component(line, source.sector, index.positions, doc.sectors,
                                  doc.sidedefs, sector_geometry(doc), index)
            owner = ids.get(node, -1)
            if owner < 0 and 0 <= source.sector < len(doc.sectors):
                # A BSP can resolve a mislabeled raw boundary. Keep that
                # line's authored collision heights even when its named
                # sector has no spatial component beside this particular side.
                if source.sector not in ghosts:
                    ghost = len(sectors)
                    ghosts[source.sector] = ghost
                    sectors.append(replace(doc.sectors[source.sector], id=ghost))
                    open_sectors.append(replace(state['sectors'][source.sector], id=ghost))
                    regions.append([])
                owner = ghosts[source.sector]
            side_ids.append(len(sidedefs))
            # Packed sidedefs can be shared by distant islands. Clone each
            # line side instead of assigning one source sidedef two owners.
            sidedefs.append(replace(source, id=len(sidedefs), sector=owner))
            owners.append(owner if owner in sources else None)
        line_nodes[line.id] = owners
        lines.append(replace(line, right_sidedef=side_ids[0], left_sidedef=side_ids[1]))
    edges = []
    for line in lines:
        a, b = line_nodes[line.id]
        if a is not None and b is not None and a != b:
            edges.append({'a': a, 'b': b, 'kind': 'walk', 'linedef': line.id})
    by_line = {line.id: line for line in doc.linedefs}
    for edge in doc.derived.sector_graph.get('edges', []):
        if edge.get('kind') != 'teleport' or edge.get('linedef') not in by_line:
            continue
        line = by_line[edge['linedef']]
        source = line_component(line, edge['a'], index.positions, doc.sectors,
                                doc.sidedefs, sector_geometry(doc), index)
        if 'destination_point' in edge:
            destination = index.point(edge['b'], *edge['destination_point'])
        elif edge.get('destination_line') in by_line:
            destination = line_component(by_line[edge['destination_line']], edge['b'], index.positions,
                                         doc.sectors, doc.sidedefs, sector_geometry(doc), index)
        else:
            destination = None
        if source in ids and destination in ids:
            edges.append({**edge, 'a': ids[source], 'b': ids[destination]})
    virtual_state = {**state, 'sectors': open_sectors,
                     'distances': {(ids[node], 0): distance for node, distance in state['distances'].items()
                                   if node in ids},
                     'supported_crossings': [(ids[a], ids[b], line_id) for a, b, line_id
                                             in state.get('supported_crossings', []) if a in ids and b in ids]}
    derived = DerivedData(sector_regions=regions, sector_graph={'edges': edges},
                          traversal_version=TRAVERSAL_VERSION)
    virtual = replace(doc, sectors=sectors, sidedefs=sidedefs, linedefs=lines, derived=derived,
                      extra={**doc.extra, '_roomgraph_component_sources': sources})
    return virtual, virtual_state, sources

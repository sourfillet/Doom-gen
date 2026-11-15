import copy
import random

import pytest

from generation.baselines import procedural_graph, retrieval_graph, geometry_fingerprint
from roomgraph.types import Room, RoomGraph, Connection
from data.roomgraph_targets import graph_to_sequence
from tests.geometry_fixtures import two_rooms


def reference():
    return RoomGraph([Room(i, [], 0, 128, 512**2, (0, 0), (0, 0, 512, 512),
                           roles=['start'] if i == 0 else ['exit']) for i in range(3)],
                     [Connection(0, 1, 'door', 128), Connection(1, 2, 'corridor', 64)])


def test_procedural_baseline_matches_counts_and_preserves_constraints_and_rng():
    nx = pytest.importorskip('networkx')
    graphs = [reference()]
    original, rng = copy.deepcopy(graphs), random.getstate()
    graph = procedural_graph(graphs, rooms=12, edges=17, seed=7)
    assert len(graph.rooms) == 12 and len(graph.connections) == 17
    assert nx.check_planarity(nx.Graph((e.a, e.b) for e in graph.connections if e.kind != 'teleport'))[0]
    assert any('start' in r.roles for r in graph.rooms) and any('exit' in r.roles for r in graph.rooms)
    assert graph_to_sequence(graph) == graph_to_sequence(procedural_graph(graphs, rooms=12, edges=17, seed=7))
    assert graphs == original and random.getstate() == rng


def test_retrieval_baseline_is_an_independent_copy_of_nearest_size_reference():
    graphs = [reference()]
    graph = retrieval_graph(graphs, rooms=7, seed=0)
    assert graph == graphs[0] and graph is not graphs[0]
    graph.rooms[0].floor_h = 99
    assert graphs[0].rooms[0].floor_h == 0


def test_geometry_copy_probe_ignores_translation_ids_and_textures_but_sees_geometry():
    doc = two_rooms()
    fingerprint = geometry_fingerprint(doc)
    changed = copy.deepcopy(doc)
    for element in [*changed.vertices, *changed.things]:
        element.x += 100.5
        element.y -= 37.25
    changed.sectors[0].floor_flat = 'OTHER'
    changed.vertices.reverse()
    mapping = {v.id: i for i, v in enumerate(changed.vertices)}
    for i, vertex in enumerate(changed.vertices):
        vertex.id = i
    for line in changed.linedefs:
        line.v1, line.v2 = mapping[line.v1], mapping[line.v2]
    assert geometry_fingerprint(changed) == fingerprint
    changed.sectors[0].floor_height += 8
    assert geometry_fingerprint(changed)['geometry'] != fingerprint['geometry']
def test_referenced_geometry_fingerprint_ignores_inert_records_but_detects_changed_walls():
    import copy
    from generation.baselines import geometry_fingerprint, geometry_fingerprint_v2
    from tests.geometry_fixtures import two_rooms
    from wad_parser.types import Vertex
    source = two_rooms()
    changed = copy.deepcopy(source)
    changed.vertices.append(Vertex(len(changed.vertices), -10000, -10000))
    unused_sector = copy.deepcopy(source.sectors[0])
    unused_sector.id = len(changed.sectors); unused_sector.floor_height = -999
    changed.sectors.append(unused_sector)
    unused_side = copy.deepcopy(source.sidedefs[0])
    unused_side.id = len(changed.sidedefs); unused_side.sector = unused_sector.id
    changed.sidedefs.append(unused_side)
    assert geometry_fingerprint(source) != geometry_fingerprint(changed)
    assert geometry_fingerprint_v2(source) == geometry_fingerprint_v2(changed)
    changed.vertices[0].x -= 8
    assert geometry_fingerprint_v2(source)['geometry'] != geometry_fingerprint_v2(changed)['geometry']


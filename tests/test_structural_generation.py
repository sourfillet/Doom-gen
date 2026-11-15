import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parent)]
from geometry_fixtures import two_rooms
from data.rasterizer import rasterize
from data.roomgraph_targets import graph_to_sequence, sequence_to_graph
from generation.vectorizer import vectorize
from generation.validator import validate_document
from roomgraph.types import Room, RoomGraph, Connection
from roomgraph.assemble import assemble
from roomgraph.teleports import require_connections
from wad_parser.map_extractor import compute_derived
from wad_parser.traversal import distances_from_start


def _graph(kind='open', difference=0):
    rooms = [Room(i, [], 0 if i == 0 else difference, 128 if i == 0 else difference + 128,
                  512**2, (0, 0), (0, 0, 512, 512)) for i in range(2)]
    return RoomGraph(rooms=rooms, connections=[Connection(0, 1, kind, 192)])


def test_equal_height_sectors_and_solid_boundaries_survive_vectorization():
    doc = two_rooms()
    raster, _ = rasterize(doc, 64)
    decoded = vectorize(raster, coarse_factor=2)
    assert len(decoded.sectors) == 2
    assert any(line.left_sidedef >= 0 and not line.flags.impassable for line in decoded.linedefs)
    shared = next(line for line in doc.linedefs if line.left_sidedef >= 0)
    shared.flags.impassable = True
    decoded = vectorize(rasterize(doc, 64)[0], coarse_factor=2)
    assert any(line.left_sidedef >= 0 and line.flags.impassable for line in decoded.linedefs)
    json.dumps(decoded.to_dict())  # vectorizer must not leak NumPy scalar types


def test_outer_wall_pixels_do_not_invent_disconnected_parallel_slivers():
    from generation.pipeline import raster_to_document
    doc = two_rooms()
    decoded, _, _ = raster_to_document(rasterize(doc, 128)[0])
    assert len(decoded.sectors) == 2
    assert not validate_document(decoded, require_connected=True)['errors']


def test_raster_boundaries_are_independent_of_sector_numbering():
    doc = two_rooms()
    changed = copy.deepcopy(doc)
    changed.sectors.reverse()
    for index, sector in enumerate(changed.sectors):
        sector.id = index
    for side in changed.sidedefs:
        side.sector = 1 - side.sector
    changed.bounds, changed.derived = compute_derived(changed.vertices, changed.sectors,
        changed.linedefs, changed.sidedefs, changed.things)
    assert np.array_equal(rasterize(doc, 64)[0], rasterize(changed, 64)[0])


def test_graph_roundtrip_retains_loops_roles_and_calibrated_area():
    graph = _graph()
    graph.rooms[0].area = 1e6
    graph.rooms[0].roles = ['start']
    graph.rooms.append(Room(2, [], 0, 128, 512**2, (0, 0), (0, 0, 512, 512), roles=['exit', 'key']))
    graph.connections.extend([Connection(1, 2, 'door', 128), Connection(0, 2, 'open', 64)])
    rebuilt = sequence_to_graph(graph_to_sequence(graph))
    assert len(rebuilt.connections) == 3
    assert abs(rebuilt.rooms[0].area / 1e6 - 1) < 0.1
    assert any({'exit', 'key'} <= set(room.roles) for room in rebuilt.rooms)
    assert {connection.width for connection in rebuilt.connections} == {64, 128, 192}


def test_exact_graph_snapshot_does_not_quantize_geometry_or_drop_components():
    graph = _graph()
    graph.rooms[1].floor_h, graph.rooms[1].area = 37, 123456.75
    graph.rooms[1].polygons = [[(0.25, 0.5), (0.25, 88.5), (100.75, 88.5)]]
    graph.rooms[1].holes = [[(10.25, 10.5), (20.75, 10.5), (20.75, 20.5)]]
    graph.rooms[1].components = [(5, 2)]
    graph.excluded_components = [(8, 1)]
    assert RoomGraph.from_dict(json.loads(json.dumps(graph.to_dict()))) == graph


def test_graph_disconnected_components_are_never_silently_dropped():
    graph = _graph()
    graph.connections.clear()
    with pytest.raises(ValueError, match='disconnected'):
        graph_to_sequence(graph)


def test_grid_assembler_routes_cycle_around_existing_rooms():
    graph = _graph()
    graph.rooms.append(Room(2, [], 0, 128, 512**2, (0, 0), (0, 0, 512, 512)))
    graph.connections.extend([Connection(0, 2, 'open', 128), Connection(1, 2, 'open', 128)])
    document = assemble(graph)
    require_connections(document)
    assert len(document.extra['connection_realization']) == 3
    assert len(graph.rooms) == 3  # routing must not mutate the requested graph
    assert not validate_document(document, require_exit=False, require_connected=True)['errors']


def test_busy_hub_keeps_all_requested_connections():
    graph = _graph()
    for index in range(2, 8):
        graph.rooms.append(Room(index, [], 0, 128, 512**2, (0, 0), (0, 0, 512, 512)))
        graph.connections.append(Connection(0, index, 'corridor', 64))
    document = assemble(graph)
    require_connections(document)
    assert len(document.extra['connection_realization']) == 7
    assert not validate_document(document, require_exit=False, require_connected=True)['errors']


@pytest.mark.parametrize('spokes', [[1, 3, 5, 2, 4, 6], [6, 4, 2, 5, 3, 1]])
def test_junction_expansion_preserves_the_planarity_of_a_wheel(spokes):
    import networkx as nx
    from roomgraph.assemble import _fanout_hubs
    rooms = [Room(i, [], 0, 128, 512**2, (0, 0), (0, 0, 512, 512)) for i in range(7)]
    edges = [(0, other) for other in spokes] + [(i, i % 6 + 1) for i in range(1, 7)]
    graph = RoomGraph(rooms, [Connection(a, b, 'corridor', 128) for a, b in edges])
    original = copy.deepcopy(graph)
    expanded, requests, routes = _fanout_hubs(graph)
    physical = nx.Graph((edge.a, edge.b) for edge in expanded.connections)
    assert nx.check_planarity(physical)[0]
    assert max(dict(physical.degree()).values()) <= 4
    assert graph == original
    assert len(routes) == len(requests) == len(edges)
    for edge, route in zip(requests, routes):
        assert route[0][0] == edge.a and route[-1][-1] == edge.b
        assert all(physical.has_edge(a, b) for a, b in route)


def test_wheel_cycle_with_a_split_hub_realizes_all_twelve_connections():
    from tests.geometry_fixtures import wheel_graph
    graph = wheel_graph()
    document = assemble(graph)
    require_connections(document)
    assert len(document.extra['connection_realization']) == 12
    assert not validate_document(document, require_exit=False, require_connected=True)['errors']


def test_nonplanar_physical_graph_is_rejected_but_teleports_are_not_planar_edges():
    from roomgraph.assemble import _fanout_hubs
    rooms = [Room(i, [], 0, 128, 512**2, (0, 0), (0, 0, 512, 512)) for i in range(6)]
    graph = RoomGraph(rooms, [Connection(a, b, 'corridor', 128) for a in range(3) for b in range(3, 6)])
    with pytest.raises(ValueError, match='non-planar'):
        _fanout_hubs(graph)
    graph.connections[0].kind = 'teleport'
    expanded, requests, routes = _fanout_hubs(graph)
    assert len(requests) == len(routes) == 9
    assert expanded.connections[0].kind == 'teleport'


def test_router_retries_edge_orders_without_losing_connections_or_rng_state():
    import random
    rooms = [Room(i, [], 0, 128, 512**2, (0, 0), (0, 0, 512, 512)) for i in range(14)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6), (5, 6),
             (4, 7), (4, 8), (6, 8), (8, 9), (9, 10), (10, 11), (11, 12), (0, 13)]
    graph = RoomGraph(rooms, [Connection(a, b, 'corridor', 128) for a, b in edges])
    unchanged, rng = copy.deepcopy(graph), random.getstate()
    document = assemble(graph)
    require_connections(document)
    routing = document.extra['routing']
    assert routing['initial_missing_edges'] > 0
    assert 1 < routing['attempts'] <= routing['max_attempts'] == 12
    assert routing['remaining_missing_edges'] == 0
    assert len(document.extra['connection_realization']) == len(edges)
    assert not validate_document(document, require_exit=False, require_connected=True)['errors']
    assert graph == unchanged and random.getstate() == rng
    assert assemble(graph).to_dict() == document.to_dict()


def test_router_tries_bounded_alternative_room_placements():
    rooms = [Room(i, [], 0, 128, 512**2, (0, 0), (0, 0, 512, 512)) for i in range(10)]
    edges = [(0, 1), (1, 2), (1, 3), (0, 4), (4, 5), (4, 6), (6, 7),
             (6, 8), (2, 9), (5, 0), (4, 9), (9, 7)]
    graph = RoomGraph(rooms, [Connection(a, b, 'corridor', 128) for a, b in edges])
    unchanged = copy.deepcopy(graph)
    doc = assemble(graph)
    require_connections(doc)
    assert 1 < doc.extra['routing']['layout_attempts'] <= 16
    assert not validate_document(doc, require_exit=False, require_connected=True)['errors']
    assert graph == unchanged


def _diagonal_grid(extra=()):
    # A 3x3 room grid with one diagonal per square: planar, but the lane
    # router leaves connections unrouted for every placement and edge order.
    index = {(i, j): 3 * i + j for i in range(3) for j in range(3)}
    edges = ([(index[(i, j)], index[(i + 1, j)]) for i in range(2) for j in range(3)]
             + [(index[(i, j)], index[(i, j + 1)]) for i in range(3) for j in range(2)]
             + [(index[(i, j)], index[(i + 1, j + 1)]) for i in range(2) for j in range(2)])
    rooms = [Room(i, [], (i % 3) * 32, (i % 3) * 32 + 160, 384**2, (0, 0), (0, 0, 384, 384),
                  roles=['start'] if i == 0 else ['exit'] if i == 8 else []) for i in range(9)]
    connections = [Connection(a, b, 'door' if (a + b) % 4 == 0 else 'corridor', 128) for a, b in edges]
    return RoomGraph(rooms, connections + [Connection(a, b, kind, 128) for a, b, kind in extra])


def test_planar_fallback_realizes_a_graph_the_lane_router_cannot():
    graph = _diagonal_grid()
    unchanged = copy.deepcopy(graph)
    lanes = assemble(graph, planar_fallback=False)
    assert any(not edge['realized'] for edge in lanes.extra['connection_realization'])
    document = assemble(graph)
    require_connections(document)
    routing = document.extra['routing']
    assert routing['router'] == 'planar_visibility' and routing['planar_fallback'] == 'used'
    assert routing['lane_remaining_missing_edges'] > 0 and routing['remaining_missing_edges'] == 0
    records = document.extra['connection_realization']
    assert len(records) == len(graph.connections)
    for request, record in zip(graph.connections, records):
        assert record['route'][0][0] == request.a and record['route'][-1][-1] == request.b
    assert not document.derived.geometry_issues
    assert not validate_document(document, require_exit=False, require_connected=True)['errors']
    assert graph == unchanged
    assert assemble(graph).to_dict() == document.to_dict()


def test_lane_routable_graphs_do_not_change_with_the_planar_fallback():
    from tests.geometry_fixtures import wheel_graph
    for graph in (_graph('door', 64), wheel_graph()):
        document = assemble(graph)
        assert document.to_dict() == assemble(graph, planar_fallback=False).to_dict()
        assert 'router' not in document.extra['routing'] and 'planar_fallback' not in document.extra['routing']


@pytest.mark.parametrize('profile', ['classic', 'varied', 'compact', 'adjoining'])
def test_planar_fallback_keeps_parallel_edges_and_teleports_and_survives_serialization(tmp_path, profile):
    from generation.baselines import realize
    from generation.validator import require_valid
    from wad_parser.map_extractor import extract_all_maps
    from wad_parser.wad_reader import WadFile
    from wad_writer.wad_writer import write_wad
    graph = _diagonal_grid(extra=[(0, 1, 'open'), (2, 6, 'teleport')])
    doc = realize(graph, geometry_profile=profile, seed=5)
    assert doc.extra['routing']['router'] == 'planar_visibility'
    records = doc.extra['connection_realization']
    assert len(records) == len(graph.connections) and all(record['realized'] for record in records)
    parallel = [record for record in records if {record['a'], record['b']} == {0, 1}]
    assert len(parallel) == 2 and parallel[0]['route'] != parallel[1]['route']
    assert records[-1]['kind'] == 'teleport' and records[-1]['pad_sectors']
    path = tmp_path / 'planar.wad'
    write_wad([doc], path)
    require_valid(extract_all_maps(WadFile(str(path)))[0], require_connected=True)


def test_st_numbering_gives_every_inner_vertex_a_lower_and_higher_neighbour():
    import networkx as nx
    from roomgraph.assemble import _st_numbering
    graphs = [nx.wheel_graph(9), nx.grid_2d_graph(4, 5), nx.cycle_graph(7),
              nx.convert_node_labels_to_integers(nx.triangular_lattice_graph(3, 4))]
    for graph in graphs:
        graph = nx.convert_node_labels_to_integers(graph)
        assert nx.is_biconnected(graph)
        adjacency = {v: set(graph[v]) for v in graph}
        s = min(adjacency)
        t = min(adjacency[s])
        order = _st_numbering(adjacency, s, t)
        assert sorted(order) == sorted(adjacency) and order[0] == s and order[-1] == t
        number = {v: i for i, v in enumerate(order)}
        for v in order[1:-1]:
            assert min(number[w] for w in adjacency[v]) < number[v] < max(number[w] for w in adjacency[v])


def test_long_room_chain_is_compact_enough_for_binary_coordinates():
    from wad_writer.assembler import fit_coordinates
    rooms = [Room(i, [], (i % 2) * 256, (i % 2) * 256 + 128,
                  2048**2, (0, 0), (0, 0, 2048, 2048)) for i in range(48)]
    graph = RoomGraph(rooms, [Connection(i - 1, i, 'corridor', 64) for i in range(1, 48)])
    doc = assemble(graph)
    require_connections(doc)
    fit_coordinates(doc)
    assert max(v.x for v in doc.vertices) - min(v.x for v in doc.vertices) <= 65535
    assert max(v.y for v in doc.vertices) - min(v.y for v in doc.vertices) <= 65535


def test_real_shape_path_preserves_holes_through_segmentation_retrieval_and_assembly(tmp_path):
    from tests.geometry_fixtures import polygon_document, square
    from roomgraph.segment import segment
    from roomgraph.shapes import build_library, save_library, load_library, apply_shapes
    from roomgraph.assemble_real import assemble_real
    from wad_parser.geometry import sector_area_perimeter
    source = polygon_document([[square(0, 0, 512, 512), list(reversed(square(192, 192, 320, 320)))]], start=(64, 64))
    graph = segment(source)
    assert len(graph.rooms[0].holes) == 1
    library_path = tmp_path / 'shapes.json'
    save_library(build_library([source]), library_path)
    apply_shapes(graph, load_library(library_path), seed=0)
    assembled = assemble_real(graph, positions={0: (0, 0)})
    assert len(assembled.derived.sector_rings[0]) == 2
    assert sector_area_perimeter(assembled, 0)[0] == pytest.approx(512**2 - 128**2)
    assert not validate_document(assembled, require_exit=False, require_connected=True)['errors']


def test_merged_room_shape_is_the_union_of_its_sector_outlines():
    from roomgraph.segment import segment
    from roomgraph.assemble_real import assemble_real
    from wad_parser.geometry import sector_area_perimeter
    source = two_rooms()
    graph = segment(source)
    assert len(graph.rooms) == 1
    assembled = assemble_real(graph)
    assert sector_area_perimeter(assembled, 0)[0] == 256 * 128


def test_door_slabs_roundtrip_as_connections_without_inventing_extra_rooms():
    from roomgraph.segment import segment
    source = assemble(_graph('door'))
    rebuilt = segment(source, playable_only=True)
    assert len(rebuilt.rooms) == 2
    assert len(rebuilt.connections) == 1 and rebuilt.connections[0].kind == 'door'
    assert len(rebuilt.collapsed_door_sectors) == 1
    assert all(room.ceil_h - room.floor_h >= 56 for room in rebuilt.rooms)


@pytest.mark.parametrize('kind,difference', [('open', 128), ('open', 384), ('door', 0), ('door', 384), ('teleport', 0)])
def test_realized_stairs_doors_and_dedicated_teleports_are_connected(kind, difference):
    doc = assemble(_graph(kind, difference))
    require_connections(doc)
    assert set(distances_from_start(doc)) == {s.id for s in doc.sectors}
    assert not validate_document(doc, require_exit=False, require_connected=True)['errors']
    if kind == 'door':
        closed = [sector for sector in doc.sectors if sector.ceil_height == sector.floor_height]
        assert len(closed) == 1
        assert len([line for line in doc.linedefs if line.special == 1]) == 2
    if kind == 'teleport':
        assert len([thing for thing in doc.things if thing.type == 14]) == 2
        assert len([line for line in doc.linedefs if line.special == 97]) == 8


@pytest.mark.parametrize('side', [192, 256])
def test_small_start_room_keeps_space_for_teleport_pads(side):
    from roomgraph.teleports import add_teleport_pair
    from geometry_fixtures import polygon_document, square
    doc = polygon_document([[square(0, 0, side, side)],
                            [square(side * 2, 0, side * 3, side)]], start=(side / 2, side / 2))
    pads = add_teleport_pair(doc, 0, 1)
    assert len(pads) == 2
    assert set(distances_from_start(doc)) == {0, 1, *pads}
    assert not validate_document(doc, require_exit=False, require_connected=True)['errors']


def test_multiple_teleport_pads_do_not_overlap_in_a_compact_room():
    from roomgraph.teleports import add_teleport_pair
    from geometry_fixtures import polygon_document, square
    doc = polygon_document([[square(i * 384, 0, i * 384 + 256, 256)] for i in range(3)], start=(128, 128))
    add_teleport_pair(doc, 0, 1)
    add_teleport_pair(doc, 0, 2)
    assert not validate_document(doc, require_exit=False, require_connected=True)['errors']


def test_censored_prefixes_have_no_false_end_label():
    torch = pytest.importorskip('torch')
    from models.roomgen.model import RoomGraphRNN
    model = RoomGraphRNN(hidden=8)
    for parameter in model.parameters():
        parameter.data.zero_()
    sequence = graph_to_sequence(_graph())
    tag = torch.zeros(256)
    complete = model.loss(sequence, tag, terminal=True)
    censored = model.loss(sequence, tag, terminal=False)
    assert (complete - censored).item() == pytest.approx(math.log(2), abs=1e-6)


def test_stop_hazard_does_not_overweight_short_map_endings():
    torch = pytest.importorskip('torch')
    from models.roomgen.model import RoomGraphRNN
    model = RoomGraphRNN(hidden=8)
    for parameter in model.parameters():
        parameter.data.zero_()
    sequence = graph_to_sequence(_graph())
    # One one-room map and one three-room map have two stops and two
    # continuations. A constant hazard of 1/2 must have zero bias gradient.
    third = dict(sequence[-1], parent=1)
    loss = model.loss(sequence[:1], torch.zeros(256)) + model.loss(sequence + [third], torch.zeros(256))
    loss.backward()
    assert model.h_stop.bias.grad.item() == pytest.approx(0, abs=1e-7)


def test_stop_targets_choose_which_sequences_end():
    from models.roomgen.stopping import stop_labels
    assert stop_labels(True, 10, 128) == (True, True)
    assert stop_labels(False, 10, 128) == (False, True)
    assert stop_labels(True, 200, 128) == (False, True)
    assert stop_labels(False, 10, 128, 'complete') == (False, False)
    assert stop_labels(True, 10, 128, 'complete') == (True, True)
    assert stop_labels(False, 10, 128, 'reachable-ends') == (True, True)
    assert stop_labels(False, 200, 128, 'reachable-ends') == (False, True)
    assert stop_labels(False, 200, 0, 'reachable-ends') == (True, True)
    with pytest.raises(ValueError, match='stop_targets'):
        stop_labels(True, 1, 0, 'never')


def test_fragments_can_train_structure_without_stop_terms():
    torch = pytest.importorskip('torch')
    from models.roomgen.model import RoomGraphRNN
    model = RoomGraphRNN(hidden=8)
    for parameter in model.parameters():
        parameter.data.zero_()
    sequence = graph_to_sequence(_graph())
    tag = torch.zeros(256)
    censored = model.loss(sequence, tag, terminal=False)
    structure = model.loss(sequence, tag, terminal=False, stop=False)
    # Each non-root room carries one continue term of log 2 at a zero logit.
    assert (censored - structure).item() == pytest.approx((len(sequence) - 1) * math.log(2), abs=1e-6)
    structure.backward()
    assert model.h_stop.bias.grad is None or model.h_stop.bias.grad.abs().sum().item() == 0
    with pytest.raises(ValueError, match='terminal'):
        model.loss(sequence, tag, terminal=True, stop=False)


def test_sampler_masks_a_complete_five_room_physical_graph():
    torch = pytest.importorskip('torch')
    from models.roomgen.model import RoomGraphRNN
    from data.roomgraph_targets import KINDS
    model = RoomGraphRNN(hidden=8)
    for parameter in model.parameters():
        parameter.data.zero_()
    model.h_stop.bias.data.fill_(-100)
    for head in (model.h_kind, model.h_extra):
        head.bias.data.fill_(-100)
        head.bias.data[KINDS.index('open')] = 100
    torch.manual_seed(7)
    graph = sequence_to_graph(model.sample(torch.zeros(256), max_rooms=5))
    assert len(graph.rooms) == 5
    physical = [edge for edge in graph.connections if edge.kind != 'teleport']
    # All ten edges would make K5, which cannot be realized by ordinary 2D
    # corridors. A five-vertex planar graph has at most nine edges.
    assert len(physical) <= 9


def test_legacy_room_checkpoint_keeps_legacy_tensor_shapes(tmp_path):
    torch = pytest.importorskip('torch')
    from models.roomgen.model import RoomGraphRNN
    from models.conditioning import TagConditioner
    from generation.roomgen import load_roomgen_model
    model, conditioner = RoomGraphRNN(hidden=8, target_version=2), TagConditioner()
    path = tmp_path / 'legacy.pt'
    torch.save({'model': model.state_dict(), 'conditioner': conditioner.state_dict(),
                'tag_to_index': {}, 'config': {'hidden': 8, 'target_version': 2}}, path)
    loaded, _ = load_roomgen_model(path)
    assert loaded.target_version == 2
    assert loaded.gru.input_size == 32 * 5


def test_failed_node_build_preserves_previous_output(tmp_path, monkeypatch):
    from generation.pipeline import documents_to_wad
    import generation.pipeline as pipeline
    def fail(_):
        raise ValueError('node builder failed')
    monkeypatch.setattr(pipeline, 'build_nodes', fail)
    output = tmp_path / 'existing.wad'
    output.write_bytes(b'previous output')
    with pytest.raises(ValueError, match='node builder failed'):
        documents_to_wad([two_rooms()], output)
    assert output.read_bytes() == b'previous output'


@pytest.mark.parametrize('playable', [False, True])
def test_node_builder_cannot_hide_corrupt_bsp_behind_valid_authoring_geometry(tmp_path, monkeypatch, playable):
    from generation.pipeline import documents_to_wad
    from generation.progression import ensure_exit
    from tests.wad_builder import _assemble_wad, build_two_sector_wad_with_nodes
    from wad_parser.wad_reader import WadFile
    import generation.pipeline as pipeline
    fixture = tmp_path / 'fixture.wad'
    fixture.write_bytes(build_two_sector_wad_with_nodes())
    lumps = WadFile(str(fixture)).lumps
    broken = _assemble_wad([(lump.name, b'broken' if lump.name == 'NODES' else lump.data) for lump in lumps])
    def corrupt(path):
        path.write_bytes(broken)
        return True
    monkeypatch.setattr(pipeline, 'build_nodes', corrupt)
    doc = two_rooms()
    ensure_exit(doc)
    output = tmp_path / 'existing.wad'
    output.write_bytes(b'previous output')
    with pytest.raises(ValueError, match='node builder produced invalid BSP'):
        documents_to_wad([doc], output, require_playable=playable)
    assert output.read_bytes() == b'previous output'


def test_node_builder_cannot_silently_drop_the_requested_maps(tmp_path, monkeypatch):
    from generation.pipeline import documents_to_wad
    import generation.pipeline as pipeline
    def corrupt(path):
        path.write_bytes(b'PWAD' + b'\0' * 8)
        return True
    monkeypatch.setattr(pipeline, 'build_nodes', corrupt)
    output = tmp_path / 'existing.wad'
    output.write_bytes(b'previous output')
    from generation.progression import ensure_exit
    doc = two_rooms()
    ensure_exit(doc)
    with pytest.raises(ValueError, match='no maps in compiled output'):
        documents_to_wad([doc], output, require_playable=True)
    assert output.read_bytes() == b'previous output'


def test_coordinate_translation_preserves_relative_geometry_and_things():
    from wad_writer.assembler import fit_coordinates
    doc = two_rooms()
    for point in doc.vertices + doc.things:
        point.x += 40000
    old_relative = doc.vertices[1].x - doc.things[0].x
    assert fit_coordinates(doc)
    assert doc.vertices[1].x - doc.things[0].x == old_relative
    assert min(v.x for v in doc.vertices) >= -32768
    assert max(v.x for v in doc.vertices) <= 32767

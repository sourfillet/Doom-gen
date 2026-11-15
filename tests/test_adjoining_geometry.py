"""Adjoining rooms retain requested routes and real solid wall thickness."""
from copy import deepcopy
import random

import pytest

from generation.baselines import geometry_fingerprint, realize
from roomgraph.adjoining import adjoining_rects
from roomgraph.assemble import assemble
from roomgraph.types import Connection, Room, RoomGraph
from tests.geometry_fixtures import room_pair
from wad_parser.map_extractor import extract_all_maps
from wad_parser.wad_reader import WadFile
from wad_writer.wad_writer import write_wad


@pytest.mark.parametrize('kind,rise,band', [('open', 0, 16), ('open', 24, 16),
                                         ('corridor', 0, 64), ('corridor', -24, 64),
                                         ('door', 0, 32)])
def test_short_thresholds_survive_binary_round_trip(tmp_path, kind, rise, band):
    from generation.validator import require_valid
    graph = room_pair(kind, rise, 512)
    before = deepcopy(graph)
    doc = realize(graph, geometry_profile='adjoining', seed=3)
    changes = doc.extra['geometry_variation']['adjacency']
    change, = changes['connections']
    assert change['wall_band'] == band < change['previous_gap']
    a, b = (changes['realized_bounds'][str(i)] for i in (0, 1))
    assert b[0] - a[2] == band
    assert doc.extra['connection_realization'][0]['realized']
    assert [(s.floor_height, s.ceil_height) for s in doc.sectors[:2]] == [(0, 128), (rise, rise + 128)]
    if kind == 'door':
        assert sum(line.special == 1 for line in doc.linedefs) == 2
        assert sum(s.floor_height == s.ceil_height for s in doc.sectors) == 1
    path = write_wad([doc], tmp_path / 'adjoining.wad')
    restored, = extract_all_maps(WadFile(str(path)))
    assert not require_valid(restored, require_connected=True)['errors']
    assert graph == before


@pytest.mark.parametrize('kind,rise', [('open', 128), ('door', 24), ('teleport', 128)])
def test_stairs_and_teleports_retain_the_existing_realization(kind, rise):
    graph = room_pair(kind, rise, 512)
    compact = realize(graph, geometry_profile='compact', seed=3)
    adjoining = realize(graph, geometry_profile='adjoining', seed=3)
    assert adjoining.extra['geometry_variation']['adjacency']['connections'] == []
    assert geometry_fingerprint(adjoining) == geometry_fingerprint(compact)
    assert adjoining.extra['connection_realization'] == compact.extra['connection_realization']


def test_growth_rejects_an_unrequested_room_contact_without_changing_input():
    rooms = [Room(i, [], 0, 128, 512**2, (0, 0), (0, 0, 512, 512)) for i in range(3)]
    graph = RoomGraph(rooms, [Connection(0, 1, 'open', 128)])
    placed = {0: (0, 0), 1: (2, 0), 2: (1, 1)}
    rects = {0: (-256, -256, 256, 256), 1: (512, -256, 1024, 256),
             2: (288, 240, 480, 432)}
    before = deepcopy(rects)
    result, joined, report = adjoining_rects(graph, placed, rects, seed=3)
    assert not joined and result == before == rects
    assert report['rejected'][0]['reason'] == 'room or transit clearance'


def test_growth_and_geometry_are_seeded_without_changing_global_rng_or_graph():
    graph = room_pair('open', 0, 512)
    before, rng = deepcopy(graph), random.getstate()
    first = assemble(graph, geometry_profile='adjoining', seed=3)
    repeat = assemble(graph, geometry_profile='adjoining', seed=3)
    assert first.to_dict() == repeat.to_dict()
    assert graph == before and random.getstate() == rng
    for rid, bounds in first.extra['geometry_variation']['adjacency']['realized_bounds'].items():
        original = first.extra['geometry_variation']['adjacency']['requested_bounds'][rid]
        assert bounds[2] - bounds[0] <= original[2] - original[0] + 768


def test_low_shared_headroom_keeps_the_existing_bridge():
    graph = room_pair('open', 24, 512)
    graph.rooms[0].ceil_h = 80
    graph.rooms[1].ceil_h = 104
    doc = assemble(graph, geometry_profile='adjoining', seed=3)
    assert doc.extra['geometry_variation']['adjacency']['connections'] == []

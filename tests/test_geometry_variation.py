import copy
import random

import pytest

from generation.baselines import geometry_fingerprint, realize
from roomgraph.assemble import assemble
from roomgraph.teleports import require_connections
from roomgraph.types import Connection, Room, RoomGraph
from roomgraph.variation import axis_positions
from tests.geometry_fixtures import wheel_graph, room_pair
from wad_parser.map_extractor import extract_all_maps
from wad_parser.wad_reader import WadFile
from wad_writer.wad_writer import write_wad


@pytest.mark.parametrize('kind,difference,side', [
    ('open', 0, 512), ('open', 128, 512), ('door', 0, 512), ('door', 384, 512),
    ('teleport', 128, 512), ('open', 0, 192), ('door', 128, 192), ('teleport', 0, 192),
])
@pytest.mark.parametrize('profile', ['varied', 'compact'])
def test_varied_rooms_keep_physical_routes_and_survive_serialization(tmp_path, kind, difference, side, profile):
    from generation.validator import require_valid
    graph = room_pair(kind, difference, side)
    before = copy.deepcopy(graph)
    doc = realize(graph, geometry_profile=profile, seed=19)
    require_connections(doc)
    path = tmp_path / 'varied.wad'
    write_wad([doc], path)
    restored = extract_all_maps(WadFile(str(path)))[0]
    require_valid(restored, require_connected=True)
    assert graph == before
    assert doc.extra['geometry_profile'] == profile
    assert doc.extra['connection_realization'][0]['realized']


@pytest.mark.parametrize('profile', ['varied', 'compact'])
def test_variation_is_seeded_without_changing_graph_routes_or_global_rng(profile):
    graph = wheel_graph()
    before, state = copy.deepcopy(graph), random.getstate()
    classic = assemble(graph)
    first = realize(graph, geometry_profile=profile, seed=3)
    repeat = realize(graph, geometry_profile=profile, seed=3)
    other = realize(graph, geometry_profile=profile, seed=4)
    assert first.to_dict() == repeat.to_dict()
    assert geometry_fingerprint(first) != geometry_fingerprint(other)
    assert first.extra['routing'] == classic.extra['routing']
    assert first.extra['connection_realization'] == classic.extra['connection_realization']
    assert first.extra['geometry_variation']['clipped_rooms'] >= 5
    assert first.extra['geometry_variation']['shifted_openings'] > 0
    assert any(first.vertices[line.v1].x != first.vertices[line.v2].x
               and first.vertices[line.v1].y != first.vertices[line.v2].y
               for line in first.linedefs if line.left_sidedef < 0)
    assert graph == before and random.getstate() == state


def test_default_geometry_remains_classic_and_independent_of_geometry_seed():
    graph = room_pair('door', 128, 512)
    default = assemble(graph)
    explicit = assemble(graph, geometry_profile='classic', seed=127)
    assert geometry_fingerprint(default) == geometry_fingerprint(explicit)
    assert default.extra['geometry_variation']['clipped_rooms'] == 0
    assert default.extra['geometry_variation']['shifted_openings'] == 0
    with pytest.raises(ValueError, match='geometry profile'):
        assemble(graph, geometry_profile='unknown')


def test_uneven_grid_only_expands_intervals_and_respects_coordinate_budget():
    cells = [-4, -2, 0, 1, 3]
    positions = axis_positions(cells, 768, 512, random.Random(3))
    gaps = [(positions[b] - positions[a]) / (b - a) for a, b in zip(cells, cells[1:])]
    assert min(gaps) > 768 and len(set(gaps)) > 1
    # Near the signed-coordinate span limit, expansion must use the remaining
    # budget rather than turning a fitting map into an unpublishable one.
    positions = axis_positions([0, 10], 6500, 512, random.Random(3))
    assert 65000 <= positions[10] - positions[0] <= 65534 - 512


@pytest.mark.parametrize('profile', ['varied', 'compact'])
def test_varied_large_rooms_fit_binary_coordinates(profile):
    from wad_writer.assembler import fit_coordinates
    rooms = [Room(i, [], (i % 2) * 256, (i % 2) * 256 + 128, 2048**2,
                  (0, 0), (0, 0, 2048, 2048)) for i in range(48)]
    graph = RoomGraph(rooms, [Connection(i - 1, i, 'corridor', 64) for i in range(1, 48)])
    doc = assemble(graph, geometry_profile=profile, seed=5)
    require_connections(doc)
    fit_coordinates(doc)
    assert all(-32768 <= v.x <= 32767 and -32768 <= v.y <= 32767 for v in doc.vertices)


@pytest.mark.parametrize('profile', ['varied', 'compact'])
def test_varied_profile_does_not_silently_use_the_source_outline_assembler(profile):
    from generation.roomgen import generate
    with pytest.raises(ValueError, match='omit shape_library'):
        generate('unused.pt', {}, shape_library={}, geometry_profile=profile)


def test_compact_spacing_removes_empty_transit_without_shrinking_rooms_or_losing_routes():
    graph = wheel_graph()
    classic = realize(graph)
    compact = realize(graph, geometry_profile='compact', seed=3)
    classic_span = max(classic.bounds.max_x - classic.bounds.min_x,
                       classic.bounds.max_y - classic.bounds.min_y)
    compact_span = max(compact.bounds.max_x - compact.bounds.min_x,
                       compact.bounds.max_y - compact.bounds.min_y)
    assert compact_span < .75 * classic_span
    assert compact.extra['connection_realization'] == classic.extra['connection_realization']
    assert compact.extra['room_sectors'] == classic.extra['room_sectors']
    assert [(s.floor_height, s.ceil_height) for s in compact.sectors] == [
        (s.floor_height, s.ceil_height) for s in classic.sectors]


def test_compact_spacing_reserves_nonlocal_stair_runs_and_largest_neighboring_extents():
    from roomgraph.variation import compact_axis_positions
    # A stair spanning several lanes must reserve its entire run; a large
    # room on another row still prevents adjacent columns from overlapping.
    half = {-4: 96, 0: 512, 7: 96, 11: 96}
    positions = compact_axis_positions(half, [(-4, 7, 1800)], random.Random(3))
    assert positions[7] - positions[-4] >= 1800
    for a, b in zip(half, list(half)[1:]):
        assert positions[b] - positions[a] >= half[a] + half[b] + 192
    assert positions[11] - positions[7] < 512

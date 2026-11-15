import math

import pytest
from shapely.geometry import Polygon

from roomgraph.assemble_real import _poly_centroid, assemble_real
from roomgraph.layout import _radius, _separate, radial_layout
from roomgraph.teleports import require_connections
from roomgraph.types import Connection, Room, RoomGraph


def _room(index):
    # Its area centroid is not the bounding-box midpoint. The old half-diagonal
    # circle misses the far vertex when assembly translates this actual shape.
    ring = [(0, 0), (0, 512), (64, 512), (512, 0)]
    return Room(index, [], 0, 128, Polygon(ring).area, (0, 0),
                (0, 0, 512, 512), polygons=[ring])


def test_layout_radius_encloses_the_polygon_around_its_assembly_center():
    room = _room(0)
    center = _poly_centroid(room.polygons[0])
    radius = _radius(room)
    assert all(math.dist(center, p) <= radius + 1e-6 for p in room.polygons[0])


@pytest.mark.parametrize('count', [2, 5])
def test_coincident_rooms_separate_deterministically(count):
    positions = {i: [0.0, 0.0] for i in range(count)}
    radii = {i: 128.0 for i in range(count)}
    _separate(positions, radii)
    repeat = {i: [0.0, 0.0] for i in range(count)}
    _separate(repeat, radii)
    assert repeat == positions
    assert all(math.dist(positions[a], positions[b]) > radii[a] + radii[b]
               for a in positions for b in positions if a < b)


def test_separation_exhaustion_rejects_overlapping_rooms():
    with pytest.raises(ValueError, match='separation'):
        _separate({0: [0.0, 0.0], 1: [32.0, 0.0]}, {0: 128.0, 1: 128.0}, iterations=0)


def test_disconnected_forest_layout_places_every_room_without_overlaps():
    from shapely.affinity import translate
    graph = RoomGraph([_room(i) for i in range(5)],
                      [Connection(0, 1, 'corridor', 64), Connection(2, 3, 'corridor', 64)])
    positions = radial_layout(graph, seed=12)
    assert set(positions) == set(range(5))
    outlines = []
    for room in graph.rooms:
        cx, cy = _poly_centroid(room.polygons[0])
        px, py = positions[room.id]
        outlines.append(translate(Polygon(room.polygons[0]), px - cx, py - cy))
    assert all(a.intersection(b).area < 1e-6 for i, a in enumerate(outlines) for b in outlines[i + 1:])
    doc = assemble_real(graph, positions=positions)
    require_connections(doc)
    assert set(doc.extra['room_sectors']) == set(positions)

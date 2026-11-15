"""Playable arenas and keyed room graphs shared by gameplay and engine tests."""
from generation.progression import ensure_exit
from roomgraph.assemble import assemble
from roomgraph.types import Connection, Room, RoomGraph
from tests.geometry_fixtures import polygon_document, square, two_rooms


def playable(name='MAP01'):
    doc = two_rooms()
    doc.meta.map_name = name
    ensure_exit(doc)
    return doc


def arena():
    doc = polygon_document([[square(0, 0, 1024, 1024)],
                             [square(1024, 0, 2048, 1024)],
                             [square(2048, 0, 3072, 1024)]], start=(96, 512))
    ensure_exit(doc)
    return doc


def terrace_arena(side=1024, headroom=128):
    doc = polygon_document([[square(0, 0, side, side)],
                            [square(side, 0, 2 * side, side)],
                            [square(2 * side, 0, 3 * side, side)]], start=(96, side / 2))
    for sector in doc.sectors:
        sector.ceil_height = headroom
    ensure_exit(doc)
    return doc


def keyed_arena(*, bypass=False):
    rooms = [Room(i, [], 0, 128, 512**2, (0, 0), (0, 0, 512, 512),
                  roles=[['start'], ['key'], ['exit']][i]) for i in range(3)]
    edges = [Connection(0, 1, 'open', 128), Connection(1, 2, 'door', 128)]
    if bypass:
        edges.append(Connection(0, 2, 'corridor', 128))
    doc = assemble(RoomGraph(rooms, edges), geometry_profile='varied', seed=3)
    ensure_exit(doc)
    return doc


def keyed_branch(profile='adjoining', *, door=True):
    from generation.baselines import realize
    rooms = [Room(i, [], 0, 128, 512**2, (0, 0), (0, 0, 512, 512),
                  roles=['start'] if i == 0 else []) for i in range(4)]
    graph = RoomGraph(rooms, [Connection(0, 1, 'open', 128), Connection(1, 2, 'open', 128),
                              Connection(1, 3, 'door' if door else 'open', 128)])
    return realize(graph, geometry_profile=profile, seed=2)

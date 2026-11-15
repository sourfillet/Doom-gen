"""Clockwise Doom sectors with shared boundaries for geometry regressions."""
from wad_parser.map_extractor import compute_derived
from wad_parser.types import (
    MapDocument, MapMeta, Vertex, Sector, Sidedef, Linedef, LinedefFlags,
    Thing, ThingFlags, resolve_doomed,
)


def square(x0, y0, x1, y1):
    return [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]


def polygon_document(regions, start=None):
    vertices, sectors, sidedefs, linedefs = [], [], [], []
    vertex_ids, edges = {}, {}
    def vid(point):
        if point not in vertex_ids:
            vertex_ids[point] = len(vertices)
            vertices.append(Vertex(id=len(vertices), x=point[0], y=point[1]))
        return vertex_ids[point]
    for sid, rings in enumerate(regions):
        sectors.append(Sector(sid, 0, 128, "FLOOR4_8", "CEIL3_5", 160, 0, 0))
        for ring in rings:
            for a, b in zip(ring, ring[1:] + ring[:1]):
                v1, v2 = vid(a), vid(b)
                sd = len(sidedefs)
                sidedefs.append(Sidedef(sd, 0, 0, "-", "-", "STARTAN2", sid))
                if (v2, v1) in edges:
                    line = linedefs[edges[(v2, v1)]]
                    line.left_sidedef = sd
                    line.flags.two_sided = True
                    line.flags.impassable = False
                    for side in (line.right_sidedef, sd):
                        sidedefs[side].mid_tex = "-"
                        sidedefs[side].upper_tex = sidedefs[side].lower_tex = "STARTAN2"
                else:
                    edges[(v1, v2)] = len(linedefs)
                    linedefs.append(Linedef(len(linedefs), v1, v2,
                                            LinedefFlags(impassable=True), 0, 0, sd, -1))
    things = []
    if start:
        name, category, _ = resolve_doomed(1)
        things.append(Thing(0, *start, 0, 1, name, category,
                            ThingFlags(skill_1_2=True, skill_3=True, skill_4_5=True)))
    bounds, derived = compute_derived(vertices, sectors, linedefs, sidedefs, things)
    return MapDocument("1.0", MapMeta("MAP01", "<fixture>", "fixture", "doom"),
                       bounds, vertices, sectors, linedefs, sidedefs, things, derived)


def two_rooms():
    return polygon_document([[square(0, 0, 128, 128)],
                             [square(128, 0, 256, 128)]], start=(64, 64))


def decorative_start():
    diamond = [(256, 240), (240, 256), (256, 272), (272, 256)]
    return polygon_document([[square(0, 0, 512, 512), list(reversed(diamond))],
                             [diamond]], start=(256, 256))


def two_room_graph():
    """Start and exit rooms joined by one door, matching two_rooms()' sectors."""
    from roomgraph.types import Room, RoomGraph, Connection
    rooms = [
        Room(0, [0], 0, 128, 512**2, (0, 0), (0, 0, 512, 512), roles=['start']),
        Room(1, [1], 0, 128, 512**2, (512, 0), (512, 0, 1024, 512), roles=['exit']),
    ]
    return RoomGraph(rooms, [Connection(0, 1, 'door', 64)])


def wheel_graph():
    """Planar wheel whose input spoke order previously broke hub expansion."""
    from roomgraph.types import Room, RoomGraph, Connection
    rooms = [Room(i, [], 0, 128, 512**2, (0, 0), (0, 0, 512, 512)) for i in range(7)]
    edges = [(0, other) for other in [1, 3, 5, 2, 4, 6]]
    edges += [(i, i % 6 + 1) for i in range(1, 7)]
    return RoomGraph(rooms, [Connection(a, b, 'corridor', 128) for a, b in edges])


def room_pair(kind, difference, side):
    from roomgraph.types import Room, RoomGraph, Connection
    rooms = [Room(i, [], i * difference, i * difference + 128, side**2,
                  (0, 0), (0, 0, side, side), roles=['start'] if i == 0 else ['exit'])
             for i in range(2)]
    return RoomGraph(rooms, [Connection(0, 1, kind, 160)])


def boundary_start():
    doc = two_rooms()
    doc.things[0].x = 128
    doc.bounds, doc.derived = compute_derived(doc.vertices, doc.sectors, doc.linedefs, doc.sidedefs, doc.things)
    return doc


def neck_map(width):
    low, high = 64 - width / 2, 64 + width / 2
    ring = [(0, 0), (0, 128), (128, 128), (128, high), (256, high), (256, 128),
            (384, 128), (384, 0), (256, 0), (256, low), (128, low), (128, 0)]
    doc = polygon_document([[ring]], start=(64, 64))
    exit_line = next(line for line in doc.linedefs if doc.vertices[line.v1].x == doc.vertices[line.v2].x == 384)
    exit_line.special = 11
    return doc, exit_line


def repair_arena():
    return polygon_document([[square(0, 0, 256, 256)],
                             [square(256, 0, 512, 256)],
                             [square(1024, 0, 1280, 256)],
                             [square(1280, 0, 1536, 256)]], start=(128, 128))

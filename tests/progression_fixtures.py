"""Small key/teleport progression puzzles with explicit physical histories."""
from tests.geometry_fixtures import polygon_document, square
from wad_parser.map_extractor import compute_derived
from wad_parser.types import Thing, ThingFlags, resolve_doomed, resolve_linedef_special


def one_shot_revisit_puzzle(special=39):
    # A -> B teleport, B -> A return. B's yellow key opens A's blue-key
    # branch; that blue key requires a second visit to B for the red exit key.
    regions = [[square(0, 0, 128, 128)], [square(128, 0, 256, 128)],
               [square(512, 0, 640, 128)], [square(640, 0, 672, 128)],
               [square(672, 0, 800, 128)], [square(512, 128, 640, 256)],
               [square(0, 128, 128, 160)], [square(0, 160, 128, 288)],
               [square(-32, 0, 0, 128)], [square(-160, 0, -32, 128)]]
    doc = polygon_document(regions, start=(64, 64))
    def portal(front, back):
        return next(line for line in doc.linedefs if line.left_sidedef >= 0
                    and doc.sidedefs[line.right_sidedef].sector == front
                    and doc.sidedefs[line.left_sidedef].sector == back)
    first, second = portal(0, 1), portal(2, 5)
    first.special, first.tag, doc.sectors[2].tag = special, 7, 7
    second.special, second.tag, doc.sectors[0].tag = 97, 8, 8
    for front, door, color in [(2, 3, 32), (0, 6, 34), (0, 8, 33)]:
        doc.sectors[door].ceil_height = 0
        portal(front, door).special = color
    exit_line = next(line for line in doc.linedefs if line.left_sidedef < 0
                     and doc.sidedefs[line.right_sidedef].sector == 9)
    exit_line.special = 11
    for x, y, kind in [(64, 64, 14), (576, 64, 14),
                       (576, 64, 6), (64, 224, 5), (736, 64, 13)]:
        name, category, unresolved = resolve_doomed(kind)
        doc.things.append(Thing(len(doc.things), x, y, 0, kind, name, category,
                                ThingFlags(skill_1_2=True, skill_3=True, skill_4_5=True), unresolved=unresolved))
    for line in doc.linedefs:
        line.special_name, line.trigger_type = resolve_linedef_special(line.special)
    doc.bounds, doc.derived = compute_derived(doc.vertices, doc.sectors, doc.linedefs, doc.sidedefs, doc.things)
    return doc


def door_document(special=26):
    doc = polygon_document([[square(0, 0, 128, 128)], [square(128, 0, 160, 128)],
                             [square(160, 0, 288, 128)]], start=(64, 64))
    doc.sectors[1].ceil_height = 0
    door = next(line for line in doc.linedefs if line.left_sidedef >= 0
                and doc.sidedefs[line.right_sidedef].sector == 0)
    door.special = special
    return doc


def add_key(doc, kind=5, x=96, *, active=True):
    name, category, _ = resolve_doomed(kind)
    doc.things.append(Thing(len(doc.things), x, 96, 0, kind, name, category,
                            ThingFlags(skill_3=active, skill_1_2=not active)))


def add_exit(doc, sid):
    line = next(line for line in doc.linedefs if line.left_sidedef < 0
                and doc.sidedefs[line.right_sidedef].sector == sid
                and doc.vertices[line.v1].x == doc.vertices[line.v2].x
                == max(v.x for v in doc.vertices))
    line.special = 11


def branch_document(*, return_from_key=False):
    doc = polygon_document([[square(128, 128, 256, 256)], [square(0, 128, 128, 256)],
                            [square(256, 128, 384, 256)], [square(384, 128, 416, 256)],
                            [square(416, 128, 544, 256)]], start=(192, 192))
    doc.sectors[0].floor_height = 64
    if return_from_key:
        doc.sectors[1].floor_height = 64
    doc.sectors[3].ceil_height = 0
    door = next(line for line in doc.linedefs if line.left_sidedef >= 0
                and doc.sidedefs[line.right_sidedef].sector == 2
                and doc.sidedefs[line.left_sidedef].sector == 3)
    door.special = 32
    add_key(doc, x=32)
    doc.things[-1].y = 192
    add_exit(doc, 4)
    return doc


def short_switch_document():
    ring = [(0, 0), (0, 56), (0, 72), (0, 128), (128, 128), (128, 0)]
    doc = polygon_document([[ring], [square(128, 0, 160, 128)],
                            [square(160, 0, 288, 128)]], start=(64, 64))
    doc.sectors[1].ceil_height, doc.sectors[1].tag = 0, 7
    switch = next(line for line in doc.linedefs if doc.vertices[line.v1].y == 56)
    switch.special, switch.tag = 103, 7
    add_exit(doc, 2)
    return doc


def shared_key_branches():
    hub = [(0, 128), (0, 256), (272, 256), (272, 128), (144, 128), (128, 128)]
    doc = polygon_document([[hub], [square(0, 0, 128, 128)],
                            [square(144, 0, 272, 128)], [square(272, 0, 304, 128)],
                            [square(304, 0, 432, 128)]], start=(64, 192))
    doc.sectors[0].floor_height = 64
    doc.sectors[3].ceil_height = 0
    line = next(line for line in doc.linedefs if line.left_sidedef >= 0
                and doc.sidedefs[line.right_sidedef].sector == 2
                and doc.sidedefs[line.left_sidedef].sector == 3)
    line.special = 32
    add_key(doc, x=127)
    doc.things[-1].y = 64
    add_exit(doc, 4)
    return doc


def consumption_walk_puzzle():
    doc = one_shot_revisit_puzzle()
    for line in doc.linedefs:
        if line.special == 11:
            line.special = 0
    exit_line = next(line for line in doc.linedefs if line.left_sidedef < 0
                     and doc.sidedefs[line.right_sidedef].sector == 1)
    exit_line.special = 11
    return doc

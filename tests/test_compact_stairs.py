"""Regressions for compact stair collision-floor modeling (8-unit tread / 16-unit rise).

An 8-unit tread / 16-unit rise fixture completes in the engine but was previously
rejected by both traversal abstractions (MapClearance and PlayerNavigation) because
the actor's radius (16) touched subsequent stair steps before the actor's center
point crossed into the immediate neighbor.

These tests verify that:
1. An 8-unit tread / 16-unit rise fixture is accepted by both traversal abstractions.
2. Steps exceeding 24 units without intermediate climbing steps (e.g. 8-unit tread /
   32-unit rise or a single 32-unit ledge) remain strictly rejected.
"""
from __future__ import annotations

from generation.validator import validate_document
from wad_parser.navigation import PlayerNavigation
from wad_parser.traversal import distances_from_start
from wad_parser.types import (
    Linedef, LinedefFlags, MapDocument, MapMeta, Sector, Sidedef, Thing, ThingFlags, Vertex,
)


def _build_stair_corridor_document(tread: float = 8.0, rise: int = 16, num_steps: int = 3, width: float = 128.0) -> MapDocument:
    """Build a map document with a start room, intermediate compact stair steps, and an end room with an exit.

    Start room: x in [-128, 0], y in [0, width], floor 0, ceil 128
    Step 1..N:  x in [(i-1)*tread, i*tread], y in [0, width], floor i*rise, ceil 128
    End room:   x in [N*tread, N*tread + 128], y in [0, width], floor (N+1)*rise, ceil (N+1)*rise + 128
    """
    vertices = []
    vert_map: dict[tuple[float, float], int] = {}

    def get_v(x: float, y: float) -> int:
        key = (round(float(x), 4), round(float(y), 4))
        if key not in vert_map:
            vid = len(vertices)
            vert_map[key] = vid
            vertices.append(Vertex(id=vid, x=key[0], y=key[1]))
        return vert_map[key]

    sectors = []
    sidedefs = []
    linedefs = []

    # Sector 0: start room
    sec_start = 0
    sectors.append(Sector(id=0, floor_height=0, ceil_height=128, floor_flat="FLOOR4_8", ceil_flat="CEIL3_5", light_level=160, special=0, tag=0))

    # Stair sectors
    stair_sectors = []
    for i in range(1, num_steps + 1):
        sid = len(sectors)
        sectors.append(Sector(id=sid, floor_height=i * rise, ceil_height=max(128, i * rise + 128),
                              floor_flat="FLOOR4_8", ceil_flat="CEIL3_5", light_level=160, special=0, tag=0))
        stair_sectors.append(sid)

    # Sector end: end room
    sec_end = len(sectors)
    end_floor = (num_steps + 1) * rise
    sectors.append(Sector(id=sec_end, floor_height=end_floor, ceil_height=end_floor + 128,
                          floor_flat="FLOOR4_8", ceil_flat="CEIL3_5", light_level=160, special=0, tag=0))

    def add_wall(p1, p2, sec: int):
        s_id = len(sidedefs)
        sidedefs.append(Sidedef(id=s_id, x_offset=0, y_offset=0, upper_tex="-", lower_tex="-", mid_tex="STARTAN2", sector=sec))
        linedefs.append(Linedef(id=len(linedefs), v1=get_v(*p1), v2=get_v(*p2),
                                flags=LinedefFlags(two_sided=False, impassable=True),
                                special=0, tag=0, right_sidedef=s_id, left_sidedef=-1))

    def add_portal(p1, p2, sec_r: int, sec_l: int, special: int = 0):
        sr = len(sidedefs)
        sidedefs.append(Sidedef(id=sr, x_offset=0, y_offset=0, upper_tex="STARTAN2", lower_tex="STARTAN2", mid_tex="-", sector=sec_r))
        sl = len(sidedefs)
        sidedefs.append(Sidedef(id=sl, x_offset=0, y_offset=0, upper_tex="STARTAN2", lower_tex="STARTAN2", mid_tex="-", sector=sec_l))
        linedefs.append(Linedef(id=len(linedefs), v1=get_v(*p1), v2=get_v(*p2),
                                flags=LinedefFlags(two_sided=True, impassable=False),
                                special=special, tag=0, right_sidedef=sr, left_sidedef=sl))

    # Build start room [-128, 0] x [0, width]
    add_wall((-128, 0), (-128, width), sec_start)
    add_wall((-128, width), (0, width), sec_start)
    add_wall((0, 0), (-128, 0), sec_start)

    # Portal start -> step 1
    add_portal((0, width), (0, 0), sec_start, stair_sectors[0])

    # Build intermediate steps
    for i, sid in enumerate(stair_sectors):
        x0 = i * tread
        x1 = (i + 1) * tread
        # North wall
        add_wall((x0, width), (x1, width), sid)
        # South wall
        add_wall((x1, 0), (x0, 0), sid)
        # East portal to next step or end room
        next_sec = stair_sectors[i + 1] if i + 1 < len(stair_sectors) else sec_end
        add_portal((x1, width), (x1, 0), sid, next_sec)

    # Build end room [num_steps*tread, num_steps*tread + 128] x [0, width]
    xe0 = num_steps * tread
    xe1 = xe0 + 128
    add_wall((xe0, width), (xe1, width), sec_end)
    # Exit switch on the far wall
    s_exit = len(sidedefs)
    sidedefs.append(Sidedef(id=s_exit, x_offset=0, y_offset=0, upper_tex="-", lower_tex="-", mid_tex="SW1EXIT", sector=sec_end))
    linedefs.append(Linedef(id=len(linedefs), v1=get_v(xe1, width), v2=get_v(xe1, 0),
                            flags=LinedefFlags(two_sided=False, impassable=True),
                            special=11, tag=0, right_sidedef=s_exit, left_sidedef=-1,
                            special_name="S1 Exit (Normal)", trigger_type="switch"))
    add_wall((xe1, 0), (xe0, 0), sec_end)

    # Player 1 start in start room
    things = [
        Thing(id=0, x=-64, y=width / 2, angle=0, type=1, type_name="Player1Start", category="player-start",
              flags=ThingFlags(skill_1_2=True, skill_3=True, skill_4_5=True))
    ]

    from wad_parser.map_extractor import compute_derived
    bounds, derived = compute_derived(vertices, sectors, linedefs, sidedefs, things)
    meta = MapMeta(map_name="MAP01", source_wad="<fixture>", wad_hash="stairs", format="doom", is_iwad=False)
    return MapDocument(schema_version="1.0", meta=meta, bounds=bounds,
                       vertices=vertices, sectors=sectors, linedefs=linedefs,
                       sidedefs=sidedefs, things=things, derived=derived)


def test_compact_stairs_8_tread_16_rise_accepted_by_both_abstractions():
    doc = _build_stair_corridor_document(tread=8.0, rise=16, num_steps=3)
    all_sector_ids = {s.id for s in doc.sectors}

    # 1. Sector graph traversal abstraction (distances_from_start via MapClearance)
    reached = set(distances_from_start(doc))
    assert reached == all_sector_ids, f"Expected all sectors reached, got {reached}"

    # 2. Configuration space navigation abstraction (PlayerNavigation)
    nav = PlayerNavigation(doc)
    start_key = (0, 0)
    assert start_key in nav.adjacency, "Start sector has no navigation regions"

    # Breadth-first search through navigation regions
    visited = {start_key}
    queue = [start_key]
    while queue:
        cur = queue.pop(0)
        for nb in nav.adjacency.get(cur, ()):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)

    visited_sectors = {key[0] for key in visited}
    assert visited_sectors == all_sector_ids, f"Navigation reached sectors {visited_sectors}, expected {all_sector_ids}"

    # 3. Document validator passes with no unreachable errors
    report = validate_document(doc, require_exit=True, require_connected=True)
    assert not report["errors"], f"Validation errors: {report['errors']}"


def test_compact_stairs_8_tread_32_rise_remains_strictly_rejected():
    # An 8-unit tread with 32-unit rise exceeds the 24-unit step limit
    doc = _build_stair_corridor_document(tread=8.0, rise=32, num_steps=2)

    # 1. Traversal abstraction rejects step 1 from start room
    reached = set(distances_from_start(doc))
    assert 0 in reached
    assert 1 not in reached, "Step 1 should be unreachable with a 32-unit rise"
    assert len(reached) < len(doc.sectors)

    # 2. PlayerNavigation does not connect start to step 1
    nav = PlayerNavigation(doc)
    start_key = (0, 0)
    visited = {start_key}
    queue = [start_key]
    while queue:
        cur = queue.pop(0)
        for nb in nav.adjacency.get(cur, ()):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)

    visited_sectors = {key[0] for key in visited}
    assert 1 not in visited_sectors
    assert len(visited_sectors) < len(doc.sectors)

    # 3. Validator reports unreachable sectors
    report = validate_document(doc, require_exit=True, require_connected=True)
    assert any("unreachable" in err for err in report["errors"])


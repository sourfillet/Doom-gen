"""Room-graph segmentation (Phase 9.1)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wad_builder import build_two_sector_wad

from roomgraph.assemble import assemble
from roomgraph.assemble_real import assemble_real
from roomgraph.corridor import plan_corridor
from roomgraph.segment import segment
from roomgraph.types import Connection, Room, RoomGraph
from wad_parser.map_extractor import extract_all_maps
from wad_parser.wad_reader import WadFile


def _doc(tmp_path):
    p = tmp_path / "src.wad"
    p.write_bytes(build_two_sector_wad())
    return extract_all_maps(WadFile(str(p)))[0]


def test_two_open_sectors_merge_into_one_room(tmp_path):
    # the fixture's two sectors share a wide (128u) open two-sided wall, no
    # door -> one room, no connections
    doc = _doc(tmp_path)
    rg = segment(doc)
    assert isinstance(rg, RoomGraph)
    assert len(rg.rooms) == 1
    assert set(rg.rooms[0].sectors) == {0, 1}
    assert rg.connections == []


def test_narrow_opening_splits_rooms(tmp_path):
    # widen the merge threshold requirement past the shared wall length so the
    # two sectors become separate rooms joined by one connection
    doc = _doc(tmp_path)
    rg = segment(doc, merge_width=200.0)
    assert len(rg.rooms) == 2
    assert len(rg.connections) == 1
    assert rg.connections[0].kind == "corridor"   # 128u opening < 200 threshold


def test_room_carries_concrete_geometry(tmp_path):
    doc = _doc(tmp_path)
    rg = segment(doc)
    room = rg.rooms[0]
    assert room.polygons and all(len(p) >= 3 for p in room.polygons)
    assert room.area > 0
    assert room.bbox[2] > room.bbox[0] and room.bbox[3] > room.bbox[1]


def test_summary_shape(tmp_path):
    rg = segment(_doc(tmp_path))
    s = rg.summary()
    assert s["rooms"] == 1 and "kinds" in s


# ---------------------------------------------------------------------------
# assembler (Phase 9.2): room graph -> playable geometry
# ---------------------------------------------------------------------------

def _two_room_graph() -> RoomGraph:
    def room(i):
        return Room(id=i, sectors=[i], floor_h=0, ceil_h=128, area=1e5,
                    centroid=(0.0, 0.0), bbox=(0.0, 0.0, 512.0, 512.0))
    return RoomGraph(rooms=[room(0), room(1)],
                     connections=[Connection(a=0, b=1, kind="door", width=192)])


def test_assemble_makes_rooms_and_a_corridor():
    asm = assemble(_two_room_graph())
    assert len(asm.sectors) == 5                    # 2 rooms + 2 approaches + closed door slab
    assert sum(s.ceil_height == s.floor_height for s in asm.sectors) == 1
    two_sided = [ld for ld in asm.linedefs if ld.left_sidedef != -1]
    assert len(two_sided) >= 2                       # corridor connects to both rooms
    # every sector closes into a polygon ring (well-formed geometry)
    assert all(len(asm.derived.sector_polygons[s.id]) >= 4 for s in asm.sectors)
    assert any(t.type == 1 for t in asm.things)     # a player start
    # a door connection puts a door special on a corridor line
    assert any(ld.special == 1 for ld in asm.linedefs)


def test_assembled_map_survives_wad_round_trip(tmp_path):
    from wad_writer.wad_writer import write_wad
    asm = assemble(_two_room_graph())
    out = write_wad([asm], tmp_path / "asm.wad")
    redoc = extract_all_maps(WadFile(str(out)))[0]
    assert len(redoc.sectors) == 5
    from wad_parser.traversal import distances_from_start
    assert set(distances_from_start(redoc)) == {s.id for s in redoc.sectors}
    assert any(ld.left_sidedef != -1 for ld in redoc.linedefs)  # openings survived


def test_assemble_from_segmented_real_fixture(tmp_path):
    # the 1-room fixture assembles to a single valid sector, no crash
    asm = assemble(segment(_doc(tmp_path)))
    assert len(asm.sectors) == 1 and asm.derived.sector_polygons


# ---------------------------------------------------------------------------
# real-geometry assembler (Phase 10.2): real polygons + carved corridors
# ---------------------------------------------------------------------------

def _square(cx, cy, half=128.0):
    return [(cx - half, cy - half), (cx + half, cy - half),
            (cx + half, cy + half), (cx - half, cy + half)]


def _two_poly_graph(kind="open") -> RoomGraph:
    def room(i):
        return Room(id=i, sectors=[i], floor_h=0, ceil_h=128, area=6.5e4,
                    centroid=(0.0, 0.0), bbox=(-128.0, -128.0, 128.0, 128.0),
                    polygons=[_square(0, 0)])
    return RoomGraph(rooms=[room(0), room(1)],
                     connections=[Connection(a=0, b=1, kind=kind, width=160)])


def test_plan_corridor_finds_facing_openings():
    # two squares 700u apart on the x-axis: the ray exits each on the facing
    # (inner) edge, so both openings sit near x = +/-128 of their own square
    plan = plan_corridor((0, 0), _square(0, 0), (700, 0), _square(700, 0))
    assert plan is not None
    ax = (plan.a.p0[0] + plan.a.p1[0]) / 2
    bx = (plan.b.p0[0] + plan.b.p1[0]) / 2
    assert abs(ax - 128) < 1 and abs(bx - 572) < 1     # facing inner walls


def test_assemble_real_carves_a_corridor():
    asm = assemble_real(_two_poly_graph(), positions={0: (0, 0), 1: (700, 0)})
    assert len(asm.sectors) == 3                        # 2 rooms + 1 corridor
    two_sided = [ld for ld in asm.linedefs if ld.left_sidedef != -1]
    assert len(two_sided) == 2                          # a mouth into each room
    # every two-sided line bridges two DIFFERENT sectors (valid opening)
    for ld in two_sided:
        assert asm.sidedefs[ld.right_sidedef].sector != asm.sidedefs[ld.left_sidedef].sector
    assert all(len(asm.derived.sector_polygons[s.id]) >= 3 for s in asm.sectors)
    assert any(t.type == 1 for t in asm.things)         # a player start


def test_assemble_real_door_connection_gets_special():
    asm = assemble_real(_two_poly_graph(kind="door"), positions={0: (0, 0), 1: (700, 0)})
    assert any(ld.special == 1 for ld in asm.linedefs)  # DR door on the mouth


def test_assemble_real_survives_wad_round_trip(tmp_path):
    from wad_writer.wad_writer import write_wad
    asm = assemble_real(_two_poly_graph(), positions={0: (0, 0), 1: (700, 0)})
    out = write_wad([asm], tmp_path / "real.wad")
    redoc = extract_all_maps(WadFile(str(out)))[0]
    assert len(redoc.sectors) == 3
    assert any(ld.left_sidedef != -1 for ld in redoc.linedefs)  # openings survived
    # every sector still closes into a ring after the writer/parser round-trip
    assert all(len(p) >= 3 for p in redoc.derived.sector_polygons)


def test_assemble_real_drops_corridor_that_tunnels_a_room(tmp_path):
    # a third room sitting on the straight line between 0 and 1 must block the
    # corridor (overlapping sectors); the two rooms then have no shared opening
    g = _two_poly_graph()
    g.rooms.append(Room(id=2, sectors=[2], floor_h=0, ceil_h=128, area=6.5e4,
                        centroid=(0.0, 0.0), bbox=(-128.0, -128.0, 128.0, 128.0),
                        polygons=[_square(0, 0)]))
    asm = assemble_real(g, positions={0: (0, 0), 1: (900, 0), 2: (450, 0)})
    # room 2 blocks 0<->1: no corridor sector bridging just those two remains
    two_sided = [ld for ld in asm.linedefs if ld.left_sidedef != -1]
    assert len(two_sided) == 0                          # the only connection was blocked


def test_assemble_real_from_segmented_real_fixture(tmp_path):
    # the 1-room fixture (no connections) assembles to one valid sector, no crash
    asm = assemble_real(segment(_doc(tmp_path)))
    assert len(asm.sectors) == 1 and asm.derived.sector_polygons

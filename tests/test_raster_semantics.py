from data.rasterizer import rasterize, SEMANTIC_SPECIALS, N_CHANNELS, RASTER_VERSION
from generation.vectorizer import vectorize
from tests.geometry_fixtures import two_rooms


def test_raster_semantic_exit_head_roundtrips_without_decoding_special_density():
    doc = two_rooms()
    wall = next(line for line in doc.linedefs if line.left_sidedef == -1
                and doc.vertices[line.v1].x == doc.vertices[line.v2].x == 0)
    wall.special = 51
    raster, _ = rasterize(doc, 64)
    assert raster.shape[0] == N_CHANNELS == 20 and RASTER_VERSION == 5
    assert raster[10 + SEMANTIC_SPECIALS.index(51)].sum() > 0
    decoded = vectorize(raster, coarse_factor=1)
    assert any(line.special == 51 for line in decoded.linedefs)
    raster[10:] = 0
    assert not any(line.special for line in vectorize(raster, coarse_factor=1).linedefs)


def test_raster_door_head_requires_a_closed_back_sector_and_correct_orientation():
    from roomgraph.assemble import assemble
    from roomgraph.types import Room, RoomGraph, Connection
    rooms = [Room(i, [], 0, 128, 256**2, (0, 0), (0, 0, 256, 256)) for i in range(2)]
    doc = assemble(RoomGraph(rooms, [Connection(0, 1, 'door', 128)]))
    raster, _ = rasterize(doc, 128)
    decoded = vectorize(raster, coarse_factor=1, cell_units=8)
    doors = [line for line in decoded.linedefs if line.special == 1]
    assert doors
    for line in doors:
        back = decoded.sectors[decoded.sidedefs[line.left_sidedef].sector]
        assert back.floor_height == back.ceil_height
    # A semantic door mark on an ordinary open partition must not close it.
    source = two_rooms()
    shared = next(line for line in source.linedefs if line.left_sidedef >= 0)
    shared.special = 1
    assert not any(line.special == 1 for line in vectorize(rasterize(source, 64)[0]).linedefs)


def test_legacy_raster_models_keep_their_ten_channel_interpretation():
    source = two_rooms()
    source.linedefs[0].special = 11
    raster = rasterize(source, 64)[0]
    assert not any(line.special for line in vectorize(raster, raster_version=3).linedefs)
    assert not any(line.special for line in vectorize(raster[:10]).linedefs)

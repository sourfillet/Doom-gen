import numpy as np

from data.rasterizer import rasterize, N_CHANNELS
from generation.vectorizer import vectorize
from tests.geometry_fixtures import two_rooms


def test_grid_face_boundaries_do_not_create_parallel_sliver_sectors():
    doc = two_rooms()
    raster, _ = rasterize(doc, 64)
    decoded = vectorize(raster, coarse_factor=2)
    assert len(decoded.sectors) == 2
    assert not decoded.derived.geometry_issues
    assert decoded.extra['raster_decoding']['effective_factor'] == 1


def test_face_directions_do_not_cut_perpendicular_neighbors():
    raster = np.zeros((N_CHANNELS, 8, 8), dtype=np.float32)
    raster[9, 1:7, 1:7] = 1
    raster[2] = 0.5
    raster[4] = 0.5625
    raster[5] = 0.5
    raster[16, 1:7, 3] = 1  # east faces only
    doc = vectorize(raster, coarse_factor=1)
    assert len(doc.sectors) == 2
    assert any(line.left_sidedef >= 0 and not line.flags.impassable for line in doc.linedefs)
    raster[18, 1:7, 3] = 1
    doc = vectorize(raster, coarse_factor=1)
    assert any(line.left_sidedef >= 0 and line.flags.impassable for line in doc.linedefs)


def test_legacy_raster_targets_remain_available_for_old_checkpoint_evaluation():
    doc = two_rooms()
    assert rasterize(doc, 32, raster_version=3)[0].shape[0] == 10
    assert rasterize(doc, 32, raster_version=4)[0].shape[0] == 16
    assert rasterize(doc, 32, raster_version=5)[0].shape[0] == 20

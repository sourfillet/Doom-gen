import numpy as np

from data.rasterizer import N_CHANNELS
from generation.vectorizer import vectorize
from wad_parser.geometry import sector_area_perimeter, ring_depths, sector_points


def raster(mask):
    result = np.zeros((N_CHANNELS, *mask.shape), dtype=np.float32)
    result[9] = mask
    result[2] = 0.5
    result[4] = 0.5625
    result[5] = 0.5
    return result


def test_decoder_keeps_single_pixel_passage_and_world_scale():
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 1:6] = mask[4:12, 10:15] = True
    mask[8, 6:10] = True
    doc = vectorize(raster(mask), coarse_factor=2, cell_units=64)
    assert doc.extra['raster_decoding']['effective_factor'] == 1
    assert len(doc.sectors) == 1
    assert sector_area_perimeter(doc, 0)[0] == mask.sum() * 32**2
    assert len([depth for depth in ring_depths(sector_points(doc, 0)) if depth % 2 == 0]) == 1


def test_decoder_keeps_a_single_pixel_hole():
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 4:12] = True
    mask[8, 8] = False
    doc = vectorize(raster(mask), coarse_factor=2, cell_units=64)
    assert doc.extra['raster_decoding']['effective_factor'] == 1
    assert sorted(ring_depths(sector_points(doc, 0))) == [0, 1]
    assert sector_area_perimeter(doc, 0)[0] == mask.sum() * 32**2

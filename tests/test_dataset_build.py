"""Tests for the Phase 3 dataset layer: JSON round-trip, rasterizer
channels, graph tensors, augmentation invariants, and the dataset loader
end to end."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wad_builder import build_two_sector_wad, build_udmf_wad

from data.augmentation import jitter_heights, jitter_lights, transform_document
from data.dataset import MapDataset
from data.graph_builder import EDGE_FEATURES, NODE_FEATURES, build_graph
from data.rasterizer import N_CHANNELS, rasterize
from wad_parser.map_extractor import extract_all_maps
from wad_parser.types import MapDocument
from wad_parser.wad_reader import WadFile


def _doc(tmp_path, builder=build_two_sector_wad, floor2: int | None = 64):
    p = tmp_path / "t.wad"
    p.write_bytes(builder())
    doc = extract_all_maps(WadFile(str(p)))[0]
    if floor2 is not None and len(doc.sectors) > 1:
        doc.sectors[1].floor_height = floor2  # give the map some relief
        from wad_parser.map_extractor import compute_derived
        doc.bounds, doc.derived = compute_derived(doc.vertices, doc.sectors, doc.linedefs, doc.sidedefs, doc.things)
    return doc


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------

def test_from_dict_round_trip(tmp_path):
    doc = _doc(tmp_path)
    reloaded = MapDocument.from_dict(json.loads(json.dumps(doc.to_dict())))
    assert reloaded.to_dict() == doc.to_dict()


def test_from_dict_round_trip_udmf(tmp_path):
    doc = _doc(tmp_path, build_udmf_wad, floor2=None)
    reloaded = MapDocument.from_dict(json.loads(json.dumps(doc.to_dict())))
    assert reloaded.to_dict() == doc.to_dict()
    assert reloaded.sectors[0].floor_slope is not None


def test_from_dict_tolerates_unknown_keys(tmp_path):
    d = _doc(tmp_path).to_dict()
    d["future_field"] = 123
    d["sectors"][0]["future_sector_field"] = "x"
    MapDocument.from_dict(d)  # must not raise


# ---------------------------------------------------------------------------
# rasterizer
# ---------------------------------------------------------------------------

def test_raster_shape_and_channels(tmp_path):
    doc = _doc(tmp_path)
    raster, meta = rasterize(doc, resolution=64)
    assert raster.shape == (N_CHANNELS, 64, 64)
    assert raster.dtype == np.float32
    assert raster[0].sum() > 0                      # walls drawn
    assert 0.0 <= raster.min() and raster.max() <= 1.0
    # floor-relative channel must distinguish the two sectors (0 vs 64)
    interior = raster[9] > 0                        # sector-id channel as area mask
    assert len(np.unique(np.round(raster[1][interior], 3))) >= 2
    # light 160 everywhere inside
    lights = np.unique(np.round(raster[5][interior], 3))
    assert np.allclose(lights, 160 / 255, atol=1e-3)
    # things splatted
    assert raster[8].max() == 1.0
    assert meta["resolution"] == 64 and meta["floor_max"] == 64


def test_raster_meta_invertibility(tmp_path):
    doc = _doc(tmp_path)
    _, meta = rasterize(doc, resolution=64)
    # a world point at map center must land at the tile center
    col = (doc.bounds.min_x + doc.bounds.max_x) / 2
    assert meta["center_x"] == col


# ---------------------------------------------------------------------------
# graph builder
# ---------------------------------------------------------------------------

def test_graph_tensors(tmp_path):
    doc = _doc(tmp_path)
    g = build_graph(doc)
    assert g["node_features"].shape == (2, len(NODE_FEATURES))
    assert g["edge_index"].shape == (2, 2)          # one link, both directions
    assert g["edge_features"].shape == (2, len(EDGE_FEATURES))
    assert set(map(tuple, g["edge_index"].T.tolist())) == {(0, 1), (1, 0)}
    assert np.isfinite(g["node_features"]).all()
    # player start in sector 0, monster in sector 1
    starts = g["node_features"][:, NODE_FEATURES.index("n_starts")]
    monsters = g["node_features"][:, NODE_FEATURES.index("n_monsters")]
    assert starts[0] > 0 and monsters[1] > 0
    # 64-unit floor step on the shared edge
    step = g["edge_features"][0][EDGE_FEATURES.index("floor_step")]
    assert np.isclose(step, 0.5)


# ---------------------------------------------------------------------------
# augmentation
# ---------------------------------------------------------------------------

def test_mirror_involution_and_sidedef_swap(tmp_path):
    doc = _doc(tmp_path)
    back = transform_document(transform_document(doc, "mirror_x"), "mirror_x")
    assert [(v.x, v.y) for v in back.vertices] == [(v.x, v.y) for v in doc.vertices]
    assert [t.angle for t in back.things] == [t.angle for t in doc.things]
    assert [(l.v1, l.v2, l.right_sidedef, l.left_sidedef) for l in back.linedefs] == \
           [(l.v1, l.v2, l.right_sidedef, l.left_sidedef) for l in doc.linedefs]
    # a single mirror reverses vertex order (sidedefs must NOT move: the
    # v1/v2 swap alone restores the right-side-front invariant)
    once = transform_document(doc, "mirror_x")
    assert once.linedefs[1].v1 == doc.linedefs[1].v2
    assert once.linedefs[1].right_sidedef == doc.linedefs[1].right_sidedef
    one_sided = [l for l in once.linedefs if l.left_sidedef == -1]
    assert len(one_sided) == len([l for l in doc.linedefs if l.left_sidedef == -1])


def test_rotation_angles_and_identity(tmp_path):
    doc = _doc(tmp_path)
    assert doc.things[0].angle == 90               # builder: player faces north
    rot = transform_document(doc, "rot90")
    assert rot.things[0].angle == 180              # CCW quarter turn -> west
    mirrored = transform_document(doc, "mirror_x")
    assert mirrored.things[0].angle == 90          # north unchanged by x-mirror
    assert mirrored.things[1].angle == 180         # east -> west
    four = doc
    for _ in range(4):
        four = transform_document(four, "rot90")
    assert [(v.x, v.y) for v in four.vertices] == [(v.x, v.y) for v in doc.vertices]


def test_augmented_derived_recomputed(tmp_path):
    doc = _doc(tmp_path)
    rot = transform_document(doc, "rot90")
    assert all(len(ring) >= 3 for ring in rot.derived.sector_polygons)
    assert len(rot.derived.sector_graph["edges"]) == len(doc.derived.sector_graph["edges"])
    assert rot.derived.reachable_from_start == doc.derived.reachable_from_start
    assert rot.derived.bsp_sanity_check is None    # no lump group to check against
    assert rot.extra["augmented"] == ["rot90"]
    # bounds actually rotated: original is wider than tall
    assert (doc.bounds.max_x - doc.bounds.min_x) == (rot.bounds.max_y - rot.bounds.min_y)


def test_mirror_raster_matches_flipped_raster(tmp_path):
    doc = _doc(tmp_path)
    r_orig, _ = rasterize(doc, resolution=64)
    r_mir, _ = rasterize(transform_document(doc, "mirror_x"), resolution=64)
    flipped = np.flip(r_orig, axis=2)
    # same mass per channel; geometry may shift by a pixel from rounding
    assert np.allclose(r_mir.sum(axis=(1, 2)), flipped.sum(axis=(1, 2)), rtol=0.06)


def test_jitters(tmp_path):
    doc = _doc(tmp_path)
    lit = jitter_lights(doc, amount=16, seed=7)
    assert all(0 <= s.light_level <= 255 for s in lit.sectors)
    assert any(s.light_level != o.light_level for s, o in zip(lit.sectors, doc.sectors))
    shifted = jitter_heights(doc, amount=16, seed=7)
    for s, o in zip(shifted.sectors, doc.sectors):
        assert (s.ceil_height - s.floor_height) == (o.ceil_height - o.floor_height)
        assert (s.floor_height - o.floor_height) % 8 == 0


# ---------------------------------------------------------------------------
# dataset loader
# ---------------------------------------------------------------------------

def test_dataset_end_to_end(tmp_path):
    doc = _doc(tmp_path)
    maps_dir = tmp_path / "corpus"
    maps_dir.mkdir()
    (maps_dir / "MAP01.json").write_text(json.dumps(doc.to_dict()))
    (maps_dir / "x.textures.json").write_text("{}")  # must be skipped

    from tagging.store import TagStore
    store = TagStore(tmp_path / "tags")
    store.tag_wad(doc.meta.wad_hash, "techbase")
    store.write_auto(doc.meta.wad_hash, {"vanilla": 1.0}, {})

    ds = MapDataset(maps_dir, tags_dir=tmp_path / "tags", resolution=64)
    assert len(ds) == 1
    sample = ds[0]
    assert sample["raster"].shape == (N_CHANNELS, 64, 64)
    assert sample["graph"]["node_features"].shape[0] == 2
    assert sample["tags"] == {"techbase": 1.0, "vanilla": 1.0}

    vocab = ds.tag_vocab()
    assert vocab == ["techbase", "vanilla"]
    vec = ds.tag_vector(sample["tags"], vocab)
    assert vec.tolist() == [1.0, 1.0]

    augmented = ds.get(0, augment_op="mirror_y")
    assert augmented["raster"].shape == (N_CHANNELS, 64, 64)

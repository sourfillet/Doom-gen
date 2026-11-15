"""Raster cache hits must match the preprocessing semantics that built them."""
import hashlib
import json

import numpy as np

from data.raster_cache import RASTER_CACHE_VERSION, RasterCache
from data.rasterizer import RASTER_VERSION
from models.training import code_fingerprint, tag_fingerprint
from wad_parser.geometry import DERIVED_VERSION


class Dataset:
    resolution = 8

    def __init__(self):
        self.calls = 0

    def get(self, index, **kwargs):
        self.calls += 1
        return {'raster': np.ones((1, 8, 8), dtype=np.float32),
                'raster_meta': {'resolution': 8}, 'tags': {'techbase': 1.0}}


def test_old_namespace_is_preserved_but_never_reused_for_current_targets(tmp_path):
    dataset = Dataset()
    legacy = f"corpus:{tag_fingerprint(None)}:8:{RASTER_VERSION}:{DERIVED_VERSION}"
    old_prefix = hashlib.sha256(legacy.encode()).hexdigest()
    old_array = tmp_path / f'{old_prefix}_0_identity.npy'
    old_meta = old_array.with_suffix('.json')
    np.save(old_array, np.zeros((1, 8, 8), dtype=np.float32))
    old_meta.write_text(json.dumps({'raster_meta': {}, 'tags': {}}))
    before = old_array.read_bytes(), old_meta.read_bytes()
    cache = RasterCache(dataset, tmp_path, 'corpus')
    result = cache.get(0)
    assert dataset.calls == 1
    assert np.all(result['raster'] == 1)
    assert result['cache_contract']['version'] == RASTER_CACHE_VERSION
    assert result['cache_contract']['preprocessing_fingerprint'] == cache.preprocessing_fingerprint
    assert (old_array.read_bytes(), old_meta.read_bytes()) == before
    repeat = RasterCache(dataset, tmp_path, 'corpus').get(0)
    assert dataset.calls == 1
    assert np.array_equal(repeat['raster'], result['raster'])


def test_preprocessing_revision_changes_miss_a_warmed_cache(tmp_path, monkeypatch):
    dataset = Dataset()
    monkeypatch.setattr('data.raster_cache.code_fingerprint', lambda scope: 'old-implementation')
    before = RasterCache(dataset, tmp_path, 'corpus')
    before.get(0)
    monkeypatch.setattr('data.raster_cache.code_fingerprint', lambda scope: 'new-implementation')
    after = RasterCache(dataset, tmp_path, 'corpus')
    after.get(0)
    assert before.prefix != after.prefix and dataset.calls == 2


def test_dependency_changes_miss_a_warmed_cache(tmp_path, monkeypatch):
    dataset = Dataset()
    monkeypatch.setattr('data.raster_cache.runtime_versions', lambda *names: {'numpy': 'old'})
    before = RasterCache(dataset, tmp_path, 'corpus')
    before.get(0)
    monkeypatch.setattr('data.raster_cache.runtime_versions', lambda *names: {'numpy': 'new'})
    after = RasterCache(dataset, tmp_path, 'corpus')
    assert before.prefix != after.prefix
    assert after.get(0)['cache_contract']['runtime']['numpy'] == 'new'
    assert dataset.calls == 2


def test_raster_source_fingerprint_tracks_transitive_code_and_ignores_generation(tmp_path):
    for directory in ('data', 'wad_parser', 'generation'):
        (tmp_path / directory).mkdir()
    (tmp_path / 'data/dataset.py').write_text('from data.augmentation import transform_document\n')
    augmentation = tmp_path / 'data/augmentation.py'
    augmentation.write_text('from wad_parser.geometry import DERIVED_VERSION\n')
    geometry = tmp_path / 'wad_parser/geometry.py'
    geometry.write_text('DERIVED_VERSION = 5\n')
    unrelated = tmp_path / 'generation/validator.py'
    unrelated.write_text('value = 1\n')
    initial = code_fingerprint('raster_preprocessing', root=tmp_path)
    unrelated.write_text('value = 2\n')
    assert code_fingerprint('raster_preprocessing', root=tmp_path) == initial
    augmentation.write_text('from wad_parser.geometry import DERIVED_VERSION\nvalue = 1\n')
    changed = code_fingerprint('raster_preprocessing', root=tmp_path)
    assert changed != initial
    geometry.write_text('DERIVED_VERSION = 6\n')
    assert code_fingerprint('raster_preprocessing', root=tmp_path) != changed

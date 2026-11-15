"""Bounded-memory, versioned raster cache; augmentation stays on demand."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from shapely import geos_version_string

from data.rasterizer import RASTER_VERSION
from wad_parser.geometry import DERIVED_VERSION
from models.training import code_fingerprint, runtime_versions, tag_fingerprint

RASTER_CACHE_VERSION = 2


class RasterCache:
    def __init__(self, dataset, directory, fingerprint, tags_dir=None):
        self.dataset = dataset
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.preprocessing_fingerprint = code_fingerprint('raster_preprocessing')
        self.runtime = {**runtime_versions('numpy', 'shapely'), 'geos': geos_version_string}
        signature = (f"raster-cache-v{RASTER_CACHE_VERSION}:{self.preprocessing_fingerprint}:"
                     f"{fingerprint}:{tag_fingerprint(tags_dir)}:{dataset.resolution}:"
                     f"{RASTER_VERSION}:{DERIVED_VERSION}:{json.dumps(self.runtime, sort_keys=True)}")
        self.prefix = hashlib.sha256(signature.encode()).hexdigest()

    def get(self, index, op=None):
        path = self.directory / f"{self.prefix}_{index}_{op or 'identity'}"
        array_path, meta_path = path.with_suffix('.npy'), path.with_suffix('.json')
        if not array_path.exists() or not meta_path.exists():
            sample = self.dataset.get(index, augment_op=op, include_graph=False)
            pid = os.getpid()
            tmp_array = path.with_suffix(f'.{pid}.npy.tmp')
            with tmp_array.open('wb') as stream:
                np.save(stream, sample['raster'])
            tmp_array.replace(array_path)
            tmp_meta = path.with_suffix(f'.{pid}.json.tmp')
            tmp_meta.write_text(
                json.dumps({'raster_meta': sample['raster_meta'], 'tags': sample['tags'],
                            'cache_contract': {'version': RASTER_CACHE_VERSION,
                                'preprocessing_fingerprint': self.preprocessing_fingerprint,
                                'runtime': self.runtime}}),
                encoding='utf-8',
            )
            tmp_meta.replace(meta_path)
        metadata = json.loads(meta_path.read_text(encoding='utf-8'))
        with array_path.open('rb') as stream:
            raster = np.load(stream)
        return {**metadata, 'raster': raster}

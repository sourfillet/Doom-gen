import json

import pytest

from data.corpus import build_manifest, training_split
from data.geometry_tokens import encode_prediction
from data.learned_geometry import build_target
from tests.geometry_fixtures import two_rooms
from tools.select_geometry_sources import select


def test_selection_retains_original_families_splits_and_exact_token_counts(tmp_path):
    maps = tmp_path / 'maps'
    maps.mkdir()
    for index in range(3):
        doc = two_rooms()
        doc.meta.wad_hash = f'family-{index}'
        doc.sectors[0].light_level += index
        (maps / f'{index}.json').write_text(json.dumps(doc.to_dict()))
    parent = build_manifest(maps)
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(parent))
    report = select(maps, path, tmp_path / 'selected', min_vertices=4, max_maps=3, per_family=1)
    assert report['selected'] == 3
    splits, child = training_split(maps, tmp_path / 'selected/corpus.json')
    assert {row['path']: row['split'] for row in child['entries']} == {row['path']: row['split'] for row in parent['entries']}
    assert all(splits[name] for name in ('train', 'validation', 'test'))
    doc = two_rooms()
    assert report['records'][0]['tokens'] == len(encode_prediction(build_target(doc)['supervision']))
    assert path.read_text() == json.dumps(parent)
    with pytest.raises(ValueError, match='fresh directory'):
        select(maps, path, tmp_path / 'selected')

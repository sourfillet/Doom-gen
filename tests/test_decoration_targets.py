import json

import numpy as np
import pytest

pytest.importorskip('torch')

from tests.geometry_fixtures import two_rooms
from data.placement_targets import build_placement_vocabs, placement_sample
from data.property_targets import graph_sample
from wad_parser.types import Thing, ThingFlags, resolve_doomed


def _thing(doc, kind, x, y, angle=0, **flags):
    name, category, _ = resolve_doomed(kind)
    doc.things.append(Thing(len(doc.things), x, y, angle, kind, name, category,
                            ThingFlags(**flags)))


def test_placement_targets_select_one_population_and_retain_rare_keys(tmp_path):
    doc = two_rooms()
    _thing(doc, 3004, 96, 32, 45, skill_3=True)
    _thing(doc, 3004, 32, 96, skill_1_2=True)
    _thing(doc, 3004, 192, 32, skill_3=True, not_single_player=True)
    for _ in range(4):
        _thing(doc, 2007, 192, 96, skill_3=True)
    _thing(doc, 5, 192, 64, skill_3=True)
    (tmp_path / 'map.json').write_text(json.dumps(doc.to_dict()))
    vocabs = build_placement_vocabs(tmp_path, top=1)
    assert vocabs == {'monster': [3004], 'pickup': [2007, 5]}
    sample = placement_sample(doc, vocabs)
    assert sample['monster_count'].tolist() == [1, 0]
    assert sample['pickup_count'].tolist() == [0, 5]
    assert sample['monster_position'][0, 4 * 16 + 12] == 1
    assert sample['monster_angle'][0, 1] == 1
    assert sample['monster_position'].sum() == 1


def test_wall_targets_distinguish_sides_of_the_same_sector_and_signed_offsets():
    doc = two_rooms()
    doc.sidedefs[0].mid_tex, doc.sidedefs[1].mid_tex = 'STONE2', 'STONE3'
    doc.sidedefs[0].x_offset, doc.sidedefs[0].y_offset = -16, 32
    vocab = {'flat': {'UNK': 0}, 'tex': {'UNK': 0, '-': 1, 'STONE2': 2, 'STONE3': 3},
             'special': {'0': 0}}
    sample = graph_sample(doc, vocab)
    assert sample['side_owner'][:2].tolist() == [0, 0]
    assert sample['side_mid_tex'][:2].tolist() == [2, 3]
    assert np.allclose(sample['side_offsets'][0], [-16 / 1024, 32 / 1024])
    assert not np.array_equal(sample['side_features'][0], sample['side_features'][1])

import copy
import hashlib
import json
import math

import pytest

from data.spatial_prediction import build_prediction_scaffold, decode_prediction, scaffold_edges, graph_from_source_record
from data.spatial_targets import authoring_digest, build_spatial_target
from tests.geometry_fixtures import two_rooms
from wad_parser.types import Thing, ThingFlags


def _target():
    doc = two_rooms()
    doc.sectors[1].ceil_height = 0  # Retain a closed, collapsed-door sector.
    next(line for line in doc.linedefs if line.left_sidedef >= 0).special = 1
    doc.things.append(Thing(1, 192, 64, 0, 14, 'TeleportDestination', 'teleport', ThingFlags(skill_3=True)))
    return doc, build_spatial_target(doc)


def test_lossless_labels_reconstruct_controls_and_shared_boundaries_without_mutation():
    source, target = _target()
    before = copy.deepcopy(target)
    scaffold, labels = build_prediction_scaffold(target)
    assert all('x' not in v and 'y' not in v for v in scaffold['vertices'] + scaffold['things'])
    assert all('floor_height' not in s and 'ceil_height' not in s for s in scaffold['sectors'])
    assert 'hierarchy' not in scaffold and 'geometry' not in scaffold
    decoded = decode_prediction(scaffold, labels)
    assert authoring_digest(decoded) == authoring_digest(source)
    assert decoded.sectors[1].ceil_height == 0
    assert any(line.special == 1 for line in decoded.linedefs)
    assert len(decoded.things) == 2
    assert sum(line.left_sidedef >= 0 for line in decoded.linedefs) == 1
    assert target == before
    assert scaffold_edges(scaffold)


def test_predicted_xy_and_each_sector_height_actually_change_output():
    source, target = _target()
    scaffold, labels = build_prediction_scaffold(target)
    prediction = copy.deepcopy(labels)
    nv, ns = len(scaffold['vertices']), len(scaffold['sectors'])
    for index, values in enumerate(prediction['values']):
        if nv <= index < nv + ns:
            values[1] = 160 / 128
        else:
            values[0] += 8 / scaffold['frame']['scale']
    changed = decode_prediction(scaffold, prediction)
    assert [v.x for v in changed.vertices] == [v.x + 8 for v in source.vertices]
    assert [t.x for t in changed.things] == [t.x + 8 for t in source.things]
    assert all(s.ceil_height == 160 for s in changed.sectors)
    assert authoring_digest(changed) != authoring_digest(source)


@pytest.mark.parametrize('failure', ['missing', 'nan', 'infinity', 'collapsed'])
def test_invalid_predictions_are_rejected_without_authored_geometry_fallback(failure):
    _, target = _target()
    scaffold, prediction = build_prediction_scaffold(target)
    before = copy.deepcopy(scaffold)
    if failure == 'missing':
        prediction['values'].pop()
    elif failure in {'nan', 'infinity'}:
        prediction['values'][0][0] = math.nan if failure == 'nan' else math.inf
    else:
        for row in prediction['values']:
            row[:] = [0, 0]
    with pytest.raises(ValueError):
        decode_prediction(scaffold, prediction)
    assert scaffold == before


def test_incomplete_geometry_v1_cannot_silently_be_used_as_lossless_scaffold():
    with pytest.raises(ValueError, match='omits source controls'):
        build_prediction_scaffold({'schema': 'doom_spatial_geometry_v1'})


def test_changed_numeric_source_cannot_keep_an_old_authoring_fingerprint():
    _, target = _target()
    target['vertices'][0]['x'] += 8 / target['frame']['scale']
    with pytest.raises(ValueError, match='authoring fingerprint'):
        build_prediction_scaffold(target)


@pytest.mark.parametrize('change', ['none', 'bytes', 'digest', 'version', 'map', 'wad', 'graph_map'])
def test_frozen_graph_is_bound_to_source_bytes_and_map_identity(change):
    from tests.geometry_fixtures import two_room_graph
    source = two_rooms()
    raw = json.dumps(source.to_dict()).encode()
    record = {'graph_snapshot_version': 1, 'graph_snapshot': two_room_graph().to_dict(),
              'map': source.meta.map_name, 'wad_hash': source.meta.wad_hash,
              'document_sha256': hashlib.sha256(raw).hexdigest()}
    before = copy.deepcopy(record)
    if change == 'none':
        assert graph_from_source_record(raw, record).to_dict() == record['graph_snapshot']
        assert record == before
        return
    if change == 'bytes':
        raw += b' '
    elif change == 'digest':
        record['document_sha256'] = '0' * 64
    elif change == 'version':
        record['graph_snapshot_version'] = 2
    elif change == 'graph_map':
        record['graph_snapshot']['map_name'] = 'MAP02'
    else:
        record['map' if change == 'map' else 'wad_hash'] = 'other'
    with pytest.raises(ValueError):
        graph_from_source_record(raw, record)

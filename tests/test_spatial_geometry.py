import pytest

from data.spatial_geometry import build_spatial_geometry_target, SCHEMA
from tests.geometry_fixtures import two_room_graph, two_rooms
from roomgraph.types import Room, RoomGraph


def test_spatial_geometry_target_two_rooms():
    doc = two_rooms()
    graph = two_room_graph()
    target = build_spatial_geometry_target(doc, graph)

    assert target['schema'] == SCHEMA
    assert target['frame']['scale'] > 0
    assert len(target['topology']['rooms']) == 2
    assert len(target['topology']['connections']) == 1

    placements = target['geometry']['placements']
    assert len(placements) == 2
    for p in placements:
        assert len(p['center']) == 2
        assert len(p['bounds']) == 4
        assert not p['is_multi_height']

    interiors = target['geometry']['interiors']
    assert len(interiors) == 2
    for room in interiors:
        assert len(room['components']) >= 1
        comp = room['components'][0]
        assert 'exterior_ring' in comp
        assert 'relative_exterior_ring' in comp
        assert len(comp['exterior_ring']) == len(comp['relative_exterior_ring'])


def test_spatial_geometry_multi_height_room():
    doc = two_rooms()
    # Modify second sector height
    doc.sectors[1].floor_height = 24
    doc.sectors[1].ceil_height = 160

    # Put both sectors into room 0 to form a multi-height interior
    rooms = [
        Room(0, [0, 1], 0, 160, 2 * 512**2, (256, 0), (0, 0, 1024, 512), roles=['start', 'exit']),
    ]
    graph = RoomGraph(rooms, [])
    target = build_spatial_geometry_target(doc, graph)

    p0 = target['geometry']['placements'][0]
    assert p0['is_multi_height']
    assert p0['distinct_floors'] == [0.0, 24.0]
    assert p0['distinct_ceilings'] == [128.0, 160.0]

    interiors = target['geometry']['interiors'][0]
    assert interiors['component_count'] == 2


def test_spatial_geometry_exclusion_masking():
    doc = two_rooms()
    graph = two_room_graph()

    audit_data = {
        'connections': [
            {
                'a': 0,
                'b': 1,
                'kind': 'door',
                'status': 'unsupported_or_unusable',
                'local_directions': {'initial': [], 'potential': []},
                'contiguous_runs': [],
            }
        ]
    }
    index_data = {
        'strict_compiled_area_match': True,
        'raw_validation_errors': 0,
        'compiled_validation_errors': 0,
        'outside_local_audit': [
            {'linedef': 5, 'special': 123, 'tag': 2, 'name': 'Lift: Lowest Neighbor Floor'}
        ],
    }

    target = build_spatial_geometry_target(doc, graph, audit_data=audit_data, index_data=index_data)

    exclusions = target['exclusions']
    assert exclusions['is_clean_geometry'] is True
    assert [0, 1] in exclusions['unsupported_edges']
    assert exclusions['outside_audit_count'] == 1
    assert exclusions['outside_audit_actions'][0]['special'] == 123


def test_spatial_geometry_requires_player_start():
    doc = two_rooms()
    doc.things = []  # Remove player 1 start
    graph = two_room_graph()
    with pytest.raises(ValueError, match='requires exactly one player-one start'):
        build_spatial_geometry_target(doc, graph)


def test_spatial_geometry_real_dmonfear_map19():
    import json
    from pathlib import Path
    from wad_parser.types import MapDocument

    stem = '6ce8b4b0e602140261a202180dbe7c317453b8f1991985258e503643caf2ee26-0018'
    corpus = Path('generated/repair-experiments/current-complete-corpus')
    base = Path('generated/repair-experiments/adjoining-architecture')
    doc_path = corpus / 'maps' / f'{stem}.json'
    if not doc_path.exists():
        pytest.skip('corpus map not available')

    doc = MapDocument.from_dict(json.loads(doc_path.read_text()))
    graph_snapshot = json.loads((corpus / 'records' / f'{stem}.json').read_text())['graph_snapshot']
    graph = RoomGraph.from_dict(graph_snapshot)
    audit = json.loads((base / 'spatial-audit-v1' / f'{stem}.json').read_text())
    index = json.loads((base / 'spatial-index-v2' / f'{stem}.json').read_text())

    target = build_spatial_geometry_target(doc, graph, audit_data=audit, index_data=index)
    assert target['schema'] == SCHEMA
    assert len(target['topology']['rooms']) == 33
    assert len(target['topology']['connections']) == 36
    assert len(target['geometry']['placements']) == 33
    assert sum(p['is_multi_height'] for p in target['geometry']['placements']) == 5
    assert target['exclusions']['unsupported_edges'] == [[8, 11]]
    assert target['exclusions']['one_way_edges'] == [[3, 4], [27, 12]]
    assert target['exclusions']['outside_audit_count'] == 2
    assert target['exclusions']['is_clean_geometry'] is True


import copy
import json

import pytest

from data.spatial_targets import build_spatial_target, authoring_digest
from roomgraph.spatial import reconstruct_spatial_target
from tests.geometry_fixtures import polygon_document, square, two_rooms
from wad_parser.geometry import sector_area_perimeter
from wad_parser.types import Thing, ThingFlags, SectorSlope, Vertex
from wad_writer.assembler import assemble_map_lumps


def reconstruct(doc, graph=None):
    target = json.loads(json.dumps(build_spatial_target(doc, graph)))
    return target, reconstruct_spatial_target(target)


def test_irregular_courtyard_and_unreferenced_vertex_round_trip():
    outer = [(0, 0), (0, 384), (192, 384), (192, 256), (384, 256), (384, 0)]
    hole = list(reversed(square(64, 64, 128, 128)))
    doc = polygon_document([[outer, hole]], start=(48, 192))
    doc.vertices.append(Vertex(len(doc.vertices), -4096, 123))
    before = copy.deepcopy(doc.to_dict())
    target, rebuilt = reconstruct(doc)
    assert target['schema'] == 'doom_spatial_reconstruction_v1'
    assert authoring_digest(doc) == authoring_digest(rebuilt)
    assert assemble_map_lumps(doc) == assemble_map_lumps(rebuilt)
    assert sector_area_perimeter(doc, 0) == sector_area_perimeter(rebuilt, 0)
    assert not rebuilt.derived.geometry_issues
    assert doc.to_dict() == before


def test_multiheight_hierarchy_preserves_distinct_interiors_and_assets():
    from roomgraph.types import Room, RoomGraph
    doc = two_rooms()
    doc.sectors[1].floor_height = 24
    doc.sectors[1].light_level = 96
    doc.sectors[1].floor_flat = 'NUKAGE1'
    graph = RoomGraph([Room(0, [0, 1], 0, 128, 32768, (128, 64), (0, 0, 256, 128),
                           components=[(0, 0), (1, 0)])])
    target, rebuilt = reconstruct(doc, graph)
    parts = target['hierarchy']['rooms'][0]['components']
    assert {p['floor'] for p in parts} == {0, 24}
    assert {p['source_component'][0] for p in parts} == {0, 1}
    assert len(rebuilt.sectors) == 2
    assert authoring_digest(doc) == authoring_digest(rebuilt)


def test_authored_actions_locks_remote_controls_unknown_flags_and_things_survive():
    doc = polygon_document([[square(-256, -128, -128, 0)], [square(-128, -128, -96, 0)],
                            [square(-96, -128, 32, 0)], [square(512, 512, 544, 544)]],
                           start=(-192, -64))
    doc.sectors[1].ceil_height = 0
    doc.sectors[1].tag = 7
    shared = next(line for line in doc.linedefs if line.left_sidedef >= 0)
    shared.special = 26
    shared.extra['unrecognized_flag_bits'] = 0x4000
    switch = doc.linedefs[-1]
    switch.special, switch.tag = 103, 7
    doc.sidedefs[-1].x_offset, doc.sidedefs[-1].y_offset = -16, 24
    doc.sidedefs[-1].mid_tex = 'SW1BRCOM'
    doc.things.append(Thing(1, -160, -64, 90, 5, 'BlueCard', 'key', ThingFlags(skill_3=True),
                            extra={'unrecognized_flag_bits': 0x1000}))
    target, rebuilt = reconstruct(doc)
    assert len(target['sectors']) == 4
    assert authoring_digest(doc) == authoring_digest(rebuilt)
    assert assemble_map_lumps(doc) == assemble_map_lumps(rebuilt)
    rebuilt.linedefs[shared.id].special = 1
    assert authoring_digest(doc) != authoring_digest(rebuilt)


def test_reconstruction_recomputes_geometry_without_trusting_cached_bsp():
    doc = two_rooms()
    doc.derived.sector_regions = [[[(0, 0), (1000, 1000), (0, 1000)]]] * 2
    target, rebuilt = reconstruct(doc)
    assert 'derived' not in target
    assert rebuilt.derived.sector_regions == []
    assert sector_area_perimeter(rebuilt, 0)[0] == 128**2


@pytest.mark.parametrize('case', ['udmf', 'hexen', 'slope', 'fractional', 'sparse_ids'])
def test_unsupported_authoring_is_explicit(case):
    doc = two_rooms()
    if case in {'udmf', 'hexen'}:
        doc.meta.format = case
    elif case == 'slope':
        doc.sectors[0].floor_slope = SectorSlope(0, 0, 1, 0)
    elif case == 'fractional':
        doc.vertices[0].x += 0.1
    else:
        doc.sectors[0].id = 8
    with pytest.raises(ValueError):
        build_spatial_target(doc)


def test_missing_ambiguous_or_duplicate_component_attribution_is_rejected():
    from roomgraph.types import Room, RoomGraph
    doc = polygon_document([[square(0, 0, 128, 128), square(512, 0, 640, 128)]], start=(64, 64))
    graph = RoomGraph([Room(0, [0], 0, 128, 32768, (64, 64), (0, 0, 640, 128))])
    with pytest.raises(ValueError, match='explicit component'):
        build_spatial_target(doc, graph)
    graph.rooms[0].components = [(0, 0), (0, 99)]
    with pytest.raises(ValueError, match='missing component'):
        build_spatial_target(doc, graph)
    graph.rooms[0].components = [(0, 0), (0, 0)]
    with pytest.raises(ValueError, match='duplicate component'):
        build_spatial_target(doc, graph)


@pytest.mark.parametrize('case', ['schema', 'scale', 'nan', 'coordinate', 'reference'])
def test_corrupted_reconstruction_targets_fail(case):
    target = build_spatial_target(two_rooms())
    if case == 'schema':
        target['schema'] = 'old'
    elif case == 'scale':
        target['frame']['scale'] = 0
    elif case == 'nan':
        target['vertices'][0]['x'] = float('nan')
    elif case == 'coordinate':
        target['vertices'][0]['x'] += 0.001
    else:
        target['linedefs'][0]['v1'] = 123456
    with pytest.raises(ValueError):
        reconstruct_spatial_target(target)

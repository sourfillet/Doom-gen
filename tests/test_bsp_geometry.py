import json
import struct
import zlib

import pytest

from tests.wad_builder import build_two_sector_wad_with_nodes
from wad_parser.wad_reader import WadFile
from wad_parser.map_extractor import extract_map, _sector_containing
from wad_parser.bsp_geometry import reconstruct_bsp, union_sector_regions
from wad_parser.geometry import sector_points, sector_geometry, sector_area_perimeter, ring_depths, rings_area_perimeter
from wad_parser.types import MapDocument
from roomgraph.segment import segment
from data.rasterizer import rasterize


def _group(tmp_path):
    path = tmp_path / 'bsp.wad'
    path.write_bytes(build_two_sector_wad_with_nodes())
    wad = WadFile(str(path))
    group = wad.find_map_groups()[0]
    # Partition northward: child 0 is the east room, child 1 the west room.
    group.lumps['NODES'] = struct.pack('<hhhhhhhhhhhhHH', 128, 0, 0, 128,
                                     128, 0, 128, 256, 128, 0, 0, 128, 0x8001, 0x8000)
    return wad, group


def test_bsp_recovers_an_unclosed_raw_sector_without_editing_authored_elements(tmp_path):
    wad, group = _group(tmp_path)
    raw = bytearray(group.lumps['SIDEDEFS'])
    struct.pack_into('<H', raw, 2 * 30 + 28, 1)  # north wall points at the wrong sector
    group.lumps['SIDEDEFS'] = bytes(raw)
    doc = extract_map(wad, group)
    assert not doc.derived.geometry_issues
    assert doc.derived.raw_geometry_issues
    assert doc.derived.bsp_geometry['method'] == 'classic-bsp-halfplanes-v1'
    assert len(doc.vertices) == 6 and doc.sidedefs[2].sector == 1
    assert len(doc.linedefs) == 7 and len(doc.sidedefs) == 8
    assert sector_area_perimeter(doc, 0)[0] == 128**2
    assert sector_area_perimeter(doc, 1)[0] == 128**2
    assert _sector_containing(64, 64, doc.sectors, sector_geometry(doc), doc.vertices) == 0
    assert _sector_containing(192, 64, doc.sectors, sector_geometry(doc), doc.vertices) == 1
    assert _sector_containing(-100, -100, doc.sectors, sector_geometry(doc), doc.vertices) is None
    graph = segment(doc, playable_only=True)
    assert len(graph.rooms) == 1 and graph.rooms[0].area == 256 * 128
    assert rasterize(doc, 64)[0][9].sum() > 0
    restored = MapDocument.from_dict(json.loads(json.dumps(doc.to_dict())))
    assert sector_points(restored, 0) == sector_points(doc, 0)
    assert restored.to_dict()['vertices'] == doc.to_dict()['vertices']


def test_malformed_bsp_cannot_hide_unclosed_raw_geometry(tmp_path):
    wad, group = _group(tmp_path)
    raw = bytearray(group.lumps['SIDEDEFS'])
    struct.pack_into('<H', raw, 2 * 30 + 28, 1)
    group.lumps['SIDEDEFS'] = bytes(raw)
    group.lumps['NODES'] = b'broken'
    doc = extract_map(wad, group)
    assert doc.derived.geometry_issues and not doc.derived.sector_regions
    assert 'truncated' in doc.derived.bsp_geometry['failure']


def test_compiled_regions_can_ignore_a_zero_length_authoring_boundary_without_deleting_it(tmp_path):
    from wad_writer.assembler import assemble_map_lumps
    wad, group = _group(tmp_path)
    group.lumps['LINEDEFS'] += struct.pack('<7H', 0, 0, 1, 0, 0, 7, 0xffff)
    doc = extract_map(wad, group)
    assert len(doc.linedefs) == 8 and doc.linedefs[-1].v1 == doc.linedefs[-1].v2 == 0
    assert any('zero-length boundary' in issue for issue in doc.derived.raw_geometry_issues)
    assert not doc.derived.geometry_issues and doc.derived.sector_regions
    assert sector_area_perimeter(doc, 0)[0] == 128**2
    assert sector_area_perimeter(doc, 1)[0] == 128**2
    restored = MapDocument.from_dict(json.loads(json.dumps(doc.to_dict())))
    assert restored.derived.sector_regions == json.loads(json.dumps(doc.derived.sector_regions))
    assert restored.derived.raw_geometry_issues == doc.derived.raw_geometry_issues
    legacy = doc.to_dict()
    legacy['derived']['derived_version'] = 4
    upgraded = MapDocument.from_dict(legacy)
    assert not upgraded.derived.geometry_issues
    assert sector_area_perimeter(upgraded, 0)[0] == 128**2
    assert sector_area_perimeter(upgraded, 1)[0] == 128**2
    with pytest.raises(ValueError, match='zero length'):
        assemble_map_lumps(doc)


@pytest.mark.parametrize('corruption', ['vertex', 'zero_line_side'])
def test_compiled_recovery_does_not_hide_other_corrupt_references_alongside_a_zero_length_line(tmp_path, corruption):
    wad, group = _group(tmp_path)
    if corruption == 'vertex':
        group.lumps['LINEDEFS'] += struct.pack('<7H', 0, 0, 1, 0, 0, 7, 0xffff)
        group.lumps['LINEDEFS'] += struct.pack('<7H', 9999, 0, 1, 0, 0, 7, 0xffff)
    else:
        group.lumps['LINEDEFS'] += struct.pack('<7H', 0, 0, 1, 0, 0, 9999, 0xffff)
    doc = extract_map(wad, group)
    assert any('invalid' in issue and 'reference' in issue for issue in doc.derived.geometry_issues)
    assert not doc.derived.sector_regions


def test_bsp_partition_side_and_sector_unions_agree_with_closed_raw_geometry(tmp_path):
    wad, group = _group(tmp_path)
    doc = extract_map(wad, group)
    shapes, info = reconstruct_bsp(group, doc.vertices, doc.linedefs, doc.sidedefs, len(doc.sectors))
    assert info['area'] == 256 * 128
    for sector in doc.sectors:
        assert rings_area_perimeter(shapes[sector.id])[0] == sector_area_perimeter(doc, sector.id)[0]


def test_a_hole_touching_its_shell_is_classified_from_a_nonshared_point():
    outer = [(0, 0), (0, 256), (256, 256), (256, 0)]
    hole = [(0, 0), (64, 32), (32, 64)]
    assert ring_depths([outer, hole]) == [0, 1]
    assert rings_area_perimeter([outer, hole])[0] == 256**2 - 1536


def test_room_union_does_not_assign_a_large_hole_to_a_nested_island():
    from tests.geometry_fixtures import square
    rings = [square(0, 0, 512, 512), list(reversed(square(128, 128, 384, 384))),
             square(224, 224, 288, 288)]
    merged = union_sector_regions([rings])
    assert rings_area_perimeter(merged)[0] == 512**2 - 256**2 + 64**2
    assert sorted(ring_depths(merged)) == [0, 1, 2]


@pytest.mark.parametrize('signature', [b'XNOD', b'ZNOD'])
def test_extended_and_compressed_nodes_preserve_supplemental_fixed_point_vertices(tmp_path, signature):
    wad, group = _group(tmp_path)
    original = extract_map(wad, group)
    segs = list(struct.iter_unpack('<HHHHHH', group.lumps['SEGS']))
    leaves, flattened = [], []
    first_seg = segs[0]
    a, b = original.vertices[first_seg[0]], original.vertices[first_seg[1]]
    new = ((a.x + b.x) / 2, (a.y + b.y) / 2)
    for leaf_index, (count, first) in enumerate(struct.iter_unpack('<HH', group.lumps['SSECTORS'])):
        part = [(v1, v2, line, side) for v1, v2, angle, line, side, offset in segs[first:first + count]]
        if leaf_index == 0:
            v1, v2, line, side = part.pop(0)
            part[:0] = [(v1, len(original.vertices), line, side), (len(original.vertices), v2, line, side)]
        leaves.append(len(part))
        flattened.extend(part)
    nodes = list(struct.iter_unpack('<hhhhhhhhhhhhHH', group.lumps['NODES']))
    body = struct.pack('<IIiiI', len(original.vertices), 1, int(new[0] * 65536), int(new[1] * 65536), len(leaves))
    body += b''.join(struct.pack('<I', count) for count in leaves)
    body += struct.pack('<I', len(flattened)) + b''.join(struct.pack('<IIHB', *seg) for seg in flattened)
    body += struct.pack('<I', len(nodes))
    for node in nodes:
        children = [0x80000000 | (value & 0x7fff) if value & 0x8000 else value for value in node[-2:]]
        body += struct.pack('<hhhhhhhhhhhhII', *node[:-2], *children)
    group.lumps['SEGS'] = group.lumps['SSECTORS'] = b''
    group.lumps['NODES'] = signature + (zlib.compress(body) if signature == b'ZNOD' else body)
    doc = extract_map(wad, group)
    assert doc.derived.bsp_geometry['node_format'] == signature.decode()
    assert doc.vertices == original.vertices  # supplemental nodes never rewrite authoring data
    assert [sector_area_perimeter(doc, sid)[0] for sid in range(2)] == [128**2, 128**2]


@pytest.mark.parametrize('data', [b'XNOD\x00', b'ZNODbroken', b'ZNOD' + zlib.compress(b'\x00')[:-1]])
def test_truncated_extended_nodes_are_explicit_failures(tmp_path, data):
    wad, group = _group(tmp_path)
    doc = extract_map(wad, group)
    group.lumps['SEGS'] = group.lumps['SSECTORS'] = b''
    group.lumps['NODES'] = data
    with pytest.raises(ValueError):
        reconstruct_bsp(group, doc.vertices, doc.linedefs, doc.sidedefs, len(doc.sectors))


def test_compiled_regions_survive_augmentation_and_derived_schema_upgrade(tmp_path):
    from data.augmentation import transform_document, jitter_heights, AUGMENT_OPS
    from wad_parser.geometry import DERIVED_VERSION
    wad, group = _group(tmp_path)
    raw = bytearray(group.lumps['SIDEDEFS'])
    struct.pack_into('<H', raw, 2 * 30 + 28, 1)
    group.lumps['SIDEDEFS'] = bytes(raw)
    source = extract_map(wad, group)
    original = source.to_dict()
    assert source.derived.raw_geometry_issues and source.derived.sector_regions
    for op in AUGMENT_OPS:
        changed = transform_document(source, op)
        assert changed.derived.sector_regions
        assert not changed.derived.geometry_issues
        assert [sector_area_perimeter(changed, sid)[0] for sid in range(2)] == [128**2, 128**2]
        assert changed.derived.raw_geometry_issues
    changed = jitter_heights(source, seed=7)
    assert changed.derived.sector_regions == source.derived.sector_regions
    legacy = source.to_dict()
    legacy['derived']['derived_version'] = DERIVED_VERSION - 1
    upgraded = MapDocument.from_dict(legacy)
    assert upgraded.derived.derived_version == DERIVED_VERSION
    assert upgraded.derived.sector_regions == source.derived.sector_regions
    assert source.to_dict() == original


@pytest.mark.parametrize('signature', [b'XGLN', b'ZGLN', b'XGL2', b'ZGL2', b'XGL3', b'ZGL3'])
def test_gl_extended_formats_decode_implied_seg_endpoints_and_fractional_nodes(signature):
    from tests.geometry_fixtures import polygon_document, square
    from wad_parser.wad_reader import MapLumpGroup
    version = int(chr(signature[3])) if signature[3:4] in {b'2', b'3'} else 1
    split = 128.5 if version == 3 else 128
    doc = polygon_document([[square(0, 0, split, 128)], [square(split, 0, 256, 128)]])
    parts = []
    for sid in range(2):
        edges = []
        for line in doc.linedefs:
            if doc.sidedefs[line.right_sidedef].sector == sid:
                edges.append((line.v1, line.v2, line.id, 0))
            elif line.left_sidedef >= 0 and doc.sidedefs[line.left_sidedef].sector == sid:
                edges.append((line.v2, line.v1, line.id, 1))
        ordered = [edges.pop(0)]
        while edges:
            index = next(i for i, edge in enumerate(edges) if edge[0] == ordered[-1][1])
            ordered.append(edges.pop(index))
        parts.append(ordered)
    body = struct.pack('<III', len(doc.vertices), 0, 2)
    body += b''.join(struct.pack('<I', len(part)) for part in parts)
    body += struct.pack('<I', sum(map(len, parts)))
    fmt = '<IIHB' if version == 1 else '<IIIB'
    body += b''.join(struct.pack(fmt, v1, 4294967295, line, side) for part in parts for v1, v2, line, side in part)
    body += struct.pack('<I', 1)
    coordinates = (split, 0, 0, 128)
    if version == 3:
        coordinates = tuple(int(value * 65536) for value in coordinates)
    body += struct.pack('<iiiihhhhhhhhII' if version == 3 else '<hhhhhhhhhhhhII',
                        *coordinates, *([0] * 8), 0x80000001, 0x80000000)
    packed = signature + (zlib.compress(body) if signature[:1] == b'Z' else body)
    # GL-only binary output stores its tree in SSECTORS, not NODES.
    group = MapLumpGroup('MAP01', 'doom', {'SSECTORS': packed, 'NODES': b'', 'SEGS': b''})
    regions, info = reconstruct_bsp(group, doc.vertices, doc.linedefs, doc.sidedefs, 2)
    assert info['node_format'] == signature.decode()
    assert info['area'] == pytest.approx(256 * 128)
    assert rings_area_perimeter(regions[0])[0] == pytest.approx(split * 128)


def test_miniseg_only_leaf_inherits_owner_without_crossing_authored_boundaries():
    from wad_parser.bsp_geometry import _gl_leaf_owners
    from tests.geometry_fixtures import two_rooms
    doc = two_rooms()
    # One authored seg and a partnered miniseg join a second mini-only leaf.
    segs = [(0, 1, 0, 0, 0, 0), (1, 2, 0, None, 0, 0), (2, 1, 0, None, 0, 0)]
    owners = _gl_leaf_owners(segs, [(2, 0), (1, 2)], [4294967295, 2, 1], doc.linedefs, doc.sidedefs)
    assert owners == [0, 0]

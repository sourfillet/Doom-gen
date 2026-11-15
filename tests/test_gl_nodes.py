import copy
import struct
import zlib

import pytest

from tests.geometry_fixtures import two_rooms
from tests.wad_builder import _assemble_wad
from wad_parser.bsp_geometry import reconstruct_bsp
from wad_parser.geometry import sector_area_perimeter
from wad_parser.map_extractor import extract_map
from wad_parser.wad_reader import MapLumpGroup, WadFile
from wad_writer.assembler import assemble_map_lumps


def gl_group(version=2):
    doc = two_rooms()
    parts = []
    flag = {1: 0x8000, 2: 0x8000, 3: 0x40000000, 5: 0x80000000}[version]
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
        if sid == 0:
            a, b, line, side = ordered.pop(0)
            ordered[:0] = [(a, flag, line, side), (flag, b, line, side)]
        parts.append(ordered)
    flattened = [seg for part in parts for seg in part]
    sentinel = 0xffffffff if version in (3, 5) else 0xffff
    packed = []
    for a, b, line, side in flattened:
        partner = next((i for i, (v1, v2, _, _) in enumerate(flattened) if v1 == b and v2 == a), sentinel)
        packed.append(struct.pack('<IIHHI' if version in (3, 5) else '<HHHHH', a, b, line, side, partner))
    y = 64 if version == 1 else 64.5
    vertices = struct.pack('<hh', 0, y) if version == 1 else struct.pack('<ii', 0, int(y * 65536))
    marker = b'' if version == 1 else (b'gNd5' if version == 5 else b'gNd2')
    counts = [(len(parts[0]), 0), (len(parts[1]), len(parts[0]))]
    leaves = b''.join(struct.pack('<II' if version in (3, 5) else '<HH', *part) for part in counts)
    nodes = struct.pack('<hhhhhhhhhhhhII' if version == 5 else '<hhhhhhhhhhhhHH',
                        128, 0, 0, 128, *([0] * 8),
                        *([0x80000001, 0x80000000] if version == 5 else [0x8001, 0x8000]))
    lumps = dict(assemble_map_lumps(doc))
    lumps.update({'GL_VERT': marker + vertices, 'GL_SEGS': (b'gNd3' if version == 3 else b'') + b''.join(packed),
                  'GL_SSECT': (b'gNd3' if version == 3 else b'') + leaves, 'GL_NODES': nodes})
    return doc, MapLumpGroup('MAP01', 'doom', lumps)


@pytest.mark.parametrize('version', [1, 2, 3, 5])
@pytest.mark.parametrize('name', ['MAP01', 'MAP100'])
def test_reader_groups_standalone_nodes_and_recovers_geometry_without_raw_mutation(tmp_path, version, name):
    original, group = gl_group(version)
    normal = [(key, value) for key, value in group.lumps.items() if not key.startswith('GL_')]
    checksum = zlib.adler32(group.lumps['LINEDEFS'], zlib.adler32(group.lumps['VERTEXES']))
    metadata = f'CHECKSUM=0x{checksum:08X}\n'
    if len(name) > 5:
        metadata = f'LEVEL={name}\n' + metadata
    path = tmp_path / 'legacy-gl.wad'
    path.write_bytes(_assemble_wad([(name, b''), *normal,
                                   ('GL_LEVEL' if len(name) > 5 else 'GL_' + name, metadata.encode()),
                                   *[(key, value) for key, value in group.lumps.items() if key.startswith('GL_')],
                                   ('GX_TEST', b'preserved'), ('GLOBAL', b'unrelated')]))
    wad = WadFile(str(path))
    groups = wad.find_map_groups()
    assert len(groups) == 1 and groups[0].name == name
    assert groups[0].lumps['GX_TEST'] == b'preserved'
    before = copy.deepcopy(groups[0].lumps)
    doc = extract_map(wad, groups[0])
    assert doc.derived.bsp_geometry['node_format'] == f'GLv{version}'
    assert [sector_area_perimeter(doc, sid)[0] for sid in range(2)] == [128**2, 128**2]
    assert doc.vertices == original.vertices
    assert groups[0].lumps == before


@pytest.mark.parametrize('mutation,match', [
    ('checksum', 'checksum'), ('vertex', 'vertex'), ('partner', 'partner'),
    ('winding', 'winding'), ('range', 'subsector'), ('truncated', 'truncated'),
    ('v4', 'unsupported'), ('missing', 'incomplete'),
])
def test_bad_gl_data_is_rejected_with_a_diagnostic(mutation, match):
    doc, group = gl_group()
    if mutation == 'checksum':
        group.gl_metadata['CHECKSUM'] = '0x00000000'
    elif mutation == 'vertex':
        data = bytearray(group.lumps['GL_SEGS'])
        struct.pack_into('<H', data, 0, len(doc.vertices))  # lacks GL vertex flag
        group.lumps['GL_SEGS'] = bytes(data)
    elif mutation == 'partner':
        data = bytearray(group.lumps['GL_SEGS'])
        struct.pack_into('<H', data, 8, 1234)
        group.lumps['GL_SEGS'] = bytes(data)
    elif mutation == 'winding':
        data = bytearray(group.lumps['GL_SEGS'])
        struct.pack_into('<H', data, 2, 1)
        group.lumps['GL_SEGS'] = bytes(data)
    elif mutation == 'range':
        group.lumps['GL_SSECT'] = struct.pack('<HH', 1000, 0)
    elif mutation == 'truncated':
        group.lumps['GL_VERT'] += b'x'
    elif mutation == 'v4':
        group.lumps['GL_VERT'] = b'gNd4' + group.lumps['GL_VERT'][4:]
    elif mutation == 'missing':
        del group.lumps['GL_NODES']
    with pytest.raises(ValueError, match=match):
        reconstruct_bsp(group, doc.vertices, doc.linedefs, doc.sidedefs, 2)


def test_gl_marker_cannot_attach_to_the_wrong_map(tmp_path):
    _, group = gl_group()
    normal = [(key, value) for key, value in group.lumps.items() if not key.startswith('GL_')]
    path = tmp_path / 'unmatched-gl.wad'
    path.write_bytes(_assemble_wad([('MAP01', b''), *normal, ('GL_MAP02', b''),
                                   *[(key, value) for key, value in group.lumps.items() if key.startswith('GL_')]]))
    maps = WadFile(str(path)).find_map_groups()
    assert len(maps) == 1 and not any(key.startswith('GL_') for key in maps[0].lumps)


def test_a_partial_replacement_gl_block_cannot_reuse_stale_nodes(tmp_path):
    doc, group = gl_group()
    normal = [(key, value) for key, value in group.lumps.items() if not key.startswith('GL_')]
    gl = [(key, value) for key, value in group.lumps.items() if key.startswith('GL_')]
    path = tmp_path / 'replaced-gl.wad'
    path.write_bytes(_assemble_wad([('MAP01', b''), *normal, ('GL_MAP01', b''), *gl,
                                   ('GX_OLD', b'stale'), ('GL_MAP01', b''),
                                   ('GL_VERT', group.lumps['GL_VERT'])]))
    replacement = WadFile(str(path)).find_map_groups()[0]
    assert 'GL_NODES' not in replacement.lumps and 'GX_OLD' not in replacement.lumps
    with pytest.raises(ValueError, match='incomplete'):
        reconstruct_bsp(replacement, doc.vertices, doc.linedefs, doc.sidedefs, 2)

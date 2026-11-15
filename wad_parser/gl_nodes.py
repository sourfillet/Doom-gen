"""Checked standalone glBSP v1/v2/v3/v5 records.

Layouts, vertex flags, mini-seg winding and optional Adler-32 map checksum:
https://glbsp.sourceforge.net/specs.php
Supplemental vertices never enter the authored VERTEXES array.
"""
from __future__ import annotations

import struct
import zlib


def decode_gl_nodes(group, vertices):
    names = ('GL_VERT', 'GL_SEGS', 'GL_SSECT', 'GL_NODES')
    if not all(name in group.lumps for name in names):
        raise ValueError('incomplete standalone GL BSP lumps')
    raw_vertices, raw_segs, raw_leaves, raw_nodes = (group.lumps[name] for name in names)
    checksum = group.gl_metadata.get('CHECKSUM')
    if checksum is not None:
        try:
            expected = int(checksum, 16)
        except ValueError as exc:
            raise ValueError('invalid standalone GL BSP checksum') from exc
        actual = zlib.adler32(group.lumps.get('LINEDEFS', b''),
                             zlib.adler32(group.lumps.get('VERTEXES', b'')))
        if actual != expected:
            raise ValueError('standalone GL BSP checksum does not match authored geometry')
    version = 1
    if raw_vertices[:4] in {b'gNd2', b'gNd5'}:
        version = int(chr(raw_vertices[3]))
        raw_vertices = raw_vertices[4:]
    elif raw_vertices.startswith(b'gNd'):
        raise ValueError('unsupported standalone GL BSP version: ' + repr(raw_vertices[:4]))
    if version == 2 and raw_segs.startswith(b'gNd3'):
        if not raw_leaves.startswith(b'gNd3'):
            raise ValueError('inconsistent standalone GL v3 subsector header')
        version = 3
        raw_segs, raw_leaves = raw_segs[4:], raw_leaves[4:]

    def records(raw, fmt):
        if len(raw) % struct.calcsize(fmt):
            raise ValueError('truncated standalone GL BSP record')
        return list(struct.iter_unpack(fmt, raw))

    positions = {vertex.id: (vertex.x, vertex.y) for vertex in vertices}
    first_added = max(positions, default=-1) + 1
    for index, (x, y) in enumerate(records(raw_vertices, '<hh' if version == 1 else '<ii'), first_added):
        positions[index] = (x, y) if version == 1 else (x / 65536, y / 65536)
    vertex_flag = {1: 0x8000, 2: 0x8000, 3: 0x40000000, 5: 0x80000000}[version]
    def vertex_index(value):
        if value & vertex_flag:
            value = first_added + (value & (vertex_flag - 1))
        elif value >= first_added:
            raise ValueError('invalid standalone GL BSP original vertex reference')
        if value not in positions:
            raise ValueError('invalid standalone GL BSP vertex reference')
        return value
    leaves = records(raw_leaves, '<II' if version in (3, 5) else '<HH')
    count = sum(size for size, _ in leaves)
    seg_format = '<IIHHI' if version in (3, 5) else '<HHHHH'
    if len(raw_segs) != count * struct.calcsize(seg_format):
        raise ValueError('standalone GL BSP seg length does not match subsector counts '
                         f'(expected {struct.calcsize(seg_format)}-byte records)')
    segs, partners = [], []
    sentinel = 0xffffffff if version in (3, 5) else 0xffff
    for v1, v2, line, side, partner in records(raw_segs, seg_format):
        mini = line in (0xffff, 0xffffffff)
        segs.append((vertex_index(v1), vertex_index(v2), 0, None if mini else line, 0 if mini else side, 0))
        partners.append(0xffffffff if partner == sentinel else partner)
    for index, partner in enumerate(partners):
        if partner == 0xffffffff:
            continue
        if partner >= len(segs) or partner == index or partners[partner] != index:
            raise ValueError('invalid standalone GL BSP partner reference')
        if segs[index][:2] != tuple(reversed(segs[partner][:2])):
            # Distinct supplemental indices can represent the same point.
            if [positions[v] for v in segs[index][:2]] != [positions[v] for v in reversed(segs[partner][:2])]:
                raise ValueError('standalone GL BSP partners do not share reversed endpoints')
    if not segs or not leaves:
        raise ValueError('standalone GL BSP has no subsectors')
    covered = set()
    for count, first in leaves:
        if not count or first + count > len(segs):
            raise ValueError('invalid standalone GL BSP subsector range')
        for index in range(first, first + count):
            if index in covered:
                raise ValueError('overlapping standalone GL BSP subsector ranges')
            covered.add(index)
            following = first + (index - first + 1) % count
            if positions[segs[index][1]] != positions[segs[following][0]]:
                raise ValueError('standalone GL BSP subsector is not a closed winding')
    if len(covered) != len(segs):
        raise ValueError('standalone GL BSP subsectors do not cover segs')
    nodes = records(raw_nodes, '<hhhhhhhhhhhhII' if version == 5 else '<hhhhhhhhhhhhHH')
    return (positions, segs, leaves, nodes, partners, (0x80000000 if version == 5 else 0x8000),
            version, struct.calcsize(seg_format))

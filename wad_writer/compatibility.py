"""Static Doom 1.9 format limits; passing is not a render-limit certificate.

Chocolate Doom retains signed short BLOCKMAP offsets and line indexes:
https://github.com/chocolate-doom/chocolate-doom/blob/master/src/doom/p_maputl.c
View-dependent visplane/drawseg/openings limits still require engine tests.
"""
from __future__ import annotations

import re
import struct

TARGET_PORTS = ('zdoom', 'vanilla')
_VANILLA_MAP_RE = re.compile(r'^(?:MAP(?:0[1-9]|[1-9]\d)|E[1-9]M[1-9])$')
_EXTENDED_NODE_SIGNATURES = (b'XNOD', b'ZNOD', b'XGLN', b'ZGLN', b'XGL2', b'ZGL2', b'XGL3', b'ZGL3')


def vanilla_issues(group):
    lumps, issues = group.lumps, []
    if not _VANILLA_MAP_RE.fullmatch(group.name.upper()):
        issues.append(f'map marker {group.name!r} is outside classic MAP01..MAP99/ExMy slots')
    sizes = {'THINGS': 10, 'VERTEXES': 4, 'LINEDEFS': 14, 'SIDEDEFS': 30, 'SECTORS': 26,
             'SEGS': 12, 'SSECTORS': 4, 'NODES': 28}
    counts = {}
    for name, size in sizes.items():
        raw = lumps.get(name, b'')
        if len(raw) % size:
            issues.append(f'{name} is not classic Doom record data')
        counts[name] = len(raw) // size
    if group.format != 'doom':
        issues.append('vanilla requires Doom binary map format')
    for name in ('VERTEXES', 'LINEDEFS', 'SIDEDEFS', 'SECTORS', 'SEGS', 'SSECTORS', 'NODES'):
        if counts[name] > 32767:
            issues.append(f'{name} count {counts[name]} exceeds signed/classic index range')
    if not counts['SEGS'] or not counts['SSECTORS']:
        issues.append('vanilla needs compiled SEGS/SSECTORS; install zdbsp')
    if (lumps.get('NODES', b'').startswith(_EXTENDED_NODE_SIGNATURES)
            or lumps.get('SSECTORS', b'').startswith(_EXTENDED_NODE_SIGNATURES)):
        issues.append('vanilla does not support extended/compressed nodes')
    if not issues:
        for v1, v2, flags, special, tag, right, left in struct.iter_unpack('<7H', lumps['LINEDEFS']):
            if (max(v1, v2) >= counts['VERTEXES'] or right >= counts['SIDEDEFS']
                    or left != 65535 and left >= counts['SIDEDEFS']):
                issues.append('invalid classic LINEDEFS reference')
                break
        for (sector,) in struct.iter_unpack('<28xH', lumps['SIDEDEFS']):
            if sector >= counts['SECTORS']:
                issues.append('invalid classic SIDEDEFS sector')
                break
        for v1, v2, angle, line, direction, offset in struct.iter_unpack('<6H', lumps['SEGS']):
            if max(v1, v2) >= counts['VERTEXES'] or line >= counts['LINEDEFS'] or direction > 1:
                issues.append('invalid classic SEGS reference')
                break
        for count, first in struct.iter_unpack('<hh', lumps['SSECTORS']):
            if count <= 0 or first < 0 or first + count > counts['SEGS']:
                issues.append('invalid classic SSECTORS seg range')
                break
        nodes = list(struct.iter_unpack('<24xHH', lumps['NODES']))
        pending = [len(nodes) - 1] if nodes else [32768]
        visited, leaves = set(), set()
        while pending:
            child = pending.pop()
            if child & 32768:
                leaf = child & 32767
                if leaf >= counts['SSECTORS'] or leaf in leaves:
                    issues.append('invalid or repeated classic BSP leaf')
                    break
                leaves.add(leaf)
            else:
                if child >= len(nodes) or child in visited:
                    issues.append('invalid or cyclic classic BSP node')
                    break
                visited.add(child)
                pending.extend(nodes[child])
        if len(leaves) != counts['SSECTORS']:
            issues.append('classic BSP does not cover every subsector')
    required = (counts['SECTORS'] ** 2 + 7) // 8
    if len(lumps.get('REJECT', b'')) < required:
        issues.append(f'REJECT needs at least {required} bytes')
    return issues + blockmap_issues(group, vanilla=True, required=True)


def blockmap_issues(group, *, vanilla=False, required=False):
    """Check supplied binary BLOCKMAPs, including unsigned offset wrapping.

    Empty lumps can be rebuilt by modern ports. A nonempty corrupt table must
    not be published as valid compiled data. Lump length may exceed 64K words
    if all list starts remain representable and every list is valid.
    """
    if group.format not in {'doom', 'hexen'}:
        return []  # UDMF has no binary LINEDEFS index table to check here.
    lumps, issues = group.lumps, []
    raw = lumps.get('BLOCKMAP', b'')
    if not raw and not required:
        return []
    if len(raw) < 8 or len(raw) % 2:
        issues.append('vanilla needs a compiled BLOCKMAP' if vanilla else 'invalid compiled BLOCKMAP size')
        return issues
    words = struct.unpack(f'<{len(raw) // 2}H', raw)
    width, height = words[2:4]
    table_end = 4 + width * height
    if (not width or not height or table_end > len(words)
            or vanilla and (width > 32767 or height > 32767)):
        issues.append('invalid BLOCKMAP dimensions/offset table')
        return issues
    offsets = set(words[4:table_end])
    high = max(offsets, default=0)
    if vanilla and high > 32767:
        issues.append(f'BLOCKMAP list offset {high} exceeds vanilla signed 16-bit range (32767)')
        return issues
    line_count = len(lumps.get('LINEDEFS', b'')) // (16 if group.format == 'hexen' else 14)
    for offset in offsets:
        if not table_end <= offset < len(words):
            issues.append('BLOCKMAP list offset is outside list data')
            break
        # Shared/suffix lists are legal. Bound every walk by the lump length.
        for index in range(offset, len(words)):
            value = words[index]
            if value == 65535:
                break
            if value >= line_count or vanilla and value > 32767:
                issues.append(f'BLOCKMAP contains invalid {"vanilla " if vanilla else ""}linedef index {value}')
                break
        else:
            issues.append('BLOCKMAP list has no terminator')
        if issues:
            break
    return issues


def require_target_port(wad, target_port):
    if target_port not in TARGET_PORTS:
        raise ValueError(f'unknown target port {target_port!r}; choose zdoom or vanilla')
    groups = wad.find_map_groups()
    if target_port == 'zdoom':
        failures = [f'{group.name}: {issue}' for group in groups
                    for issue in blockmap_issues(group)]
        if not groups:
            failures.append('no maps in compiled output')
        if failures:
            raise ValueError('zdoom compatibility rejected output: ' + '; '.join(failures))
        return None
    records = [{'map': group.name, 'issues': vanilla_issues(group)} for group in groups]
    failures = [f"{record['map']}: {issue}" for record in records for issue in record['issues']]
    if not records:
        failures.append('no maps in compiled output')
    if failures:
        raise ValueError('vanilla compatibility rejected output: ' + '; '.join(failures))
    return {'target': 'vanilla', 'maps': [record['map'] for record in records],
            'scope': 'Static format/index/BLOCKMAP checks; view-dependent engine limits still require playtesting.'}

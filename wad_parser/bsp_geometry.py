"""Recover filled classic Doom sector regions from the compiled BSP.

Raw authoring lines can be unclosed while the engine's subsectors still
define playable space. Clip each leaf by its BSP path and front-facing segs,
then union leaves by the sector assigned by its first seg. No raw elements
are edited. Normal and GL extended/compressed nodes share checked clipping.
"""
from __future__ import annotations

import struct
import zlib
from collections import defaultdict

GRID = 1 / 65536  # Doom world-coordinate fixed-point precision
EXTENDED_SIGNATURES = {b'XNOD', b'ZNOD', b'XGLN', b'ZGLN', b'XGL2', b'ZGL2', b'XGL3', b'ZGL3'}


def _extended_nodes(raw, vertices):
    """Decode extended normal/GL layouts emitted by zdbsp (processor.cpp).

    https://github.com/rheit/zdbsp/blob/master/processor.cpp
    Supplemental fixed-point vertices belong only to the derived layer.
    """
    payload = raw[4:]
    gl = raw[1:3] == b'GL'
    version = int(chr(raw[3])) if gl and raw[3:4] in {b'2', b'3'} else 1
    if raw[:1] == b'Z':
        try:
            decoder = zlib.decompressobj()
            payload = decoder.decompress(payload, 128 * 1024 * 1024 + 1)
            if len(payload) > 128 * 1024 * 1024 or not decoder.eof or decoder.unused_data:
                raise ValueError('invalid, truncated, or oversized compressed BSP')
        except zlib.error as exc:
            raise ValueError('invalid compressed BSP stream') from exc
    cursor = 0
    def read(fmt, count=1):
        nonlocal cursor
        size = struct.calcsize(fmt) * count
        if cursor + size > len(payload):
            raise ValueError('truncated extended BSP record')
        values = list(struct.iter_unpack(fmt, memoryview(payload)[cursor:cursor + size]))
        cursor += size
        return values
    def integer():
        return read('<I')[0][0]
    original, added = integer(), integer()
    if original > len(vertices):
        raise ValueError('extended BSP references missing original vertices')
    positions = {vertex.id: (vertex.x, vertex.y) for vertex in vertices if vertex.id < original}
    for index, (x, y) in enumerate(read('<ii', added), original):
        positions[index] = (x / 65536, y / 65536)
    leaves, first = [], 0
    for (count,) in read('<I', integer()):
        leaves.append((count, first))
        first += count
    partners = None
    if gl:
        records = read('<IIHB' if version == 1 else '<IIIB', integer())
        partners = [record[1] for record in records]
        sentinel = 65535 if version == 1 else 4294967295
        segs = []
        if first != len(records):
            raise ValueError('GL BSP subsectors do not cover segs')
        for count, start in leaves:
            if not count:
                raise ValueError('GL BSP has an empty subsector')
            for index in range(start, start + count):
                v1, partner, line, side = records[index]
                if partner != 4294967295 and partner >= len(records):
                    raise ValueError('invalid GL BSP partner reference')
                v2 = records[start + (index - start + 1) % count][0]
                segs.append((v1, v2, 0, None if line == sentinel else line, side, 0))
    else:
        segs = [(v1, v2, 0, line, side, 0) for v1, v2, line, side in read('<IIHB', integer())]
    if first != len(segs):
        raise ValueError('extended BSP subsector counts do not cover segs')
    nodes = read('<iiiihhhhhhhhII' if version == 3 else '<hhhhhhhhhhhhII', integer())
    if version == 3:
        nodes = [tuple(value / 65536 for value in node[:4]) + node[4:] for node in nodes]
    if cursor != len(payload):
        raise ValueError('unexpected trailing extended BSP data')
    return positions, segs, leaves, nodes, partners


def _gl_leaf_owners(segs, leaves, partners, linedefs, sidedefs):
    """Minisegs have no authored side; propagate ownership across them only."""
    parent = list(range(len(leaves)))
    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    segment_leaf, known = {}, {}
    for leaf, (count, start) in enumerate(leaves):
        for index in range(start, start + count):
            segment_leaf[index] = leaf
            line, direction = segs[index][3:5]
            if line is not None and leaf not in known:
                if line >= len(linedefs) or direction not in (0, 1):
                    raise ValueError('invalid GL BSP authored side')
                side = linedefs[line].left_sidedef if direction else linedefs[line].right_sidedef
                if not 0 <= side < len(sidedefs):
                    raise ValueError('GL BSP seg has no front sector')
                known[leaf] = sidedefs[side].sector
    for index, partner in enumerate(partners):
        if segs[index][3] is None and partner != 4294967295:
            if segs[partner][3] is not None:
                raise ValueError('GL miniseg partner is an authored boundary')
            a, b = root(segment_leaf[index]), root(segment_leaf[partner])
            parent[a] = b
    owners = {}
    for leaf, owner in known.items():
        component = root(leaf)
        if component in owners and owners[component] != owner:
            raise ValueError('GL miniseg component has conflicting sector owners')
        owners[component] = owner
    if any(root(leaf) not in owners for leaf in range(len(leaves))):
        raise ValueError('GL miniseg component has no authored sector owner')
    return [owners[root(leaf)] for leaf in range(len(leaves))]


def _union(polygons):
    from shapely import union_all
    from shapely.errors import GEOSException
    try:
        return union_all(polygons, grid_size=GRID)
    except GEOSException as exc:
        raise ValueError(f'cannot union derived geometry: {exc}') from exc


def geometry_rings(geometry):
    from shapely import Polygon
    from wad_parser.geometry import signed_area
    rings = []
    parts = [geometry] if geometry.geom_type == 'Polygon' else list(geometry.geoms)
    for polygon in sorted(parts, key=lambda poly: -poly.area):
        if polygon.geom_type != 'Polygon' or polygon.area < GRID:
            continue
        shell = list(polygon.exterior.coords)[:-1]
        rings.append(shell if signed_area(shell) < 0 else list(reversed(shell)))
        for interior in sorted(polygon.interiors, key=lambda ring: -Polygon(ring).area):
            hole = list(interior.coords)[:-1]
            rings.append(hole if signed_area(hole) > 0 else list(reversed(hole)))
    return rings


def component_polygons(rings):
    """Return (shell index, polygon) for every disconnected filled component."""
    from shapely import Polygon
    from wad_parser.geometry import ring_depths
    depths = ring_depths(rings)
    shells = {index: Polygon(ring) for index, (ring, depth) in enumerate(zip(rings, depths)) if depth % 2 == 0}
    holes = defaultdict(list)
    for ring, depth in zip(rings, depths):
        if depth % 2:
            hole = Polygon(ring)
            containers = [index for index, shell in shells.items()
                          if shell.area >= hole.area and shell.covers(hole)]
            if not containers:
                raise ValueError('derived hole has no containing shell')
            holes[min(containers, key=lambda index: shells[index].area)].append(ring)
    result = []
    for index, shell in shells.items():
        polygon = Polygon(list(shell.exterior.coords), holes[index])
        if not polygon.is_valid:
            raise ValueError('invalid derived sector shell/hole assignment')
        result.append((index, polygon))
    return result


def union_sector_regions(regions):
    """Merge effective sector regions into one room without losing holes."""
    polygons = [polygon for rings in regions for _, polygon in component_polygons(rings)]
    geometry = _union(polygons)
    if not geometry.is_valid:
        raise ValueError('invalid union of derived room regions')
    return geometry_rings(geometry)


def _clip(points, a, vector, side):
    if not points:
        return []
    def distance(point):
        return side * (vector[0] * (point[1] - a[1]) - vector[1] * (point[0] - a[0]))
    result = []
    previous, old = points[-1], distance(points[-1])
    for point in points:
        new = distance(point)
        if (new >= -1e-8) != (old >= -1e-8):
            fraction = old / (old - new)
            result.append((previous[0] + (point[0] - previous[0]) * fraction,
                           previous[1] + (point[1] - previous[1]) * fraction))
        if new >= -1e-8:
            result.append(point)
        previous, old = point, new
    return result


def reconstruct_bsp(group, vertices, linedefs, sidedefs, n_sectors):
    """Return (coordinate rings per sector, diagnostics), or None without BSP.

    Invalid/unsupported compiled data raises ValueError so callers can retain
    raw-boundary diagnostics instead of silently inventing a closure.
    """
    from shapely import Polygon, union_all
    from wad_parser.geometry import signed_area
    raw_segs, raw_leaves, raw_nodes = (group.lumps.get(name, b'') for name in ('SEGS', 'SSECTORS', 'NODES'))
    raw_nodes = group.lumps.get('ZNODES') or raw_nodes
    if not raw_nodes and not raw_segs and raw_leaves[:4] in EXTENDED_SIGNATURES:
        raw_nodes = raw_leaves  # zdbsp GL-only output embeds the tree here
    extended = raw_nodes[:4] in EXTENDED_SIGNATURES
    gl_owners = None
    gl_record_bytes = None
    node_format = raw_nodes[:4].decode('ascii') if extended else 'classic'
    method = 'extended-bsp-halfplanes-v1' if extended else 'classic-bsp-halfplanes-v1'
    if extended:
        positions, segs, leaves, nodes, partners = _extended_nodes(raw_nodes, vertices)
        if partners is not None:
            gl_owners = _gl_leaf_owners(segs, leaves, partners, linedefs, sidedefs)
        leaf_flag = 0x80000000
    else:
        if not raw_segs or not raw_leaves:
            if not any(name in group.lumps for name in ('GL_VERT', 'GL_SEGS', 'GL_SSECT', 'GL_NODES')):
                return None
            from wad_parser.gl_nodes import decode_gl_nodes
            positions, segs, leaves, nodes, partners, leaf_flag, version, gl_record_bytes = decode_gl_nodes(group, vertices)
            gl_owners = _gl_leaf_owners(segs, leaves, partners, linedefs, sidedefs)
            node_format, method = f'GLv{version}', 'standalone-gl-bsp-halfplanes-v1'
        else:
            if len(raw_segs) % 12 or len(raw_leaves) % 4 or len(raw_nodes) % 28:
                raise ValueError('truncated classic BSP record')
            positions = {v.id: (v.x, v.y) for v in vertices}
            segs = list(struct.iter_unpack('<HHHHHH', raw_segs))
            leaves = list(struct.iter_unpack('<HH', raw_leaves))
            nodes = [struct.unpack_from('<hhhhhhhhhhhhHH', raw_nodes, index)
                     for index in range(0, len(raw_nodes), 28)]
            leaf_flag = 0x8000
    if not positions:
        raise ValueError('BSP has no vertices')
    xs, ys = zip(*positions.values())
    bounds = [(min(xs) - 1, min(ys) - 1), (max(xs) + 1, min(ys) - 1),
              (max(xs) + 1, max(ys) + 1), (min(xs) - 1, max(ys) + 1)]
    sectors = defaultdict(list)
    seen_leaves, seen_nodes, collapsed = set(), set(), 0
    stack = [(len(nodes) - 1, bounds)] if nodes else [(leaf_flag | i, bounds) for i in range(len(leaves))]
    while stack:
        index, points = stack.pop()
        if index & leaf_flag:
            leaf = index & (leaf_flag - 1)
            if leaf >= len(leaves) or leaf in seen_leaves:
                raise ValueError('invalid or multiply referenced BSP subsector')
            seen_leaves.add(leaf)
            count, first = leaves[leaf]
            if not count or first + count > len(segs):
                raise ValueError('invalid BSP subsector seg range')
            owner = gl_owners[leaf] if gl_owners is not None else None
            for v1, v2, angle, line_id, direction, offset in segs[first:first + count]:
                if (v1 not in positions or v2 not in positions or direction not in {0, 1}
                        or line_id is not None and line_id >= len(linedefs)):
                    raise ValueError('invalid classic BSP seg reference')
                if line_id is not None:
                    line = linedefs[line_id]
                    side = line.left_sidedef if direction else line.right_sidedef
                    if not 0 <= side < len(sidedefs):
                        raise ValueError('BSP seg has no front sector')
                    if owner is None:
                        owner = sidedefs[side].sector
                a, b = positions[v1], positions[v2]
                points = _clip(points, a, (b[0] - a[0], b[1] - a[1]), -1)
            if not 0 <= owner < n_sectors:
                raise ValueError('invalid BSP sector reference')
            if len(points) < 3 or abs(signed_area(points)) < GRID:
                collapsed += 1
                continue
            # Half-plane intersections are convex. Normalize duplicate/tiny
            # floating-point turns before precision snapping and union.
            polygon = Polygon(points).convex_hull
            if polygon.area - abs(signed_area(points)) > max(GRID, polygon.length * GRID):
                raise ValueError('BSP clipping lost convexity beyond fixed-point tolerance')
            sectors[owner].append(polygon)
        else:
            if index >= len(nodes) or index in seen_nodes:
                raise ValueError('invalid or cyclic BSP node reference')
            seen_nodes.add(index)
            node = nodes[index]
            x, y, dx, dy = node[:4]
            if dx == dy == 0:
                raise ValueError('zero-length BSP partition')
            stack.append((node[-2], _clip(points, (x, y), (dx, dy), -1)))
            stack.append((node[-1], _clip(points, (x, y), (dx, dy), 1)))
    if len(seen_leaves) != len(leaves):
        raise ValueError('BSP does not cover every subsector')
    regions = [[] for _ in range(n_sectors)]
    merged = []
    for sid, cells in sectors.items():
        geometry = _union(cells)
        if not geometry.is_valid:
            raise ValueError('invalid BSP sector union')
        merged.append(geometry)
        regions[sid] = geometry_rings(geometry)
    area = sum(geometry.area for geometry in merged)
    if area - _union(merged).area > max(GRID, area * 1e-9):
        raise ValueError('compiled BSP sector regions overlap')
    return regions, {'method': method, 'node_format': node_format,
                     **({'seg_record_bytes': gl_record_bytes} if gl_record_bytes is not None else {}),
                     'subsectors': len(leaves),
                     'collapsed_subsectors': collapsed, 'coordinate_grid': GRID,
                     'sectors_with_geometry': sum(bool(parts) for parts in regions), 'area': area}

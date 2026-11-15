"""Geometry assembler: a RoomGraph -> a playable MapDocument. The
graph-native analog of the vectorizer - it turns abstract room structure
into real sectors, walls, corridors and doorways.

Rooms retain their requested sizes in a compact grid with routing lanes.
Bounded deterministic routing orders place corridors and bend junctions;
dedicated door slabs and teleport pads realize the requested connection types.
Every required edge has a realization record, and callers reject missing edges.
"""
from __future__ import annotations

import math
import random
from collections import deque
from copy import deepcopy
from dataclasses import replace

from wad_parser.map_extractor import compute_derived
from wad_parser.types import (
    SCHEMA_VERSION, MapDocument, MapMeta, Thing, ThingFlags, Vertex, resolve_doomed,
)
from roomgraph.types import RoomGraph, Room, Connection
from roomgraph.building import (
    CEIL, FLOOR, WALL, GeometryBuilder, complement_spans, emit_corridor,
)
from roomgraph.teleports import finalize_connections
from roomgraph.variation import (PROFILES, VERSION, COMPACT_GAP, axis_positions,
                                compact_axis_positions, opening_center, room_outline)

SLOT = 768               # grid cell; rooms sit inside with a gap for corridors
CORR_W = 160             # corridor width
MIN_ROOM, MAX_ROOM = 192, SLOT - 192
_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
_complement = complement_spans


def _layout(graph: RoomGraph, seed=0) -> dict[int, tuple[int, int]]:
    adj = graph.neighbors()
    rng = random.Random(seed)
    placed: dict[int, tuple[int, int]] = {}
    occupied: dict[tuple[int, int], int] = {}

    def compactness(cell):
        xs = [x for x, y in occupied] + [cell[0]]
        ys = [y for x, y in occupied] + [cell[1]]
        width, height = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
        return max(width, height), width * height, abs(cell[0]) + abs(cell[1])

    def nearest_free(gx, gy):
        for radius in range(1, 400):
            candidates = []
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    c = (gx + dx, gy + dy)
                    if c not in occupied:
                        candidates.append(c)
            if candidates:
                return min(candidates, key=compactness)
        raise ValueError('no free room layout cell within the search bound')

    for root in (r.id for r in graph.rooms):
        if root in placed:
            continue
        placed[root] = (0, 0) if not occupied else nearest_free(0, 0)
        occupied[placed[root]] = root
        queue = [root]
        while queue:
            r = queue.pop(0)
            gx, gy = placed[r]
            neighbors = sorted(adj[r])
            if seed:
                rng.shuffle(neighbors)
            for nb in neighbors:
                if nb in placed:
                    continue
                candidates = [(gx + d[0], gy + d[1]) for d in _DIRS
                              if (gx + d[0], gy + d[1]) not in occupied]
                cell = min(candidates, key=compactness) if candidates else None
                if cell is None:
                    cell = nearest_free(gx, gy)
                placed[nb] = cell
                occupied[cell] = nb
                queue.append(nb)
    # Empty lanes between room cells give the router space to reach interior
    # nodes. Balanced extents also prevent a chain becoming an enormous row
    # that cannot fit the binary Doom coordinate range.
    return {rid: (2 * x, 2 * y) for rid, (x, y) in placed.items()}


def _room_dims(room) -> tuple[float, float]:
    w0, h0 = room.size
    if w0 <= 0 or h0 <= 0:
        return (384.0, 384.0)
    aspect = min(max(w0 / h0, 0.45), 2.2)
    base = min(max(math.sqrt(max(room.area, 1.0)), MIN_ROOM), 4096)
    w = min(max(base * math.sqrt(aspect), MIN_ROOM), 4096)
    h = min(max(base / math.sqrt(aspect), MIN_ROOM), 4096)
    return (w, h)


def _route_once(graph, placed):
    """Route grid paths around occupied cells, adding explicit bend sectors."""
    original = list(graph.connections)
    graph = deepcopy(graph)
    graph.connections = []
    occupied = set(placed.values())
    reserved = set()
    plans, plan_conn_ids = [], []
    room_by_id = {room.id: room for room in graph.rooms}
    for edge in original:
        if edge.kind == 'teleport':
            cid = len(graph.connections)
            graph.connections.append(edge)
            plans.append([(edge.a, edge.b)])
            plan_conn_ids.append([cid])
            continue
        start, end = placed[edge.a], placed[edge.b]
        xs, ys = zip(*occupied)
        bounds = min(xs) - 3, min(ys) - 3, max(xs) + 3, max(ys) + 3
        previous = {start: None}
        queue = deque([start])
        while queue and end not in previous:
            current = queue.popleft()
            for dx, dy in _DIRS:
                candidate = current[0] + dx, current[1] + dy
                if not bounds[0] <= candidate[0] <= bounds[2] or not bounds[1] <= candidate[1] <= bounds[3]:
                    continue
                if candidate in previous or candidate != end and (candidate in occupied or candidate in reserved):
                    continue
                previous[candidate] = current
                queue.append(candidate)
        if end not in previous:
            plans.append([])
            plan_conn_ids.append([])
            continue
        cells, current = [], end
        while current is not None:
            cells.append(current)
            current = previous[current]
        cells.reverse()
        reserved.update(cells[1:-1])
        waypoints = [edge.a]
        for index in range(1, len(cells) - 1):
            before, here, after = cells[index - 1:index + 2]
            if (here[0] - before[0], here[1] - before[1]) == (after[0] - here[0], after[1] - here[1]):
                continue
            rid = len(graph.rooms)
            fraction = index / (len(cells) - 1)
            floor = round((room_by_id[edge.a].floor_h * (1 - fraction) + room_by_id[edge.b].floor_h * fraction) / 8) * 8
            width = max(192, edge.width + 64)
            room = Room(rid, [], floor, floor + 128, width**2, (0, 0), (0, 0, width, width), role='corridor')
            graph.rooms.append(room)
            room_by_id[rid] = room
            placed[rid] = here
            occupied.add(here)
            waypoints.append(rid)
        waypoints.append(edge.b)
        segments = list(zip(waypoints, waypoints[1:]))
        plans.append(segments)
        cids = []
        for index, (a, b) in enumerate(segments):
            cids.append(len(graph.connections))
            graph.connections.append(Connection(a, b, edge.kind if index == 0 else 'corridor', edge.width))
        plan_conn_ids.append(cids)
    return graph, original, plans, plan_conn_ids


def _route_connections(graph, placed, max_attempts=12):
    """Retry bounded edge orders before accepting a greedy routing failure.

    Early long corridors can fence off another room's required ports. Short
    edges first and a few local-RNG permutations recover many such cases;
    unsuccessful routing remains explicit and never invents connections.
    """
    edges = graph.connections
    indices = list(range(len(edges)))
    def length(index):
        edge = edges[index]
        a, b = placed[edge.a], placed[edge.b]
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    orders = [indices, sorted(indices, key=length),
              sorted(indices, key=length, reverse=True), list(reversed(indices))]
    rng = random.Random(0)  # independent of model/training RNG state
    for _ in range(max(0, max_attempts - len(orders))):
        order = indices[:]
        rng.shuffle(order)
        orders.append(order)
    seen, best, attempts, first_missing = set(), None, 0, None
    for order in orders[:max_attempts]:
        key = tuple(order)
        if key in seen:
            continue
        seen.add(key)
        positions = dict(placed)
        candidate = replace(graph, connections=[edges[index] for index in order])
        routed, requests, plans, plan_conn_ids = _route_once(candidate, positions)
        attempts += 1
        missing = sum(not plan for plan in plans)
        if first_missing is None:
            first_missing = missing
        score = (missing, sum(len(plan) for plan in plans))
        if best is None or score < best[0]:
            by_edge = [([], [])] * len(edges)
            for pos, edge_idx in enumerate(order):
                by_edge[edge_idx] = (plans[pos], plan_conn_ids[pos])
            best = (score, routed, requests, plans, by_edge, positions)
        if missing == 0:
            break
    if best is None:
        raise ValueError('routing attempt budget must be positive')
    score, routed, requests, plans, by_edge, positions = best
    placed.update(positions)
    return routed, requests, plans, by_edge, {'attempts': attempts, 'max_attempts': max_attempts,
                                              'initial_missing_edges': first_missing,
                                              'remaining_missing_edges': score[0]}


def _expand_hubs(source, rotations=None):
    """Split hubs in source edge order, or in the supplied planar rotation."""
    graph = deepcopy(source)
    original = list(graph.connections)
    incident = {room.id: [] for room in graph.rooms}
    for index, edge in enumerate(original):
        if edge.kind != 'teleport':
            incident[edge.a].append(index)
            incident[edge.b].append(index)
    if rotations is not None:
        for rid, indices in incident.items():
            if len(indices) > 4:
                rank = {other: index for index, other in enumerate(rotations[rid])}
                indices.sort(key=lambda index: (rank[original[index].b if original[index].a == rid
                                                     else original[index].a], index))
    endpoint_paths = {}
    internal = []
    for room in list(graph.rooms):
        indices = incident[room.id]
        for index in indices[:3] if len(indices) > 4 else indices:
            endpoint_paths[(room.id, index)] = [room.id]
        if len(indices) <= 4:
            continue
        pending, path = indices[3:], [room.id]
        while pending:
            rid = len(graph.rooms)
            width = max(192, max(original[index].width for index in pending) + 64)
            junction = Room(rid, [], room.floor_h, room.ceil_h, width**2,
                            (0, 0), (0, 0, width, width), role='corridor')
            graph.rooms.append(junction)
            internal.append(Connection(path[-1], rid, 'corridor', min(width - 32, 192)))
            path = path + [rid]
            count = 3 if len(pending) <= 3 else 2
            for index in pending[:count]:
                endpoint_paths[(room.id, index)] = path
            pending = pending[count:]
    graph.connections = internal[:]
    requests = []
    for index, edge in enumerate(original):
        left = endpoint_paths.get((edge.a, index), [edge.a])
        right = endpoint_paths.get((edge.b, index), [edge.b])
        graph.connections.append(Connection(left[-1], right[-1], edge.kind, edge.width))
        route = left + list(reversed(right))
        requests.append(list(zip(route, route[1:])))
    return graph, original, requests


def _fanout_hubs(source):
    """Keep valid hub expansions; repair splits that destroy source planarity.

    A coarse rectangle has four facing directions. Extra ports use junction
    trees, preserving every original edge and its realization path. Splitting
    consecutive ports in a planar rotation avoids creating non-planar graphs.
    """
    import networkx as nx
    expanded = _expand_hubs(source)
    def physical(graph):
        result = nx.Graph()
        result.add_nodes_from(sorted(room.id for room in graph.rooms))
        result.add_edges_from(sorted({tuple(sorted((edge.a, edge.b)))
                                      for edge in graph.connections if edge.kind != 'teleport'}))
        return result
    if nx.check_planarity(physical(expanded[0]))[0]:
        return expanded
    planar, embedding = nx.check_planarity(physical(source))
    if not planar:
        raise ValueError('physical room connections are non-planar; cannot split hubs into planar corridors')
    # The embedding records cyclic incident order. Splitting consecutive
    # groups preserves planarity. Keep already valid expansions unchanged.
    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.planarity.check_planarity.html
    return _expand_hubs(source, embedding.get_data())


def _st_numbering(adjacency, s, t):
    """Tarjan's st-numbering of a biconnected graph containing edge (s, t).

    Depth-first preorder from s with t visited first; each vertex is inserted
    beside its parent according to the sign of its low point. Every vertex
    other than s and t then has a lower and a higher neighbour.
    https://doi.org/10.1016/0304-3975(86)90041-8
    """
    pre, parent, low, order = {s: 0}, {s: None}, {s: s}, [s]
    stack = [(s, iter([t] + sorted(w for w in adjacency[s] if w != t)))]
    while stack:
        v, neighbors = stack[-1]
        for w in neighbors:
            if w not in pre:
                pre[w], parent[w], low[w] = len(order), v, w
                order.append(w)
                stack.append((w, iter(sorted(adjacency[w]))))
                break
            if w != parent[v] and pre[w] < pre[low[v]]:
                low[v] = w
        else:
            stack.pop()
            above = parent[v]
            if above is not None and pre[low[v]] < pre[low[above]]:
                low[above] = low[v]
    sign, numbering = {s: -1}, [s, t]
    for v in order[2:]:
        index = numbering.index(parent[v])
        if sign[low[v]] < 0:
            numbering.insert(index, v)
            sign[parent[v]] = 1
        else:
            numbering.insert(index + 1, v)
            sign[parent[v]] = -1
    return numbering


def _planar_route(source):
    """Realize every physical edge with a planar visibility representation.

    Fallback for graphs the lane router cannot route. After a biconnecting
    augmentation (extra edges are never built) the graph gets an st-numbering;
    rows are longest paths in that orientation and columns longest paths in
    its dual (Tamassia and Tollis). Each room becomes a horizontal bar on one
    row: its room cell plus corridor junctions joined along the row. Each
    connection becomes one straight vertical corridor at its left face's
    column. Bars on a row are disjoint and no vertical corridor crosses a bar,
    so nothing crosses; the result is still checked before use. Parallel edges
    are subdivided by a junction. Returns (graph, requests, conn_ids, placed)
    or None for graphs with fewer than three physical vertices.
    https://doi.org/10.1007/BF02187705
    """
    import networkx as nx
    from networkx.algorithms.planar_drawing import make_bi_connected
    graph = deepcopy(source)
    original = list(graph.connections)
    rooms = {room.id: room for room in graph.rooms}

    def junction(floor, ceil, width):
        rid = len(graph.rooms)
        room = Room(rid, [], floor, ceil, width**2, (0, 0), (0, 0, width, width), role='corridor')
        graph.rooms.append(room)
        rooms[rid] = room
        return rid

    simple = nx.Graph()
    simple.add_nodes_from(rooms)
    segment_info, segments = {}, {}
    for index, edge in enumerate(original):
        if edge.kind == 'teleport':
            continue
        if edge.a == edge.b:
            raise ValueError('a room cannot connect to itself')
        if not simple.has_edge(edge.a, edge.b):
            pieces = [(edge.a, edge.b, edge.kind)]
        else:
            a, b = rooms[edge.a], rooms[edge.b]
            floor = round((a.floor_h + b.floor_h) / 16) * 8
            middle = junction(floor, floor + 128, max(192, edge.width + 64))
            simple.add_node(middle)
            pieces = [(edge.a, middle, edge.kind), (middle, edge.b, 'corridor')]
        for a, b, kind in pieces:
            simple.add_edge(a, b)
            segment_info[frozenset((a, b))] = (a, b, kind, edge.width)
        segments[index] = [(a, b) for a, b, _ in pieces]
    if simple.number_of_nodes() < 3:
        return None
    planar, embedding = nx.check_planarity(simple)
    if not planar:
        raise ValueError('physical room connections are non-planar')
    starts = [min(component) for component in nx.connected_components(simple)]
    for a, b in zip(starts, starts[1:]):
        embedding.connect_components(a, b)
    counted = set()
    for v in sorted(embedding.nodes()):
        for w in list(embedding.neighbors_cw_order(v)):
            make_bi_connected(embedding, v, w, counted)
    embedding.check_structure()
    undirected = nx.Graph(embedding.edges())
    if not nx.is_biconnected(undirected):
        raise ValueError('planar augmentation did not produce a biconnected graph')
    start = next((room.id for room in source.rooms if room.role == 'start' or 'start' in room.roles),
                 source.rooms[0].id)
    adjacency = {v: set(undirected[v]) for v in undirected}
    order = _st_numbering(adjacency, start, min(adjacency[start]))
    number = {v: i for i, v in enumerate(order)}
    if len(number) != len(adjacency) or any(
            v not in (order[0], order[-1]) and not (min(number[w] for w in adjacency[v]) < number[v]
                                                     < max(number[w] for w in adjacency[v]))
            for v in adjacency):
        raise ValueError('st-numbering failed')
    face_of, faces = {}, 0
    for v in sorted(embedding.nodes()):
        for w in embedding.neighbors_cw_order(v):
            if (v, w) not in face_of:
                marked = set()
                embedding.traverse_face(v, w, mark_half_edges=marked)
                for half in marked:
                    face_of[half] = faces
                faces += 1
    s, t = order[0], order[-1]
    outer = face_of[(t, s)]
    dual, left_face = nx.DiGraph(), {}
    dual.add_nodes_from(['S', 'T'])
    for a, b in undirected.edges():
        low, high = (a, b) if number[a] < number[b] else (b, a)
        left, right = face_of[(high, low)], face_of[(low, high)]
        left, right = ('S' if left == outer else left), ('T' if right == outer else right)
        dual.add_edge(left, right)
        left_face[frozenset((a, b))] = left
    x = {}
    for node in nx.topological_sort(dual):
        x[node] = max((x[p] + 1 for p in dual.predecessors(node)), default=0)
    y = {}
    for v in order:
        y[v] = max((y[w] + 1 for w in adjacency[v] if number[w] < number[v]), default=0)

    columns, widths = {v: set() for v in adjacency}, {}
    for key, (a, b, kind, width) in segment_info.items():
        column = x[left_face[key]]
        for v in (a, b):
            columns[v].add(column)
            widths[(v, column)] = max(widths.get((v, column), 0), width)
    cells, placed, room_column = {}, {}, {}
    for v in sorted(adjacency):
        candidates = sorted(columns[v]) or sorted(x[left_face[frozenset((v, w))]] for w in adjacency[v])
        room_column[v] = candidates[len(candidates) // 2]
        for column in sorted(columns[v] | {room_column[v]}):
            if column == room_column[v]:
                cell = v
            else:
                cell = junction(rooms[v].floor_h, rooms[v].ceil_h, max(192, widths[(v, column)] + 64))
            cells[(v, column)] = cell
            placed[cell] = (column, y[v])
    graph.connections = []
    index_of = {}

    def connect(a, b, kind, width):
        index_of[frozenset((a, b))] = len(graph.connections)
        graph.connections.append(Connection(a, b, kind, width))

    def bar(v, column):
        lo, hi = sorted((room_column[v], column))
        path = [cells[(v, c)] for c in sorted(c for vertex, c in cells if vertex == v and lo <= c <= hi)]
        return path if column >= room_column[v] else path[::-1]

    for v in sorted(adjacency):
        row = [cells[(v, c)] for c in sorted(c for vertex, c in cells if vertex == v)]
        for a, b in zip(row, row[1:]):
            connect(a, b, 'corridor', CORR_W)
    for key, (a, b, kind, width) in segment_info.items():
        column = x[left_face[key]]
        connect(cells[(a, column)], cells[(b, column)], kind, width)
    requests, conn_ids = [], []
    for index, edge in enumerate(original):
        if edge.kind == 'teleport':
            connect(edge.a, edge.b, 'teleport', edge.width)
            pairs = [(edge.a, edge.b)]
        else:
            route = []
            for a, b in segments[index]:
                column = x[left_face[frozenset((a, b))]]
                route += bar(a, column) + bar(b, column)[::-1]
            route = [cell for i, cell in enumerate(route) if i == 0 or cell != route[i - 1]]
            pairs = list(zip(route, route[1:]))
        requests.append(pairs)
        conn_ids.append([index_of[frozenset(pair)] for pair in pairs])
    _check_planar_cells(graph, placed)
    return graph, requests, conn_ids, placed


def _check_planar_cells(graph, placed):
    """Reject a planar layout whose straight corridors cross, overlap or hit a cell."""
    occupied = set(placed.values())
    if len(occupied) != len(placed):
        raise ValueError('planar layout placed two cells at one position')
    spans = []
    for edge in graph.connections:
        if edge.kind == 'teleport':
            continue
        (ax, ay), (bx, by) = placed[edge.a], placed[edge.b]
        if (ax != bx) == (ay != by):
            raise ValueError('planar layout produced a non-axis-aligned corridor')
        if ax == bx:
            lo, hi = sorted((ay, by))
            if any((ax, k) in occupied for k in range(lo + 1, hi)):
                raise ValueError('planar layout corridor passes through a room cell')
            spans.append(('V', ax, lo, hi))
        else:
            lo, hi = sorted((ax, bx))
            if any((k, ay) in occupied for k in range(lo + 1, hi)):
                raise ValueError('planar layout corridor passes through a room cell')
            spans.append(('H', ay, lo, hi))
    for i, (kind, at, lo, hi) in enumerate(spans):
        for other, at2, lo2, hi2 in spans[i + 1:]:
            if kind == other and at == at2 and max(lo, lo2) < min(hi, hi2):
                raise ValueError('planar layout corridors overlap')
            if kind != other and lo < at2 < hi and lo2 < at < hi2:
                raise ValueError('planar layout corridors cross')


def _routed_layout(graph, max_layout_attempts=16, *, geometry_profile='classic'):
    """Bounded alternative placements when edge-order rerouting is exhausted."""
    best, seen, attempts = None, set(), 0
    for seed in range(max_layout_attempts):
        placed = _layout(graph, seed=seed)
        signature = tuple(sorted(placed.items()))
        if signature in seen:
            continue
        seen.add(signature)
        routed, edges, plans, by_edge, report = _route_connections(graph, placed)
        attempts += 1
        rooms = {room.id: room for room in routed.rooms}
        maximum_step = max((abs(rooms[e.a].floor_h - rooms[e.b].floor_h)
                            for e in routed.connections if e.kind != 'teleport'), default=0)
        slot = max(SLOT, max(max(_room_dims(room)) for room in routed.rooms) + maximum_step * 3 + 192)
        spans = []
        for axis in (0, 1):
            if geometry_profile in {'compact', 'adjoining'}:
                cells = [cell[axis] for cell in placed.values()]
                half_extents = {cell: 0.0 for cell in cells}
                for room in routed.rooms:
                    cell = placed[room.id][axis]
                    half_extents[cell] = max(half_extents[cell], _room_dims(room)[axis] / 2)
                min_spans = []
                for e in routed.connections:
                    if e.kind == 'teleport':
                        continue
                    a, b = placed[e.a], placed[e.b]
                    if a[1 - axis] != b[1 - axis] or a[axis] == b[axis]:
                        continue
                    left, right = sorted((a[axis], b[axis]))
                    run = abs(rooms[e.a].floor_h - rooms[e.b].floor_h) * 3 + COMPACT_GAP
                    dist = _room_dims(rooms[e.a])[axis] / 2 + _room_dims(rooms[e.b])[axis] / 2 + run + 2
                    min_spans.append((left, right, dist))
                pos = compact_axis_positions(half_extents, min_spans, random.Random(0))
                lows = [pos[placed[room.id][axis]] - _room_dims(room)[axis] / 2 for room in routed.rooms]
                highs = [pos[placed[room.id][axis]] + _room_dims(room)[axis] / 2 for room in routed.rooms]
                spans.append(max(highs) - min(lows))
            else:
                lows, highs = [], []
                for room in routed.rooms:
                    center = (placed[room.id][axis] + 0.5) * slot
                    half = _room_dims(room)[axis] / 2
                    lows.append(center - half)
                    highs.append(center + half)
                spans.append(max(highs) - min(lows))
        score = (report['remaining_missing_edges'], max(spans) > 65535, max(spans), len(routed.rooms))
        if best is None or score < best[0]:
            best = (score, routed, edges, plans, by_edge, report, placed, seed)
        if score[0] == 0 and not score[1]:
            break
    score, routed, edges, plans, by_edge, report, placed, seed = best
    report.update(layout_attempts=attempts, max_layout_attempts=max_layout_attempts, layout_seed=seed)
    return routed, edges, plans, by_edge, report, placed


def assemble(graph: RoomGraph, map_name: str | None = None, *,
             geometry_profile: str = 'classic', seed: int = 0, planar_fallback: bool = True) -> MapDocument:
    """Lane routing first; if it leaves edges unrouted, the planar visibility layout.

    Graphs the lane router realizes are unchanged. The fallback places cells
    with compact spacing in every profile, and its report names the router.
    """
    if geometry_profile not in PROFILES:
        raise ValueError(f'unknown geometry profile: {geometry_profile}')
    if not graph.rooms:
        raise ValueError("cannot assemble an empty room graph")
    source = graph
    graph, original_connections, requests = _fanout_hubs(graph)
    n_internal = len(graph.connections) - len(original_connections)
    internal_idx = {
        frozenset((e.a, e.b)): i for i, e in enumerate(graph.connections[:n_internal])
    }
    graph, physical_connections, physical_plans, by_edge, routing, placed = _routed_layout(
        graph, geometry_profile=geometry_profile)
    route_plans, route_conn_ids = [], []
    for idx, request in enumerate(requests):
        edge_indices = [internal_idx.get(frozenset(pair), n_internal + idx) for pair in request]
        seg_plans = [by_edge[e_idx][0] for e_idx in edge_indices]
        seg_cids = [by_edge[e_idx][1] for e_idx in edge_indices]
        if all(seg_plans):
            route_plans.append([pair for segment in seg_plans for pair in segment])
            route_conn_ids.append([cid for group in seg_cids for cid in group])
        else:
            route_plans.append([])
            route_conn_ids.append([])
    planar = False
    if planar_fallback and routing['remaining_missing_edges']:
        try:
            fallback = _planar_route(source)
        except ValueError as exc:
            routing['planar_fallback'] = f'rejected: {exc}'
        else:
            if fallback is not None:
                graph, route_plans, route_conn_ids, placed = fallback
                planar = True
                routing.update(router='planar_visibility', planar_fallback='used',
                               lane_remaining_missing_edges=routing['remaining_missing_edges'],
                               remaining_missing_edges=0)
    room_by_id = {room.id: room for room in graph.rooms}
    maximum_step = max((abs(room_by_id[edge.a].floor_h - room_by_id[edge.b].floor_h)
                        for edge in graph.connections if edge.kind != 'teleport'), default=0)
    slot = max(SLOT, max(max(_room_dims(room)) for room in graph.rooms) + maximum_step * 3 + 192)
    rng = random.Random(seed)
    varied = geometry_profile != 'classic'
    axes, spacing = [], []
    for axis in range(2):
        cells = [cell[axis] for cell in placed.values()]
        extent = max(_room_dims(room)[axis] for room in graph.rooms)
        if planar or geometry_profile in {'compact', 'adjoining'}:
            half_extents = {cell: 0 for cell in cells}
            for room in graph.rooms:
                cell = placed[room.id][axis]
                half_extents[cell] = max(half_extents[cell], _room_dims(room)[axis] / 2)
            spans = []
            for edge in graph.connections:
                if edge.kind == 'teleport':
                    continue
                a, b = placed[edge.a], placed[edge.b]
                if a[1 - axis] != b[1 - axis] or a[axis] == b[axis]:
                    continue
                left, right = sorted((a[axis], b[axis]))
                run = abs(room_by_id[edge.a].floor_h - room_by_id[edge.b].floor_h) * 3 + COMPACT_GAP
                distance = (_room_dims(room_by_id[edge.a])[axis] / 2
                            + _room_dims(room_by_id[edge.b])[axis] / 2 + run + 2)
                spans.append((left, right, distance))
            positions = compact_axis_positions(half_extents, spans, rng)
            axes.append(positions)
            spacing.append({'axis': axis, 'constraints': len(spans),
                            'classic_center_span': (max(cells) - min(cells)) * slot,
                            'compact_center_span': max(positions.values()) - min(positions.values())})
        else:
            axes.append(axis_positions(cells, slot, extent, rng) if varied else
                        {cell: (cell + 0.5) * slot for cell in cells})
    rects: dict[int, tuple[float, float, float, float]] = {}
    for r in graph.rooms:
        gx, gy = placed[r.id]
        cx, cy = axes[0][gx], axes[1][gy]
        w, h = _room_dims(r)
        rects[r.id] = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    joined, adjacency = set(), None
    if geometry_profile == 'adjoining':
        from roomgraph.adjoining import adjoining_rects
        rects, joined, adjacency = adjoining_rects(graph, placed, rects, seed=seed)

    builder = GeometryBuilder(wall=WALL, floor=FLOOR, ceil=CEIL)
    verts, linedefs, sidedefs, sectors = (
        builder.verts, builder.linedefs, builder.sidedefs, builder.sectors)
    add_line = builder.add_line
    new_sector = builder.new_sector

    # room sectors keep their graph id; corridors get ids after all rooms
    for r in graph.rooms:
        new_sector(r.floor_h, r.ceil_h, sid=r.id)

    gaps: dict[int, dict[str, list]] = {
        r.id: {"E": [], "W": [], "N": [], "S": []} for r in graph.rooms}
    corridors = []   # (rect, orient, roomA, roomB, is_door)
    occupied = set(placed.values())

    realized = set()
    shifted_openings = 0
    for conn_idx, c in enumerate(graph.connections):
        if c.kind == "teleport":
            continue
        if c.a not in placed or c.b not in placed:
            continue
        (ax, ay), (bx, by) = placed[c.a], placed[c.b]
        dx, dy = bx - ax, by - ay
        if (dx != 0) == (dy != 0):     # diagonal (or same cell): can't route straight
            continue
        # a straight corridor is only valid if every cell between is empty
        if dx != 0:
            step = 1 if dx > 0 else -1
            if any((ax + i, ay) in occupied for i in range(step, dx, step)):
                continue
        else:
            step = 1 if dy > 0 else -1
            if any((ax, ay + i) in occupied for i in range(step, dy, step)):
                continue
        is_door = c.kind == "door"
        width = max(48, min(c.width or CORR_W, min(*_room_dims(room_by_id[c.a]), *_room_dims(room_by_id[c.b])) - 32))
        if dx != 0:        # horizontal corridor (may span several empty cells)
            west, east = (c.a, c.b) if dx > 0 else (c.b, c.a)
            wx1 = rects[west][2]
            ex0 = rects[east][0]
            low, high = max(rects[west][1], rects[east][1]), min(rects[west][3], rects[east][3])
            yc = opening_center(low, high, width, rng) if varied else (low + high) / 2
            shifted_openings += abs(yc - (low + high) / 2) >= 1
            lo, hi = yc - width / 2, yc + width / 2
            gaps[west]["E"].append((lo, hi))
            gaps[east]["W"].append((lo, hi))
            corridors.append(((wx1, lo, ex0, hi), "H", west, east, is_door, conn_idx in joined))
            realized.add(conn_idx)
        else:              # vertical corridor (may span several empty cells)
            south, north = (c.a, c.b) if dy > 0 else (c.b, c.a)
            sy1 = rects[south][3]
            ny0 = rects[north][1]
            low, high = max(rects[south][0], rects[north][0]), min(rects[south][2], rects[north][2])
            xc = opening_center(low, high, width, rng) if varied else (low + high) / 2
            shifted_openings += abs(xc - (low + high) / 2) >= 1
            lo, hi = xc - width / 2, xc + width / 2
            gaps[south]["N"].append((lo, hi))
            gaps[north]["S"].append((lo, hi))
            corridors.append(((lo, sy1, hi, ny0), "V", south, north, is_door, conn_idx in joined))
            realized.add(conn_idx)

    # Room boundaries retain their actual mouths; bevels only remove unused
    # corners. Axis-aligned boundary pieces skip the shared corridor spans.
    clipped_rooms = 0
    for r in graph.rooms:
        x0, y0, x1, y1 = rects[r.id]
        g = gaps[r.id]
        if varied:
            outline = room_outline(rects[r.id], g, rng)
            clipped_rooms += len(outline) > 4
            for p, q in zip(outline, outline[1:] + outline[:1]):
                if p[0] == q[0]:
                    side_name = 'W' if p[0] == x0 else 'E'
                    for a, b in _complement(min(p[1], q[1]), max(p[1], q[1]), g[side_name]):
                        start, end = ((p[0], a), (p[0], b))
                        if q[1] < p[1]:
                            start, end = end, start
                        add_line(start, end, r.id)
                elif p[1] == q[1]:
                    side_name = 'S' if p[1] == y0 else 'N'
                    for a, b in _complement(min(p[0], q[0]), max(p[0], q[0]), g[side_name]):
                        start, end = ((a, p[1]), (b, p[1]))
                        if q[0] < p[0]:
                            start, end = end, start
                        add_line(start, end, r.id)
                else:
                    add_line(p, q, r.id)
        else:
            for a, b in _complement(y0, y1, g["E"]):
                add_line((x1, b), (x1, a), r.id)
            for a, b in _complement(y0, y1, g["W"]):
                add_line((x0, a), (x0, b), r.id)
            for a, b in _complement(x0, x1, g["S"]):
                add_line((b, y0), (a, y0), r.id)
            for a, b in _complement(x0, x1, g["N"]):
                add_line((a, y1), (b, y1), r.id)

    # corridor sectors: floor/ceil bridging the two rooms
    for (cx0, cy0, cx1, cy1), orient, room_a, room_b, is_door, is_joined in corridors:
        fa, fb = room_by_id[room_a], room_by_id[room_b]
        if orient == "H":
            quad = [(cx0, cy0), (cx0, cy1), (cx1, cy1), (cx1, cy0)]
        else:
            quad = [(cx1, cy0), (cx0, cy0), (cx0, cy1), (cx1, cy1)]
        emitter = emit_corridor
        if is_joined:
            from roomgraph.adjoining import emit_threshold
            emitter = emit_threshold
        emitter(quad, room_a, room_b, fa.floor_h, fb.floor_h,
                      fa.ceil_h, fb.ceil_h, is_door,
                      lambda floor, ceil, closed: new_sector(floor, ceil, closed=closed), add_line)

    r0 = next((room for room in graph.rooms if room.role == 'start' or 'start' in room.roles), graph.rooms[0])
    x0, y0, x1, y1 = rects[r0.id]
    name, cat, unres = resolve_doomed(1)
    things = [Thing(id=0, x=int((x0 + x1) / 2), y=int((y0 + y1) / 2), angle=90,
                    type=1, type_name=name, category=cat,
                    flags=ThingFlags(skill_1_2=True, skill_3=True, skill_4_5=True),
                    unresolved=unres)]

    vertex_objs = [Vertex(id=i, x=float(p[0]), y=float(p[1]))
                   for p, i in sorted(verts.items(), key=lambda kv: kv[1])]
    bounds, derived = compute_derived(vertex_objs, sectors, linedefs, sidedefs, things)
    meta = MapMeta(map_name=map_name or graph.map_name, source_wad="<assembled>",
                   wad_hash="", format="doom", is_iwad=False)
    doc = MapDocument(
        schema_version=SCHEMA_VERSION, meta=meta, bounds=bounds,
        vertices=vertex_objs, sectors=sectors, linedefs=linedefs,
        sidedefs=sidedefs, things=things, derived=derived,
        extra={"assembled": True, "routing": routing,
               "geometry_profile": geometry_profile, "geometry_seed": seed,
               "geometry_variation": {"version": VERSION, "clipped_rooms": clipped_rooms,
                                      "shifted_openings": shifted_openings,
                                      "corridors": len(corridors)},
               "connection_realization": [
            {"a": c.a, "b": c.b, "kind": c.kind,
             "realized": bool(plan) and bool(cids) and all(cid in realized for cid in cids),
             "reason": "" if plan and cids and all(cid in realized for cid in cids) else "no corridor route",
             "route": [list(pair) for pair in plan], 'requested_width': c.width}
            for c, plan, cids in zip(original_connections, route_plans, route_conn_ids)]})

    if spacing:
        doc.extra['geometry_variation']['spacing'] = {'version': 1, 'axes': spacing}
    if adjacency is not None:
        doc.extra['geometry_variation']['adjacency'] = adjacency
    doc.extra['room_roles'] = {r.id: r.roles or [r.role] for r in graph.rooms}
    room_sectors = {r.id: r.id for r in graph.rooms}
    doc.extra['room_sectors'] = room_sectors
    return finalize_connections(doc, room_sectors)

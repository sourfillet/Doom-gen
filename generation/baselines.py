"""Training-only graph baselines and exact authored-geometry copy probes."""
from __future__ import annotations

import copy
import hashlib
import json
import random


def procedural_graph(references, *, rooms, edges, seed):
    """Match sampled room/edge counts; draw binned attributes from training.

    A random growing tree supplies connectivity. Extra edges preserve physical
    planarity; the shared assembler supplies exactly the same routing budget.
    Start/exit presence is imposed, rather than learned, for this baseline.
    """
    import networkx as nx
    from data.roomgraph_targets import graph_to_sequence, sequence_to_graph, KINDS, ROOT_FLOOR_IDX
    rng = random.Random(seed)
    tokens = [token for graph in references for token in graph_to_sequence(graph)]
    if not tokens or rooms < 1:
        raise ValueError('baseline needs training references and a positive room count')
    sequence, physical, linked, degree = [], nx.Graph(), set(), [0] * rooms
    physical.add_nodes_from(range(rooms))
    for index in range(rooms):
        token = copy.deepcopy(rng.choice(tokens))
        token.update(extra_edges=[], roles=token.get('roles', 0) & 4)
        if index == 0:
            token.update(parent=-1, kind=KINDS.index('none'), floor=ROOT_FLOOR_IDX, roles=1)
        else:
            parent = rng.choices(range(index), weights=[max(1, 4 - degree[i]) for i in range(index)])[0]
            if KINDS[token['kind']] == 'none':
                token['kind'] = KINDS.index('corridor')
            token['parent'] = parent
            linked.add((parent, index))
            degree[parent] += 1
            degree[index] += 1
            if KINDS[token['kind']] != 'teleport':
                physical.add_edge(parent, index)
        sequence.append(token)
    sequence[-1]['roles'] |= 2
    choices = [(a, b) for b in range(rooms) for a in range(b) if (a, b) not in linked]
    rng.shuffle(choices)
    for a, b in choices:
        if len(linked) >= edges:
            break
        token = rng.choice(tokens)
        kind = token['kind'] if KINDS[token['kind']] != 'none' else KINDS.index('corridor')
        if KINDS[kind] != 'teleport':
            physical.add_edge(a, b)
            if not nx.check_planarity(physical)[0]:
                physical.remove_edge(a, b)
                continue
        sequence[b]['extra_edges'].append({'parent': a, 'kind': kind, 'width': token.get('width', 2)})
        linked.add((a, b))
    return sequence_to_graph(sequence)


def retrieval_graph(references, *, rooms, seed):
    if not references:
        raise ValueError('baseline needs training references')
    rng = random.Random(seed)
    difference = min(abs(len(graph.rooms) - rooms) for graph in references)
    return copy.deepcopy(rng.choice([graph for graph in references if abs(len(graph.rooms) - rooms) == difference]))


def geometry_fingerprint(doc):
    """Exact XY-translation/ID-independent wall/height and thing fingerprints.

    Does not detect rotated, partial, approximate, or retrieved-outline copies.
    Texture changes do not hide an exact structural copy.
    """
    return _geometry_fingerprint(doc, referenced_only=False)


def geometry_fingerprint_v2(doc):
    """Referenced-wall geometry/thing hashes; inert records cannot move the frame.

    V1 remains available for reproducing older reports. V2 anchors XY/Z only
    on vertices/sectors actually referenced by walls, retaining zero-height
    controls that do have boundaries. It is still an exact-copy probe, not an
    approximate, rotated, partial-copy or gameplay-equivalence assessment.
    """
    return _geometry_fingerprint(doc, referenced_only=True)


def _geometry_fingerprint(doc, *, referenced_only):
    positions = {v.id: (v.x, v.y) for v in doc.vertices}
    if referenced_only:
        used = {index for line in doc.linedefs for index in (line.v1, line.v2)}
        basis = [positions[index] for index in used]
        sectors = {doc.sidedefs[index].sector for line in doc.linedefs
                   for index in (line.right_sidedef, line.left_sidedef) if 0 <= index < len(doc.sidedefs)}
        floors = [doc.sectors[index].floor_height for index in sectors if 0 <= index < len(doc.sectors)]
    else:
        basis = list(positions.values())
        floors = [s.floor_height for s in doc.sectors]
    x0 = min((p[0] for p in basis), default=0)
    y0 = min((p[1] for p in basis), default=0)
    z0 = min(floors, default=0)
    def point(index):
        x, y = positions[index]
        return round(float(x - x0), 6), round(float(y - y0), 6)
    def side(index):
        if index < 0 or index >= len(doc.sidedefs):
            return None
        sector_id = doc.sidedefs[index].sector
        if not 0 <= sector_id < len(doc.sectors):
            return None
        sector = doc.sectors[sector_id]
        return sector.floor_height - z0, sector.ceil_height - z0
    walls = []
    for line in doc.linedefs:
        a, b = point(line.v1), point(line.v2)
        right, left = side(line.right_sidedef), side(line.left_sidedef)
        if b < a:
            a, b, right, left = b, a, left, right
        walls.append((a, b, right, left))
    # JSON strings have a total ordering even with null sidedefs.
    walls = sorted(json.dumps(wall) for wall in walls)
    things = sorted((round(float(t.x - x0), 6), round(float(t.y - y0), 6), t.type, t.angle,
                     t.flags.skill_1_2, t.flags.skill_3, t.flags.skill_4_5, t.flags.not_single_player)
                    for t in doc.things)
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return {'geometry': digest(walls), 'things': digest(things)}


def realize(graph, *, geometry_profile='classic', seed=0):
    """The undecorated neural evaluation pipeline, shared by both baselines."""
    from roomgraph.assemble import assemble
    from roomgraph.teleports import require_connections
    from generation.validator import validate_and_repair, require_valid
    from generation.progression import ensure_exit
    from wad_parser.map_extractor import compute_derived
    from wad_writer.assembler import fit_coordinates
    doc = assemble(graph, geometry_profile=geometry_profile, seed=seed)
    require_connections(doc)
    validate_and_repair(doc)
    ensure_exit(doc)
    doc.bounds, doc.derived = compute_derived(doc.vertices, doc.sectors, doc.linedefs, doc.sidedefs, doc.things)
    fit_coordinates(doc)
    require_valid(doc, require_connected=True)
    return doc

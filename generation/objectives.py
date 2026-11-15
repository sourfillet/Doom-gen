"""Stable semantic goal ranking for the opt-in introductory gameplay profile."""
from __future__ import annotations

import math
from collections import deque


def select_intro_objective(doc, walls, *, start_sector, sector_distances):
    """Choose among already clearance-checked eligible encounter-room walls.

    Requested room hops rank goals independently of assembler-added sectors.
    This is a ranking heuristic, not a directed physical reachability proof;
    the caller must still verify a player-region route and action history.
    Old/source documents without complete semantic metadata retain the legacy
    ranking, explicitly identified in the returned policy report.
    """
    if not walls:
        raise ValueError('intro layout has no usable objective walls')
    start = next(thing for thing in doc.things if thing.type == 1)

    def wall_rank(line):
        a, b = (doc.vertices[v] for v in (line.v1, line.v2))
        return (math.dist((start.x, start.y), ((a.x + b.x) / 2, (a.y + b.y) / 2)),
                math.hypot(a.x - b.x, a.y - b.y), -line.id)

    mappings = doc.extra.get('room_sectors', {})
    requests = doc.extra.get('connection_realization', [])
    roles = {int(rid): value for rid, value in doc.extra.get('room_roles', {}).items()}
    reason = 'no complete semantic room metadata'
    if mappings and requests:
        if not all(edge.get('realized') for edge in requests):
            reason = 'incomplete requested-connection realization'
        else:
            room_sectors = {int(rid): sid for rid, sid in mappings.items()}
            adjacency = {}
            for edge in requests:
                a, b = edge['a'], edge['b']
                adjacency.setdefault(a, set()).add(b)
                adjacency.setdefault(b, set()).add(a)
            # Only original request endpoints participate. Artificial bends
            # and fanout corridors must not become semantic encounter goals.
            starts = [rid for rid, sid in room_sectors.items() if sid == start_sector and rid in adjacency]
            if len(starts) == 1 and all(rid in room_sectors for rid in adjacency):
                distance, queue = {starts[0]: 0}, deque(starts)
                while queue:
                    rid = queue.popleft()
                    for neighbor in sorted(adjacency[rid]):
                        if neighbor not in distance:
                            distance[neighbor] = distance[rid] + 1
                            queue.append(neighbor)
                candidates = {}
                for rid in distance:
                    if 'corridor' in roles.get(rid, []):
                        continue
                    candidates[rid] = [line for line in walls
                                       if doc.sidedefs[line.right_sidedef].sector == room_sectors[rid]]
                candidates = {rid: lines for rid, lines in candidates.items() if lines}
                if candidates:
                    # Prefer an end room over a junction when distance/role
                    # agree. A stable ID alone can move the exit from a door
                    # leaf into a bypassable central room on the same graph.
                    rid = max(candidates, key=lambda rid: (distance[rid], 'exit' in roles.get(rid, []),
                                                          len(adjacency[rid]) == 1, -rid))
                    objective = max(candidates[rid], key=wall_rank)
                    return objective, {'policy': 'semantic_room_hops_v2', 'room': rid,
                                       'distance': distance[rid], 'start_room': starts[0],
                                       'scope': 'goal ranking only; physical region route checked separately'}
                reason = 'no eligible wall belongs to a reachable semantic room'
            else:
                reason = 'semantic endpoint/start ownership is incomplete or ambiguous'
    objective = max(walls, key=lambda line: (sector_distances[doc.sidedefs[line.right_sidedef].sector],
                                            *wall_rank(line)))
    return objective, {'policy': 'legacy_sector_hops_v1', 'reason': reason,
                       'scope': 'source/legacy fallback; not invariant to sector subdivision'}


def select_keyed_intro_objective(doc, walls, *, start_sector, sector_distances,
                                 seed=0, game=None, max_goals=16):
    """Bounded non-mutating gate/key preflight for an explicitly requested key.

    Try the ordinary semantic ranking first, then the next eligible room/sector.
    A trial uses the actual key planner and supported history validator. The
    final populated map still needs a fresh key plan; placement may change.
    """
    from copy import deepcopy
    from generation.keyed import add_blue_key
    from wad_parser.types import resolve_linedef_special

    if not isinstance(max_goals, int) or isinstance(max_goals, bool) or max_goals < 1:
        raise ValueError('keyed objective max_goals must be positive')
    remaining, rejected = list(walls), []
    for _ in range(max_goals):
        if not remaining:
            break
        objective, policy = select_intro_objective(doc, remaining, start_sector=start_sector,
                                                   sector_distances=sector_distances)
        sid = doc.sidedefs[objective.right_sidedef].sector
        trial = deepcopy(doc)
        for line in trial.linedefs:
            if line.special == 11:
                line.special, line.special_name, line.trigger_type = 0, None, None
        line = trial.linedefs[objective.id]
        line.special, line.tag = 11, 0
        line.special_name, line.trigger_type = resolve_linedef_special(11)
        try:
            plan = add_blue_key(trial, seed=seed, game=game)
        except ValueError as exc:
            rejected.append({'sector': sid, 'exit_line': objective.id, 'reason': str(exc)})
            # One representative wall per eligible room/sector is the explicit
            # search scope. Do not claim exhaustive solving of arbitrary maps.
            remaining = [candidate for candidate in remaining
                         if doc.sidedefs[candidate.right_sidedef].sector != sid]
            continue
        return objective, {**policy, 'required_progression': 'blue-key',
                            'key_preflight': {'status': 'supported', 'door_sector': plan['door_sector'],
                                              'key_sector': plan['key_sector'], 'max_goals': max_goals,
                                              'goals_attempted': len(rejected) + 1, 'rejected_goals': rejected,
                                              'scope': 'Pre-population preflight; final populated key plan/history required.'}}
    state = 'budget exhausted' if remaining else 'no supported goal among eligible room/sector representatives'
    details = '; '.join(f'sector {row["sector"]}: {row["reason"]}' for row in rejected)
    raise ValueError(f'blue-key objective search: {state} ({len(rejected)}/{max_goals} goals): {details}')

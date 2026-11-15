"""Non-mutating sector identity across node-builder pruning/reindexing.

This supplies correspondence for diagnostic comparisons, not permission to drop
controls or a geometry/playability certificate. Ambiguous ownership stays open.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import json


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _sector(sector):
    return {key: value for key, value in asdict(sector).items() if key != 'id'}


def _side(side):
    return {key: value for key, value in asdict(side).items() if key not in {'id', 'sector'}}


def _line(line, positions):
    return _encoded({'points': [positions[line.v1], positions[line.v2]], 'flags': asdict(line.flags),
                     'special': line.special, 'tag': line.tag, 'args': line.args, 'extra': line.extra})


def sector_correspondence(source, rebuilt):
    """Map by oriented authored boundaries, assets/actions and sector fields.

    IDs/array positions may change. A repeated identical line is usable only if
    its candidate sides agree on ownership. No nearest-centroid/area guess is
    made. Unreferenced source sectors and unresolved/conflicting anchors remain
    explicit; even an identified mapping does not prove unchanged geometry.
    """
    originals = defaultdict(list)
    source_positions = {v.id: (float(v.x), float(v.y)) for v in source.vertices}
    target_positions = {v.id: (float(v.x), float(v.y)) for v in rebuilt.vertices}
    for line in source.linedefs:
        originals[_line(line, source_positions)].append(line)
    source_sectors = {sector.id: sector for sector in source.sectors}
    target_sectors = {sector.id: sector for sector in rebuilt.sectors}
    pairs, referenced, ambiguous, unmatched = defaultdict(set), set(), [], []
    for line in source.linedefs:
        referenced.update(source.sidedefs[side].sector for side in (line.right_sidedef, line.left_sidedef) if side >= 0)
    for line in rebuilt.linedefs:
        candidates = originals.get(_line(line, target_positions), [])
        if not candidates:
            unmatched.append(line.id)
            continue
        for attr in ('right_sidedef', 'left_sidedef'):
            target_side_id = getattr(line, attr)
            target_side = rebuilt.sidedefs[target_side_id] if target_side_id >= 0 else None
            owners = set()
            for candidate in candidates:
                original_id = getattr(candidate, attr)
                original_side = source.sidedefs[original_id] if original_id >= 0 else None
                if original_side is None or target_side is None:
                    continue
                if (_side(original_side) == _side(target_side)
                        and _sector(source_sectors[original_side.sector]) == _sector(target_sectors[target_side.sector])):
                    owners.add(original_side.sector)
            if target_side is not None:
                if len(owners) == 1:
                    pairs[next(iter(owners))].add(target_side.sector)
                else:
                    ambiguous.append({'linedef': line.id, 'side': attr, 'target_sector': target_side.sector,
                                      'candidate_source_sectors': sorted(owners)})
    reverse = defaultdict(set)
    for sid, targets in pairs.items():
        for tid in targets:
            reverse[tid].add(sid)
    mapping = {sid: next(iter(targets)) for sid, targets in pairs.items() if len(targets) == 1
               and len(reverse[next(iter(targets))]) == 1}
    conflicts = [{'source_sector': sid, 'target_sectors': sorted(targets)} for sid, targets in pairs.items()
                 if sid not in mapping]
    # Ambiguous duplicate lines can be corroborated by other unique boundaries.
    unresolved = [entry for entry in ambiguous
                  if sum(mapping.get(sid) == entry['target_sector'] for sid in entry['candidate_source_sectors']) != 1]
    unreferenced = sorted(source_sectors.keys() - referenced)
    unmapped = sorted(referenced - mapping.keys())
    target_unmapped = sorted(target_sectors.keys() - set(mapping.values()))
    return {'schema': 'spatial_sector_correspondence_v1',
            'status': 'complete' if not (unmatched or conflicts or unresolved or unmapped or target_unmapped) else 'incomplete',
            'mapping': [{'source_sector': sid, 'target_sector': tid} for sid, tid in sorted(mapping.items())],
            'unreferenced_source_sectors': [{'sector': sid, 'attributes': _sector(source_sectors[sid])} for sid in unreferenced],
            'unmapped_referenced_source_sectors': unmapped, 'unmapped_target_sectors': target_unmapped,
            'unmatched_target_linedefs': unmatched, 'conflicts': conflicts, 'unresolved_anchors': unresolved,
            'scope': 'Identity from authored boundary/side/action evidence only; pruning semantics and geometry require separate checks.'}

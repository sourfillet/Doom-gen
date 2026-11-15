"""Bounded procedural cover columns in already validated generated rooms.

Columns are holes with inward-facing one-sided walls, not unreachable dummy
sectors. Every accepted addition preserves the existing playable regions/exit.
This is procedural interior detail, not a learned room-shape representation.
"""
from __future__ import annotations

from copy import deepcopy
import random

INTERIOR_PROFILES = ('columns', 'stepped')


def add_terraces(doc, *, seed=0, max_terraces=2, wall_texture=None, floor_flat=None):
    """Insert reachable nested floor sectors, with wide treads and low risers.

    New floors rise 16 units per 64-unit tread, to at most 64 units. They
    retain at least 64 units of headroom and 64 units of clearance from the
    original boundary/actors. Tagged/special sectors and the start are excluded.
    Each trial is fully validated before it replaces the current document.
    """
    from shapely import Point, Polygon
    from wad_parser.bsp_geometry import component_polygons
    from wad_parser.geometry import sector_points, sector_geometry
    from wad_parser.map_extractor import compute_derived, _sector_containing
    from wad_parser.types import Vertex, Sector, Sidedef, Linedef, LinedefFlags
    from generation.validator import require_valid

    if max_terraces < 0:
        raise ValueError('max_terraces must be non-negative')
    require_valid(doc, require_connected=True)
    rng = random.Random(seed)
    start = next(thing for thing in doc.things if thing.type == 1)
    start_sid = _sector_containing(start.x, start.y, doc.sectors, sector_geometry(doc), doc.vertices)
    candidates = []
    for sector in doc.sectors:
        if (sector.id == start_sid or sector.special or sector.tag
                or sector.floor_slope is not None or sector.ceil_slope is not None):
            continue
        for _, polygon in component_polygons(sector_points(doc, sector.id)):
            x0, y0, x1, y1 = polygon.bounds
            small = min(x1 - x0, y1 - y0)
            if small >= 512 and max(x1 - x0, y1 - y0) / small <= 2.5:
                levels = min(4 if small >= 896 else 3 if small >= 768 else 2,
                             int((sector.ceil_height - sector.floor_height - 64) // 16))
                if levels >= 1:
                    candidates.append((polygon.area, sector.id, polygon.bounds, levels))
    candidates.sort(reverse=True)
    result = {'version': 1, 'terraces': [], 'rejected_candidates': 0,
              'scope': 'Procedural within-room floors; topology/entrances retained, no learned architecture claim.'}

    def octagon(cx, cy, width):
        half, clip = width // 2, min(64, width // 4)
        # Counterclockwise: surrounding lower floor is on the right.
        return [(cx - half + clip, cy - half), (cx + half - clip, cy - half),
                (cx + half, cy - half + clip), (cx + half, cy + half - clip),
                (cx + half - clip, cy + half), (cx - half + clip, cy + half),
                (cx - half, cy + half - clip), (cx - half, cy - half + clip)]

    for _, sid, (x0, y0, x1, y1), levels in candidates:
        if len(result['terraces']) >= max_terraces:
            break
        width = 128 * levels
        fractions = [(.5, .5), (.4, .55), (.6, .45), (.4, .4), (.6, .6)]
        rng.shuffle(fractions)
        polygons = [p for _, p in component_polygons(sector_points(doc, sid))]
        for fx, fy in fractions:
            cx = round((x0 + (x1 - x0) * fx) / 8) * 8
            cy = round((y0 + (y1 - y0) * fy) / 8) * 8
            footprint = Polygon(octagon(cx, cy, width))
            if not any(p.covers(footprint.buffer(64, join_style='mitre')) for p in polygons):
                continue
            if any(footprint.buffer(64).covers(Point(t.x, t.y)) for t in doc.things):
                continue
            trial = deepcopy(doc)
            base = trial.sectors[sid]
            texture = wall_texture or next((s.mid_tex for s in trial.sidedefs
                if s.sector == sid and s.mid_tex not in {'-', ''}
                and not s.mid_tex.startswith(('SW1', 'SW2'))), 'STARTAN2')
            previous_sid, steps = sid, []
            for level in range(1, levels + 1):
                new_sid = len(trial.sectors)
                height = base.floor_height + 16 * level
                light = min(208, max(144, base.light_level) + 8 * (level - 1))
                trial.sectors.append(Sector(new_sid, height, base.ceil_height,
                    floor_flat or base.floor_flat, base.ceil_flat, light, 0, 0))
                points = octagon(cx, cy, width - 128 * (level - 1))
                ids = []
                for x, y in points:
                    ids.append(len(trial.vertices))
                    trial.vertices.append(Vertex(len(trial.vertices), x, y))
                for a, b in zip(ids, ids[1:] + ids[:1]):
                    right = len(trial.sidedefs)
                    trial.sidedefs.extend([
                        Sidedef(right, 0, 0, texture, texture, '-', previous_sid),
                        Sidedef(right + 1, 0, 0, texture, texture, '-', new_sid)])
                    trial.linedefs.append(Linedef(len(trial.linedefs), a, b,
                        LinedefFlags(two_sided=True), 0, 0, right, right + 1,
                        extra={'interior_detail': 'terrace_step'}))
                steps.append({'sector': new_sid, 'floor_height': height,
                              'width': width - 128 * (level - 1), 'light_level': light})
                previous_sid = new_sid
            trial.bounds, trial.derived = compute_derived(trial.vertices, trial.sectors,
                trial.linedefs, trial.sidedefs, trial.things)
            try:
                require_valid(trial, require_connected=True)
            except ValueError:
                result['rejected_candidates'] += 1
                continue
            doc.__dict__.update(trial.__dict__)
            result['terraces'].append({'base_sector': sid, 'center': [cx, cy],
                'base_floor_height': base.floor_height, 'footprint': list(footprint.exterior.coords)[:-1],
                'riser_height': 16, 'tread_width': 64, 'steps': steps})
            break
    doc.extra['terrace_details'] = result
    return result


def add_columns(doc, *, seed=0, max_columns=4, wall_texture=None):
    from shapely import Point, box
    from wad_parser.bsp_geometry import component_polygons
    from wad_parser.geometry import sector_points
    from wad_parser.map_extractor import compute_derived
    from wad_parser.types import Vertex, Sidedef, Linedef, LinedefFlags
    from generation.validator import require_valid

    if max_columns < 0:
        raise ValueError('max_columns must be non-negative')
    rng = random.Random(seed)
    require_valid(doc, require_connected=True)
    candidates = []
    for sector in doc.sectors:
        if (sector.special or sector.tag
                or sector.floor_slope is not None or sector.ceil_slope is not None):
            continue
        for _, polygon in component_polygons(sector_points(doc, sector.id)):
            x0, y0, x1, y1 = polygon.bounds
            if min(x1 - x0, y1 - y0) >= 512 and sector.ceil_height - sector.floor_height >= 64:
                candidates.append((polygon.area, sector.id, polygon.bounds))
    candidates.sort(reverse=True)
    result = {'version': 1, 'columns': [], 'rejected_candidates': 0,
              'scope': 'Procedural cover; original rooms and required connections retained.'}
    for _, sid, (x0, y0, x1, y1) in candidates:
        if len(result['columns']) >= max_columns:
            break
        fractions = [(.35, .35), (.65, .65), (.35, .65), (.65, .35)]
        rng.shuffle(fractions)
        half = 48 if min(x1 - x0, y1 - y0) >= 768 else 32
        parts = [polygon for _, polygon in component_polygons(sector_points(doc, sid))]
        for fx, fy in fractions:
            cx = round((x0 + (x1 - x0) * fx) / 8) * 8
            cy = round((y0 + (y1 - y0) * fy) / 8) * 8
            obstacle = box(cx - half, cy - half, cx + half, cy + half)
            if not any(polygon.covers(obstacle.buffer(64, join_style='mitre')) for polygon in parts):
                continue
            if any(obstacle.buffer(64).covers(Point(thing.x, thing.y)) for thing in doc.things):
                continue
            trial = deepcopy(doc)
            # Counterclockwise walls keep the surrounding sector on their right.
            points = [(cx - half, cy - half), (cx + half, cy - half),
                      (cx + half, cy + half), (cx - half, cy + half)]
            texture = wall_texture or next((side.mid_tex for side in trial.sidedefs
                if side.sector == sid and side.mid_tex not in {'-', ''}
                and not side.mid_tex.startswith(('SW1', 'SW2'))), 'STARTAN2')
            ids = []
            for x, y in points:
                ids.append(len(trial.vertices))
                trial.vertices.append(Vertex(len(trial.vertices), x, y))
            for a, b in zip(ids, ids[1:] + ids[:1]):
                side = len(trial.sidedefs)
                trial.sidedefs.append(Sidedef(side, 0, 0, '-', '-', texture, sid))
                trial.linedefs.append(Linedef(len(trial.linedefs), a, b,
                    LinedefFlags(impassable=True), 0, 0, side, -1,
                    extra={'interior_detail': 'column'}))
            trial.bounds, trial.derived = compute_derived(trial.vertices, trial.sectors,
                trial.linedefs, trial.sidedefs, trial.things)
            try:
                require_valid(trial, require_connected=True)
            except ValueError:
                result['rejected_candidates'] += 1
                continue
            doc.__dict__.update(trial.__dict__)
            result['columns'].append({'sector': sid, 'center': [cx, cy], 'width': half * 2})
            break
    doc.extra['interior_details'] = result
    return result

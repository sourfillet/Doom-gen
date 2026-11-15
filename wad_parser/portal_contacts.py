"""Order-independent contacts along a shared authored portal.

Oblique polygon boundaries can differ from their linedef by floating-point
roundoff. Clip each region in the same portal frame, then intersect its 1D
intervals. Only uncertain normal signs are snapped; player clearance, genuine
gaps and the caller's minimum contact length remain separate checks.
"""
from __future__ import annotations

import math
import sys


def portal_intervals(polygon, mouth):
    """Closed polygon contact as sorted disjoint fractions of the linedef."""
    from shapely import LineString, Polygon, make_valid
    from shapely.errors import GEOSException

    origin, end = list(mouth.coords)
    dx, dy = end[0] - origin[0], end[1] - origin[1]
    length = math.hypot(dx, dy)
    if length <= 0:
        return []

    def project(point):
        x, y = point[0] - origin[0], point[1] - origin[1]
        along = math.fsum([x * dx, y * dy]) / length
        normal = math.fsum([dx * y, -dy * x]) / length
        # Account for coordinate subtraction and product rounding. In Doom's
        # coordinate range this is far below one fixed-point world unit and
        # below the existing clearance tolerance for exact 32-unit passages.
        error = (8 * sys.float_info.epsilon * (abs(dx * y) + abs(dy * x)) / length
                 + 4 * (math.ulp(point[0]) + math.ulp(point[1])
                        + math.ulp(origin[0]) + math.ulp(origin[1])))
        return along, 0.0 if abs(normal) <= error else normal

    local = Polygon([project(point) for point in polygon.exterior.coords],
                    [[project(point) for point in ring.coords] for ring in polygon.interiors])
    segment = LineString([(0, 0), (length, 0)])
    try:
        contact = local.intersection(segment)
    except GEOSException:
        # Snapping normals onto the portal can leave a ring touching itself
        # (GEOS "side location conflict"). Rebuild only the polygonal area.
        contact = make_valid(local, method='structure', keep_collapsed=False).intersection(segment)

    def intervals(part):
        if part.is_empty:
            return []
        if part.geom_type in {'LineString', 'LinearRing'}:
            values = [point[0] / length for point in part.coords]
            return [(max(0, min(values)), min(1, max(values)))]
        return [interval for child in getattr(part, 'geoms', []) for interval in intervals(child)]

    merged = []
    for low, high in sorted(intervals(contact)):
        if high <= low:
            continue
        if merged and low <= merged[-1][1]:
            merged[-1] = merged[-1][0], max(high, merged[-1][1])
        else:
            merged.append((low, high))
    return merged


def overlap_length(left, right, length):
    """World length of overlapping sorted, disjoint portal intervals."""
    i = j = 0
    overlap = 0.0
    while i < len(left) and j < len(right):
        low, high = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        overlap += max(0, high - low)
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return overlap * length

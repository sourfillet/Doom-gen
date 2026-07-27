"""Lightweight top-down SVG preview of a parsed map, for the tagging CLI.

Line art only, per plan/tagging.md - one-sided walls dark, two-sided
(passable) lines light, things as category-colored dots. SVG because it
needs zero dependencies and opens in any browser/editor.
"""
from __future__ import annotations

from wad_parser.types import MapDocument

_THING_COLORS = {
    "player-start": "#2e9e4f",
    "deathmatch-start": "#7bc96f",
    "teleport-dest": "#9b59d0",
    "monster": "#d04b3e",
    "key": "#e0b520",
    "weapon": "#3273c4",
    "ammo": "#7fa8d8",
    "item": "#4fb8b0",
    "obstacle": "#8a6d4f",
}
_DEFAULT_THING_COLOR = "#aaaaaa"


def map_to_svg(doc: MapDocument, width: int = 1024) -> str:
    b = doc.bounds
    span_x = max(b.max_x - b.min_x, 1.0)
    span_y = max(b.max_y - b.min_y, 1.0)
    margin = 0.03 * max(span_x, span_y)
    height = int(width * (span_y + 2 * margin) / (span_x + 2 * margin))

    def sx(x: float) -> float:
        return (x - b.min_x + margin) / (span_x + 2 * margin) * width

    def sy(y: float) -> float:
        # Doom is y-up, SVG is y-down
        return height - (y - b.min_y + margin) / (span_y + 2 * margin) * height

    vpos = {v.id: (v.x, v.y) for v in doc.vertices}
    stroke_w = max(width / 1024.0, 0.75)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#14161a"/>',
        f'<title>{doc.meta.map_name}</title>',
    ]

    two_sided, one_sided, special = [], [], []
    for ld in doc.linedefs:
        if ld.v1 not in vpos or ld.v2 not in vpos:
            continue
        (x1, y1), (x2, y2) = vpos[ld.v1], vpos[ld.v2]
        seg = f'<line x1="{sx(x1):.1f}" y1="{sy(y1):.1f}" x2="{sx(x2):.1f}" y2="{sy(y2):.1f}"/>'
        if ld.special:
            special.append(seg)
        elif ld.left_sidedef == -1:
            one_sided.append(seg)
        else:
            two_sided.append(seg)

    parts.append(f'<g stroke="#3d4450" stroke-width="{stroke_w:.2f}">' + "".join(two_sided) + "</g>")
    parts.append(f'<g stroke="#c9d1d9" stroke-width="{2 * stroke_w:.2f}">' + "".join(one_sided) + "</g>")
    parts.append(f'<g stroke="#d7a53e" stroke-width="{2 * stroke_w:.2f}">' + "".join(special) + "</g>")

    dots = []
    r = 3.5 * stroke_w
    for t in doc.things:
        color = _THING_COLORS.get(t.category, _DEFAULT_THING_COLOR)
        dots.append(f'<circle cx="{sx(t.x):.1f}" cy="{sy(t.y):.1f}" r="{r:.1f}" fill="{color}"/>')
    parts.append("".join(dots))
    parts.append("</svg>")
    return "\n".join(parts)

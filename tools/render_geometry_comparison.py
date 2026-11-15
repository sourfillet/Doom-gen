"""Render a v2 geometry diagnostic report without inventing rates or evidence."""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path

from wad_parser.map_extractor import extract_all_maps
from wad_parser.wad_reader import WadFile


def render_svg(doc, width=400, height=350):
    x0, y0, x1, y1 = doc.bounds.min_x, doc.bounds.min_y, doc.bounds.max_x, doc.bounds.max_y
    scale = min((width - 40) / max(x1 - x0, 1), (height - 40) / max(y1 - y0, 1))
    def point(x, y):
        return 20 + (x - x0) * scale, height - 20 - (y - y0) * scale
    lines = []
    positions = {v.id: (v.x, v.y) for v in doc.vertices}
    for line in doc.linedefs:
        a, b = point(*positions[line.v1]), point(*positions[line.v2])
        color = '#e67e22' if line.left_sidedef >= 0 else '#3498db'
        lines.append(f'<line x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}" '
                     f'stroke="{color}" stroke-width="1.5"/>')
    for thing in doc.things:
        if thing.type == 1:
            x, y = point(thing.x, thing.y)
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#4ade80"/>')
    return f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}">' + ''.join(lines) + '</svg>'


def build_html_report(exp_dir, out_path=None):
    exp_dir = Path(exp_dir)
    out_path = Path(out_path) if out_path else exp_dir / 'index.html'
    if out_path.exists():
        raise ValueError('gallery destination already exists; preserve prior evidence and choose a fresh path')
    data = json.loads((exp_dir / 'report.json').read_text())
    if data.get('schema') != 'doom_geometry_comparison_v2':
        raise ValueError('requires v2 diagnostic report; preserve historical v1 galleries unchanged')
    html = ['<!doctype html><html lang="en"><meta charset="utf-8">',
            '<title>Geometry reconstruction diagnostics</title>',
            '<style>body{font:16px system-ui;background:#111827;color:#e5e7eb;margin:2rem}'
            'table{border-collapse:collapse;width:100%}td,th{padding:.7rem;text-align:left;border-bottom:1px solid #374151}'
            'article{margin-top:2rem;padding:1rem;background:#1f2937}svg{background:#111827}'
            'p{max-width:75ch}code{overflow-wrap:anywhere}</style>',
            '<h1>Geometry reconstruction diagnostics</h1>',
            f'<p>{escape(data["scope"])}</p>',
            f'<p>Seed: {data["seed"]}. Engine: {escape(data["engine_status"])}. '
            f'{escape(data.get("engine_reason", ""))}</p>',
            '<table><tr><th>Method</th><th>Conditioning</th><th>Published / attempted</th><th>Status</th></tr>']
    for method, summary in data['summary'].items():
        row = next(r for r in data['records'] if r['method'] == method)
        html.append(f'<tr><td>{escape(method.replace("_", " "))}</td>'
                    f'<td>{escape(row["conditioning"])}</td>'
                    f'<td>{summary["published_count"]} / {summary["total_count"]}</td>'
                    f'<td>{escape(summary["status"])}</td></tr>')
    html.append('</table>')
    for row in data['records']:
        html.append(f'<article><h2>{escape(row["method"].replace("_", " "))}</h2>')
        if row.get('reason'):
            html.append(f'<p>{escape(row["reason"])}</p>')
        for stage in ('in_memory_validation', 'serialized_validation', 'final_validation'):
            if stage in row:
                errors = row[stage]['errors']
                html.append(f'<p>{escape(stage.replace("_", " "))}: '
                            f'{escape("; ".join(errors)) if errors else "passed"}</p>')
        if row.get('publication'):
            html.append(f'<p>Nodes built: {row["publication"]["nodes_built"]}.</p>')
        wad_name = row.get('wad_file') or row.get('candidate_wad')
        if wad_name:
            doc, = extract_all_maps(WadFile(str(exp_dir / wad_name)))
            html.append(render_svg(doc))
        html.append('</article>')
    html.append('</html>')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(html) + '\n')
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    print(build_html_report(args.experiment, args.out))

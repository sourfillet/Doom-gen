"""Make a portable SVG comparison of an evaluate-roomgen report.

    python -m tools.render_evaluation generated/experiment --limit 16

Run from an installed project environment. Includes rejected seeds and actual
world dimensions; successful serialization is not labelled engine completion.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def render(directory, *, limit=16):
    from tagging.preview import map_to_svg
    from wad_parser.map_extractor import extract_all_maps
    from wad_parser.wad_reader import WadFile

    directory = Path(directory)
    report = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
    if 'panel' in report:
        panels = [('neural', report['panel'], report.get('geometry', {}), 'sample-panel.wad')]
        panels += [(name, record['panel'], record['geometry'], f'{name}-panel.wad')
                   for name, record in report.get('baselines', {}).items()]
    else:
        panels = []
        for method, summary in report['panels'].items():
            records = [row for row in report['records'] if row['method'] == method]
            panel = [{**row, 'cycles': max(0, row['edges'] - row['rooms'] + 1)} for row in records]
            geometry = {'written_maps': summary['written'],
                        'failed_samples': [row for row in records if not row['written']],
                        'summaries': [row for row in records if row['written']]}
            panels.append((method, panel, geometry, summary.get('path', f'{method}-panel.wad')))
    cards, columns = {}, []
    for name, panel, geometry, filename in panels:
        columns.append(name)
        path = directory / filename
        documents = extract_all_maps(WadFile(str(path))) if path.exists() else []
        failed = {record['seed']: record.get('reason', '') for record in geometry.get('failed_samples', [])}
        if name == 'neural':
            by_name = {doc.meta.map_name: doc for doc in documents}
            written = {record['seed']: by_name[record['map']] for record in geometry.get('summaries', [])}
            assert len(written) == len(documents), 'neural report/WAD map mismatch'
        else:
            attempted = geometry.get('written_maps', 0) + len(failed)
            seeds = [record['seed'] for record in panel[:attempted] if record['seed'] not in failed]
            assert len(seeds) == len(documents), f'{name} report/WAD map mismatch'
            written = dict(zip(seeds, documents))
        for record in panel[:limit]:
            seed = record['seed']
            text = f"{record['rooms']} rooms · {record['edges']} edges · {record['cycles']} cycles"
            doc = written.get(seed)
            if doc is not None:
                svg_name = f'{name}-seed-{seed}.svg'
                (directory / svg_name).write_text(map_to_svg(doc, width=768), encoding='utf-8')
                width, height = doc.bounds.max_x - doc.bounds.min_x, doc.bounds.max_y - doc.bounds.min_y
                body = (f'<p class="ok">Written · {html.escape(doc.meta.map_name)} · '
                        f'{width:g} × {height:g} world units</p>'
                        f'<a href="{html.escape(svg_name)}"><img src="{html.escape(svg_name)}" '
                        f'alt="{html.escape(name)} seed {seed} map"></a>'
                        f'<p><a href="{html.escape(filename)}">Panel WAD</a></p>')
            elif seed in failed:
                body = f'<p class="failed">Rejected</p><pre>{html.escape(failed[seed])}</pre>'
            else:
                body = '<p>Geometry not evaluated.</p>'
                if geometry.get('failure'):
                    body += f'<pre>{html.escape(geometry["failure"])}</pre>'
            cards[name, seed] = f'<article><p>{html.escape(text)}</p>{body}</article>'
    seeds = list(dict.fromkeys(record['seed'] for _, panel, _, _ in panels for record in panel[:limit]))
    rows = ''.join(f'<section><h2>Seed {seed}</h2><div class="row">' + ''.join(
        f'<div><h3>{html.escape(name)}</h3>{cards.get((name, seed), "No record")}</div>'
        for name in columns) + '</div></section>' for seed in seeds)
    counts = ' · '.join(f'{name}: {geometry.get("written_maps", 0)} written / '
                        f'{geometry.get("written_maps", 0) + len(geometry.get("failed_samples", []))} attempted'
                        for name, _, geometry, _ in panels)
    content = ('<!doctype html><html lang="en"><meta charset="utf-8">'
               '<meta name="viewport" content="width=device-width,initial-scale=1">'
               '<title>Room generation comparison</title><style>'
               'body{font:16px system-ui;background:#11151b;color:#e4e9ee;margin:2rem}'
               'a{color:#88beff}.row{display:grid;grid-template-columns:repeat('
               f'{len(columns)},minmax(0,1fr));gap:1rem}}'
               'article{background:#1b222d;padding:1rem;min-height:18rem}'
               'img{width:100%;height:24rem;object-fit:contain}pre{white-space:pre-wrap;overflow-wrap:anywhere}'
               '.ok{color:#85d9a0}.failed{color:#ffad9c}section{margin:2rem 0}'
               '@media(max-width:850px){.row{grid-template-columns:1fr}}'
               '</style><h1>Room generation comparison</h1><p>' + html.escape(counts) + '</p>'
               '<p>Same realization constraints and one attempt per seed. Maps are scaled to fit; '
               'world dimensions are shown. Written maps have passed static validation. '
               'This report does not establish engine completion, combat balance, novelty or human quality.</p>'
               '<p><a href="report.json">Raw report</a></p>' + rows + '</html>')
    output = directory / 'comparison.html'
    output.write_text(content, encoding='utf-8')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--limit', type=int, default=16)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    print(render(args.directory, limit=args.limit))

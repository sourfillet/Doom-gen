"""Trace experimental BSP region differences without changing parser semantics.

Run with ``python -m tools.diagnose_bsp_regions --help``. The candidate decoder
is an explicit, archived Python experiment, never installed over the parser.
Distance, area, topology and empty mismatches remain separate diagnostics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil

from shapely import geos_version_string, union_all

from data.spatial_correspondence import sector_correspondence
from models.training import code_fingerprint, source_preprocessing_fingerprint
from wad_parser.bsp_geometry import GRID, component_polygons
from wad_parser.geometry import sector_points
from wad_parser.map_extractor import extract_all_maps
from wad_parser.wad_reader import WadFile


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _topology(shape):
    parts = [] if shape.is_empty else [shape] if shape.geom_type == 'Polygon' else list(shape.geoms)
    return {'components': len(parts), 'holes': sum(len(p.interiors) for p in parts)}


def compare_regions(source, rebuilt):
    """Keep empty/invalid outcomes and thin interior cracks visible.

    A boundary-band measurement describes localization only. It is deliberately
    not an acceptance threshold: small area does not establish safe collision.
    """
    result = {'source_area': source.area, 'rebuilt_area': rebuilt.area,
              'source_topology': _topology(source), 'rebuilt_topology': _topology(rebuilt)}
    result['topology_equal'] = result['source_topology'] == result['rebuilt_topology']
    if not source.is_valid or not rebuilt.is_valid:
        return {**result, 'status': 'invalid_geometry'}
    delta = source.symmetric_difference(rebuilt)
    result.update(symmetric_difference_area=delta.area,
                  relative_area_error=abs(source.area - rebuilt.area) / max(source.area, rebuilt.area, 1))
    if source.is_empty or rebuilt.is_empty:
        return {**result, 'status': 'empty_both' if source.is_empty and rebuilt.is_empty else 'empty_mismatch',
                'hausdorff': None}
    result.update(status='compared', hausdorff=source.hausdorff_distance(rebuilt),
                  boundary_band_width=4 * GRID,
                  difference_outside_source_boundary_band=delta.difference(source.boundary.buffer(4 * GRID)).area)
    return result


def summarize(sectors):
    """Never classify an empty or unresolved sector as a resolved map."""
    return {'statuses': dict(Counter(s['status'] for s in sectors)),
            'max_hausdorff': max((s.get('hausdorff') or 0 for s in sectors), default=0),
            'beyond_one_unit': sum((s.get('hausdorff') or 0) > 1 + 1e-6 for s in sectors),
            'empty_mismatches': sum(s['status'] == 'empty_mismatch' for s in sectors),
            'topology_changes': sum(s.get('topology_equal') is False for s in sectors),
            'area_mismatches': sum(s.get('relative_area_error', 0) >= 1e-5 for s in sectors),
            'unresolved': sum(s['status'] not in {'compared', 'empty_both', 'deleted_unreferenced'} for s in sectors)}


def _shape(rings):
    return union_all([polygon for _, polygon in component_polygons(rings)])


def _instrument_decoder(raw, path):
    # Capture polygons before union/precision snapping, retaining the candidate's
    # algorithm verbatim. Refuse incompatible experiments instead of guessing.
    anchor = '            sectors[owner].append(polygon)'
    if raw.count(anchor) != 1:
        raise ValueError('candidate decoder has no unique leaf capture anchor')
    instrumented = raw.replace(anchor, anchor + '\n            _diagnostic_leaves.append('
                               '(leaf, owner, polygon, segs[first:first + count]))')
    namespace = {'__name__': 'isolated_bsp_diagnostic', '__file__': str(path), '_diagnostic_leaves': []}
    exec(compile(instrumented, str(path), 'exec'), namespace)
    return namespace, instrumented


def trace_leaves(leaves, doc, delta):
    """Retain authored owners/actions for each leaf intersecting a discrepancy."""
    rows = []
    for leaf, owner, polygon, segs in leaves:
        overlap = polygon.intersection(delta).area
        if overlap <= GRID:
            continue
        boundaries = []
        for seg in segs:
            line_id, direction = seg[3:5]
            if line_id is None:
                boundaries.append({'linedef': None, 'direction': direction})
                continue
            line = doc.linedefs[line_id]
            right, left = (doc.sidedefs[sid].sector if sid >= 0 else None
                           for sid in (line.right_sidedef, line.left_sidedef))
            boundaries.append({'linedef': line_id, 'direction': direction,
                               'right_sector': right, 'left_sector': left,
                               'self_referencing': right is not None and right == left,
                               'special': line.special, 'tag': line.tag})
        rows.append({'leaf': leaf, 'owner': owner, 'area': polygon.area,
                     'difference_overlap_area': overlap, 'bounds': list(polygon.bounds),
                     'all_boundaries_self_referencing': all(b.get('self_referencing') for b in boundaries),
                     'boundaries': boundaries})
    return rows


def diagnose_pair(raw_path, compiled_path, namespace):
    raw, = extract_all_maps(WadFile(str(raw_path)))
    wad = WadFile(str(compiled_path))
    compiled, = extract_all_maps(wad)
    correspondence = sector_correspondence(raw, compiled)
    result = {'correspondence': correspondence}
    if correspondence['status'] != 'complete':
        return {**result, 'status': 'incomplete_correspondence'}
    group, = wad.find_map_groups()
    namespace['_diagnostic_leaves'] = []
    regions, info = namespace['reconstruct_bsp'](group, compiled.vertices, compiled.linedefs,
                                                compiled.sidedefs, len(compiled.sectors))
    mapping = {p['source_sector']: p['target_sector'] for p in correspondence['mapping']}
    sectors = []
    for sector in raw.sectors:
        sid = sector.id
        if sid not in mapping:
            sectors.append({'sector': sid, 'status': 'deleted_unreferenced'})
            continue
        tid = mapping[sid]
        source = _shape(sector_points(raw, sid))
        rebuilt = _shape(regions[tid])
        row = {'sector': sid, 'other_sector': tid, **compare_regions(source, rebuilt)}
        if row['status'] == 'empty_mismatch' or (row.get('hausdorff') or 0) > 1 + 1e-6:
            cells = [entry for entry in namespace['_diagnostic_leaves'] if entry[1] == tid]
            # Sensitivity is reported, never used to replace candidate geometry.
            row['union_sensitivity'] = [
                {'grid_size': grid, **compare_regions(source, union_all([c[2] for c in cells], grid_size=grid))}
                for grid in (None, GRID, 1 / 1024)]
            delta = source.symmetric_difference(rebuilt)
            row['leaf_evidence'] = trace_leaves(namespace['_diagnostic_leaves'], compiled, delta)
        sectors.append(row)
    return {**result, 'status': 'diagnosed', 'bsp': info, 'sectors': sectors, 'summary': summarize(sectors)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-report', type=Path, required=True,
                        help='Exact-line v2 report defining the immutable map panel.')
    parser.add_argument('--reconstruction-root', type=Path, required=True,
                        help='Directory containing <map stem>/{raw,compiled}.wad.')
    parser.add_argument('--candidate-decoder', type=Path, required=True,
                        help='Trusted experimental Python decoder to run in isolation.')
    parser.add_argument('--out', type=Path, required=True, help='New experiment directory (must not exist).')
    args = parser.parse_args(argv)
    prior_bytes = args.prior_report.read_bytes()
    prior = json.loads(prior_bytes)
    if prior.get('schema') != 'exact_authored_line_bsp_probe_v2' or prior.get('status') != 'complete':
        raise ValueError('requires a complete exact-line v2 report')
    candidate = args.candidate_decoder.read_bytes()
    if _sha(candidate) != prior['candidate_decoder_sha256']:
        raise ValueError('candidate decoder does not match prior report')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'prior-report.json').write_bytes(prior_bytes)
    (args.out / 'candidate_bsp_geometry.py').write_bytes(candidate)
    namespace, instrumented = _instrument_decoder(candidate.decode(), args.out / 'instrumented_decoder.py')
    (args.out / 'instrumented_decoder.py').write_text(instrumented)
    root = Path(__file__).resolve().parents[1]
    shutil.copyfile(__file__, args.out / 'runner.py')
    for directory in ('data', 'models', 'cli', 'wad_parser', 'wad_writer', 'generation', 'roomgraph', 'tagging', 'tools'):
        for path in (root / directory).rglob('*.py'):
            dest = args.out / 'source' / path.relative_to(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    signature = source_preprocessing_fingerprint()
    report = {'schema': 'bsp_region_diagnostics_v1', 'status': 'running',
              'source_preprocessing_fingerprint': signature, 'code_fingerprint': code_fingerprint(),
              'candidate_sha256': _sha(candidate), 'instrumented_decoder_sha256': _sha(instrumented.encode()),
              'prior_report_sha256': _sha(prior_bytes), 'runner_sha256': _sha(Path(__file__).read_bytes()),
              'versions': {'python': platform.python_version(), 'geos': geos_version_string,
                           **{p: importlib.metadata.version(p) for p in ('shapely', 'numpy', 'networkx')}},
              'policy': {'panel': 'all prior records, no resampling', 'attempts_per_map': 1, 'seed': None,
                         'scope': 'diagnostic only; no source eligibility, parser, cache or gameplay certification'},
              'records': []}
    for row in prior['records']:
        stem = Path(row['path']).stem
        inputs = args.out / 'inputs' / stem
        inputs.mkdir(parents=True)
        hashes = {}
        for name in ('raw.wad', 'compiled.wad'):
            original = args.reconstruction_root / stem / name
            raw = original.read_bytes()
            (inputs / name).write_bytes(raw)
            hashes[name] = {'path': str(original.resolve()), 'sha256': _sha(raw)}
        try:
            result = diagnose_pair(inputs / 'raw.wad', inputs / 'compiled.wad', namespace)
        except Exception as exc:
            result = {'status': 'error', 'reason': f'{type(exc).__name__}: {exc}'}
        result.update(path=row['path'], map=row['map'], split=row['split'], inputs=hashes)
        payload = (json.dumps(result, indent=2, allow_nan=False) + '\n').encode()
        (args.out / row['path']).write_bytes(payload)
        brief = {k: v for k, v in result.items() if k not in {'sectors', 'correspondence', 'bsp'}}
        brief['sha256'] = _sha(payload)
        report['records'].append(brief)
        (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(row['path'], result['status'], result.get('summary', result.get('reason')), flush=True)
    report.update(status='complete', source_changed=source_preprocessing_fingerprint() != signature
                  or code_fingerprint() != report['code_fingerprint'])
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return int(report['source_changed'] or any(r['status'] != 'diagnosed' for r in report['records']))


if __name__ == '__main__':
    raise SystemExit(main())

"""Save model-free geometry reconstructions, overlays, and structural metrics."""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import click
import numpy as np

from data.rasterizer import rasterize, _fill_polygon
from generation.vectorizer import vectorize
from generation.validator import validate_and_repair, validate_document
from generation.style import transfer_source_attributes
from roomgraph.segment import segment
from roomgraph.assemble_real import assemble_real
from roomgraph.teleports import require_connections
from tagging.preview import map_to_svg
from wad_parser.geometry import sector_points
from wad_parser.types import MapDocument


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    sizes = []
    while remaining:
        queue = deque([remaining.pop()])
        size = 0
        while queue:
            row, col = queue.popleft()
            size += 1
            for other in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if other in remaining:
                    remaining.remove(other)
                    queue.append(other)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def _decoded_occupancy(doc, resolution, units_per_pixel):
    occupancy = np.zeros((resolution, resolution), dtype=bool)
    for sector in doc.sectors:
        mask = np.zeros_like(occupancy)
        for ring in sector_points(doc, sector.id):
            points = np.array(ring)
            rows, cols = _fill_polygon(mask.shape, points[:, 0] / units_per_pixel,
                                       resolution - points[:, 1] / units_per_pixel)
            mask[rows, cols] ^= True
        occupancy |= mask
    return occupancy


def _stats(doc):
    from wad_parser.traversal import distances_from_start
    return {'sectors': len(doc.sectors), 'linedefs': len(doc.linedefs),
            'things': len(doc.things), 'boundary_issues': doc.derived.geometry_issues,
            'raw_boundary_issues': doc.derived.raw_geometry_issues,
            'bsp_geometry': doc.derived.bsp_geometry,
            'reachable_sectors': len(distances_from_start(doc)),
            'solid_wall_fraction': sum(line.left_sidedef == -1 or line.flags.impassable for line in doc.linedefs) / max(1, len(doc.linedefs)),
            'floor_range': [min((s.floor_height for s in doc.sectors), default=0),
                            max((s.floor_height for s in doc.sectors), default=0)]}


def diagnose(path, out_dir, resolution=128, coarse_factor=2):
    path, out_dir = Path(path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = MapDocument.from_dict(json.loads(path.read_text()))
    raster, metadata = rasterize(source, resolution)
    np.savez_compressed(out_dir / 'raster.npz', raster=raster, metadata=json.dumps(metadata))
    (out_dir / 'source.json').write_text(json.dumps(source.to_dict()))
    (out_dir / 'source.svg').write_text(map_to_svg(source))
    report = {'source_path': str(path), 'resolution': resolution, 'coarse_factor': coarse_factor,
              'source': _stats(source), 'source_components': _components(raster[9] > 0.5)}
    try:
        cell_units = coarse_factor / metadata['scale']
        doc = vectorize(raster, cell_units=cell_units, coarse_factor=coarse_factor)
        (out_dir / 'before-repair.json').write_text(json.dumps(doc.to_dict()))
        (out_dir / 'before-repair.svg').write_text(map_to_svg(doc))
        occupied = _decoded_occupancy(doc, resolution, cell_units / coarse_factor)
        original = raster[9] > 0.5
        report['raster_reconstruction'] = {
            'occupancy_iou': float((occupied & original).sum() / max(1, (occupied | original).sum())),
            'components': _components(occupied), 'stats_before_repair': _stats(doc)}
        report['source_transfer'] = transfer_source_attributes(doc, source, metadata, cell_units, coarse_factor)
        report['repairs'] = validate_and_repair(doc, repair_heights=False)
        report['validation'] = validate_document(doc, require_exit=False)
        (out_dir / 'decoded.json').write_text(json.dumps(doc.to_dict()))
        (out_dir / 'decoded.svg').write_text(map_to_svg(doc))
    except ValueError as exc:
        report['raster_failure'] = str(exc)
    try:
        graph = segment(source, playable_only=True)
        report['room_graph'] = {**graph.summary(), 'excluded_sectors': graph.excluded_sectors,
                                'source_components_reduced_to_representative_polygons':
                                    sum(max(0, len(room.polygons) - 1) for room in graph.rooms),
                                'interior_sectors_reduced_to_representative_heights':
                                    sum(max(0, len(room.sectors) - 1) for room in graph.rooms),
                                'preserved_holes': sum(len(room.holes) for room in graph.rooms)}
        assembled = assemble_real(graph)
        (out_dir / 'assembled.svg').write_text(map_to_svg(assembled))
        (out_dir / 'assembled.json').write_text(json.dumps(assembled.to_dict()))
        report['room_graph']['connection_realization'] = assembled.extra.get('connection_realization', [])
        try:
            require_connections(assembled)
        except ValueError as exc:
            report['room_graph']['failure'] = str(exc)
        report['room_graph']['validation'] = validate_document(assembled, require_exit=False, require_connected=True)
    except ValueError as exc:
        report['room_graph_failure'] = str(exc)
    (out_dir / 'report.json').write_text(json.dumps(report, indent=2))
    return report


@click.command()
@click.option('--maps-dir', required=True, type=click.Path(exists=True, file_okay=False))
@click.option('--out', required=True, type=click.Path(file_okay=False))
@click.option('--resolution', default=128, type=click.IntRange(min=8))
@click.option('--coarse-factor', default=2, type=click.IntRange(min=1))
@click.option('--limit', default=8, type=click.IntRange(min=0), help='0 processes the complete corpus.')
def main(maps_dir, out, resolution, coarse_factor, limit):
    paths = sorted(path for path in Path(maps_dir).glob('*.json') if not path.name.endswith('.textures.json'))
    if limit:
        paths = paths[:limit]
    reports = []
    for path in paths:
        report = diagnose(path, Path(out) / path.stem, resolution, coarse_factor)
        reports.append(report)
        click.echo(f"{path.name}: IoU {report.get('raster_reconstruction', {}).get('occupancy_iou', 'failed')}")
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / 'summary.json').write_text(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()

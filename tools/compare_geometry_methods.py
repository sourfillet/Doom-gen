"""Compare authored/scaffold reconstruction with procedural geometry diagnostics.

A fixed-source scaffold control has more conditioning than a RoomGraph sampler.
Reports disclose that distinction and never infer engine execution from imports.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from data.spatial_prediction import build_prediction_scaffold, decode_prediction, digest, graph_from_source_record
from data.spatial_targets import build_spatial_target
from generation.pipeline import documents_to_wad
from generation.progression import ensure_exit
from generation.validator import validate_document
from roomgraph.assemble import assemble
from roomgraph.assemble_real import assemble_real
from roomgraph.layout import radial_layout
from roomgraph.shapes import apply_shapes, build_library, save_library
from roomgraph.spatial import reconstruct_spatial_target
from roomgraph.teleports import require_connections
from wad_parser.geometry import sector_area_perimeter
from wad_parser.map_extractor import extract_all_maps
from wad_parser.types import MapDocument
from wad_parser.wad_reader import WadFile
from wad_writer.wad_writer import write_wad

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEM = '6ce8b4b0e602140261a202180dbe7c317453b8f1991985258e503643caf2ee26-0018'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _geometry_metrics(doc):
    width, height = doc.bounds.max_x - doc.bounds.min_x, doc.bounds.max_y - doc.bounds.min_y
    envelope = width * height
    area = sum(sector_area_perimeter(doc, s.id)[0] for s in doc.sectors)
    floors = sorted({s.floor_height for s in doc.sectors})
    ceilings = sorted({s.ceil_height for s in doc.sectors})
    return {'sectors': len(doc.sectors), 'linedefs': len(doc.linedefs), 'vertices': len(doc.vertices),
            'width': width, 'height': height, 'envelope_area': envelope, 'total_floor_area': area,
            'area_efficiency_pct': 100 * area / envelope if envelope else 0,
            'distinct_floor_heights': floors, 'distinct_ceil_heights': ceilings,
            'distinct_floor_count': len(floors), 'distinct_ceil_count': len(ceilings),
            'one_sided_linedefs': sum(l.left_sidedef < 0 for l in doc.linedefs),
            'two_sided_linedefs': sum(l.left_sidedef >= 0 for l in doc.linedefs),
            'geometry_issues': doc.derived.geometry_issues}


def ensure_training_shape_library(out_dir):
    manifest_path = ROOT / 'generated/repair-experiments/adjoining-architecture/spatial-manifest-v1.json'
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    corpus = ROOT / 'generated/repair-experiments/current-complete-corpus/maps'
    docs, inputs = [], []
    for entry in manifest['entries']:
        if entry['split'] != 'train':
            continue
        path = corpus / entry['path']
        raw = path.read_bytes()  # Missing training inputs must not be silently omitted.
        if hashlib.sha256(raw).hexdigest() != entry.get('source_document_sha256'):
            raise ValueError(f'frozen training source fingerprint mismatch: {entry["path"]}')
        docs.append(MapDocument.from_dict(json.loads(raw)))
        inputs.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()})
    lib = build_library(docs, playable_only=True)
    save_library(lib, out_dir / 'training_shape_library.json')
    (out_dir / 'shape-library-manifest.json').write_bytes(manifest_bytes)
    (out_dir / 'shape-library-provenance.json').write_text(json.dumps({
        'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(), 'split': 'train', 'inputs': inputs}, indent=2) + '\n')
    return lib


def evaluate_case(graph, target, out_dir, shape_library, *, seed=7, checkpoint=None):
    """One attempt per arm, preserving raw candidates and all failed checks.

    No edge masks are inferred and no failed candidate is resampled. The source
    target/scaffold retains authored controls/assets/population; procedural arms
    do not share that conditioning. This is not a matched gameplay experiment.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for method in ('authored_reconstruction', 'learned_scaffold', 'procedural_adjoining', 'outline_retrieval'):
        row = {'method': method, 'seed': seed, 'attempts': 1, 'status': 'pending',
               'graph_sha256': digest(graph.to_dict()), 'requested_connections': len(graph.connections),
               'engine_status': 'not_run', 'engine_reason': 'no engine controller invoked',
               'conditioning': 'source authoring scaffold' if method in
               {'authored_reconstruction', 'learned_scaffold'} else 'RoomGraph and procedural rules'}
        if method == 'learned_scaffold' and checkpoint is None:
            row.update(status='skipped', reason='no versioned spatial checkpoint supplied')
            records.append(row)
            continue
        try:
            if method == 'authored_reconstruction':
                doc = reconstruct_spatial_target(target)
            elif method == 'learned_scaffold':
                from models.spatial import predict_checkpoint
                scaffold, _ = build_prediction_scaffold(target)
                loaded, prediction = predict_checkpoint(checkpoint, expected_scaffold=scaffold)
                row.update(checkpoint_sha256=_sha(checkpoint), prediction_sha256=digest(prediction))
                (out_dir / 'learned-prediction.json').write_text(json.dumps(prediction, indent=2) + '\n')
                doc = decode_prediction(loaded, prediction)
            elif method == 'procedural_adjoining':
                doc = assemble(copy.deepcopy(graph), geometry_profile='adjoining', seed=seed)
                require_connections(doc)
            else:
                retrieved = copy.deepcopy(graph)
                apply_shapes(retrieved, shape_library, seed=seed)
                doc = assemble_real(retrieved, positions=radial_layout(retrieved, seed=seed), seed=seed)
                require_connections(doc)
            doc.meta.map_name = 'MAP01'
            # An exit-only procedural policy is common to every arm. Never
            # reconnect, alter floors or insert substitute teleport edges.
            row['exit_policy'] = ensure_exit(doc)
            raw_path = out_dir / f'{method}-candidate.wad'
            write_wad([doc], raw_path)
            row.update(candidate_wad=raw_path.name, candidate_sha256=_sha(raw_path), metrics=_geometry_metrics(doc))
            row['in_memory_validation'] = validate_document(doc, require_exit=True, require_connected=True)
            serialized, = extract_all_maps(WadFile(str(raw_path)))
            row['serialized_validation'] = validate_document(serialized, require_exit=True, require_connected=True)
            if row['in_memory_validation']['errors'] or row['serialized_validation']['errors']:
                row.update(status='rejected', reason='physical/gameplay validation failed; candidate retained')
            else:
                path = out_dir / f'{method}.wad'
                publication = documents_to_wad([doc], path, require_playable=True)
                row.update(status='published', publication=publication,
                           wad_file=path.name, wad_sha256=_sha(path))
                built, = extract_all_maps(WadFile(str(path)))
                row['final_validation'] = validate_document(built, require_exit=True, require_connected=True)
                if row['final_validation']['errors']:
                    raise ValueError('final serialized/compiled geometry failed validation')
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            row.update(status='rejected', reason=f'{type(exc).__name__}: {exc}')
        records.append(row)
    return records


def run_comparison(out_dir, stem=DEFAULT_STEM, seed=7, *, checkpoint=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    corpus = ROOT / 'generated/repair-experiments/current-complete-corpus'
    source_path, graph_path = corpus / 'maps' / f'{stem}.json', corpus / 'records' / f'{stem}.json'
    source_bytes, record_bytes = source_path.read_bytes(), graph_path.read_bytes()
    graph = graph_from_source_record(source_bytes, json.loads(record_bytes))
    source = MapDocument.from_dict(json.loads(source_bytes))
    target = build_spatial_target(source, graph)
    (out_dir / 'source.json').write_bytes(source_bytes)
    (out_dir / 'graph-record.json').write_bytes(record_bytes)
    (out_dir / 'target.json').write_text(json.dumps(target, indent=2) + '\n')
    for name in ('compare_geometry_methods.py', 'render_geometry_comparison.py'):
        shutil.copyfile(ROOT / 'tools' / name, out_dir / name)
    from models.training import code_fingerprint, source_preprocessing_fingerprint
    signature = code_fingerprint()
    for package in ('data', 'models', 'cli', 'wad_parser', 'wad_writer', 'generation', 'roomgraph', 'tagging'):
        for path in (ROOT / package).rglob('*.py'):
            destination = out_dir / 'source' / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    if checkpoint is not None:
        shutil.copyfile(checkpoint, out_dir / 'model.pt')
        checkpoint = out_dir / 'model.pt'
    shape_library = ensure_training_shape_library(out_dir)
    records = evaluate_case(graph, target, out_dir / 'panel', shape_library, seed=seed, checkpoint=checkpoint)
    for row in records:
        for field in ('wad_file', 'candidate_wad'):
            if field in row:
                row[field] = 'panel/' + row[field]
        row['panel'] = stem
    report = {'schema': 'doom_geometry_comparison_v2', 'stem': stem, 'seed': seed,
              'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
              'graph_record_sha256': hashlib.sha256(record_bytes).hexdigest(),
              'code_fingerprint': signature, 'source_preprocessing_fingerprint': source_preprocessing_fingerprint(),
              'source_changed': code_fingerprint() != signature,
              'engine_status': 'not_run', 'engine_reason': 'no engine controller invoked',
              'scope': 'Source-scaffold memorization/reconstruction controls and procedural diagnostics. '
                       'Different conditioning/population/action policies; no matched gameplay, holdout or learned advantage claim.',
              'records': records,
              'summary': {r['method']: {'published_count': int(r['status'] == 'published'), 'total_count': 1,
                                        'status': r['status']} for r in records}}
    (out_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='Fresh output directory, never overwrites v1 evidence.')
    parser.add_argument('--checkpoint', type=Path, help='Versioned spatial memorization v2 checkpoint.')
    parser.add_argument('--stem', default=DEFAULT_STEM)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    report = run_comparison(args.out, stem=args.stem, seed=args.seed, checkpoint=args.checkpoint)
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()

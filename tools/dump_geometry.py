"""Faithful diagnostic PWAD export, including geometrically invalid predictions.

This is deliberately separate from playable publication. It validates binary
syntax/ranges/references, preserves every authoring record, embeds provenance,
and checks core bytes after reading the WAD back. It never builds/repairs geometry
or certifies engine safety. Incomplete streams can be retained as trace-only WADs
with no map marker; they must never be described as generated complete maps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from data.geometry_tokens import _record_bytes, encode_prediction
from data.learned_geometry import AUTHORING_FIELDS, CORE_LUMPS
from wad_parser.wad_reader import WadFile
from wad_writer.assembler import MAP_LUMP_ORDER
from wad_writer.wad_writer import write_lumps


def dump_result(result, path, *, provenance=None):
    """Write one unmodified prediction or a clearly marked incomplete trace.

    Physical validation is retained from inference, not bypassed or relabeled.
    Absence of a complete prediction never invents geometry or list endings.
    """
    path = Path(path)
    if path.exists():
        raise ValueError('diagnostic WAD already exists; preserve it and choose a fresh path')
    prediction = result.get('prediction')
    metadata = {'schema': 'doom_geometry_diagnostic_wad_v1',
                'kind': 'raw_authoring' if prediction is not None else 'incomplete_trace_only',
                'inference_status': result['status'], 'engine': 'not_run', 'nodes': 'not_built',
                'playable_publication': False, 'repairs': [],
                'provenance': provenance or {}, 'sample': result['sample'],
                'assessment': result.get('assessment'), 'reason': result.get('reason')}
    lumps, core = [], {}
    if prediction is not None:
        encode_prediction(prediction)  # File grammar/ranges/references, not geometry.
        core = {lump: b''.join(_record_bytes(name, record) for record in prediction[name])
                for name, lump in zip(AUTHORING_FIELDS, CORE_LUMPS)}
        sections = {name: b'' for name in MAP_LUMP_ORDER}
        sections.update(core)
        # This visibility matrix is serialization housekeeping, not geometry.
        sections['REJECT'] = bytes((len(prediction['sectors'])**2 + 7)//8)
        lumps = [('MAP01', b''), *[(name, sections[name]) for name in MAP_LUMP_ORDER]]
        metadata['counts'] = {name: len(prediction[name]) for name in AUTHORING_FIELDS}
        metadata['core_sha256'] = {name: hashlib.sha256(data).hexdigest() for name, data in core.items()}
    payload = json.dumps(metadata, sort_keys=True, allow_nan=False).encode()
    lumps.append(('GENMETA', payload))
    write_lumps(lumps, path)
    readback = WadFile(str(path))
    entries = {entry.name: entry.data for entry in readback.lumps}
    if entries['GENMETA'] != payload or any(entries.get(name) != data for name, data in core.items()):
        raise ValueError('diagnostic WAD readback differs; retain file for diagnosis')
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'kind': metadata['kind'], 'core_bytes_verified': bool(core),
            'playable_publication': False, 'inference_status': result['status']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, help='Retained inference JSON, not an authored source scaffold.')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    path = Path(args.report)
    raw = path.read_bytes()
    try:
        result = dump_result(json.loads(raw), args.out,
                             provenance={'inference_report_sha256': hashlib.sha256(raw).hexdigest()})
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

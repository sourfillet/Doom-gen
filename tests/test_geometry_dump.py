import copy
import json

import pytest

from data.geometry_tokens import _record_bytes
from data.learned_geometry import AUTHORING_FIELDS, CORE_LUMPS, build_target
from generation.learned_geometry import decode_prediction
from tests.geometry_fixtures import two_rooms
from tools.dump_geometry import dump_result
from wad_parser.wad_reader import WadFile


def test_rejected_geometry_wad_retains_exact_authoring_and_failed_verdict(tmp_path):
    prediction = build_target(two_rooms())['supervision']
    prediction['linedefs'][0]['v2'] = prediction['linedefs'][0]['v1']
    before = copy.deepcopy(prediction)
    with pytest.raises(ValueError, match='zero length'):
        decode_prediction(prediction)
    result = {'status': 'rejected', 'prediction': prediction,
              'sample': {'status': 'complete', 'tokens': [1, 2]}, 'assessment': {'status': 'rejected'}}
    path = tmp_path / 'rejected.wad'
    report = dump_result(result, path)
    lumps = {entry.name: entry.data for entry in WadFile(str(path)).lumps}
    for name, lump in zip(AUTHORING_FIELDS, CORE_LUMPS):
        assert lumps[lump] == b''.join(_record_bytes(name, record) for record in prediction[name])
    meta = json.loads(lumps['GENMETA'])
    assert meta['inference_status'] == 'rejected'
    assert meta['playable_publication'] is False
    assert report['core_bytes_verified']
    assert prediction == before
    with pytest.raises(ValueError, match='already exists'):
        dump_result(result, path)


def test_incomplete_wad_is_trace_only_and_does_not_invent_map_records(tmp_path):
    result = {'status': 'budget_exhausted', 'prediction': None,
              'sample': {'status': 'budget_exhausted', 'tokens': [1, 4]}}
    path = tmp_path / 'incomplete.wad'
    report = dump_result(result, path)
    wad = WadFile(str(path))
    assert wad.find_map_groups() == []
    assert wad.lump_names() == ['GENMETA']
    assert json.loads(wad.lumps[0].data)['sample']['tokens'] == [1, 4]
    assert report['kind'] == 'incomplete_trace_only'


def test_raw_export_still_rejects_out_of_range_references_without_creating_file(tmp_path):
    prediction = build_target(two_rooms())['supervision']
    prediction['linedefs'][0]['v1'] = 60000
    path = tmp_path / 'bad-reference.wad'
    with pytest.raises(ValueError):
        dump_result({'prediction': prediction, 'status': 'rejected', 'sample': {}}, path)
    assert not path.exists()


def test_dump_import_and_help_work_without_torch():
    import subprocess
    import sys
    code = '''
import importlib.abc, runpy, sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('Torch imported')
sys.meta_path.insert(0, NoTorch())
sys.argv = ['dump', '--help']
try: runpy.run_module('tools.dump_geometry', run_name='__main__')
except SystemExit as exc: assert exc.code == 0
'''
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)

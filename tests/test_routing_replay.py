import copy
import hashlib
import json

import pytest

from roomgraph.types import Connection
from tests.geometry_fixtures import wheel_graph
from tools.evaluate_routing import evaluate
from tools.render_evaluation import render


@pytest.mark.parametrize('geometry_profile', ['classic', 'varied', 'compact', 'adjoining'])
def test_snapshot_replay_keeps_rejected_and_previously_unattempted_seeds(tmp_path, geometry_profile):
    graph = wheel_graph()
    invalid = copy.deepcopy(graph)
    invalid.rooms = invalid.rooms[:6]
    invalid.connections = [Connection(a, b, 'corridor', 128) for a in range(3) for b in range(3, 6)]
    panel = [{'seed': seed, 'graph': g.to_dict()} for seed, g in [(5, graph), (6, invalid), (7, graph)]]
    source = tmp_path / 'input.json'
    source.write_text(json.dumps({'panel': panel, 'geometry': {
        'written_maps': 1, 'summaries': [{'seed': 5, 'map': 'MAP01'}],
        'failed_samples': [{'seed': 6, 'reason': 'non-planar'}]}}))
    before = source.read_bytes()
    directory = tmp_path / 'replay'
    report = evaluate(source, directory, samples=3, geometry_profile=geometry_profile)
    assert [row['original_written'] for row in report['records']] == [True, False, None]
    assert [row['written'] for row in report['records']] == [True, False, True]
    assert 'non-planar' in report['records'][1]['reason']
    assert report['panels']['neural']['written'] == 2
    assert report['status'] == 'complete'
    assert report['geometry_profile'] == geometry_profile
    assert report['records'][0]['geometry_seed'] == 5
    assert report['records'][0]['geometry_profile'] == geometry_profile
    assert not report['code_changed_during_evaluation']
    assert report['input_sha256'] == hashlib.sha256(before).hexdigest()
    assert (directory / report['input_snapshot']).read_bytes() == source.read_bytes() == before
    assert (directory / 'evaluate_routing.py').exists()
    html = render(directory).read_text()
    assert html.count('<article>') == 3 and html.count('<img ') == 2
    assert 'Rejected' in html
    with pytest.raises(ValueError, match='must be empty'):
        evaluate(source, directory)


def test_unknown_snapshot_version_is_rejected_before_creating_outputs(tmp_path):
    source = tmp_path / 'input.json'
    source.write_text(json.dumps({'graph_snapshot_version': 2}))
    with pytest.raises(ValueError, match='snapshot version'):
        evaluate(source, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_static_success_is_not_reported_as_a_written_map_when_publication_fails(tmp_path, monkeypatch):
    import generation.pipeline
    def fail_publication(*args, **kwargs):
        raise ValueError('compiled map is invalid')
    monkeypatch.setattr(generation.pipeline, 'documents_to_wad', fail_publication)
    source = tmp_path / 'input.json'
    source.write_text(json.dumps({'records': [{'method': 'neural', 'seed': 0, 'graph': wheel_graph().to_dict()}]}))
    report = evaluate(source, tmp_path / 'out')
    assert report['records'][0]['validated'] and not report['records'][0]['written']
    assert report['records'][0]['failure_stage'] == 'publication'
    assert report['panels']['neural']['written'] == 0
    assert 'compiled map is invalid' in report['panels']['neural']['publication_error']


def test_replay_retains_other_seeds_when_one_compiled_map_is_invalid(tmp_path, monkeypatch):
    from tests.pipeline_fixtures import rejecting_builder
    calls = rejecting_builder(monkeypatch, {2})
    graph = wheel_graph().to_dict()
    source = tmp_path / 'input.json'
    source.write_text(json.dumps({'records': [{'method': 'neural', 'seed': i, 'graph': graph}
                                             for i in range(3)]}))
    report = evaluate(source, tmp_path / 'out', samples=3)
    assert len(calls) == 3
    assert [r['validated'] for r in report['records']] == [True, True, True]
    assert [r['written'] for r in report['records']] == [True, False, True]
    assert report['records'][1]['failure_stage'] == 'publication'
    assert report['records'][2]['map'] == 'MAP02'
    assert report['panels']['neural']['validated'] == 3
    assert report['panels']['neural']['written'] == 2


def test_replay_respects_explicit_failed_publication_instead_of_legacy_count(tmp_path, monkeypatch):
    monkeypatch.setattr('generation.pipeline.build_nodes', lambda path: False)
    source = tmp_path / 'input.json'
    source.write_text(json.dumps({'panel': [{'seed': 7, 'graph': wheel_graph().to_dict()}],
                                 'geometry': {'written_maps': 0, 'failed_samples': [],
                                              'outcomes': [{'seed': 7, 'validated': True, 'written': False,
                                                            'failure_stage': 'publication'}]}}))
    report = evaluate(source, tmp_path / 'out')
    assert report['records'][0]['original_written'] is False
    assert report['records'][0]['written'] is True

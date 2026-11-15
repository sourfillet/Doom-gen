from data.source_endings import assess_source_ending, SOURCE_ENDINGS_VERSION
from roomgraph.segment import segment
from tests.geometry_fixtures import neck_map
from tests.progression_fixtures import branch_document
from tests.geometry_fixtures import polygon_document
import pytest


def test_an_exit_behind_an_intra_sector_neck_is_not_a_complete_training_ending():
    doc, _ = neck_map(24)
    graph = segment(doc, playable_only=True)
    assert assess_source_ending(doc, graph, version='reachable_exit_v1')['complete']
    ending = assess_source_ending(doc, graph)
    assert not ending['complete'] and 'physical' in ending['reason']
    assert ending['version'] == SOURCE_ENDINGS_VERSION


def test_incompatible_key_branches_are_censored_without_rewriting_historical_targets():
    doc = branch_document()
    graph = segment(doc, playable_only=True)
    assert assess_source_ending(doc, graph, version='reachable_exit_v1')['complete']
    ending = assess_source_ending(doc, graph)
    assert not ending['complete'] and ending['history']['status'] == 'failure'


def test_a_physical_key_return_route_is_a_supported_complete_example():
    doc = branch_document(return_from_key=True)
    ending = assess_source_ending(doc, segment(doc, playable_only=True))
    assert ending['complete'] and ending['history']['status'] == 'success'


@pytest.mark.parametrize('step, complete', [(0, True), (24, True), (25, False)])
def test_source_ending_uses_stable_oblique_contact_with_the_climb_limit(step, complete):
    doc = polygon_document([[[(0, 0), (0, 40), (129, 40), (256, 0)]],
                            [[(256, 0), (129, 40), (512, 40), (512, 0)]]], start=(32, 20))
    doc.sectors[1].floor_height = step
    exit_line = next(line for line in doc.linedefs
                     if doc.vertices[line.v1].x == doc.vertices[line.v2].x == 512)
    exit_line.special = 11
    ending = assess_source_ending(doc, segment(doc, playable_only=True))
    assert ending['complete'] == complete

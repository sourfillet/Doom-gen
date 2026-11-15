import copy

import pytest

from tests.progression_fixtures import one_shot_revisit_puzzle, consumption_walk_puzzle
from wad_parser.progression import search_exit_history


@pytest.mark.parametrize('special,expected', [(39, 'failure'), (97, 'success')])
def test_key_histories_cannot_reuse_a_consumed_one_shot_teleport(special, expected):
    from generation.validator import validate_document
    doc = one_shot_revisit_puzzle(special)
    before = copy.deepcopy(doc.to_dict())
    result = search_exit_history(doc)
    assert result['status'] == expected
    assert doc.to_dict() == before
    errors = validate_document(doc)['errors']
    assert bool(errors) is (special == 39)


def test_a_single_trip_can_collect_the_keys_before_leaving_the_destination():
    doc = one_shot_revisit_puzzle()
    blue = next(thing for thing in doc.things if thing.type == 5)
    blue.x, blue.y = 576, 64
    before = copy.deepcopy(doc.to_dict())
    result = search_exit_history(doc)
    assert result['status'] == 'success'
    assert sum(event['action'] == 'teleport-once' for event in result['trace']) == 1
    assert doc.to_dict() == before


def test_consuming_the_trigger_restores_ordinary_walking_after_the_return_trip():
    from generation.validator import validate_document
    from data.source_endings import assess_source_ending
    from roomgraph.segment import segment
    doc = consumption_walk_puzzle()
    assert search_exit_history(doc, consume_one_shot_teleports=False)['status'] == 'failure'
    result = search_exit_history(doc)
    assert result['status'] == 'success'
    assert [event['action'] for event in result['trace']] == ['teleport-once']
    assert not validate_document(doc)['errors']
    assert assess_source_ending(doc, segment(doc, playable_only=True))['complete']


def test_one_shot_event_states_respect_the_existing_search_budgets():
    doc = consumption_walk_puzzle()
    assert search_exit_history(doc, max_states=1)['status'] == 'inconclusive'
    assert search_exit_history(doc, max_geometries=1)['status'] == 'inconclusive'


def test_historical_source_endings_keep_their_prior_teleport_assumptions():
    from data.source_endings import assess_source_ending, SOURCE_ENDINGS_VERSION
    from roomgraph.segment import segment
    doc = one_shot_revisit_puzzle()
    graph = segment(doc, playable_only=True)
    old = assess_source_ending(doc, graph, version='physical_exit_v2')
    new = assess_source_ending(doc, graph)
    assert old['complete'] and not new['complete']
    assert old['history']['abstraction'] == 'location_keys_potential_doors_v1'
    assert new['version'] == SOURCE_ENDINGS_VERSION == 'physical_exit_v3'


def test_active_trigger_start_and_same_sector_planes_are_explicitly_unsupported():
    from tests.geometry_fixtures import polygon_document, square
    from wad_parser.types import Thing, ThingFlags
    doc = one_shot_revisit_puzzle()
    doc.things[0].x = 128
    assert search_exit_history(doc)['status'] == 'unsupported'
    doc = polygon_document([[square(0, 0, 128, 128), square(128, 0, 256, 128)],
                             [square(512, 0, 640, 128)]], start=(64, 64))
    line = next(line for line in doc.linedefs if line.left_sidedef >= 0)
    line.special, line.tag, doc.sectors[1].tag = 39, 7, 7
    doc.things.append(Thing(1, 576, 64, 0, 14, 'TeleportDest', 'teleport-dest', ThingFlags(skill_3=True)))
    assert search_exit_history(doc)['status'] == 'unsupported'


def test_monster_only_teleports_do_not_make_player_action_histories_unsupported():
    doc = one_shot_revisit_puzzle(97)
    line = next(line for line in doc.linedefs if line.left_sidedef < 0
                and doc.sidedefs[line.right_sidedef].sector == 1)
    line.special, line.tag = 125, 7
    assert search_exit_history(doc)['status'] == 'success'


def test_a_backside_crossing_is_not_treated_as_a_free_reusable_front_activation():
    doc = one_shot_revisit_puzzle()
    doc.things[0].x = 192
    assert search_exit_history(doc, consume_one_shot_teleports=False)['status'] == 'success'
    assert search_exit_history(doc)['status'] == 'failure'

"""Walkable within-room floor detail retains supported map traversal."""
from copy import deepcopy

import pytest

from generation.interiors import add_terraces
from generation.gameplay import apply_intro
from generation.validator import require_valid
from tests.gameplay_fixtures import terrace_arena
from wad_parser.geometry import sector_points, point_in_rings
from wad_parser.navigation import PlayerNavigation


@pytest.mark.parametrize('side,headroom,levels', [(512, 128, 2), (768, 128, 3),
                                                (1024, 128, 4), (1024, 80, 1)])
def test_terrace_has_reachable_nested_floors_and_preserves_authored_geometry(side, headroom, levels):
    doc = terrace_arena(side, headroom)
    before = deepcopy(doc)
    report = add_terraces(doc, seed=3, max_terraces=1, floor_flat='FLAT23')
    terrace, = report['terraces']
    assert len(terrace['steps']) == levels
    assert terrace['base_sector'] != 0
    assert [s.floor_height for s in doc.sectors[:3]] == [0, 0, 0]
    assert len(doc.sectors) == 3 + levels
    assert doc.vertices[:len(before.vertices)] == before.vertices
    assert doc.linedefs[:len(before.linedefs)] == before.linedefs
    assert doc.things == before.things
    assert [doc.sectors[s['sector']].floor_height for s in terrace['steps']] == list(range(16, 16 * levels + 1, 16))
    assert all(s.ceil_height - s.floor_height >= 64 for s in doc.sectors)
    top = terrace['steps'][-1]['sector']
    center = terrace['center']
    assert point_in_rings(*center, sector_points(doc, top))
    assert not point_in_rings(*center, sector_points(doc, terrace['base_sector']))
    navigation = PlayerNavigation(doc)
    assert navigation.reaches_point(*center, sid=top)
    assert not require_valid(doc, require_connected=True)['errors']
    assert all(s.floor_flat == 'FLAT23' for s in doc.sectors[3:])
    duplicate = add_terraces(before, seed=3, max_terraces=1, floor_flat='FLAT23')
    assert duplicate == report and before.to_dict() == doc.to_dict()


def test_terraces_skip_protected_rooms_and_reject_invalid_budgets_without_mutation():
    doc = terrace_arena()
    doc.sectors[1].special = 1
    doc.sectors[2].tag = 7
    before = doc.to_dict()
    with pytest.raises(ValueError, match='non-negative'):
        add_terraces(doc, max_terraces=-1)
    assert doc.to_dict() == before
    report = add_terraces(doc)
    assert report['terraces'] == []
    assert len(doc.sectors) == 3


def test_stepped_intro_preserves_safe_supplies_and_complete_population_budgets():
    doc = terrace_arena()
    report = apply_intro(doc, seed=4, monster_budget=4, interior_profile='stepped')
    assert len(report['interiors']['terraces']) == 2
    assert report['monsters'] == 4
    assert report['shells_supplied_before_combat'] >= report['shells_required_by_plan']
    assert report['health_supplied'] == 25 * report['health_stations_planned']
    assert all(t['sector'] == 0 for t in report['things'] if t['role'].startswith('opening_'))
    assert not require_valid(doc, require_connected=True)['errors']
    before = doc.to_dict()
    with pytest.raises(ValueError, match='interior_profile'):
        apply_intro(doc, interior_profile='unknown')
    assert doc.to_dict() == before


def test_cli_requires_intro_for_stepped_interiors_before_loading_models(tmp_path):
    from click.testing import CliRunner
    from cli.generate_roomgen import main
    checkpoint = tmp_path / 'stub.pt'
    checkpoint.touch()
    result = CliRunner().invoke(main, ['--checkpoint', str(checkpoint), '--interior-profile', 'stepped'])
    assert result.exit_code != 0 and 'require --gameplay intro' in result.output


def test_coordinate_fitting_updates_aliased_interior_reports_and_accumulates_translations():
    from wad_writer.assembler import fit_coordinates
    from wad_parser.types import Vertex
    from shapely import Polygon
    doc = terrace_arena()
    plan = apply_intro(doc, seed=4, monster_budget=4, interior_profile='stepped')
    doc.extra['decoration'] = {'gameplay': plan}
    original = deepcopy(plan['interiors'])
    assert original['columns'] and original['terraces']
    doc.vertices.append(Vertex(len(doc.vertices), 40000, 0))
    assert fit_coordinates(doc)
    dx, dy = doc.extra['coordinate_translation']
    assert dx != 0
    details = plan['interiors']
    assert details is doc.extra['gameplay']['interiors']
    assert details is doc.extra['decoration']['gameplay']['interiors']
    for terrace, before in zip(details['terraces'], original['terraces']):
        assert terrace['center'] == [before['center'][0] + dx, before['center'][1] + dy]
        assert list(terrace['footprint'][0]) == [before['footprint'][0][0] + dx, before['footprint'][0][1] + dy]
        top = terrace['steps'][-1]['sector']
        assert PlayerNavigation(doc).reaches_point(*terrace['center'], sid=top)
        assert Polygon(terrace['footprint']).contains(__import__('shapely').Point(*terrace['center']))
    for column, before in zip(details['columns'], original['columns']):
        assert column['center'] == [before['center'][0] + dx, before['center'][1] + dy]
    for terrace in doc.extra['terrace_details']['terraces']:
        assert PlayerNavigation(doc).reaches_point(*terrace['center'], sid=terrace['steps'][-1]['sector'])
    # A second fit must update total translation, not erase the first offset.
    doc.vertices.append(Vertex(len(doc.vertices), 0, 40000))
    assert fit_coordinates(doc)
    total_x, total_y = doc.extra['coordinate_translation']
    for terrace, before in zip(details['terraces'], original['terraces']):
        assert terrace['center'] == [before['center'][0] + total_x, before['center'][1] + total_y]
    assert total_y != 0 and total_x == dx
    assert not require_valid(doc, require_connected=True)['errors']


def test_roomgen_fits_coordinates_before_intro_validation(tmp_path, monkeypatch):
    from generation import roomgen
    from tests.pipeline_fixtures import stub_roomgen
    from generation.style import StyleModel
    from wad_parser.map_extractor import compute_derived
    stub_roomgen(monkeypatch)
    def shifted_document(*args, **kwargs):
        doc = terrace_arena()
        for element in doc.vertices + doc.things:
            element.x += 40000
        doc.bounds, doc.derived = compute_derived(doc.vertices, doc.sectors, doc.linedefs, doc.sidedefs, doc.things)
        return doc
    monkeypatch.setattr(roomgen, 'assemble', shifted_document)
    monkeypatch.setattr('generation.pipeline.build_nodes', lambda path: False)
    result = roomgen.generate('unused.pt', {}, seed=4, max_attempts=1, style=StyleModel(),
                              gameplay='intro', interior_profile='stepped', out_path=tmp_path / 'shifted.wad')
    assert result['maps'] == ['MAP01']
    assert any('translated XY' in repair for repair in result['summaries'][0]['repairs'])
    assert result['summaries'][0]['decoration']['gameplay']['interiors']['terraces']

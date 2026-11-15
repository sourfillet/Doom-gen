import copy

from data.spatial_correspondence import sector_correspondence
from tests.geometry_fixtures import two_rooms
from wad_parser.types import Sector, Sidedef


def test_reindexed_sectors_with_identical_properties_match_boundaries_without_mutation():
    source = two_rooms()
    rebuilt = copy.deepcopy(source)
    rebuilt.sectors = [rebuilt.sectors[1], rebuilt.sectors[0]]
    for sid, sector in enumerate(rebuilt.sectors):
        sector.id = sid
    for side in rebuilt.sidedefs:
        side.sector = 1 - side.sector
    for vertex in rebuilt.vertices:
        vertex.x, vertex.y = float(vertex.x), float(vertex.y)
    before = copy.deepcopy((source.to_dict(), rebuilt.to_dict()))
    result = sector_correspondence(source, rebuilt)
    assert result['status'] == 'complete'
    assert result['mapping'] == [{'source_sector': 0, 'target_sector': 1}, {'source_sector': 1, 'target_sector': 0}]
    assert (source.to_dict(), rebuilt.to_dict()) == before


def test_unused_tagged_control_is_reported_when_pruned_instead_of_shifting_comparisons():
    source = two_rooms()
    control = Sector(1, 16, 144, 'FLOOR4_8', 'CEIL3_5', 160, 8, 7)
    source.sectors[1].id = 2
    source.sectors.insert(1, control)
    for side in source.sidedefs:
        if side.sector == 1:
            side.sector = 2
    # Even an unused sidedef must not make the control a physical owner.
    source.sidedefs.append(Sidedef(len(source.sidedefs), 0, 0, '-', '-', '-', 1))
    rebuilt = two_rooms()
    result = sector_correspondence(source, rebuilt)
    assert result['status'] == 'complete'
    assert result['mapping'][-1] == {'source_sector': 2, 'target_sector': 1}
    assert result['unreferenced_source_sectors'][0]['attributes']['tag'] == 7
    assert result['unreferenced_source_sectors'][0]['attributes']['special'] == 8


def test_changed_action_or_texture_cannot_be_accepted_as_an_identical_anchor():
    source, rebuilt = two_rooms(), two_rooms()
    rebuilt.linedefs[0].special = 11
    result = sector_correspondence(source, rebuilt)
    assert result['status'] == 'incomplete' and result['unmatched_target_linedefs'] == [0]
    rebuilt = two_rooms()
    rebuilt.sidedefs[0].mid_tex = 'STONE2'
    result = sector_correspondence(source, rebuilt)
    assert result['status'] == 'incomplete' and result['unresolved_anchors']


def test_identical_ambiguous_lines_do_not_guess_by_id_centroid_or_sector_area():
    source = two_rooms()
    # Two same-geometry copies with different sector owners and no unique
    # boundary evidence. Array order alone cannot identify the original copy.
    line = next(line for line in source.linedefs if line.left_sidedef == -1)
    source.linedefs = [copy.deepcopy(line), copy.deepcopy(line)]
    source.linedefs[1].right_sidedef = len(source.sidedefs)
    source.sidedefs.append(copy.deepcopy(source.sidedefs[line.right_sidedef]))
    source.sidedefs[-1].id, source.sidedefs[-1].sector = len(source.sidedefs) - 1, 1
    for lid, element in enumerate(source.linedefs):
        element.id = lid
    result = sector_correspondence(source, source)
    assert result['status'] == 'incomplete'
    assert result['mapping'] == [] and result['unresolved_anchors']

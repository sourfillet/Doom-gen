"""Per-candidate publication, bounded retries, and exact compiled-lump merging."""
from copy import deepcopy
import json
import struct

import pytest

from generation.pipeline import MapBatch, NoUsableMapsError
from generation.progression import ensure_exit
from tests.gameplay_fixtures import playable
from tests.pipeline_fixtures import rejecting_builder, stub_roomgen
from wad_parser.wad_reader import WadFile
from wad_writer.assets import AssetBundle


def test_batch_keeps_successful_map_bytes_and_appends_assets_once(tmp_path, monkeypatch):
    calls = rejecting_builder(monkeypatch, {2})
    destination = tmp_path / 'out.wad'
    destination.write_bytes(b'previous output')
    assets = AssetBundle(flats={'NEWFLAT': bytes(4096)})
    with MapBatch(destination, assets=assets) as batch:
        batch.add(playable())
        with pytest.raises(ValueError, match='BLOCKMAP'):
            batch.add(playable('MAP02'))
        assert destination.read_bytes() == b'previous output'
        batch.add(playable('MAP02'))
        result = batch.publish()
    assert len(calls) == 3  # Final combine never reruns the builder.
    groups = WadFile(str(destination)).find_map_groups()
    assert [g.name for g in groups] == result['maps'] == ['MAP01', 'MAP02']
    for group, raw in zip(groups, (calls[0], calls[2])):
        path = tmp_path / 'before.wad'
        path.write_bytes(raw)
        assert group.lumps == WadFile(str(path)).find_map_groups()[0].lumps
    assert WadFile(str(destination)).lump_names().count('NEWFLAT') == 1
    assert len(result['map_publication']) == 2
    assert not list(tmp_path.glob('.map-batch-*'))


def test_final_validation_and_failed_empty_batch_preserve_previous_output(tmp_path, monkeypatch):
    import generation.pipeline as pipeline
    rejecting_builder(monkeypatch, set())
    destination = tmp_path / 'out.wad'
    destination.write_bytes(b'previous output')
    with MapBatch(destination) as batch:
        with pytest.raises(ValueError, match='no accepted maps'):
            batch.publish()
        batch.add(playable())
        original = pipeline.write_lumps
        def corrupt(lumps, path):
            return original([(n, b'' if n == 'THINGS' else b) for n, b in lumps], path)
        monkeypatch.setattr(pipeline, 'write_lumps', corrupt)
        with pytest.raises(ValueError):
            batch.publish()
    assert destination.read_bytes() == b'previous output'
    assert not list(tmp_path.glob('.map-batch-*'))


def test_roomgen_retries_compiled_failure_with_original_seed_budget(tmp_path, monkeypatch):
    roomgen, seen = stub_roomgen(monkeypatch)
    calls = rejecting_builder(monkeypatch, {2})
    result = roomgen.generate('unused.pt', {}, n_maps=2, seed=10, max_attempts=2,
                              out_path=tmp_path / 'out.wad')
    assert seen == [(10, 'MAP01'), (12, 'MAP02'), (13, 'MAP02')]
    assert len(calls) == 3
    assert [row['seed'] for row in result['summaries']] == [10, 13]
    failure, = result['failed_samples']
    assert (failure['sample'], failure['attempt'], failure['seed']) == (1, 1, 12)
    assert failure['failure_stage'] == 'publication' and failure['validated']
    assert result['maps'] == ['MAP01', 'MAP02']


def test_roomgen_keeps_other_maps_after_publication_budget_exhausted(tmp_path, monkeypatch):
    roomgen, seen = stub_roomgen(monkeypatch)
    rejecting_builder(monkeypatch, {2, 3})
    result = roomgen.generate('unused.pt', {}, n_maps=3, seed=0, max_attempts=1,
                              out_path=tmp_path / 'out.wad')
    assert seen == [(0, 'MAP01'), (1, 'MAP02'), (2, 'MAP02')]
    assert result['maps'] == ['MAP01'] and result['requested_maps'] == 3
    assert [r['seed'] for r in result['failed_samples']] == [1, 2]


def test_all_failed_roomgen_retains_per_attempt_evidence_and_old_output(tmp_path, monkeypatch):
    roomgen, _ = stub_roomgen(monkeypatch)
    rejecting_builder(monkeypatch, {1, 2})
    destination = tmp_path / 'out.wad'
    destination.write_bytes(b'previous output')
    with pytest.raises(NoUsableMapsError) as failure:
        roomgen.generate('unused.pt', {}, seed=7, max_attempts=2, out_path=destination)
    assert failure.value.report['maps'] == []
    assert [r['seed'] for r in failure.value.report['failed_samples']] == [7, 8]
    assert destination.read_bytes() == b'previous output'


def test_cli_writes_failure_report_even_when_all_samples_fail(tmp_path, monkeypatch):
    from click.testing import CliRunner
    import cli.generate_roomgen as cli
    checkpoint = tmp_path / 'stub.pt'
    checkpoint.touch()
    failure = NoUsableMapsError([{'seed': 7, 'reason': 'bad BLOCKMAP',
                                 'failure_stage': 'publication', 'validated': True}], 1)
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(cli, 'generate', fail)
    report = tmp_path / 'report.json'
    result = CliRunner().invoke(cli.main, ['--checkpoint', str(checkpoint), '--report', str(report)])
    assert result.exit_code != 0 and 'bad BLOCKMAP' in result.output
    assert json.loads(report.read_text()) == failure.report


def test_raster_generation_discards_only_failed_compiled_candidate(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from generation import pipeline
    import models.checkpoint
    class Conditioner:
        def encode(self, tags, **kwargs):
            return torch.zeros(len(tags), 256)
    class Diffusion:
        def ddim_sample(self, shape, *args, **kwargs):
            return torch.zeros(shape)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(models.checkpoint, 'load_checkpoint', lambda *args:
                        (None, Diffusion(), Conditioner(), {'in_channels': 9, 'resolution': 8}))
    names = []
    def document(*args, map_name, **kwargs):
        names.append(map_name)
        return playable(map_name), ['test repair'], {'test': 'decoration'}
    monkeypatch.setattr(pipeline, 'raster_to_document', document)
    calls = rejecting_builder(monkeypatch, {2})
    result = pipeline.generate('unused.pt', {}, n_maps=3, seed=7, out_path=tmp_path / 'out.wad')
    assert len(calls) == 3 and names == ['MAP01', 'MAP02', 'MAP02']
    assert result['maps'] == ['MAP01', 'MAP02'] and result['requested_maps'] == 3
    assert result['failed_samples'][0]['seed'] == 7
    assert result['failed_samples'][0]['style_seed'] == 8
    assert result['failed_samples'][0]['failure_stage'] == 'publication'
    assert result['repairs'] == {'MAP01': ['test repair'], 'MAP02': ['test repair']}
    assert result['decoration']['MAP02'] == {'test': 'decoration'}


def test_publication_combines_existing_compiled_maps_without_rebuilding(tmp_path, monkeypatch):
    from generation.pipeline import documents_to_wad, map_wads_to_wad
    calls = rejecting_builder(monkeypatch, set())
    paths = []
    for name in ['MAP01', 'MAP02']:
        path = tmp_path / f'{name}.wad'
        documents_to_wad([playable(name)], path, require_playable=True)
        paths.append(path)
    raw = [WadFile(str(path)).find_map_groups()[0].lumps for path in paths]
    result = map_wads_to_wad(paths, tmp_path / 'panel.wad')
    assert len(calls) == 2
    assert result['maps'] == ['MAP01', 'MAP02']
    assert [g.lumps for g in WadFile(result['path']).find_map_groups()] == raw
    destination = tmp_path / 'previous.wad'
    destination.write_bytes(b'previous output')
    with pytest.raises(ValueError, match='duplicate map marker'):
        map_wads_to_wad([paths[0], paths[0]], destination)
    assert destination.read_bytes() == b'previous output'
    assets = AssetBundle(flats={'NEWFLAT': bytes(4096)})
    documents_to_wad([playable('MAP03')], tmp_path / 'assets.wad', assets=assets)
    with pytest.raises(ValueError, match='global assets'):
        map_wads_to_wad([tmp_path / 'assets.wad'], destination)
    assert destination.read_bytes() == b'previous output'


def test_orphan_duplicate_map_marker_from_builder_cannot_shadow_valid_geometry(tmp_path, monkeypatch):
    from generation.pipeline import documents_to_wad
    from wad_writer.wad_writer import write_lumps
    def builder(path):
        wad = WadFile(str(path))
        # The extra MAP01 is not recognized as a map group, but classic engine
        # global lookup still selects it instead of the actual level marker.
        write_lumps([(l.name, l.data) for l in wad.lumps] + [('MAP01', b'not a map')], path)
        return False
    monkeypatch.setattr('generation.pipeline.build_nodes', builder)
    out = tmp_path / 'previous.wad'
    out.write_bytes(b'previous output')
    with pytest.raises(ValueError, match='orphan marker'):
        documents_to_wad([playable()], out, require_playable=True)
    assert out.read_bytes() == b'previous output'


def test_compiled_physical_validation_rejects_collapsed_bsp_and_preserves_destination(tmp_path, monkeypatch):
    from generation.pipeline import documents_to_wad
    from tests.wad_builder import build_two_sector_wad_with_nodes
    from wad_parser.map_extractor import extract_map
    from generation.progression import ensure_exit
    from wad_writer.wad_writer import write_lumps
    from wad_writer.assembler import MAP_LUMP_ORDER

    (tmp_path / 'seed.wad').write_bytes(build_two_sector_wad_with_nodes())
    source_wad = WadFile(str(tmp_path / 'seed.wad'))
    ref_group, = source_wad.find_map_groups()
    ref_group.lumps['NODES'] = struct.pack('<hhhhhhhhhhhhHH', 128, 0, 0, 128, 128, 0, 128, 256, 128, 0, 0, 128, 0x8001, 0x8000)
    doc = extract_map(source_wad, ref_group)
    ensure_exit(doc)

    def builder(path):
        current = WadFile(str(path))
        out_group, = current.find_map_groups()
        for name in ['SEGS', 'SSECTORS', 'NODES']:
            out_group.lumps[name] = ref_group.lumps[name]
        # Reverse root children in NODES: subsectors collapse to zero area
        bad = bytearray(out_group.lumps['NODES'])
        struct.pack_into('<HH', bad, 24, 0x8000, 0x8001)
        out_group.lumps['NODES'] = bytes(bad)
        write_lumps([(out_group.name, b''),
                     *[(name, out_group.lumps[name]) for name in MAP_LUMP_ORDER if name in out_group.lumps]], path)
        return True

    monkeypatch.setattr('generation.pipeline.build_nodes', builder)
    out = tmp_path / 'destination.wad'
    out.write_bytes(b'original destination')
    with pytest.raises(ValueError, match='compiled BSP has no physical sector regions'):
        documents_to_wad([doc], out, require_playable=True)
    assert out.read_bytes() == b'original destination'


def test_compiled_physical_validation_rejects_player_outside_compiled_space(tmp_path, monkeypatch):
    from generation.pipeline import documents_to_wad
    from tests.wad_builder import build_two_sector_wad_with_nodes
    from wad_parser.map_extractor import extract_map
    from generation.progression import ensure_exit
    from wad_writer.wad_writer import write_lumps
    from wad_writer.assembler import MAP_LUMP_ORDER

    (tmp_path / 'seed.wad').write_bytes(build_two_sector_wad_with_nodes())
    source_wad = WadFile(str(tmp_path / 'seed.wad'))
    ref_group, = source_wad.find_map_groups()
    ref_group.lumps['NODES'] = struct.pack('<hhhhhhhhhhhhHH', 128, 0, 0, 128, 128, 0, 128, 256, 128, 0, 0, 128, 0x8001, 0x8000)
    doc = extract_map(source_wad, ref_group)
    ensure_exit(doc)
    start = next(t for t in doc.things if t.type == 1)
    start.x = 100
    start.y = 64

    def builder(path):
        current = WadFile(str(path))
        out_group, = current.find_map_groups()
        for name in ['SEGS', 'SSECTORS', 'NODES']:
            out_group.lumps[name] = ref_group.lumps[name]
        # Partition at x=80 instead of x=128; leaf 0 is clipped to x <= 80, excluding player at (100, 64)
        bad = bytearray(out_group.lumps['NODES'])
        struct.pack_into('<h', bad, 0, 80)
        out_group.lumps['NODES'] = bytes(bad)
        write_lumps([(out_group.name, b''),
                     *[(name, out_group.lumps[name]) for name in MAP_LUMP_ORDER if name in out_group.lumps]], path)
        return True

    monkeypatch.setattr('generation.pipeline.build_nodes', builder)
    out = tmp_path / 'destination.wad'
    out.write_bytes(b'original destination')
    with pytest.raises(ValueError, match='player start is outside compiled physical space'):
        documents_to_wad([doc], out, require_playable=True)
    assert out.read_bytes() == b'original destination'


@pytest.mark.parametrize('return_route', [False, True])
def test_compiled_physical_validation_requires_a_single_key_history(return_route):
    from generation.pipeline import _check_compiled_output
    from tests.progression_fixtures import branch_document
    from wad_parser.geometry import sector_points
    doc = branch_document(return_from_key=return_route)
    # Model the fresh decoder's physical regions without recomputing them in
    # the publication check. The optimistic union reaches every region in both
    # controls, but the irreversible branch cannot bring its key to the door.
    doc.derived.sector_regions = [sector_points(doc, s.id) for s in doc.sectors]
    doc.derived.bsp_geometry = {'sectors_with_geometry': 5, 'area': 70000,
                                'subsectors': 5, 'collapsed_subsectors': 0}
    before = deepcopy(doc.to_dict())
    if return_route:
        _check_compiled_output(doc, require_playable=True)
    else:
        with pytest.raises(ValueError, match='compiled door/key progression'):
            _check_compiled_output(doc, require_playable=True)
    assert doc.to_dict() == before


@pytest.mark.parametrize('batch_mode', [False, True])
def test_failed_companion_publication_preserves_both_destinations(tmp_path, monkeypatch, batch_mode):
    from generation.pipeline import documents_to_wad
    rejecting_builder(monkeypatch, {1})
    out, companion = tmp_path / 'maps.wad', tmp_path / 'resources.wad'
    out.write_bytes(b'previous map')
    companion.write_bytes(b'previous resources')
    assets = AssetBundle(flats={'NEWFLAT': bytes(4096)})
    with pytest.raises(ValueError, match='BLOCKMAP'):
        if batch_mode:
            with MapBatch(out, assets=assets) as batch:
                batch.add(playable())
                batch.publish(companion_textures=companion)
        else:
            documents_to_wad([playable()], out, assets=assets, require_playable=True,
                              companion_textures=companion)
    assert out.read_bytes() == b'previous map'
    assert companion.read_bytes() == b'previous resources'


def test_companion_validates_final_batch_before_replacing_resources(tmp_path, monkeypatch):
    import generation.pipeline as pipeline
    rejecting_builder(monkeypatch, set())
    out, companion = tmp_path / 'maps.wad', tmp_path / 'resources.wad'
    out.write_bytes(b'previous map')
    companion.write_bytes(b'previous resources')
    with MapBatch(out, assets=AssetBundle(flats={'NEWFLAT': bytes(4096)})) as batch:
        batch.add(playable())
        original = pipeline.write_lumps
        def corrupt_map(lumps, path):
            return original([(name, b'' if name == 'THINGS' else data) for name, data in lumps], path)
        monkeypatch.setattr(pipeline, 'write_lumps', corrupt_map)
        with pytest.raises(ValueError):
            batch.publish(companion_textures=companion)
    assert out.read_bytes() == b'previous map'
    assert companion.read_bytes() == b'previous resources'


def test_companion_publication_restores_resources_on_map_replace_error(tmp_path, monkeypatch):
    from pathlib import Path
    from generation.pipeline import documents_to_wad
    out, companion = tmp_path / 'maps.wad', tmp_path / 'resources.wad'
    out.write_bytes(b'previous map')
    companion.write_bytes(b'previous resources')
    replace = Path.replace
    def fail_map_replace(source, destination):
        if source.name == 'map.wad' and Path(destination) == out:
            raise OSError('simulated map replacement failure')
        return replace(source, destination)
    monkeypatch.setattr(Path, 'replace', fail_map_replace)
    with pytest.raises(OSError, match='simulated'):
        documents_to_wad([playable()], out, try_nodes=False,
            assets=AssetBundle(flats={'NEWFLAT': bytes(4096)}), companion_textures=companion)
    assert out.read_bytes() == b'previous map'
    assert companion.read_bytes() == b'previous resources'


def test_companion_publication_keeps_map_and_resources_separate(tmp_path):
    from generation.pipeline import documents_to_wad
    out, companion = tmp_path / 'maps.wad', tmp_path / 'resources.wad'
    result = documents_to_wad([playable()], out, try_nodes=False, require_playable=True,
        assets=AssetBundle(flats={'NEWFLAT': bytes(4096)}), companion_textures=companion)
    assert result['path'] == str(out)
    assert result['companion_textures'] == str(companion)
    assert WadFile(str(companion)).find_flats()['NEWFLAT'] == bytes(4096)
    assert 'NEWFLAT' not in WadFile(str(out)).lump_names()
    assert not list(tmp_path.glob('.*-output-*'))
    before = out.read_bytes()
    with pytest.raises(ValueError, match='different paths'):
        documents_to_wad([playable()], out, companion_textures=out)
    assert out.read_bytes() == before


def test_empty_companion_does_not_report_an_unwritten_file(tmp_path):
    from generation.pipeline import documents_to_wad
    companion = tmp_path / 'resources.wad'
    result = documents_to_wad([playable()], tmp_path / 'map.wad', try_nodes=False,
                               companion_textures=companion)
    assert 'companion_textures' not in result
    assert not companion.exists()


@pytest.mark.parametrize('actor_y, accepted', [(108, False), (32, True)])
def test_compiled_physical_validation_checks_monster_spawn_regions(tmp_path, monkeypatch, actor_y, accepted):
    from generation.pipeline import documents_to_wad
    from generation.validator import require_valid
    from tests.wad_builder import build_two_sector_wad_with_nodes
    from wad_parser.map_extractor import extract_map
    from wad_parser.types import Thing, ThingFlags, resolve_doomed
    from wad_writer.wad_writer import write_lumps
    seed = tmp_path / 'seed.wad'
    seed.write_bytes(build_two_sector_wad_with_nodes())
    wad = WadFile(str(seed))
    group, = wad.find_map_groups()
    # Clip the west room to y <= 96 without removing the route to the east
    # room. The extra north leaf collapses; both sectors still have geometry.
    group.lumps['NODES'] = (
        struct.pack('<hhhhhhhhhhhhHH', 0, 96, 128, 0, 96, 0, 0, 128, 128, 96, 0, 128, 0x8000, 0x8002)
        + struct.pack('<hhhhhhhhhhhhHH', 128, 0, 0, 128, 128, 0, 128, 256, 128, 0, 0, 128, 0x8001, 0))
    group.lumps['SSECTORS'] += struct.pack('<HH', 4, 4)
    doc = extract_map(wad, group)
    ensure_exit(doc)
    doc.things[0].x, doc.things[0].y = 96, 64
    name, category, _ = resolve_doomed(3001)
    doc.things.append(Thing(len(doc.things), 32, actor_y, 0, 3001, name, category,
        ThingFlags(skill_1_2=True, skill_3=True, skill_4_5=True)))
    require_valid(doc, require_connected=True)  # Both controls have valid raw spawns.
    def builder(path):
        current = WadFile(str(path))
        write_lumps([(lump.name, group.lumps[lump.name]
                      if lump.name in {'NODES', 'SSECTORS', 'SEGS'} else lump.data)
                     for lump in current.lumps], path)
        return True
    monkeypatch.setattr('generation.pipeline.build_nodes', builder)
    out = tmp_path / 'destination.wad'
    out.write_bytes(b'previous map')
    if accepted:
        assert documents_to_wad([doc], out, require_playable=True)['nodes_built']
    else:
        with pytest.raises(ValueError, match='thing .*compiled.*outside playable space'):
            documents_to_wad([doc], out, require_playable=True)
        assert out.read_bytes() == b'previous map'

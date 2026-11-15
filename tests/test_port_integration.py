"""Opt-in classic-node and Chocolate Doom checks (DOOM_GEN_PORT_TESTS=1)."""
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.skipif(os.environ.get('DOOM_GEN_PORT_TESTS') != '1',
                                reason='opt-in external node-builder/Chocolate Doom tests')


@pytest.fixture(scope='module')
def compiled_panel(tmp_path_factory):
    if shutil.which('zdbsp') is None:
        pytest.skip('zdbsp is not available')
    from generation.pipeline import MapBatch
    from generation.progression import ensure_exit
    from roomgraph.assemble import assemble
    from roomgraph.types import Room, RoomGraph, Connection
    directory = tmp_path_factory.mktemp('classic-nodes')
    docs = []
    for index, (kind, rise, side) in enumerate([('open', 128, 512), ('door', 384, 512),
                                               ('teleport', 0, 192)], 1):
        rooms = [Room(i, [], rise * i, rise * i + 128, side**2,
                      (0, 0), (0, 0, side, side)) for i in range(2)]
        doc = assemble(RoomGraph(rooms, [Connection(0, 1, kind, 128)]), map_name=f'MAP{index:02d}')
        ensure_exit(doc)
        docs.append(doc)
    path = directory / 'compiled.wad'
    with MapBatch(path, target_port='vanilla') as batch:
        for doc in docs:
            batch.add(doc)
        result = batch.publish()
    assert result['nodes_built']
    assert result['compatibility']['target'] == 'vanilla'
    assert len(result['map_publication']) == 3
    return path


def test_real_node_builder_preserves_valid_classic_geometry(compiled_panel):
    from generation.validator import validate_document
    from wad_parser.map_extractor import extract_all_maps
    from wad_parser.wad_reader import WadFile
    wad = WadFile(str(compiled_panel))
    for group in wad.find_map_groups():
        for name in ('SEGS', 'SSECTORS', 'NODES', 'BLOCKMAP'):
            assert group.lumps[name], (group.name, name)
        count = len(group.lumps['SECTORS']) // 26
        assert group.lumps['REJECT'] == bytes((count * count + 7) // 8)
    docs = extract_all_maps(wad)
    assert len(docs) == 3
    for doc in docs:
        assert doc.derived.bsp_geometry['method'] == 'classic-bsp-halfplanes-v1'
        assert not validate_document(doc, require_connected=True)['errors']


@pytest.mark.parametrize('options,signature', [(('-X',), 'XNOD'), (('-z',), 'ZNOD'),
                                             (('-x', '-X'), 'XGLN'), (('-x', '-z'), 'ZGLN')])
def test_real_node_builder_extended_geometry(compiled_panel, tmp_path, options, signature):
    from wad_parser.map_extractor import extract_all_maps
    from wad_parser.wad_reader import WadFile
    from wad_parser.geometry import sector_area_perimeter
    output = tmp_path / 'extended.wad'
    subprocess.run(['zdbsp', str(compiled_panel), *options, '-R', '-o', str(output)], check=True,
                   capture_output=True, timeout=30)
    before = extract_all_maps(WadFile(str(compiled_panel)))
    after = extract_all_maps(WadFile(str(output)))
    assert len(before) == len(after) == 3
    for original, rebuilt in zip(before, after):
        assert rebuilt.derived.bsp_geometry['node_format'] == signature
        assert sum(sector_area_perimeter(rebuilt, s.id)[0] for s in rebuilt.sectors) == pytest.approx(
            sum(sector_area_perimeter(original, s.id)[0] for s in original.sectors), abs=1e-3)


@pytest.mark.parametrize('options,version', [(('-x',), 2), (('-x', '-5'), 5)])
def test_real_node_builder_standalone_gl_geometry(compiled_panel, tmp_path, options, version):
    from wad_parser.map_extractor import extract_all_maps
    from wad_parser.wad_reader import WadFile
    from wad_parser.geometry import sector_area_perimeter
    output = tmp_path / 'standalone-gl.wad'
    subprocess.run(['zdbsp', str(compiled_panel), *options, '-R', '-o', str(output)], check=True,
                   capture_output=True, timeout=30)
    before = extract_all_maps(WadFile(str(compiled_panel)))
    wad = WadFile(str(output))
    groups = wad.find_map_groups()
    after = extract_all_maps(wad)
    assert len(before) == len(after) == 3
    for original, rebuilt, group in zip(before, after, groups):
        # The isolated 2018 Ubuntu builder writes 20-byte in-memory records
        # for -5 instead of glBSP's packed 16-byte records, plus excess node
        # bytes. Diagnose it rather than inventing an alternate file format.
        if version == 5:
            import struct
            count = sum(size for size, _ in struct.iter_unpack('<II', group.lumps['GL_SSECT']))
            if len(group.lumps['GL_SEGS']) != count * 16:
                assert 'expected 16-byte records' in rebuilt.derived.bsp_geometry['failure']
                continue
        assert rebuilt.derived.bsp_geometry['node_format'] == f'GLv{version}'
        assert sum(sector_area_perimeter(rebuilt, s.id)[0] for s in rebuilt.sectors) == pytest.approx(
            sum(sector_area_perimeter(original, s.id)[0] for s in original.sectors), abs=1e-3)


def test_chocolate_doom_renders_node_built_maps(compiled_panel):
    executable = shutil.which('chocolate-doom')
    if executable is None:
        pytest.skip('Chocolate Doom is not available')
    iwad = os.environ.get('DOOM_GEN_IWAD')
    if not iwad:
        engine = pytest.importorskip('vizdoom')
        iwad = str(Path(engine.__file__).parent / 'freedoom2.wad')
    directory = compiled_panel.parent
    env = dict(os.environ, SDL_VIDEODRIVER='dummy', SDL_AUDIODRIVER='dummy',
               XDG_DATA_HOME=str(directory / 'data'))
    for map_number in range(1, 4):
        demo = directory / f'map{map_number}.lmp'
        # Vanilla 1.9 header, skill 3, no monsters, one player; three seconds
        # of ordinary forward movement followed by the demo-end marker.
        demo.write_bytes(bytes([109, 2, 1, map_number, 0, 0, 0, 1, 0, 1, 0, 0, 0])
                         + bytes([10, 0, 0, 0]) * 105 + b'\x80')
        result = subprocess.run([executable, '-iwad', iwad, '-file', str(compiled_panel),
                                 '-timedemo', str(demo), '-nosound', '-nogui',
                                 '-config', str(directory / 'default.cfg'),
                                 '-extraconfig', str(directory / 'chocolate.cfg'),
                                 '-savedir', str(directory)], env=env,
                                capture_output=True, text=True, timeout=30)
        output = result.stdout + result.stderr
        # Chocolate Doom reports successful timedemo statistics through its
        # historical error-exit path (255), so require the completed tic count.
        assert result.returncode in (0, 255) and re.search(r'timed 105 gametics in \d+ realtics', output), output
        assert 'REJECT lump too short' not in output


def test_bundled_assets_preserve_stock_resources_and_render_in_chocolate(tmp_path):
    engine = pytest.importorskip('vizdoom')
    executable = shutil.which('chocolate-doom')
    if executable is None or shutil.which('zdbsp') is None:
        pytest.skip('Chocolate Doom and zdbsp are required')
    from tests.pipeline_fixtures import asset_fixture
    from generation.pipeline import documents_to_wad
    iwad = Path(engine.__file__).parent / 'freedoom2.wad'
    doc, assets = asset_fixture(iwad)
    panel = tmp_path / 'assets.wad'
    documents_to_wad([doc], panel, assets=assets, target_port='vanilla', require_playable=True)
    demo = tmp_path / 'assets.lmp'
    demo.write_bytes(bytes([109, 2, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0])
                     + bytes([10, 0, 0, 0]) * 105 + b'\x80')
    env = dict(os.environ, SDL_VIDEODRIVER='dummy', SDL_AUDIODRIVER='dummy',
               XDG_DATA_HOME=str(tmp_path / 'data'))
    result = subprocess.run([executable, '-iwad', str(iwad), '-file', str(panel), '-timedemo', str(demo),
                             '-nosound', '-nogui', '-config', str(tmp_path / 'default.cfg'),
                             '-extraconfig', str(tmp_path / 'chocolate.cfg'), '-savedir', str(tmp_path)],
                            env=env, capture_output=True, text=True, timeout=30)
    output = result.stdout + result.stderr
    assert result.returncode in (0, 255) and re.search(r'timed 105 gametics in \d+ realtics', output), output

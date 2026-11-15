import struct

import pytest

from wad_parser.wad_reader import MapLumpGroup
from wad_writer.compatibility import vanilla_issues, blockmap_issues, require_target_port


def compiled_group():
    # One block with a dummy zero, line zero, and a terminator.
    return MapLumpGroup('MAP01', 'doom', {
        'VERTEXES': bytes(8), 'LINEDEFS': bytes(14), 'SIDEDEFS': bytes(30),
        'SECTORS': bytes(26), 'SEGS': bytes(12), 'SSECTORS': struct.pack('<HH', 1, 0), 'NODES': b'',
        'REJECT': bytes(1), 'BLOCKMAP': struct.pack('<8H', 0, 0, 1, 1, 5, 0, 0, 65535)})


def test_vanilla_blockmap_unsigned_offset_cannot_be_used_as_signed_short():
    group = compiled_group()
    group.lumps['BLOCKMAP'] = struct.pack('<5H', 0, 0, 1, 1, 32768) + bytes(65530)
    assert any('signed 16-bit' in issue for issue in vanilla_issues(group))


@pytest.mark.parametrize('words,problem', [
    ([0, 0, 1, 1, 4, 0, 65535], 'outside list'),
    ([0, 0, 1, 1, 5, 0], 'terminator'),
    ([0, 0, 1, 1, 5, 0, 1, 65535], 'linedef index'),
    ([0, 0, 3, 3, 5], 'dimensions'),
])
def test_vanilla_blockmap_rejects_invalid_lists(words, problem):
    group = compiled_group()
    group.lumps['BLOCKMAP'] = struct.pack(f'<{len(words)}H', *words)
    assert any(problem in issue for issue in vanilla_issues(group))


def test_vanilla_allows_classic_single_leaf_and_shared_block_lists():
    group = compiled_group()
    assert not vanilla_issues(group)
    group.lumps['BLOCKMAP'] = struct.pack('<9H', 0, 0, 2, 1, 6, 6, 0, 0, 65535)
    assert not vanilla_issues(group)


@pytest.mark.parametrize('name,data', [
    ('NODES', struct.pack('<24xHH', 0, 32768)),
    ('SSECTORS', struct.pack('<HH', 2, 0)),
    ('SEGS', struct.pack('<6H', 2, 0, 0, 0, 0, 0)),
])
def test_vanilla_rejects_corrupt_compiled_references(name, data):
    group = compiled_group()
    group.lumps[name] = data
    assert vanilla_issues(group)


def test_vanilla_requires_nodes_without_replacing_previous_output(tmp_path, monkeypatch):
    from generation.pipeline import documents_to_wad
    from tests.geometry_fixtures import two_rooms
    import generation.pipeline as pipeline
    monkeypatch.setattr(pipeline, 'build_nodes', lambda path: False)
    destination = tmp_path / 'previous.wad'
    destination.write_bytes(b'previous output')
    with pytest.raises(ValueError, match='vanilla compatibility'):
        documents_to_wad([two_rooms()], destination, target_port='vanilla')
    assert destination.read_bytes() == b'previous output'


def test_cli_forwards_target_port(tmp_path, monkeypatch):
    from click.testing import CliRunner
    import cli.generate_roomgen as cli
    checkpoint = tmp_path / 'stub.pt'
    checkpoint.touch()
    seen = {}
    def generate(*args, **kwargs):
        seen.update(kwargs)
        return {'path': 'out.wad', 'maps': ['MAP01'], 'summaries': [], 'nodes_built': True}
    monkeypatch.setattr(cli, 'generate', generate)
    result = CliRunner().invoke(cli.main, ['--checkpoint', str(checkpoint), '--target-port', 'vanilla'])
    assert result.exit_code == 0, result.output
    assert seen['target_port'] == 'vanilla'


def test_modern_ports_accept_valid_unsigned_offsets_and_missing_tables():
    group = compiled_group()
    words = [0, 0, 1, 1, 32768] + [0] * (32768 - 5) + [0, 65535]
    group.lumps['BLOCKMAP'] = struct.pack(f'<{len(words)}H', *words)
    assert not blockmap_issues(group)
    assert any('signed 16-bit' in issue for issue in vanilla_issues(group))
    group.lumps['BLOCKMAP'] = b''
    assert not blockmap_issues(group)


def test_modern_ports_reject_unsigned_offset_wrap_into_the_table():
    from types import SimpleNamespace
    group = compiled_group()
    width, height = 229, 286
    words = [0, 0, width, height] + [4] * (width * height) + [0, 0, 65535]
    group.lumps['BLOCKMAP'] = struct.pack(f'<{len(words)}H', *words)
    with pytest.raises(ValueError, match='zdoom compatibility.*outside list data'):
        require_target_port(SimpleNamespace(find_map_groups=lambda: [group]), 'zdoom')


def test_invalid_compiled_blockmap_preserves_previous_modern_output(tmp_path, monkeypatch):
    from generation.pipeline import documents_to_wad
    from tests.geometry_fixtures import two_rooms
    from tests.wad_builder import _assemble_wad
    from wad_parser.wad_reader import WadFile
    def invalid_builder(path):
        wad = WadFile(str(path))
        bad = struct.pack('<7H', 0, 0, 1, 1, 4, 0, 65535)
        path.write_bytes(_assemble_wad([(l.name, bad if l.name == 'BLOCKMAP' else l.data) for l in wad.lumps]))
        return True
    monkeypatch.setattr('generation.pipeline.build_nodes', invalid_builder)
    destination = tmp_path / 'previous.wad'
    destination.write_bytes(b'previous output')
    with pytest.raises(ValueError, match='zdoom compatibility'):
        documents_to_wad([two_rooms()], destination)
    assert destination.read_bytes() == b'previous output'

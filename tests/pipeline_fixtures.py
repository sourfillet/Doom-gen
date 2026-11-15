"""Pipeline stubs, embedded assets and source-WAD trees for publication tests."""
from copy import deepcopy
import struct

from generation.progression import ensure_exit
from tests.gameplay_fixtures import playable
from tests.geometry_fixtures import two_rooms
from tests.wad_builder import _assemble_wad, build_two_sector_wad
from wad_parser.texture_parser import parse_pnames, parse_texture_lump
from wad_parser.wad_reader import WadFile
from wad_writer.assets import AssetBundle, EmbeddedTexture
from wad_writer.wad_writer import write_wad


def stub_roomgen(monkeypatch):
    from generation import roomgen
    from roomgraph.types import Room, RoomGraph
    graph = RoomGraph([Room(0, [], 0, 128, 256**2, (0, 0), (0, 0, 256, 256))], [])
    seen = []
    def sample(*args, **kwargs):
        seen.append((kwargs['seed'], kwargs['map_name']))
        result = deepcopy(graph)
        result.map_name = kwargs['map_name']
        return result
    monkeypatch.setattr(roomgen, 'load_roomgen_model', lambda *args: (None, None))
    monkeypatch.setattr(roomgen, 'sample_room_graph', sample)
    return roomgen, seen


def rejecting_builder(monkeypatch, rejected):
    calls = []
    def build(path):
        calls.append(path.read_bytes())
        if len(calls) in rejected:
            wad = WadFile(str(path))
            bad = struct.pack('<7H', 0, 0, 1, 1, 4, 0, 65535)
            path.write_bytes(_assemble_wad([(l.name, bad if l.name == 'BLOCKMAP' else l.data)
                                           for l in wad.lumps]))
            return True
        return False
    monkeypatch.setattr('generation.pipeline.build_nodes', build)
    return calls


def asset_fixture(iwad):
    """Mix stock walls/ceiling with a new floor and a composed wall texture."""
    base = WadFile(str(iwad))
    pnames = parse_pnames(base.get_all_lumps('PNAMES')[-1])
    textures = {}
    for field in ['TEXTURE1', 'TEXTURE2']:
        for table in base.get_all_lumps(field)[-1:]:
            textures.update(parse_texture_lump(table, pnames))
    source = textures['STARTAN2']
    bundle = AssetBundle(flats={'TESTFLAT': bytes([96]) * 4096},
                         textures=[EmbeddedTexture('TESTWALL', source.width, source.height,
                             [(p.name, p.x_origin, p.y_origin) for p in source.patches])]).with_base_wad(iwad)
    doc = playable()
    doc.sidedefs[0].mid_tex = 'TESTWALL'
    doc.sectors[0].floor_flat = 'TESTFLAT'
    return doc, bundle


def source_wads(tmp_path):
    root = tmp_path / 'wads'
    root.mkdir()
    doc = two_rooms()
    ensure_exit(doc)
    write_wad([doc], root / 'complete.WAD')
    (root / 'fragment.wad').write_bytes(build_two_sector_wad())
    (root / 'resources.WaD').write_bytes(_assemble_wad([('PALETTE', b'abc')]))
    return root

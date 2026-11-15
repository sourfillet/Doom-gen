"""Resource tables must preserve target assets and avoid global name shadows."""
import struct

import pytest

from generation.pipeline import documents_to_wad
from tests.gameplay_fixtures import playable
from wad_writer.assets import AssetBundle, EmbeddedTexture
from wad_writer.wad_writer import write_wad
from wad_parser.texture_parser import parse_pnames, parse_texture_lump


@pytest.mark.parametrize('name', ['MAP01', 'map99', 'E1M1', 'PNAMES', 'TEXTURE1',
                                  'F_START', 'FF_END', 'S_END', 'F1_START', 'GL_MAP01', 'GL_NODES'])
def test_reserved_asset_lumps_are_rejected_without_replacing_previous_output(tmp_path, name):
    out = tmp_path / 'previous.wad'
    out.write_bytes(b'previous output')
    with pytest.raises(ValueError, match='reserved'):
        write_wad([playable()], out, assets=AssetBundle(flats={name: bytes(4096)}))
    assert out.read_bytes() == b'previous output'


@pytest.mark.parametrize('texture,problem', [
    (EmbeddedTexture('TOO_LONG_NAME', 64, 64, [('PATCH', 0, 0)]), '8 bytes'),
    (EmbeddedTexture('CAFÉ', 64, 64, [('PATCH', 0, 0)]), 'ASCII'),
    (EmbeddedTexture('TEX', 0, 64, [('PATCH', 0, 0)]), 'range'),
    (EmbeddedTexture('TEX', 64, 32768, [('PATCH', 0, 0)]), 'range'),
    (EmbeddedTexture('TEX', 64, 64, [('PATCH', 32768, 0)]), 'range'),
    (EmbeddedTexture('TEX', 64, 64, [('PATCH', 1.5, 0)]), 'integer'),
    (EmbeddedTexture('TEX', 64, 64, []), 'patches'),
])
def test_internal_texture_fields_fail_with_actionable_errors(texture, problem):
    with pytest.raises(ValueError, match=problem):
        AssetBundle(textures=[texture]).build_lumps()


def test_encoded_name_collisions_are_rejected_but_texture_and_patch_can_share_a_name():
    with pytest.raises(ValueError, match='duplicate'):
        AssetBundle(flats={'TEST': bytes(4096)}, patches={'test': bytes(32)}).build_lumps()
    with pytest.raises(ValueError, match='duplicate'):
        AssetBundle(textures=[EmbeddedTexture('TEX', 64, 64, [('PATCH', 0, 0)]),
                              EmbeddedTexture('tex', 64, 64, [('PATCH', 0, 0)])]).build_lumps()
    assert AssetBundle(patches={'TEX': bytes(32)},
                       textures=[EmbeddedTexture('TEX', 64, 64, [('TEX', 0, 0)])]).build_lumps()


def test_patch_only_bundle_preserves_base_tables_and_flat_range_markers():
    patches = dict(AssetBundle(patches={'PATCH': bytes(32)}).build_lumps())
    assert 'PNAMES' not in patches and 'TEXTURE1' not in patches
    names = [name for name, _ in AssetBundle(flats={'NEWFLAT': bytes(4096)}).build_lumps()]
    assert names == ['FF_START', 'NEWFLAT', 'FF_END', 'F_END']


def test_base_texture_prefix_and_secondary_definitions_survive_custom_overrides(tmp_path):
    # Build a minimal two-table resource fixture, including a duplicate PNAMES
    # slot. The original record references that slot and must remain unchanged.
    from wad_writer.wad_writer import write_lumps
    def texture(name, patch):
        body = name.encode().ljust(8, b'\0') + struct.pack('<ihhih', 0, 64, 64, 0, 1)
        body += struct.pack('<hhhhh', 0, 0, patch, 1, 0)
        return struct.pack('<ii', 1, 8) + body
    base = tmp_path / 'base.wad'
    pnames = struct.pack('<i', 3) + b'BASEPAT\0BASEPAT\0SECONDP\0'
    write_lumps([('PNAMES', pnames), ('TEXTURE1', texture('BASEWALL', 1)),
                 ('TEXTURE2', texture('SECOND', 2))], base)
    bundle = AssetBundle(patches={'NEWPATCH': bytes(32)},
                         textures=[EmbeddedTexture('NEWALL', 64, 64, [('NEWPATCH', 0, 0)])]).with_base_wad(base)
    original = AssetBundle(textures=[EmbeddedTexture('NEWALL', 64, 64, [('NEWPATCH', 0, 0)])])
    assert not original.base_tables
    lumps = dict(bundle.build_lumps())
    names = parse_pnames(lumps['PNAMES'])
    assert names == ['BASEPAT', 'BASEPAT', 'SECONDP', 'NEWPATCH']
    textures = parse_texture_lump(lumps['TEXTURE1'], names)
    assert list(textures) == ['BASEWALL', 'SECOND', 'NEWALL']
    assert textures['BASEWALL'].patches[0].name == 'BASEPAT'
    assert textures['SECOND'].patches[0].name == 'SECONDP'
    assert textures['NEWALL'].patches[0].name == 'NEWPATCH'
    assert lumps['TEXTURE2'] == struct.pack('<i', 0)
    # Multiple overrides must keep a usable patch-name mapping throughout.
    bundle.textures += [EmbeddedTexture('BASEWALL', 64, 64, [('NEWPATCH', 0, 0)]),
                        EmbeddedTexture('SECOND', 64, 64, [('NEWPATCH', 0, 0)])]
    lumps = dict(bundle.build_lumps())
    textures = parse_texture_lump(lumps['TEXTURE1'], parse_pnames(lumps['PNAMES']))
    assert all(textures[n].patches[0].name == 'NEWPATCH' for n in ['BASEWALL', 'SECOND'])


def test_vanilla_requires_base_tables_before_publishing_custom_textures(tmp_path):
    bundle = AssetBundle(textures=[EmbeddedTexture('NEWALL', 64, 64, [('PATCH', 0, 0)])])
    out = tmp_path / 'previous.wad'
    out.write_bytes(b'previous output')
    with pytest.raises(ValueError, match='base texture tables'):
        documents_to_wad([playable()], out, assets=bundle, target_port='vanilla')
    assert out.read_bytes() == b'previous output'


def test_vanilla_rejects_wrong_flat_size_and_cross_namespace_replacements():
    tables = {'PNAMES': struct.pack('<i', 1) + b'BASEPAT\0', 'TEXTURE1': struct.pack('<i', 0)}
    with pytest.raises(ValueError, match='4096'):
        AssetBundle(flats={'NEWFLAT': bytes(16)}, base_tables=tables).require_target('vanilla')
    with pytest.raises(ValueError, match='new names'):
        AssetBundle(flats={'BASEPAT': bytes(4096)}, base_tables=tables).require_target('vanilla')
    with pytest.raises(ValueError, match='base flats'):
        AssetBundle(patches={'BASEFLAT': bytes(32)}, base_tables=tables,
                    base_flat_names={'BASEFLAT'}).require_target('vanilla')


def test_directory_loader_does_not_silently_overwrite_case_collisions(tmp_path):
    directory = tmp_path / 'flats'
    directory.mkdir()
    (directory / 'TEST.lump').write_bytes(bytes(4096))
    (directory / 'test.lump').write_bytes(bytes([1]) * 4096)
    if len(list(directory.iterdir())) != 2:
        pytest.skip('case-insensitive filesystem')
    with pytest.raises(ValueError, match='duplicate encoded flat'):
        AssetBundle.from_dir(tmp_path)


def test_unicode_case_expansion_cannot_turn_into_an_ascii_asset_name(tmp_path):
    # Uppercasing the non-ASCII character would produce ASCII SS; validate
    # original input bytes first so names are never silently transformed.
    with pytest.raises(ValueError, match='ASCII'):
        AssetBundle(textures=[EmbeddedTexture('ß', 64, 64, [('PATCH', 0, 0)])]).build_lumps()
    directory = tmp_path / 'flats'
    directory.mkdir()
    (directory / 'ß.lump').write_bytes(bytes(4096))
    with pytest.raises(ValueError, match='ASCII'):
        AssetBundle.from_dir(tmp_path)

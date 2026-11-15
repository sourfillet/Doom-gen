"""Tests for automated missing-texture and flat extraction across source WADs."""
from pathlib import Path
import struct

import pytest

from generation.asset_extractor import AssetExtractor
from generation.game_profile import GameProfile
from wad_parser.wad_reader import WadFile
from wad_writer.assets import AssetBundle, EmbeddedTexture
from wad_writer.wad_writer import write_lumps


def _make_flat_lump() -> bytes:
    # Standard 64x64 8-bit flat lump: exactly 4096 bytes
    return bytes(i % 256 for i in range(4096))


def _make_patch_lump() -> bytes:
    # Minimal Doom picture format patch (header + column pointers + columns)
    # width=8, height=8, left=0, top=0
    header = struct.pack("<hhhh", 8, 8, 0, 0)
    column = bytes([0, 8, 0]) + bytes([10] * 8) + bytes([0, 0xFF])
    col_offsets = struct.pack("<8i", *(40 + i * len(column) for i in range(8)))
    return header + col_offsets + column * 8


def _build_synthetic_wad(path: Path, *, pnames: list[str], textures: list[tuple[str, int, int, list[tuple[str, int, int]]]],
                         patches: dict[str, bytes], flats: dict[str, bytes]) -> Path:
    bundle = AssetBundle(
        flats=flats,
        patches=patches,
        textures=[EmbeddedTexture(name, w, h, p_list) for name, w, h, p_list in textures]
    )
    if pnames:
        # Construct explicit PNAMES and TEXTURE1
        p_data = struct.pack("<i", len(pnames)) + b"".join(name.encode("ascii").ljust(8, b"\x00")[:8] for name in pnames)
        p_index = {n: i for i, n in enumerate(pnames)}
        records = []
        for name, w, h, p_list in textures:
            body = (name.encode("ascii").ljust(8, b"\x00")[:8]
                    + struct.pack("<ihhih", 0, w, h, 0, len(p_list)))
            for pname, ox, oy in p_list:
                body += struct.pack("<hhhhh", ox, oy, p_index[pname], 1, 0)
            records.append(body)
        offset = 4 + len(records) * 4
        offsets = []
        for rec in records:
            offsets.append(offset)
            offset += len(rec)
        t_data = (struct.pack("<i", len(records))
                  + b"".join(struct.pack("<i", off) for off in offsets)
                  + b"".join(records))

        lumps = [("PNAMES", p_data), ("TEXTURE1", t_data)]
        if patches:
            lumps.append(("P_START", b""))
            lumps.extend(patches.items())
            lumps.append(("P_END", b""))
        if flats:
            lumps.append(("F_START", b""))
            lumps.extend(flats.items())
            lumps.append(("F_END", b""))
        return write_lumps(lumps, path)

    return write_lumps(bundle.build_lumps(), path)


def test_asset_extractor_synthetic(tmp_path):
    base_wad_path = tmp_path / "base.wad"
    _build_synthetic_wad(
        base_wad_path,
        pnames=["BASEP1"],
        textures=[("BASETEX", 64, 64, [("BASEP1", 0, 0)])],
        patches={"BASEP1": _make_patch_lump()},
        flats={"BASEFLAT": _make_flat_lump()}
    )

    src_wad_path = tmp_path / "source.wad"
    _build_synthetic_wad(
        src_wad_path,
        pnames=["BASEP1", "SRCP1"],
        textures=[("SRCTEX1", 128, 64, [("BASEP1", 0, 0), ("SRCP1", 64, 0)])],
        patches={"SRCP1": _make_patch_lump()},
        flats={"SRCFLAT": _make_flat_lump()}
    )

    extractor = AssetExtractor([src_wad_path])
    assert "SRCTEX1" in extractor.available_textures
    assert "SRCFLAT" in extractor.available_flats

    # Create map document referencing SRCTEX1 and SRCFLAT
    from tests.geometry_fixtures import two_rooms
    doc = two_rooms()
    for sd in doc.sidedefs:
        if sd.mid_tex not in ("-", ""):
            sd.mid_tex = "BASETEX"
    for s in doc.sectors:
        s.floor_flat = "BASEFLAT"
        s.ceil_flat = "BASEFLAT"
    doc.sidedefs[1].mid_tex = "SRCTEX1"
    doc.sectors[0].floor_flat = "SRCFLAT"

    game = GameProfile.from_wad(base_wad_path)
    assert "BASETEX" in game.textures
    assert "SRCTEX1" not in game.textures
    assert "SRCFLAT" not in game.flats

    # Resolve and register
    bundle = extractor.resolve_and_register(doc, game)
    assert not bundle.is_empty
    assert "SRCTEX1" in bundle.texture_names()
    assert "SRCFLAT" in bundle.flat_names()

    # Patches: SRCP1 was extracted, but BASEP1 was NOT duplicated because it already exists in base WAD!
    assert "SRCP1" in bundle.patches
    assert "BASEP1" not in bundle.patches

    # After registration, GameProfile should recognize SRCTEX1 and SRCFLAT
    assert "SRCTEX1" in game.textures
    assert "SRCFLAT" in game.flats

    # Enforcement must keep SRCTEX1 and SRCFLAT intact
    summary = game.enforce(doc)
    assert doc.sidedefs[1].mid_tex == "SRCTEX1"
    assert doc.sectors[0].floor_flat == "SRCFLAT"

    # Companion WAD output
    companion_path = tmp_path / "companion.wad"
    extractor.write_companion_wad(bundle, companion_path)
    assert companion_path.exists()
    comp_wad = WadFile(str(companion_path))
    lump_names = [l.name for l in comp_wad.lumps]
    assert "TEXTURE1" in lump_names
    assert "PNAMES" in lump_names
    assert "SRCP1" in lump_names
    assert "SRCFLAT" in comp_wad.find_flats()

    # Embedded WAD output
    from generation.pipeline import documents_to_wad
    embedded_path = tmp_path / "embedded.wad"
    documents_to_wad([doc], embedded_path, assets=bundle)
    assert embedded_path.exists()
    emb_wad = WadFile(str(embedded_path))
    emb_lump_names = [l.name for l in emb_wad.lumps]
    assert "MAP01" in emb_lump_names
    assert "TEXTURE1" in emb_lump_names
    assert "PNAMES" in emb_lump_names
    assert "SRCP1" in emb_lump_names
    assert "SRCFLAT" in emb_wad.find_flats()


def test_asset_extractor_missing_raises(tmp_path):
    extractor = AssetExtractor()
    with pytest.raises(KeyError, match="Texture 'MISSING' not found"):
        extractor.extract(textures=["MISSING"])
    with pytest.raises(KeyError, match="Flat 'MISSING' not found"):
        extractor.extract(flats=["MISSING"])


def test_changed_source_patch_is_preserved_without_replacing_base_patch(tmp_path):
    base, source = tmp_path / 'base.wad', tmp_path / 'source.wad'
    original = _make_patch_lump()
    changed = original[:43] + bytes([11]) + original[44:]  # First column's first pixel.
    _build_synthetic_wad(base, pnames=['SHARED'],
                         textures=[('BASETEX', 64, 64, [('SHARED', 0, 0)])],
                         patches={'SHARED': original}, flats={})
    _build_synthetic_wad(source, pnames=['SHARED'],
                         textures=[('SRCTEX', 64, 64, [('SHARED', 0, 0)])],
                         patches={'SHARED': changed}, flats={})
    bundle = AssetExtractor([source]).extract(textures=['SRCTEX'], target_iwad_path=base)
    patch_name, _, _ = bundle.textures[0].patches[0]
    assert patch_name != 'SHARED'
    assert bundle.patches[patch_name] == changed
    assert bundle.patches[patch_name][43] == 11 and original[43] == 10
    assert 'SHARED' not in bundle.patches
    bundle.validate()
    tables = dict(bundle.build_lumps())
    from wad_parser.texture_parser import parse_pnames, parse_texture_lump
    textures = parse_texture_lump(tables['TEXTURE1'], parse_pnames(tables['PNAMES']))
    assert textures['BASETEX'].patches[0].name == 'SHARED'
    assert textures['SRCTEX'].patches[0].name == patch_name


def test_missing_patch_does_not_register_an_incomplete_texture(tmp_path):
    source = tmp_path / 'source.wad'
    _build_synthetic_wad(source, pnames=['ABSENT'],
                         textures=[('SRCTEX', 64, 64, [('ABSENT', 0, 0)])], patches={}, flats={})
    extractor = AssetExtractor([source])
    bundle = extractor.extract(textures=['SRCTEX'], strict=False)
    assert bundle.is_empty


@pytest.mark.parametrize('companion', [False, True])
def test_regeneration_resolves_source_assets_before_enforcement(tmp_path, monkeypatch, companion):
    import json
    from copy import deepcopy
    from generation.pipeline import regenerate_from_json
    from tests.geometry_fixtures import two_rooms
    from wad_parser.map_extractor import extract_all_maps
    base, source = tmp_path / 'base.wad', tmp_path / 'source.wad'
    _build_synthetic_wad(base, pnames=['BASEP'],
                         textures=[('BASETEX', 64, 64, [('BASEP', 0, 0)])],
                         patches={'BASEP': _make_patch_lump()}, flats={'BASEFLAT': _make_flat_lump()})
    _build_synthetic_wad(source, pnames=['SRCP'],
                         textures=[('SRCTEX', 64, 64, [('SRCP', 0, 0)])],
                         patches={'SRCP': _make_patch_lump()}, flats={'SRCFLAT': _make_flat_lump()})
    doc = two_rooms()
    for side in doc.sidedefs:
        for attr in ('upper_tex', 'lower_tex', 'mid_tex'):
            if getattr(side, attr) != '-':
                setattr(side, attr, 'SRCTEX')
    for sector in doc.sectors:
        sector.floor_flat = sector.ceil_flat = 'SRCFLAT'
    path = tmp_path / 'source.json'
    path.write_text(json.dumps(doc.to_dict()))
    monkeypatch.setattr('generation.pipeline.build_nodes', lambda path: False)
    game = GameProfile.from_wad(base)
    before = deepcopy(game)
    out = tmp_path / 'out.wad'
    resource_path = tmp_path / 'resources.wad'
    result = regenerate_from_json(path, out, resolution=32, game=game, sources=[source],
                                 companion_textures=resource_path if companion else None)
    wad = WadFile(str(out))
    restored, = extract_all_maps(wad)
    assert any(side.mid_tex == 'SRCTEX' for side in restored.sidedefs)
    assert all(sector.floor_flat == 'SRCFLAT' for sector in restored.sectors)
    resources = WadFile(str(resource_path)) if companion else wad
    assert resources.get_lump('SRCP') == _make_patch_lump()
    assert (wad.get_lump('SRCP') is None) == companion
    assert result['asset_resolution']['sources'][0]['sha256'] == WadFile(str(source)).hash
    assert {asset['name'] for asset in result['asset_resolution']['resolved']} == {'SRCTEX', 'SRCFLAT'}
    assert game == before


def test_source_patch_variants_are_stable_across_map_resolution(tmp_path):
    from copy import deepcopy
    from generation.asset_extractor import AssetResolution
    from tests.geometry_fixtures import two_rooms
    base = tmp_path / 'base.wad'
    _build_synthetic_wad(base, pnames=['BASEP'],
        textures=[('BASETEX', 64, 64, [('BASEP', 0, 0)])],
        patches={'BASEP': _make_patch_lump()}, flats={'BASEFL': _make_flat_lump()})
    sources = []
    for name, pixel in [('SRCONE', 10), ('SRCTWO', 20)]:
        source = tmp_path / (name + '.wad')
        _build_synthetic_wad(source, pnames=['SHARED'],
            textures=[(name, 64, 64, [('SHARED', 0, 0)])],
            patches={'SHARED': bytes([pixel]) * 64}, flats={})
        sources.append(source)
    game = GameProfile.from_wad(base)
    assets = AssetBundle(flats={'MANUAL': bytes(4096)}).with_base_wad(base)
    before_game, before_assets = deepcopy(game), deepcopy(assets)
    context = AssetResolution(game, assets, sources)
    for name in ['SRCONE', 'SRCTWO', 'SRCONE']:
        doc = two_rooms()
        doc.sidedefs[0].mid_tex = name
        context.resolve(doc)
    textures = {texture.name: texture for texture in context.bundle.textures}
    first, second = textures['SRCONE'].patches[0][0], textures['SRCTWO'].patches[0][0]
    assert first != second
    assert context.bundle.patches[first] == bytes([10]) * 64
    assert context.bundle.patches[second] == bytes([20]) * 64
    assert game == before_game and assets == before_assets
    context.bundle.validate()


def test_vocabulary_only_profile_cannot_bind_source_texture_tables(tmp_path):
    from generation.asset_extractor import AssetResolution
    from tests.geometry_fixtures import two_rooms
    source = tmp_path / 'source.wad'
    _build_synthetic_wad(source, pnames=['SRC'],
        textures=[('SRCTEX', 64, 64, [('SRC', 0, 0)])], patches={'SRC': _make_patch_lump()}, flats={})
    context = AssetResolution(GameProfile('profile'), None, [source])
    doc = two_rooms()
    doc.sidedefs[0].mid_tex = 'SRCTEX'
    with pytest.raises(ValueError, match='target IWAD'):
        context.resolve(doc)
    assert 'SRCTEX' not in context.game.textures
    assert context.bundle.is_empty


@pytest.mark.parametrize('bad_table', ['pnames_count', 'texture_directory', 'patch_index', 'patch_count'])
def test_source_index_rejects_malformed_tables_without_partial_registration(tmp_path, bad_table):
    good = tmp_path / 'good.wad'
    _build_synthetic_wad(good, pnames=['PATCH'],
        textures=[('GOODTEX', 8, 8, [('PATCH', 0, 0)])],
        patches={'PATCH': _make_patch_lump()}, flats={})
    valid = WadFile(str(good))
    pnames, texture = valid.get_lump('PNAMES'), valid.get_lump('TEXTURE1')
    if bad_table == 'pnames_count':
        pnames = struct.pack('<i', -1)
    elif bad_table == 'texture_directory':
        texture = struct.pack('<i', 2)  # Missing both directory offsets.
    else:
        texture = bytearray(texture)
        offset, = struct.unpack_from('<i', texture, 4)
        struct.pack_into('<h', texture, offset + (26 if bad_table == 'patch_index' else 20),
                         99 if bad_table == 'patch_index' else -1)
        texture = bytes(texture)
    bad = tmp_path / 'bad.wad'
    write_lumps([('PNAMES', pnames), ('TEXTURE1', texture),
                 ('F_START', b''), ('BADFLAT', bytes(4096)), ('F_END', b'')], bad)
    extractor = AssetExtractor([good])
    before = list(extractor.sources), extractor.available_textures, extractor.available_flats
    with pytest.raises(ValueError, match='source texture tables'):
        extractor.add_source(bad)
    assert (extractor.sources, extractor.available_textures, extractor.available_flats) == before


def test_source_index_accepts_valid_texture2_only_source(tmp_path):
    seed = tmp_path / 'seed.wad'
    _build_synthetic_wad(seed, pnames=['PATCH'],
        textures=[('SRCTEX', 8, 8, [('PATCH', 0, 0)])],
        patches={'PATCH': _make_patch_lump()}, flats={})
    source = tmp_path / 'texture2.wad'
    write_lumps([('TEXTURE2' if lump.name == 'TEXTURE1' else lump.name, lump.data)
                 for lump in WadFile(str(seed)).lumps], source)
    extractor = AssetExtractor([source])
    bundle = extractor.extract(textures=['SRCTEX'])
    assert bundle.texture_names() == {'SRCTEX'}
    assert bundle.patches['PATCH'] == _make_patch_lump()


def test_asset_extractor_real_wads():
    plut_path = Path("wads/PLUTONIA.WAD")
    d2_path = Path("wads/DOOM2.WAD")
    if not (plut_path.exists() and d2_path.exists()):
        pytest.skip("Local IWADs not available")

    extractor = AssetExtractor([plut_path])
    assert "A-BROWN1" in extractor.available_textures

    game = GameProfile.from_wad(d2_path)
    assert "A-BROWN1" not in game.textures

    bundle = extractor.extract(textures=["A-BROWN1"], target_iwad_path=d2_path)
    assert "A-BROWN1" in bundle.texture_names()
    assert "BROWN1" in bundle.patches  # Plutonia-specific patch extracted
    bundle.validate()

    lumps = dict(bundle.build_lumps())
    assert "TEXTURE1" in lumps
    assert "PNAMES" in lumps
    assert "BROWN1" in lumps


"""Game-vocabulary constraint: enforce() must leave a map referencing only
assets the target game can render, and asset bundles must round-trip into a
PWAD and be accepted by enforcement."""
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wad_builder import build_two_sector_wad

from generation.game_profile import GameProfile
from wad_parser.map_extractor import extract_all_maps
from wad_parser.texture_parser import parse_pnames, parse_texture_lump
from wad_parser.types import resolve_doomed
from wad_parser.wad_reader import WadFile
from wad_writer.assets import AssetBundle, EmbeddedTexture
from wad_writer.wad_writer import write_wad

_IWAD = Path(__file__).resolve().parent.parent / "wads" / "DOOM.WAD"


def _doc(tmp_path):
    p = tmp_path / "src.wad"
    p.write_bytes(build_two_sector_wad())
    return extract_all_maps(WadFile(str(p)))[0]


def _hand_profile() -> GameProfile:
    prof = GameProfile(name="T")
    prof.textures = {"STARTAN2", "BROWN1"}
    prof.flats = {"FLOOR4_8", "CEIL3_5"}
    prof.thing_types = {3004, 1}
    prof.texture_use = Counter({"STARTAN2": 5, "BROWN1": 2})
    prof.flat_use = Counter({"FLOOR4_8": 3, "CEIL3_5": 3})
    prof.thing_use = Counter({3004: 10, 1: 1})
    return prof


# ---------------------------------------------------------------------------
# enforcement
# ---------------------------------------------------------------------------

def test_enforce_remaps_everything_invalid(tmp_path):
    prof = _hand_profile()
    doc = _doc(tmp_path)
    doc.sectors[0].floor_flat = "NOFLAT77"
    doc.sidedefs[0].mid_tex = "NOTATEX"
    # an Imp (3001) is a monster not in the profile -> nearest monster (3004)
    from wad_parser.types import Thing, ThingFlags
    doc.things.append(Thing(id=len(doc.things), x=64, y=64, angle=0, type=3001,
                            type_name="Imp", category="monster",
                            flags=ThingFlags(skill_1_2=True, skill_3=True, skill_4_5=True)))

    summary = prof.enforce(doc)

    assert summary["flats_remapped"] >= 1
    assert summary["textures_remapped"] >= 1
    assert summary["things_remapped"] >= 1
    # nothing invalid remains
    for s in doc.sectors:
        assert s.floor_flat in prof.flats and s.ceil_flat in prof.flats
    for sd in doc.sidedefs:
        for tex in (sd.upper_tex, sd.lower_tex, sd.mid_tex):
            assert tex == "-" or tex in prof.textures
    for t in doc.things:
        assert t.type in prof.thing_types or t.type in {1, 2, 3, 4, 11, 14}
    # the Imp became the profile's only monster, the Zombieman
    assert all(resolve_doomed(t.type)[1] != "monster" or t.type == 3004
               for t in doc.things)


def test_enforce_drops_thing_with_no_category_substitute(tmp_path):
    prof = _hand_profile()   # has a monster and a player start, no "key"
    doc = _doc(tmp_path)
    from wad_parser.types import Thing, ThingFlags
    doc.things = [Thing(id=0, x=64, y=64, angle=0, type=5, type_name="Blue keycard",
                        category="key", flags=ThingFlags())]
    summary = prof.enforce(doc)
    assert summary["things_dropped"] == 1
    assert doc.things == []


def test_enforce_preserves_essential_things(tmp_path):
    prof = _hand_profile()
    doc = _doc(tmp_path)
    types_before = {t.type for t in doc.things if t.type in {1, 2, 3, 4}}
    prof.enforce(doc)
    assert {t.type for t in doc.things if t.type in {1, 2, 3, 4}} == types_before


def test_profile_json_round_trip():
    prof = _hand_profile()
    restored = GameProfile.from_json(prof.to_json())
    assert restored.textures == prof.textures
    assert restored.flats == prof.flats
    assert restored.thing_types == prof.thing_types
    assert restored.thing_use == prof.thing_use


# ---------------------------------------------------------------------------
# asset embedding
# ---------------------------------------------------------------------------

def _bundle() -> AssetBundle:
    return AssetBundle(
        flats={"MYFLAT": bytes(4096)},                 # one 64x64 flat
        patches={"MYPATCH": b"\x00" * 32},              # opaque bytes; names are what matter
        textures=[EmbeddedTexture("MYWALL", 64, 128, [("MYPATCH", 0, 0)])],
    )


def test_asset_bundle_round_trips_through_pwad(tmp_path):
    doc = _doc(tmp_path)
    out = write_wad([doc], tmp_path / "withassets.wad", assets=_bundle())
    wad = WadFile(str(out))

    # flat present between F_START/F_END
    assert "MYFLAT" in wad.find_flats()
    # PNAMES + TEXTURE1 parse back to the composed texture
    pnames = parse_pnames(wad.get_lump("PNAMES"))
    assert "MYPATCH" in pnames
    textures = parse_texture_lump(wad.get_lump("TEXTURE1"), pnames)
    assert "MYWALL" in textures
    tex = textures["MYWALL"]
    assert (tex.width, tex.height) == (64, 128)
    assert [p.name for p in tex.patches] == ["MYPATCH"]
    # patch lump itself is present
    assert wad.get_lump("MYPATCH") == b"\x00" * 32


def test_from_wad_reads_embedded_assets(tmp_path):
    doc = _doc(tmp_path)
    out = write_wad([doc], tmp_path / "withassets.wad", assets=_bundle())
    prof = GameProfile.from_wad(out, name="EMB")
    assert "MYWALL" in prof.textures
    assert "MYFLAT" in prof.flats


def test_registered_assets_survive_enforcement(tmp_path):
    prof = _hand_profile()
    bundle = _bundle()
    prof.extend_assets(bundle.texture_names(), bundle.flat_names())
    doc = _doc(tmp_path)
    doc.sidedefs[0].mid_tex = "MYWALL"
    doc.sectors[0].floor_flat = "MYFLAT"
    prof.enforce(doc)
    assert doc.sidedefs[0].mid_tex == "MYWALL"   # kept, not remapped
    assert doc.sectors[0].floor_flat == "MYFLAT"


# ---------------------------------------------------------------------------
# real IWAD (skipped when the gitignored WAD isn't present)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _IWAD.exists(), reason="DOOM.WAD not present")
def test_from_real_iwad_has_expected_vocabulary():
    prof = GameProfile.from_wad(_IWAD, name="DOOM")
    assert "STARTAN3" in prof.textures
    assert "FLOOR4_8" in prof.flats
    assert 3004 in prof.thing_types          # Zombieman
    assert 66 not in prof.thing_types        # Revenant is Doom II only

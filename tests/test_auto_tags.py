"""Tests for auto-derived fact tags: rule correctness, WAD-level hoisting,
store precedence (user > auto), vetoes surviving regeneration, and
rename/merge leaving the auto layer alone."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wad_builder import build_hexen_wad, build_two_sector_wad, build_udmf_wad

from tagging.auto import derive_map_facts, derive_wad_facts
from tagging.store import TagStore
from wad_parser.map_extractor import extract_all_maps
from wad_parser.types import ThingFlags, Thing
from wad_parser.wad_reader import WadFile

WAD = "c" * 64


def _doc(tmp_path, builder=build_two_sector_wad):
    p = tmp_path / "t.wad"
    p.write_bytes(builder())
    wad = WadFile(str(p))
    return wad, extract_all_maps(wad)[0]


def _mk_thing(type_num: int, thing_id: int = 99) -> Thing:
    return Thing(id=thing_id, x=64, y=64, angle=0, type=type_num, type_name="x",
                 category="special", flags=ThingFlags())


# ---------------------------------------------------------------------------
# derivation rules
# ---------------------------------------------------------------------------

def test_basic_facts(tmp_path):
    _, doc = _doc(tmp_path)
    tags = derive_map_facts(doc)
    assert tags.get("single-player") == 1.0   # builder places a Player1Start
    assert "coop" not in tags
    assert "deathmatch" not in tags
    assert "secrets" not in tags
    assert tags.get("vanilla") == 1.0


def test_coop_and_deathmatch_thresholds(tmp_path):
    _, doc = _doc(tmp_path)
    doc.things.append(_mk_thing(2))
    doc.things += [_mk_thing(11, i) for i in range(3)]  # only 3 DM starts
    tags = derive_map_facts(doc)
    assert tags.get("coop") == 1.0
    assert "deathmatch" not in tags  # vanilla requires four
    doc.things.append(_mk_thing(11, 50))
    assert derive_map_facts(doc).get("deathmatch") == 1.0


def test_secrets_vanilla_and_boom_bit(tmp_path):
    _, doc = _doc(tmp_path)
    doc.sectors[0].special = 9
    assert derive_map_facts(doc).get("secrets") == 1.0
    doc.sectors[0].special = 0x0080  # Boom generalized secret bit
    assert derive_map_facts(doc).get("secrets") == 1.0


def test_boss_keys_teleporters(tmp_path):
    _, doc = _doc(tmp_path)
    doc.things.append(_mk_thing(16))  # Cyberdemon
    doc.things.append(_mk_thing(5))
    doc.things[-1].category = "key"
    doc.derived.sector_graph["edges"].append(
        {"a": 0, "b": 1, "linedef": 0, "kind": "teleport"})
    tags = derive_map_facts(doc)
    assert tags.get("boss") == 1.0
    assert tags.get("keys") == 1.0
    assert tags.get("teleporters") == 1.0


def test_feature_tiers(tmp_path):
    _, doc = _doc(tmp_path)
    assert derive_map_facts(doc).get("vanilla") == 1.0

    doc.linedefs[0].special = 243  # Boom line teleport
    tags = derive_map_facts(doc)
    assert tags.get("boom") == 1.0 and "vanilla" not in tags

    doc.linedefs[1].special = 271  # MBF sky transfer outranks Boom
    assert derive_map_facts(doc).get("mbf") == 1.0

    doc.linedefs[2].special = 1024  # MBF21 scroller outranks all
    assert derive_map_facts(doc).get("mbf21") == 1.0


def test_junk_flag_bits_do_not_upgrade_tier(tmp_path):
    # The IWAD's E2M7 has special-0 linedefs with garbage high flag bits;
    # port-extension flags must never count as feature-tier evidence.
    _, doc = _doc(tmp_path)
    doc.linedefs[0].flags.block_land_monsters = True
    doc.linedefs[0].flags.block_players = True
    doc.linedefs[1].flags.pass_through = True
    doc.things[0].flags.friendly = True
    assert derive_map_facts(doc).get("vanilla") == 1.0


def test_format_tags(tmp_path):
    _, hexen_doc = _doc(tmp_path, build_hexen_wad)
    assert derive_map_facts(hexen_doc).get("hexen-format") == 1.0
    _, udmf_doc = _doc(tmp_path, build_udmf_wad)
    tags = derive_map_facts(udmf_doc)
    assert tags.get("udmf") == 1.0
    assert tags.get("slopes") == 1.0  # builder's TEXTMAP has a floorplane + zfloor


def test_wad_level_hoisting(tmp_path):
    wad, doc = _doc(tmp_path)
    wad_tags, per_map = derive_wad_facts(wad, [doc])
    # single-map WAD: every map fact is shared, so all hoist to WAD level
    assert wad_tags.get("single-player") == 1.0
    assert per_map["MAP01"] == {}
    assert "iwad" not in wad_tags  # builder emits a PWAD
    assert "megawad" not in wad_tags


# ---------------------------------------------------------------------------
# store semantics
# ---------------------------------------------------------------------------

def test_user_tag_outranks_auto(tmp_path):
    store = TagStore(tmp_path / "tags")
    store.write_auto(WAD, {"deathmatch": 1.0}, {"MAP01": {"secrets": 1.0}})
    store.tag_map(WAD, "MAP01", "secrets", weight=0.5)  # user overrides weight
    detailed = store.effective_tags_detailed(WAD, "MAP01")
    assert detailed["secrets"] == (0.5, "user")
    assert detailed["deathmatch"] == (1.0, "auto")


def test_veto_survives_regeneration(tmp_path):
    store = TagStore(tmp_path / "tags")
    store.write_auto(WAD, {}, {"MAP01": {"boss": 1.0}})
    assert "boss" in store.effective_tags(WAD, "MAP01")
    store.deny_auto(WAD, "boss", "MAP01")
    assert "boss" not in store.effective_tags(WAD, "MAP01")
    store.write_auto(WAD, {}, {"MAP01": {"boss": 1.0}})  # regenerate
    assert "boss" not in store.effective_tags(WAD, "MAP01")


def test_regeneration_replaces_wholesale(tmp_path):
    store = TagStore(tmp_path / "tags")
    store.write_auto(WAD, {"iwad": 1.0}, {"MAP01": {"boss": 1.0}, "MAP02": {"keys": 1.0}})
    store.write_auto(WAD, {}, {"MAP01": {"secrets": 1.0}})
    assert store.wad_auto_tags(WAD) == {}
    assert store.map_auto_tags(WAD, "MAP01") == {"secrets": 1.0}
    assert store.map_auto_tags(WAD, "MAP02") == {}  # stale file removed


def test_excluded_inherited_drops_auto_wad_tags(tmp_path):
    store = TagStore(tmp_path / "tags")
    store.write_auto(WAD, {"deathmatch": 1.0}, {})
    store.exclude_inherited(WAD, "MAP07", "deathmatch")
    assert "deathmatch" not in store.effective_tags(WAD, "MAP07")
    assert "deathmatch" in store.effective_tags(WAD, "MAP01")


def test_rename_does_not_touch_auto_layer(tmp_path):
    store = TagStore(tmp_path / "tags")
    store.write_auto(WAD, {"deathmatch": 1.0}, {})
    store.tag_wad(WAD, "deathmatch")  # user also asserts it
    store.rename_tag("deathmatch", "dm")
    assert store.wad_tags(WAD) == {"dm": 1.0}          # user layer renamed
    assert store.wad_auto_tags(WAD) == {"deathmatch": 1.0}  # auto layer untouched


def test_auto_tags_registered_with_descriptions(tmp_path):
    store = TagStore(tmp_path / "tags")
    from tagging.auto import AUTO_TAG_DESCRIPTIONS
    store.write_auto(WAD, {"deathmatch": 1.0}, {}, AUTO_TAG_DESCRIPTIONS)
    entry = store.registry.get("deathmatch")
    assert entry is not None and entry.description.startswith("[auto]")

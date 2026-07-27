"""Tests for the tag registry, store (inheritance/exclusion/rename/merge),
and SVG preview rendering."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wad_builder import build_two_sector_wad

from tagging.registry import Registry, canonicalize
from tagging.preview import map_to_svg
from tagging.store import TagStore
from wad_parser.map_extractor import extract_all_maps
from wad_parser.wad_reader import WadFile

WAD = "a" * 64
WAD2 = "b" * 64


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def test_canonicalization():
    assert canonicalize("  Tech  Base ") == "tech base"
    assert canonicalize("TECHBASE") == "techbase"
    with pytest.raises(ValueError):
        canonicalize("   ")


def test_registry_create_persist_complete(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    reg.ensure("TechBase", "techy maps")
    reg.ensure("tech demo")
    reg.ensure("hell")

    reloaded = Registry(tmp_path / "registry.json")
    assert reloaded.get("techbase").display == "TechBase"
    assert reloaded.get("techbase").description == "techy maps"
    # prefix matches first, then substring
    assert reloaded.complete("tech") == ["tech demo", "techbase"]
    assert reloaded.complete("demo") == ["tech demo"]


def test_registry_rename_conflict_requires_merge(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    reg.ensure("techbase")
    reg.ensure("tech-base")
    with pytest.raises(ValueError):
        reg.rename("tech-base", "techbase")


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------

def test_wad_and_map_tags_with_inheritance(tmp_path):
    store = TagStore(tmp_path)
    store.tag_wad(WAD, "Multiplayer", wad_name="dm.wad")
    store.tag_wad(WAD, "slaughter", weight=0.5)
    store.tag_map(WAD, "map01", "techbase")

    assert store.wad_tags(WAD) == {"multiplayer": 1.0, "slaughter": 0.5}
    # map name canonicalized to uppercase, effective = own ∪ inherited
    assert store.effective_tags(WAD, "MAP01") == {
        "multiplayer": 1.0, "slaughter": 0.5, "techbase": 1.0,
    }
    # untouched sibling map only inherits
    assert store.effective_tags(WAD, "MAP02") == {"multiplayer": 1.0, "slaughter": 0.5}


def test_map_weight_overrides_inherited(tmp_path):
    store = TagStore(tmp_path)
    store.tag_wad(WAD, "hell", weight=1.0)
    store.tag_map(WAD, "MAP01", "hell", weight=0.3)
    assert store.effective_tags(WAD, "MAP01") == {"hell": 0.3}


def test_exclusion_and_reinclusion(tmp_path):
    store = TagStore(tmp_path)
    store.tag_wad(WAD, "multiplayer")
    store.exclude_inherited(WAD, "MAP07", "multiplayer")
    assert store.effective_tags(WAD, "MAP07") == {}
    assert store.effective_tags(WAD, "MAP01") == {"multiplayer": 1.0}
    store.include_inherited(WAD, "MAP07", "multiplayer")
    assert store.effective_tags(WAD, "MAP07") == {"multiplayer": 1.0}


def test_untag(tmp_path):
    store = TagStore(tmp_path)
    store.tag_wad(WAD, "a")
    store.tag_map(WAD, "MAP01", "b")
    assert store.untag_wad(WAD, "a") is True
    assert store.untag_wad(WAD, "a") is False
    assert store.untag_map(WAD, "MAP01", "b") is True
    assert store.effective_tags(WAD, "MAP01") == {}


def test_rename_rewrites_all_files(tmp_path):
    store = TagStore(tmp_path)
    store.tag_wad(WAD, "tech base")
    store.tag_map(WAD, "MAP01", "tech base", weight=0.8)
    store.tag_wad(WAD2, "tech base")
    store.exclude_inherited(WAD2, "E1M1", "tech base")

    touched = store.rename_tag("tech base", "techbase")
    assert touched == 4
    assert store.wad_tags(WAD) == {"techbase": 1.0}
    assert store.map_own_tags(WAD, "MAP01") == {"techbase": 0.8}
    assert store.map_record(WAD2, "E1M1")["excluded_inherited"] == ["techbase"]
    assert "tech base" not in store.registry.tags
    assert store.registry.get("techbase") is not None


def test_merge_keeps_existing_target_assignment(tmp_path):
    store = TagStore(tmp_path)
    store.tag_wad(WAD, "tech-base", weight=0.9)
    store.tag_wad(WAD, "techbase", weight=0.4)   # both on same WAD
    store.tag_wad(WAD2, "tech-base", weight=0.7)  # only source on this one

    store.merge_tags("tech-base", "techbase")
    assert store.wad_tags(WAD) == {"techbase": 0.4}   # existing target wins
    assert store.wad_tags(WAD2) == {"techbase": 0.7}  # source folded in
    assert "tech-base" not in store.registry.tags


def test_tag_usage_counts(tmp_path):
    store = TagStore(tmp_path)
    store.tag_wad(WAD, "hell")
    store.tag_map(WAD, "MAP01", "hell")
    store.tag_map(WAD, "MAP02", "city")
    assert store.tag_usage() == {"hell": 2, "city": 1}


def test_store_files_match_planned_layout(tmp_path):
    store = TagStore(tmp_path)
    store.tag_wad(WAD, "x", wad_name="foo.wad")
    store.tag_map(WAD, "map01", "y")
    assert (tmp_path / f"{WAD}.wad.json").exists()
    assert (tmp_path / WAD / "MAP01.json").exists()
    rec = json.loads((tmp_path / f"{WAD}.wad.json").read_text())
    assert rec["wad_name"] == "foo.wad"
    assert rec["tags"] == [{"tag": "x", "weight": 1.0}]


# ---------------------------------------------------------------------------
# preview
# ---------------------------------------------------------------------------

def test_svg_preview_renders(tmp_path):
    p = tmp_path / "two.wad"
    p.write_bytes(build_two_sector_wad())
    doc = extract_all_maps(WadFile(str(p)))[0]
    svg = map_to_svg(doc)
    assert svg.startswith("<svg")
    assert "<line" in svg          # walls drawn
    assert 'fill="#2e9e4f"' in svg  # player start dot
    assert 'fill="#d04b3e"' in svg  # monster dot
    assert "<title>MAP01</title>" in svg

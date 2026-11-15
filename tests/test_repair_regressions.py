"""Regression coverage for the first batch in REPO_REVIEW.md."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wad_builder import build_two_sector_wad

from cli.parse_wad import main as parse_wad
from data.dataset import MapDataset
from data.rasterizer import rasterize
from generation.pipeline import raster_to_document
from wad_parser.map_extractor import extract_all_maps
from wad_parser.wad_reader import WadFile
from wad_writer.assembler import assemble_map_lumps
from wad_writer.wad_writer import write_wad


def _doc(tmp_path):
    path = tmp_path / "source.wad"
    path.write_bytes(build_two_sector_wad())
    return extract_all_maps(WadFile(str(path)))[0]


def test_extraction_preserves_same_named_maps_and_is_idempotent(tmp_path):
    out = tmp_path / "maps"
    runner = CliRunner()
    for name, suffix in [("a.wad", b""), ("b.wad", b"different source")]:
        path = tmp_path / name
        path.write_bytes(build_two_sector_wad() + suffix)
        result = runner.invoke(parse_wad, [str(path), "--output", str(out)])
        assert result.exit_code == 0, result.output
    ds = MapDataset(out)
    assert len(ds) == 2
    docs = [ds.load_document(i) for i in range(len(ds))]
    assert {d.meta.map_name for d in docs} == {"MAP01"}
    assert len({d.meta.wad_hash for d in docs}) == 2
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in ds.paths}
    result = runner.invoke(parse_wad, [str(tmp_path / "a.wad"), "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in ds.paths} == before
    manifests = [json.loads(p.read_text()) for p in (out / ".manifests").glob("*.json")]
    assert len(manifests) == 2
    for manifest in manifests:
        entry, = manifest["maps"]
        assert (out / entry["path"]).is_file()
        assert entry["map_name"] == "MAP01"
        assert entry["content_hash"]
    # Repackaging identical map content retains both provenance records and
    # exposes the duplicate content in the manifests for later split/dedupe.
    assert manifests[0]["maps"][0]["content_hash"] == manifests[1]["maps"][0]["content_hash"]


def test_extraction_refuses_to_replace_unrelated_document(tmp_path):
    path = tmp_path / "source.wad"
    path.write_bytes(build_two_sector_wad())
    wad = WadFile(str(path))
    out = tmp_path / "maps"
    out.mkdir()
    dest = out / f"{wad.hash}_MAP01.json"
    dest.write_text('{"meta": {"wad_hash": "another-wad", "map_name": "MAP01"}}')
    before = dest.read_bytes()
    result = CliRunner().invoke(parse_wad, [str(path), "--output", str(out)])
    assert result.exit_code != 0
    assert "collision" in result.output.lower()
    assert dest.read_bytes() == before


def test_incremental_extraction_keeps_existing_manifest_entries(tmp_path):
    first = _doc(tmp_path)
    second = copy.deepcopy(first)
    second.meta.map_name = "MAP02"
    path = write_wad([first, second], tmp_path / "two.wad")
    out = tmp_path / "maps"
    for name in ("MAP02", "MAP01"):
        result = CliRunner().invoke(parse_wad, [str(path), "--output", str(out), "--map", name])
        assert result.exit_code == 0, result.output
    manifest_path, = (out / ".manifests").glob("*.json")
    manifest = json.loads(manifest_path.read_text())
    assert {entry["map_name"] for entry in manifest["maps"]} == {"MAP01", "MAP02"}
    assert len(MapDataset(out)) == 2


def test_map_filter_does_not_parse_unrequested_maps(tmp_path, monkeypatch):
    import wad_parser.map_extractor as extractor
    first = _doc(tmp_path)
    second = copy.deepcopy(first)
    second.meta.map_name = 'MAP02'
    path = write_wad([first, second], tmp_path / 'two.wad')
    original = extractor.extract_map
    visited = []
    def extract(wad, group, *args):
        visited.append(group.name)
        if group.name == 'MAP02':
            raise RuntimeError('unrequested map must not be parsed')
        return original(wad, group, *args)
    monkeypatch.setattr(extractor, 'extract_map', extract)
    result = CliRunner().invoke(parse_wad, [str(path), '--map', 'MAP01', '--output', str(tmp_path / 'maps')])
    assert result.exit_code == 0, (result.output, result.exception)
    assert visited == ['MAP01']


@pytest.mark.parametrize("value", [40000, -40000, float("inf"), float("nan")])
def test_writer_rejects_unrepresentable_coordinates_without_touching_output(tmp_path, value):
    doc = _doc(tmp_path)
    doc.vertices[0].x = value
    dest = tmp_path / "out.wad"
    dest.write_bytes(b"previous valid output")
    with pytest.raises(ValueError, match="vertex 0.*x"):
        write_wad([doc], dest)
    assert dest.read_bytes() == b"previous valid output"


@pytest.mark.parametrize("field,value", [("special", 65536), ("tag", -1), ("v1", 99),
                                          ("right_sidedef", -1)])
def test_writer_rejects_invalid_linedef_fields(tmp_path, field, value):
    doc = _doc(tmp_path)
    setattr(doc.linedefs[0], field, value)
    with pytest.raises(ValueError, match=f"linedef 0.*{field}"):
        assemble_map_lumps(doc)


def test_writer_rejects_edge_collapsed_by_integer_rounding(tmp_path):
    doc = _doc(tmp_path)
    doc.vertices[1].x = 0.25
    doc.vertices[1].y = 0.25
    with pytest.raises(ValueError, match="linedef 0.*rounding"):
        assemble_map_lumps(doc)


def test_writer_keeps_signed_coordinate_limits(tmp_path):
    doc = _doc(tmp_path)
    doc.vertices[0].x = -32768
    doc.vertices[1].x = 32767
    out = write_wad([doc], tmp_path / "limits.wad")
    reread = extract_all_maps(WadFile(str(out)))[0]
    assert [v.x for v in reread.vertices[:2]] == [-32768, 32767]


@pytest.mark.parametrize('count', [1, 8, 9, 81])
def test_writer_provides_sized_all_visible_reject_matrix(count):
    from tests.geometry_fixtures import polygon_document, square
    doc = polygon_document([[square(0, 0, 128, 128)]], start=(64, 64))
    template = doc.sectors[0]
    doc.sectors = [copy.deepcopy(template) for _ in range(count)]
    for sid, sector in enumerate(doc.sectors):
        sector.id = sid
    lumps = assemble_map_lumps(doc)
    assert lumps['REJECT'] == bytes((count * count + 7) // 8)
    assert lumps['REJECT']  # even a one-sector map requires one visibility bit


@pytest.mark.parametrize("kind,field,value", [
    ("sectors", "ceil_height", 40000), ("sectors", "tag", 32768),
    ("sidedefs", "sector", -1), ("sidedefs", "x_offset", -40000),
    ("things", "x", 40000), ("linedefs", "left_sidedef", 65535),
    ("linedefs", "v1", 0.0), ("vertices", "id", 99)])
def test_writer_checks_other_field_ranges_and_references(tmp_path, kind, field, value):
    doc = _doc(tmp_path)
    setattr(getattr(doc, kind)[0], field, value)
    with pytest.raises(ValueError):
        assemble_map_lumps(doc)


def test_generation_supplies_exit_without_style(tmp_path):
    raster, _ = rasterize(_doc(tmp_path))
    generated, repairs, _ = raster_to_document(raster)
    assert any(ld.special == 11 for ld in generated.linedefs)
    assert any("exit" in repair.lower() for repair in repairs)


def test_generation_preserves_existing_exit(tmp_path):
    from generation.progression import ensure_exit

    doc = _doc(tmp_path)
    doc.linedefs[0].special = 11
    before = copy.deepcopy(doc.to_dict())
    assert ensure_exit(doc) is None
    assert doc.to_dict() == before


def test_exit_insertion_fails_instead_of_silently_skipping(tmp_path):
    from generation.progression import ensure_exit

    doc = _doc(tmp_path)
    for ld in doc.linedefs:
        ld.special = 1
    with pytest.raises(ValueError, match="exit"):
        ensure_exit(doc)


@pytest.mark.parametrize("shaped", [False, True])
def test_roomgen_writes_exit_without_optional_style(tmp_path, monkeypatch, shaped):
    from generation import roomgen
    from roomgraph.types import Connection, Room, RoomGraph

    rooms = [Room(id=i, sectors=[], floor_h=0, ceil_h=128, area=384**2,
                  centroid=(0, 0), bbox=(0, 0, 384, 384)) for i in range(2)]
    graph = RoomGraph(rooms=rooms, connections=[Connection(0, 1, "corridor", 160)])
    monkeypatch.setattr(roomgen, "load_roomgen_model", lambda *args: (None, None))
    monkeypatch.setattr(roomgen, "sample_room_graph", lambda *args, **kwargs: graph)
    import generation.pipeline as pipeline
    monkeypatch.setattr(pipeline, "build_nodes", lambda *args: False)
    library = {"square": [{"poly": [(0, 0), (0, 384), (384, 384), (384, 0)],
                           "area": 384**2}]} if shaped else None
    result = roomgen.generate("unused.pt", {}, out_path=tmp_path / "out.wad", shape_library=library)
    generated, = extract_all_maps(WadFile(result["path"]))
    assert any(ld.special == 11 for ld in generated.linedefs)
    assert generated.derived.reachable_from_start


def test_generation_modules_import_without_torch():
    code = """
import importlib.abc
import sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ImportError('torch deliberately unavailable')
sys.meta_path.insert(0, NoTorch())
import cli.generate_wad
import cli.generate_roomgen
import generation.pipeline
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True,
                   cwd=Path(__file__).resolve().parent.parent)


def test_roomgen_cli_forwards_shape_library_and_room_limit(tmp_path, monkeypatch):
    from cli import generate_roomgen

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    library = tmp_path / "shapes.json"
    library.write_text("{}")
    args = {}

    def generate(*pos, **kw):
        args.update(kw)
        return {"path": "out.wad", "maps": ["MAP01"], "nodes_built": True, "summaries": []}

    monkeypatch.setattr(generate_roomgen, "generate", generate)
    result = CliRunner().invoke(generate_roomgen.main, [
        "--checkpoint", str(checkpoint), "--shape-library", str(library), "--max-rooms", "12"])
    assert result.exit_code == 0, result.output
    assert args["shape_library"] == str(library)
    assert args["max_rooms"] == 12


@pytest.mark.parametrize('spec', ['theme:nan', 'theme:inf', 'theme:bad', ':1', 'theme:-1'])
def test_cli_rejects_invalid_tag_weights(spec):
    from cli.options import parse_tags
    import click
    with pytest.raises(click.BadParameter):
        parse_tags(spec)

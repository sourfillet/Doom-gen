# Doom-gen

Doom-gen learns from Doom WAD maps and generates new PWADs. It is an
experimental research project: generated maps pass static, serialized and
compiled validation, but layout quality and balance are not established.

## How it works

- **Room-graph path (main).** An RNN samples a room graph: rooms, sizes,
  heights, connections and start/exit/key roles. Procedural routing then places
  the rooms and corridors, keeping every requested connection.
- **Raster path.** A tag-conditioned diffusion model paints a top-down layout
  raster that is vectorized into sectors.
- **Decoration.** Optional property and placement GNNs predict textures and
  things; a simpler style pass fitted on the corpus can be used instead.
- **Experimental full-map geometry RNN.** Emits complete map records with no
  procedural assembly. Its outputs are diagnostics, not playable maps yet.

Geometry variation, retrieved outlines, intro encounters, terraces and the
blue-key objective are procedural additions, not learned.

Before a map is published it must have a valid start and a reachable, usable
exit, checked with player clearance and key/door history, both in memory and
after serialization. Nodes are built with zdbsp when available, and a failed
publication never replaces existing output.

## Install

Python 3.10 or newer:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,ml]'
```

Use `.[dev]` alone for extraction and diagnostics without Torch, and add
`engine` for ViZDoom checks. A locked environment is available with
`uv sync --locked --extra dev --extra ml`. Install `zdbsp` separately to build
nodes.

## Quick start

```sh
parse-wad /path/to/DOOM2.WAD --output dataset/maps
prepare-corpus --maps-dir dataset/maps --out dataset/corpus.json --seed 7
train-roomgen --maps-dir dataset/maps --manifest dataset/corpus.json --out checkpoints/rooms --steps 15000 --seed 7
generate-roomgen --checkpoint checkpoints/rooms/best.pt --seed 7 --max-rooms 20 --max-attempts 3 --out generated.wad
gzdoom -iwad /path/to/DOOM2.WAD -file generated.wad -warp 1
```

Add `--gameplay intro --game /path/to/DOOM2.WAD --style dataset/maps` to
`generate-roomgen` to add small encounters, a weapon cache and health.
GZDoom-family ports rebuild the empty node/BLOCKMAP lumps on load.

## Commands

| Command | Purpose |
| --- | --- |
| `parse-wad` | Extract each map in a WAD to a JSON document. |
| `tag-maps` | Assign and manage WAD/map tags used for conditioning. |
| `prepare-corpus` | Build deterministic family-held-out train/validation/test splits. |
| `diagnose-maps` | Save per-map conversion diagnostics (JSON, SVG, metrics). |
| `train-roomgen` / `generate-roomgen` | Train the room-graph model / sample and publish maps. |
| `train-layout` / `generate-wad` | Train the raster diffusion model / sample and publish maps. |
| `train-properties` / `train-placement` | Train the texture and thing-placement GNNs. |
| `overfit-roomgen` / `evaluate-roomgen` | Memorization probes / held-out evaluation with baselines. |

Every command supports `--help`. Source-tree tools for audits, evaluation and
the experimental geometry model live in `tools/` and run with
`python -m tools.<name>`.

## Documentation

- [Preparing data](docs/data-preparation.md): extraction, corpora, source
  audits and format versions.
- [Room-graph generation](docs/room-graph-generation.md): training, routing,
  geometry profiles, resume and decoration.
- [Procedural gameplay](docs/gameplay.md): intro encounters, the blue-key
  objective and interior terraces.
- [Evaluation](docs/evaluation.md): probes, held-out evaluation, seed pooling
  and routing replay.
- [Raster diffusion](docs/raster-diffusion.md): the raster path, decoding and
  raster versions.
- [Output, validation and assets](docs/output-and-validation.md): what
  validation guarantees, publication, target ports and textures.
- [Experimental learned geometry](docs/learned-geometry.md): the full-map
  geometry RNN and spatial reconstruction research.
- [Development](docs/development.md): environment, tests, engine/port checks
  and CI.

## Tests

```sh
python -m pytest -q
```

Engine (`DOOM_GEN_ENGINE_TESTS=1`) and source-port (`DOOM_GEN_PORT_TESTS=1`)
checks are opt-in; see [docs/development.md](docs/development.md).

## Repository layout

| Path | Contents |
| --- | --- |
| `wad_parser/` | Binary/UDMF extraction, geometry, BSP regions and traversal. |
| `data/` | Augmentation, raster/graph targets, corpora and caches. |
| `models/` | Models, conditioning, checkpoints and reproducible training. |
| `roomgraph/` | Segmentation, routing, geometry and requested connections. |
| `generation/` | Sampling, validation, decoration and publication. |
| `wad_writer/` | Serialization, bundled resources and port compatibility. |
| `tagging/` | Tag registry, assignments and previews. |
| `cli/`, `tools/` | Installed commands and reproducible evaluation utilities. |
| `tests/` | Regression tests and optional engine controllers. |
| `plan/` | Design documents and roadmap. |

## Game data

No IWADs, commercial graphics, training corpora or checkpoints are included or
committed. Use your own legally obtained WADs. Engine tests use the Freedoom II
IWAD bundled with ViZDoom.

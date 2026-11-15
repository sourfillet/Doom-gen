# Roadmap

> Historical roadmap. The current proposed sequence is
> [ML-first migration](../ML_FIRST_MIGRATION.md). Phase 9's claim below that raster
> diffusion cannot learn coherent room structure is not established by the
> repaired repository; its procedural geometry stage is not the intended
> ML-first endpoint.

Phased so each phase produces something runnable/checkable before the next
begins, rather than a big-bang rewrite.

## Phase 0 — Extraction core
- Binary-format lump parser (`THINGS`/`LINEDEFS`/`SIDEDEFS`/`VERTEXES`/`SECTORS`)
  producing the raw+derived JSON schema from [data-model.md](data-model.md).
- Texture-system parsing (`TEXTURE1/2`, `PNAMES`, flats, `PLAYPAL`, `COLORMAP`).
- CLI: `parse-wad <file> --output <dir>`.
- **Checkpoint:** round-trip a handful of well-known WADs (`DOOM2.WAD`,
  a couple of popular PWADs) and manually spot-check the JSON against a known
  map editor's view.

## Phase 1 — Extraction completeness
- UDMF/`TEXTMAP` parser (slopes, 3D floors, per-field extras).
- `ANIMATED`/`SWITCHES`, `MAPINFO`/`ZMAPINFO`/`EMAPINFO` parsing.
- Derived-data pass: sector polygons, adjacency, reachability, texture
  alignment, symmetry detection.
- **Checkpoint:** a UDMF map with slopes and a Boom map with animated
  textures both parse without falling back to the `extra` bag for anything
  that has a typed field defined.

## Phase 2 — Tagging system
- Tag registry (create/rename/merge, canonicalization, autocomplete).
- WAD-level and per-map tag storage with inheritance/exclusion.
- Interactive tagging CLI with lightweight top-down preview.
- Heuristic tag *suggestions* (non-authoritative).
- **Checkpoint:** tag a small real corpus end-to-end (WAD-level tags on
  single-map WADs, per-map tags on a megawad), confirm inheritance resolves
  correctly, confirm re-tagging doesn't touch parsed data at all.

## Phase 3 — Dataset build
- Rasterizer (map document → multi-channel array).
- Graph builder (map document → attributed graph tensors, full field set).
- Augmentation (mirror/rotate/jitter with correct thing-angle and
  sidedef-reassignment handling).
- Tag embedding table + text-encoder cold-start fallback.
- **Checkpoint:** dataset loader produces (raster, graph, tag_vector) triples
  for the whole tagged corpus; augmented samples still pass the Stage-3
  validity checks structurally (no broken sector rings after mirroring).

## Phase 4 — Stage 1: Layout diffusion
- U-Net + DDPM/DDIM training loop, tag-conditioned via CFG.
- **Checkpoint:** unconditional samples look like plausible Doom floorplans;
  conditional samples visibly shift with different tag sets.

## Phase 5 — Stage 2: Vectorization
- Algorithmic contour-tracing vectorizer.
- **Checkpoint:** sampled rasters convert to valid sector polygon sets without
  manual cleanup on the majority of samples.

## Phase 6 — Stage 3: Property GNN
- All classification/regression heads, tag-consistency repair pass.
- **Checkpoint:** filled-in maps have consistent textures/heights/specials
  and pass the linedef/sector tag consistency check.

## Phase 7 — Stage 4: Thing placement + validation/postprocessing + WAD writer
- Thing budget/type/position model, deterministic special-thing placement.
- Full validity gauntlet + repair passes.
- Node-builder integration, PWAD assembly.
- **Checkpoint:** end-to-end `generate --tags ...` produces a WAD that loads
  and is completable in a source port.

## Phase 8 — Stretch goals
- Retrieval-augmented generation (nearest-neighbor structural exemplars for
  niche tag combinations).
- Incremental/online fine-tuning of tag embeddings as new tags accumulate
  labeled examples, without a full retrain.
- Native UDMF-format generation output (slopes, 3D floors) rather than only
  binary-format maps.

## Phase 9 — Graph-native geometry (course correction)

**Finding (2026-07):** Phases 4-5 (raster diffusion + vectorization)
produce decoration-ready geometry but *cannot* produce coherent room
structure — the model fills the canvas (no void) or, with an occupancy
channel, produces occupancy *noise* (no room-scale coherence). Rooms /
corridors / doors are a graph property a pixel model of this scale doesn't
capture. Evidence in `memory/model-fills-void-root-cause`.

**Direction:** replace the raster geometry generator with a graph-native
(floorplan-style) one — generate a room-adjacency graph, lay it out, and
assemble sectors/corridors/doors from it. Stages 3/4, tagging, WAD writer,
validator all reuse. Full design in [graph-native.md](graph-native.md).
De-risk order: room-graph representation + segmentation → geometry assembler
(graph→WAD, validated on real extracted graphs, the regen-harness analog) →
deterministic layout → learned autoregressive graph generator → re-plug
Stage 3/4. Prior art (House-GAN++/Graph2Plan/HouseDiffusion, GraphRNN,
Doom-GAN 2018) makes this an engineering build adapting known methods, not
novel research.

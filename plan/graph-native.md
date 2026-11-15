# Graph-Native Map Generation (design)

> Historical design notes. The claims below about a settled raster dead end
> and rooms floating in void describe an early interpretation, not a general
> result established by the repaired repo. Current limits and evidence are in
> `REPO_REVIEW.md`; the next spatial representation/assembly work is tracked in
> [adjoining-architecture.md](adjoining-architecture.md).

Replacement for the raster-diffusion geometry generator (Stage 1/2), which
we proved cannot produce coherent room structure. Everything *downstream* of
geometry — textures, thing placement, WAD writing, tagging — is kept and
reused. This is the one component we swap.

## Why: the raster dead end (settled with evidence)

See [[../memory]] `model-fills-void-root-cause`. Summary of the finding:

- A real Doom map is discrete **rooms floating in void, joined by
  corridors** — 57% of its linedefs are solid walls. Rooms are first-class
  objects with deliberate connections.
- The raster diffusion model represents a map as a filled pixel field. With
  the original 9-channel raster it **fills 100% of the canvas** (learns no
  void) → one blob fractured into height patches, 9% walls. Adding an
  explicit binary occupancy channel + weighted loss fixed the *amount* of
  void (100% → 41%, target ~35%) but the occupancy came out as **per-pixel
  salt-and-pepper noise** — 318 disconnected blobs, largest 9% of the
  occupied area. Weight it down → mush; weight it up → static. Neither is
  rooms.
- Conclusion: a pixel model of this size learns **marginals, not room-scale
  structure**. Rooms/corridors/doors are a *graph* property; a raster is the
  wrong representation to generate them in.

## Prior art (adaptation of a *current* line, not invention)

Two columns — what to learn *from* vs. what to learn *not* to do:

- **DON'T (image-based, same family as our dead end)**: Giacomello et al.,
  *DOOM Level Generation using GANs* (2018) — generates levels as
  multi-channel *bitmaps* via a DCGAN and hit the same structural-coherence
  limits we did. It is evidence the pixel approach fails, NOT a template for
  this design. Also dated technique: GANs → diffusion since ~2021.
- **DO — modern floorplan / structured layout** (a Doom map *is* a
  floorplan; these generate *geometry*, never pixels):
  - **Topology**: GraphRNN (2018) → **DiGress** (discrete graph diffusion,
    2023) for generating the room-adjacency graph.
  - **Layout**: **HouseDiffusion** (2023) diffuses over a floorplan's
    polygon *corner coordinates* conditioned on a room bubble-diagram;
    Graph2Plan (2020) and House-GAN++ (2021) are the same lineage
    (graph-conditioned geometry).
- **Classic PCG**: graph grammars, Wave Function Collapse, dungeon
  generators — decades of "rooms + connections" methods, useful for the
  deterministic v1 layout.

Caveat: floorplan lineage above is solid; the very latest *Doom-specific*
PCGML / transformer-sequence work is outside a reliable citation here
(knowledge cutoff), so treat this as the well-established core, not an
exhaustive survey. The uncertainty is *quality on our data* (variable
heights, non-rectangular sectors, ~1370 maps, engine constraints) —
execution/tuning risk, not novelty risk.

## Representation: the room graph

A map becomes `G = (rooms, connections)`:

- **Room node** — `centroid`, bounding `size (w,h)`, `shape_class`
  (rect / L / polygon / round), `floor_h`, `ceil_h`, `area`, `role`
  (start / exit / key / arena / hub / closet), plus the tag context.
- **Connection edge** — `type` ∈ {door, open-threshold, corridor, lift,
  teleport, window}, endpoints, and an approximate side/position hint.

This is the natural Doom abstraction and, crucially, it makes **doors and
corridors first-class** — the exact things the current output lacks.

## Deriving room graphs from the corpus (segmentation)

The parser already yields sectors + a sector-adjacency graph. A *room* is
usually several sectors (a room with a raised ledge is 2+ sectors), so we
need a **sector → room segmentation**:

- Merge sectors joined by *open, low-step, wide* two-sided lines (same room).
- Split at solid walls, doors (door specials / narrow blocking lines), and
  thin necks (corridors become their own room-nodes or edges).
- Heuristic v1: union-find over sectors, merging across two-sided lines whose
  |floor step| < 48 and shared length > a corridor-width threshold; classify
  a merged component as a *room*; classify long thin components as
  *corridors* (edges). Doors are two-sided lines with a door special.

This is itself a real sub-task (and imperfect), but it only needs to be
*good enough* to produce training graphs — it is not on the generation path.

## Architecture — three stages

### G1. Room-graph generator (the learned core)

Input: tags (conditioning). Output: a room graph (nodes with attributes +
edges with types). Options, with recommendation:

| Option | Fit | Notes |
|---|---|---|
| Autoregressive (add room, attach edges) | **recommended v1** | Controllable, sample-efficient on small data, natural stopping; GraphRNN/GRAN lineage |
| Graph VAE | good | Latent interpolation, but reconstruction of discrete graphs is fiddly |
| Discrete graph diffusion (DiGress) | later | Strong but data-hungry; revisit if corpus grows |

Recommendation: a **conditional autoregressive generator** — emit rooms
one at a time (attributes from small MLP/GRU heads), and for each new room
predict connections to existing rooms (a pointer/attention over prior
nodes). Tag conditioning via the existing `TagConditioner`. Trained with
teacher forcing on a canonical node ordering (e.g. BFS from the start room).

### G2. Geometric layout (graph → coordinates)

Turn the abstract graph into non-overlapping placed rooms realizing the
adjacencies. v1 = **deterministic** (coherent by construction):
grid/slicing-tree packing seeded by the graph, room sizes from node
attributes, then route corridors along edges. **v2 = graph-conditioned
diffusion over vector coordinates (HouseDiffusion-style)** — diffuse room
corner positions conditioned on the room graph. This is the current SOTA and
sits in the diffusion machinery we already have; note it diffuses over
*geometry*, not pixels, which is exactly why it avoids our failure mode. It
is a *layout* method (graph in, coordinates out) — it assumes G1 supplies the
graph.

### G3. Geometry assembly (rooms + corridors + doors → sectors)

Instantiate real geometry: each room → a sector polygon (its `shape_class`
at its placed size), each corridor edge → a thin connecting sector, each
door edge → a short sector with a door linedef special + door-track
textures. Emit vertices/linedefs/sidedefs with correct one-sided (wall) vs
two-sided (opening) assignment. **This is where walls, corridors and doors
finally come from** — by construction, not by luck.

## What we reuse (most of the pipeline is already built)

- **Extraction / parsing** — already produces sectors + graph.
- **Tagging + conditioning** — `TagConditioner`, tag store, auto-tags.
- **Stage 3 property GNN** — textures/flats/specials, now on *room* nodes
  (retrain on room-graphs; the model is unchanged, the input graph differs).
- **Stage 4 placement GNN** — monster/pickup counts + types per room (reuse
  directly — it already keys on sector/room features).
- **WAD writer, game profiles, asset bundling, validator** — unchanged.

Retired: the rasterizer / vectorizer / layout-diffusion for *geometry* (kept
in-tree; the occupancy channel + fixes stay, they're harmless and correct
for the regen harness).

## Phased build plan (de-risk in the same order as the raster pipeline)

1. **Room-graph representation + segmentation** — derive room-graphs for the
   corpus; a `RoomGraph` type + `segment(doc) -> RoomGraph`. Checkpoint:
   segmentation of real maps looks sane (room counts, corridor/door counts in
   a believable range vs the source).
2. **G3 assembler first** — `RoomGraph -> MapDocument` (geometry + doors +
   corridors), then reuse the WAD writer. Checkpoint: feed a *real map's
   extracted* room-graph through assembly and get a playable WAD with
   recognizable rooms/doors (the graph-native analog of the regen harness —
   proves the plumbing before any model).
3. **G2 deterministic layout** — so a graph with only adjacencies (no exact
   coords) still assembles cleanly.
4. **G1 learned generator** — train the autoregressive room-graph model,
   tag-conditioned. Checkpoint: sampled graphs assemble into coherent,
   connected, playable maps with actual rooms and corridors.
5. **Re-plug Stage 3/4** — textures + placement on generated room-graphs.
6. **Quality pass** — learned G2 layout, door/lock/key logic, difficulty
   pacing along the room graph.

## Risks & open questions

- **Segmentation quality** — garbage room-graphs → garbage training. Needs
  spot-checking; may need iteration on the merge/split heuristics.
- **Small corpus (~1370)** — autoregressive/VAE chosen partly for
  sample-efficiency; more WADs still help.
- **Layout feasibility** — packing rooms to satisfy all adjacencies without
  overlap can be over-constrained; v1 may drop hard-to-place edges (turn
  them into teleporters, which we already support).
- **Non-rectangular rooms** — v1 may restrict `shape_class` to a small set;
  richer shapes are a later upgrade.
- **Realism vs. correctness** — the validator still guarantees playability;
  the open question is aesthetic quality, which only iteration answers.

## Honest effort

A real build — a new generative model (G1), a layout step (G2), and a
geometry assembler (G3), plus segmentation. But it reuses ~70% of the
existing pipeline, follows a published architecture family, and de-risks in
the same "assembler-first, model-second" order that worked for the raster
stack. Weeks of iteration, not a research program — and the correct
architectural bet for this domain.

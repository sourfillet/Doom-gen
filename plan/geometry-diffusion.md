# Graph-conditioned geometry diffusion (learned stage-2)

Goal: replace the procedural assembler (apply_shapes retrieval + radial_layout +
_separate) with a LEARNED model that, conditioned on the stage-1 room graph,
generates each room's real polygon AND its global position. Corridors stay
procedural (the carver is validated, 0 crossings). This is the piece that puts
the ML back in charge of the geometry — the honest answer to "is the ML even
doing anything."

Prior art: HouseDiffusion (Shabani 2023) — diffusion over floorplan CORNER
COORDINATES, graph as condition, transformer denoiser with structured attention
(intra-room / inter-room-if-connected / global). We adapt, we don't port:
houses are a wall-to-wall PARTITION of a rectangle (no void, no corridors);
Doom has void + thin corridors + irregular connectivity. So we learn ROOM
polygons + positions only, and keep the procedural corridor carve.

## Representation
- Per room i: K corner coords (2D), global frame, normalized per-map to [-1,1].
- Fixed K (candidate 24-32) with padding + mask; rooms with >K corners are
  polygon-simplified (Douglas-Peucker) to K; rooms with <K corners padded.
- Condition = the graph: node feats (size-class, aspect, height, floor-step
  embeddings — SAME bins stage-1 emits, so it transfers), edges (connection
  kind). Diffusion over all corners of all rooms jointly; attention masked by
  intra-room / graph-adjacent / global.
- Target at train time = the REAL room polygons from segment.py (we already
  have graph+polygons for the whole corpus — training data EXISTS).

## Milestones (de-risk cheapest-first, assembler-first philosophy)
1. REPRESENTATION VALIDATION (do first, commits nothing): corpus rooms ->
   simplify to K corners -> normalize -> reconstruct -> raster-IoU vs original +
   overlay renders. Pick K. If K-corner normalized polys can't round-trip real
   rooms, the whole approach is dead. <1 hr.
2. Data tensors: (graph, [N x K x 2] corners, masks) for the corpus + heavy
   augmentation (rot/reflect/jitter — critical, see data risk).
3. Overfit ONE map (graph -> its geometry). If it can't overfit one, arch/repr
   is wrong. Cheap.
4. Train on corpus; generate geometry for held-out graphs. Metrics: room
   overlap rate, connected-room adjacency, corner-count realism, shape variety.
5. Integrate: stage-1 graph -> geometry model -> procedural corridor carve ->
   WAD. A/B vs the procedural assembler (0-crossing + connectivity + look).
6. Polish: discrete-coordinate branch for crisp walls; overlap/validity repair.

## Honest risks
- DATA SCARCITY is the big one. HouseDiffusion trained on RPLAN (~60k plans);
  we have ~1370 maps (~40x less). Mitigations: strong augmentation, condition
  hard on the graph (graph->realization, not free gen), simpler/smaller model.
  This is where OBLIGE/OBSIDIAN as a data factory would matter most — thousands
  of clean guaranteed-connected maps. Flag to user: geometry diffusion may need
  the data factory to shine.
- Coordinate diffusion can emit self-intersecting / overlapping polygons ->
  still need a validity + overlap repair pass (we have the pieces).
- Connected-room adjacency is soft (corridors bridge gaps) so less critical than
  in houses; long corridors if the model places connected rooms far apart.
- Compute is CHEAP (hundreds of corners vs 65k pixels) — GPU-feasible.

## Milestone 1 RESULT (validated) — strong green light + a refinement
9332 real rooms: corner counts median 4 (!), p90 12, p99 60. K=24 covers 96%
of rooms outright, reconstruction IoU 0.983 (p10 1.000); K=32 -> 98%/0.988. So
K=24 fixed-corner normalized polys round-trip real rooms almost perfectly.
FINDING: most Doom rooms are RECTANGLES (median 4 corners) — a learned model
will (correctly) make mostly orthogonal rooms; sets expectations (authentic,
not more "organic").

## DECOMPOSITION (data-driven revision of the plan)
Don't build one joint HouseDiffusion model — split by training-unit count:
- ROOM SHAPE model: polygon from (size,aspect,height) attrs. ~85k training
  examples (one per room) = DATA-RICH, low-risk. Replaces apply_shapes
  retrieval -> genuinely GENERATED outlines. BUILD THIS FIRST.
- LAYOUT model: positions from graph. ~1370 examples (one per map) = data-
  scarce, high-risk. Keep radial_layout procedural for now; revisit with a
  data factory (OBLIGE/OBSIDIAN).
Shape-first: data-rich, directly answers "shapes are retrieved not generated,"
A/B-able vs retrieval, keeps the validated procedural layout+corridor carve.

## Expectation setting
Highest-effort thing in the project; most likely to underwhelm on pass 1 (data)
if done as the joint model — the shape-first split sidesteps most of that risk.
Still the ONLY path to ML-owned geometry. plan/ private, never commit.

# Room-graph generation

The room-graph model learns room attributes and connections. Room placement,
corridor routing and the optional geometry profiles are procedural; optional
polygon outlines come from retrieval.

## Train and sample

```sh
train-roomgen --maps-dir dataset/maps --manifest dataset/corpus.json --out checkpoints/rooms --steps 15000 --seed 7
generate-roomgen --checkpoint checkpoints/rooms/best.pt --seed 7 --max-rooms 20 --max-attempts 3 --out generated.wad
```

`best.pt` is selected on complete, usable validation graphs. Without any, use
the numbered final checkpoint instead.

## Training targets and stopping

- Version-3 room targets keep loop edges, connection widths and
  start/exit/key roles, and use calibrated world-unit size bins.
- Prefixes cut by `--max-train-rooms` are censored rather than labeled as true
  endings. The stop hazard uses a sequence likelihood to avoid overweighting
  short-map endings.
- Training defaults to graphs with a supported reachable exit and requires at
  least one supported complete ending. `--include-incomplete` adds source
  fragments with censored endings; fragments never select `best.pt`.
- Fragments are conservative reachable portions of maps whose exit could not
  be certified, so they are mostly the larger maps. With the default objective
  their continuations outnumber the complete maps' endings, and the learned
  stop hazard drifts toward zero as training continues.
- `--stop-targets complete` trains the stop hazard on complete maps only;
  fragments still train room and connection structure.
  `--stop-targets reachable-ends` ends every map after its last reachable room,
  so the hazard learns the size of reachable regions.
- `--complete-repeat N` lists each complete training map N times in the
  sampled pool.
- These settings are recorded in the checkpoint, and resume keeps them.

Observed on the v6 corpus: most complete training maps are one-room maps from a
single WAD, so the `complete` mode learns to stop almost immediately (local
evidence: `generated/repair-experiments/stop-routing-training-v1/`).

## Segmentation

Segmentation version 2 separates disconnected sector islands and represents
thin door slabs as connections.

## Routing and assembly

- Sampling masks physically non-planar loops. The assembler reserves routing
  lanes in a compact layout and tries up to sixteen deterministic room
  placements and twelve corridor edge orders per placement.
- If edges remain unrouted, a planar fallback realizes every physical
  connection with a visibility representation: each room becomes a bar of
  corridor junctions on one grid row, and each connection becomes one straight
  corridor. Graphs that lane routing realizes are unchanged. Fallback maps use
  compact spacing and report `router: planar_visibility`. Pass
  `planar_fallback=False` to `roomgraph.assemble.assemble` for lane routing
  only.
- Attempts and unresolved edges remain in generation reports. Each candidate is
  serialized, node-built when zdbsp is available, and checked for
  target-port/BSP/playability errors before acceptance. Compiled-map failures
  consume the same bounded attempt budget, and accepted maps keep their
  compiled bytes when combined. `--report` also saves per-attempt evidence when
  every candidate fails.
- `--shape-library library.json` retrieves merged outlines and holes; interior
  sector height changes are reduced to one room height.
- Still limited: finer interior geometry, directed connection types and
  scripted progression.

## Geometry profiles

All profiles keep the routing budget, stairs, doors, requested topology and
required-edge checks. The model does not predict these outlines, and `--seed`
also controls the procedural variation. Only the default `classic` profile can
be combined with `--shape-library`, which uses a separate assembler.

| `--geometry-profile` | Behavior |
| --- | --- |
| `classic` (default) | Rectangular geometry, for reproducible comparisons. |
| `varied` | Unequal row/column spacing, off-center corridor openings and independently clipped room corners. |
| `compact` | Keeps the varied corners and entrances but spaces occupied rows/columns by their actual room extents, reserving a full stair run per routed connection, instead of multiplying one largest-room spacing across empty lanes. Room dimensions, floor heights and connections are unchanged; decisions are recorded in `geometry_variation.spacing`. This is a density option; it does not retrain the model or solve missing routes. |
| `adjoining` | Starts from compact routing and grows eligible straight neighboring rooms toward a shared wall band: open thresholds 16 units deep, flat doors 32 and short corridors 64. Larger stairs and routed bends keep their existing realization. Growth is bounded and checked against other rooms and transit space. Requested/realized bounds and accepted/rejected joins are in `geometry_variation.adjacency`, in the assembly coordinate frame (any later `coordinate_translation` is applied separately). |

## Training, caching and resume

All four trainers (`train-layout`, `train-roomgen`, `train-properties` and
`train-placement`) use family holdouts, training-only vocabularies, seeded
validation, per-step JSONL metrics, best-validation checkpoints and complete
optimizer/RNG state.

```sh
train-roomgen --maps-dir dataset/maps --out checkpoints/rooms --resume checkpoints/rooms/roomgen_015000.pt --steps 20000
```

- `--steps` is the total step count. Resume restores the checkpoint settings
  and rejects changed corpus documents, tag data, dependency versions, or
  training/preprocessing source.
- Starting a fresh run in an existing checkpoint directory is rejected; use a
  new directory or resume explicitly.
- Old checkpoints without complete training state are inference-only.
- Layout rasters are cached on disk and loaded on demand. Raster cache
  namespace v2 records and checks the scoped dataset, augmentation and
  rasterization source plus Python/NumPy/Shapely/GEOS versions. Older or
  mismatched entries are kept but rebuilt on demand; the tensor contract
  remains raster version 5. A cache hit does not override the trainer's own
  checkpoint/resume compatibility checks.

## Source-ending assessment

New room-model training uses source-ending assessment `physical_exit_v3`. An
exit-room role also needs a physical player approach and a supported key/door
history, including supported one-shot type-39 teleport consumption.

- Unsupported by this event solver: active backside/failed attempts,
  same-sector trigger planes and extension teleport histories.
- Incomplete examples are excluded by default, or have their stop label
  censored with `--include-incomplete`.
- Evaluating historical checkpoints preserves their original assessment
  version; `physical_exit_v2` keeps its earlier teleport-reuse assumption for
  historical comparisons. Exact source references also require matching
  preprocessing versions or the archived source.

## Decoration

`--style`, `--properties` and `--placement` optionally decorate generated maps.

- Property models predict wall textures and signed offsets per sidedef.
- Placement models learn counts/types, local position distributions and facing
  for single-player skill 3.
- Both placers check actor dimensions, wall clearance, reachability and solid
  collisions, and report requested versus placed budgets. Solid actors use
  their actual radii; non-solid pickups can be placed close to walls. Inactive
  monsters and coop start markers do not block single-player spawns.
- `--game` restricts assets and thing types to a supplied game profile or IWAD.
- Still open: broader difficulty/mode conditioning and resource/key
  progression.

For procedural encounters, objectives and interior detail, see
[gameplay.md](gameplay.md).

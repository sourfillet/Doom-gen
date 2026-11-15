# Experimental learned geometry

This is research toward moving placement, boundaries, openings and interior
geometry into learned generation, while keeping deterministic compilation and
physical validation. It is separate from the established room-graph and raster
paths, and all of it is opt-in: legacy defaults and checkpoints are unchanged.

## Status

- A separate full-map supervision contract, a prediction-only decoder and a
  provenance-preserving preparation tool are implemented.
- The shared model emits complete variable-length map records, and an opt-in
  source-tree backend compiles them through the existing checks.
- The first real-data experiment produced literal squares.
- A separate learned control reproduces a nonrectangular source map and
  completes native exits, but every accepted control output is an exact
  training copy.
- A two-map capacity control produces nonrectangular, multi-level, 14-sector
  geometry in 7/8 final seeds, all exact training wall-geometry copies.
  Compiled geometry checks pass; static actor/progression checks fail.
- These are learning controls, not new playable levels.
- The bounded multi-map runs still produce no valid geometry in their fixed
  panels. Novel-layout quality and held-out learning remain open.
- None of the options below supplies room shapes, outline templates, target
  counts, stopping rules, or angle/length/closure/routing rules. None of them
  establishes useful new architecture.

## Commands

These are source-tree tools; they register no console commands.

```sh
python -m tools.prepare_learned_geometry --maps-dir dataset/maps --manifest dataset/corpus.json --out generated/learned-geometry
python -m tools.train_geometry --prepared generated/learned-geometry --out generated/geometry-training ...
python -m tools.sample_geometry --checkpoint generated/geometry-training/geometry_000500.pt --out generated/geometry-samples --samples 8 --max-tokens 4096
python -m tools.generate_geometry --checkpoint generated/geometry-training/geometry_001000.pt --game /path/to/DOOM2.WAD \
  --out generated/learned.wad --report-dir generated/learned-report --maps 4 --max-attempts 1
```

- Preparation reuses the frozen family split and keeps every rejection and
  source snapshot. Exported does **not** mean physically eligible, complete or
  ready for training.
- Sampling keeps every requested sample, including invalid or incomplete token
  streams. Its candidate WADs are uncompiled diagnostics, not playable
  publication.
- `generate_geometry` publishes with target assets and compiled checks, and
  requires zdbsp on PATH. Every bounded attempt is retained. It applies no
  procedural layout, geometry/exit repair, population or asset substitution.
- Use a compatible, freshly verified preparation; source signatures are
  checked.

The matched v7 configuration, for example:

```sh
.venv/bin/python -m tools.train_geometry --prepared generated/learned-geometry \
  --out generated/geometry-dependencies-new --token-codec interleaved \
  --record-memory --memory-read --record-anchors --relative-pointers \
  --spatial-pointers --dependency-memory --sampling coverage-family \
  --source-policy geometry --max-tokens 12000 --min-outline-corners 6 \
  --hidden 48 --embedding 32 --layers 1 --context-dim 16 --batch-size 1 \
  --seed 7 --threads 1 --lr 0.002 --stop-weight 5 \
  --steps 96 --save-every 24 --validate-every 24 \
  --checkpoint-wads generated/geometry-dependencies-wads-new
```

Adding `--ending-loss-weight 0.1` reproduces the separately evaluated ending
objective.

## Token codecs (`--token-codec`)

- `compact` losslessly compresses individual numeric and material fields with
  a training-only vocabulary and byte escapes.
- `relative` losslessly models preceding-vertex offsets and floor-relative
  ceilings.
- `interleaved` (with `--record-memory --memory-read --record-anchors`) is a
  separate learned typed-record stream. Source serialization places
  dependencies near their first use; inference chooses its own record types,
  coordinates, references and global ending. Source reindexing keeps every
  record and has an audit-only inverse permutation; index-sensitive mechanics
  still need independent correspondence checks. The original field-balanced
  objective does not support this stream.

## Source selection

- `--source-policy geometry` checks authored and serialized boundaries
  separately from actor/progression eligibility. Those sources are **not**
  positive gameplay examples; the policy is for reconstruction research and
  does not certify source gameplay eligibility.
- `--min-outline-corners 6` declares an explicit source-selection filter.
- Publication still runs the full checks. The generator's
  `--reject-flat-rectangles` rejects a single flat rectangular footprint
  instead of replacing it.

## Losses

- `--stop-weight` is an experimental loss weight for rare list endings. The
  trainer reports list-stop recall separately from continuation accuracy.
- `--loss-balance field` balances section/field/stop-class training losses
  (requires compact or relative tokens and stop weight 1).
- `--ending-loss-weight 0.1` adds a class-balanced binary global-ending loss to
  interleaved token NLL; the default 0 keeps the original loss. In the 96-step
  comparison it raised true-ending recall from 0/28 (training) and 0/9
  (validation) to 28/28 and 8/9, while predicting 582 premature validation
  endings. All 8 sampled maps still failed raw geometry.

## Record memory and pointers

- `--record-memory --token-codec compact` adds learned attention and reference
  heads over records the model has already emitted. Sampling starts with an
  empty bank and builds its own geometry history.
- `--memory-read` adds a separately versioned learned connection from the
  memory read into the reference query. Earlier memory checkpoints keep their
  original behavior.
- `--record-anchors` adds learned use of already completed references in the
  current record.
- `--relative-pointers` adds learned relative-index scores with no preset
  endpoint preference.
- `--spatial-pointers` adds learned candidate XY relations to completed vertex
  anchors; it requires `--relative-pointers`.
- `--dependency-memory` also learns sector-to-side and vertex/side-to-wall
  messages over emitted references. It requires the spatial pointer path and
  interleaved tokens. Its separate v7 checkpoint/cache contract starts at the
  v6 function; its 96-step trial still produced 0/8 valid raw geometries.

## Sampling (`--sampling`)

- `family` balances training families before choosing maps; validation still
  reports a complete map mean.
- `coverage-family` first draws every admitted training map once in a seeded
  shuffled pass, then uses family sampling. Each optimizer batch records source
  paths and lengths in `sampling-trace-STARTSTEP.jsonl`; resumed training
  writes a new segment.
- In a 96-step comparison, family sampling used only 15/28 maps for
  optimization. Complete coverage used 28/28 but still produced 0/8 valid raw
  geometries. Coverage is a data control, not a quality guarantee.

## Diagnostic WADs and copy audits

- `train_geometry --checkpoint-wads DIR` writes a diagnostic WAD at each
  initial, saved, validation and final checkpoint.
- `sample_geometry --raw-wads` keeps one for every sampled attempt.
- Complete records keep exact raw authoring. Incomplete streams become
  trace-only WADs **without a map marker**. Neither is compiled or playable
  publication.
- New copy audits use `geometry_fingerprint_v2`, which ignores unreferenced
  records when anchoring the geometry. The original fingerprint remains
  available to reproduce older reports.
- Local WAD inventories and galleries from these runs are under
  `generated/repair-experiments/` (for example
  `learned-topology-checkpoints-v1/` and `generation-checkpoints-v3/`), which
  git ignores.

## Spatial memorization probe

The initial source evidence export keeps detailed geometry separate from the
current v3 training labels. The usable-opening audit and the lossless authored
reconstruction contract preserve multi-height interiors, shared boundaries,
actions, materials and objects without treating source reconstruction as
learned generation.

The spatial probe predicts every shared vertex and thing XY and every sector
floor/ceiling, including collapsed doors and remote controls:

```sh
.venv/bin/python -m cli.overfit_spatial \
  --out generated/repair-experiments/spatial-memorization-new --steps 1200 --seed 7
.venv/bin/python -m tools.compare_geometry_methods \
  --checkpoint generated/repair-experiments/spatial-memorization-new/model.pt \
  --out generated/repair-experiments/spatial-comparison-new
.venv/bin/python -m tools.render_geometry_comparison \
  --experiment generated/repair-experiments/spatial-comparison-new
```

- The default source is frozen Dmonfear MAP19; `--source-json` selects another
  extracted classic map. `--target-json` accepts the lossless
  `doom_spatial_reconstruction_v1` contract. The earlier `geometry_v1` target
  omits collapsed doors/controls and is rejected for this prediction path.
- This is a single-source memorization control with learned slot identities.
  The source supplies authoring topology, assets, action/thing metadata and the
  affine frame. No source XY or sector elevation is kept as an inference
  fallback. It does not generate a new scaffold from a RoomGraph or show
  generalization.
- Saved-checkpoint inference reconstructs the candidate through explicit
  integer quantization, with separate basic, connected, exit/progression and
  serialized checks. Invalid predictions and disconnected candidates remain
  rejected evidence. The candidate WAD is diagnostic, and engine execution is
  reported separately.
- The comparison labels its differing conditioning/population policies; it is
  not a matched gameplay experiment. Missing checkpoints skip the learned arm,
  and missing node builders never count as successful compilation.
- Frozen source/graph pairs must match exact document hashes and WAD/map
  identity; shape-library inputs must match their training manifest. Changed
  authored targets, corrupt or unversioned checkpoints and mismatched source
  scaffolds are rejected. The checkpoint supports inference, not exact training
  resume.
- Use new output directories; historical v1 checkpoints and panels remain
  intact. The historical `geometry-comparison-v1` panel uses authored target
  reconstruction for its arm labeled `learned_geometry`; it does not evaluate
  checkpoint inference or run engine combat/progression checks.

## BSP region diagnostics

For the remaining BSP reconstruction discrepancies, run the isolated
diagnostics against the preserved exact-line panel, with a new output
directory each time:

```sh
.venv/bin/python -m tools.diagnose_bsp_regions \
  --prior-report generated/repair-experiments/adjoining-architecture/bsp-exact-line-probe-v2/report.json \
  --reconstruction-root generated/repair-experiments/adjoining-architecture/spatial-reconstruction-v1 \
  --candidate-decoder generated/repair-experiments/adjoining-architecture/bsp-exact-line-probe-v2/candidate_bsp_geometry.py \
  --out generated/repair-experiments/bsp-region-diagnostics-new
```

This runs the explicitly selected experimental decoder and archives its code,
source helpers, input WADs, hashes and dependency versions. Empty-sector
differences, topology changes, thin interior cracks and self-referencing
subsector owners stay visible. Union-grid sensitivity is diagnostic: it never
changes parser behavior, source eligibility or existing training caches.
Successful diagnosis does not certify geometry or gameplay. The input paths
above are local artifacts that git ignores.

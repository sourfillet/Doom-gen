# Doom-gen

Experimental Doom map generation with a raster diffusion model and a room-graph
generator. Both paths support optional property and placement GNNs. The graph
model learns room attributes and connections; room placement is procedural,
and optional polygon outlines come from retrieval. See
[REPO_REVIEW.md](REPO_REVIEW.md) for completed repairs and remaining limits.

The [ML-first migration plan](ML_FIRST_MIGRATION.md) proposes moving
placement, boundaries, openings and interior geometry into learned generation,
with deterministic compilation and physical validation retained. It inventories
the current learned/procedural split and defines implementation/evaluation gates;
the full spatial migration and its evaluation gates remain incomplete.

Initial implementation is tracked in
[ML_FIRST_PROGRESS.md](ML_FIRST_PROGRESS.md): a separate full-map supervision
contract, prediction-only decoder and provenance-preserving preparation tool.
The experimental shared model now emits complete variable-length map records,
and an opt-in source-tree backend compiles them through the existing checks.
The initial real-data experiment produced literal squares. A separate learned
control now reproduces a nonrectangular source map and completes native exits,
but every accepted control output is an exact training copy. Novel-layout
quality and held-out learning remain open. The bounded multi-map runs still
produce no valid geometry in their fixed panels; the progress document records
these failures and the next model work.

Source-tree commands are `python -m tools.train_geometry`,
`python -m tools.sample_geometry`, and `python -m tools.generate_geometry`.
The progress document provides preparation, bounded sampling, publication and
reproduction commands. Legacy defaults/checkpoints are unchanged.

For larger geometry experiments, `tools.train_geometry --token-codec compact`
compresses individual numeric/material fields losslessly using a training-only
vocabulary and byte escapes. It never supplies room/outline templates.
`--source-policy geometry` checks authored and serialized boundaries separately
from actor/progression eligibility; those sources are **not** positive gameplay
examples. `--min-outline-corners 6` declares an explicit source-selection filter.
Publication still runs the full checks; the generator's
`--reject-flat-rectangles` option also rejects a single flat rectangular footprint
rather than replacing it. These options do not establish useful new architecture.
The trainer reports list-stop recall separately from continuation accuracy;
`--stop-weight` is an experimental loss weight for rare list endings. It supplies
no target counts or stopping rules to inference.

`--record-memory --token-codec compact` enables experimental learned attention
and reference heads over records the model has already emitted. Sampling starts
with an empty bank and builds its own geometry history. `--sampling family`
balances training families before choosing maps; validation still reports a
complete map mean. Adding `--memory-read` enables a separately versioned learned
connection from the memory read into the reference query; earlier memory
checkpoints retain their original behavior. These are opt-in learning changes;
map quality remains
unestablished. [Progress and evidence](ML_FIRST_PROGRESS.md) track their tests
and bounded comparisons.

`--sampling coverage-family` first draws every admitted training map once in a
seeded shuffled pass, then uses family sampling. Each optimizer batch records
source paths and lengths in `sampling-trace-STARTSTEP.jsonl`; resumed training
writes a new segment. A96-step family run used only15/28 optimizer maps; complete
coverage used28/28 but still produced0/8 valid raw geometries. Coverage is a data
control, not a quality guarantee.

`--ending-loss-weight 0.1` optionally adds a class-balanced binary global-ending
loss to interleaved token NLL. In the96-step comparison it improves true-ending
recall from0/28 training and0/9 validation to28/28 and8/9, while predicting582
premature validation endings. All8 sampled maps still fail raw geometry. It adds
no inference threshold or record-count rule; default0 keeps the original loss.

`--spatial-pointers` adds learned candidate XY relations to completed vertex
anchors; it requires `--relative-pointers`. `--dependency-memory` additionally
learns sector-to-side and vertex/side-to-wall messages over emitted references.
It requires the spatial pointer path and interleaved tokens. Its separate v7
checkpoint/cache contract starts at the v6 function; its96-step trial still
produces0/8 valid raw geometries.
No angle, length, closure or routing rule is added. The matched v7 configuration is:

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

Use a compatible freshly verified preparation; source signatures are checked.
The experiment plan records its exact38-map manifest and source hashes. Adding
`--ending-loss-weight 0.1` reproduces the separately evaluated ending objective.
The geometry-only policy is for reconstruction research and does not certify
source gameplay eligibility. Existing publication validation remains required.

`--token-codec interleaved --record-memory --memory-read --record-anchors`
enables a separate learned typed-record stream. Source serialization places
dependencies near their first use; inference chooses its own record types,
coordinates, references and global ending. `--relative-pointers` adds learned
relative-index scores, with no preset endpoint preference. Source reindexing
retains every record and has an audit-only inverse permutation; index-sensitive
mechanics still need independent correspondence checks. The original field-
balanced objective does not support this new stream.

The multi-map trials still yield no valid geometry. A separate two-map capacity
control now produces nonrectangular, multi-level,14-sector geometry in7/8 final
seeds, all exact training wall-geometry copies. Compiled geometry checks pass;
static actor/progression checks fail. These are learning controls, not new
playable levels. [Current WADs and compiled diagnostics](generated/repair-experiments/learned-topology-checkpoints-v1/README.md)
and [every fixed sample](generated/repair-experiments/learned-topology-checkpoints-v1/gallery-v4/index.html)
are retained. New copy audits use `geometry_fingerprint_v2`, which ignores
unreferenced records when anchoring the geometry; the original fingerprint
remains available to reproduce older reports.

`--token-codec relative` losslessly models preceding-vertex offsets and floor-
relative ceilings. With memory enabled, `--record-anchors` adds learned use of
already completed references in the current record. `--loss-balance field`
optionally balances section/field/stop-class training losses (requires compact
or relative tokens and stop weight1). None supplies room shapes or target counts.
The current bounded trials still produce invalid geometry.

`train_geometry --checkpoint-wads DIR` emits a diagnostic WAD at each initial,
saved, validation and final checkpoint. `sample_geometry --raw-wads` preserves
one for every sampled attempt. Complete records retain exact raw authoring;
incomplete streams become trace-only WADs **without a map marker**. They are
not compiled/playable publication. See the [checkpoint file index](generated/repair-experiments/generation-checkpoints-v3/README.md)
and [progress/handoff](ML_FIRST_PROGRESS.md) for verdicts and reproducible runs.

## Install and test

Python 3.10 or newer:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,ml]'
python -m pytest -q
```

Use `.[dev]` for extraction and model-free diagnostics without torch. All twelve
console commands support `--help` without torch. The supplied repair environment
used Python 3.14.4, NumPy 2.5.1, Click 8.4.2, Shapely 2.1.2, NetworkX 3.6.1,
PyTorch 2.13.0, pytest 9.1.1, and setuptools 83.0.0. These describe the repair
environment. Shapely/GEOS is a core geometry dependency;
NetworkX is also a core dependency: model-free graph assembly uses its planar
embeddings to split high-degree rooms without making their connections non-planar.

`uv.lock` pins resolved packages, hashes, Python-version markers and optional
extras. Use `uv sync --locked --extra dev --extra ml` for a fresh locked training
environment, or omit `--extra ml` for the base tools. Isolated locked base/dev
environments were tested on Python 3.10, 3.12 and 3.14; OS/accelerator runtime coverage
is still limited. The lock selects newer packages than the original repair
environment, so do not switch dependencies in the middle of an exact-resume run.

## Prepare data

Start with classic Doom/Doom II binary maps. Other parsed formats do not imply
support for their mechanics in training or generation.

```sh
parse-wad /path/to/DOOM2.WAD --output dataset/maps
parse-wad /path/to/another.wad --output dataset/maps
prepare-corpus --maps-dir dataset/maps --out dataset/corpus.json --seed 7
```

Extraction uses `<full-wad-sha256>_<escaped-map-name>.json`, retaining the source
marker in metadata. Per-WAD `.manifests/` record paths, formats, and content
fingerprints. Repeated extraction is idempotent; incremental `--map MAP01`
preserves previous manifest entries. Use a fresh directory when migrating a
legacy corpus so old filenames do not duplicate newly extracted documents.

`prepare-corpus` groups duplicate content and source WADs into deterministic
train/validation/test splits, with duplicate provenance retained. Related WAD
releases need an explicit hash-to-family mapping through `--families`; hashes
cannot infer authorship. A corpus with fewer than three families is explicitly
an overfit/smoke experiment. Trainers can construct the same manifest themselves,
but passing a frozen `--manifest` makes comparisons easier to audit.

For a larger local WAD collection, audit current training eligibility before
spending time on a new training run:

```sh
python -m tools.audit_sources --wads wads --out generated/source-audit --workers 4 --timeout 30 --export-documents
python -m tools.audit_sources --wads wads --out generated/source-audit --workers 4 --timeout 30 --export-documents --resume
python -m tools.materialize_sources --audit generated/source-audit/report.json --out generated/complete-corpus
```

Run these source-tree tools from the checkout. The audit records mapless WADs,
extraction/target failures, per-map stages, timeouts, and complete/fragment ending
evidence. `--limit N` bounds a batch; resume processes remaining maps without
repeating completed results. A timeout is inconclusive. Resume requires matching
WAD content, relevant preprocessing dependencies, Python/library versions, and
time budget. Unrelated model/generation/diagnostic edits do not invalidate new
scoped source caches; whole-repository provenance is recorded separately.
Older whole-tree signatures retain their conservative compatibility checks.
Use a new directory
for a changed assessment. To retry only timeouts with a longer budget while
preserving an exported audit, create an independent verified branch:

```sh
python -m tools.retry_source_audit --audit generated/source-audit/report.json --wads wads --out generated/source-audit-retry --timeout 120 --workers 2
```

The branch copies verified terminal records/documents, retains parent provenance
and original budgets, then reassesses only timeouts. Source inventory, assessment
code/runtime and snapshot changes are rejected. It does not resume an interrupted
retry branch. Per-map records survive an
interruption even before the aggregate report is refreshed.
`--export-documents` saves assessed source JSON and exact targets during that
same pass. Materialization verifies and reuses these snapshots; without the
option it re-extracts and re-assesses. Changed snapshots are rejected, and the
aggregate audit stores hashes rather than repeating full target geometry.

The materializer requires a finished stable audit with matching code/runtime,
rechecks selected complete examples, and produces `maps/` and `corpus.json`.
Pass those paths to a trainer. Related releases can be grouped with its
`--families` option. Its `--include-fragments` option also exports assessed
incomplete examples; train these with `train-roomgen --include-incomplete` so
their endings stay censored. Use `--workers` and `--timeout` to bound parallel
re-extraction. `--validation-fraction` and `--test-fraction` configure the family
holdout (defaults 0.1 each); choose these before inspecting model results.
Individual records retain exact graphs and target sequences as
well as ending evidence. Complete here means a supported potential exit history;
combat, resources, moving-floor puzzles, and compact-stair exclusions remain
separate limitations. These filtered examples do not represent the distribution
of the whole WAD collection.

Prepared target snapshots can avoid repeating the expensive geometry/progression
pass during an unconditional room-model run:

```sh
python -m tools.train_prepared_roomgen --corpus generated/complete-corpus --out checkpoints/prepared --include-incomplete --steps 1000
```

The tool verifies the source documents, target snapshots, graph/sequence agreement,
assessment code and runtime. It reuses the existing model, censored-ending loss,
validation selection and resume loop, and binds preparation/tool signatures into
its checkpoints. Only complete examples enter validation or test-loss pools.
Use `--resume checkpoints/prepared/roomgen_001000.pt --steps 2000` to continue
with unchanged inputs. The regular `train-roomgen` command remains available
for raw source documents and tagged training.
`python -m tools.evaluate_prepared_roomgen --corpus generated/complete-corpus
--checkpoint checkpoints/prepared/best.pt --out generated/prepared-evaluation`
reuses the verified source targets in the ordinary evaluator. Sampling,
procedural/retrieval baselines, raw copy probes, and geometry checks remain
the same; reports record the cached inputs and exact tool sources.

Older JSON-derived geometry upgrades from raw elements on load. Re-extract WADs
to recover compiled BSP regions: old JSON does not contain the node/seg
lumps needed for this recovery. Derived geometry version 3 preserves authored
elements and raw-boundary diagnostics while storing validated coordinate regions.
Derived version 4 also handles pinched boundary cycles and preserves compiled
regions through rotations, mirrors, height jitter and JSON upgrades.
Derived version 5 permits validated compiled recovery around zero-length
authoring boundaries while retaining the raw warnings and checking references.
It also gives covered sector boundaries consistent geometric ownership.
Traversal version 3 lets valid starts on open boundaries bootstrap reachability;
physical clearance still rejects wall spawns and void interiors. Reassess source
eligibility and regenerate preprocessing caches when moving to these versions.
Traversal version 4 includes possible ordinary passage after consuming a type-39
teleporter in the optimistic geometry union. Final histories track actual
consumption; repeatable type 97 continues to suppress false front walk-through.
Traversal version 5 follows a drop across landing components too thin to stand
in (a strip or step below a ledge): while the player box still overlaps the
ledge, nearby boundaries are crossed at the ledge height, and the room graph
keeps a connection for such a crossing. Reassess sources and rebuild
preprocessing caches for this version.
Historical checkpoint evaluation requires its recorded preprocessing versions;
use archived source to reproduce old reference populations. Inference sampling
continues to support older checkpoint architectures.
Normal XNOD/ZNOD and extended/compressed XGLN/ZGLN, XGL2/ZGL2 and XGL3/ZGL3
nodes are supported, including bounded decompression, supplemental fixed-point
vertices and fractional GL3 partitions. Miniseg ownership propagates only across
miniseg partners; ambiguous ownership is rejected. Standalone glBSP v1/v2/v3/v5
GL_* blocks are associated with their map marker (including GL_LEVEL), with
checked windings, partners and optional map checksums. Original vertices remain
untouched. V2 is checked against real zdbsp; the installed older builder emits
nonstandard v5 records, which are explicitly rejected. Obsolete glBSP v4,
external GWA companions, slopes and 3D sectors remain unsupported. Multi-ring
geometry, rasters, graph inputs, and room targets have changed; regenerate caches
and retrain to benefit. Existing weights remain inference-compatible with their
older input/target shapes.

## Diagnose conversion before training

```sh
diagnose-maps --maps-dir dataset/maps --out generated/diagnostics --limit 8
generate-wad --from-json dataset/maps/<hash>_MAP01.json --out reconstructed.wad
```

Diagnostics save source/raster/decoded/assembled JSON, SVGs, and metrics, including
failures and excluded source geometry. `--limit 0` processes the whole corpus.
Source reconstruction preserves authored progression and reports losses; its
output is a diagnostic artifact. It does not receive the generated-map
playability guarantee.

The decoder defaults to factor-two cells, preserves small regions, and removes
only collinear contour vertices. Arbitrary contour smoothing is opt-in through
the Python API. If coarse sampling changes occupancy/void components or erases
a hole, decoding falls back to factor one while preserving world scale.
On three local Doom II maps at 128 pixels, occupancy IoU improved
from 0.865/0.796/0.826 to 0.957/0.904/0.941. At factor one it was 1.0 against the
source raster, which still loses source geometry at rasterization. Occupancy
overlap does not establish topology or gameplay fidelity.

## Train and sample

```sh
train-roomgen --maps-dir dataset/maps --manifest dataset/corpus.json --out checkpoints/rooms --steps 15000 --seed 7
generate-roomgen --checkpoint checkpoints/rooms/best.pt --seed 7 --max-rooms 20 --max-attempts 3 --out generated.wad
```

Without complete usable validation graphs, use the numbered final checkpoint instead of
`best.pt`. Version-3 room targets retain loop edges, connection widths, and
start/exit/key roles, and use calibrated world-unit size bins. Prefixes cut by
`--max-train-rooms` are censored rather than labeled as true endings. The stop
hazard uses a sequence likelihood to avoid overweighting short-map endings.
Training defaults to graphs with a supported reachable exit. Use
`--include-incomplete` to include source fragments with censored endings;
fragments never select `best.pt`. Training requires at least one supported
complete ending. Fragments are conservative reachable portions of maps whose
exit could not be certified, so they are mostly the larger maps. With the
default objective their continuations outnumber the complete maps' endings and
the learned stop hazard drifts toward zero as training continues. The opt-in
`--stop-targets complete` option trains the stop hazard on complete maps only;
fragments still train room and connection structure. With
`--stop-targets reachable-ends`, every map ends after its last reachable room,
so the hazard learns the size of reachable regions. `--complete-repeat N`
lists each complete training map N times in the sampled pool. These settings are
recorded in the checkpoint, and resume keeps them. In the v6 corpus most
complete training maps are one-room maps from a single WAD. As a result, the
`complete` mode learns to stop almost immediately (`stop-routing-training-v1/`). Segmentation version 2
separates disconnected sector islands
and represents thin door slabs as connections. Sampling masks physically
non-planar loops, and the assembler reserves routing lanes in a compact layout.
It tries up to sixteen deterministic room placements and twelve corridor edge
orders per placement. If edges remain unrouted, a planar fallback realizes
every physical connection with a visibility representation. Each room becomes
a bar of corridor junctions on one grid row, and each connection becomes one
straight corridor. Graphs that lane routing realizes are unchanged. Fallback maps use compact
spacing and report `router: planar_visibility`. Pass `planar_fallback=False`
to `roomgraph.assemble.assemble` for lane routing only. Attempts and unresolved
edges remain in generation reports. Each candidate is serialized, node-built when
available, and checked for target-port/BSP/playability errors before acceptance.
Compiled-map failures consume the same bounded attempt budget; accepted maps
retain their compiled bytes when combined. `--report` also saves per-attempt
evidence when every candidate fails. Raster generation similarly skips only
the rejected candidate, without adding unrequested sampling retries.
Finer interior geometry, directed connection types, and scripted progression
are still limited. `--shape-library library.json` retrieves merged outlines
and holes; interior sector height changes are reduced to one room height.

`generate-roomgen --geometry-profile varied` uses the routed assembler with
unequal row/column spacing, off-center corridor openings, and independently
clipped room corners. `--seed` also controls this procedural variation. It
retains the same routing budget, stairs, doors, and required-edge checks;
the model does not predict these outlines. The default `classic` profile keeps
the rectangular geometry for reproducible comparisons. The profile cannot be
combined with `--shape-library`, which uses a separate assembler.

`--geometry-profile compact` retains the varied corners and entrances while
reducing empty transit between rooms. It spaces occupied rows/columns by their
actual room extents and reserves a full stair run for each routed connection,
instead of multiplying one largest-room spacing across empty grid lanes.
Room dimensions, floor heights and requested connections remain unchanged.
Spacing decisions are recorded in `geometry_variation.spacing`. This is a
procedural density option; it does not retrain the model or solve missing routes.

`--geometry-profile adjoining` starts from compact routing and grows eligible
straight neighboring rooms toward a shared wall band. Open thresholds are 16
units deep, flat doors 32, and short corridors 64; larger stairs and routed
bends keep their existing realization. Growth is bounded and checked against
other rooms and transit space. This changes procedural room dimensions while
retaining requested topology and source floor heights. Requested/realized bounds
and accepted/rejected joins are in `geometry_variation.adjacency`; bounds use
the assembly coordinate frame, with any later `coordinate_translation` applied
separately. The default profiles retain their existing behavior.

The initial source evidence export preserves detailed geometry separately from
the current v3 training labels; the new playtest profile does not retrain the
model. The usable-opening audit and lossless authored reconstruction contract
preserve multiheight interiors, shared boundaries, actions, materials and
objects without treating source reconstruction as learned generation.

For the remaining BSP reconstruction discrepancies, run the isolated diagnostics
against the preserved exact-line panel (choose a new output directory each time):

```sh
.venv/bin/python -m tools.diagnose_bsp_regions \
  --prior-report generated/repair-experiments/adjoining-architecture/bsp-exact-line-probe-v2/report.json \
  --reconstruction-root generated/repair-experiments/adjoining-architecture/spatial-reconstruction-v1 \
  --candidate-decoder generated/repair-experiments/adjoining-architecture/bsp-exact-line-probe-v2/candidate_bsp_geometry.py \
  --out generated/repair-experiments/bsp-region-diagnostics-new
```

This executes the explicitly selected experimental decoder and archives its code,
source helpers, input WADs, hashes and dependency versions. Empty-sector differences,
topology changes, thin interior cracks and self-referencing subsector owners stay
visible. Union-grid sensitivity is diagnostic; it never changes parser behavior,
source eligibility or existing training caches. Successful diagnosis does not
certify geometry or gameplay. The historical `geometry-comparison-v1` panel uses
authored target reconstruction for its arm labeled `learned_geometry`; it does
not evaluate checkpoint inference or execute engine combat/progression checks.

The replacement spatial probe predicts every shared vertex and thing XY and
every sector floor/ceiling, including collapsed doors and remote controls:

```sh
.venv/bin/python -m cli.overfit_spatial \
  --out generated/repair-experiments/spatial-memorization-new --steps 1200 --seed 7
.venv/bin/python -m tools.compare_geometry_methods \
  --checkpoint generated/repair-experiments/spatial-memorization-new/model.pt \
  --out generated/repair-experiments/spatial-comparison-new
.venv/bin/python -m tools.render_geometry_comparison \
  --experiment generated/repair-experiments/spatial-comparison-new
```

The default source is frozen Dmonfear MAP19; `--source-json` selects another
extracted classic map. `--target-json` accepts the lossless
`doom_spatial_reconstruction_v1` contract. The earlier geometry_v1 target omits
collapsed doors/controls and is rejected for this prediction path. All outputs
use new directories; historical v1 checkpoints and panels remain intact.

This is a single-source memorization control with learned slot identities.
The source supplies authoring topology, assets, action/thing metadata and the
affine frame; no source XY or sector elevation is retained as inference fallback.
It does not generate a new scaffold from a RoomGraph or establish generalization.
Saved-checkpoint inference reconstructs the candidate through explicit integer
quantization, with separate basic, connected, exit/progression and serialized
checks. Invalid predictions and disconnected candidates remain rejected evidence.
The candidate WAD remains diagnostic, and engine execution is reported separately.
The comparison labels its differing conditioning/population policies; it is not
a matched gameplay experiment. Missing checkpoints skip the learned arm, and
missing node builders never count as successful compilation.
Frozen source/graph pairs must match exact document hashes and WAD/map identity;
shape-library inputs must match their training manifest. Changed authored targets,
corrupt or unversioned checkpoints, and mismatched source scaffolds are rejected.
The checkpoint supports inference; it does not claim exact training resume.

For a procedural gameplay baseline, `generate-roomgen --gameplay intro` adds
cover columns, small zombie/shotgunner/imp encounters, a directly reachable
shotgun/shell cache and armor, recovery health, and a distant usable exit.
Supply counts are reported; enemy drops and the initial pistol ammo are not
credited. Use `--style` with a style fitted on training maps and `--game` with
the target IWAD/profile. The profile excludes `--placement` to keep one owner
of the encounter budget. It defaults to `none`, preserving bare structural
comparisons. These additions are procedural and do not retrain the room model.
With complete assembly metadata, the intro objective ranks original requested
rooms, so extra connector sectors do not change which room becomes the goal.
Equal-distance ties prefer an exit role, then a terminal room and stable room ID.
The selected room/policy is reported in `objective.selection`; source or legacy
layouts without that metadata explicitly report their sector-hop fallback.
Actual player-region routing and progression checks remain separate.
When `--progression blue-key` is requested, a bounded preflight tries semantic
goal rooms against the actual key/gate planner before encounters are populated.
Rejected goals and the attempt budget appear in `objective.selection.key_preflight`.
The populated map must still pass final key-placement/history and publication
checks. This can select a different goal from an ordinary intro run.
For newly sampled graphs, this profile also constrains combat-room dimensions
to 384–1024 units and headroom to 128–256, retaining topology and floor heights.
Every changed size prediction is recorded in `gameplay_geometry`; the explicit
policy reduces the long empty transit caused by a single oversized room setting
global routing spacing. The default `none` profile retains raw sampled sizes.

```sh
generate-roomgen --checkpoint generated/repair-experiments/current-control/training/best.pt --style generated/repair-experiments/gameplay-intro/style.json --game generated/repair-experiments/gameplay-intro/doom2-profile.json --gameplay intro --geometry-profile varied --seed 1 --max-rooms 64 --max-attempts 1 --out generated/intro.wad --report generated/intro.json
```

Those example checkpoint/style paths are local repair-run artifacts.
For walkable height detail inside rooms, add `--interior-profile stepped` to
an intro run. It inserts bounded octagonal terraces with 16-unit risers and
64-unit treads, retains at least 64 units of headroom, and leaves the original
entrances and opening equipment clear. Every addition passes the existing
navigation checks; its floor sectors and positions are reported. This is
procedural architecture. The default `columns` profile retains the earlier
intro behavior. Learning authored interior structure remains separate work.

`--report` saves sizing policies, actual populations, supply budgets and failed
attempts to a fresh JSON file; existing reports are preserved. To compare
decoration on exactly the same existing layouts, run
`python -m tools.build_gameplay_demo --input layout.wad --style style.json --game game-profile.json --out generated/gameplay-demo`.
It preserves input/style/profile/source snapshots, reports each failed map,
and validates compiled output before publication. Materials use bounded,
coherent palettes and learned light levels; old style files without light
statistics retain their existing levels. Cover columns preserve player routes.
The ammo allowance and recovery schedule are baseline assumptions; human
encounter balance and finer learned interior architecture remain open.
Replay the same saved graphs with either profile, without resampling a model:

```sh
python -m tools.evaluate_routing --input generated/evaluation/report.json --out generated/varied-replay --geometry-profile varied --samples 16
python -m tools.render_evaluation generated/varied-replay
```

All four trainers (`train-layout`, `train-roomgen`, `train-properties`, and
`train-placement`) use family holdouts, training-only vocabularies, seeded
validation, per-step JSONL metrics, best-validation checkpoints, and complete
optimizer/RNG state. Layout rasters are cached on disk and loaded on demand.
Raster cache namespace v2 also records and checks the scoped dataset,
augmentation and rasterization source identity plus Python/NumPy/Shapely/GEOS
versions. Older or mismatched entries are preserved but rebuilt on demand;
the tensor contract remains raster version 5. A cache hit does not override
the trainer's independent checkpoint/resume compatibility checks.

```sh
train-roomgen --maps-dir dataset/maps --out checkpoints/rooms --resume checkpoints/rooms/roomgen_015000.pt --steps 20000
```

`--steps` is the total step count. Resume restores checkpoint settings and
rejects changed corpus documents, tag data, dependency versions, or
training/preprocessing source. Starting a fresh run in an existing checkpoint
directory is rejected; use a new directory or explicitly resume.
Old checkpoints without complete training state are inference-only.
New room-model training uses source-ending assessment `physical_exit_v3`:
an exit-room role also needs a physical player approach and a supported key/door
history, including supported one-shot type-39 teleport consumption. Active
backside/failed attempts, same-sector trigger planes and extension teleport
histories remain unsupported by this event solver. Incomplete examples are
excluded by default or have their stop label
censored with `--include-incomplete`. Evaluating historical checkpoints preserves
their original assessment version; `physical_exit_v2` retains its prior
teleport-reuse assumption for historical comparisons. Exact source references
also require matching preprocessing versions or the archived source.

Optional `--style`, `--properties`, and `--placement` decorate generated maps.
New property models predict wall textures and signed offsets per sidedef.
Placement models learn counts/types, local position distributions, and facing
for single-player skill 3. Both placers check actor dimensions, wall clearance,
reachability, and solid collisions, and report requested versus placed budgets.
Solid actors use their actual radii; non-solid pickups can be placed close to
walls. Inactive monsters and coop start markers do not block single-player spawns.
Broader difficulty/mode conditioning and resource/key progression remain open.
`--game` restricts assets and thing types to a supplied game profile/IWAD.

GitHub Actions runs locked base environments on Linux (Python 3.10/3.12/3.14),
Windows and macOS (3.12), plus a Linux CPU ML job. A manual workflow run also
enables ViZDoom, zdbsp and Chocolate Doom interaction checks with bundled Freedoom.
The workflow has not been executed on hosted runners during this repair session;
the local verification below does not establish the full OS/accelerator matrix.
Earlier locked Linux base/dev suites passed on Python 3.10, 3.12 and 3.14.
The current minimum-Python base check passes 384 tests on Python 3.10 with
NetworkX 3.4.2; optional ML/engine checks are skipped there. GPU/MPS and hosted
Windows/macOS executions remain unverified. See the review for current full-suite counts.

## Reproduce learning probes

```sh
OMP_NUM_THREADS=1 overfit-roomgen --out generated/toy-overfit --steps 800
OMP_NUM_THREADS=1 overfit-roomgen --source-json dataset/maps/<hash>_MAP01.json --out generated/map-overfit --steps 1200
OMP_NUM_THREADS=1 evaluate-roomgen --checkpoint checkpoints/rooms/best.pt --maps-dir dataset/maps --out generated/evaluation --samples 32 --write-maps 8
```

The overfit command saves targets, full checkpoints, fixed-seed comparisons,
and a WAD panel. Incomplete source fragments have censored endings by default;
prefix matches do not establish that the full source can be reproduced.
`--force-terminal` explicitly trains an artificial END for an encoded-target
memorization experiment, with that limitation recorded in the report. Both
overfitting and evaluation require empty output directories to preserve earlier
evidence. Evaluation uses the checkpoint's frozen corpus manifest and
reports complete and fragment held-out unconditional prefix losses separately,
graph distributions, nearest training descriptors, exact binned-graph matches,
and generation failures. Training references follow the checkpoint's source
selection settings, so excluded fragments do not enter copy/similarity checks.
Evaluation now includes matched-count procedural and nearest-size binned-graph
retrieval panels with the same realization budget. Exact wall/height copy probes
ignore XY translation, IDs, and texture changes; rotated, partial and approximate
copies remain outside that metric. `--no-baselines` skips the extra panels.
New reports bind checkpoint hashes and retain per-seed validation/publication
outcomes, distinguishing failed geometry from individually rejected compiled
maps and final batch publication failures. Pool disjoint panels from the
same checkpoint, corpus, settings and source with
`python -m tools.summarize_evaluations --out generated/seed-summary generated/evaluation-a/report.json generated/evaluation-b/report.json`.
The summary includes publication-rate Wilson intervals and paired discordant
seed counts. These describe sampling uncertainty for one fixed model and routing
budget; they do not measure training-run, dataset, gameplay or human judgments.
Historical reports without checkpoint hashes are explicitly marked as having
weaker identity evidence. Overlapping seeds or incompatible runs are rejected.
Create a portable comparison with per-seed SVG maps, rejected samples and world
dimensions using `python -m tools.render_evaluation generated/evaluation --limit 16`.
It writes `comparison.html` and SVG files beside the raw report; written maps
are labelled separately from any engine-completion evidence.
New evaluation reports include version-1 exact room-graph snapshots. Replay the
same sampled graphs without loading a checkpoint or re-encoding training maps:

```sh
python -m tools.evaluate_routing --input generated/evaluation/report.json --out generated/routing-replay --samples 16
python -m tools.render_evaluation generated/routing-replay --limit 16
```

Use a fresh output directory. Replays save input and tool snapshots, runtime
versions, failed seeds, routing details and separate validation/publication
results. Graph snapshots retain world coordinates, heights, roles, holes and
component IDs; they do not round-trip through lossy attribute bins.
Descriptor similarity cannot prove authored-geometry copying. Human review,
procedural/retrieval comparisons, and tag ablations are still needed.

Local single-example probes memorized a three-room graph and a 15-room MAP01
graph across 32 seeds each; each produced eight validated maps. These establish
that the graph learner and assembler can work together on a training example.
They do not establish generalization. Reports and panels from this session are
under `generated/repair-experiments/`; this directory is intentionally ignored
by git.

## Raster compatibility

New raster models learn world scale through a numeric condition and a
training-only scale prior. `generate-wad --world-span ...` overrides that prior.
Legacy checkpoints retain their numeric sector-color decoding and conditioner
shape. Raster version 2 introduced an order-independent boundary channel;
version 3 refreshes caches for BSP-derived geometry. Version 4 uses sixteen
channels: six explicit masks for specials 11/51/52/124/1/31 accompany the original
ten channels. Decoding restores supported exit and closed-door actions only
when the generated geometry supports them; it does not infer mechanics from
generic special density. Retrain raster models for these heads; old checkpoint
shapes and decoding versions remain supported for inference.
Version 5 adds four binary east/south face channels for boundaries and solid
walls, bringing the total to twenty. These eliminate ambiguous pixel-line
crossings that invented parallel cuts and hundreds of tiny sectors in dense
maps. Face targets decode at native pixel resolution, preserving world scale;
`--coarse-factor` still controls legacy decoding. Cached rasters and new training
use version 5; use `rasterize(..., raster_version=3/4)` to evaluate old targets.
Even perfect source rasters still lose some physical routes, so valid sampled
rasters and faithful full-map reconstruction remain separate research gates.

Guidance uses `unconditional_plus_scale_v1`: zero is unconditional, one is
ordinary conditional prediction, and values above one strengthen conditioning.
An old positive guidance value `g` corresponds to `g + 1`; old zero corresponds
to new one. Sampling steps must not exceed the checkpoint's diffusion timesteps.

## Load and check output

The verified engine target is ViZDoom 1.3.1 with its bundled Freedoom II IWAD.
GZDoom-family ports can rebuild the empty node/BLOCKMAP lumps on load:

```sh
gzdoom -iwad /path/to/DOOM2.WAD -file generated.wad -warp 1
python -m pip install -e '.[engine]'
DOOM_GEN_ENGINE_TESTS=1 xvfb-run -a python -m pytest tests/test_engine_integration.py -q
```

Repair-run artifacts in this checkout include thirteen local playtest panels in
[generated/playtest/README.md](generated/playtest/README.md). The newest compiled
panels include populated intro maps with verified pickups, combat and exits.
The newest, `neural-reachable-ends.wad` (`-warp 1` to `-warp 4`), comes from a
`reachable-ends` checkpoint trained on the v6 corpora, with 11–42 rooms per map.
Three of its maps use the planar large-graph router.
The latest single-map keyed variant keeps adjoining architecture and selects
an existing gated goal. Both IWAD checks collect the card, open/cross the gate,
clear combat, tour both terraces and exit:

```sh
gzdoom -iwad wads/DOOM2.WAD -file generated/playtest/neural-adjoining-blue-key.wad -warp 1 -skill 3
```

Use `neural-semantic-objectives.wad` for the ordinary three-map variant, with
`-warp 1`, `-warp 2` or `-warp 3`. These ignored local artifacts are not bundled
with the installed package.

Use an existing display instead of `xvfb-run` where appropriate. Engine tests
exercise stairs, doors from both sides, bidirectional teleports in small and
large rooms, exit activation, decorative-sector crossings, keys, and remote
door switches. Thirty maps across three generated panels also passed load/spawn
checks. A separate waypoint controller reached and activated all sixteen exits
in the two single-example overfit panels. Run a panel completion check with:

```sh
DOOM_GEN_ENGINE_TESTS=1 DOOM_GEN_ENGINE_PANEL=/absolute/path/to/sample-panel.wad DOOM_GEN_ENGINE_REPORT=/absolute/path/to/completion.json xvfb-run -a python -m pytest tests/test_engine_integration.py::test_generated_panel_can_reach_its_exits -q
```

The controller uses ordinary movement/use actions and a bounded route search;
failure does not prove a map impossible. These empty-map exit checks do not
measure combat, encounter balance, or human quality.
After bounded rerouting, the broader complete-source experiment completed all
ten written maps from its best/final panels. Write success rose from 2/8 to 3/8
and 5/8 to 7/8 on unchanged seeds/weights. Other seeds still fail corridor routing,
and the learner shows validation overfitting; see the review for counts and losses.

Generated maps require a valid start, supported potential traversal to playable geometry,
and a usable reachable exit after decoration and again after serialization.
The exit fallback prefers a predicted exit room, then distant reachable walls.
Closed doors and authored darkness survive default repair. Floor lowering,
brightening, and improvised teleport reconnection require explicit opt-in.
Supported vanilla doors open only after potentially reachable activation and
required skill-3 keys. Extraction keeps optimistic potential-access distances.
Final validation also searches location-consistent key/door histories, checking
physical pickup/use approaches and retaining irreversible branches. The default
budgets are 4,096 states and 256 geometry configurations; exhaustion is an explicit
failure. Doors remain open in this abstraction. Timed puzzles, consumed one-shot
door activators, failed/backside or extension teleport histories, lifts, scripts
and combat/resource progression remain unsupported.
Player configuration space checks narrow intra-sector passages, directed drops,
and exit approaches; this is not a full engine completion certificate. A drop
keeps the player at the ledge height until its box clears the ledge, so a low
ceiling just beyond a narrow trench blocks it, as in the engine. The player may
land in any lower sector under that ledge-height sweep, so a thin step that
touches the ledge does not hide the room beyond it. Walk-trigger lines passed
over while still supported by the ledge are not recorded as crossings.
Traversal metadata version 2 recomputes older cached distances, including the
correct skill-3 teleport population and flag-independent player-one start.

The writer rejects invalid references and out-of-range fields, translates XY
coordinates when the span fits without rescaling, and checks the actual written
geometry. Publication is atomic: failed validation or node building preserves
the previous output. Install `zdbsp` separately to build nodes. The real builder
receives `-R` to size the all-visible REJECT table after sector pruning; the
writer also supplies this table before node building. Empty REJECT data caused
invalid-table warnings in Chocolate Doom.
Prefer externally compiled nodes for engine checks. A measured ViZDoom 1.3.1
case completes all six maps but crashes in native shutdown when runtime node
building is used; an externally compiled copy completes and closes cleanly in
three repeated runs. Python's success status alone misses that backend failure.
Preserve the original and build a separate copy with
`zdbsp input.wad -R -o compiled.wad`. This is a verified workaround on that panel,
not a diagnosis of the native engine bug. Logs and both WADs are retained under
`generated/repair-experiments/fragment-seed-panels/`.
All supplied binary BLOCKMAPs now receive structural/index checks for ZDoom
publication too. The current retrieval control exposed wrapped unsigned
16-bit pointers from a large zdbsp table; those malformed compiled outputs are
rejected atomically. Empty modern-port tables remain a runtime-build fallback.
Engine tests isolate and reject native crash logs even when Python succeeds.
The pipeline was verified with the real builder, and Chocolate Doom 3.0.0 rendered and ran
movement demos on node-built stair/door/compact-teleport fixtures. With `zdbsp`
and `chocolate-doom` on PATH, run these optional checks using:

```sh
DOOM_GEN_PORT_TESTS=1 python -m pytest tests/test_port_integration.py -q
```

The port test uses ViZDoom's Freedoom II by default, or `DOOM_GEN_IWAD` when set.
All 23 maps from the four repair panels also passed these rendered movement
checks; logs and statistics are under `generated/repair-experiments/port-compatibility/`.
After rerouting, 9/10 maps pass Chocolate Doom movement checks. One larger map
has a 215-by-190-cell BLOCKMAP whose offsets exceed Chocolate Doom's signed
16-bit range and crashes there; it completes in ViZDoom. The report is
`generated/repair-experiments/port-compatibility-routed/report.json`.
Broad vanilla engine limits, other ports, and full Chocolate Doom progression
remain unverified.
Both generation commands accept `--target-port vanilla` to reject known static
format/index/BLOCKMAP incompatibilities before publication. It requires zdbsp
and preserves an existing output on failure. View-dependent vanilla rendering
limits and gameplay still require engine tests. The default target is `zdoom`.


Bundled graphics use strict Doom names (ASCII, at most eight bytes). Raw flats
and patches cannot use map markers or engine/namespace lump names. For vanilla
output with `generate-wad --assets ...`, pass `--game path/to/IWAD.wad` so the
bundle can preserve the complete base texture tables and patch indices. A saved
vocabulary-only game profile does not contain those tables. In Python, bind
`bundle = bundle.with_base_wad(iwad_path)` before publication. This copies table
metadata, without bundling the base images. Vanilla new flats need unique names
and exactly 4096 indexed pixels; patch replacements cannot shadow base flats.
Patch-only bundles retain the base texture tables. Modern-port standalone
resource definitions remain available through the low-level writer.

Both generators also accept `--sources source.wad,another.wad` (or a directory)
with `--game path/to/IWAD.wad` to extract missing referenced textures/flats before
game enforcement substitutes them. Directory entries are sorted; the first
indexed definition of a name wins. Changed source patches receive private names
so existing base textures retain their original pixels. Incomplete textures are
left for ordinary fallback enforcement. Generation results include source WAD
hashes, resolved names and output asset hashes under `asset_resolution`.

Assets are embedded by default. Add `--companion-textures resources.wad` to write
a separate resource PWAD, and load it alongside the generated map. Both outputs
are validated before either destination is replaced; an ordinary publication
failure preserves existing files. The two file replacements cannot be atomic
across a process or power failure. No companion file is reported when there are
no assets to write. Source WAD graphics retain their original licensing; the
repository does not include extracted commercial graphics.
Custom palettes, animation/switch tables and source-port scripts are not
transferred by this image extractor.


For a required item/gate objective, `generate-roomgen --gameplay intro
--progression blue-key` preflights bounded semantic goal candidates against
existing mandatory doors, puts a blue card in a reachable earlier room, and validates the complete
pickup/door/exit history. `--interior-profile stepped` can be combined with it.
Layouts without a real mandatory gate are rejected within the usual retry
budget. Room-role key predictions are preferred where usable; general lock
relationships and multi-key/script progression are still open. The verified
current one-map variant is `generated/playtest/neural-adjoining-blue-key.wad`
(`-warp 1`), with eight enemies and two terraces, directly checked in both IWADs.
The earlier `neural-blue-key.wad` remains unchanged. Screenshots:
`generated/playtest/adjoining-keyed-comparison/index.html`.

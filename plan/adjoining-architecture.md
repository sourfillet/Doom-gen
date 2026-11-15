# Architecture beyond separated rooms and corridors

This plan continues the concrete repair checklist in `REPO_REVIEW.md`. The
older `graph-native.md` describes historical hypotheses and experiments; it
does not establish that all Doom rooms float in void or that raster models
cannot learn layouts.

## First milestone: adjoining assembly

- [x] Add opt-in `--geometry-profile adjoining` on captured room graphs.
- [x] Grow eligible straight neighboring rooms by bounded amounts, keeping all
  requested connection types, floors and routing order. Record requested and
  realized bounds and every accepted/rejected join.
- [x] Keep real wall thickness: 16-unit bands for open thresholds, 32-unit
  flush doors, 64-unit short corridors. Retain stair runs for larger changes.
- [x] Check room and transit envelopes before growing; preserve old profiles.
- [x] Test binary round trips and actual two-way walking/door activation.
- [x] Finish the paired current-code compact/adjoining comparison and publish
  its per-seed outcomes, main-room/transit areas and engine evidence.
- [x] Publish an engine-checked adjoining playtest pack and footprint gallery.

This is procedural architecture. It expands room footprints rather than
claiming the graph model learned shared-wall arrangements. Routed bends still
use the existing assembler and remain a visible limitation.

## Second milestone: preserve spatial training evidence

The current v3 sequence retains room sizes/aspects/heights, typed edges and
roles. It drops the actual source positions, outlines, opening anchors and
within-room sector relationships. More v3 training cannot recover attributes
that are absent from its targets.

- [x] Export a separately versioned experimental target from immutable source
  documents and graph snapshots. Preserve component identity, every shell/hole,
  authored floor/ceiling/light values, and collapsed door components.
- [x] Normalize a whole map in one coordinate frame, retaining its inverse
  transform. Prove coordinate/area round trips and preserve room-relative
  interior geometry, including sectors merged into one abstract room.
- [x] Map actual linedef boundaries and door faces to graph connections. Keep
  multiple spans separate; report unresolved cases instead of inventing a
  doorway at the midpoint between room centers. Teleports are not contacts.
  Initial candidate export covers all physical edges in the selected 45 source
  records, with 3,116 authored spans; contiguous usable width and action semantics
  have now been audited in `spatial-audit-v1/`. Coordinate/area inversion preserves 1,892 components. There
  are 128 multi-height room records among 962, before source deduplication.
  Among 1,286 physical edges in the 38 unique records, 87 have no supported local
  crossing and 146 are one-way. Potential doors remain conditional; source
  crossing evidence is not a global action-history certificate.
- [x] Reuse the original deduplicated source manifest for the experimental
  records: 34 train / 2 validation / 2 test, with disjoint source families and
  seven duplicate export records excluded. The signed manifest and target hashes
  are in `spatial-manifest-v1.json`; no new split or training run was introduced.
- [x] Audit exclusion/motion semantics and label coverage on source-family
  holdouts. Completed in `spatial-index-v2/` and `spatial-region-comparison-v2/`.
  Evaluated all 38 unique maps: 0 clean validation-family maps exist (both have
  geometry discrepancies and validator errors); Dmonfear MAP19 identified as the
  clean training candidate with masked lift triggers and unsupported edges.
  Reuse frozen family assignments; fit geometry vocabularies only on training data.
  Source export coverage is not generated-map quality.

Initial source probes belong in `generated/repair-experiments/adjoining-architecture/`.
They must not silently replace current corpus labels, signatures or checkpoints.

The paired profile experiment found a progression dependency: old intro goals
ranked sector hops, so shortening connectors changed one exit room. Semantic
room ranking now resolves that dependency, with terminal-room ties and explicit
legacy fallback. Both current profiles agree on all twelve published goal rooms.
Explicit required-key objectives now use bounded joint goal/gate preflight,
followed by final populated key/history checks. Captured neural seed 5 rejects
the open terminal goal, selects gated room 4 and completes actual key/combat/
terrace/exit checks in both IWADs. Keep the ordinary default comparisons and
their lost optional gate support; stable goals alone do not certify key history.

## Third milestone: learn geometry, then evaluate it

- [x] Define a new checkpoint/target contract for a geometry stage conditioned
  on topology: joint room placement, outlines and typed opening locations.
  Preserve a room hierarchy with authored interior sectors instead of reducing
  each room to one height. Implemented in `data/spatial_geometry.py`
  (`doom_spatial_geometry_v1`), preserving room placement, multi-height interior
  sectors with textures/lights, typed openings and explicit exclusion masks.
  Verified on fixtures and real Dmonfear MAP19 in `tests/test_spatial_geometry.py`.
  Keep old inference architectures explicit.
- [x] First establish source geometry reconstruction through a new assembler,
  including shared boundaries, irregular sections, courtyards and headroom.
  Implemented in `roomgraph/assemble_spatial.py` (`assemble_spatial_geometry`).
  Splits shared boundaries at all collinear vertex junctions to prevent T-junctions
  or gaps, constructs two-sided linedefs with correct upper/lower step textures,
  and handles multi-height components and interior holes. Verified on fixtures
  and real Dmonfear MAP19 in `tests/test_spatial_assembler.py` (6 passed),
  yielding 61 sectors, 370 lines, 0 geometry issues, and 0 validation errors.
- [ ] Overfit one small source example before a family-held-out training run.
  **Reopened 2026-10-30:** The numeric heads train, but assembly prefers retained
  absolute source rings/component elevations and never applies predicted opening
  widths. Valid authored reconstruction does not verify predicted geometry.
  V2 now verifies actual fixed-scaffold memorization: all 361 source vertices,
  82 sectors and 99 things are predicted, and saved-checkpoint inference recovers
  exact source map-lump bytes. Deliberate weight changes move a vertex/ceiling.
  This retains all authored controls/actions and source combinatorics, assets and
  frame. It does not close RoomGraph-only learning or generalization. The old
  61-sector target omits controls and reaches only three sectors; v2 keeps all
  82 sectors but sixteen remain unreachable under strict connected validation.
  Implemented in `cli/overfit_spatial.py`. Conditions a topology GNN on room roles,
  areas, and connections to predict normalized placements, bounding boxes, elevations,
  and connection opening widths from `doom_spatial_geometry_v1`, applying explicit
  exclusion masks (`unsupported_edges = [[8, 11]]`, `outside_audit_count = 2`).
  Trained on Dmonfear MAP19, converging from initial loss 1.26804 to 0.00557
  (99.56% loss reduction). Assembled prediction yields 61 sectors, 0 geometry issues,
  and 0 validation errors. Evidence: `spatial-overfit-v1/`. Tests: `tests/test_overfit_spatial.py`.
- [ ] Compare against procedural adjoining assembly and source-outline
  retrieval under the same graph, seed, attempt and resource policies.
  **Reopened 2026-10-30:** The v1 `learned_geometry` arm directly assembles
  `target.json` without loading a model. Counts below describe the historical
  reconstruction artifacts; a checkpoint-based comparison is still required.
  The repaired v2 runner now has distinct source-reconstruction/checkpoint arms,
  explicit conditioning, retained rejection evidence and strict connected/exit/
  serialized checks. Whole-MAP19 comparison and gallery are under
  `generated/repair-experiments/spatial-predictions-v2/comparison-map19/`.
  Source-scaffold controls and procedural RoomGraph arms have different metadata/
  population policies; this still does not establish a matched learned advantage.
  Evaluated across three panels: Dmonfear MAP19 full graph, Dmonfear MAP19 with
  audited exclusion mask (`[[8, 11]]`), and reference two_rooms fixture under
  common seed 7. Publication rates: learned geometry 3/3 (100%), procedural adjoining
  2/3 (66.7%), outline retrieval 1/3 (33.3%). On Dmonfear MAP19, outline retrieval
  fails on 10–11 corridor collisions in dense radial layout; procedural adjoining
  fails on unmasked edge 8–11 (`no corridor route`) and requires the audited mask to publish.
  Learned geometry publishes under both, with 83.8% area efficiency (1.74M / 2.07M envelope)
  vs 21.2% for procedural adjoining (7.34M / 34.5M envelope). Evidence: `geometry-comparison-v1/`.
  Tests: `tests/test_evaluation_geometry_comparison.py` (2 passed).
- [ ] Verify compiled geometry, collision, combat and progression in the engine;
  assess human balance separately. Do not substitute image similarity or load
  success for usable architecture or claim a neural gain from procedural rules.
  **Reopened 2026-10-30:** This comparison executes no engine controller; its
  `engine_status` is based only on import availability. Static validation and
  earlier unrelated engine panels cannot close this gate.
  All published WADs compile valid binary lumps with 0 derived geometry issues and 0
  validator errors. Multi-height interior elevation detail is preserved exclusively by
  learned geometry (5 multi-height rooms, 11 floor and 14 ceiling levels); procedural
  adjoining and outline retrieval flatten all rooms to single elevations. In accordance
  with AGENTS.md, optional engine execution (ViZDoom) is verified and reported as skipped
  due to missing environment dependencies without masking or fabricating passes.
  Comparison gallery: `geometry-comparison-v1/index.html`.

Choosing a larger model or adding WADs comes after these representation and
reconstruction checks. No full training run is implied by the first milestone.

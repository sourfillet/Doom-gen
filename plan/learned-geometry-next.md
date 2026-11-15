# Next learned-geometry experiments

This is an implementation handoff, not a claim that the current generator makes
useful new maps. Read `WORK_LOG.md` for active jobs and current verification.
Do not repeat old bounded runs, relabel preparations, or replace failed seeds.

## Evidence that changes the next implementation choice

- Dependency-first-use v4 makes references much closer, but its96-step rollout
  panel remains0/8 raw-valid. V5 relative pointer scores largely learn recency.
- V6 coordinate relation scores and full initial optimizer coverage do not fix
  geometry or true global stopping. The coverage model gets0/28 train and0/9
  validation true EOS argmax at96; mean true EOS probability.02691/.02647.
- V7 learned parent messages improve conditional side accuracy to1906/3397
  (coverage v6:1766), but both miss every true ending. Its final8 outputs are
  rejected or budget traces. This is no novel-map result.
- Strong causal serialization control gets3397/3397 side references, not merely
  the1755 achieved by optional-NULL/required-latest. Every source side in this
  validation panel is emitted immediately before its first wall use. Required
  right=first new side since last completed wall; optional left=last new side
  when two were emitted, otherwise NULL. A separate v1 endpoint control reaches
  785/1703; latest-only v2 reaches931/1703. These are validation baselines only;
  do not implement their geometric/serialization assumptions as inference rules.
- Training has3725 vertices and1773 distinct relative XY pairs. Each scalar mode
  is0 but the joint(0,0) pair occurs only3 times. Validation has8/1736 zero pairs.
  Current model modes produce many coincident points. Autoregressive Y already
  can condition on X; this is an observed fit gap, not proof of independence.
- A train-fit128-pair vocabulary covers1751/3725 training and765/1736 validation
  vertices. Five-token rare-pair escapes would *lengthen* coordinate streams
  (11621 vs9221 train XY tokens;5620 vs4380 validation).256 pairs still lengthen
  both. Do not assume a large joint codebook is automatically better.

Exact reports and helpers are under
`generated/repair-experiments/learned-topology-checkpoints-v1/`:
`serialization-controls.json`, `reference-controls.json`,
`sampling-coverage-formal.json`, `coordinate-distribution.json`.
Teacher, fit-gap and parity reports stay inside their respective experiments.

## Finish the current ending-loss comparison first

`--ending-loss-weight=.1` retains token/type/value NLL and adds per-map equal
means for binary true-ending and continuing-header loss. It shares current
logits and adds no parameters, threshold or count input. Compare both true and
premature endings; a higher ending recall alone may shorten broken outputs.
The completed bounded run is `ending-auxiliary-geometry-v1/`, with96 updates and the same
v7 architecture, initialization and every optimizer source draw. Do not extend
its budget or compare total weighted loss directly against another objective.
Use unweighted fit-gap, whole reference fields and actual fixed raw panels.

These helpers have already completed and their reports exist. Do not rerun
them over those outputs; inspect the preserved results:

```sh
.venv/bin/python generated/repair-experiments/ending-auxiliary-geometry-v1/diagnose.py --arms ending --out teacher-diagnostic.json
.venv/bin/python generated/repair-experiments/ending-auxiliary-geometry-v1/parity.py
.venv/bin/python generated/repair-experiments/ending-auxiliary-geometry-v1/fit_gap.py
```

Check files and current checkpoint first: do not run over an existing report.
The full suite and source compatibility verification have separate logs.

Completed ending result:28/28 train,8/9 validation true endings, but1012
training and582/6849 eligible validation premature stops. Final8/8 streams
complete;0/8 raw/static-valid. Macro validation terminal rank AUC improves from
v7.95109 to.97012, while mean true EOS probability rises.03360->.74723. Only1/9
terminal probabilities beat every continuing header in their own map. This
captures better discrimination plus a large stop-prior shift, not usable global
stopping or generated geometry. Do not simply increase the ending weight.

## First representation change: encode references relative to the emitted bank (COMPLETED — `relative-reference-geometry-v8/`)

Keep every field, reference, action, actor and record. A referenced ID is
currently a numeric absolute-ID token even though the pointer head scores its
relative age. Propose a separate, lossless token schema where a non-null
reference encodes `emitted_count(kind)-1-ID`. This is a coding transform, not a
preference for recent or nearest endpoints: every emitted ID must remain legal.

**Completed status (`2026-11-10`):**
- Implemented in `data/geometry_interleaved.py`, `data/geometry_codec.py`,
  `data/geometry_memory.py`, `models/geometry_memory.py`, `generation/geometry_model.py`,
  `tools/train_geometry.py`, and tested in `tests/test_geometry_{interleaved,anchors,dependencies,training}.py`
  (`66 passed in 87.61s`, `generated/repair-experiments/relative-reference-geometry-v8/focused-v3.stdout.txt`).
- Completed 96-step matched run and diagnostics in `generated/repair-experiments/relative-reference-geometry-v8/`:
  - Validation teacher whole-field accuracy at step 96 (`teacher-diagnostic.json`):
    `vertices = 1,547 / 3,406` (`45.4%`, beating `latest_id_correct = 1,406 / 3,406`),
    `sectors = 957 / 2,480` (`38.6%`), `sidedefs = 2,653 / 3,397` (`78.1%`,
    `linedefs.left_sidedef = 1,703 / 1,703 = 100.0%`).
  - Trained-prefix parity (`trained-prefix-parity.json`): `max_abs_logit_error = 7.62939453125e-06` over 596 steps.
  - Fit-gap (`fit-gap.json`): step 0->96 unweighted decision NLL `4.63985 -> 1.74476` (train),
    `4.70646 -> 2.48546` (validation); true EOS argmax `0/28` train, `0/9` validation (`0` premature).
  - Free-running rollout (`panel-096/report.json`): `7/8` seeds emitted `END_MAP` before 12,000
    tokens (`0/8` raw-valid due to `linedef 0 collapses to zero length after integer rounding`).

Implementation boundaries:

1. Add a separately dispatched codec/prefix schema. Fit integer/name vocabulary
   only on the actual training event stream, including reference ages at that
   event, not full-map final counts. Source first-use/inverse-ID audits stay in
   their current actual-ID semantics. Never give an inference source count.
2. Decode an emitted age using the current causal kind count before the base
   `_finish`. The memory bank and completed/current record references remain
   actual IDs. Keep null separate, handle two-byte escapes, and allow non-null
   vertex/sector ID65535 when their count is65536. Optional-side65535 is NULL.
3. Version the memory/query contract: the low byte in a pending reference query
   now describes age. Candidate byte filtering and first/high-token projection
   must encode `count-1-candidate_ID`, not candidate_ID. Masks must permit every
   prior ID and reject every future ID, with no self-edge/closure rule added.
4. Keep old codecs/checkpoints loadable and strictly reject mismatched codec/
   memory contracts. Add exact byte/source inverse tests, counts crossing255,
   low-byte and null cases, full/incremental active cache equivalence, zero
   future gradients, ragged exact resume and source-free raw export tests.
5. Rebuild source preparation when scoped data/generation bytes change. Audit
   equality of original target supervision/actions/heights/context, but do not
   treat equality as compatibility or relabel old reports. Train from scratch
   when code/helper signatures change.

This experiment changes token meaning and vocabulary. Do not promise identical
untrained streams or compare token NLL across codecs as equivalent tasks. Keep
original source cohort/families/splits, report changed sequence lengths and
actual optimizer token exposure, then compare whole fields and raw geometry.
One seed/small budget gives a diagnostic only, not a generalization result.

## Then test direct causal numeric conditioning within a record (COMPLETED 2026-11-14, NULL RESULT — `record-values-geometry-v9/`)

Before fitting a512+ joint-pair codebook, test a zero-initialized learned skip
from already completed scalar fields into the next-token/query head. Current
anchors expose completed *references*, not completed coordinate values as an
explicit numeric input. This could make the X-to-Y dependency easier to fit
without a vocabulary full of rare pair escapes.

**Current status (`2026-11-10`):**
- Code and focused tests (`73 passed in 92.75s`, `generated/repair-experiments/record-values-geometry-v9/focused.stdout.txt`;
  initial `isinstance(codec, CompactCodec)` subclass test failure retained in `focused-initial.stdout.txt`)
  are complete in `data/geometry_memory.py`, `models/geometry_memory.py`,
  `models/geometry.py`, `generation/geometry_model.py`, `tools/train_geometry.py`,
  and `tests/test_geometry_{dependencies,training}.py`.
- Frozen `source/` snapshot and fresh 38-source `prepared/` targets are built in
  `generated/repair-experiments/record-values-geometry-v9/` (partial artifacts from
  an interrupted run stopped at step 48 by a WSL/server restart are retained in
  `interrupted-restart-attempt-1/`).
- Completed 2026-11-14. Conditional metrics match v8 within a few fields
  (vertex references 1,517 vs 1,547 of 3,406; `vertices.y` 428 vs 428 of 1,736;
  validation NLL 2.48706 vs 2.48546); 8/8 rollouts end, 0/8 raw-valid.
  `rollout-degeneracy.json`: 649/2,453 sampled walls have `v1 == v2` and
  566/2,707 vertices repeat the previous XY; training has 0/4,014 and 3/3,725.
  The skip does not, at 96 steps and one seed, make zero offsets rarer. Explain
  the repeated endpoint choice before the fit curve; add no self-edge mask.

Define a new typed prefix contract with seven numeric values and presence bits;
include a slot only after its complete scalar field has been consumed. Never
expose an unknown Y, a half-read byte field, future reference or target count.
Specify world vs relative field values exactly and test wraparound. Existing
GRU prefix/type/count inputs and reference/parent memory remain available.
Projection weights start at zero so common initial tensors/function/RNG match.
Test an active known-X effect on Y, missing-field exclusion, full/incremental
prefix parity, future value noninterference/zero gradients and exact resume.
A model must still learn zero-offset rarity. Do not add zero-offset masks,
minimum wall lengths, polygon closure or positive-height enforcement; legitimate
source duplicates, unused records and closed control sectors must remain legal.

## Fit curve and eligibility gates

After a representation probe, predeclare a bounded *fitting* curve with complete
source coverage and more than one training seed, using a fresh experiment and
explicit compute/attempt budgets. Do not silently extend a failed96-step run.
The512-step two-map hidden96 control already reproduces14-sector training wall
geometry; it is a capacity/memorization result, not full-cohort architecture.
Measure train/validation separately, rare endings, premature endings, new vs
reused references, whole XY pairs, source-copy checks and actual raw validity.
Keep the test split untouched for tuning. Generate WADs at all milestones.

Source gameplay eligibility remains a separate open gate. The geometry-only
28/9/1 cohort has retained unsupported/unreachable mechanics and actors; it is
not a certified playable-map corpus. Do not weaken static publication checks to
make an ML panel look successful. Recheck serialized/compiled physical/action
correspondence on any valid geometry; native combat/objective/exit completion
and human quality are separate evidence. No procedural fallback should be
reported as learned generation.

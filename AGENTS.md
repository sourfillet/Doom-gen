# Repository instructions for coding agents

These instructions apply throughout this repository. Read this file before
working here, then read the **Current resume checkpoint** in `WORK_LOG.md` and
the relevant items in `REPO_REVIEW.md`. User instructions take precedence.
Keep this file focused on lasting conventions; put session state and results
in the work log.

## Purpose and architecture

Doom-gen learns from Doom WAD maps and generates new PWADs. Established layout
paths are raster diffusion followed by vectorization and a room-graph RNN
followed by procedural routing/assembly. An experimental full-map geometry RNN
emits variable-length authoring records without procedural layout assembly;
its raw diagnostics are not automatically playable maps. Geometry variation, retrieved
outlines, corpus-frequency styling, introductory encounters, terraces and the
bounded blue-key objective are procedural additions, not learned architecture.

- `wad_parser/`: binary/UDMF extraction, geometry, BSP regions and traversal.
- `data/`: augmentation, raster/graph targets, corpora and caches.
- `models/`: learning, conditioning, checkpoints and reproducible training.
- `roomgraph/`: segmentation, routing, geometry and requested connections.
- `generation/`: sampling, validation, decoration and publication pipelines.
- `wad_writer/`: serialization, bundled resources and port compatibility.
- `cli/`, `tools/`: installed commands and reproducible evaluation utilities.
- `tests/`: regressions and optional ordinary-input engine controllers.

## Preserve ongoing work and evidence

- Expect an extensively modified working tree. Inspect changes before editing;
  do not reset, clean, overwrite another agent's work or create commits unless
  the user asks. Do not infer authorship from `git diff` alone.
- Log each newly discovered issue and completed item in `WORK_LOG.md` as it
  happens. Record evidence, validation, limitations and the next action. Update
  `REPO_REVIEW.md` checkboxes when the corresponding work is actually verified.
- Before handing off, update the current resume checkpoint. Distinguish
  completed, failed, skipped, pending and blocked checks, with exact commands
  or artifact paths. Never present an older test count as current verification.
- Keep old checkpoints, source snapshots, reports, seeds, failure cases and
  playtest WADs intact. Use a fresh experiment directory under
  `generated/repair-experiments/`; generated artifacts are normally ignored.
- Preserve input hashes, code/helper snapshots, preprocessing contracts,
  dependency versions, seeds and attempt budgets for experiments. Retain
  rejected candidates. Do not silently resample or combine unlike panels.
- Source-preprocessing and training fingerprints are compatibility checks.
  When their inputs change, audit/rebuild affected caches and corpora through
  the supported tools. Never relabel old snapshots or bypass a signature check
  to claim compatibility or exact resume.

## Implementation contracts

- Keep parsing and validation diagnostic and non-mutating where promised.
  Preserve raw authoring data and report unsupported mechanics explicitly.
- Preserve every requested room connection, including loops. Reject unresolved
  routes instead of silently dropping edges or substituting teleports.
- Use player/actor clearance and supported action history for reachability;
  center-point containment or a connected sector graph is insufficient.
- Population must respect target-IWAD assets, actor dimensions, skill flags,
  starting equipment and explicit resource budgets. Do not place arbitrary keys
  as a substitute for required-progression planning.
- Validate serialized and node-built output as well as in-memory documents.
  Publish atomically, preserve an existing destination on failure, and keep
  accepted per-map compiled bytes when combining a batch.
- Preserve default profiles and checkpoint compatibility unless deliberately
  changing their contract. Make experimental procedural behavior opt-in.
- Keep Torch optional for parsing, geometry, base-package imports and CLI help.
  Avoid eager ML imports in those paths. Supported Python starts at 3.10;
  dependencies, extras and console commands are defined in `pyproject.toml`.
- Spatial evidence/audits and authored reconstruction targets are experimental
  contracts, separate from v3 room-graph labels/checkpoints. Retain full source
  controls/actions and every interior height. A candidate boundary, an optimistic
  door union and a lossless authored-byte round trip each need separate physical,
  history and compiled-geometry checks; none is a gameplay certificate.
- Generated intro goals use original semantic rooms when complete assembly
  metadata exists. Added connector sectors must not change semantic goal ranks.
  Keep legacy fallbacks explicit and verify physical routes independently.
  An explicitly requested key objective may constrain the goal through bounded
  preflight. Retain rejected goals/budgets and validate the final populated key
  placement/history independently of its pre-population trial.

## Verification

Use the repository virtual environment where available:

```sh
.venv/bin/python -m pytest -q
```

Run focused regressions first, then the checks appropriate to the change. The
default suite skips optional engine/port checks. Do not install dependencies or
start long training runs merely to claim broader coverage.

- Real engine tests require the optional `engine` dependency, a working display
  or Xvfb, and `DOOM_GEN_ENGINE_TESTS=1`.
- Port tests require `DOOM_GEN_PORT_TESTS=1` and the necessary tools on `PATH`
  (notably zdbsp and Chocolate Doom). Engine panel/demo tests also require the
  artifact environment variables described in the tests and work log.
- Historical `/tmp` engine/tool installations in the log may not survive a new
  session. Check availability; never treat a skipped test as a pass.
- For packaging changes, build a wheel and run `tools/check_install.py` against
  its installed package outside the checkout. This checks the console commands
  and base imports with Torch deliberately unavailable.
- Distinguish static validation, successful serialization, engine load/movement,
  actual combat/objective/exit completion, and human quality/balance. None
  automatically establishes the next. Inspect native crash logs and shutdown.
- A bot timeout is inconclusive about human playability. Preserve failed
  traces when fixing the controller and identify the changed verifier.
- Do not claim better learned layouts from procedural changes or one sampling
  panel. Use matched baselines and state the limits of sampling/training seeds.

## Assets and documentation

Do not commit or redistribute commercial IWADs, copied stock graphics, large
training corpora or generated checkpoints. Use locally available IWADs only
for authorized testing; keep resource provenance clear. Keep `README.md` a
short overview with a quick start. Document user-visible options and details
in the relevant `docs/` page, and reproducible playtest commands in
`generated/playtest/README.md` when applicable.

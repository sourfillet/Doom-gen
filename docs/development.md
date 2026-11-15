# Development, testing and engines

## Environment

- Python 3.10 or newer. `.[dev]` covers extraction and model-free diagnostics
  without Torch; add `ml` for training and sampling, and `engine` for ViZDoom
  checks. All twelve console commands support `--help` without Torch.
- Shapely/GEOS and NetworkX are core dependencies. Model-free graph assembly
  uses NetworkX planar embeddings to split high-degree rooms without making
  their connections non-planar.
- `uv.lock` pins resolved packages, hashes, Python-version markers and optional
  extras:

  ```sh
  uv sync --locked --extra dev --extra ml   # training environment
  uv sync --locked --extra dev              # base tools only
  ```

  Isolated locked base/dev environments were tested on Python 3.10, 3.12 and
  3.14; OS and accelerator coverage is still limited.
- The original repair environment used Python 3.14.4, NumPy 2.5.1, Click
  8.4.2, Shapely 2.1.2, NetworkX 3.6.1, PyTorch 2.13.0, pytest 9.1.1 and
  setuptools 83.0.0. The lock selects newer packages, so do not switch
  dependencies in the middle of an exact-resume run.

## Tests

```sh
python -m pytest -q
```

The default suite skips the optional engine and port checks, and tests that
need local corpora, reports or IWADs skip when those are absent. A skipped test
is not a pass.

### Engine checks (ViZDoom)

The verified engine target is ViZDoom 1.3.1 with its bundled Freedoom II IWAD.

```sh
python -m pip install -e '.[engine]'
DOOM_GEN_ENGINE_TESTS=1 xvfb-run -a python -m pytest tests/test_engine_integration.py -q
```

- Use an existing display instead of `xvfb-run` where appropriate.
- The tests cover stairs, doors from both sides, bidirectional teleports in
  small and large rooms, exit activation, decorative-sector crossings, keys
  and remote door switches. They isolate native crash logs and reject them
  even when Python reports success.
- Check that a generated panel reaches its exits:

  ```sh
  DOOM_GEN_ENGINE_TESTS=1 DOOM_GEN_ENGINE_PANEL=/absolute/path/to/sample-panel.wad DOOM_GEN_ENGINE_REPORT=/absolute/path/to/completion.json xvfb-run -a python -m pytest tests/test_engine_integration.py::test_generated_panel_can_reach_its_exits -q
  ```

  The controller uses ordinary movement/use actions and a bounded route
  search; failure does not prove a map impossible. These empty-map exit checks
  do not measure combat, encounter balance or human quality.
- Prefer externally compiled nodes for engine checks. In one measured ViZDoom
  1.3.1 case, a panel completed all six maps but crashed in native shutdown
  when nodes were built at runtime, while an externally compiled copy completed
  and closed cleanly in three repeated runs. Keep the original and build a
  separate copy with `zdbsp input.wad -R -o compiled.wad`. This is a verified
  workaround for that panel, not a diagnosis of the native bug. Local logs
  and both WADs: `generated/repair-experiments/fragment-seed-panels/`.
- Earlier results: thirty maps across three generated panels passed load/spawn
  checks, and a waypoint controller reached and activated all sixteen exits in
  the two single-example overfit panels.

### Source-port checks (zdbsp and Chocolate Doom)

With `zdbsp` and `chocolate-doom` on PATH:

```sh
DOOM_GEN_PORT_TESTS=1 python -m pytest tests/test_port_integration.py -q
```

- Uses ViZDoom's Freedoom II by default, or `DOOM_GEN_IWAD` when set.
- Chocolate Doom 3.0.0 rendered and ran movement demos on node-built
  stair/door/compact-teleport fixtures. All 23 maps from four earlier repair
  panels also passed these rendered movement checks (local logs and
  statistics: `generated/repair-experiments/port-compatibility/`).
- After rerouting, 9/10 maps passed Chocolate Doom movement checks. One larger
  map has a 215-by-190-cell BLOCKMAP whose offsets exceed Chocolate Doom's
  signed 16-bit range and crashes there; it completes in ViZDoom (local
  report: `generated/repair-experiments/port-compatibility-routed/report.json`).
- Broad vanilla engine limits, other ports and full Chocolate Doom progression
  remain unverified.

## Continuous integration

GitHub Actions (`.github/workflows/ci.yml`) runs locked base environments on
Linux (Python 3.10/3.12/3.14), Windows and macOS (3.12), plus a Linux CPU ML
job. A manual workflow run also enables the ViZDoom, zdbsp and Chocolate Doom
checks with bundled Freedoom. GPU/MPS and hosted Windows/macOS runs have not
been verified yet. An earlier minimum-Python base check passed 384 tests on
Python 3.10 with NetworkX 3.4.2, with the optional ML/engine checks skipped.

For packaging changes, build a wheel and run `tools/check_install.py` against
the installed package outside the checkout; it checks the console commands and
base imports with Torch deliberately unavailable.

## Local playtest panels

A working checkout may hold playtest WADs under `generated/playtest/` (see
`generated/playtest/README.md` there). They are local artifacts: git ignores
them and they are not part of the package. Examples from the repair runs:

- `neural-reachable-ends.wad` (`-warp 1` to `-warp 4`): from a
  `reachable-ends` checkpoint trained on the v6 corpora, 11–42 rooms per map;
  three maps use the planar large-graph router.
- `neural-adjoining-blue-key.wad` (`-warp 1`): single keyed map with adjoining
  architecture, eight enemies and two terraces. Checks in both IWADs collect
  the card, open and cross the gate, clear combat, tour both terraces and exit.
  The earlier `neural-blue-key.wad` is unchanged. Screenshots are in
  `generated/playtest/adjoining-keyed-comparison/index.html`.
- `neural-semantic-objectives.wad` (`-warp 1` to `-warp 3`): the ordinary
  three-map variant.

```sh
gzdoom -iwad wads/DOOM2.WAD -file generated/playtest/neural-adjoining-blue-key.wad -warp 1 -skill 3
```

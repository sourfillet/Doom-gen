# Evaluating the room model

All of these commands require empty output directories, so earlier evidence is
never overwritten.

## Learning probes

```sh
OMP_NUM_THREADS=1 overfit-roomgen --out generated/toy-overfit --steps 800
OMP_NUM_THREADS=1 overfit-roomgen --source-json dataset/maps/<hash>_MAP01.json --out generated/map-overfit --steps 1200
```

- The overfit command saves targets, full checkpoints, fixed-seed comparisons
  and a WAD panel. Without `--source-json` it uses a synthetic three-room
  graph.
- Incomplete source fragments have censored endings by default, and prefix
  matches do not show that the full source can be reproduced.
- `--force-terminal` explicitly trains an artificial END for an encoded-target
  memorization experiment and records that limitation in the report.

## Held-out evaluation

```sh
OMP_NUM_THREADS=1 evaluate-roomgen --checkpoint checkpoints/rooms/best.pt --maps-dir dataset/maps --out generated/evaluation --samples 32 --write-maps 8
```

- Uses the checkpoint's frozen corpus manifest. Training references follow the
  checkpoint's source-selection settings, so excluded fragments do not enter
  copy/similarity checks.
- Reports complete and fragment held-out unconditional prefix losses
  separately, graph distributions, nearest training descriptors, exact
  binned-graph matches and generation failures.
- Includes matched-count procedural and nearest-size binned-graph retrieval
  panels with the same realization budget. `--no-baselines` skips them.
- Exact wall/height copy probes ignore XY translation, IDs and texture changes.
  Rotated, partial and approximate copies are outside that metric.
- New reports bind checkpoint hashes and keep per-seed validation/publication
  outcomes, distinguishing failed geometry from individually rejected compiled
  maps and final batch publication failures. Historical reports without
  checkpoint hashes are marked as having weaker identity evidence.
- New reports include version-1 exact room-graph snapshots for
  [routing replay](#routing-replay).

## Pooling seed panels

Pool disjoint panels from the same checkpoint, corpus, settings and source:

```sh
python -m tools.summarize_evaluations --out generated/seed-summary generated/evaluation-a/report.json generated/evaluation-b/report.json
```

The summary includes publication-rate Wilson intervals and paired discordant
seed counts. These describe sampling uncertainty for one fixed model and
routing budget, not training-run, dataset, gameplay or human judgments.
Overlapping seeds or incompatible runs are rejected.

## Rendering

```sh
python -m tools.render_evaluation generated/evaluation --limit 16
```

Writes `comparison.html` and SVG files beside the raw report, with per-seed
maps, rejected samples and world dimensions. Written maps are labeled
separately from any engine-completion evidence.

## Routing replay

Replay the same sampled graphs without loading a checkpoint or re-encoding the
training maps, optionally with a different geometry profile:

```sh
python -m tools.evaluate_routing --input generated/evaluation/report.json --out generated/routing-replay --samples 16
python -m tools.evaluate_routing --input generated/evaluation/report.json --out generated/varied-replay --geometry-profile varied --samples 16
python -m tools.render_evaluation generated/routing-replay --limit 16
```

Replays save input and tool snapshots, runtime versions, failed seeds, routing
details and separate validation/publication results. Graph snapshots keep world
coordinates, heights, roles, holes and component IDs; they do not round-trip
through lossy attribute bins.

## Limits and results so far

- Descriptor similarity cannot prove authored-geometry copying. Human review,
  procedural/retrieval comparisons and tag ablations are still needed.
- Local single-example probes memorized a three-room graph and a 15-room MAP01
  graph across 32 seeds each, and each produced eight validated maps. This
  shows that the graph learner and assembler work together on a training
  example; it does not show generalization.
- After bounded rerouting, the broader complete-source experiment completed all
  ten written maps from its best/final panels. Write success rose from 2/8 to
  3/8 and from 5/8 to 7/8 on unchanged seeds and weights. Other seeds still fail
  corridor routing, and the learner shows validation overfitting.
- Reports and panels from these runs are under `generated/repair-experiments/`,
  which git ignores.

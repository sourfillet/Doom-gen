# Preparing data

Start with classic Doom/Doom II binary maps. Other formats parse, but parsing
them does not mean their mechanics are supported in training or generation.

## Extract maps and build a corpus

```sh
parse-wad /path/to/DOOM2.WAD --output dataset/maps
parse-wad /path/to/another.wad --output dataset/maps
prepare-corpus --maps-dir dataset/maps --out dataset/corpus.json --seed 7
```

- Extraction writes `<full-wad-sha256>_<escaped-map-name>.json` and keeps the
  source marker in metadata. Per-WAD `.manifests/` record paths, formats and
  content fingerprints.
- Repeated extraction is idempotent. Incremental `--map MAP01` preserves
  previous manifest entries. Use a fresh directory when migrating a legacy
  corpus, so old filenames do not duplicate newly extracted documents.
- `prepare-corpus` groups duplicate content and source WADs into
  deterministic train/validation/test splits and keeps duplicate provenance.
  Related WAD releases need an explicit hash-to-family mapping through
  `--families`; hashes cannot infer authorship.
- A corpus with fewer than three families is explicitly an overfit/smoke
  experiment.
- Trainers can build the same manifest themselves, but a frozen `--manifest`
  makes comparisons easier to audit.

## Audit a large WAD collection

Check training eligibility before spending time on a new training run. These
are source-tree tools; run them from the checkout.

```sh
python -m tools.audit_sources --wads wads --out generated/source-audit --workers 4 --timeout 30 --export-documents
python -m tools.audit_sources --wads wads --out generated/source-audit --workers 4 --timeout 30 --export-documents --resume
python -m tools.materialize_sources --audit generated/source-audit/report.json --out generated/complete-corpus
```

**Audit (`tools.audit_sources`)**

- Records mapless WADs, extraction/target failures, per-map stages, timeouts
  and complete/fragment ending evidence. A timeout is inconclusive.
- `--limit N` bounds a batch. `--resume` processes the remaining maps without
  repeating completed results. Per-map records survive an interruption even
  before the aggregate report is refreshed.
- Resume requires matching WAD content, relevant preprocessing dependencies,
  Python/library versions and time budget. Unrelated model, generation or
  diagnostic edits do not invalidate new scoped source caches; whole-repository
  provenance is recorded separately. Older whole-tree signatures keep their
  conservative compatibility checks. Use a new directory for a changed
  assessment.
- `--export-documents` saves the assessed source JSON and exact targets in the
  same pass. The aggregate audit stores hashes rather than repeating full target
  geometry.

**Retry timeouts (`tools.retry_source_audit`)** with a longer budget, keeping
an exported audit intact:

```sh
python -m tools.retry_source_audit --audit generated/source-audit/report.json --wads wads --out generated/source-audit-retry --timeout 120 --workers 2
```

The retry creates an independent, verified branch. It copies verified terminal
records and documents, keeps the parent provenance and original budgets, and
reassesses only the timeouts. Changes to the source inventory, assessment code,
runtime or snapshots are rejected. It does not resume an interrupted retry
branch.

**Materialize (`tools.materialize_sources`)**

- Requires a finished, stable audit with matching code and runtime. It rechecks
  the selected complete examples and writes `maps/` and `corpus.json`; pass
  those paths to a trainer.
- Verifies and reuses exported snapshots. Without `--export-documents` it
  re-extracts and re-assesses. Changed snapshots are rejected.
- `--families` groups related releases. `--include-fragments` also exports
  assessed incomplete examples; train these with
  `train-roomgen --include-incomplete` so their endings stay censored.
- `--workers` and `--timeout` bound parallel re-extraction.
  `--validation-fraction` and `--test-fraction` (default 0.1 each) configure
  the family holdout. Choose them before inspecting model results.
- Records keep exact graphs, target sequences and ending evidence. "Complete"
  means a supported potential exit history. Combat, resources, moving-floor
  puzzles and compact-stair exclusions remain separate limitations. The
  filtered examples do not represent the distribution of the whole collection.

## Reuse prepared room targets

Prepared target snapshots avoid repeating the expensive geometry/progression
pass during an unconditional room-model run:

```sh
python -m tools.train_prepared_roomgen --corpus generated/complete-corpus --out checkpoints/prepared --include-incomplete --steps 1000
python -m tools.train_prepared_roomgen --corpus generated/complete-corpus --out checkpoints/prepared --include-incomplete --resume checkpoints/prepared/roomgen_001000.pt --steps 2000
python -m tools.evaluate_prepared_roomgen --corpus generated/complete-corpus --checkpoint checkpoints/prepared/best.pt --out generated/prepared-evaluation
```

- Training verifies the source documents, target snapshots, graph/sequence
  agreement, assessment code and runtime. It reuses the regular model,
  censored-ending loss, validation selection and resume loop, and binds the
  preparation/tool signatures into its checkpoints.
- Only complete examples enter validation or test-loss pools. Resume with
  unchanged inputs.
- Evaluation reuses the verified source targets in the ordinary evaluator.
  Sampling, procedural/retrieval baselines, raw copy probes and geometry checks
  are unchanged; reports record the cached inputs and exact tool sources.
- The regular `train-roomgen` command remains available for raw source
  documents and tagged training.

## Diagnose conversion before training

```sh
diagnose-maps --maps-dir dataset/maps --out generated/diagnostics --limit 8
generate-wad --from-json dataset/maps/<hash>_MAP01.json --out reconstructed.wad
```

- Diagnostics save source, raster, decoded and assembled JSON, SVGs and
  metrics, including failures and excluded source geometry. `--limit 0`
  processes the whole corpus.
- Source reconstruction preserves authored progression and reports losses. Its
  output is a diagnostic artifact and does not receive the generated-map
  playability guarantee.
- Raster decoding details are in [raster-diffusion.md](raster-diffusion.md).

## Format and cache versions

Older JSON-derived geometry upgrades from raw elements on load. Re-extract WADs
to recover compiled BSP regions, because old JSON lacks the node/seg lumps
needed for this recovery. When moving to a new derived or traversal version,
reassess source eligibility and regenerate preprocessing caches.

| Version | Change |
| --- | --- |
| Derived 3 | Preserves authored elements and raw-boundary diagnostics while storing validated coordinate regions. |
| Derived 4 | Handles pinched boundary cycles; preserves compiled regions through rotations, mirrors, height jitter and JSON upgrades. |
| Derived 5 | Permits validated compiled recovery around zero-length authoring boundaries (keeping the raw warnings and checking references); gives covered sector boundaries consistent geometric ownership. |
| Traversal 3 | Valid starts on open boundaries bootstrap reachability; physical clearance still rejects wall spawns and void interiors. |
| Traversal 4 | Includes possible ordinary passage after consuming a type-39 teleporter in the optimistic geometry union. Final histories track actual consumption; repeatable type 97 still suppresses false front walk-through. |
| Traversal 5 | Follows a drop across landing components too thin to stand in (a strip or step below a ledge). While the player box still overlaps the ledge, nearby boundaries are crossed at ledge height, and the room graph keeps a connection for that crossing. |

Historical checkpoint evaluation requires its recorded preprocessing versions;
use archived source to reproduce old reference populations. Inference sampling
still supports older checkpoint architectures. Multi-ring geometry, rasters,
graph inputs and room targets have changed, so regenerate caches and retrain to
benefit. Existing weights remain inference-compatible with their older
input/target shapes.

### Node formats

- Supported: normal XNOD/ZNOD and extended/compressed XGLN/ZGLN, XGL2/ZGL2 and
  XGL3/ZGL3 nodes, including bounded decompression, supplemental fixed-point
  vertices and fractional GL3 partitions.
- Miniseg ownership propagates only across miniseg partners; ambiguous
  ownership is rejected.
- Standalone glBSP v1/v2/v3/v5 `GL_*` blocks are associated with their map
  marker (including `GL_LEVEL`), with checked windings, partners and optional
  map checksums. Original vertices are untouched.
- V2 is checked against real zdbsp. The installed older builder emits
  nonstandard v5 records, which are explicitly rejected.
- Unsupported: obsolete glBSP v4, external GWA companions, slopes and 3D
  sectors.

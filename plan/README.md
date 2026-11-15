# Doom Map Generation — Architecture Plan

**Current direction (2026-11-05):** See [ML-first migration](../ML_FIRST_MIGRATION.md)
for the repository audit and proposed next implementation sequence. The original
plans below retain historical designs; implemented behavior and experimental
limits are tracked in `../REPO_REVIEW.md` and `../WORK_LOG.md`.

## Goal

Two decoupled halves, connected by a shared data schema and a shared tag vocabulary:

1. **Extraction** — read Doom-engine WADs (and the newer UDMF-format maps used by
   Boom/MBF/ZDoom/GZDoom ports) and turn every piece of information they contain
   into a structured, lossless, versioned dataset.
2. **Generation** — train models on that dataset and use them to generate new
   maps, conditioned on tags the user supplies at generation time, and write the
   result back out as a playable WAD.

Everything in between (tagging, dataset building) is shared plumbing that both
halves depend on.

```
WAD/PK3 corpus
     │
     ▼
[Extraction]  ──► raw + derived JSON per map (lossless)
     │
     ▼
[Tagging]     ──► user-defined tags, WAD-level and per-map, stored separately
     │              from the parsed data so re-tagging never requires re-parsing
     ▼
[Dataset build] ─► raster tensors + graph tensors + tag vectors, built from
     │              *all* captured fields, not a hand-picked subset
     ▼
[Training]    ──► layout diffusion → vectorizer → property GNN → thing placer
     │
     ▼
[Generation]  ──► user supplies a tag set (+ per-tag weight) → sample → validate
     │              → repair → write PWAD
     ▼
Playable .wad
```

See:
- [data-model.md](data-model.md) — what gets extracted from a WAD and how, with
  the principle that *nothing observable in the file is allowed to be silently
  dropped*.
- [tagging.md](tagging.md) — the user-defined tag system: WAD-level vs. map-level,
  storage, workflow, and how open-vocabulary tags get turned into something a
  model can condition on.
- [generation.md](generation.md) — the training/generation pipeline, how tag
  weighting works at sample time, and how the whole corpus (not just
  tag-matching maps) keeps contributing to geometry and texture-alignment
  quality.
- [roadmap.md](roadmap.md) — phased build order with concrete milestones.

## Non-negotiable constraints from the original ask

- **Comprehensive capture.** Heights, light levels, textures and their offsets,
  where things (monsters/items/decorations/player starts) are placed, sector
  specials, linedef specials/flags, texture animation sequences, palette/colormap
  data, per-map metadata (music, sky, cluster) — all of it goes into the
  extracted dataset, not just the "core five" lumps. See
  [data-model.md](data-model.md#capture-completeness-principle).
- **All of that geometry/data is available for training**, not just a coarse
  raster. The dataset exposes both a raster view and a full attributed graph
  view so no stage of the pipeline is starved of information the WAD actually
  contains.
- **Tags are completely user-defined.** No hardcoded taxonomy. Users invent tag
  strings as they go (`"multiplayer"`, `"techbase"`, `"city"`, `"ctf"`,
  `"joke"`, whatever). See [tagging.md](tagging.md).
- **Tags exist at two scopes.** Some tags describe a whole WAD (e.g. a WAD that's
  entirely deathmatch maps), others describe one map within a multi-map WAD
  (e.g. MAP01 is "techbase", MAP02 is "city" in the same WAD). Map-level tagging
  inherits WAD-level tags by default.
- **Generation is tag-weighted, not tag-exclusive.** The user hands the
  generator a set of tags (optionally with per-tag strength), and the output is
  pulled toward those tags — but the model still leans on the entire training
  corpus for things tags don't dictate, like plausible geometry, texture
  alignment, and structural validity. See
  [generation.md](generation.md#blending-tag-conditioning-with-corpus-wide-quality).

# Tagging System

The previous prototype used a fixed enum taxonomy (`doom1`/`doom2`/...,
`tech`/`hell`/..., etc.). That's explicitly *not* what's wanted: tags must be
entirely user-defined, invented as the user goes, with no hardcoded vocabulary
anywhere in the schema or the model's output space.

## Scope: WAD-level vs. map-level

A WAD is a container that can hold one map or dozens. Tags need to describe
both levels:

- **WAD-level tags** — describe the WAD as a whole (e.g. a WAD that's entirely
  deathmatch maps gets `"multiplayer"` once, not repeated per map; author name,
  gameplay-mod compatibility, `"joke-wad"`).
- **Map-level tags** — describe one map inside a (possibly multi-map) WAD
  (MAP01 is `"techbase"`, MAP02 in the same WAD is `"city"`).

**Inheritance rule:** every map's *effective* tag set = its own map-level tags
∪ its WAD's tags, unless the user explicitly excludes an inherited tag for
that one map (an exclusion list per map, for the rare case a WAD-level tag
doesn't fit one map in the set). This keeps single-map WADs simple (tag it
once, at the WAD) while still letting multi-map megawads differentiate.

## Storage: separate from parsed data

Tags live in their own store, not inside the extracted map JSON from
[data-model.md](data-model.md), for one reason: **re-tagging must never
require re-parsing.** The store is keyed by WAD hash (and map name within it),
so:

```
tags/
├── registry.json          # all tag names + user-authored descriptions ever created
├── <wad_hash>.wad.json    # { "tags": [...], "weights": {...} }
└── <wad_hash>/
    └── <map_name>.json     # { "tags": [...], "excluded_inherited": [...], "weights": {...} }
```

A tag *assignment* is `{ "tag": "techbase", "weight": 1.0 }` — weight defaults
to 1.0 (fully applies) but can be lowered for partial/ambiguous cases ("mostly
techbase, a little hell" → two tags at 0.7 and 0.3). This is optional; plain
presence/absence is the common case and weight is just an escape hatch, not a
required field the user has to think about every time.

## The tag registry is organizational, not restrictive

`registry.json` records every tag name a user has ever created, plus a
free-text description they write for themselves (and optionally a color/group
for their own UI organization). Its purpose is to:

- Power autocomplete while tagging, so near-duplicate tags (`"techbase"` vs.
  `"tech-base"` vs. `"Tech Base"`) get caught and merged/reused instead of
  silently forking the vocabulary.
- Support `tag rename` / `tag merge` operations across the whole corpus (all
  WAD/map tag files get rewritten) when a user decides two tags should really
  be one.

Canonicalization: tags are stored lowercase/trimmed with whitespace collapsed;
the registry keeps the user's preferred *display* spelling separately. This is
the only normalization applied — there's no fixed category system, no
required prefix, no closed set of allowed values. A user can create a tag with
a single command the first time they use it; nothing needs pre-registering.

## Tagging workflow

- **Interactive CLI**: browse parsed maps (with a lightweight rendered
  preview — top-down line art from the derived sector polygons is enough,
  no need for a full 3D renderer), assign tags with registry-backed
  autocomplete, set weights when needed.
- **Bulk import**: CSV/JSON import for a user who already has a labeling
  spreadsheet from prior organizing work.
- **Heuristic *suggestions*, never silent auto-tagging**: the derived layer
  (monster density, texture-name frequency, symmetry, connectivity shape)
  can propose candidate tags (e.g. "high monster density → maybe
  `hard`-ish, if you use a tag like that"), surfaced for the user to accept,
  edit, or reject. Suggestions never write a tag without user confirmation —
  tags are ground truth precisely because a human chose them, and letting
  heuristics write silently would undermine that.

## Auto-derived fact tags (amendment, 2026-07-26)

The no-silent-auto-tagging rule above is about *judgment calls* (theme,
difficulty, style). A second category exists: **objective, mechanically
checkable facts** — has ≥4 deathmatch starts (vanilla's hard requirement),
has coop starts, has secret sectors, uses Boom/MBF21 features, map format,
teleporters/keys/bosses/slopes present, IWAD origin, megawad size. These are
implemented in `tagging/auto.py` + `tag-maps auto` and ARE written without
per-tag confirmation, under derived-layer rules that keep the user the
source of truth:

- Stored in parallel `*.auto.json` files, regenerated **wholesale** on each
  run — the same raw-vs-derived split as the data model. Rename/merge skip
  them (regeneration would recreate the rule's canonical name anyway).
- User assignments always outrank auto tags at the same scope; resolution
  order is WAD-auto < WAD-user < map-auto < map-user, with
  `excluded_inherited` dropping WAD-level tags of either provenance.
- `untag` on an auto tag records a **veto** (`denied_auto` in the user's own
  file) so the override survives regeneration.
- Facts shared by every map in a WAD are hoisted to the WAD level, matching
  the inheritance model's intent.
- Every rule cites its basis (doomwiki engine behavior, or an explicitly
  documented project convention like megawad ≥ 15 maps); anything requiring
  taste stays in the suggestion pipeline.

## From open-vocabulary tags to model conditioning

Because the tag vocabulary isn't fixed and keeps growing, the embedding
strategy has to accommodate new tags appearing after training has already
started, not just a `nn.Embedding` sized to a vocabulary frozen at training
time:

1. **Learned per-tag embedding table**, keyed by canonical tag string. Grows
   as new tags are created; existing rows keep training, new rows start
   fresh and pick up signal as maps get labeled with them.
2. **Text-encoder fallback for cold-start tags.** A tag with zero or very few
   labeled examples doesn't have enough signal to learn a good embedding from
   data alone. Instead of leaving it at a random init with no gradient
   signal, embed the tag's name + its user-authored registry description with
   a small frozen sentence-embedding model, and initialize (or, for
   near-zero-example tags, entirely substitute) its vector from similarity to
   already-well-trained tags. This means a brand new tag behaves sensibly
   from the first generation request, even before it has training examples.
3. **Weighted multi-hot conditioning.** A generation request's tag set (each
   with a weight) becomes a conditioning vector by looking up each tag's
   embedding, scaling by its weight, and combining (sum or attention pooling)
   — fed into the diffusion/GNN/thing-placement stages the same way as any
   other conditioning signal, detailed in [generation.md](generation.md).

## Module layout (tagging half)

```
tagging/
├── registry.py       # tag registry: create/rename/merge, canonicalization, autocomplete
├── store.py          # WAD-level and per-map tag file read/write, inheritance resolution
├── suggest.py        # heuristic tag *suggestions* from derived map data (non-authoritative)
└── embedding.py       # learned tag embedding table + text-encoder cold-start fallback
cli/
└── tag_maps.py        # interactive + bulk-import tagging CLI
```

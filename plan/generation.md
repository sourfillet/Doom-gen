# Training & Generation Pipeline

## Representation strategy

A Doom map is a planar subdivision graph (sectors = faces, linedefs = edges,
vertices = nodes) with a rich attribute set hanging off every element. The
pipeline uses both views of the same data, because each stage wants a
different one:

1. **Raster image** — coarse top-down multi-channel image, good for a
   diffusion model to lay out overall shape/topology.
2. **Attributed planar graph** — sectors/linedefs/sidedefs/things with *every*
   field captured in [data-model.md](data-model.md) available as node/edge
   features, good for filling in exact per-element properties.

Generation runs as four staged passes, each consuming the richest data it can
use:

## Stage 1 — Layout Diffusion (raster)

A conditional DDPM/DDIM model over a multi-channel top-down raster at fixed
tile resolution.

| Channel | Encodes |
|---|---|
| 0 | Binary wall mask |
| 1–2 | Floor height (normalized) |
| 3–4 | Ceiling height |
| 5 | Light level |
| 6 | Sector-ID coloring (topology hint) |
| 7 | Linedef special density |
| 8 | Thing density |

Tag conditioning (from [tagging.md](tagging.md#from-open-vocabulary-tags-to-model-conditioning))
is injected via cross-attention/AdaLN at every U-Net resolution level.
Classifier-free guidance dropout during training gives an adjustable
tag-strength knob at sample time.

## Stage 2 — Vectorization (raster → graph)

Converts the sampled raster into an explicit `DoomMap` skeleton: contour-trace
the wall-mask channel → polygon extraction → sector assignment → linedef graph
construction, with heights pulled from their raster channels as an initial
estimate. Textures/specials are left as placeholders here — Stage 3 owns them.
A learned polygon-regression fallback is a documented option for noisy rasters
where algorithmic contour tracing struggles, not required for v1.

## Stage 3 — Property Filling (graph GNN)

> **Interim implementation note (2026-07):** until this GNN exists,
> `generation/style.py` provides a corpus-statistics stopgap - flat/wall
> texture distributions applied with adjacency coherence, corpus-sampled
> light-effect sector specials, monster/pickup placement from empirical
> density and type distributions, DM starts, and an S1 exit switch in the
> graph-farthest sector. The validator additionally reconnects
> void-separated islands with walk-over teleporter pairs. Doors need
> geometry insertion and stay out of scope for the stopgap.
>
> Two refinements make placement less arbitrary than pure corpus sampling:
> (1) `PlacementDensity` importance-samples monster/pickup *positions* from
> the generated raster's thing-density channel (channel 8) instead of
> uniformly within area-weighted sectors, so things follow where the layout
> model actually put density; it falls back to area-weighting when the
> channel is empty. Placement is gated for safety: a candidate is rejected
> unless it clears every sector wall by `_THING_WALL_MARGIN` (no things
> wedged against geometry) and sits in a sector reachable from the player
> start - reachability following shared walls AND teleporter links (the
> WR-97 pairs the validator adds), so nothing lands in a disconnected
> island; the monster/pickup budget scales with the reachable area only.
> (2) The `regenerate_from_json` harness no longer applies
> the corpus decorator by default - it calls
> `transfer_source_attributes`, carrying the *source map's own* real flats,
> wall textures, sector lighting/specials and things onto the regenerated
> skeleton (the raster round-trip loses only textures and thing types, not
> the source document). This keeps the harness diagnostic: any remaining
> deviation is attributable to geometry vectorization, not to invented
> decoration. Passing an explicit `--style` still exercises the corpus path.

A tag-conditioned GNN over the Stage 2 skeleton fills in everything the raster
couldn't represent at pixel resolution — this is where the "use *all* the
geometry and data" requirement really lives, since the GNN's input/output
space is the full attribute set from the data model, not a lossy raster:

| Target | Head | Notes |
|---|---|---|
| Floor/ceiling flat | Classification | over all flats seen in training, including animation-cycle grouping from `ANIMATED` |
| Upper/lower/mid texture per sidedef | Classification | same, using `TEXTURE1/2`+`PNAMES` vocabulary |
| Sidedef X/Y texture offset | Regression | learns real-world alignment habits (see texture-alignment metric in the data model) |
| Sector light level | Regression | 0–255 |
| Sector special | Classification | ~50 standard specials, extensible for Boom/MBF/ZDoom generalized specials |
| Linedef special + tag/args | Classification + grouping | tag-consistency pass described below |
| Linedef flags | Multi-label classification | full flag set, not a truncated subset |

Node features: normalized floor/ceiling height, area, perimeter, adjacency
count. Edge features: length, angle (sin/cos), flag bitmask, two-sided flag.
Global conditioning: tag embedding, via FiLM or concatenation.

**Tag consistency pass**: linedefs with non-zero specials must reference
sector tags that actually exist on a target sector. A post-processing step
groups trigger linedefs with their target sectors and assigns consistent
integer tags not already in use in that map — this is a small deterministic
repair, not a learned head, because it's a hard correctness constraint rather
than a stylistic one.

## Stage 4 — Thing Placement

Per-sector: a monster/item budget (conditioned on sector area + difficulty-ish
tags + game tags, if the user has created such tags), a type distribution
(multinomial, conditioned on sector context and global tag conditioning), and
a position (predicted offset from sector centroid, rejection-sampled to stay
inside the polygon). Player starts, teleport destinations, and keys are placed
by deterministic rule (start near the entrance, keys gated behind reachability
from the derived reachability graph) rather than purely probabilistically,
since these have hard correctness requirements the same way linedef tags do.

## Blending tag-conditioning with corpus-wide quality

The user's stated expectation is important here: tags should *steer* the
output, but general structural quality — plausible geometry, correct texture
alignment, valid connectivity — should still benefit from the whole training
corpus, not just the (possibly small) slice of maps that happen to share the
requested tags.

This falls directly out of standard classifier-free-guidance training and is
worth stating explicitly as a design requirement rather than an incidental
side effect:

- Every stage trains on the **full corpus**, with tag conditioning randomly
  dropped during training (standard CFG dropout). This forces the model to
  learn a strong *unconditional* prior over valid Doom geometry/texture
  alignment from every map it's ever seen, plus a *conditional delta* that
  tags contribute on top.
- At generation time, sampling mixes the unconditional prior with the
  tag-conditional signal via the CFG guidance scale. Turning the scale down
  leans more on general corpus-wide structural competence; turning it up
  leans more on the specific requested tags. This one knob directly answers
  "generate for a rare tag combination without the output falling apart
  structurally."
- **Stretch goal, not required for v1**: retrieval augmentation — at
  generation time, look up nearest-neighbor maps by tag overlap + geometry
  embedding and use them as structural exemplars (e.g., seed Stage 2's
  vectorizer with a retrieved skeleton, or use a retrieved sector shape as a
  prior). Documented in the roadmap as a Phase 8 item because it meaningfully
  helps with the "niche tag + still want good geometry" case beyond what CFG
  alone provides, but it's an enhancement on top of the CFG-based approach
  above, not a substitute for it.

## Training data pipeline

```
WAD corpus
  ├── extraction     → raw+derived JSON per map (data-model.md)
  ├── tagging        → user-defined tags, WAD- and map-level (tagging.md)
  ├── rasterize      → multi-channel arrays (Stage 1)
  ├── graph-build    → attributed graph tensors, full field set (Stage 3+4)
  └── augment        → mirror X/Y, rotate 90° (with thing-angle correction
                        and sidedef left/right re-assignment), light/height jitter
```

Dataset classes expose raster view, graph view, and tag-vector for the same
underlying map document, so every stage trains from the same raw extracted
data — no stage is working from a separately hand-curated subset.

## Validation & post-processing

Generated maps go through a validity gauntlet before being written out:

| Check | Repair |
|---|---|
| At least one Player1Start | Insert at centroid of largest sector |
| Unclosed sectors | Snap near-coincident vertices; insert missing linedefs |
| Zero-length linedefs | Remove |
| Ceiling ≥ floor height | Swap/clamp |
| All sectors reachable from a player start | Flag; optionally connect or drop |
| Crossed/overlapping linedefs | Split or remove |
| Unknown texture names | Replace with nearest match in vocabulary |
| Linedef tag / sector tag consistency | Deterministic repair pass (Stage 3) |

## Game vocabulary & asset bundling

Generation draws geometry/style from the whole training corpus, but a
playable output must reference only assets its *target* game can render.
`generation/game_profile.py` builds a `GameProfile` straight from an IWAD:
its renderable wall textures and flats (TEXTURE1/2 + PNAMES + F_* spans) and
the DoomEd thing numbers that game's own maps use (there is no lump listing a
game's monster roster, so its maps are the authoritative sample). Usage
counts are kept so substitution prefers common assets.

`GameProfile.enforce(doc)` runs last in the pipeline and rewrites every
texture/flat/thing to something valid: an unknown texture/flat becomes the
closest name (difflib) or the game's most-common one; an unknown thing
becomes the most-common valid thing of the same category, or is dropped if
the game has none. Structural things (player/DM starts, teleport dests) are
never substituted. This is the guarantee behind `generate-wad --game
<iwad-or-profile.json>`: no missing-texture errors, no unspawnable things,
even when the corpus spans multiple games. (Map-slot naming — ExMy vs MAPxx —
is a separate concern the writer does not yet remap.)

The escape hatch is `wad_writer/assets.py`: an `AssetBundle` (custom flats,
patches, and composed TEXTURE1 entries, loadable from a `flats/ patches/
textures.json` directory) is written into the PWAD with correct
markers/PNAMES/TEXTURE1, and its names are registered on the profile so
enforcement keeps them. So a generated map can use graphics beyond the IWAD:
`generate-wad --game ... --assets <dir>`.

## WAD Writer

1. Assemble lumps in correct order: map marker → THINGS → LINEDEFS → SIDEDEFS
   → VERTEXES → (placeholder SEGS/SSECTORS/NODES/REJECT/BLOCKMAP), then any
   bundled asset lumps (PNAMES/TEXTURE1, P_*/F_* spans).
2. Shell out to a node-builder (e.g. `zdbsp`) to fill in the BSP/visibility
   lumps — these are exactly the lumps the data model deliberately excludes
   from training, since they're deterministic outputs of the geometry, not
   design choices.
3. Write PWAD header + directory + lump data. Requires the relevant IWAD
   present at runtime to actually play.

## Module layout (generation half)

```
data/
├── rasterizer.py       # map document → multi-channel array
├── graph_builder.py     # map document → attributed graph tensors
├── dataset.py           # dataset wrapper: (raster, graph, tag_vector)
└── augmentation.py       # mirror/rotate/jitter, both views
models/
├── conditioning.py       # tag embedding lookup + CFG null-token logic
├── layout/{unet,diffusion}.py
├── properties/{gnn,heads}.py
└── things/placer.py
generation/
├── pipeline.py           # orchestrates Stage 1→2→3→4→validate→write
├── vectorizer.py          # raster → DoomMap skeleton
├── validator.py           # validity checks
└── postprocessor.py       # repairs
wad_writer/
├── assembler.py           # DoomMap → lump binary dicts
└── wad_writer.py           # PWAD + node-builder invocation
```

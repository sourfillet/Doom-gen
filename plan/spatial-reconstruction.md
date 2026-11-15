# Spatial evidence and source reconstruction

The current graph RNN does not predict placement, shared walls, source outlines,
portal anchors or within-room heights. Its optional adjoining assembler changes
physical architecture procedurally. More training with v3 graph targets cannot
recover fields that those targets do not contain.

The next geometry learner has three distinct prerequisites. Keep their evidence
separate and preserve the old targets/checkpoints.

## Candidate boundaries and physical labels

`data/spatial_portals.py` audits the frozen
`experimental_spatial_evidence_v1` records. It checks the original linedef,
endpoint coordinates, action/tag and component ownership. Touching collinear
spans can form a contiguous opening; disconnected windows are never summed.
Player navigation then measures actual directed crossings and entry intervals
for an axis-aligned radius-16, height-56 player. A collapsed door needs usable
faces connected through the same internal free region.

Initial geometry and optimistic potential-door geometry have separate direction
lists. Preserve those lists rather than converting an edge-level status into a
bidirectional label. The potential union can combine incompatible histories or
overestimate access to an activator; it is conditional local geometry, not a
global progression certificate. Active teleport triggers remain active in both
views. Teleports are remote relationships, never inferred shared-wall doors.

All 45 frozen exports audited successfully. The original 38 unique records have
1,286 physical edges: 1,045 initially local, 154 potential-only and 87 with no
supported local crossing; 146 of the physical edges are one-way. There are also
26 remote teleport edges. Seven duplicate exports are excluded from these
training/holdout counts. No new split or vocabulary was fitted.

Before fitting geometry labels, retain supported directions and door/lock/tag
references, classify unsupported motion and history semantics, and report
exclusions by the frozen source-family split. Do not interpret unsupported or
unusable as proof of engine impossibility.

## Authored reconstruction contract

`data/spatial_targets.py` exports `doom_spatial_reconstruction_v1`;
`roomgraph/spatial.py` decodes it directly, without room templates or corridor
routing. This is a reconstruction contract, not a final neural checkpoint
architecture or migrated training corpus.

- A single inverse frame contains player-start XY, whole-map XY scale and the
  start floor origin. Classic authored XY decodes to integral coordinates;
  fractional predictions are rejected instead of silently snapped to a grid.
- All authored vertices, linedefs, sidedefs, sectors and things survive,
  including unreferenced elements, controls, textures, offsets, tags, lock
  actions and unknown flag bits. Shared boundaries retain their original IDs.
- Optional room hierarchy retains exact sector components, holes and distinct
  floor/ceiling values. Ambiguous multi-island ownership is rejected. Ungrouped
  geometry remains present; excluded or control sectors are not deleted.
- Compiled BSP regions and cached traversal are absent from decoder inputs.
  Derived geometry is recomputed from authored boundaries and rebuilt again
  after serialization through zdbsp. Unsupported dialects, slopes and extension
  fields fail explicitly; this version handles classic Doom binary maps.
- `authoring_digest` covers every authored field, including actions and assets.
  This is deliberately stronger than a shape-only geometry fingerprint.

Focused fixtures preserve authored lump bytes exactly for irregular rooms,
courtyards, shared-wall doors, multiheight interiors and remote controls. Native
checks establish courtyard circulation, key pickup, door opening from both
sides after closure, and step ascent/descent. They do not establish corpus-wide
completion, combat balance or a learned gain.

Lossless authoring does not imply byte-identical nodes or identical effective
source polygons. Original BSP recovery and a different node build can produce
different sector regions; source actors may also fail the current conservative
validator. The frozen corpus reconstruction report records those deltas and all
validation errors/timeouts independently. Never copy the old compiled regions
into the decoded target to make that comparison pass.

## Learner gate

Before a new overfit experiment, identify a small source with verified authored,
serialized and compiled geometry plus usable supported connections. Define a
fresh model/target contract for placement, shared boundaries and typed openings,
including exact interior heights and explicit exclusion masks. Fit any geometry
vocabulary on training families only. Keep unsupported motion/history cases
visible and out of positive labels. Then test one-example reconstruction before
a frozen-family holdout run and matched procedural/retrieval comparisons.

Current artifacts are under
`generated/repair-experiments/adjoining-architecture/`. Source reconstruction
artifacts are diagnostic; they are not published generated playtest maps.

## Fixed-scaffold memorization control (2026-10-30)

The earlier `doom_spatial_geometry_v1` overfit did not decode predictions: its
absolute source rings/component elevations overrode numeric placement heads,
and opening predictions were unused. It also omitted collapsed doors/controls;
the 61-sector MAP19 reconstruction reaches only three sectors under connected
validation. Historical reports remain preserved, with their acceptance claims
reopened.

`data/spatial_prediction.py` now splits the lossless authored reconstruction
target into an explicit source scaffold and numeric supervision. The scaffold
retains line/side references, all controls, assets, actions, thing metadata and
room incidence. All vertex/thing XY and every sector elevation are removed from
the scaffold and learned by `models/spatial.py`. Shared linedef vertex references
stay shared in the output. No source BSP regions or absolute-ring fallback enters
the inference path.

This control uses learned slot identities and fixed authoring combinatorics. It
is not a model of a novel scaffold, a RoomGraph-only decoder, a held-out result or
an independent opening-width learner. Widths follow the predicted vertices.
Explicit nearest-integer prediction quantization is a separate contract from
the unchanged strict lossless inverse transform. Versioned checkpoints reject
old unversioned heads and mismatched source scaffolds.

The CLI archives source/target/scaffold/supervision, checkpoints, predictions,
source helpers, dependency versions and all decode/physical outcomes. Tests must
delete target coordinates before inference and change predicted values to prove
the assembled map follows the model. Separate connected/exit/serialized checks
prevent valid boundaries from being presented as a playable result. The repaired
comparison labels authored reconstruction separately, never substitutes it for
missing model inference, and reports engine execution only when it actually runs.

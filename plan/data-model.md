# Extraction & Data Model

## Capture-completeness principle

The failure mode to avoid is picking a convenient subset of lumps up front
(e.g. "just THINGS/LINEDEFS/SIDEDEFS/VERTEXES/SECTORS") and never coming back
for the rest. Instead, extraction is split into three layers with different
stability guarantees:

1. **Raw layer** — a byte-accurate structured record of every lump the parser
   recognizes, plus an `extra` bag (keyed by originating lump/port) for bytes
   it doesn't yet have a typed field for. Nothing recognized is dropped;
   nothing unrecognized crashes the parser — it's preserved untyped instead of
   discarded, so future schema versions can grow into it without re-parsing
   the corpus from WAD files.
2. **Derived layer** — computed, cache-able, *regenerate-able-from-raw* data:
   sector polygons, adjacency graphs, reachability from player starts, texture
   alignment stats, symmetry detection. Never authoritative — always a pure
   function of the raw layer so it can be recomputed after a bugfix without
   touching stored raw data.
3. **Training layer** — the rasterized/graph tensors actually fed to models.
   Built from raw + derived, per model stage. This is the only layer that's
   allowed to be lossy, and even then the default is "include everything
   feasible," revisited as the pipeline matures — not truncated early for
   convenience.

## What gets parsed

### Binary (Doom/Hexen) format map lumps — authoritative, not regenerable

| Lump | Captures |
|---|---|
| `THINGS` | Every placed object: position, angle, DoomEd type, skill/mode/flag bits (including MBF friendly-flag and port-specific extensions), Hexen's extra fields (special/args/id) when present |
| `LINEDEFS` | Vertex pair, full flag set (impassable, block-monsters, two-sided, upper/lower unpegged, secret, block-sound, not-on-map, already-on-map, plus Boom/MBF/ZDoom extension flags), special + tag/args, both sidedef references |
| `SIDEDEFS` | X/Y texture offset, upper/lower/mid texture names, owning sector |
| `VERTEXES` | Raw x/y (fixed-point where applicable) |
| `SECTORS` | Floor height, ceiling height, floor flat, ceiling flat, light level, special, tag |

`SEGS`/`SSECTORS`/`NODES`/`REJECT`/`BLOCKMAP` are BSP/visibility artifacts fully
derivable from the five lumps above — they are **not** stored as training data
(storing them would just be noise correlated with a node-builder's internal
choices), but the parser still reads them opportunistically to sanity-check
sector polygon reconstruction against a known-good BSP.

### UDMF (`TEXTMAP`) format — the modern text-based map format

Boom is binary-only, but ZDoom/GZDoom/Eternity maps increasingly ship as UDMF
(`TEXTMAP` lump instead of binary `THINGS`/`LINEDEFS`/...). UDMF adds fields the
binary format can't express at all: per-vertex Z, sector floor/ceiling slopes,
3D floors, per-line/per-sector arbitrary user-defined key/value fields. A
parallel `udmf_parser.py` reads `TEXTMAP`/`ENDMAP` and maps its namespaced
fields onto the same logical schema as the binary parser, with slope/3D-floor
fields carried in explicit optional fields rather than dropped — WADs using
these features are exactly the kind of richer geometry we want represented,
not skipped.

### Global/texture-system lumps (WAD-wide, not per-map)

| Lump | Captures |
|---|---|
| `TEXTURE1`/`TEXTURE2` + `PNAMES` | Every composite texture: name, width/height, and its patch list (name + x/y origin per patch) |
| Flats (between `F_START`/`F_END`) | Every 64×64 (or larger, in newer ports) flat, raw indexed pixels |
| `PLAYPAL` | All 14 palettes |
| `COLORMAP` | All 34 colormaps (needed to reproduce in-game lighting, not just the raw light-level integer) |
| `ANIMATED` (Boom) / animation definitions | Which textures/flats are animation-cycle frames of each other — critical so the model learns "these N texture names are interchangeable," not N unrelated textures |
| `SWITCHES` (Boom) | Switch-texture on/off pairs |

### Per-map/per-WAD metadata

| Source | Captures |
|---|---|
| `UMAPINFO`/`MAPINFO`/`ZMAPINFO`/`EMAPINFO` (when present) | Level name (free text — also a strong signal for auto-tag *suggestions*), music lump, sky texture, cluster/episode grouping, next-map linkage |
| Cluster/episode grouping | Which maps in a WAD are structurally related (same hub/episode) — informs WAD-level vs. map-level tag inheritance and cross-map consistency checks |
| WAD header | IWAD vs. PWAD, lump count, embedded WAD-level identifying info |

Unresolved/custom `THINGS` types (from DECORATE-defined actors in complex mod
WADs) are recorded with whatever DoomEd number/name is present and an
`unresolved: true` flag rather than being dropped — the geometry/placement
signal is still useful for training even when the semantic category is
unknown.

## Derived data

Computed once from the raw layer, cached, versioned:

- Sector polygons (ordered vertex ring per sector, walking the linedef graph)
- Sector adjacency graph (which sectors share a two-sided linedef, and which
  linedef)
- Reachability graph from each player start (which sectors/things are
  actually reachable — flags disconnected geometry rather than silently
  training on it as if it were normal)
- Linedef trigger → target-sector map (tag matching)
- Texture-alignment metrics (does a texture's offset line up across adjacent
  sidedefs, common in hand-built maps — a feature worth learning to reproduce)
- Symmetry detection (mirrored/rotated layouts are common in deathmatch/CTF
  maps — a useful WAD/map-level auto-tag hint, see [tagging.md](tagging.md))
- Normalized coordinate space (re-centered, unit-scaled) stored *alongside*,
  not instead of, raw coordinates — absolute scale still matters (an 8/16/24
  unit step height is a stair; a 64-unit step is not climbable), so both views
  are kept.

## Output format

One self-contained, versioned JSON document per map (plus one per WAD for
WAD-wide texture/palette/metadata lumps, referenced by hash so multiple maps
sharing a WAD don't duplicate texture data):

```json
{
  "schema_version": "1.0",
  "meta": {
    "map_name": "MAP01",
    "source_wad": "DOOM2.WAD",
    "wad_hash": "sha256:...",
    "format": "doom" ,
    "mapinfo": { "level_name": "Entryway", "music": "D_RUNNIN", "sky": "SKY1", "cluster": null }
  },
  "bounds": { "min_x": -768, "max_x": 1792, "min_y": -704, "max_y": 1216 },
  "vertices": [ { "id": 0, "x": 0, "y": 0, "z": null } ],
  "sectors": [
    {
      "id": 0, "floor_h": 0, "ceil_h": 128,
      "floor_flat": "FLOOR4_8", "ceil_flat": "CEIL3_5",
      "light": 160, "special": 0, "tag": 0,
      "slope": null
    }
  ],
  "linedefs": [
    {
      "id": 0, "v1": 0, "v2": 1,
      "flags": { "impassable": true, "two_sided": false, "...": "..." },
      "special": 1, "tag": 0, "args": null,
      "right": 0, "left": -1
    }
  ],
  "sidedefs": [
    { "id": 0, "x_off": 0, "y_off": 0, "upper": "-", "lower": "-", "mid": "STARTAN2", "sector": 0 }
  ],
  "things": [
    {
      "id": 0, "x": 512, "y": 0, "angle": 90,
      "type": 1, "type_name": "Player1Start", "category": "player-start",
      "unresolved": false,
      "flags": { "skill_1_2": true, "skill_3": true, "skill_4_5": true }
    }
  ],
  "derived": {
    "sector_graph": { "nodes": [ "..." ], "edges": [ "..." ] },
    "sector_polygons": [ [0, 1, 2, 3] ],
    "trigger_map": { "0": [12, 13] },
    "reachable_from_start": [0, 1, 2, 3, "..."]
  },
  "extra": {}
}
```

`schema_version` and the `extra` bag exist specifically so the schema can grow
(new UDMF fields, new port extensions) without invalidating already-parsed
data — a migration reprocesses `extra` into typed fields; it never needs to
re-read the original WAD.

## Module layout (extraction half)

```
wad_parser/
├── types.py            # dataclasses, versioned schema, special resolvers
├── lookup_tables.py     # GENERATED DoomEd/linedef/action-special tables — do not hand-edit
├── wad_reader.py        # binary WAD I/O: header, lump directory, raw lump bytes
├── lump_parser.py       # struct-unpack for binary-format map lumps (incl. Hexen SPAC flags)
├── udmf_parser.py        # TEXTMAP/ENDMAP parser (slopes, 3D floors, per-field extras)
├── map_extractor.py      # assemble a map document from parsed lumps; compute derived data
├── texture_parser.py     # TEXTURE1/2, PNAMES, flats, PLAYPAL, COLORMAP, ANIMATED, SWITCHES
└── mapinfo_parser.py      # UMAPINFO/MAPINFO/ZMAPINFO/EMAPINFO
tools/
└── gen_lookup_tables.py  # regenerates lookup_tables.py from doomwiki.org + ZDoom zspecial.acs
```

The lookup tables are **generated, not hand-written**: doomwiki.org's "Thing
types" and "Linedef type" pages are the source of truth for DoomEd numbers
and vanilla/Boom linedef specials, and the ZDoom ACS compiler's
`zspecial.acs` for the Hexen/ZDoom action-special numbering. Hand-editing
them invites the transcription errors that generation exists to prevent —
fix the generator (or its sources) and regenerate instead.

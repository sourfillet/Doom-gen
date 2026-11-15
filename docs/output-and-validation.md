# Output, validation and assets

## What a published map guarantees

Generated maps require a valid start, supported potential traversal to playable
geometry, and a usable reachable exit, checked after decoration and again after
serialization. These are static checks, not a full engine completion
certificate.

- **Exit fallback.** Prefers a predicted exit room, then distant reachable
  walls.
- **Repair.** Closed doors and authored darkness survive default repair. Floor
  lowering, brightening and improvised teleport reconnection require explicit
  opt-in.
- **Keys and doors.** Supported vanilla doors open only after potentially
  reachable activation and the required skill-3 keys. Extraction keeps
  optimistic potential-access distances. Final validation also searches
  location-consistent key/door histories, checking physical pickup/use
  approaches and keeping irreversible branches. The default budgets are 4,096
  states and 256 geometry configurations; exhausting them is an explicit
  failure. Doors stay open in this abstraction.
- **Player clearance.** Configuration-space checks cover narrow intra-sector
  passages, directed drops and exit approaches. A drop keeps the player at the
  ledge height until its box clears the ledge, so a low ceiling just beyond a
  narrow trench blocks it, as in the engine. The player may land in any lower
  sector under that ledge-height sweep, so a thin step touching the ledge does
  not hide the room beyond it. Walk-trigger lines passed over while still
  supported by the ledge are not recorded as crossings.
- **Unsupported.** Timed puzzles, consumed one-shot door activators,
  failed/backside or extension teleport histories, lifts, scripts and
  combat/resource progression.
- Traversal metadata version 2 recomputes older cached distances, including the
  correct skill-3 teleport population and flag-independent player-one start.

## Writer and publication

- The writer rejects invalid references and out-of-range fields, translates XY
  coordinates when the span fits without rescaling, and checks the actual
  written geometry.
- Publication is atomic: failed validation or node building preserves the
  previous output.
- Install `zdbsp` separately to build nodes. The builder receives `-R` to size
  the all-visible REJECT table after sector pruning, and the writer also
  supplies this table before node building (empty REJECT data caused
  invalid-table warnings in Chocolate Doom).
- All supplied binary BLOCKMAPs receive structural/index checks, including for
  ZDoom publication. A retrieval control exposed wrapped unsigned 16-bit
  pointers from a large zdbsp table; such malformed compiled outputs are
  rejected atomically. Empty modern-port tables remain a runtime-build
  fallback.

## Target ports

Both generation commands accept `--target-port vanilla` to reject known static
format/index/BLOCKMAP incompatibilities before publication. It requires zdbsp
and preserves an existing output on failure. View-dependent vanilla rendering
limits and gameplay still require engine tests. The default target is `zdoom`.

## Textures and assets

**Bundled graphics**

- Use strict Doom names: ASCII, at most eight bytes. Raw flats and patches
  cannot use map markers or engine/namespace lump names.
- For vanilla output with `generate-wad --assets ...`, pass
  `--game path/to/IWAD.wad` so the bundle can keep the complete base texture
  tables and patch indices. A saved vocabulary-only game profile does not
  contain those tables. In Python, bind
  `bundle = bundle.with_base_wad(iwad_path)` before publication; this copies
  table metadata without bundling the base images.
- Vanilla new flats need unique names and exactly 4096 indexed pixels. Patch
  replacements cannot shadow base flats. Patch-only bundles keep the base
  texture tables. Modern-port standalone resource definitions remain available
  through the low-level writer.

**Extracting from source WADs**

- Both generators accept `--sources source.wad,another.wad` (or a directory)
  with `--game path/to/IWAD.wad` to extract missing referenced textures and
  flats before game enforcement substitutes them.
- Directory entries are sorted, and the first indexed definition of a name
  wins. Changed source patches get private names, so existing base textures
  keep their original pixels. Incomplete textures are left for ordinary
  fallback enforcement.
- Generation results include source WAD hashes, resolved names and output
  asset hashes under `asset_resolution`.
- Custom palettes, animation/switch tables and source-port scripts are not
  transferred.

**Embedded or companion resources**

- Assets are embedded by default. `--companion-textures resources.wad` writes a
  separate resource PWAD to load alongside the generated map.
- Both outputs are validated before either destination is replaced, and an
  ordinary publication failure preserves existing files. The two replacements
  cannot be atomic across a process or power failure.
- No companion file is reported when there are no assets to write.

Source WAD graphics keep their original licensing. The repository does not
include extracted commercial graphics.

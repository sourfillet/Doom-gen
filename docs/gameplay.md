# Procedural gameplay

These additions are procedural and do not retrain the room model. The default
`--gameplay none` keeps bare structural maps for comparisons.

## Intro encounters (`--gameplay intro`)

`generate-roomgen --gameplay intro` adds cover columns, small
zombie/shotgunner/imp encounters, a directly reachable shotgun/shell cache and
armor, recovery health and a distant usable exit.

```sh
generate-roomgen --checkpoint checkpoints/rooms/best.pt --style dataset/maps --game /path/to/DOOM2.WAD \
  --gameplay intro --geometry-profile varied --seed 1 --max-rooms 64 --max-attempts 1 \
  --out generated/intro.wad --report generated/intro.json
```

- `--style` takes a saved style `.json` or a directory of training maps to fit
  one from. `--game` takes the target IWAD or a saved game profile. The profile excludes `--placement`, so one
  component owns the encounter budget.
- Supply counts are reported. Enemy drops and the initial pistol ammo are not
  credited. The ammo allowance and recovery schedule are baseline assumptions.
- Materials use bounded, coherent palettes and learned light levels; older
  style files without light statistics keep their existing levels. Cover
  columns preserve player routes.
- For newly sampled graphs, combat rooms are constrained to 384–1024 units and
  headroom to 128–256, keeping topology and floor heights. Every changed size
  prediction is recorded in `gameplay_geometry`. This avoids the long empty
  transit caused by one oversized room setting the global routing spacing. The
  `none` profile keeps raw sampled sizes.
- Still open: human encounter balance and finer learned interior architecture.

### Goal selection

- With complete assembly metadata, the intro objective ranks the original
  requested rooms, so extra connector sectors do not change which room becomes
  the goal.
- Equal-distance ties prefer an exit role, then a terminal room, then a stable
  room ID.
- The selected room and policy are reported in `objective.selection`. Source or
  legacy layouts without that metadata explicitly report their sector-hop
  fallback.
- Actual player-region routing and progression checks remain separate.

## Blue-key objective (`--progression blue-key`)

`generate-roomgen --gameplay intro --progression blue-key` requires an
item/gate objective.

- A bounded preflight tries semantic goal rooms against existing mandatory
  doors and the actual key/gate planner before encounters are populated. It can
  therefore choose a different goal from an ordinary intro run. Rejected goals
  and the attempt budget appear in `objective.selection.key_preflight`.
- A blue card goes in a reachable earlier room, and the complete
  pickup/door/exit history is validated. The populated map must still pass
  final key-placement/history and publication checks.
- Layouts without a real mandatory gate are rejected within the usual retry
  budget. Room-role key predictions are preferred where usable.
- Still open: general lock relationships and multi-key or scripted
  progression.

## Walkable interior detail (`--interior-profile stepped`)

Add `--interior-profile stepped` to an intro run (it also combines with the
blue-key objective). It inserts bounded octagonal terraces with 16-unit risers
and 64-unit treads, keeps at least 64 units of headroom, and leaves the
original entrances and opening equipment clear. Every addition passes the
existing navigation checks, and its floor sectors and positions are reported.
The default `columns` profile keeps the earlier intro behavior. Learning
authored interior structure is separate work.

## Reports and fixed-layout comparisons

`--report` saves sizing policies, actual populations, supply budgets and failed
attempts to a fresh JSON file; existing reports are preserved.

To compare decoration on exactly the same existing layouts:

```sh
python -m tools.build_gameplay_demo --input layout.wad --style style.json --game game-profile.json --out generated/gameplay-demo
```

It keeps input/style/profile/source snapshots, reports each failed map and
validates compiled output before publication. To replay saved graphs with a
different geometry profile, see [routing replay](evaluation.md#routing-replay).

# Raster diffusion path

A tag-conditioned diffusion model paints a top-down multi-channel layout
raster, which is then vectorized into sectors and written as a PWAD.

```sh
train-layout --maps-dir dataset/maps --manifest dataset/corpus.json --out checkpoints/layout --steps 20000 --seed 7
generate-wad --checkpoint checkpoints/layout/best.pt --tags "single-player,secrets:0.7" --n 3 --out generated.wad
```

`generate-wad` also takes `--style`, `--game`, `--properties` and
`--placement` (see [decoration](room-graph-generation.md#decoration)), plus the
asset options in [output-and-validation.md](output-and-validation.md#textures-and-assets).
Raster generation skips only a rejected candidate and does not add unrequested
sampling retries.

## Scale and guidance

- New raster models learn world scale through a numeric condition and a
  training-only scale prior. `generate-wad --world-span ...` overrides the
  prior.
- Guidance uses `unconditional_plus_scale_v1`: 0 is unconditional, 1 is
  ordinary conditional prediction, and values above 1 strengthen the
  conditioning. An old positive guidance value `g` corresponds to `g + 1`; old
  0 corresponds to new 1.
- Sampling steps must not exceed the checkpoint's diffusion timesteps.

## Decoding

- The decoder defaults to factor-two cells, preserves small regions and removes
  only collinear contour vertices. Arbitrary contour smoothing is opt-in through
  the Python API.
- If coarse sampling changes occupancy/void components or erases a hole,
  decoding falls back to factor one while preserving world scale.
- On three local Doom II maps at 128 pixels, occupancy IoU improved from
  0.865/0.796/0.826 to 0.957/0.904/0.941. At factor one it was 1.0 against the
  source raster, which itself still loses source geometry at rasterization.
  Occupancy overlap does not establish topology or gameplay fidelity.
- Even perfect source rasters still lose some physical routes, so valid sampled
  rasters and faithful full-map reconstruction remain separate research gates.

## Raster versions

| Version | Change |
| --- | --- |
| 2 | Order-independent boundary channel. |
| 3 | Refreshed caches for BSP-derived geometry. |
| 4 | Sixteen channels: six explicit masks for specials 11/51/52/124/1/31 join the original ten. Decoding restores supported exit and closed-door actions only when the generated geometry supports them; it does not infer mechanics from generic special density. |
| 5 | Twenty channels: four binary east/south face channels for boundaries and solid walls. These remove ambiguous pixel-line crossings that invented parallel cuts and hundreds of tiny sectors in dense maps. Face targets decode at native pixel resolution, preserving world scale; `--coarse-factor` still controls legacy decoding. |

- Cached rasters and new training use version 5. Use
  `rasterize(..., raster_version=3/4)` to evaluate old targets.
- Retrain raster models to get the newer heads. Legacy checkpoints keep their
  numeric sector-color decoding, conditioner shape and decoding version for
  inference.

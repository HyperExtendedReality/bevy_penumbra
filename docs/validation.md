# Validation — October 5, 2026

These measurements cover the Cornell-box example, not a large production scene.
The [README](../README.md) lists supported materials and remaining paper features.

## Environment and method

- NVIDIA GeForce RTX 4070 Laptop GPU, Vulkan, NVIDIA driver 617.14.
- Windows 11, AMD Ryzen 7 8845HS, 31.3 GiB RAM.
- Bevy 0.19.1, Slang 2026.19, Slang optimization level 2.
- Optimized **dev** profile: crate level 1, dependencies level 3; not release timings.
- Headless 640×480 Cornell box, 96 triangles, emissive ceiling panel, no ambient
  or conventional lights for quality comparisons. Separate runs exercise point,
  directional, and spot sources.

GPU timestamps cover GI compute and compositing, excluding other Bevy rendering,
CPU extraction, allocation, and compilation. Benchmarks warm up for three seconds
and sample for approximately nine seconds. Laptop clocks, power state, background
activity, and scene content affect results. GI timings alone do not establish FPS.

## Visual quality

The former implementation relied on randomly reaching emissive triangles and
normalized sparse nine-term SH by reconstructed angular coverage. Moving sources
repeatedly exposed sparse samples, producing bright/dark patches and flicker.

The revision balances explicit emissive-triangle samples against cosine-direction
samples in solid-angle measure, using the existing total sample budget. This
prevents tiny emitter faces and nearby lights from producing unbounded sample
weights without radiance clamping. Triangle selection uses an area × luminance CDF
and independent world-space sample rotations. Light selection shares a phase within
each probe for coherence; different probes have independent phases. The CDF is
built before BVH partitioning so moving geometry cannot reorder emitter samples.
Cascades carry scattered radiance;
direct emission uses next-event estimation without double-counting. Six nonnegative
ambient-cube lobes replace SH normalization. Visibility checks reject interpolation
through walls. Separate screen history uses previous-world surface-plane/normal rejection, bilinear
reprojection, neighborhood clamping, temporal variance, and reactive blending.
Variance-aware à trous filtering never feeds spatially filtered values into history.
Reprojection and upsampling invert the actual integer prepass sample centers;
the nominal half-resolution grid previously shifted stationary history by a
quarter GI pixel. Hard plane rejection in filtering and compositing prevents
nearly parallel surfaces from exchanging illumination.

Quality defaults use 76,800 primary rays, up to 76,800 secondary rays, 32-frame
retention, three filtering steps, 16 combined direct-emission samples per receiver (eight per hit),
and 0.92 temporal blending. Previous defaults used 19,200 primary rays, 16-frame
retention, two filtering steps, 0.8 blending, and no explicit emitter samples.
Comparisons include both algorithm and quality-setting changes.

`scripts/check_quality.py` measures RGB residual RMS after fitting a quadratic
illumination gradient to fixed, unobstructed regions. This catches broad blotches
that adjacent-pixel differences miss. It also measures absolute RGB differences
between 16 successive captures, checks receiver brightness, and rejects frame-pair
changes above 2/255. Screenshot readback separates captures by approximately
30–60 ms here; they are not every rendered frame.

| Wall-region metric, RGB units out of 255 | Earlier | Revised |
|---|---:|---:|
| Static residual RMS | 5.682 | 2.507 (56% lower) |
| Moving-emitter residual RMS | 11.506 | 2.770 (76% lower) |
| Static captured-frame MAE, mean / maximum | unavailable | 0.522 / 0.699 |
| Moving captured-frame MAE, mean / maximum | unavailable | 0.834 / 1.187 |

Moving-source MAE includes real lighting changes. Before images are saved in
`tmp/quality-before/`; after sequences are `screenshots/quality_emissive_*.png`
and `quality_moving_emissive_*.png`. Reports are `tmp/quality-metrics.json` and
`tmp/quality-moving-metrics.json`. Reproduce using Python, NumPy, and Pillow:

```sh
cargo run --example cornell_box -- --capture-quality
cargo run --example cornell_box -- --capture-quality --capture-moving
python scripts/check_quality.py --before tmp/quality-before/feedback.png --after screenshots/quality_emissive_015.png --sequence "screenshots/quality_emissive_*.png"
python scripts/check_quality.py --before tmp/quality-before/moving.png --after screenshots/quality_moving_emissive_015.png --sequence "screenshots/quality_moving_emissive_*.png" --output tmp/quality-moving-metrics.json
```

These regional metrics and visual inspection demonstrate improvement in this
scene. They are not a path-traced accuracy comparison, a convergence proof, or a
guarantee that arbitrary scenes are artifact-free. Ambient-cube angular blur,
softened contacts, finite-budget soft-shadow noise, and reprojection ghosting remain.

The immediately preceding revision is preserved in `tmp/quality-followup-before/`.
Its wall RMS was 2.432 static / 2.880 moving: this follow-up targets near-emitter
speckles, history drift, and surface leaks rather than uniformly lowering wall RMS.
In a fixed ceiling strip (x=220..419, y=74..85), moving-source captured-frame MAE
dropped from 4.621 to 2.064/255 over 16 captures (55% lower). The report is
`tmp/quality-artifact-metrics.json`. Motion phase and readback intervals
differ, so this is a local stability observation, not an equal-time accuracy test.

## Performance and memory

| GI GPU timing, ms | Quality defaults | Reduced cost |
|---|---:|---:|
| Primary rays | 76,800 | 19,200 |
| Retention / filter steps / emitter samples | 32 / 3 / 16 | 16 / 2 / 8 |
| Timestamp samples | 461 | 532 |
| Mean | 13.514 | 3.488 |
| Median | 9.871 | 2.973 |
| p95 | 28.150 | 6.102 |
| Maximum | 30.297 | 9.692 |
| Approximate per-camera GI allocation | 254.5 MiB | 63.6 MiB |

Both configurations enable 0.8 feedback and 0.92 temporal blending. Memory excludes
geometry, Bevy prepasses, and render targets. Device limits can reduce the ray
budget. Each camera owns independent buffers. The reduced-cost configuration uses
the former ray count and filtering/retention budgets but changes sampling and
accumulation; this is not an equal-work or equal-quality comparison with the old code.
The quality preset exceeds a 16.7 ms GI budget in its slowest frames on this GPU;
it does not guarantee 60 FPS. Use the reduced-cost settings when frame time matters.
Compared with the preceding revision's 10.834 / 3.375 ms means, this run costs
more despite the same sample counts: cosine samples require closest-hit traversal
rather than early-out visibility rays. There are no new GPU buffers or dependencies.

```sh
cargo run --example cornell_box -- --benchmark
cargo run --example cornell_box -- --benchmark --divisor=4 --history=16 --filter=2 --emitter-samples=8
```

Each run overwrites `tmp/benchmark.txt`. Final reports are saved as
`tmp/benchmark-quality-final.txt` and `tmp/benchmark-low-cost-final.txt`.
Earlier SH timing tables do not describe this workload.

## Lighting and regression checks

Formatting, all-target compilation, Clippy with warnings denied, and all seven Rust
tests pass. The Slang test compiles both modules, checks all 17 named entry points,
and validates WGSL with Naga. CPU tests cover settings/ray budgets, emitter areas
and power-weighted CDFs, stable emitter ordering across BVH changes, BVH coverage,
geometry/material/transform edits and removals,
static reuse, and light motion/visibility/removal. Light-only edits share geometry.

All 18 lighting lifecycle scenarios pass, plus camera motion at divisor three,
two light-quality sequences and one camera-quality sequence of 16 images each.
Static and continuously moving emissive, point, directional, and spot
sources light all 1,056 tested front pixels. With GI disabled, this indirect-only
receiver has zero lit pixels, even with conventional direct lighting elsewhere.
Switch-off first proves illumination, then requests capture about 100 ms after
removing the source; all four sources leave zero lit pixels in the 38,400-pixel
lower receiver region.

Constant environment: expected color approximately 188.1/255, measured mean absolute
error 0.08/255 with tonemapping disabled. Offscreen emission lights the indirect-only
receiver too. Directional mode opens the ceiling; a closed room blocks sunlight.
Camera motion exercises reprojection without rebuilding static geometry.

Near-field energy: a 20×20 unit-radiance emitter sits 0.3 units above a white
diffuse plane; the central receiver sees almost the entire cosine hemisphere.
At 16 direct-emission samples, measured color MAE is 0.68/255 with defaults and
0.87/255 with `--history=0 --filter=0`, below the 2/255 regression threshold.
Screen temporal accumulation remains enabled in the latter test. An unfiltered
four-sample run has 3.78/255 MAE and correctly fails this quality threshold;
low sampling budgets still leave stochastic noise.

```sh
cargo run --example cornell_box -- --capture-near-field
cargo run --example cornell_box -- --capture-near-field --history=0 --filter=0
cargo run --example cornell_box -- --capture-quality --capture-camera-motion
cargo run --example cornell_box -- --capture-camera-motion --divisor=3
```

World-cache invalidation preserves the frame sequence. Screen history survives
scene edits with rejection/clamping; viewport/settings changes invalidate both.
With feedback disabled, explicit emitter sampling still supplies direct emission
and one reflected cascade bounce; feedback adds deeper transport in unchanged frames.

These GPU region checks catch inactive lighting and lifecycle regressions, without
quantifying every shadow edge or radiometric bias. Large moving scenes still need
BVH refitting/instancing. Textured ray-hit materials, adaptive distance LODs, and
secondary hierarchical ordering remain unsupported.

## Surface-cache and far-occlusion revision

This revision separates directional caches by the dominant surface-normal axis
and sign, including hierarchical ray ranking, cascade merging, irradiance
prefiltering, and secondary feedback. Opposing faces no longer share hemisphere
samples. Reduction and prefiltering retain an actual sample nearest the cell
center, rather than a centroid that may lie inside geometry. Cascade merging
uses that stored surface position. The exact directional fallback now performs
the same BVH visibility rejection as the irradiance-probe path.

Primary and secondary rays continue to the closest hit or scene escape. The
last cascade represents the remaining ray rather than treating its finite cutoff
as open sky. `--capture-distant-occlusion` first proves illumination from a unit
environment, then enables a 200-unit black enclosure and requires a dark receiver.
The preceding renderer fails its final occlusion check: the central receiver
averages 188/255. The corrected renderer averages exactly 0/255. The new test
passes with its positive control, so disabled lighting cannot satisfy it.

Screen accumulation applies its reactive history reduction on scene edits.
Unchanged scenes accumulate noisy samples with the full configured weight;
geometry rejection and neighborhood clipping still apply. In the final 16-frame
static sequence, back-wall captured-frame MAE falls from 0.522 to 0.210/255
(60% lower), and maximum pair MAE falls from 0.699 to 0.248/255. Wall brightness
is 137.48 before and 137.25 after; spatial residual RMS is 2.507 before and 2.490
after. Moving-source wall MAE is 0.895 mean / 1.245 maximum, versus the preceding
0.834 / 1.187. Moving-frame differences include real illumination changes and
depend on motion phase/readback intervals; these runs do not establish improved
moving-source spatial noise or radiometric accuracy. Both remain below the
existing 2/255 frame-pair limit.

Same-setting sequential GPU runs measure 7.840 ms mean / 7.937 ms p95 before and
8.937 ms mean / 9.044 ms p95 after, with 531 / 530 timestamp samples respectively.
Both use quality defaults. These are GI-only times in the existing optimized dev
profile on the hardware above, not total frame times. The additional cache
separation and visibility work cost about 1.1 ms in this workload. No GPU buffers,
dependencies, or sample budgets were added.

All seven Rust tests, including compilation and Naga validation of all 17 Slang
entry points, pass. All-target compilation, formatting, and Clippy checks pass.
GPU checks cover constant environment, near-field energy without retention or
spatial filtering, distant occlusion with its positive control, all four source
types under motion and switch-off, offscreen emission, and camera motion at
divisor three. Static and moving quality sequences each contain 16 images.

Sources and sequences preceding this revision are saved in
`tmp/quality-surface-before/`. The temporary baseline crate is
`tmp/quality-surface-baseline/`; reports are `tmp/quality-surface-report.json`,
`tmp/benchmark-surface-baseline.txt`, and `tmp/benchmark-surface-quality.txt`.

```sh
cargo run --example cornell_box -- --capture-distant-occlusion
cargo run --example cornell_box -- --capture-quality
cargo run --example cornell_box -- --capture-quality --capture-moving
```

This remains an approximate diffuse GI implementation. Dominant-normal sectors
can lose reuse or produce seams on curved surfaces; same-sector thin geometry,
sparse angular coverage, ambient-cube blur, and temporal disocclusions can still
produce artifacts. The tests demonstrate these fixes in the specified scenes,
not a perfect or universally artifact-free renderer.

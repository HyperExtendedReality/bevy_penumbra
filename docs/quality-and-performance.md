# GI quality and performance validation

This is a historical report from earlier revisions. The current crate supports
only the Split Radiance Cascades strict base. Every measurement and transport
comparison below describes previous source revisions and does not certify the
current implementation.

The optimization target is a responsive 1280×720 view at 30 FPS on a
GTX 1650 / RX 5500 XT class 4 GB GPU. A passing GI timestamp alone does not
establish that target: the complete application frame must fit 33.3 ms.
Development results on the RTX 4070 Laptop GPU are not target certification.

## Diagnosis

The shared sparse grid couples receiver coverage to work in the original
implementation. Lowering `max_rays` also coarsens that grid. Reconstruction
cannot recover a surface that has no compatible current sample or history.

The transport audit also found independent lighting defects:

- Direct emitter/environment lighting could be discarded when the indirect
  cache had no support. Direct and indirect validity must remain distinct.
- Static surface-owned receivers repeated a spatially keyed sample sequence
  instead of collecting new directions during refinement.
- Outgoing radiance was clipped to a luminance limit tied to exposure, biasing
  bright emitter and HDR paths before reconstruction.
- Fast irradiance-probe interpolation used a receiver-plane test without
  sufficient geometry/visibility validation for nearby thin surfaces.

The project already uses Bevy 0.20 and wgpu 30. Historical adapter logs show
Vulkan. Vulkan is a [backend of wgpu](https://wgpu.rs/doc/wgpu/enum.Backend.html),
alongside DirectX 12 and other graphics APIs. These findings
do not establish an upstream Bevy defect. A backend-specific failure needs a
completed reproduction with adapter, driver, shader path, and validation output.

## Implementation details recorded in this validation history

The low-end preset separates receiver coverage from per-frame trace work: a
160×90 candidate grid is serviced over time, while at most 1,024 receiver
updates and 16,384 admitted traversal queries are scheduled in one frame. Cache
table sizing and per-bucket admission honor the configured retention limit;
`RadianceCascades::low_end()` uses 65,536 records with 1 m base probes and a
1.6 m cascade-zero cutoff. Its Split gather uses the cosine-prefiltered interval
cache, with rays starting at the biased surface rather than a separate c-1
Monte Carlo correction. At that revision, all six methods could resolve into
640×360 storage for a 1280×720 view; the four irradiance and five temporal metadata images then used
about 31.6 MiB, before scene data, the receiver identity image, or the optional
depth pyramid. The pyramid is allocated only for methods that actually screen
trace.

The raw resolve now reconstructs direct and indirect lighting with separate
support before combining them for temporal accumulation. This keeps direct
lighting when an indirect cache is empty and preserves supported black samples.
Temporal reprojection and denoising use the configured resolve scale. The
traversal diagnostic allocation also covers every counter the shader writes.
These are structural fixes. The reduced-resolution runs below check transport
dispatch, raw-output readiness, and timing; reference-image comparisons are
still required to establish image quality.

Progressive receiver updates now use a bijective shear ordering of the
rectangular candidate grid. Each frame still obeys its exact update budget, and
every candidate is selected once during each finite schedule cycle. The
per-view cycle phase is independent of the wrapping render-frame counter and
advances only after a frame is submitted. This removes the horizontal bands
caused by selecting contiguous flattened grid indices. For the 160×90 grid
with a 1,024-update budget, each full batch touches every 8×8-pixel tile row
and column, covers at least 216 of 240 coarse tiles, and places no more than
eight candidates in any one tile.

No additional rendering crate was needed. Bevy already exposes the wgpu
compute, storage-buffer, texture, and render-graph facilities used here. The
missing piece was bounded persistent priority scheduling, which remains an
application-level work queue rather than a reason to add another dependency.

The Sponza development benchmark overrides the shared-scene cap to 2 GiB and
the material-atlas cap to 768 MiB so its supplied 512-pixel materials and
roughly 0.97 GiB candidate snapshot can become resident with replacement
overlap. This scene-specific configuration is above the library's 4 GB low-end
defaults and does not certify target-device memory use.

## Reproducible checks

On Windows, direct temporary files to a writable workspace directory:

```powershell
$env:TEMP = "$PWD/tmp"
$env:TMP = $env:TEMP
$env:BEVY_PENUMBRA_SHADER_CACHE_DIR = "$PWD/tmp/gi_shader_cache"
cargo test --locked --offline --lib
$env:WGPU_BACKEND = 'vulkan'
cargo test --locked --offline --lib -- --ignored --test-threads=1
```

Repeat GPU checks with `WGPU_BACKEND=dx12`. Run one GPU benchmark at a time.
Record actual enabled device features and shader format; selecting DX12 does
not establish that the same native shader path was exercised as Vulkan.

Quality comparisons must use scene-linear, exposure-independent output with
artistic occlusion disabled and matched material/bounce semantics. Use an
uncached reference estimator, analytical diffuse/environment cases, and an
independent small-scene intersection oracle. Tonemapped PNG stability does not
establish energy conservation, coverage, or convergence.

Separate missing samples from supported black samples. Test empty caches with
direct lighting, bright HDR sources, static refinement, thin walls and folds,
alpha masking, disocclusion, moving blockers, lighting steps, resizing, and
multiple views. Measure response in elapsed time, rather than only frame count.

## Acceptance gates

- Analytical lighting energy error at most 5%.
- At least 25% less aggregate reference error than the matched starting build,
  with no increased geometry leakage.
- Lighting-step response within 100 ms on previously supported surfaces.
- p95 complete frame time at most 33.3 ms on the target hardware; 16.7 ms is the
  provisional GI allocation within that frame.
- At most 1 GiB of aggregate GI allocations, including replacement overlap.

The archived verification plan compared methods at equal measured work and GPU time. It included direct
shadow rays, secondaries, validation, screen fallback, reconstruction, and CPU
preparation. Missing timestamp support must be reported as unavailable.

## Starting verification

Before this implementation, the ordinary library suite passed 65 tests with 11
ignored GPU/compiler checks. All 11 ignored checks passed serially on Vulkan.
Starting sources and the existing development executable were saved under
`tmp/gi_quality_baseline/` in that run; the bundle is not present in this
checkout. Its captures must not be presented as fresh release-build or
post-change results.

At that earlier stage, no method had been shown to be inherently incapable of
improvement, and the project retained six selectors pending controlled
comparisons. That selector API was later removed. The diagnosed issues fit the
existing crates; no additional rendering crate was warranted.

### Release baseline on the development GPU

The saved starting sources were also compiled in release mode. Only the
benchmark harness was changed: the benchmark loop runs without a sleep and
discards 120 fresh GPU measurements before collecting 120 measurements.
Transport shaders and settings were retained. Asset and sibling-crate paths
were adjusted to run the isolated snapshot. Adapter logs confirm Vulkan on the
RTX 4070 Laptop GPU, NVIDIA driver 617.42.

| Method | Median GI ms | p95 GI ms | p95 GPU queue envelope ms |
| --- | ---: | ---: | ---: |
| Hybrid SPWI | 30.566 | 35.313 | 57.936 |
| PAIC | 21.977 | 23.725 | 46.514 |
| Surface PAIC | 22.783 | 22.865 | 44.378 |
| Surfel RC | 21.942 | 22.135 | 44.080 |
| Split | 65.766 | 69.048 | 91.107 |
| Cascade Guided | 267.980 | 283.389 | 305.823 |

These runs use 1280×720 and `--max-rays=1024`. The starting Surface PAIC and
Surfel RC settings still cap effective updates at 256; the other methods use
their legacy grid sizing. A nominal setting does not establish equal measured
traversal work. The queue envelope includes GPU uploads and idle gaps, and
excludes CPU extraction and presentation. It is not complete application FPS.
Machine-readable results and per-pass logs were saved under
`tmp/gi_quality_baseline/release_benchmarks.json` in the original run. Those
figures are historical and predate the current implementation.

### Reduced-resolution post-change smoke

On 2026-10-14, five transports completed 120 warmup and 120 measured GPU
samples on the RTX 4070 Laptop GPU using Vulkan. The adapter reported
`passthrough_shaders`; the benchmark infers the Slang-to-SPIR-V passthrough path
from that feature because the actual output format is not directly exposed.
These diagnostic runs used Sponza at 320×180, grid divisor 16, a 256-ray
ceiling, and resolve scale 2. They are not directly comparable with the
1280×720 historical baseline, nor do they certify target-device performance.

| Method | Median GI ms | p95 GI ms | p95 Penumbra GPU frame ms |
| --- | ---: | ---: | ---: |
| PAIC | 16.275 | 17.680 | 28.110 |
| Surface PAIC | 21.934 | 23.254 | 33.590 |
| Surfel RC | 23.882 | 25.141 | 35.260 |
| Hybrid SPWI | 23.034 | 24.970 | 41.480 |
| Split | 39.825 | 43.142 | 53.632 |
| Cascade Guided | not measured | not measured | not measured |

Cascade Guided submitted a frame but did not produce a benchmark result in the
run window. Its shader and pipeline initialize successfully; raw GI readiness
was the remaining unknown. The Sponza benchmark now emits bounded raw-readback
health logs to distinguish an unsupported/black result from slow progress.
Reduced-run JSON and logs are under `tmp/quality-vulkan-small/`.

### Native shader optimization and surface reconstruction

All runtime stages start with Slang's O2 WGSL output. Vulkan validates it with
Naga and emits bounds-checked SPIR-V using the device binding map, retaining
helper calls to bound NVIDIA driver compilation. The direct Slang SPIR-V
compiler, manual resource relocation, and entry-point parser have been removed. Expensive
visibility and guided-path loops remain loops instead of explicitly expanding
the BVH/material call graph at every sample. Shader cache keys include the
source, compiler, target, and optimization level; driver caches remain separate.

At that revision, Sponza adapted only after GI dispatches and prewarming completed, using top-level
GPU work timings without counting their nested spans twice. CPU frame intervals
and queue envelopes can include VSync wait and must not reduce GI work.

Final composition checks raster receiver identity, geometric normal, and world
plane distance with a curved-surface allowance. When bilinear support is empty,
it may use the nearest compatible sample inside the same four-tap footprint.
The optional surface-group constraint remains enforced. This addresses nearby
surface leakage and zero-weight support holes without increasing ray counts.

CPU scene updates combine pending-receiver eligibility and component-change
detection in one geometry pass, and reuse light-vector capacity on unchanged
frames. BLAS construction remains on Bevy's `AsyncComputeTaskPool`; traversal
and filtering remain GPU work. No additional async runtime was introduced.
CPU savings from these changes have not been timed independently.

Sponza offscreen captures now use GPU completion for back-pressure instead of
a fixed 250 ms frame delay. `--warmup-frames=N` also controls image capture
refinement. Benchmark JSON reports the configured scene, atlas, and view budgets
from the same constants used to configure the plugin. Earlier JSON files that
reported 256 MiB scene/atlas budgets were incorrect: these runs configured
2 GiB for the scene, 768 MiB for the atlas, and 512 MiB per view.

#### Matched Sponza measurement, 2026-10-14

RTX 4070 Laptop GPU, Vulkan, driver 617.42; optimized development profile
(crate O1, dependencies O3). Both runs used Hybrid SPWI, 1280×720, grid divisor
8, 1,024 requested updates, one near-field sample, resolve scale 2, and 120
warmup plus 120 measured GI samples. A saved baseline executable was run before
changing native shader optimization. It already includes the geometry-pass
fusion, so these numbers do not isolate that CPU improvement.

| Metric | Baseline | Final |
| --- | ---: | ---: |
| Median GI GPU time | 16.338 ms | 11.787 ms |
| p95 GI GPU time | 17.614 ms | 12.805 ms |
| Median Penumbra GPU frame envelope | 39.421 ms | 33.322 ms |
| p95 Penumbra GPU frame envelope | 41.609 ms | 36.401 ms |

Median GI time decreased 27.9%; the frame envelope decreased 15.5%. This is
one matched workload on one laptop, not a 60 FPS Sponza result. The final
per-pass snapshot reports merge at 0.941 ms versus 3.253 ms before, and the
three denoise passes total 0.319 ms versus 0.887 ms. Per-pass snapshots are not
percentile statistics.

This command records the previous Hybrid CLI and is not accepted by the current
Split-only executable:

```powershell
cargo run --example sponza -- --benchmark --method=hybrid --width=1280 --height=720 --warmup-frames=120 --measured-frames=120 --json-output=captures/sponza_quality_performance_final.json
```

Raw results are in `captures/sponza_before_hybrid.json`,
`captures/sponza_quality_performance_final.json`, and
`captures/gi_optimization_comparison.json`. The associated logs include adapter
details, pipeline initialization, readiness, and GPU pass diagnostics.

An interactive Cornell capture reached 600 GI frames at 60.0 FPS, with a
5.486 ms GI GPU timestamp and a physical target of 1500×1125 (1200×900 logical
pixels at 125% desktop scaling). See `captures/cornell_quality_60fps.png` and
its log. The 88 non-ignored library checks and the focused production-compositor
GPU regression passed. The GPU regression covers receiver separation, parallel
planes, zero-weight support recovery, curved surfaces, and optional group
separation. These checks do not establish a reduction in reference-image energy
error; no new such measurement is claimed.

Sponza day, GI-off, and night captures completed under
`captures/sponza_quality_final_*.png`, with 600 ready GI frames before each lit
capture. Their readiness and day/night/GI-toggle assertions passed. The saved
baseline images used the older 32-frame readiness gate, so the image pair is
not a matched convergence comparison.

### Progressive-schedule validation

After the scan fix, Hybrid SPWI completed a second Sponza run at 640×360 on
2026-10-14. It used an 80×45 candidate grid, 1,024 receiver updates per frame,
resolve scale 2, and 120 warmup plus 120 measured GPU samples. Vulkan adapter
and driver details match the reduced-resolution smoke above. The final raw
readback reported 20,008 supported output pixels with a 165-frame ready
streak. Median and p95 GI times were 18.789 ms and 21.129 ms; p95 Penumbra GPU
frame time was 41.553 ms. This verifies startup and supported transport after
the schedule change, but it is not a matched before/after image-quality test or
a target-device performance certification. The JSON and log are under
`tmp/quality-vulkan-small/scan-fix/`.

# Validation

## Hardware ray-query traversal — October 29

Same device, driver, scene and settings as the section below. With
`hardware_ray_queries`, `traceDeferredHardware` and `traceVisibilityHardware`
walk a hardware TLAS/BLAS; every candidate triangle is re-tested by the
software triangle, culling, layer and alpha-coverage code.

| Traversal | GI p50 / p95 | Penumbra GPU frame p50 / p95 | Frame interval p50 / p95 | Trace / visibility |
| --- | ---: | ---: | ---: | ---: |
| Software (inlined, fused merge), fast mode | 9.61 / 12.18 ms | 17.64 / 20.26 ms | 17.91 / 20.85 ms | 0.83 / 0.62 ms |
| Hardware | 8.15 / 8.36 ms | 16.03 / 16.31 ms | 16.07 / 17.31 ms | 0.15 / 0.08 ms |

The hardware run supported all 129,542 eligible raw pixels and showed no slow
mode. It meets the 16.7 ms GPU-frame p95 in this run; the frame-interval p95
(17.31 ms) does not. Cold pipeline creation took 10 s and 46 s for the two
hardware kernels; later starts load the driver pipeline cache.

At pose 127, 2,048 updates and 512 warmup frames, matched raw stability
captures compare as follows. Hardware differs from software by no more than
two software runs differ from each other:

| Pair | Raw relative L2 | PNG mean | Coverage |
| --- | ---: | ---: | ---: |
| Software vs software repeat | 0.0833 | 0.56/255 | 100% |
| Software A vs hardware | 0.0842 | 0.62/255 | 100% |
| Software B vs hardware | 0.0801 | 0.59/255 | 100% |

Records: `tmp/perf2/hw1.json`, `tmp/rq/q_*.json`.

## Traversal inlining and fused child merge — October 29

RTX 4070 Laptop GPU, driver 621.01, Vulkan, 1920×1080, balanced assets and
shadows, 2,048 updates, divisor 4, 120 warmup and 240 measured samples.

Vulkan SPIR-V previously marked every helper `DontInline`. In the
traversal-only kernels (`traceDeferred` and `traceVisibility`), that kept the
traversal state in call frames on every step. Those two kernels now leave
inlining to the driver. Kernels with hit shading keep helper calls, because
inlining that call graph takes the NVIDIA compiler minutes. With
per-dispatch timestamps, closest-hit trace fell from 4.47 to 0.83–0.86 ms and
visibility from 3.66 to 0.55–0.68 ms.

Merge interpolated each of its four child directions once per transport
source, repeating the same eight hash-chain lookups three times. One walk now
feeds every active source. Per-source support, weights and light-hit bounds
keep their original accumulation order; dependency boxes are order-independent
unions. Merge levels 0–2 fell from about 2.3 ms to about 1.6 ms.

| Build | GI p50 | Penumbra GPU frame p50 |
| --- | ---: | ---: |
| Before | 16.59 ms | 24.66 ms |
| Inlined traversal, fast mode | 9.98–10.69 ms | 17.98–18.80 ms |
| Plus fused merge, fast mode | 9.61 ms | 17.64 ms |
| Either build, slow mode | 12.24–12.54 ms | 20.29–20.49 ms |

Runs alternate between two stable modes with identical workload counters
(2,048 receivers, about 6,100 admitted queries). In the slow mode only the two
traversal kernels lengthen (trace about 3.1 ms, visibility about 1.3 ms), while
SM and memory clocks stay at 2,625 / 8,001 MHz. The cause is not established;
buffer residency or placement is a hypothesis. The 16.7 ms frame target is
still not met. Records are under `tmp/perf2/`.

## Converted daytime assets and presentation pacing

The current local cook uses indexed float-attribute glTF and 512-pixel PNGs;
it contains 6,276,355 triangles and 34,369,063 PNG bytes versus 10,809,942
triangles and 2,180,411,830 PNG bytes in the source. These are disk counts,
not GPU-residency measurements. The source and generated bundles remain
separate; `--asset-profile=balanced` selects the generated files.

On the RTX 4070 Laptop GPU, NVIDIA driver 621.01, Vulkan, 1080p, 2,048 updates
and balanced shadows, the second pair of 120-warmup/240-sample screens measured:

| Asset profile | GI | Frame p50 / p95 |
| --- | --- | --- |
| source | off | 8.662 / 9.325 ms |
| converted | off | 6.246 / 6.880 ms |
| source | on | 33.708 / 37.073 ms |
| converted | on | 30.196 / 33.881 ms |

These are independent screens, not 60 FPS certification. With converted
assets, Split GI itself measured 21.695 ms p50 and 25.418 ms p95, including
13.792 / 17.571 ms of tracing. `--no-vsync` requests AutoNoVsync for interactive
windows; actual platform fallback and presentation behavior are not recorded.
The active display adapter reported 60 Hz via Windows. A separate 600-sample
converted GI-off screen requested AutoVsync and measured 16.649 ms median;
the AutoNoVsync-requested screen above measured 6.246 ms. This is consistent
with display pacing, not proof of a particular compositor or swapchain mode.
One interactive AutoNoVsync-requested launch showed 33.3 FPS with GI on and
156.9 FPS after toggling GI off; these are snapshots, not distributions.

Five converted-versus-source daytime views passed the static image bounds at
8,192 updates and reference shadows, with raw relative L2 from 0.028 to 0.070.
The same five poses also passed with converted assets, balanced shadows and
2,048 updates: raw relative L2 ranged from 0.038 to 0.102. Reference captures
used driver 617.42 and converted captures used 621.01. These views do not
establish moving-camera or all-lighting fidelity. Records:
`tmp/format-converted-quality-pose*.json`, `tmp/format-combined-quality-pose*.json`,
`tmp/format-source-driver621-repeat/`, `tmp/format-converted-driver621-repeat/`
and `tmp/format-converted-vsync-no-gi.json`.

The raster textures remain RGBA8 in GPU memory; PNG compression reduces disk
size, not resident texel cost. BC7/BC5 KTX2 could reduce raster residency but
is not a drop-in here: the GI CPU atlas currently rejects block-compressed
images and Bevy 0.20 does not support the `KHR_texture_basisu` extension syntax.

## Daytime interactive workload

On the RTX 4070 Laptop GPU at 1920×1080, Vulkan and requested AutoNoVsync,
the original 8,192-update compact workload measured 49.284 / 52.034 ms
frame-interval p50 / p95. The current source-asset, balanced-shadow default
uses 2,048 updates and requested grid divisor 4. A 120-warmup, 240-sample
screen measured 31.997 / 35.034 ms frame p50 / p95 (31.3 / 28.5 FPS);
the GI-off control and complete distributions are in
`tmp/daytime-grid-4-2048/`. These are windowed screens, not interactive
AutoVsync measurements or 60 FPS certification. A separate default interactive
launch displayed 34.1 FPS in its window title at one observed moment; this is
not a frame-time distribution.

The 2,048-update setting uses the same trace estimator on an 8×8 receiver
grid; larger requested update budgets retain compact indirect dispatch.
At 1,024 updates and divisor 12, the grid screen measured 24.655 / 27.875 ms
versus 34.801 / 46.648 ms with compact dispatch in a separate screen. The
dispatch cutoff is based on requested updates, not an exact GPU selection count.
At higher settings compact remains available with `--max-rays=8192`.

Five daytime source/reference-shadow captures at 2,048 updates, divisor 4,
and 512 warmup frames passed the static image bounds. Raw relative L2 ranged
from 0.047 to 0.091, with at least 99.94% common eligible coverage. They do
not prove motion quality at this lower update rate. Divisor 12 was not selected:
its 2,048-update screen reached 28.992 / 33.296 ms, but pose 127 had raw
relative L2 0.159 versus the 0.15 limit. A same-settings reference repeat
at pose 127 scored 0.046. Records are under `tmp/day-grid-*`.

An earlier reduced asset cook lowered the early deferred prepass median from
about 2.55 ms to 1.88 ms and the receiver-identity median from 1.40 ms to
0.95 ms in those screens. Full-frame p95 did not improve in those two cooked
runs, so the default remained source assets. Those independent runs do not isolate
the cause of the full-frame difference.

## 1080p balanced-profile screening

Windows, RTX 4070 Laptop GPU, driver 617.42, Vulkan, native 1920×1080,
AutoNoVsync requested, 120 warmup and 240 measured samples. These are screening
runs, not 1,800-sample certification. The source asset profile was retained:
the candidate cook and its preservation fallbacks failed the midnight
scene-linear fidelity bound. The default sun uses two 2048-pixel cascades over
40 m; five fixed GI-off day/night views met the image bounds. At 1024 pixels,
the first daytime view visibly lost architectural shadows.

The following workload screen used the original, non-compact trace dispatch:

| Requested grid divisor | Receiver updates | GI-on frame p50 / p95 | GI-off frame p95 |
| ---: | ---: | ---: | ---: |
| 4 | 8,192 | 52.851 / 58.765 ms | 16.917 ms |
| 4 | 4,096 | 37.251 / 41.664 ms | 16.951 ms |
| 4 | 2,048 | 30.583 / 34.094 ms | 16.982 ms |
| 4 | 1,024 | 26.738 / 29.385 ms | 16.973 ms |
| 12 | 8,192 | 56.023 / 62.281 ms | 16.947 ms |
| 12 | 4,096 | 34.572 / 38.205 ms | 16.943 ms |
| 12 | 2,048 | 28.274 / 32.224 ms | 16.981 ms |
| 12 | 1,024 | 24.792 / 27.854 ms | 16.931 ms |

None met the 16.667 ms frame gate. At the time the demo retained 8,192 updates,
requested divisor 4 and resolve scale 4. At 1080p the receiver cap produces
32,400 candidates (240×135); the raw resolve was 480×270. The lowest-budget
divisor-12 run reported 14,400 candidates, 1,024 selected updates, 3,085
admitted queries, 99.915% eligible support, 475,010,056 live view bytes and
1,632,938,912 live scene bytes. Query counts and memory are endpoint diagnostics,
not distributions. A requested cap of 131,072 is not an executed-ray count.

Compact trace dispatch remains enabled for larger workloads, following explicit
acceptance of its unresolved quality risk. It reduced median direction-generation, trace and argument-copy
mean cost by 14.4% across three source/reference runs. Its midnight raw
relative-L2 error was 0.659 against the 0.15 bound; an unchanged-reference
repeat scored 1.457, demonstrating baseline variability but not equivalence.
With balanced shadows, a later compact 8,192-update run recorded 49.284 ms
frame p50 and 52.034 ms p95; GI-off frame p95 was 17.018 ms. These do not
meet the 60 FPS gate.

The initial reduced asset cook had 6,887,437 triangles versus
10,809,942 source triangles and 34,369,063 bytes of PNGs versus 2,180,411,830
source bytes, but its midnight relative L2 was 1.754. The ivy, curtains,
full-geometry and source-texture fallbacks also failed that bound. The
near-black midnight baseline itself failed the visible-contribution assertion,
so these failures do not identify a specific asset or renderer defect.

The PNG texture loader caches complete color-space-specific mip chains, using
older base-level entries to build the new cache on first use. One warm 512-pixel
run loaded 80 textures from cache with no PNG decodes in 1.457 seconds; an
earlier warm base-level-cache run took 1.590 seconds, so the separate runs do
not establish a reliable startup speedup. The resident 256-pixel screen had
15.905 ms queue p95, equal at displayed precision to the final 512-pixel run;
the demo retains 512 pixels. This cache change does not reduce GPU texture
residency or demonstrate a steady-state frame-time improvement.

The source/reference moving-blocker baseline stopped at an extinguished-light
residual of 33.6% versus the 5% limit. This measures encoded submissions, not
GPU completion or displayed latency. No 60 FPS or complete quality
certification is claimed. Records are under `tmp/rendering-before-*`,
`tmp/rendering-compact-*`, `tmp/assets-*-quality-0-0.json`,
`tmp/shadow-balanced-benchmark/` and `tmp/workload-*/`.

## Active CPU and Split trace profiling

The Sponza benchmark now records the Split trace pass distribution in
`samples.trace_gpu_ms`. It uses independently delivered GPU diagnostic samples
within the benchmark window; its sample count can differ from the GI total.
The trace pass includes traversal, hit shading, visibility and cache deposits;
the timestamp does not separate those costs.

On Windows, RTX 4070 Laptop GPU, Vulkan driver 617.42, release Sponza at
1920×1080, 8,192 updates and a 131,072-query cap, these separate windowed runs
requested AutoNoVsync, discarded 120 warmup samples and measured 240 GI samples.
The original trace run included CPU thread sampling; the changed run did not.
Conditions and scene readiness can vary.

| Run | Trace GPU p50 / p95 | Total GI GPU p50 / p95 | Frame interval p50 / p95 |
| --- | --- | --- | --- |
| Before shader change | 34.470 / 39.143 ms | 42.879 / 47.268 ms | 58.192 / 63.188 ms |
| After shader change | 34.811 / 39.475 ms | 42.929 / 47.308 ms | 58.358 / 63.815 ms |

The trace pass is the largest measured GI component. Both runs supported all
129,542 eligible raw pixels at readiness. The change removes UV and color
interpolation from the hit-attribute structure where the results were unused;
material sampling still interpolates them where needed. Receiver queueing also
reuses its entity lookup allocation and extraction avoids a duplicate lookup.
These runs do not demonstrate a speedup, and the 60 FPS target is not met.
Records: `tmp/cpu-sampled-baseline-gi.json` and `tmp/cpu-perf-optimized-gi.json`.

Process CPU clocks, sampled separately during warmed measurement windows,
showed 15.375 CPU seconds over 10.399 wall seconds with GI and 7.906 CPU
seconds over 6.516 wall seconds without GI: approximately 1.48 and 1.21
logical cores respectively. CPU time is summed across threads and may overlap
GPU work. The GI-on run's busiest thread accumulated 8.203 CPU seconds during
the 10.399-second interval. These are separate runs and do not isolate the
cost of a particular Bevy system.

An additional run sampled user-mode instruction pointers of the four most
active threads approximately every 45 ms, weighting snapshots by intervening
thread CPU cycles. The busiest thread was often sampled in
`RtlQueryPerformanceCounter` and NVIDIA's `nvoglv64.dll`; this is consistent
with polling, but snapshots do not provide call stacks or identify the caller,
and interval weighting can attribute earlier work to the sampled endpoint.
The sampling itself can perturb timings. WPR CPU tracing was also attempted but
Windows rejected the profiling policy with `0xc5585011`, so no stack-based
attribution is available. The CPU samples are in
`tmp/cpu-samples-baseline-gi.json`; process-clock runs are
`tmp/cpu-profile-named-gi.json` and `tmp/cpu-profile-nogi-extended.json`.

A 320×180 multi-bounce capture completed day, GI-off and night stages with
visible GI contribution (`tmp/cpu-perf-smoke_{day,no_gi,night}.png`); this is
not a proof of radiometric equivalence across scenes or backends.
The workspace all-target suite passed 124 tests, all 26 ignored Vulkan
shader/GPU regressions passed, and strict all-target Clippy passed.

## Base rendering with inactive GI

When no active camera has positive GI intensity, scene synchronization and
receiver-identity extraction stop doing per-frame mesh work. Re-enabling GI
revalidates referenced assets and pending BLAS work, including assets that
were absent when the scene resumed. Existing allocations may remain resident
after a toggle; a scene started with GI disabled does not build the GI scene.
Rasterization, GI-on sampling and Bevy's deferred prepass are unchanged.

Release Sponza measurements at 1920×1080 on an RTX 4070 Laptop GPU used Vulkan,
windowed rendering with `AutoNoVsync` requested, 120 warmup samples and at
least 240 measurements. These are separate runs; scene readiness and scheduling
can vary. The baseline predates this optimization.

| Run | Frame interval p50 / p95 | Main schedule p50 / p95 |
| --- | --- | --- |
| GI disabled, before | 16.664 / 16.964 ms | 1.955 / 2.475 ms |
| GI disabled, after | 16.663 / 17.435 ms | 1.702 / 2.417 ms |
| GI enabled, after | 58.725 / 64.962 ms | See benchmark record |

The inactive path removes work but these runs do not demonstrate a reliable
frame-rate improvement, and the GI-on 60 FPS target remains unmet. The
GI-disabled run did not construct or upload a GI scene. Records:
`tmp/60fps-no-gi-1080.json`, `tmp/base-final-no-gi.json`, and
`tmp/base-final-gi.json`.

The earlier 16.6 ms GI-disabled GPU figure is a **queue-time envelope**, not
proof of 16.6 ms of active GPU work: timestamps are submitted separately and
can span CPU preparation or presentation waits. Requested `AutoNoVsync` does
not establish the backend's selected present mode. A new
`--benchmark-offscreen-no-gi` diagnostic mode removes the window but explicitly
drains the GPU; its 17.868 ms median queue envelope
(`tmp/base-final-offscreen-no-gi.json`) is not directly comparable to
windowed throughput or active GPU execution. Bevy 0.20's built-in occlusion
culling documents unspecified behavior with the required deferred prepass, so
it was not enabled for GI cameras.

The final workspace suite passed 100 tests, all 26 ignored Vulkan GPU
regressions and 14 Sponza example tests passed, and Clippy passed. A 320×180
Sponza multi-bounce capture passed its GI-on, GI-off and resumed night stages.
These checks do not establish
visual equivalence for every scene or backend.

## 1080p traversal and environment optimization

Closest-hit and shadow traversal now reuse each instance's inverse-transform
rows for both origin and direction. HDR environment sampling reuses its mapped
UV for radiance and PDF evaluation, reuses an alias record when possible, and
defers non-alias texture reads for directions rejected by the surface. Ray
selection, budgets, intersection tests, and lighting estimators are unchanged.

The table records separate release, windowed, VSync-off Sponza runs on an RTX
4070 Laptop GPU with Vulkan, 120 warmup samples and at least 240 measured
samples. GI-on runs use the same 1 m probes, 8,192 updates, and 131,072-query
cap. The pre-change row retains the environment optimization but uses the
original traversal transforms; the final row includes both optimizations.
Timings vary with scheduling and GPU conditions.

| 1920×1080 run | GI GPU p50 / p95 | Frame interval p50 / p95 |
| --- | --- | --- |
| GI disabled | — | 16.664 / 16.964 ms |
| Original traversal | 42.346 / 46.775 ms | 57.506 / 62.682 ms |
| Final traversal | 42.549 / 46.103 ms | 57.983 / 62.093 ms |

The 60 FPS target is not met and these runs do not establish a 1080p speedup.
The final run supported all 129,542 eligible raw pixels at readiness. At
640×360 offscreen with 1,024 updates and a 16,384-query cap, GI p50 was
12.872 ms before the transform change and 12.669 ms after; this small
difference is not sufficient to establish a robust speedup. Records are
`tmp/60fps-{no-gi-1080,paired-before-1080,final-1080,paired-before-low,final-low}.json`.

The Vulkan GPU suite passed all 26 regressions, including brute-force traversal
comparisons and rotated HDR environment PDF and radiance checks. The workspace
suite passed 97 tests; Clippy passed. A 320×180 Sponza multi-bounce day/GI-off/
night capture passed. Neither that capture nor support counts establish
radiometric equivalence across all scenes or backends.

## Paper-style cached hit sky

`cached_hit_sky` (Sponza `--cached-hit-sky`, requires `--multi-bounce`) follows
the paper's multibounce shading: secondary-cache rays that escape carry the
environment radiance, and primary and secondary ray hits covered by that cache
take sky light from it instead of tracing environment visibility rays. Hits
without secondary coverage, including the cold cache, keep visibility-tested
environment sampling. The cache's sky term is diffuse-only and one frame late;
GGX sky reflection at hit points is dropped.

Release Sponza, Vulkan, RTX 4070 Laptop GPU, 1920×1080 windowed AutoNoVsync,
balanced asset and shadow profiles, divisor 4, 2,048 updates, 131,072 queries,
`--multi-bounce`, 120 warmup and 600 measured samples, alternating runs from
one binary:

| Run | GI GPU p50 / p95 | Penumbra frame p50 | Admitted queries |
| --- | --- | --- | --- |
| Sky NEE, run 1 | 20.310 / 21.316 ms | 28.300 ms | 3,011 |
| Sky NEE, run 2 | 20.335 / 21.445 ms | 28.361 ms | 3,024 |
| Cached sky, run 1 | 19.005 / 19.876 ms | 27.014 ms | 2,102 |
| Cached sky, run 2 | 18.962 / 20.409 ms | 27.008 ms | 2,108 |

GI GPU p50 fell by about 1.34 ms (6.6%); the secondary trace pass and the
primary visibility pass account for most of it. Matched 160×90 raw stability
captures (16 frames) against sky NEE gave relative L2 0.133 and a −1% to −3%
per-channel mean bias; two sky-NEE runs differed by relative L2 0.171 and
+1.8% to +2.3% bias, so the difference is within run-to-run variation at this
sample count. Tone-mapped PNGs differed by 1.49/255 mean absolute error. This
is not a reference comparison. Records are `tmp/perf-pass/sky_mb_*.json` and
`tmp/sky-cache/`.

## Optional secondary cache and traversal pass

The optional `multi_bounce` path seeds a separate, two-LOD-coarser cache
from previous primary ray hits. Both primary and secondary hit shading read
the previous completed secondary field. Adjacent secondary LODs now blend
their independently supported source values over the LOD overlap, instead of
switching abruptly. The lookup retains the reference implementation's nearest
coarse probe; unsupported or invalidated observations remain excluded.

Traversal compares complete layer-mask words together, avoids reading an
immutable geometry page when an animated overlay supplies the word, and defers
the geometry header until the root bounds are hit. These preserve the selected
words and intersections. The Vulkan GPU regressions, including traversal,
secondary generation/invalidation, seed/clear, and LOD-overlap cases, passed
(26 ignored GPU tests). The ordinary workspace suite passed (97 tests), as
did Clippy.

An isolated release Sponza run at 640×360, divisor 8, 1,024 updates and
16,384 traversal queries used Vulkan on an RTX 4070 Laptop GPU, 120 warmup
samples and 240 measured samples. The earlier baseline below is from the
preceding vector-finiteness pass; it was not collected in the same session.

| Run | GI GPU p50 / p95 | Supported / eligible at readiness |
| --- | --- | --- |
| Earlier baseline | 12.770 / 13.868 ms | 14,335 / 14,396 |
| This pass | 12.763 / 13.915 ms | 14,335 / 14,396 |

This comparison does not demonstrate an end-to-end speedup; its difference is
small relative to run-to-run variation. Records are
`tmp/perf-final-low.json` and `tmp/current-traversal-isolated.json`.
At 320×180, separate Sponza day/GI-off/night captures passed both with
multi-bounce enabled (1,024 total updates, 16,384 total queries) and disabled
(512 updates, 8,192 queries), matching the primary quota. The daytime RGB
means were 32.543 and 29.636 respectively. The multi-bounce run dispatched
the secondary trace and both captures reached 3,501/3,599 supported pixels.
These independently accumulated, tone-mapped captures are consistent with
visible feedback, but do not isolate it from sampling differences or establish
radiometric accuracy or convergence against a reference.

The remaining dated sections describe the state at their stated revisions;
statements that multi-bounce was absent are historical.

## Vector finiteness and sparse-cache performance

Slang lowered vector `isfinite` checks into dynamically indexed component
loops. Branchless IEEE-754 exponent tests preserve the classification of
finite values, infinities, and NaNs without those loops. Resolve computes its
octahedral lookup once per receiver, while merging, retention, probe packing,
material sampling, and scene repacking avoid redundant work. Independent source
support, history dependencies, and traversal limits remain in force.

The following release Sponza runs used Vulkan on the RTX 4070 Laptop GPU
with Slang → WGSL → Naga SPIR-V, 120 warmup samples, and at least 240
measured GI GPU samples. The before runs use the saved
`tmp/perf-autotune-stack.exe`; the after runs use the updated source.
Per-frame scheduling and scene readiness can vary between runs.

| View / receiver updates / traversal cap | Before GI p50 / p95 | After GI p50 / p95 | Supported / eligible at readiness |
| --- | --- | --- | --- |
| 640×360 offscreen, 1,024 / 16,384, divisor 8 | 24.671 / 26.153 ms | 12.770 / 13.868 ms | 14,335 / 14,396 in both |
| 640×360 offscreen, 8,192 / 131,072, divisor 4 | 62.466 / 64.742 ms | 46.837 / 48.894 ms | 14,396 / 14,396 in both |
| 1280×720 windowed, 8,192 / 131,072, divisor 4 | 48.856 / 60.359 ms | 34.457 / 42.045 ms | 57,570 / 57,580 in both |

The two offscreen GPU queue-envelope p95 values after the changes were 35.195 ms
and 70.065 ms. The windowed run requested VSync off and measured a 56.691 ms
p95 frame interval after the changes, above the 33.3 ms target. These results
do not certify target GPUs or radiometric equivalence. The measurements and
settings are in `tmp/perf-current-baseline{,-full,-window}.json` and
`tmp/perf-final-{low,full,window}.json`.

The 94 ordinary library tests and all 24 ignored GPU/compiler regressions
passed on Vulkan, including fractional support, octahedral seams, scene edits,
and traversal. The DirectX 12 composite and transport-helper regressions passed,
and the retention pipeline compiled; its pre-change shader failed FXC loop
unrolling. The full DirectX 12 suite was not completed: FXC trace-pipeline
compilation exceeded 180 seconds and the merge regression exceeded 120 seconds.
Selecting the installed SDK's dynamic DXC failed while requesting a device
(`Device(Lost)`). These checks do not certify the DX12 path.

## Deferred BVH branch timing

Closest-hit traversal now retains the entry distance of each deferred BVH
branch. When a nearer hit shortens the ray, that distance decides whether the
branch is still eligible, without loading and testing its box again. Both the
top-level and instance walks retain their existing stack depth and overflow
fallback. Shadow traversal already avoids the repeated box test.

Two separate release, windowed Vulkan Sponza runs per version used an RTX
4070 Laptop GPU, 1280×720, VSync off, 8,192 receiver updates, a 131,072-query
cap, 120 warmup samples and 240 measured GI GPU samples. The same
57,570/57,580 eligible raw pixels were supported at readiness in all runs.

| Run | Before GI p50 / p95 | After GI p50 / p95 | Before / after GPU queue envelope p50 |
| --- | --- | --- | --- |
| 1 | 53.623 / 64.625 ms | 51.994 / 62.902 ms | 68.414 / 66.843 ms |
| 2 | 62.610 / 65.659 ms | 61.643 / 65.246 ms | 77.468 / 76.405 ms |

The approximately 1–1.6 ms median GI improvement is small relative to the
run-to-run variation. A separate 640×360 offscreen test with 1,024 updates and
a 16,384-query cap measured 25.743 / 27.329 ms before and 25.818 / 27.089 ms
after (GI p50 / p95), with 14,335/14,396 eligible pixels supported in both.
The records are `tmp/perf-autotune-window-{baseline,baseline-repeat,stack,stack-repeat}.json`
and `tmp/perf-autotune-low-{baseline,stack}.json`.

These runs do not establish the 30 FPS target or performance on the low-end
target GPUs. The deferred-node stack now uses two words per entry instead of
one, so register or local-memory pressure may differ on other devices. The
paper's OptiX tracing measurements are not directly comparable to this crate's
software BVH.

## Paper-guided performance pass

The paper's Section 6 uses workgroup-level reuse to preintegrate the 6×6
irradiance field. The probe-packing pass now shares its canonical-directory
lookup and the 32 direction vectors within each workgroup. It still keeps
background, emissive, and punctual support separate and preserves the
dependency bounds. The trace pass rejects unselected receivers before loading
the G-buffer, reuses per-ray triangle setup and traversal bounds, and avoids
repeating atlas metadata reads. Animation pose snapshots are moved into the
previous-frame state instead of cloned.

A prototype that cached the four child-direction lookups across lighting
sources was not retained: it did not improve the measured Sponza or Cornell
runs. Directly pre-averaging children as in the paper would also need to
preserve this implementation's independent fractional support, missing samples,
and conservative history dependencies; averaging conditional radiance alone
is not equivalent.

The following are separate release, offscreen-drained Vulkan runs on an RTX
4070 Laptop GPU with the Slang → WGSL → Naga SPIR-V path, 120 warmup samples,
and at least 240 measured GI GPU samples. Both versions used the same Sponza
scene, 640×360 viewport, and requested budgets. Asynchronous scene readiness
and temporal sampling can differ between runs.

| Receiver updates / traversal cap | Before GI p50 / p95 | After GI p50 / p95 |
| --- | --- | --- |
| 1,024 / 16,384, grid divisor 8 | 24.575 / 25.865 ms | 24.916 / 26.274 ms |
| 8,192 / 131,072, grid divisor 4 | 58.629 / 62.010 ms | 59.827 / 63.325 ms |

These runs did **not** establish an end-to-end GPU speedup. At readiness,
both versions supported 14,335/14,396 eligible receivers in the first setting
and 14,396/14,396 in the second. The daylight, GI-disabled, and night capture
gates passed after the changes. These are not matched scene-linear reference
comparisons, native-resolution frame-time certification, or evidence that the
previously documented dynamic-light extinction failure is fixed. The paper
uses OptiX for tracing and excludes G-buffer and direct-shadow costs from its
reported breakdown, so its timings are not directly comparable.

The final offline all-target Vulkan suite passed 142 tests, including the
ignored shader and GPU checks. Strict all-target Clippy and the six transport
metric-tool tests also passed.

Benchmark records are `tmp/perf-before.json`, `tmp/perf-before-default.json`,
`tmp/perf-atlas.json`, and `tmp/perf-atlas-default.json`; final captures use the
`tmp/perf-atlas_` prefix.

## Vulkan dynamic lighting, alpha coverage, and sub-grid receivers

On the RTX 4070 Laptop GPU, a 640×360 Sponza windowed, pipelined debug-profile
benchmark with 8,192 updates and a 131,072-query cap recorded 240 GI samples:
55.951 ms median and 64.505 ms p95 for GI, and 78.209 ms p95 frame interval.
All 14,396 eligible raw receiver samples were supported at readiness. This is
not a controlled before/after comparison and exceeds the existing frame-time
target. See `tmp/terra-improved-benchmark.json`.

The final moving-camera, moving-blocker, and dynamic-light capture retained
14,313/14,313 eligible raw samples at the light-off pose, but failed the
unchanged extinction gate: 41.7% residual versus 5% allowed. The logged
83.8 ms response is an encoded-submission interval, not GPU completion or
display latency. The failure occurs before the relight gate, so relight is
not verified. See `tmp/terra-final-motion.log` and `tmp/terra_final_motion_*.png`.

Vulkan GPU fixtures exercise masked and blended closest-hit/shadow traversal,
including fully transparent, fully opaque, and fractional texture alpha;
independent background retention on punctual changes; and a small receiver
without any compatible reduced-resolution screen sample. These checks do not
establish the outstanding Sponza response or performance targets.
The complete offline all-target Vulkan run passed 118 library and 24 example
tests, including ignored GPU fixtures; strict all-target Clippy and example
builds passed.

## Earlier source-separated response investigation

The current implementation retains directional/environment, emissive, and
finite-punctual radiance independently through interval merging, angular
prefiltering, and spatial/LOD reconstruction. These groups share the same
transmittance estimator. The Sponza profile requests 8,192 receiver updates,
131,072 traversal queries, and a 1 GiB view budget; the library low-end
profile is unchanged.

The probe directory now compares canonical keys without their ray-direction
bits. Before this fix, a traced nonzero direction could win the directory slot
but fail the subsequent packing lookup. The Vulkan GPU regression reproduced
the missing irradiance before the fix and passes afterward. In a stationary
Sponza capture, recorded packed probe count stabilized at 781 instead of
varying around 366–401.

On the RTX 4070 Laptop GPU with Vulkan, the 640×360 motion capture supported
14,313/14,313 eligible receivers after reset and at light-off. It still failed
the unchanged extinction gate: residual was 46.2%, versus the required 5%.
Pre-change midpoint-to-baseline scene-linear drift was 9.5%; the settled
midpoint-to-final drift was 3.8%. A longer-retention experiment measured 20.7%
residual after more than 1,000 dispatches per state; it was not adopted as the
default. These figures do not establish convergence or a 100 ms visible
response. Submission timestamps do not measure GPU completion or display
latency. See `tmp/terra-fixed-probe.log` and `tmp/terra-stable-beta.log`.

The Vulkan suite passed 119 library and 24 example tests, including the ignored
GPU fixtures, and strict all-target Clippy passed. Vulkan is the target backend
for this crate; DirectX 12 compatibility is outside the current scope.

Vulkan Cornell and Sponza static captures passed after the probe lookup fix.
Cornell had 38,398/38,400 sampled image pixels lit. Sponza's day, GI-off, and
night display means were 29.180, 7.545, and 0.720; the matched GI-on/off
difference was 21.654/255. The recorded day sample had 14,396/14,396 eligible
receivers supported. The Sponza static log reported a 57.42 ms Split GPU sample
at 640×360; this is not a native-resolution performance benchmark. Images were
inspected in `tmp/terra-probe-cornell.png` and `tmp/terra_probe_static_*.png`.
The camera-only motion and cache-revisit capture also passed: immediate return
and the final revisit differed from the initial image by 0.569/255 and
0.570/255 respectively, below the 8/255 gates; eligible support was 100% at
the captured return. See `tmp/terra-probe-camera.log`.

The following October 26 results describe an earlier implementation, not this
source-separated revision.

## Reference completion and runtime gates — October 26

The implementation was reviewed against the supplied PDF, parsed paper, and
Chandra source. This remains the strict diffuse base, not the optional
multi-bounce, glossy, or c−1 extensions.

Cold initialization now weights probe admission by projected receiver
population instead of treating every visible probe equally. Three bounded
rank phases build a logarithmic population histogram, admit denser probes,
then fill unused capacity from the existing fair window. The histogram reuses
parent-count scratch; no cache allocation, ray-budget increase, directional
sampling change, or fabricated support is involved. The GPU regression checks
density precedence, phase separation, budget limits, and unchanged warm/no-history
scheduling. A separate resolve case preserves partially supported black.

In the Vulkan 640×360 Sponza moving-blocker/light capture, with 512 receiver
updates, 16,384 traversal queries, and 120 warmup frames, the full reset at frame
128 previously supported 13,018/14,313 eligible receivers (90.95%). The weighted
admission run supported 14,306/14,313 (99.951%) at that reset and reached frame
192 with 14,313/14,313 supported. These are recorded samples, not a guarantee
of first-frame coverage in arbitrary scenes. The log is
`tmp/sol-weighted-motion.log`; captures use `tmp/sol_weighted_motion_motion_*.png`.

The lighting-response comparison now excludes unsupported observations rather
than rejecting an entire sample when a single receiver is missing. Each sample
must still have 95% support, and at least 95% of baseline-supported receivers
must remain comparable. The 100 ms response window, 5% extinction residual,
minimum 16 affected receivers, and measurable-contribution checks are unchanged.
This exposes a real unresolved acceptance failure: the recorded extinction
comparison at frame 192 reports 128.8% residual. No timed extinction or relight
acceptance is claimed from this run.

Immutable scene light and layer arrays are now shared by history snapshots,
removing per-view per-frame vector copies while preserving last-encoded
snapshot comparisons. `scripts/benchmark_sponza.ps1` consumes the current
schema-13 report and its `updates_per_frame` field instead of rejecting it
against the obsolete schema-11 names.

Verification passes all 139 Rust tests (116 library and 23 example tests),
including every ignored Slang/GPU fixture, plus the six Python transport-metric
tests. Strict all-target Clippy and formatting checks pass:

```sh
cargo test --offline --all --workspace --all-targets -- --include-ignored --test-threads=1
cargo clippy --offline --all --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
python -m unittest discover -s scripts -p test_measure_transport.py -v
```

The weighted-admission build also passed the 640×360 Sponza daylight/GI-off/night
capture, with display RGB means 28.210, 7.545, and 0.720 and GI-on/off difference
20.686/255. All 14,400 raw values were finite; the recorded daylight sample
supported 14,396/14,396 eligible receivers. The camera-return/cache-revisit
sequence passed: immediate-return RGB difference was 1.279/255 and the final
revisit was 1.851/255, below the unchanged 8/255 gates. Cornell at 640×480
reported 38,398/38,400 sampled image pixels lit. Images were inspected;
these are smoke and temporal-reuse checks, not radiometric accuracy. Logs are
`tmp/sol-final-{static,camera,cornell}.log`; images use
`tmp/sol_final_{static,camera,cornell}*`.

A release, windowed, VSync-off 1920×1080 benchmark on the RTX 4070 Laptop GPU
recorded 120 warmup and 240 measured GI samples. GI median/p95 was
15.554/17.837 ms; the GPU queue envelope p95 was 32.903 ms and the observed
frame-interval p95 was 34.211 ms. The JSON is `tmp/sol_native_1080.json`.
This benchmark includes the snapshot-sharing change but predates weighted cold
admission. It does not pass the script's unchanged 16.7 ms complete-frame gate
and does not certify low-end target hardware.

### Split-only strict-base status — October 24

This change set reduces the crate to the Split Radiance Cascades strict base
described by revision 2 of the paper. Hybrid SPWI, alternate estimators, c−1
continuation, and secondary feedback have been removed from the active API. The
Sponza and Cornell examples and validation scripts use Split only; they expose
no method selector, transport hotkey, or method-switch prewarm check.

Reference: [Split Radiance Cascades v2 (arXiv:2607.20384v2)](https://arxiv.org/abs/2607.20384v2).

Sponza reports the first GI dispatch and the first existing raw readback with at
least 95% support over eligible pixels. The support report is independent of
the capture warmup and does not add a GPU readback. Captures still require their
configured warmup and the unchanged 95% support gate.

At this October 24 snapshot, the Sponza profile used 0.15 m probe spacing, a 0.24 m first interval,
32,768 receiver cells, 262,144 retained interval records, a 544 MiB view budget,
64 history frames, and 0.985 temporal blend. Sponza uses a 32 MiB/frame geometry
upload cap to shorten initial scene staging; the library default is unchanged.

## Strict-base paper conformance checklist

These are source and fixture review targets for the frozen implementation,
not claims established by a benchmark. The paper is linked above.

| Paper section | Contract to review |
| --- | --- |
| §3.2, Eq. 7 | Cascade composition follows the paper's interval-ratio recurrence. |
| §4 | Probe initialization uses nearest world-grid positions; sparse interpolation reads only existing probes. |
| §4.1 | Chebyshev LOD selection uses the 0.9 overlap and does not merge probes across LODs. |
| §5 | Lower intervals pass through transparent hits, stop at opaque hits, and leave farther intervals unobserved. |
| §5.1 | Hierarchical R₂ prefixes count selected rays and assign contiguous, non-overlapping parent/child blocks. |
| §5.2 | Temporal reuse is keyed and accumulated in world space. |

This strict base omits optional §6 multi-bounce and §7.1 c−1 extensions. Bevy
continues to provide native PBR lighting; Split supplies the paper's diffuse
indirect transport. Method selection, Hybrid estimators, custom near-field
continuation, secondary bounce feedback, and their settings were removed from
the active API.

## Bounded-support reference completion — October 25

The bundled paper and Chandra implementation were reviewed against the
existing strict diffuse base. This pass preserves native Bevy light ownership
and secondary-hit Disney/GGX material evaluation; it adds no optional glossy
receiver or multi-bounce transport.

Sparse interval interpolation, irradiance packing, angular lookup, spatial
lookup, and LOD blending now preserve finite positive support below the former
epsilon cutoffs. Per-axis weight selection also avoids cancellation from
arithmetic `lerp` at tiny positive fractions. GPU fixtures exercise conditional
radiance and positive support down to approximately `2.43e-14`; at that coverage,
the raw alpha retains its supported bit but cannot represent the fractional
confidence.

Cold initialization nominates receivers with one atomic exchange rather than
retrying weak compare-exchange. Its remaining updates now require membership
in the current fair window. The cold-rank fixture preserves nominated
descriptor IDs separately from the count storage that replaces them, and the
R₂ fixture includes the current production rank helper and its dependencies.
The previously missing fixture word serialization is complete.

Bounded traversal no longer publishes exhausted or clipped misses as supported
lighting observations. Closest-hit traversal must be complete; a real accepted
blocker still proves any-hit occlusion even after another branch was incomplete.
The GPU regression failed with the original exhaustion branch reporting an
exhausted clear ray as blocked, and passes with explicit unsupported propagation.

Final verification passes 138 tests: 116 library tests, including every ignored
Slang/GPU fixture, plus 22 example tests. No tests are skipped. The default-only
control-value assertion was removed rather than pinning incidental tuning.
All examples build; strict Clippy and formatting checks pass:

```sh
cargo test --offline --all --workspace --all-targets -- --include-ignored --test-threads=1
cargo clippy --offline --all --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo build --offline --examples
```

Debug-profile Vulkan smoke captures on the RTX 4070 Laptop GPU completed with
120 warmup frames. Cornell at 640×480 lit 38,398/38,400 sampled pixels and saved
`tmp/luna_complete_cornell.png`. Sponza at 640×360 completed day, GI-off, and
night stages with all 14,400 raw values finite and 14,396/14,396 eligible
receivers supported in the recorded daylight samples. The display RGB means
were 28.758, 7.545, and 0.720 respectively; the GI-on/off difference was
21.234/255. Captures are `tmp/luna_complete_sponza_{day,no_gi,night}.png`.
The images were inspected; these are smoke results, not radiometric accuracy.

The camera-only run also completed its pan, immediate return, 64-frame
look-away, and cache-revisit stages at the same workload. Immediate-return
display RGB difference was 1.055/255; the first look-away return was
1.740/255 and the final revisit was 1.702/255, below the unchanged 8/255 gates.
The return captures had 14,396/14,396 eligible receivers supported. Images are
`tmp/luna_complete_camera_camera_motion_*.png`.

The matching moving-blocker/light run passed its captured support gates through
frame 192, whose raw output supported 14,313/14,313 eligible receivers with
14,400 finite values. It then exited 101 because the extinction-response
comparison returned no measurable ratio. The metric also rejects missing
support in a comparison sample, so that message alone does not establish that
the changed lights contributed no energy. Captures through frame 192 are
`tmp/luna_complete_motion_motion_*.png`. No 100 ms response, 5% residual, or
native-1080p performance acceptance is claimed.

## Reference-completion follow-up — October 25

The bundled paper and Chandra source were checked against the active transport.
The emitter transition was incomplete: primary next-event estimation had been
removed, but first-hit emission still required the former ray-emitter flag.
Ordinary emissive meshes now supply first-hit emission without a classification
component. All supported emitters remain in the next-event distribution used
to shade ray-hit surfaces; those reflected paths are distinct from primary
first-hit emission. The obsolete classification component and flag plumbing
have been removed.

Temporal reprojection now inverts the actual clamped resolve anchors at partial
viewport-edge tiles. The regression reproduced an unchanged edge value of 5
being incorrectly blended to 3 with the old shader. Composite binding-cache
keys now include the selected filter output, so changing spatial filter steps
cannot keep displaying a previously selected texture.

Additional GPU fixtures exercise four-level far-to-near merging, independent
raw/merged geometry dependencies, local-light invalidation with transmittance
reuse, sparse directional support, valid black versus missing lighting, and
LOD isolation/overlap. The dense SDF reference harness also initializes its
bounded-search outputs explicitly for Slang compilation. The emissive-mesh
fixture uses the renderer's validated, layout-aware SPIR-V lowering on Vulkan
to avoid the driver's long helper-inlining path.

Validation on this source passes all 106 library tests, including the 22
normally ignored Slang/GPU checks, and all 22 example tests:

```sh
cargo test --offline --all --workspace --all-targets -- --include-ignored --test-threads=1
cargo clippy --offline --all --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

On restricted Windows workspaces, set `TEMP` and `TMP` to the repository's
`tmp` directory before running the tests. The complete test log is
`tmp/split-final-tests.log`; Clippy and formatting pass. The six Python
transport-measurement tests also pass. No tests are skipped in the full Rust
run.

The Vulkan 640×480 Cornell smoke capture completed after 120 warmup frames,
with 38,394/38,400 sampled image pixels lit. The fixed image is
`target/reference_review_cornell.png`; the capture before the emitter fix is
`target/reference_review_cornell_before_emission.png`. These display-space
captures demonstrate the missing light path's effect, not reference accuracy.

The debug-profile Vulkan 640×360 Sponza smoke run completed the day, GI-off,
and night stages with unchanged capture gates. Its last printed daylight
readback had 14,395/14,396 eligible receivers supported, with all 14,400 raw
values finite. Display RGB means were 20.429 (day), 7.545 (GI off), and 0.720
(night); the GI-on/off RGB difference was 12.905/255. Images are
`target/reference_review_sponza_{day,no_gi,night}.png`, and the log is
`tmp/split-sponza-final.log`. This is a static smoke test, not a frame-budget
or moving-scene acceptance run.

The earlier captures and timings below predate these fixes. They do not
establish radiometric accuracy, dynamic-scene acceptance, or target-hardware
performance for this revision.

## Earlier strict-base evidence

The final strict-base source/API cleanup passes `cargo test --offline --lib`
(85 passed, 15 ignored) and `cargo test --offline --examples` (22 passed), for
107 active CPU tests. The 17 production compute entries compile to WGSL with
Slang; the composite vertex and fragment entries compile as well. Focused
Vulkan fixtures passed for sparse binding validation, hierarchical selected-ray
prefixes, visible zero-ray cache reuse, first-hit interval cutoffs, temporal
support, transport helpers, cache clearing, and cache commit/remapping. Those
GPU fixtures preceded the final resolve-scale uniform propagation; the release
Sponza runs below exercised the final host/shader scale path.

The 8 MiB/frame release 640×360 static capture completed its day, GI-off, and
night stages with exit code zero. At day capture, 14,394/14,396 eligible pixels were
supported after 141 dispatches; all 14,400 raw pixels were finite. Its reported
RGB means were 32.416 (day), 7.545 (GI off), and 0.720 (night). Images and logs
are `target/split_only_sponza_static_{day,no_gi,night}.png` and
`target/split_only_sponza_static.log`.

A second-process, warm-cache offscreen benchmark with the earlier 8 MiB/frame
upload cap completed 120 warmup and 240 measured GPU samples at the unchanged
640×360 quality profile: 512 updates/frame,
resolve scale 4, 32,768 receiver cells, and 262,144 cache entries. Split GPU
time averaged 18.052 ms (p95 20.485 ms); total Penumbra frame GPU time averaged
43.872 ms (p95 47.065 ms). Final eligible-pixel support was 14,387/14,396
(99.937%), with zero invalid raw values. The JSON report is
`target/split_only_sponza_warm.json`.

The 8 MiB/frame native 1920×1080 release baseline completed 120 warmup and 240
measured samples at 512 updates/frame and resolve scale 4. Final support was
125,247/129,542 eligible pixels (96.69%), all 129,600 raw values were finite,
and memory high-water was 1.992 GB. Split GPU time averaged 30.965 ms (p95
33.215 ms); total Penumbra GPU frame time averaged 59.580 ms (p95 64.162 ms).
The observed frame interval averaged 60.955 ms (p95 65.675 ms), so this profile
does not meet a real-time frame budget at 1080p. Its first dispatch came 13.322
s after startup; 95% support arrived at 16.001 s (2.678 s after that dispatch).
Scene loading took 3.256 s, and the scene-ready-to-dispatch interval was 10.067
s. The benchmark is `target/split_only_sponza_native1080.json`.

For the single Sponza-only upload-cap experiment, the cap was raised from 8 to
32 MiB/frame without changing GI sampling or reconstruction settings. The warm
1920×1080 run reached first dispatch in 8.514 s and 95% support in 11.356 s,
compared with 13.322 s and 16.001 s for the 8 MiB baseline. Time from first
dispatch to 95% support was 2.841 s vs 2.678 s. Both pipeline-cache reports
were hits. Texture/scene asset loading was 2.017 s vs 2.548 s; the renderer's
startup scene-load telemetry was 2.757 s vs 3.256 s.

About 4.1 s before first dispatch, the 32 MiB run reported no geometry bytes
and 32,544 acceleration bytes remaining, with atlas layer 8/175 and dynamic
uploads still pending. At the corresponding point in the 8 MiB run, 602,784,064
geometry bytes and 207,254,192 acceleration bytes remained. The sampled startup
`penumbra_frame_gpu_ms` peak was 49.851 ms vs 48.231 ms; the readback log samples
only every 60 observations, so it does not establish a startup p95.

The 32 MiB run retained the same quality profile and finished with 125,243/
129,542 eligible pixels supported (96.68%), all 129,600 raw values finite, and
the same 1.992 GB memory high-water. Split GPU time was 31.755 ms mean / 34.775
ms p95 vs 30.965 / 33.215 ms. Total Penumbra GPU frame time was 60.518 / 64.568
ms vs 59.580 / 64.162 ms; observed frame interval was 61.856 / 66.041 ms vs
60.955 / 65.675 ms. These one-run steady-state costs are close, with a small
measured increase; the startup improvement is clear, while startup p95 remains
unmeasured. The schema-13 result is
`target/split_only_sponza_upload32_native1080.json` and its startup log is
`target/split_only_sponza_upload32_native1080.log`.

With the earlier 8 MiB/frame cap, the 640×360 motion capture failed at frame
eight with 7,028/14,395 eligible pixels supported (48.82%), below the unchanged
95% gate. All 14,400 raw values
were finite; scene/geometry/light revisions were 10/10/9. It saved only
`target/split_only_sponza_motion_motion_000.png` and exited 101. The 90%
response, 5% extinction-residual, and 100 ms response checks did not run; no
motion-quality result is established.

In the earlier 640×360 runs with the 8 MiB/frame cap, the first release process
reached 95% raw support in 19.594 s from application startup (1.023 s from its
first dispatch). The second process reached it in
13.324 s (0.971 s from first dispatch), a 6.270 s shorter startup-to-support
time. Both runs reported a Vulkan pipeline-cache hit, so this is a first-run vs
second-process warm-cache comparison, not an isolated driver-compilation
measurement. Scene asset loading was 4.316 s and 2.289 s, respectively.

Compared with the earlier static day image at
`captures/gi_continue_resolve_static_day.png`, the new image's mean luminance
is 33.44 vs 23.20; image RMSE is 25.22/255. This rules out a darker mean-energy
regression in this capture, but the per-pixel difference is substantial and
does not establish visual-quality parity or radiometric accuracy. The dated
records below remain historical evidence for earlier implementations only.

## Historical validation records

The dated entries below describe earlier source revisions. Their multi-method
results and implementation details are not evidence about the current
Split-only strict-base implementation.

### Historical October 22 capacity and light-history follow-up

Sponza's Split profile now requests 32,768 receiver cells and 262,144 retained
interval records; switching methods restores the low-end receiver/cache caps
before applying the next profile. The library sanitizer's documented cache
ceiling is now 262,144, with library defaults unchanged. The demo view budget
is 544 MiB. At 1920×1080, the allocation calculation is about 479.4 MiB of view
buffers plus 31.64 MiB of full-resolution receiver data before lights. The CPU
admission regression verifies the sanitized profile retains its 240×135
candidate grid and requested cache capacity within that budget, with 512 MiB
buffer/binding limits. This is an allocation check, not a measured runtime result.

Persistent Split shading records up to four matching page slots during its
existing directory search and reuses that list for directional misses. List
overflow invokes the full search. Match order, closest-bin selection, owner,
LOD, finite geometry, and normal/plane filtering are preserved. This avoids
repeating the same hash-table and entry reads for each direction; the local
candidate storage can affect GPU occupancy and still needs runtime measurement.

Finite point/spot changes now have an independent radiance-dirty bound covering
both old and new influence ranges. History can survive only with known
dependencies disjoint from that bound and any proven geometry-dirty bound.
Raw and merged history use their respective dependencies; old packed-page reuse
also requires a valid aggregate dependency. The global radiance-invalid flag
remains set on a light revision. Unknown dependencies, unsupported light types,
emissive/layer/environment/material changes, and active secondary feedback use
conservative global invalidation. Bounded rigid blocker motion and finite-light
motion can compose their separate proofs. Light changes do not invalidate beta.

The combined CPU suite passes 106 library, six Cornell, five path-reference,
and 17 Sponza tests (134 total; 21 library GPU/tool tests ignored), including
cache sanitization, profile switching, old/new light bounds, global fallbacks,
material protection, and combined geometry/light motion. GPU/runtime results
below predate these final capacity and local-history changes unless stated.

### October 22 follow-up validation

GI-disabled startup no longer waits forever for shared GI pipelines whose shader
sources were intentionally not requested. Pending shared pipeline states count
only when GI is required or all methods are explicitly prewarmed; already-started
compile workers still count.

The corrected native 1920×1080 release baseline completed 120 warmup and 240
measured GPU samples on the RTX 4070 Laptop GPU, Vulkan driver 617.42, with
`AutoNoVsync`. The GPU queue-envelope p95 was 17.459 ms, observed frame-interval
p95 16.793 ms, main-schedule CPU p95 2.826 ms, and render-schedule CPU p95
16.093 ms. Queue envelopes include gaps and presentation waits; these are not
summed GPU execution times. The result is saved in
`target/gi_continue_1080_baseline_fixed.json` and does not pass every 16.7 ms gate.

A subsequent native Split run exposed adaptive-budget starvation: the budget
fell toward one update while supported coverage remained about 82% of 32,388
eligible receivers at resolve scale eight. It never reached the unchanged 95%
readiness gate and produced no accepted benchmark result. Its progress log is
`target/gi_continue_1080_group8.err.log`. The smaller resident-ray workgroups
were active, but this changing-workload run cannot establish their speedup.

Benchmarks now preserve the requested update ceiling throughout readiness and
measurement. Existing windowed resolve-scale adaptation is unchanged. Interactive
update adaptation has a floor of 64, capped by any smaller requested ceiling,
so the default profile retains eight resident updates rather than rounding them
down to zero. This preserves progress, not a guarantee of the frame-time target.

The motion harness now warms up once before its trajectory starts, with its
initial occluder and lights already at the trajectory's frame-zero positions.
It no longer makes an unnecessary initial emissive-material edit. After motion
starts it continues through submitted revisions rather than restarting warmup
after every change. Response samples must arrive within 100 ms; the separate
settled reference must be at least 250 ms after the change. Previously both
deadlines were applied to the same sample, making acceptance impossible. The
90% response, 5% extinction-residual, finite-value, and 95% receiver-support
requirements remain. CPU regressions pass; this does not yet certify moving
scene quality.

The following exact-bin fallback and convolution details describe the
pre-integration resolver revision and are superseded by the current 6×6 resolve
snapshot below.

Persistent Split resolution at that revision fell back to another matching probe page when
the selected page lacks a directional bin or that bin's physical anchor is
incompatible. Directory headers remain a search hint; packed-bin support,
instance ownership, normal/plane compatibility, and finite geometry determine
acceptance. This also handles matching headers whose representative anchors are
incompatible while their packed directional anchors are valid. The fallback is
lazy, bounded by the existing directory-search limit, and adds no BVH queries.
Ordinary methods retain their existing visibility path.

That revision's 32-bin convolution passes its known bin index directly to interpolation,
avoiding direction encoding and zero-weight angular taps at exact bin centers.
A truly empty low/high-LOD neighborhood returns before environment sampling.
Resident Split tracing uses eight-thread groups to distribute a small update
batch across more workgroups; the host dispatch count matches that group size.
These scheduling changes preserve the requested work count.

The ordinary CPU suite after the follow-up changes passes 102 library, six
Cornell, five path-reference, and 16 Sponza tests (129 total; 21 library GPU/tool
tests remain ignored). Formatting and diff whitespace checks pass. The focused
GPU neighborhood regression also passes (124.14 s), covering all 32 exact-bin
interpolations, high-LOD-only support, truly empty neighborhoods, missing-bin
and incompatible-anchor fallback, unknown-owner rejection, header sentinels,
and invalid persistent normals. Floating-point convolution comparisons use a
5e-4 absolute tolerance; support/owner checks retain their strict expectations.

The release 640×360 static capture completes day, GI-off, and night stages with
exit code zero. A populated-cache sample supports 14,062/14,396 eligible
receivers (97.68%), with all 14,400 raw pixels finite and no directory
rejections. Images are `captures/gi_continue_resolve_static_{day,no_gi,night}.png`;
the day and GI-off images visibly differ on the floor and shaded walls. That
sample reports 32.45 ms for Split, including 7.52 ms for world resolve. This is
a capture diagnostic, not an accepted native-resolution timing window or a
controlled before/after speed comparison.

The release motion capture fails the unchanged 95% support gate at frame eight:
2,728/14,395 eligible receivers (18.95%) are supported after the submitted
revisions advance to scene/geometry/light 8/8/7. All 14,400 raw pixels remain
finite. It saves only `captures/gi_continue_resolve_motion_motion_000.png` and
exits 101 before the lighting-response checks; no 90%/5%/100 ms response result
is established. Punctual/emissive-light revisions conservatively withdraw Split
radiance globally while retaining transmittance. The existing bounded geometry
dependency path cannot safely stand in for light-aware invalidation. Moving-light
support remains unresolved; the capture gate has not been weakened.

The native 1920×1080 follow-up also remains below readiness: it was stopped
after 2,051 dispatches at 26,519/32,388 eligible receivers supported (81.88%),
with all 32,400 raw pixels finite. Requested updates stayed at 512; retained
cache demand reached its 131,072-record capacity. Resolve scale adapted to
eight. The final active sample reports 34.80 ms for Split and 14.69 ms for world
resolve, with a 49.85 ms frame queue envelope. These progress samples are not a
completed benchmark window. No `target/gi_continue_resolve_1080.json` was
produced. Logs are `target/gi_continue_resolve_1080.stderr.log` and `.stdout.log`.

### October 25 preintegrated resolve and timing snapshot

The current Split resolve follows the paper's Lambertian preintegration
description: each probe carries a 6×6 octahedral irradiance field. Its 36
RGBA32F cells store radiance with observed angular coverage, followed by the
two aggregate dependency vectors, for a 608-byte probe record. Resolve uses
four-tap bilinear sampling with explicit octahedral seam folding; no copied
border is allocated. It visits positive-weight spatial neighbors at the active
low/high LODs and skips missing probes or angular samples. Coverage stays
separate from conditional radiance, so unsupported directions are not treated
as black. Angular lookup is limited to those active spatial neighbors.

The host schedules cold-priority rank phases only when persistent world
intervals are enabled, `history_frames > 0`, and world history is invalid. The
two extra phases precede parent and prefix construction; this cold-priority
frame does not consume the ordinary fair cursor. With history disabled, those
phases remain off and the circular fair schedule continues to advance. Update
windows wrap across uneven candidate-list tails while preserving the requested
per-frame budget.

The active Sponza profile is 1 m probe spacing, 1.6 m first interval, 512
updates/frame, grid divisor 4, 32,768 receiver cells, resolve scale 4, 16,384
traversal queries, 262,144 cache entries, and 64 history frames. A matched
1920×1080 Vulkan snapshot on an NVIDIA RTX 4070 Laptop GPU, using those settings
and 120 warmup plus 240 measured samples, reports GI p50 30.54 ms to 15.37 ms
and observed frame-interval p50 59.12 ms to 41.32 ms. The before and after JSON
files are [`tmp/perf-before-1080.json`](../tmp/perf-before-1080.json) and
[`tmp/perf-cooperative-1080.json`](../tmp/perf-cooperative-1080.json). Both
measured windows report full support, but the after window's GI p95 is 17.52 ms
and frame-interval p95 is 44.20 ms, above the 16.7 ms target. This is a measured
snapshot, not a frame-budget, cold-history, moving-scene, or lighting-response
acceptance result.

A separate cold/reset readback at raw frame 286 measured 93.88668% eligible-
pixel support (`tmp/cold-classifier-sponza.log`). The temporary classifier found
no missing directory descriptors; 724 unsupported pixels had no covered cell
in their neighboring tables, and 151 had table coverage away from the target
normal. This diagnostic does not pass the unchanged 95% cold-support gate, and
full-response checks remain unestablished.

### October 22 Split corrections

The Split implementation was checked against the latest
[Split Radiance Cascades revision, October 11, 2026](https://arxiv.org/abs/2607.20384v2).
Its four-cascade interval layout and hierarchical direction allocation remain.
Secondary diffuse/GGX feedback is an experimental extension and now defaults to
zero in the library as well as the demos; B no longer changes feedback. Ordinary
indirect interval transport and temporal accumulation do not depend on this field.

Interval records and probe heads are published once, after composition has
finished using the live cache. Publication changes heads from working-entry to
history-entry offsets, so running it again would interpret the wrong buffer's
indices. Scheduling follows the same requirements used to prepare the commit stage.
Probe-directory seeding also dispatches over current canonical interval records,
using the same entry mapping as merging, rather than walking bucket links.

An explicit Split cache capacity is independent of screen-grid resolution.
For example, a 131,072-record request previously retained only 4,096 records
with a small receiver grid. The configured capacity now determines the rounded
bucket count, including the separate secondary layer when enabled. Per-view
memory admission still applies; the explicit capacity can allocate more memory
than earlier low-resolution runs.

Split follows the paper's interval recurrence and sparse interpolation without
additional per-neighbor BVH visibility rays during merging or final shading.
Visibility/transmittance comes from the traced intervals. Instance, geometric
normal, and receiver-plane compatibility checks still filter reused samples.
Final Split shading uses the same bounded curved-surface tolerance as sparse
reconstruction, so normal-compatible anchors on curtains and pillars are not
rejected solely by a flat-plane approximation. Parallel, separated receivers
and other instance owners remain rejected. The persistent probe directory uses
one shared 128-slot search limit for insertion and both lookups, capped by its
table size. This admits crowded patch/normal variants under their shared spatial
hash without changing the allocation; generic methods keep their 32-slot limit.
Other methods retain their visibility-validated interpolation. Completed taps
retain fractional coverage; an entirely unobserved neighborhood remains unknown,
while an observed black interval still contributes valid support. As in the
paper, interpolating sparse intervals can still leak light near occluders.

On the RTX 4070 Laptop GPU with Vulkan, the custom no-inline SPIR-V merge path
repeatedly lost the device once Split history populated. Routing only merge
through wgpu's standard WGSL pipeline completed 2,288 GI frames at 320×180 with
512 requested updates per frame and finite values in all 3,600 raw pixels.
Its first merge compilation took about 53 seconds; the driver cache is reused
on later starts. Other stages retain their existing Vulkan path.

This diagnostic run was stopped manually: raw supported coverage remained about
53%, below the capture gate's 95%. Matching the candidate grid to the resolve
grid (`--resolution-divisor=4`) raised supported coverage to about 70% at the
same update budget; Sponza now uses that grid divisor by default, subject to its
existing receiver-cell cap. Curved-surface acceptance subsequently raised raw
coverage to about 86–88%. Expanding the directory search reduced rejections from
roughly 6,000–7,000 to about 800 per frame at comparable populated-cache stages;
the coarse-cell comparison ran for 2,175 GI frames with all raw values finite
and about 86% supported coverage. Existing GPU cache counters are included in the demo's
raw-readback progress log to distinguish lookup rejection, retention pressure,
and traversal admission.

Sponza's Split profile now uses 0.15 m probe spacing and a 0.24 m first interval,
preserving the paper's 1.6× ratio. The old 1.2 m cells could put several distinct
surface layers under one persistent anchor. With the finer profile, the complete
640×360 day/GI-off/night capture passes its unchanged 95% gate after 120 supported
warmup frames. A day readback reports 14,065/14,396 eligible pixels supported
(97.7%), all 14,400 raw pixels finite, and no directory rejections. The measured
GI-on/off RGB difference is 15.266/255; the night capture also passes. Images are
saved as `captures/gi_fix_sponza_finer_{day,no_gi,night}.png`. Other transport
profiles retain their previous probe spacing. This establishes static capture
readiness at that resolution; 1080p performance and moving-scene acceptance remain
unverified. Automated captures and benchmarks now compile their selected
estimator only; interactive method switches
and the explicit switch-validation run still prewarm all methods.

After these corrections, 102 library tests, six Cornell tests, and 11 Sponza
tests pass. The ignored GPU neighborhood/interpolation regression passes,
including 48 colliding probe identities inserted concurrently, curved-anchor
acceptance, separate-surface/owner rejection, and partial interval support.
The bounded cache publication/remap GPU regression also passes. Formatting and
diff whitespace checks pass; the complete ignored GPU suite was not run.

The current API implements Split only. Validated temporal history and the
geometry-aware a-trous filter serve it. Split's interval cache ages radiance
and transmittance independently;
`temporal_blend` controls radiance retention and `split_transmittance_blend`
controls visibility retention. Dirty-geometry checks reject affected old
intervals before reuse. No 4 GB low-end GPU measurement has been recorded, so
30 FPS at 1280×720 remains a target rather than a result. Earlier experimental
method comparisons are preserved below as historical evidence.

## Current Split acceptance target

The active target covers Split at native 1920×1080 on the RTX 4070 Laptop GPU
using Vulkan: release CPU-frame and complete GPU-frame p95 at or below
16.7 ms, at least 120 warmup and 240 measured samples, at least 95% reconstructed
support over eligible visible receivers, and a 100 ms lighting response. The
lighting response check must compare submitted geometry/light revisions and
require 90% of the new steady response within 100 ms, with extinguished-light
residual below 5%. `scripts/benchmark_sponza.ps1` runs the VSync-off windowed
benchmark plus a GI-disabled baseline; `scripts/validate_sponza_gi.ps1` covers
Split static and moving-scene captures on Vulkan or DX12. These criteria are
acceptance gates, not measured results. The historical measurements below do
not certify this target.

The current benchmark JSON (schema 11) separates the GPU queue envelope, main
schedule CPU time, render schedule CPU time, and observed frame interval. Its
PowerShell gate requires a complete timing window and checks each p95 against
16.7 ms. The capture run checks Split static, stability, camera-return, and
moving-light cases. Moving-light captures change
point and spot intensity, color, range, and cone, and toggle a test emissive
source through five logged states while the blocker moves. A stable-pose sequence
measures a submitted light extinction and re-enable: response is sampled within
100 ms, the extinguished residual must be under 5%, and the re-enabled response
must reach 90% of its steady image. Validation requires all five states
and multiple submitted lighting revisions. These runtime gates are implemented
but still need a fresh GPU run.

The benchmark and capture results below document earlier implementation
revisions; they are historical evidence, not post-change measurements. At that
revision, the low-end profile separated a 160×90 candidate grid from a
1,024-update frame budget, capped admitted traversal queries and retained cache
records at 16,384, and resolved Split at 640×360 for a 1280×720 viewport.
Direct and indirect support were reconstructed separately before temporal
accumulation. Post-change captures are recorded in the October 14 section
below; they do not establish the low-end hardware performance target.

### Historical experimental transport comparisons (retired)

At that revision, Hybrid SPWI remained the default. Hybrid, Split, and Surfel RC
composed neighboring interval samples using each sample's support,
receiver-normal compatibility, and a BVH visibility check. The weights are normalized over
supported neighbors; a neighborhood with no current support stays empty so
screen-history methods can reuse a compatible prior sample. A supported black
sample remains distinguishable from a missing sample.

Surfel RC is an opt-in stable surface-cache variant. Its persistent population
uses instance-and-connected-patch ownership, world-space subcell sampling,
bounded updates, cache aging, and dirty-scene rejection. A scene revision
invalidates its prior cache wholesale before the updated scene is traced, which
avoids stale lighting at the cost of restarting refinement after edits. Its
standalone settings are 256 updates per frame, a 1,024 nominal traversal-query
budget, two bounces, 0.5-second refinement, half-resolution reconstruction,
and a 512 MiB per-view cap. The shared `RadianceCascades::low_end()` profile
raises the update budget to 1,024 and sets 16,384-query and 16,384-record caps.
Direct-light visibility rays are outside the nominal query reservation; shared
scene geometry/material allocations are outside the per-view cap.

Surface PAIC, Surfel RC, Split, and Cascade Guided now include a bounded
resident update sweep over eligible scene triangles, so off-screen receivers
can enter their world caches. Screen observations still provide the visible
transport path. The candidate list is triangle-uniform rather than area-weighted,
and frame updates are capped; this does not perform scene-wide baking.

Cascade Guided is an opt-in Split-cache gather that mixes an eight-axis
cache-derived direction proposal with cosine sampling. It evaluates the full
mixture PDF and sets the guide probability to zero when the cache has no valid
support. Existing direct emissive and environment next-event sampling remains
enabled as a separate first-bounce direct-light estimator. The cache stores
scattered radiance, so these terms do not share paths that need cross-MIS
weights. The default settings are two bounces and a 0.5 guide mix. The guide's
variance, mean error, and equal-ray path-traced comparison have not yet been
measured.

At 1280×720 and an equal 256-ray budget, the Sponza benchmark completed 120 GPU
timing samples per method on the RTX 4070 Laptop GPU (Vulkan, driver 617.42):

| Method | Median GI time | p95 GI time |
| --- | ---: | ---: |
| Hybrid SPWI | 30.438 ms | 33.631 ms |
| Surfel RC | 28.831 ms | 29.741 ms |
| Cascade Guided | 166.792 ms | 180.917 ms |

Cascade Guided's gather took 38.391 ms in this run and its complete GI pass was
about 5.5× slower than Hybrid. These timings establish the experimental cost at
that revision, not a quality or variance improvement. The 256-ray budget is the
Surfel RC default update cap and was also passed to the other methods. The GPU suite covers
unsupported and black current samples, invalid temporal history, moving geometry,
masked occluders, production shader compilation, and binding layouts. It does
not yet measure method-specific image error against a path-traced reference or
compare method output under those scene cases.

The existing embedded-shader test validates production WGSL, SPIR-V lowering,
entry points, and binding layouts, but is ignored in the ordinary suite because
it requires `slangc`. A cache-entry ABI change from 96 to 112 bytes keeps the
Rust allocation stride aligned with the production shader's normal field.
The ordinary library suite passed (65 passed, 11 ignored) when its temporary
directory was redirected to the writable workspace, and all 11 ignored GPU and
shader-compiler checks passed on this host. The Sponza examples also compile.

### Historical October 10 screen-history reconstruction update

The full-resolution resolve now accumulates only history that matches the
current receiver identity, normal, and depth, with optional motion-vector
validation. A receiver that has valid raster depth but no current sparse support
can retain compatible lighting without treating the gap as a black sample. A
supported black sample still updates the history. Setting either
`history_frames=0` or `temporal_blend=0` disables screen history; setting
`spatial_filter_steps=0` disables the screen-space filter. At that revision
Surface PAIC kept separate world-cache history and did not use screen temporal
accumulation or spatial filtering; the current explicit sampling profile enables
the shared screen-history path for Surface PAIC and Surfel RC as well.

Screen-space PAIC, Hybrid, and Split allocate nine viewport-sized Rgba32Float
images for irradiance, history metadata, and receiver validation. At 1280×720
these images occupied about 126.6 MiB, excluding other per-view resources.
Surface PAIC then used four resolve-sized images and five 1×1 placeholders. At
the low-end profile at that revision, all six methods resolved at 640×360 and
the four irradiance plus five temporal metadata images occupied about 31.6 MiB
combined.
Cornell and
Sponza now use 64 history frames; their capture blends are 0.97 and 0.985,
respectively. Longer retention averages the sparse observations without adding
per-frame dispatch work. The retained weight is bounded by
`oldLength / (oldLength + 1)`, so 64 samples cap it at 64/65 (about 0.9846).
Cornell waits for at least `max(32, history_frames)` distinct GPU measurements,
and Sponza applies the same threshold to valid GI readbacks. The image metrics
below predate these settings. In a 640×480 Cornell before/after capture, the
back-wall residual RMS changed from 7.954 to 7.477 while its mean stayed near
215.6/255; the front region was effectively unchanged (0.5175 to 0.5217 RMS).
The after-image removes the visible grid pattern on the wall, while these
regional metrics do not show a broad noise reduction. The capture files are
`tmp/grid_before_cornell.png` and `tmp/grid_final_cornell.png`. The latest 16-frame
Sponza fixed-camera sequence passed its stability gates: adjacent-frame mean
RGB change was 0.394/255, frame-mean p95 was 0.568/255, and average per-pixel
temporal p95 was 1.422/255. Across 256 motion frames at the default 2,048-ray
budget, the separate Sponza camera-motion capture saved 22 PNG samples and
measured 5.125/255 settled/restarted RGB difference against an 8/255 gate and
19.525/255 mean RGB.
The production GPU checks passed: temporal
resolve retained validated history through unsupported pixels, updated from
supported black samples, rejected mismatched receiver identity, and disabled
history at zero frame capacity or blend; transport helper checks covered
reconstruction noise, lattice seams, and the environment sampler. These targeted
checks do not establish all-angle coverage or radiometric accuracy. The CPU
library suite passed 60 tests, with 11 GPU-dependent tests ignored. Earlier
benchmark and capture results below describe previous builds and do not establish
the quality or cost of this change.

### October 10 PAIC complete-path reduction and compact images

PAIC now stores the full path radiance already computed by tracing, with an
explicit known-path weight, then averages equal directional cache keys during
reduction. Black paths remain valid contributors. The cascade hierarchy still
assigns the ray sequence, but PAIC skips the four sparse interval recomposition
dispatches. Split and Hybrid retain their original merge path. This is an
experimental full-path cache approximation, not an equivalent implementation of
the Split paper's interval algorithm.

The first 720p, 2,048-budget Sponza benchmark measured 26.418 ms median GI and
27.178 ms p95 across 120 samples, compared with 54.407/54.795 ms before this
change on the same development GPU. This run measured GI only. The capture
`screenshots/paic_complete_path_720p_day.png` shows more illumination on some
surfaces, but large dark regions and insufficient sampling remain.

After compact texture allocation, the isolated benchmark measured 26.412 ms
median GI and 27.178 ms p95. The separately timestamped render queue envelope
was 53.398 ms median and 61.840 ms p95 across 120 samples. The envelope includes
uploads and CPU-induced queue idle gaps, and excludes CPU extraction and
presentation. It therefore does not establish 30 FPS; GI time alone must not
be converted to application FPS. The benchmark now enables `FrameTimingPlugin`
and reports both distributions, or marks the envelope unavailable if unsupported.

Sixteen 720p fixed-camera captures with the compact allocation passed the
stability gate: all 15 adjacent pairs had zero RGB difference at screenshot
precision. They are saved under `screenshots/paic_compact_720p_stability_*.png`.
This validates static repeatability, not motion, temporal AA, or radiometric
accuracy. The ordinary library suite passed 58 tests; the production reduction
GPU regression and all embedded shader compilation checks also passed.

The full-path day capture's mean RGB is 13.671/255 versus 13.928/255 before the
change, and 7.406/255 with GI disabled. Pixels with all RGB channels below 2/255
increased from 51.76% to 55.05%. These tonemapped statistics do not measure
receiver coverage or physical correctness, but they do show that this performance
change is not an overall brightness/quality win. The image still needs denser
angular/receiver coverage and a validated multi-bounce solution. Image statistics
are recorded in `target/paic_complete_path_image_statistics.json`.

In that previous build, current-frame PAIC and Hybrid replaced six full-resolution
history/filter textures with distinct 1×1 placeholders. The calculated saving
was 88,473,504 bytes (84.375 MiB) at 720p, covering those textures only. Screen
history has since been added to those methods, so that allocation result is
historical. The reduction regression described there dispatches the production
shader with duplicate keys, different colors, and known black paths.

### October 10 PAIC interval and reconstruction fixes

The disabled c-1 pass previously left cascade rays starting one probe spacing
away from the surface. Rays now start at the biased surface when c-1 is disabled,
so nearby geometry is included. Both distance levels reuse that trace when their
origins coincide. Adaptive receiver coordinates no longer add the viewport origin
twice. Reconstruction uses a compact continuous kernel over a 4×4 candidate
stencil, rejecting zero-weight candidates before loading receiver metadata.

At 1280×720 and a 2,048 sample budget on the development RTX 4070 Laptop GPU,
120 completed PAIC GPU measurements gave 54.407 ms median and 54.795 ms p95,
versus 64.752/65.585 ms immediately before the near-interval fix. The trace
pass average fell from 21.543 to 12.234 ms. Upsampling averaged 1.026 ms;
gather averaged 6.944 ms. These are GI timings, not whole-frame FPS, and this
development run does not establish minimum-spec performance. Shader-helper GPU
tests were also run during startup; use an isolated run for formal benchmarking.

The merge diagnostic averages one dispatch: 7.146 ms, with four cascade
dispatches per frame (about 28.6 ms combined). This remains a major bottleneck.
Historical per-pass averages below likewise must not be read as summed costs.

The 720p day, GI-off, and night captures completed under
`screenshots/paic_near_interval_720p_*.png`. The day image still has dark regions
and incomplete receiver coverage; neither these fixes nor the 59×33 sample grid
establish smooth high-quality GI. The GPU helper regression checks cover
nonzero viewport coordinate round-trips, continuous reconstruction across lattice
boundaries, compact kernel support, and ownership of the c-1 interval. All
embedded shader stages compile; the ordinary library suite passed 56 tests.

```sh
cargo run --offline --example sponza -- --benchmark --method=paic --width=1280 --height=720 --max-rays=2048
cargo run --offline --example sponza -- --capture --method=paic --width=1280 --height=720 --max-rays=2048 --capture-prefix=screenshots/paic_validation
```

Use `--method=hybrid` or `--method=split` with the same parameters for comparison.
Benchmark results print to stdout even when the logging filter suppresses info
messages. This renderer currently requires MSAA off; vendor portability and
compatibility with every AA/upscaling method have not been validated.

### October 10 Hybrid SPWI benchmark

At 1280×720 with the Sponza demo's 2,048-ray budget, the current Hybrid path
measured 63.561 ms median and 66.901 ms p95 across 120 GPU timing samples on an
RTX 4070 Laptop GPU (8 GB, Vulkan, driver 617.42). The trace pass averaged
21.686 ms, sparse gather 4.322 ms, and cascade merge 7.852 ms. Hybrid's
receiver-plane gather cut the previous 14.696 ms gather cost, but the full pass
is still below real time even on that GPU; the 4 GB Radeon Pro 5500M target is
not met or validated. The 720p capture is available as
`screenshots/hybrid_spwi_720p_day.png`; its 2,048-ray grid is only 59×33, so it
does not establish smooth visual quality.

## Run the current examples

```sh
cargo run --example cornell_box -- --capture
cargo run --example cornell_box -- --capture --capture-output=tmp/cornell_after.png
cargo run --example cornell_box -- --capture-raw --raw-output=screenshots/cornell_src.pfm
cargo run --example cornell_box -- --benchmark --width=1280 --height=720
cargo run --example sponza -- --capture
cargo run --offline --example sponza -- --capture-stability --width=160 --height=120 --max-rays=2048 --capture-prefix=screenshots/split_rc_fullres
cargo run --offline --example sponza -- --capture --width=160 --height=120 --max-rays=2048 --capture-prefix=screenshots/split_rc_src
cargo run --example lighting_ownership
```

The Cornell image capture defaults to `screenshots/cornell_box.png`; use
`--capture-output=path.png` to preserve a comparison image under another name.
It checks for lit receiver pixels. Its raw capture also checks
that the SRC output contains finite values and valid receivers before writing a
PFM image and JSON sidecar. The benchmark reports completed SRC GPU timings only.
Each run depends on a working Slang compiler and a supported desktop GPU.
The Sponza fixed-camera capture saves 16 consecutive frames and reports adjacent
RGB differences to reveal temporal flicker.
Its `--width` and `--height` options set the offscreen target resolution, and
`--capture-prefix` selects the output stem for the day, GI-off, and night PNGs.
Use a fresh prefix when comparing captures so existing images are preserved.

### October 10 full-resolution Cornell smoke capture

```sh
cargo run --offline --example cornell_box -- --capture-raw --width=160 --height=120 --resolution-divisor=4 --max-rays=2048 --raw-output=target/split_rc_cornell_fixed.pfm
```

The JSON sidecar reported a 160×120 viewport and a 160×120 GI output, with
17,748 valid receiver pixels. The linear RGB capture had a 5.86 maximum and
0.528 mean; 47,515 of its 57,600 channels were above `1e-6`. This confirms
that sparse gathering reconstructs finite, nonzero lighting across the full
viewport and avoids the old low-resolution composite upsample. It is a runtime
smoke check, not a radiometric reference or a measurement of residual noise.

### October 10 full-scene Sponza validation

The three-scene day/GI-off/night capture completed at 160×120 with a 2,048-ray
budget, one near-field sample per sparse receiver, and the paper-aligned 1.6×
cascade interval. Its output means were 20.855, 7.585, and 0.720/255,
respectively; the day-to-GI-off mean absolute difference was 13.285/255. The
fixed-camera 16-frame sequence at the same size measured 0.547/255 adjacent-
frame mean RGB, 1.659/255 average per-pixel temporal p95, and 0.625/255 maximum
frame-mean RGB change. The sequence passes the capture gates: mean flicker below
1.0/255, frame-mean p95 below 1.5/255, and average pixel p95 below 2.0/255.
These checks exercise full-viewport reconstruction on the larger scene; they do
not establish radiometric accuracy or detail below the selected receiver grid.

The seed pass reads the scene-instance index and geometric normal from the
receiver raster, then verifies the pixel depth against that instance's
committed BLAS. This avoids a scene-wide TLAS walk for every receiver while
preserving the matched entity and triangle-patch identity through cache updates.

All backends start with Slang's O2 WGSL output. Vulkan validates it with Naga and
uses Naga's binding map to emit SPIR-V for the exact device layouts. A function
control hint preserves remaining helper calls to bound NVIDIA driver compilation;
the direct Slang SPIR-V compiler, manual binding relocation, and entry-point
binary parser have been removed. Other backends use wgpu's WGSL translation.
GPU regressions exercise the same shader translation as Sponza. Captures remain necessary to establish
visual quality; support and GI-toggle checks alone do not establish correct materials.

Sponza requires the local assets described in [sponza-assets.md](sponza-assets.md).
The lighting ownership example exercises SRC enable/disable transitions alongside
Bevy's native ambient, environment, volume, and reflection lighting. Bevy shares
the environment cubemap enable state between diffuse and specular lighting, so a
native environment map can continue to contribute diffuse light while SRC is
active. For an SRC environment contribution, use `RadianceCascadesEnvironment`.

`python scripts/check_quality.py --help` shows how to compare externally supplied
before/after images and a capture sequence. The script reports regional spatial
and temporal image metrics; it does not create captures or establish radiometric
accuracy.

### October 9 performance and temporal quality pass

On the RTX 4070 Laptop GPU (Vulkan, driver 617.42), the same 160×120
receiver grid went from 46.47 ms median to approximately 15.9 ms. Cornell now
uses eight emitter samples, selectively inlines small shader helpers, and skips
recomposing retained intervals that already contain valid lighting. The default
960×720 demo uses a divisor of six for sparse ray tracing. Final receiver
gathering, temporal reconstruction, denoising, and composition now run at full
viewport resolution; the measurements above predate this quality change and do
not represent its current GPU cost.
Its 32-sample benchmark measured 15.91 ms median and 16.21 ms p95. These are GI
GPU timings, not total frame times. The final capture differed from the previous
capture by 0.91/255 mean RGB after resizing to the same display resolution.

Sponza uses geometric receiver normals for temporal and spatial compatibility,
validates history depth in the previous camera's projection, and retains history
through empty Monte Carlo neighborhoods. Cache history capacity includes both
ray layers. The imported scene uses two-sided native materials to match software
GI visibility. `bevy_texture_cache` supplies PNG mip levels generated in linear
color space for sRGB images. The disk cache stores complete mip chains separately
for sRGB and linear requests; older base-pixel entries are converted on first use.

### October 10 Sponza performance and temporal pass

The Sponza demo disables temporal anti-aliasing and uses spatial SMAA. SRC keeps
its own geometry-aware temporal history (64 frames, 0.985 requested blend, capped
at 64/65 by the available history). Its low-
discrepancy screen lattice advances independently of raster TAA jitter, so the
sample pattern does not inherit camera jitter or add a second temporal filter.
The renderer still aligns GI reconstruction with temporal jitter for applications
that choose to attach Bevy TAA themselves.

That revision used one trace layer, a 2,048-ray budget, 1.2 m probe
spacing with a 1.92 m cascade-zero cutoff (the paper's 1.6× ratio), one
progressive near-field sample per sparse receiver, four emissive visibility
samples and three spatial filter steps. At 1280×720 the budget resolves to a
59×33 (1,947-ray) receiver grid. The renderer gathers at jittered sparse
anchors, then uses geometry-aware reconstruction, temporal accumulation and
denoising at full viewport resolution. The secondary feedback layer was an
optional demo control at that revision. On the RTX 4070 Laptop GPU (Vulkan,
driver 617.42), the 2,048-ray,
1280×720 benchmark measured 99.835 ms median and 110.251 ms p95 across 120
samples. Average stage times included gather 12.663 ms, trace 21.679 ms, merge
13.994 ms and full-resolution upsample 0.314 ms. This is a costly,
below-real-time GI pass. The earlier 41.80 ms measurement predates the current
full-resolution reconstruction path and is not a like-for-like comparison.

With the default settings and all three supplied scenes loaded, the noon capture
passed its GI-on check and differed from GI-off by 13.285/255 mean absolute RGB;
the night capture also passed. Comparing the 160×120 one-sample capture with the
earlier per-pixel reference gives a 4.947/255 RGB MAE and 20.855/255 versus
21.687/255 daytime mean. This verifies approximate energy retention and a stable
capture sequence, not radiometric accuracy or every camera angle. Cache history
retains four entries per bucket and ray layer; the capture still reported 9,607
entries dropped under Sponza's cache load.

The fixed-camera 16-frame capture exposed that the denoiser started from the raw
current sample and overwrote the temporal result. Denoising now starts from the
accumulated texture and ping-pongs without touching the saved history. Temporal
weight no longer falls as spatial variance rises; validated history is retained
on noisy receivers, while scene edits remain reactive. With a moving sparse
lattice and one c(-1) sample per receiver, adjacent-frame mean RGB change was
0.547/255, frame-mean p95 was 0.621/255, and average per-pixel temporal p95 was
1.659/255. The default stability capture fails above the documented limits.

The native path uses Slang compute entry points compiled to SPIR-V and
dispatched through Bevy's compute passes. An earlier blanket O1/O2 experiment made NVIDIA
pipeline creation take over 30 seconds for the trace entry and 43.8 seconds for
gather at O2. Ray tracing and material evaluation therefore retain O0. Bounded
cache, merge, depth-pyramid, temporal, denoise, and upsample entries now use O2;
their optimization policy has a distinct shader-cache key. The shader cache
stores compiler output; the separate device pipeline cache stores driver data.

### October 14 sparse-grid and energy corrections

The GI grid failure involved missing observations being discarded or interpreted
as black, and a shared traversal limit being exhausted before direct lighting.
Direct visibility now runs after receiver seeding, before indirect transport.
Explicit update requests are bounded by the configured query limit and light
sampling cost. A denied query or empty cache retains a compatible previous
candidate; a supported black observation still updates it. Partial update grids
keep their receiver anchors fixed and spread each batch across both image axes.

Directional prefiltering and exact diffuse gathers now apply the same cache
support weight to their radiance numerator and normalization denominator. This
removes the inverse-support amplification that made Surfel RC and Hybrid much
too bright. Split's c(-1) continuation uses the original biased receiver cache
anchor, whose c0 interval already excludes the near segment; missing continuation
directions are excluded from its conditional mean. Resolve and composite sampling
use the actual integer pixel-center anchors. Three denoising passes now use
distinct default strides of 1, 2, and 4 without adding filter passes.

The regression tests exercise production GPU helpers for fractional-support
prefiltering, near-field cache anchors, missing-versus-black continuations,
constant sparse-field reconstruction at resolve scales one and two, and temporal
history rejection. Vulkan and DX12 use the same translated WGSL path. This is a plugin implementation
correction; the investigation did not establish a Bevy engine defect.

Cornell captures now wait for 16 complete update epochs by default. Use
`--capture-warmup-frames=N` to override convergence time and
`--capture-frame-count=N` to capture successive displayed frames. Raw reference
comparisons use linear PFM data with receiver support masks. The CPU reference's
`--bounces=2` shades the primary receiver and one reflected vertex, matching
Cornell's GPU one-bounce setting; CPU `--bounces=1` is direct lighting only.

On the RTX 4070 Laptop GPU through Vulkan, the corrected PAIC capture at
1195×819 illuminated 121,800 of 122,590 sampled receiver pixels (99.36%). An
oversized 20,000-update request retained that coverage: the 15,450-candidate
grid selected 1,408 visible updates in the recorded frame and admitted 6,952
queries, below the 16,384 cap. A 16-frame 640×480 PAIC sequence had adjacent-frame
mean RGB change of 0.044/255, with a maximum regional frame-pair mean of
0.066/255. At 640×480, PAIC lit 96.41% of sampled pixels; Surface PAIC, Surfel RC,
and Hybrid each lit 98.23%. These captures show continuous lighting with residual
soft Monte Carlo mottling, rather than disconnected lit cells. These are static
Cornell measurements, not moving-camera or Sponza certification.

### October 14 Cornell Split traversal and capture correction

Split's explicit sampling mode now reserves the last third of the traversal
budget for receiver gathering. The earlier stages use a smaller admission
ceiling, and selected-update sizing includes nominal first-hit lighting and
transport work. The frame-wide hard cap still applies; the estimate does not
guarantee that every selected receiver completes in arbitrary geometry.

When a near-field gather exhausts its queries after completing some samples,
it keeps those completed observations. An unobserved tail is excluded, and a
completed black observation remains valid. This is a partial conditional
estimate under budget pressure, not a proof of an unbiased stopped estimator.

Cornell raw captures attach `ReadbackOnce` so each queued readback contributes
exactly one observation. Requests for two and sixteen raw frames were checked
to produce exactly two and sixteen frames and observations per supported pixel.
Older multi-frame raw captures without this marker can contain repeated
observations and must not be used as evidence of independent sampling.

The ordinary library suite passes 82 tests with 13 ignored. The embedded Slang
compilation check and the production GPU near-field, temporal-history, and
sparse-upsample checks also pass on the development GPU.

The selected low-end preset uses 1 m probes, a 1.6 m cascade-zero cutoff,
65,536 retained cache records, and `near_field_samples = 0`. In this mode the
transport ray starts at the biased surface, so the cache includes the near
interval rather than dropping it. The cosine-prefiltered gather removes the
separate high-variance c-1 estimator at this work budget. Library `Default`
settings and the opt-in near-field implementation are preserved.

The final sixteen-frame raw capture reported indirect support on all 210,339
supported receiver pixels; the viewport total of 245,180 pixels includes
background outside those receivers. A 960×720 capture checks the ordinary demo's aspect
ratio and shows the same smooth block faces and wall reconstruction.

A matched 1195×819, feedback-off Split comparison against the starting working
build reduced detrended block-front RMS from 15.52 to 1.91/255 and back-wall RMS
from 9.51 to 3.05/255. All six measured regions improved; ceiling RMS changed
only slightly, from 15.56 to 15.34/255, and broad cache approximation remains
around the emitter. Across sixteen serialized captures, regional mean RGB
changes ranged from 0.0024 to 0.0341/255. Screenshot readbacks can be separated
by multiple rendered frames, so these are capture-pair measurements, not a
per-rendered-frame flicker bound. They do not establish radiometric accuracy,
moving-camera quality, or target-hardware frame time.

```sh
cargo run --offline --example cornell_box -- --capture --method=split --width=1195 --height=819 --capture-frame-count=16 --capture-output=target/cornell_split_smooth.png
```

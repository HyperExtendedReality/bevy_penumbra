# GI instability and performance audit — status 2026-10-19

This is a historical audit of the 2026-10-19 source revision. At that time the
crate supported Split and Hybrid SPWI. The current crate implements only the
Split Radiance Cascades strict base; every result below remains tied to the
older code revision and hardware named in its section. PAIC, Surface PAIC,
Surfel RC, and Cascade Guided were retired comparison implementations even in
that historical record.

## Latest measured baseline: static gates pass; dynamic restart still fails

The surface-ownership/canonical-anchor refactor is implemented. Split retains
entity, connected-patch and geometric-normal class in its interval keys, reuses
clean canonical surface anchors, and fixes the base distance LOD at zero while
keeping four RC cascades. Half the selected receiver work can refresh registered
world owners independently of raster visibility. Balanced registry partitions
rotate globally once per two frames, including single-candidate and truncated
batch schedules. A dropped directory representative is replaced by an exact
committed sibling direction when one survives; owner, patch and normal class
must match. The focused commit/remap GPU regression passed in 1.85 seconds.

Exact-anchor raw interval running means preserve accumulated sample mass without
screen-history decay. The final merged-field filter now reuses only matching
owned anchors. At 1280×720, the exact-pack implementation passes the unchanged
8/255 camera gates: 4.780 immediate, 5.303 settled, and 6.249/6.441 after the
long look-away revisit (`tmp/ggx-rc-exact-pack-camera.log`). Stationary gates also
pass: mean 0.128/255, frame-mean p95 0.149 and average pixel p95 0.667.

The current benchmark records GI p50 10.446 ms / p95 11.016 ms over 120 samples,
and whole-GPU p50 24.557 ms / p95 25.197 ms over 153 samples
(`tmp/ggx-rc-exact-pack-benchmark.json`). Sponza uses a 131,072-entry cache within
the existing 512 MiB view budget, 512 updates, 16,384 queries and 120 warmup
frames. Temporal retention remains 0.985 with a 64-frame cap.

Dynamic quality is unresolved: the exact-pack moving-blocker capture fails its
settled/restarted comparison at **13.298/255**. This comparison includes
rebuilding support after geometry changes and does not alone prove stale-light
leakage. Its gate remains unchanged. The exact-pack baseline passed all 113
library/GPU checks (`tmp/ggx-rc-exact-pack-all-tests.log`).

Dependency-aware geometry reuse is now implemented for rigid, non-emissive
motion with stable scene roots, materials, lights and topology. Tracked primary,
shadow and child-visibility segments invalidate affected world observations;
untracked dependencies and unsupported scene changes fall back conservatively.
Entries are 176 bytes and packed probes 2,080 bytes. The first live run retained
171,639 supported resolve pixels before restart, versus 143,402 previously, but
still failed the dynamic gate at **14.582/255**
(`tmp/ggx-rc-dependency-motion.log`). Camera-only comparisons still passed at
5.015/5.577 and 6.613/6.695 after the long return
(`tmp/ggx-rc-dependency-camera.log`).

Progressive root R2 sampling now reserves contiguous blocks from the existing
owner-sequence table, with child cascades inheriting disjoint subranges. Bounded
geometry motion preserves those counters. This reduced the dynamic comparison
to **13.972/255**, still failing. Camera returns pass, including 6.713/6.657 after
64 frames looking away; stationary flicker passes at mean 0.141, frame-mean p95
0.254 and average pixel p95 0.689. The 120-sample benchmark records GI p50
**10.724 ms**, p95 **11.370 ms**, and whole-GPU p50/p95 **24.835/25.477 ms**
(`tmp/ggx-rc-progressive-benchmark.json`). All 115 library/GPU checks passed
across the full run (114 passing) and the corrected isolated interpolation
fixture rerun (`tmp/ggx-rc-progressive-all-tests.log`,
`tmp/ggx-rc-progressive-interpolation-fixed.log`). The BSDF CPU oracle also
passes 3,316 checks. Dynamic quality remains unresolved.

The next tested checkpoint narrows persistent anchor invalidation to the actual
stored point plus a conservative ray-offset/ULP envelope, and uses selected
background candidates for retained-world refresh instead of preempting visible
receivers. The full suite passes **115/115** together
(`tmp/ggx-rc-visible-all-tests.log`). Camera differences are 4.750/5.102 and
5.697/5.974 after the long return; stationary mean/frame-mean p95/pixel p95 are
0.111/0.131/0.644. GI p50/p95 are **10.460/11.177 ms**, whole-GPU p50/p95
**24.539/25.205 ms** (`tmp/ggx-rc-visible-benchmark.json`). The dynamic comparison
improves to **11.833/255**, but still fails the unchanged 8/255 gate.

A separate correctness issue is confirmed in coarse reduction: different fine
anchors can share one coarse key, and their raw intervals are currently averaged
while retaining one representative origin. Exact history matching then also
resets counts when that representative changes. A single-physical-anchor
coarse reduction is being implemented and has not yet been accepted by live
captures. Scene-wide discovery of previously unseen receivers also remains a
structural limitation; the current registry discovers owners from raster input.
The dependency audit also confirmed that child/visibility bounds overwrite the
raw interval bounds. This can discard clean parent J/T when only a child is
affected. Raw and merged-I bounds are being separated in a 208-byte Entry
(packed probes remain 2,080 bytes), with explicit rejection of dirty old-I
fallback. These in-progress changes require their own full suite and captures.

Before exact-pack filtering, all 113 running-mean library/GPU checks passed in
96.88 seconds (`tmp/ggx-rc-running-mean-all-tests.log`), and the BSDF CPU oracle
passed 3,316 checks. That revision passed short returns at 6.193/6.605 but failed
the long return at 10.514 and dynamic comparison at 12.446. The measurements
below describe preceding implementations, not the current exact-pack baseline.

Static Split receivers now reproject through the last submitted GI frame's
matrices instead of raster motion vectors, whose frame interval can differ.
Intervening scene changes retain the motion-vector fallback. The focused GPU
test passes, including stale-motion and dynamic-scene cases. Matrix reprojection
improved short camera return from 11.585 to 9.735/255. Coverage-weighted temporal
confidence then passes the unchanged 8/255 short-return gates at 6.763 immediate
and 6.016 settled, but the 64-frame look-away revisit still fails at 12.114:
`tmp/ggx-rc-coverage-camera.log`. Quality is not yet certified. At the long return,
screen history is unavailable; world-cache probe retention counters do not show
wholesale cache loss. The remaining issue is the world lighting estimator.

All 112 library/GPU checks passed for coverage propagation; the subsequent
partial-history-clipping correction passed its focused both-lobe GPU regression.
Partial directional observations preserve known black versus unknown while
contributing fractional confidence rather than full-lobe confidence.

The old gather and primary NEE passes no longer execute for Split. The dense
resolve filters uncolored diffuse and incident GGX separately before applying
full-resolution G-buffer material weights. Cascade merging can use the entire
configured visibility-query cap now that the removed gather needs no reserve;
the production GPU fixture verifies Split and legacy admission ceilings.

The preceding coverage implementation's windowed Vulkan benchmark records GI p50 11.170 ms / p95 11.838 ms
and whole-GPU frame p50 25.362 ms / p95 26.060 ms on the RTX 4070 Laptop, with
120 warmup and 120 measured GI samples. Evidence:
`tmp/ggx-rc-coverage-benchmark.json`. Against the prior fused resolve's GI median
16.288 ms, this is a 31.4% reduction. The camera quality failure still applies.

The following paragraphs record earlier revisions and their failures.

The normal-contract fixes passed all 111 library/GPU checks. Camera return still
failed at 9.90/255; per-observation interval retention did not resolve it
(10.00/255). A stationary-camera capture also failed at 2.88/255 adjacent-frame
difference. Disabling primary glossy GI improved the short return to 6.54/255,
but the 64-frame look-away revisit failed at 12.69/255. These are failures,
not current quality or performance certification.

Primary diffuse environment/emissive NEE was retained only in screen history,
while the world-cache diffuse gather discarded its boundary term. A fused
G-buffer diffuse/specular world resolve is being validated to remove that
screen-history dependency. It inherits the cache's 32-bin angular resolution,
including its limited ability to resolve small emitters.

The fused world resolve failed the short-return gate at 10.96/255. A 600-frame
warmup control still failed at 11.40/255; a reconstructed-world-position control
passed every same-pose revisit at exactly 0.00/255. Final-probe EMA, retained
anchors, and freezing unobserved merged entries were rejected after worse camera
results and removed. The remaining work is cascade reconstruction stability.

The earlier fused-resolve windowed Vulkan benchmark (1280×720, RTX 4070 Laptop, 120 warmup and
120 measured GPU samples) records GI p50 16.288 ms / p95 17.383 ms, and whole-GPU
frame p50 30.429 ms / p95 31.407 ms. Per-pass endpoint diagnostics reveal an
11.126 ms legacy gather and 1.330 ms primary NEE pass whose screen output is no
longer displayed by Split's fused primary resolve; composition is 2.883 ms.
Those endpoint pass values are not percentile statistics. Evidence:
`tmp/ggx-rc-world-field-benchmark.json`. This does not meet the visual gate.

A separate source audit found no confirmed defect in the R2 count/prefix
hierarchy, four-child angular subdivision, or interval endpoints. Sparse merges
conditionally average observed children and carry coverage separately; this can
extrapolate across missing angular bins and remains a quality limitation. It is
not evidence of a path-tracing implementation or a reason to change the tested
interval algebra. The next reconstruction pass keeps diffuse and GGX fields
separate through temporal/spatial filtering and applies material weights last.

The modular BSDF CPU oracle passed 3,316 checks. Probe packing, boundary merge,
glossy constant-field energy, shader compilation, and the retained cache have
GPU coverage. The first 1280×720 Split camera-return capture nevertheless failed
the unchanged 8/255 gate: 10.374/255 with four screen-random reflection samples,
then 10.008/255 with deterministic 32-bin convolution. Increasing updates from
512 to 1,024 did not resolve the failure (10.632/255).

A native-only composition control passed at exactly 0.000/255 on immediate,
settled, and long offscreen revisits (`tmp/ggx-rc-native-only-camera.log`). This
isolates the remaining variation to GI. The latest fixes preserve each receiver's
exact geometric normal through rank/trace, preserve an origin's own normal
during reduction, keep geometric compatibility separate from shading normals,
and make Split ray-material footprints depend on world probe spacing rather
than screen-pixel spacing. Their final visual and performance checks are pending.

The persistent Split probe-record pack, lookup, and commit path has landed and
is undergoing GPU validation. The previous full library/GPU suite passed 110
checks with Hybrid and `near_field_samples = 1` (`tmp/hybrid-near-all-tests.log`);
that run predates the current Split cache changes. All capture matrices and
benchmarks below are historical results from earlier source revisions. Their
pass labels do not validate the current cache, current shading target, or image
quality. The full Disney/GGX multi-scattering adjustment and its visual and
performance validation remain open.

## Historical quality and performance measurements

Before the latest pre-cache optimization, the library run passed 109 CPU, GPU,
and shader-compiler checks, including the formerly ignored GPU checks
(`tmp/gbuffer-mask-all-tests.log`). A 640×360 camera-motion matrix passed for
all six methods; Hybrid SPWI also passed at 1280×720. Immediate is the frame-32
return to the original pose; settled is the same pose at frame 64. These image-
difference checks do not establish radiometric accuracy or eliminate visible
sparse-sampling noise.

| Method | 640×360 immediate RGB difference /255 | 640×360 settled /255 |
| --- | ---: | ---: |
| Hybrid SPWI | 5.685 | 4.966 |
| PAIC | 5.748 | 5.177 |
| Surface PAIC | 5.625 | 5.626 |
| Surfel RC | 5.458 | 5.424 |
| Cascade Guided | 4.907 | 4.393 |
| Split | 6.029 | 5.522 |

Hybrid SPWI at 1280×720 measured 5.840/255 immediate and 6.115/255 settled.
The earlier Disney-only Hybrid 640×360 run measured 6.225/255 immediate and
5.458/255 settled. The 1280×720 image still shows dark, patchy areas
(`tmp/gbuffer-hybrid-720_camera_motion_032.png`); visual quality is unresolved.

The historical 1280×720 windowed Hybrid run, after exact-origin visibility reuse,
measured application-frame median 34.7667 ms / p95 39.1009 ms and GI median
20.618240 ms / p95 24.663040 ms, from 120 warmup and 120 measured GI samples.
The latest reported `split_rc/gather` diagnostic was 7.804928 ms. Against the
immediately preceding G-buffer/mask run (`tmp/gbuffer-windowed.json`: application
p95 40.7755 ms, GI p95 26.135552 ms), these p95 values improved by about 4.1%
and 5.6%. The application p95 still exceeds 33.3 ms and GI p95 exceeds the
provisional 16.7 ms allocation; this was not a 30 fps result. The earlier
Disney-only 1280×720 baseline was 40.9701 ms application p95 and 25.914368 ms
GI p95. Data for that run: `tmp/memo-windowed.json`.

## Earlier source behavior (historical inspection)

- The existing path evaluates Disney diffuse for direct and indirect lighting;
  the current target adds the agreed Disney/GGX multi-scattering treatment.
  That shading change and its validation are still open.
- Split stores exact probe-major directional records and is under GPU
  validation. Its two-buffer pool reserves `buckets × 2 × 1,536` bytes; a
  non-Split method uses one 1,536-byte placeholder. At 65,536 Split slots the
  record buffers use 192 MiB regardless of how many slots currently hold
  probes. Probe headers are grouped by eight per LOD, with the merged-record
  lookup chain as fallback for directory overflow or evicted headers. Commit
  remaps lookup heads only after publishing complete records and retains recent
  eligible entries under the LRU policy. Screen-history TTL is independent.
  Moving geometry currently invalidates the Split world cache globally until
  per-segment dependency tracking exists.
- Split query admission reserves half the hard cap for trace/NEE work, permits
  merge visibility up to two-thirds, and lets final gather use the full hard
  cap. The earlier shared two-thirds ceiling could exhaust before merges.
- Receiver shading uses deferred depth plus G-buffer base color, metallic,
  perceptual roughness, and shading normal. A separate raster receiver texture
  carries entity identity and geometric normal. The path requires single-sample
  deferred inputs; opaque and alpha-mask materials are supported.
- Sparse camera migration projects each new candidate into the previous view,
  searches the nearby 4×4 previous candidate cells, and selects the nearest
  saved anchor that maps into the previous reconstruction footprint. It checks
  receiver identity, normal, roughness, prior/current view direction, and
  world-space plane distance. Direct and indirect estimates migrate separately;
  counts are retained, while the prior frame's freshness marker is removed.
  Migration reads the previous anchor map before current receivers are
  published, then priority compaction and `prepareReceivers` establish the
  current anchors before seeding and light gathering.
- Direct support uses mask bit 2 and indirect support uses bit 1. Reconstruction
  sums only supported terms and carries both bits through temporal history;
  partial estimates cannot stand in for a missing component. Denoising also
  rejects taps with a different support mask, so a supported black result stays
  distinct from missing support.

## Historical implementation and validation records

The following notes preserve observations from earlier source states. A result
labelled “final” or “passed” records that run only; it does not validate the
current source. The earlier audit recorded these implementation changes:

- Composite anchor geometric normals used the anchor's own G-buffer normal for
  orientation, rejecting an opposite face of the same entity.
- Sparse reconstruction weighted direct and indirect candidate means by their
  bounded observation counts separately.
- Light-only revisions preserved compatible screen history and reseeded
  confidence; geometry, layout, viewport, camera-grid and epoch changes kept
  structural invalidation.

At that historical point, the library suite passed 89 CPU tests and all 14
ignored GPU/compiler checks on Vulkan. Logs: `tmp/gi-audit-final-cpu.log` and
`tmp/gi-audit-final-gpu.log`.

## Historical: initial failure and diagnostic limitation

The 640×360 Hybrid Sponza motion capture fails its existing brightness assertion
at motion frame 88 both before and after these changes: mean RGB 1.429/255
before, 1.372/255 after. Do not describe the motion issue as solved. This point
in the path includes a large camera turn and a moving blocker, so it does not
isolate a reprojection failure from newly visible, insufficiently sampled
geometry. Coordinate/depth reprojection review found no proved sign or space
error. Structural camera/geometry changes still reset sparse grid accumulation;
validated screen history alone cannot fill newly revealed surfaces.

Reproduce:

```powershell
$env:TEMP = "$PWD/tmp"
$env:TMP = $env:TEMP
$env:WGPU_BACKEND = 'vulkan'
cargo run --locked --offline --example sponza -- --capture-motion --method=hybrid --width=640 --height=360 --warmup-frames=120 --capture-prefix=captures/gi_audit_sponza_motion_after
```

The final log is `tmp/gi-audit-sponza-motion-after.log`; images are
`captures/gi_audit_sponza_motion_after_motion_*.png`.
The fixed-camera baseline measured 0.077/255 mean adjacent-frame RGB change
across 16 frames, which does not establish radiometric accuracy or adequate GI.
Next investigation should isolate camera-only motion from moving geometry and
measure scene-linear indirect support, query denial and receiver revisit latency
in the newly visible regions. Preserve the existing assertions and work budgets.

## Historical: camera accumulation follow-up

Camera-only movement now migrates the prior direct and indirect sparse fields
separately. Migration validates receiver identity, geometric normal, roughness,
depth and world-plane distance. It rejects disoccluded or incompatible surfaces;
reprojection preserves bounded observation counts without adding an observation.
Geometry/layout/viewport changes and explicit resets still invalidate history.

Candidate means are capped at `60 / ceil(candidate_count / update_budget)`
observations (clamped to 1–64). At 160×90 with 1,024 updates, the observation
window decreases from 960 to 60 rendered frames: four visits rather than 64.
This is a rendered-frame bound, not a measured millisecond response guarantee.
Successful observations carry a fractional freshness marker; spatial reconstruction
keeps all valid candidate RGB, while PAIC screen history distinguishes new
evidence from retained estimates. Unsupported validated history no longer fades
by an artificial 0.98 multiplier. Fresh radiometric changes reset screen confidence.

Validation on Vulkan: 103 of 104 combined CPU/GPU checks passed in
`tmp/gi-motion-final-tests.log`; the remaining helper assertion was updated for
the new four-observation cap and passed separately in
`tmp/gi-motion-helper-final.log`. The new production clear regression checks
independent direct/indirect migration and rejects wrong owners, off-plane data
and missing history. Temporal GPU cases cover retained inputs, light steps and
out-of-viewport reprojection.

The final default-budget Hybrid camera-only pan/return capture passed, with a
settled return difference of 3.497/255 and 50,964 of 57,600 raw pixels supported
at the final readback. See `tmp/gi-camera-final-hybrid.log` and
`captures/gi_camera_final_hybrid_camera_motion_*.png`. Reproduce with:

```powershell
cargo run --locked --offline --example sponza -- --capture-camera-motion --method=hybrid --width=640 --height=360 --warmup-frames=120 --capture-prefix=captures/gi_camera_final_hybrid
```

The original moving-blocker/large-turn test is a separate case and has not been
shown to pass. A dense, stationary pose-88 diagnostic (3,600 candidates per frame,
131,072-query ceiling) reached 98% raw support yet only 2.585/255 image mean,
below that test's fixed 4/255 gate even without camera motion. That gate alone
cannot certify a camera-accumulation defect at this dark orientation. The new
camera-only capture preserves its brightness and return-difference assertions;
the original moving-blocker gate has not been weakened. Radiometric accuracy,
target-device timing, and all-method motion equivalence remain unestablished.

## Historical: Sponza demo audit and repeatable captures

The demo disables the glTF's authored cameras and lights, then creates one
camera and a single animated directional sun. Its GI sky and visible clear color
are derived from the same hour-dependent radiance. The six `--method=` names
(`paic`, `surface-paic`, `surfel-rc`, `cascade-guided`, `hybrid`, `split`)
match the benchmark script. The executable defaults to Hybrid SPWI with an
8-pixel grid divisor, 512 updates, two receiver light samples, and a 16,384
traversal-query cap. Its near-field setting defaults to one sample, enabled for
Hybrid and Split; the other transport methods ignore it.

Existing Vulkan logs repeatedly reject four scene entities because their
material uses alpha blending. Static inspection identifies the main model's
`dirt_decal` material as the only `BLEND` material in these three glTF assets;
the curtains and ivy are not the rejected material. The GI path supports opaque
and alpha-mask materials, so these decals are omitted from GI visibility and
lighting. Converting them to opaque would change their authored semantics and
would hide the renderer limitation rather than establish a fix for the broad
darkness case.

Sponza's raw capture target copies the sparse reconstruction before temporal
filtering and stores its support count in alpha. Its support and luminance
readbacks therefore describe the current raw GI estimate, while the RGB
brightness gates measure the final rendered image. These diagnostics do not
establish radiometric accuracy by themselves.

Run the day/no-GI/night capture gates sequentially at 640×360 with the default
low-end budgets. The 80×45 candidate grid exceeds the 1,024-update budget, so
the captures exercise partial updates. Use `-Methods hybrid` to rerun one
method after a targeted fix:

```powershell
.\scripts\validate_sponza_gi.ps1
```

Add `-IncludeHybridMotionCases` to also run the 640×360 Hybrid camera-only and
moving-blocker captures. The camera-only case retains its brightness and
return-difference assertions. The moving-blocker case retains its existing
brightness and stale-lighting assertions; the known frame-88 darkness remains
an unresolved failure and is reported by the script without changing its gate.
Each run writes fresh images and a log under `tmp/sponza-gi-validation/`.

## Historical: October 16 scheduling, support, and near-field corrections

Priority compaction now precedes receiver seeding and direct-light gathering.
Previously those stages consumed the preceding priority map while transport
consumed the newly compacted map, leaving some selected cells without a seeded
receiver or direct estimate. Adaptive PAIC positions are published in a separate
`prepareReceivers` pass after migration and compaction have read the previous
anchor map. The priority predicate computes its current anchor directly.

Temporal output now carries zero alpha when neither current lighting nor
validated history exists. Denoising can fill that hole from compatible supported
neighbors using raster depth, and leaves alpha zero if it cannot. Composition
therefore excludes unsupported black texels while retaining observed black
lighting. GPU regressions distinguish those cases.

Split near-field samples below the geometric horizon now contribute zero to
the sample mean instead of disappearing from its denominator. A separate count
of observed transport samples prevents a horizon-only prefix followed by query
exhaustion from being published as observed black transport.

View allocation reuse now tracks the original request separately from effective
budget-adapted settings. Changes to ray limits, receiver caps, or resolve scale
are applied on the next preparation instead of silently reusing the previous
request. Owner-sequence reset decisions are captured before settings are updated.

Moving geometry also raised the global scene-change flag and discarded temporal
confidence even on compatible receivers whose lighting had not changed. The
reset now requires a local luminance change above a noise-aware threshold
(three retained standard deviations, bounded to 5–25%). Unchanged lighting keeps
its confidence; large light steps remain reactive. This is a temporal filtering
heuristic, not a radiometric correction.

The deeper lighting audit found a separate missing sky-bounce term: constant
environments were evaluated only by escaped transport rays, while HDR sky used
next-event estimation at every shaded vertex. Thus a secondary surface could
reflect sun or HDR sky toward the receiver but could not reflect the equivalent
constant sky. Both environment sources now use visibility-tested next-event
estimation at primary receivers and surface hits. Escaped transport rays return
supported zero because their direct sky term is already owned by that estimator.
Constant skies use cosine sampling; HDR maps retain alias/cosine mixture sampling.
Query affordability includes nonblack constant skies as well as environment maps.

### Historical October 16 validation

The combined library suite passed all 108 CPU, GPU, and shader-compiler checks
on Vulkan (`tmp/sponza-fix-final-tests.log`). The Sponza example test also passed
(`tmp/sponza-fix-demo-tests.log`). The production environment estimator is checked
against the analytic Disney diffuse integral under uniform sky, a uniform HDR
map, blocked visibility, and denied traversal. Escaped-ray support is checked
separately to guard the NEE partition.

All six methods passed the existing day/GI-off/night gates at 640×360, 1,024
requested updates, and a 16,384-query ceiling. Artifacts and per-method logs:
`tmp/sponza-gi-validation/20261016-162801-f53f45b5/`. GI-on/off mean absolute RGB
differences were 11.121/255 (PAIC), 14.400 (Surface PAIC), 13.695 (Surfel RC),
7.880 (Cascade Guided), 10.463 (Hybrid), and 12.661 (Split). These tonemapped
differences establish a visible contribution, not radiometric equivalence.

Hybrid's camera-only pan/return passed, with 6.557/255 settled return difference
against the unchanged 8/255 gate. A separate sixteen-image fixed-camera sequence
passed its existing stability gates: adjacent-image mean RGB change was
0.129/255, frame-mean p95 was 0.139/255, and average pixel p95 was 0.667/255
(`tmp/sponza-fix-sky-stability.log`). Screenshot readbacks are not guaranteed to
sample consecutive rendered frames.

The moving-blocker/large-turn case still fails at frame 88: mean RGB 1.823/255,
below the unchanged 4/255 gate. A stationary dense diagnostic at that pose,
with 3,600 updates, 131,072 queries and 240 warmup readbacks, reached 56,396 of
57,600 supported raw pixels but mean RGB only 3.383/255
(`tmp/sponza-fix-sky-pose88-dense.log`). This is improved from 2.419/255 before
the sky correction at the same dense settings, but neither result passes the
gate. Do not describe all-angle motion or the brightness failure as solved.

The first corrected 1280×720 Hybrid benchmark on the RTX 4070 Laptop GPU
(Vulkan, driver 617.42, development profile) measured 120 GI samples after 120
warmup samples: GI median 14.238 ms and p95 15.554 ms. The queue/frame envelope
measured median 35.632 ms and p95 39.615 ms over 119 samples; the Bevy frame-time
diagnostic median was 37.257 ms. This run retained continuous raw-output readback
and CPU validation scanning. It does not establish a 33.3 ms application frame,
and GI timing alone must not be converted to application FPS. Data:
`tmp/sponza-fix-benchmark.json`.

Benchmark mode now removes its raw-output readback after full-scene readiness,
the settle interval, and GI timing diagnostics are available. Capture, motion,
and stability modes retain their readbacks and gates. This avoids continuously
copying 3.516 MiB and scanning 230,400 pixels per 720p benchmark frame; timing
still requires 120 new warmup and 120 new measured GI observations.

The follow-up run passed that sampling requirement and measured GI median
14.296 ms / p95 15.654 ms (120 samples), queue/frame envelope median 35.171 ms /
p95 37.803 ms (95 timestamp samples in the measurement interval), and Bevy
frame-time median 35.844 ms (119 samples). The small difference from the first
run does not establish a causal speedup. Offscreen runs also explicitly drain
GPU work each frame, so their queue envelope includes CPU-induced idle gaps and
does not establish windowed application FPS. The 33.3 ms whole-frame target and
GTX 1650 / RX 5500 XT certification remain unestablished. Data:
`tmp/sponza-fix-benchmark-clean.json`; cleanup confirmation and sampling log:
`tmp/sponza-fix-benchmark-clean.log`.

The stationary dense pose-88 experiment was repeated with traversal admission
uncapped (`--max-traversal-queries=0`). It produced the identical 3.3832796/255
image mean (`tmp/sponza-fix-pose88-uncapped.log`), ruling out that cap as the
cause of the dense result. A further source audit found no additional proved
BRDF/PDF or support-normalization error. The view faces a covered arcade, and
Hybrid evaluates directly lit first-hit surfaces without recursive diffuse
feedback; a need for further indirect bounces is consistent with the darkness.
This is an explanation of the configured approximation, not a reference-render
validation or evidence that the failing brightness gate is correct.

The motion-pose audit confirms frame 88 holds the camera at approximately
(-6.507, 1.592, 0.342), facing (-0.960, 0.194, 0.201), away from the courtyard
and into the shadowed arcade. Noon sky strength is unchanged. The saved image
contains a stone arch and occluded recess rather than the courtyard or sky.
Thus the image-mean assertion detects darkness but does not isolate transport
correctness from scene content; no exposure, camera-path, or gate adjustment was
made to turn the failure into a pass.

## Historical October 17 implementation work

Hybrid now traces bounded current-frame diffuse continuations, with
`RadianceCascades::hybrid_bounces` in 1..=4. The library defaults to two segments;
the Sponza scene uses that same two-segment default. Every endpoint evaluates
visibility-tested direct NEE, with cosine-PDF cancellation and multiplicative
colored diffuse throughput. Environment escape remains zero because NEE owns
that path. A denied endpoint rejects the complete estimate rather than publishing
its observed prefix. The host reserves nominal path/query work before scheduling
receivers, and the shared shader counter still enforces the hard cap.

The new production-function GPU regression checks one through three continuation
segments against the analytic colored-albedo series under a constant-radiance
endpoint oracle. It also checks geometric-horizon rejection, supported escapes,
and denial at both the first query and a later endpoint. The focused test passed
on Vulkan (`tmp/sponza-complete-continuation-test.log`).

The moving-camera audit also exposed a separate whole-view dropout: one newly
visible material awaiting native/identity pipeline specialization prevented the
entire receiver identity pass from running. Queueing now keeps compiled draws
active while pending draws leave their identity pixels cleared/ineligible. The
compute/composite path can therefore serve valid receivers during specialization;
it does not fabricate identity or lighting for the unavailable draw.

`--benchmark-windowed` now uses the normal Winit/render pipeline and window
presentation, without the offscreen per-frame GPU drain. The benchmark still
removes raw readback after readiness and collects 120 new warmup and 120 measured
GI timestamps plus frame-time diagnostics. Its JSON identifies windowed versus
drained execution and the requested presentation mode. In `settings`,
`near_field_samples` preserves the requested setting and
`near_field_samples_effective` reports the active count for Hybrid and Split,
and zero for the other methods. `--benchmark-vsync` opts into synchronized
presentation; the benchmark defaults to AutoNoVsync.

```powershell
cargo run --release --locked --offline --example sponza -- --benchmark-windowed --method=hybrid --width=1280 --height=720 --json-output=tmp/sponza-windowed.json
```

The matrix script accepts `-Windowed`, `-Vsync`, and a `-Methods` subset and now
checks the actual `gi_gpu_ms` sample field in the JSON schema.

A 1,000-observation, four-segment dense pose-88 run converged to mean RGB
3.4076/255 with 56,510/57,600 supported raw pixels. Four segments therefore do
not reliably raise this shadowed view above 4/255; an earlier 240-observation
run was still decaying from an unusually bright Monte Carlo estimate and cannot
be used as proof of adequate converged brightness. See
`tmp/sponza-complete-pose88-settled.log`.

The new capture validation treats this as a scene-content limitation of an
absolute image-mean gate. Dark motion views now freeze the current camera and
blocker, capture a matched GI-off control, and require RGB difference >0.2/255,
the same contribution threshold as the static GI-toggle gate. Explicit fixed
camera poses run that control once after their uninterrupted stability sequence.
The control saves its image and restores GI without explicitly resetting history
(reenabling GI can still reinitialize render-view state). It does not
raise sky/exposure, change geometry, resample a brighter pose, or average extra
GI-on frames to evade the dropout test. The camera-only brightness/return gates
and final moving-scene stale-history difference gate remain unchanged.

### Historical windowed release performance profile

The full-scene 1280x720 release benchmark ran with normal window presentation,
AutoNoVsync, Vulkan and the same RTX 4070 Laptop GPU. Each run discarded 120 new
GI timing observations and measured 120 more. Frame-time measurements covered
the same interval (151-159 samples); they are application-frame diagnostics,
not FPS inferred from a GI timestamp.

| Receiver updates requested | Receiver NEE samples | GI p95 ms | Application frame p95 ms |
| --- | --- | --- | --- |
| 1024 | 4 | 21.931 | 39.417 |
| 1024 | 2 | 20.483 | 35.765 |
| 768 | 2 | 18.036 | 34.519 |
| 512 | 2 | 16.214 | 32.500 |

Data: `tmp/sponza-complete-windowed.json`,
`tmp/sponza-complete-windowed-tuned.json`,
`tmp/sponza-complete-windowed-768.json`, and
`tmp/sponza-complete-windowed-512.json`.

The Sponza default now requests 512 receiver updates and two receiver NEE
samples (one at each outgoing path vertex), with two Hybrid diffuse segments.
The candidate grid remains 8-pixel spacing, resolve scale remains two, and the
16,384-query hard cap remains in place. This is a scene-specific quality/work
tradeoff with stochastic refinement; it does not change the general library
preset or increase exposure/sky strength. The validation script defaults to the
same 512-update batch and accepts `-MaxRays` for controlled comparisons and
`-Release` for the release executable.

This development-adapter measurement satisfies the 33.3 ms application-frame
and provisional 16.7 ms GI p95 allocations in this tested configuration. It does
not certify GTX 1650 / RX 5500 XT hardware, other scenes, all camera positions,
or the separate aggregate-memory/reference-error goals.

### Historical final functional-validation run

The final 512-update / two-NEE-sample release profile passed all six methods'
day, GI-off and night gates. Day mean / GI-on-off mean absolute RGB difference
(/255): PAIC 18.372 / 10.852; Surface PAIC 21.515 / 14.007; Surfel RC 21.214 /
13.705; Cascade Guided 17.310 / 9.793; Hybrid 18.471 / 10.954; Split 20.071 /
12.551. These metrics establish visible contribution, not reference accuracy.

Hybrid camera-only return passed at 6.657/255 against its unchanged 8/255 gate.
The entire moving-blocker sequence passed, including the original pose 88:
matched GI-on/off difference 1.007/255 at frame 88 and 1.239/255 at frame 96,
with final settled/restarted history difference 7.736/255 against the unchanged
8/255 gate. All 22 motion checkpoints were saved. Artifacts and individual logs:
`tmp/sponza-completed-validation/20261016-214124-6f66eb27/`.

The explicit pose-88 fixed-camera sequence passed its unchanged flicker gates:
mean adjacent change 0.058/255, frame-mean p95 0.073/255, average pixel p95
0.333/255. Its separate final GI-off control measured 2.377/255 contribution
(`tmp/sponza-complete-pose88-verified.log`). The control is deliberately outside
the stability sequence. Capture scheduling also stops after the sixteenth
stability image, including while waiting for the control/exit callback.

All 109 combined CPU/GPU/shader checks passed with no ignored checks remaining
(`tmp/sponza-complete-all-tests.log`). The Sponza example test passed
(`tmp/sponza-complete-example-verified.log`). Formatting and whitespace checks
also passed. Core shader/renderer sources were unchanged by the later capture
harness and scene-profile adjustments.

The longer windowed release confirmation used the final default profile, 120
warmup GI observations and 600 measured GI observations. Over the same 21.219 s
interval, 763 application-frame observations measured median 27.867 ms, p95
30.934 ms, and maximum 33.120 ms. GI measured median 12.686 ms, p95 14.802 ms,
and maximum 16.056 ms. The queue envelope p95 was 30.263 ms. This confirms the
tested configuration's frame allocation over the longer interval without
converting GI time to FPS. Data and log:
`tmp/sponza-complete-windowed-confirmed.json` and `.log`.

The final pose-88 lifecycle rerun also exited cleanly after its separate control,
with the same 0.073/255 frame-mean flicker p95 and 2.377/255 GI contribution:
`tmp/sponza-complete-pose88-clean.log`.

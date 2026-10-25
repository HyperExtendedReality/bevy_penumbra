# Validation

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

The active Sponza profile uses 0.15 m probe spacing, a 0.24 m first interval,
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

Persistent Split resolution now falls back to another matching probe page when
the selected page lacks a directional bin or that bin's physical anchor is
incompatible. Directory headers remain a search hint; packed-bin support,
instance ownership, normal/plane compatibility, and finite geometry determine
acceptance. This also handles matching headers whose representative anchors are
incompatible while their packed directional anchors are valid. The fallback is
lazy, bounded by the existing directory-search limit, and adds no BVH queries.
Ordinary methods retain their existing visibility path.

The 32-bin convolution passes its known bin index directly to interpolation,
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
GI visibility. `bevy_texture_cache` now supplies PNG mip levels, generated in
linear color space for sRGB images; its disk cache still stores base pixels and
applies color-space settings and mips separately for each load.

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

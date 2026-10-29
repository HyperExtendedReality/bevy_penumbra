# Historical Split Radiance Cascades design record

This page records an earlier implementation plan and status. Its descriptions
of Hybrid, near-field sampling, secondary feedback, owner-specific cache records,
and layouts are historical, not the current source or public API. The current
crate provides Split Radiance Cascades and an optional secondary multi-bounce
cache; see [`rendering-boundaries.md`](rendering-boundaries.md) and
[`validation.md`](validation.md) for current behavior and validation.

At the revision documented here, the persistent directional-record pack,
lookup, and commit path had landed, including hierarchical direction allocation
and surface-owned canonical anchors. Secondary-cascade feedback and visual
quality still needed acceptance. Sponza selected Split by default while Hybrid
SPWI remained the comparison transport.

Sponza's selected Split path traces cascade intervals, merges their radiance and
transmittance, and evaluates the resulting directional field at G-buffer
receivers. It does not extend recursive per-pixel paths. The supplied
`RCPG_Ikkala2026.pdf` describes product path guiding for a path tracer; its
interval-bound discussion is reference material, not the selected renderer.
The separate offline reference example was independent of Sponza and the GI
plugin's execution path.

The historical comparison point is the pre-persistence 1280×720 windowed Split run
in `tmp/split-before-persistence.json`: RTX 4070 Laptop GPU, Vulkan, driver
617.42, 512 receiver updates, zero near-field samples, 120 warmup and 120
measured GI samples. GI p95 was 14.779392 ms and the Bevy frame-time diagnostic
p95 was 30.6804 ms. These timings predate the current cache refactor and do not
establish that the image is visually solved.

## The RC contract

Split RC stores radiance and transmittance over distance intervals. For cascade
`n`, probe `p`, and direction `ω`, let `Jₙ(p, ω)` be radiance returned by a hit
inside that interval, and `βₙ(p, ω)` be its transmittance: zero after an
occluding hit and one when the interval is empty. Merge from the farthest
cascade toward the nearest with `Iₙ = Jₙ + βₙ Iₙ₊₁`. A ray therefore provides
interval samples to several cascade levels; this remains radiance-cascade
transport, not a recursively extended per-pixel path chain.

The paper's reference configuration uses four cascades, a base direction count
of 32, branching factor `K = 4`, length scale `l = 4`, and `t₀ ≈ 1.6 Δs`.
Its level spacing grows as `2ⁿ`, directions as `Kⁿ`, and interval cutoffs as
`lⁿ`. Those values are starting points for controlled measurements, not a
promise that they fit this software-BVH implementation's budgets.

The paper's sparse layout seeds base probes from visible surface positions,
derives coarser probe locations from their parents, and interpolates over the
existing sparse neighbors rather than allocating every trilinear neighbor.
Its ray-splitting method assigns a low-discrepancy `R₂` sequence to each probe:
count visible source pixels per base probe, propagate those counts through the
cascade parents, prefix-sum the ranges, then use each range to generate
semi-uniform directions. This gives sparsely populated higher levels better
directional coverage than independent random sampling.

## Probe-major world records

The landed refactor replaces the old cosine irradiance prefilter with exact
probe-major directional records. A probe key identifies a quantized spatial
cell and its LOD. Each record carries 32 direction samples with radiance, actual
ray origins and surface normals, and explicit validity/visibility state. A
separate boundary array stores visible emission and escaped-sky visibility,
so secondary feedback can consume scattered radiance without counting NEE twice.
Its single directional record is 2,080 bytes, including aggregate dependency
bounds, versus 592 bytes in the previous layout. Non-Split methods use one
2,080-byte placeholder record; they do not
allocate the Split pool.

The Split pool currently reserves `buckets × 2 × 2,080` bytes for its
double-buffered records, based on pool capacity rather than the number of slots
currently holding probes. At 65,536 slots, the directional-record buffers
alone use 260 MiB, before hash tables, scratch, interval metadata, or secondary
data. The cap and eviction policy must be measured before increasing pool
density.

`seedProbes` and `packProbes` reuse the former seed/prefilter stage slots and
binding 12. Lookup groups eight spatial-probe headers per LOD; when that compact
directory overflows or a header has been evicted, it falls back to the existing
merged-record lookup chain. After a successful commit, lookup heads are remapped
to the completed records. Commit is bounded and retains the most recently used
eligible records, rather than depending on key-sort order. The retained world
cache survives camera changes independently of the screen-history TTL, subject
to LRU eviction and validity checks. Moving geometry currently invalidates the
Split world cache globally; per-segment dependency tracking is not implemented.

Split interval keys retain entity, connected-patch and geometric-normal class.
Clean surface-owned cells reuse their canonical anchor independently of the
current visible receiver. Primary Split uses base distance LOD zero; all four
distance cascades remain. G-buffer lookup can select geometrically compatible
probes across patch/class boundaries while retaining entity ownership.

Split's screen observations seed visible probes, while the bounded resident
triangle sweep supplies off-screen receiver candidates to its directional cache.
The sweep advances through a stratified candidate list across frames and respects
the per-view receiver and update caps. Candidate density is uniform by triangle,
not surface area, so highly tessellated objects can receive a larger share.
Balanced registry partitions rotate globally once per two frames, so truncated
batches cannot permanently starve an adjacent header. After commit, an evicted
directory representative can be replaced by a surviving directional record with
the exact same cell, owner, patch and class. The packed probe slot stays intact.

Frame-local ray counts, offsets, and reduction scratch remain separate from
retained world records. A frame must never publish an unsupported result as
observed black, and a reader must not see partially written records. The full
hierarchical direction allocator is implemented; its visual acceptance remains open.
Under an explicit query cap, trace/NEE work is limited to half the hard cap,
and merge visibility can use the full frame-wide hard cap. Split's dense world
resolve replaces the old visibility-ray final gather.

## Frame dataflow

1. **Read the deferred receiver inputs.** Use depth/world position, base color,
   metallic, roughness, shading normal, and the separate receiver identity and
   geometric-normal attachment. Opaque and alpha-mask support stays consistent
   with the existing scene path. Forward materials, transparency, and MSAA are
   not added by this refactor.
2. **Seed active probes.** Map visible receiver positions to stable base LOD
   cells, then derive parent cells for higher LODs. Keep the persistent world
   lookup distinct from this frame's visible active-probe list.
3. **Allocate directions hierarchically.** Count ray sources by base probe,
   propagate the counts to parent probes, and prefix-sum contiguous `R₂`
   sequence ranges. Use a stable probe key plus frame jitter so probe order
   changes do not silently reuse a direction with a different owner.
4. **Trace and deposit intervals.** Trace the generated directions through the
   selected scene backend. Split each result at the cascade cutoffs and deposit
   supported `Jₙ` and `βₙ` values to the corresponding probe/direction records.
   Preserve exact per-direction visibility and reduce multiple deposits with
   explicit support counts; keep an empty interval distinct from a supported
   zero-radiance interval.
5. **Merge far to near.** Apply `Jₙ + βₙ Iₙ₊₁` in reverse LOD order. Pre-average
   only data that the next merge consumes; do not keep redundant full-resolution
   intermediate fields without a measured reason.
6. **Shade the full-resolution receiver.** Sparse-interpolate compatible
   supported probes, then evaluate the agreed full Disney/GGX shading model,
   including its multi-scattering adjustment, from G-buffer material values and
   the receiver normal. The fixed-normal diffuse-only moment experiment below
   is not a substitute. Direct and indirect support remain explicit through
   temporal filtering and composition.
7. **Commit world state.** Publish completed records and remap their lookup
   heads only after the current deposits and merges are complete. The bounded
   commit retains recent eligible records; camera changes alter the active
   visible set, not the identity of retained world cells. Moving-geometry
   invalidation is currently global until segment dependencies are tracked.

## Secondary-cascade feedback

The current shader implements bounded diffuse and GGX feedback from the distinct,
coarser secondary RC cache; GPU and visual validation are pending. This extension
is disabled by default (`bounce_feedback = 0.0`), including in Sponza and Cornell.
The demos have no B feedback control. Applications can opt in through the existing
field; temporal interval accumulation and ordinary indirect transport remain active
without secondary feedback. Seed that
cache from first-hit world positions and normals produced by primary rays, as in
the paper's multibounce extension. Shade primary hits from the committed
secondary cache, and let the secondary cache read its prior committed state to
propagate further bounce energy over time. The paper uses secondary probes two
LODs coarser than the corresponding primary probes to reduce cost. This provides
feedback between radiance fields without turning each visible receiver into a
long sequence of recursively traced path vertices. Dynamic hit records need
the same owner and revision checks as primary records; stale geometry must not
feed persistent lighting.

## Bevy material integration

Keep imported glTF `StandardMaterial` assets and Bevy's native deferred punctual
lighting. In Bevy 0.20, `MaterialExtension::deferred_fragment_shader` changes the
mesh's G-buffer output; the later fullscreen deferred-lighting pass still owns
the BRDF evaluation. An extension alone cannot replace that pass's diffuse
function. Converting every imported material to `ExtendedMaterial` would also
require new scene extraction and receiver support in this crate.

The selected integration uses the connected `bevy_slang` compiler for the RC
resolve and a separate modular BSDF shader, reading the existing G-buffer.
This keeps native direct GGX and shadow bindings intact. Primary punctual NEE
must not be added again by GI; secondary ray hits still require punctual NEE
because Bevy never raster-shades those hit points for the transport cache.

The material response separates multi-scatter GGX reflection from the diffuse
lobe. It uses the native correlated DFG lookup table for GGX compensation.
The mobile analytic DFG polynomial is not an interchangeable substitute:
the CPU investigation measured nearly 1.9 reflected energy from a unit-energy
white environment for one rough, grazing metal case when that polynomial was
paired with correlated Smith visibility. The correct lookup table and sampler
PDF must pass energy checks before visual acceptance.

The CPU oracle passed 3,316 checks, including unit white metal and dielectric
furnaces (maximum energy error 0.000687) and VNDF PDF checks. This validates the
BSDF math, not the sparse lighting reconstruction. The normalized Burley lobe
uses a view-conditioned energy correction; this is an approximation and is not
reciprocal. The primary reflection resolve uses a deterministic convolution of
the 32 RC angular bins: a four-sample screen-random VNDF resolve produced visible
grain and failed the first 1280×720 camera-return gate (10.374/255 against 8/255).
The 32-bin split-sum resolve and coplanar geometry
compatibility checks also remain approximations: sharp reflections blur, and
separated coplanar surfaces can share light without an explicit visibility test.

Incident data must remain separate from the receiver material. Baking a full
colored BSDF into the existing sparse diffuse RGB history would blur material
textures and apply albedo twice at composition. The dense world resolve evaluates
Burley and GGX kernels at GI resolution using the G-buffer normal, view and
roughness. Their uncolored fields have separate temporal and spatial histories;
glossy reuse also checks shading normals, roughness and view direction. Bilateral
full-resolution composition applies albedo, metallic/reflectance, native DFG
energy and diffuse normalization once. Sky visibility and directly visible
emission remain separate boundary payloads. Primary diffuse and reflection both
read those terms, while secondary feedback keeps scattered radiance separate
from its own direct NEE to avoid double counting.

True BTDF transmission is not implemented by the supported opaque/alpha-mask
material path. It needs IOR, thickness or entry/exit surfaces, medium state and
refracted transport; an alpha blend cannot stand in for that model.

References: pinned Bevy 0.20 `extended_material.rs`,
`deferred/deferred_lighting.wesl`, `deferred/functions.wesl`,
`render/pbr_lighting.wesl`; the sibling `bevy_slang` compiler/loader; and
[Filament's multiple-scattering discussion](https://google.github.io/filament/Filament.html).

## Historical Disney diffuse moments and rejected normal-atlas shortcut

The Split RC paper's optimized final shade prefilters a 6×6 octahedral
irradiance field under a Lambertian assumption. That shortcut is not compatible
with the target full Disney/GGX response. The selected probe records retain
per-direction radiance, origins, normals, and visibility so the full material
response can be evaluated at the receiver. The fixed-normal moments below were
an exact CPU experiment for the Disney diffuse shape term only; they are a
historical, rejected optimization for this target. They do not represent the
specular BRDF, its multiple-scattering adjustment, or complete material
evaluation.

For a fixed normal `n`, incident directions `l = (x,y,z)`, and incoming RGB
radiance `L`, the CPU check forms 14 RGB moments
`Mₖ = Σᵢ (4/N) max(0,n·lᵢ) confidenceᵢ Lᵢ φₖ(lᵢ)`. Its basis is
`φ = [1,x,y,z,p,px,py,pz,px²,py²,pz²,pxy,pxz,pyz]`, where
`p = (1-max(0,n·l))⁵`. To evaluate the Disney diffuse shape for view `v`, set
`r = max(0.089,roughness)²`, `b = r−0.5`, `s = (1−n·v)⁵`, then
`A = 1+bs`, `B = sr v`, `C = b+b²s`, `D = r(1+2bs)v`, and
`E = sr² vvᵀ`. The BRDF factor expands to
`A + B·l + p(C + D·l + lᵀEl)`, which contracts against those 14 moments;
apply the renderer's diffuse material factor and `1/π` as in the direct
evaluation. `scripts/check_disney_rc_prefilter.py` verifies this fixed-normal
identity over 6,048 CPU cases, with maximum relative error `1.87e-15`. It does
not validate interpolation between normals or fp16 storage. A proposed
64-normal fp16 atlas would use `14 × 64 × 8 = 7,168` bytes per probe, or about
448 MiB at 65,536 probes, before any other cache data. The 6×6 normal-atlas
experiment also reached up to 8.6% error at incident-light peaks for a constant
input. No normal atlas has been implemented or GPU-measured; the diffuse-only
fit is not the selected full-BRDF representation.

## Validation and timing boundaries

Split's primary G-buffer resolve convolves the same complete world incident
field for Burley diffuse and multi-scatter GGX: scattered radiance plus emitter
boundary radiance plus transmitted environment. It no longer obtains primary
diffuse sky/emitter lighting from screen-only sparse history. Secondary feedback
still uses scattered radiance with its own direct NEE to avoid double counting.
The dense resolve replaces Split's old visibility-ray diffuse gather and primary
NEE pass. It reuses the two sparse image pairs for raw/filtered specular and
unfiltered specular history, with four additional moments/receiver images.
Candidate-grid clearing skips these reused images, which now have GI extent.
The fused loop reuses its 32 directional cache reads; tiny emitters and sharp
reflections remain limited by that angular resolution. Camera and stationary
captures must validate the resulting change before it is considered stable.

The resolve carries observed spatial/angular kernel coverage independently from
conditional radiance. A lone supported direction no longer claims a full-lobe
observation: temporal confidence grows by its coverage, with proportional blend
authority. Partial observations cannot clamp or radiometrically reset complete
history. Full coverage preserves the existing behavior; supported black remains
distinct from unknown. This is confidence accounting for sparse extrapolation,
not an unbiased reconstruction of missing directions. GPU fixtures cover unknown,
partial unit/black, full unit, spatial coverage, and both temporal lobes.

Coverage-aware world-probe averaging was tested and removed: the long offscreen
revisit worsened to 14.671/255 despite passing its focused GPU tests. It cannot
repair an estimator whose surface contributors and anchor change behind one
world-cell key. Fresh anchors whose child queries are denied remain unknown
instead of acquiring an old anchor's radiance; that ownership fix is retained.

The implemented structural change preserves entity/connected-patch ownership and a
surface-orientation class through Split's existing hierarchy. Each discovered
surface-owned cell keeps a clean canonical anchor rather than replacing it with
this frame's visible receiver. Full-resolution G-buffer lookup selects compatible
owned probes. Exact-anchor raw interval running means retain accumulated sample
mass; final merged-field filtering reuses only matching owned anchors. The
current exact-pack camera capture passes the unchanged 8/255 gates: 4.780
immediate, 5.303 settled and 6.249/6.441 on long revisits
(`tmp/ggx-rc-exact-pack-camera.log`). Stationary gates pass at mean 0.128/255,
frame-mean p95 0.149 and average pixel p95 0.667. The moving-blocker
settled/restarted comparison still fails at **13.298/255**; dynamic quality is
unresolved, and dependency-aware geometry reuse remains a plan.

The exact-pack benchmark records GI p50 10.446 ms / p95 11.016 ms (120 samples)
and whole-GPU p50 24.557 ms / p95 25.197 ms (153 samples):
`tmp/ggx-rc-exact-pack-benchmark.json`. The 131,072-entry Sponza cache remains
within its 512 MiB view budget; updates (512), queries (16,384), warmup (120)
and quality gates are unchanged. The corrected exact-pack GPU fixture awaits
rerun and a full suite, so this is not full certification. The preceding raw
running-mean revision passed all 113 checks in 96.88 seconds and the BSDF CPU
oracle passed 3,316 checks, but its long camera return still failed at 10.514.
Earlier remap/coverage timings belong to their own preceding implementations.

Final-probe temporal filtering and retained-anchor experiments failed the camera
gate (13.37/255 and 12.84/255 respectively) and were removed. One initial anchor
lookup experiment also caused a native GPU device-loss error; the bounded seed
variant ran but did not stabilize the image. These are rejected approaches.

The new record, lookup, and commit path is under GPU validation. Keep the
existing capture and motion gates unchanged. Validate interval algebra and
support handling, then run the static, camera-only, moving-blocker, and
fixed-pose captures with raw support and scene-linear diagnostics. Include light
and geometry revisions to prove that retained records neither leak stale energy
nor discard compatible static history. Report failures as failures; do not
infer correctness from GI-on/off brightness alone.

The Split RC paper reports 8.6 ms for its Sponza case on an RTX 3080 Laptop GPU,
using LuisaCompute over CUDA and OptiX. Its timing excludes G-buffer creation and
shadow rays, and it is not a performance estimate for this crate's software BVH
or the RTX 4070 Laptop GPU. The local pre-persistence JSON above is a historical
comparison point. Current GPU validation, secondary feedback, full allocation
policy, visual quality, and target-device performance remain incomplete.

References: [Split Radiance Cascades, v2](https://arxiv.org/abs/2607.20384),
especially Sections 3–7; [original Radiance Cascades
paper](https://github.com/Raikiri/RadianceCascadesPaper/blob/main/out_latexmk2%2FRadianceCascades.pdf); and the local CPU check
`scripts/check_disney_rc_prefilter.py`.

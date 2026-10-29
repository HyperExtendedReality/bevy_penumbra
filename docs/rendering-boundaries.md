# Rendering boundaries on Bevy 0.20

Penumbra extends Bevy's deferred renderer. It consumes receiver normals,
material data, depth, and motion vectors from the existing prepasses.

## Visibility buffers and traditional geometry

Bevy 0.20 already has the hybrid path in `bevy_pbr::meshlet`: a material
resolve pass creates the deferred G-buffer from meshlet visibility records.
Traditional meshes use Bevy's ordinary deferred prepass. A separate rendering
crate would duplicate that mechanism.

Meshlets require the `meshlet` feature, preprocessed `MeshletMesh3d` assets,
supported Vulkan or Metal hardware, `Msaa::Off`, and a consistent global opaque
rendering method. Penumbra's receiver identity pass currently supports
`Mesh3d` with `MeshMaterial3d<StandardMaterial>` geometry; meshlet receiver identity extraction and GI
scene registration still need an adapter before meshlets can participate in SRC.
The native hybrid path's existence does not make Penumbra meshlet-compatible.

See Bevy's [meshlet implementation](https://github.com/bevyengine/bevy/tree/v0.20.0/crates/bevy_pbr/src/meshlet)
and [0.20 release notes](https://bevy.org/news/bevy-0-20/).

## Sparse scene uploads

`AtomicSparseBufferVec` uses a GPU scatter pass, owns its allocation, and can
fall back to a complete upload when it grows or when many elements change.
Its `write_buffers` operation does not expose a byte budget.

Penumbra's scene uploads must respect per-frame byte limits and publish a
replacement acceleration structure only after all buffer and texture pages
have arrived. The native helper therefore cannot replace that staging path
without also changing scheduling and memory accounting. Dynamic overlays can
instead compare CPU word snapshots and upload changed contiguous spans into
the existing allocation. GPU-written refit nodes require particular care:
the CPU comparison snapshot describes CPU writes, not GPU readback contents.

Attribute-only changes repack vertex words while retaining the existing triangle
indices, BLAS nodes, and mesh spans. Texture-atlas cache hits avoid copying the
sorted texture identifiers; full atlas rebuilds still validate all inputs before
publishing the replacement.

Split world-cache invalidation checks the tracked ray-segment and hit-lighting
dependencies against changed geometry and light bounds. Unknown dependencies
invalidate conservatively. Raw intervals and composed cones track separate
dependencies so a dirty child can invalidate the cone without discarding a
clean local interval.

## Split transport and light ownership

Split RC uses ray tracing to build a sparse world-space radiance field. Rays
start at visible surfaces, deposit transparent lower intervals and an opaque
first-hit interval, then leave farther intervals unobserved. The final cascade
is a terminal interval: it accepts distant hits and supported-zero escapes.
Four distance cascades are distinct from camera-distance LODs.

Primary diffuse shading uses the paper's Lambertian preintegration. Each world
probe stores three 6×6 octahedral irradiance fields: directional/environment,
emissive, and finite punctual. Each RGBA32F texel carries conditional irradiance
and its own angular support. Two aggregate dependency float4s bring the probe
record to 1,760 bytes; each interval entry occupies 272 bytes.
Probe-directory identity includes position, cascade level, and LOD, but not
ray direction; the canonical representative may be any traced direction.

Resolve folds bilinear taps across octahedral seams and independently normalizes
each source through angular, spatial, and LOD interpolation before summing.
The four wrapped octahedral texels and bilinear weights are computed once per
receiver normal and reused for every neighboring probe and both LODs.
Output coverage is the minimum of active source coverages. An inactive emissive
or punctual group is analytically zero; an active group without observations is
unsupported, not black. Source changes preserve independently valid fields;
geometry changes remain conservatively bounded by aggregate dependencies.
Screen reconstruction also retains a separate background estimate across
punctual or emissive changes. Combined history is rejected on those changes;
unchanged background history is reused only after receiver validation and is
discarded when geometry is dirty. When an active source has no current
observation, the display can show the independently supported sources while
raw output continues to report incomplete coverage.
The increased probe size can cause memory-constrained views to adapt their grid.
Receiver albedo and metallic attenuation are applied at composition. Bevy
retains native punctual and environment lighting, their shadows, and visible
surface specular shading. Penumbra adds no glossy receiver transport.
The optional `multi_bounce` cache uses prior primary hit surfaces and feeds
previously completed, two-LOD-coarser secondary irradiance back into ray-hit
shading. Its workload shares the configured update, query, and cache limits.

Opaque and mask materials use their authored binary coverage. Blend and
premultiplied materials participate in transport with stochastic coverage from
base-color, vertex, and texture alpha; shadow and closest-hit rays pass through
rejected samples. Mesh-emitter samples weight fractional coverage explicitly.
Transparent forward surfaces do not receive the deferred GI composite, and
alpha coverage is not refraction or participating-media transmission.

The update scheduler walks a stable candidate-rank permutation with a circular
window, including budgets that cross the end of the candidate list. When
persistent world history is enabled but not yet valid, three extra rank phases
run after the ordinary phase and before parent/prefix construction; that cold
priority frame does not advance the fair cursor. With `history_frames=0`, the
extra phases stay off and the ordinary fair cursor continues to advance.

Cold initialization picks a frame-seeded eligible receiver per base probe and
counts the probe's projected candidate population. A bounded logarithmic
histogram admits denser probes first, up to the existing update cap; atomic
tickets break ties within the boundary population bin. Remaining updates come
only from the current fair window. The histogram and population counts reuse
parent-count scratch, which parent construction overwrites before hierarchical
prefixing. Ray directions and radiometric support are unchanged.

All participating emissive surfaces contribute radiance when a primary ray
hits them. There is no `RadianceCascadesEmitterSampling` classification or
separate primary next-event estimator. Removing the old component from scenes
is the only migration needed. Small emitters can require more ray samples to
converge.

Next-event estimation is independent of radiance cascades. This implementation
uses it to evaluate direct lighting at secondary ray-hit surfaces, including
mesh emitters, punctual lights, and the configured transport environment.
Those paths contain a reflected hit surface; they do not duplicate first-hit
emission. Secondary material evaluation uses the existing Disney/GGX BSDF
helpers. The cascade algorithm itself does not require NEE.

Merged RGB is a conditional radiance mean and its support is tracked
separately. Sparse interpolation weights both quantities consistently; observed
black is valid and unavailable traversal is not black. Temporal interval
retention is per observation, with independent radiance and transmittance
weights; frames without observations do not decay unchanged world state.

Finite positive interpolation weights and coverage are retained without an
epsilon cutoff, including sparse corner and LOD-overlap tails. Per-axis weights
select the fraction or its complement directly; arithmetic interpolation between
those endpoints can cancel a small positive fraction. The raw alpha's supported
bit survives even when its fractional coverage is too small to represent.
Vector finiteness checks classify IEEE-754 exponent bits directly, preserving
infinities and NaNs as invalid while avoiding Slang's dynamically indexed
vector loops. This also prevents those loops from forcing enclosing DX12 FXC
loops to unroll.

Traversal step limits and clipped leaf scans produce unsupported queries, not
supported black lighting or proven-clear shadows. Closest-hit queries require
complete traversal. An accepted blocker still proves an any-hit shadow query,
even when another branch was incomplete; unknown dependencies cannot enter
retained lighting as valid observations.

Perspective LOD uses Chebyshev camera distance and a 0.9 overlap between
adjacent levels. Orthographic LOD depends on projected spacing rather than
camera distance. Both use the same five-bit world-key LOD field, capped at 24
for finite spacing and interval arithmetic; merging never crosses LODs.

Full-resolution reconstruction uses current receiver identity, geometric
normal, world-plane compatibility, and lighting-support metadata. Sampling and
interpolation share clamped integer anchors, including partial edge tiles and
nonzero viewport origins. If no compatible screen sample exists, the compositor
queries the world probes at the actual pixel position and shading normal; it
does not borrow an unrelated receiver's screen lighting. A supported black
sample remains black.

Raw `RadianceCascadesRawOutput` RGB is scene-linear diffuse transport divided
by pi, before receiver albedo, intensity, occlusion, and camera exposure.
Its alpha distinguishes background, missing lighting, and supported lighting;
the fractional supported alpha carries coverage. See the public component
documentation for its encoding.

Reference: the supplied [Split Radiance Cascades paper](../ref/Split%20Radiance%20Cascades.pdf)
and [Chandra source](../ref/chandra/), especially paper §§4–6.

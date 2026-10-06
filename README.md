# bevy_penumbra

Experimental 3D diffuse global illumination for **Bevy 0.19.1**, based on
**Split Radiance Cascades**. GPU kernels use **Slang**, compiled through the
sibling `bevy_slang` crate. World-space triangle traversal includes offscreen
geometry, emissive surfaces, and secondary bounce positions.

## Run

Requires Rust 1.95+, Slang 2026.19+, and a desktop GPU with compute shaders,
eight compute storage bindings, and RGBA32F storage textures. Put `slangc` on
PATH or set `SLANGC`. This checkout expects `../bevy_slang`.

```sh
cargo run --release --example cornell_box
```

Space toggles GI, B toggles additional bounces, arrows move the emitter, P saves a screenshot,
and Escape exits. The quality defaults enable secondary bounces, a half-resolution
sampling grid (subject to the ray budget), 32-frame probe retention, three spatial
filtering steps, and 16 direct-emission samples per receiver.
It contains an emissive ceiling panel, with no ambient or conventional lights.

## Integrate

```toml
[dependencies]
bevy = { version = "0.19.1", default-features = false, features = ["3d"] }
bevy_penumbra = { path = "path/to/bevy_penumbra" }
```

```rust
use bevy::prelude::*;
use bevy_penumbra::{RadianceCascades, RadianceCascadesGeometry, RadianceCascadesPlugin};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(RadianceCascadesPlugin::default())
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn((
        RadianceCascades { bounce_feedback: 0.8, ..default() },
        Transform::from_xyz(0.0, 2.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    // Tag opaque Mesh3d + MeshMaterial3d<StandardMaterial> entities with
    // RadianceCascadesGeometry, including individual loaded-scene descendants.
}
```

The camera component supplies `Camera3d`, HDR, depth/normal/deferred prepasses,
and `Msaa::Off`. Set `Msaa::Off` explicitly when an existing camera already has
MSAA. The plugin selects the **global default deferred opaque renderer**;
explicit forward materials do not receive GI. Meshes must retain CPU-readable
positions/indices (`RenderAssetUsages::MAIN_WORLD`, the default for primitives).
All tagged, inherited-visible meshes affect every GI camera; render layers are
currently not filtered in the tracing scene.

GI is added after opaque lighting, before transparency and tonemapping. Receiver
albedo, metallic, baked occlusion, normals, and camera exposure come from Bevy's
prepasses. Unlit/black and fully metallic receivers are bypassed. Bevy continues
to render direct and specular lighting. Ray hits use flat material colors and
emission, plus point, directional, and spot lighting with BVH shadow rays.
Spot cone falloff follows [Bevy's inner/outer angle controls](https://docs.rs/bevy/0.19.1/bevy/light/struct.SpotLight.html).
Light radius affects Bevy's specular response; diffuse GI uses inverse-square
attenuation with the same smooth range cutoff as Bevy.

## Pipeline and research

The latest directly relevant paper found as of October 5, 2026 is Freeman and
Sannikov, **[Split Radiance Cascades: Real-Time Global Illumination via Sparse
Radiance Probes](https://arxiv.org/abs/2607.20384)**, July 22, 2026 (v1).

1. Count visible rays in base probes, propagate counts from unique children to
   parents, then assign contiguous R2 sequence segments down the hierarchy
   (Algorithm 3). Independent root probes use independent sequence phases.
   Integer phase arithmetic avoids long-session float sequence quantization.
2. Trace one hemisphere ray per sample with a software triangle BVH. Split it
   into transparent intervals and one terminal interval; never deposit an
   occluded ray into higher cascades (Section 5).
3. Insert full spatial/angular keys into hashed chains, separated by dominant
   surface-normal orientation to keep opposing faces from sharing hemisphere
   samples. A bucket-owned reduction consolidates duplicate samples once,
   without float atomics or spinlocks. Keep an actual surface origin nearest
   the cell center; averaging positions can put visibility origins inside walls.
4. Merge four cascades from coarse to fine (Eq. 7), with spatial spacing ×2,
   direction counts ×4, and cutoff ×4. Cascade zero has 32 equal-area directions.
   Unknown angular samples are excluded rather than treated as black.
5. Retain unsampled world-space entries for a bounded lifetime and blend fresh
   values with history. Only fresh canonical entries need to merge again.
6. Split the direct-emission budget between power-weighted triangle samples and
   cosine-direction samples, combined with the
   [balance heuristic](https://www.pbr-book.org/4ed/Monte_Carlo_Integration/Improving_Efficiency).
   This bounds near-emitter sample weights without clamping light energy.
   Stratified selection, independent world-space rotations, and BVH visibility
   reduce variance. Ray hits evaluate this direct irradiance too, so moving emitters supply
   reflected lighting immediately. Emission is excluded from the hemisphere-ray
   estimator to avoid counting direct emitter paths twice.
   Emitter order stays stable across BVH rebuilds to prevent sampling jumps.
7. With feedback enabled, spawn an extra ray at each primary hit into a secondary
   cache with four-times probe spacing/cutoff (two higher LODs). Both primary and
   secondary hits read its previous-frame irradiance, allowing offscreen surfaces
   to contribute additional bounces once the scene is unchanged.
8. Prefilter diffuse irradiance into six nonnegative ambient-cube lobes, each
   normalized by its known angular coverage. This preserves constant illumination
   and avoids the ringing and bright patches caused by sparse SH normalization.
   Interpolation checks visibility to actual surface sample origins to reduce
   wall leaks, including the directional fallback. Shading blends eight probes
   instead of gathering directional bins. Cascade merging samples at the stored
   surface origin rather than a cell center inside geometry.
9. Reproject irradiance using the actual prepass pixel centers, previous-world
   surface-plane and normal rejection, neighborhood
   clamping, temporal variance, and a reactive history weight on scene edits.
   Unchanged scenes retain their full history weight instead of treating noisy
   samples as lighting changes. Apply variance-aware,
   normal/plane-aware à trous filtering and bilateral upsampling. Only temporal
   output becomes screen history; spatial filtering never feeds back recursively.

All sources are embedded; no shader-asset copying is needed. `bevy_slang` compiles
two modules containing 17 entry points once, at optimization level 2, to validated
WGSL. Slang is not itself a guarantee of faster execution. Compiler failure logs a diagnostic and
disables GI. No handwritten WGSL or unsafe SPIR-V passthrough is used.

This is a portable adaptation of the paper's CUDA/OptiX implementation:

- Primary probes use fixed world-space spacing; adaptive camera-distance LODs
  and transitions remain to be implemented.
- Secondary rays use independently hashed R2 phases without their own hierarchical ordering.
- Sparse directional entries replace dense angular atlases. Bucket reduction
  is serial; unusually long collision chains can still be expensive.
- An ambient cube replaces the paper's octahedral irradiance texture. Sharp angular
  distributions lose detail. Probe-hash searches are bounded to
  32 attempts, with an exact directional-gather fallback.
- Retention has both an age and a capacity limit; it is not a permanent world cache.
  Geometry, material, and lighting edits invalidate world-space history. Screen
  history survives these edits with geometry rejection and lighting clamping;
  viewport/settings changes invalidate both histories.
  Invalidation preserves the frame sequence, so continuously moving scenes sample
  new directions instead of repeatedly tracing the first frame's rays.
- Geometry edits rebuild the median BVH. Static frames reuse it; light-only edits
  reuse CPU geometry and upload only the light buffer. Large moving scenes still
  need BVH refitting/instancing.

## Tuning and memory

`RadianceCascades` exposes spacing, cutoff, resolution divisor, primary ray budget,
bias, environment radiance, strength, temporal blend, retained frame count,
secondary feedback, emissive sample count, and spatial filter steps. Start with
`interval_length = 1.6 * probe_spacing`. The cascade cutoffs scale by four;
secondary cutoffs are four times larger. The last cascade covers all remaining
distance: rays continue to the nearest hit or escape the scene, so distant
geometry still blocks the environment.

Smaller spacing retains detail but reduces reuse. More rays improve coverage.
`history_frames = 0` disables retention; `temporal_blend = 0` disables accumulation.
`bounce_feedback = 0` disables secondary rays and halves interval allocation;
explicit emitter sampling still supplies direct and reflected lighting.
`spatial_filter_steps = 0` disables denoising; three is the default, four the maximum.
`emissive_samples` ranges from 1 to 32 (default 16); ray hits use half this count.
This total budget is split between triangle and cosine samples; a count of one
uses triangle sampling alone. Increase it for small emitters or noisy soft shadows. Selection cost is logarithmic
in emitter count; shadow-ray cost depends on scene complexity.
The default temporal blend is 0.92. Moving scenes retain reactive screen history
and a freshly evaluated reflected bounce; deeper feedback settles in static frames.
For a lower-cost starting point, use divisor 4, 16 retention frames, two filtering
steps, and eight emissive samples. Reprojection can still ghost at disocclusions,
and spatial filtering can soften contact shadows.

For `R` primary rays and `L` ray layers (one or two):

| Allocation | Bytes per camera |
|---|---:|
| Two interval caches, each with fresh/retained halves | `1280 * R * L` |
| Two ambient-cube irradiance-probe grids | `224 * B` |
| Two hash buffers | approximately `16 * B` |
| Two irradiance histories, two display textures, two metadata textures | `96 * R` |

`B` is the next power of two at or above `max(1024, 2 * R)`. The 640×480 default
uses 76,800 primary rays plus up to 76,800 secondary rays and about 254.5 MiB,
excluding geometry and Bevy's prepasses. The divisor-4 configuration uses about
63.6 MiB. The 131,072-primary-ray ceiling can use about 392 MiB with feedback.
Device binding limits automatically reduce the ray budget (for example to at most
104,857 primary rays with feedback and a 128 MiB binding limit). Each camera owns
independent buffers.

Tracing supports opaque indexed/non-indexed `TriangleList` meshes with flat
`StandardMaterial` base color, metallic, and emission. Textured ray-hit materials,
alpha masks, transmission, skinning, morphs, custom materials, `RectLight`, light texture masks,
glossy GI, and browser builds are unsupported. Emissive triangle importance
sampling is supported. Thin surfaces, sparse coverage, and normal-sector
boundaries can still cause interpolation artifacts; finite sample budgets can
leave soft-shadow noise.

## Checks and measurements

```sh
cargo fmt --check
cargo check --all-targets
cargo test --lib -- --include-ignored
cargo run --release --example cornell_box -- --capture-reference
cargo run --release --example cornell_box -- --capture-near-field --history=0 --filter=0
cargo run --release --example cornell_box -- --capture-distant-occlusion
cargo run --release --example cornell_box -- --capture-feedback
cargo run --release --example cornell_box -- --capture-offscreen
cargo run --release --example cornell_box -- --capture-dynamic
cargo run --release --example cornell_box -- --capture-moving
cargo run --release --example cornell_box -- --capture-camera-motion
cargo run --release --example cornell_box -- --capture-quality
cargo run --release --example cornell_box -- --capture-feedback --light=spot
cargo run --release --example cornell_box -- --capture-dynamic --light=point
cargo run --release --example cornell_box -- --capture-moving --light=directional
cargo run --release --example cornell_box -- --capture-no-gi
cargo run --release --example cornell_box -- --benchmark
```

Captures run headlessly, save PNGs to `screenshots/`, and return a failing exit code
when checks fail. The reference uses a constant unit environment with tonemapping
disabled and checks the expected exposed diffuse color. Near-field mode places a
unit-radiance emitter just above a diffuse plane and requires color error below
2/255, including with retention and filtering disabled at the default sample budget.
Distant-occlusion mode first proves environment illumination, then encloses the
receiver in black walls beyond the old far cutoff and requires it to become
black. This also checks scene-edit invalidation and prevents an inactive GI
pipeline from passing the occlusion test.
The feedback check requires
lighting on the small box's front face. Offscreen mode places the emitter behind
the camera. Dynamic mode first checks that the indirect-only front receiver is
lit, switches off the source after three seconds, and requests a black-receiver
capture about 100 ms later. Moving mode animates the source continuously and checks
indirect lighting while the world cache is invalidated each frame. Camera-motion
mode exercises reprojection. Quality mode saves 16 frames for temporal measurements.

Headless modes accept `--light=emissive|point|directional|spot` (default: emissive).
Run `--capture-feedback`, `--capture-moving`, `--capture-dynamic`, and
`--capture-no-gi` for each source. Directional mode opens the ceiling to admit
sunlight. With conventional lights, the GI-disabled check requires a black front
receiver while allowing direct lighting elsewhere. Conventional captures append
the source name to their PNG filename.

`--benchmark` records GPU timestamp spans after a three-second warm-up and writes
mean, median, p95, and maximum times to `tmp/benchmark.txt`. Missing timestamps are
reported as unavailable. Headless modes accept `--divisor=N`, `--history=N`,
`--filter=N`, `--emitter-samples=N`, and `--single-bounce`. To measure reduced-cost GI:

```sh
cargo run --release --example cornell_box -- --benchmark --divisor=4 --history=16 --filter=2 --emitter-samples=8
```

See [validation measurements](docs/validation.md) for the tested hardware, workload,
results, and comparison limits. Benchmark your own scenes and target hardware.

Licensed under MIT OR Apache-2.0.


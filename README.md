# bevy_penumbra

`bevy_penumbra` implements diffuse global illumination in Bevy with Split
Radiance Cascades, following the strict base of the v2 paper. It uses persistent
world-space intervals with validated temporal reconstruction.

The crate and its local helper crates target Bevy 0.20.0.

## Add GI to a scene

Install `RadianceCascadesPlugin`, attach `RadianceCascades` to a camera, and mark
participating meshes with `RadianceCascadesGeometry`:

```rust
use bevy::pbr::DefaultOpaqueRendererMethod;
use bevy::prelude::*;
use bevy_penumbra::{RadianceCascades, RadianceCascadesGeometry, RadianceCascadesPlugin};

App::new()
    .insert_resource(DefaultOpaqueRendererMethod::deferred())
    .add_plugins(DefaultPlugins)
    .add_plugins(RadianceCascadesPlugin::default())
    .add_systems(Startup, setup)
    .run();

fn setup(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, mut materials: ResMut<Assets<StandardMaterial>>) {
    commands.spawn((
        Camera3d::default(),
        RadianceCascades::default(),
        Transform::from_xyz(0.0, 2.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(10.0, 10.0))),
        MeshMaterial3d(materials.add(Color::WHITE)),
        RadianceCascadesGeometry,
    ));
}
```

GI uses deferred rendering so the renderer can read receiver material and
surface data. Receiver inputs are depth, packed G-buffer base color, metallic,
roughness and shading normal, plus a separate raster attachment for receiver
identity and geometric normal. Opaque and alpha-mask surfaces receive GI.
Alpha-blended and premultiplied meshes can participate as shadow casters and
emitters using stochastic texture-alpha coverage; they do not receive the
deferred GI composite. This is not refractive or volumetric transmission.
Forward-only receivers and MSAA remain unsupported.
The examples show setup, controls, raw output, and scene registration patterns.

## Tune a view

`RadianceCascades` contains the per-camera quality and appearance settings.

For the 1280×720, 4 GB GPU profile, use `RadianceCascadesPlugin::low_end()`
and `RadianceCascades::low_end()`. Cornell and Sponza use this profile by default;
the library's `Default` settings retain their work and memory budgets. The profile
uses 1 m base probes, a 1.6 m cascade-zero cutoff, and 65,536 retained cache
records. Smaller probes need more completed transport work to converge smoothly.
It targets GTX 1650 / RX 5500 XT class hardware; that performance target has not
been certified on those GPUs. See [quality and performance validation](docs/quality-and-performance.md)
for the measurement protocol and current evidence.

`sampling` separates candidate-grid density from selected receiver updates,
cache capacity, traversal work, and resolve scale. In explicit sampling mode,
lowering `max_rays` reduces selected updates while retaining the candidate grid.
An all-zero workload configuration retains the legacy grid/budget relationship.
Use `..Default::default()` or `..RadianceCascades::low_end()` in struct literals
to account for the new `sampling` field.

Use `..Default::default()` in struct literals to inherit shared settings.

| Setting | Purpose |
| --- | --- |
| `probe_spacing`, `interval_length` | Set the world-space cascade spacing and interval scale. |
| `resolution_divisor`, `max_rays` | Set the sparse receiver grid and trace budget; geometry-aware reconstruction fills the full viewport. |
| `normal_bias` | Offset surface samples to reduce self-intersection. |
| `intensity`, `environment` | Set GI strength and the diffuse environment contribution. |
| `multi_bounce` | Enable an additional off-screen cache for diffuse multi-bounce lighting; disabled by default. |
| `cached_hit_sky` | With `multi_bounce`, shade ray-hit sky light from the secondary cache instead of environment visibility rays; disabled by default. |
| `hardware_ray_queries` | Trace the primary closest-hit and visibility passes with hardware ray queries; falls back to the software BVH when unavailable. Disabled by default. |
| `history_frames`, `temporal_blend` | Control validated screen and world-space history. Zero disables primary history; the optional secondary cache still retains history for temporal feedback. |
| `split_transmittance_blend` | Set per-observation retention for interval transmittance while lighting radiance refreshes. |
| `spatial_filter_steps` | Control geometry-aware a-trous filtering for screen history; zero disables the screen-space filter. |
| `emissive_samples` | Tune emitter sampling. |
| Diffuse transport | Split evaluates diffuse illumination, including emissive first hits. Bevy retains native punctual/environment lighting, their shadows, and visible-surface specular shading. Screen history checks receiver identity and surface compatibility. |
| Split transport | Each world probe stores separate 6×6 octahedral Lambertian irradiance and support fields for background, emissive, and finite punctual sources. Resolve normalizes each field independently; Penumbra adds no glossy receiver transport. |
| `artistic_occlusion` | Enable the optional occlusion adjustment. |

Call `reset_history()` after discontinuous camera or lighting changes. Bevy
still owns scene transforms and asset lifetime. The low-end profile uses a
160×90 candidate grid, updates at most 1,024 receivers per frame, caps admitted
traversal queries at 16,384, retains at most 65,536 world-cache records, and
resolves at 640×360 for a 1280×720 view. Depth, receiver identity, normal, and
plane checks gate reconstruction back to the output viewport. Split combines
persistent world-space intervals with temporal reconstruction. Surfaces without
a compatible traced receiver can still have missing GI; reconstruction cannot
recover untraced lighting.

When no active camera has positive GI intensity, scene updates and receiver
identity extraction are suspended. Previously allocated resources can remain
resident. On re-enabling GI, referenced assets are revalidated and scene
preparation may take time before GI resumes.

When `multi_bounce` is enabled, prior primary ray hits seed a separate cache
whose probes are two LOD levels coarser. Its completed irradiance feeds later
primary and secondary hits. The configured receiver-update, traversal and cache
entry limits are shared between the caches, and the extra allocations count
toward the view budget. The secondary pass waits when the update budget is one,
or when a nonzero traversal-query or cache-entry limit is one. A zero query or
entry limit retains its existing unrestricted meaning. Secondary feedback uses
the nearest coarse probe and blends supported adjacent LODs across their
overlap. Secondary history is required for temporal multi-bounce feedback even
when primary screen history is disabled.

With `cached_hit_sky`, secondary rays that escape carry the environment, and
ray hits covered by the secondary cache take their sky light from it, as in the
paper, instead of tracing environment visibility rays. Uncovered hits keep the
visibility-tested sample. The cached sky is diffuse-only and one frame late; in
Sponza it lowered GI GPU time by about 1.3 ms (6.6%) with matched captures
within run-to-run variation (see validation notes).

With `hardware_ray_queries`, the primary closest-hit and visibility passes walk
a hardware TLAS/BLAS with wgpu ray queries instead of the software BVH. The
hardware only proposes candidate triangles. Every BLAS is non-opaque, and each
candidate is re-tested with the software triangle, culling, layer and
alpha-coverage tests. Accepted hits therefore keep the software distance and
barycentrics, and toggling the setting keeps all GI history. Hit shading,
query recording and replay, the overflow trace and the secondary cache stay on
the software traversal. This requires Vulkan with the SPIR-V passthrough path,
wgpu's experimental ray-query feature (Bevy requests adapter features by
default), and unpaged scene geometry. BLASes are built from the scene geometry
buffer when a view first requests hardware traversal; deformed meshes rebuild
theirs every frame. This acceleration-structure memory is outside the scene and
view budgets. While the structures or pipelines are not ready, the software
traversal runs. `RadianceCascadesStatus::hardware_ray_queries` reports which
traversal the latest dispatch used.

The Sponza demo and library use Split Radiance Cascades. In the interactive
<<<<<<< Updated upstream
Sponza demo, `F12` toggles cursor grab.
Sponza defaults to 8,192 receiver updates and 131,072 traversal queries per
frame, resolve scale 4, and two direct-light samples. The library's low-end
preset remains at 1,024 updates and 16,384 queries. The demo keeps its requested
profile fixed while running; `--max-rays`, `--max-traversal-queries`, and
`--resolution-divisor` adjust it; `--multi-bounce` enables the secondary cache.
=======
Sponza demo, `F12` toggles cursor grab and `H` toggles hardware ray queries,
which Sponza requests by default; the window title shows the traversal in use.
`--software-traversal` starts with the software traversal.
Sponza defaults to 2,048 receiver updates and 131,072 traversal queries per
frame, requested grid divisor 4, resolve scale 4, and two direct-light samples.
The library's low-end preset remains at 1,024 updates and 16,384 queries. The demo keeps its requested
profile fixed while running; `--max-rays`, `--max-traversal-queries`, and
`--resolution-divisor` adjust it; `--multi-bounce` enables the secondary cache,
and `--cached-hit-sky` adds cached hit sky lighting.
Interactive mode requests VSync by default; `--no-vsync` requests AutoNoVsync
instead and may produce tearing. The platform can choose a fallback present mode.
>>>>>>> Stashed changes
Split's Sponza profile uses 1 m probe spacing
and a 1.6 m first interval; it permits 32,768 receiver cells and 262,144 cached
intervals within a 1 GiB view budget, subject to device allocation limits. Run
`scripts/benchmark_sponza.ps1` for the release 1920×1080 windowed benchmark
with VSync-off requested. It records a GI-disabled baseline and requires 120
warmup plus 240 measured CPU/GPU samples. The offscreen benchmark remains
available for drained measurements; use `--benchmark-offscreen-no-gi` for an
offscreen GI-disabled baseline. Queue-envelope timings can include GPU idle
time and are not measurements of active GPU execution. Schedule CPU diagnostics
measure elapsed time, including waits, not active CPU time. The JSON report
includes `samples.trace_gpu_ms`, a distribution of Split trace pass GPU
timestamps observed within the measurement window; it is empty when GI is
disabled. Startup logs record the first GI dispatch and the first raw readback
reaching 95% eligible-pixel support; neither event waits for warmup.
`scripts/validate_sponza_gi.ps1 -Release`
checks Split with static captures, a moving blocker, and a moving point
light. Dark camera directions use matched GI-on/off controls instead of
requiring courtyard brightness in a shadowed arcade. These scripts enforce the
target budgets; their current results are not certified until the acceptance run passes.
See the [validation status and historical measurements](docs/validation.md).
With the low-end profile, four irradiance images and five temporal metadata
images use 640×360 storage for a 1280×720 viewport, about 31.6 MiB combined.
The sparse GI resolve, receiver identity, scene cache, and Bevy-owned rendering
resources are accounted separately. Split reconstructs the paper's diffuse
indirect term from persistent world-space intervals. Unsupported samples remain
distinct from observed black radiance.

Bevy's native environment maps continue to control native PBR lighting.
`RadianceCascadesEnvironment` supplies a separate input to Split's indirect
transport; assigning it does not automatically enable or replace Bevy's native
`EnvironmentMap`.

Emissive geometry contributes radiance through first-hit cascade rays. No
primary next-event estimator or emitter classification is required. Remove
the former `RadianceCascadesEmitterSampling` component from existing scenes.
The renderer still uses next-event estimation to shade secondary hit surfaces;
these reflected paths do not duplicate first-hit emission. Small emitters can
require more samples to converge.

See [rendering boundaries](docs/rendering-boundaries.md) for the native Bevy
visibility-buffer path, sparse-upload constraints, and the diffuse transport
and light-ownership boundary.

On native targets, Slang's generated WGSL is cached between runs under the
platform user cache directory (`%LOCALAPPDATA%\bevy_penumbra\shaders` on
Windows). Set `BEVY_PENUMBRA_SHADER_CACHE_DIR` to use another directory. The
cache key includes the embedded shader sources, Slang version, target, and O2
optimization level. Every backend starts with validated WGSL; Naga maps resource
bindings for the selected backend. Device pipeline caches are saved atomically when supported.
Sponza compiles the Split pipeline set on demand at startup.

## Examples

The Cornell box is a compact GI scene with raw capture and GPU timing support.
Sponza loads the included courtyard, curtains, and ivy assets. The lighting
ownership example checks that GI view state is restored when GI is disabled.
The capture profiles retain 64 history frames: Cornell blends at 0.97 and
Sponza at 0.985, averaging sparse samples without adding per-frame dispatches.

```sh
cargo run --example cornell_box
cargo run --example cornell_box -- --capture
cargo run --example cornell_box -- --capture --capture-output=tmp/cornell_after.png
cargo run --example cornell_box -- --capture-raw --raw-output=screenshots/cornell_src.pfm
cargo run --example cornell_box -- --live-capture --width=1280 --height=720 --output=captures/cornell_1280x720.png
cargo run --example cornell_box -- --live-capture --capture-frame-count=8 --output=captures/cornell_flicker.png
cargo run --example cornell_box -- --benchmark --width=1280 --height=720
cargo run --example sponza
cargo run --example sponza -- --capture
cargo run --example sponza -- --capture-motion
cargo run --example sponza -- --capture-stability
cargo run --example sponza -- --capture-motion
cargo run --example lighting_ownership
```

The Cornell window uses Split at a requested logical size of 960×720. It accepts
`--width=N --height=N` for other sizes. Interactive windows use the monitor's
DPI scale factor, so at 125% DPI a 960×720 request becomes a 1200×900 physical
target. `--live-capture` uses that same DPI scale, controls, and diagnostics
overlay; it automates saving the interactive window after warmup. The receiver
grid follows the physical
window size and raises the divisor as needed to stay within
`--max-receiver-cells=N`; renderer device and per-view memory limits can lower
the active grid further. The renderer reserves the full-resolution receiver
texture before choosing GI allocations and can raise the GI resolve divisor
through 8 when a large target would exceed the memory budget. Reconstruction
still produces the full-resolution image. Allocation adjustments are logged
with the viewport, effective grid, resolve divisor, and memory budget.
The F3 diagnostics
show the physical target, the grid allowed by the configured cell cap, the
renderer-reported active candidate count, and successful updates alongside the
scheduled update budget. The warmup estimate uses that configured budget, not
the phase-varying successful-update count. The interactive Cornell demo defaults
to 2,048 receiver updates per frame and an 8,192-cell receiver cap. The active
grid adapts its divisor
to the physical window size and may be lowered further by renderer memory
limits. `--max-rays=N` and `--max-receiver-cells=N` can lower those budgets for
slower GPUs. `--resolution-divisor=N`, `--max-receiver-cells=N`, `--max-rays=N`,
and `--resolve-scale=N` also apply to interactive runs. The window title includes
the initial scene and requested size; use `--window-title=NAME` to supply a
recording-friendly title.

`--live-capture` waits for the live window's GI warmup, captures its rendered
primary window, and exits after saving. It defaults to
`captures/cornell_live.png`; use `--output=PATH` to choose another file. Pair it
with `--capture-frame-count=N` to save a numbered PNG sequence, sampling once
per fresh GI render result while earlier PNG writes finish asynchronously. A raw
capture writes a little-endian RGB PFM file and a JSON sidecar with its dimensions
and settings. Benchmarks report completed GI GPU samples; they do not represent
total application frame time.

GI scene updates filter image and material asset events to the assets referenced
by visible GI geometry. Debug text and unrelated UI atlas updates preserve GI
history. Idle controls leave the ceiling emitter's transform untouched, and
identical geometry transforms do not trigger a rebuild. The warmup counter shows
rendered GI frames, rather than independent samples per receiver.
`RUST_LOG=bevy_penumbra::render=debug,bevy_penumbra::scene=debug` logs scene rebuild
and sparse-history reset causes.
`--capture-warmup-frames=N` can require a longer live observation before saving.

The Sponza example expects the supplied files under `assets/`. See
[the asset inventory](docs/sponza-assets.md) for scene contents. Its demo limits
material texture images to 512 pixels and uses `--max-rays=N` to adjust the
per-view work budget. To admit those assets and replacement overlap, the
benchmark allows a 2 GiB scene and a 768 MiB material atlas; this is separate
from the library's 4 GB low-end memory profile.

## Research and validation

The implementation is checked against
[Split Radiance Cascades v2, revised October 11, 2026](https://arxiv.org/abs/2607.20384v2).
Background references include the local
[Holographic Radiance Cascades paper](../GI%20Research/2505.02041v1.pdf) and
[Neural Radiance Caching paper](../GI%20Research/Neural%20Radiance%20Caching.pdf).
Holographic Radiance Cascades is a 2D reference here; its dense 3D form is not
added because its memory scales as N⁴. Online neural radiance caching is also
deferred because it needs a separate training and inference path. Related code
and shader references include [RadianceCascadesGIVulkan](https://github.com/tlegoc/RadianceCascadesGIVulkan),
[the 3D merge shader](https://www.shadertoy.com/view/X3XfRM),
[SRC-DGI](https://github.com/mxcop/src-dgi), and
[vlam](https://github.com/entropylost/vlam). The supplied [Chandra source](ref/chandra/)
and [Split Radiance Cascades paper](ref/Split%20Radiance%20Cascades.pdf) are the
local algorithm references.

[Holographic Radiance Cascades](https://arxiv.org/abs/2505.02041) describes a
2D interval-extension method. Its radiance/transmittance algebra overlaps the
existing Split cache, which already stores sparse 3D intervals. A distinct 3D
An HRC implementation would need its own sparse ray-extension algorithm; directly building
the paper's dense representation in 3D has N⁴ memory scaling, so this crate does
not implement it.

[Validation notes](docs/validation.md) explain how to run the current examples.
They do not carry forward measurements from earlier implementations. The
`scripts/check_quality.py` utility compares supplied image sequences and reports
regional spatial and temporal metrics; it does not generate images or establish
radiometric accuracy.

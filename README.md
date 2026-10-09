# bevy_penumbra

`bevy_penumbra` implements Split Radiance Cascades (SRC) for diffuse global
illumination in Bevy. The crate exposes one transport method: Split Radiance
Cascades. The algorithm follows the supplied paper and the local Chandra
implementation for its cascade construction and updates.

## Add SRC to a scene

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

SRC uses deferred opaque rendering so the renderer can read receiver material and
surface data. The examples show the full setup, controls, raw output, and scene
registration patterns.

## Tune a view

`RadianceCascades` contains the per-camera quality and appearance settings.
Typical controls are:

| Setting | Purpose |
| --- | --- |
| `probe_spacing`, `interval_length` | Set the world-space cascade spacing and interval scale. |
| `resolution_divisor`, `max_rays` | Set the screen grid and the work budget. |
| `normal_bias` | Offset surface samples to reduce self-intersection. |
| `intensity`, `environment` | Set GI strength and the diffuse environment contribution. |
| `history_frames`, `temporal_blend` | Control temporal reuse. |
| `spatial_filter_steps` | Control the spatial reconstruction filter. |
| `emissive_samples`, `bounce_feedback` | Tune emissive sampling and cascade feedback. |
| `artistic_occlusion` | Enable the optional occlusion adjustment. |

Call `reset_history()` after discontinuous camera or lighting changes that should
discard accumulated history. Bevy still owns scene transforms and asset lifetime.
Native Bevy environment cubemaps share an enable state for diffuse and specular
lighting; keeping a cubemap for reflections can also retain native diffuse
irradiance. Use `RadianceCascadesEnvironment` when you want to provide an SRC
diffuse environment source.

On native targets, Slang's generated WGSL is cached between runs under the
platform user cache directory (`%LOCALAPPDATA%\bevy_penumbra\shaders` on
Windows). Set `BEVY_PENUMBRA_SHADER_CACHE_DIR` to use another directory. The
cache key includes the embedded shader sources and Slang version; a cold Vulkan
driver pipeline build still occurs when a pipeline has not been compiled before.

## Examples

The Cornell box is a compact SRC scene with raw capture and GPU timing support.
Sponza loads the included courtyard, curtains, and ivy assets. The lighting
ownership example checks that SRC view state is restored when SRC is disabled.

```sh
cargo run --example cornell_box
cargo run --example cornell_box -- --capture
cargo run --example cornell_box -- --capture-raw --raw-output=screenshots/cornell_src.pfm
cargo run --example cornell_box -- --benchmark --width=1280 --height=720
cargo run --example sponza
cargo run --example sponza -- --capture
cargo run --example sponza -- --capture-motion
cargo run --example lighting_ownership
```

The Cornell box accepts `--resolution-divisor=N` and `--max-rays=N` to adjust the
SRC work grid and budget. A raw capture writes a little-endian RGB PFM file and a
JSON sidecar with its dimensions and settings. Benchmarks report completed SRC
GPU samples; they do not represent total application frame time.

The Sponza example expects the supplied files under `assets/`. See
[the asset inventory](docs/sponza-assets.md) for scene contents. Its demo limits
material texture images to 512 pixels and uses `--max-rays=N` to adjust the
per-view work budget.

## Research and validation

The local references used for this implementation are the
[Split Radiance Cascades paper](../GI%20Research/Split%20Radiance%20Cascades.pdf)
and [Chandra](../GI%20Research/chandra/). These research copies live beside the
crate in the supplied workspace.

[Validation notes](docs/validation.md) explain how to run the current examples.
They do not carry forward measurements from earlier implementations. The
`scripts/check_quality.py` utility compares supplied image sequences and reports
regional spatial and temporal metrics; it does not generate images or establish
radiometric accuracy.

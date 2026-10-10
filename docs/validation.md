# Validation

The numerical results previously listed in this document described an earlier
renderer, including its guided and reference modes. Those results are obsolete
and do not describe this crate's Split Radiance Cascades implementation. This
file now records how to exercise the current examples without carrying forward
those historical measurements.

## Run the current examples

```sh
cargo run --example cornell_box -- --capture
cargo run --example cornell_box -- --capture-raw --raw-output=screenshots/cornell_src.pfm
cargo run --example cornell_box -- --benchmark --width=1280 --height=720
cargo run --example sponza -- --capture
cargo run --example lighting_ownership
```

The Cornell capture checks for lit receiver pixels. Its raw capture also checks
that the SRC output contains finite values and valid receivers before writing a
PFM image and JSON sidecar. The benchmark reports completed SRC GPU timings only.
Each run depends on a working Slang compiler and a supported desktop GPU.

The seed pass resolves each visible receiver by tracing the pixel's depth ray
against the same GI scene used by the cascades. It carries the matched entity
and triangle identity through temporal filtering and composition, so the screen
receiver and cached radiance use the same geometry identity.

The native Vulkan SPIR-V path relocates logical shader bindings to the dense
physical bindings used by wgpu. In particular, receiver-history storage at
logical binding 23 becomes physical binding 20, while deferred material sampling
at logical binding 27 becomes physical binding 23. Omitting this relocation can
overwrite the deferred material buffer and produce sharp black regions, incorrect
colors and unstable lighting. WGSL uses Naga's automatic relocation. The shader
cache retains the original compiler output; relocation happens at pipeline creation.

`cargo test --lib render::native_compute_binding_relocation_respects_backend -- --exact`
checks these resource assignments and preserves the dense second descriptor set
and non-Vulkan shader output. Captures remain necessary to establish visual quality;
the lit-pixel and GI-toggle checks alone do not reject corrupted material buffers.

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
960×720 demo uses a divisor of six; raster geometry remains at full resolution.
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

Sponza also enables Bevy TAA with MSAA disabled. GI reconstruction applies the
same projection jitter and resolution override as Bevy's raster view. Demo GI/time
controls invalidate both temporal histories. The final noon/GI-off/
midnight captures completed with an on/off RGB difference of 17.29/255.
The final courtyard motion/reset capture completed with a settled/restarted
image difference of 5.00/255; TAA jitters raster samples between captures.
The courtyard captures exercise the main glTF scene; they do not establish visual
quality for every optional curtain/ivy asset or GPU backend.

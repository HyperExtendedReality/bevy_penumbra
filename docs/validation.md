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

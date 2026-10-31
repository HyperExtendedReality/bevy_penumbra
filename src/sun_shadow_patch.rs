//! Routes Bevy's directional shadow lookup to ray-traced sun visibility.
//!
//! Bevy 0.20 has no hook for an external shadow term, so two of its PBR shader
//! modules are patched in place once they load. Every material Bevy shades,
//! deferred or forward, then takes the traced visibility for a marked light.
//! The anchors are exact Bevy 0.20 text: when they are missing, nothing is
//! patched, a warning is logged and Bevy's shadow maps stay in use.

use bevy::{
    prelude::*,
    shader::{Shader, Source},
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

/// `shadow_depth_bias` of a directional light whose shadow Penumbra traces.
/// Its visibility is layer 0 of the directional shadow texture, in screen space.
pub(crate) const RAY_TRACED_SHADOW_BIAS: f32 = -1.0e30;
const SENTINEL: &str = "-1e30";
const MARKER: &str = "bevy_penumbra: ray-traced directional shadow";
const PATHS: [&str; 2] = [
    "embedded://bevy_pbr/render/pbr_functions.wesl",
    "embedded://bevy_pbr/render/shadows.wesl",
];

/// True once both Bevy modules carry the patch.
#[derive(Resource, Clone, Default, bevy::render::extract_resource::ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
pub(crate) struct SunShadowPatch(pub Arc<AtomicBool>);

impl SunShadowPatch {
    pub(crate) fn applied(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// Also enters the shadow branch for a marked light that has no shadow maps.
fn patch_functions(source: &str) -> Option<String> {
    let flags = "(view_bindings::lights.directional_lights[i].flags & \
                 mesh_view_types::DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u)";
    // The direct and diffuse-transmission lookups; fog in-scattering is left unshadowed.
    (source.matches(flags).count() == 2).then(|| {
        let marked = format!(
            "({} // {MARKER}\n                    || view_bindings::lights.directional_lights[i].shadow_depth_bias == {SENTINEL}))",
            &flags[..flags.len() - 1]
        );
        source.replace(flags, &marked)
    })
}

/// Returns the traced visibility at the shaded pixel for a marked light.
fn patch_shadows(source: &str) -> Option<String> {
    let anchor = "    let light = &view_bindings::lights.directional_lights[light_id];\n    \
                  let cascade_index = get_cascade_index(light_id, view_z);\n";
    (source.matches(anchor).count() == 1).then(|| {
        source.replacen(
            anchor,
            &format!(
                "    let light = &view_bindings::lights.directional_lights[light_id];
    // {MARKER}.
    @if(!NO_ARRAY_TEXTURES_SUPPORT) {{
    if ((*light).shadow_depth_bias == {SENTINEL}) {{
        return textureLoad(view_bindings::directional_shadow_textures,
            vec2<i32>(frag_coord_xy), 0, 0);
    }}
    }}
    let cascade_index = get_cascade_index(light_id, view_z);
"
            ),
            1,
        )
    })
}

/// Patches each module once it has loaded; a no-op after both are patched.
pub(crate) fn patch_bevy_shadow_shaders(
    server: Res<AssetServer>,
    mut shaders: ResMut<Assets<Shader>>,
    patch: Res<SunShadowPatch>,
    mut handles: Local<Option<[Handle<Shader>; 2]>>,
    mut done: Local<[bool; 2]>,
) {
    if done.iter().all(|done| *done) {
        return;
    }
    let handles = handles.get_or_insert_with(|| PATHS.map(|path| server.load(path)));
    for (index, handle) in handles.iter().enumerate() {
        if done[index] {
            continue;
        }
        let Some(Source::Wesl(source)) = shaders.get(handle).map(|shader| &shader.source) else {
            continue;
        };
        if source.contains(MARKER) {
            done[index] = true;
            continue;
        }
        let patched = if index == 0 {
            patch_functions(source)
        } else {
            patch_shadows(source)
        };
        let Some(patched) = patched else {
            warn!(
                path = PATHS[index],
                "Bevy shader text changed; ray-traced sun shadows stay on Bevy's shadow maps"
            );
            *done = [true; 2];
            return;
        };
        // Replacing the source makes Bevy recompile every pipeline importing it.
        shaders.get_mut(handle).expect("loaded shader").source = Source::Wesl(patched.into());
        done[index] = true;
    }
    let patched = handles.iter().all(|handle| {
        shaders.get(handle).is_some_and(
            |shader| matches!(&shader.source, Source::Wesl(source) if source.contains(MARKER)),
        )
    });
    patch.0.store(patched, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The patched Bevy release's sources, from Cargo's registry checkout.
    fn bevy_source(name: &str) -> String {
        let home = std::env::var_os("CARGO_HOME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(
                    std::env::var_os("USERPROFILE")
                        .or_else(|| std::env::var_os("HOME"))
                        .unwrap(),
                )
                .join(".cargo")
            });
        let crate_dir = std::fs::read_dir(home.join("registry/src"))
            .unwrap()
            .flatten()
            .map(|index| index.path().join("bevy_pbr-0.20.0"))
            .find(|path| path.is_dir())
            .expect("bevy_pbr 0.20.0 sources in the Cargo registry");
        std::fs::read_to_string(crate_dir.join("src/render").join(name)).unwrap()
    }

    #[test]
    fn bevy_pbr_shaders_accept_the_patch() {
        let functions = patch_functions(&bevy_source("pbr_functions.wesl")).expect("anchors");
        assert_eq!(functions.matches(MARKER).count(), 2);
        assert_eq!(functions.matches("shadow_depth_bias == -1e30").count(), 2);
        let shadows = patch_shadows(&bevy_source("shadows.wesl")).expect("anchor");
        assert_eq!(shadows.matches(MARKER).count(), 1);
        // A changed Bevy source is rejected rather than half patched.
        assert!(patch_shadows("fn other() {}").is_none());
        assert!(patch_functions("fn other() {}").is_none());
        assert_eq!(RAY_TRACED_SHADOW_BIAS, -1e30);
    }
}

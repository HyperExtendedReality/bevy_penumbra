//! World-space diffuse GI based on Freeman and Sannikov's Split Radiance Cascades.
//!
//! Add [`RadianceCascadesPlugin`] after Bevy's render plugins, attach
//! [`RadianceCascades`] to a 3D camera, and tag opaque `Mesh3d` /
//! `MeshMaterial3d<StandardMaterial>` entities with [`RadianceCascadesGeometry`].
//! Shaders are embedded Slang sources compiled once using `bevy_slang`.
//! See the crate README for requirements, costs, and research deviations.

#![recursion_limit = "256"]

mod render;
mod scene;

use bevy::{
    camera::{Hdr, visibility::VisibilitySystems},
    core_pipeline::prepass::{DeferredPrepass, DepthPrepass, NormalPrepass},
    pbr::DefaultOpaqueRendererMethod,
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        extract_resource::ExtractResourcePlugin,
    },
    transform::TransformSystems,
};
use bevy_slang::SlangCompiler;

/// Opts an opaque, CPU-readable triangle mesh into world-space GI ray tracing.
///
/// Offscreen meshes still occlude and emit. Base color, metallic, and emission
/// come from `StandardMaterial`; texture sampling and vertex deformation are
/// not currently represented in the software ray tracing scene.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct RadianceCascadesGeometry;

/// Per-camera quality and lighting controls. Distances are in world units.
///
/// Requires an HDR, single-sample deferred 3D camera. Required components are
/// inserted automatically; do not override `Msaa::Off` on this camera.
#[derive(Component, Clone, Copy, Debug, PartialEq, ExtractComponent)]
#[require(Camera3d, Hdr, DepthPrepass, NormalPrepass, DeferredPrepass, Msaa::Off)]
pub struct RadianceCascades {
    /// Spacing of cascade zero. Higher cascades double it (paper Eq. 4).
    pub probe_spacing: f32,
    /// Cascade-zero cutoff. Higher cascades quadruple it; four cascades total.
    pub interval_length: f32,
    /// One GI ray per this many pixels along each image axis (minimum 1).
    pub resolution_divisor: u32,
    /// Maximum rays per camera per frame; resolution is reduced to respect it.
    /// Clamped to 1..=131072. The device's storage limit may reduce this budget.
    pub max_rays: u32,
    /// Offset of rays along the surface normal to prevent self intersection.
    pub normal_bias: f32,
    /// Strength of the added diffuse lighting; zero bypasses the effect.
    pub intensity: f32,
    /// Uniform environment radiance sampled by rays escaping the scene.
    pub environment: LinearRgba,
    /// Blend for world-space directional bins and reprojected irradiance, 0..0.99.
    /// Scene edits invalidate world bins; screen history rejects geometry
    /// mismatches and clamps changed lighting to its current neighborhood.
    pub temporal_blend: f32,
    /// Frames to retain unsampled world-space directional entries, 0..64.
    /// Zero disables retention; scene edits always invalidate the cache.
    pub history_frames: u32,
    /// Edge-aware spatial denoising steps, 0..4. Zero disables the filter.
    pub spatial_filter_steps: u32,
    /// Combined emissive-triangle and cosine-direction samples per receiver, 1..32.
    /// Ray hits use half this budget. More samples improve soft shadow quality.
    pub emissive_samples: u32,
    /// Previous-frame diffuse feedback at hits, 0..0.95. Zero disables feedback;
    /// explicit emitter sampling still supplies direct and reflected lighting.
    /// Enables one extra ray per primary hit to fill a secondary world-space
    /// cache at two higher probe LODs, including offscreen hit positions.
    /// Explicit emitter sampling supplies a fresh reflected bounce during
    /// motion; further bounces accumulate while the scene is unchanged.
    pub bounce_feedback: f32,
}

impl Default for RadianceCascades {
    fn default() -> Self {
        Self {
            probe_spacing: 0.25,
            interval_length: 0.4,
            resolution_divisor: 2,
            max_rays: 131_072,
            normal_bias: 0.005,
            intensity: 1.0,
            environment: LinearRgba::BLACK,
            temporal_blend: 0.92,
            history_frames: 32,
            spatial_filter_steps: 3,
            emissive_samples: 16,
            bounce_feedback: 0.8,
        }
    }
}

impl RadianceCascades {
    pub(crate) fn sanitized(mut self) -> Self {
        fn finite(value: f32, fallback: f32, min: f32, max: f32) -> f32 {
            if value.is_finite() {
                value.clamp(min, max)
            } else {
                fallback
            }
        }
        self.probe_spacing = finite(self.probe_spacing, 0.25, 0.001, 10_000.0);
        self.interval_length = finite(self.interval_length, 0.4, 0.001, 100_000.0);
        self.normal_bias = finite(self.normal_bias, 0.005, 0.00001, self.probe_spacing);
        self.intensity = finite(self.intensity, 1.0, 0.0, 100.0);
        self.temporal_blend = finite(self.temporal_blend, 0.92, 0.0, 0.99);
        self.history_frames = self.history_frames.min(64);
        self.spatial_filter_steps = self.spatial_filter_steps.min(4);
        self.emissive_samples = self.emissive_samples.clamp(1, 32);
        self.bounce_feedback = finite(self.bounce_feedback, 0.8, 0.0, 0.95);
        self.resolution_divisor = self.resolution_divisor.clamp(1, 64);
        self.max_rays = self.max_rays.clamp(1, 131_072);
        self.environment = LinearRgba::new(
            finite(self.environment.red, 0.0, 0.0, 100_000.0),
            finite(self.environment.green, 0.0, 0.0, 100_000.0),
            finite(self.environment.blue, 0.0, 0.0, 100_000.0),
            1.0,
        );
        self
    }
}

/// Installs extraction, a CPU triangle BVH, and Slang GPU passes into `Core3d`.
///
/// The default opaque renderer is set to deferred to provide receiver albedo.
/// Explicitly forward materials remain forward and do not receive GI.
#[derive(Default)]
pub struct RadianceCascadesPlugin {
    /// Slang compiler configuration (`SLANGC` or `slangc` on PATH by default).
    pub compiler: SlangCompiler,
}

impl Plugin for RadianceCascadesPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(DefaultOpaqueRendererMethod::deferred())
            .init_resource::<scene::GiScene>()
            .add_plugins((
                ExtractComponentPlugin::<RadianceCascades>::default(),
                ExtractResourcePlugin::<scene::GiScene>::default(),
            ))
            .add_systems(
                PostUpdate,
                scene::update_scene
                    .after(TransformSystems::Propagate)
                    .after(VisibilitySystems::VisibilityPropagate),
            );
        render::install(app, self.compiler.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_settings_are_bounded() {
        let settings = RadianceCascades {
            probe_spacing: f32::NAN,
            resolution_divisor: 0,
            max_rays: u32::MAX,
            bounce_feedback: 1.0,
            history_frames: u32::MAX,
            spatial_filter_steps: u32::MAX,
            emissive_samples: u32::MAX,
            ..default()
        }
        .sanitized();
        assert_eq!(settings.probe_spacing, 0.25);
        assert_eq!(settings.resolution_divisor, 1);
        assert_eq!(settings.max_rays, 131_072);
        assert_eq!(settings.bounce_feedback, 0.95);
        assert_eq!(settings.history_frames, 64);
        assert_eq!(settings.spatial_filter_steps, 4);
        assert_eq!(settings.emissive_samples, 32);
    }
}

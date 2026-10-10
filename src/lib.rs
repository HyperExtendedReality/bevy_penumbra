//! World-space global illumination using Split Radiance Cascades.
//!
//! Add [`RadianceCascadesPlugin`] after Bevy's render plugins, attach
//! [`RadianceCascades`] to a 3D camera, and tag opaque/alpha-masked `Mesh3d` /
//! `MeshMaterial3d<StandardMaterial>` entities with [`RadianceCascadesGeometry`].
//! Shaders are embedded Slang sources compiled once using `bevy_slang`.
//! See the crate README for requirements, costs, and research deviations.

#![recursion_limit = "256"]

mod deformation;
mod environment;
mod frame_timing;
pub use frame_timing::FrameTimingPlugin;
mod material;
pub use environment::RadianceCascadesEnvironment;
mod receiver;
mod render;
mod scene;

use bevy::{
    camera::{Hdr, visibility::VisibilitySystems},
    core_pipeline::prepass::{DeferredPrepass, DepthPrepass, MotionVectorPrepass, NormalPrepass},
    prelude::*,
    render::{
        RenderApp,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        extract_resource::ExtractResourcePlugin,
        sync_world::SyncToRenderWorld,
    },
    transform::TransformSystems,
};
use bevy_slang::SlangCompiler;

/// Supplies opaque or alpha-masked triangle geometry to Split Radiance Cascades.
///
/// Offscreen meshes still occlude and emit. Base color, metallic, and emission
/// come from `StandardMaterial`, vertex colors and CPU-retained material images.
/// Indexed normals, tangents, and UV0/UV1 are retained for cascade shading.
/// Alpha-masked surfaces preserve their material coverage.
/// Morphing and skinning are supported.
/// Split Radiance Cascades approximates GI with diffuse response and ignores
/// clearcoat, anisotropy, and specular lobes and maps; Bevy keeps native specular shading.
/// Requires render-world synchronization for the raster receiver identity pass.
#[derive(Component, Clone, Copy, Debug, Default)]
#[require(SyncToRenderWorld)]
pub struct RadianceCascadesGeometry;

/// Optional destination for the raw linear GI estimate, before reconstruction.
///
/// Supply an `Rgba32Float` image with `COPY_DST | COPY_SRC` usage and exactly the
/// resolved GI dimensions (viewport divided by `resolution_divisor`, adjusted
/// as needed for the cascade budget and device limits). RGB contains linear GI
/// before receiver albedo, intensity, occlusion, and exposure; alpha is receiver
/// depth. Read it asynchronously using Bevy's `Readback::texture`. No readback
/// occurs unless the application requests one. Invalid destinations are diagnosed.
#[derive(Component, Clone, Debug, ExtractComponent)]
#[extract_app(RenderApp)]
pub struct RadianceCascadesRawOutput(pub Handle<Image>);

/// Per-camera Split Radiance Cascades quality and lighting controls. Distances
/// are in world units.
///
/// Requires an HDR, single-sample deferred 3D camera. Required components are
/// inserted automatically; do not override `Msaa::Off` on this camera.
#[derive(Component, Clone, Copy, Debug, PartialEq, ExtractComponent)]
#[extract_app(RenderApp)]
#[require(
    Camera3d,
    Hdr,
    DepthPrepass,
    NormalPrepass,
    MotionVectorPrepass,
    DeferredPrepass,
    Msaa::Off
)]
pub struct RadianceCascades {
    /// Per-view invalidation epoch. Call `reset_history()` after a game-driven
    /// origin rebase or camera teleport. The game retains ownership of its origin.
    pub history_epoch: u32,
    /// Spacing of cascade zero. Higher cascades double it (paper Eq. 4).
    pub probe_spacing: f32,
    /// Cascade-zero cutoff. Higher cascades quadruple it; four cascades total.
    pub interval_length: f32,
    /// One cascade sample per this many pixels along each image axis (minimum 1).
    pub resolution_divisor: u32,
    /// Maximum cascade rays per camera per frame, clamped to 1..=131072.
    /// Resolution, device limits, and per-view memory budgets may reduce the work.
    pub max_rays: u32,
    /// Offset of rays along the surface normal to prevent self intersection.
    pub normal_bias: f32,
    /// Strength of the added GI; zero bypasses the effect.
    pub intensity: f32,
    /// Apply Bevy's material and screen-space occlusion to composed GI.
    /// Raw output bypasses these artistic factors.
    pub artistic_occlusion: bool,
    /// Uniform environment radiance for cascade directions that exit the scene.
    pub environment: LinearRgba,
    /// Per-frame blend with the previous cascade result, from 0 to 0.99.
    pub temporal_blend: f32,
    /// Number of frames to retain cascade history, clamped to 0..=64.
    pub history_frames: u32,
    /// Edge-aware spatial denoising steps, 0..4. Zero disables the filter.
    pub spatial_filter_steps: u32,
    /// Samples used to evaluate emissive surfaces per receiver, clamped to 1..=32.
    pub emissive_samples: u32,
    /// Previous-frame diffuse feedback strength, clamped to 0..=0.95.
    pub bounce_feedback: f32,
}

impl Default for RadianceCascades {
    fn default() -> Self {
        Self {
            history_epoch: 0,
            probe_spacing: 0.25,
            interval_length: 0.4,
            resolution_divisor: 2,
            max_rays: 131_072,
            normal_bias: 0.005,
            intensity: 1.0,
            artistic_occlusion: true,
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
    /// Invalidates this view's probe and screen history on the next rendered frame.
    /// Scene transforms and asset residency continue to come from the game.
    pub fn reset_history(&mut self) {
        self.history_epoch = self.history_epoch.wrapping_add(1);
    }

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

/// Installs scene extraction and Slang GPU passes into `Core3d`.
///
/// The application must select deferred opaque rendering to provide receiver
/// albedo. Explicitly forward materials remain forward and do not receive GI.
pub struct RadianceCascadesPlugin {
    /// Slang compiler configuration (`SLANGC` or `slangc` on PATH by default).
    pub compiler: SlangCompiler,
    /// Maximum bytes of RGBA32F material atlas pages. Default: 256 MiB.
    /// Residency failure is diagnosed; source textures are never downsampled
    /// or silently discarded to fit this budget.
    pub texture_atlas_budget_bytes: usize,
    /// Maximum live shared GPU scene bytes, including replacement overlap.
    /// Default: 256 MiB. Excludes the separately budgeted atlas, per-camera
    /// transport/reconstruction, and Bevy-owned resources.
    pub scene_gpu_budget_bytes: u64,
    /// Aggregate camera transport/reconstruction allocations, including resize
    /// overlap. HDR/light buffers are checked separately when packed. Default:
    /// 2 GiB, to admit a typical 1080p cascade allocation.
    pub view_gpu_budget_bytes: u64,
    /// Scene and atlas transfer bytes per render frame. Default: 2 MiB; minimum
    /// one 16-byte word.
    pub geometry_upload_bytes_per_frame: usize,
    /// Maximum number of GI compute pipelines to compile concurrently. Default: 2.
    /// Higher values can increase driver memory use during pipeline creation.
    pub max_concurrent_pipeline_compiles: usize,
}

impl Default for RadianceCascadesPlugin {
    fn default() -> Self {
        Self {
            compiler: SlangCompiler::default(),
            texture_atlas_budget_bytes: 256 * 1024 * 1024,
            scene_gpu_budget_bytes: 256 * 1024 * 1024,
            view_gpu_budget_bytes: 2 * 1024 * 1024 * 1024,
            geometry_upload_bytes_per_frame: 2 * 1024 * 1024,
            max_concurrent_pipeline_compiles: 2,
        }
    }
}

impl Plugin for RadianceCascadesPlugin {
    fn build(&self, app: &mut App) {
        let sampler = app
            .get_added_plugins::<bevy::image::ImagePlugin>()
            .first()
            .map_or_else(bevy::image::ImageSamplerDescriptor::linear, |plugin| {
                plugin.default_sampler.clone()
            });
        app.insert_resource(material::TextureOptions {
            sampler,
            budget: self.texture_atlas_budget_bytes,
        });
        app.init_resource::<scene::GiScene>()
            .insert_resource(render::SceneBudget {
                live_bytes: self.scene_gpu_budget_bytes,
                view_bytes: self.view_gpu_budget_bytes,
                upload_bytes: self.geometry_upload_bytes_per_frame.max(16),
            })
            .init_resource::<environment::EnvironmentTables>()
            .add_plugins((
                ExtractComponentPlugin::<RadianceCascades>::default(),
                ExtractComponentPlugin::<RadianceCascadesRawOutput>::default(),
                ExtractComponentPlugin::<RadianceCascadesEnvironment>::default(),
                ExtractResourcePlugin::<scene::GiScene>::default(),
                ExtractResourcePlugin::<render::SceneBudget>::default(),
                ExtractResourcePlugin::<environment::EnvironmentTables>::default(),
            ))
            .add_systems(
                PostUpdate,
                scene::update_scene
                    .after(bevy::asset::AssetEventSystems)
                    .after(TransformSystems::Propagate)
                    .after(VisibilitySystems::VisibilityPropagate),
            );
        app.add_systems(
            PostUpdate,
            environment::update_environment_tables.after(bevy::asset::AssetEventSystems),
        );
        render::install(
            app,
            self.compiler.clone(),
            self.max_concurrent_pipeline_compiles,
        );
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
            temporal_blend: f32::INFINITY,
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
        assert_eq!(settings.temporal_blend, 0.92);
        assert_eq!(settings.bounce_feedback, 0.95);
        assert_eq!(settings.history_frames, 64);
        assert_eq!(settings.spatial_filter_steps, 4);
        assert_eq!(settings.emissive_samples, 32);
    }

    #[test]
    fn split_radiance_cascades_defaults_are_active_and_preserved() {
        let settings = RadianceCascades { ..default() }.sanitized();
        assert_eq!(settings.temporal_blend, 0.92);
        assert_eq!(settings.history_frames, 32);
        assert_eq!(settings.spatial_filter_steps, 3);
        assert_eq!(settings.bounce_feedback, 0.8);
        assert!(settings.artistic_occlusion);
    }

    #[test]
    fn plugin_preserves_application_opaque_renderer_choice() {
        use bevy::pbr::DefaultOpaqueRendererMethod;
        let mut app = App::new();
        app.insert_resource(DefaultOpaqueRendererMethod::forward());
        app.add_plugins(RadianceCascadesPlugin::default());
        assert_eq!(
            format!(
                "{:?}",
                app.world().resource::<DefaultOpaqueRendererMethod>()
            ),
            format!("{:?}", DefaultOpaqueRendererMethod::forward())
        );
    }
}

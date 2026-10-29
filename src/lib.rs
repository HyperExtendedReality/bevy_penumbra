//! Split radiance cascades global illumination.
//!
//! Add [`RadianceCascadesPlugin`] after Bevy's render plugins, attach
//! [`RadianceCascades`] to a 3D camera, and tag participating `Mesh3d` /
//! `MeshMaterial3d<StandardMaterial>` entities with [`RadianceCascadesGeometry`].
//! Shaders are embedded Slang sources compiled once using `bevy_slang`.
//! See the crate README for requirements, costs, and research deviations.

#![recursion_limit = "256"]

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::Instant,
};

mod deformation;
mod environment;
mod frame_timing;
pub use frame_timing::FrameTimingPlugin;
mod material;
pub use environment::RadianceCascadesEnvironment;
mod receiver;
mod render;
mod scene;

/// Shader/pipeline failures and scene revisions submitted by the GI renderer.
#[derive(Resource, Clone, Default, bevy::render::extract_resource::ExtractResource)]
#[extract_app(RenderApp)]
pub struct RadianceCascadesStatus {
    state: Arc<Mutex<RadianceCascadesStatusState>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
/// Scene and lighting snapshots accepted by the renderer for an eligible view.
pub struct RadianceCascadesRevisions {
    /// Scene snapshot revision.
    pub scene: u64,
    /// Geometry snapshot revision.
    pub geometry: u64,
    /// Punctual and emissive-light revision.
    pub lighting: u64,
}

/// Metadata for the latest GI frame encoded by the renderer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RadianceCascadesDispatch {
    /// Monotonic within this renderer instance.
    pub sequence: u64,
    /// Scene and lighting revisions used by the encoded frame.
    pub revisions: RadianceCascadesRevisions,
}

/// Metadata for one raw GI texture copy, captured when the renderer encoded it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RadianceCascadesRawFrame {
    /// Per-view frame counter for the copied GI result.
    pub frame_id: u64,
    /// Dispatch metadata for the copied GI result.
    pub dispatch: RadianceCascadesDispatch,
    /// Time the matching GI commands were encoded for submission.
    pub submitted_at: Instant,
}

/// Traversal used by the latest GI dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HardwareRayQueries {
    /// The device lacks ray queries on the Vulkan SPIR-V path.
    Unsupported,
    /// Supported, but the software traversal ran: not requested, or the
    /// hardware structures or pipelines were not ready.
    Inactive,
    /// Hardware ray queries traced the closest-hit and visibility passes.
    Active,
}

#[derive(Default)]
struct RadianceCascadesStatusState {
    failures: Vec<String>,
    submitted: RadianceCascadesRevisions,
    dispatch: Option<RadianceCascadesDispatch>,
    raw_frames: VecDeque<Option<RadianceCascadesRawFrame>>,
    next_dispatch_sequence: u64,
    shader_compile_jobs: usize,
    pipeline_compile_jobs: usize,
    shader_backend: u8,
    hardware_ray_queries_supported: bool,
    hardware_ray_queries_active: bool,
}

impl RadianceCascadesStatus {
    /// Returns shader or pipeline errors from the renderer, if any.
    pub fn failures(&self) -> Vec<String> {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .failures
            .clone()
    }

    /// Revisions for the latest scene and lights uploaded for an eligible view.
    pub fn submitted_revisions(&self) -> RadianceCascadesRevisions {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .submitted
    }

    /// Metadata for the most recent GI frame whose commands were encoded.
    pub fn last_dispatch(&self) -> Option<RadianceCascadesDispatch> {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .dispatch
    }

    /// Removes metadata for the oldest completed raw-output readback.
    ///
    /// Call once for every `ReadbackComplete` event from the matching raw
    /// output texture. The outer `None` means no frame metadata is queued;
    /// `Some(None)` means that frame had no committed dispatch.
    pub fn take_raw_frame(&self) -> Option<Option<RadianceCascadesRawFrame>> {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .raw_frames
            .pop_front()
    }

    pub(crate) fn record_raw_frame(&self, frame: Option<RadianceCascadesRawFrame>) {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .raw_frames
            .push_back(frame);
    }

    /// Whether Slang or driver pipeline work is still running.
    pub fn compilation_in_flight(&self) -> bool {
        let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.shader_compile_jobs != 0 || state.pipeline_compile_jobs != 0
    }

    /// Shader format selected from the active device backend and features.
    pub fn shader_backend(&self) -> Option<&'static str> {
        match self
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .shader_backend
        {
            1 => Some("slang_wgsl"),
            2 => Some("slang_wgsl_naga_spirv"),
            _ => None,
        }
    }

    /// Which traversal the latest GI dispatch used.
    pub fn hardware_ray_queries(&self) -> HardwareRayQueries {
        let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if state.hardware_ray_queries_active {
            HardwareRayQueries::Active
        } else if state.hardware_ray_queries_supported {
            HardwareRayQueries::Inactive
        } else {
            HardwareRayQueries::Unsupported
        }
    }

    pub(crate) fn set_hardware_ray_queries_supported(&self, supported: bool) {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .hardware_ray_queries_supported = supported;
    }

    pub(crate) fn set_hardware_ray_queries_active(&self, active: bool) {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .hardware_ray_queries_active = active;
    }

    pub(crate) fn record_failure(&self, failure: String) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if !state.failures.contains(&failure) {
            state.failures.push(failure);
        }
    }

    pub(crate) fn record_submitted_revisions(&self, submitted: RadianceCascadesRevisions) {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .submitted = submitted;
    }

    pub(crate) fn record_dispatch(
        &self,
        revisions: RadianceCascadesRevisions,
    ) -> RadianceCascadesDispatch {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.next_dispatch_sequence = state.next_dispatch_sequence.wrapping_add(1).max(1);
        state.submitted = revisions;
        let dispatch = RadianceCascadesDispatch {
            sequence: state.next_dispatch_sequence,
            revisions,
        };
        state.dispatch = Some(dispatch);
        dispatch
    }

    pub(crate) fn set_shader_compile_jobs(&self, count: usize) {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .shader_compile_jobs = count;
    }

    pub(crate) fn set_pipeline_compile_jobs(&self, count: usize) {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .pipeline_compile_jobs = count;
    }

    pub(crate) fn set_shader_backend(&self, backend: u8) {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .shader_backend = backend;
    }
}

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

/// Supplies triangle geometry to radiance cascades GI.
///
/// Offscreen meshes still occlude and emit. Base color, metallic, and emission
/// come from `StandardMaterial`, vertex colors and CPU-retained material images.
/// Indexed normals, tangents, and UV0/UV1 are retained for cascade shading.
/// Opaque and alpha-masked surfaces can receive deferred GI. Blended and
/// premultiplied surfaces use stochastic texture-alpha coverage for transport
/// visibility and emission, but do not receive the deferred GI composite.
/// Alpha coverage is not refractive or volumetric transmission.
/// Morphing and skinning are supported.
/// Split computes diffuse indirect illumination from transported radiance and
/// applies receiver albedo from Bevy's deferred material data. Native Bevy
/// lighting retains direct and specular illumination. The GI estimate does not
/// model clearcoat, anisotropy, transmission, or specular tint maps.
/// Requires render-world synchronization for the raster receiver identity pass.
#[derive(Component, Clone, Copy, Debug, Default)]
#[require(SyncToRenderWorld)]
pub struct RadianceCascadesGeometry;

/// Optional per-view cap for adaptive runtime transport scheduling.
///
/// This only limits work issued in a frame; it does not change cache capacity,
/// allocation, or history. A zero value still permits one update.
#[derive(Component, Clone, Copy, Debug, ExtractComponent)]
#[extract_app(RenderApp)]
pub struct RadianceCascadesRuntimeBudget {
    /// Maximum selected receiver updates this view may issue in one frame.
    pub max_updates_per_frame: u32,
}

/// Optional destination for the linear diffuse GI estimate before composition.
/// Raw output excludes receiver albedo and artistic occlusion.
///
/// Supply an `Rgba32Float` image with `COPY_DST | COPY_SRC` usage and the active
/// resolve dimensions: full viewport by default and half viewport (rounded up)
/// with the low-end preset. RGB contains scene-linear diffuse transport divided
/// by pi, before receiver albedo, intensity, occlusion, and camera exposure.
/// No exposure conversion is required for scene-linear reference comparisons.
/// Alpha is zero for background, negative for an
/// eligible receiver without lighting support, and `1 + 0.25 * coverage` for
/// supported diffuse GI (including observed black). Coverage is in `[0, 1]`.
/// Read it asynchronously using Bevy's `Readback::texture`. No readback occurs
/// unless the application requests one. Invalid destinations are diagnosed.
#[derive(Component, Clone, Debug, ExtractComponent)]
#[extract_app(RenderApp)]
pub struct RadianceCascadesRawOutput(pub Handle<Image>);

/// Shared receiver sampling, update, cache, traversal, and reconstruction budgets.
///
/// Every view uses one amortized scheme: a sparse candidate grid, a bounded
/// number of receiver updates per frame rotated fairly across the grid, and
/// a persistent world cache. Zero fields inherit `resolution_divisor` and
/// `max_rays`; `max_rays` also caps the per-frame selected updates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SamplingSettings {
    /// Candidate-grid pixel divisor. Zero inherits `RadianceCascades::resolution_divisor`.
    pub grid_divisor: u32,
    /// Maximum visible screen-grid candidates. Zero uses the hard safety cap
    /// of 131,072 candidates.
    pub max_receiver_cells: u32,
    /// Receiver updates selected per frame. Zero inherits `max_rays`.
    pub max_updates_per_frame: u32,
    /// Maximum retained world-cache entries, capped at 262,144. Zero derives
    /// capacity from the candidate grid.
    pub max_cache_entries: u32,
    /// Hard traversal-query cap per frame across receiver seeding, transport,
    /// and visibility queries. Zero disables the cap.
    pub max_traversal_queries: u32,
    /// Resolve scale: 1 is full resolution; values 2 through 8 divide the
    /// viewport dimensions by that factor, rounding up.
    pub resolve_scale: u32,
}

impl SamplingSettings {
    /// No workload limits beyond `resolution_divisor` and `max_rays`.
    pub const fn unlimited() -> Self {
        Self {
            grid_divisor: 0,
            max_receiver_cells: 0,
            max_updates_per_frame: 0,
            max_cache_entries: 0,
            max_traversal_queries: 0,
            resolve_scale: 1,
        }
    }

    /// Defaults aimed at a 4 GB GTX 1650 / RX 5500 XT at 1280×720.
    pub const fn low_end() -> Self {
        Self {
            grid_divisor: 8,
            max_receiver_cells: 16_384,
            max_updates_per_frame: 1_024,
            max_cache_entries: 16_384,
            max_traversal_queries: 16_384,
            resolve_scale: 2,
        }
    }

    fn sanitized(mut self) -> Self {
        if self.grid_divisor != 0 {
            self.grid_divisor = self.grid_divisor.clamp(1, 256);
        }
        if self.max_receiver_cells != 0 {
            self.max_receiver_cells = self.max_receiver_cells.clamp(1, 131_072);
        }
        if self.max_updates_per_frame != 0 {
            self.max_updates_per_frame = self.max_updates_per_frame.clamp(1, 131_072);
        }
        if self.max_cache_entries != 0 {
            self.max_cache_entries = self.max_cache_entries.clamp(1, 262_144);
        }
        if self.max_traversal_queries != 0 {
            self.max_traversal_queries = self.max_traversal_queries.clamp(1, 131_072);
        }
        self.resolve_scale = self.resolve_scale.clamp(1, 8);
        self
    }
}

impl Default for SamplingSettings {
    fn default() -> Self {
        Self::unlimited()
    }
}

/// Per-camera radiance-cascades quality and lighting controls. Distances are
/// in world units.
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
    /// Shared receiver-grid, update, traversal, cache, and reconstruction limits.
    pub sampling: SamplingSettings,
    /// Per-view invalidation epoch. Call `reset_history()` after a game-driven
    /// origin rebase or camera teleport. The game retains ownership of its origin.
    pub history_epoch: u32,
    /// Spacing of cascade zero. Higher cascades double it (paper Eq. 4).
    pub probe_spacing: f32,
    /// Cascade-zero cutoff in world units. About 1.6× `probe_spacing` follows
    /// the paper's recommended base interval. Higher cascades quadruple it;
    /// four cascades are evaluated in total.
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
    /// Enable the paper's coarser, off-screen secondary cache for diffuse multi-bounce lighting.
    /// This requires additional GPU memory and tracing work.
    pub multi_bounce: bool,
    /// Paper-style hit lighting: with `multi_bounce`, ray-hit surfaces take sky
    /// light from the secondary cache, whose escaped rays carry the environment,
    /// instead of tracing environment visibility rays. Hits without cache
    /// coverage still use visibility-tested sampling. The cache's sky is
    /// diffuse only and one frame late; ignored without `multi_bounce`.
    pub cached_hit_sky: bool,
    /// Traverse the scene with hardware ray queries instead of the software
    /// BVH, for the primary closest-hit and visibility passes. Hits keep the
    /// software triangle, culling, layer and alpha-coverage tests. Requires
    /// Vulkan with wgpu's experimental ray-query feature and unpaged scene
    /// geometry. Until the hardware structures are ready, or without that
    /// support, the software traversal runs instead. See
    /// [`RadianceCascadesStatus::hardware_ray_queries`].
    pub hardware_ray_queries: bool,
    /// Apply Bevy's material and screen-space occlusion to composed GI.
    /// Raw output bypasses these artistic factors.
    pub artistic_occlusion: bool,
    /// Constant source radiance for visibility-tested diffuse indirect estimates
    /// at Split ray-hit surfaces. Bevy PBR shades visible surfaces with its own
    /// ambient and environment-map lighting.
    pub environment: LinearRgba,
    /// Blend with validated previous irradiance.
    /// Split also uses this as per-observation radiance interval retention;
    /// zero disables screen history.
    pub temporal_blend: f32,
    /// Split-only per-observation retention for cached interval transmittance. Higher
    /// values preserve visibility while sparse probes refresh; dirty geometry
    /// still invalidates affected records. Clamped to 0..=0.999.
    pub split_transmittance_blend: f32,
    /// Screen-history retention. Clamped to 0..=64.
    /// Zero disables screen history and sparse-lattice jitter.
    /// With history enabled, Split's world intervals persist until cache
    /// eviction or invalidation, independently of this screen-history age.
    pub history_frames: u32,
    /// Geometry-aware a-trous denoising steps. Clamped to
    /// 0..=4. Zero disables the screen-space filter.
    pub spatial_filter_steps: u32,
    /// Samples used for direct lighting at ray-hit surfaces, clamped to 1..=32.
    pub emissive_samples: u32,
}

impl Default for RadianceCascades {
    fn default() -> Self {
        Self {
            sampling: SamplingSettings::default(),
            history_epoch: 0,
            probe_spacing: 0.25,
            interval_length: 0.4,
            resolution_divisor: 2,
            max_rays: 131_072,
            normal_bias: 0.005,
            intensity: 1.0,
            multi_bounce: false,
            cached_hit_sky: false,
            hardware_ray_queries: false,
            artistic_occlusion: true,
            environment: LinearRgba::BLACK,
            temporal_blend: 0.88,
            split_transmittance_blend: 0.98,
            history_frames: 24,
            spatial_filter_steps: 3,
            emissive_samples: 16,
        }
    }
}

impl RadianceCascades {
    /// Conservative receiver-grid and reconstruction settings for 4 GB GPUs at
    /// 1280×720. Use this when an explicit low-end workload cap is preferred.
    pub fn low_end() -> Self {
        Self {
            sampling: SamplingSettings {
                // Preserve useful Cornell-scale detail while retaining enough
                // directional keys to avoid churn across the room's surfaces.
                max_cache_entries: 65_536,
                ..SamplingSettings::low_end()
            },
            probe_spacing: 1.0,
            interval_length: 1.6,
            resolution_divisor: 8,
            max_rays: 1_024,
            emissive_samples: 4,
            history_frames: 24,
            temporal_blend: 0.95,
            spatial_filter_steps: 3,
            ..Self::default()
        }
    }
}

impl RadianceCascades {
    /// Invalidates this view's screen history and Split world cache on the next rendered frame.
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
        self.temporal_blend = finite(self.temporal_blend, 0.88, 0.0, 0.99);
        self.split_transmittance_blend = finite(self.split_transmittance_blend, 0.98, 0.0, 0.999);
        self.history_frames = self.history_frames.min(64);
        self.spatial_filter_steps = self.spatial_filter_steps.min(4);
        self.emissive_samples = self.emissive_samples.clamp(1, 32);
        self.sampling = self.sampling.sanitized();
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

impl RadianceCascadesPlugin {
    /// Memory limits for a 4 GB GPU: 256 MiB scene, 256 MiB material atlas,
    /// and 512 MiB aggregate camera allocations (including resize overlap,
    /// receiver identity textures, and packed lights).
    pub fn low_end() -> Self {
        Self {
            scene_gpu_budget_bytes: 256 * 1024 * 1024,
            texture_atlas_budget_bytes: 256 * 1024 * 1024,
            view_gpu_budget_bytes: 512 * 1024 * 1024,
            max_concurrent_pipeline_compiles: 1,
            ..Self::default()
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
        let status = RadianceCascadesStatus::default();
        app.init_resource::<scene::GiScene>()
            .insert_resource(status.clone())
            .insert_resource(render::SceneBudget {
                live_bytes: self.scene_gpu_budget_bytes,
                view_bytes: self.view_gpu_budget_bytes,
                upload_bytes: self.geometry_upload_bytes_per_frame.max(16),
            })
            .init_resource::<environment::EnvironmentTables>()
            .add_plugins((
                ExtractComponentPlugin::<RadianceCascades>::default(),
                ExtractComponentPlugin::<RadianceCascadesRuntimeBudget>::default(),
                ExtractComponentPlugin::<RadianceCascadesRawOutput>::default(),
                ExtractComponentPlugin::<RadianceCascadesEnvironment>::default(),
                ExtractResourcePlugin::<scene::GiScene>::default(),
                ExtractResourcePlugin::<render::SceneBudget>::default(),
                ExtractResourcePlugin::<environment::EnvironmentTables>::default(),
                ExtractResourcePlugin::<RadianceCascadesStatus>::default(),
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
            status,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renderer_status_clones_share_failures_and_submitted_revisions() {
        let status = RadianceCascadesStatus::default();
        let render_status = status.clone();
        render_status.record_failure("shader failed".into());
        let revisions = RadianceCascadesRevisions {
            scene: 8,
            geometry: 5,
            lighting: 3,
        };
        render_status.record_submitted_revisions(revisions);
        render_status.record_dispatch(revisions);
        assert_eq!(status.failures(), ["shader failed"]);
        assert_eq!(status.submitted_revisions(), revisions);
        assert_eq!(
            status.last_dispatch(),
            Some(RadianceCascadesDispatch {
                sequence: 1,
                revisions,
            })
        );
        let raw_frame = RadianceCascadesRawFrame {
            frame_id: 7,
            dispatch: status.last_dispatch().unwrap(),
            submitted_at: Instant::now(),
        };
        render_status.record_raw_frame(Some(raw_frame));
        render_status.record_raw_frame(None);
        render_status.record_raw_frame(Some(raw_frame));
        assert_eq!(status.take_raw_frame(), Some(Some(raw_frame)));
        assert_eq!(status.take_raw_frame(), Some(None));
        assert_eq!(status.take_raw_frame(), Some(Some(raw_frame)));
        assert_eq!(status.take_raw_frame(), None);
        assert!(!status.compilation_in_flight());
        render_status.set_shader_compile_jobs(1);
        assert!(status.compilation_in_flight());
        render_status.set_shader_compile_jobs(0);
        render_status.set_pipeline_compile_jobs(1);
        assert!(status.compilation_in_flight());
        render_status.set_pipeline_compile_jobs(0);
        assert!(!status.compilation_in_flight());
    }

    #[test]
    fn invalid_settings_are_bounded() {
        let settings = RadianceCascades {
            probe_spacing: f32::NAN,
            resolution_divisor: 0,
            max_rays: u32::MAX,
            temporal_blend: f32::INFINITY,
            history_frames: u32::MAX,
            spatial_filter_steps: u32::MAX,
            emissive_samples: u32::MAX,
            ..default()
        }
        .sanitized();
        assert_eq!(settings.probe_spacing, 0.25);
        assert_eq!(settings.resolution_divisor, 1);
        assert_eq!(settings.max_rays, 131_072);
        assert_eq!(settings.temporal_blend, 0.88);
        assert_eq!(settings.history_frames, 64);
        assert_eq!(settings.spatial_filter_steps, 4);
        assert_eq!(settings.emissive_samples, 32);
    }

    #[test]
    fn shared_sampling_and_split_low_end_preset_are_bounded() {
        let unlimited = SamplingSettings::default();
        assert_eq!(unlimited, SamplingSettings::unlimited());
        assert_eq!(unlimited.resolve_scale, 1);

        let low_end = RadianceCascades::low_end();
        assert_eq!(low_end.sampling.grid_divisor, 8);
        assert_eq!(low_end.sampling.max_receiver_cells, 16_384);
        assert_eq!(low_end.sampling.max_updates_per_frame, 1_024);
        assert_eq!(low_end.sampling.max_cache_entries, 65_536);
        assert_eq!(low_end.sampling.max_traversal_queries, 16_384);
        assert_eq!(low_end.sampling.resolve_scale, 2);
        assert_eq!(low_end.max_rays, 1_024);
        assert_eq!(low_end.emissive_samples, 4);
        assert_eq!(low_end.probe_spacing, 1.0);
        assert_eq!(low_end.interval_length, 1.6);
        assert_eq!(low_end.history_frames, 24);
        assert_eq!(low_end.temporal_blend, 0.95);
        assert_eq!(low_end.spatial_filter_steps, 3);
        let default_settings = RadianceCascades::default();
        assert_eq!(default_settings.resolution_divisor, 2);
        assert_eq!(default_settings.max_rays, 131_072);
        assert_eq!(default_settings.emissive_samples, 16);
    }

    #[test]
    fn cache_entry_limit_accepts_262144_and_clamps_larger_values() {
        let base = SamplingSettings::unlimited();
        let maximum = SamplingSettings {
            max_cache_entries: 262_144,
            ..base
        }
        .sanitized();
        assert_eq!(maximum.max_cache_entries, 262_144);

        let over_limit = SamplingSettings {
            max_cache_entries: u32::MAX,
            ..base
        }
        .sanitized();
        assert_eq!(over_limit.max_cache_entries, 262_144);
    }

    #[test]
    fn low_end_plugin_sets_separate_aggregate_resource_caps() {
        let plugin = RadianceCascadesPlugin::low_end();
        assert_eq!(plugin.scene_gpu_budget_bytes, 256 * 1024 * 1024);
        assert_eq!(plugin.texture_atlas_budget_bytes, 256 * 1024 * 1024);
        assert_eq!(plugin.view_gpu_budget_bytes, 512 * 1024 * 1024);
        assert_eq!(plugin.max_concurrent_pipeline_compiles, 1);
        assert_eq!(
            plugin.scene_gpu_budget_bytes
                + plugin.texture_atlas_budget_bytes as u64
                + plugin.view_gpu_budget_bytes,
            1024 * 1024 * 1024
        );
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

#[cfg(test)]
mod transport_tests;

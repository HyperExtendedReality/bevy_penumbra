//! Local Sponza courtyard, curtains, ivy, and a controllable 24-hour sun.
//! Brackets scrub time; Space animates; 1/2/3/4 select dawn/noon/dusk/night.
//! G toggles GI. Right mouse + WASD/QE flies.
//! P captures; Escape exits.
//! `--benchmark-windowed` measures normal rendering/presentation without draining
//! the GPU each frame. Add `--benchmark-vsync` to request synchronized presentation.
#![recursion_limit = "256"]

use bevy::{
    app::{AppExit, ScheduleRunnerPlugin},
    asset::{RenderAssetUsages, UntypedAssetLoadFailedEvent},
    camera::{Exposure, RenderTarget, ShadowLodOrigin},
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    gltf::{GltfAssetLabel, GltfLoaderSettings},
    pbr::DefaultOpaqueRendererMethod,
    prelude::*,
    render::{
        Render, RenderApp, RenderPlugin,
        diagnostic::RenderDiagnosticsPlugin,
        gpu_readback::{Readback, ReadbackComplete},
        render_resource::{PollType, TextureFormat, TextureUsages},
        renderer::{RenderAdapterInfo, RenderDevice, RenderGraph, RenderGraphSystems},
        view::screenshot::{Screenshot, ScreenshotCaptured, save_to_disk},
    },
    window::{ExitCondition, PresentMode, WindowResolution},
    winit::WinitPlugin,
    world_serialization::{WorldInstanceReady, WorldInstanceSpawner},
};
use bevy_penumbra::{
    FrameTimingPlugin, RadianceCascades, RadianceCascadesDispatch, RadianceCascadesGeometry,
    RadianceCascadesPlugin, RadianceCascadesRawOutput, RadianceCascadesRevisions,
    RadianceCascadesStatus, HardwareRayQueries,
};
use bevy_texture_cache::{TextureCachePlugin, TextureCacheStats};
use std::{
    collections::HashSet,
    path::Path,
    sync::{Arc, Mutex},
};

const COURTYARD: &str = "NewSponza_Main_glTF_003.gltf";
const CURTAINS: &str = "NewSponza_Curtains_glTF.gltf";
const IVY: &str = "NewSponza_IvyGrowth_glTF.gltf";
const TEXTURE_SIZE: u32 = 512;
// Secondary mesh/environment NEE each use one sample per ray-hit surface.
const SPONZA_LIGHT_SAMPLES: u32 = 2;
// Keep the requested Sponza quality profile fixed while measuring its cost.
const SPONZA_UPDATE_BUDGET: u32 = 8_192;
const SPONZA_TRAVERSAL_QUERY_BUDGET: u32 = 131_072;
const SPONZA_RESOLVE_SCALE: u32 = 4;
const SPONZA_PROBE_SPACING_METERS: f32 = 1.0;
const SPONZA_INTERVAL_LENGTH_METERS: f32 = 1.6;
// Stream more scene geometry per frame to shorten Sponza startup; library defaults stay unchanged.
const SPONZA_GEOMETRY_UPLOAD_BYTES_PER_FRAME: usize = 32 * 1024 * 1024;
const SCENE_GPU_BUDGET: u64 = 2 * 1024 * 1024 * 1024;
const ATLAS_GPU_BUDGET: usize = 768 * 1024 * 1024;
// Source-separated intervals and irradiance tables retain the native candidate
// grid within this view cap; scene and atlas budgets are accounted separately.
const VIEW_GPU_BUDGET: u64 = 1024 * 1024 * 1024;
const SPONZA_SPLIT_MAX_RECEIVER_CELLS: u32 = 32_768;
const SPONZA_SPLIT_MAX_CACHE_ENTRIES: u32 = 262_144;
const CAMERA_CURSOR_GRAB_TOGGLE_KEY: KeyCode = KeyCode::F12;
const CONTROLS: &str = "[ / ] time | Space play | 1 dawn 2 noon 3 dusk 4 night | G GI | H hardware rays | RMB hold + WASD/QE fly | F12 cursor grab toggle | P screenshot";

#[derive(Resource)]
struct Demo {
    startup_started: std::time::Instant,
    scene_ready_at: Option<std::time::Instant>,
    first_gi_dispatch_at: Option<std::time::Instant>,
    first_95pct_support_logged: bool,
    scenes: Vec<String>,
    capture_size: (u32, u32),
    capture_prefix: Option<String>,
    hour: f32,
    playing: bool,
    capture: bool,
    target: Option<Handle<Image>>,
    loaded: usize,
    day_mean: f32,
    day_pixels: Vec<u8>,
    gi_ready: bool,
    gi_frames: u32,
    last_gi_dispatch: Option<RadianceCascadesRevisions>,
    capture_stage: usize,
    requested: bool,
    settle_until: f64,
    benchmark: bool,
    benchmark_windowed: bool,
    benchmark_vsync: bool,
    benchmark_no_gi: bool,
    multi_bounce: bool,
    cached_hit_sky: bool,
    hardware_ray_queries: bool,
    resolution_divisor: u32,
    max_rays: u32,
    max_traversal_queries: u32,
    warmup_frames: usize,
    measured_frames: usize,
    benchmark_output: String,
    motion_capture: bool,
    camera_motion_capture: bool,
    camera_pose_frame: Option<u32>,
    motion_frame: u32,
    latest_raw_sample: Option<RawGiSample>,
    motion_light_baseline_raw: Option<RawGiSample>,
    motion_light_response_samples: Vec<RawGiSample>,
    motion_light_change: Option<MotionLightChange>,
    motion_reset_settle: MotionSettleObservation,
    motion_capture_pending: Option<PendingMotionCapture>,
    raw_gi: RawGiDiagnostics,
    stability_capture: bool,
    stability_frame: u32,
    stability_pixels: Vec<u8>,
    stability_differences: Vec<(f32, f32, f32)>,
}

#[derive(Clone, Copy, Default)]
struct RawGiDiagnostics {
    sample_count: u32,
    total_pixels: usize,
    eligible_pixels: usize,
    finite_pixels: usize,
    supported_pixels: usize,
    mean_luminance: f32,
    supported_mean_luminance: f32,
    max_luminance: f32,
    irradiance_cache_rejections: Option<f64>,
    persistent_cache_dropped: Option<f64>,
    directional_probe_count: Option<f64>,
    transport_query_attempts: Option<f64>,
    transport_queries_admitted: Option<f64>,
    retention_demand: Option<f64>,
    frame_id: u64,
    dispatch_sequence: u64,
    scene_revision: u64,
    geometry_revision: u64,
    light_revision: u64,
}

#[derive(Clone)]
struct RawGiSample {
    frame_id: u64,
    dispatch: RadianceCascadesDispatch,
    submitted_at: std::time::Instant,
    radiance: Vec<[f32; 3]>,
    eligible: Vec<bool>,
    supported: Vec<bool>,
}
const MOTION_SETTLE_DISPATCHES: u64 = 128;
const MOTION_SETTLE_MIDPOINT_DISPATCHES: u64 = 64;

#[derive(Clone)]
struct MotionLightChange {
    revision: u64,
    encoded_started_at: std::time::Instant,
    settle: MotionSettleObservation,
}

#[derive(Clone)]
struct PendingMotionCapture {
    frame: u32,
    camera_transform: Transform,
    history_epoch: u32,
    occluder_position: Option<Vec3>,
    minimum_dispatch_sequence: u64,
    previous_geometry_revision: Option<u64>,
    previous_lighting_revision: Option<u64>,
}

#[derive(Clone, Default)]
struct MotionSettleObservation {
    start_sequence: Option<u64>,
    midpoint_sample: Option<RawGiSample>,
}

impl MotionSettleObservation {
    fn begin(&mut self, start_sequence: u64) {
        self.start_sequence = Some(start_sequence);
        self.midpoint_sample = None;
    }

    fn observe(&mut self, sample: &RawGiSample) {
        let Some(start_sequence) = self.start_sequence else {
            return;
        };
        if self.midpoint_sample.is_none()
            && sample.dispatch.sequence.saturating_sub(start_sequence)
                >= MOTION_SETTLE_MIDPOINT_DISPATCHES
        {
            self.midpoint_sample = Some(sample.clone());
        }
    }

    fn encoded_samples(&self, current_sequence: u64) -> u64 {
        self.start_sequence
            .map_or(0, |start| current_sequence.saturating_sub(start))
    }

    fn midpoint_drift(&self, settled: &RawGiSample) -> Option<f32> {
        self.midpoint_sample
            .as_ref()
            .and_then(|middle| stationary_sample_drift(middle, settled))
    }
}

// The raw bytes are retained only for the opt-in moving-sun capture.
#[derive(Resource, Default)]
struct SunCaptureRaw(Vec<u8>);

#[derive(Default)]
struct SunCaptureState {
    stage: u8,
    frame: u32,
    start_hour: f32,
    baseline: Option<RawGiSample>,
    immediate: Option<RawGiSample>,
    midpoint: Option<RawGiSample>,
    warmup_frame_drift: Option<f32>,
    motion_previous: Option<RawGiSample>,
    motion_drift_sum: f64,
    motion_drift_max: f32,
    motion_drift_count: u32,
    final_sequence_floor: u64,
    final_revision: u64,
    stable_revision_frames: u32,
}

const SUN_MOTION_FRAMES: u32 = 64;
const SUN_HOURS_PER_FRAME: f32 = 0.02;

fn gi_capture_ready(
    scene_loaded: bool,
    dispatch: Option<RadianceCascadesDispatch>,
    eligible_pixels: usize,
    support_ratio: f32,
    warmup_frames: u32,
    required_frames: u32,
) -> bool {
    let Some(dispatch) = dispatch else {
        return false;
    };
    let revisions = dispatch.revisions;
    scene_loaded
        && revisions.scene > 0
        && revisions.geometry > 0
        && revisions.lighting > 0
        && eligible_pixels > 0
        && support_ratio >= 0.95
        && warmup_frames >= required_frames
}

#[derive(Component)]
struct Sun;

#[derive(Component)]
struct PendingScene(Vec<Entity>);

#[derive(Component)]
struct MotionCaptureOccluder;

#[derive(Component)]
struct MotionCaptureLight;

#[derive(Component)]
struct MotionCaptureSpot;

#[derive(Component)]
struct MotionCaptureEmissive;

#[derive(Component)]
struct DebugOverlay;

fn main() -> AppExit {
    let startup_started = std::time::Instant::now();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut scenes = [COURTYARD, CURTAINS, IVY].map(str::to_owned).to_vec();
    let mut hour = 12.0;
    let mut capture = false;
    let mut benchmark = false;
    let mut benchmark_windowed = false;
    let mut benchmark_vsync = false;
    let mut benchmark_no_gi = false;
    let mut multi_bounce = false;
<<<<<<< Updated upstream
    let mut resolution_divisor = SPONZA_RESOLVE_SCALE;
=======
    let mut cached_hit_sky = false;
    let mut hardware_ray_queries = true;
    let mut resolution_divisor = SPONZA_GRID_DIVISOR;
>>>>>>> Stashed changes
    let mut max_rays = SPONZA_UPDATE_BUDGET;
    let mut max_traversal_queries = SPONZA_TRAVERSAL_QUERY_BUDGET;
    let mut warmup_frames = 120;
    let mut measured_frames = 240;
    let mut benchmark_output = String::from("tmp/sponza_benchmark.json");
    let mut motion_capture = false;
    let mut camera_motion_capture = false;
    let mut camera_pose_frame = None;
    let mut stability_capture = false;
    let mut sun_capture = false;
    let mut capture_prefix = None;
    for arg in &args {
        if let Some(value) = arg.strip_prefix("--scene=") {
            assert!(
                !value.is_empty(),
                "--scene requires a path relative to assets/"
            );
            scenes = vec![value.to_owned()];
        } else if let Some(value) = arg.strip_prefix("--time=") {
            hour = value
                .parse::<f32>()
                .expect("--time requires a number in [0,24)");
            assert!(
                hour.is_finite() && (0.0..24.0).contains(&hour),
                "--time requires a number in [0,24)"
            );
        } else if arg == "--capture" {
            capture = true;
        } else if arg == "--capture-moving-sun" {
            capture = true;
            sun_capture = true;
        } else if arg == "--capture-motion" {
            capture = true;
            motion_capture = true;
        } else if arg == "--capture-camera-motion" {
            capture = true;
            motion_capture = true;
            camera_motion_capture = true;
        } else if let Some(value) = arg.strip_prefix("--camera-pose-frame=") {
            let frame = value
                .parse::<u32>()
                .expect("--camera-pose-frame requires an integer");
            assert!(frame <= 127, "--camera-pose-frame must be 0..=127");
            camera_pose_frame = Some(frame);
        } else if arg == "--capture-stability" {
            capture = true;
            stability_capture = true;
        } else if arg == "--benchmark" {
            benchmark = true;
        } else if arg == "--benchmark-windowed" {
            benchmark = true;
            benchmark_windowed = true;
        } else if arg == "--benchmark-vsync" {
            benchmark_vsync = true;
        } else if arg == "--benchmark-no-gi" {
            benchmark = true;
            benchmark_windowed = true;
            benchmark_no_gi = true;
        } else if arg == "--benchmark-offscreen-no-gi" {
            benchmark = true;
            benchmark_windowed = false;
            benchmark_no_gi = true;
        } else if arg == "--multi-bounce" {
            multi_bounce = true;
        } else if arg == "--cached-hit-sky" {
            cached_hit_sky = true;
        } else if arg == "--software-traversal" {
            hardware_ray_queries = false;
        } else if let Some(value) = arg.strip_prefix("--max-rays=") {
            max_rays = value.parse().expect("--max-rays requires an integer");
            assert!(
                (1..=131072).contains(&max_rays),
                "--max-rays must be 1..=131072"
            );
        } else if let Some(value) = arg.strip_prefix("--max-traversal-queries=") {
            max_traversal_queries = value
                .parse()
                .expect("--max-traversal-queries requires an integer");
            assert!(
                max_traversal_queries <= 131_072,
                "--max-traversal-queries must be 0..=131072 (0 disables the cap)"
            );
        } else if let Some(value) = arg.strip_prefix("--resolution-divisor=") {
            resolution_divisor = value
                .parse()
                .expect("--resolution-divisor requires an integer");
            assert!(
                (1..=256).contains(&resolution_divisor),
                "--resolution-divisor must be 1..=256"
            );
        } else if let Some(value) = arg.strip_prefix("--warmup-frames=") {
            warmup_frames = value.parse().expect("--warmup-frames requires an integer");
        } else if let Some(value) = arg.strip_prefix("--measured-frames=") {
            measured_frames = value
                .parse()
                .expect("--measured-frames requires an integer");
        } else if let Some(value) = arg.strip_prefix("--json-output=") {
            assert!(!value.is_empty(), "--json-output requires a path");
            benchmark_output = value.to_owned();
        } else if let Some(value) = arg.strip_prefix("--capture-prefix=") {
            assert!(!value.is_empty(), "--capture-prefix requires a path prefix");
            capture_prefix = Some(value.to_owned());
        } else if arg.starts_with("--width=") || arg.starts_with("--height=") {
            // Parsed below once capture/benchmark mode is known.
        } else {
            panic!(
                "Unknown argument {arg}; use --scene=path.gltf, --time=10, --max-rays=1024, --max-traversal-queries=131072, --resolution-divisor=8, --width=160, --height=120, --camera-pose-frame=88 with --capture-stability, --capture-prefix=path, --capture, --capture-motion, --capture-camera-motion, --capture-moving-sun, --capture-stability, --multi-bounce, --benchmark, or --benchmark-offscreen-no-gi"
            );
        }
    }
    assert!(
        !sun_capture || (!motion_capture && !stability_capture),
        "--capture-moving-sun cannot be combined with another capture mode"
    );
    assert!(
        camera_pose_frame.is_none() || stability_capture,
        "--camera-pose-frame requires --capture-stability"
    );
    assert!(
        !benchmark || (warmup_frames >= 120 && measured_frames >= 240),
        "Sponza benchmark requires at least 120 warmup and 240 measured frame samples"
    );
    let default_size = if capture { (640, 360) } else { (1920, 1080) };
    let capture_size = (
        dimension_arg(&args, "--width=", default_size.0),
        dimension_arg(&args, "--height=", default_size.1),
    );
    let mut app = App::new();
    // Keep shader compilation off the window thread so the demo stays responsive.
    let plugins = DefaultPlugins
        .set(AssetPlugin {
            file_path: concat!(env!("CARGO_MANIFEST_DIR"), "/assets").into(),
            ..default()
        })
        .set(RenderPlugin {
            synchronous_pipeline_compilation: false,
            ..default()
        });
    assert!(!(capture && benchmark), "choose --capture or --benchmark");
    if capture || (benchmark && !benchmark_windowed) {
        app.add_plugins(
            plugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .disable::<WinitPlugin>(),
        )
        // drain_gpu below provides back-pressure for the offscreen target.
        // A fixed capture delay changes temporal refinement from normal play.
        .add_plugins(ScheduleRunnerPlugin::run_loop(std::time::Duration::ZERO));
        app.sub_app_mut(RenderApp)
            .add_systems(RenderGraph, drain_gpu.in_set(RenderGraphSystems::Finish));
    } else {
        app.add_plugins(
            plugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: CONTROLS.into(),
                    resolution: WindowResolution::new(capture_size.0, capture_size.1)
                        .with_scale_factor_override(1.0),
                    present_mode: if benchmark_windowed && !benchmark_vsync {
                        PresentMode::AutoNoVsync
                    } else {
                        PresentMode::AutoVsync
                    },
                    ..default()
                }),
                ..default()
            }),
        )
        .add_plugins(FreeCameraPlugin);
    }
    if benchmark_windowed {
        app.insert_resource(bevy::winit::WinitSettings::continuous());
    }
    app.insert_resource(DefaultOpaqueRendererMethod::deferred())
        .insert_resource(GlobalAmbientLight {
            brightness: 0.0,
            ..default()
        })
        .insert_resource(Demo {
            startup_started,
            scene_ready_at: None,
            first_gi_dispatch_at: None,
            first_95pct_support_logged: false,
            scenes,
            capture_size,
            capture_prefix,
            hour,
            playing: false,
            capture,
            target: None,
            loaded: 0,
            day_mean: 0.0,
            day_pixels: Vec::new(),
            gi_ready: false,
            gi_frames: 0,
            last_gi_dispatch: None,
            capture_stage: 0,
            requested: false,
            settle_until: 0.0,
            benchmark,
            benchmark_windowed,
            benchmark_vsync,
            benchmark_no_gi,
            multi_bounce,
            cached_hit_sky,
            hardware_ray_queries,
            resolution_divisor,
            max_rays,
            max_traversal_queries,
            warmup_frames,
            measured_frames,
            benchmark_output,
            motion_capture,
            camera_motion_capture,
            camera_pose_frame,
            motion_frame: 0,
            latest_raw_sample: None,
            motion_light_baseline_raw: None,
            motion_light_response_samples: Vec::new(),
            motion_light_change: None,
            motion_reset_settle: MotionSettleObservation::default(),
            motion_capture_pending: None,
            raw_gi: RawGiDiagnostics::default(),
            stability_capture,
            stability_frame: 0,
            stability_pixels: Vec::new(),
            stability_differences: Vec::new(),
        })
        .add_plugins(RadianceCascadesPlugin {
            scene_gpu_budget_bytes: SCENE_GPU_BUDGET,
            texture_atlas_budget_bytes: ATLAS_GPU_BUDGET,
            view_gpu_budget_bytes: VIEW_GPU_BUDGET,
            geometry_upload_bytes_per_frame: SPONZA_GEOMETRY_UPLOAD_BYTES_PER_FRAME,
            max_concurrent_pipeline_compiles: 2,
            ..RadianceCascadesPlugin::low_end()
        })
        .add_plugins(RenderDiagnosticsPlugin)
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(TextureCachePlugin {
            max_dimension: TEXTURE_SIZE,
            cache_dir: "tmp/texture-cache".into(),
            ..default()
        })
        .add_observer(scene_ready)
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                prepare_scene,
                fail_capture_on_gi_error,
                controls,
                update_debug_overlay,
                lighting,
                capture_frame,
                capture_motion,
                capture_stability,
                capture_moving_sun,
                asset_errors,
                stop_benchmark_raw_output,
                performance,
                resize_raw_gi_output,
            )
                .chain(),
        );
    if sun_capture {
        app.insert_resource(SunCaptureRaw::default());
    }
    if benchmark {
        app.add_plugins(FrameTimingPlugin);
        let shared_metadata = SharedRendererMetadata::default();
        app.insert_resource(shared_metadata.clone());
        app.sub_app_mut(RenderApp)
            .insert_resource(shared_metadata)
            .add_systems(Render, collect_renderer_metadata);
    }
    app.run()
}

fn dimension_arg(args: &[String], prefix: &str, fallback: u32) -> u32 {
    args.iter()
        .find_map(|arg| {
            arg.strip_prefix(prefix).map(|value| {
                value
                    .parse::<u32>()
                    .expect("capture dimension must be an integer")
            })
        })
        .unwrap_or(fallback)
        .max(1)
}

fn apply_sponza_profile(settings: &mut RadianceCascades, demo: &Demo) {
    settings.multi_bounce = demo.multi_bounce;
    settings.cached_hit_sky = demo.cached_hit_sky;
    settings.hardware_ray_queries = demo.hardware_ray_queries;
    settings.probe_spacing = SPONZA_PROBE_SPACING_METERS;
    settings.interval_length = SPONZA_INTERVAL_LENGTH_METERS;
    settings.emissive_samples = SPONZA_LIGHT_SAMPLES;
    settings.resolution_divisor = demo.resolution_divisor;
    settings.sampling.grid_divisor = demo.resolution_divisor;
    settings.sampling.resolve_scale = SPONZA_RESOLVE_SCALE;
    settings.max_rays = demo.max_rays;
    settings.sampling.max_updates_per_frame = demo.max_rays;
    settings.sampling.max_traversal_queries = demo.max_traversal_queries;
    settings.sampling.max_receiver_cells = SPONZA_SPLIT_MAX_RECEIVER_CELLS;
    settings.sampling.max_cache_entries = SPONZA_SPLIT_MAX_CACHE_ENTRIES;
    settings.temporal_blend = 0.985;
    settings.history_frames = 64;
}
fn setup(
    mut commands: Commands,
    assets: Res<AssetServer>,
    mut demo: ResMut<Demo>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    info!(
        "{CONTROLS}; demo material textures limited to {TEXTURE_SIZE} pixels, GI scene cap 2 GiB, texture-atlas cap 768 MiB"
    );
    if !demo.capture && !demo.benchmark {
        commands.spawn((
            DebugOverlay,
            Text::new("FPS: 0.0"),
            TextFont {
                font_size: FontSize::Px(18.0),
                ..default()
            },
            TextColor(Color::WHITE),
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(12.0),
                left: Val::Px(12.0),
                padding: UiRect::all(Val::Px(10.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.02, 0.02, 0.02, 0.78)),
        ));
    }
    commands.spawn(WorldAssetRoot(load_scene(&assets, &demo.scenes[0])));
    if demo.motion_capture && !demo.camera_motion_capture {
        let initial_motion_positions = motion_capture_positions(0);
        commands.spawn((
            Name::new("Deterministic moving GI occluder"),
            Mesh3d(meshes.add(Cuboid::from_size(Vec3::splat(0.7)))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.75, 0.12, 0.05),
                ..default()
            })),
            Transform::from_translation(initial_motion_positions.occluder),
            RadianceCascadesGeometry,
            MotionCaptureOccluder,
        ));
        commands.spawn((
            Name::new("Deterministic moving GI point light"),
            PointLight {
                color: Color::linear_rgb(1.0, 0.62, 0.34),
                intensity: 18_000.0,
                range: 12.0,
                ..default()
            },
            Transform::from_translation(initial_motion_positions.point_light),
            MotionCaptureLight,
        ));
        commands.spawn((
            Name::new("Deterministic moving GI spot light"),
            SpotLight {
                color: Color::linear_rgb(0.45, 0.68, 1.0),
                intensity: 12_000.0,
                range: 14.0,
                inner_angle: 0.35,
                outer_angle: 0.65,
                ..default()
            },
            Transform::from_translation(initial_motion_positions.spot_light)
                .looking_at(initial_motion_positions.occluder, Vec3::Y),
            MotionCaptureSpot,
        ));
        commands.spawn((
            Name::new("Deterministic GI emissive test source"),
            Mesh3d(meshes.add(Cuboid::from_size(Vec3::splat(0.35)))),
            MeshMaterial3d(materials.add(StandardMaterial {
                emissive: LinearRgba::new(14.0, 4.0, 0.6, 1.0),
                ..default()
            })),
            Transform::from_xyz(-6.5, 1.0, 0.5),
            RadianceCascadesGeometry,
            MotionCaptureEmissive,
        ));
    }
    // IvyLeaf_BaseColor.png is RGB and the supplied ivy is leaf-shaped geometry.
    // Add explicit cutout cards to exercise alpha rejection without inventing an
    // opacity channel for the authored ivy or changing its material.
    let mask = images.add(leaf_mask());
    let foliage = materials.add(StandardMaterial {
        base_color_texture: Some(mask),
        alpha_mode: AlphaMode::Mask(0.5),
        double_sided: true,
        cull_mode: None,
        metallic: 0.0,
        perceptual_roughness: 0.9,
        ..default()
    });
    let card = meshes.add(Rectangle::new(1.2, 1.8));
    for z in [-1.5, 0.0, 1.5] {
        commands.spawn((
            Name::new("Alpha foliage test card"),
            Mesh3d(card.clone()),
            MeshMaterial3d(foliage.clone()),
            Transform::from_xyz(-12.0, 1.5, z)
                .with_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2)),
            RadianceCascadesGeometry,
        ));
    }
    commands.spawn((
        Sun,
        DirectionalLight {
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::default(),
    ));
    let mut settings = RadianceCascades::low_end();
    apply_sponza_profile(&mut settings, &demo);
    settings.intensity = if demo.benchmark_no_gi { 0.0 } else { 1.0 };
    info!(
        grid_divisor = settings.sampling.grid_divisor,
        max_updates_per_frame = settings.sampling.max_updates_per_frame,
        max_traversal_queries = settings.sampling.max_traversal_queries,
        max_cache_entries = settings.sampling.max_cache_entries,
        history_frames = settings.history_frames,
        temporal_blend = settings.temporal_blend,
        "Sponza GI workload settings"
    );
    let mut camera = commands.spawn((
        Camera3d::default(),
        Msaa::Off,
        // Keep the raster stable for screen-history reconstruction; this demo
        // uses spatial SMAA instead of adding a second temporal accumulation.
        bevy::anti_alias::smaa::Smaa::default(),
        settings,
        ShadowLodOrigin,
        Exposure::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: std::f32::consts::FRAC_PI_3,
            ..default()
        }),
        // Use the supplied courtyard camera position, clear of pillars and ivy.
        demo.camera_pose_frame
            .map_or_else(|| motion_capture_pose(0), motion_capture_pose),
    ));
    if (demo.capture || demo.benchmark) && !demo.benchmark_windowed {
        let (width, height) = demo.capture_size;
        let target = images.add(Image::new_target_texture(
            width,
            height,
            TextureFormat::Rgba8UnormSrgb,
            None,
        ));
        camera.insert(RenderTarget::Image(target.clone().into()));
        demo.target = Some(target);
    }
    if demo.capture || (demo.benchmark && !demo.benchmark_no_gi) {
        let (width, height) = demo.capture_size;
        let resolve_scale = settings.sampling.resolve_scale.max(1);
        let raw_width = width.div_ceil(resolve_scale).max(1);
        let raw_height = height.div_ceil(resolve_scale).max(1);
        let mut raw =
            Image::new_target_texture(raw_width, raw_height, TextureFormat::Rgba32Float, None);
        raw.texture_descriptor.usage |= TextureUsages::COPY_SRC | TextureUsages::COPY_DST;
        let raw = images.add(raw);
        camera.insert(RadianceCascadesRawOutput(raw.clone()));
        let mut readback = commands.spawn(Readback::texture(raw));
        if demo.benchmark {
            readback.insert(BenchmarkRawReadback);
        }
        readback.observe(
            |event: On<ReadbackComplete>,
             mut demo: ResMut<Demo>,
             mut readback_count: Local<u32>,
             diagnostics: Res<DiagnosticsStore>,
             status: Res<RadianceCascadesStatus>,
             mut sun_raw: Option<ResMut<SunCaptureRaw>>,
             views: Query<&RadianceCascades>| {
                let raw_frame = status.take_raw_frame().flatten();
                let diagnostic_value = |name: &str| {
                    diagnostics
                        .iter()
                        .find(|diagnostic| {
                            let path = diagnostic.path().as_str();
                            path.contains("penumbra/view_") && path.rsplit('/').next() == Some(name)
                        })
                        .and_then(|diagnostic| diagnostic.value())
                };
                let gpu_timing_ms = |path: &str| {
                    diagnostics
                        .iter()
                        .find(|diagnostic| diagnostic.path().as_str() == path)
                        .and_then(|diagnostic| diagnostic.value())
                };
                let mut supported_pixels = 0usize;
                let mut eligible_pixels = 0usize;
                let mut finite_pixels = 0usize;
                let mut luminance_sum = 0.0f32;
                let mut supported_luminance_sum = 0.0f32;
                let mut max_luminance = 0.0f32;
                let mut radiance = Vec::with_capacity(event.data.len() / 16);
                let mut eligible = Vec::with_capacity(event.data.len() / 16);
                let mut supported = Vec::with_capacity(event.data.len() / 16);
                for pixel in event.data.as_chunks::<16>().0 {
                    let values: [f32; 4] = std::array::from_fn(|i| {
                        f32::from_le_bytes(pixel[i * 4..i * 4 + 4].try_into().unwrap())
                    });
                    let finite = values.iter().all(|value| value.is_finite());
                    if finite {
                        let luminance =
                            (0.2126 * values[0] + 0.7152 * values[1] + 0.0722 * values[2])
                                .max(0.0);
                        finite_pixels += 1;
                        luminance_sum += luminance;
                        max_luminance = max_luminance.max(luminance);
                        if values[3] != 0.0 {
                            eligible_pixels += 1;
                        }
                        if values[3] > 0.0 {
                            supported_pixels += 1;
                            supported_luminance_sum += luminance;
                        }
                    }
                    radiance.push([values[0], values[1], values[2]]);
                    eligible.push(finite && values[3] != 0.0);
                    supported.push(finite && values[3] > 0.0);
                }
                let support_ratio = supported_pixels as f32 / eligible_pixels.max(1) as f32;
                let frame_id = raw_frame.map_or(0, |frame| frame.frame_id);
                let dispatch = raw_frame.map(|frame| frame.dispatch).filter(|dispatch| {
                    dispatch.revisions.scene > 0
                        && dispatch.revisions.geometry > 0
                        && dispatch.revisions.lighting > 0
                        && finite_pixels == event.data.len() / 16
                        && event.data.len().is_multiple_of(16)
                });
                if let (Some(dispatch), Some(frame)) = (dispatch, raw_frame) {
                    record_first_gi_dispatch(
                        &mut demo,
                        dispatch,
                        frame.submitted_at,
                        "raw_readback_submission",
                    );
                }
                let revisions = raw_frame.map_or(
                    RadianceCascadesRevisions::default(),
                    |frame| frame.dispatch.revisions,
                );
                let signature = dispatch.map(|value| value.revisions);
                if signature != demo.last_gi_dispatch {
                    demo.last_gi_dispatch = signature;
                    demo.gi_frames = 0;
                    demo.gi_ready = false;
                }
                demo.gi_frames = if dispatch.is_some()
                    && eligible_pixels > 0
                    && support_ratio >= 0.95
                {
                    demo.gi_frames + 1
                } else {
                    0
                };
                *readback_count = (*readback_count).saturating_add(1);
                demo.raw_gi = RawGiDiagnostics {
                    sample_count: *readback_count,
                    total_pixels: event.data.len() / 16,
                    eligible_pixels,
                    finite_pixels,
                    supported_pixels,
                    mean_luminance: luminance_sum / finite_pixels.max(1) as f32,
                    supported_mean_luminance: supported_luminance_sum
                        / supported_pixels.max(1) as f32,
                    max_luminance,
                    irradiance_cache_rejections: diagnostic_value("irradiance_cache_rejections"),
                    persistent_cache_dropped: diagnostic_value("persistent_cache_dropped"),
                    directional_probe_count: diagnostic_value("directional_probe_count"),
                    transport_query_attempts: diagnostic_value("transport_query_attempts"),
                    transport_queries_admitted: diagnostic_value("transport_queries_admitted"),
                    retention_demand: diagnostic_value("retention_demand"),
                    frame_id,
                    dispatch_sequence: raw_frame.map_or(0, |value| value.dispatch.sequence),
                    scene_revision: revisions.scene,
                    geometry_revision: revisions.geometry,
                    light_revision: revisions.lighting,
                };
                if !demo.first_95pct_support_logged
                    && eligible_pixels > 0
                    && support_ratio >= 0.95
                    && dispatch.is_some()
                    && let Some(frame) = raw_frame
                {
                    demo.first_95pct_support_logged = true;
                    let startup_ms = frame
                        .submitted_at
                        .saturating_duration_since(demo.startup_started)
                        .as_secs_f64()
                        * 1_000.0;
                    let dispatch_to_coverage_ms = demo.first_gi_dispatch_at.map(|first| {
                        frame
                            .submitted_at
                            .saturating_duration_since(first)
                            .as_secs_f64()
                            * 1_000.0
                    });
                    info!(
                        startup_to_95pct_raw_support_ms = startup_ms,
                        first_dispatch_to_95pct_raw_support_ms = ?dispatch_to_coverage_ms,
                        supported_pixels,
                        eligible_pixels,
                        coverage = support_ratio,
                        "Sponza first raw GI sample reached 95% eligible-pixel support"
                    );
                }
                let latest_raw_sample = dispatch.zip(raw_frame).map(|(dispatch, frame)| {
                    RawGiSample {
                        frame_id,
                        dispatch,
                        submitted_at: frame.submitted_at,
                        radiance,
                        eligible,
                        supported,
                    }
                });
                if let Some(raw) = sun_raw.as_mut() {
                    raw.0.clear();
                    if latest_raw_sample.is_some() {
                        raw.0.extend_from_slice(&event.data);
                    }
                }
                if let Some(sample) = latest_raw_sample.as_ref() {
                    demo.motion_reset_settle.observe(sample);
                    let baseline_revision = demo
                        .motion_light_baseline_raw
                        .as_ref()
                        .map(|baseline| baseline.dispatch.revisions.lighting);
                    if demo.motion_capture
                        && baseline_revision.is_some_and(|baseline| {
                            baseline != sample.dispatch.revisions.lighting
                        })
                    {
                        let revision = sample.dispatch.revisions.lighting;
                        let started = match demo.motion_light_change.as_ref() {
                            Some(change) if change.revision == revision => change.encoded_started_at,
                            _ => {
                                demo.motion_light_response_samples.clear();
                                let mut settle = MotionSettleObservation::default();
                                settle.begin(sample.dispatch.sequence.saturating_sub(1));
                                settle.observe(sample);
                                demo.motion_light_change = Some(MotionLightChange {
                                    revision,
                                    encoded_started_at: sample.submitted_at,
                                    settle,
                                });
                                sample.submitted_at
                            }
                        };
                        if let Some(change) = demo.motion_light_change.as_mut() {
                            change.settle.observe(sample);
                        }
                        if sample
                            .submitted_at
                            .checked_duration_since(started)
                            .is_some_and(|elapsed| {
                                elapsed <= std::time::Duration::from_millis(100)
                            })
                        {
                            demo.motion_light_response_samples.push(sample.clone());
                        }
                    }
                }
                demo.latest_raw_sample = latest_raw_sample;
                if (demo.benchmark || demo.capture)
                    && (*readback_count).is_multiple_of(60)
                {
                    info!(
                        "Sponza GI readback progress: samples={}, dispatch={}, ready_streak={}, scene_revision={}, geometry_revision={}, light_revision={}, support={}/{}, eligible={}, raw={}, finite={}, mean_luminance={:.5}, supported_mean_luminance={:.5}, max_luminance={:.5}, irradiance_cache_rejections={:?}, persistent_cache_dropped={:?}, directional_probe_count={:?}, transport_query_attempts={:?}, transport_queries_admitted={:?}, retention_demand={:?}, split_rc_gpu_ms={:?}, trace_gpu_ms={:?}, world_resolve_gpu_ms={:?}, merge_gpu_ms={:?}, penumbra_frame_gpu_ms={:?}",
                        *readback_count,
                        demo.raw_gi.dispatch_sequence,
                        demo.gi_frames,
                        revisions.scene,
                        revisions.geometry,
                        revisions.lighting,
                        supported_pixels,
                        eligible_pixels,
                        eligible_pixels,
                        demo.raw_gi.total_pixels,
                        finite_pixels,
                        demo.raw_gi.mean_luminance,
                        demo.raw_gi.supported_mean_luminance,
                        max_luminance,
                        demo.raw_gi.irradiance_cache_rejections,
                        demo.raw_gi.persistent_cache_dropped,
                        demo.raw_gi.directional_probe_count,
                        demo.raw_gi.transport_query_attempts,
                        demo.raw_gi.transport_queries_admitted,
                        demo.raw_gi.retention_demand,
                        gpu_timing_ms("render/split_rc/elapsed_gpu"),
                        gpu_timing_ms("render/split_rc/trace/elapsed_gpu"),
                        gpu_timing_ms("render/split_rc/worldResolve/elapsed_gpu"),
                        gpu_timing_ms("render/split_rc/merge/elapsed_gpu"),
                        gpu_timing_ms("render/penumbra_frame/elapsed_gpu"),
                    );
                }
                let required_frames = views
                    .iter()
                    .map(|gi| gi.history_frames.max(32))
                    .max()
                    .unwrap_or(32);
                // Honor the requested refinement interval in image captures,
                // too: a sparse grid needs several complete update cycles.
                let required_frames = if demo.capture {
                    required_frames.max(demo.warmup_frames.min(u32::MAX as usize) as u32)
                } else {
                    required_frames
                };
                demo.gi_ready = gi_capture_ready(
                    demo.loaded == demo.scenes.len(),
                    dispatch,
                    eligible_pixels,
                    support_ratio,
                    demo.gi_frames,
                    required_frames,
                );
            },
        );
    } else if !demo.benchmark || demo.benchmark_windowed {
        camera.insert(FreeCamera {
            walk_speed: 3.0,
            run_speed: 9.0,
            keyboard_key_toggle_cursor_grab: CAMERA_CURSOR_GRAB_TOGGLE_KEY,
            ..default()
        });
    }
}

fn scene_ready(
    ready: On<WorldInstanceReady>,
    spawner: Res<WorldInstanceSpawner>,
    mut commands: Commands,
) {
    commands.entity(ready.entity).insert(PendingScene(
        spawner.iter_instance_entities(ready.instance_id).collect(),
    ));
}

fn load_scene(assets: &AssetServer, path: &str) -> Handle<WorldAsset> {
    assets
        .load_builder()
        .with_settings(|settings: &mut GltfLoaderSettings| {
            // The demo owns its view; Sponza includes six active authoring cameras.
            settings.load_cameras = false;
            settings.load_lights = false;
            // Materials must be available to the native renderer as well as GI.
            // The texture-cache loader limits images before their GPU upload.
            settings.load_materials = RenderAssetUsages::default();
        })
        .load(GltfAssetLabel::Scene(0).from_asset(path.to_owned()))
}

#[allow(clippy::too_many_arguments)]
fn prepare_scene(
    assets: Res<AssetServer>,
    pending: Query<(Entity, &PendingScene)>,
    meshes: Query<(Entity, &MeshMaterial3d<StandardMaterial>), With<Mesh3d>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut commands: Commands,
    mut demo: ResMut<Demo>,
    time: Res<Time<Real>>,
    cameras: Query<(&Camera, Option<&RadianceCascadesRawOutput>)>,
    cache: Res<TextureCacheStats>,
) {
    for (root, pending) in &pending {
        let mut textures = HashSet::new();
        let mut count = 0;
        for entity in &pending.0 {
            let Ok((_, material)) = meshes.get(*entity) else {
                continue;
            };
            let mut material = materials
                .get_mut(&material.0)
                .expect("Loaded glTF material");
            // The supplied scene mixes mirrored transforms and thin surfaces.
            // Match software GI's two-sided visibility in the native renderer.
            material.cull_mode = None;
            material.double_sided = true;
            for texture in [
                &material.base_color_texture,
                &material.normal_map_texture,
                &material.metallic_roughness_texture,
                &material.emissive_texture,
                &material.occlusion_texture,
            ]
            .into_iter()
            .flatten()
            {
                textures.insert(texture.id());
            }
            count += 1;
        }
        // Instance readiness precedes external PNG dependencies. Commit the entire
        // instance only after every sampled image is CPU resident.
        if textures.iter().any(|id| !images.contains(*id)) {
            continue;
        }
        assert!(
            textures.iter().all(|id| images
                .get(*id)
                .is_some_and(|image| image.width().max(image.height()) <= TEXTURE_SIZE)),
            "Demo PNG loader did not apply the texture preset"
        );
        assert!(
            count > 0,
            "The selected scene has no StandardMaterial meshes"
        );
        // Pack GI geometry once. Repacking after each import keeps the previous
        // multi-hundred-MiB scene alive alongside its larger replacement.
        if demo.loaded + 1 == demo.scenes.len() {
            demo.gi_frames = 0;
            demo.gi_ready = false;
            assert_eq!(
                cameras.iter().count(),
                1,
                "Sponza must have exactly one camera"
            );
            // Discard the tiny startup scene's readback while the full GI scene
            // uploads. Fresh nonzero results now imply the new bindings ran.
            for (_, raw) in &cameras {
                if let Some(raw) = raw {
                    let mut image = images.get_mut(&raw.0).expect("GI capture target");
                    image.data = Some(vec![0; (image.width() * image.height() * 16) as usize]);
                }
            }
            for (entity, _) in &meshes {
                commands.entity(entity).insert(RadianceCascadesGeometry);
            }
        }
        commands.entity(root).remove::<PendingScene>();
        info!(
            "Sponza scene ready: {count} mesh primitives, {} material textures ({TEXTURE_SIZE}px demo preset)",
            textures.len()
        );
        demo.loaded += 1;
        if demo.loaded == demo.scenes.len() {
            info!(
                "Texture cache: {} hits, {} PNG decodes; scene loading {:.3}s",
                cache.hits(),
                cache.decodes(),
                time.elapsed_secs_f64()
            );
        }
        if demo.loaded < demo.scenes.len() {
            commands.spawn(WorldAssetRoot(load_scene(
                &assets,
                &demo.scenes[demo.loaded],
            )));
        }
        demo.settle_until = time.elapsed_secs_f64() + 3.0;
    }
}

fn sun_direction(hour: f32) -> Vec3 {
    let angle = (hour.rem_euclid(24.0) - 6.0) * std::f32::consts::TAU / 24.0;
    Vec3::new(angle.cos(), 0.9 * angle.sin(), 0.4 * angle.sin()).normalize()
}

fn leaf_mask() -> Image {
    let mut pixels = Vec::with_capacity(64 * 64 * 4);
    for y in 0..64 {
        for x in 0..64 {
            let uv = Vec2::new((x as f32 + 0.5) / 64.0, (y as f32 + 0.5) / 64.0);
            let p = (uv - Vec2::splat(0.5)) * Vec2::new(2.6, 2.1);
            let opaque = p.length_squared() < 1.0
                && (uv.x - 0.5).abs() < (1.0 - (uv.y * 2.0 - 1.0).abs()) * 0.5;
            pixels.extend_from_slice(&[45, 120, 24, if opaque { 255 } else { 0 }]);
        }
    }
    Image::new(
        bevy::render::render_resource::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        bevy::render::render_resource::TextureDimension::D2,
        pixels,
        TextureFormat::Rgba8UnormSrgb,
        bevy::asset::RenderAssetUsages::default(),
    )
}

fn controls(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut demo: ResMut<Demo>,
    mut cameras: Query<&mut RadianceCascades>,
    mut exit: MessageWriter<AppExit>,
) {
    if demo.capture || demo.benchmark {
        return;
    }
    if keys.just_pressed(KeyCode::Escape) {
        exit.write(AppExit::Success);
    }
    if keys.just_pressed(KeyCode::Space) {
        demo.playing = !demo.playing;
    }
    let mut jump = false;
    for (key, hour) in [
        (KeyCode::Digit1, 6.0),
        (KeyCode::Digit2, 12.0),
        (KeyCode::Digit3, 18.0),
        (KeyCode::Digit4, 0.0),
    ] {
        if keys.just_pressed(key) {
            demo.hour = hour;
            jump = true;
        }
    }
    let scrub = f32::from(keys.pressed(KeyCode::BracketRight))
        - f32::from(keys.pressed(KeyCode::BracketLeft));
    demo.hour = (demo.hour
        + time.delta_secs() * (scrub * 2.0 + if demo.playing { 0.1 } else { 0.0 }))
    .rem_euclid(24.0);
    // Hardware and software traversal produce the same transport, so the
    // toggle keeps all GI history.
    if keys.just_pressed(KeyCode::KeyH) {
        demo.hardware_ray_queries = !demo.hardware_ray_queries;
    }
    for mut gi in &mut cameras {
        if gi.hardware_ray_queries != demo.hardware_ray_queries {
            gi.hardware_ray_queries = demo.hardware_ray_queries;
        }
        if keys.just_pressed(KeyCode::KeyG) {
            gi.intensity = if gi.intensity > 0.0 { 0.0 } else { 1.0 };
            jump = true;
        }
        if jump {
            gi.reset_history();
        }
    }
}

fn update_debug_overlay(
    diagnostics: Res<DiagnosticsStore>,
    settings: Query<&RadianceCascades>,
    mut overlay: Query<&mut Text, With<DebugOverlay>>,
) {
    let Some(_) = settings.iter().next() else {
        return;
    };
    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|diagnostic| diagnostic.smoothed())
        .unwrap_or(0.0);
    for mut text in &mut overlay {
        text.0 = format!("FPS: {fps:.1}");
    }
}

fn record_first_gi_dispatch(
    demo: &mut Demo,
    dispatch: RadianceCascadesDispatch,
    dispatch_at: std::time::Instant,
    timestamp_source: &'static str,
) {
    let revisions = dispatch.revisions;
    if revisions.scene == 0
        || revisions.geometry == 0
        || revisions.lighting == 0
        || demo.first_gi_dispatch_at.is_some()
    {
        return;
    }
    demo.first_gi_dispatch_at = Some(dispatch_at);
    let startup_ms = dispatch_at
        .saturating_duration_since(demo.startup_started)
        .as_secs_f64()
        * 1_000.0;
    let scene_load_ms = demo.scene_ready_at.map(|ready| {
        ready
            .saturating_duration_since(demo.startup_started)
            .as_secs_f64()
            * 1_000.0
    });
    let scene_ready_to_dispatch_ms = demo
        .scene_ready_at
        .map(|ready| dispatch_at.saturating_duration_since(ready).as_secs_f64() * 1_000.0);
    info!(
        startup_to_first_gi_dispatch_ms = startup_ms,
        dispatch_timestamp_source = timestamp_source,
        scene_load_ms = ?scene_load_ms,
        scene_ready_to_dispatch_ms = ?scene_ready_to_dispatch_ms,
        "Sponza first GI dispatch observed"
    );
}

fn lighting(
    demo: Res<Demo>,
    mut sun: Query<(&mut DirectionalLight, &mut Transform), With<Sun>>,
    mut views: Query<&mut RadianceCascades>,
    mut clear: ResMut<ClearColor>,
    mut previous_hour: Local<Option<f32>>,
) {
    if *previous_hour == Some(demo.hour) {
        return;
    }
    *previous_hour = Some(demo.hour);
    let direction = sun_direction(demo.hour);
    let daylight = direction.y.max(0.0);
    for (mut light, mut transform) in &mut sun {
        light.illuminance = 100_000.0 * daylight;
        light.color = Color::linear_rgb(1.0, 0.55 + daylight * 0.4, 0.3 + daylight * 0.6);
        *transform = Transform::default().looking_to(-direction, Vec3::Y);
    }
    let sky = ((direction.y + 0.1) / 1.0).clamp(0.0, 1.0);
    for mut gi in &mut views {
        gi.environment = LinearRgba::new(
            500.0 * sky + 0.02,
            700.0 * sky + 0.03,
            1000.0 * sky + 0.06,
            1.0,
        );
    }
    // ClearColor bypasses PBR's exposure multiplication. Match the radiance
    // supplied to GI so visible sky is not an unexposed, clipped white field.
    let exposure = Exposure::default().exposure();
    clear.0 = Color::linear_rgb(
        (500.0 * sky + 0.02) * exposure,
        (700.0 * sky + 0.03) * exposure,
        (1000.0 * sky + 0.06) * exposure,
    );
}

#[derive(Default)]
struct PerformanceSamples {
    title_updated: f64,
    gi: Vec<f64>,
    trace: Vec<f64>,
    frame: Vec<f64>,
    frame_interval: Vec<f64>,
    main_schedule_cpu: Vec<f64>,
    render_schedule_cpu: Vec<f64>,
    frame_points: Vec<(bevy::platform::time::Instant, f64)>,
    trace_points: Vec<(bevy::platform::time::Instant, f64)>,
    frame_interval_points: Vec<(bevy::platform::time::Instant, f64)>,
    main_schedule_cpu_points: Vec<(bevy::platform::time::Instant, f64)>,
    render_schedule_cpu_points: Vec<(bevy::platform::time::Instant, f64)>,
    warmup_seen: usize,
    measurement_start: Option<bevy::platform::time::Instant>,
    measurement_end: Option<bevy::platform::time::Instant>,
    measurement_start_s: Option<f64>,
    measurement_end_s: Option<f64>,
    last_gi: Option<bevy::platform::time::Instant>,
    last_trace: Option<bevy::platform::time::Instant>,
    last_frame: Option<bevy::platform::time::Instant>,
    last_frame_interval: Option<bevy::platform::time::Instant>,
    last_main_schedule_cpu: Option<bevy::platform::time::Instant>,
    last_render_schedule_cpu: Option<bevy::platform::time::Instant>,
}

#[derive(Component)]
struct BenchmarkRawReadback;

#[derive(Clone, Default)]
struct RendererMetadata {
    name: String,
    vendor: u32,
    device: u32,
    device_type: String,
    driver: String,
    driver_info: String,
    backend: String,
    shader_path: String,
    timestamp_queries: bool,
    pipeline_cache: bool,
    passthrough_shaders: bool,
    max_buffer_size: u64,
    max_storage_buffer_binding_size: u64,
    max_texture_dimension_2d: u32,
}

#[derive(Resource, Clone, Default)]
struct SharedRendererMetadata(Arc<Mutex<Option<RendererMetadata>>>);

fn collect_renderer_metadata(
    shared: Res<SharedRendererMetadata>,
    adapter: Res<RenderAdapterInfo>,
    device: Res<RenderDevice>,
    status: Res<RadianceCascadesStatus>,
) {
    let Some(shader_path) = status.shader_backend() else {
        return;
    };
    let mut slot = shared.0.lock().expect("renderer metadata lock poisoned");
    if slot.is_some() {
        return;
    }
    let features = device.features();
    let limits = device.limits();
    let passthrough_shaders = features.contains(wgpu::Features::PASSTHROUGH_SHADERS);
    *slot = Some(RendererMetadata {
        name: adapter.name.clone(),
        vendor: adapter.vendor,
        device: adapter.device,
        device_type: format!("{:?}", adapter.device_type),
        driver: adapter.driver.clone(),
        driver_info: adapter.driver_info.clone(),
        backend: format!("{:?}", adapter.backend),
        shader_path: shader_path.into(),
        timestamp_queries: features.contains(wgpu::Features::TIMESTAMP_QUERY),
        pipeline_cache: features.contains(wgpu::Features::PIPELINE_CACHE),
        passthrough_shaders,
        max_buffer_size: limits.max_buffer_size,
        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
        max_texture_dimension_2d: limits.max_texture_dimension_2d,
    });
}

#[allow(clippy::too_many_arguments)]
fn performance(
    mut demo: ResMut<Demo>,
    status: Res<RadianceCascadesStatus>,
    time: Res<Time<Real>>,
    diagnostics: Res<DiagnosticsStore>,
    renderer_metadata: Option<Res<SharedRendererMetadata>>,
    mut windows: Query<&mut Window>,
    mut state: Local<PerformanceSamples>,
    mut exit: MessageWriter<AppExit>,
) {
    let now = time.elapsed_secs_f64();
    let latest_dispatch = status.last_dispatch();
    if demo.scene_ready_at.is_none() && demo.loaded == demo.scenes.len() {
        demo.scene_ready_at = Some(std::time::Instant::now());
    }
    let raw_readback_enabled = demo.capture || (demo.benchmark && !demo.benchmark_no_gi);
    if !raw_readback_enabled
        && demo.first_gi_dispatch_at.is_none()
        && let Some(dispatch) = latest_dispatch
    {
        record_first_gi_dispatch(
            &mut demo,
            dispatch,
            std::time::Instant::now(),
            "status_observation",
        );
    }
    if now >= state.title_updated + 0.5 {
        state.title_updated = now;
        let fps = diagnostics
            .get(&FrameTimeDiagnosticsPlugin::FPS)
            .and_then(|d| d.smoothed());
        for mut window in &mut windows {
            window.title = format!(
                "Sponza | {:.1} FPS | {:02}:{:02} {} | rays: {} | {CONTROLS}",
                fps.unwrap_or(0.0),
                demo.hour as u32,
                ((demo.hour.fract() * 60.0) as u32).min(59),
                if demo.playing { "playing" } else { "paused" },
                match status.hardware_ray_queries() {
                    HardwareRayQueries::Active => "hardware",
                    HardwareRayQueries::Inactive if demo.hardware_ray_queries => {
                        "software (hardware pending)"
                    }
                    HardwareRayQueries::Inactive => "software",
                    HardwareRayQueries::Unsupported => "software (no hardware support)",
                }
            );
        }
    }
    // Offscreen benchmarks report GI GPU timings and separate frame-time
    // diagnostics. Readback confirms the full scene has produced fresh GI.
    if demo.benchmark
        && (demo.benchmark_no_gi || demo.gi_ready)
        && !status.compilation_in_flight()
        && demo.loaded == demo.scenes.len()
        && now > demo.settle_until + 3.0
        && (demo.benchmark_no_gi
            || diagnostics.iter().any(|d| {
                d.path().as_str().contains("split_rc")
                    && d.path().as_str().ends_with("elapsed_gpu")
                    && d.value().is_some()
            }))
    {
        if !demo.benchmark_no_gi
            && let Some(measurement) = diagnostics
                .iter()
                .find(|d| d.path().as_str() == "render/split_rc/elapsed_gpu")
                .and_then(|d| d.measurement())
            && state.last_gi != Some(measurement.time)
        {
            state.last_gi = Some(measurement.time);
            if state.gi.len() < demo.warmup_frames {
                state.gi.push(measurement.value);
                state.warmup_seen += 1;
            } else {
                if state.measurement_start.is_none() {
                    state.measurement_start = Some(measurement.time);
                    state.measurement_start_s = Some(now);
                }
                state.gi.push(measurement.value);
                state.measurement_end = Some(measurement.time);
                state.measurement_end_s = Some(now);
            }
        }
        if let Some(measurement) = diagnostics
            .iter()
            .find(|d| d.path().as_str() == "render/split_rc/trace/elapsed_gpu")
            .and_then(|d| d.measurement())
            && state.last_trace != Some(measurement.time)
        {
            state.last_trace = Some(measurement.time);
            if state.measurement_start.is_some() {
                state
                    .trace_points
                    .push((measurement.time, measurement.value));
            }
        }
        if let Some(measurement) = diagnostics
            .iter()
            .find(|d| d.path().as_str() == "render/penumbra_frame/elapsed_gpu")
            .and_then(|d| d.measurement())
            && state.last_frame != Some(measurement.time)
        {
            state.last_frame = Some(measurement.time);
            if demo.benchmark_no_gi {
                if state.warmup_seen < demo.warmup_frames {
                    state.warmup_seen += 1;
                } else {
                    if state.measurement_start.is_none() {
                        state.measurement_start = Some(measurement.time);
                        state.measurement_start_s = Some(now);
                    }
                    state.frame.push(measurement.value);
                    state.measurement_end = Some(measurement.time);
                    state.measurement_end_s = Some(now);
                }
            } else if state.measurement_start.is_some() {
                state
                    .frame_points
                    .push((measurement.time, measurement.value));
            }
        }
        if let Some(measurement) = diagnostics
            .iter()
            .find(|d| d.path().as_str() == "frame_time")
            .and_then(|d| d.measurement())
            && state.last_frame_interval != Some(measurement.time)
        {
            state.last_frame_interval = Some(measurement.time);
            if state.measurement_start.is_some() {
                state
                    .frame_interval_points
                    .push((measurement.time, measurement.value));
            }
        }
        if let Some(measurement) = diagnostics
            .iter()
            .find(|d| d.path().as_str() == "render/penumbra_frame/main_schedule_cpu")
            .and_then(|d| d.measurement())
            && state.last_main_schedule_cpu != Some(measurement.time)
        {
            state.last_main_schedule_cpu = Some(measurement.time);
            if state.measurement_start.is_some() {
                state
                    .main_schedule_cpu_points
                    .push((measurement.time, measurement.value));
            }
        }
        if let Some(measurement) = diagnostics
            .iter()
            .find(|d| d.path().as_str() == "render/penumbra_frame/render_schedule_cpu")
            .and_then(|d| d.measurement())
            && state.last_render_schedule_cpu != Some(measurement.time)
        {
            state.last_render_schedule_cpu = Some(measurement.time);
            if state.measurement_start.is_some() {
                state
                    .render_schedule_cpu_points
                    .push((measurement.time, measurement.value));
            }
        }
        let measurement_complete = if demo.benchmark_no_gi {
            state.frame.len() >= demo.measured_frames
        } else {
            state.gi.len() >= demo.warmup_frames + demo.measured_frames
        };
        if measurement_complete {
            let start = state.measurement_start.expect("measured GI window starts");
            let end = state.measurement_end.expect("measured GI window ends");
            // GPU and CPU diagnostics arrive independently. Extend the same
            // time window until each stream has the requested sample count.
            let enough = |points: &[(bevy::platform::time::Instant, f64)]| {
                points
                    .iter()
                    .filter(|(timestamp, _)| *timestamp >= start && *timestamp <= end)
                    .count()
                    >= demo.measured_frames
            };
            if (!demo.benchmark_no_gi && !enough(&state.frame_points))
                || !enough(&state.frame_interval_points)
                || !enough(&state.main_schedule_cpu_points)
                || !enough(&state.render_schedule_cpu_points)
            {
                if now > 600.0 {
                    error!("Sponza benchmark timed out waiting for aligned timing samples");
                    exit.write(AppExit::error());
                }
                return;
            }
            if !demo.benchmark_no_gi {
                state.trace = state
                    .trace_points
                    .iter()
                    .filter_map(|(timestamp, value)| {
                        (*timestamp >= start && *timestamp <= end).then_some(*value)
                    })
                    .collect();
                state.frame = state
                    .frame_points
                    .iter()
                    .filter_map(|(timestamp, value)| {
                        (*timestamp >= start && *timestamp <= end).then_some(*value)
                    })
                    .collect();
                state.gi.drain(..demo.warmup_frames);
            }
            state.frame_interval = state
                .frame_interval_points
                .iter()
                .filter_map(|(timestamp, value)| {
                    (*timestamp >= start && *timestamp <= end).then_some(*value)
                })
                .collect();
            state.main_schedule_cpu = state
                .main_schedule_cpu_points
                .iter()
                .filter_map(|(timestamp, value)| {
                    (*timestamp >= start && *timestamp <= end).then_some(*value)
                })
                .collect();
            state.render_schedule_cpu = state
                .render_schedule_cpu_points
                .iter()
                .filter_map(|(timestamp, value)| {
                    (*timestamp >= start && *timestamp <= end).then_some(*value)
                })
                .collect();
            let measured = if demo.benchmark_no_gi {
                &mut state.frame
            } else {
                &mut state.gi
            };
            measured.sort_by(f64::total_cmp);
            let median = measured[measured.len() / 2];
            let metadata = renderer_metadata
                .as_deref()
                .and_then(|shared| shared.0.lock().ok()?.clone());
            let traversal = match status.hardware_ray_queries() {
                HardwareRayQueries::Active => "hardware",
                HardwareRayQueries::Inactive | HardwareRayQueries::Unsupported => "software",
            };
            let json = benchmark_json(&demo, &state, &diagnostics, metadata.as_ref(), traversal);
            if let Some(parent) = Path::new(&demo.benchmark_output).parent()
                && !parent.as_os_str().is_empty()
                && let Err(error) = std::fs::create_dir_all(parent)
            {
                error!("Could not create benchmark output directory: {error}");
                exit.write(AppExit::error());
                return;
            }
            if let Err(error) = std::fs::write(&demo.benchmark_output, &json) {
                error!("Could not write benchmark JSON: {error}");
                exit.write(AppExit::error());
                return;
            }
            println!("{json}");
            println!(
                "Sponza Split {}x{}: {} measured {} GPU samples after {} warmup samples, median {median:.3} ms, JSON {}",
                demo.capture_size.0,
                demo.capture_size.1,
                demo.measured_frames,
                if demo.benchmark_no_gi { "frame" } else { "GI" },
                demo.warmup_frames,
                demo.benchmark_output
            );
            exit.write(AppExit::Success);
        }
    }
    if demo.benchmark && now > 600.0 {
        error!("Sponza benchmark timed out waiting for GI");
        exit.write(AppExit::error());
    }
}

fn json_string(value: &str) -> String {
    let mut output = String::with_capacity(value.len() + 2);
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            value if value.is_control() => output.push_str(&format!("\\u{:04x}", value as u32)),
            value => output.push(value),
        }
    }
    output.push('"');
    output
}

fn timing_json(samples: &[f64]) -> String {
    if samples.is_empty() {
        return "{\"samples\":0,\"mean_ms\":null,\"p50_ms\":null,\"p95_ms\":null}".into();
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let median = sorted[sorted.len() / 2];
    let p95 = sorted[(sorted.len() * 95 / 100).min(sorted.len() - 1)];
    format!(
        "{{\"samples\":{},\"mean_ms\":{mean:.6},\"p50_ms\":{median:.6},\"p95_ms\":{p95:.6},\"min_ms\":{:.6},\"max_ms\":{:.6}}}",
        sorted.len(),
        sorted[0],
        sorted[sorted.len() - 1]
    )
}

fn diagnostics_json(diagnostics: &DiagnosticsStore) -> String {
    let mut values = diagnostics
        .iter()
        .filter_map(|diagnostic| {
            let name = diagnostic.path().as_str();
            (name.starts_with("penumbra/")
                || name.starts_with("render/penumbra/")
                || (name.starts_with("render/") && name.ends_with("elapsed_gpu")))
            .then(|| diagnostic.value().map(|value| (name.to_owned(), value)))
            .flatten()
        })
        .collect::<Vec<_>>();
    values.sort_by(|a, b| a.0.cmp(&b.0));
    format!(
        "{{{}}}",
        values
            .into_iter()
            .map(|(name, value)| format!("{}:{value:.6}", json_string(&name)))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn renderer_json(renderer: Option<&RendererMetadata>) -> String {
    let Some(renderer) = renderer else {
        return "null".into();
    };
    format!(
        "{{\"name\":{},\"vendor_id\":{},\"device_id\":{},\"device_type\":{},\"driver\":{},\"driver_info\":{},\"backend\":{},\"shader_path\":{},\"shader_path_is_inferred\":false,\"shader_path_basis\":\"RenderStartup backend selection\",\"shader_output_actual\":{},\"features\":{{\"timestamp_queries\":{},\"pipeline_cache\":{},\"passthrough_shaders\":{}}},\"limits\":{{\"max_buffer_size\":{},\"max_storage_buffer_binding_size\":{},\"max_texture_dimension_2d\":{}}}}}",
        json_string(&renderer.name),
        renderer.vendor,
        renderer.device,
        json_string(&renderer.device_type),
        json_string(&renderer.driver),
        json_string(&renderer.driver_info),
        json_string(&renderer.backend),
        json_string(&renderer.shader_path),
        json_string(&renderer.shader_path),
        renderer.timestamp_queries,
        renderer.pipeline_cache,
        renderer.passthrough_shaders,
        renderer.max_buffer_size,
        renderer.max_storage_buffer_binding_size,
        renderer.max_texture_dimension_2d,
    )
}

fn benchmark_json(
    demo: &Demo,
    samples: &PerformanceSamples,
    diagnostics: &DiagnosticsStore,
    renderer: Option<&RendererMetadata>,
    traversal: &str,
) -> String {
    let backend_requested = std::env::var("WGPU_BACKEND").unwrap_or_default();
    format!(
<<<<<<< Updated upstream
        "{{\"schema_version\":13,\"crate_version\":{},\"os\":{},\"architecture\":{},\"requested_backend\":{},\"renderer\":{},\"benchmark_mode\":{},\"present_mode_requested\":{},\"gi_enabled\":{},\"scene\":{{\"name\":\"sponza_courtyard_curtains_ivy\",\"width\":{},\"height\":{},\"shader_path\":{}}},\"transport\":\"Split\",\"settings\":{{\"multi_bounce\":{},\"grid_divisor\":{},\"updates_per_frame\":{},\"max_traversal_queries\":{},\"emissive_samples\":{SPONZA_LIGHT_SAMPLES},\"resolve_scale\":{},\"geometry_upload_bytes_per_frame\":{SPONZA_GEOMETRY_UPLOAD_BYTES_PER_FRAME},\"max_receiver_cells\":{SPONZA_SPLIT_MAX_RECEIVER_CELLS},\"max_cache_entries\":{SPONZA_SPLIT_MAX_CACHE_ENTRIES},\"probe_spacing_m\":{SPONZA_PROBE_SPACING_METERS},\"interval_length_m\":{SPONZA_INTERVAL_LENGTH_METERS},\"history_frames\":64,\"temporal_blend\":0.985,\"scene_gpu_budget_bytes\":{SCENE_GPU_BUDGET},\"atlas_budget_bytes\":{ATLAS_GPU_BUDGET},\"view_gpu_budget_bytes\":{VIEW_GPU_BUDGET}}},\"quality\":{{\"supported_pixels\":{},\"eligible_pixels\":{},\"support_ratio\":{},\"total_pixels\":{},\"frame_id\":{},\"dispatch_sequence\":{},\"scene_revision\":{},\"geometry_revision\":{},\"lighting_revision\":{}}},\"samples\":{{\"warmup_requested\":{},\"warmup_observed\":{},\"measured_gpu_requested\":{},\"measured_interval_s\":{{\"start\":{},\"end\":{}}},\"gi_gpu_ms\":{},\"trace_gpu_ms\":{},\"penumbra_gpu_frame_ms\":{},\"frame_interval_diagnostic_ms\":{},\"main_schedule_cpu_ms\":{},\"render_schedule_cpu_ms\":{}}},\"diagnostics\":{}}}\n",
=======
        "{{\"schema_version\":14,\"crate_version\":{},\"os\":{},\"architecture\":{},\"requested_backend\":{},\"renderer\":{},\"benchmark_mode\":{},\"present_mode_requested\":{},\"gi_enabled\":{},\"scene\":{{\"name\":\"sponza_courtyard_curtains_ivy\",\"width\":{},\"height\":{},\"shader_path\":{},\"asset_profile\":{},\"asset_paths\":[{asset_paths}]}},\"transport\":\"Split\",\"settings\":{{\"shadow_profile\":{},\"multi_bounce\":{},\"cached_hit_sky\":{},\"hardware_ray_queries\":{},\"traversal\":{},\"grid_divisor\":{},\"updates_per_frame\":{},\"max_traversal_queries\":{},\"emissive_samples\":{SPONZA_LIGHT_SAMPLES},\"resolve_scale\":{},\"geometry_upload_bytes_per_frame\":{SPONZA_GEOMETRY_UPLOAD_BYTES_PER_FRAME},\"max_receiver_cells\":{SPONZA_SPLIT_MAX_RECEIVER_CELLS},\"max_cache_entries\":{SPONZA_SPLIT_MAX_CACHE_ENTRIES},\"probe_spacing_m\":{SPONZA_PROBE_SPACING_METERS},\"interval_length_m\":{SPONZA_INTERVAL_LENGTH_METERS},\"history_frames\":64,\"temporal_blend\":0.985,\"scene_gpu_budget_bytes\":{SCENE_GPU_BUDGET},\"atlas_budget_bytes\":{ATLAS_GPU_BUDGET},\"view_gpu_budget_bytes\":{VIEW_GPU_BUDGET}}},\"quality\":{{\"supported_pixels\":{},\"eligible_pixels\":{},\"support_ratio\":{},\"total_pixels\":{},\"frame_id\":{},\"dispatch_sequence\":{},\"scene_revision\":{},\"geometry_revision\":{},\"lighting_revision\":{}}},\"samples\":{{\"warmup_requested\":{},\"warmup_observed\":{},\"measured_gpu_requested\":{},\"measured_interval_s\":{{\"start\":{},\"end\":{}}},\"gi_gpu_ms\":{},\"trace_gpu_ms\":{},\"penumbra_gpu_frame_ms\":{},\"frame_interval_diagnostic_ms\":{},\"main_schedule_cpu_ms\":{},\"render_schedule_cpu_ms\":{},\"gpu_passes\":{{{gpu_passes}}}}},\"diagnostics\":{}}}\n",
>>>>>>> Stashed changes
        json_string(env!("CARGO_PKG_VERSION")),
        json_string(std::env::consts::OS),
        json_string(std::env::consts::ARCH),
        json_string(&backend_requested),
        renderer_json(renderer),
        json_string(if demo.benchmark_windowed {
            "windowed_pipelined"
        } else {
            "offscreen_drained"
        }),
        json_string(if !demo.benchmark_windowed {
            "none"
        } else if demo.benchmark_vsync {
            "AutoVsync"
        } else {
            "AutoNoVsync"
        }),
        !demo.benchmark_no_gi,
        demo.capture_size.0,
        demo.capture_size.1,
        renderer
            .map(|metadata| json_string(&metadata.shader_path))
            .unwrap_or_else(|| "null".into()),
        demo.multi_bounce,
        demo.cached_hit_sky,
        demo.hardware_ray_queries,
        json_string(traversal),
        demo.resolution_divisor,
        demo.max_rays,
        demo.max_traversal_queries,
        SPONZA_RESOLVE_SCALE,
        demo.raw_gi.supported_pixels,
        demo.raw_gi.eligible_pixels,
        demo.raw_gi.supported_pixels as f32 / demo.raw_gi.eligible_pixels.max(1) as f32,
        demo.raw_gi.total_pixels,
        demo.raw_gi.frame_id,
        demo.raw_gi.dispatch_sequence,
        demo.raw_gi.scene_revision,
        demo.raw_gi.geometry_revision,
        demo.raw_gi.light_revision,
        demo.warmup_frames,
        samples.warmup_seen,
        demo.measured_frames,
        samples.measurement_start_s.unwrap_or(0.0),
        samples.measurement_end_s.unwrap_or(0.0),
        timing_json(&samples.gi),
        timing_json(&samples.trace),
        timing_json(&samples.frame),
        timing_json(&samples.frame_interval),
        timing_json(&samples.main_schedule_cpu),
        timing_json(&samples.render_schedule_cpu),
        diagnostics_json(diagnostics),
    )
}

fn capture_frame(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time<Real>>,
    diagnostics: Res<DiagnosticsStore>,
    mut demo: ResMut<Demo>,
    sun_raw: Option<Res<SunCaptureRaw>>,
    mut exit: MessageWriter<AppExit>,
) {
    if demo.benchmark || demo.motion_capture || demo.stability_capture || sun_raw.is_some() {
        return;
    }
    if !demo.capture {
        if keys.just_pressed(KeyCode::KeyP) {
            commands
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk("screenshots/sponza.png"));
        }
        return;
    }
    if time.elapsed_secs_f64() > 600.0 {
        error!(
            loaded_scenes = demo.loaded,
            expected_scenes = demo.scenes.len(),
            gi_frames = demo.gi_frames,
            raw_sample = demo.raw_gi.sample_count,
            eligible_pixels = demo.raw_gi.eligible_pixels,
            supported_pixels = demo.raw_gi.supported_pixels,
            scene_revision = demo.raw_gi.scene_revision,
            geometry_revision = demo.raw_gi.geometry_revision,
            light_revision = demo.raw_gi.light_revision,
            "Sponza capture timed out before valid GI and scene readiness"
        );
        exit.write(AppExit::error());
        return;
    }
    if demo.requested
        || demo.loaded != demo.scenes.len()
        || (demo.capture_stage != 1 && !demo.gi_ready)
        || time.elapsed_secs_f64() < demo.settle_until
    {
        return;
    }
    demo.requested = true;
    if demo.capture_stage == 0 {
        for diagnostic in diagnostics.iter().filter(|d| {
            d.path().as_str().contains("/penumbra/") || d.path().as_str().ends_with("elapsed_gpu")
        }) {
            if let Some(value) = diagnostic.value() {
                info!("{}: {value}", diagnostic.path());
            }
        }
    }
    let stage = ["day", "no_gi", "night"][demo.capture_stage];
    let path = demo.capture_prefix.as_ref().map_or_else(
        || format!("screenshots/sponza_{stage}.png"),
        |prefix| format!("{prefix}_{stage}.png"),
    );
    commands
        .spawn(Screenshot::image(demo.target.clone().unwrap()))
        .observe(
            move |event: On<ScreenshotCaptured>,
                  mut demo: ResMut<Demo>,
                  mut views: Query<&mut RadianceCascades>,
                  time: Res<Time<Real>>,
                  mut exit: MessageWriter<AppExit>| {
                let image = event.image.clone();
                save_to_disk(&path)(event);
                let data = image.data.as_ref().unwrap();
                let pixels = data.as_chunks::<4>().0;
                let mean = pixels
                    .iter()
                    .map(|p| (f32::from(p[0]) + f32::from(p[1]) + f32::from(p[2])) / 3.0)
                    .sum::<f32>()
                    / pixels.len() as f32;
                if demo.capture_stage == 0 {
                    assert!(mean > 4.0, "Sunny Sponza capture is dark: {mean}");
                    demo.day_mean = mean;
                    demo.day_pixels = data.clone();
                } else if demo.capture_stage == 1 {
                    let change = demo
                        .day_pixels
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .zip(pixels)
                        .map(|(on, off)| {
                            (0..3)
                                .map(|i| (f32::from(on[i]) - f32::from(off[i])).abs())
                                .sum::<f32>()
                        })
                        .sum::<f32>()
                        / (pixels.len() * 3) as f32;
                    assert!(
                        change > 0.2,
                        "GI toggle did not change Sponza lighting: {change}"
                    );
                    info!("Sponza GI on/off RGB difference: {change:.3}/255");
                    demo.day_pixels.clear();
                } else {
                    assert!(
                        mean < demo.day_mean * 0.2,
                        "Changing to midnight did not darken the scene: {mean} vs {}",
                        demo.day_mean
                    );
                }
                info!(
                    "Sponza capture stage {} passed at {:.2}h, RGB mean {mean:.3}",
                    demo.capture_stage, demo.hour
                );
                if demo.capture_stage == 2 {
                    exit.write(AppExit::Success);
                } else {
                    demo.capture_stage += 1;
                    if demo.capture_stage == 2 {
                        demo.hour = 0.0;
                    }
                    demo.requested = false;
                    demo.gi_frames = 0;
                    demo.gi_ready = false;
                    demo.settle_until = time.elapsed_secs_f64() + 3.0;
                    for mut gi in &mut views {
                        gi.intensity = if demo.capture_stage == 1 { 0.0 } else { 1.0 };
                        gi.reset_history();
                    }
                }
            },
        );
}

fn motion_capture_pose(frame: u32) -> Transform {
    let start = Vec3::new(-8.80743, 1.59222, -0.85826);
    let target = Vec3::new(-3.00422, 2.76208, 0.23737);
    let forward = (target - start).normalize();
    let pan = Vec3::new(0.8, 0.0, 1.2);
    let strafe = Vec3::new(1.5, 0.0, 0.0);
    let teleport = Vec3::new(-3.0, 0.0, 2.5);
    match frame {
        0..=32 => {
            let t = frame as f32 / 32.0;
            let position = start + pan * t;
            Transform::from_translation(position).looking_at(target, Vec3::Y)
        }
        33..=64 => {
            let t = (frame - 32) as f32 / 32.0;
            let delta = strafe * t;
            Transform::from_translation(start + pan + delta).looking_at(target + delta, Vec3::Y)
        }
        65..=96 => {
            let t = (frame - 64) as f32 / 32.0;
            let position = start + pan + strafe;
            let turned = Quat::from_rotation_y(t * std::f32::consts::PI * 1.5) * forward;
            Transform::from_translation(position).looking_to(turned, Vec3::Y)
        }
        _ => {
            let t = ((frame.min(127).saturating_sub(97)) as f32 / 30.0).clamp(0.0, 1.0);
            let position = start + teleport + Vec3::new(0.45, 0.1, -0.35) * t;
            let look_target = target + teleport + Vec3::new(0.45, 0.1, -0.35) * t;
            Transform::from_translation(position).looking_at(look_target, Vec3::Y)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct MotionCapturePositions {
    occluder: Vec3,
    point_light: Vec3,
    spot_light: Vec3,
}

fn motion_capture_positions(frame: u32) -> MotionCapturePositions {
    let frame = frame.min(127);
    let phase = frame as f32 * 0.055;
    let occluder = Vec3::new(
        -5.0 + phase.cos() * 1.8,
        1.2 + phase.sin() * 0.15,
        0.5 + phase.sin() * 1.4,
    );
    let light_phase = frame as f32 * 0.047;
    let point_light = Vec3::new(
        -5.0 + light_phase.cos() * 2.1,
        2.5 + light_phase.sin() * 0.3,
        0.5 + light_phase.sin() * 1.8,
    );
    MotionCapturePositions {
        occluder,
        point_light,
        spot_light: point_light + Vec3::new(1.4, 0.4, -0.6),
    }
}

fn motion_light_state_needs_emissive_update(previous: &mut Option<usize>, current: usize) -> bool {
    previous
        .replace(current)
        .is_some_and(|previous| previous != current)
}

fn camera_motion_capture_pose(frame: u32) -> Transform {
    let start = motion_capture_pose(0);
    if (65..=128).contains(&frame) {
        return start.with_rotation(start.rotation * Quat::from_rotation_y(std::f32::consts::PI));
    }
    let pan_fraction = match frame {
        0..=16 => frame as f32 / 16.0,
        17..=32 => (32 - frame) as f32 / 16.0,
        _ => 0.0,
    };
    Transform::from_translation(start.translation + Vec3::new(0.35, 0.0, 0.5) * pan_fraction)
        .with_rotation(Quat::from_rotation_y(pan_fraction * 0.12) * start.rotation)
}

fn motion_light_state(frame: u32) -> usize {
    match frame {
        0..=31 => 0,
        32..=63 => 1,
        64..=95 => 2,
        96..=159 => 0,
        160..=256 => 3,
        _ => 4,
    }
}

fn motion_capture_can_advance(
    frame: u32,
    gi_ready: bool,
    camera_only: bool,
    latest_sample: Option<&RawGiSample>,
    reset_settle: &MotionSettleObservation,
    light_change: Option<&MotionLightChange>,
) -> bool {
    if frame == 0 {
        return gi_ready;
    }
    if camera_only {
        return true;
    }
    let Some(sample) = latest_sample else {
        return !matches!(frame, 129 | 161 | 258);
    };
    match frame {
        129 => reset_settle.encoded_samples(sample.dispatch.sequence) >= MOTION_SETTLE_DISPATCHES,
        161 | 258 => light_change.is_some_and(|change| {
            change.revision == sample.dispatch.revisions.lighting
                && change.settle.encoded_samples(sample.dispatch.sequence)
                    >= MOTION_SETTLE_DISPATCHES
                && sample
                    .submitted_at
                    .checked_duration_since(change.encoded_started_at)
                    .is_some_and(|elapsed| elapsed >= std::time::Duration::from_millis(250))
        }),
        _ => true,
    }
}

fn pending_motion_sample_ready(pending: &PendingMotionCapture, sample: &RawGiSample) -> bool {
    let revisions = sample.dispatch.revisions;
    sample.dispatch.sequence >= pending.minimum_dispatch_sequence
        && pending.previous_geometry_revision != Some(revisions.geometry)
        && pending.previous_lighting_revision != Some(revisions.lighting)
}

// Motion advances once per rendered app frame, not according to wall-clock time.
// Readbacks carry the dispatch revisions that produced their raw pixels.
#[allow(clippy::too_many_arguments)]
fn capture_moving_sun(
    mut commands: Commands,
    mut demo: ResMut<Demo>,
    raw: Option<Res<SunCaptureRaw>>,
    status: Res<RadianceCascadesStatus>,
    diagnostics: Res<DiagnosticsStore>,
    time: Res<Time<Real>>,
    mut state: Local<SunCaptureState>,
    mut exit: MessageWriter<AppExit>,
) {
    let Some(raw) = raw else { return };
    if time.elapsed_secs_f64() > 600.0 {
        error!(
            stage = state.stage,
            frame = state.frame,
            "Moving-sun capture timed out"
        );
        exit.write(AppExit::error());
        return;
    }
    if demo.requested || demo.loaded != demo.scenes.len() {
        return;
    }

    if state.stage == 1 {
        if let Some(sample) = demo.latest_raw_sample.as_ref()
            && state
                .motion_previous
                .as_ref()
                .is_none_or(|previous| sample.dispatch.sequence > previous.dispatch.sequence)
        {
            if let Some(drift) = state
                .motion_previous
                .as_ref()
                .and_then(|previous| stationary_sample_drift(previous, sample))
            {
                state.motion_drift_sum += f64::from(drift);
                state.motion_drift_max = state.motion_drift_max.max(drift);
                state.motion_drift_count += 1;
            }
            state.motion_previous = Some(sample.clone());
        }
        state.frame += 1;
        demo.hour = (state.start_hour + SUN_HOURS_PER_FRAME * state.frame as f32).rem_euclid(24.0);
        if state.frame == SUN_MOTION_FRAMES {
            state.final_sequence_floor = status.last_dispatch().map_or(0, |d| d.sequence);
            state.stage = 2;
        }
        return;
    }

    let Some(sample) = demo.latest_raw_sample.as_ref() else {
        return;
    };
    if state.stage == 0
        && state
            .motion_previous
            .as_ref()
            .is_none_or(|previous| sample.dispatch.sequence > previous.dispatch.sequence)
    {
        state.warmup_frame_drift = state
            .motion_previous
            .as_ref()
            .filter(|previous| previous.dispatch.revisions == sample.dispatch.revisions)
            .and_then(|previous| stationary_sample_drift(previous, sample));
        state.motion_previous = Some(sample.clone());
    }
    let stage = match state.stage {
        0 if demo.gi_ready => {
            state.start_hour = demo.hour;
            state.baseline = Some(sample.clone());
            state.motion_previous = Some(sample.clone());
            state.stage = 1;
            "before"
        }
        2 => {
            let Some(dispatch) = status.last_dispatch() else {
                return;
            };
            if dispatch.revisions.lighting != state.final_revision {
                state.final_revision = dispatch.revisions.lighting;
                state.stable_revision_frames = 0;
            } else {
                state.stable_revision_frames += 1;
            }
            let baseline_revision = state
                .baseline
                .as_ref()
                .map(|s| s.dispatch.revisions.lighting);
            if state.stable_revision_frames < 3
                || dispatch.sequence < state.final_sequence_floor.saturating_add(3)
                || Some(dispatch.revisions.lighting) == baseline_revision
                || sample.dispatch.revisions.lighting != dispatch.revisions.lighting
                || sample.dispatch.sequence < state.final_sequence_floor.saturating_add(3)
            {
                return;
            }
            state.immediate = Some(sample.clone());
            state.stage = 3;
            "immediate"
        }
        3 => {
            let Some(immediate) = state.immediate.as_ref() else {
                return;
            };
            if sample.dispatch.revisions.lighting != immediate.dispatch.revisions.lighting {
                error!("Moving-sun final light revision changed while settling");
                exit.write(AppExit::error());
                return;
            }
            let elapsed = sample
                .dispatch
                .sequence
                .saturating_sub(immediate.dispatch.sequence);
            if elapsed >= MOTION_SETTLE_MIDPOINT_DISPATCHES && state.midpoint.is_none() {
                state.midpoint = Some(sample.clone());
            }
            if elapsed < MOTION_SETTLE_DISPATCHES {
                return;
            }
            state.stage = 4;
            "settled"
        }
        _ => return,
    };

    let prefix = demo
        .capture_prefix
        .as_deref()
        .unwrap_or("screenshots/sponza");
    let path = format!("{prefix}_moving_sun_{stage}");
    let raw_path = format!("{path}.rgba32f");
    if let Some(parent) = std::path::Path::new(&raw_path).parent()
        && !parent.as_os_str().is_empty()
        && let Err(error) = std::fs::create_dir_all(parent)
    {
        error!(%error, "Failed to create moving-sun capture directory");
        exit.write(AppExit::error());
        return;
    }
    if let Err(error) = std::fs::write(&raw_path, &raw.0) {
        error!(%error, %raw_path, "Failed to write moving-sun raw capture");
        exit.write(AppExit::error());
        return;
    }
    let (width, height) = demo.capture_size;
    let raw_width = width.div_ceil(SPONZA_RESOLVE_SCALE).max(1);
    let raw_height = height.div_ceil(SPONZA_RESOLVE_SCALE).max(1);
    info!(
        stage,
        motion_frames = state.frame,
        hour = demo.hour,
        raw_width,
        raw_height,
        raw_path,
        frame_id = sample.frame_id,
        dispatch_sequence = sample.dispatch.sequence,
        scene_revision = sample.dispatch.revisions.scene,
        geometry_revision = sample.dispatch.revisions.geometry,
        light_revision = sample.dispatch.revisions.lighting,
        support_ratio = ?motion_sample_support_ratio(sample),
        mean_luminance = demo.raw_gi.mean_luminance,
        supported_mean_luminance = demo.raw_gi.supported_mean_luminance,
        max_luminance = demo.raw_gi.max_luminance,
        before_to_current_drift = ?state.baseline.as_ref().and_then(|s| stationary_sample_drift(s, sample)),
        immediate_to_current_drift = ?state.immediate.as_ref().and_then(|s| stationary_sample_drift(s, sample)),
        midpoint_to_current_drift = ?state.midpoint.as_ref().and_then(|s| stationary_sample_drift(s, sample)),
        warmup_frame_drift = ?state.warmup_frame_drift,
        motion_drift_count = state.motion_drift_count,
        motion_drift_mean = ?(state.motion_drift_count > 0).then(|| state.motion_drift_sum / f64::from(state.motion_drift_count)),
        motion_drift_max = state.motion_drift_max,
        diagnostics = %diagnostics_json(&diagnostics),
        "Sponza moving-sun raw capture"
    );
    demo.requested = true;
    let png_path = format!("{path}.png");
    commands
        .spawn(Screenshot::image(
            demo.target.clone().expect("capture target"),
        ))
        .observe(
            move |event: On<ScreenshotCaptured>,
                  mut demo: ResMut<Demo>,
                  mut exit: MessageWriter<AppExit>| {
                save_to_disk(&png_path)(event);
                demo.requested = false;
                if stage == "settled" {
                    exit.write(AppExit::Success);
                }
            },
        );
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn capture_motion(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    time: Res<Time<Real>>,
    status: Res<RadianceCascadesStatus>,
    mut demo: ResMut<Demo>,
    mut camera: Query<
        (&mut Transform, &GlobalTransform, &mut RadianceCascades),
        (
            Without<MotionCaptureOccluder>,
            Without<MotionCaptureLight>,
            Without<MotionCaptureSpot>,
        ),
    >,
    mut occluders: Query<&mut Transform, With<MotionCaptureOccluder>>,
    mut moving_lights: Query<
        (&mut Transform, &mut PointLight),
        (With<MotionCaptureLight>, Without<MotionCaptureOccluder>),
    >,
    mut moving_spots: Query<
        (&mut Transform, &mut SpotLight),
        (
            With<MotionCaptureSpot>,
            Without<MotionCaptureOccluder>,
            Without<MotionCaptureLight>,
        ),
    >,
    emissive_emitters: Query<&MeshMaterial3d<StandardMaterial>, With<MotionCaptureEmissive>>,
    mut previous_light_state: Local<Option<usize>>,
    mut exit: MessageWriter<AppExit>,
) {
    if !demo.motion_capture {
        return;
    }
    if time.elapsed_secs_f64() > 600.0 {
        error!(
            loaded_scenes = demo.loaded,
            expected_scenes = demo.scenes.len(),
            gi_frames = demo.gi_frames,
            raw_sample = demo.raw_gi.sample_count,
            eligible_pixels = demo.raw_gi.eligible_pixels,
            supported_pixels = demo.raw_gi.supported_pixels,
            scene_revision = demo.raw_gi.scene_revision,
            geometry_revision = demo.raw_gi.geometry_revision,
            light_revision = demo.raw_gi.light_revision,
            "Sponza motion capture timed out"
        );
        exit.write(AppExit::error());
        return;
    }
    if demo.loaded != demo.scenes.len() || demo.requested {
        return;
    }
    let camera_only = demo.camera_motion_capture;
    let cache_revisit = camera_only;
    let pending_capture = demo.motion_capture_pending.clone();
    let (frame, camera_transform, history_epoch, occluder_position, raw_sample) =
        if let Some(pending) = pending_capture {
            let Some(raw_sample) = demo
                .latest_raw_sample
                .as_ref()
                .filter(|sample| pending_motion_sample_ready(&pending, sample))
                .cloned()
            else {
                return;
            };
            demo.motion_capture_pending = None;
            (
                pending.frame,
                pending.camera_transform,
                pending.history_epoch,
                pending.occluder_position,
                raw_sample,
            )
        } else {
            if !motion_capture_can_advance(
                demo.motion_frame,
                demo.gi_ready,
                camera_only,
                demo.latest_raw_sample.as_ref(),
                &demo.motion_reset_settle,
                demo.motion_light_change.as_ref(),
            ) {
                return;
            }
            let frame = demo.motion_frame;
            let capture_this_frame = if camera_only {
                matches!(frame, 0 | 16 | 32 | 64)
                    || (cache_revisit && matches!(frame, 80 | 128 | 129 | 160))
            } else {
                (frame <= 120 && frame.is_multiple_of(8))
                    || matches!(
                        frame,
                        127 | 128 | 159 | 160 | 166 | 192 | 224 | 256 | 257 | 263 | 289
                    )
            };
            assert_eq!(
                demo.raw_gi.finite_pixels, demo.raw_gi.total_pixels,
                "Camera motion produced non-finite GI at frame {frame}"
            );
            let mut camera_transform = Transform::IDENTITY;
            let mut history_epoch = 0;
            let mut minimum_dispatch_sequence = None;
            let mut capture_pose_ready = true;
            for (mut transform, global_transform, mut gi) in &mut camera {
                *transform = if camera_only {
                    camera_motion_capture_pose(frame)
                } else {
                    motion_capture_pose(frame)
                };
                if !camera_only && frame == 128 {
                    let start_sequence = status
                        .last_dispatch()
                        .map(|dispatch| dispatch.sequence)
                        .or_else(|| {
                            demo.latest_raw_sample
                                .as_ref()
                                .map(|sample| sample.dispatch.sequence)
                        })
                        .unwrap_or(0);
                    minimum_dispatch_sequence = Some(start_sequence.wrapping_add(1).max(1));
                    demo.motion_reset_settle.begin(start_sequence);
                    gi.reset_history();
                }
                camera_transform = *transform;
                capture_pose_ready &= global_transform
                    .to_matrix()
                    .abs_diff_eq(transform.to_matrix(), 1e-5);
                history_epoch = gi.history_epoch;
            }
            // Update applies the pose before transform propagation; capture its first propagated frame.
            if camera_only && capture_this_frame && !capture_pose_ready {
                return;
            }
            let mut geometry_changed = false;
            let mut lighting_state_changed = false;
            let occluder_position = if camera_only {
                None
            } else {
                let positions = motion_capture_positions(frame);
                for mut transform in &mut occluders {
                    geometry_changed |= transform.translation != positions.occluder;
                    transform.translation = positions.occluder;
                }
                let light_state = motion_light_state(frame);
                lighting_state_changed = frame > 0 && motion_light_state(frame - 1) != light_state;
                if motion_light_state_needs_emissive_update(&mut previous_light_state, light_state)
                {
                    for material in &emissive_emitters {
                        if let Some(mut material) = materials.get_mut(&material.0) {
                            material.emissive = [
                                LinearRgba::new(14.0, 4.0, 0.6, 1.0),
                                LinearRgba::new(3.0, 10.0, 18.0, 1.0),
                                LinearRgba::BLACK,
                                LinearRgba::BLACK,
                                LinearRgba::new(4.0, 10.0, 20.0, 1.0),
                            ][light_state];
                        }
                    }
                }
                for (mut transform, mut light) in &mut moving_lights {
                    transform.translation = positions.point_light;
                    light.intensity = [18_000.0, 30_000.0, 18_000.0, 0.0, 24_000.0][light_state];
                    light.color = [
                        Color::linear_rgb(1.0, 0.62, 0.34),
                        Color::linear_rgb(0.35, 0.72, 1.0),
                        Color::linear_rgb(1.0, 0.3, 0.12),
                        Color::linear_rgb(1.0, 0.62, 0.34),
                        Color::linear_rgb(0.25, 1.0, 0.72),
                    ][light_state];
                    light.range = [12.0, 20.0, 12.0, 12.0, 24.0][light_state];
                }
                for (mut transform, mut light) in &mut moving_spots {
                    transform.translation = positions.spot_light;
                    transform.look_at(positions.occluder, Vec3::Y);
                    light.intensity = [12_000.0, 18_000.0, 0.0, 0.0, 24_000.0][light_state];
                    light.color = [
                        Color::linear_rgb(0.45, 0.68, 1.0),
                        Color::linear_rgb(1.0, 0.8, 0.45),
                        Color::linear_rgb(1.0, 0.8, 0.45),
                        Color::linear_rgb(0.7, 0.4, 1.0),
                        Color::linear_rgb(0.4, 1.0, 0.55),
                    ][light_state];
                    light.range = [14.0, 22.0, 22.0, 8.0, 25.0][light_state];
                    light.inner_angle = [0.35, 0.25, 0.25, 0.4, 0.25][light_state];
                    light.outer_angle = [0.65, 0.8, 0.8, 0.5, 0.75][light_state];
                }
                Some(positions.occluder)
            };
            demo.motion_frame += 1;
            if !capture_this_frame {
                return;
            }
            let previous_revisions = demo
                .latest_raw_sample
                .as_ref()
                .map(|sample| sample.dispatch.revisions);
            demo.motion_capture_pending = Some(PendingMotionCapture {
                frame,
                camera_transform,
                history_epoch,
                occluder_position,
                minimum_dispatch_sequence: minimum_dispatch_sequence.unwrap_or_else(|| {
                    status
                        .last_dispatch()
                        .map_or(1, |dispatch| dispatch.sequence.wrapping_add(1).max(1))
                }),
                previous_geometry_revision: if geometry_changed {
                    previous_revisions.map(|revisions| revisions.geometry)
                } else {
                    None
                },
                previous_lighting_revision: if lighting_state_changed {
                    previous_revisions.map(|revisions| revisions.lighting)
                } else {
                    None
                },
            });
            return;
        };
    demo.requested = true;
    let path = demo.capture_prefix.as_ref().map_or_else(
        || {
            if camera_only {
                format!("screenshots/sponza_src_camera_motion_{frame:03}.png")
            } else {
                format!("screenshots/sponza_src_motion_{frame:03}.png")
            }
        },
        |prefix| {
            if camera_only {
                format!("{prefix}_camera_motion_{frame:03}.png")
            } else {
                format!("{prefix}_motion_{frame:03}.png")
            }
        },
    );
    commands
        .spawn(Screenshot::image(demo.target.clone().unwrap()))
        .observe(
            move |event: On<ScreenshotCaptured>,
                  mut commands: Commands,
                  mut demo: ResMut<Demo>,
                  mut views: Query<&mut RadianceCascades>,
                  diagnostics: Res<DiagnosticsStore>,
                  mut exit: MessageWriter<AppExit>| {
                let pixels = event.image.data.as_ref().unwrap().clone();
                save_to_disk(&path)(event);
                let revisions = raw_sample.dispatch.revisions;
                assert!(revisions.scene > 0 && revisions.geometry > 0);
                assert!(raw_sample.radiance.iter().flatten().all(|value| value.is_finite()),
                    "motion capture raw GI contains non-finite radiance at frame {frame}");
                if !camera_only {
                    assert!(revisions.lighting > 0, "light motion was not submitted");
                }
                let mean = pixels
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|p| (f32::from(p[0]) + f32::from(p[1]) + f32::from(p[2])) / 3.0)
                    .sum::<f32>()
                    / (pixels.len() / 4) as f32;
                let light_state = motion_light_state(frame) as u32;
                info!(
                    frame,
                    light_state,
                    raw_frame_id = raw_sample.frame_id,
                    dispatch_sequence = raw_sample.dispatch.sequence,
                    encoded_submitted_at = ?raw_sample.submitted_at,
                    scene_revision = revisions.scene,
                    geometry_revision = revisions.geometry,
                    light_revision = revisions.lighting,
                    history_epoch,
                    camera_position = ?camera_transform.translation,
                    camera_rotation = ?camera_transform.rotation,
                    occluder_position = ?occluder_position,
                    mean_rgb = mean,
                    raw_sample = demo.raw_gi.sample_count,
                    raw_support = demo.raw_gi.supported_pixels,
                    raw_pixels = demo.raw_gi.total_pixels,
                    raw_finite_pixels = demo.raw_gi.finite_pixels,
                    raw_mean_luminance = demo.raw_gi.mean_luminance,
                    raw_supported_mean_luminance = demo.raw_gi.supported_mean_luminance,
                    raw_max_luminance = demo.raw_gi.max_luminance,
                    raw_support_ratio = ?motion_sample_support_ratio(&raw_sample),
                    directional_probe_count = ?demo.raw_gi.directional_probe_count,
                    retention_demand = ?demo.raw_gi.retention_demand,
                    transport_queries_admitted = ?demo.raw_gi.transport_queries_admitted,
                    transport_query_attempts = ?demo.raw_gi.transport_query_attempts,
                    diagnostics = %diagnostics_json(&diagnostics),
                    "Sponza motion capture state"
                );
                assert!(motion_sample_support_ratio(&raw_sample)
                    .is_some_and(|support| support >= 0.95),
                    "motion capture raw GI support fell below 95% at frame {frame}: support={:?}, revisions={revisions:?}",
                    motion_sample_support_ratio(&raw_sample));
                if camera_only {
                    if !(cache_revisit && (65..=128).contains(&frame)) {
                        assert!(
                            mean > 4.0,
                            "Camera-only Sponza lighting became dark: {mean}"
                        );
                    }
                    match frame {
                        0 => demo.day_pixels = pixels.clone(),
                        16 => {
                            let difference = rgb_temporal_difference(&demo.day_pixels, &pixels).0;
                            info!(
                                "Sponza camera-only pan RGB difference from start: {difference:.3}/255"
                            );
                            assert!(
                                difference > 0.2,
                                "Camera-only pan did not change the view: {difference}"
                            );
                        }
                        32 => {
                            // Compare the identical pose immediately: waiting
                            // for a stationary sweep hides movement-induced loss.
                            let difference = rgb_temporal_difference(&demo.day_pixels, &pixels).0;
                            info!("Sponza immediate camera-return RGB difference: {difference:.3}/255");
                            assert!(difference < 8.0,
                                "Sponza GI broke during camera motion: {difference}");
                        }
                        64 => {
                            let difference = rgb_temporal_difference(&demo.day_pixels, &pixels).0;
                            info!(
                                "Sponza camera-only return RGB difference after settling: {difference:.3}/255"
                            );
                            assert!(
                                difference < 8.0,
                                "Sponza camera-only history did not converge after return: {difference}"
                            );
                            if !cache_revisit {
                                exit.write(AppExit::Success);
                            }
                        }
                        129 | 160 if cache_revisit => {
                            let difference = rgb_temporal_difference(&demo.day_pixels, &pixels).0;
                            info!(frame, difference,
                                "Sponza same-pose cache revisit after 64 frames looking away");
                            assert!(difference < 8.0,
                                "Sponza world lighting was lost during the offscreen revisit: {difference}");
                            if frame == 160 {
                                exit.write(AppExit::Success);
                            }
                        }
                        _ => {}
                    }
                } else {
                    if matches!(frame, 32 | 127) {
                        demo.day_pixels = pixels.clone();
                    }
                    if frame == 159 {
                        let encoded_samples = demo
                            .motion_reset_settle
                            .encoded_samples(raw_sample.dispatch.sequence);
                        assert!(
                            encoded_samples >= MOTION_SETTLE_DISPATCHES,
                            "pre-change baseline contains only {encoded_samples} encoded GI dispatches after reset"
                        );
                        let scene_linear_drift =
                            demo.motion_reset_settle.midpoint_drift(&raw_sample);
                        info!(
                            encoded_samples_after_reset = encoded_samples,
                            midpoint_to_baseline_scene_linear_drift = ?scene_linear_drift,
                            "Sponza pre-change baseline convergence observation"
                        );
                        demo.motion_light_baseline_raw = Some(raw_sample.clone());
                        demo.motion_light_change = None;
                        demo.motion_light_response_samples.clear();
                    } else if frame == 192 {
                        let baseline = demo
                            .motion_light_baseline_raw
                            .as_ref()
                            .expect("scene-linear baseline must precede extinction");
                        let (revision, started, encoded_samples, scene_linear_drift) = {
                            let change = demo
                                .motion_light_change
                                .as_ref()
                                .expect("the changed light revision must be encoded");
                            (
                                change.revision,
                                change.encoded_started_at,
                                change.settle.encoded_samples(raw_sample.dispatch.sequence),
                                change.settle.midpoint_drift(&raw_sample),
                            )
                        };
                        assert!(
                            encoded_samples >= MOTION_SETTLE_DISPATCHES,
                            "extinction baseline contains only {encoded_samples} encoded GI dispatches after the light step"
                        );
                        let response = latest_response_sample(
                            &demo.motion_light_response_samples,
                            started,
                        )
                        .expect("extinction response must be sampled within 100 ms");
                        assert_eq!(response.dispatch.revisions.lighting, revision);
                        assert_eq!(raw_sample.dispatch.revisions.lighting, revision);
                        assert!(
                            motion_response_timing_valid(response, &raw_sample, started),
                            "encoded extinction response sample must be within 100 ms; settled encoded sample must be at least 250 ms after the light revision"
                        );
                        let residual = extinguished_light_residual_ratio(
                            baseline,
                            response,
                            &raw_sample,
                        )
                        .expect("extinguished-light response must have measurable contribution");
                        info!(
                            residual_ratio = residual,
                            light_revision = revision,
                            frame_id = raw_sample.frame_id,
                            response_encoded_elapsed_ms = response
                                .submitted_at
                                .saturating_duration_since(started)
                                .as_secs_f64()
                                * 1_000.0,
                            settled_encoded_elapsed_ms = raw_sample
                                .submitted_at
                                .saturating_duration_since(started)
                                .as_secs_f64()
                                * 1_000.0,
                            encoded_samples_after_light_step = encoded_samples,
                            midpoint_to_settled_scene_linear_drift = ?scene_linear_drift,
                            baseline_to_response_drift = ?stationary_sample_drift(baseline, response),
                            response_to_settled_drift = ?stationary_sample_drift(response, &raw_sample),
                            response_geometry_revision = response.dispatch.revisions.geometry,
                            settled_geometry_revision = raw_sample.dispatch.revisions.geometry,
                            "Sponza extinguished-light response and convergence observation"
                        );
                        assert!(
                            residual <= 0.05,
                            "extinguished-light residual is {:.1}% of its preceding contribution",
                            residual * 100.0
                        );
                        demo.day_pixels = pixels.clone();
                    }
                    if frame == 256 {
                        let difference = rgb_temporal_difference(&demo.day_pixels, &pixels).0;
                        info!("Sponza settled/restarted history RGB difference: {difference:.3}/255");
                        assert!(
                            difference < 8.0,
                            "Sponza retained stale lighting after camera motion: {difference}"
                        );
                        demo.motion_light_baseline_raw = Some(raw_sample.clone());
                        demo.motion_light_change = None;
                        demo.motion_light_response_samples.clear();
                    }
                    if frame == 289 {
                        let baseline = demo
                            .motion_light_baseline_raw
                            .as_ref()
                            .expect("scene-linear off baseline must precede the light change");
                        let (revision, started, encoded_samples, scene_linear_drift) = {
                            let change = demo
                                .motion_light_change
                                .as_ref()
                                .expect("new light revision must be encoded");
                            (
                                change.revision,
                                change.encoded_started_at,
                                change.settle.encoded_samples(raw_sample.dispatch.sequence),
                                change.settle.midpoint_drift(&raw_sample),
                            )
                        };
                        assert!(
                            encoded_samples >= MOTION_SETTLE_DISPATCHES,
                            "new-light baseline contains only {encoded_samples} encoded GI dispatches after the light step"
                        );
                        let response = latest_response_sample(
                            &demo.motion_light_response_samples,
                            started,
                        )
                        .expect("new-light response must be sampled within 100 ms");
                        assert_eq!(response.dispatch.revisions.lighting, revision);
                        assert_eq!(raw_sample.dispatch.revisions.lighting, revision);
                        assert!(
                            motion_response_timing_valid(response, &raw_sample, started),
                            "encoded new-light response sample must be within 100 ms; settled encoded sample must be at least 250 ms after the light revision"
                        );
                        let error = lighting_response_error(baseline, response, &raw_sample)
                            .expect("new-light response must affect previously supported receivers");
                        info!(
                            response_error = error,
                            light_revision = revision,
                            frame_id = raw_sample.frame_id,
                            response_encoded_elapsed_ms = response
                                .submitted_at
                                .saturating_duration_since(started)
                                .as_secs_f64()
                                * 1_000.0,
                            settled_encoded_elapsed_ms = raw_sample
                                .submitted_at
                                .saturating_duration_since(started)
                                .as_secs_f64()
                                * 1_000.0,
                            encoded_samples_after_light_step = encoded_samples,
                            midpoint_to_settled_scene_linear_drift = ?scene_linear_drift,
                            baseline_to_response_drift = ?stationary_sample_drift(baseline, response),
                            response_to_settled_drift = ?stationary_sample_drift(response, &raw_sample),
                            response_geometry_revision = response.dispatch.revisions.geometry,
                            settled_geometry_revision = raw_sample.dispatch.revisions.geometry,
                            "Sponza new-light response and convergence observation"
                        );
                        assert!(
                            error <= 0.1,
                            "new-light scene-linear response error is {:.1}%",
                            error * 100.0
                        );
                        exit.write(AppExit::Success);
                    }
                }
                if !camera_only && mean <= 4.0 {
                    // An enclosed noon view can be physically dark. Verify GI
                    // at this exact frozen pose instead of imposing the
                    // courtyard's brightness floor on every camera direction.
                    for mut gi in &mut views { gi.intensity = 0.0; }
                    queue_gi_control(&mut commands, &demo, pixels,
                        format!("{path}.no_gi.png"), false);
                } else {
                    demo.requested = false;
                }
            },
        );
}

fn capture_stability(
    mut commands: Commands,
    time: Res<Time<Real>>,
    mut demo: ResMut<Demo>,
    mut exit: MessageWriter<AppExit>,
) {
    if !demo.stability_capture || demo.stability_frame >= 16 {
        return;
    }
    if time.elapsed_secs_f64() > 600.0 {
        error!(
            loaded_scenes = demo.loaded,
            expected_scenes = demo.scenes.len(),
            gi_frames = demo.gi_frames,
            raw_sample = demo.raw_gi.sample_count,
            eligible_pixels = demo.raw_gi.eligible_pixels,
            supported_pixels = demo.raw_gi.supported_pixels,
            scene_revision = demo.raw_gi.scene_revision,
            geometry_revision = demo.raw_gi.geometry_revision,
            light_revision = demo.raw_gi.light_revision,
            "Sponza fixed-camera capture timed out"
        );
        exit.write(AppExit::error());
        return;
    }
    if demo.loaded != demo.scenes.len() || !demo.gi_ready || demo.requested {
        return;
    }
    let frame = demo.stability_frame;
    demo.requested = true;
    let path = demo.capture_prefix.as_ref().map_or_else(
        || format!("screenshots/sponza_src_stability_{frame:03}.png"),
        |prefix| format!("{prefix}_stability_{frame:03}.png"),
    );
    commands
        .spawn(Screenshot::image(demo.target.clone().unwrap()))
        .observe(
            move |event: On<ScreenshotCaptured>,
                  mut commands: Commands,
                  mut demo: ResMut<Demo>,
                  mut views: Query<&mut RadianceCascades>,
                  mut exit: MessageWriter<AppExit>| {
                let pixels = event.image.data.as_ref().unwrap().clone();
                save_to_disk(&path)(event);
                let mean_rgb = pixels
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|pixel| {
                        (f32::from(pixel[0]) + f32::from(pixel[1]) + f32::from(pixel[2])) / 3.0
                    })
                    .sum::<f32>()
                    / (pixels.len() / 4).max(1) as f32;
                info!(
                    frame,
                    camera_pose_frame = ?demo.camera_pose_frame,
                    mean_rgb,
                    raw_sample = demo.raw_gi.sample_count,
                    raw_support = demo.raw_gi.supported_pixels,
                    raw_pixels = demo.raw_gi.total_pixels,
                    raw_finite_pixels = demo.raw_gi.finite_pixels,
                    raw_mean_luminance = demo.raw_gi.mean_luminance,
                    raw_supported_mean_luminance = demo.raw_gi.supported_mean_luminance,
                    raw_max_luminance = demo.raw_gi.max_luminance,
                    "Sponza fixed-camera diagnostic sample"
                );
                // Keep the stability sequence uninterrupted; the deliberately
                // disabled-lighting control belongs after its final frame.
                let needs_control = demo.camera_pose_frame.is_some() && frame == 15;
                if !demo.stability_pixels.is_empty() {
                    let difference = rgb_temporal_difference(&demo.stability_pixels, &pixels);
                    info!(
                        "Sponza fixed-camera adjacent frame {} -> {}: mean {:.3}, p95 {:.3}, max {:.3}/255",
                        frame - 1,
                        frame,
                        difference.0,
                        difference.1,
                        difference.2
                    );
                    demo.stability_differences.push(difference);
                }
                demo.stability_pixels = pixels.clone();
                demo.stability_frame += 1;
                if demo.stability_frame == 16 {
                    let mut means: Vec<_> = demo
                        .stability_differences
                        .iter()
                        .map(|difference| difference.0)
                        .collect();
                    means.sort_by(f32::total_cmp);
                    let mean = means.iter().sum::<f32>() / means.len().max(1) as f32;
                    let frame_mean_p95 = means[(means.len() * 95 / 100).min(means.len() - 1)];
                    let pixel_p95 = demo
                        .stability_differences
                        .iter()
                        .map(|difference| difference.1)
                        .sum::<f32>()
                        / demo.stability_differences.len().max(1) as f32;
                    let max = *means.last().unwrap_or(&0.0);
                    info!(
                        "Sponza fixed-camera flicker summary across {} adjacent frames: mean {mean:.3}, frame-mean p95 {frame_mean_p95:.3}, average pixel p95 {pixel_p95:.3}, max frame mean {max:.3}/255",
                        means.len()
                    );
                    assert!(mean < 1.0, "Sponza fixed-camera mean flicker is {mean}/255");
                    assert!(
                        frame_mean_p95 < 1.5,
                        "Sponza fixed-camera frame-mean p95 is {frame_mean_p95}/255"
                    );
                    assert!(
                        pixel_p95 < 2.0,
                        "Sponza fixed-camera average pixel p95 is {pixel_p95}/255"
                    );
                    if !needs_control { exit.write(AppExit::Success); }
                }
                if needs_control {
                    for mut gi in &mut views { gi.intensity = 0.0; }
                    queue_gi_control(&mut commands, &demo, pixels,
                        format!("{path}.no_gi.png"), true);
                } else {
                    demo.requested = false;
                }
            },
        );
}

// The requesting capture keeps `requested` set, freezing its deterministic
// camera/occluder until this matched control has completed. It does not reset
// history or change exposure, environment, sampling work, or scene geometry.
fn queue_gi_control(
    commands: &mut Commands,
    demo: &Demo,
    on: Vec<u8>,
    path: String,
    exit_after: bool,
) {
    commands
        .spawn(Screenshot::image(demo.target.clone().unwrap()))
        .observe(
            move |event: On<ScreenshotCaptured>,
                  mut demo: ResMut<Demo>,
                  mut views: Query<&mut RadianceCascades>,
                  mut exit: MessageWriter<AppExit>| {
                let off = event.image.data.as_ref().unwrap();
                let difference = rgb_temporal_difference(&on, off).0;
                info!(
                    "Sponza matched-pose GI on/off difference: {difference:.3}/255; control {path}"
                );
                save_to_disk(&path)(event);
                assert!(
                    difference > 0.2,
                    "GI has no visible contribution at the frozen pose: {difference}/255"
                );
                for mut gi in &mut views {
                    gi.intensity = 1.0;
                }
                demo.requested = false;
                if exit_after {
                    exit.write(AppExit::Success);
                }
            },
        );
}

fn latest_response_sample(
    samples: &[RawGiSample],
    submitted_at: std::time::Instant,
) -> Option<&RawGiSample> {
    samples.iter().rfind(|sample| {
        sample
            .submitted_at
            .checked_duration_since(submitted_at)
            .is_some_and(|elapsed| elapsed <= motion_response_window())
    })
}

fn motion_response_window() -> std::time::Duration {
    std::time::Duration::from_millis(100)
}

fn motion_response_timing_valid(
    response: &RawGiSample,
    settled: &RawGiSample,
    started: std::time::Instant,
) -> bool {
    response
        .submitted_at
        .checked_duration_since(started)
        .is_some_and(|elapsed| elapsed <= motion_response_window())
        && settled
            .submitted_at
            .checked_duration_since(started)
            .is_some_and(|elapsed| elapsed >= std::time::Duration::from_millis(250))
}

fn motion_sample_support_ratio(sample: &RawGiSample) -> Option<f32> {
    if sample.radiance.len() != sample.eligible.len()
        || sample.eligible.len() != sample.supported.len()
    {
        return None;
    }
    let eligible = sample.eligible.iter().filter(|&&value| value).count();
    let supported = sample
        .eligible
        .iter()
        .zip(&sample.supported)
        .filter(|(eligible, supported)| **eligible && **supported)
        .count();
    (eligible > 0).then_some(supported as f32 / eligible as f32)
}

fn stationary_sample_drift(reference: &RawGiSample, settled: &RawGiSample) -> Option<f32> {
    let pixels = reference.radiance.len();
    if pixels < 16
        || settled.radiance.len() != pixels
        || reference.eligible.len() != pixels
        || settled.eligible.len() != pixels
        || reference.supported.len() != pixels
        || settled.supported.len() != pixels
        || !motion_sample_support_ratio(reference).is_some_and(|ratio| ratio >= 0.95)
        || !motion_sample_support_ratio(settled).is_some_and(|ratio| ratio >= 0.95)
    {
        return None;
    }

    let baseline_supported = reference
        .eligible
        .iter()
        .zip(&reference.supported)
        .filter(|(eligible, supported)| **eligible && **supported)
        .count();
    let mut difference_energy = 0.0f64;
    let mut reference_energy = 0.0f64;
    let mut comparable = 0usize;
    for index in 0..pixels {
        if !reference.eligible[index]
            || !reference.supported[index]
            || !settled.eligible[index]
            || !settled.supported[index]
        {
            continue;
        }
        comparable += 1;
        for channel in 0..3 {
            let before = reference.radiance[index][channel];
            let after = settled.radiance[index][channel];
            if !before.is_finite() || !after.is_finite() {
                return None;
            }
            difference_energy += f64::from(after - before).powi(2);
            reference_energy += f64::from(before).powi(2);
        }
    }
    if (comparable as u128) * 100 < (baseline_supported as u128) * 95
        || comparable < 16
        || reference_energy <= 1e-8
    {
        return None;
    }
    Some((difference_energy / reference_energy).sqrt() as f32)
}

fn lighting_response_error(
    before: &RawGiSample,
    response: &RawGiSample,
    settled: &RawGiSample,
) -> Option<f32> {
    let pixels = before.radiance.len();
    if pixels < 16
        || response.radiance.len() != pixels
        || settled.radiance.len() != pixels
        || before.eligible.len() != pixels
        || response.eligible.len() != pixels
        || settled.eligible.len() != pixels
        || before.supported.len() != pixels
        || response.supported.len() != pixels
        || settled.supported.len() != pixels
    {
        return None;
    }

    // Individual Split samples are stochastic: a few pixels can be unsupported
    // in one frame and supported in the next. Keep the capture's 95% support
    // gate for every sample, then measure only their common supported receivers.
    if [before, response, settled]
        .iter()
        .any(|sample| !motion_sample_support_ratio(sample).is_some_and(|ratio| ratio >= 0.95))
    {
        return None;
    }
    let baseline_supported = before
        .eligible
        .iter()
        .zip(&before.supported)
        .filter(|(eligible, supported)| **eligible && **supported)
        .count();

    let mut error_energy = 0.0f64;
    let mut target_energy = 0.0f64;
    let mut comparable = 0usize;
    let mut affected = 0usize;
    for index in 0..pixels {
        if !before.eligible[index] || !before.supported[index] {
            continue;
        }
        if !response.eligible[index]
            || !response.supported[index]
            || !settled.eligible[index]
            || !settled.supported[index]
        {
            continue;
        }
        comparable += 1;

        let target = std::array::from_fn::<_, 3, _>(|channel| {
            settled.radiance[index][channel] - before.radiance[index][channel]
        });
        if !target.iter().all(|value| value.is_finite()) {
            return None;
        }
        let target_norm = target
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>();
        if target_norm <= 1e-8 {
            continue;
        }
        let residual = std::array::from_fn::<_, 3, _>(|channel| {
            settled.radiance[index][channel] - response.radiance[index][channel]
        });
        if !residual.iter().all(|value| value.is_finite()) {
            return None;
        }
        error_energy += residual
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>();
        target_energy += target_norm;
        affected += 1;
    }
    if (comparable as u128) * 100 < (baseline_supported as u128) * 95 {
        return None;
    }
    (affected >= 16 && target_energy > 1e-8).then(|| (error_energy / target_energy).sqrt() as f32)
}

fn extinguished_light_residual_ratio(
    before: &RawGiSample,
    response: &RawGiSample,
    settled: &RawGiSample,
) -> Option<f32> {
    lighting_response_error(before, response, settled)
}

fn rgb_temporal_difference(previous: &[u8], current: &[u8]) -> (f32, f32, f32) {
    let mut differences: Vec<_> = previous
        .as_chunks::<4>()
        .0
        .iter()
        .zip(current.as_chunks::<4>().0)
        .map(|(a, b)| {
            (0..3)
                .map(|channel| (f32::from(a[channel]) - f32::from(b[channel])).abs())
                .sum::<f32>()
                / 3.0
        })
        .collect();
    differences.sort_by(f32::total_cmp);
    let mean = differences.iter().sum::<f32>() / differences.len().max(1) as f32;
    let p95 = differences[(differences.len() * 95 / 100).min(differences.len() - 1)];
    let max = *differences.last().unwrap_or(&0.0);
    (mean, p95, max)
}

fn drain_gpu(device: Res<RenderDevice>) {
    // Automated offscreen runs lack swapchain back-pressure. Bound outstanding
    // GPU work even when a frame takes longer than the capture interval.
    device
        .poll(PollType::wait_indefinitely())
        .expect("offscreen GPU completion");
}

fn stop_benchmark_raw_output(
    mut commands: Commands,
    demo: Res<Demo>,
    time: Res<Time<Real>>,
    diagnostics: Res<DiagnosticsStore>,
    readbacks: Query<Entity, With<BenchmarkRawReadback>>,
    raw_outputs: Query<Entity, With<RadianceCascadesRawOutput>>,
    mut stopped: Local<bool>,
) {
    let now = time.elapsed_secs_f64();
    if *stopped
        || !demo.benchmark
        || !demo.gi_ready
        || demo.loaded != demo.scenes.len()
        || now <= demo.settle_until + 3.0
        || !diagnostics.iter().any(|diagnostic| {
            diagnostic.path().as_str().contains("split_rc")
                && diagnostic.path().as_str().ends_with("elapsed_gpu")
                && diagnostic.value().is_some()
        })
    {
        return;
    }

    for entity in &readbacks {
        commands.entity(entity).despawn();
    }
    for entity in &raw_outputs {
        commands
            .entity(entity)
            .remove::<RadianceCascadesRawOutput>();
    }
    *stopped = true;
    info!(
        "Sponza benchmark: disabled raw GI output/readback after readiness; GI timing diagnostics remain active"
    );
}

fn resize_raw_gi_output(
    views: Query<(&Camera, &RadianceCascades, &RadianceCascadesRawOutput)>,
    mut images: ResMut<Assets<Image>>,
) {
    for (camera, settings, raw) in &views {
        let Some(viewport) = camera.physical_viewport_size() else {
            continue;
        };
        let scale = settings.sampling.resolve_scale.max(1);
        let size = wgpu::Extent3d {
            width: viewport.x.div_ceil(scale).max(1),
            height: viewport.y.div_ceil(scale).max(1),
            depth_or_array_layers: 1,
        };
        if let Some(mut image) = images.get_mut(&raw.0)
            && image.texture_descriptor.size != size
        {
            image.resize(size);
        }
    }
}

fn asset_errors(
    mut errors: MessageReader<UntypedAssetLoadFailedEvent>,
    mut exit: MessageWriter<AppExit>,
) {
    for error in errors.read() {
        error!("Sponza asset failed: {}: {}", error.path, error.error);
        exit.write(AppExit::error());
    }
}

fn fail_capture_on_gi_error(
    demo: Res<Demo>,
    status: Res<RadianceCascadesStatus>,
    mut exit: MessageWriter<AppExit>,
) {
    if !capture_requires_gi(demo.capture, demo.benchmark, demo.benchmark_no_gi) {
        return;
    }
    let failures = status.failures();
    if !failures.is_empty() {
        error!(failures = ?failures, "Sponza GI capture failed");
        exit.write(AppExit::error());
    }
}

fn capture_requires_gi(capture: bool, benchmark: bool, benchmark_no_gi: bool) -> bool {
    capture || (benchmark && !benchmark_no_gi)
}

#[cfg(test)]
mod tests {
    #[test]
    fn gi_readiness_requires_committed_revisions_support_and_warmup() {
        let revisions = bevy_penumbra::RadianceCascadesRevisions {
            scene: 1,
            geometry: 2,
            lighting: 3,
        };
        let dispatch = |revisions| {
            Some(bevy_penumbra::RadianceCascadesDispatch {
                sequence: 4,
                revisions,
            })
        };
        assert!(super::gi_capture_ready(
            true,
            dispatch(revisions),
            95,
            0.95,
            32,
            32
        ));
        assert!(!super::gi_capture_ready(true, None, 95, 1.0, 32, 32));
        assert!(!super::gi_capture_ready(
            true,
            dispatch(bevy_penumbra::RadianceCascadesRevisions {
                lighting: 0,
                ..revisions
            }),
            95,
            0.95,
            32,
            32,
        ));
        assert!(!super::gi_capture_ready(
            true,
            dispatch(revisions),
            0,
            1.0,
            32,
            32
        ));
        assert!(!super::gi_capture_ready(
            true,
            dispatch(revisions),
            95,
            0.94,
            32,
            32
        ));
        assert!(!super::gi_capture_ready(
            true,
            dispatch(revisions),
            95,
            0.95,
            31,
            32
        ));
    }

    #[test]
    fn no_gi_baseline_does_not_fail_on_gi_pipeline_errors() {
        assert!(super::capture_requires_gi(true, false, false));
        assert!(super::capture_requires_gi(false, true, false));
        assert!(!super::capture_requires_gi(false, true, true));
        assert!(!super::capture_requires_gi(false, false, false));
    }

    #[test]
    fn benchmark_includes_custom_render_counters() {
        use bevy::diagnostic::{
            Diagnostic, DiagnosticMeasurement, DiagnosticPath, DiagnosticsStore,
        };
        let mut store = DiagnosticsStore::default();
        let mut diagnostic = Diagnostic::new(DiagnosticPath::new(
            "render/penumbra/view_test/persistent_cache_dropped",
        ));
        diagnostic.add_measurement(DiagnosticMeasurement {
            time: bevy::platform::time::Instant::now(),
            value: 7.0,
        });
        store.add(diagnostic);
        assert!(
            super::diagnostics_json(&store)
                .contains("\"render/penumbra/view_test/persistent_cache_dropped\":7.000000")
        );
    }
    #[test]
    fn camera_cache_revisit_keeps_the_reference_pose_and_looks_away_for_64_frames() {
        let reference = camera_motion_capture_pose(0);
        for frame in [32, 64, 129, 160] {
            let pose = camera_motion_capture_pose(frame);
            assert_eq!(pose.translation, reference.translation);
            assert_eq!(pose.rotation, reference.rotation);
        }
        for frame in 65..=128 {
            let pose = camera_motion_capture_pose(frame);
            assert_eq!(pose.translation, reference.translation);
            assert!(pose.forward().dot(*reference.forward()) < -0.9999);
        }
    }

    #[test]
    fn motion_lights_end_in_an_extinguished_state_after_a_stable_baseline() {
        assert_eq!(motion_light_state(31), 0);
        assert_eq!(motion_light_state(32), 1);
        assert_eq!(motion_light_state(64), 2);
        assert_eq!(motion_light_state(96), 0);
        assert_eq!(motion_light_state(159), 0);
        assert_eq!(motion_light_state(160), 3);
        assert_eq!(motion_light_state(256), 3);
        assert_eq!(motion_light_state(257), 4);
        assert_eq!(motion_light_state(u32::MAX), 4);

        let make_sample = |radiance: Vec<[f32; 3]>, supported: bool| RawGiSample {
            frame_id: 1,
            dispatch: RadianceCascadesDispatch {
                sequence: 1,
                revisions: RadianceCascadesRevisions {
                    scene: 1,
                    geometry: 1,
                    lighting: 1,
                },
            },
            submitted_at: std::time::Instant::now(),
            eligible: vec![true; radiance.len()],
            supported: vec![supported; radiance.len()],
            radiance,
        };
        let baseline = make_sample(vec![[0.0, 0.0, 0.0]; 32], true);
        let mut settled_values = vec![[0.0, 0.0, 0.0]; 32];
        settled_values[..16].fill([10.0, 0.0, 0.0]);
        let settled = make_sample(settled_values, true);
        let mut response_values = vec![[0.0, 0.0, 0.0]; 32];
        response_values[..16].fill([9.0, 0.0, 0.0]);
        let response = make_sample(response_values, true);
        let error = lighting_response_error(&baseline, &response, &settled).unwrap();
        assert!((error - 0.1).abs() < 1e-6);
        assert_eq!(
            extinguished_light_residual_ratio(&baseline, &response, &settled),
            Some(error)
        );

        let mut wrong_color_values = vec![[0.0, 0.0, 0.0]; 32];
        wrong_color_values[..16].fill([0.0, 9.0, 0.0]);
        let wrong_color = make_sample(wrong_color_values, true);
        assert!(lighting_response_error(&baseline, &wrong_color, &settled).unwrap() > 1.0);

        let mut overshoot_values = vec![[0.0, 0.0, 0.0]; 32];
        overshoot_values[..16].fill([20.0, 0.0, 0.0]);
        let overshoot = make_sample(overshoot_values, true);
        assert!(lighting_response_error(&baseline, &overshoot, &settled).unwrap() > 0.9);

        let unsupported = make_sample(response.radiance.clone(), false);
        assert!(lighting_response_error(&baseline, &unsupported, &settled).is_none());

        let start = std::time::Instant::now();
        let mut early = response.clone();
        early.submitted_at = start + std::time::Duration::from_millis(90);
        let mut late = response;
        late.submitted_at = start + std::time::Duration::from_millis(101);
        assert_eq!(
            latest_response_sample(&[early], start)
                .unwrap()
                .submitted_at,
            start + std::time::Duration::from_millis(90)
        );
        assert!(latest_response_sample(&[late], start).is_none());
    }

    #[test]
    fn light_response_compares_common_support_without_relaxing_coverage_or_residual_gates() {
        let make_sample = |radiance: Vec<[f32; 3]>, supported: Vec<bool>| RawGiSample {
            frame_id: 1,
            dispatch: RadianceCascadesDispatch {
                sequence: 1,
                revisions: RadianceCascadesRevisions {
                    scene: 1,
                    geometry: 1,
                    lighting: 1,
                },
            },
            submitted_at: std::time::Instant::now(),
            eligible: vec![true; radiance.len()],
            supported,
            radiance,
        };
        let baseline = make_sample(vec![[0.0, 0.0, 0.0]; 100], vec![true; 100]);
        let mut settled_values = vec![[0.0, 0.0, 0.0]; 100];
        settled_values[..32].fill([20.0, 0.0, 0.0]);
        let settled = make_sample(settled_values, vec![true; 100]);
        let mut response_values = vec![[0.0, 0.0, 0.0]; 100];
        response_values[..32].fill([19.0, 0.0, 0.0]);
        let mut response = make_sample(response_values, vec![true; 100]);

        // Independent sparse observations may miss different pixels. The
        // common receiver set still meets the same 95% comparison-coverage gate.
        response.supported[..3].fill(false);
        let mut settled_partial = settled.clone();
        settled_partial.supported[3..5].fill(false);
        let error = lighting_response_error(&baseline, &response, &settled_partial).unwrap();
        assert!((error - 0.05).abs() < 1e-6);

        // Each sample individually still has >=95% support, but the common
        // comparison set is only 94% of the baseline and cannot pass.
        response.supported[5] = false;
        assert!(motion_sample_support_ratio(&response).unwrap() >= 0.95);
        assert!(lighting_response_error(&baseline, &response, &settled_partial).is_none());

        // Exactly 95% per-sample support remains admissible; falling below it
        // cannot be hidden by the common-pixel intersection.
        response.supported.fill(true);
        response.supported[..5].fill(false);
        assert_eq!(motion_sample_support_ratio(&response), Some(0.95));
        assert!(lighting_response_error(&baseline, &response, &settled).is_some());
        response.supported[5] = false;
        assert!(motion_sample_support_ratio(&response).unwrap() < 0.95);
        assert!(lighting_response_error(&baseline, &response, &settled).is_none());

        // A response outside the unchanged 5% residual limit still fails.
        response.supported.fill(true);
        response.radiance[..32].fill([18.9, 0.0, 0.0]);
        assert!(lighting_response_error(&baseline, &response, &settled).unwrap() > 0.05);

        // Supported output without an actual measured lighting delta is not a pass.
        assert!(lighting_response_error(&baseline, &baseline, &baseline).is_none());
    }

    #[test]
    fn motion_capture_waits_for_encoded_samples_after_each_history_change() {
        let make_sample = |sequence: u64, lighting: u64, value: f32| RawGiSample {
            frame_id: sequence,
            dispatch: RadianceCascadesDispatch {
                sequence,
                revisions: RadianceCascadesRevisions {
                    scene: 1,
                    geometry: 2,
                    lighting,
                },
            },
            submitted_at: std::time::Instant::now(),
            radiance: vec![[value, value, value]; 32],
            eligible: vec![true; 32],
            supported: vec![true; 32],
        };
        assert!(!motion_capture_can_advance(
            0,
            false,
            false,
            None,
            &MotionSettleObservation::default(),
            None
        ));
        assert!(motion_capture_can_advance(
            0,
            true,
            false,
            None,
            &MotionSettleObservation::default(),
            None
        ));

        let mut reset_settle = MotionSettleObservation::default();
        reset_settle.begin(100);
        let immediate = make_sample(101, 3, 1.0);
        reset_settle.observe(&immediate);
        assert!(motion_capture_can_advance(
            128,
            true,
            false,
            Some(&immediate),
            &reset_settle,
            None
        ));
        let before_reset_settle = make_sample(227, 3, 1.0);
        assert!(!motion_capture_can_advance(
            129,
            true,
            false,
            Some(&before_reset_settle),
            &reset_settle,
            None
        ));
        let reset_settled = make_sample(228, 3, 1.0);
        assert!(motion_capture_can_advance(
            129,
            true,
            false,
            Some(&reset_settled),
            &reset_settle,
            None
        ));

        let mut light_settle = MotionSettleObservation::default();
        light_settle.begin(500);
        let change = MotionLightChange {
            revision: 7,
            encoded_started_at: std::time::Instant::now()
                .checked_sub(std::time::Duration::from_millis(300))
                .unwrap(),
            settle: light_settle,
        };
        let before_light_settle = make_sample(627, 7, 1.0);
        assert!(!motion_capture_can_advance(
            161,
            true,
            false,
            Some(&before_light_settle),
            &reset_settle,
            Some(&change)
        ));
        let light_settled = make_sample(628, 7, 1.1);
        assert!(motion_capture_can_advance(
            161,
            true,
            false,
            Some(&light_settled),
            &reset_settle,
            Some(&change)
        ));
        let fast_change = MotionLightChange {
            revision: 7,
            encoded_started_at: light_settled.submitted_at,
            settle: change.settle.clone(),
        };
        assert!(!motion_capture_can_advance(
            161,
            true,
            false,
            Some(&light_settled),
            &reset_settle,
            Some(&fast_change)
        ));
        assert!(motion_capture_can_advance(
            258,
            true,
            false,
            Some(&light_settled),
            &reset_settle,
            Some(&change)
        ));
        let wrong_revision = make_sample(700, 8, 1.1);
        assert!(!motion_capture_can_advance(
            161,
            true,
            false,
            Some(&wrong_revision),
            &reset_settle,
            Some(&change)
        ));
        assert!(motion_capture_can_advance(
            129,
            false,
            true,
            Some(&immediate),
            &reset_settle,
            None
        ));

        let mut convergence = MotionSettleObservation::default();
        convergence.begin(100);
        let midpoint = make_sample(164, 3, 1.0);
        convergence.observe(&midpoint);
        let late = make_sample(228, 3, 1.1);
        assert_eq!(convergence.encoded_samples(late.dispatch.sequence), 128);
        assert!(
            convergence
                .midpoint_drift(&late)
                .is_some_and(|drift| drift > 0.09)
        );
    }

    #[test]
    fn pending_motion_capture_uses_the_encoded_pose_revisions() {
        let sample = |sequence, geometry, lighting| RawGiSample {
            frame_id: sequence,
            dispatch: RadianceCascadesDispatch {
                sequence,
                revisions: RadianceCascadesRevisions {
                    scene: 1,
                    geometry,
                    lighting,
                },
            },
            submitted_at: std::time::Instant::now(),
            radiance: Vec::new(),
            eligible: Vec::new(),
            supported: Vec::new(),
        };
        let pending = PendingMotionCapture {
            frame: 160,
            camera_transform: Transform::IDENTITY,
            history_epoch: 1,
            occluder_position: Some(Vec3::ZERO),
            minimum_dispatch_sequence: 10,
            previous_geometry_revision: Some(4),
            previous_lighting_revision: Some(8),
        };
        assert!(!pending_motion_sample_ready(&pending, &sample(9, 5, 9)));
        assert!(!pending_motion_sample_ready(&pending, &sample(10, 4, 9)));
        assert!(!pending_motion_sample_ready(&pending, &sample(11, 5, 8)));
        assert!(pending_motion_sample_ready(&pending, &sample(11, 5, 9)));
    }

    #[test]
    fn motion_response_window_and_settled_sample_have_distinct_deadlines() {
        let make_sample = |submitted_at| RawGiSample {
            frame_id: 1,
            dispatch: RadianceCascadesDispatch {
                sequence: 1,
                revisions: RadianceCascadesRevisions {
                    scene: 1,
                    geometry: 1,
                    lighting: 1,
                },
            },
            submitted_at,
            radiance: vec![[1.0, 1.0, 1.0]; 32],
            eligible: vec![true; 32],
            supported: vec![true; 32],
        };
        let started = std::time::Instant::now();
        let response = make_sample(started + std::time::Duration::from_millis(100));
        let settled = make_sample(started + std::time::Duration::from_millis(250));
        assert!(motion_response_timing_valid(&response, &settled, started));

        let late_response = make_sample(started + std::time::Duration::from_millis(101));
        assert!(!motion_response_timing_valid(
            &late_response,
            &settled,
            started
        ));
        let early_settled = make_sample(started + std::time::Duration::from_millis(249));
        assert!(!motion_response_timing_valid(
            &response,
            &early_settled,
            started
        ));
        let early_response = make_sample(started - std::time::Duration::from_millis(1));
        assert!(!motion_response_timing_valid(
            &early_response,
            &settled,
            started
        ));
    }

    #[test]
    fn motion_capture_support_gate_uses_only_eligible_pixels() {
        let mut sample = RawGiSample {
            frame_id: 1,
            dispatch: RadianceCascadesDispatch {
                sequence: 1,
                revisions: RadianceCascadesRevisions {
                    scene: 1,
                    geometry: 1,
                    lighting: 1,
                },
            },
            submitted_at: std::time::Instant::now(),
            radiance: vec![[1.0, 1.0, 1.0]; 20],
            eligible: vec![true; 10]
                .into_iter()
                .chain(std::iter::repeat_n(false, 10))
                .collect(),
            supported: vec![true; 20],
        };
        sample.supported[0] = false;
        assert_eq!(motion_sample_support_ratio(&sample), Some(0.9));
        assert!(!motion_sample_support_ratio(&sample).is_some_and(|ratio| ratio >= 0.95));
        sample.supported[0] = true;
        assert_eq!(motion_sample_support_ratio(&sample), Some(1.0));
        assert!(motion_sample_support_ratio(&sample).is_some_and(|ratio| ratio >= 0.95));
        sample.supported.pop();
        assert_eq!(motion_sample_support_ratio(&sample), None);
    }

    #[test]
    fn motion_objects_start_at_the_frame_zero_capture_pose() {
        let positions = motion_capture_positions(0);
        assert!(
            positions
                .occluder
                .abs_diff_eq(Vec3::new(-3.2, 1.2, 0.5), 1e-6)
        );
        assert!(
            positions
                .point_light
                .abs_diff_eq(Vec3::new(-2.9, 2.5, 0.5), 1e-6)
        );
        assert!(
            positions
                .spot_light
                .abs_diff_eq(Vec3::new(-1.5, 2.9, -0.1), 1e-6)
        );
        assert_eq!(motion_capture_positions(127), motion_capture_positions(128));
    }

    #[test]
    fn emissive_state_initialization_skips_a_redundant_material_write() {
        let mut previous = None;
        assert!(!motion_light_state_needs_emissive_update(&mut previous, 0));
        assert_eq!(previous, Some(0));
        assert!(!motion_light_state_needs_emissive_update(&mut previous, 0));
        assert!(motion_light_state_needs_emissive_update(&mut previous, 1));
        assert_eq!(previous, Some(1));
    }

    use super::*;
    #[test]
    fn resolve_scale_change_resizes_raw_output_without_resetting_world_history() {
        let mut app = App::new();
        app.init_resource::<Assets<Image>>()
            .add_systems(Update, resize_raw_gi_output);
        let raw = app
            .world_mut()
            .resource_mut::<Assets<Image>>()
            .add(Image::new_target_texture(
                480,
                270,
                TextureFormat::Rgba32Float,
                None,
            ));
        let mut settings = RadianceCascades::low_end();
        settings.sampling.resolve_scale = 8;
        settings.history_epoch = 22;
        let entity = app
            .world_mut()
            .spawn((
                Camera {
                    viewport: Some(bevy::camera::Viewport {
                        physical_size: UVec2::new(1920, 1080),
                        ..default()
                    }),
                    ..default()
                },
                settings,
                RadianceCascadesRawOutput(raw.clone()),
            ))
            .id();
        app.update();
        let images = app.world().resource::<Assets<Image>>();
        assert_eq!(images.get(&raw).unwrap().size(), UVec2::new(240, 135));
        assert_eq!(
            app.world()
                .get::<RadianceCascades>(entity)
                .unwrap()
                .history_epoch,
            22
        );
    }

    #[test]
    fn clock_wraps_sun_crosses_the_horizon_and_capture_queries_initialize() {
        assert!(sun_direction(12.0).y > 0.9);
        assert!(sun_direction(0.0).y < -0.9);
        assert!(sun_direction(6.0).y.abs() < 1e-6);
        assert!(sun_direction(18.0).y.abs() < 1e-6);
        assert!(sun_direction(34.0).abs_diff_eq(sun_direction(10.0), 1e-6));
        for hour in 0..24 {
            assert!((sun_direction(hour as f32).length() - 1.0).abs() < 1e-6);
        }
        let mask = leaf_mask();
        let pixels = mask.data.as_ref().unwrap().as_chunks::<4>().0;
        assert!(pixels.iter().any(|p| p[3] == 0));
        assert!(pixels.iter().any(|p| p[3] == 255));
        assert_eq!(pixels[0][3], 0);
        assert_eq!(pixels[32 * 64 + 32][3], 255);
        let dimensions = vec!["--width=160".to_owned(), "--height=120".to_owned()];
        assert_eq!(dimension_arg(&dimensions, "--width=", 640), 160);
        assert_eq!(dimension_arg(&dimensions, "--height=", 360), 120);
        let zero_dimension = vec!["--width=0".to_owned()];
        assert_eq!(dimension_arg(&zero_dimension, "--width=", 640), 1);
        let mut app = App::new();
        app.insert_resource(Demo {
            startup_started: std::time::Instant::now(),
            scene_ready_at: None,
            first_gi_dispatch_at: None,
            first_95pct_support_logged: false,
            scenes: Vec::new(),
            capture_size: (640, 360),
            capture_prefix: None,
            hour: 23.0,
            playing: false,
            capture: false,
            target: None,
            loaded: 0,
            day_mean: 0.0,
            day_pixels: Vec::new(),
            gi_ready: false,
            gi_frames: 0,
            last_gi_dispatch: None,
            capture_stage: 0,
            requested: false,
            settle_until: 0.0,
            benchmark: false,
            benchmark_windowed: false,
            benchmark_vsync: false,
            benchmark_no_gi: false,
            multi_bounce: true,
            cached_hit_sky: false,
            hardware_ray_queries: false,
            resolution_divisor: RadianceCascades::low_end().sampling.grid_divisor,
            max_rays: 32_768,
            max_traversal_queries: 16_384,
            warmup_frames: 120,
            measured_frames: 120,
            benchmark_output: String::from("tmp/sponza_benchmark.json"),
            motion_capture: false,
            camera_motion_capture: false,
            camera_pose_frame: None,
            motion_frame: 0,
            latest_raw_sample: None,
            motion_light_baseline_raw: None,
            motion_light_response_samples: Vec::new(),
            motion_light_change: None,
            motion_reset_settle: MotionSettleObservation::default(),
            motion_capture_pending: None,
            raw_gi: RawGiDiagnostics::default(),
            stability_capture: false,
            stability_frame: 0,
            stability_pixels: Vec::new(),
            stability_differences: Vec::new(),
        })
        .insert_resource(RadianceCascadesStatus::default())
        .init_resource::<ButtonInput<KeyCode>>()
        .init_resource::<Time>()
        .init_resource::<Time<Real>>()
        .init_resource::<Assets<StandardMaterial>>()
        .add_message::<AppExit>()
        .add_systems(Update, (controls, capture_motion));
        let mut profile = RadianceCascades::low_end();
        apply_sponza_profile(&mut profile, app.world().resource::<Demo>());
        assert!(profile.multi_bounce);
        assert_eq!(
            (profile.probe_spacing, profile.interval_length),
            (SPONZA_PROBE_SPACING_METERS, SPONZA_INTERVAL_LENGTH_METERS)
        );
        let profile_json = benchmark_json(
            app.world().resource::<Demo>(),
            &PerformanceSamples::default(),
            &DiagnosticsStore::default(),
            None,
        );
        assert!(profile_json.contains("\"probe_spacing_m\":1,\"interval_length_m\":1.6"));
        assert!(profile_json.contains("\"multi_bounce\":true"));
        assert_eq!(
            profile.sampling.max_receiver_cells,
            SPONZA_SPLIT_MAX_RECEIVER_CELLS
        );
        assert_eq!(
            profile.sampling.max_cache_entries,
            SPONZA_SPLIT_MAX_CACHE_ENTRIES
        );
        assert_eq!(
            (profile.temporal_blend, profile.history_frames),
            (0.985, 64)
        );
        let camera = app.world_mut().spawn(RadianceCascades::default()).id();
        app.world_mut()
            .resource_mut::<Time>()
            .advance_by(std::time::Duration::from_secs(1));
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::BracketRight);
        app.update();
        assert_eq!(app.world().resource::<Demo>().hour, 1.0);
        let mut keys = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
        keys.reset_all();
        keys.press(KeyCode::Digit2);
        keys.press(KeyCode::KeyG);
        keys.press(KeyCode::Space);
        app.world_mut()
            .resource_mut::<Time>()
            .advance_by(std::time::Duration::ZERO);
        app.update();
        assert_eq!(app.world().resource::<Demo>().hour, 12.0);
        assert!(app.world().resource::<Demo>().playing);
        let gi = app.world().get::<RadianceCascades>(camera).unwrap();
        assert_eq!(gi.intensity, 0.0);
        assert_eq!(gi.history_epoch, 1);
<<<<<<< Updated upstream
=======
        // H switches traversal without resetting GI history.
        let mut keys = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
        keys.reset_all();
        keys.press(KeyCode::KeyH);
        app.update();
        assert!(app.world().resource::<Demo>().hardware_ray_queries);
        let gi = app.world().get::<RadianceCascades>(camera).unwrap();
        assert!(gi.hardware_ray_queries);
        assert_eq!(gi.history_epoch, 1);
        let mut keys = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
        keys.reset_all();
        keys.press(KeyCode::KeyH);
        app.update();
        assert!(!app.world().get::<RadianceCascades>(camera).unwrap().hardware_ray_queries);
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .reset_all();
        app.world_mut().resource_mut::<Demo>().playing = false;
        app.insert_resource(ClearColor::default());
        let sun = app
            .world_mut()
            .spawn((Sun, DirectionalLight::default(), Transform::default()))
            .id();
        app.add_systems(Update, lighting);
        for (hour, shadows) in [
            (12.0, true),
            (0.0, false),
            (12.0, true),
            (5.99, false),
            (6.01, true),
            (17.99, true),
            (18.01, false),
        ] {
            app.world_mut().resource_mut::<Demo>().hour = hour;
            app.update();
            let light = app.world().get::<DirectionalLight>(sun).unwrap();
            assert_eq!(light.shadow_maps_enabled, shadows, "hour {hour}");
            assert_eq!(light.illuminance > 0.0, shadows, "hour {hour}");
        }
>>>>>>> Stashed changes
    }
}

//! Cornell-box Split Radiance Cascades demo.
//!
//! Run interactively with `cargo run --example cornell_box`.
//! `--width=N --height=N` set the requested
//! logical window size; the receiver grid adapts to its physical size and
//! configured limits. Live captures retain the same monitor DPI, controls, and
//! diagnostics overlay as the normal interactive window. Use
//! `--live-capture` to save a ready live window to `captures/cornell_live.png`;
//! add `--capture-frame-count=N` for a numbered sequence. Press G to toggle
//! GI and use the arrow keys to move the ceiling panel. P saves a screenshot
//! after GI is ready; F3 toggles GI work counters, and Escape exits.
//! `--capture` saves a headless PNG,
//! `--capture-raw` writes the raw linear GI estimate as PFM, and `--benchmark`
//! records completed Split Radiance Cascades GPU timings.
//! `--capture-response` writes baseline, camera-moved, and doubled-emitter raw
//! GI samples with phase suffixes; `--response-settle-frames=N` controls the
//! post-change rendered samples before each settled readback.
//! Captures wait for 16 complete sparse update epochs; override this with
//! `--capture-warmup-frames=N` when testing startup or convergence.
//! `--capture-frame-count=N` saves a sequence for checking display flicker.

use bevy::{
    app::{AppExit, ScheduleRunnerPlugin},
    camera::{Exposure, RenderTarget, ShadowLodOrigin},
    core_pipeline::tonemapping::Tonemapping,
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    pbr::DefaultOpaqueRendererMethod,
    platform::time::Instant,
    prelude::*,
    render::{
        RenderPlugin,
        diagnostic::RenderDiagnosticsPlugin,
        gpu_readback::{Readback, ReadbackComplete, ReadbackOnce},
        render_resource::{TextureFormat, TextureUsages},
        view::screenshot::{Screenshot, ScreenshotCaptured, save_to_disk},
    },
    window::{ExitCondition, PrimaryWindow},
    winit::WinitPlugin,
};

const CORNELL_CAMERA_EV100: f32 = 0.0;
use bevy_penumbra::{
    RadianceCascades, RadianceCascadesGeometry, RadianceCascadesPlugin, RadianceCascadesRawOutput,
};

#[derive(Resource)]
struct Capture {
    target: Handle<Image>,
    image_path: String,
    image_frame_count: u32,
    image_frames_completed: u32,
    last_image_sample: Option<Instant>,
    raw_target: Option<Handle<Image>>,
    raw_path: String,
    viewport: UVec2,
    raw_size: UVec2,
    resolution_divisor: u32,
    max_receiver_cells: u32,
    max_rays: u32,
    warmup_frames: u32,
    resolve_scale: u32,
    raw_frame_count: usize,
    raw_frames_queued: usize,
    raw_frames_completed: usize,
    last_raw_sample: Option<Instant>,
    raw_rgb_sum: Vec<[f64; 3]>,
    raw_sample_count: Vec<u32>,
    raw_indirect_count: Vec<u32>,
    raw_direct_count: Vec<u32>,
    started_at: Option<f64>,
    ready_frames: u32,
    last_gi_sample: Option<Instant>,
    requested: bool,
}

#[derive(Resource, Default)]
struct GiDebugState {
    samples: u32,
    required_samples: u32,
    last_sample: Option<Instant>,
    started_at: Option<f64>,
    viewport: Option<UVec2>,
    enabled: bool,
    history_epoch: Option<u32>,
    render_ms: Option<f64>,
    timing_source: Option<&'static str>,
    receiver_updates: Option<u64>,
    receiver_candidates: Option<u64>,
    query_attempts: Option<u64>,
    query_admitted: Option<u64>,
    show_details: bool,
}

impl GiDebugState {
    fn restart_warmup(&mut self) {
        self.samples = 0;
        self.started_at = None;
    }

    fn ready(&self, now: f64) -> bool {
        self.enabled
            && self
                .receiver_candidates
                .is_some_and(|candidates| candidates > 0)
            && self.samples >= self.required_samples
            && self
                .started_at
                .is_some_and(|started_at| now >= started_at + 3.0)
    }

    fn elapsed(&self, now: f64) -> f64 {
        self.started_at
            .map_or(0.0, |started_at| (now - started_at).max(0.0))
    }
}

#[derive(Resource)]
struct Benchmark {
    started_at: Option<f64>,
    warmup_frames: u32,
    measured_frames: u32,
    repetitions: u32,
    repetition: u32,
    frame: u32,
    last_sample: Option<Instant>,
    timings_ms: Vec<f64>,
}

#[derive(Resource)]
struct LiveCapture {
    output_path: String,
    frame_count: u32,
    minimum_samples: u32,
    frames_requested: u32,
    frames_completed: u32,
    started_at: Option<f64>,
    last_sample: Option<Instant>,
    viewport: Option<UVec2>,
}

#[derive(Component)]
struct LightMover;

#[derive(Component)]
struct RawCaptureOutput {
    path: String,
    size: UVec2,
    viewport: UVec2,
    camera_exposure: f32,
    resolution_divisor: u32,
    max_receiver_cells: u32,
    max_rays: u32,
    resolve_scale: u32,
    response_phase: Option<RawResponsePhase>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RawResponsePhase {
    Baseline,
    CameraImmediate,
    CameraSettled,
    LightImmediate,
    LightSettled,
}

impl RawResponsePhase {
    fn suffix(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::CameraImmediate => "camera_1",
            Self::CameraSettled => "camera_settled",
            Self::LightImmediate => "light_1",
            Self::LightSettled => "light_settled",
        }
    }

    fn capture_path(self, base_path: &str) -> String {
        let base = std::path::Path::new(base_path);
        let stem = base
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("cornell_response");
        let extension = base
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("pfm");
        base.with_file_name(format!("{stem}_{}.{extension}", self.suffix()))
            .to_string_lossy()
            .into_owned()
    }

    fn frames_until_capture(self, settled_frames: u32) -> u32 {
        match self {
            Self::Baseline => 0,
            Self::CameraImmediate | Self::LightImmediate => 1,
            Self::CameraSettled | Self::LightSettled => settled_frames.max(1),
        }
    }
}

#[derive(Resource)]
struct RawResponseSequence {
    phase: RawResponsePhase,
    settle_frames: u32,
    frames_since_change: u32,
    last_sample: Option<Instant>,
}

#[derive(Component)]
struct ResponseCamera;

#[derive(Resource)]
struct InteractiveTransport {
    resolution_divisor: u32,
    max_receiver_cells: u32,
    max_rays: u32,
    resolve_scale: u32,
}

fn main() -> AppExit {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let benchmark = args.iter().any(|arg| arg == "--benchmark");
    let response_capture = args.iter().any(|arg| arg == "--capture-response");
    let raw_capture = args.iter().any(|arg| arg == "--capture-raw") || response_capture;
    let image_capture = args.iter().any(|arg| arg == "--capture");
    let live_capture = args.iter().any(|arg| arg == "--live-capture");
    assert!(
        !(benchmark && (raw_capture || image_capture || live_capture)
            || live_capture && (raw_capture || image_capture)),
        "choose one capture or benchmark operation"
    );

    let size = UVec2::new(
        number(&args, "--width=", if benchmark { 1920 } else { 640 }),
        number(&args, "--height=", if benchmark { 1080 } else { 480 }),
    );
    let window_size = UVec2::new(
        number(&args, "--width=", 960).max(1),
        number(&args, "--height=", 720).max(1),
    );
    let low_end = RadianceCascades::low_end();
    let requested_divisor = number(
        &args,
        "--resolution-divisor=",
        low_end.sampling.grid_divisor,
    )
    .clamp(1, 256);
    let max_receiver_cells = number(&args, "--max-receiver-cells=", 8_192).min(131_072);
    // The demo exposes receiver density and per-frame work as separate budgets.
    let requested_max_rays = number(&args, "--max-rays=", low_end.max_rays.saturating_mul(2));
    // Bound per-frame transport updates while leaving receiver-grid density as
    // an independent, separately budgeted setting.
    let max_rays = requested_max_rays.min(if raw_capture { 32_768 } else { 131_072 });
    let resolve_scale =
        number(&args, "--resolve-scale=", low_end.sampling.resolve_scale).clamp(1, 2);
    let mut budget_settings = low_end;
    budget_settings.max_rays = max_rays;
    budget_settings.sampling.max_updates_per_frame = max_rays;
    let update_budget = scheduled_update_budget(budget_settings);
    let (candidate_grid, resolution_divisor) =
        resolved_size(size, requested_divisor, max_receiver_cells);
    let (window_grid, window_grid_divisor) =
        resolved_size(window_size, requested_divisor, max_receiver_cells);
    if !benchmark && !raw_capture && !image_capture {
        info!(
            requested_width = window_size.x,
            requested_height = window_size.y,
            grid_width = window_grid.x,
            grid_height = window_grid.y,
            grid_divisor = window_grid_divisor,
            max_receiver_cells,
            scheduled_updates_per_frame = update_budget,
            resolve_scale,
            "Cornell Box interactive GI settings"
        );
    }
    // Reusing an unchanged sparse field is not a new candidate observation.
    // Wait through complete update epochs before judging capture quality.
    let warmup_frames = number(
        &args,
        "--capture-warmup-frames=",
        capture_warmup_frames(
            candidate_grid,
            update_budget,
            low_end.history_frames.max(32),
            16,
        ),
    );
    let raw_frame_count = number(&args, "--raw-frame-count=", 1).clamp(1, 512) as usize;
    let raw_frame_count = if response_capture { 1 } else { raw_frame_count };
    let image_frame_count = number(&args, "--capture-frame-count=", 1).clamp(1, 512);
    let raw_size = resolve_size(size, resolve_scale);
    let raw_path = args
        .iter()
        .find_map(|arg| arg.strip_prefix("--raw-output=").map(str::to_owned))
        .unwrap_or_else(|| "screenshots/cornell_raw.pfm".into());
    let image_path = args
        .iter()
        .find_map(|arg| arg.strip_prefix("--capture-output=").map(str::to_owned))
        .unwrap_or_else(|| "screenshots/cornell_box.png".into());
    assert!(
        !image_path.is_empty(),
        "--capture-output requires an output path"
    );
    let live_output_path = args
        .iter()
        .find_map(|arg| arg.strip_prefix("--output=").map(str::to_owned))
        .unwrap_or_else(|| "captures/cornell_live.png".into());
    let live_frame_count = number(&args, "--capture-frame-count=", 1).clamp(1, 512);
    let window_title = args
        .iter()
        .find_map(|arg| arg.strip_prefix("--window-title="))
        .filter(|title| !title.trim().is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| format!("Cornell Box | {}×{}", window_size.x, window_size.y));
    if live_capture && live_output_path.trim().is_empty() {
        error!("--output requires a file path");
        return AppExit::error();
    }
    if live_capture {
        info!(
            title = %window_title,
            output = %live_output_path,
            frames = live_frame_count,
            "Live Cornell Box capture configured"
        );
    }

    if live_capture
        && let Some(parent) = std::path::Path::new(&live_output_path)
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        && let Err(error) = std::fs::create_dir_all(parent)
    {
        error!("Could not create live capture output directory: {error}");
        return AppExit::error();
    }

    let mut app = App::new();
    app.insert_resource(DefaultOpaqueRendererMethod::deferred())
        .insert_resource(InteractiveTransport {
            resolution_divisor: requested_divisor,
            max_receiver_cells,
            max_rays,
            resolve_scale,
        })
        .insert_resource(GlobalAmbientLight {
            brightness: 0.0,
            ..default()
        });
    if response_capture {
        app.insert_resource(RawResponseSequence {
            phase: RawResponsePhase::Baseline,
            settle_frames: number(&args, "--response-settle-frames=", 16),
            frames_since_change: 0,
            last_sample: None,
        });
    }
    if live_capture {
        app.insert_resource(LiveCapture {
            output_path: live_output_path,
            frame_count: live_frame_count,
            minimum_samples: warmup_frames,
            frames_requested: 0,
            frames_completed: 0,
            started_at: None,
            last_sample: None,
            viewport: None,
        });
    }
    let plugins = DefaultPlugins.set(RenderPlugin {
        synchronous_pipeline_compilation: false,
        ..default()
    });

    if benchmark || raw_capture || image_capture {
        app.add_plugins(
            plugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .disable::<WinitPlugin>(),
        );
        app.add_plugins(ScheduleRunnerPlugin::run_loop(if benchmark {
            std::time::Duration::ZERO
        } else {
            std::time::Duration::from_secs_f64(1.0 / 60.0)
        }));

        let (target, raw_target) = {
            let mut images = app.world_mut().resource_mut::<Assets<Image>>();
            let target = images.add(Image::new_target_texture(
                size.x,
                size.y,
                TextureFormat::Rgba8UnormSrgb,
                None,
            ));
            let raw_target = raw_capture.then(|| {
                let mut image = Image::new_target_texture(
                    raw_size.x,
                    raw_size.y,
                    TextureFormat::Rgba32Float,
                    None,
                );
                image.texture_descriptor.usage |= TextureUsages::COPY_SRC | TextureUsages::COPY_DST;
                images.add(image)
            });
            (target, raw_target)
        };
        app.insert_resource(Capture {
            target,
            image_path,
            image_frame_count,
            image_frames_completed: 0,
            last_image_sample: None,
            raw_target,
            raw_path,
            viewport: size,
            raw_size,
            resolution_divisor,
            max_receiver_cells,
            max_rays,
            warmup_frames,
            resolve_scale,
            raw_frame_count,
            raw_frames_queued: 0,
            raw_frames_completed: 0,
            last_raw_sample: None,
            raw_rgb_sum: if raw_capture {
                vec![[0.0; 3]; (raw_size.x * raw_size.y) as usize]
            } else {
                Vec::new()
            },
            raw_sample_count: if raw_capture {
                vec![0; (raw_size.x * raw_size.y) as usize]
            } else {
                Vec::new()
            },
            raw_indirect_count: if raw_capture {
                vec![0; (raw_size.x * raw_size.y) as usize]
            } else {
                Vec::new()
            },
            raw_direct_count: if raw_capture {
                vec![0; (raw_size.x * raw_size.y) as usize]
            } else {
                Vec::new()
            },
            started_at: None,
            ready_frames: 0,
            last_gi_sample: None,
            requested: false,
        });
    } else {
        let primary_window = Window {
            title: window_title,
            resolution: (window_size.x, window_size.y).into(),
            ..default()
        };
        app.add_plugins(plugins.set(WindowPlugin {
            primary_window: Some(primary_window),
            ..default()
        }))
        .add_plugins(FrameTimeDiagnosticsPlugin::default());
    }

    app.add_plugins(RenderDiagnosticsPlugin);
    if benchmark {
        app.add_plugins(bevy_penumbra::FrameTimingPlugin);
        app.insert_resource(Benchmark {
            started_at: None,
            warmup_frames: number(&args, "--warmup-frames=", 300),
            measured_frames: number(&args, "--measured-frames=", 1000),
            repetitions: number(&args, "--repetitions=", 3),
            repetition: 0,
            frame: 0,
            last_sample: None,
            timings_ms: Vec::new(),
        });
    }

    app.add_plugins(RadianceCascadesPlugin::low_end())
        .add_systems(Startup, setup)
        .init_resource::<GiDebugState>()
        .add_systems(
            Update,
            (
                update_gi_debug_state,
                controls,
                benchmark_frame,
                capture_frame,
                live_capture_frame,
                update_debug_overlay,
            )
                .chain(),
        )
        .run()
}

#[derive(Component)]
struct DebugOverlay;

fn update_debug_overlay(
    diagnostics: Res<DiagnosticsStore>,
    debug: Res<GiDebugState>,
    time: Res<Time<Real>>,
    settings: Query<&RadianceCascades>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut overlay: Query<&mut Text, With<DebugOverlay>>,
) {
    let Some(settings) = settings.iter().next() else {
        return;
    };
    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|diagnostic| diagnostic.smoothed())
        .unwrap_or(0.0);
    let now = time.elapsed_secs_f64();
    let gi_status = if settings.intensity <= 0.0 {
        "GI: OFF".to_owned()
    } else if debug.samples == 0 {
        "GI: waiting for the first dispatch".to_owned()
    } else if debug.ready(now) {
        format!("GI: READY ({} rendered frames)", debug.samples)
    } else {
        format!(
            "GI: WARMING ({} / {} frames, {:.1} / 3.0 s)",
            debug.samples,
            debug.required_samples,
            debug.elapsed(now)
        )
    };
    let render_time = debug.render_ms.map_or_else(
        || "n/a".to_owned(),
        |ms| format!("{ms:.2} ms {}", debug.timing_source.unwrap_or("CPU")),
    );
    let viewport = windows.iter().next().map(|window| {
        UVec2::new(
            window.resolution.physical_width(),
            window.resolution.physical_height(),
        )
    });
    let target = viewport.map_or_else(
        || "Target: offscreen".to_owned(),
        |size| format!("Target: {}×{} physical", size.x, size.y),
    );
    let grid_divisor = if settings.sampling.grid_divisor == 0 {
        settings.resolution_divisor
    } else {
        settings.sampling.grid_divisor
    };
    let max_receiver_cells = if settings.sampling.max_receiver_cells == 0 {
        131_072
    } else {
        settings.sampling.max_receiver_cells
    };
    let (grid, grid_divisor) = viewport.map_or((UVec2::ONE, grid_divisor), |size| {
        resolved_size(size, grid_divisor, max_receiver_cells)
    });
    let grid_cells = u64::from(grid.x) * u64::from(grid.y);
    let update_budget = scheduled_update_budget(*settings);
    let details = if debug.show_details {
        format!(
            "\n{target}\nGrid limit: {}×{} /{} ({} cells)\nActive GI candidates: {}\nSeed updates/frame: {} successful / {} scheduled\nTraversal queries: {} / {} admitted\nGI render: {render_time}",
            grid.x,
            grid.y,
            grid_divisor,
            grid_cells,
            display_counter(debug.receiver_candidates),
            display_counter(debug.receiver_updates),
            update_budget,
            display_counter(debug.query_admitted),
            display_counter(debug.query_attempts),
        )
    } else {
        "\nF3: GI diagnostics".to_owned()
    };
    for mut text in &mut overlay {
        text.0 = format!(
            "{gi_status}\nFPS: {fps:.1}{details}\nP: save screenshot as cornell_box_live.png when GI is ready"
        );
    }
}

fn update_gi_debug_state(
    diagnostics: Res<DiagnosticsStore>,
    time: Res<Time<Real>>,
    settings: Query<&RadianceCascades>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut debug: ResMut<GiDebugState>,
) {
    let Some(settings) = settings.iter().next() else {
        return;
    };

    let viewport = windows.iter().next().map(|window| {
        UVec2::new(
            window.resolution.physical_width(),
            window.resolution.physical_height(),
        )
    });
    let reset = debug.history_epoch != Some(settings.history_epoch) || debug.viewport != viewport;
    if reset {
        debug.restart_warmup();
        debug.receiver_updates = None;
        debug.receiver_candidates = None;
        debug.query_attempts = None;
        debug.query_admitted = None;
        debug.history_epoch = Some(settings.history_epoch);
        debug.viewport = viewport;
    }
    debug.enabled = settings.intensity > 0.0;
    let mut new_sample = false;
    if let Some((sample, elapsed_ms, source)) = gi_render_timing(&diagnostics) {
        debug.render_ms = Some(elapsed_ms);
        debug.timing_source = Some(source);
        if debug.last_sample != Some(sample) {
            debug.last_sample = Some(sample);
            debug.samples = debug.samples.saturating_add(1);
            debug.started_at.get_or_insert(time.elapsed_secs_f64());
            new_sample = true;
        }
    } else {
        debug.render_ms = None;
        debug.timing_source = None;
    }

    if !reset || new_sample {
        debug.receiver_updates = gi_counter(&diagnostics, "/selected_receiver_updates");
        debug.receiver_candidates = gi_counter(&diagnostics, "/receiver_candidates");
        debug.query_attempts = gi_counter(&diagnostics, "/transport_query_attempts");
        debug.query_admitted = gi_counter(&diagnostics, "/transport_queries_admitted");
    }

    let history_samples = settings.history_frames.max(32);
    let scheduled_updates = u64::from(scheduled_update_budget(*settings));
    let coverage_samples = match debug.receiver_candidates {
        Some(candidates) if candidates > 0 => candidates
            .div_ceil(scheduled_updates)
            .saturating_mul(16)
            .min(u64::from(u32::MAX)) as u32,
        _ => 0,
    };
    debug.required_samples = history_samples.max(coverage_samples);
}

fn scheduled_update_budget(settings: RadianceCascades) -> u32 {
    let legacy = settings.max_rays.max(1);
    let requested = if settings.sampling.max_updates_per_frame == 0 {
        legacy
    } else {
        settings.sampling.max_updates_per_frame
    };
    // Cornell has one emissive distribution and no HDR/punctual sources. Match
    // the renderer's direct+seed share when estimating sparse-grid sweeps.
    let query_budget = if settings.sampling.max_traversal_queries == 0 {
        u32::MAX
    } else {
        settings.sampling.max_traversal_queries.div_ceil(2)
            / settings.emissive_samples.max(1).saturating_add(1)
    };
    requested.min(legacy).min(query_budget).max(1)
}

fn gi_render_timing(diagnostics: &DiagnosticsStore) -> Option<(Instant, f64, &'static str)> {
    for (path, source) in [
        ("render/split_rc/elapsed_gpu", "GPU"),
        ("render/split_rc/elapsed_cpu", "CPU"),
    ] {
        let Some(measurement) = diagnostics
            .iter()
            .find(|diagnostic| diagnostic.path().as_str() == path)
            .and_then(|diagnostic| diagnostic.measurement())
        else {
            continue;
        };
        return Some((measurement.time, measurement.value, source));
    }
    None
}

fn gi_counter(diagnostics: &DiagnosticsStore, suffix: &str) -> Option<u64> {
    diagnostics
        .iter()
        .find(|diagnostic| {
            let path = diagnostic.path().as_str();
            path.starts_with("render/penumbra/view_") && path.ends_with(suffix)
        })
        .and_then(|diagnostic| diagnostic.value())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .map(|value| value.round() as u64)
}

fn display_counter(counter: Option<u64>) -> String {
    counter.map_or_else(|| "n/a".to_owned(), |value| value.to_string())
}

fn number(args: &[String], prefix: &str, fallback: u32) -> u32 {
    args.iter()
        .find_map(|arg| {
            arg.strip_prefix(prefix)
                .and_then(|value| value.parse().ok())
        })
        .unwrap_or(fallback)
        .max(1)
}

fn resolved_size(viewport: UVec2, requested_divisor: u32, max_receiver_cells: u32) -> (UVec2, u32) {
    let size_at = |divisor: u32| {
        UVec2::new(
            viewport.x.div_ceil(divisor).max(1),
            viewport.y.div_ceil(divisor).max(1),
        )
    };
    let budget = u64::from(max_receiver_cells.max(1));
    let mut low = requested_divisor.max(1);
    let mut high = viewport.max_element().max(low);
    while low < high {
        let middle = low + (high - low) / 2;
        let size = size_at(middle);
        if u64::from(size.x) * u64::from(size.y) > budget {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    (size_at(low), low)
}

fn capture_image_path(path: &str, frame: u32, frame_count: u32) -> String {
    if frame_count == 1 {
        return path.to_owned();
    }
    let path = std::path::Path::new(path);
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("cornell_box");
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("png");
    path.with_file_name(format!("{stem}_{frame:03}.{extension}"))
        .to_string_lossy()
        .into_owned()
}

fn capture_warmup_frames(
    candidate_grid: UVec2,
    updates_per_frame: u32,
    minimum_frames: u32,
    observations_per_candidate: u32,
) -> u32 {
    let candidates = u64::from(candidate_grid.x) * u64::from(candidate_grid.y);
    let epoch_frames = candidates.div_ceil(u64::from(updates_per_frame.max(1)));
    let converged_frames =
        epoch_frames.saturating_mul(u64::from(observations_per_candidate.max(1)));
    converged_frames
        .max(u64::from(minimum_frames))
        .min(u64::from(u32::MAX)) as u32
}

fn resolve_size(viewport: UVec2, scale: u32) -> UVec2 {
    let scale = scale.max(1);
    UVec2::new(viewport.x.div_ceil(scale), viewport.y.div_ceil(scale)).max(UVec2::ONE)
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    capture: Option<Res<Capture>>,
    response: Option<Res<RawResponseSequence>>,
    interactive_transport: Res<InteractiveTransport>,
) {
    if capture.is_none() {
        commands.spawn((
            DebugOverlay,
            Text::new("GI: waiting for the first dispatch\nF3: GI diagnostics | P: screenshot when GI is ready"),
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
    let white = materials.add(StandardMaterial {
        base_color: Color::srgb(0.8, 0.8, 0.8),
        perceptual_roughness: 1.0,
        ..default()
    });
    let red = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.12, 0.08),
        perceptual_roughness: 1.0,
        ..default()
    });
    let green = materials.add(StandardMaterial {
        base_color: Color::srgb(0.1, 0.7, 0.2),
        perceptual_roughness: 1.0,
        ..default()
    });
    let emitter = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        // This demo has no direct/static light; the ceiling panel must carry
        // enough radiance to light the room through Split Radiance Cascades.
        emissive: LinearRgba::new(30.0, 25.0, 18.0, 1.0),
        ..default()
    });

    // Open-front Cornell room, two diffuse blocks and a small ceiling emitter.
    for (size, position, material) in [
        (
            Vec3::new(6.0, 0.1, 6.0),
            Vec3::new(0.0, -0.05, 0.0),
            white.clone(),
        ),
        (
            Vec3::new(6.0, 4.0, 0.1),
            Vec3::new(0.0, 2.0, -3.05),
            white.clone(),
        ),
        (
            Vec3::new(6.0, 0.1, 6.0),
            Vec3::new(0.0, 4.05, 0.0),
            white.clone(),
        ),
        (Vec3::new(0.1, 4.0, 6.0), Vec3::new(-3.05, 2.0, 0.0), red),
        (Vec3::new(0.1, 4.0, 6.0), Vec3::new(3.05, 2.0, 0.0), green),
        (
            Vec3::new(2.0, 0.05, 2.0),
            Vec3::new(0.0, 3.97, -0.8),
            emitter,
        ),
    ] {
        let ceiling_emitter = size.y < 0.1 && position.y > 3.9;
        let mesh = if ceiling_emitter {
            meshes.add(Rectangle::new(size.x, size.z))
        } else {
            meshes.add(Cuboid::from_size(size))
        };
        let transform = if ceiling_emitter {
            // Emit into the room only. A glowing box also emits upward across
            // the tiny ceiling gap, producing noisy near-contact light samples.
            Transform::from_translation(position)
                .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2))
        } else {
            Transform::from_translation(position)
        };
        let mut entity = commands.spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material),
            transform,
            RadianceCascadesGeometry,
        ));
        if ceiling_emitter {
            entity.insert(LightMover);
        }
    }
    for (size, position, rotation) in [
        (Vec3::new(1.4, 1.5, 1.4), Vec3::new(-1.1, 0.75, 0.0), 0.3),
        (Vec3::new(1.2, 2.8, 1.2), Vec3::new(1.0, 1.4, -1.0), -0.3),
    ] {
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::from_size(size))),
            MeshMaterial3d(white.clone()),
            Transform::from_translation(position).with_rotation(Quat::from_rotation_y(rotation)),
            RadianceCascadesGeometry,
        ));
    }

    let mut camera = commands.spawn((
        Camera3d::default(),
        ShadowLodOrigin,
        Msaa::Off,
        Camera {
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        Exposure {
            ev100: CORNELL_CAMERA_EV100,
        },
        Tonemapping::Linear,
        {
            let mut settings = RadianceCascades::low_end();
            settings.intensity = 1.0;
            settings.resolution_divisor = capture
                .as_ref()
                .map_or(interactive_transport.resolution_divisor, |capture| {
                    capture.resolution_divisor
                });
            settings.sampling.grid_divisor = settings.resolution_divisor;
            settings.sampling.max_receiver_cells = capture
                .as_ref()
                .map_or(interactive_transport.max_receiver_cells, |capture| {
                    capture.max_receiver_cells
                });
            settings.sampling.resolve_scale = capture
                .as_ref()
                .map_or(interactive_transport.resolve_scale, |capture| {
                    capture.resolve_scale
                });
            settings.max_rays = capture
                .as_ref()
                .map_or(interactive_transport.max_rays, |capture| capture.max_rays);
            settings.sampling.max_updates_per_frame = settings.max_rays;
            settings
        },
        Transform::from_xyz(0.0, 2.0, 8.5).looking_at(Vec3::new(0.0, 1.8, -0.5), Vec3::Y),
    ));
    if response.is_some() {
        camera.insert(ResponseCamera);
    }
    if let Some(capture) = capture {
        camera.insert(RenderTarget::Image(capture.target.clone().into()));
        if let Some(raw) = &capture.raw_target {
            camera.insert(RadianceCascadesRawOutput(raw.clone()));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn controls(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    capture: Option<Res<Capture>>,
    mut gi_state: ResMut<GiDebugState>,
    mut cameras: Query<&mut RadianceCascades>,
    mut lights: Query<&mut Transform, With<LightMover>>,
    mut commands: Commands,
    mut exit: MessageWriter<AppExit>,
) {
    if capture.is_some() {
        return;
    }
    if keys.just_pressed(KeyCode::Escape) {
        exit.write(AppExit::Success);
    }
    if keys.just_pressed(KeyCode::F3) {
        gi_state.show_details = !gi_state.show_details;
    }
    let direction =
        i32::from(keys.pressed(KeyCode::ArrowRight)) - i32::from(keys.pressed(KeyCode::ArrowLeft));
    let mut history_reset = direction != 0;
    for mut settings in &mut cameras {
        if keys.just_pressed(KeyCode::KeyG) {
            settings.intensity = if settings.intensity > 0.0 { 0.0 } else { 1.0 };
            settings.reset_history();
            history_reset = true;
        }
    }
    if history_reset {
        gi_state.restart_warmup();
    }
    if direction != 0 {
        for mut transform in &mut lights {
            transform.translation.x += direction as f32 * time.delta_secs() * 2.0;
        }
    }
    if keys.just_pressed(KeyCode::KeyP) {
        if direction == 0 && gi_state.ready(time.elapsed_secs_f64()) {
            commands
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk("cornell_box_live.png"));
        } else {
            info!(
                samples = gi_state.samples,
                required = gi_state.required_samples,
                elapsed_seconds = gi_state.elapsed(time.elapsed_secs_f64()),
                "GI screenshot skipped while the transport history is warming"
            );
        }
    }
}

fn live_capture_frame(
    mut commands: Commands,
    time: Res<Time<Real>>,
    debug: Res<GiDebugState>,
    live_capture: Option<ResMut<LiveCapture>>,
    mut exit: MessageWriter<AppExit>,
) {
    let Some(mut live_capture) = live_capture else {
        return;
    };
    let now = time.elapsed_secs_f64();
    let started_at = *live_capture.started_at.get_or_insert(now);
    if now >= started_at + 600.0 {
        error!("Live capture timed out before GI became ready");
        exit.write(AppExit::error());
        return;
    }
    if live_capture.frames_requested > 0
        && live_capture.viewport.is_some()
        && debug.viewport.is_some()
        && live_capture.viewport != debug.viewport
    {
        error!("Live capture stopped because the window was resized during a sequence");
        exit.write(AppExit::error());
        return;
    }
    if live_capture.frames_requested >= live_capture.frame_count
        || !debug.ready(now)
        || debug.samples < live_capture.minimum_samples
    {
        return;
    }
    let Some(sample) = debug.last_sample else {
        return;
    };
    if live_capture.last_sample == Some(sample) {
        return;
    }
    let Some(viewport) = debug.viewport else {
        error!("Live capture requires a primary window");
        exit.write(AppExit::error());
        return;
    };

    live_capture.viewport.get_or_insert(viewport);
    live_capture.last_sample = Some(sample);
    let output_path = capture_image_path(
        &live_capture.output_path,
        live_capture.frames_requested + 1,
        live_capture.frame_count,
    );
    let gi_samples = debug.samples;
    let receiver_candidates = debug.receiver_candidates;
    let successful_receiver_seeds = debug.receiver_updates;
    let gi_render_ms = debug.render_ms;
    let timing_source = debug.timing_source;
    info!(
        frame = live_capture.frames_requested + 1,
        total = live_capture.frame_count,
        width = viewport.x,
        height = viewport.y,
        gi_samples,
        ?receiver_candidates,
        ?successful_receiver_seeds,
        ?gi_render_ms,
        ?timing_source,
        output = %output_path,
        "Capturing ready live Cornell Box frame"
    );
    live_capture.frames_requested = live_capture.frames_requested.saturating_add(1);
    commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(output_path))
        .observe(live_capture_saved);
}

fn live_capture_saved(
    _event: On<ScreenshotCaptured>,
    mut live_capture: ResMut<LiveCapture>,
    mut exit: MessageWriter<AppExit>,
) {
    live_capture.frames_completed = live_capture.frames_completed.saturating_add(1);
    if live_capture.frames_completed >= live_capture.frame_count {
        info!(
            frames = live_capture.frames_completed,
            "Live Cornell Box capture complete"
        );
        exit.write(AppExit::Success);
    }
}

fn benchmark_frame(
    time: Res<Time<Real>>,
    diagnostics: Res<DiagnosticsStore>,
    benchmark: Option<ResMut<Benchmark>>,
    mut exit: MessageWriter<AppExit>,
) {
    let Some(mut benchmark) = benchmark else {
        return;
    };
    if let Some(measurement) = diagnostics
        .iter()
        .find(|diagnostic| diagnostic.path().as_str() == "render/split_rc/elapsed_gpu")
        .and_then(|diagnostic| diagnostic.measurement())
        && benchmark.last_sample != Some(measurement.time)
    {
        benchmark.last_sample = Some(measurement.time);
        if benchmark.started_at.is_none() {
            benchmark.started_at = Some(time.elapsed_secs_f64());
            info!(
                "Split Radiance Cascades benchmark: {} warmup and {} measured completed GPU samples, {} repetitions",
                benchmark.warmup_frames, benchmark.measured_frames, benchmark.repetitions
            );
        }
        if benchmark.frame < benchmark.warmup_frames {
            benchmark.frame += 1;
        } else {
            benchmark.timings_ms.push(measurement.value);
        }
    }
    if benchmark.timings_ms.len() < benchmark.measured_frames as usize {
        if benchmark
            .started_at
            .is_some_and(|started| time.elapsed_secs_f64() > started + 600.0)
        {
            error!("Benchmark timed out waiting for Split Radiance Cascades GPU samples");
            exit.write(AppExit::error());
        }
        return;
    }

    benchmark.timings_ms.sort_by(f64::total_cmp);
    let samples = &benchmark.timings_ms;
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let p50 = samples[samples.len() / 2];
    let p95 = samples[(samples.len() * 95 / 100).min(samples.len() - 1)];
    let report = format!(
        "Split Radiance Cascades GPU time, repetition {}/{}: n={} mean={mean:.3} ms p50={p50:.3} ms p95={p95:.3} ms max={:.3} ms\n",
        benchmark.repetition + 1,
        benchmark.repetitions,
        samples.len(),
        samples[samples.len() - 1],
    );
    let path = format!("tmp/cornell_src_benchmark_{}.txt", benchmark.repetition + 1);
    if let Err(error) = std::fs::create_dir_all("tmp").and_then(|_| std::fs::write(&path, &report))
    {
        error!("Could not save benchmark: {error}");
        exit.write(AppExit::error());
        return;
    }
    info!("{report}");
    for diagnostic in diagnostics
        .iter()
        .filter(|d| d.path().as_str().ends_with("elapsed_gpu"))
    {
        if let Some(value) = diagnostic.value() {
            info!(path = %diagnostic.path(), gpu_ms = value, "Cornell GPU stage");
        }
    }
    benchmark.repetition += 1;
    if benchmark.repetition >= benchmark.repetitions {
        exit.write(AppExit::Success);
    } else {
        benchmark.frame = 0;
        benchmark.started_at = None;
        benchmark.last_sample = None;
        benchmark.timings_ms.clear();
    }
}

#[allow(clippy::too_many_arguments)]
fn capture_frame(
    mut commands: Commands,
    time: Res<Time<Real>>,
    diagnostics: Res<DiagnosticsStore>,
    camera_settings: Query<&RadianceCascades>,
    capture: Option<ResMut<Capture>>,
    response: Option<ResMut<RawResponseSequence>>,
    benchmark: Option<Res<Benchmark>>,
    mut exit: MessageWriter<AppExit>,
) {
    let Some(mut capture) = capture else {
        return;
    };
    if benchmark.is_some()
        || (capture.raw_target.is_none()
            && (capture.requested || capture.image_frames_completed >= capture.image_frame_count))
        || capture.raw_frames_queued >= capture.raw_frame_count
    {
        return;
    }
    let measurement = gi_render_timing(&diagnostics);
    let Some((measurement_time, _, _)) = measurement else {
        if time.elapsed_secs_f64() > 600.0 {
            error!("Capture timed out waiting for GI output");
            exit.write(AppExit::error());
        }
        return;
    };
    let new_sample = capture.last_gi_sample != Some(measurement_time);
    if new_sample {
        capture.last_gi_sample = Some(measurement_time);
        capture.ready_frames += 1;
    }
    let started_at = *capture.started_at.get_or_insert(time.elapsed_secs_f64());
    let minimum_samples = camera_settings
        .iter()
        .map(|settings| settings.history_frames)
        .max()
        .unwrap_or_default()
        .max(32);
    let required_samples = capture.warmup_frames.max(minimum_samples);
    let mut response = response;
    let response_ready = if let Some(response) = response.as_mut() {
        if response.phase == RawResponsePhase::Baseline {
            capture.ready_frames >= required_samples && time.elapsed_secs_f64() >= started_at + 3.0
        } else {
            if response.last_sample != Some(measurement_time) {
                response.last_sample = Some(measurement_time);
                response.frames_since_change = response.frames_since_change.saturating_add(1);
            }
            response.frames_since_change
                >= response.phase.frames_until_capture(response.settle_frames)
        }
    } else {
        capture.ready_frames >= required_samples && time.elapsed_secs_f64() >= started_at + 3.0
    };
    if !response_ready {
        return;
    }
    if capture.raw_target.is_some() && capture.last_raw_sample == Some(measurement_time) {
        return;
    }
    if capture.raw_target.is_none() && capture.last_image_sample == Some(measurement_time) {
        return;
    }
    if capture.raw_frames_queued == 0 && capture.image_frames_completed == 0 && !capture.requested {
        for diagnostic in diagnostics.iter().filter(|diagnostic| {
            let path = diagnostic.path().as_str();
            path.starts_with("render/penumbra/view_")
                && [
                    "/transport_query_attempts",
                    "/transport_queries_admitted",
                    "/selected_receiver_updates",
                    "/receiver_candidates",
                ]
                .iter()
                .any(|suffix| path.ends_with(suffix))
        }) {
            if let Some(value) = diagnostic.value() {
                info!(path = %diagnostic.path(), value, "GI capture work counters");
            }
        }
    }
    let response_phase = response.as_ref().map(|response| response.phase);
    let response_path = response_phase.map(|phase| phase.capture_path(&capture.raw_path));
    let output_path = capture
        .raw_target
        .as_ref()
        .map_or(capture.image_path.as_str(), |_| {
            response_path.as_deref().unwrap_or(&capture.raw_path)
        })
        .to_owned();
    if let Some(parent) = std::path::Path::new(&output_path)
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        && let Err(error) = std::fs::create_dir_all(parent)
    {
        error!("Could not create capture output directory: {error}");
        exit.write(AppExit::error());
        return;
    }
    if let Some(raw) = capture.raw_target.clone() {
        capture.raw_frames_queued += 1;
        capture.last_raw_sample = Some(measurement_time);
        commands
            .spawn((
                Readback::texture(raw),
                ReadbackOnce,
                RawCaptureOutput {
                    path: output_path.clone(),
                    size: capture.raw_size,
                    viewport: capture.viewport,
                    camera_exposure: Exposure {
                        ev100: CORNELL_CAMERA_EV100,
                    }
                    .exposure(),
                    resolution_divisor: capture.resolution_divisor,
                    max_receiver_cells: capture.max_receiver_cells,
                    max_rays: capture.max_rays,
                    resolve_scale: capture.resolve_scale,
                    response_phase,
                },
            ))
            .observe(save_raw_capture);
    } else {
        capture.requested = true;
        capture.last_image_sample = Some(measurement_time);
        let path = capture_image_path(
            &capture.image_path,
            capture.image_frames_completed,
            capture.image_frame_count,
        );
        commands
            .spawn(Screenshot::image(capture.target.clone()))
            .observe(save_to_disk(path))
            .observe(
                |captured: On<ScreenshotCaptured>,
                 mut capture: ResMut<Capture>,
                 mut exit: MessageWriter<AppExit>| {
                    let Ok(image) = captured.image.clone().try_into_dynamic() else {
                        exit.write(AppExit::error());
                        return;
                    };
                    let image = image.to_rgb8();
                    let mut lit = 0_u64;
                    let mut total = 0_u64;
                    for y in image.height() / 2..image.height() * 3 / 4 {
                        for x in image.width() / 4..image.width() * 3 / 4 {
                            total += 1;
                            lit +=
                                u64::from(image.get_pixel(x, y).0.iter().any(|&value| value > 8));
                        }
                    }
                    if lit == 0 {
                        error!("Cornell SRC capture has no lit receiver pixels");
                        exit.write(AppExit::error());
                    } else {
                        capture.image_frames_completed += 1;
                        capture.requested = false;
                        info!(
                            frame = capture.image_frames_completed,
                            frames = capture.image_frame_count,
                            "Cornell SRC capture: {lit}/{total} sampled receiver pixels are lit"
                        );
                        if capture.image_frames_completed >= capture.image_frame_count {
                            exit.write(AppExit::Success);
                        }
                    }
                },
            );
    }
}

#[allow(clippy::too_many_arguments)]
fn save_raw_capture(
    event: On<ReadbackComplete>,
    output: Query<&RawCaptureOutput>,
    mut capture: ResMut<Capture>,
    response: Option<ResMut<RawResponseSequence>>,
    mut response_camera: Query<&mut Transform, With<ResponseCamera>>,
    light_material: Query<&MeshMaterial3d<StandardMaterial>, With<LightMover>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut exit: MessageWriter<AppExit>,
) {
    let Ok(output) = output.get(event.entity) else {
        exit.write(AppExit::error());
        return;
    };
    let width = output.size.x as usize;
    let height = output.size.y as usize;
    let stride = (width * 16).div_ceil(256) * 256;
    if event.data.len() != stride * height {
        error!("Raw GI readback had an unexpected padded size");
        exit.write(AppExit::error());
        return;
    }

    if capture.raw_rgb_sum.len() != width * height
        || capture.raw_sample_count.len() != width * height
        || capture.raw_indirect_count.len() != width * height
        || capture.raw_direct_count.len() != width * height
    {
        error!("Raw GI accumulation dimensions changed during capture");
        exit.write(AppExit::error());
        return;
    }
    let mut pixel_index = 0;
    for row in event.data.chunks_exact(stride) {
        for pixel in row[..width * 16].as_chunks::<16>().0 {
            let rgba: [f32; 4] = std::array::from_fn(|index| {
                f32::from_le_bytes(pixel[index * 4..index * 4 + 4].try_into().unwrap())
            });
            if !rgba.iter().all(|value| value.is_finite()) || rgba[3] < 0.0 {
                error!("Raw GI readback contains invalid values");
                exit.write(AppExit::error());
                return;
            }
            if rgba[3] > 0.0 {
                let sum = &mut capture.raw_rgb_sum[pixel_index];
                for (channel, value) in rgba[..3].iter().copied().enumerate() {
                    sum[channel] += f64::from(value);
                }
                capture.raw_sample_count[pixel_index] =
                    capture.raw_sample_count[pixel_index].saturating_add(1);
                // The low two alpha bits encode direct/indirect support. The
                // fractional part (bounded to 0.25) reports fresh samples.
                let support = rgba[3].floor() as u32;
                let freshness = rgba[3] - support as f32;
                if freshness > 0.2501 || support & !3 != 0 {
                    error!(
                        "Raw GI support alpha is not a 2-bit mask with bounded freshness: {}",
                        rgba[3]
                    );
                    exit.write(AppExit::error());
                    return;
                }
                if support & 1 != 0 {
                    capture.raw_indirect_count[pixel_index] =
                        capture.raw_indirect_count[pixel_index].saturating_add(1);
                }
                if support & 2 != 0 {
                    capture.raw_direct_count[pixel_index] =
                        capture.raw_direct_count[pixel_index].saturating_add(1);
                }
            }
            pixel_index += 1;
        }
    }

    capture.raw_frames_completed += 1;
    if capture.raw_frames_completed < capture.raw_frame_count {
        return;
    }

    let mut rgb = Vec::with_capacity(width * height * 12);
    let mut mask = Vec::with_capacity(width * height);
    let mut sample_counts = Vec::with_capacity(width * height);
    let mut indirect_mask = Vec::with_capacity(width * height);
    let mut direct_mask = Vec::with_capacity(width * height);
    let mut direct_only_mask = Vec::with_capacity(width * height);
    let mut receivers = 0_usize;
    let mut lit_receivers = 0_usize;
    let mut indirect_pixels = 0_usize;
    let mut direct_pixels = 0_usize;
    let mut direct_only_pixels = 0_usize;
    let mut sample_total = 0_u64;
    let mut indirect_observations = 0_u64;
    let mut direct_observations = 0_u64;
    let mut max_samples = 0_u32;
    let mut scene_linear_rgb_sum = [0.0_f64; 3];
    for (pixel_index, (sum, &count)) in capture
        .raw_rgb_sum
        .iter()
        .zip(&capture.raw_sample_count)
        .enumerate()
    {
        let indirect_count = capture.raw_indirect_count[pixel_index];
        let direct_count = capture.raw_direct_count[pixel_index];
        let mean = if count > 0 {
            receivers += 1;
            sample_total += u64::from(count);
            max_samples = max_samples.max(count);
            let mean = sum.map(|value| (value / f64::from(count)) as f32);
            lit_receivers += usize::from(mean.iter().any(|&value| value > 1e-6));
            for (channel, value) in mean.iter().copied().enumerate() {
                scene_linear_rgb_sum[channel] += f64::from(value);
            }
            mean
        } else {
            [0.0; 3]
        };
        mask.push(if count > 0 { 255 } else { 0 });
        indirect_mask.push(if indirect_count > 0 { 255 } else { 0 });
        direct_mask.push(if direct_count > 0 { 255 } else { 0 });
        direct_only_mask.push(if direct_count > 0 && indirect_count == 0 {
            255
        } else {
            0
        });
        indirect_pixels += usize::from(indirect_count > 0);
        direct_pixels += usize::from(direct_count > 0);
        direct_only_pixels += usize::from(direct_count > 0 && indirect_count == 0);
        indirect_observations += u64::from(indirect_count);
        direct_observations += u64::from(direct_count);
        sample_counts.push(count.min(255) as u8);
        for value in mean {
            rgb.extend_from_slice(&value.to_le_bytes());
        }
    }
    if receivers == 0 {
        error!("Raw GI readback contains no receiver pixels");
        exit.write(AppExit::error());
        return;
    }

    let mut pfm = format!("PF\n{width} {height}\n-1.0\n").into_bytes();
    for row in rgb.chunks_exact(width * 12).rev() {
        pfm.extend_from_slice(row);
    }
    let total = width * height;
    let coverage = receivers as f64 / total as f64;
    let mean_samples = sample_total as f64 / receivers as f64;
    let mean_scene_linear_rgb_all = scene_linear_rgb_sum.map(|sum| sum / total as f64);
    let mean_scene_linear_rgb_supported = scene_linear_rgb_sum.map(|sum| sum / receivers as f64);
    let camera_x_offset = match output.response_phase {
        Some(
            RawResponsePhase::CameraImmediate
            | RawResponsePhase::CameraSettled
            | RawResponsePhase::LightImmediate
            | RawResponsePhase::LightSettled,
        ) => 0.25,
        _ => 0.0,
    };
    let emitter_scale = match output.response_phase {
        Some(RawResponsePhase::LightImmediate | RawResponsePhase::LightSettled) => 2.0,
        _ => 1.0,
    };
    let path = std::path::Path::new(&output.path);
    let result = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .map_or(Ok(()), std::fs::create_dir_all)
        .and_then(|()| {
            std::fs::write(path, pfm)?;
            let mut pgm = format!("P5\n{width} {height}\n255\n").into_bytes();
            pgm.extend_from_slice(&mask);
            std::fs::write(path.with_extension("mask.pgm"), pgm)?;
            let mut sample_pgm = format!("P5\n{width} {height}\n255\n").into_bytes();
            sample_pgm.extend_from_slice(&sample_counts);
            std::fs::write(path.with_extension("samples.pgm"), sample_pgm)?;
            for (extension, values) in [
                ("indirect.pgm", &indirect_mask),
                ("direct.pgm", &direct_mask),
                ("direct_only.pgm", &direct_only_mask),
            ] {
                let mut component_pgm = format!("P5\n{width} {height}\n255\n").into_bytes();
                component_pgm.extend_from_slice(values);
                std::fs::write(path.with_extension(extension), component_pgm)?;
            }
            std::fs::write(
                path.with_extension("json"),
                format!(
                    "{{\n  \"algorithm\": {:?},\n  \"response_phase\": {:?},\n  \"camera_x_offset\": {camera_x_offset:.2},\n  \"emitter_emissive_scale\": {emitter_scale:.1},\n  \"units\": \"PFM RGB is scene-linear diffuse transport divided by pi, before receiver albedo and camera exposure\",\n  \"camera_exposure\": {:.8},\n  \"scene_linear_rgb_scale\": 1.0,\n  \"mean_scene_linear_rgb_all_pixels\": [{:.8}, {:.8}, {:.8}],\n  \"mean_scene_linear_rgb_supported_pixels\": [{:.8}, {:.8}, {:.8}],\n  \"viewport\": [{}, {}],\n  \"gi_size\": [{width}, {height}],\n  \"resolution_divisor\": {},\n  \"max_receiver_cells\": {},\n  \"max_updates_per_frame\": {},\n  \"resolve_scale\": {},\n  \"captured_frames\": {},\n  \"supported_pixels\": {receivers},\n  \"pixel_count\": {total},\n  \"support_coverage\": {coverage:.8},\n  \"indirect_supported_pixels\": {indirect_pixels},\n  \"direct_supported_pixels\": {direct_pixels},\n  \"direct_only_pixels\": {direct_only_pixels},\n  \"indirect_observations\": {indirect_observations},\n  \"direct_observations\": {direct_observations},\n  \"mean_observations_per_supported_pixel\": {mean_samples:.4},\n  \"max_observations_per_pixel\": {max_samples},\n  \"lit_supported_pixels\": {lit_receivers},\n  \"mask_file\": {:?},\n  \"sample_count_file\": {:?},\n  \"indirect_mask_file\": {:?},\n  \"direct_mask_file\": {:?},\n  \"direct_only_mask_file\": {:?}\n}}\n",
                    "Split Radiance Cascades",
                    output.response_phase.map_or("static", RawResponsePhase::suffix),
                    output.camera_exposure,
                    mean_scene_linear_rgb_all[0],
                    mean_scene_linear_rgb_all[1],
                    mean_scene_linear_rgb_all[2],
                    mean_scene_linear_rgb_supported[0],
                    mean_scene_linear_rgb_supported[1],
                    mean_scene_linear_rgb_supported[2],
                    output.viewport.x,
                    output.viewport.y,
                    output.resolution_divisor,
                    output.max_receiver_cells,
                    output.max_rays,
                    output.resolve_scale,
                    capture.raw_frames_completed,
                    path.with_extension("mask.pgm").to_string_lossy(),
                    path.with_extension("samples.pgm").to_string_lossy(),
                    path.with_extension("indirect.pgm").to_string_lossy(),
                    path.with_extension("direct.pgm").to_string_lossy(),
                    path.with_extension("direct_only.pgm").to_string_lossy(),
                ),
            )
        });
    if let Err(error) = result {
        error!("Could not save raw GI output: {error}");
        exit.write(AppExit::error());
    } else {
        info!(
            path = %output.path,
            response_phase = output.response_phase.map_or("static", RawResponsePhase::suffix),
            ?mean_scene_linear_rgb_all,
            ?mean_scene_linear_rgb_supported,
            "Saved {receivers} raw GI receiver pixels at {width}x{height}; {lit_receivers} are illuminated across {} samples",
            capture.raw_frames_completed
        );
        if let Some(mut response) = response {
            if output.response_phase != Some(response.phase) {
                error!("Raw response phase changed before its readback completed");
                exit.write(AppExit::error());
                return;
            }
            match response.phase {
                RawResponsePhase::Baseline => {
                    let Some(mut transform) = response_camera.iter_mut().next() else {
                        error!("Response capture camera was not found");
                        exit.write(AppExit::error());
                        return;
                    };
                    transform.translation.x += 0.25;
                    response.phase = RawResponsePhase::CameraImmediate;
                }
                RawResponsePhase::CameraImmediate => {
                    response.phase = RawResponsePhase::CameraSettled;
                }
                RawResponsePhase::CameraSettled => {
                    let Some(material_handle) = light_material.iter().next() else {
                        error!("Response capture emitter was not found");
                        exit.write(AppExit::error());
                        return;
                    };
                    let Some(mut material) = materials.get_mut(&material_handle.0) else {
                        error!("Response capture emitter material was not found");
                        exit.write(AppExit::error());
                        return;
                    };
                    material.emissive *= 2.0;
                    response.phase = RawResponsePhase::LightImmediate;
                }
                RawResponsePhase::LightImmediate => {
                    response.phase = RawResponsePhase::LightSettled;
                }
                RawResponsePhase::LightSettled => {
                    info!("Cornell response capture sequence complete");
                    exit.write(AppExit::Success);
                    return;
                }
            }
            response.frames_since_change = 0;
            response.last_sample = capture.last_gi_sample;
            capture.raw_frames_queued = 0;
            capture.raw_frames_completed = 0;
            capture.raw_rgb_sum.fill([0.0; 3]);
            capture.raw_sample_count.fill(0);
            capture.raw_indirect_count.fill(0);
            capture.raw_direct_count.fill(0);
            info!(
                phase = response.phase.suffix(),
                "Advancing Cornell GI response capture"
            );
        } else {
            exit.write(AppExit::Success);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Resource, Default)]
    struct ControlsChangeSnapshot {
        light_changed: bool,
        settings_changed: bool,
        light_x: f32,
    }

    fn snapshot_controls_changes(
        lights: Query<Ref<Transform>, With<LightMover>>,
        cameras: Query<Ref<RadianceCascades>>,
        mut snapshot: ResMut<ControlsChangeSnapshot>,
    ) {
        let light = lights.iter().next().expect("test light exists");
        snapshot.light_changed = light.is_changed();
        snapshot.light_x = light.translation.x;
        let settings = cameras.iter().next().expect("test camera exists");
        snapshot.settings_changed = settings.is_changed();
    }

    #[test]
    fn idle_controls_do_not_mark_light_or_gi_settings_changed() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_message::<AppExit>()
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<GiDebugState>()
            .init_resource::<ControlsChangeSnapshot>()
            .insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
                std::time::Duration::from_millis(16),
            ))
            .add_systems(Update, (controls, snapshot_controls_changes).chain());
        app.world_mut().spawn((LightMover, Transform::default()));
        app.world_mut().spawn(RadianceCascades::low_end());

        // The first tick establishes the initial component change ticks.
        app.update();
        // A second idle tick must not make either component look edited.
        app.update();
        let idle = app.world().resource::<ControlsChangeSnapshot>();
        assert!(!idle.light_changed);
        assert!(!idle.settings_changed);
        let idle_x = idle.light_x;

        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::ArrowRight);
        app.update();
        let moving = app.world().resource::<ControlsChangeSnapshot>();
        assert!(moving.light_changed);
        assert!(moving.light_x > idle_x);
        assert!(!moving.settings_changed);
        let moved_x = moving.light_x;

        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .release(KeyCode::ArrowRight);
        app.update();
        let released = app.world().resource::<ControlsChangeSnapshot>();
        assert!(!released.light_changed);
        assert!(!released.settings_changed);
        assert_eq!(released.light_x, moved_x);
    }

    #[test]
    fn receiver_density_is_independent_of_per_frame_update_budget() {
        let viewport = UVec2::new(1280, 720);
        let (full_low_end_grid, divisor) = resolved_size(viewport, 8, 16_384);
        assert_eq!(divisor, 8);
        assert_eq!(full_low_end_grid, UVec2::new(160, 90));
        assert!(full_low_end_grid.x * full_low_end_grid.y > 1_024);

        let (legacy_matched_grid, divisor) = resolved_size(viewport, 8, 1_024);
        assert_eq!(divisor, 31);
        assert_eq!(legacy_matched_grid, UVec2::new(42, 24));
        assert!(legacy_matched_grid.x * legacy_matched_grid.y <= 1_024);
    }

    #[test]
    fn raw_capture_target_uses_configured_resolve_scale() {
        let viewport = UVec2::new(1280, 720);
        assert_eq!(resolve_size(viewport, 2), UVec2::new(640, 360));
        assert_eq!(resolve_size(viewport, 1), viewport);
    }

    #[test]
    fn capture_warmup_waits_for_complete_sparse_observation_epochs() {
        let (grid, _) = resolved_size(UVec2::new(1195, 819), 8, 16_384);
        assert_eq!(grid, UVec2::new(150, 103));
        assert_eq!(capture_warmup_frames(grid, 1_024, 32, 16), 256);
        assert_eq!(capture_warmup_frames(grid, 20_000, 32, 16), 32);
    }

    #[test]
    fn response_capture_names_and_sample_gates_are_distinct() {
        let base = "captures/cornell_split.pfm";
        assert_eq!(
            std::path::Path::new(&RawResponsePhase::Baseline.capture_path(base))
                .file_name()
                .unwrap(),
            "cornell_split_baseline.pfm"
        );
        assert_eq!(
            std::path::Path::new(&RawResponsePhase::CameraImmediate.capture_path(base))
                .file_name()
                .unwrap(),
            "cornell_split_camera_1.pfm"
        );
        assert_eq!(
            std::path::Path::new(&RawResponsePhase::LightSettled.capture_path(base))
                .file_name()
                .unwrap(),
            "cornell_split_light_settled.pfm"
        );
        assert_eq!(RawResponsePhase::Baseline.frames_until_capture(16), 0);
        assert_eq!(
            RawResponsePhase::CameraImmediate.frames_until_capture(16),
            1
        );
        assert_eq!(RawResponsePhase::CameraSettled.frames_until_capture(16), 16);
        assert_eq!(RawResponsePhase::LightImmediate.frames_until_capture(16), 1);
        assert_eq!(RawResponsePhase::LightSettled.frames_until_capture(16), 16);
    }
}

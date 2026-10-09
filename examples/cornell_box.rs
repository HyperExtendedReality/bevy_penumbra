//! Cornell-box Split Radiance Cascades demo.
//!
//! Run interactively with `cargo run --example cornell_box`. Press G to toggle
//! GI, B to toggle cascade feedback, the arrow keys to move the ceiling panel,
//! P to save a screenshot, and Escape to exit. `--capture` saves a headless PNG,
//! `--capture-raw` writes the raw linear GI estimate as PFM, and `--benchmark`
//! records completed Split Radiance Cascades GPU timings.

use bevy::{
    app::{AppExit, ScheduleRunnerPlugin},
    camera::{Exposure, RenderTarget, ShadowLodOrigin},
    core_pipeline::tonemapping::Tonemapping,
    diagnostic::DiagnosticsStore,
    pbr::DefaultOpaqueRendererMethod,
    platform::time::Instant,
    prelude::*,
    render::{
        RenderPlugin,
        diagnostic::RenderDiagnosticsPlugin,
        gpu_readback::{Readback, ReadbackComplete},
        render_resource::{TextureFormat, TextureUsages},
        view::screenshot::{Screenshot, ScreenshotCaptured, save_to_disk},
    },
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_penumbra::{
    RadianceCascades, RadianceCascadesGeometry, RadianceCascadesPlugin, RadianceCascadesRawOutput,
};

#[derive(Resource)]
struct Capture {
    target: Handle<Image>,
    raw_target: Option<Handle<Image>>,
    raw_path: String,
    viewport: UVec2,
    raw_size: UVec2,
    resolution_divisor: u32,
    max_rays: u32,
    started_at: Option<f64>,
    requested: bool,
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

#[derive(Component)]
struct LightMover;

#[derive(Component)]
struct RawCaptureOutput {
    path: String,
    size: UVec2,
    viewport: UVec2,
    resolution_divisor: u32,
    max_rays: u32,
}

fn main() -> AppExit {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let benchmark = args.iter().any(|arg| arg == "--benchmark");
    let raw_capture = args.iter().any(|arg| arg == "--capture-raw");
    let image_capture = args.iter().any(|arg| arg == "--capture");
    assert!(
        !(benchmark && (raw_capture || image_capture)),
        "choose one headless operation"
    );

    let size = UVec2::new(
        number(&args, "--width=", if benchmark { 1920 } else { 640 }),
        number(&args, "--height=", if benchmark { 1080 } else { 480 }),
    );
    let requested_divisor = number(&args, "--resolution-divisor=", 2).clamp(1, 64);
    let requested_max_rays = number(&args, "--max-rays=", 131_072);
    // Keep raw capture's exact-size copy grid within WGPU's guaranteed storage binding limits.
    let max_rays = requested_max_rays.min(if raw_capture { 32_768 } else { 131_072 });
    let (raw_size, resolution_divisor) = resolved_size(size, requested_divisor, max_rays);
    let raw_path = args
        .iter()
        .find_map(|arg| arg.strip_prefix("--raw-output=").map(str::to_owned))
        .unwrap_or_else(|| "screenshots/cornell_raw.pfm".into());

    let mut app = App::new();
    app.insert_resource(DefaultOpaqueRendererMethod::deferred())
        .insert_resource(GlobalAmbientLight {
            brightness: 0.0,
            ..default()
        });
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
            raw_target,
            raw_path,
            viewport: size,
            raw_size,
            resolution_divisor,
            max_rays,
            started_at: None,
            requested: false,
        });
    } else {
        app.add_plugins(plugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Split Radiance Cascades | G: GI | B: feedback | arrows: move light | P: screenshot".into(),
                resolution: (960, 720).into(),
                ..default()
            }),
            ..default()
        }));
    }

    if benchmark || raw_capture || image_capture {
        app.add_plugins(RenderDiagnosticsPlugin);
    }
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

    app.add_plugins(RadianceCascadesPlugin::default())
        .add_systems(Startup, setup)
        .add_systems(Update, (controls, benchmark_frame, capture_frame).chain())
        .run()
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

fn resolved_size(viewport: UVec2, requested_divisor: u32, max_rays: u32) -> (UVec2, u32) {
    let size_at = |divisor: u32| {
        UVec2::new(
            viewport.x.div_ceil(divisor).max(1),
            viewport.y.div_ceil(divisor).max(1),
        )
    };
    let budget = u64::from(max_rays.max(1));
    let mut divisor = requested_divisor.max(1);
    while u64::from(size_at(divisor).x) * u64::from(size_at(divisor).y) > budget {
        divisor += 1;
    }
    (size_at(divisor), divisor)
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    capture: Option<Res<Capture>>,
) {
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
        emissive: LinearRgba::new(12.0, 10.0, 7.0, 1.0),
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
        (Vec3::new(0.1, 4.0, 6.0), Vec3::new(-3.05, 2.0, 0.0), red),
        (Vec3::new(0.1, 4.0, 6.0), Vec3::new(3.05, 2.0, 0.0), green),
        (
            Vec3::new(2.0, 0.05, 2.0),
            Vec3::new(0.0, 3.97, -0.8),
            emitter,
        ),
    ] {
        let mut entity = commands.spawn((
            Mesh3d(meshes.add(Cuboid::from_size(size))),
            MeshMaterial3d(material),
            Transform::from_translation(position),
            RadianceCascadesGeometry,
        ));
        if size.y < 0.1 && position.y > 3.9 {
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

    // Keep the room visible while the async split GI pipelines initialize.
    // The emissive panel above remains the indirect-light source for cascades.
    commands.spawn((
        PointLight {
            intensity: 2_400.0,
            range: 8.0,
            ..default()
        },
        Transform::from_xyz(0.0, 3.85, -0.8),
        LightMover,
    ));

    let mut camera = commands.spawn((
        Camera3d::default(),
        ShadowLodOrigin,
        Msaa::Off,
        Camera {
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        Exposure { ev100: 0.0 },
        Tonemapping::Linear,
        RadianceCascades {
            intensity: 1.0,
            resolution_divisor: capture
                .as_ref()
                .map_or(2, |capture| capture.resolution_divisor),
            max_rays: capture.as_ref().map_or(131_072, |capture| capture.max_rays),
            bounce_feedback: 0.0,
            ..default()
        },
        Transform::from_xyz(0.0, 2.0, 8.5).looking_at(Vec3::new(0.0, 1.8, -0.5), Vec3::Y),
    ));
    if let Some(capture) = capture {
        camera.insert(RenderTarget::Image(capture.target.clone().into()));
        if let Some(raw) = &capture.raw_target {
            camera.insert(RadianceCascadesRawOutput(raw.clone()));
        }
    }
}

fn controls(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    capture: Option<Res<Capture>>,
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
    if keys.just_pressed(KeyCode::KeyP) {
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk("cornell_box.png"));
    }
    for mut settings in &mut cameras {
        if keys.just_pressed(KeyCode::KeyG) {
            settings.intensity = if settings.intensity > 0.0 { 0.0 } else { 1.0 };
            settings.reset_history();
        }
        if keys.just_pressed(KeyCode::KeyB) {
            settings.bounce_feedback = if settings.bounce_feedback > 0.0 {
                0.0
            } else {
                0.8
            };
            settings.reset_history();
        }
        let direction = i32::from(keys.pressed(KeyCode::ArrowRight))
            - i32::from(keys.pressed(KeyCode::ArrowLeft));
        for mut transform in &mut lights {
            transform.translation.x += direction as f32 * time.delta_secs() * 2.0;
        }
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

fn capture_frame(
    mut commands: Commands,
    time: Res<Time<Real>>,
    diagnostics: Res<DiagnosticsStore>,
    capture: Option<ResMut<Capture>>,
    benchmark: Option<Res<Benchmark>>,
    mut exit: MessageWriter<AppExit>,
) {
    let Some(mut capture) = capture else {
        return;
    };
    if benchmark.is_some() || capture.requested {
        return;
    }
    let rendered = diagnostics.iter().any(|diagnostic| {
        diagnostic.path().as_str() == "render/split_rc/elapsed_gpu"
            && diagnostic.measurement().is_some()
    });
    if !rendered {
        if time.elapsed_secs_f64() > 600.0 {
            error!("Capture timed out waiting for Split Radiance Cascades output");
            exit.write(AppExit::error());
        }
        return;
    }
    let started_at = *capture.started_at.get_or_insert(time.elapsed_secs_f64());
    if time.elapsed_secs_f64() < started_at + 3.0 {
        return;
    }
    if let Err(error) = std::fs::create_dir_all("screenshots") {
        error!("Could not create screenshots directory: {error}");
        exit.write(AppExit::error());
        return;
    }
    capture.requested = true;

    if let Some(raw) = capture.raw_target.clone() {
        commands
            .spawn((
                Readback::texture(raw),
                RawCaptureOutput {
                    path: capture.raw_path.clone(),
                    size: capture.raw_size,
                    viewport: capture.viewport,
                    resolution_divisor: capture.resolution_divisor,
                    max_rays: capture.max_rays,
                },
            ))
            .observe(save_raw_capture);
    } else {
        let path = "screenshots/cornell_box.png";
        commands
            .spawn(Screenshot::image(capture.target.clone()))
            .observe(save_to_disk(path))
            .observe(
                |captured: On<ScreenshotCaptured>, mut exit: MessageWriter<AppExit>| {
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
                        info!("Cornell SRC capture: {lit}/{total} sampled receiver pixels are lit");
                        exit.write(AppExit::Success);
                    }
                },
            );
    }
}

fn save_raw_capture(
    event: On<ReadbackComplete>,
    output: Query<&RawCaptureOutput>,
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

    let mut rgb = Vec::with_capacity(width * height * 12);
    let mut receivers = 0_usize;
    let mut lit_receivers = 0_usize;
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
            receivers += usize::from(rgba[3] > 0.0);
            lit_receivers += usize::from(rgba[3] > 0.0 && rgba[..3].iter().any(|&v| v > 1e-6));
            for value in &rgba[..3] {
                rgb.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    if receivers == 0 {
        error!("Raw GI readback contains no receiver pixels");
        exit.write(AppExit::error());
        return;
    }
    if lit_receivers == 0 {
        error!("Raw GI readback contains receivers but no emitted or bounced light");
        exit.write(AppExit::error());
        return;
    }

    let mut pfm = format!("PF\n{width} {height}\n-1.0\n").into_bytes();
    for row in rgb.chunks_exact(width * 12).rev() {
        pfm.extend_from_slice(row);
    }
    let path = std::path::Path::new(&output.path);
    let result = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .map_or(Ok(()), std::fs::create_dir_all)
        .and_then(|()| {
            std::fs::write(path, pfm)?;
            std::fs::write(
                path.with_extension("json"),
                format!(
                    "{{\n  \"algorithm\": \"Split Radiance Cascades\",\n  \"units\": \"linear RGB GI estimate\",\n  \"viewport\": [{}, {}],\n  \"gi_size\": [{width}, {height}],\n  \"resolution_divisor\": {},\n  \"max_rays\": {},\n  \"valid_pixels\": {receivers}\n}}\n",
                    output.viewport.x,
                    output.viewport.y,
                    output.resolution_divisor,
                    output.max_rays,
                ),
            )
        });
    if let Err(error) = result {
        error!("Could not save raw GI output: {error}");
        exit.write(AppExit::error());
    } else {
        info!(
            "Saved {receivers} raw GI receiver pixels at {width}x{height}; {lit_receivers} are illuminated"
        );
        exit.write(AppExit::Success);
    }
}

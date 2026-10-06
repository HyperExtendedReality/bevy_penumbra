//! Emissive-only Cornell box: space toggles GI, B toggles diffuse feedback,
//! arrows move the light, P saves a screenshot, Escape exits. --capture runs headlessly.

use bevy::{
    app::{AppExit, ScheduleRunnerPlugin},
    camera::{Exposure, RenderTarget, ShadowLodOrigin},
    core_pipeline::tonemapping::Tonemapping,
    diagnostic::DiagnosticsStore,
    platform::time::Instant,
    prelude::*,
    render::{
        diagnostic::RenderDiagnosticsPlugin,
        render_resource::TextureFormat,
        view::screenshot::{Screenshot, ScreenshotCaptured, save_to_disk},
    },
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_penumbra::{RadianceCascades, RadianceCascadesGeometry, RadianceCascadesPlugin};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[derive(Clone, Copy, PartialEq)]
enum LightSource {
    Emissive,
    Point,
    Directional,
    Spot,
}

#[derive(Resource)]
struct Capture {
    target: Handle<Image>,
    requested: bool,
    no_gi: bool,
    offscreen: bool,
    benchmark: bool,
    reference: bool,
    near_field: bool,
    distant_occlusion: bool,
    bounce_check: bool,
    divisor: u32,
    history_frames: u32,
    feedback: bool,
    filter_steps: u32,
    dynamic: bool,
    moving: bool,
    camera_motion: bool,
    light_source: LightSource,
    initial_requested: bool,
    initial_passed: Arc<AtomicBool>,
    switched_at: Option<f64>,
    quality: bool,
    sequence: u32,
    sequence_index: u32,
    emissive_samples: u32,
}

#[derive(Resource, Default)]
struct Benchmark(std::collections::BTreeMap<String, (Instant, Vec<f64>)>);

#[derive(Component)]
struct LightMover;

#[derive(Component)]
struct DistantOccluder;

fn main() -> AppExit {
    let benchmark = std::env::args().any(|a| a == "--benchmark");
    let capture = benchmark || std::env::args().any(|a| a.starts_with("--capture"));
    let no_gi = std::env::args().any(|a| a == "--capture-no-gi");
    let offscreen = std::env::args().any(|a| a == "--capture-offscreen");
    let reference = std::env::args().any(|a| a == "--capture-reference");
    let near_field = std::env::args().any(|a| a == "--capture-near-field");
    let distant_occlusion = std::env::args().any(|a| a == "--capture-distant-occlusion");
    let bounce_check = std::env::args().any(|a| a == "--capture-feedback");
    let feedback = !reference && !std::env::args().any(|a| a == "--single-bounce");
    let dynamic = std::env::args().any(|a| a == "--capture-dynamic");
    let moving = std::env::args().any(|a| a == "--capture-moving");
    let camera_motion = std::env::args().any(|a| a == "--capture-camera-motion");
    let quality = std::env::args().any(|a| a == "--capture-quality");
    let sequence = if quality { 16 } else { 1 };
    let emissive_samples = std::env::args()
        .find_map(|a| {
            a.strip_prefix("--emitter-samples=")
                .and_then(|s| s.parse::<u32>().ok())
        })
        .unwrap_or(16)
        .clamp(1, 32);
    let light_source = match std::env::args()
        .find_map(|a| a.strip_prefix("--light=").map(str::to_owned))
        .as_deref()
    {
        None | Some("emissive") => LightSource::Emissive,
        Some("point") => LightSource::Point,
        Some("directional") => LightSource::Directional,
        Some("spot") => LightSource::Spot,
        Some(_) => panic!("--light must be emissive, point, directional, or spot"),
    };
    let filter_steps = std::env::args()
        .find_map(|a| {
            a.strip_prefix("--filter=")
                .and_then(|s| s.parse::<u32>().ok())
        })
        .unwrap_or(3)
        .min(4);
    let divisor = std::env::args()
        .find_map(|a| {
            a.strip_prefix("--divisor=")
                .and_then(|s| s.parse::<u32>().ok())
        })
        .unwrap_or(2)
        .clamp(1, 64);
    let history_frames = std::env::args()
        .find_map(|a| {
            a.strip_prefix("--history=")
                .and_then(|s| s.parse::<u32>().ok())
        })
        .unwrap_or(32)
        .min(64);
    let mut app = App::new();
    app.insert_resource(GlobalAmbientLight {
        brightness: 0.0,
        ..default()
    });
    if capture {
        app.add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .disable::<WinitPlugin>(),
        );
        app.add_plugins(ScheduleRunnerPlugin::run_loop(
            std::time::Duration::from_secs_f64(1.0 / 60.0),
        ));
        let target =
            app.world_mut()
                .resource_mut::<Assets<Image>>()
                .add(Image::new_target_texture(
                    640,
                    480,
                    TextureFormat::Rgba8UnormSrgb,
                    None,
                ));
        app.insert_resource(Capture {
            target,
            requested: false,
            no_gi,
            offscreen,
            benchmark,
            reference,
            near_field,
            distant_occlusion,
            bounce_check,
            divisor,
            history_frames,
            feedback,
            filter_steps,
            dynamic,
            moving,
            camera_motion,
            light_source,
            initial_requested: false,
            initial_passed: Arc::new(AtomicBool::new(false)),
            switched_at: None,
            quality,
            sequence,
            sequence_index: 0,
            emissive_samples,
        });
    } else {
        app.add_plugins(
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title:
                        "Penumbra | Space: GI | B: feedback | Arrows: move light | P: screenshot"
                            .into(),
                    resolution: (960, 720).into(),
                    ..default()
                }),
                ..default()
            }),
        );
    }
    if benchmark {
        app.add_plugins(RenderDiagnosticsPlugin)
            .init_resource::<Benchmark>();
    }
    app.add_plugins(RadianceCascadesPlugin::default())
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                controls,
                animate_lighting,
                animate_camera,
                change_lighting,
                benchmark_frame,
                capture_frame,
            )
                .chain(),
        )
        .run()
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    capture: Option<Res<Capture>>,
) {
    let reference = capture.as_ref().is_some_and(|c| c.reference);
    let light_source = capture
        .as_ref()
        .map_or(LightSource::Emissive, |c| c.light_source);
    let white = materials.add(StandardMaterial {
        base_color: Color::srgb(0.8, 0.8, 0.8),
        perceptual_roughness: 1.0,
        ..default()
    });
    if let Some(capture) = capture.as_ref().filter(|c| c.distant_occlusion) {
        // Every hemisphere ray meets a black enclosure beyond the former
        // 25.6-unit far cutoff. Environment lighting must remain occluded.
        let black = materials.add(StandardMaterial {
            base_color: Color::BLACK,
            ..default()
        });
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(200.0, 200.0, 200.0))),
            MeshMaterial3d(black),
            RadianceCascadesGeometry,
            Visibility::Hidden,
            DistantOccluder,
        ));
        commands.spawn((
            Mesh3d(meshes.add(Rectangle::new(20.0, 20.0))),
            MeshMaterial3d(white),
            Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
            RadianceCascadesGeometry,
        ));
        commands.spawn((
            RadianceCascades {
                environment: LinearRgba::WHITE,
                bounce_feedback: 0.0,
                resolution_divisor: capture.divisor,
                history_frames: capture.history_frames,
                spatial_filter_steps: capture.filter_steps,
                ..default()
            },
            Exposure { ev100: 0.0 },
            Tonemapping::None,
            RenderTarget::Image(capture.target.clone().into()),
            Transform::from_xyz(0.0, 5.0, 0.0).looking_at(Vec3::ZERO, Vec3::Z),
        ));
        return;
    }
    if let Some(capture) = capture.as_ref().filter(|c| c.near_field) {
        // A unit-radiance square fills almost the entire cosine hemisphere:
        // E/pi is within 0.1% of one here. Uniform area sampling is very noisy.
        let panel = materials.add(StandardMaterial {
            base_color: Color::BLACK,
            emissive: LinearRgba::WHITE,
            ..default()
        });
        for (height, angle, material) in [
            (0.0, -std::f32::consts::FRAC_PI_2, white),
            (0.3, std::f32::consts::FRAC_PI_2, panel),
        ] {
            commands.spawn((
                Mesh3d(meshes.add(Rectangle::new(20.0, 20.0))),
                MeshMaterial3d(material),
                Transform::from_xyz(0.0, height, 0.0).with_rotation(Quat::from_rotation_x(angle)),
                RadianceCascadesGeometry,
            ));
        }
        commands.spawn((
            RadianceCascades {
                bounce_feedback: 0.0,
                resolution_divisor: capture.divisor,
                history_frames: capture.history_frames,
                spatial_filter_steps: capture.filter_steps,
                emissive_samples: capture.emissive_samples,
                ..default()
            },
            Exposure { ev100: 0.0 },
            Tonemapping::None,
            RenderTarget::Image(capture.target.clone().into()),
            Transform::from_xyz(0.0, 0.15, 0.0).looking_at(Vec3::ZERO, Vec3::Z),
        ));
        return;
    }
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
    let emission = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        emissive: if light_source == LightSource::Emissive {
            LinearRgba::new(12.0, 10.0, 7.0, 1.0)
        } else {
            LinearRgba::BLACK
        },
        ..default()
    });
    for (size, position, material) in [
        (
            Vec3::new(6.0, 0.1, 6.0),
            Vec3::new(0.0, -0.05, 0.0),
            white.clone(),
        ),
        (
            Vec3::new(6.0, 0.1, 6.0),
            Vec3::new(0.0, 4.05, 0.0),
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
            Vec3::new(
                0.0,
                3.97,
                if capture.as_ref().is_some_and(|c| c.offscreen) {
                    11.0
                } else {
                    -0.8
                },
            ),
            emission,
        ),
    ] {
        // Sunlight needs an opening; a closed ceiling correctly shadows the
        // entire room in both Bevy's direct pass and the GI shadow rays.
        if !reference && light_source == LightSource::Directional && position.y > 4.0 {
            continue;
        }
        let mut entity = commands.spawn((
            Mesh3d(meshes.add(Cuboid::from_size(size))),
            MeshMaterial3d(if reference { white.clone() } else { material }),
            Transform::from_translation(position),
            RadianceCascadesGeometry,
        ));
        if size.y < 0.1 && position.y > 3.9 && light_source == LightSource::Emissive {
            entity.insert(LightMover);
        }
        if reference {
            entity.remove::<RadianceCascadesGeometry>();
        }
    }
    if !reference {
        let transform = Transform::from_xyz(0.0, 3.6, -0.8);
        match light_source {
            LightSource::Emissive => {}
            LightSource::Point => {
                commands.spawn((
                    PointLight {
                        intensity: 800.0,
                        range: 15.0,
                        shadow_maps_enabled: true,
                        ..default()
                    },
                    transform,
                    LightMover,
                ));
            }
            LightSource::Directional => {
                commands.spawn((
                    DirectionalLight {
                        illuminance: 8.0,
                        shadow_maps_enabled: true,
                        ..default()
                    },
                    transform.looking_to(Vec3::new(0.0, -1.0, 0.3), Vec3::Y),
                    LightMover,
                ));
            }
            LightSource::Spot => {
                commands.spawn((
                    SpotLight {
                        intensity: 1600.0,
                        range: 15.0,
                        inner_angle: 0.3,
                        outer_angle: 0.75,
                        shadow_maps_enabled: true,
                        ..default()
                    },
                    transform.looking_at(Vec3::new(0.0, 0.0, -1.0), Vec3::Y),
                    LightMover,
                ));
            }
        }
    }
    for (size, position, rotation) in [
        (Vec3::new(1.4, 1.5, 1.4), Vec3::new(-1.1, 0.75, 0.0), 0.3),
        (Vec3::new(1.2, 2.8, 1.2), Vec3::new(1.0, 1.4, -1.0), -0.3),
    ] {
        let mut entity = commands.spawn((
            Mesh3d(meshes.add(Cuboid::from_size(size))),
            MeshMaterial3d(white.clone()),
            Transform::from_translation(position).with_rotation(Quat::from_rotation_y(rotation)),
            RadianceCascadesGeometry,
        ));
        if reference {
            entity.remove::<RadianceCascadesGeometry>();
        }
    }
    let mut camera = commands.spawn((
        Camera3d::default(),
        ShadowLodOrigin,
        Msaa::Off,
        Camera {
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        Exposure { ev100: 0.0 },
        RadianceCascades {
            probe_spacing: 0.18,
            interval_length: 0.288,
            resolution_divisor: capture.as_ref().map_or(2, |c| c.divisor),
            max_rays: 131_072,
            history_frames: capture.as_ref().map_or(32, |c| c.history_frames),
            spatial_filter_steps: capture.as_ref().map_or(3, |c| c.filter_steps),
            emissive_samples: capture.as_ref().map_or(16, |c| c.emissive_samples),
            bounce_feedback: if capture.as_ref().is_none_or(|c| c.feedback) {
                0.8
            } else {
                0.0
            },
            environment: if reference {
                LinearRgba::WHITE
            } else {
                LinearRgba::BLACK
            },
            intensity: if capture.as_ref().is_some_and(|c| c.no_gi) {
                0.0
            } else {
                1.0
            },
            ..default()
        },
        Transform::from_xyz(0.0, 2.0, 8.5).looking_at(Vec3::new(0.0, 1.8, -0.5), Vec3::Y),
    ));
    if reference {
        camera.insert(Tonemapping::None);
    }
    if let Some(capture) = capture {
        camera.insert(RenderTarget::Image(capture.target.clone().into()));
    }
}

fn controls(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut cameras: Query<&mut RadianceCascades>,
    mut lights: Query<&mut Transform, With<LightMover>>,
    mut commands: Commands,
    mut exit: MessageWriter<AppExit>,
) {
    if keys.just_pressed(KeyCode::Escape) {
        exit.write(AppExit::Success);
    }
    if keys.just_pressed(KeyCode::KeyP) {
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk("cornell_box.png"));
    }
    for mut settings in &mut cameras {
        if keys.just_pressed(KeyCode::Space) {
            settings.intensity = if settings.intensity > 0.0 { 0.0 } else { 1.0 };
        }
        if keys.just_pressed(KeyCode::KeyB) {
            settings.bounce_feedback = if settings.bounce_feedback > 0.0 {
                0.0
            } else {
                0.8
            };
        }
        let direction = i32::from(keys.pressed(KeyCode::ArrowRight))
            - i32::from(keys.pressed(KeyCode::ArrowLeft));
        if direction != 0 {
            for mut transform in &mut lights {
                transform.translation.x += direction as f32 * time.delta_secs() * 2.0;
            }
        }
    }
}

fn change_lighting(
    time: Res<Time<Real>>,
    capture: Option<ResMut<Capture>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut points: Query<&mut PointLight>,
    mut spots: Query<&mut SpotLight>,
    mut directional: Query<&mut DirectionalLight>,
    mut occluders: Query<&mut Visibility, With<DistantOccluder>>,
) {
    let Some(mut capture) = capture else {
        return;
    };
    if capture.switched_at.is_some()
        || time.elapsed_secs() < 3.0
        || !(capture.dynamic || capture.distant_occlusion)
        || !capture.initial_passed.load(Ordering::Relaxed)
    {
        return;
    }
    for mut visibility in &mut occluders {
        *visibility = Visibility::Visible;
    }
    for (_, material) in materials.iter_mut() {
        material.emissive = LinearRgba::BLACK;
    }
    for mut light in &mut points {
        light.intensity = 0.0;
    }
    for mut light in &mut spots {
        light.intensity = 0.0;
    }
    for mut light in &mut directional {
        light.illuminance = 0.0;
    }
    capture.switched_at = Some(time.elapsed_secs_f64());
}

fn animate_lighting(
    time: Res<Time<Real>>,
    capture: Option<Res<Capture>>,
    mut lights: Query<&mut Transform, With<LightMover>>,
) {
    let Some(capture) = capture else {
        return;
    };
    if !capture.moving {
        return;
    }
    let phase = (time.elapsed_secs() * 2.0).sin();
    for mut transform in &mut lights {
        transform.translation.x = 0.35 * phase;
        if capture.light_source == LightSource::Directional {
            transform.rotation = Quat::from_rotation_y(phase * 0.1)
                * Transform::IDENTITY
                    .looking_to(Vec3::new(0.0, -1.0, 0.3), Vec3::Y)
                    .rotation;
        } else if capture.light_source == LightSource::Spot {
            transform.look_at(Vec3::new(0.0, 0.0, -1.0), Vec3::Y);
        }
    }
}

fn animate_camera(
    time: Res<Time<Real>>,
    capture: Option<Res<Capture>>,
    mut cameras: Query<&mut Transform, With<RadianceCascades>>,
) {
    if !capture.is_some_and(|capture| capture.camera_motion) {
        return;
    }
    for mut transform in &mut cameras {
        transform.translation.x = 0.15 * (time.elapsed_secs() * 1.5).sin();
        transform.look_at(Vec3::new(0.0, 1.8, -0.5), Vec3::Y);
    }
}

fn capture_frame(
    mut commands: Commands,
    time: Res<Time<Real>>,
    capture: Option<ResMut<Capture>>,
    mut exit: MessageWriter<AppExit>,
) {
    let Some(mut capture) = capture else {
        return;
    };
    if time.elapsed_secs() > 60.0 {
        exit.write(AppExit::error());
        return;
    }
    // A dark final capture alone can pass when GI never ran. First prove that
    // the receiver was lit before removing its light or enclosing it.
    if (capture.dynamic || capture.distant_occlusion)
        && !capture.initial_passed.load(Ordering::Relaxed)
    {
        if !capture.initial_requested && time.elapsed_secs() > 2.0 {
            capture.initial_requested = true;
            let passed = capture.initial_passed.clone();
            commands
                .spawn(Screenshot::image(capture.target.clone()))
                .observe(
                    move |captured: On<ScreenshotCaptured>, mut exit: MessageWriter<AppExit>| {
                        let Ok(image) = captured.image.clone().try_into_dynamic() else {
                            exit.write(AppExit::error());
                            return;
                        };
                        let image = image.to_rgb8();
                        let mut lit = 0;
                        let mut total = 0;
                        for y in image.height() * 64 / 100..image.height() * 71 / 100 {
                            for x in image.width() * 36 / 100..image.width() * 41 / 100 {
                                lit += u32::from(image.get_pixel(x, y).0.iter().any(|&v| v > 8));
                                total += 1;
                            }
                        }
                        if lit > total / 2 {
                            info!("Dynamic initial indirect receiver: {lit}/{total} lit");
                            passed.store(true, Ordering::Relaxed);
                        } else {
                            error!("Dynamic initial indirect receiver was dark: {lit}/{total} lit");
                            exit.write(AppExit::error());
                        }
                    },
                );
        }
        return;
    }
    let waiting = if capture.dynamic || capture.distant_occlusion {
        capture
            .switched_at
            .is_none_or(|at| time.elapsed_secs_f64() < at + 0.1)
    } else {
        time.elapsed_secs() < if capture.benchmark { 12.0 } else { 6.0 }
    };
    if capture.requested || waiting {
        return;
    }
    capture.requested = true;
    if let Err(error) = std::fs::create_dir_all("screenshots") {
        error!("{error}");
        exit.write(AppExit::error());
        return;
    }
    let no_gi = capture.no_gi || capture.dynamic || capture.distant_occlusion;
    let reference = capture.reference;
    let near_field = capture.near_field;
    let conventional = capture.light_source != LightSource::Emissive;
    let bounce_check = !no_gi
        && (capture.bounce_check
            || capture.moving
            || capture.camera_motion
            || capture.quality
            || conventional);
    let direct_only = capture.no_gi && conventional && !capture.dynamic;
    let path = if capture.distant_occlusion {
        "screenshots/distant_occlusion.png"
    } else if capture.dynamic {
        "screenshots/dynamic.png"
    } else if capture.moving {
        "screenshots/moving.png"
    } else if capture.camera_motion {
        "screenshots/camera_motion.png"
    } else if no_gi {
        "screenshots/no_gi.png"
    } else if near_field {
        "screenshots/near_field.png"
    } else if reference {
        "screenshots/reference.png"
    } else if bounce_check {
        "screenshots/feedback.png"
    } else if capture.offscreen {
        "screenshots/offscreen.png"
    } else {
        "screenshots/cornell_box.png"
    };
    let source = match capture.light_source {
        LightSource::Point => "point",
        LightSource::Spot => "spot",
        LightSource::Directional => "directional",
        LightSource::Emissive => "emissive",
    };
    let path = if capture.quality {
        let motion = if capture.moving {
            "moving_"
        } else if capture.camera_motion {
            "camera_"
        } else {
            ""
        };
        format!(
            "screenshots/quality_{motion}{source}_{:03}.png",
            capture.sequence_index
        )
    } else if conventional {
        format!("{}_{source}.png", path.trim_end_matches(".png"))
    } else {
        path.to_owned()
    };
    commands
        .spawn(Screenshot::image(capture.target.clone()))
        .observe(save_to_disk(path))
        .observe(
            move |captured: On<ScreenshotCaptured>, mut exit: MessageWriter<AppExit>, mut capture: ResMut<Capture>| {
                let Ok(image) = captured.image.clone().try_into_dynamic() else {
                    exit.write(AppExit::error());
                    return;
                };
                let image = image.to_rgb8();
                // Lower center contains only non-emissive receivers, so it must be
                // black with GI off and lit with GI on (including offscreen mode).
                let mut lit = 0;
                let mut total = 0;
                for y in image.height() / 2..image.height() * 3 / 4 {
                    for x in image.width() / 4..image.width() * 3 / 4 {
                        lit += u32::from(image.get_pixel(x, y).0.iter().any(|&v| v > 8));
                        total += 1;
                    }
                }
                let mut correct = if direct_only { true } else if no_gi { lit == 0 } else { lit > total / 100 };
                if reference || near_field {
                    // No tracing geometry: constant environment has E/pi=1.
                    // With tonemapping disabled, the expected displayed white
                    // material value follows directly from albedo and exposure.
                    let linear = Color::srgb(0.8, 0.8, 0.8).to_linear().red * Exposure { ev100: 0.0 }.exposure();
                    let expected = Color::linear_rgb(linear, linear, linear).to_srgba().red * 255.0;
                    let mut error = 0.0;
                    for y in image.height() / 2..image.height() * 3 / 4 {
                        for x in image.width() / 4..image.width() * 3 / 4 {
                            for channel in image.get_pixel(x, y).0 { error += (channel as f32 - expected).abs(); }
                        }
                    }
                    let mean_error = error / (total * 3) as f32;
                    info!("Diffuse-energy reference: expected={expected:.1}, mean absolute error={mean_error:.2}/255");
                    correct &= mean_error < if near_field { 2.0 } else { 12.0 };
                }
                // The small box's central front face cannot directly see the
                // ceiling emitter. Additional bounces must light this region.
                let mut front_lit = 0; let mut front_total = 0; let mut front_sum = 0u32;
                for y in image.height() * 64 / 100..image.height() * 71 / 100 {
                    for x in image.width() * 36 / 100..image.width() * 41 / 100 {
                        let value = *image.get_pixel(x, y).0.iter().max().unwrap_or(&0);
                        front_lit += u32::from(value > 8); front_total += 1; front_sum += u32::from(value);
                    }
                }
                info!("Front receiver: {front_lit}/{front_total} lit, mean={:.2}/255", front_sum as f32 / front_total as f32);
                if bounce_check { correct &= front_lit > front_total / 2; }
                // Conventional direct lighting remains on the floor with GI
                // disabled; this front region must still be black.
                if direct_only { correct &= front_lit == 0; }
                if correct {
                    info!("GPU image check passed: {lit}/{total} receiver pixels lit");
                } else {
                    error!("GPU image check failed: {lit}/{total} receiver pixels lit");
                }
                capture.sequence_index += 1;
                if !correct { exit.write(AppExit::error()); }
                else if capture.sequence_index >= capture.sequence { exit.write(AppExit::Success); }
                else { capture.requested = false; }
            },
        );
}

fn benchmark_frame(
    time: Res<Time<Real>>,
    diagnostics: Res<DiagnosticsStore>,
    benchmark: Option<ResMut<Benchmark>>,
    capture: Option<Res<Capture>>,
) {
    let (Some(mut benchmark), Some(capture)) = (benchmark, capture) else {
        return;
    };
    if time.elapsed_secs() < 3.0 || capture.requested {
        return;
    }
    for diagnostic in diagnostics.iter() {
        let path = diagnostic.path().as_str();
        if !path.contains("split_rc") || !path.ends_with("elapsed_gpu") {
            continue;
        }
        let Some(measurement) = diagnostic.measurement() else {
            continue;
        };
        let row = benchmark
            .0
            .entry(path.to_string())
            .or_insert((measurement.time, Vec::new()));
        if row.0 != measurement.time || row.1.is_empty() {
            row.0 = measurement.time;
            row.1.push(measurement.value);
        }
    }
    if time.elapsed_secs() >= 12.0 {
        let mut report = format!(
            "640x480 Cornell box, divisor={}, history={}, feedback={}, filter={}, emitter_samples={}, warm-up 3s, GPU milliseconds\n",
            capture.divisor,
            capture.history_frames,
            capture.feedback,
            capture.filter_steps,
            capture.emissive_samples
        );
        for (path, (_, samples)) in &mut benchmark.0 {
            samples.sort_unstable_by(f64::total_cmp);
            let mean = samples.iter().sum::<f64>() / samples.len() as f64;
            let p95 = samples[(samples.len() - 1) * 95 / 100];
            report.push_str(&format!(
                "{path}: n={} mean={mean:.3} p50={:.3} p95={p95:.3} max={:.3}\n",
                samples.len(),
                samples[samples.len() / 2],
                samples[samples.len() - 1]
            ));
        }
        if benchmark.0.is_empty() {
            report.push_str(if capture.no_gi {
                "GI disabled; no pass submitted\n"
            } else {
                "GPU timestamps unavailable\n"
            });
        }
        info!("{report}");
        if let Err(error) =
            std::fs::create_dir_all("tmp").and_then(|_| std::fs::write("tmp/benchmark.txt", report))
        {
            error!("Could not save benchmark: {error}");
        }
    }
}

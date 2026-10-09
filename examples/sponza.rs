//! Local Sponza courtyard, curtains, ivy, and a controllable 24-hour sun.
//! Brackets scrub time; Space animates; 1/2/3/4 select dawn/noon/dusk/night.
//! G toggles GI; right mouse + WASD/QE flies; P captures; Escape exits.
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
        RenderApp, RenderPlugin,
        diagnostic::RenderDiagnosticsPlugin,
        gpu_readback::{Readback, ReadbackComplete},
        render_resource::{PollType, TextureFormat, TextureUsages},
        renderer::{RenderDevice, RenderGraph, RenderGraphSystems},
        view::screenshot::{Screenshot, ScreenshotCaptured, save_to_disk},
    },
    window::ExitCondition,
    winit::WinitPlugin,
    world_serialization::{WorldInstanceReady, WorldInstanceSpawner},
};
use bevy_penumbra::{
    RadianceCascades, RadianceCascadesGeometry, RadianceCascadesPlugin, RadianceCascadesRawOutput,
};
use bevy_texture_cache::{TextureCachePlugin, TextureCacheStats};
use std::collections::HashSet;

const COURTYARD: &str = "NewSponza_Main_glTF_003.gltf";
const CURTAINS: &str = "NewSponza_Curtains_glTF.gltf";
const IVY: &str = "NewSponza_IvyGrowth_glTF.gltf";
const TEXTURE_SIZE: u32 = 512;
const CONTROLS: &str = "[ / ] time | Space play | 1 dawn 2 noon 3 dusk 4 night | G GI | B feedback | RMB + WASD/QE fly | P screenshot";

#[derive(Resource)]
struct Demo {
    scenes: Vec<String>,
    hour: f32,
    playing: bool,
    capture: bool,
    target: Option<Handle<Image>>,
    loaded: usize,
    day_mean: f32,
    day_pixels: Vec<u8>,
    gi_ready: bool,
    gi_frames: u32,
    capture_stage: usize,
    requested: bool,
    settle_until: f64,
    benchmark: bool,
    max_rays: u32,
    motion_capture: bool,
    motion_frame: u32,
}

#[derive(Component)]
struct Sun;

#[derive(Component)]
struct PendingScene(Vec<Entity>);

fn main() -> AppExit {
    let mut scenes = [COURTYARD, CURTAINS, IVY].map(str::to_owned).to_vec();
    let mut hour = 12.0;
    let mut capture = false;
    let mut benchmark = false;
    let mut max_rays = 8192;
    let mut motion_capture = false;
    for arg in std::env::args().skip(1) {
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
        } else if arg == "--capture-motion" {
            capture = true;
            motion_capture = true;
        } else if arg == "--benchmark" {
            benchmark = true;
        } else if let Some(value) = arg.strip_prefix("--max-rays=") {
            max_rays = value.parse().expect("--max-rays requires an integer");
            assert!(
                (1..=131072).contains(&max_rays),
                "--max-rays must be 1..=131072"
            );
        } else {
            panic!(
                "Unknown argument {arg}; use --scene=path.gltf, --time=10, --max-rays=8192, --capture, --capture-motion, or --benchmark"
            );
        }
    }
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
    if capture || benchmark {
        app.add_plugins(
            plugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .disable::<WinitPlugin>(),
        )
        .add_plugins(ScheduleRunnerPlugin::run_loop(
            // Offscreen capture has no swapchain to limit frames in flight.
            std::time::Duration::from_millis(250),
        ));
        app.sub_app_mut(RenderApp)
            .add_systems(RenderGraph, drain_gpu.in_set(RenderGraphSystems::Finish));
    } else {
        app.add_plugins(plugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: CONTROLS.into(),
                resolution: (1280, 720).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(FreeCameraPlugin);
    }
    app.insert_resource(DefaultOpaqueRendererMethod::deferred())
        .insert_resource(GlobalAmbientLight {
            brightness: 0.0,
            ..default()
        })
        .insert_resource(Demo {
            scenes,
            hour,
            playing: false,
            capture,
            target: None,
            loaded: 0,
            day_mean: 0.0,
            day_pixels: Vec::new(),
            gi_ready: false,
            gi_frames: 0,
            capture_stage: 0,
            requested: false,
            settle_until: 0.0,
            benchmark,
            max_rays,
            motion_capture,
            motion_frame: 0,
        })
        .add_plugins(RadianceCascadesPlugin {
            texture_atlas_budget_bytes: 768 * 1024 * 1024,
            scene_gpu_budget_bytes: 2 * 1024 * 1024 * 1024,
            geometry_upload_bytes_per_frame: 16 * 1024 * 1024,
            ..default()
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
                controls,
                lighting,
                capture_frame,
                capture_motion,
                asset_errors,
                performance,
            )
                .chain(),
        );
    app.run()
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
        "{CONTROLS}; demo material textures limited to {TEXTURE_SIZE} pixels, GI atlas budget 768 MiB"
    );
    commands.spawn(WorldAssetRoot(load_scene(&assets, &demo.scenes[0])));
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
    let settings = RadianceCascades {
        intensity: 1.0,
        probe_spacing: 0.4,
        interval_length: 0.6,
        resolution_divisor: 4,
        max_rays: demo.max_rays,
        emissive_samples: 4,
        bounce_feedback: 0.0,
        history_frames: 16,
        temporal_blend: 0.98,
        spatial_filter_steps: 2,
        ..default()
    };
    let mut camera = commands.spawn((
        Camera3d::default(),
        settings,
        ShadowLodOrigin,
        Exposure { ev100: 8.0 },
        Projection::Perspective(PerspectiveProjection {
            fov: std::f32::consts::FRAC_PI_3,
            ..default()
        }),
        // Use the supplied courtyard camera position, clear of pillars and ivy.
        Transform::from_xyz(-8.80743, 1.59222, -0.85826)
            .looking_at(Vec3::new(-3.00422, 2.76208, 0.23737), Vec3::Y),
    ));
    if demo.capture || demo.benchmark {
        let (width, height) = if demo.benchmark {
            (1280, 720)
        } else {
            (640, 360)
        };
        let target = images.add(Image::new_target_texture(
            width,
            height,
            TextureFormat::Rgba8UnormSrgb,
            None,
        ));
        camera.insert(RenderTarget::Image(target.clone().into()));
        demo.target = Some(target);
        let mut divisor = settings.resolution_divisor;
        while width.div_ceil(divisor) * height.div_ceil(divisor) > settings.max_rays {
            divisor += 1;
        }
        let mut raw = Image::new_target_texture(
            width.div_ceil(divisor),
            height.div_ceil(divisor),
            TextureFormat::Rgba32Float,
            None,
        );
        raw.texture_descriptor.usage |= TextureUsages::COPY_SRC | TextureUsages::COPY_DST;
        let raw = images.add(raw);
        camera.insert(RadianceCascadesRawOutput(raw.clone()));
        commands.spawn(Readback::texture(raw)).observe(
            |event: On<ReadbackComplete>, mut demo: ResMut<Demo>| {
                let valid = event.data.as_chunks::<16>().0.iter().any(|pixel| {
                    let values: [f32; 4] = std::array::from_fn(|i| {
                        f32::from_le_bytes(pixel[i * 4..i * 4 + 4].try_into().unwrap())
                    });
                    values.iter().all(|v| v.is_finite())
                        && values[3] > 0.0
                        && values[..3].iter().any(|v| *v > 0.001)
                });
                demo.gi_frames = if valid { demo.gi_frames + 1 } else { 0 };
                demo.gi_ready = demo.gi_frames >= 32;
            },
        );
    } else {
        camera.insert(FreeCamera {
            walk_speed: 3.0,
            run_speed: 9.0,
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
            // Prevent uploading 4K source images before the demo preset is applied.
            settings.load_materials = RenderAssetUsages::MAIN_WORLD;
        })
        .load(GltfAssetLabel::Scene(0).from_asset(path.to_owned()))
}

#[allow(clippy::too_many_arguments)]
fn prepare_scene(
    assets: Res<AssetServer>,
    pending: Query<(Entity, &PendingScene)>,
    meshes: Query<(Entity, &MeshMaterial3d<StandardMaterial>), With<Mesh3d>>,
    materials: Res<Assets<StandardMaterial>>,
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
            let material = materials.get(&material.0).expect("Loaded glTF material");
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
    for mut gi in &mut cameras {
        if keys.just_pressed(KeyCode::KeyG) {
            gi.intensity = if gi.intensity > 0.0 { 0.0 } else { 1.0 };
            jump = true;
        }
        if keys.just_pressed(KeyCode::KeyB) {
            gi.bounce_feedback = if gi.bounce_feedback > 0.0 { 0.0 } else { 0.8 };
            jump = true;
        }
        if jump {
            gi.reset_history();
        }
    }
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
    clear.0 = Color::linear_rgb(500.0 * sky + 0.02, 700.0 * sky + 0.03, 1000.0 * sky + 0.06);
}

fn performance(
    demo: Res<Demo>,
    time: Res<Time<Real>>,
    diagnostics: Res<DiagnosticsStore>,
    mut windows: Query<&mut Window>,
    mut state: Local<(f64, Vec<f64>, Option<bevy::platform::time::Instant>)>,
    mut exit: MessageWriter<AppExit>,
) {
    let now = time.elapsed_secs_f64();
    if now >= state.0 + 0.5 {
        state.0 = now;
        let fps = diagnostics
            .get(&FrameTimeDiagnosticsPlugin::FPS)
            .and_then(|d| d.smoothed());
        for mut window in &mut windows {
            window.title = format!(
                "Sponza | {:.1} FPS | {:02}:{:02} {} | {CONTROLS}",
                fps.unwrap_or(0.0),
                demo.hour as u32,
                ((demo.hour.fract() * 60.0) as u32).min(59),
                if demo.playing { "playing" } else { "paused" }
            );
        }
    }
    // Offscreen benchmarks report GPU timings, not their intentionally capped
    // application FPS. Readback confirms the full scene has produced fresh GI.
    if demo.benchmark
        && demo.gi_ready
        && demo.loaded == demo.scenes.len()
        && now > demo.settle_until + 3.0
        && diagnostics.iter().any(|d| {
            d.path().as_str().contains("split_rc")
                && d.path().as_str().ends_with("elapsed_gpu")
                && d.value().is_some()
        })
    {
        if let Some(measurement) = diagnostics
            .iter()
            .find(|d| d.path().as_str() == "render/split_rc/elapsed_gpu")
            .and_then(|d| d.measurement())
            && state.2 != Some(measurement.time)
        {
            state.2 = Some(measurement.time);
            state.1.push(measurement.value);
        }
        if state.1.len() >= 120 {
            state.1.sort_by(f64::total_cmp);
            let median = state.1[state.1.len() / 2];
            let p95 = state.1[state.1.len() * 95 / 100];
            info!(
                "Sponza 1280x720 Split Radiance Cascades GPU benchmark: {} sample budget, {} timing samples, median {median:.3} ms, p95 {p95:.3} ms",
                demo.max_rays,
                state.1.len()
            );
            for d in diagnostics
                .iter()
                .filter(|d| d.path().as_str().ends_with("elapsed_gpu"))
            {
                if let Some(ms) = d.average() {
                    info!("{}: {ms:.3} ms", d.path());
                }
            }
            exit.write(AppExit::Success);
        }
    }
    if demo.benchmark && now > 600.0 {
        error!("Sponza benchmark timed out waiting for GI");
        exit.write(AppExit::error());
    }
}

fn capture_frame(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time<Real>>,
    diagnostics: Res<DiagnosticsStore>,
    mut demo: ResMut<Demo>,
    mut exit: MessageWriter<AppExit>,
) {
    if demo.benchmark || demo.motion_capture {
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
        error!("Sponza capture timed out before valid GI and scene readiness");
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
    let path = format!(
        "screenshots/sponza_{}.png",
        ["day", "no_gi", "night"][demo.capture_stage]
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

fn capture_motion(
    mut commands: Commands,
    time: Res<Time<Real>>,
    mut demo: ResMut<Demo>,
    mut camera: Query<(&mut Transform, &mut RadianceCascades)>,
    mut exit: MessageWriter<AppExit>,
) {
    if !demo.motion_capture {
        return;
    }
    if time.elapsed_secs_f64() > 600.0 {
        error!("Sponza motion capture timed out");
        exit.write(AppExit::error());
        return;
    }
    if demo.loaded != demo.scenes.len() || !demo.gi_ready || demo.requested {
        return;
    }
    let frame = demo.motion_frame;
    let t = frame.min(32) as f32 / 32.0;
    for (mut transform, mut gi) in &mut camera {
        *transform = Transform::from_xyz(-8.80743, 1.59222, -0.85826)
            .with_translation(Vec3::new(-8.80743, 1.59222, -0.85826) + Vec3::new(0.8, 0.0, 1.2) * t)
            .looking_at(Vec3::new(-3.00422, 2.76208, 0.23737), Vec3::Y);
        if frame == 128 {
            gi.reset_history();
        }
    }
    demo.motion_frame += 1;
    if frame > 32 && frame != 127 && frame != 256 || frame <= 32 && frame % 8 != 0 {
        return;
    }
    demo.requested = true;
    let path = format!("screenshots/sponza_src_motion_{frame:03}.png");
    commands
        .spawn(Screenshot::image(demo.target.clone().unwrap()))
        .observe(
            move |event: On<ScreenshotCaptured>,
                  mut demo: ResMut<Demo>,
                  mut exit: MessageWriter<AppExit>| {
                let pixels = event.image.data.as_ref().unwrap().clone();
                save_to_disk(&path)(event);
                let mean = pixels
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|p| (f32::from(p[0]) + f32::from(p[1]) + f32::from(p[2])) / 3.0)
                    .sum::<f32>()
                    / (pixels.len() / 4) as f32;
                assert!(mean > 4.0, "Moving Sponza lighting became dark: {mean}");
                if frame == 32 || frame == 127 {
                    demo.day_pixels = pixels.clone();
                }
                if frame == 256 {
                    let difference = pixels
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .zip(demo.day_pixels.as_chunks::<4>().0)
                        .map(|(a, b)| {
                            (0..3)
                                .map(|i| (f32::from(a[i]) - f32::from(b[i])).abs())
                                .sum::<f32>()
                        })
                        .sum::<f32>()
                        / ((pixels.len() / 4) * 3) as f32;
                    info!("Sponza settled/restarted history RGB difference: {difference:.3}/255");
                    assert!(
                        difference < 8.0,
                        "Sponza retained stale lighting after camera motion: {difference}"
                    );
                    exit.write(AppExit::Success);
                }
                info!("Sponza motion frame {frame}: RGB mean {mean:.3}");
                demo.requested = false;
            },
        );
}

fn drain_gpu(device: Res<RenderDevice>) {
    // Automated offscreen runs lack swapchain back-pressure. Bound outstanding
    // GPU work even when a frame takes longer than the capture interval.
    device
        .poll(PollType::wait_indefinitely())
        .expect("offscreen GPU completion");
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn clock_wraps_and_sun_crosses_the_horizon() {
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
        let mut app = App::new();
        app.insert_resource(Demo {
            scenes: Vec::new(),
            hour: 23.0,
            playing: false,
            capture: false,
            target: None,
            loaded: 0,
            day_mean: 0.0,
            day_pixels: Vec::new(),
            gi_ready: false,
            gi_frames: 0,
            capture_stage: 0,
            requested: false,
            settle_until: 0.0,
            benchmark: false,
            max_rays: 8192,
            motion_capture: false,
            motion_frame: 0,
        })
        .init_resource::<ButtonInput<KeyCode>>()
        .init_resource::<Time>()
        .add_message::<AppExit>()
        .add_systems(Update, controls);
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
    }
}

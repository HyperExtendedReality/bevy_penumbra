//! Native multi-camera lighting ownership and removal/disable regression.
//! Run with WGPU_BACKEND=vulkan or dx12 (matched DXC on DX12).
#![recursion_limit = "256"]
use bevy::{
    app::{AppExit, ScheduleRunnerPlugin},
    asset::RenderAssetUsages,
    camera::{Exposure, Hdr, RenderTarget},
    core_pipeline::{
        prepass::{DeferredPrepass, DepthPrepass, NormalPrepass},
        tonemapping::Tonemapping,
    },
    light::IrradianceVolume,
    pbr::{
        DefaultOpaqueRendererMethod, RenderViewLightProbes, ScreenSpaceReflections,
        ScreenSpaceReflectionsPipelineId, prepare_ssr_pipelines, prepare_ssr_settings,
    },
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        gpu_readback::{Readback, ReadbackComplete},
        render_resource::{
            Extent3d, TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
            TextureViewDimension,
        },
        view::screenshot::{Screenshot, ScreenshotCaptured},
    },
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_penumbra::{
    RadianceCascades, RadianceCascadesGeometry, RadianceCascadesPlugin, RadianceCascadesRawOutput,
};
use std::sync::{Arc, Mutex};

#[derive(Component, Clone, Copy, ExtractComponent)]
#[extract_app(RenderApp)]
struct ViewIndex(usize);
#[derive(Clone, Copy, Debug, PartialEq)]
struct Lighting {
    ambient: f32,
    environment: bool,
    volume: bool,
    ssr: bool,
}
#[derive(Resource, Clone, Default)]
struct Audit(Arc<Mutex<[Option<Lighting>; 3]>>);
#[derive(Resource)]
struct Check {
    targets: Vec<Handle<Image>>,
    raw_valid: bool,
    phase: usize,
    ready_at: f64,
    requested: bool,
    means: [Option<f32>; 3],
}

fn main() -> AppExit {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                ..default()
            })
            .disable::<WinitPlugin>(),
    )
    .add_plugins(ScheduleRunnerPlugin::run_loop(
        std::time::Duration::from_secs_f64(1.0 / 60.0),
    ))
    .insert_resource(DefaultOpaqueRendererMethod::deferred())
    .insert_resource(GlobalAmbientLight {
        brightness: 80.0,
        ..default()
    })
    .add_plugins((
        RadianceCascadesPlugin::default(),
        ExtractComponentPlugin::<ViewIndex>::default(),
    ))
    .init_resource::<Audit>()
    .add_systems(Startup, setup)
    .add_systems(Update, check);
    let audit = app.world().resource::<Audit>().clone();
    app.sub_app_mut(RenderApp)
        .insert_resource(audit)
        .add_systems(
            Render,
            audit_preparation
                .in_set(RenderSystems::PrepareBindGroups)
                .after(prepare_ssr_pipelines)
                .after(prepare_ssr_settings),
        );
    app.run()
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    commands.spawn((
        Mesh3d(meshes.add(Rectangle::new(20.0, 20.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::linear_rgb(0.5, 0.5, 0.5),
            perceptual_roughness: 1.0,
            ..default()
        })),
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
        RadianceCascadesGeometry,
    ));
    let mut cube = Image::new_fill(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 6,
        },
        TextureDimension::D2,
        &[128, 64, 32, 255],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::default(),
    );
    cube.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::Cube),
        ..default()
    });
    let cube = images.add(cube);
    let volume = images.add(Image::new_fill(
        Extent3d {
            width: 6,
            height: 1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D3,
        &[32, 64, 128, 255],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::default(),
    ));
    commands.spawn((
        LightProbe::default(),
        IrradianceVolume {
            voxels: volume,
            intensity: 10.0,
            ..default()
        },
        Transform::from_scale(Vec3::splat(20.0)),
    ));
    // Both global and local environment probes use native Bevy gathering.
    let environment = EnvironmentMapLight {
        diffuse_map: cube.clone(),
        specular_map: cube,
        intensity: 10.0,
        ..default()
    };
    commands.spawn((
        LightProbe::default(),
        environment.clone(),
        Transform::from_scale(Vec3::splat(20.0)),
    ));
    let mut raw = Image::new_target_texture(32, 32, TextureFormat::Rgba32Float, None);
    raw.texture_descriptor.usage |= TextureUsages::COPY_DST | TextureUsages::COPY_SRC;
    let raw = images.add(raw);
    let mut targets = Vec::new();
    for index in 0..3 {
        let target = images.add(Image::new_target_texture(
            32,
            32,
            TextureFormat::Rgba8UnormSrgb,
            None,
        ));
        targets.push(target.clone());
        let mut camera = commands.spawn((
            Camera {
                order: index as isize,
                ..default()
            },
            Camera3d::default(),
            Hdr,
            Msaa::Off,
            DepthPrepass,
            NormalPrepass,
            DeferredPrepass,
            Projection::Orthographic(OrthographicProjection {
                near: 0.01,
                far: 50.0,
                scaling_mode: bevy::camera::ScalingMode::FixedHorizontal {
                    viewport_width: 2.0,
                },
                ..OrthographicProjection::default_3d()
            }),
            Transform::from_xyz(0.0, 0.5, 0.0).looking_at(Vec3::ZERO, Vec3::Z),
            RenderTarget::Image(target.into()),
            Exposure { ev100: 0.0 },
            Tonemapping::Linear,
            environment.clone(),
            ScreenSpaceReflections::default(),
            ViewIndex(index),
        ));
        if index < 2 {
            camera.insert(RadianceCascades {
                resolution_divisor: 1,
                environment: LinearRgba::BLACK,
                ..default()
            });
        }
        if index == 0 {
            camera.insert((
                AmbientLight {
                    brightness: 7.0,
                    ..default()
                },
                RadianceCascadesRawOutput(raw.clone()),
            ));
        }
    }
    commands.spawn(Readback::texture(raw.clone())).observe(
        |event: On<ReadbackComplete>, mut check: ResMut<Check>| {
            // width 32 gives aligned 512-byte RGBA32F rows. Demand a real GI dispatch.
            check.raw_valid = event.data.len() == 32 * 32 * 16
                && event
                    .data
                    .as_chunks::<16>()
                    .0
                    .iter()
                    .all(|p| f32::from_le_bytes(p[12..16].try_into().unwrap()) > 0.0);
        },
    );
    commands.insert_resource(Check {
        targets,
        raw_valid: false,
        phase: 0,
        ready_at: 1.0,
        requested: false,
        means: [None; 3],
    });
}

#[allow(clippy::type_complexity)]
fn audit_preparation(
    views: Query<(
        &ViewIndex,
        Option<&AmbientLight>,
        Has<RenderViewLightProbes<EnvironmentMapLight>>,
        Has<RenderViewLightProbes<IrradianceVolume>>,
        Has<ScreenSpaceReflectionsPipelineId>,
    )>,
    global: Res<GlobalAmbientLight>,
    audit: Res<Audit>,
) {
    let mut audit = audit.0.lock().unwrap();
    for (index, ambient, environment, volume, ssr) in &views {
        audit[index.0] = Some(Lighting {
            ambient: ambient.map_or(global.brightness, |a| a.brightness),
            environment,
            volume,
            ssr,
        });
    }
}

fn check(
    mut commands: Commands,
    time: Res<Time<Real>>,
    mut check: ResMut<Check>,
    audit: Res<Audit>,
    mut global: ResMut<GlobalAmbientLight>,
    views: Query<(Entity, &ViewIndex, Option<&AmbientLight>)>,
    mut exit: MessageWriter<AppExit>,
) {
    if time.elapsed_secs() > 240.0 {
        error!(
            "Lighting ownership check timed out in phase {}: {:?}",
            check.phase,
            *audit.0.lock().unwrap()
        );
        exit.write(AppExit::error());
        return;
    }
    if check.means.iter().all(Option::is_some) {
        let means = check.means.map(Option::unwrap);
        assert!(
            means.iter().all(|mean| mean.is_finite() && *mean > 0.0),
            "A lighting ownership view rendered black or invalid pixels: {means:?}"
        );
        info!(
            "Lighting ownership phase {} passed: RGB means {means:?}",
            check.phase
        );
        // The game components remain untouched; only render-world state changes.
        let (_, _, ambient) = views.iter().find(|(_, i, _)| i.0 == 0).unwrap();
        assert_eq!(
            ambient.map(|a| a.brightness),
            [Some(7.0), Some(13.0), Some(13.0), None][check.phase]
        );
        if check.phase == 3 {
            exit.write(AppExit::Success);
            return;
        }
        check.phase += 1;
        check.means = [None; 3];
        check.requested = false;
        check.ready_at = time.elapsed_secs_f64() + 0.5;
        for (entity, index, _) in &views {
            if index.0 >= 2 {
                continue;
            }
            let mut view = commands.entity(entity);
            match check.phase {
                1 => {
                    if index.0 == 0 {
                        view.remove::<RadianceCascades>().insert(AmbientLight {
                            brightness: 13.0,
                            ..default()
                        });
                    } else {
                        view.insert(RadianceCascades {
                            intensity: 0.0,
                            ..default()
                        });
                    }
                    global.brightness = 21.0;
                }
                2 => {
                    view.insert(RadianceCascades {
                        resolution_divisor: 1,
                        environment: LinearRgba::BLACK,
                        ..default()
                    });
                }
                _ => {
                    if index.0 == 0 {
                        view.remove::<(RadianceCascades, AmbientLight)>();
                    } else {
                        view.insert(RadianceCascades {
                            intensity: 0.0,
                            ..default()
                        });
                    }
                }
            }
        }
    }
    if check.requested || time.elapsed_secs_f64() < check.ready_at || !check.raw_valid {
        return;
    }
    let on = |ambient| {
        Some(Lighting {
            ambient,
            // Bevy shares the environment-cubemap enable bit between diffuse
            // irradiance and specular reflection, so SRC leaves that native
            // source enabled. Use RadianceCascadesEnvironment for SRC diffuse.
            environment: true,
            volume: false,
            ssr: true,
        })
    };
    let off = |ambient| {
        Some(Lighting {
            ambient,
            environment: true,
            volume: true,
            ssr: true,
        })
    };
    let expected = match check.phase {
        0 => [on(0.0), on(0.0), off(80.0)],
        1 => [off(13.0), off(21.0), off(21.0)],
        2 => [on(0.0), on(0.0), off(21.0)],
        _ => [off(21.0), off(21.0), off(21.0)],
    };
    if *audit.0.lock().unwrap() != expected {
        return;
    }
    check.requested = true;
    for (index, target) in check.targets.iter().enumerate() {
        commands.spawn(Screenshot::image(target.clone())).observe(
            move |event: On<ScreenshotCaptured>,
                  mut check: ResMut<Check>,
                  mut commands: Commands| {
                let data = event.image.data.as_ref().unwrap();
                assert_eq!(data.len(), 32 * 32 * 4);
                check.means[index] = Some(
                    data.as_chunks::<4>()
                        .0
                        .iter()
                        .map(|p| (f32::from(p[0]) + f32::from(p[1]) + f32::from(p[2])) / 3.0)
                        .sum::<f32>()
                        / (32.0 * 32.0),
                );
                commands.entity(event.entity).despawn();
            },
        );
    }
}

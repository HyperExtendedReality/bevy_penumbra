use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use bevy::{
    core_pipeline::{
        Core3dSystems,
        core_3d::{main_opaque_pass_3d, main_transparent_pass_3d},
        prepass::ViewPrepassTextures,
        schedule::Core3d,
    },
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        camera::ExtractedCamera,
        diagnostic::RecordDiagnostics,
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
        view::{ExtractedView, ViewTarget},
    },
    shader::Shader,
};
use bevy_slang::{SlangCompiler, SlangLoaderError, SlangSettings, SlangTarget};

use crate::{
    RadianceCascades,
    scene::{BvhNode, GiLight, GiScene, Triangle},
};

const LEVELS: usize = 4;
const ENTRY_SIZE: u64 = 80;
const PROBE_SIZE: u64 = 112;
const COMPUTE_ENTRIES: [&str; 15] = [
    "clear",
    "seed",
    "parents",
    "rank",
    "prefix",
    "generateDirections",
    "retain",
    "trace",
    "reduce",
    "merge",
    "seedIrradiance",
    "prefilter",
    "gather",
    "temporal",
    "denoise",
];
const MODULES: [(&str, &[&str]); 2] = [
    ("cascades.slang", &COMPUTE_ENTRIES),
    ("composite.slang", &["vertex", "fragment"]),
];
const SOURCES: &[(&str, &str)] = &[
    ("common.slang", include_str!("../shaders/common.slang")),
    ("cascades.slang", include_str!("../shaders/cascades.slang")),
    (
        "composite.slang",
        include_str!("../shaders/composite.slang"),
    ),
];

fn compile(
    compiler: &SlangCompiler,
    file: &str,
    entries: &[&str],
) -> Result<Shader, SlangLoaderError> {
    let mut shader = compiler.compile_bundle(
        file,
        SOURCES,
        &SlangSettings {
            target: SlangTarget::Wgsl,
            entry_points: entries.iter().map(|entry| (*entry).to_owned()).collect(),
            optimization: Some(2),
            ..default()
        },
    )?;
    shader.validate_shader = bevy::shader::ValidateShader::Enabled;
    Ok(shader)
}

#[derive(Clone, Copy, ShaderType)]
struct Parameters {
    world_from_clip: Mat4,
    previous_clip_from_world: Mat4,
    previous_world_from_clip: Mat4,
    viewport: UVec4,
    size: UVec4,
    tuning: Vec4,
    environment: Vec4,
    history: Vec4,
    frame: UVec4,
    light_counts: UVec4,
    stability: Vec4,
}

#[derive(Resource)]
struct Pipelines {
    compute_layout: BindGroupLayoutDescriptor,
    composite_layout: BindGroupLayoutDescriptor,
    compute: [CachedComputePipelineId; COMPUTE_ENTRIES.len()],
    composite: CachedRenderPipelineId,
}

#[derive(Resource)]
struct PipelineShaders(Vec<Handle<Shader>>);

pub(crate) fn install(app: &mut App, compiler: SlangCompiler) {
    if app.get_sub_app(RenderApp).is_none() {
        return;
    }
    let compiled: Result<Vec<_>, _> = MODULES
        .into_iter()
        .map(|(file, entries)| compile(&compiler, file, entries))
        .collect();
    let compiled = match compiled {
        Ok(shaders) => shaders,
        Err(error) => {
            error!("Radiance Cascades disabled: {error}. Install Slang 2026.19+ and set SLANGC.");
            return;
        }
    };
    let handles: Vec<_> = compiled
        .into_iter()
        .map(|s| app.world_mut().resource_mut::<Assets<Shader>>().add(s))
        .collect();
    app.sub_app_mut(RenderApp)
        .insert_resource(PipelineShaders(handles))
        .add_systems(RenderStartup, init_pipelines)
        .add_systems(Render, prepare.in_set(RenderSystems::PrepareBindGroups))
        .add_systems(
            Core3d,
            render
                .in_set(Core3dSystems::MainPass)
                .after(main_opaque_pass_3d)
                .before(main_transparent_pass_3d),
        );
}

fn init_pipelines(
    mut commands: Commands,
    cache: Res<PipelineCache>,
    shaders: Res<PipelineShaders>,
) {
    let handles = &shaders.0;
    let compute_layout = BindGroupLayoutDescriptor::new(
        "split_rc_compute",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<Parameters>(false),
                storage_buffer_read_only::<Vec<Triangle>>(false),
                storage_buffer_read_only::<Vec<BvhNode>>(false),
                storage_buffer_read_only::<Vec<GiLight>>(false),
                texture_2d(TextureSampleType::Depth),
                texture_2d(TextureSampleType::Float { filterable: false }),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                texture_2d(TextureSampleType::Float { filterable: false }),
                storage_buffer_sized(false, None),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                texture_2d(TextureSampleType::Float { filterable: false }),
            ),
        ),
    );
    let composite_layout = BindGroupLayoutDescriptor::new(
        "split_rc_composite",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                uniform_buffer::<Parameters>(false),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Depth),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Uint),
            ),
        ),
    );
    let compute = std::array::from_fn(|i| {
        cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(format!("split_rc_{}", COMPUTE_ENTRIES[i]).into()),
            layout: vec![compute_layout.clone()],
            shader: handles[0].clone(),
            entry_point: Some(COMPUTE_ENTRIES[i].into()),
            ..default()
        })
    });
    let composite = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("split_rc_composite".into()),
        layout: vec![composite_layout.clone()],
        vertex: VertexState {
            shader: handles[1].clone(),
            entry_point: Some("vertex".into()),
            ..default()
        },
        fragment: Some(FragmentState {
            shader: handles[1].clone(),
            entry_point: Some("fragment".into()),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba16Float,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        ..default()
    });
    commands.insert_resource(Pipelines {
        compute_layout,
        composite_layout,
        compute,
        composite,
    });
}

#[derive(Resource, Default)]
struct GpuScene {
    revision: u64,
    geometry_revision: u64,
    triangles: StorageBuffer<Vec<Triangle>>,
    nodes: StorageBuffer<Vec<BvhNode>>,
    lights: StorageBuffer<Vec<GiLight>>,
}

struct CacheBuffers {
    heads: Buffer,
    entries: Buffer,
}

#[derive(Component)]
struct ViewBuffers {
    size: UVec2,
    ray_layers: u32,
    buckets: u32,
    caches: [CacheBuffers; 2],
    irradiance_probes: Buffer,
    irradiance_textures: [Texture; 4],
    irradiance_views: [TextureView; 4],
    _metadata: [Texture; 2],
    metadata_views: [TextureView; 2],
    previous_clip_from_world: Mat4,
    previous_world_from_clip: Mat4,
    viewport: UVec4,
    frame: AtomicU32,
    history_valid: AtomicBool,
    screen_history_valid: AtomicBool,
    revision: u64,
    settings: RadianceCascades,
}

#[derive(Component)]
struct ViewBindings {
    compute: [BindGroup; LEVELS],
    denoise: [BindGroup; LEVELS],
    temporal: BindGroup,
    composite: [(TextureViewId, BindGroup); 2],
}

fn resolution(viewport: UVec2, settings: RadianceCascades) -> UVec2 {
    let size_at = |divisor: u32| {
        UVec2::new(
            viewport.x.div_ceil(divisor).max(1),
            viewport.y.div_ceil(divisor).max(1),
        )
    };
    let mut low = settings.resolution_divisor.max(1);
    let mut high = viewport.max_element().max(low);
    while low < high {
        let middle = low + (high - low) / 2;
        let size = size_at(middle);
        if u64::from(size.x) * u64::from(size.y) > u64::from(settings.max_rays) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    size_at(low)
}

impl ViewBuffers {
    fn new(device: &RenderDevice, size: UVec2, settings: RadianceCascades, revision: u64) -> Self {
        let samples = size.x * size.y;
        let ray_layers = 1 + u32::from(settings.bounce_feedback > 0.0);
        let buckets = (samples * 2).next_power_of_two().max(1024);
        let caches = std::array::from_fn(|_| CacheBuffers {
            heads: device.create_buffer(&BufferDescriptor {
                label: Some("split_rc_hash_heads"),
                size: u64::from(buckets * 2 + 1) * 4,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            entries: device.create_buffer(&BufferDescriptor {
                label: Some("split_rc_intervals"),
                size: u64::from(samples * ray_layers) * LEVELS as u64 * ENTRY_SIZE * 2,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
        });
        let irradiance_probes = device.create_buffer(&BufferDescriptor {
            label: Some("split_rc_irradiance_probes"),
            size: u64::from(buckets) * PROBE_SIZE * 2,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let make_texture = || {
            device.create_texture(&TextureDescriptor {
                label: Some("split_rc_irradiance"),
                size: Extent3d {
                    width: size.x,
                    height: size.y,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba32Float,
                usage: TextureUsages::STORAGE_BINDING
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_SRC
                    | TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let irradiance: [Texture; 4] = std::array::from_fn(|_| make_texture());
        let metadata: [Texture; 2] = std::array::from_fn(|_| make_texture());
        let metadata_views =
            std::array::from_fn(|i| metadata[i].create_view(&TextureViewDescriptor::default()));
        let irradiance_views =
            std::array::from_fn(|i| irradiance[i].create_view(&TextureViewDescriptor::default()));
        Self {
            size,
            ray_layers,
            buckets,
            caches,
            irradiance_probes,
            irradiance_textures: irradiance,
            irradiance_views,
            _metadata: metadata,
            metadata_views,
            previous_clip_from_world: Mat4::IDENTITY,
            previous_world_from_clip: Mat4::IDENTITY,
            viewport: UVec4::ZERO,
            frame: AtomicU32::new(0),
            history_valid: AtomicBool::new(false),
            screen_history_valid: AtomicBool::new(false),
            revision,
            settings,
        }
    }
}

// Bevy system parameters and query tuples describe the render-world inputs.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn prepare(
    mut commands: Commands,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    cache: Res<PipelineCache>,
    pipelines: Res<Pipelines>,
    scene: Res<GiScene>,
    mut gpu_scene: Option<ResMut<GpuScene>>,
    mut views: Query<(
        Entity,
        &ExtractedView,
        &ExtractedCamera,
        &ViewTarget,
        &ViewPrepassTextures,
        &RadianceCascades,
        Option<&mut ViewBuffers>,
    )>,
) {
    if device.limits().max_storage_buffers_per_shader_stage < 8 {
        warn_once!("Radiance Cascades needs eight storage buffers per compute stage");
        return;
    }
    if scene.data.triangles.len() as u64 * 80 > device.limits().max_storage_buffer_binding_size
        || scene.data.nodes.len() as u64 * 32 > device.limits().max_storage_buffer_binding_size
        || (scene.data.lights.len() + scene.data.emissive_lights.len()) as u64 * 80
            > device.limits().max_storage_buffer_binding_size
    {
        warn_once!("GI scene exceeds this device's storage buffer limit");
        for (entity, ..) in &views {
            commands.entity(entity).remove::<ViewBindings>();
        }
        return;
    }
    let mut new_gpu_scene = None;
    if gpu_scene
        .as_ref()
        .is_none_or(|s| s.revision != scene.revision)
    {
        let upload_geometry = gpu_scene
            .as_ref()
            .is_none_or(|s| s.geometry_revision != scene.geometry_revision);
        let mut next = gpu_scene
            .as_mut()
            .map_or_else(GpuScene::default, |s| std::mem::take(&mut **s));
        if upload_geometry {
            let mut triangles = StorageBuffer::from(if scene.data.triangles.is_empty() {
                vec![Triangle::default()]
            } else {
                scene.data.triangles.as_ref().clone()
            });
            let mut nodes = StorageBuffer::from(if scene.data.nodes.is_empty() {
                vec![BvhNode::default()]
            } else {
                scene.data.nodes.as_ref().clone()
            });
            triangles.write_buffer(&device, &queue);
            nodes.write_buffer(&device, &queue);
            next.triangles = triangles;
            next.nodes = nodes;
            next.geometry_revision = scene.geometry_revision;
        }
        let mut lights = StorageBuffer::from(
            if scene.data.lights.is_empty() && scene.data.emissive_lights.is_empty() {
                vec![GiLight::default()]
            } else {
                scene
                    .data
                    .lights
                    .iter()
                    .chain(scene.data.emissive_lights.iter())
                    .copied()
                    .collect()
            },
        );
        lights.write_buffer(&device, &queue);
        next.revision = scene.revision;
        next.lights = lights;
        if let Some(gpu_scene) = gpu_scene.as_mut() {
            **gpu_scene = next;
        } else {
            new_gpu_scene = Some(next);
        }
    }
    let Some(gpu_scene_ref) = new_gpu_scene.as_ref().or(gpu_scene.as_deref()) else {
        return;
    };
    let (Some(triangles), Some(nodes), Some(lights)) = (
        gpu_scene_ref.triangles.binding(),
        gpu_scene_ref.nodes.binding(),
        gpu_scene_ref.lights.binding(),
    ) else {
        return;
    };
    for (entity, view, camera, target, prepass, settings, existing) in &mut views {
        let mut settings = settings.sanitized();
        if settings.intensity <= 0.0 {
            if let Some(buffers) = existing {
                buffers.history_valid.store(false, Ordering::Relaxed);
                buffers.screen_history_valid.store(false, Ordering::Relaxed);
            }
            commands.entity(entity).remove::<ViewBindings>();
            continue;
        }
        let layers = 1 + u32::from(settings.bounce_feedback > 0.0);
        settings.max_rays = settings
            .max_rays
            .min(
                (device.limits().max_storage_buffer_binding_size
                    / (ENTRY_SIZE * LEVELS as u64 * 2 * u64::from(layers))) as u32,
            )
            .max(1);
        let max_buckets = (device.limits().max_storage_buffer_binding_size / (PROBE_SIZE * 2))
            .min(u64::from(u32::MAX)) as u32;
        let max_buckets = 1u32 << (31 - max_buckets.max(1).leading_zeros());
        settings.max_rays = settings.max_rays.min((max_buckets / 2).max(1));
        let (Some(depth), Some(normal), Some(gbuffer)) = (
            prepass.depth_view(),
            prepass.normal_view(),
            prepass.deferred_view(),
        ) else {
            if let Some(buffers) = existing {
                buffers.history_valid.store(false, Ordering::Relaxed);
                buffers.screen_history_valid.store(false, Ordering::Relaxed);
            }
            commands.entity(entity).remove::<ViewBindings>();
            continue;
        };
        if target.main_texture_format() != TextureFormat::Rgba16Float
            || view.viewport.z == 0
            || view.viewport.w == 0
        {
            if let Some(buffers) = existing {
                buffers.history_valid.store(false, Ordering::Relaxed);
                buffers.screen_history_valid.store(false, Ordering::Relaxed);
            }
            commands.entity(entity).remove::<ViewBindings>();
            continue;
        }
        let size = resolution(UVec2::new(view.viewport.z, view.viewport.w), settings);
        let mut created = None;
        let mut existing = existing;
        if existing
            .as_ref()
            .is_none_or(|v| v.size != size || v.ray_layers != layers)
        {
            created = Some(ViewBuffers::new(&device, size, settings, scene.revision));
        }
        let Some(buffers) = created.as_mut().or(existing.as_deref_mut()) else {
            continue;
        };
        let scene_changed = buffers.revision != scene.revision;
        if scene_changed || buffers.settings != settings || buffers.viewport != view.viewport {
            if buffers.settings != settings || buffers.viewport != view.viewport {
                buffers.screen_history_valid.store(false, Ordering::Relaxed);
            }
            // Keep cache parity and the R2 sequence advancing during motion.
            // Only history validity resets; restarting samples repeats noise.
            buffers.history_valid.store(false, Ordering::Relaxed);
            buffers.revision = scene.revision;
            buffers.settings = settings;
            buffers.viewport = view.viewport;
        }
        let frame = buffers.frame.load(Ordering::Relaxed);
        let current = &buffers.caches[(frame % 2) as usize];
        let previous = &buffers.caches[(frame.wrapping_add(1) % 2) as usize];
        let irradiance = &buffers.irradiance_views[(frame % 2) as usize];
        let previous_irradiance = &buffers.irradiance_views[(frame.wrapping_add(1) % 2) as usize];
        let metadata = &buffers.metadata_views[(frame % 2) as usize];
        let previous_metadata = &buffers.metadata_views[(frame.wrapping_add(1) % 2) as usize];
        let clip_from_world = view.clip_from_world.unwrap_or_else(|| {
            view.clip_from_view * Mat4::from(view.world_from_view.affine().inverse())
        });
        let params = Parameters {
            world_from_clip: clip_from_world.inverse(),
            previous_clip_from_world: buffers.previous_clip_from_world,
            previous_world_from_clip: buffers.previous_world_from_clip,
            viewport: view.viewport,
            size: UVec4::new(size.x, size.y, size.x * size.y, buffers.buckets),
            tuning: Vec4::new(
                settings.probe_spacing,
                settings.interval_length,
                settings.normal_bias,
                settings.intensity,
            ),
            environment: settings.environment.to_vec4().with_w(camera.exposure),
            history: Vec4::new(
                settings.temporal_blend,
                settings.bounce_feedback,
                if buffers.history_valid.load(Ordering::Relaxed) {
                    1.0
                } else {
                    0.0
                },
                settings.history_frames as f32,
            ),
            frame: UVec4::new(
                frame,
                0,
                scene.data.triangles.len() as u32,
                scene.data.lights.len() as u32,
            ),
            light_counts: UVec4::new(
                scene.data.lights.len() as u32,
                scene.data.emissive_lights.len() as u32,
                settings.emissive_samples,
                0,
            ),
            stability: Vec4::new(
                if buffers.screen_history_valid.load(Ordering::Relaxed) {
                    1.0
                } else {
                    0.0
                },
                if scene_changed { 1.0 } else { 0.0 },
                0.0,
                0.0,
            ),
        };
        let uniforms: [UniformBuffer<Parameters>; LEVELS] = std::array::from_fn(|level| {
            let mut p = params;
            p.frame.y = level as u32;
            let mut buffer = UniformBuffer::from(p);
            buffer.write_buffer(&device, &queue);
            buffer
        });
        let bind_compute = |level: usize,
                            output: &TextureView,
                            input: &TextureView,
                            sample_input: &TextureView,
                            metadata_output: &TextureView,
                            metadata_input: &TextureView| {
            device.create_bind_group(
                "split_rc_compute",
                &cache.get_bind_group_layout(&pipelines.compute_layout),
                &BindGroupEntries::sequential((
                    &uniforms[level],
                    triangles.clone(),
                    nodes.clone(),
                    lights.clone(),
                    depth,
                    normal,
                    current.heads.as_entire_binding(),
                    current.entries.as_entire_binding(),
                    previous.heads.as_entire_binding(),
                    previous.entries.as_entire_binding(),
                    output,
                    input,
                    buffers.irradiance_probes.as_entire_binding(),
                    sample_input,
                    metadata_output,
                    metadata_input,
                )),
            )
        };
        let compute = std::array::from_fn(|level| {
            bind_compute(
                level,
                irradiance,
                previous_irradiance,
                previous_irradiance,
                metadata,
                previous_metadata,
            )
        });
        let temporal = bind_compute(
            0,
            &buffers.irradiance_views[2],
            previous_irradiance,
            irradiance,
            metadata,
            previous_metadata,
        );
        let denoise = std::array::from_fn(|level| {
            let output = &buffers.irradiance_views[2 + level % 2];
            let input = if level == 0 {
                irradiance
            } else {
                &buffers.irradiance_views[2 + (level - 1) % 2]
            };
            bind_compute(level, output, input, metadata, previous_metadata, metadata)
        });
        // Raw ping-pong irradiance is temporal history. Filtering only the
        // display output avoids recursively diffusing lighting across surfaces.
        let display = if settings.spatial_filter_steps == 0 {
            irradiance
        } else {
            &buffers.irradiance_views[2 + (settings.spatial_filter_steps as usize - 1) % 2]
        };
        let composite =
            [target.main_texture_view(), target.main_texture_other_view()].map(|source| {
                (
                    source.id(),
                    device.create_bind_group(
                        "split_rc_composite",
                        &cache.get_bind_group_layout(&pipelines.composite_layout),
                        &BindGroupEntries::sequential((
                            &uniforms[0],
                            source,
                            display,
                            depth,
                            normal,
                            gbuffer,
                        )),
                    ),
                )
            });
        commands.entity(entity).insert(ViewBindings {
            compute,
            denoise,
            temporal,
            composite,
        });
        buffers.previous_clip_from_world = clip_from_world;
        buffers.previous_world_from_clip = params.world_from_clip;
        if let Some(created) = created {
            commands.entity(entity).insert(created);
        }
    }
    if let Some(gpu_scene) = new_gpu_scene {
        commands.insert_resource(gpu_scene);
    }
}

fn render(
    view: ViewQuery<(&ViewTarget, &RadianceCascades, &ViewBuffers, &ViewBindings)>,
    pipelines: Res<Pipelines>,
    cache: Res<PipelineCache>,
    mut context: RenderContext,
) {
    let (target, settings, buffers, bindings) = view.into_inner();
    if settings.intensity <= 0.0 {
        return;
    }
    let [
        Some(clear),
        Some(seed),
        Some(parents),
        Some(rank),
        Some(prefix),
        Some(sample),
        Some(retain),
        Some(trace),
        Some(reduce),
        Some(merge),
        Some(seed_irradiance),
        Some(prefilter),
        Some(gather),
        Some(temporal),
        Some(denoise),
    ] = pipelines.compute.map(|id| cache.get_compute_pipeline(id))
    else {
        return;
    };
    let Some(composite) = cache.get_render_pipeline(pipelines.composite) else {
        return;
    };
    let diagnostics = context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let total_span = diagnostics.time_span(context.command_encoder(), "split_rc");
    let mut dispatch =
        |name: &'static str, pipeline: &ComputePipeline, level: usize, x: u32, y: u32| {
            let span = diagnostics.time_span(context.command_encoder(), name);
            let mut pass = context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("split_rc"),
                    ..default()
                });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bindings.compute[level], &[]);
            pass.dispatch_workgroups(x, y, 1);
            drop(pass);
            span.end(context.command_encoder());
        };
    dispatch("clear", clear, 0, buffers.buckets.div_ceil(64), 1);
    dispatch(
        "seed",
        seed,
        0,
        buffers.size.x.div_ceil(8),
        buffers.size.y.div_ceil(8),
    );
    dispatch("rank", rank, 0, buffers.buckets.div_ceil(64), 1);
    for level in 1..LEVELS {
        dispatch(
            ["parents_0", "parents_1", "parents_2", "parents_3"][level],
            parents,
            level,
            (buffers.size.x * buffers.size.y).div_ceil(64),
            1,
        );
        dispatch(
            ["rank_0", "rank_1", "rank_2", "rank_3"][level],
            rank,
            level,
            buffers.buckets.div_ceil(64),
            1,
        );
    }
    for level in (0..LEVELS).rev() {
        dispatch(
            ["prefix_0", "prefix_1", "prefix_2", "prefix_3"][level],
            prefix,
            level,
            buffers.buckets.div_ceil(64),
            1,
        );
    }
    dispatch(
        "sample",
        sample,
        0,
        buffers.size.x.div_ceil(8),
        buffers.size.y.div_ceil(8),
    );
    dispatch("reset_heads", clear, 0, buffers.buckets.div_ceil(64), 1);
    dispatch("retain", retain, 0, buffers.buckets.div_ceil(64), 1);
    dispatch(
        "trace",
        trace,
        0,
        buffers.size.x.div_ceil(8),
        buffers.size.y.div_ceil(8),
    );
    dispatch("reduce", reduce, 0, buffers.buckets.div_ceil(64), 1);
    for level in (0..LEVELS).rev() {
        dispatch(
            ["merge_0", "merge_1", "merge_2", "merge_3"][level],
            merge,
            level,
            (buffers.size.x * buffers.size.y * buffers.ray_layers).div_ceil(64),
            1,
        );
    }
    dispatch(
        "seed_irradiance",
        seed_irradiance,
        0,
        buffers.buckets.div_ceil(64),
        1,
    );
    dispatch("prefilter", prefilter, 0, buffers.buckets.div_ceil(64), 1);
    dispatch(
        "gather",
        gather,
        0,
        buffers.size.x.div_ceil(8),
        buffers.size.y.div_ceil(8),
    );
    let temporal_span = diagnostics.time_span(context.command_encoder(), "temporal");
    let mut pass = context
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("split_rc_temporal"),
            ..default()
        });
    pass.set_pipeline(temporal);
    pass.set_bind_group(0, &bindings.temporal, &[]);
    pass.dispatch_workgroups(buffers.size.x.div_ceil(8), buffers.size.y.div_ceil(8), 1);
    drop(pass);
    // Only temporal results become history; spatially filtered display values
    // never feed back and spread lighting recursively across contacts.
    let frame = buffers.frame.load(Ordering::Relaxed);
    context.command_encoder().copy_texture_to_texture(
        buffers.irradiance_textures[2].as_image_copy(),
        buffers.irradiance_textures[(frame & 1) as usize].as_image_copy(),
        Extent3d {
            width: buffers.size.x,
            height: buffers.size.y,
            depth_or_array_layers: 1,
        },
    );
    temporal_span.end(context.command_encoder());
    let steps = buffers.settings.spatial_filter_steps;
    for level in 0..steps as usize {
        let span = diagnostics.time_span(
            context.command_encoder(),
            ["denoise_0", "denoise_1", "denoise_2", "denoise_3"][level],
        );
        let mut pass = context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("split_rc_denoise"),
                ..default()
            });
        pass.set_pipeline(denoise);
        pass.set_bind_group(0, &bindings.denoise[level], &[]);
        pass.dispatch_workgroups(buffers.size.x.div_ceil(8), buffers.size.y.div_ceil(8), 1);
        drop(pass);
        span.end(context.command_encoder());
    }

    let post = target.post_process_write();
    let group = if bindings.composite[0].0 == post.source.id() {
        &bindings.composite[0].1
    } else {
        &bindings.composite[1].1
    };
    let mut pass = context
        .command_encoder()
        .begin_render_pass(&RenderPassDescriptor {
            label: Some("split_rc_composite"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: post.destination,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                },
            })],
            ..default()
        });
    pass.set_pipeline(composite);
    pass.set_bind_group(0, group, &[]);
    pass.draw(0..3, 0..1);
    drop(pass);
    total_span.end(context.command_encoder());
    buffers.frame.fetch_add(1, Ordering::Relaxed);
    buffers.history_valid.store(true, Ordering::Relaxed);
    buffers.screen_history_valid.store(true, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolution_respects_budget_and_zero_viewports() {
        let settings = RadianceCascades {
            max_rays: 8192,
            ..default()
        };
        for viewport in [
            UVec2::ZERO,
            UVec2::ONE,
            UVec2::new(7680, 4320),
            UVec2::new(1, 50000),
            UVec2::splat(u32::MAX),
        ] {
            let size = resolution(viewport, settings);
            assert!(size.x > 0 && size.y > 0 && size.x * size.y <= settings.max_rays);
        }
    }

    #[test]
    #[ignore = "requires slangc (or SLANGC) on PATH"]
    fn compile_all_embedded_slang_stages() {
        for (file, entries) in MODULES {
            let shader = compile(&SlangCompiler::default(), file, entries).expect(file);
            let bevy::shader::Source::Wgsl(source) = shader.source else {
                panic!("expected Slang's WGSL output")
            };
            let module = naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|error| panic!("{file}: {}", error.emit_to_string(&source)));
            for entry in entries {
                assert!(module.entry_points.iter().any(|point| point.name == *entry));
            }
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .unwrap_or_else(|error| panic!("{file}: {error:?}"));
        }
    }
}

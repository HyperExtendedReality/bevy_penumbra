//! Ignored GPU checks for the production transport helpers.
use bevy::{
    asset::RenderAssetUsages,
    math::{Mat4, UVec2, UVec4, Vec3, Vec4},
    prelude::{Image, default},
    render::{
        render_resource::*,
        renderer::initialize_renderer,
        settings::{Backends, WgpuSettings},
    },
    tasks::block_on,
};
use bevy_slang::{SlangCompiler, SlangSettings, SlangTarget};

#[path = "composite_tests.rs"]
mod composite_tests;
#[path = "emitter_tests.rs"]
mod emitter_tests;
#[path = "lod_tests.rs"]
mod lod_tests;
#[path = "split_merge_tests.rs"]
mod split_merge_tests;
#[path = "world_resolve_tests.rs"]
mod world_resolve_tests;

#[derive(Clone, Copy, ShaderType)]
struct TransportParameters {
    world_from_clip: Mat4,
    clip_from_world: Mat4,
    previous_clip_from_world: Mat4,
    previous_world_from_clip: Mat4,
    camera_position: Vec4,
    viewport: UVec4,
    size: UVec4,
    tuning: Vec4,
    environment: Vec4,
    history: Vec4,
    interval_history: Vec4,
    frame: UVec4,
    light_counts: UVec4,
    stability: Vec4,
    layers: UVec4,
    environment_info: UVec4,
    environment_rotation: Vec4,
    environment_scale: Vec4,
    geometry_pages: UVec4,
    scene_pages: UVec4,
    scene_overlay: UVec4,
    dirty_min: Vec4,
    dirty_max: Vec4,
    transport: Vec4,
    sampling: UVec4,
    owner_sequence: UVec4,
    cache_time: UVec4,
    world_receivers: UVec4,
    world_receiver_state: UVec4,
    light_dirty_min: Vec4,
    light_dirty_max: Vec4,
}

const TRANSPORT_SHADER: &str = r#"
#include "common.slang"

[[vk::binding(0, 0)]] ConstantBuffer<Parameters> params;
[[vk::binding(3, 0)]] StructuredBuffer<Light> lights;
bool traceOccluded(float3 origin, float3 direction, float limit,
                   out bool querySupported, uint admissionLimit) {
    uint mode = params.ownerSequence.z;
    querySupported = mode != 2u;
    return mode == 1u;
}
bool traceOccludedTracked(float3 origin, float3 direction, float limit,
                          out bool querySupported, out DependencyContext dependency,
                          uint admissionLimit) {
    bool blocked = traceOccluded(origin, direction, limit, querySupported, admissionLimit);
    dependency = querySupported ? dependencyEmpty() : dependencyUnknown();
    if (querySupported)
        dependencyIncludeSegment(dependency, origin, origin + direction * limit, 0.0);
    return blocked;
}
#include "bsdf.slang"
#include "environment.slang"
[[vk::binding(4, 0)]] RWStructuredBuffer<float4> results;

[shader("compute")]
[numthreads(1, 1, 1)]
void checkTransport(uint3 id : SV_DispatchThreadID) {
    float3 normal = float3(0, 0, 1);
    float3 light = normalize(float3(0.997, 0, 0.08));
    float3 view = normalize(float3(-0.3, 0.4, 0.866));
    for (uint index = 0; index < 3; ++index) {
        float roughness = float(index) * 0.5;
        results[index] = float4(
            disneyDiffuseMultiplier(normal, light, view, roughness),
            disneyDiffuseMultiplier(normal, view, light, roughness),
            roughness, 1.0);
    }
    results[3] = float4(clampOutgoingRadiance(float3(100, 50, 25), params), 1.0);
    results[4] = float4(clampOutgoingRadiance(float3(-1, 1, 1), params), 1.0);
    results[5] = float4(clampOutgoingRadiance(
        float3(asfloat(0x7fc00000u), 1, 2), params), 1.0);

    for (uint index = 0; index < 8; ++index) {
        float4 random = float4((float(index) + 0.25) / 8.0, 0, 0.5, 0.5);
        float3 direction, radiance;
        float pdf;
        bool valid = sampleEnvironment(random, direction, pdf, radiance);
        uint output = 6 + index * 2;
        results[output] = float4(direction, pdf);
        results[output + 1] = float4(
            radiance, valid ? environmentPdf(direction) : -1.0);
    }
    results[22] = float4(
        rawGiSupportAlpha(false, false, 1.0),
        rawGiSupportAlpha(true, false, 1.0),
        rawGiSupportAlpha(true, true, 0.25),
        rawGiSupportAlpha(true, true, 1.0));
    float3 rayZero = decodeDirection(frac(0.5 + r2(1u)));
    results[23] = float4(rayZero, length(rayZero));
    float3 tangent = cosineDirection(float3(0, 1, 0), float2(1, 0));
    results[24] = float4(tangent, length(tangent));
    uint north = directionIndex(float3(0, 1, 0), 3u);
    uint south = directionIndex(float3(0, -1, 0), 3u);
    results[25] = float4(float(north), float(south), 2048.0, 0.0);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void checkEnvironment(uint3 id : SV_DispatchThreadID) {
    float3 normal = float3(0, 1, 0);
    bool primarySupported;
    float3 primary = environmentIrradiance(
        float3(0, 0, 0), normal, normal, normal, params.tuning.y,
        params.sampling.x, params.ownerSequence.w, primarySupported);
    bool secondSupported;
    float3 second = environmentIrradiance(
        float3(3, 2, 1), normal, normal, normal, params.tuning.y,
        params.sampling.x, params.ownerSequence.w + 1u, secondSupported);
    uint outputIndex = params.ownerSequence.x;
    results[outputIndex] = float4(primary, primarySupported ? 1.0 : 0.0);
    results[outputIndex + 1u] = float4(second, secondSupported ? 1.0 : 0.0);
}
"#;
const DENSE_REFERENCE_SHADER: &str = r#"
#include "common.slang"
[[vk::binding(0, 0)]] RWStructuredBuffer<float4> results;

float denseBoxSdf(float3 point, float3 center, float3 halfSize) {
    float3 q = abs(point - center) - halfSize;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float denseSceneSdf(float3 point, out uint material) {
    float distance = denseBoxSdf(point, float3(-3.0, 0.0, 0.0), float3(0.9));
    material = 1u;
    float sphere = length(point - float3(3.0, 1.0, 0.0)) - 1.0;
    if (sphere < distance) { distance = sphere; material = 2u; }
    float floorDistance = denseBoxSdf(point, float3(0.0, 3.0, 0.0), float3(2.0, 0.04, 2.0));
    if (floorDistance < distance) { distance = floorDistance; material = 3u; }
    float blocker = denseBoxSdf(point, float3(0.0, 0.0, 4.0), float3(2.0, 0.5, 0.03));
    if (blocker < distance) { distance = blocker; material = 4u; }
    return distance;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void checkDenseSdf(uint3 thread : SV_DispatchThreadID) {
    if (thread.x >= 245760u) return;
    uint level = 0u;
    uint offset = 0u;
    // Keep a valid division fallback: Slang's compile-time diagnostics do not
    // prove this bounded search always finds the active record's level.
    uint dimension = 1u;
    uint directionCount = 1u;
    bool found = false;
    [loop] for (uint candidate = 0u; candidate < 4u; ++candidate) {
        dimension = 16u >> candidate;
        directionCount = 32u << (2u * candidate);
        uint count = dimension * dimension * dimension * directionCount;
        if (thread.x < offset + count) {
            level = candidate;
            found = true;
            break;
        }
        offset += count;
    }
    if (!found) return;
    uint local = thread.x - offset;
    uint directionIndexValue = local % directionCount;
    uint probe = local / directionCount;
    uint x = probe % dimension;
    uint y = (probe / dimension) % dimension;
    uint z = probe / (dimension * dimension);
    float cellSize = float(1u << level);
    float3 origin = -8.0 + (float3(x, y, z) + 0.5) * cellSize;
    float3 direction = binDirection(directionIndexValue, level);
    float start = float((1u << level) - 1u) + 0.001;
    float end = float((1u << (level + 1u)) - 1u);
    float4 interval = float4(0.0, 0.0, 0.0, 1.0);
    [loop] for (uint step = 0u; step < 96u && start <= end; ++step) {
        uint material;
        float distance = denseSceneSdf(origin + direction * start, material);
        if (distance < 0.001) {
            interval = material == 3u
                ? float4(6.0, 2.0, 0.5, 0.0)
                : float4(0.0, 0.0, 0.0, 0.0);
            break;
        }
        start += max(distance, 0.005);
    }
    results[thread.x * 2u] = interval;
    results[thread.x * 2u + 1u] = float4(direction, float(directionIndexValue));
}
"#;
fn hdr_test_image() -> Image {
    let mut pixels: Vec<[f32; 4]> = vec![[0.0, 0.0, 0.0, 1.0]; 2 * 4];
    pixels[0] = [100.0, 60.0, 20.0, 1.0];
    Image::new(
        Extent3d {
            width: 2,
            height: 4,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        pixels
            .iter()
            .flat_map(|pixel| pixel.iter().flat_map(|channel| channel.to_le_bytes()))
            .collect(),
        TextureFormat::Rgba32Float,
        RenderAssetUsages::MAIN_WORLD,
    )
}

fn expected_environment_pdf(table: &crate::environment::EnvironmentTable, direction: Vec3) -> f64 {
    let direction = direction.normalize();
    let u = (direction.z.atan2(direction.x) / std::f32::consts::TAU + 0.5).rem_euclid(1.0);
    let v = direction.x.hypot(direction.z).atan2(direction.y) / std::f32::consts::PI;
    let x = ((u * table.width as f32) as u32).min(table.width - 1);
    let y = ((v * table.height as f32) as u32).min(table.height - 1);
    let index = (x + y * table.width) as usize;
    let pmf = f64::from(f32::from_bits(table.aliases[index].z));
    let theta0 = std::f64::consts::PI * f64::from(y) / f64::from(table.height);
    let theta1 = std::f64::consts::PI * f64::from(y + 1) / f64::from(table.height);
    let solid_angle =
        (std::f64::consts::TAU / f64::from(table.width)) * (theta0.cos() - theta1.cos());
    pmf / solid_angle
}

fn output_vec4(bytes: &[u8], index: usize) -> [f32; 4] {
    let start = index * 16;
    std::array::from_fn(|component| {
        let offset = start + component * 4;
        f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    })
}

fn output_u32x4(bytes: &[u8], index: usize) -> [u32; 4] {
    let start = index * 16;
    std::array::from_fn(|component| {
        let offset = start + component * 4;
        u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    })
}

fn temporal_test_texture_sized(
    device: &bevy::render::renderer::RenderDevice,
    label: &'static str,
    format: TextureFormat,
    usage: TextureUsages,
    size: UVec2,
) -> Texture {
    device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

fn write_rgba32f_pixel(
    queue: &bevy::render::renderer::RenderQueue,
    texture: &Texture,
    value: [f32; 4],
) {
    let bytes: Vec<_> = value.into_iter().flat_map(f32::to_le_bytes).collect();
    queue.write_texture(
        TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        &bytes,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(16),
            rows_per_image: Some(1),
        },
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
}

fn write_rgba32f_row(
    queue: &bevy::render::renderer::RenderQueue,
    texture: &Texture,
    pixels: &[[f32; 4]],
) {
    let mut bytes = Vec::with_capacity(pixels.len() * 16);
    for pixel in pixels {
        for value in pixel {
            bytes.extend(value.to_le_bytes());
        }
    }
    queue.write_texture(
        TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        &bytes,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some((pixels.len() * 16) as u32),
            rows_per_image: Some(1),
        },
        Extent3d {
            width: pixels.len() as u32,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
}

fn write_rgba32u_row(
    queue: &bevy::render::renderer::RenderQueue,
    texture: &Texture,
    pixels: &[[u32; 4]],
) {
    let mut bytes = Vec::with_capacity(pixels.len() * 16);
    for pixel in pixels {
        for value in pixel {
            bytes.extend(value.to_le_bytes());
        }
    }
    queue.write_texture(
        TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        &bytes,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some((pixels.len() * 16) as u32),
            rows_per_image: Some(1),
        },
        Extent3d {
            width: pixels.len() as u32,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
}

struct TemporalTestTextures {
    size: UVec2,
    depth: Texture,
    current: Texture,
    previous: Texture,
    previous_metadata: Texture,
    previous_receiver: Texture,
    motion: Texture,
    raster_receiver: Texture,
    gbuffer: Texture,
    irradiance_output: Texture,
    metadata_output: Texture,
    receiver_output: Texture,
}

impl TemporalTestTextures {
    fn new(
        device: &bevy::render::renderer::RenderDevice,
        queue: &bevy::render::renderer::RenderQueue,
    ) -> Self {
        Self::new_sized(device, queue, UVec2::ONE)
    }

    fn new_sized(
        device: &bevy::render::renderer::RenderDevice,
        queue: &bevy::render::renderer::RenderQueue,
        size: UVec2,
    ) -> Self {
        assert_eq!(
            size.y, 1,
            "temporal test helper currently uses a single row"
        );
        let sampled_float = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
        let current = temporal_test_texture_sized(
            device,
            "temporal_test_current",
            TextureFormat::Rgba32Float,
            sampled_float,
            size,
        );
        let previous = temporal_test_texture_sized(
            device,
            "temporal_test_previous",
            TextureFormat::Rgba32Float,
            sampled_float,
            size,
        );
        let previous_metadata = temporal_test_texture_sized(
            device,
            "temporal_test_previous_metadata",
            TextureFormat::Rgba32Float,
            sampled_float,
            size,
        );
        let previous_receiver = temporal_test_texture_sized(
            device,
            "temporal_test_previous_receiver",
            TextureFormat::Rgba32Float,
            sampled_float,
            size,
        );
        let motion = temporal_test_texture_sized(
            device,
            "temporal_test_motion",
            TextureFormat::Rgba32Float,
            sampled_float,
            size,
        );
        let raster_receiver = temporal_test_texture_sized(
            device,
            "temporal_test_raster_receiver",
            TextureFormat::Rgba32Uint,
            sampled_float,
            size,
        );
        let gbuffer = temporal_test_texture_sized(
            device,
            "temporal_test_gbuffer",
            TextureFormat::Rgba32Uint,
            sampled_float,
            size,
        );
        let output = TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC;
        let irradiance_output = temporal_test_texture_sized(
            device,
            "temporal_test_irradiance_output",
            TextureFormat::Rgba32Float,
            output,
            size,
        );
        let metadata_output = temporal_test_texture_sized(
            device,
            "temporal_test_metadata_output",
            TextureFormat::Rgba32Float,
            output,
            size,
        );
        let receiver_output = temporal_test_texture_sized(
            device,
            "temporal_test_receiver_output",
            TextureFormat::Rgba32Float,
            output,
            size,
        );
        let depth = temporal_test_texture_sized(
            device,
            "temporal_test_depth",
            TextureFormat::Depth32Float,
            TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
            size,
        );
        let depth_view = depth.create_view(&TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("temporal_test_depth_clear"),
        });
        {
            let _pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("temporal_test_depth_clear"),
                color_attachments: &[],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(0.5),
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..default()
            });
        }
        queue.submit([encoder.finish()]);

        let previous_row = vec![[1.0, 1.0, 1.0, 0.5]; size.x as usize];
        let metadata_row = vec![[0.5, 0.5, 16.0, 0.0]; size.x as usize];
        let receiver_row = vec![[1.0, 2.0, 8_390_656.0, 65_537.0,]; size.x as usize];
        let motion_row = vec![[0.0; 4]; size.x as usize];
        let raster_row = vec![[1.0f32.to_bits(), 2.0f32.to_bits(), 0, 0]; size.x as usize];
        let gbuffer_row = vec![[0, 0, 0, 0x0080_0800]; size.x as usize];
        write_rgba32f_row(queue, &previous, &previous_row);
        write_rgba32f_row(queue, &previous_metadata, &metadata_row);
        write_rgba32f_row(queue, &previous_receiver, &receiver_row);
        write_rgba32f_row(queue, &motion, &motion_row);
        write_rgba32u_row(queue, &raster_receiver, &raster_row);
        write_rgba32u_row(queue, &gbuffer, &gbuffer_row);

        Self {
            size,
            depth,
            current,
            previous,
            previous_metadata,
            previous_receiver,
            motion,
            raster_receiver,
            gbuffer,
            irradiance_output,
            metadata_output,
            receiver_output,
        }
    }
}

fn temporal_test_result(
    device: &bevy::render::renderer::RenderDevice,
    queue: &bevy::render::renderer::RenderQueue,
    pipeline: &ComputePipeline,
    textures: &TemporalTestTextures,
    parameters: TransportParameters,
) -> [[f32; 4]; 3] {
    temporal_test_result_full(device, queue, pipeline, textures, parameters, false)[0]
}

fn temporal_test_result_full(
    device: &bevy::render::renderer::RenderDevice,
    queue: &bevy::render::renderer::RenderQueue,
    pipeline: &ComputePipeline,
    textures: &TemporalTestTextures,
    parameters: TransportParameters,
    advance_history: bool,
) -> Vec<[[f32; 4]; 3]> {
    let head_words = parameters.size.w as usize * 2 + 36;
    let heads = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("temporal_test_heads"),
        contents: &vec![0u8; head_words * 4],
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    let mut parameters = UniformBuffer::from(parameters);
    parameters.write_buffer(device, queue);
    let view = |texture: &Texture| texture.create_view(&TextureViewDescriptor::default());
    let depth = view(&textures.depth);
    let current = view(&textures.current);
    let previous = view(&textures.previous);
    let previous_metadata = view(&textures.previous_metadata);
    let previous_receiver = view(&textures.previous_receiver);
    let motion = view(&textures.motion);
    let raster_receiver = view(&textures.raster_receiver);
    let gbuffer = view(&textures.gbuffer);
    let irradiance_output = view(&textures.irradiance_output);
    let metadata_output = view(&textures.metadata_output);
    let receiver_output = view(&textures.receiver_output);
    let group = device.create_bind_group(
        "temporal_test",
        &pipeline.get_bind_group_layout(0).into(),
        &[
            BindGroupEntry {
                binding: 0,
                resource: parameters.binding().unwrap(),
            },
            BindGroupEntry {
                binding: 4,
                resource: BindingResource::TextureView(&depth),
            },
            BindGroupEntry {
                binding: 6,
                resource: heads.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 10,
                resource: BindingResource::TextureView(&irradiance_output),
            },
            BindGroupEntry {
                binding: 11,
                resource: BindingResource::TextureView(&previous),
            },
            BindGroupEntry {
                binding: 13,
                resource: BindingResource::TextureView(&current),
            },
            BindGroupEntry {
                binding: 14,
                resource: BindingResource::TextureView(&metadata_output),
            },
            BindGroupEntry {
                binding: 15,
                resource: BindingResource::TextureView(&previous_metadata),
            },
            BindGroupEntry {
                binding: 20,
                resource: BindingResource::TextureView(&motion),
            },
            BindGroupEntry {
                binding: 23,
                resource: BindingResource::TextureView(&receiver_output),
            },
            BindGroupEntry {
                binding: 25,
                resource: BindingResource::TextureView(&previous_receiver),
            },
            BindGroupEntry {
                binding: 26,
                resource: BindingResource::TextureView(&raster_receiver),
            },
            BindGroupEntry {
                binding: 27,
                resource: BindingResource::TextureView(&gbuffer),
            },
        ],
    );
    let row_pitch = (textures.size.x * 16).div_ceil(256) * 256;
    let image_pitch = row_pitch * textures.size.y;
    let readback = device.create_buffer(&BufferDescriptor {
        label: Some("temporal_test_readback"),
        size: u64::from(3 * image_pitch),
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("temporal_test_dispatch"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("temporal_test_dispatch"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(textures.size.x.div_ceil(8), textures.size.y.div_ceil(8), 1);
    }
    let extent = Extent3d {
        width: textures.size.x,
        height: textures.size.y,
        depth_or_array_layers: 1,
    };
    if advance_history {
        for (source, destination) in [
            (&textures.irradiance_output, &textures.previous),
            (&textures.metadata_output, &textures.previous_metadata),
            (&textures.receiver_output, &textures.previous_receiver),
        ] {
            encoder.copy_texture_to_texture(
                TexelCopyTextureInfo {
                    texture: source,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                TexelCopyTextureInfo {
                    texture: destination,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                extent,
            );
        }
    }
    for (index, texture) in [
        &textures.irradiance_output,
        &textures.metadata_output,
        &textures.receiver_output,
    ]
    .into_iter()
    .enumerate()
    {
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &readback,
                layout: TexelCopyBufferLayout {
                    offset: index as u64 * u64::from(image_pitch),
                    bytes_per_row: Some(row_pitch),
                    rows_per_image: Some(textures.size.y),
                },
            },
            extent,
        );
    }
    let submission = queue.submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::channel();
    readback.slice(..).map_async(MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    device
        .poll(PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(std::time::Duration::from_secs(30)),
        })
        .unwrap();
    receiver
        .recv_timeout(std::time::Duration::from_secs(1))
        .unwrap()
        .unwrap();
    let bytes = readback
        .slice(..)
        .get_mapped_range()
        .expect("temporal output readback should map")
        .to_vec();
    readback.unmap();
    let mut pixels = Vec::with_capacity((textures.size.x * textures.size.y) as usize);
    for y in 0..textures.size.y {
        for x in 0..textures.size.x {
            pixels.push(std::array::from_fn(|texture_index| {
                let start = texture_index * image_pitch as usize
                    + y as usize * row_pitch as usize
                    + x as usize * 16;
                std::array::from_fn(|component| {
                    let offset = start + component * 4;
                    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
                })
            }));
        }
    }
    pixels
}

fn assert_denoise_rejects_other_receiver(
    device: &bevy::render::renderer::RenderDevice,
    queue: &bevy::render::renderer::RenderQueue,
    pipeline: &ComputePipeline,
) {
    for curved in [false, true] {
        let size = UVec2::new(11, 1);
        let sampled = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
        let input = temporal_test_texture_sized(
            device,
            "denoise_test_input",
            TextureFormat::Rgba32Float,
            sampled,
            size,
        );
        let moments = temporal_test_texture_sized(
            device,
            "denoise_test_moments",
            TextureFormat::Rgba32Float,
            sampled,
            size,
        );
        let current_receiver = temporal_test_texture_sized(
            device,
            "denoise_test_current_receiver",
            TextureFormat::Rgba32Float,
            sampled,
            size,
        );
        let output = temporal_test_texture_sized(
            device,
            "denoise_test_output",
            TextureFormat::Rgba32Float,
            TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
            size,
        );
        let output_receiver = temporal_test_texture_sized(
            device,
            "denoise_test_output_receiver",
            TextureFormat::Rgba32Float,
            TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
            size,
        );
        let depth = temporal_test_texture_sized(
            device,
            "denoise_test_depth",
            TextureFormat::Depth32Float,
            TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
            size,
        );
        let depth_view = depth.create_view(&TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("denoise_test_depth_clear"),
        });
        {
            let _pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("denoise_test_depth_clear"),
                color_attachments: &[],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(0.5),
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..default()
            });
        }
        queue.submit([encoder.finish()]);

        let receiver = |identity: [u32; 2], supported: bool| {
            [
                f32::from_bits(identity[0]),
                f32::from_bits(identity[1]),
                8_390_656.0,
                32_769.0 + if supported { 65_536.0 } else { 0.0 },
            ]
        };
        let mut input_pixels = [[0.0; 4]; 11];
        let mut moment_pixels = [[0.0; 4]; 11];
        let mut receiver_pixels = [receiver([90, 91], false); 11];

        input_pixels[0] = [1.0, 0.0, 0.0, 0.5];
        input_pixels[1] = [0.0, 1.0, 0.0, 0.0];
        input_pixels[2] = [0.0, 0.0, 1.0, 0.5];
        moment_pixels[0] = [0.0, 0.0, 1.0, 1.0];
        moment_pixels[2] = [0.0, 0.0, 1.0, 1.0];
        receiver_pixels[0] = receiver([1, 2], true);
        receiver_pixels[1] = receiver([1, 2], false);
        receiver_pixels[2] = receiver([3, 4], true);
        if curved {
            let packed_normal = |angle: f32| {
                let x = angle.sin() / (angle.sin().abs() + angle.cos());
                let oct_x = ((x * 0.5 + 0.5) * 4095.0).round() as u32;
                (oct_x | (2048 << 12)) as f32
            };
            receiver_pixels[0][2] = packed_normal(-0.1);
            receiver_pixels[1][2] = packed_normal(0.1);
        }

        write_rgba32f_row(queue, &input, &input_pixels);
        write_rgba32f_row(queue, &moments, &moment_pixels);
        write_rgba32f_row(queue, &current_receiver, &receiver_pixels);

        let mut params = temporal_test_parameters(64);
        params.tuning.x = 0.15;
        params.viewport = UVec4::new(0, 0, size.x, size.y);
        params.size = UVec4::new(size.x, size.y, size.x * size.y, 1);
        params.frame = UVec4::ZERO;
        let mut params = UniformBuffer::from(params);
        params.write_buffer(device, queue);
        let input_view = input.create_view(&TextureViewDescriptor::default());
        let moments_view = moments.create_view(&TextureViewDescriptor::default());
        let current_receiver_view = current_receiver.create_view(&TextureViewDescriptor::default());
        let output_view = output.create_view(&TextureViewDescriptor::default());
        let output_receiver_view = output_receiver.create_view(&TextureViewDescriptor::default());
        let group = device.create_bind_group(
            "denoise_test",
            &pipeline.get_bind_group_layout(0).into(),
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: params.binding().unwrap(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(&depth_view),
                },
                BindGroupEntry {
                    binding: 10,
                    resource: BindingResource::TextureView(&output_view),
                },
                BindGroupEntry {
                    binding: 11,
                    resource: BindingResource::TextureView(&input_view),
                },
                BindGroupEntry {
                    binding: 13,
                    resource: BindingResource::TextureView(&moments_view),
                },
                BindGroupEntry {
                    binding: 23,
                    resource: BindingResource::TextureView(&output_receiver_view),
                },
                BindGroupEntry {
                    binding: 25,
                    resource: BindingResource::TextureView(&current_receiver_view),
                },
            ],
        );
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("denoise_test_readback"),
            size: 512,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("denoise_test_dispatch"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("denoise_test_dispatch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(size.x.div_ceil(8), 1, 1);
        }
        for (index, source) in [&output, &output_receiver].into_iter().enumerate() {
            encoder.copy_texture_to_buffer(
                TexelCopyTextureInfo {
                    texture: source,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                TexelCopyBufferInfo {
                    buffer: &readback,
                    layout: TexelCopyBufferLayout {
                        offset: index as u64 * 256,
                        bytes_per_row: Some(256),
                        rows_per_image: Some(1),
                    },
                },
                Extent3d {
                    width: size.x,
                    height: size.y,
                    depth_or_array_layers: 1,
                },
            );
        }
        let submission = queue.submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::channel();
        readback.slice(..).map_async(MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        device
            .poll(PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(std::time::Duration::from_secs(30)),
            })
            .unwrap();
        receiver
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap()
            .unwrap();
        let bytes = readback
            .slice(..)
            .get_mapped_range()
            .expect("denoise output readback should map")
            .to_vec();
        readback.unmap();
        let hole_center = output_vec4(&bytes, 1);
        assert!(
            hole_center[0] > 0.9,
            "same-receiver red neighbor did not fill unsupported center: {hole_center:?}"
        );
        assert!(
            hole_center[1] < 1e-5 && hole_center[2] < 1e-5,
            "unsupported center was treated as observed black or another receiver bled in: {hole_center:?}"
        );
        assert!((hole_center[3] - 0.5).abs() < 1e-5);
        let filled_receiver = output_vec4(&bytes[256..], 1);
        assert_eq!(
            (filled_receiver[3] as u32 >> 16) & 1,
            1,
            "denoised support must reach composition: {filled_receiver:?}"
        );
        assert_eq!(
            (output_vec4(&bytes[256..], 10)[3] as u32 >> 16) & 1,
            0,
            "an unreconstructed receiver must remain unsupported"
        );
    }
}
fn temporal_test_parameters(history_frames: u32) -> TransportParameters {
    TransportParameters {
        world_from_clip: Mat4::IDENTITY,
        clip_from_world: Mat4::IDENTITY,
        previous_clip_from_world: Mat4::IDENTITY,
        previous_world_from_clip: Mat4::IDENTITY,
        camera_position: Vec4::ZERO,
        viewport: UVec4::new(0, 0, 1, 1),
        size: UVec4::new(1, 1, 1, 1),
        tuning: Vec4::new(1.2, 1.92, 0.005, 1.0),
        environment: Vec4::new(0.0, 0.0, 0.0, 1.0),
        history: Vec4::new(0.9, 0.0, 0.0, history_frames as f32),
        interval_history: Vec4::ZERO,
        frame: UVec4::ZERO,
        light_counts: UVec4::ZERO,
        stability: Vec4::new(1.0, 0.0, 0.0, 0.0),
        layers: UVec4::ZERO,
        environment_info: UVec4::ZERO,
        environment_rotation: Vec4::new(0.0, 0.0, 0.0, 1.0),
        environment_scale: Vec4::ZERO,
        geometry_pages: UVec4::ZERO,
        scene_pages: UVec4::ZERO,
        scene_overlay: UVec4::ZERO,
        dirty_min: Vec4::ZERO,
        dirty_max: Vec4::ZERO,
        transport: Vec4::ZERO,
        sampling: UVec4::ZERO,
        owner_sequence: UVec4::ZERO,
        cache_time: UVec4::ZERO,
        world_receivers: UVec4::ZERO,
        world_receiver_state: UVec4::ZERO,
        light_dirty_min: Vec4::ZERO,
        light_dirty_max: Vec4::ZERO,
    }
}

#[test]
#[ignore = "requires slangc and a desktop GPU; dispatches production Split temporal and denoise shaders"]
fn gpu_temporal_resolve_preserves_unobserved_history_and_valid_black() {
    let options = WgpuSettings::default();
    let resources = block_on(initialize_renderer(
        options.backends.unwrap_or(Backends::PRIMARY),
        None,
        &options,
    ));
    let device = &resources.0;
    let queue = &resources.1;
    let shader = SlangCompiler::default()
        .compile_bundle(
            "cascades.slang",
            &[
                ("common.slang", include_str!("../shaders/common.slang")),
                ("bsdf.slang", include_str!("../shaders/bsdf.slang")),
                (
                    "receiver_metadata.slang",
                    include_str!("../shaders/receiver_metadata.slang"),
                ),
                (
                    "traversal.slang",
                    include_str!("../shaders/traversal.slang"),
                ),
                ("material.slang", include_str!("../shaders/material.slang")),
                (
                    "environment.slang",
                    include_str!("../shaders/environment.slang"),
                ),
                (
                    "linked_merge_sort.slang",
                    include_str!("../shaders/linked_merge_sort.slang"),
                ),
                ("split.slang", include_str!("../shaders/split.slang")),
                (
                    "world_resolve.slang",
                    include_str!("../shaders/world_resolve.slang"),
                ),
                ("cascades.slang", include_str!("../shaders/cascades.slang")),
            ],
            &SlangSettings {
                target: SlangTarget::Wgsl,
                entry_points: vec!["temporal".into(), "denoise".into()],
                optimization: Some(2),
                ..default()
            },
        )
        .unwrap();
    let bevy::shader::Source::Wgsl(source) = shader.source else {
        panic!("expected WGSL from the temporal resolve test shader");
    };
    let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
        label: Some("temporal_resolve_regression"),
        source: ShaderSource::Wgsl(source),
    });
    let pipeline = device.create_compute_pipeline(&RawComputePipelineDescriptor {
        label: Some("temporal_resolve_regression"),
        layout: None,
        module: &module,
        entry_point: Some("temporal"),
        compilation_options: default(),
        cache: None,
    });
    let denoise_pipeline = device.create_compute_pipeline(&RawComputePipelineDescriptor {
        label: Some("denoise_receiver_history_regression"),
        layout: None,
        module: &module,
        entry_point: Some("denoise"),
        compilation_options: default(),
        cache: None,
    });
    let support_mask = |receiver: [f32; 4]| ((receiver[3] as u32) >> 16) & 1;

    // Negative raw alpha means this eligible receiver has no current sample.
    // Its compatible supported history remains available without gaining age.
    let unobserved = TemporalTestTextures::new(device, queue);
    write_rgba32f_pixel(queue, &unobserved.current, [0.0, 0.0, 0.0, -1.0]);
    let mut parameters = temporal_test_parameters(30);
    parameters.interval_history.y = 1.0;
    let retained = temporal_test_result(device, queue, &pipeline, &unobserved, parameters);
    assert!(
        retained[0][..3]
            .iter()
            .all(|value| (*value - 1.0).abs() < 1e-4)
    );
    assert!((retained[0][3] - 0.5).abs() < 1e-5);
    assert!((retained[1][2] - 16.0).abs() < 1e-5);
    assert_eq!(support_mask(retained[2]), 1);

    // A black observation is supported data. A lighting revision clears old
    // radiance and retains a valid black result with the one-bit support marker.
    let black = TemporalTestTextures::new(device, queue);
    write_rgba32f_pixel(queue, &black.current, [0.0, 0.0, 0.0, 1.25]);
    let mut parameters = temporal_test_parameters(30);
    parameters.interval_history.y = 1.0;
    parameters.stability.y = 1.0;
    parameters.transport.w = 1.0;
    let observed_black = temporal_test_result(device, queue, &pipeline, &black, parameters);
    assert!(
        observed_black[0][..3]
            .iter()
            .all(|value| value.abs() < 1e-6)
    );
    assert!((observed_black[1][2] - 1.0).abs() < 1e-5);
    assert_eq!(support_mask(observed_black[2]), 1);

    // Raw alpha carries angular coverage in quarter steps. With screen history
    // disabled, a quarter-covered sample contributes one quarter of a frame.
    let partial = TemporalTestTextures::new(device, queue);
    write_rgba32f_pixel(queue, &partial.current, [0.25, 0.25, 0.25, 1.0625]);
    let mut parameters = temporal_test_parameters(30);
    parameters.interval_history.y = 1.0;
    parameters.history.x = 0.0;
    let partial_result = temporal_test_result(device, queue, &pipeline, &partial, parameters);
    assert!(
        partial_result[0][..3]
            .iter()
            .all(|value| (*value - 0.25).abs() < 1e-5)
    );
    assert!((partial_result[1][2] - 0.25).abs() < 1e-5);
    assert_eq!(support_mask(partial_result[2]), 1);

    // A world-space dirty bound disjoint from the receiver keeps its history
    // through a geometry-only revision. An intersecting bound rejects it.
    let mut parameters = temporal_test_parameters(30);
    parameters.interval_history.y = 1.0;
    parameters.stability.y = 1.0;
    parameters.dirty_min = Vec4::new(10.0, 10.0, 10.0, 1.0);
    parameters.dirty_max = Vec4::new(11.0, 11.0, 11.0, 1.0);
    let disjoint = temporal_test_result(device, queue, &pipeline, &unobserved, parameters);
    assert!(
        disjoint[0][..3]
            .iter()
            .all(|value| (*value - 1.0).abs() < 1e-4)
    );
    assert!((disjoint[1][2] - 16.0).abs() < 1e-5);
    assert_eq!(support_mask(disjoint[2]), 1);

    parameters.dirty_min = Vec4::new(0.0, -2.0, 0.0, 1.0);
    parameters.dirty_max = Vec4::new(2.0, 0.0, 1.0, 1.0);
    let intersecting = temporal_test_result(device, queue, &pipeline, &unobserved, parameters);
    assert!(intersecting[0][..3].iter().all(|value| value.abs() < 1e-6));
    assert_eq!(intersecting[0][3], 0.0);
    assert_eq!(support_mask(intersecting[2]), 0);

    // Screen history has no transport dependencies: a lighting revision must
    // discard old radiance even when the refreshed world field is incomplete.
    for (alpha, expected_length, expected_support) in [(-1.0, 0.0, 0), (1.0625, 0.25, 1)] {
        let changed = TemporalTestTextures::new(device, queue);
        write_rgba32f_pixel(queue, &changed.current, [0.0, 0.0, 0.0, alpha]);
        let mut parameters = temporal_test_parameters(30);
        parameters.interval_history.y = 1.0;
        parameters.stability.y = 1.0;
        parameters.transport.w = 1.0;
        let result =
            temporal_test_result_full(device, queue, &pipeline, &changed, parameters, true)[0];
        assert!(
            result[0][..3].iter().all(|value| value.abs() < 1e-6),
            "old lighting survived incomplete refresh: {result:?}"
        );
        assert!((result[1][2] - expected_length).abs() < 1e-5);
        assert_eq!(support_mask(result[2]), expected_support);
        parameters.stability.y = 0.0;
        parameters.transport.w = 0.0;
        write_rgba32f_pixel(queue, &changed.current, [0.0, 0.0, 0.0, -1.0]);
        let next = temporal_test_result(device, queue, &pipeline, &changed, parameters);
        assert!(
            next[0][..3].iter().all(|value| value.abs() < 1e-6),
            "invalidated lighting returned on the next frame: {next:?}"
        );
        assert_eq!(support_mask(next[2]), expected_support);
    }

    // The final partial resolve tile is clamped to the viewport edge. Static
    // reprojection must read that exact anchor without mixing its neighbor.
    for scale in 2..=8 {
        let size = UVec2::new(scale + 1, 1);
        let edge = TemporalTestTextures::new_sized(device, queue, size);
        let mut previous = vec![[1.0, 1.0, 1.0, 0.5]; size.x as usize];
        previous[1] = [5.0, 5.0, 5.0, 0.5];
        write_rgba32f_row(queue, &edge.previous, &previous);
        write_rgba32f_row(
            queue,
            &edge.current,
            &vec![[0.0, 0.0, 0.0, -1.0]; size.x as usize],
        );
        let mut parameters = temporal_test_parameters(30);
        parameters.interval_history.y = 1.0;
        parameters.viewport.z = size.x;
        parameters.sampling.w = scale;
        let result = temporal_test_result_full(device, queue, &pipeline, &edge, parameters, false);
        assert!(
            result[1][0][..3]
                .iter()
                .all(|value| (*value - 5.0).abs() < 1e-5),
            "scale {scale} changed a static edge history: {:?}",
            result[1]
        );
        assert_eq!(support_mask(result[1][2]), 1);
    }

    // A supported same-receiver neighbor may fill an unsupported center, while
    // another receiver cannot lend its blue radiance.
    assert_denoise_rejects_other_receiver(device, queue, &denoise_pipeline);
}
#[test]
#[ignore = "requires slangc and a desktop GPU; dispatches production BSDF and environment helpers"]
fn gpu_transport_helpers_preserve_reciprocity_hue_and_environment_pdf() {
    let options = WgpuSettings::default();
    let resources = block_on(initialize_renderer(
        options.backends.unwrap_or(Backends::PRIMARY),
        None,
        &options,
    ));
    let device = &resources.0;
    let queue = &resources.1;

    let image = hdr_test_image();
    let table = crate::environment::validate_image(&image, 1).unwrap();
    let mut packed_lights = Vec::new();
    let alias_offset = table.append_alias_lights(&mut packed_lights);
    assert_eq!(alias_offset, 0);
    assert_eq!(table.aliases.len(), 8);

    let shader = SlangCompiler::default()
        .compile_bundle(
            "transport_check.slang",
            &[
                ("common.slang", include_str!("../shaders/common.slang")),
                ("bsdf.slang", include_str!("../shaders/bsdf.slang")),
                (
                    "environment.slang",
                    include_str!("../shaders/environment.slang"),
                ),
                ("transport_check.slang", TRANSPORT_SHADER),
            ],
            &SlangSettings {
                target: SlangTarget::Wgsl,
                entry_points: ["checkTransport", "checkEnvironment"]
                    .into_iter()
                    .map(str::to_owned)
                    .collect(),
                optimization: Some(2),
                ..default()
            },
        )
        .unwrap();
    let bevy::shader::Source::Wgsl(source) = shader.source else {
        panic!("expected WGSL from transport helper test shader");
    };
    let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
        label: Some("transport_helper_check"),
        source: ShaderSource::Wgsl(source),
    });
    let pipeline = device.create_compute_pipeline(&RawComputePipelineDescriptor {
        label: Some("transport_helper_check"),
        layout: None,
        module: &module,
        entry_point: Some("checkTransport"),
        compilation_options: default(),
        cache: None,
    });
    let environment_pipeline = device.create_compute_pipeline(&RawComputePipelineDescriptor {
        label: Some("environment_nee_regression"),
        layout: None,
        module: &module,
        entry_point: Some("checkEnvironment"),
        compilation_options: default(),
        cache: None,
    });

    let mut base_parameters = temporal_test_parameters(0);
    base_parameters.size = UVec4::new(2, 4, 8, 0);
    base_parameters.environment = Vec4::new(0.0, 0.0, 0.0, 2.0);
    base_parameters.environment_info = UVec4::new(alias_offset, table.width, table.height, 1);
    base_parameters.environment_rotation = Vec4::new(0.0, 0.0, 0.0, 1.0);
    base_parameters.environment_scale = Vec4::ONE;
    let mut parameters = UniformBuffer::from(base_parameters);
    parameters.write_buffer(device, queue);
    let mut lights = StorageBuffer::from(packed_lights);
    lights.write_buffer(device, queue);

    let texture = device.create_texture(&TextureDescriptor {
        label: Some("transport_test_environment"),
        size: Extent3d {
            width: 2,
            height: 4,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let texture_bytes = image.data.as_ref().expect("HDR bytes are present");
    queue.write_texture(
        TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        texture_bytes,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(2 * 16),
            rows_per_image: Some(4),
        },
        Extent3d {
            width: 2,
            height: 4,
            depth_or_array_layers: 1,
        },
    );
    let texture_view = texture.create_view(&TextureViewDescriptor::default());
    let output_bytes = 26 * 16;
    let output = device.create_buffer(&BufferDescriptor {
        label: Some("transport_helper_output"),
        size: output_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let group = device.create_bind_group(
        "transport_helper_check",
        &pipeline.get_bind_group_layout(0).into(),
        &[
            BindGroupEntry {
                binding: 0,
                resource: parameters.binding().unwrap(),
            },
            BindGroupEntry {
                binding: 3,
                resource: lights.binding().unwrap(),
            },
            BindGroupEntry {
                binding: 4,
                resource: output.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 22,
                resource: BindingResource::TextureView(&texture_view),
            },
        ],
    );
    let bytes = crate::scene::traversal_tests::dispatch_readback(
        device,
        queue,
        &pipeline,
        &group,
        &output,
        output_bytes,
        1,
    );

    for index in 0..3 {
        let result = output_vec4(&bytes, index);
        assert!(result[0].is_finite() && result[1].is_finite());
        assert!(result[0] > 0.0 && result[1] > 0.0);
        assert!((result[0] - result[1]).abs() <= 1e-5);
        assert_eq!(result[2], index as f32 * 0.5);
    }
    assert_eq!(output_vec4(&bytes, 3), [100.0, 50.0, 25.0, 1.0]);
    assert_eq!(output_vec4(&bytes, 4), [0.0, 1.0, 1.0, 1.0]);
    assert_eq!(output_vec4(&bytes, 5), [0.0, 0.0, 0.0, 1.0]);
    for index in 0..8 {
        let direction_pdf = output_vec4(&bytes, 6 + index * 2);
        let radiance_lookup_pdf = output_vec4(&bytes, 7 + index * 2);
        let direction = Vec3::new(direction_pdf[0], direction_pdf[1], direction_pdf[2]);
        assert!(direction.is_finite() && (direction.length() - 1.0).abs() < 1e-4);
        assert!(direction_pdf[3].is_finite() && direction_pdf[3] > 0.0);
        assert!(
            radiance_lookup_pdf[..3]
                .iter()
                .all(|component| component.is_finite() && *component >= 0.0)
        );
        assert!((direction_pdf[3] - radiance_lookup_pdf[3]).abs() < 1e-4);
        let expected_pdf = expected_environment_pdf(&table, direction) as f32;
        assert!((direction_pdf[3] - expected_pdf).abs() <= expected_pdf * 2e-4);
    }
    assert_eq!(
        output_vec4(&bytes, 22),
        [0.0, -1.0, 1.0625, 1.25],
        "raw GI alpha distinguishes ineligible, unobserved, and supported coverage"
    );
    for result in [output_vec4(&bytes, 23), output_vec4(&bytes, 24)] {
        assert!(result.iter().all(|value| value.is_finite()));
        assert!((result[3] - 1.0).abs() < 1e-5);
    }
    let poles = output_vec4(&bytes, 25);
    assert!(poles[..3].iter().all(|value| value.is_finite()));
    assert!(poles[0] >= 0.0 && poles[0] < poles[2]);
    assert!(poles[1] >= 0.0 && poles[1] < poles[2]);
    assert_eq!(poles[2], 2048.0);
    let polar_radiance = output_vec4(&bytes, 7);
    assert!(polar_radiance[0] > 20.0);
    assert!((polar_radiance[1] / polar_radiance[0] - 0.6).abs() < 1e-3);

    let sky = [4.0f32, 2.0, 1.0];
    let uniform_pixels = [[sky[0], sky[1], sky[2], 1.0]; 8];
    let uniform_image = Image::new(
        Extent3d {
            width: 2,
            height: 4,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        uniform_pixels
            .iter()
            .flat_map(|pixel| pixel.iter().flat_map(|channel| channel.to_le_bytes()))
            .collect(),
        TextureFormat::Rgba32Float,
        RenderAssetUsages::MAIN_WORLD,
    );
    let uniform_table = crate::environment::validate_image(&uniform_image, 2).unwrap();
    let mut uniform_packed_lights = Vec::new();
    let uniform_alias_offset = uniform_table.append_alias_lights(&mut uniform_packed_lights);
    let mut uniform_lights = StorageBuffer::from(uniform_packed_lights);
    uniform_lights.write_buffer(device, queue);
    queue.write_texture(
        TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        uniform_image.data.as_ref().unwrap(),
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(2 * 16),
            rows_per_image: Some(4),
        },
        Extent3d {
            width: 2,
            height: 4,
            depth_or_array_layers: 1,
        },
    );

    let run_environment_case = |case_parameters: TransportParameters| -> [[f32; 4]; 2] {
        let mut case_uniform = UniformBuffer::from(case_parameters);
        case_uniform.write_buffer(device, queue);
        let case_output_bytes = 2 * 16;
        let case_output = device.create_buffer(&BufferDescriptor {
            label: Some("environment_nee_regression_output"),
            size: case_output_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let case_group = device.create_bind_group(
            "environment_nee_regression",
            &environment_pipeline.get_bind_group_layout(0).into(),
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: case_uniform.binding().unwrap(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: uniform_lights.binding().unwrap(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: case_output.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 22,
                    resource: BindingResource::TextureView(&texture_view),
                },
            ],
        );
        let case_bytes = crate::scene::traversal_tests::dispatch_readback(
            device,
            queue,
            &environment_pipeline,
            &case_group,
            &case_output,
            case_output_bytes,
            1,
        );
        std::array::from_fn(|index| output_vec4(&case_bytes, index))
    };
    let make_environment_parameters = |is_hdr: bool, visibility_mode: u32, sample_count: u32| {
        let mut case_parameters = base_parameters;
        case_parameters.environment = Vec4::new(sky[0], sky[1], sky[2], 0.0);
        case_parameters.environment_info = UVec4::new(
            uniform_alias_offset,
            uniform_table.width,
            uniform_table.height,
            u32::from(is_hdr),
        );
        case_parameters.environment_rotation = Vec4::new(0.0, 0.0, 0.0, 1.0);
        case_parameters.environment_scale = Vec4::ONE;
        case_parameters.tuning = Vec4::new(1.0, 0.4f32.sqrt(), 0.001, 0.0);
        case_parameters.sampling = UVec4::new(sample_count, 0, 0, 0);
        case_parameters.owner_sequence = UVec4::new(0, 0, visibility_mode, 0);
        case_parameters
    };

    let constant_sky = run_environment_case(make_environment_parameters(false, 0, 4096));
    let uniform_hdr = run_environment_case(make_environment_parameters(true, 0, 4096));
    for result in [
        constant_sky[0],
        constant_sky[1],
        uniform_hdr[0],
        uniform_hdr[1],
    ] {
        assert_eq!(
            result[3], 1.0,
            "visible environment query lost support: {result:?}"
        );
        assert!(
            result[..3]
                .iter()
                .zip(sky)
                .all(|(actual, expected)| (actual - expected).abs() <= expected * 0.025),
            "unoccluded constant sky should integrate to L: {result:?}"
        );
    }
    for index in 0..3 {
        assert!(
            (constant_sky[0][index] - uniform_hdr[0][index]).abs() <= sky[index] * 0.03,
            "constant and uniform-HDR estimates diverged: {:?} vs {:?}",
            constant_sky[0],
            uniform_hdr[0]
        );
    }
    for mode in [1u32, 2u32] {
        let result = run_environment_case(make_environment_parameters(false, mode, 32));
        if mode == 1 {
            assert_eq!(result, [[0.0, 0.0, 0.0, 1.0]; 2]);
        } else {
            assert_eq!(
                result, [[0.0; 4]; 2],
                "denied visibility remains unobserved"
            );
        }
    }
}
#[test]
#[ignore = "requires slangc and a desktop GPU; dispatches production Split prefix helpers"]
fn gpu_split_r2_prefix_uses_selected_rays_and_world_grid_keys() {
    fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        let start_at = source.find(start).expect("production shader section start");
        let tail = &source[start_at..];
        let end_at = tail
            .find(end)
            .unwrap_or_else(|| panic!("production shader section end `{end}`"));
        &tail[..end_at]
    }

    const CANDIDATES: u32 = 64;
    const CACHE_BUCKETS: u32 = 16;
    const SELECTED_BUDGET: u32 = 48;
    const ENTRY_WORDS: usize = 208 / 4;
    let directory_capacity = (CANDIDATES * 4).next_power_of_two();
    let probe_bucket = |cell: i32, lod: u32, level: u32| {
        let mut hash = (cell as u32).wrapping_mul(0x8da6_b343)
            ^ ((lod << 18) | (level << 12)).wrapping_mul(0x1656_67b1);
        hash ^= hash >> 16;
        hash = hash.wrapping_mul(0x7feb_352d);
        hash ^= hash >> 15;
        hash & (directory_capacity - 1)
    };

    // Distinct world-grid keys in one open-addressing cluster exercise linear
    // probing alongside a much larger set of duplicate pixel descriptors.
    let collision_bucket = probe_bucket(0, 0, 0);
    let mut collision_cells = Vec::new();
    let mut collision_roots = Vec::new();
    for cell in 1..1_000_000i32 {
        if probe_bucket(cell, 0, 0) == collision_bucket && !collision_roots.contains(&(cell >> 3)) {
            collision_cells.push(cell);
            collision_roots.push(cell >> 3);
            if collision_cells.len() == 4 {
                break;
            }
        }
    }
    assert_eq!(
        collision_cells.len(),
        4,
        "find distinct keys in one probe cluster"
    );
    assert!(
        collision_cells
            .iter()
            .all(|cell| probe_bucket(*cell, 0, 0) == collision_bucket)
    );
    let mut unique_cells = collision_cells;
    for cell in [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96] {
        if !unique_cells.contains(&cell) {
            unique_cells.push(cell);
        }
    }
    unique_cells.truncate(15);
    assert_eq!(unique_cells.len(), 15);

    let mut cells = vec![0i32; CANDIDATES as usize];
    cells[0] = 1_000_000; // unique descriptor 0 is the first selected child
    cells[1..48].fill(1_000_003);
    cells[48..63].copy_from_slice(&unique_cells[..15]);
    cells[61] = -2;
    cells[62] = -1;
    cells[63] = 1_000_007; // visible, but never selected for a ray

    let selected_variant = |sample: usize| -> Option<u32> {
        match sample {
            0..=47 => Some(0),
            48..=63 => None,
            _ => unreachable!(),
        }
    };
    let selected_count = (0..CANDIDATES as usize)
        .filter(|sample| selected_variant(*sample).is_some())
        .count() as u32;
    assert_eq!(selected_count, 48);
    assert!(selected_count <= SELECTED_BUDGET);

    let descriptor_count = (CANDIDATES * 2 * 4) as usize;
    let mut entry_words = vec![0u32; descriptor_count * ENTRY_WORDS];
    let mut selected_descriptors = vec![false; descriptor_count];
    for (sample, &cell) in cells.iter().enumerate() {
        for variant in 0..2usize {
            for level in 0..4usize {
                let index = (variant * CANDIDATES as usize + sample) * 4 + level;
                let base = index * ENTRY_WORDS;
                entry_words[base] = (cell >> level) as u32;
                entry_words[base + 3] = ((variant as u32) << 18) | ((level as u32) << 12);
                let spacing = 1.2 * (1u32 << (level + variant)) as f32;
                let center = (cell as f32 / (1u32 << level) as f32).floor() + 0.5;
                entry_words[base + 12] = (center * spacing).to_bits();
                entry_words[base + 13] = (0.5 * spacing).to_bits();
                entry_words[base + 14] = (-0.5 * spacing).to_bits();
                entry_words[base + 15] = 1.0f32.to_bits();
                entry_words[base + 16] = u32::MAX; // EMPTY next
                entry_words[base + 17] = 1; // initialized world-grid descriptor
                entry_words[base + 24] = (sample as f32 * 0.01).to_bits();
                entry_words[base + 26] = (sample as f32 * -0.02).to_bits();
                let selected = level == 0 && selected_variant(sample) == Some(variant as u32);
                entry_words[base + 22] = if selected { 1 } else { 0 }; // padding.x ray marker
                entry_words[base + 23] = 1; // padding.y visible-init marker
                selected_descriptors[index] = selected;
            }
        }
    }
    let entry_bytes: Vec<u8> = entry_words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();

    let mut shader_source = String::from(
        r#"
#include "common.slang"
[[vk::binding(0, 0)]] ConstantBuffer<Parameters> params;
[[vk::binding(6, 0)]] RWStructuredBuffer<Atomic<uint>> heads;
[[vk::binding(7, 0)]] RWStructuredBuffer<Entry> entries;
[[vk::binding(13, 0)]] RWStructuredBuffer<uint4> testResults;
[[vk::binding(32, 0)]] RWStructuredBuffer<Atomic<uint>> rootRayOffset;
"#,
    );
    let split = include_str!("../shaders/split.slang");
    for source in [
        section(
            split,
            "float splitIntervalEnd(",
            "uint probeDirectoryCapacity(",
        ),
        section(
            split,
            "uint probeDirectoryCapacity(",
            "uint probeDescriptorIndex(",
        ),
        section(split, "uint probeDescriptorIndex(", "uint findProbeNode("),
        section(split, "uint findProbeNode(", "uint insertProbeDescriptor("),
        section(split, "uint insertProbeDescriptor(", "uint findEntry("),
        section(split, "void rank(", "// Every visible point seeds"),
        section(split, "void parents(", "[shader(\"compute\")]"),
        section(split, "void prefix(", "[shader(\"compute\")]"),
        section(split, "void generateDirections(", "uint currentCapacity("),
    ] {
        shader_source.push_str(source);
        shader_source.push('\n');
    }
    shader_source.push_str(
        r#"
[shader("compute")]
[numthreads(64, 1, 1)]
void rankStage(uint3 id : SV_DispatchThreadID) { rank(id); }
[shader("compute")]
[numthreads(64, 1, 1)]
void parentsStage(uint3 id : SV_DispatchThreadID) { parents(id); }
[shader("compute")]
[numthreads(64, 1, 1)]
void prefixStage(uint3 id : SV_DispatchThreadID) { prefix(id); }
[shader("compute")]
[numthreads(8, 8, 1)]
void directionsStage(uint3 id : SV_DispatchThreadID) { generateDirections(id); }

[shader("compute")]
[numthreads(64, 1, 1)]
void inspectDirectory(uint3 id : SV_DispatchThreadID) {
    uint count = params.size.z * 2u * CASCADES;
    if (id.x < count) {
        Entry descriptor = entries[id.x];
        uint canonical = descriptor.next;
        uint lookup = canonical == EMPTY ? EMPTY : findProbeNode(descriptor.key);
        uint offset = canonical == EMPTY ? EMPTY : entries[canonical].valid;
        uint level = (uint(descriptor.key.w) >> 12u) & 3u;
        uint slot = canonical == EMPTY ? EMPTY : findProbeNodeSlot(descriptor.key);
        uint samples = slot == EMPTY ? 0u : heads[probeCountHead(level, slot, params)].load();
        testResults[id.x] = uint4(canonical, lookup, offset, samples);
        testResults[count + id.x] = uint4(
            descriptor.lastSample, descriptor.padding.x,
            asuint(descriptor.normal.x), asuint(descriptor.normal.z));
    }
    if (id.x == 0u)
        testResults[2u * count] = uint4(
            rootRayOffset[0].load(), probeDirectoryCapacity(params), 0u, 0u);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void inspectIntervals(uint3 id : SV_DispatchThreadID) {
    if (id.x != 0u) return;
    uint base = params.size.z * 2u * CASCADES * 2u + 1u;
    float t0 = splitIntervalEnd(0u, 0u);
    float t1 = splitIntervalEnd(1u, 0u);
    float t2 = splitIntervalEnd(2u, 0u);
    float t3 = splitIntervalEnd(3u, 0u);
    float t0After = asfloat(asuint(t0) + 1u);
    float t1After = asfloat(asuint(t1) + 1u);
    float t2After = asfloat(asuint(t2) + 1u);
    float t3After = asfloat(asuint(t3) + 1u);
    testResults[base] = uint4(
        splitIntervalContainsHit(t0, 0u, 0u) ? 1u : 0u,
        splitIntervalContainsHit(t0, 1u, 0u) ? 1u : 0u,
        splitIntervalContainsHit(t0After, 0u, 0u) ? 1u : 0u,
        splitIntervalContainsHit(t0After, 1u, 0u) ? 1u : 0u);
    testResults[base + 1u] = uint4(
        splitIntervalContainsHit(t1, 1u, 0u) ? 1u : 0u,
        splitIntervalContainsHit(t1, 2u, 0u) ? 1u : 0u,
        splitIntervalContainsHit(t1After, 1u, 0u) ? 1u : 0u,
        splitIntervalContainsHit(t1After, 2u, 0u) ? 1u : 0u);
    testResults[base + 2u] = uint4(
        splitIntervalContainsHit(t2, 2u, 0u) ? 1u : 0u,
        splitIntervalContainsHit(t2, 3u, 0u) ? 1u : 0u,
        splitIntervalContainsHit(t2After, 2u, 0u) ? 1u : 0u,
        splitIntervalContainsHit(t2After, 3u, 0u) ? 1u : 0u);
    testResults[base + 3u] = uint4(
        splitIntervalContainsHit(t3, 3u, 0u) ? 1u : 0u,
        splitIntervalContainsHit(t3After, 0u, 0u) ? 1u : 0u,
        splitIntervalContainsHit(t3After, 1u, 0u) ? 1u : 0u,
        splitIntervalContainsHit(t3After, 2u, 0u) ? 1u : 0u);
    testResults[base + 4u] = uint4(
        splitIntervalContainsHit(t3After, 3u, 0u) ? 1u : 0u, 0u, 0u, 0u);
}
"#,
    );
    let shader = SlangCompiler::default()
        .compile_bundle(
            "split_r2_prefix_check.slang",
            &[
                ("common.slang", include_str!("../shaders/common.slang")),
                ("split_r2_prefix_check.slang", &shader_source),
            ],
            &SlangSettings {
                target: SlangTarget::Wgsl,
                entry_points: [
                    "rankStage",
                    "parentsStage",
                    "prefixStage",
                    "directionsStage",
                    "inspectDirectory",
                    "inspectIntervals",
                ]
                .into_iter()
                .map(str::to_owned)
                .collect(),
                optimization: Some(2),
                ..default()
            },
        )
        .unwrap();
    let bevy::shader::Source::Wgsl(source) = shader.source else {
        panic!("expected WGSL from the Split prefix regression shader");
    };
    let options = WgpuSettings::default();
    let resources = block_on(initialize_renderer(
        options.backends.unwrap_or(Backends::PRIMARY),
        None,
        &options,
    ));
    let device = &resources.0;
    let queue = &resources.1;
    let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
        label: Some("split_r2_prefix_regression"),
        source: ShaderSource::Wgsl(source),
    });
    let make_pipeline = |entry| {
        device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some(entry),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: default(),
            cache: None,
        })
    };
    let rank = make_pipeline("rankStage");
    let parents = make_pipeline("parentsStage");
    let prefix = make_pipeline("prefixStage");
    let directions = make_pipeline("directionsStage");
    let inspect = make_pipeline("inspectDirectory");
    let inspect_intervals = make_pipeline("inspectIntervals");

    let head_words = (2 * CACHE_BUCKETS + 36 + 8 * directory_capacity) as usize;
    let count_base = (2 * CACHE_BUCKETS + 36 + 4 * directory_capacity) as usize;
    let mut initial_heads = vec![u32::MAX; head_words];
    initial_heads[count_base..].fill(0);
    let head_bytes: Vec<u8> = initial_heads
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    let heads = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("split_r2_prefix_heads"),
        contents: &head_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    let entries = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("split_r2_prefix_entries"),
        contents: &entry_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    });
    let root_ray_offset = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("split_r2_root_ray_offset"),
        contents: &[0u32.to_le_bytes()].concat(),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    });
    let result_rows = descriptor_count * 2 + 6;
    let results = device.create_buffer(&BufferDescriptor {
        label: Some("split_r2_prefix_results"),
        size: (result_rows * 16) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut base_parameters = temporal_test_parameters(0);
    base_parameters.size = UVec4::new(CANDIDATES, 1, CANDIDATES, CACHE_BUCKETS);
    base_parameters.tuning.y = 0.25;
    base_parameters.sampling.x = SELECTED_BUDGET;
    let make_parameters = |level: u32| {
        let mut value = base_parameters;
        value.frame.y = level;
        let mut uniform = UniformBuffer::from(value);
        uniform.write_buffer(device, queue);
        uniform
    };
    let make_group = |pipeline: &ComputePipeline,
                      parameters: &UniformBuffer<TransportParameters>,
                      has_heads: bool,
                      has_entries: bool,
                      has_root_offset: bool,
                      has_results: bool| {
        let mut bindings = vec![BindGroupEntry {
            binding: 0,
            resource: parameters.binding().unwrap(),
        }];
        if has_heads {
            bindings.push(BindGroupEntry {
                binding: 6,
                resource: heads.as_entire_binding(),
            });
        }
        if has_entries {
            bindings.push(BindGroupEntry {
                binding: 7,
                resource: entries.as_entire_binding(),
            });
        }
        if has_results {
            bindings.push(BindGroupEntry {
                binding: 13,
                resource: results.as_entire_binding(),
            });
        }
        if has_root_offset {
            bindings.push(BindGroupEntry {
                binding: 32,
                resource: root_ray_offset.as_entire_binding(),
            });
        }
        device.create_bind_group(
            "split_r2_prefix_test",
            &pipeline.get_bind_group_layout(0).into(),
            &bindings,
        )
    };
    let entry_bytes_len = (descriptor_count * ENTRY_WORDS * 4) as u64;
    let dispatch_and_wait = |pipeline: &ComputePipeline,
                             group: &BindGroup,
                             output: &Buffer,
                             bytes: u64,
                             workgroups: u32| {
        crate::scene::traversal_tests::dispatch_readback(
            device, queue, pipeline, group, output, bytes, workgroups,
        )
    };

    let rank_parameters = make_parameters(0);
    let rank_group = make_group(&rank, &rank_parameters, true, true, false, false);
    let _ = dispatch_and_wait(&rank, &rank_group, &entries, entry_bytes_len, 1);
    for level in 1..4 {
        let parameters = make_parameters(level);
        let group = make_group(&parents, &parameters, true, true, false, false);
        let _ = dispatch_and_wait(
            &parents,
            &group,
            &entries,
            entry_bytes_len,
            directory_capacity.div_ceil(64),
        );
    }
    for level in (0..4).rev() {
        let parameters = make_parameters(level);
        let group = make_group(&prefix, &parameters, true, true, true, false);
        let _ = dispatch_and_wait(
            &prefix,
            &group,
            &entries,
            entry_bytes_len,
            directory_capacity.div_ceil(64),
        );
    }
    let direction_parameters = make_parameters(0);
    let direction_group = make_group(
        &directions,
        &direction_parameters,
        false,
        true,
        false,
        false,
    );
    let _ = dispatch_and_wait(
        &directions,
        &direction_group,
        &entries,
        entry_bytes_len,
        CANDIDATES.div_ceil(8),
    );
    let inspect_parameters = make_parameters(0);
    let inspect_group = make_group(&inspect, &inspect_parameters, true, true, true, true);
    let result_bytes = (result_rows * 16) as u64;
    let _ = dispatch_and_wait(
        &inspect,
        &inspect_group,
        &results,
        result_bytes,
        (descriptor_count as u32).div_ceil(64),
    );
    let interval_group = make_group(
        &inspect_intervals,
        &inspect_parameters,
        false,
        false,
        false,
        true,
    );
    let result_bytes_read = dispatch_and_wait(
        &inspect_intervals,
        &interval_group,
        &results,
        result_bytes,
        1,
    );

    #[derive(Clone, Copy)]
    struct Node {
        lod: u32,
        level: u32,
        cell: i32,
        canonical: u32,
        offset: u32,
        samples: u32,
    }
    let mut nodes: Vec<Node> = Vec::new();
    let mut ray_indices = Vec::new();
    let mut selected_lods = [0u32; 2];
    for (sample, &cell) in cells.iter().enumerate() {
        for lod in 0..2u32 {
            for level in 0..4u32 {
                let index = (lod as usize * CANDIDATES as usize + sample) * 4 + level as usize;
                let probe = output_u32x4(&result_bytes_read, index);
                let ray = output_u32x4(&result_bytes_read, descriptor_count + index);
                assert_ne!(
                    probe[0],
                    u32::MAX,
                    "initialized descriptor {index} was not ranked"
                );
                assert_eq!(
                    probe[0], probe[1],
                    "directory lookup failed for descriptor {index}"
                );
                assert!((probe[0] as usize) < descriptor_count);
                if selected_descriptors[index] {
                    selected_lods[lod as usize] += 1;
                    ray_indices.push(ray[0]);
                }
                let key_cell = cell >> level;
                if let Some(existing) = nodes
                    .iter()
                    .find(|node| node.lod == lod && node.level == level && node.cell == key_cell)
                {
                    assert_eq!(existing.canonical, probe[0]);
                    assert_eq!(existing.offset, probe[2]);
                    assert_eq!(existing.samples, probe[3]);
                } else {
                    nodes.push(Node {
                        lod,
                        level,
                        cell: key_cell,
                        canonical: probe[0],
                        offset: probe[2],
                        samples: probe[3],
                    });
                }
            }
        }
    }
    ray_indices.sort_unstable();
    assert_eq!(ray_indices, (0..selected_count).collect::<Vec<_>>());
    assert_eq!(
        selected_lods,
        [selected_count, 0],
        "overlap selected only its low LOD this frame"
    );
    let root_and_capacity = output_u32x4(&result_bytes_read, descriptor_count * 2);
    assert_eq!(root_and_capacity[0], selected_count);
    assert_eq!(root_and_capacity[1], directory_capacity);
    assert!(root_and_capacity[0] <= SELECTED_BUDGET);
    assert_eq!(
        output_u32x4(&result_bytes_read, 0)[0],
        0,
        "descriptor index zero is canonical"
    );
    assert_eq!(
        output_u32x4(&result_bytes_read, 0)[2],
        0,
        "the first child has a valid zero prefix"
    );
    assert_ne!(
        output_u32x4(&result_bytes_read, descriptor_count + 4)[2..],
        output_u32x4(&result_bytes_read, descriptor_count + 8)[2..],
        "different physical origins sharing one grid key must share one probe"
    );

    let zero_probe = nodes
        .iter()
        .find(|node| node.lod == 0 && node.level == 0 && node.cell == cells[63])
        .expect("initialized zero-ray probe");
    assert_eq!(zero_probe.samples, 0);
    let negative_parent = nodes
        .iter()
        .find(|node| node.lod == 0 && node.level == 1 && node.cell == -1)
        .expect("-2 and -1 children must share their arithmetic-shift parent -1");
    assert_eq!(negative_parent.samples, 0);
    let mut negative_children: Vec<_> = nodes
        .iter()
        .filter(|node| node.lod == 0 && node.level == 0 && node.cell >> 1 == -1)
        .map(|node| node.cell)
        .collect();
    negative_children.sort_unstable();
    assert_eq!(negative_children, [-2, -1]);
    assert_eq!(
        nodes
            .iter()
            .find(|node| node.lod == 0 && node.level == 0 && node.cell == 1_000_003)
            .unwrap()
            .samples,
        47,
        "all selected physical origins in one world-grid cell must aggregate"
    );
    for parent in nodes.iter().filter(|node| node.level > 0) {
        let mut children: Vec<_> = nodes
            .iter()
            .filter(|child| {
                child.lod == parent.lod
                    && child.level + 1 == parent.level
                    && child.cell >> 1 == parent.cell
            })
            .collect();
        children.sort_by_key(|child| child.cell);
        let mut cursor = parent.offset;
        for child in children {
            assert_eq!(
                child.offset, cursor,
                "LOD {}, level {} child {} is not contiguous under parent {}",
                parent.lod, child.level, child.cell, parent.cell
            );
            cursor += child.samples;
        }
        assert_eq!(cursor, parent.offset + parent.samples);
    }
    let mut roots: Vec<_> = nodes
        .iter()
        .filter(|node| node.level == 3 && node.samples > 0)
        .collect();
    roots.sort_by_key(|root| root.offset);
    let mut root_cursor = 0;
    for root in roots {
        assert_eq!(
            root.offset, root_cursor,
            "root ray blocks overlap or have a hole"
        );
        root_cursor += root.samples;
    }
    assert_eq!(root_cursor, selected_count);
    assert!(
        nodes
            .iter()
            .any(|node| node.level == 3 && node.samples > 0 && node.offset == 0)
    );
    assert!(
        ray_indices.contains(&0),
        "a valid zero prefix must produce ray index zero"
    );

    let interval_base = descriptor_count * 2 + 1;
    assert_eq!(
        output_u32x4(&result_bytes_read, interval_base),
        [1, 0, 0, 1]
    );
    assert_eq!(
        output_u32x4(&result_bytes_read, interval_base + 1),
        [1, 0, 0, 1]
    );
    assert_eq!(
        output_u32x4(&result_bytes_read, interval_base + 2),
        [1, 0, 0, 1]
    );
    assert_eq!(
        output_u32x4(&result_bytes_read, interval_base + 3),
        [1, 0, 0, 0]
    );
    assert_eq!(output_u32x4(&result_bytes_read, interval_base + 4)[0], 0);
}

#[test]
#[ignore = "requires slangc and a desktop GPU; dispatches production Split probe packing"]
fn gpu_visible_zero_ray_probe_packs_cached_radiance() {
    fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        let start = source.find(start).expect("production shader section start");
        let tail = &source[start..];
        let end = tail.find(end).expect("production shader section end");
        &tail[..end]
    }

    const CACHE_BUCKETS: u32 = 16;
    const DIRECTORY_CAPACITY: u32 = 4;
    const ENTRY_WORDS: usize = 208 / 4;
    let key = [3i32, -2, 1, 0];
    let hash_key = |key: [i32; 4], buckets: u32| {
        let key = key.map(|value| value as u32);
        let mut hash = key[0].wrapping_mul(0x8da6_b343)
            ^ key[1].wrapping_mul(0xd816_3841)
            ^ key[2].wrapping_mul(0xcb1a_b31f)
            ^ key[3].wrapping_mul(0x1656_67b1);
        hash ^= hash >> 16;
        hash = hash.wrapping_mul(0x7feb_352d);
        hash ^= hash >> 15;
        hash & (buckets - 1)
    };
    let cache_bucket = hash_key(key, CACHE_BUCKETS) as usize;
    let directory_slot = hash_key(key, DIRECTORY_CAPACITY) as usize;

    let mut shader_source = String::from(
        r#"
#include "common.slang"
[[vk::binding(0, 0)]] ConstantBuffer<Parameters> params;
[[vk::binding(6, 0)]] RWStructuredBuffer<Atomic<uint>> heads;
[[vk::binding(7, 0)]] RWStructuredBuffer<Entry> entries;
[[vk::binding(8, 0)]] StructuredBuffer<uint> previousHeads;
[[vk::binding(9, 0)]] RWStructuredBuffer<Entry> previousEntries;
[[vk::binding(12, 0)]] RWStructuredBuffer<DirectionalProbe> directionalProbes;
"#,
    );
    let production = include_str!("../shaders/split.slang");
    for source in [
        section(
            production,
            "uint transportCacheBucket(",
            "float splitIntervalEnd(",
        ),
        section(
            production,
            "uint probeDirectoryCapacity(",
            "uint probeDescriptorIndex(",
        ),
        section(production, "uint findEntry(", "uint findIrradianceProbe("),
        section(production, "uint currentCapacity()", "[shader"),
        section(production, "void seedProbes(", "[shader"),
        section(production, "void packProbes(", "[shader"),
    ] {
        shader_source.push_str(source);
        shader_source.push('\n');
    }
    shader_source.push_str(
        r#"
[shader("compute")]
[numthreads(64, 1, 1)]
void seedProbeStage(uint3 id : SV_DispatchThreadID) {
    seedProbes(uint3(id.x / 4u, id.x % 4u, 0u));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void packProbeStage(uint3 id : SV_DispatchThreadID) { packProbes(id); }
"#,
    );
    let shader = SlangCompiler::default()
        .compile_bundle(
            "split_visible_zero_probe_check.slang",
            &[
                ("common.slang", include_str!("../shaders/common.slang")),
                ("split_visible_zero_probe_check.slang", &shader_source),
            ],
            &SlangSettings {
                target: SlangTarget::Wgsl,
                entry_points: ["seedProbeStage", "packProbeStage"]
                    .into_iter()
                    .map(str::to_owned)
                    .collect(),
                optimization: Some(2),
                ..default()
            },
        )
        .unwrap();
    let bevy::shader::Source::Wgsl(source) = shader.source else {
        panic!("expected WGSL from the Split probe packing regression shader");
    };
    let options = WgpuSettings::default();
    let resources = block_on(initialize_renderer(
        options.backends.unwrap_or(Backends::PRIMARY),
        None,
        &options,
    ));
    let device = &resources.0;
    let queue = &resources.1;
    let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
        label: Some("split_visible_zero_probe_regression"),
        source: ShaderSource::Wgsl(source),
    });
    let make_pipeline = |entry| {
        device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some(entry),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: default(),
            cache: None,
        })
    };
    let seed = make_pipeline("seedProbeStage");
    let pack = make_pipeline("packProbeStage");

    let mut parameters = temporal_test_parameters(0);
    parameters.size = UVec4::new(1, 1, 1, CACHE_BUCKETS);
    parameters.frame.x = 0;
    let mut parameters = UniformBuffer::from(parameters);
    parameters.write_buffer(device, queue);

    let mut heads_words =
        vec![u32::MAX; (2 * CACHE_BUCKETS + 36 + 8 * DIRECTORY_CAPACITY) as usize];
    heads_words[cache_bucket] = 0;
    let heads_bytes: Vec<u8> = heads_words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    let heads = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("split_visible_zero_probe_heads"),
        contents: &heads_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    });

    let mut entry_words = vec![0u32; 8 * ENTRY_WORDS];
    for index in 0..8 {
        entry_words[index * ENTRY_WORDS + 16] = u32::MAX;
    }
    for axis in 0..4 {
        entry_words[axis] = key[axis] as u32;
    }
    entry_words[8] = 4.0f32.to_bits();
    entry_words[9] = 2.0f32.to_bits();
    entry_words[10] = 1.0f32.to_bits();
    entry_words[11] = 1.0f32.to_bits();
    entry_words[17] = 0; // valid zero prefix
    entry_words[23] = 1; // visible initialized candidate, even with no ray
    let entry_bytes: Vec<u8> = entry_words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    let entries = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("split_visible_zero_probe_entries"),
        contents: &entry_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    });
    let previous_heads_bytes: Vec<u8> = vec![u32::MAX; CACHE_BUCKETS as usize]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    let previous_heads = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("split_visible_zero_probe_previous_heads"),
        contents: &previous_heads_bytes,
        usage: BufferUsages::STORAGE,
    });
    let previous_entries = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("split_visible_zero_probe_previous_entries"),
        contents: &entry_bytes,
        usage: BufferUsages::STORAGE,
    });
    let directional_bytes = (2 * DIRECTORY_CAPACITY * 544) as usize;
    let directional_probes = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("split_visible_zero_probe_radiance"),
        contents: &vec![0u8; directional_bytes],
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    });
    let seed_group = device.create_bind_group(
        "split_visible_zero_probe_seed",
        &seed.get_bind_group_layout(0).into(),
        &[
            BindGroupEntry {
                binding: 0,
                resource: parameters.binding().unwrap(),
            },
            BindGroupEntry {
                binding: 6,
                resource: heads.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 7,
                resource: entries.as_entire_binding(),
            },
        ],
    );
    let _ = crate::scene::traversal_tests::dispatch_readback(
        device,
        queue,
        &seed,
        &seed_group,
        &entries,
        entry_bytes.len() as u64,
        1,
    );
    let pack_group = device.create_bind_group(
        "split_visible_zero_probe_pack",
        &pack.get_bind_group_layout(0).into(),
        &[
            BindGroupEntry {
                binding: 0,
                resource: parameters.binding().unwrap(),
            },
            BindGroupEntry {
                binding: 6,
                resource: heads.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 7,
                resource: entries.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 8,
                resource: previous_heads.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 9,
                resource: previous_entries.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 12,
                resource: directional_probes.as_entire_binding(),
            },
        ],
    );
    let packed = crate::scene::traversal_tests::dispatch_readback(
        device,
        queue,
        &pack,
        &pack_group,
        &directional_probes,
        directional_bytes as u64,
        1,
    );
    assert_eq!(
        output_vec4(&packed, directory_slot * (544 / 16)),
        [4.0, 2.0, 1.0, 1.0],
        "a visible zero-ray grid probe must still pack its cached directional radiance"
    );
}
#[test]
#[ignore = "requires slangc and a desktop GPU; dispatches production Split interval deposition"]
fn gpu_split_deposit_stops_at_first_hit_and_leaves_far_intervals_unfilled() {
    fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        let start = source.find(start).expect("production shader section start");
        let tail = &source[start..];
        let end = tail.find(end).expect("production shader section end");
        &tail[..end]
    }

    const CACHE_BUCKETS: u32 = 16;
    const ENTRY_WORDS: usize = 208 / 4;
    const SENTINEL: [f32; 4] = [9.0, 8.0, 7.0, 6.0];
    let production = include_str!("../shaders/split.slang");

    let mut shader_source = String::from(
        r#"
#include "common.slang"
[[vk::binding(0, 0)]] ConstantBuffer<Parameters> params;
[[vk::binding(6, 0)]] RWStructuredBuffer<Atomic<uint>> heads;
[[vk::binding(7, 0)]] RWStructuredBuffer<Entry> entries;
[[vk::binding(13, 0)]] RWStructuredBuffer<uint4> results;
struct Hit {
    float distance;
    uint triangle;
    uint instance;
    uint errors;
    float3 barycentric;
    float footprint;
};
static const uint HIT_QUERY_UNSUPPORTED = 4u;
"#,
    );
    for source in [
        section(
            production,
            "float4 emptyLightHitBounds(",
            "float4 unionConservativeSpheres(",
        ),
        section(
            production,
            "uint transportCacheBucket(",
            "float splitIntervalEnd(",
        ),
        section(
            production,
            "float splitIntervalEnd(",
            "uint probeDirectoryCapacity(",
        ),
        section(production, "void deposit(", "void traceTransport("),
    ] {
        shader_source.push_str(source);
        shader_source.push('\n');
    }
    shader_source.push_str(
        r#"
[shader("compute")]
[numthreads(1, 1, 1)]
void depositCheck(uint3 id : SV_DispatchThreadID) {
    Hit hit = (Hit)0;
    hit.distance = splitIntervalEnd(1u, 0u);
    DependencyContext rayDependency = dependencyEmpty();
    DependencyContext outgoingDependency = dependencyEmpty();
    deposit(0u, float3(0.5, 0.5, 0.5), float3(0.0),
        float3(0, 0, 1), hit, float3(4, 2, 1), 0u,
        rayDependency, 0.0, hit.distance, 0.001, outgoingDependency);
    [unroll] for (uint level = 0u; level < 4u; ++level)
        results[level] = asuint(entries[level].interval);
    results[4] = uint4(entries[0].valid, entries[1].valid,
                       entries[2].valid, entries[3].valid);
}
"#,
    );
    let shader = SlangCompiler::default()
        .compile_bundle(
            "split_deposit_check.slang",
            &[
                ("common.slang", include_str!("../shaders/common.slang")),
                ("split_deposit_check.slang", &shader_source),
            ],
            &SlangSettings {
                target: SlangTarget::Wgsl,
                entry_points: vec!["depositCheck".into()],
                optimization: Some(2),
                ..default()
            },
        )
        .unwrap();
    let bevy::shader::Source::Wgsl(source) = shader.source else {
        panic!("expected WGSL from the Split interval deposition regression shader");
    };
    let options = WgpuSettings::default();
    let resources = block_on(initialize_renderer(
        options.backends.unwrap_or(Backends::PRIMARY),
        None,
        &options,
    ));
    let device = &resources.0;
    let queue = &resources.1;
    let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
        label: Some("split_deposit_regression"),
        source: ShaderSource::Wgsl(source),
    });
    let pipeline = device.create_compute_pipeline(&RawComputePipelineDescriptor {
        label: Some("split_deposit_regression"),
        layout: None,
        module: &module,
        entry_point: Some("depositCheck"),
        compilation_options: default(),
        cache: None,
    });
    let mut parameters = temporal_test_parameters(0);
    parameters.size = UVec4::new(1, 1, 1, CACHE_BUCKETS);
    parameters.tuning.x = 0.25;
    parameters.tuning.y = 0.25;
    parameters.frame.x = 7;
    let mut parameters = UniformBuffer::from(parameters);
    parameters.write_buffer(device, queue);

    let head_words = 2 * CACHE_BUCKETS + 36;
    let heads_bytes: Vec<u8> = vec![u32::MAX; head_words as usize]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    let heads = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("split_deposit_heads"),
        contents: &heads_bytes,
        usage: BufferUsages::STORAGE,
    });
    let mut entry_words = vec![0u32; 4 * ENTRY_WORDS];
    for level in 0..4 {
        let base = level * ENTRY_WORDS;
        entry_words[base + 16] = u32::MAX;
        for (component, value) in SENTINEL.iter().enumerate() {
            entry_words[base + 4 + component] = value.to_bits();
        }
    }
    let entry_bytes: Vec<u8> = entry_words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    let entries = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("split_deposit_entries"),
        contents: &entry_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    let output = device.create_buffer(&BufferDescriptor {
        label: Some("split_deposit_results"),
        size: 5 * 16,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let group = device.create_bind_group(
        "split_deposit_check",
        &pipeline.get_bind_group_layout(0).into(),
        &[
            BindGroupEntry {
                binding: 0,
                resource: parameters.binding().unwrap(),
            },
            BindGroupEntry {
                binding: 6,
                resource: heads.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 7,
                resource: entries.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 13,
                resource: output.as_entire_binding(),
            },
        ],
    );
    let bytes = crate::scene::traversal_tests::dispatch_readback(
        device,
        queue,
        &pipeline,
        &group,
        &output,
        5 * 16,
        1,
    );
    assert_eq!(output_vec4(&bytes, 0), [0.0, 0.0, 0.0, 1.0]);
    assert_eq!(output_vec4(&bytes, 1), [4.0, 2.0, 1.0, 0.0]);
    assert_eq!(output_vec4(&bytes, 2), SENTINEL);
    assert_eq!(output_vec4(&bytes, 3), SENTINEL);
    let written = output_u32x4(&bytes, 4);
    assert_eq!(written[..2], [0x8000_0007; 2]);
    assert_eq!(written[2..], [0, 0]);
}
fn dense_reference_box_sdf(point: [f64; 3], center: [f64; 3], half_size: [f64; 3]) -> f64 {
    let q =
        std::array::from_fn::<_, 3, _>(|axis| (point[axis] - center[axis]).abs() - half_size[axis]);
    let outside = q
        .iter()
        .map(|value| value.max(0.0).powi(2))
        .sum::<f64>()
        .sqrt();
    outside + q[0].max(q[1].max(q[2])).min(0.0)
}

fn dense_reference_scene_sdf(point: [f64; 3]) -> (f64, u32) {
    let mut result = (
        dense_reference_box_sdf(point, [-3.0, 0.0, 0.0], [0.9, 0.9, 0.9]),
        1,
    );
    let sphere =
        ((point[0] - 3.0).powi(2) + (point[1] - 1.0).powi(2) + point[2].powi(2)).sqrt() - 1.0;
    if sphere < result.0 {
        result = (sphere, 2);
    }
    for (center, half_size, material) in [
        ([0.0, 3.0, 0.0], [2.0, 0.04, 2.0], 3),
        ([0.0, 0.0, 4.0], [2.0, 0.5, 0.03], 4),
    ] {
        let distance = dense_reference_box_sdf(point, center, half_size);
        if distance < result.0 {
            result = (distance, material);
        }
    }
    result
}

fn dense_reference_direction(index: u32, level: u32) -> [f64; 3] {
    let theta = 4u32 << level;
    let x = index % (2 * theta);
    let y = index / (2 * theta);
    let u = (f64::from(x) + 0.5) / f64::from(2 * theta);
    let v = (f64::from(y) + 0.5) / f64::from(theta);
    let phi = u * 2.0 * std::f64::consts::PI;
    let vertical = v * 2.0 - 1.0;
    let radius = (1.0 - vertical * vertical).max(0.0).sqrt();
    [radius * phi.cos(), vertical, radius * phi.sin()]
}

fn dense_reference_direction_index(direction: [f64; 3], level: u32) -> u32 {
    let theta = 4u32 << level;
    let u = ((direction[2].atan2(direction[0]) / (2.0 * std::f64::consts::PI) + 1.0)
        .rem_euclid(1.0))
    .clamp(0.0, 1.0);
    let v = (direction[1] * 0.5 + 0.5).clamp(0.0, 1.0);
    let x = ((u * f64::from(2 * theta)) as u32).min(2 * theta - 1);
    let y = ((v * f64::from(theta)) as u32).min(theta - 1);
    x + y * 2 * theta
}

fn dense_reference_trace(
    origin: [f64; 3],
    direction: [f64; 3],
    near_distance: f64,
    far_distance: f64,
) -> ([f32; 4], u32) {
    let mut distance_along_ray = near_distance + 0.001;
    for _ in 0..96 {
        if distance_along_ray > far_distance {
            break;
        }
        let point = std::array::from_fn::<_, 3, _>(|axis| {
            origin[axis] + direction[axis] * distance_along_ray
        });
        let (distance, material) = dense_reference_scene_sdf(point);
        if distance < 0.001 {
            return (
                if material == 3 {
                    [6.0, 2.0, 0.5, 0.0]
                } else {
                    [0.0, 0.0, 0.0, 0.0]
                },
                material,
            );
        }
        distance_along_ray += distance.max(0.005);
    }
    ([0.0, 0.0, 0.0, 1.0], 0)
}

fn dense_reference_cell(position: f32, spacing: f32) -> Option<u32> {
    ((-8.0..8.0).contains(&position)).then_some(((position + 8.0) / spacing).floor() as u32)
}

#[test]
#[ignore = "requires slangc and a desktop GPU; compares dense 3D SDF intervals with a CPU reference"]
fn gpu_dense_16_cubed_sdf_matches_cpu_and_production_spherical_mapping() {
    const RECORDS: usize = 245_760;
    let options = WgpuSettings::default();
    let resources = block_on(initialize_renderer(
        options.backends.unwrap_or(Backends::PRIMARY),
        None,
        &options,
    ));
    let device = &resources.0;
    let queue = &resources.1;
    let shader = SlangCompiler::default()
        .compile_bundle(
            "dense_sdf_reference.slang",
            &[
                ("common.slang", include_str!("../shaders/common.slang")),
                ("dense_sdf_reference.slang", DENSE_REFERENCE_SHADER),
            ],
            &SlangSettings {
                target: SlangTarget::Wgsl,
                entry_points: vec!["checkDenseSdf".into()],
                optimization: Some(2),
                ..default()
            },
        )
        .unwrap();
    let bevy::shader::Source::Wgsl(source) = shader.source else {
        panic!("expected WGSL from dense SDF reference shader");
    };
    let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
        label: Some("dense_sdf_reference"),
        source: ShaderSource::Wgsl(source),
    });
    let pipeline = device.create_compute_pipeline(&RawComputePipelineDescriptor {
        label: Some("dense_sdf_reference"),
        layout: None,
        module: &module,
        entry_point: Some("checkDenseSdf"),
        compilation_options: default(),
        cache: None,
    });
    let output = device.create_buffer(&BufferDescriptor {
        label: Some("dense_sdf_reference_output"),
        size: (RECORDS * 2 * 16) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let group = device.create_bind_group(
        "dense_sdf_reference",
        &pipeline.get_bind_group_layout(0).into(),
        &[BindGroupEntry {
            binding: 0,
            resource: output.as_entire_binding(),
        }],
    );
    let bytes = crate::scene::traversal_tests::dispatch_readback(
        device,
        queue,
        &pipeline,
        &group,
        &output,
        (RECORDS * 2 * 16) as u64,
        RECORDS.div_ceil(64) as u32,
    );

    let mut seen = vec![false; RECORDS];
    let mut expected = vec![[0.0; 4]; RECORDS];
    let mut expected_directions = vec![[0.0; 4]; RECORDS];
    let mut material_hits = [0usize; 5];
    let mut offset = 0usize;
    for level in 0..4u32 {
        let dimension = 16u32 >> level;
        let directions = 32u32 << (2 * level);
        let spacing = (1u32 << level) as f64;
        let near_distance = f64::from((1u32 << level) - 1);
        let far_distance = f64::from((1u32 << (level + 1)) - 1);
        for z in 0..dimension {
            for y in 0..dimension {
                for x in 0..dimension {
                    let probe = x + y * dimension + z * dimension * dimension;
                    let origin = [
                        -8.0 + (f64::from(x) + 0.5) * spacing,
                        -8.0 + (f64::from(y) + 0.5) * spacing,
                        -8.0 + (f64::from(z) + 0.5) * spacing,
                    ];
                    for direction_index in 0..directions {
                        let address = offset + (probe * directions + direction_index) as usize;
                        assert!(!seen[address], "duplicate dense address {address}");
                        seen[address] = true;
                        let direction = dense_reference_direction(direction_index, level);
                        let (interval, material) =
                            dense_reference_trace(origin, direction, near_distance, far_distance);
                        material_hits[material as usize] += usize::from(material != 0);
                        expected[address] = interval;
                        expected_directions[address] = [
                            direction[0] as f32,
                            direction[1] as f32,
                            direction[2] as f32,
                            direction_index as f32,
                        ];
                    }
                }
            }
        }
        offset += (dimension.pow(3) * directions) as usize;
    }
    assert_eq!(offset, RECORDS);
    assert!(seen.into_iter().all(|address| address));
    assert!(material_hits[1] > 0, "box must be hit by dense rays");
    assert!(material_hits[2] > 0, "sphere must be hit by dense rays");
    assert!(
        material_hits[3] > 0,
        "emissive surface must be hit by dense rays"
    );
    assert!(
        material_hits[4] > 0,
        "thin blocker must be hit by dense rays"
    );

    for level in 0..4 {
        let theta = 4u32 << level;
        for direction_index in 0..(32u32 << (2 * level)) {
            let direction = dense_reference_direction(direction_index, level);
            assert_eq!(
                dense_reference_direction_index(direction, level),
                direction_index
            );
        }
        for direction in [
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [1e-9, 0.0, -1.0],
            [-1e-9, 0.0, -1.0],
        ] {
            assert!(dense_reference_direction_index(direction, level) < 2 * theta * theta);
        }
        let dimension = 16u32 >> level;
        assert_eq!(dense_reference_cell(-8.0, (1u32 << level) as f32), Some(0));
        assert_eq!(dense_reference_cell(8.0, (1u32 << level) as f32), None);
        assert_eq!(
            dense_reference_cell(7.999, (1u32 << level) as f32),
            Some(dimension - 1)
        );
        for cell in [0, dimension - 1] {
            let center = -8.0 + (cell as f32 + 0.5) * (1u32 << level) as f32;
            assert_eq!(
                dense_reference_cell(center, (1u32 << level) as f32),
                Some(cell)
            );
        }
    }

    for address in 0..RECORDS {
        let interval = std::array::from_fn::<_, 4, _>(|component| {
            let start = (address * 2) * 16 + component * 4;
            f32::from_le_bytes(bytes[start..start + 4].try_into().unwrap())
        });
        let direction = std::array::from_fn::<_, 4, _>(|component| {
            let start = (address * 2 + 1) * 16 + component * 4;
            f32::from_le_bytes(bytes[start..start + 4].try_into().unwrap())
        });
        assert_eq!(
            interval[3], expected[address][3],
            "transmittance at {address}"
        );
        assert!(
            interval[..3]
                .iter()
                .zip(expected[address][..3].iter())
                .all(|(actual, wanted)| (actual - wanted).abs() < 1e-4),
            "radiance mismatch at dense address {address}: {interval:?} != {:?}",
            expected[address]
        );
        assert_eq!(
            direction[3], expected_directions[address][3],
            "spherical bin {address}"
        );
        assert!(
            direction[..3]
                .iter()
                .zip(expected_directions[address][..3].iter())
                .all(|(actual, wanted)| (actual - wanted).abs() < 1e-5),
            "direction mismatch at dense address {address}: {direction:?} != {:?}",
            expected_directions[address]
        );
    }
}

//! Ignored GPU checks for the production transport helpers.
use bevy::{
    asset::RenderAssetUsages,
    math::{Mat4, Quat, UVec2, UVec4, Vec2, Vec3, Vec4},
    prelude::{Image, default},
    render::{
        render_resource::*,
        renderer::initialize_renderer,
        settings::{Backends, WgpuSettings},
    },
    tasks::block_on,
};
use bevy_slang::{SlangCompiler, SlangSettings, SlangTarget};

#[path = "compact_clear_tests.rs"]
mod compact_clear_tests;
#[path = "compact_trace_tests.rs"]
mod compact_trace_tests;
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
    source_history: UVec4,
    sampling: UVec4,
    owner_sequence: UVec4,
    cache_time: UVec4,
    emissive_dirty_min: Vec4,
    emissive_dirty_max: Vec4,
    light_dirty_min: Vec4,
    light_dirty_max: Vec4,
    ray_traced: UVec4,
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
    // Classify finite f32 boundaries and every non-finite component at runtime.
    uint payload = params.frame.x & 1u;
    float4 finiteValues = asfloat(uint4(0u, 0x80000000u, 1u, 0x7f7fffffu - payload));
    float4 nonFinite = asfloat(uint4(0x7f800000u, 0xff800000u,
                                     0x7fc00000u | payload, 0xffc00000u | payload));
    results[26] = float4(all(isfinite(finiteValues.xy)) ? 1.0 : 0.0,
                          all(isfinite(finiteValues.xyz)) ? 1.0 : 0.0,
                          all(isfinite(finiteValues)) ? 1.0 : 0.0,
                          any(isfinite(nonFinite)) ? 1.0 : 0.0);
    results[27] = float4(any(isfinite(nonFinite.xy)) ? 1.0 : 0.0,
                          any(isfinite(nonFinite.xyz)) ? 1.0 : 0.0,
                          isfinite(nonFinite.zxyw).w ? 1.0 : 0.0,
                          isfinite(nonFinite.wzyx).w ? 1.0 : 0.0);
    // Exercise both UV-reuse paths against independent CPU PDF/radiance
    // expectations with a rotated, non-uniform HDR texture.
    for (uint index = 0; index < 8; ++index) {
        float4 random = float4((float(index) + 0.25) / 8.0, 0, 0.5, 0.5);
        float3 direction, radiance;
        float pdf;
        float2 uv;
        bool valid = sampleEnvironment(random, direction, pdf, radiance, uv);
        results[28 + index * 2] = float4(direction, environmentPdfUv(uv));
        results[29 + index * 2] = float4(environmentRadianceUv(uv), valid ? pdf : -1.0);

        direction = cosineDirection(float3(0, 1, 0),
            float2((float(index) + 0.3) / 8.0, (float(index) + 0.37) / 8.0));
        uv = environmentUv(direction);
        results[44 + index * 2] = float4(direction, environmentPdfUv(uv));
        results[45 + index * 2] = float4(environmentRadianceUv(uv), 1.0);
    }
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

fn expected_hdr_test_radiance(direction: Vec3) -> Vec3 {
    let direction = direction.normalize();
    let u = (direction.z.atan2(direction.x) / std::f32::consts::TAU + 0.5).rem_euclid(1.0);
    let v = direction.x.hypot(direction.z).atan2(direction.y) / std::f32::consts::PI;
    let x = u * 2.0 - 0.5;
    let y = v * 4.0 - 0.5;
    let low_x = x.floor() as i32;
    let low_y = y.floor() as i32;
    let fx = x - x.floor();
    let fy = y - y.floor();
    let x_weight = if low_x.rem_euclid(2) == 0 {
        1.0 - fx
    } else {
        fx
    };
    let y_weight = if low_y.clamp(0, 3) == 0 {
        1.0 - fy
    } else {
        0.0
    } + if (low_y + 1).clamp(0, 3) == 0 {
        fy
    } else {
        0.0
    };
    Vec3::new(100.0, 60.0, 20.0) * (x_weight * y_weight)
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
    current_background: Texture,
    previous_background: Texture,
    motion: Texture,
    raster_receiver: Texture,
    gbuffer: Texture,
    irradiance_output: Texture,
    metadata_output: Texture,
    receiver_output: Texture,
    background_output: Texture,
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
        let current_background = temporal_test_texture_sized(
            device,
            "temporal_test_current_background",
            TextureFormat::Rgba32Float,
            sampled_float,
            size,
        );
        let previous_background = temporal_test_texture_sized(
            device,
            "temporal_test_previous_background",
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
        let background_output = temporal_test_texture_sized(
            device,
            "temporal_test_background_output",
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
            current_background,
            previous_background,
            motion,
            raster_receiver,
            gbuffer,
            irradiance_output,
            metadata_output,
            receiver_output,
            background_output,
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
    let current_background = view(&textures.current_background);
    let previous_background = view(&textures.previous_background);
    let background_output = view(&textures.background_output);
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
            BindGroupEntry {
                binding: 38,
                resource: BindingResource::TextureView(&background_output),
            },
            BindGroupEntry {
                binding: 39,
                resource: BindingResource::TextureView(&current_background),
            },
            BindGroupEntry {
                binding: 40,
                resource: BindingResource::TextureView(&previous_background),
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
            (&textures.background_output, &textures.previous_background),
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
        source_history: UVec4::ZERO,
        sampling: UVec4::ZERO,
        owner_sequence: UVec4::ZERO,
        cache_time: UVec4::ZERO,
        emissive_dirty_min: Vec4::ZERO,
        emissive_dirty_max: Vec4::ZERO,
        light_dirty_min: Vec4::ZERO,
        light_dirty_max: Vec4::ZERO,
        ray_traced: UVec4::ZERO,
    }
}

fn reference_candidate_permute(mut value: u32, mask: u32, seed: u32) -> u32 {
    value = (value ^ seed) & mask;
    value = (value ^ (value >> 16)) & mask;
    value = value.wrapping_mul(0x7feb_352d) & mask;
    value = (value ^ (value >> 15)) & mask;
    value = value.wrapping_mul(0x846c_a68b) & mask;
    (value ^ (value >> 16)) & mask
}

fn reference_inverse_xor_shift_right(value: u32, shift: u32) -> u32 {
    let mut result = value;
    let mut offset = shift;
    while offset < 32 {
        result ^= value >> offset;
        offset += shift;
    }
    result
}

fn reference_candidate_inverse_permute(value: u32, mask: u32, seed: u32) -> u32 {
    let value = reference_inverse_xor_shift_right(value, 16) & mask;
    let value = value.wrapping_mul(0x4302_1123) & mask;
    let value = reference_inverse_xor_shift_right(value, 15) & mask;
    let value = value.wrapping_mul(0x1d69_e2a5) & mask;
    (reference_inverse_xor_shift_right(value, 16) ^ seed) & mask
}

fn reference_candidate_rank(sample: u32, width: u32, height: u32, count: u32) -> u32 {
    if count <= 1 {
        return 0;
    }
    let mut mask = 0;
    while mask < count - 1 {
        mask = (mask << 1) | 1;
    }
    let seed = (width.wrapping_mul(0x9e37_79b9) ^ height.wrapping_mul(0x85eb_ca6b) ^ count) & mask;
    let mut rank = reference_candidate_permute(sample, mask, seed);
    while rank >= count {
        rank = reference_candidate_permute(rank, mask, seed);
    }
    rank
}

fn reference_cold_candidate_rank(
    sample: u32,
    width: u32,
    height: u32,
    count: u32,
    frame: u32,
) -> u32 {
    if count <= 1 {
        return 0;
    }
    let mut mask = 0;
    while mask < count - 1 {
        mask = (mask << 1) | 1;
    }
    let seed = (width.wrapping_mul(0x9e37_79b9)
        ^ height.wrapping_mul(0x85eb_ca6b)
        ^ count
        ^ frame.wrapping_mul(0x9e37_79b9))
        & mask;
    let mut rank = reference_candidate_permute(sample, mask, seed);
    while rank >= count {
        rank = reference_candidate_permute(rank, mask, seed);
    }
    rank
}

fn reference_candidate_from_rank(rank: u32, width: u32, height: u32, count: u32) -> u32 {
    if count <= 1 {
        return 0;
    }
    let mut mask = 0;
    while mask < count - 1 {
        mask = (mask << 1) | 1;
    }
    let seed = (width.wrapping_mul(0x9e37_79b9) ^ height.wrapping_mul(0x85eb_ca6b) ^ count) & mask;
    let mut sample = reference_candidate_inverse_permute(rank, mask, seed);
    while sample >= count {
        sample = reference_candidate_inverse_permute(sample, mask, seed);
    }
    sample
}

fn reference_candidate_selected(
    sample: u32,
    width: u32,
    height: u32,
    count: u32,
    cursor: u32,
    quota: u32,
) -> bool {
    if count == 0 || quota == 0 {
        return false;
    }
    let rank = reference_candidate_rank(sample, width, height, count);
    let cursor = cursor % count;
    let distance = if rank >= cursor {
        rank - cursor
    } else {
        count - cursor + rank
    };
    distance < quota
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
    parameters.source_history.y = 7;
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
        parameters.source_history.y = 7;
        let result =
            temporal_test_result_full(device, queue, &pipeline, &changed, parameters, true)[0];
        assert!(
            result[0][..3].iter().all(|value| value.abs() < 1e-6),
            "old lighting survived incomplete refresh: {result:?}"
        );
        assert!((result[1][2] - expected_length).abs() < 1e-5);
        assert_eq!(support_mask(result[2]), expected_support);
        parameters.stability.y = 0.0;
        parameters.source_history.y = 0;
        write_rgba32f_pixel(queue, &changed.current, [0.0, 0.0, 0.0, -1.0]);
        let next = temporal_test_result(device, queue, &pipeline, &changed, parameters);
        assert!(
            next[0][..3].iter().all(|value| value.abs() < 1e-6),
            "invalidated lighting returned on the next frame: {next:?}"
        );
        assert_eq!(support_mask(next[2]), expected_support);
    }
    // Repeated small sun rotations keep validated support and blend current
    // observations instead of restarting. Once motion stops, fresh black
    // observations converge; an abrupt extinction still invalidates at once.
    let sun = TemporalTestTextures::new(device, queue);
    write_rgba32f_pixel(queue, &sun.current, [0.0, 0.0, 0.0, 1.25]);
    write_rgba32f_pixel(queue, &sun.current_background, [0.0, 0.0, 0.0, 1.25]);
    let mut sun_params = temporal_test_parameters(30);
    sun_params.interval_history.y = 1.0;
    sun_params.source_history.y = 1;
    sun_params.source_history.z = 3;
    let mut last = 1.0;
    for frame in 30..50 {
        sun_params.frame.x = frame;
        sun_params.source_history.w = frame;
        let result = temporal_test_result_full(device, queue, &pipeline, &sun, sun_params, true)[0];
        assert_eq!(support_mask(result[2]), 1);
        assert!(result[0][0] < last && result[0][0] > 0.0, "{result:?}");
        last = result[0][0];
    }
    sun_params.source_history.y = 0;
    sun_params.source_history.z = 2;
    for frame in 50..90 {
        sun_params.frame.x = frame;
        let result = temporal_test_result_full(device, queue, &pipeline, &sun, sun_params, true)[0];
        last = result[0][0];
    }
    assert!(last < 0.05, "stopped sun did not converge: {last}");
    let extinct = TemporalTestTextures::new(device, queue);
    write_rgba32f_pixel(queue, &extinct.current, [0.0, 0.0, 0.0, 1.25]);
    sun_params.source_history.y = 1;
    sun_params.source_history.z = 2;
    let result = temporal_test_result(device, queue, &pipeline, &extinct, sun_params);
    assert!(result[0][..3].iter().all(|value| value.abs() < 1e-6));
    let unsampled = TemporalTestTextures::new(device, queue);
    write_rgba32f_pixel(queue, &unsampled.current, [0.0, 0.0, 0.0, -1.0]);
    sun_params.source_history.y = 0;
    sun_params.source_history.z = 2;
    sun_params.source_history.w = 49;
    sun_params.frame.x = 170;
    let expired = temporal_test_result(device, queue, &pipeline, &unsampled, sun_params);
    assert_eq!(
        support_mask(expired[2]),
        0,
        "unsampled old sun survived: {expired:?}"
    );
    assert_eq!(expired[0][3], 0.0);

    // Moving punctual lights invalidate their old radiance, not the separate
    // background. Incomplete punctual coverage may display the valid background
    // while raw support remains incomplete.
    for (alpha, current, expected) in [(1.25, 12.0, 21.0), (-1.0, 10.0, 19.0)] {
        let moving = TemporalTestTextures::new(device, queue);
        write_rgba32f_pixel(queue, &moving.current, [current, current, current, alpha]);
        write_rgba32f_pixel(queue, &moving.current_background, [10.0, 10.0, 10.0, 1.25]);
        write_rgba32f_pixel(queue, &moving.previous_background, [20.0, 20.0, 20.0, 16.0]);
        let mut parameters = temporal_test_parameters(30);
        parameters.interval_history.y = 1.0;
        parameters.source_history.y = 4;
        let result = temporal_test_result(device, queue, &pipeline, &moving, parameters);
        assert!(
            result[0][..3]
                .iter()
                .all(|value| (*value - expected).abs() < 1e-4),
            "unchanged background was not retained independently: {result:?}"
        );
        assert_eq!(support_mask(result[2]), 1);

        parameters.source_history.y = 5;
        let changed = temporal_test_result(device, queue, &pipeline, &moving, parameters);
        assert!(
            changed[0][..3]
                .iter()
                .all(|value| (*value - current).abs() < 1e-4),
            "changed background must not retain old light: {changed:?}"
        );
        parameters.source_history.y = 4;
        parameters.dirty_min = Vec4::new(10.0, 10.0, 10.0, 1.0);
        parameters.dirty_max = Vec4::new(11.0, 11.0, 11.0, 1.0);
        let geometry = temporal_test_result(device, queue, &pipeline, &moving, parameters);
        assert!(
            geometry[0][..3]
                .iter()
                .all(|value| (*value - current).abs() < 1e-4),
            "dirty geometry must not retain untracked background paths: {geometry:?}"
        );
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
    let environment_rotation = Quat::from_rotation_y(0.73);
    base_parameters.environment_rotation = Vec4::from_array(environment_rotation.to_array());
    base_parameters.environment_scale = Vec4::new(1.0, 0.0, 0.0, 0.0);
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
    let output_bytes = 60 * 16;
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
        let expected_pdf =
            expected_environment_pdf(&table, environment_rotation * direction) as f32;
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
    assert_eq!(output_vec4(&bytes, 26), [1.0, 1.0, 1.0, 0.0]);
    assert_eq!(output_vec4(&bytes, 27), [0.0; 4]);
    let polar_radiance = output_vec4(&bytes, 7);
    assert!(polar_radiance[0] > 20.0);
    assert!((polar_radiance[1] / polar_radiance[0] - 0.6).abs() < 1e-3);

    for index in 0..8 {
        for start in [28, 44] {
            let direction_pdf = output_vec4(&bytes, start + index * 2);
            let radiance = output_vec4(&bytes, start + index * 2 + 1);
            let direction = Vec3::new(direction_pdf[0], direction_pdf[1], direction_pdf[2]);
            let local_direction = environment_rotation * direction;
            let expected_pdf = expected_environment_pdf(&table, local_direction) as f32;
            let expected_radiance = expected_hdr_test_radiance(local_direction);
            assert!((direction_pdf[3] - expected_pdf).abs() <= expected_pdf * 2e-4 + 1e-6);
            for (actual, expected) in radiance[..3].iter().zip(expected_radiance.to_array()) {
                assert!((*actual - expected).abs() <= 2e-4 * expected.max(1.0));
            }
            if start == 28 {
                let original = output_vec4(&bytes, 7 + index * 2);
                assert!((radiance[3] - direction_pdf[3]).abs() < 1e-4);
                for component in 0..3 {
                    assert!((radiance[component] - original[component]).abs() < 1e-4);
                }
            }
        }
    }
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
        case_parameters.environment_scale = Vec4::new(1.0, 0.0, 0.0, 0.0);
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
    const ENTRY_WORDS: usize = 272 / 4;
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
[[vk::binding(10, 0)]] [format("rgba32f")] WTexture2D<float4> irradiance;
[[vk::binding(13, 0)]] RWStructuredBuffer<uint4> testResults;
"#,
    );
    let split = include_str!("../shaders/split.slang");
    for source in [
        section(split, "float splitIntervalEnd(", "uint probeCountHead("),
        section(split, "uint probeCountHead(", "uint probeDescriptorIndex("),
        section(split, "uint probeDescriptorIndex(", "uint findProbeNode("),
        section(split, "uint findProbeNode(", "uint insertProbeDescriptor("),
        section(split, "uint insertProbeDescriptor(", "uint findEntry("),
        section(
            split,
            "uint permuteCandidateDomain(",
            "bool coldPriorityEnabled(",
        ),
        section(split, "bool coldPriorityEnabled(", "// Secondary seeds"),
        section(split, "void rankCandidate(", "// Every visible point seeds"),
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
            heads[rootRayOffsetHead(params)].load(), probeDirectoryCapacity(params), 0u, 0u);
    if (id.x < params.size.z) {
        uint base = 2u * count + 6u;
        testResults[base + id.x] = uint4(
            heads[traceSamplesHead(params) + id.x].load(), 0u, 0u, 0u);
        if (id.x == 0u)
            testResults[base + params.size.z] = uint4(
                heads[traceCountHead(params)].load(),
                heads[traceArgsHead(params)].load(),
                heads[traceArgsHead(params) + 1u].load(),
                heads[traceArgsHead(params) + 2u].load());
    }
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

    let head_words =
        (2 * CACHE_BUCKETS + 36 + 8 * directory_capacity + 1 + 4 + CANDIDATES) as usize;
    let count_base = (2 * CACHE_BUCKETS + 36 + 4 * directory_capacity) as usize;
    let mut initial_heads = vec![u32::MAX; head_words];
    initial_heads[count_base..].fill(0);
    let root = (2 * CACHE_BUCKETS + 36 + 8 * directory_capacity) as usize;
    initial_heads[root + 3] = 1;
    initial_heads[root + 4] = 1;
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
    let result_rows = descriptor_count * 2 + 7 + CANDIDATES as usize;
    let results = device.create_buffer(&BufferDescriptor {
        label: Some("split_r2_prefix_results"),
        size: (result_rows * 16) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let hit_texture = temporal_test_texture_sized(
        device,
        "split_prefix_secondary_hits",
        TextureFormat::Rgba32Float,
        TextureUsages::STORAGE_BINDING,
        UVec2::new(CANDIDATES, 1),
    );
    let hit_view = hit_texture.create_view(&TextureViewDescriptor::default());

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
                      _has_root_offset: bool,
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
        if std::ptr::eq(pipeline, &directions) {
            bindings.push(BindGroupEntry {
                binding: 10,
                resource: BindingResource::TextureView(&hit_view),
            });
        }
        if has_results {
            bindings.push(BindGroupEntry {
                binding: 13,
                resource: results.as_entire_binding(),
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
    let direction_group = make_group(&directions, &direction_parameters, true, true, false, false);
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
    let list_base = descriptor_count * 2 + 6;
    let list_header = output_u32x4(&result_bytes_read, list_base + CANDIDATES as usize);
    // Compact traces space small budgets over more lanes (`traceLaneStride`).
    let lane_stride = (8192 / SELECTED_BUDGET).clamp(1, 8);
    assert_eq!(
        list_header,
        [
            selected_count,
            (selected_count - 1) * lane_stride / 64 + 1,
            1,
            1
        ]
    );
    let mut compact: Vec<_> = (0..selected_count as usize)
        .map(|slot| output_u32x4(&result_bytes_read, list_base + slot)[0])
        .collect();
    compact.sort_unstable();
    assert_eq!(compact, (0..selected_count).collect::<Vec<_>>());
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
fn candidate_inverse_permutation_round_trips_irregular_counts() {
    for count in [1, 3, 5, 7, 9, 17, 31, 33, 63, 65, 127, 129] {
        let width = count;
        let height = 1;
        let mut visited = vec![false; count as usize];
        for rank in 0..count {
            let sample = reference_candidate_from_rank(rank, width, height, count);
            assert!(
                !visited[sample as usize],
                "count {count} repeated sample {sample}"
            );
            visited[sample as usize] = true;
            assert_eq!(
                reference_candidate_rank(sample, width, height, count),
                rank,
                "count {count} did not round-trip rank {rank}"
            );
        }
        assert!(visited.into_iter().all(|sample| sample), "count {count}");
    }
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
    const DIRECTORY_CAPACITY: u32 = 8;
    const ENTRY_WORDS: usize = 272 / 4;
    const ENTRY_COUNT: usize = 65;
    let key = [3i32, -2, 1, 0];
    let empty_key = [7i32, 4, -3, 0];
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
    let directory_slot = hash_key(key, DIRECTORY_CAPACITY) as usize;
    let empty_directory_slot = hash_key(empty_key, DIRECTORY_CAPACITY) as usize;
    assert_ne!(directory_slot, empty_directory_slot);

    let bin_direction = |direction: usize| {
        let x = direction % 8;
        let y = direction / 8;
        let phi = (x as f32 + 0.5) * std::f32::consts::TAU / 8.0;
        let vertical = (y as f32 + 0.5) * 0.5 - 1.0;
        let radius = (1.0 - vertical * vertical).max(0.0).sqrt();
        Vec3::new(radius * phi.cos(), vertical, radius * phi.sin())
    };
    let decode_normal = |uv: Vec2| {
        let p = uv * 2.0 - Vec2::ONE;
        let mut normal = Vec3::new(p.x, p.y, 1.0 - p.x.abs() - p.y.abs());
        let fold = (-normal.z).clamp(0.0, 1.0);
        normal.x += if normal.x >= 0.0 { -fold } else { fold };
        normal.y += if normal.y >= 0.0 { -fold } else { fold };
        normal.normalize()
    };
    let raw_bins: [[f32; 4]; 32] = std::array::from_fn(|direction| {
        let support = match direction % 4 {
            0 => 0.0,
            1 => 0.25,
            2 => 0.5,
            _ => 1.0,
        };
        let color = if direction == 5 {
            [0.0; 3]
        } else {
            [
                0.5 + direction as f32,
                (31 - direction) as f32 * 0.25,
                (direction % 7) as f32 + 0.125,
            ]
        };
        [color[0], color[1], color[2], support]
    });
    let prefilter_table = |raw_bins: &[[f32; 4]; 32]| -> [[f32; 4]; 36] {
        std::array::from_fn(|texel| {
            let x = texel % 6;
            let y = texel / 6;
            let normal = decode_normal(Vec2::new((x as f32 + 0.5) / 6.0, (y as f32 + 0.5) / 6.0));
            let mut sum = [0.0; 3];
            let mut nominal = 0.0;
            let mut observed = 0.0;
            for (direction, sample) in raw_bins.iter().enumerate() {
                let cosine = normal.dot(bin_direction(direction)).max(0.0);
                nominal += cosine;
                if cosine <= 0.0 || sample[3] <= 0.0 {
                    continue;
                }
                let weight = cosine * sample[3].clamp(0.0, 1.0);
                for (sum, component) in sum.iter_mut().zip(&sample[..3]) {
                    *sum += component * weight;
                }
                observed += weight;
            }
            if observed <= 1e-6 || nominal <= 1e-6 {
                [0.0; 4]
            } else {
                let scale = nominal * (4.0 / 32.0) / observed;
                [
                    sum[0] * scale,
                    sum[1] * scale,
                    sum[2] * scale,
                    (observed / nominal).clamp(0.0, 1.0),
                ]
            }
        })
    };
    let expected_table = prefilter_table(&raw_bins);
    let emissive_bins: [[f32; 4]; 32] = std::array::from_fn(|direction| {
        [
            2.0 + direction as f32 * 0.25,
            3.0,
            4.0 - direction as f32 * 0.125,
            1.0,
        ]
    });
    let punctual_bins: [[f32; 4]; 32] = std::array::from_fn(|direction| {
        [
            8.0,
            6.0 + direction as f32 * 0.1,
            4.0,
            if direction >= 24 && direction % 2 == 0 {
                0.75
            } else {
                0.0
            },
        ]
    });
    let expected_emissive_table = prefilter_table(&emissive_bins);
    let expected_punctual_table = prefilter_table(&punctual_bins);

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
    let pack_shader = section(
        production,
        "groupshared float4 packProbeBackground[32];",
        "void commitCache(",
    );
    let pack_shader = &pack_shader[..pack_shader
        .rfind("[shader")
        .expect("next production shader entry")];
    for source in [
        section(
            production,
            "uint transportCacheBucket(",
            "float splitIntervalEnd(",
        ),
        section(
            production,
            "uint probeCountHead(",
            "uint probeDescriptorIndex(",
        ),
        section(
            production,
            "uint findProbeNodeSlot(",
            "uint insertProbeDescriptor(",
        ),
        section(production, "uint findEntry(", "uint findIrradianceProbe("),
        section(production, "uint currentCapacity()", "[shader"),
        section(
            production,
            "void seedProbes(",
            "groupshared float4 packProbeBackground[32];",
        ),
        pack_shader,
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
                entry_points: ["seedProbeStage", "packProbes"]
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
    let pack = make_pipeline("packProbes");

    let mut parameters = temporal_test_parameters(0);
    parameters.size = UVec4::new(1, 1, 2, CACHE_BUCKETS);
    parameters.frame.x = 0;
    parameters.source_history.x = 6;
    let mut parameters = UniformBuffer::from(parameters);
    parameters.write_buffer(device, queue);

    let mut heads_words =
        vec![u32::MAX; (2 * CACHE_BUCKETS + 36 + 8 * DIRECTORY_CAPACITY) as usize];
    heads_words[(2 * CACHE_BUCKETS + 9) as usize] = 0;
    let mut entry_words = vec![0u32; ENTRY_COUNT * ENTRY_WORDS];
    for index in 0..ENTRY_COUNT {
        entry_words[index * ENTRY_WORDS + 16] = u32::MAX;
    }
    for (probe, probe_key) in [key, empty_key].into_iter().enumerate() {
        for (direction, source_radiance) in raw_bins.iter().copied().enumerate() {
            let index = match (probe, direction) {
                (0, 3) => 0,
                (1, 0) => 4,
                // The first probe's canonical descriptor is a traced nonzero direction.
                (0, 0) => 64,
                (0, _) => direction + 1,
                _ => direction + 32,
            };
            let base = index * ENTRY_WORDS;
            let mut direction_key = probe_key;
            direction_key[3] = direction as i32;
            for (word, value) in direction_key.iter().enumerate() {
                entry_words[base + word] = *value as u32;
            }
            let radiance = if probe == 0 {
                source_radiance
            } else {
                [9.0, 8.0, 7.0, 0.0]
            };
            for (component, value) in radiance.iter().enumerate() {
                entry_words[base + 8 + component] = value.to_bits();
            }
            let emissive = if probe == 0 {
                emissive_bins[direction]
            } else {
                [9.0, 8.0, 7.0, 0.0]
            };
            let punctual = if probe == 0 {
                punctual_bins[direction]
            } else {
                [9.0, 8.0, 7.0, 0.0]
            };
            for (component, value) in emissive.iter().enumerate() {
                entry_words[base + 60 + component] = value.to_bits();
            }
            for (component, value) in punctual.iter().enumerate() {
                entry_words[base + 64 + component] = value.to_bits();
            }
            if probe == 0 && direction == 3 {
                entry_words[base + 17] = 0x8000_0000; // traced canonical
            } else if probe == 1 && direction == 0 {
                entry_words[base + 23] = 1; // visible initialized candidate, even with no ray
            }
            if probe == 0 && (radiance[3] > 0.0 || emissive[3] > 0.0 || punctual[3] > 0.0) {
                entry_words[base + 44] = (direction as f32).to_bits();
                entry_words[base + 45] = (-2.0f32).to_bits();
                entry_words[base + 46] = 1.0f32.to_bits();
                entry_words[base + 47] = 1.0f32.to_bits();
                entry_words[base + 48] = (direction as f32 + 1.0).to_bits();
                entry_words[base + 49] = 2.0f32.to_bits();
                entry_words[base + 50] = 3.0f32.to_bits();
                entry_words[base + 51] = 1.0f32.to_bits();
            }
            let bucket = hash_key(direction_key, CACHE_BUCKETS) as usize;
            entry_words[base + 16] = heads_words[bucket];
            heads_words[bucket] = index as u32;
        }
    }
    let heads_bytes: Vec<u8> = heads_words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    let heads = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("split_visible_zero_probe_heads"),
        contents: &heads_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    });
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
    let directional_bytes = 2 * DIRECTORY_CAPACITY as usize * crate::render::PROBE_SIZE as usize;
    let untouched = (-123.0f32).to_le_bytes();
    let directional_probes = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("split_visible_zero_probe_radiance"),
        contents: &untouched.repeat(directional_bytes / untouched.len()),
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
        8,
    );
    // Groups 2 and 3 point at noncanonical entries for the first probe;
    // groups 4 through 7 are padding. Only the two seeded probes may pack.
    let packed_heads = crate::scene::traversal_tests::dispatch_readback(
        device,
        queue,
        &pack,
        &pack_group,
        &heads,
        heads_bytes.len() as u64,
        8,
    );
    let counter = (2 * CACHE_BUCKETS + 9) as usize * 4;
    assert_eq!(
        u32::from_le_bytes(packed_heads[counter..counter + 4].try_into().unwrap()),
        4,
        "only two canonical probes should pack on each dispatch"
    );
    let probe_words = crate::render::PROBE_SIZE as usize / 16;
    let first = directory_slot * probe_words;
    for (offset, table) in [
        (0, &expected_table),
        (36, &expected_emissive_table),
        (72, &expected_punctual_table),
    ] {
        for (texel, expected) in table.iter().enumerate() {
            let actual = output_vec4(&packed, first + offset + texel);
            for (actual, expected) in actual.iter().zip(expected) {
                assert!(
                    (actual - expected).abs() <= expected.abs().max(1.0) * 2e-4,
                    "packed source texel {offset}/{texel}: {actual:?} != {expected:?}"
                );
            }
        }
    }
    assert_eq!(
        output_vec4(&packed, first + 108),
        [0.0, -2.0, 1.0, 1.0],
        "supported source bins retain their aggregate dependency minimum"
    );
    assert_eq!(
        output_vec4(&packed, first + 109),
        [32.0, 2.0, 3.0, 1.0],
        "supported source bins retain their aggregate dependency maximum"
    );
    for slot in 0..2 * DIRECTORY_CAPACITY as usize {
        if slot == directory_slot || slot == empty_directory_slot {
            continue;
        }
        for word in 0..probe_words {
            assert_eq!(
                output_vec4(&packed, slot * probe_words + word),
                [-123.0; 4],
                "noncanonical or padded group wrote probe slot {slot}, word {word}"
            );
        }
    }
    let empty = empty_directory_slot * probe_words;
    for texel in 0..36 {
        assert_eq!(
            output_vec4(&packed, empty + texel),
            [0.0; 4],
            "zero-support probe must pack no irradiance at texel {texel}"
        );
    }
}

#[test]
#[ignore = "requires slangc and a desktop GPU; dispatches production Split rank phases"]
fn gpu_cold_priority_nominates_each_probe_then_fills_fair_budget() {
    fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        let start = source.find(start).expect("production shader section start");
        let tail = &source[start..];
        &tail[..tail.find(end).expect("production shader section end")]
    }

    const CANDIDATES: u32 = 8;
    const WIDTH: u32 = 8;
    const HEIGHT: u32 = 1;
    const BUCKETS: u32 = 16;
    const DIRECTORY_CAPACITY: usize = 32;
    const ENTRY_WORDS: usize = 272 / 4;
    const DESCRIPTOR_COUNT: usize = CANDIDATES as usize * 2 * 4;
    const BUDGET: u32 = 6;
    const CURSOR: u32 = 2;
    const COLD_FRAME: u32 = 128;
    let results_words = (3 * CANDIDATES as usize + 5) * 4;
    let results_bytes = (results_words * 4) as u64;
    let production = include_str!("../shaders/split.slang");
    let rank_source = section(
        production,
        "// Deduplicate immutable per-pixel descriptors",
        "void parents(",
    );
    let rank_source = &rank_source[..rank_source.rfind("[shader").expect("next shader entry")];
    let mut shader_source = String::from(
        r#"
#include "common.slang"
[[vk::binding(0, 0)]] ConstantBuffer<Parameters> params;
[[vk::binding(6, 0)]] RWStructuredBuffer<Atomic<uint>> heads;
[[vk::binding(7, 0)]] RWStructuredBuffer<Entry> entries;
[[vk::binding(13, 0)]] RWStructuredBuffer<uint4> results;
"#,
    );
    for source in [
        section(
            production,
            "uint permuteCandidateDomain(",
            "bool coldPriorityEnabled(",
        ),
        section(
            production,
            "bool coldPriorityEnabled(",
            "// Secondary seeds",
        ),
        section(production, "uint probeCountHead(", "uint findProbeNode("),
        section(production, "uint findProbeNodeSlot(", "uint findEntry("),
        rank_source,
    ] {
        shader_source.push_str(source);
        shader_source.push('\n');
    }
    shader_source.push_str(
        r#"
[shader("compute")]
[numthreads(64, 1, 1)]
void readRankState(uint3 id : SV_DispatchThreadID) {
    uint count = params.size.z;
    if (id.x < count) {
        Entry low = entries[probeDescriptorIndex(id.x, 0u, 0u)];
        Entry high = entries[probeDescriptorIndex(id.x, 1u, 0u)];
        results[id.x] = uint4(low.padding.x, high.padding.x,
            low.lastSample, high.lastSample);
        results[count + id.x] = uint4(transportCandidateSelected(id.x) ? 1u : 0u,
            0u, 0u, 0u);
        uint variant = id.x % 2u;
        uint descriptor = probeDescriptorIndex(id.x, variant, 0u);
        uint slot = findProbeNodeSlot(entries[descriptor].key);
        uint population = slot == EMPTY ? 0u
            : heads[probeCountHead(1u, slot, params)].load();
        results[2u * count + 5u + id.x] = uint4(population, slot, descriptor, 0u);
    }
    if (id.x < 4u) {
        uint group = id.x / 2u;
        uint variant = id.x % 2u;
        uint sample = group * 4u + variant;
        if (params.frame.z == 1u) {
            sample = id.x == 0u ? 0u : id.x == 1u ? 1u : id.x == 2u ? 5u : 7u;
            variant = sample % 2u;
        }
        uint descriptor = probeDescriptorIndex(sample, variant, 0u);
        uint slot = findProbeNodeSlot(entries[descriptor].key);
        uint rays = slot == EMPTY ? 0u
            : heads[probeCountHead(0u, slot, params)].load();
        uint nomineeIndex = EMPTY;
        if (coldPriorityEnabled(params) && params.frame.y == 0u && rays != 0u) {
            uint nomineeRank = 0xffffffffu - rays;
            uint nomineeSample = coldCandidateFromRank(nomineeRank);
            nomineeIndex = probeDescriptorIndex(nomineeSample, 0u, 0u);
            if (entries[nomineeIndex].valid == 0u
                || any(entries[nomineeIndex].key != entries[descriptor].key))
                nomineeIndex = probeDescriptorIndex(nomineeSample, 1u, 0u);
            rays = nomineeIndex + 1u;
        }
        results[2u * count + id.x] = uint4(rays, slot, descriptor, nomineeIndex);
    }
    if (id.x == 0u)
        results[2u * count + 4u] = uint4(heads[2u * params.size.w + 7u].load(), 0u, 0u, 0u);
}
"#,
    );
    let shader = SlangCompiler::default()
        .compile_bundle(
            "split_cold_priority_rank_check.slang",
            &[
                ("common.slang", include_str!("../shaders/common.slang")),
                ("split_cold_priority_rank_check.slang", &shader_source),
            ],
            &SlangSettings {
                target: SlangTarget::Wgsl,
                entry_points: ["rank", "readRankState"]
                    .into_iter()
                    .map(str::to_owned)
                    .collect(),
                optimization: Some(2),
                ..default()
            },
        )
        .unwrap();
    let bevy::shader::Source::Wgsl(source) = shader.source else {
        panic!("expected WGSL from the Split cold-priority regression shader");
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
        label: Some("split_cold_priority_rank_regression"),
        source: ShaderSource::Wgsl(source),
    });
    let pipeline = |entry| {
        device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some(entry),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: default(),
            cache: None,
        })
    };
    let rank = pipeline("rank");
    let read_state = pipeline("readRankState");
    let result = device.create_buffer(&BufferDescriptor {
        label: Some("split_cold_priority_rank_results"),
        size: results_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let make_entries = |fair_mask: Option<&[bool]>, weighted: bool| {
        let mut words = vec![0u32; DESCRIPTOR_COUNT * ENTRY_WORDS];
        for sample in 0..CANDIDATES {
            for variant in 0..2u32 {
                for level in 0..4u32 {
                    let index = ((variant * CANDIDATES + sample) * 4 + level) as usize;
                    let base = index * ENTRY_WORDS;
                    let group = if !weighted {
                        sample / 4
                    } else if variant == 0 {
                        0
                    } else {
                        match sample {
                            1 | 3 => 0,
                            5 => 1,
                            7 => 2,
                            _ => 3,
                        }
                    };
                    words[base] = group >> level;
                    words[base + 1] = 0;
                    words[base + 2] = 0;
                    words[base + 3] = (variant << 18) | (level << 12);
                    words[base + 16] = u32::MAX;
                    words[base + 17] = 1;
                    let selected_lod = sample % 2;
                    words[base + 22] = if level == 0
                        && variant == selected_lod
                        && fair_mask.is_none_or(|selected| selected[sample as usize])
                    {
                        1
                    } else {
                        0
                    };
                    words[base + 23] = 1;
                }
            }
        }
        words
    };
    let make_state = |fair_mask: Option<&[bool]>, selected_count: u32, weighted: bool| {
        let mut heads_words =
            vec![
                u32::MAX;
                2 * BUCKETS as usize + 36 + 8 * DIRECTORY_CAPACITY + 5 + CANDIDATES as usize
            ];
        let count_base = 2 * BUCKETS as usize + 36 + 4 * DIRECTORY_CAPACITY;
        for count in 0..4 * DIRECTORY_CAPACITY {
            heads_words[count_base + count] = 0;
        }
        let root = 2 * BUCKETS as usize + 36 + 8 * DIRECTORY_CAPACITY;
        heads_words[root..].fill(0);
        heads_words[root + 3] = 1;
        heads_words[root + 4] = 1;
        heads_words[2 * BUCKETS as usize + 7] = selected_count;
        let heads = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("split_cold_priority_heads"),
            contents: &heads_words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });
        let entry_words = make_entries(fair_mask, weighted);
        let entry_bytes: Vec<_> = entry_words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        let entries = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("split_cold_priority_entries"),
            contents: &entry_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });
        (heads, entries)
    };
    let make_parameters = |history_valid: bool, history_frames: u32, phase: u32| {
        let mut parameters = temporal_test_parameters(history_frames);
        parameters.size = UVec4::new(WIDTH, HEIGHT, CANDIDATES, BUCKETS);
        parameters.history.z = if history_valid { 1.0 } else { 0.0 };
        parameters.interval_history.y = 1.0;
        parameters.sampling.x = BUDGET;
        parameters.owner_sequence.z = BUDGET;
        parameters.owner_sequence.w = CURSOR;
        parameters.frame.y = phase;
        parameters
    };
    let make_groups = |parameters: TransportParameters, heads: &Buffer, entries: &Buffer| {
        let mut uniform = UniformBuffer::from(parameters);
        uniform.write_buffer(device, queue);
        let rank_group = device.create_bind_group(
            "split_cold_priority_rank",
            &rank.get_bind_group_layout(0).into(),
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: uniform.binding().unwrap(),
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
        let state_group = device.create_bind_group(
            "split_cold_priority_state",
            &read_state.get_bind_group_layout(0).into(),
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: uniform.binding().unwrap(),
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
                    resource: result.as_entire_binding(),
                },
            ],
        );
        (uniform, rank_group, state_group)
    };
    let read_state = |group: &BindGroup| {
        let bytes = crate::scene::traversal_tests::dispatch_readback(
            device,
            queue,
            &read_state,
            group,
            &result,
            results_bytes,
            1,
        );
        (0..3 * CANDIDATES as usize + 5)
            .map(|index| output_u32x4(&bytes, index))
            .collect::<Vec<_>>()
    };
    let fair_selected: Vec<bool> = (0..CANDIDATES)
        .map(|sample| {
            reference_candidate_selected(sample, WIDTH, HEIGHT, CANDIDATES, CURSOR, BUDGET)
        })
        .collect();

    let (cold_heads, cold_entries) = make_state(None, 0, false);
    let mut cold_uniforms = Vec::new();
    let mut cold_rank_groups = Vec::new();
    let mut cold_state_groups = Vec::new();
    for phase in 0..4 {
        let mut parameters = make_parameters(false, 24, phase);
        parameters.frame.x = COLD_FRAME;
        let (uniform, rank_group, state_group) =
            make_groups(parameters, &cold_heads, &cold_entries);
        cold_uniforms.push(uniform);
        cold_rank_groups.push(rank_group);
        cold_state_groups.push(state_group);
    }
    let head_bytes = cold_heads.size();
    let mut nominees = [0u32; 4];
    for phase in 0..4 {
        let _ = crate::scene::traversal_tests::dispatch_readback(
            device,
            queue,
            &rank,
            &cold_rank_groups[phase],
            &cold_heads,
            head_bytes,
            1,
        );
        let state = read_state(&cold_state_groups[phase]);
        let keys = &state[2 * CANDIDATES as usize..2 * CANDIDATES as usize + 4];
        let selected_counter = state[2 * CANDIDATES as usize + 4][0];
        if phase == 0 {
            assert_eq!(
                selected_counter, 0,
                "cold nominations do not consume the ray budget"
            );
            for (key, record) in keys.iter().enumerate() {
                let group = key / 2;
                let variant = key % 2;
                let first_sample = (group * 4 + variant) as u32;
                let second_sample = first_sample + 2;
                let first_index = ((variant as u32 * CANDIDATES + first_sample) * 4) as usize;
                let second_index = ((variant as u32 * CANDIDATES + second_sample) * 4) as usize;
                let expected_index = if reference_cold_candidate_rank(
                    first_sample,
                    WIDTH,
                    HEIGHT,
                    CANDIDATES,
                    COLD_FRAME,
                ) < reference_cold_candidate_rank(
                    second_sample,
                    WIDTH,
                    HEIGHT,
                    CANDIDATES,
                    COLD_FRAME,
                ) {
                    first_index
                } else {
                    second_index
                };
                assert_eq!(
                    record[3] as usize, expected_index,
                    "probe {key} must choose the frame-seeded lowest-rank eligible descriptor"
                );
                assert_eq!(record[0] as usize, expected_index + 1);
            }
            nominees = std::array::from_fn(|key| keys[key][3]);

            // Make the canonical owner and nominee unambiguous for the next
            // phase. Key zero's canonical descriptor is index 0, but descriptor
            // 8 is another eligible receiver for the same probe. This catches
            // confusing its nominated value 9 with the committed ray count 1.
            for (key, record) in keys.iter().enumerate() {
                let group = key / 2;
                let variant = key % 2;
                let owner_sample = (group * 4) as u32;
                let owner_index = (variant as u32 * CANDIDATES + owner_sample) * 4;
                let directory_head = 2 * BUCKETS as usize + 36 + record[1] as usize;
                queue.write_buffer(
                    &cold_heads,
                    (directory_head * 4) as u64,
                    &owner_index.to_le_bytes(),
                );
            }
            let key_zero_count_head =
                2 * BUCKETS as usize + 36 + 4 * DIRECTORY_CAPACITY + keys[0][1] as usize;
            let nominee_priority =
                u32::MAX - reference_cold_candidate_rank(2, WIDTH, HEIGHT, CANDIDATES, COLD_FRAME);
            queue.write_buffer(
                &cold_heads,
                (key_zero_count_head * 4) as u64,
                &nominee_priority.to_le_bytes(),
            );
            nominees[0] = 8;
        } else if phase == 1 {
            assert_eq!(
                selected_counter, 0,
                "building the population histogram does not reserve updates"
            );
        } else if phase == 2 {
            assert_eq!(
                selected_counter, 4,
                "one nominee is admitted per base probe"
            );
            assert_eq!(keys.iter().map(|record| record[0]).sum::<u32>(), 4);
            let cold_markers = state[..CANDIDATES as usize]
                .iter()
                .map(|record| (record[0] == 2) as usize + (record[1] == 2) as usize)
                .sum::<usize>();
            assert_eq!(cold_markers, 4, "canonical owners admit each nominee once");
            assert_eq!(
                state[2][0], 2,
                "non-owner descriptor 8 is admitted by owner 0"
            );
            assert_eq!(
                state[0][0], 1,
                "owner 0 stays eligible after nominating descriptor 8"
            );
            for nominee in nominees {
                let sample = (nominee as usize / 4) % CANDIDATES as usize;
                let variant = nominee as usize / (CANDIDATES as usize * 4);
                let record = state[sample];
                assert_eq!(record[variant], 2, "phase 2 marks only the nominee cold");
            }
        } else if phase == 3 {
            assert_eq!(
                selected_counter, BUDGET,
                "fair fill reaches the traversal budget"
            );
            assert_eq!(keys.iter().map(|record| record[0]).sum::<u32>(), BUDGET);
            let mut total_selected = 0;
            for sample in 0..CANDIDATES as usize {
                let record = state[sample];
                assert!(record[0] <= 1 && record[1] <= 1);
                assert!(
                    record[0] + record[1] <= 1,
                    "sample {sample} traces one LOD only"
                );
                let chosen = if record[0] == 1 {
                    Some((0, record[2]))
                } else if record[1] == 1 {
                    Some((1, record[3]))
                } else {
                    None
                };
                if let Some((variant, local_ordinal)) = chosen {
                    total_selected += 1;
                    let key = (sample / 4) * 2 + variant;
                    let nominee = nominees[key];
                    let descriptor = ((variant * CANDIDATES as usize + sample) * 4) as u32;
                    if descriptor != nominee {
                        assert!(
                            fair_selected[sample],
                            "non-nominee sample {sample} must be fair-selected"
                        );
                    }
                    assert!(
                        local_ordinal < keys[key][0],
                        "local ray ordinal stays within its probe"
                    );
                }
            }
            assert_eq!(total_selected, BUDGET as usize);
            for (key, probe) in keys.iter().enumerate() {
                let group = key / 2;
                let variant = key % 2;
                let mut ordinals = Vec::new();
                for sample in [group * 4 + variant, group * 4 + variant + 2] {
                    let record = state[sample];
                    let (selected, ordinal) = if variant == 0 {
                        (record[0], record[2])
                    } else {
                        (record[1], record[3])
                    };
                    if selected == 1 {
                        ordinals.push(ordinal);
                    }
                }
                ordinals.sort_unstable();
                assert_eq!(ordinals, (0..probe[0]).collect::<Vec<_>>());
            }
        }
    }

    // Warm or history-disabled frames retain the original fair selection exactly.
    for (history_valid, history_frames) in [(true, 24), (false, 0)] {
        let (warm_heads, warm_entries) = make_state(Some(&fair_selected), BUDGET, false);
        let (warm_uniform, warm_rank_group, warm_state_group) = make_groups(
            make_parameters(history_valid, history_frames, 0),
            &warm_heads,
            &warm_entries,
        );
        let _ = crate::scene::traversal_tests::dispatch_readback(
            device,
            queue,
            &rank,
            &warm_rank_group,
            &warm_heads,
            warm_heads.size(),
            1,
        );
        let state = read_state(&warm_state_group);
        let records = &state[..CANDIDATES as usize];
        let mut selected_count = 0;
        for sample in 0..CANDIDATES as usize {
            let expected_low = fair_selected[sample] && sample % 2 == 0;
            let expected_high = fair_selected[sample] && sample % 2 == 1;
            assert_eq!(records[sample][0], if expected_low { 1 } else { 0 });
            assert_eq!(records[sample][1], if expected_high { 1 } else { 0 });
            assert_eq!(
                state[CANDIDATES as usize + sample][0],
                if fair_selected[sample] { 1 } else { 0 }
            );
            selected_count += records[sample][0] + records[sample][1];
        }
        assert_eq!(selected_count, BUDGET);
        assert_eq!(state[2 * CANDIDATES as usize + 4][0], BUDGET);
        let _ = &warm_uniform;
    }
    // A cold budget follows projected candidate population before fair fill:
    // dense visible probes outrank the sparse probes they cover less often.
    let (weighted_heads, weighted_entries) = make_state(None, 0, true);
    for phase in 0..3 {
        let mut parameters = make_parameters(false, 24, phase);
        parameters.sampling.x = 2;
        parameters.owner_sequence.z = 2;
        parameters.frame.x = COLD_FRAME;
        parameters.frame.z = 1;
        let (uniform, rank_group, state_group) =
            make_groups(parameters, &weighted_heads, &weighted_entries);
        let _ = crate::scene::traversal_tests::dispatch_readback(
            device,
            queue,
            &rank,
            &rank_group,
            &weighted_heads,
            weighted_heads.size(),
            1,
        );
        let state = read_state(&state_group);
        let selected_counter = state[2 * CANDIDATES as usize + 4][0];
        let populations = &state[2 * CANDIDATES as usize + 5..3 * CANDIDATES as usize + 5];
        assert_eq!(
            populations
                .iter()
                .map(|record| record[0])
                .collect::<Vec<_>>(),
            vec![4u32, 2, 4, 2, 4, 1, 4, 1],
            "source populations count selected-LOD candidates per probe"
        );
        if phase < 2 {
            assert_eq!(selected_counter, 0, "cold ranking phases spend no rays");
        } else {
            let keys = &state[2 * CANDIDATES as usize..2 * CANDIDATES as usize + 4];
            assert_eq!(
                keys.iter().map(|record| record[0]).collect::<Vec<_>>(),
                vec![1u32, 1, 0, 0],
                "the fixed budget admits populations 4 and 2 before population 1"
            );
            assert_eq!(selected_counter, 2);
        }
        let _ = uniform;
    }

    // When unique cold probes outnumber the update budget, admission must
    // remain bounded even though every probe has a nominated descriptor.
    let (limited_heads, limited_entries) = make_state(None, 0, false);
    for phase in 0..4 {
        let mut parameters = make_parameters(false, 24, phase);
        parameters.sampling.x = 2;
        parameters.owner_sequence.z = 2;
        parameters.frame.x = COLD_FRAME;
        let (uniform, rank_group, state_group) =
            make_groups(parameters, &limited_heads, &limited_entries);
        let _ = crate::scene::traversal_tests::dispatch_readback(
            device,
            queue,
            &rank,
            &rank_group,
            &limited_heads,
            limited_heads.size(),
            1,
        );
        let state = read_state(&state_group);
        let selected_counter = state[2 * CANDIDATES as usize + 4][0];
        if phase < 2 {
            assert_eq!(
                selected_counter, 0,
                "cold preparation does not spend the budget"
            );
            let _ = uniform;
            continue;
        }
        let keys = &state[2 * CANDIDATES as usize..2 * CANDIDATES as usize + 4];
        assert_eq!(
            selected_counter, 2,
            "cold admission and fair fill must not exceed the update budget"
        );
        assert_eq!(
            keys.iter().map(|record| record[0]).sum::<u32>(),
            2,
            "only two of the four nominated probes may receive a ray"
        );
        let admitted_marker = if phase == 2 { 2 } else { 1 };
        let admitted = state[..CANDIDATES as usize]
            .iter()
            .map(|record| {
                (record[0] == admitted_marker) as usize + (record[1] == admitted_marker) as usize
            })
            .sum::<usize>();
        assert_eq!(
            admitted, 2,
            "each reserved update belongs to exactly one cold nominee"
        );
        let _ = uniform;
    }

    let _ = (&cold_uniforms, cold_rank_groups, cold_state_groups);
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
    const ENTRY_WORDS: usize = 272 / 4;
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
            "uint probeCountHead(",
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
    TransportRadiance radiance = (TransportRadiance)0;
    radiance.background = float3(4, 2, 1);
    deposit(0u, float3(0.5, 0.5, 0.5), float3(0.0),
        float3(0, 0, 1), hit, radiance, 0u,
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

#[test]
#[ignore = "requires slangc and a desktop GPU"]
fn gpu_compact_direction_group_boundaries() {
    const WIDTH: u32 = 13;
    const COUNT: u32 = 65;
    const BUCKETS: u32 = 16;
    const ENTRY_WORDS: usize = 272 / 4;
    let split = include_str!("../shaders/split.slang");
    let section = |start: &str, end: &str| {
        let tail = &split[split.find(start).unwrap()..];
        &tail[..tail.find(end).unwrap()]
    };
    let source = format!(
        r#"#include "common.slang"
[[vk::binding(0, 0)]] ConstantBuffer<Parameters> params;
[[vk::binding(6, 0)]] RWStructuredBuffer<Atomic<uint>> heads;
[[vk::binding(7, 0)]] RWStructuredBuffer<Entry> entries;
[[vk::binding(10, 0)]] [format("rgba32f")] WTexture2D<float4> irradiance;
{}
{}"#,
        section("uint probeDescriptorIndex(", "uint findProbeNode("),
        section("void generateDirections(", "uint currentCapacity("),
    );
    // The extraction starts at the function, so supply its compute entry attributes.
    let source = source.replace(
        "void generateDirections(",
        "[shader(\"compute\")]\n[numthreads(8, 8, 1)]\nvoid generateDirections(",
    );
    let shader = SlangCompiler::default()
        .compile_bundle(
            "compact_directions_test.slang",
            &[
                ("common.slang", include_str!("../shaders/common.slang")),
                ("compact_directions_test.slang", &source),
            ],
            &SlangSettings {
                target: SlangTarget::Wgsl,
                entry_points: vec!["generateDirections".into()],
                optimization: Some(2),
                ..default()
            },
        )
        .unwrap();
    let bevy::shader::Source::Wgsl(source) = shader.source else {
        panic!("expected WGSL")
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
        label: Some("compact_boundaries"),
        source: ShaderSource::Wgsl(source),
    });
    let pipeline = device.create_compute_pipeline(&RawComputePipelineDescriptor {
        label: Some("compact_boundaries"),
        layout: None,
        module: &module,
        entry_point: Some("generateDirections"),
        compilation_options: default(),
        cache: None,
    });
    let mut parameters = temporal_test_parameters(0);
    parameters.size = UVec4::new(WIDTH, 5, COUNT, BUCKETS);
    let mut uniform = UniformBuffer::from(parameters);
    uniform.write_buffer(device, queue);
    let texture = temporal_test_texture_sized(
        device,
        "compact_hits",
        TextureFormat::Rgba32Float,
        TextureUsages::STORAGE_BINDING,
        UVec2::new(WIDTH, 5),
    );
    let view = texture.create_view(&TextureViewDescriptor::default());
    let root = (2 * BUCKETS + 36 + 8 * (COUNT * 4).next_power_of_two()) as usize;
    for selected in [0u32, 1, 63, 64, 65] {
        let mut words = vec![0u32; root + 5 + COUNT as usize];
        words[root + 3] = 1;
        words[root + 4] = 1;
        let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        let heads = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("compact_heads"),
            contents: &bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let mut descriptors = vec![0u32; COUNT as usize * 8 * ENTRY_WORDS];
        for sample in 0..COUNT {
            for variant in 0..2 {
                let index = ((variant * COUNT + sample) * 4) as usize;
                let chosen = sample < selected && (sample % 3 == 0 || variant == sample % 2);
                // A marked but invalid descriptor must never nominate a sample.
                let invalid = sample >= selected && variant == 0;
                descriptors[index * ENTRY_WORDS + 16] =
                    if invalid { u32::MAX } else { index as u32 };
                descriptors[index * ENTRY_WORDS + 22] = u32::from(chosen || invalid);
            }
        }
        let bytes: Vec<_> = descriptors
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        let entries = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("compact_entries"),
            contents: &bytes,
            usage: BufferUsages::STORAGE,
        });
        let group = device.create_bind_group(
            "compact_boundaries",
            &pipeline.get_bind_group_layout(0).into(),
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: uniform.binding().unwrap(),
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
                    binding: 10,
                    resource: BindingResource::TextureView(&view),
                },
            ],
        );
        let output = crate::scene::traversal_tests::dispatch_readback(
            device,
            queue,
            &pipeline,
            &group,
            &heads,
            (words.len() * 4) as u64,
            WIDTH.div_ceil(8),
        );
        let word =
            |index: usize| u32::from_le_bytes(output[index * 4..index * 4 + 4].try_into().unwrap());
        assert_eq!(word(root + 1), selected);
        assert_eq!(
            [word(root + 2), word(root + 3), word(root + 4)],
            [selected.div_ceil(64), 1, 1]
        );
        let mut ids: Vec<_> = (0..selected as usize).map(|i| word(root + 5 + i)).collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..selected).collect::<Vec<_>>());
    }
}

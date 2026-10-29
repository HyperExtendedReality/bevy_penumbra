//! GPU regressions for distance LODs, separate from the four radiance cascades.
use super::*;

const LOD_ROWS: usize = 6;
const MAX_TEST_LOD: u32 = 24;

fn with_lod_gpu(
    check: impl FnOnce(
        &bevy::render::renderer::RenderDevice,
        &bevy::render::renderer::RenderQueue,
        &ComputePipeline,
    ),
) {
    fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        let source = &source[source.find(start).expect("production LOD helper")..];
        &source[..source.find(end).expect("production LOD helper end")]
    }
    let split = include_str!("../shaders/split.slang");
    let source = format!(
        r#"
#include "common.slang"
[[vk::binding(0, 0)]] ConstantBuffer<Parameters> params;
[[vk::binding(1, 0)]] StructuredBuffer<float4> positions;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint4> results;
{lod}
{interval}
[shader("compute")]
[numthreads(1, 1, 1)]
void checkLod(uint3 thread : SV_DispatchThreadID) {{
    uint low, high;
    float blend;
    splitLodBlend(positions[thread.x].xyz, low, high, blend);
    uint output = thread.x * 6u;
    results[output] = asuint(float4(float(low), float(high), blend,
        spacing(params, 0u, low)));
    results[output + 1u] = asuint(float4(spacing(params, 3u, low),
        splitIntervalEnd(0u, low), splitIntervalEnd(3u, low), float(CASCADES)));
    float3 negative = float3(-1.25, -2.5, -0.125) * spacing(params, 3u, low);
    int4 key = cacheKey(negative, 2047u, 3u, params, low);
    results[output + 2u] = uint4(key);
    results[output + 3u] = uint4((uint(key.w) & PROBE_LOD) >> 18u,
        (uint(key.w) >> 12u) & 3u, uint(key.w) & 4095u,
        (uint(cacheKey(negative, 2047u, 3u, params, low + 4u).w) & PROBE_LOD) >> 18u);
    results[output + 4u] = asuint(float4(spacing(params, 3u, 0xffffffffu),
        splitIntervalEnd(3u, 0xffffffffu), spacing(params, 0u, high),
        splitIntervalEnd(0u, high)));
    results[output + 5u] = asuint(float4(splitIntervalStart(1u, low),
        splitIntervalEnd(0u, low), splitIntervalStart(3u, low),
        splitIntervalEnd(2u, low)));
}}
"#,
        lod = section(split, "void splitLodBlend(", "uint transportCacheBucket("),
        interval = section(split, "float splitIntervalEnd(", "uint probeCountHead("),
    );
    let shader = SlangCompiler::default()
        .compile_bundle(
            "lod_check.slang",
            &[
                ("common.slang", include_str!("../shaders/common.slang")),
                ("lod_check.slang", &source),
            ],
            &SlangSettings {
                target: SlangTarget::Wgsl,
                entry_points: vec!["checkLod".to_owned()],
                optimization: Some(2),
                ..default()
            },
        )
        .unwrap();
    let bevy::shader::Source::Wgsl(source) = shader.source else {
        panic!("expected WGSL for production LOD helpers");
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
        label: Some("split_lod_check"),
        source: ShaderSource::Wgsl(source),
    });
    let pipeline = device.create_compute_pipeline(&RawComputePipelineDescriptor {
        label: Some("split_lod_check"),
        layout: None,
        module: &module,
        entry_point: Some("checkLod"),
        compilation_options: default(),
        cache: None,
    });
    check(device, queue, &pipeline);
}

fn run_lod_gpu(
    device: &bevy::render::renderer::RenderDevice,
    queue: &bevy::render::renderer::RenderQueue,
    pipeline: &ComputePipeline,
    parameters: TransportParameters,
    positions: &[Vec3],
) -> Vec<u8> {
    let mut uniform = UniformBuffer::from(parameters);
    uniform.write_buffer(device, queue);
    let bytes: Vec<u8> = positions
        .iter()
        .flat_map(|position| {
            position
                .extend(1.0)
                .to_array()
                .into_iter()
                .flat_map(f32::to_le_bytes)
        })
        .collect();
    let input = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("split_lod_positions"),
        contents: &bytes,
        usage: BufferUsages::STORAGE,
    });
    let output_bytes = (positions.len() * LOD_ROWS * 16) as u64;
    let output = device.create_buffer(&BufferDescriptor {
        label: Some("split_lod_results"),
        size: output_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let group = device.create_bind_group(
        "split_lod_check",
        &pipeline.get_bind_group_layout(0).into(),
        &[
            BindGroupEntry {
                binding: 0,
                resource: uniform.binding().unwrap(),
            },
            BindGroupEntry {
                binding: 1,
                resource: input.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: output.as_entire_binding(),
            },
        ],
    );
    crate::scene::traversal_tests::dispatch_readback(
        device,
        queue,
        pipeline,
        &group,
        &output,
        output_bytes,
        positions.len() as u32,
    )
}

fn lod_parameters(projection: Mat4, camera: Vec3, target: Vec3) -> TransportParameters {
    let mut parameters = temporal_test_parameters(0);
    let view_from_world = bevy::prelude::Transform::from_translation(camera)
        .looking_at(target, Vec3::Y)
        .to_matrix()
        .inverse();
    parameters.clip_from_world = projection * view_from_world;
    parameters.world_from_clip = parameters.clip_from_world.inverse();
    parameters.camera_position = camera.extend(1.0);
    parameters.viewport = UVec4::new(13, 29, 512, 512);
    parameters.tuning = Vec4::new(1.0, 1.6, 0.0, 1.0);
    parameters.owner_sequence.x = 8;
    parameters
}

fn close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() <= expected.abs().max(1.0) * 2e-5,
        "actual {actual}, expected {expected}"
    );
}

#[test]
#[ignore = "requires slangc and a desktop GPU; dispatches production Split LOD helpers"]
fn gpu_split_lod_grows_beyond_four_cascades_and_preserves_keys() {
    with_lod_gpu(|device, queue, pipeline| {
        let projection =
            bevy::math::proj::perspective_infinite_reverse(std::f32::consts::FRAC_PI_2, 1.0, 0.1);
        let camera = Vec3::new(3.0, -5.0, 7.0);
        let parameters = lod_parameters(projection, camera, camera - Vec3::Z);
        // B = focalPixels * baseSpacing / targetPixels = 256 / 8.
        let base = 32.0;
        let positions: Vec<_> = [4.0, 8.0, 16.0, 32.0, 2.0f32.powi(30)]
            .map(|radius| camera + Vec3::new(radius * base, 0.0, 0.0))
            .into();
        let output = run_lod_gpu(device, queue, pipeline, parameters, &positions);
        for (sample, lod) in [2u32, 3, 4, 5, MAX_TEST_LOD].into_iter().enumerate() {
            let row = sample * LOD_ROWS;
            let selected = output_vec4(&output, row);
            close(selected[0], lod as f32);
            close(selected[1], lod as f32);
            close(selected[2], 0.0);
            close(selected[3], 2.0f32.powi(lod as i32));
            let scale = output_vec4(&output, row + 1);
            close(scale[0], 2.0f32.powi(lod as i32 + 3));
            close(scale[1], 1.6 * 2.0f32.powi(lod as i32));
            close(scale[2], 1.6 * 2.0f32.powi(lod as i32 + 6));
            close(scale[3], 4.0);
            let key = output_u32x4(&output, row + 2);
            assert_eq!([key[0] as i32, key[1] as i32, key[2] as i32], [-2, -3, -1]);
            assert_eq!(
                output_u32x4(&output, row + 3),
                [lod, 3, 2047, (lod + 4).min(MAX_TEST_LOD)]
            );
            let upper = output_vec4(&output, row + 4);
            assert!(upper.into_iter().all(f32::is_finite));
            close(upper[0], 2.0f32.powi(MAX_TEST_LOD as i32 + 3));
            close(upper[1], 1.6 * 2.0f32.powi(MAX_TEST_LOD as i32 + 6));
            let adjacent = output_vec4(&output, row + 5);
            close(adjacent[0], adjacent[1]);
            close(adjacent[2], adjacent[3]);
        }
        // Chebyshev, not Euclidean distance, determines perspective bands.
        let output = run_lod_gpu(
            device,
            queue,
            pipeline,
            parameters,
            &[camera + Vec3::new(7.0 * base, 7.0 * base, 7.0 * base)],
        );
        assert_eq!(&output_vec4(&output, 0)[..2], &[2.0, 2.0]);
    });
}

#[test]
#[ignore = "requires slangc and a desktop GPU; dispatches production Split LOD helpers"]
fn gpu_split_lod_overlap_is_continuous_beyond_lod_three() {
    with_lod_gpu(|device, queue, pipeline| {
        let projection =
            bevy::math::proj::perspective_infinite_reverse(std::f32::consts::FRAC_PI_2, 1.0, 0.1);
        let parameters = lod_parameters(projection, Vec3::ZERO, -Vec3::Z);
        let mut positions = Vec::new();
        for next_lod in 1..=7 {
            let boundary = 32.0 * 2.0f32.powi(next_lod);
            for fraction in [0.9, 0.95, 0.999999, 1.0, 1.00001] {
                positions.push(Vec3::new(boundary * fraction, 0.0, 0.0));
            }
        }
        let output = run_lod_gpu(device, queue, pipeline, parameters, &positions);
        for next_lod in 1..=7usize {
            let mut previous_scale = 0.0;
            for (step, expected_blend) in [0.0, 0.5, 0.99999, 0.0, 0.0].into_iter().enumerate() {
                let row = ((next_lod - 1) * 5 + step) * LOD_ROWS;
                let selected = output_vec4(&output, row);
                let expected_low = if step < 3 { next_lod - 1 } else { next_lod };
                close(selected[0], expected_low as f32);
                close(selected[2], expected_blend);
                let high_spacing = output_vec4(&output, row + 4)[2];
                let scale = selected[3] * (1.0 - selected[2]) + high_spacing * selected[2];
                assert!(scale >= previous_scale);
                previous_scale = scale;
                if step >= 2 {
                    close(scale, 2.0f32.powi(next_lod as i32));
                }
            }
        }
    });
}

#[test]
#[ignore = "requires slangc and a desktop GPU; dispatches production Split LOD helpers"]
fn gpu_split_lod_orthographic_uses_projected_spacing_not_camera_distance() {
    with_lod_gpu(|device, queue, pipeline| {
        // The 8px target is 36 times the projected base spacing: LOD 5.
        let projection =
            bevy::math::proj::orthographic(-1152.0, 1152.0, -1152.0, 1152.0, 0.1, 100_000.0);
        let target = Vec3::new(-3.0, 2.0, 1.0);
        let direction = Vec3::new(2.0, 1.0, 3.0).normalize();
        let positions = [target, target + Vec3::new(-7.0, 5.0, 12.0)];
        let mut previous = None;
        for distance in [10.0, 1000.0, 20_000.0] {
            let parameters = lod_parameters(projection, target + direction * distance, target);
            let output = run_lod_gpu(device, queue, pipeline, parameters, &positions);
            for sample in 0..positions.len() {
                let selected = output_vec4(&output, sample * LOD_ROWS);
                close(selected[0], 5.0);
                close(selected[1], 5.0);
                close(selected[2], 0.0);
                close(selected[3], 32.0);
            }
            if let Some(previous) = previous.as_ref() {
                assert_eq!(
                    &output, previous,
                    "orthographic camera translation must not change LOD identity or scale"
                );
            }
            previous = Some(output);
        }
        // Zooming changes projected spacing even though distance is unchanged.
        let zoom = bevy::math::proj::orthographic(-288.0, 288.0, -288.0, 288.0, 0.1, 100_000.0);
        let parameters = lod_parameters(zoom, target + direction * 1000.0, target);
        let output = run_lod_gpu(device, queue, pipeline, parameters, &positions);
        close(output_vec4(&output, 0)[0], 3.0);
    });
}

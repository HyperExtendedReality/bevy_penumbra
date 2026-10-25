//! Production-GPU regression for ray-visible emissive meshes.
use super::*;
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only_sized, storage_buffer_sized, texture_2d, texture_2d_array,
    uniform_buffer,
};

const EMITTER_FIXTURE: &str = r#"
#include "cascades.slang"
[[vk::binding(38, 0)]] RWStructuredBuffer<float4> fixtureResults;
[shader("compute")]
[numthreads(64, 1, 1)]
void fixtureEmitter(uint3 thread : SV_DispatchThreadID) {
    uint sample = thread.x;
    // Uniform hemisphere quadrature at the actual diffuse receiver. The
    // receiver's albedo is deliberately absent from this E/pi response.
    float2 uv = r2(sample + 1u);
    float z = uv.x;
    float radius = sqrt(max(0.0, 1.0-z*z));
    float angle = 2.0*PI*uv.y;
    float3 direction = float3(radius*cos(angle), radius*sin(angle), z);
    Hit hit = traceRay(float3(0,0,0.005), direction, 100.0, false, 0.0, 0.0, params.sampling.y);
    bool supported;
    float3 value = outgoingRadiance(float3(0,0,0.005), direction, hit, supported, params.sampling.y);
    fixtureResults[sample] = float4(value*(2.0*z), supported ? 1.0 : 0.0);
}
"#;

#[test]
#[ignore = "requires slangc and a desktop GPU; dispatches production mesh-emitter traversal and outgoing radiance"]
fn gpu_emissive_mesh_first_hits_illuminate_diffuse_receivers() {
    const SAMPLES: usize = 4096;
    let options = WgpuSettings::default();
    let resources = block_on(initialize_renderer(
        options.backends.unwrap_or(Backends::PRIMARY),
        None,
        &options,
    ));
    let (device, queue) = (&resources.0, &resources.1);
    let spirv = resources.2.backend == wgpu::Backend::Vulkan
        && device
            .features()
            .contains(wgpu::Features::PASSTHROUGH_SHADERS);
    let shader = SlangCompiler::default()
        .compile_bundle(
            "emitter_fixture.slang",
            &[
                ("common.slang", include_str!("../shaders/common.slang")),
                (
                    "receiver_metadata.slang",
                    include_str!("../shaders/receiver_metadata.slang"),
                ),
                ("bsdf.slang", include_str!("../shaders/bsdf.slang")),
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
                ("emitter_fixture.slang", EMITTER_FIXTURE),
            ],
            &SlangSettings {
                target: SlangTarget::Wgsl,
                entry_points: vec!["fixtureEmitter".into()],
                optimization: Some(2),
                ..default()
            },
        )
        .unwrap();
    let bevy::shader::Source::Wgsl(source) = shader.source else {
        panic!("expected WGSL");
    };
    let layout_entries = BindGroupLayoutEntries::with_indices(
        ShaderStages::COMPUTE,
        (
            (0, uniform_buffer::<TransportParameters>(false)),
            (1, storage_buffer_read_only_sized(false, None)),
            (2, storage_buffer_read_only_sized(false, None)),
            (3, storage_buffer_read_only_sized(false, None)),
            (6, storage_buffer_sized(false, None)),
            (
                16,
                texture_2d_array(TextureSampleType::Float { filterable: false }),
            ),
            (
                22,
                texture_2d(TextureSampleType::Float { filterable: false }),
            ),
            (28, texture_2d_array(TextureSampleType::Uint)),
            (29, texture_2d_array(TextureSampleType::Uint)),
            (
                37,
                texture_2d(TextureSampleType::Float { filterable: false }),
            ),
            (38, storage_buffer_sized(false, None)),
        ),
    );
    let module = if spirv {
        let (words, workgroup_size) = crate::render::vulkan_compute_source(
            source.as_ref(),
            "fixtureEmitter",
            &layout_entries,
        )
        .unwrap();
        // SAFETY: Naga validates the WGSL and emits SPIR-V using this fixture's exact layout.
        unsafe {
            device.wgpu_device().create_shader_module_passthrough(
                wgpu::ShaderModuleDescriptorPassthrough {
                    label: Some("emitter_fixture"),
                    spirv: Some(std::borrow::Cow::Owned(words)),
                    entry_points: std::borrow::Cow::Owned(vec![
                        wgpu::PassthroughShaderEntryPoint {
                            name: std::borrow::Cow::Borrowed("fixtureEmitter"),
                            workgroup_size,
                        },
                    ]),
                    ..default()
                },
            )
        }
    } else {
        device.create_and_validate_shader_module(ShaderModuleDescriptor {
            label: Some("emitter_fixture"),
            source: ShaderSource::Wgsl(source),
        })
    };
    let layout = device.create_bind_group_layout("emitter_fixture", &layout_entries);
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("emitter_fixture"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&RawComputePipelineDescriptor {
        label: Some("emitter_fixture"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("fixtureEmitter"),
        compilation_options: default(),
        cache: None,
    });
    let bits = |values: [f32; 4]| UVec4::from_array(values.map(f32::to_bits));
    // One facing emitter triangle, plus a separate broad black blocker. All
    // geometry, materials, instances, BVHs, and query admission are production.
    let vertices = [
        [-3.0, -3.0, 2.0, 1.0],
        [0.0, 3.0, 2.0, 1.0],
        [3.0, -3.0, 2.0, 1.0],
        [-100.0, -100.0, 1.0, 1.0],
        [0.0, 100.0, 1.0, 1.0],
        [100.0, -100.0, 1.0, 1.0],
    ];
    let mut geometry_words = vec![
        UVec4::new(1, 3, 5, 6),
        UVec4::new(0, 1, 2, 0),
        UVec4::new(3, 4, 5, 0),
    ];
    for position in vertices {
        geometry_words.extend([
            bits(position),
            bits([0.0, 0.0, -1.0, 0.0]),
            bits([1.0, 0.0, 0.0, 1.0]),
            UVec4::ZERO,
            bits([1.0; 4]),
        ]);
    }
    let mut geometry = StorageBuffer::from(geometry_words);
    geometry.write_buffer(device, queue);
    let sampled = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    let atlas = device.create_texture(&TextureDescriptor {
        label: Some("emitter_fixture_atlas"),
        size: Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Float,
        usage: sampled,
        view_formats: &[],
    });
    let pages = device.create_texture(&TextureDescriptor {
        label: Some("emitter_fixture_pages"),
        format: TextureFormat::Rgba32Uint,
        ..atlas_descriptor()
    });
    let atlas_array_view = atlas.create_view(&TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    let atlas_view = atlas.create_view(&TextureViewDescriptor::default());
    let page_view = pages.create_view(&TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    let output = device.create_buffer(&BufferDescriptor {
        label: Some("emitter_fixture_output"),
        size: (SAMPLES * 16) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let run = |emission: f32, blocked: bool, denied: bool| {
        let mut words = vec![
            UVec4::new(3, 7, 0, 9),
            bits([-100.0, -100.0, 0.9, 0.0]),
            bits([100.0, 100.0, 2.1, 0.0]),
            bits([-3.0, -3.0, 2.0, 0.0]),
            bits([3.0, 3.0, 2.0, 0.0]),
            bits([-100.0, -100.0, 1.0, 0.0]),
            bits([100.0, 100.0, 1.0, 0.0]),
        ];
        words[1].w = 0;
        words[2].w = if blocked { 2 } else { 1 };
        words[3].w = 0;
        words[4].w = 1;
        words[5].w = 1;
        words[6].w = 1;
        for instance in 0..2 {
            for _ in 0..2 {
                words.extend([
                    bits([1.0, 0.0, 0.0, 0.0]),
                    bits([0.0, 1.0, 0.0, 0.0]),
                    bits([0.0, 0.0, 1.0, 0.0]),
                ]);
            }
            words.extend([
                // The instance flag word's w lane stays zero: this is ordinary
                // emissive mesh geometry, with no special emitter classification.
                UVec4::new(41, 1, 0, 0),
                UVec4::new(instance + 1, 0, 0, 25 + instance * 8),
                UVec4::new(instance + 1, instance, 1, 0),
            ]);
        }
        for value in [emission, 0.0] {
            words.extend([
                bits([0.0, 0.0, 0.0, 1.0]),
                bits([value, value, value, -1.0]),
                bits([0.0, 1.0, 0.5, 0.0]),
                UVec4::ZERO,
                bits([1.0, 0.0, 0.0, 0.0]),
                bits([0.0, 1.0, 0.0, 0.0]),
                UVec4::splat(u32::MAX),
                UVec4::new(u32::MAX, 0, 0, 0),
            ]);
        }
        words.push(UVec4::new(1, 0, 0, 0));
        let mut scene = StorageBuffer::from(words);
        scene.write_buffer(device, queue);
        let mut lights = StorageBuffer::from(vec![crate::scene::GiLight {
            source: UVec4::new(1, 0, 0, 0),
            ..default()
        }]);
        lights.write_buffer(device, queue);
        let mut head_words = vec![0u32; 68];
        if denied {
            head_words[38] = 1;
        }
        let heads = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("emitter_fixture_queries"),
            contents: &head_words
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
            usage: BufferUsages::STORAGE,
        });
        let mut parameters = temporal_test_parameters(0);
        parameters.frame = UVec4::new(0, 0, if blocked { 2 } else { 1 }, 0);
        parameters.size = UVec4::new(1, 1, 1, 16);
        parameters.layers = UVec4::new(0, 1, 0, 0);
        parameters.sampling.y = if denied { 1 } else { 0 };
        let mut uniforms = UniformBuffer::from(parameters);
        uniforms.write_buffer(device, queue);
        let group = device.create_bind_group(
            "emitter_fixture",
            &layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: uniforms.binding().unwrap(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: geometry.binding().unwrap(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: scene.binding().unwrap(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: lights.binding().unwrap(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: heads.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 16,
                    resource: BindingResource::TextureView(&atlas_array_view),
                },
                BindGroupEntry {
                    binding: 22,
                    resource: BindingResource::TextureView(&atlas_view),
                },
                BindGroupEntry {
                    binding: 28,
                    resource: BindingResource::TextureView(&page_view),
                },
                BindGroupEntry {
                    binding: 29,
                    resource: BindingResource::TextureView(&page_view),
                },
                BindGroupEntry {
                    binding: 37,
                    resource: BindingResource::TextureView(&atlas_view),
                },
                BindGroupEntry {
                    binding: 38,
                    resource: output.as_entire_binding(),
                },
            ],
        );
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("emitter_fixture_readback"),
            size: (SAMPLES * 16) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&default());
        {
            let mut pass = encoder.begin_compute_pass(&default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((SAMPLES as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, (SAMPLES * 16) as u64);
        let submission = queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback
            .slice(..)
            .map_async(MapMode::Read, move |result| tx.send(result).unwrap());
        device
            .poll(PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(std::time::Duration::from_secs(30)),
            })
            .unwrap();
        rx.recv().unwrap().unwrap();
        let bytes = readback.slice(..).get_mapped_range().unwrap();
        let values = (0..SAMPLES)
            .map(|index| output_vec4(&bytes, index))
            .collect::<Vec<_>>();
        drop(bytes);
        readback.unmap();
        values
    };
    let visible = run(4.0, false, false);
    let energy = visible.iter().map(|sample| sample[0]).sum::<f32>() / SAMPLES as f32;
    assert!(visible.iter().all(|sample| sample[3] == 1.0));
    assert!(
        energy > 0.1,
        "an unclassified emissive mesh must contribute when a primary GI ray hits it: {energy}"
    );
    // Independent area-domain integration of the same emitter checks estimator
    // energy, not just a bright accidental ray or shader wiring.
    let mut reference = 0.0f64;
    const GRID: usize = 256;
    for y in 0..GRID {
        for x in 0..GRID {
            let root = ((x as f64 + 0.5) / GRID as f64).sqrt();
            let v = (y as f64 + 0.5) / GRID as f64;
            let px = -3.0 + 3.0 * root * (1.0 - v) + 6.0 * root * v;
            let py = -3.0 + 6.0 * root * (1.0 - v);
            let z = 1.995f64;
            let distance2 = px * px + py * py + z * z;
            reference += 4.0 * 18.0 * z * z
                / (std::f64::consts::PI * distance2 * distance2)
                / (GRID * GRID) as f64;
        }
    }
    assert!(
        (f64::from(energy) - reference).abs() < reference * 0.04,
        "hemisphere and area estimates disagree: {energy} vs {reference}"
    );
    for samples in [run(4.0, true, false), run(0.0, false, false)] {
        assert!(
            samples.iter().all(
                |sample| sample[3] == 1.0 && sample[..3].iter().all(|value| value.abs() < 1e-6)
            ),
            "blocker and zero emission must produce supported black"
        );
    }
    assert!(
        run(4.0, false, true).iter().all(|sample| sample[3] == 0.0),
        "denied production traversal is unavailable, never a supported black observation"
    );
}

fn atlas_descriptor() -> TextureDescriptor<'static> {
    TextureDescriptor {
        label: Some("emitter_fixture_pages"),
        size: Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Uint,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    }
}

//! Compact dispatch exercises the production trace and interval deposition.
use super::*;
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only_sized, storage_buffer_sized, texture_2d, texture_2d_array,
    texture_storage_2d, uniform_buffer,
};

const COUNT: u32 = 8;
const BUCKETS: u32 = 16;
const WORDS: usize = 68;
const TRACE_ENTRIES: [&str; 5] = [
    "trace",
    "traceDeferred",
    "traceRecord",
    "traceVisibility",
    "traceResolve",
];

#[derive(Clone, Copy)]
enum TraceDispatch {
    Direct,
    Indirect,
    /// The four deferred passes, with heads sized for their scratch.
    Deferred,
}

/// Mirrors the private `deferred_trace_words` in render.rs.
fn deferred_trace_words(samples: u32) -> usize {
    let slots = samples.min(8192) as usize * 2;
    let queries = slots * 4;
    4 + queries + slots * 32 + queries * 12
}

/// Runs the shared triangle fixture once per `(dispatch, cap)` and returns the
/// entries followed by the heads words. `primary` selects the primary cache
/// (depth/G-buffer receivers) instead of secondary seeds.
fn compact_trace_runs(primary: bool, runs: &[(TraceDispatch, u32)]) -> Vec<Vec<u32>> {
    let options = WgpuSettings::default();
    let resources = block_on(initialize_renderer(
        options.backends.unwrap_or(Backends::PRIMARY),
        None,
        &options,
    ));
    let (device, queue) = (&resources.0, &resources.1);
    let shader = SlangCompiler::default()
        .compile_bundle(
            "cascades.slang",
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
            ],
            &SlangSettings {
                target: SlangTarget::Wgsl,
                entry_points: TRACE_ENTRIES.map(Into::into).to_vec(),
                optimization: Some(2),
                ..default()
            },
        )
        .unwrap();
    let bevy::shader::Source::Wgsl(source) = shader.source else {
        panic!("expected WGSL")
    };
    let entries_layout = BindGroupLayoutEntries::with_indices(
        ShaderStages::COMPUTE,
        (
            (0, uniform_buffer::<TransportParameters>(false)),
            (1, storage_buffer_read_only_sized(false, None)),
            (2, storage_buffer_read_only_sized(false, None)),
            (3, storage_buffer_read_only_sized(false, None)),
            (4, texture_2d(TextureSampleType::Depth)),
            (6, storage_buffer_sized(false, None)),
            (7, storage_buffer_sized(false, None)),
            (8, storage_buffer_read_only_sized(false, None)),
            (9, storage_buffer_sized(false, None)),
            (
                10,
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
            ),
            (
                11,
                texture_2d(TextureSampleType::Float { filterable: false }),
            ),
            (12, storage_buffer_sized(false, None)),
            (
                16,
                texture_2d_array(TextureSampleType::Float { filterable: false }),
            ),
            (
                22,
                texture_2d(TextureSampleType::Float { filterable: false }),
            ),
            (26, texture_2d(TextureSampleType::Uint)),
            (27, texture_2d(TextureSampleType::Uint)),
            (28, texture_2d_array(TextureSampleType::Uint)),
            (29, texture_2d_array(TextureSampleType::Uint)),
            (
                37,
                texture_2d(TextureSampleType::Float { filterable: false }),
            ),
        ),
    );
    let passthrough = resources.2.backend == wgpu::Backend::Vulkan
        && device
            .features()
            .contains(wgpu::Features::PASSTHROUGH_SHADERS);
    let wgsl_module = (!passthrough).then(|| {
        device.create_and_validate_shader_module(ShaderModuleDescriptor {
            label: Some("compact_trace"),
            source: ShaderSource::Wgsl(source.clone()),
        })
    });
    let layout = device.create_bind_group_layout("compact_trace", &entries_layout);
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("compact_trace"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipelines = TRACE_ENTRIES.map(|entry| {
        let module = wgsl_module.clone().unwrap_or_else(|| {
            let (words, workgroup_size) =
                crate::render::vulkan_compute_source(source.as_ref(), entry, &entries_layout)
                    .unwrap();
            // SAFETY: Naga validates WGSL and emits SPIR-V using this exact layout.
            unsafe {
                device.wgpu_device().create_shader_module_passthrough(
                    wgpu::ShaderModuleDescriptorPassthrough {
                        label: Some("compact_trace"),
                        spirv: Some(std::borrow::Cow::Owned(words)),
                        entry_points: std::borrow::Cow::Owned(vec![
                            wgpu::PassthroughShaderEntryPoint {
                                name: std::borrow::Cow::Borrowed(entry),
                                workgroup_size,
                            },
                        ]),
                        ..default()
                    },
                )
            }
        });
        device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("compact_trace"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: default(),
            cache: None,
        })
    });
    let bits = |v: [f32; 4]| UVec4::from_array(v.map(f32::to_bits));
    // A broad diffuse triangle receives a directional light at each ray hit.
    let mut geometry_words = vec![UVec4::new(1, 2, 5, 3), UVec4::new(0, 1, 2, 0)];
    for p in [
        [-10000.0, -10000.0, 2.0, 1.0],
        [0.0, 10000.0, 2.0, 1.0],
        [10000.0, -10000.0, 2.0, 1.0],
    ] {
        geometry_words.extend([
            bits(p),
            bits([0.0, 0.0, -1.0, 0.0]),
            bits([1.0, 0.0, 0.0, 1.0]),
            UVec4::ZERO,
            bits([1.0; 4]),
        ]);
    }
    let mut geometry = StorageBuffer::from(geometry_words);
    geometry.write_buffer(device, queue);
    let mut scene_words = vec![
        UVec4::new(2, 5, 0, 9),
        bits([-10000.0, -10000.0, 1.9, 0.0]),
        bits([10000.0, 10000.0, 2.1, 0.0]),
        bits([-10000.0, -10000.0, 1.9, 0.0]),
        bits([10000.0, 10000.0, 2.1, 0.0]),
    ];
    scene_words[1].w = 0;
    scene_words[2].w = 1;
    scene_words[3].w = 0;
    scene_words[4].w = 1;
    for _ in 0..2 {
        scene_words.extend([
            bits([1.0, 0.0, 0.0, 0.0]),
            bits([0.0, 1.0, 0.0, 0.0]),
            bits([0.0, 0.0, 1.0, 0.0]),
        ]);
    }
    scene_words.extend([
        UVec4::new(22, 1, 0, 0),
        UVec4::new(1, 0, 0, 14),
        UVec4::new(1, 0, 1, 0),
    ]);
    scene_words.extend([
        bits([1.0; 4]),
        bits([4.0, 2.0, 1.0, -1.0]),
        bits([0.0, 1.0, 0.5, 0.0]),
        UVec4::ZERO,
        bits([1.0, 0.0, 0.0, 0.0]),
        bits([0.0, 1.0, 0.0, 0.0]),
        UVec4::splat(u32::MAX),
        UVec4::new(u32::MAX, 0, 0, 0),
        UVec4::new(1, 0, 0, 0),
    ]);
    let mut scene = StorageBuffer::from(scene_words);
    scene.write_buffer(device, queue);
    let mut lights = StorageBuffer::from(vec![crate::scene::GiLight {
        position: Vec4::new(0.0, 0.0, -1.0, 0.0),
        color: Vec4::new(4.0, 2.0, 1.0, 0.0),
        source: UVec4::new(1, 0, 0, 0),
        ..default()
    }]);
    lights.write_buffer(device, queue);
    let sampled = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    let float_tex = temporal_test_texture_sized(
        device,
        "compact_trace_float",
        TextureFormat::Rgba32Float,
        sampled,
        UVec2::new(COUNT, 1),
    );
    // Octahedral +Z, encoded in the same packed form as a secondary hit.
    write_rgba32f_row(
        queue,
        &float_tex,
        &vec![[0.0, 0.0, 0.0, (2048 + 2048 * 4096 + 1) as f32]; COUNT as usize],
    );
    let float_view = float_tex.create_view(&TextureViewDescriptor::default());
    let float_array = float_tex.create_view(&TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    let uint_tex = temporal_test_texture_sized(
        device,
        "compact_trace_uint",
        TextureFormat::Rgba32Uint,
        sampled,
        UVec2::ONE,
    );
    let uint_view = uint_tex.create_view(&TextureViewDescriptor::default());
    let uint_array = uint_tex.create_view(&TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    let depth = temporal_test_texture_sized(
        device,
        "compact_trace_depth",
        TextureFormat::Depth32Float,
        TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
        UVec2::ONE,
    );
    let depth_view = depth.create_view(&TextureViewDescriptor::default());
    // Primary receivers sit at (0, 0, 0.5) under the identity camera; the zero
    // G-buffer normal orients toward the camera, i.e. toward the triangle.
    let mut encoder = device.create_command_encoder(&default());
    encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some("compact_trace_depth_clear"),
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
    queue.submit([encoder.finish()]);
    let output_tex = temporal_test_texture_sized(
        device,
        "compact_trace_output",
        TextureFormat::Rgba32Float,
        TextureUsages::STORAGE_BINDING,
        UVec2::new(COUNT, 1),
    );
    let output_view = output_tex.create_view(&TextureViewDescriptor::default());
    let root = (2 * BUCKETS + 36 + 8 * (COUNT * 4).next_power_of_two()) as usize;
    let run = |dispatch: TraceDispatch, cap: u32| {
        let mut params = temporal_test_parameters(0);
        params.size = UVec4::new(COUNT, 1, COUNT, BUCKETS);
        params.frame = UVec4::new(7, 0, 1, u32::from(cap == 0));
        params.environment_scale = Vec4::new(0.0, 0.0, if primary { 0.0 } else { 1.0 }, 1.0);
        params.source_history.x = 2;
        params.sampling.y = cap;
        params.layers = UVec4::new(0, 1, 0, 0);
        let mut uniform = UniformBuffer::from(params);
        uniform.write_buffer(device, queue);
        let samples_end = root + 5 + COUNT as usize;
        let deferred = matches!(dispatch, TraceDispatch::Deferred);
        let scratch = if deferred {
            deferred_trace_words(COUNT)
        } else {
            0
        };
        let mut h = vec![0u32; samples_end + scratch];
        h[..BUCKETS as usize].fill(u32::MAX);
        h[root + 1] = COUNT;
        h[root + 2] = 1;
        h[root + 3] = 1;
        h[root + 4] = 1;
        for i in 0..COUNT as usize {
            h[root + 5 + i] = i as u32;
        }
        if deferred {
            // The clear pass's job: query count 0, args (0, 1, 1); zero slot
            // tags never match since live tags set bit 31.
            h[samples_end + 2] = 1;
            h[samples_end + 3] = 1;
        }
        let heads = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("compact_trace_heads"),
            contents: &h.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let mut data = vec![0u32; COUNT as usize * 8 * WORDS];
        for i in 0..COUNT as usize {
            data[i * 4 * WORDS + 16] = (i * 4) as u32;
            data[i * 4 * WORDS + 22] = 1;
        }
        for i in 0..COUNT as usize {
            data[(COUNT as usize * 4 + i * 4) * WORDS + 16] = u32::MAX;
        }
        let entries = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("compact_trace_entries"),
            contents: &data
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let previous_heads = device.create_buffer(&BufferDescriptor {
            label: Some("compact_trace_previous_heads"),
            size: 4,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let previous_entries = device.create_buffer(&BufferDescriptor {
            label: Some("compact_trace_previous_entries"),
            size: 272,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let probes = device.create_buffer(&BufferDescriptor {
            label: Some("compact_trace_probes"),
            size: crate::render::PROBE_SIZE,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(
            "compact_trace",
            &layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: uniform.binding().unwrap(),
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
                    binding: 4,
                    resource: BindingResource::TextureView(&depth_view),
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
                    binding: 10,
                    resource: BindingResource::TextureView(&output_view),
                },
                BindGroupEntry {
                    binding: 11,
                    resource: BindingResource::TextureView(&float_view),
                },
                BindGroupEntry {
                    binding: 12,
                    resource: probes.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 16,
                    resource: BindingResource::TextureView(&float_array),
                },
                BindGroupEntry {
                    binding: 22,
                    resource: BindingResource::TextureView(&float_view),
                },
                BindGroupEntry {
                    binding: 26,
                    resource: BindingResource::TextureView(&uint_view),
                },
                BindGroupEntry {
                    binding: 27,
                    resource: BindingResource::TextureView(&uint_view),
                },
                BindGroupEntry {
                    binding: 28,
                    resource: BindingResource::TextureView(&uint_array),
                },
                BindGroupEntry {
                    binding: 29,
                    resource: BindingResource::TextureView(&uint_array),
                },
                BindGroupEntry {
                    binding: 37,
                    resource: BindingResource::TextureView(&float_view),
                },
            ],
        );
        let args = device.create_buffer(&BufferDescriptor {
            label: Some("compact_trace_indirect"),
            size: 12,
            usage: BufferUsages::COPY_DST | BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });
        let size = (data.len() * 4) as u64;
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("compact_trace_readback"),
            size: size + (h.len() * 4) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&default());
        if matches!(dispatch, TraceDispatch::Indirect) {
            encoder.copy_buffer_to_buffer(&heads, ((root + 2) * 4) as u64, &args, 0, 12);
        }
        let passes: &[(usize, u32)] = match dispatch {
            TraceDispatch::Direct | TraceDispatch::Indirect => &[(0, COUNT.div_ceil(64))],
            // Ticket passes, then one lane per possible query (stride 1).
            TraceDispatch::Deferred => &[
                (1, COUNT.div_ceil(64)),
                (2, COUNT.div_ceil(64)),
                (3, (COUNT * 2 * 4).div_ceil(64)),
                (4, COUNT.div_ceil(64)),
            ],
        };
        for &(entry, groups) in passes {
            let mut pass = encoder.begin_compute_pass(&default());
            pass.set_pipeline(&pipelines[entry]);
            pass.set_bind_group(0, &group, &[]);
            if matches!(dispatch, TraceDispatch::Indirect) {
                pass.dispatch_workgroups_indirect(&args, 0);
            } else {
                pass.dispatch_workgroups(groups, 1, 1);
            }
        }
        encoder.copy_buffer_to_buffer(&entries, 0, &readback, 0, size);
        encoder.copy_buffer_to_buffer(&heads, 0, &readback, size, (h.len() * 4) as u64);
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
        let words = bytes
            .chunks_exact(4)
            .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
            .collect::<Vec<_>>();
        drop(bytes);
        readback.unmap();
        words
    };
    runs.iter()
        .map(|&(dispatch, cap)| run(dispatch, cap))
        .collect()
}

/// Same deposited transport, and every selected ray deposits lit radiance.
fn assert_same_transport(direct: &[u32], other: &[u32]) {
    for sample in 0..COUNT as usize {
        for level in 0..4 {
            let base = (sample * 4 + level) * WORDS;
            assert_eq!(
                direct[base + 17],
                other[base + 17],
                "sample {sample} level {level} word 17"
            );
            assert_eq!(
                direct[base + 19],
                other[base + 19],
                "sample {sample} level {level} word 19"
            );
            for component in (4..8).chain(36..60) {
                let a = f32::from_bits(direct[base + component]);
                let b = f32::from_bits(other[base + component]);
                assert!(
                    (a - b).abs() <= 1e-5 + 1e-4 * a.abs().max(b.abs()),
                    "sample {sample} level {level} word {component}: {a} vs {b}"
                );
            }
        }
        assert_eq!(
            direct[sample * 4 * WORDS + 19],
            1,
            "selected ray must deposit"
        );
        assert!(
            (0..4).any(|level| f32::from_bits(direct[(sample * 4 + level) * WORDS + 4]) > 0.0),
            "real lit hit must deposit radiance"
        );
    }
}

#[test]
#[ignore = "requires slangc and a desktop GPU; dispatches production compact transport"]
fn gpu_compact_trace_direct_and_indirect_preserve_transport() {
    use TraceDispatch::*;
    let runs = compact_trace_runs(
        false,
        &[(Direct, 0), (Indirect, 0), (Direct, 1), (Indirect, 1)],
    );
    assert_same_transport(&runs[0], &runs[1]);
    for saturated in &runs[2..] {
        let offset = COUNT as usize * 8 * WORDS;
        assert_eq!(saturated[offset + 2 * BUCKETS as usize + 6], 1);
        let supported = (0..COUNT as usize)
            .filter(|sample| saturated[sample * 4 * WORDS + 19] == 1)
            .count();
        for sample in 0..COUNT as usize {
            let base = sample * 4 * WORDS;
            if saturated[base + 19] == 1 {
                assert_eq!(
                    f32::from_bits(saturated[base + 39]),
                    1.0,
                    "admitted observation must retain a known dependency"
                );
            } else {
                assert_eq!(
                    saturated[base + 17],
                    0,
                    "denied observation must remain invalid"
                );
            }
        }
        assert_eq!(
            supported, 1,
            "denied queries must not become supported black"
        );
    }
}

#[test]
#[ignore = "requires slangc and a desktop GPU; dispatches production compact transport"]
fn gpu_deferred_trace_matches_immediate_trace() {
    use TraceDispatch::*;
    // The deferred scratch exists only for the primary cache.
    let runs = compact_trace_runs(true, &[(Direct, 0), (Deferred, 0)]);
    assert_same_transport(&runs[0], &runs[1]);
    let root = (2 * BUCKETS + 36 + 8 * (COUNT * 4).next_power_of_two()) as usize;
    let query_count = runs[1][COUNT as usize * 8 * WORDS + root + 5 + COUNT as usize];
    assert!(
        query_count > 0,
        "deferred shading must record visibility queries"
    );
}

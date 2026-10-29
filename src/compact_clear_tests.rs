//! GPU regression for compact direction scratch and secondary-hit clearing.
use super::*;

#[test]
#[ignore = "requires slangc and a desktop GPU; dispatches production Split clear and direction generation"]
fn gpu_compact_direction_clears_primary_hits_and_isolates_scratch() {
    const WIDTH: u32 = 5;
    const HEIGHT: u32 = 3;
    const COUNT: u32 = WIDTH * HEIGHT;
    const BUCKETS: u32 = 16;
    const ENTRY_WORDS: usize = 272 / 4;
    const ROW_PITCH: u32 = 256;
    let root = (2 * BUCKETS + 36 + 8 * (COUNT * 4).next_power_of_two()) as usize;
    let head_bytes = (root + 5 + COUNT as usize) * 4;
    let split = include_str!("../shaders/split.slang");
    let section = |start: &str, end: &str| {
        let tail = &split[split.find(start).expect("production section start")..];
        &tail[..tail.find(end).expect("production section end")]
    };
    let source = format!(
        r#"#include "common.slang"
[[vk::binding(0, 0)]] ConstantBuffer<Parameters> params;
[[vk::binding(6, 0)]] RWStructuredBuffer<Atomic<uint>> heads;
[[vk::binding(7, 0)]] RWStructuredBuffer<Entry> entries;
[[vk::binding(10, 0)]] [format("rgba32f")] WTexture2D<float4> irradiance;
{}
{}
{}
[shader("compute")]
[numthreads(64, 1, 1)]
void clearStage(uint3 id : SV_DispatchThreadID) {{ clear(id); }}
[shader("compute")]
[numthreads(8, 8, 1)]
void directionsStage(uint3 id : SV_DispatchThreadID) {{ generateDirections(id); }}
"#,
        section("uint probeCountHead(", "uint probeDescriptorIndex("),
        section("uint probeDescriptorIndex(", "uint findProbeNode("),
        format!(
            "{}\n{}",
            section("void clear(", "uint permuteCandidateDomain("),
            section("void generateDirections(", "uint currentCapacity(")
        ),
    );
    let shader = SlangCompiler::default()
        .compile_bundle(
            "compact_clear_test.slang",
            &[
                ("common.slang", include_str!("../shaders/common.slang")),
                ("compact_clear_test.slang", &source),
            ],
            &SlangSettings {
                target: SlangTarget::Wgsl,
                entry_points: vec!["clearStage".into(), "directionsStage".into()],
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
        label: Some("compact_clear"),
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
    let clear = pipeline("clearStage");
    let directions = pipeline("directionsStage");
    let mut primary_params = temporal_test_parameters(0);
    primary_params.size = UVec4::new(WIDTH, HEIGHT, COUNT, BUCKETS);
    primary_params.owner_sequence.z = 1;
    primary_params.environment_scale = Vec4::new(0.0, 1.0, 0.0, 0.0);
    let mut secondary_params = primary_params;
    secondary_params.environment_scale.z = 1.0;
    let mut second_clear_params = primary_params;
    second_clear_params.owner_sequence.z = 0;
    let uniform = |params| {
        let mut buffer = UniformBuffer::from(params);
        buffer.write_buffer(device, queue);
        buffer
    };
    let primary_uniform = uniform(primary_params);
    let secondary_uniform = uniform(secondary_params);
    let second_clear_uniform = uniform(second_clear_params);
    let heads = || {
        device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("compact_clear_heads"),
            contents: &vec![0u8; head_bytes],
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        })
    };
    let primary_heads = heads();
    let secondary_heads = heads();
    let entries = |selected: &[u32]| {
        let mut words = vec![0u32; COUNT as usize * 8 * ENTRY_WORDS];
        for sample in 0..COUNT {
            for variant in 0..2 {
                let index = ((variant * COUNT + sample) * 4) as usize;
                words[index * ENTRY_WORDS + 16] = index as u32;
                words[index * ENTRY_WORDS + 17] = 1;
                words[index * ENTRY_WORDS + 22] =
                    u32::from(selected.contains(&sample) && variant == 0);
            }
        }
        let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("compact_clear_entries"),
            contents: &bytes,
            usage: BufferUsages::STORAGE,
        })
    };
    let primary_entries = entries(&[0, 6, 14]);
    let secondary_entries = entries(&[2, 8]);
    let empty_entries = entries(&[]);
    let texture = temporal_test_texture_sized(
        device,
        "compact_hit_image",
        TextureFormat::Rgba32Float,
        TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC | TextureUsages::COPY_DST,
        UVec2::new(WIDTH, HEIGHT),
    );
    let view = texture.create_view(&TextureViewDescriptor::default());
    let group = |pipeline: &ComputePipeline,
                 uniform: &UniformBuffer<TransportParameters>,
                 heads: &Buffer,
                 entries: &Buffer,
                 is_clear: bool| {
        let mut bindings = vec![
            BindGroupEntry {
                binding: 0,
                resource: uniform.binding().unwrap(),
            },
            BindGroupEntry {
                binding: 6,
                resource: heads.as_entire_binding(),
            },
        ];
        if !is_clear {
            bindings.push(BindGroupEntry {
                binding: 7,
                resource: entries.as_entire_binding(),
            });
        }
        bindings.push(BindGroupEntry {
            binding: 10,
            resource: BindingResource::TextureView(&view),
        });
        device.create_bind_group(
            "compact_clear",
            &pipeline.get_bind_group_layout(0).into(),
            &bindings,
        )
    };
    let dispatch = |pipeline: &ComputePipeline, group: &BindGroup, x: u32, y: u32| {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("compact_clear_dispatch"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("compact_clear_dispatch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(x, y, 1);
        }
        queue.submit([encoder.finish()]);
    };
    let readback = |heads: &Buffer| {
        let image_offset = (head_bytes as u64 + 255) & !255;
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("compact_clear_readback"),
            size: image_offset + u64::from(ROW_PITCH * HEIGHT),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("compact_clear_readback"),
        });
        encoder.copy_buffer_to_buffer(heads, 0, &buffer, 0, head_bytes as u64);
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &buffer,
                layout: TexelCopyBufferLayout {
                    offset: image_offset,
                    bytes_per_row: Some(ROW_PITCH),
                    rows_per_image: Some(HEIGHT),
                },
            },
            Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
        );
        let submission = queue.submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer.slice(..).map_async(MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        device
            .poll(PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(std::time::Duration::from_secs(30)),
            })
            .unwrap();
        receiver.recv().unwrap().unwrap();
        let data = buffer.slice(..).get_mapped_range().unwrap().to_vec();
        buffer.unmap();
        let word = |i: usize| u32::from_le_bytes(data[i * 4..i * 4 + 4].try_into().unwrap());
        let count = word(root + 1);
        let mut ids: Vec<_> = (0..count as usize).map(|i| word(root + 5 + i)).collect();
        ids.sort_unstable();
        let args = [word(root + 2), word(root + 3), word(root + 4)];
        let pixels: Vec<_> = (0..COUNT)
            .map(|i| {
                let offset =
                    image_offset as usize + (i / WIDTH * ROW_PITCH + i % WIDTH * 16) as usize;
                std::array::from_fn::<_, 4, _>(|c| {
                    f32::from_le_bytes(data[offset + c * 4..offset + c * 4 + 4].try_into().unwrap())
                })
            })
            .collect();
        (ids, args, pixels)
    };
    let fill = |value: [f32; 4]| {
        let pixels = vec![value; COUNT as usize];
        let bytes: Vec<_> = pixels
            .iter()
            .flat_map(|pixel| pixel.iter().flat_map(|v| v.to_le_bytes()))
            .collect();
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &bytes,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(WIDTH * 16),
                rows_per_image: Some(HEIGHT),
            },
            Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
        );
    };
    let clear_groups = (4 * (COUNT * 4).next_power_of_two()).div_ceil(64);
    dispatch(
        &clear,
        &group(
            &clear,
            &primary_uniform,
            &primary_heads,
            &primary_entries,
            true,
        ),
        clear_groups,
        1,
    );
    dispatch(
        &clear,
        &group(
            &clear,
            &secondary_uniform,
            &secondary_heads,
            &secondary_entries,
            true,
        ),
        clear_groups,
        1,
    );
    let sentinel = [1.0, 2.0, 3.0, 4.0];
    fill(sentinel);
    dispatch(
        &directions,
        &group(
            &directions,
            &primary_uniform,
            &primary_heads,
            &primary_entries,
            false,
        ),
        1,
        1,
    );
    let (ids, args, pixels) = readback(&primary_heads);
    assert_eq!(ids, [0, 6, 14]);
    assert_eq!(args, [1, 1, 1]);
    assert!(
        pixels.iter().all(|pixel| *pixel == [0.0; 4]),
        "selected and unselected primary hit pixels must clear"
    );
    fill(sentinel);
    dispatch(
        &directions,
        &group(
            &directions,
            &secondary_uniform,
            &secondary_heads,
            &secondary_entries,
            false,
        ),
        1,
        1,
    );
    let (ids, args, pixels) = readback(&secondary_heads);
    assert_eq!(ids, [2, 8]);
    assert_eq!(args, [1, 1, 1]);
    assert!(
        pixels.iter().all(|pixel| *pixel == sentinel),
        "secondary generation must preserve primary hits"
    );
    let (ids, args, _) = readback(&primary_heads);
    assert_eq!(ids, [0, 6, 14]);
    assert_eq!(args, [1, 1, 1]);
    dispatch(
        &clear,
        &group(
            &clear,
            &second_clear_uniform,
            &primary_heads,
            &primary_entries,
            true,
        ),
        clear_groups,
        1,
    );
    let (ids, args, _) = readback(&primary_heads);
    assert_eq!(ids, [0, 6, 14], "second clear preserves the compact list");
    assert_eq!(args, [1, 1, 1]);
    dispatch(
        &clear,
        &group(
            &clear,
            &primary_uniform,
            &primary_heads,
            &empty_entries,
            true,
        ),
        clear_groups,
        1,
    );
    fill(sentinel);
    dispatch(
        &directions,
        &group(
            &directions,
            &primary_uniform,
            &primary_heads,
            &empty_entries,
            false,
        ),
        1,
        1,
    );
    let (ids, args, pixels) = readback(&primary_heads);
    assert!(ids.is_empty(), "next frame must reset selection count");
    assert_eq!(args, [0, 1, 1]);
    assert!(
        pixels.iter().all(|pixel| *pixel == [0.0; 4]),
        "zero-selection frame clears old hits"
    );
    let (ids, args, _) = readback(&secondary_heads);
    assert_eq!(
        ids,
        [2, 8],
        "primary reset must not touch secondary scratch"
    );
    assert_eq!(args, [1, 1, 1]);
}

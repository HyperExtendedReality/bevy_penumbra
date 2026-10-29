//! GPU coverage of the production full-resolution reconstruction helper.
use super::*;
use bevy::render::render_resource::binding_types::{texture_2d, uniform_buffer};

const CHECK_COMPOSITE: &str = r#"
#include "composite.slang"
[[vk::binding(12, 0)]] RWStructuredBuffer<float4> results;
[shader("compute")]
[numthreads(8, 8, 1)]
void checkComposite(uint3 thread : SV_DispatchThreadID) {
    if (any(thread.xy >= params.viewport.zw)) return;
    uint2 pixel = params.viewport.xy + thread.xy;
    results[thread.y * params.viewport.z + thread.x] = float4(
        reconstructCompositeGi(pixel, depthTexture.Load(int3(pixel, 0)),
                               gbuffer.Load(int3(pixel, 0))), 1.0);
}
"#;

const DEPTH_SHADER: &str = r#"
@group(0) @binding(0) var depths: texture_2d<f32>;
@vertex fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(select(-1.0, 3.0, index == 1u), select(-1.0, 3.0, index == 2u), 0.0, 1.0);
}
@fragment fn fragment(@builtin(position) position: vec4<f32>) -> @builtin(frag_depth) f32 {
    return textureLoad(depths, vec2<i32>(position.xy), 0).x;
}
"#;

struct CompositeFixture {
    parameters: TransportParameters,
    full_size: UVec2,
    resolve_size: UVec2,
    depth: Vec<[f32; 4]>,
    raster: Vec<[u32; 4]>,
    gbuffer: Vec<[u32; 4]>,
    irradiance: Vec<[f32; 4]>,
    receivers: Vec<[f32; 4]>,
    probes: Vec<([i32; 4], [f32; 3])>,
}

impl CompositeFixture {
    fn new(scale: u32, origin: UVec2, extent: UVec2) -> Self {
        let mut parameters = temporal_test_parameters(0);
        parameters.size.z = 16;
        parameters.viewport = UVec4::new(origin.x, origin.y, extent.x, extent.y);
        parameters.sampling.w = scale;
        let full_size = origin + extent;
        let resolve_size = UVec2::new(extent.x.div_ceil(scale), extent.y.div_ceil(scale));
        let count = (full_size.x * full_size.y) as usize;
        let resolve_count = (resolve_size.x * resolve_size.y) as usize;
        Self {
            parameters,
            full_size,
            resolve_size,
            depth: vec![[0.5, 0.0, 0.0, 0.0]; count],
            raster: vec![[1.0f32.to_bits(), 2.0f32.to_bits(), 0xa000_4000, 0]; count],
            gbuffer: vec![[0, 0, 0, 0x0080_0800]; count],
            irradiance: vec![[2.0, 3.0, 5.0, 0.5]; resolve_count],
            receivers: vec![[1.0, 2.0, 8_390_656.0, 65_537.0]; resolve_count],
            probes: Vec::new(),
        }
    }

    fn full_index(&self, local: UVec2) -> usize {
        let full = UVec2::new(self.parameters.viewport.x, self.parameters.viewport.y) + local;
        (full.y * self.full_size.x + full.x) as usize
    }

    fn run(
        &self,
        device: &bevy::render::renderer::RenderDevice,
        queue: &bevy::render::renderer::RenderQueue,
        pipeline: &ComputePipeline,
        depth_pipeline: &RenderPipeline,
    ) -> Vec<[f32; 4]> {
        let sampled = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
        let float_texture = |label, size| {
            temporal_test_texture_sized(device, label, TextureFormat::Rgba32Float, sampled, size)
        };
        let irradiance = float_texture("composite_test_irradiance", self.resolve_size);
        let receivers = float_texture("composite_test_receivers", self.resolve_size);
        let depths = float_texture("composite_test_depth_values", self.full_size);
        let raster = temporal_test_texture_sized(
            device,
            "composite_test_raster",
            TextureFormat::Rgba32Uint,
            sampled,
            self.full_size,
        );
        let gbuffer = temporal_test_texture_sized(
            device,
            "composite_test_gbuffer",
            TextureFormat::Rgba32Uint,
            sampled,
            self.full_size,
        );
        let upload = |texture: &Texture, bytes: Vec<u8>, size: UVec2| {
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
                    bytes_per_row: Some(size.x * 16),
                    rows_per_image: Some(size.y),
                },
                Extent3d {
                    width: size.x,
                    height: size.y,
                    depth_or_array_layers: 1,
                },
            );
        };
        for (texture, pixels, size) in [
            (&irradiance, &self.irradiance, self.resolve_size),
            (&receivers, &self.receivers, self.resolve_size),
            (&depths, &self.depth, self.full_size),
        ] {
            upload(
                texture,
                pixels
                    .iter()
                    .flatten()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
                size,
            );
        }
        for (texture, pixels) in [(&raster, &self.raster), (&gbuffer, &self.gbuffer)] {
            upload(
                texture,
                pixels
                    .iter()
                    .flatten()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
                self.full_size,
            );
        }
        let depth = temporal_test_texture_sized(
            device,
            "composite_test_depth",
            TextureFormat::Depth32Float,
            TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
            self.full_size,
        );
        let view = |texture: &Texture| texture.create_view(&TextureViewDescriptor::default());
        let irradiance_view = view(&irradiance);
        let receiver_view = view(&receivers);
        let depths_view = view(&depths);
        let raster_view = view(&raster);
        let gbuffer_view = view(&gbuffer);
        let depth_view = view(&depth);
        let depth_group = device.create_bind_group(
            "composite_test_depth",
            &depth_pipeline.get_bind_group_layout(0).into(),
            &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&depths_view),
            }],
        );
        let mut parameters = UniformBuffer::from(self.parameters);
        parameters.write_buffer(device, queue);
        let capacity = (self.parameters.size.z * 4).next_power_of_two() as usize;
        let base = (2 * self.parameters.size.w + 36) as usize;
        let mut heads = vec![u32::MAX; base + capacity];
        let mut entries = vec![0u32; self.probes.len().max(1) * 68];
        let mut directional = vec![0u32; capacity * 2 * 440];
        for (index, (key, color)) in self.probes.iter().enumerate() {
            for (word, value) in key.iter().enumerate() {
                entries[index * 68 + word] = *value as u32;
            }
            let mut hash = (key[0] as u32).wrapping_mul(0x8da6_b343)
                ^ (key[1] as u32).wrapping_mul(0xd816_3841)
                ^ (key[2] as u32).wrapping_mul(0xcb1a_b31f)
                ^ (key[3] as u32).wrapping_mul(0x1656_67b1);
            hash ^= hash >> 16;
            hash = hash.wrapping_mul(0x7feb_352d);
            hash ^= hash >> 15;
            let mut slot = hash as usize & (capacity - 1);
            while heads[base + slot] != u32::MAX {
                slot = (slot + 1) & (capacity - 1);
            }
            heads[base + slot] = index as u32;
            for texel in 0..36 {
                for component in 0..3 {
                    directional[slot * 440 + texel * 4 + component] = color[component].to_bits();
                }
                directional[slot * 440 + texel * 4 + 3] = 1.0f32.to_bits();
            }
        }
        let storage = |label, words: &[u32]| {
            device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some(label),
                contents: &words
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect::<Vec<_>>(),
                usage: BufferUsages::STORAGE,
            })
        };
        let heads = storage("composite_test_probe_heads", &heads);
        let entries = storage("composite_test_probe_entries", &entries);
        let directional = storage("composite_test_directional", &directional);
        let output_size = u64::from(self.parameters.viewport.z * self.parameters.viewport.w) * 16;
        let output = device.create_buffer(&BufferDescriptor {
            label: Some("composite_test_output"),
            size: output_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("composite_test_readback"),
            size: output_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(
            "composite_test",
            &pipeline.get_bind_group_layout(0).into(),
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: parameters.binding().unwrap(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&irradiance_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&depth_view),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::TextureView(&gbuffer_view),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::TextureView(&receiver_view),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: BindingResource::TextureView(&raster_view),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: heads.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 10,
                    resource: entries.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 11,
                    resource: directional.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 12,
                    resource: output.as_entire_binding(),
                },
            ],
        );
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("composite_test"),
        });
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("composite_test_depth"),
                color_attachments: &[],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(0.0),
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..default()
            });
            pass.set_pipeline(depth_pipeline);
            pass.set_bind_group(0, &depth_group, &[]);
            pass.draw(0..3, 0..1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("composite_test"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(
                self.parameters.viewport.z.div_ceil(8),
                self.parameters.viewport.w.div_ceil(8),
                1,
            );
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_size);
        let submission = queue.submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::channel();
        readback
            .slice(..)
            .map_async(MapMode::Read, move |result| sender.send(result).unwrap());
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
        let bytes = readback.slice(..).get_mapped_range().unwrap().to_vec();
        readback.unmap();
        (0..(output_size / 16) as usize)
            .map(|index| output_vec4(&bytes, index))
            .collect()
    }
}

#[test]
#[ignore = "requires slangc and a desktop GPU; dispatches production composite reconstruction"]
fn gpu_composite_upsample_respects_receiver_geometry_and_support() {
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
            "composite_check.slang",
            &[
                ("common.slang", include_str!("../shaders/common.slang")),
                (
                    "receiver_metadata.slang",
                    include_str!("../shaders/receiver_metadata.slang"),
                ),
                (
                    "composite.slang",
                    include_str!("../shaders/composite.slang"),
                ),
                (
                    "world_resolve.slang",
                    include_str!("../shaders/world_resolve.slang"),
                ),
                ("composite_check.slang", CHECK_COMPOSITE),
            ],
            &SlangSettings {
                target: SlangTarget::Wgsl,
                entry_points: vec!["checkComposite".into()],
                optimization: Some(2),
                ..default()
            },
        )
        .unwrap();
    let bevy::shader::Source::Wgsl(source) = shader.source else {
        panic!("expected WGSL");
    };
    let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
        label: Some("composite_test"),
        source: ShaderSource::Wgsl(source),
    });
    // Explicit layout intentionally accepts resources unused by the broken helper,
    // so the same fixture establishes failing-before and passing-after behavior.
    let layout = device.create_bind_group_layout(
        Some("composite_test"),
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                (0, uniform_buffer::<TransportParameters>(false)),
                (
                    2,
                    texture_2d(TextureSampleType::Float { filterable: false }),
                ),
                (3, texture_2d(TextureSampleType::Depth)),
                (5, texture_2d(TextureSampleType::Uint)),
                (
                    6,
                    texture_2d(TextureSampleType::Float { filterable: false }),
                ),
                (7, texture_2d(TextureSampleType::Uint)),
                (
                    9,
                    bevy::render::render_resource::binding_types::storage_buffer_read_only_sized(
                        false, None,
                    ),
                ),
                (
                    10,
                    bevy::render::render_resource::binding_types::storage_buffer_read_only_sized(
                        false, None,
                    ),
                ),
                (
                    11,
                    bevy::render::render_resource::binding_types::storage_buffer_read_only_sized(
                        false, None,
                    ),
                ),
                (
                    12,
                    bevy::render::render_resource::binding_types::storage_buffer::<Vec<Vec4>>(
                        false,
                    ),
                ),
            ),
        ),
    );
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("composite_test"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&RawComputePipelineDescriptor {
        label: Some("composite_test"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("checkComposite"),
        compilation_options: default(),
        cache: None,
    });
    let depth_module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
        label: Some("composite_test_depth"),
        source: ShaderSource::Wgsl(DEPTH_SHADER.into()),
    });
    let depth_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("composite_test_depth"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &depth_module,
            entry_point: Some("vertex"),
            compilation_options: default(),
            buffers: &[],
        },
        primitive: default(),
        depth_stencil: Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(CompareFunction::Always),
            stencil: default(),
            bias: default(),
        }),
        multisample: default(),
        fragment: Some(wgpu::FragmentState {
            module: &depth_module,
            entry_point: Some("fragment"),
            compilation_options: default(),
            targets: &[],
        }),
        multiview_mask: None,
        cache: None,
    });
    let check = |fixture: &CompositeFixture| fixture.run(device, queue, &pipeline, &depth_pipeline);
    let assert_rgb = |actual: [f32; 4], expected: [f32; 3], label: &str| {
        for component in 0..3 {
            assert!(
                (actual[component] - expected[component]).abs() < 2e-4,
                "{label}: {actual:?} != {expected:?}"
            );
        }
    };
    for scale in [1, 2, 4] {
        for origin in [UVec2::ZERO, UVec2::new(3, 2)] {
            let fixture = CompositeFixture::new(scale, origin, UVec2::new(9, 5));
            for actual in check(&fixture) {
                assert_rgb(
                    actual,
                    [2.0, 3.0, 5.0],
                    "constant field including partial tiles",
                );
            }
        }
    }
    let mut ramp = CompositeFixture::new(4, UVec2::new(3, 2), UVec2::new(9, 5));
    for y in 0..ramp.resolve_size.y {
        for x in 0..ramp.resolve_size.x {
            ramp.irradiance[(y * ramp.resolve_size.x + x) as usize] = [
                (x * 4 + 2).min(8) as f32,
                (y * 4 + 2).min(4) as f32,
                1.0,
                0.5,
            ];
        }
    }
    let reconstructed = check(&ramp);
    assert_rgb(
        reconstructed[4 * 9 + 8],
        [8.0, 4.0, 1.0],
        "partial tile endpoint interpolation",
    );
    assert_rgb(
        reconstructed[3 * 9 + 7],
        [7.0, 3.0, 1.0],
        "partial tile interior interpolation",
    );
    let mut fixture = CompositeFixture::new(4, UVec2::new(3, 2), UVec2::new(9, 5));
    let target = UVec2::new(4, 2);
    let output_index = (target.y * 9 + target.x) as usize;
    // A bright incompatible sample must not leak through bilinear weights. A
    // denoise-filled value is eligible only when its post-filter support bit is set.
    for (case, expected) in [
        ("identity", [2.0, 3.0, 5.0]),
        ("normal", [2.0, 3.0, 5.0]),
        ("raster normal", [2.0, 3.0, 5.0]),
        ("depth", [2.0, 3.0, 5.0]),
        ("missing", [2.0, 3.0, 5.0]),
        ("denoised support", [51.0, 101.5, 152.5]),
        ("source ineligible", [2.0, 3.0, 5.0]),
        ("black", [1.0, 1.5, 2.5]),
        ("no compatible fallback", [0.0; 3]),
        (
            "compatible neighbor outside bilinear footprint",
            [2.0, 3.0, 5.0],
        ),
        ("supported black bilinear suppresses fallback", [0.0; 3]),
        ("target identity", [0.0; 3]),
        ("target normal", [0.0; 3]),
        ("target depth", [0.0; 3]),
        ("target ineligible", [0.0; 3]),
    ] {
        fixture.irradiance.fill([2.0, 3.0, 5.0, 0.5]);
        fixture.receivers.fill([1.0, 2.0, 8_390_656.0, 65_537.0]);
        fixture
            .raster
            .fill([1.0f32.to_bits(), 2.0f32.to_bits(), 0xa000_4000, 0]);
        fixture.depth.fill([0.5, 0.0, 0.0, 0.0]);
        for y in 0..fixture.resolve_size.y {
            let other = (y * fixture.resolve_size.x + 1) as usize;
            fixture.irradiance[other] = [100.0, 200.0, 300.0, 0.5];
            match case {
                "identity" => fixture.receivers[other][0] = 7.0,
                "normal" => fixture.receivers[other][2] = 8_392_703.0, // +X instead of +Z
                "depth" => fixture.irradiance[other][3] = 0.9,
                "missing" => fixture.receivers[other][3] = 1.0,
                "denoised support" => fixture.receivers[other][3] = 65_537.0,
                "black" => fixture.irradiance[other] = [0.0, 0.0, 0.0, 0.5],
                _ => {}
            }
        }
        if case == "source ineligible" || case == "raster normal" {
            for y in 0..fixture.resolve_size.y {
                let full = fixture.full_index(UVec2::new(6, (y * 4 + 2).min(4)));
                fixture.raster[full][2] = if case == "source ineligible" {
                    0
                } else {
                    0xa000_7fff
                };
            }
        }
        let full = fixture.full_index(target);
        match case {
            "no compatible fallback" => fixture
                .receivers
                .iter_mut()
                .for_each(|receiver| receiver[0] = 7.0),
            "compatible neighbor outside bilinear footprint" => {
                // At target (4, 2), the 2x2 bilinear footprint is grid x=0..1.
                // Only grid x=2 has a compatible, supported sample. This catches
                // a black seam when sparse anchors miss a narrow receiver facet.
                fixture
                    .receivers
                    .iter_mut()
                    .for_each(|receiver| receiver[0] = 7.0);
                fixture.receivers[2][0] = 1.0;
            }
            "supported black bilinear suppresses fallback" => {
                fixture.irradiance.fill([0.0, 0.0, 0.0, 0.5]);
                fixture.irradiance[2] = [100.0, 200.0, 300.0, 0.5];
            }
            "target identity" => fixture.raster[full][0] = 7.0f32.to_bits(),
            "target normal" => fixture.raster[full][2] = 0xa000_7fff, // packed +X
            "target depth" => fixture.depth[full][0] = 0.9,
            "target ineligible" => fixture.raster[full][2] = 0,
            _ => {}
        }
        assert_rgb(check(&fixture)[output_index], expected, case);
    }
    // No reduced-resolution anchor lands on this receiver. Resolve the world
    // field at its own position and normal, rather than returning black or
    // accepting another receiver's bright screen sample.
    let mut narrow = CompositeFixture::new(4, UVec2::new(3, 2), UVec2::new(9, 5));
    narrow
        .receivers
        .iter_mut()
        .for_each(|receiver| receiver[0] = 7.0);
    for z in -1..=0 {
        for y in -1..=0 {
            for x in -1..=0 {
                narrow.probes.push(([x, y, z, 0], [7.0, 11.0, 13.0]));
            }
        }
    }
    assert_rgb(
        check(&narrow)[output_index],
        [7.0, 11.0, 13.0],
        "sub-grid receiver uses world probes",
    );
    narrow
        .receivers
        .iter_mut()
        .for_each(|receiver| receiver[0] = 1.0);
    narrow.irradiance.fill([0.0, 0.0, 0.0, 0.5]);
    assert_rgb(
        check(&narrow)[output_index],
        [0.0; 3],
        "supported black screen result is not replaced by probes",
    );
    // On a sloped plane, depth must be evaluated at the clamped integer anchor
    // of the final partial tile, not at its off-viewport nominal center.
    let mut edge = CompositeFixture::new(4, UVec2::new(3, 2), UVec2::new(9, 5));
    edge.parameters.world_from_clip = Mat4::from_cols(Vec4::X, Vec4::Y, Vec4::Z, Vec4::W);
    for y in 0..5 {
        for x in 0..9 {
            let index = edge.full_index(UVec2::new(x, y));
            edge.depth[index][0] = 0.1 + 0.08 * x as f32;
            // Oct encoding of normalize(-0.36, 0, 1) and its raster equivalent.
            edge.raster[index][2] = 0xa000_2f0f;
        }
    }
    for y in 0..edge.resolve_size.y {
        for x in 0..edge.resolve_size.x {
            let index = (y * edge.resolve_size.x + x) as usize;
            edge.irradiance[index][3] = 0.1 + 0.08 * (x * 4 + 2).min(8) as f32;
            edge.receivers[index][2] = 8_390_114.0;
        }
    }
    let actual = check(&edge);
    for y in 0..5 {
        assert_rgb(
            actual[(y * 9 + 8) as usize],
            [2.0, 3.0, 5.0],
            "clamped partial-tile anchor on a world plane",
        );
    }

    // The target and sparse anchor lie on a shallow curved surface. Their
    // chord is offset from either endpoint tangent, but remains on the average
    // tangent plane. This must interpolate at the normal tolerance above.
    let mut curved = CompositeFixture::new(4, UVec2::new(3, 2), UVec2::new(9, 5));
    curved.parameters.tuning.x = 0.15;
    curved.parameters.tuning.z = 0.005;
    for y in 0..5 {
        for x in 0..9 {
            let index = curved.full_index(UVec2::new(x, y));
            let world_x = (x as f32 + 0.5) * (2.0 / 9.0) - 1.0;
            curved.depth[index][0] = 0.5 + 0.08 * world_x * world_x;
            curved.raster[index][2] = packed_geometric_normal(Vec3::new(-0.16 * world_x, 0.0, 1.0));
        }
    }
    for y in 0..curved.resolve_size.y {
        for x in 0..curved.resolve_size.x {
            let anchor_x = (x * 4 + 2).min(8);
            let world_x = (anchor_x as f32 + 0.5) * (2.0 / 9.0) - 1.0;
            curved.irradiance[(y * curved.resolve_size.x + x) as usize][3] =
                0.5 + 0.08 * world_x * world_x;
        }
    }
    curved.irradiance[1][..3].copy_from_slice(&[100.0, 200.0, 300.0]);
    let curved_actual = check(&curved);
    assert_rgb(
        curved_actual[output_index],
        [51.0, 101.5, 152.5],
        "shallow curved surface across sparse anchors",
    );

    // The average-normal plane does not admit a parallel surface offset: when
    // both normals agree, the distance remains the full normal separation.
    let mut parallel = CompositeFixture::new(4, UVec2::new(3, 2), UVec2::new(9, 5));
    parallel.irradiance[1] = [100.0, 200.0, 300.0, 0.6];
    assert_rgb(
        check(&parallel)[output_index],
        [2.0, 3.0, 5.0],
        "parallel offset surface remains rejected",
    );
}

fn packed_geometric_normal(normal: Vec3) -> u32 {
    let normal = normal.normalize();
    let scale = normal.x.abs() + normal.y.abs() + normal.z.abs();
    let oct = bevy::math::Vec2::new(normal.x, normal.y) / scale;
    let encoded = (oct * 0.5 + bevy::math::Vec2::splat(0.5)) * 32_767.0;
    let x = encoded.x.round().clamp(0.0, 32_767.0) as u32;
    let y = encoded.y.round().clamp(0.0, 32_767.0) as u32;
    0x8000_0000 | x | (y << 15)
}

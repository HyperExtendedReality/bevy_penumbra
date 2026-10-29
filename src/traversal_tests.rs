//! GPU traversal versus independent double-precision, world-space brute force.
use super::*;
use bevy::{
    render::{
        render_resource::*,
        renderer::initialize_renderer,
        settings::{Backends, WgpuSettings},
    },
    tasks::block_on,
};
use bevy_slang::{SlangCompiler, SlangSettings, SlangTarget};

#[derive(ShaderType)]
struct CheckParameters {
    frame: UVec4,
    tuning: Vec4,
    size: UVec4,
    layers: UVec4,
}
const TRACKED_RESULT_SLOTS: usize = 14;

#[derive(Clone, Copy, Debug)]
struct Ray {
    origin: Vec3,
    direction: Vec3,
    limit: f32,
    any_hit: bool,
}

fn traversal_pipeline(
    device: &bevy::render::renderer::RenderDevice,
    entry: &str,
) -> ComputePipeline {
    let shader = SlangCompiler::default()
        .compile_bundle(
            if entry == "deformVertices" || entry == "refitBlas" {
                "deformation.slang"
            } else if entry == "checkLinkedSort" {
                "sort_check.slang"
            } else {
                "traversal_check.slang"
            },
            &[
                (
                    "deformation.slang",
                    include_str!("../shaders/deformation.slang"),
                ),
                (
                    "sort_check.slang",
                    include_str!("../shaders/sort_check.slang"),
                ),
                (
                    "linked_merge_sort.slang",
                    include_str!("../shaders/linked_merge_sort.slang"),
                ),
                ("common.slang", include_str!("../shaders/common.slang")),
                (
                    "traversal.slang",
                    include_str!("../shaders/traversal.slang"),
                ),
                ("material.slang", include_str!("../shaders/material.slang")),
                (
                    "traversal_check.slang",
                    include_str!("../shaders/traversal_check.slang"),
                ),
            ],
            &SlangSettings {
                target: SlangTarget::Wgsl,
                entry_points: vec![entry.into()],
                optimization: Some(2),
                ..default()
            },
        )
        .unwrap();
    let bevy::shader::Source::Wgsl(source) = shader.source else {
        panic!("expected WGSL");
    };
    let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
        label: Some(entry),
        source: ShaderSource::Wgsl(source),
    });
    if entry == "deformVertices" || entry == "refitBlas" {
        // Validate the actual shared renderer layout as well as the inferred dispatch layout.
        let descriptor = crate::render::deformation_layout();
        let group_layout =
            device.create_bind_group_layout("deformation_production_check", &descriptor.entries);
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("deformation_production_check"),
            bind_group_layouts: &[Some(&group_layout)],
            immediate_size: 0,
        });
        device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("deformation_production_check"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: default(),
            cache: None,
        });
    }
    device.create_compute_pipeline(&RawComputePipelineDescriptor {
        label: Some(entry),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: default(),
        cache: None,
    })
}

pub(crate) fn dispatch_readback(
    device: &bevy::render::renderer::RenderDevice,
    queue: &bevy::render::renderer::RenderQueue,
    pipeline: &ComputePipeline,
    group: &BindGroup,
    output: &Buffer,
    bytes: u64,
    workgroups: u32,
) -> Vec<u8> {
    let staging = device.create_buffer(&BufferDescriptor {
        label: Some("ray_scene_test_readback"),
        size: bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(output, 0, &staging, 0, bytes);
    let submission = queue.submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::channel();
    staging.slice(..).map_async(MapMode::Read, move |result| {
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
    let result = staging
        .slice(..)
        .get_mapped_range()
        .expect("staging buffer mapping should be valid")
        .to_vec();
    staging.unmap();
    result
}

#[test]
#[ignore = "requires slangc and a desktop GPU; no demo scene is launched"]
fn gpu_deformation_and_blas_refit_match_cpu() {
    #[derive(ShaderType)]
    struct Parameters {
        dispatch: UVec4,
        scene_pages: UVec4,
        scene_overlay: UVec4,
    }
    let options = WgpuSettings::default();
    let resources = block_on(initialize_renderer(
        options.backends.unwrap_or(Backends::PRIMARY),
        None,
        &options,
    ));
    let device = &resources.0;
    let queue = &resources.1;
    let word = |value: Vec4| UVec4::from_array(value.to_array().map(f32::to_bits));
    let positions = [
        Vec3::ZERO,
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        Vec3::ONE,
        Vec3::new(-1.0, 2.0, 3.0),
    ];
    let normal = Vec3::new(1.0, 2.0, 3.0);
    let tangent = Vec3::new(2.0, -1.0, 1.0);
    let delta = Vec3::new(0.5, -1.0, 2.0);
    let normal_delta = Vec3::new(-0.5, 1.0, 0.0);
    let tangent_delta = Vec3::new(1.0, 0.0, -0.5);
    let weight = -0.5;
    let matrix = Mat4::from_scale_rotation_translation(
        Vec3::new(-2.0, 3.0, 0.5),
        Quat::from_rotation_z(0.3),
        Vec3::new(4.0, -2.0, 1.0),
    );
    let mut geometry = vec![
        UVec4::new(1, 3, 5, 12),
        UVec4::new(6, 7, 8, 0),
        UVec4::new(9, 10, 11, 0),
    ];
    for position in positions {
        geometry.extend([
            word(position.extend(17.0)),
            word(normal.extend(23.0)),
            word(tangent.extend(-1.0)),
            UVec4::new(11, 12, 13, 14),
            UVec4::new(21, 22, 23, 24),
        ]);
    }
    geometry.resize(63, UVec4::ZERO);
    let geometry = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("deformation_geometry_check"),
        contents: &geometry
            .iter()
            .flat_map(|v| v.to_array().into_iter().flat_map(u32::to_le_bytes))
            .collect::<Vec<_>>(),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    // One job, six vertex mappings, two influence words per vertex, three morph words per vertex.
    let mut deformation = vec![
        UVec4::new(1, 1, 4, 0),
        UVec4::new(0, 6, 6, 10),
        UVec4::new(22, 1, 0, 4),
        UVec4::new(1, 0, 0, 0),
    ];
    deformation.extend((0..6).map(|i| UVec4::new(0, i, i, 0)));
    for _ in 0..6 {
        deformation.extend([UVec4::ZERO, word(Vec4::X)]);
    }
    for _ in 0..6 {
        deformation.extend([
            word(delta.extend(0.0)),
            word(normal_delta.extend(0.0)),
            word(tangent_delta.extend(0.0)),
        ]);
    }
    let mut deformation = StorageBuffer::from(deformation);
    deformation.write_buffer(device, queue);
    let mut poses = Vec::new();
    crate::deformation::pose_words(
        &crate::deformation::DeformationPose {
            matrices: vec![matrix],
            weights: vec![weight],
        },
        &mut poses,
    );
    let mut poses = StorageBuffer::from(poses);
    poses.write_buffer(device, queue);
    let mut params = UniformBuffer::from(Parameters {
        dispatch: UVec4::new(0, 6, 0, 1),
        scene_pages: UVec4::ZERO,
        scene_overlay: UVec4::ZERO,
    });
    params.write_buffer(device, queue);
    // Dynamic paging branches still need texture bindings in the buffer-backed shader.
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("deformation_unused_pages"),
        size: Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Uint,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    let output_texture = device.create_texture(&TextureDescriptor {
        label: Some("deformation_unused_output"),
        size: Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Uint,
        usage: TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    let output_view = output_texture.create_view(&TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    let pipeline = traversal_pipeline(device, "deformVertices");
    let group = device.create_bind_group(
        "deformation_check",
        &pipeline.get_bind_group_layout(0).into(),
        &[
            BindGroupEntry {
                binding: 0,
                resource: params.binding().unwrap(),
            },
            BindGroupEntry {
                binding: 1,
                resource: geometry.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: deformation.binding().unwrap(),
            },
            BindGroupEntry {
                binding: 4,
                resource: poses.binding().unwrap(),
            },
            BindGroupEntry {
                binding: 7,
                resource: BindingResource::TextureView(&view),
            },
            BindGroupEntry {
                binding: 8,
                resource: BindingResource::TextureView(&output_view),
            },
        ],
    );
    let bytes = dispatch_readback(device, queue, &pipeline, &group, &geometry, 63 * 16, 1);
    let words = bytes
        .as_chunks::<16>()
        .0
        .iter()
        .map(|b| {
            UVec4::from_array(std::array::from_fn(|i| {
                u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap())
            }))
        })
        .collect::<Vec<_>>();
    let vector = |word: UVec4| {
        Vec3::new(
            f32::from_bits(word.x),
            f32::from_bits(word.y),
            f32::from_bits(word.z),
        )
    };
    let expected_positions = positions.map(|p| matrix.transform_point3(p + weight * delta));
    for (i, expected) in expected_positions.iter().enumerate() {
        let base = 3 + (6 + i) * 5;
        assert!((vector(words[base]) - *expected).length() < 2e-5);
        let expected_normal = matrix
            .inverse()
            .transpose()
            .transform_vector3(normal + weight * normal_delta)
            .normalize();
        let expected_tangent = matrix
            .transform_vector3(tangent + weight * tangent_delta)
            .normalize();
        assert!((vector(words[base + 1]) - expected_normal).length() < 2e-5);
        assert!((vector(words[base + 2]) - expected_tangent).length() < 2e-5);
        assert_eq!(words[base].w, 17.0f32.to_bits());
        assert_eq!(words[base + 1].w, 23.0f32.to_bits());
        assert_eq!(words[base + 2].w, (-1.0f32).to_bits());
        assert_eq!(words[base + 3], UVec4::new(11, 12, 13, 14));
        assert_eq!(words[base + 4], UVec4::new(21, 22, 23, 24));
    }
    let scene_words = [
        UVec4::ZERO,
        UVec4::new(0, 0, 0, 1),
        UVec4::ZERO,
        UVec4::ZERO,
        UVec4::new(0, 0, 0, 1),
        UVec4::new(0, 0, 0, 1),
        UVec4::new(0, 0, 0, 1),
    ];
    let scene = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("deformation_blas_check"),
        contents: &scene_words
            .iter()
            .flat_map(|v| v.to_array().into_iter().flat_map(u32::to_le_bytes))
            .collect::<Vec<_>>(),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let mut nodes = StorageBuffer::from(vec![1u32, 2, 0]);
    nodes.write_buffer(device, queue);
    let pipeline = traversal_pipeline(device, "refitBlas");
    let mut result = Vec::new();
    for (offset, count) in [(0, 2), (2, 1)] {
        params.get_mut().dispatch = UVec4::new(offset, count, 0, 1);
        params.write_buffer(device, queue);
        let group = device.create_bind_group(
            "deformation_refit_check",
            &pipeline.get_bind_group_layout(0).into(),
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: params.binding().unwrap(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: geometry.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: scene.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: nodes.binding().unwrap(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::TextureView(&view),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: BindingResource::TextureView(&view),
                },
            ],
        );
        result = dispatch_readback(device, queue, &pipeline, &group, &scene, 7 * 16, 1);
    }
    let words = result
        .as_chunks::<16>()
        .0
        .iter()
        .map(|b| {
            UVec4::from_array(std::array::from_fn(|i| {
                u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap())
            }))
        })
        .collect::<Vec<_>>();
    for (node, range) in [(0, 0..6), (1, 0..3), (2, 3..6)] {
        let (minimum, maximum) = expected_positions[range].iter().fold(
            (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
            |(lo, hi), p| (lo.min(*p), hi.max(*p)),
        );
        assert!((vector(words[1 + node * 2]) - minimum).length() < 2e-5);
        assert!((vector(words[2 + node * 2]) - maximum).length() < 2e-5);
        assert_eq!(words[1 + node * 2].w, scene_words[1 + node * 2].w);
        assert_eq!(words[2 + node * 2].w, scene_words[2 + node * 2].w);
    }

    // Force records across 4x4 page boundaries; append output after four immutable layers.
    let paged = device.create_texture(&TextureDescriptor {
        label: Some("deformation_paged_check"),
        size: Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 6,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Uint,
        usage: TextureUsages::TEXTURE_BINDING
            | TextureUsages::STORAGE_BINDING
            | TextureUsages::COPY_DST
            | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let mut base_bytes = bytes.clone();
    base_bytes.resize(64 * 16, 0);
    queue.write_texture(
        TexelCopyTextureInfo {
            texture: &paged,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        &base_bytes,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * 16),
            rows_per_image: Some(4),
        },
        Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 4,
        },
    );
    let input_view = paged.create_view(&TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        base_array_layer: 0,
        array_layer_count: Some(4),
        ..default()
    });
    let output_view = paged.create_view(&TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        base_array_layer: 4,
        array_layer_count: Some(2),
        ..default()
    });
    let full_view = paged.create_view(&TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    let mut overlay = StorageBuffer::from(vec![UVec4::new(1, 1, 64, 6), UVec4::new(6, 6, 0, 0)]);
    overlay.add_usages(BufferUsages::COPY_SRC);
    overlay.write_buffer(device, queue);
    params.get_mut().dispatch = UVec4::new(0, 6, 2, 4);
    params.write_buffer(device, queue);
    let pipeline = traversal_pipeline(device, "deformVertices");
    let group = device.create_bind_group(
        "paged_deformation_check",
        &pipeline.get_bind_group_layout(0).into(),
        &[
            BindGroupEntry {
                binding: 0,
                resource: params.binding().unwrap(),
            },
            BindGroupEntry {
                binding: 1,
                resource: overlay.binding().unwrap(),
            },
            BindGroupEntry {
                binding: 3,
                resource: deformation.binding().unwrap(),
            },
            BindGroupEntry {
                binding: 4,
                resource: poses.binding().unwrap(),
            },
            BindGroupEntry {
                binding: 7,
                resource: BindingResource::TextureView(&input_view),
            },
            BindGroupEntry {
                binding: 8,
                resource: BindingResource::TextureView(&output_view),
            },
        ],
    );
    // Serialize deformation before rebinding the appended layers as sampled refit inputs.
    dispatch_readback(
        device,
        queue,
        &pipeline,
        &group,
        overlay.buffer().unwrap(),
        32,
        1,
    );
    let readback = device.create_buffer(&BufferDescriptor {
        label: Some("deformation_pages_readback"),
        size: 256 * 4 * 6,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture: &paged,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &readback,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(4),
            },
        },
        Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 6,
        },
    );
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
    let mapped = readback
        .slice(..)
        .get_mapped_range()
        .expect("readback buffer mapping should be valid");
    let page_bytes = mapped
        .as_chunks::<256>()
        .0
        .iter()
        .flat_map(|row| row[..64].iter().copied())
        .collect::<Vec<_>>();
    // Immutable input layers must survive, while sparse destination vertices use appended layers.
    assert_eq!(&page_bytes[..64 * 16], &base_bytes);
    assert_eq!(&page_bytes[64 * 16..94 * 16], &bytes[33 * 16..63 * 16]);
    drop(mapped);
    readback.unmap();

    let mut paged_scene = vec![scene_words[0], UVec4::new(0, 3, 2, 0)];
    paged_scene.extend_from_slice(&scene_words[1..]);
    let mut paged_scene = StorageBuffer::from(paged_scene);
    paged_scene.add_usages(BufferUsages::COPY_SRC);
    paged_scene.write_buffer(device, queue);
    params.get_mut().scene_pages = UVec4::new(1, 1, 1, 1);
    params.get_mut().scene_overlay = UVec4::new(7, 0, 0, 6);
    let pipeline = traversal_pipeline(device, "refitBlas");
    let mut paged_result = Vec::new();
    for (offset, count) in [(0, 2), (2, 1)] {
        params.get_mut().dispatch = UVec4::new(offset, count, 2, 4);
        params.write_buffer(device, queue);
        let group = device.create_bind_group(
            "paged_deformation_refit_check",
            &pipeline.get_bind_group_layout(0).into(),
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: params.binding().unwrap(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: overlay.binding().unwrap(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: paged_scene.binding().unwrap(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: nodes.binding().unwrap(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::TextureView(&view),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: BindingResource::TextureView(&full_view),
                },
            ],
        );
        paged_result = dispatch_readback(
            device,
            queue,
            &pipeline,
            &group,
            paged_scene.buffer().unwrap(),
            8 * 16,
            1,
        );
    }
    assert_eq!(&paged_result[..16], &result[..16]);
    assert_eq!(&paged_result[32..], &result[16..]);
}

fn check_texture_sampling(
    device: &bevy::render::renderer::RenderDevice,
    queue: &bevy::render::renderer::RenderQueue,
) {
    use bevy::{
        asset::RenderAssetUsages,
        image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    };
    let pipeline = traversal_pipeline(device, "checkTextureSampling");
    let mut queries = Vec::new();
    for lod in [
        -2.0, -0.2, 0.0, 0.2, 0.49, 0.51, 0.9, 1.0, 1.6, 3.3, 9.0, 12.0,
    ] {
        // Stay on either side of nearest-filter discontinuities: native
        // sampler coordinate rounding may choose either texel exactly on one.
        for x in [
            -2.10003,
            -0.70003,
            -0.0001,
            0.0,
            0.1234,
            509.5 / 530.0,
            509.99 / 530.0,
            510.01 / 530.0,
            510.5 / 530.0,
            1.0,
            1.0001,
            2.10003,
        ] {
            for y in [-1.1, -0.001, 0.0, 0.37, 1.0, 1.001] {
                queries.push(Vec4::new(x, y, lod, 0.0));
            }
        }
    }
    let mut input = StorageBuffer::from(queries.clone());
    input.write_buffer(device, queue);
    let output = device.create_buffer(&BufferDescriptor {
        label: Some("texture_sampling_output"),
        size: queries.len() as u64 * 32,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut cases = 0;
    let mut worst = 0.0f32;
    for format in [TextureFormat::Rgba8Unorm, TextureFormat::Rgba8UnormSrgb] {
        let size = Extent3d {
            width: 530,
            height: 17,
            depth_or_array_layers: 1,
        };
        let mut pixels = Vec::new();
        let mut mips = Vec::new();
        for level in 0..10 {
            let extent = Extent3d {
                width: (size.width >> level).max(1),
                height: (size.height >> level).max(1),
                ..size
            };
            let start = pixels.len();
            for y in 0..extent.height {
                for x in 0..extent.width {
                    pixels.extend([
                        (x * 7 + y * 13 + level * 19) as u8,
                        (x * 3 + y * 5 + level * 29) as u8,
                        (x * 11 + y * 2 + level * 37) as u8,
                        (x + y * 17 + level * 41) as u8,
                    ]);
                }
            }
            mips.push((start, pixels.len(), extent));
        }
        let mut image = Image::new_uninit(
            size,
            TextureDimension::D2,
            format,
            RenderAssetUsages::default(),
        );
        image.data = Some(pixels.clone());
        image.texture_descriptor.mip_level_count = 10;
        let native = device.create_texture(&TextureDescriptor {
            label: Some("native_sampling_oracle"),
            size,
            mip_level_count: 10,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (level, &(start, end, extent)) in mips.iter().enumerate() {
            queue.write_texture(
                TexelCopyTextureInfo {
                    texture: &native,
                    mip_level: level as u32,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                &pixels[start..end],
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(extent.width * 4),
                    rows_per_image: Some(extent.height),
                },
                extent,
            );
        }
        let native_view = native.create_view(&TextureViewDescriptor::default());
        for address in [
            ImageAddressMode::ClampToEdge,
            ImageAddressMode::Repeat,
            ImageAddressMode::MirrorRepeat,
        ] {
            for filters in 0..4 {
                let descriptor = ImageSamplerDescriptor {
                    address_mode_u: address,
                    address_mode_v: match address {
                        ImageAddressMode::Repeat => ImageAddressMode::MirrorRepeat,
                        _ => ImageAddressMode::Repeat,
                    },
                    mag_filter: if filters & 1 != 0 {
                        ImageFilterMode::Linear
                    } else {
                        ImageFilterMode::Nearest
                    },
                    min_filter: if filters & 2 != 0 {
                        ImageFilterMode::Linear
                    } else {
                        ImageFilterMode::Nearest
                    },
                    mipmap_filter: if filters & 1 != 0 {
                        ImageFilterMode::Linear
                    } else {
                        ImageFilterMode::Nearest
                    },
                    lod_min_clamp: if filters == 3 { 0.3 } else { 0.0 },
                    lod_max_clamp: if filters == 3 { 4.7 } else { 32.0 },
                    ..default()
                };
                image.sampler = ImageSampler::Descriptor(descriptor.clone());
                let mut images = Assets::<Image>::default();
                let handle = images.add(image.clone());
                let mut cache = crate::material::TextureCache::default();
                cache
                    .update(
                        vec![handle.id()],
                        Some(&images),
                        &crate::material::TextureOptions::default(),
                        true,
                    )
                    .unwrap();
                let (_atlas, atlas_view) =
                    crate::render::upload_material_atlas(device, queue, &cache.atlas);
                let mut words = vec![UVec4::new(1, 0, 0, 0)];
                words.extend(&cache.atlas.words);
                let mut scene = StorageBuffer::from(words);
                scene.write_buffer(device, queue);
                let mut params = UniformBuffer::from(CheckParameters {
                    frame: UVec4::new(cache.atlas.textures[&handle.id()], 0, 0, 0),
                    tuning: Vec4::ZERO,
                    size: UVec4::new(queries.len() as u32, 0, 0, 0),
                    layers: UVec4::ZERO,
                });
                params.write_buffer(device, queue);
                let sampler = device.create_sampler(&descriptor.as_wgpu());
                let layout = pipeline.get_bind_group_layout(0).into();
                let group = device.create_bind_group(
                    "texture_sampling_check",
                    &layout,
                    &[
                        BindGroupEntry {
                            binding: 0,
                            resource: params.binding().unwrap(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: scene.binding().unwrap(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: input.binding().unwrap(),
                        },
                        BindGroupEntry {
                            binding: 4,
                            resource: output.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 5,
                            resource: BindingResource::TextureView(&atlas_view),
                        },
                        BindGroupEntry {
                            binding: 6,
                            resource: BindingResource::TextureView(&native_view),
                        },
                        BindGroupEntry {
                            binding: 7,
                            resource: BindingResource::Sampler(&sampler),
                        },
                    ],
                );
                let result = dispatch_readback(
                    device,
                    queue,
                    &pipeline,
                    &group,
                    &output,
                    queries.len() as u64 * 32,
                    (queries.len() as u32).div_ceil(64),
                );
                for (query, record) in queries.iter().zip(result.as_chunks::<32>().0.iter()) {
                    let values: Vec<_> = record
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|v| f32::from_le_bytes(*v))
                        .collect();
                    let atlas = Vec4::from_slice(&values[..4]);
                    let native = Vec4::from_slice(&values[4..8]);
                    // Independent f64 filtering of the original source bytes.
                    // Native fractional LOD is approximate even on this device;
                    // do not use that approximation as the transport oracle.
                    let lod = f64::from(query.z.clamp(
                        descriptor.lod_min_clamp.min(9.0),
                        descriptor.lod_max_clamp.min(9.0),
                    ));
                    let linear = if lod > 0.0 {
                        descriptor.min_filter
                    } else {
                        descriptor.mag_filter
                    } == ImageFilterMode::Linear;
                    let level_color = |level: usize| {
                        let (start, _, extent) = mips[level];
                        let address = |coordinate: i32, length: u32, mode| match mode {
                            ImageAddressMode::Repeat => coordinate.rem_euclid(length as i32) as u32,
                            ImageAddressMode::MirrorRepeat => {
                                let index = coordinate.rem_euclid(length as i32 * 2) as u32;
                                index.min(length * 2 - 1 - index)
                            }
                            _ => coordinate.clamp(0, length as i32 - 1) as u32,
                        };
                        let pixel = |x, y| {
                            let x = address(x, extent.width, descriptor.address_mode_u);
                            let y = address(y, extent.height, descriptor.address_mode_v);
                            let offset = start + ((y * extent.width + x) * 4) as usize;
                            let rgba =
                                std::array::from_fn(|i| f32::from(pixels[offset + i]) / 255.0);
                            let color = if format == TextureFormat::Rgba8UnormSrgb {
                                Color::srgba(rgba[0], rgba[1], rgba[2], rgba[3])
                                    .to_linear()
                                    .to_vec4()
                            } else {
                                Vec4::from_array(rgba)
                            };
                            color.as_dvec4()
                        };
                        let normalize = |uv: f32, mode| match mode {
                            ImageAddressMode::Repeat => f64::from(uv).rem_euclid(1.0),
                            ImageAddressMode::MirrorRepeat => {
                                let uv = f64::from(uv).rem_euclid(2.0);
                                uv.min(2.0 - uv)
                            }
                            _ => f64::from(uv).clamp(0.0, 1.0),
                        };
                        let x =
                            normalize(query.x, descriptor.address_mode_u) * f64::from(extent.width);
                        let y = normalize(query.y, descriptor.address_mode_v)
                            * f64::from(extent.height);
                        if !linear {
                            return pixel(x.floor() as i32, y.floor() as i32);
                        }
                        let x = x - 0.5;
                        let y = y - 0.5;
                        let ix = x.floor() as i32;
                        let iy = y.floor() as i32;
                        let fx = x - x.floor();
                        let fy = y - y.floor();
                        pixel(ix, iy)
                            .lerp(pixel(ix + 1, iy), fx)
                            .lerp(pixel(ix, iy + 1).lerp(pixel(ix + 1, iy + 1), fx), fy)
                    };
                    let low = level_color(lod.floor() as usize);
                    let high = level_color(lod.ceil() as usize);
                    let reference = if descriptor.mipmap_filter == ImageFilterMode::Linear {
                        low.lerp(high, lod.fract())
                    } else {
                        level_color((lod + 0.5).floor() as usize)
                    };
                    assert!(
                        (atlas.as_dvec4() - reference).abs().max_element() < 0.0001,
                        "source-byte filtering {descriptor:?} {query:?}: atlas {atlas:?}, f64 {reference:?}"
                    );
                    let error = (atlas - native).abs().max_element();
                    worst = worst.max(error);
                    // This device has 8 sub-texel precision bits: two axes can
                    // each move a bilinear weight by one 1/256 step. Native
                    // sRGB conversion gets a separate 1/256 channel bound.
                    // Fractional native LOD additionally
                    // permits 1/64 of adjacent-mip contrast; the strict f64
                    // comparison above still verifies exact trilinear weights.
                    let lod_bound = if descriptor.mipmap_filter == ImageFilterMode::Linear {
                        (high - low).abs().max_element() as f32 / 64.0
                    } else {
                        0.0
                    };
                    let native_bound = if linear { 2.0 / 256.0 } else { 1e-6 }
                        + if format == TextureFormat::Rgba8UnormSrgb {
                            1.0 / 256.0
                        } else {
                            0.0
                        }
                        + lod_bound;
                    assert!(
                        atlas.is_finite() && native.is_finite() && error <= native_bound,
                        "{format:?} {descriptor:?} query {query:?}: atlas {atlas:?}, native {native:?}, error {error} (native bound {native_bound})"
                    );
                }
                cases += 1;
            }
        }
    }
    println!(
        "{} texture/native comparisons in {cases} sampler cases; max channel error {worst:.8}",
        queries.len() * cases
    );
}

// Moller-Trumbore in world space, with f64 arithmetic. This deliberately shares
// neither the shader's sheared edge algorithm nor its BVH or local-ray transforms.
fn brute_force(data: &SceneData, ray: Ray, layers: &RenderLayers) -> Vec<(f64, u32, u32)> {
    let mut hits = Vec::new();
    for (index, instance) in data.instances.iter().enumerate() {
        if !layers.intersects(&instance.layers) {
            continue;
        }
        for triangle in instance.first..instance.first + instance.count {
            let local = data.triangles[triangle as usize];
            let matrix = Mat4::from(instance.world_from_local);
            let [a, b, c] = [local.a, local.b, local.c].map(|v| {
                matrix.x_axis.truncate().as_dvec3() * f64::from(v.x)
                    + matrix.y_axis.truncate().as_dvec3() * f64::from(v.y)
                    + matrix.z_axis.truncate().as_dvec3() * f64::from(v.z)
                    + matrix.w_axis.truncate().as_dvec3()
            });
            let direction = ray.direction.as_dvec3();
            let e1 = b - a;
            let e2 = c - a;
            let facing = e1.cross(e2).dot(direction);
            let cull = instance.flags & 3;
            if (cull == 1 && facing >= 0.0) || (cull == 2 && facing <= 0.0) {
                continue;
            }
            let p = direction.cross(e2);
            let determinant = e1.dot(p);
            if determinant == 0.0 {
                continue;
            }
            let relative = ray.origin.as_dvec3() - a;
            let u = relative.dot(p) / determinant;
            let q = relative.cross(e1);
            let v = direction.dot(q) / determinant;
            let t = e2.dot(q) / determinant;
            // Compare within a componentwise f32 position uncertainty interval
            // at a boundary. Barycentric gradients avoid treating a tall, thin
            // axis-aligned triangle as equally uncertain along every axis.
            let normal = e1.cross(e2);
            let gradient_u = e2.cross(normal) / normal.length_squared();
            let gradient_v = normal.cross(e1) / normal.length_squared();
            let error = (ray.origin.as_dvec3().abs()
                + direction.abs() * t.abs()
                + a.abs()
                + b.abs()
                + c.abs())
                * (8.0 * f64::from(f32::EPSILON));
            if u < -gradient_u.abs().dot(error)
                || v < -gradient_v.abs().dot(error)
                || u + v > 1.0 + (gradient_u + gradient_v).abs().dot(error)
            {
                continue;
            }
            if t > 0.000025 && t < f64::from(ray.limit) {
                hits.push((t, triangle, index as u32));
            }
        }
    }
    hits.sort_by(|a, b| a.0.total_cmp(&b.0));
    hits
}

fn rays_for(data: &SceneData) -> Vec<Ray> {
    let mut rays = Vec::new();
    for instance in data.instances.iter() {
        for local in data.triangles
            [instance.first as usize..(instance.first + instance.count) as usize]
            .iter()
            .take(12)
        {
            let triangle = instance.triangle(*local);
            let [a, b, c] = [triangle.a, triangle.b, triangle.c].map(|v| v.truncate());
            let normal = (b - a).cross(c - a).normalize();
            let center = (a + b + c) / 3.0;
            // Interiors from both faces and a face's shared diagonal. Distances
            // span four orders of magnitude without normalizing local rays.
            let edges = [(local.a, local.b), (local.b, local.c), (local.c, local.a)];
            let (start, end) = edges
                .into_iter()
                .max_by(|(a, b), (c, d)| {
                    (b - a)
                        .length_squared()
                        .total_cmp(&(d - c).length_squared())
                })
                .unwrap();
            let diagonal = instance
                .world_from_local
                .transform_point3((start.truncate() + end.truncate()) * 0.5);
            let targets = if instance.world_from_local.matrix3 == bevy::math::Mat3A::IDENTITY {
                vec![center, diagonal, a]
            } else {
                vec![center, diagonal]
            };
            for target in targets {
                for sign in [-1.0, 1.0] {
                    for distance in [0.01, 1.0, 100.0] {
                        let origin = target + normal * (sign * distance);
                        let direction = (target - origin).normalize();
                        if direction.is_finite() {
                            rays.push(Ray {
                                origin,
                                direction,
                                limit: distance * 1.1,
                                any_hit: false,
                            });
                        }
                    }
                }
            }
        }
    }
    let mut state = 7u32;
    let mut random = || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (state >> 8) as f32 / 16777216.0
    };
    for _ in 0..4096 {
        let origin = Vec3::new(random(), random(), random()) * 100.0 - Vec3::splat(50.0);
        let direction = (Vec3::new(random(), random(), random()) * 2.0 - Vec3::ONE).normalize();
        rays.push(Ray {
            origin,
            direction,
            limit: 200.0,
            any_hit: false,
        });
    }
    // Exact axis-parallel directions exercise zero slab components.
    for instance in data.instances.iter() {
        let center = (instance.min + instance.max) * 0.5;
        for direction in [Vec3::X, Vec3::Y, Vec3::Z, -Vec3::X, -Vec3::Y, -Vec3::Z] {
            rays.push(Ray {
                origin: center - direction * 100.0,
                direction,
                limit: 200.0,
                any_hit: false,
            });
        }
    }
    let shadows: Vec<_> = rays
        .iter()
        .map(|ray| Ray {
            any_hit: true,
            ..*ray
        })
        .collect();
    rays.extend(shadows);
    rays
}

fn check_attributes(data: &SceneData, ray: Ray, words: &[u32]) -> Result<(), String> {
    use bevy::math::{DMat3, DVec3, DVec4};
    let instance = &data.instances[words[2] as usize];
    let triangle = data.triangles[words[1] as usize];
    let vertices = triangle
        .indices
        .to_array()
        .map(|i| data.vertices[i as usize]);
    let matrix = Mat4::from(instance.world_from_local);
    let linear = DMat3::from_cols(
        matrix.x_axis.truncate().as_dvec3(),
        matrix.y_axis.truncate().as_dvec3(),
        matrix.z_axis.truncate().as_dvec3(),
    );
    let world =
        vertices.map(|v| linear * v.position.as_dvec3() + matrix.w_axis.truncate().as_dvec3());
    let geometric = (world[1] - world[0]).cross(world[2] - world[0]).normalize();
    let bary = Vec3::new(
        f32::from_bits(words[8]),
        f32::from_bits(words[9]),
        f32::from_bits(words[10]),
    )
    .as_dvec3();
    if !bary.is_finite() || bary.min_element() < -2e-6 || (bary.element_sum() - 1.0).abs() > 2e-6 {
        return Err(format!("invalid barycentrics {bary:?}"));
    }
    // Independent f64 Moller-Trumbore barycentrics, with the same componentwise
    // f32 input uncertainty used by the traversal oracle at thin edge boundaries.
    let e1 = world[1] - world[0];
    let e2 = world[2] - world[0];
    let p = ray.direction.as_dvec3().cross(e2);
    let determinant = e1.dot(p);
    let relative = ray.origin.as_dvec3() - world[0];
    let q = relative.cross(e1);
    let u = relative.dot(p) / determinant;
    let v = ray.direction.as_dvec3().dot(q) / determinant;
    let normal = e1.cross(e2);
    let grad_u = e2.cross(normal) / normal.length_squared();
    let grad_v = normal.cross(e1) / normal.length_squared();
    let error = (ray.origin.as_dvec3().abs()
        + ray.direction.as_dvec3().abs() * f64::from(f32::from_bits(words[0]))
        + world[0].abs()
        + world[1].abs()
        + world[2].abs())
        * (8.0 * f64::from(f32::EPSILON));
    if (bary.y - u).abs() > grad_u.abs().dot(error) + 2e-6
        || (bary.z - v).abs() > grad_v.abs().dot(error) + 2e-6
    {
        return Err(format!(
            "barycentrics {bary:?}, independent ({}, {u}, {v})",
            1.0 - u - v
        ));
    }
    let normal_matrix = linear.inverse().transpose();
    let sign = linear.determinant().signum();
    let mut normal = DVec3::ZERO;
    let mut tangent = DVec4::ZERO;
    let mut uv = DVec4::ZERO;
    let mut color = DVec4::ZERO;
    for (i, vertex) in vertices.iter().enumerate() {
        let n = normal_matrix * vertex.normal.as_dvec3();
        normal += if vertex.flags & 1 != 0 && n.length_squared() > 0.0 {
            n.normalize() * bary[i]
        } else {
            geometric * bary[i]
        };
        let t = linear * vertex.tangent.truncate().as_dvec3();
        tangent += if t.length_squared() > 0.0 {
            t.normalize()
        } else {
            DVec3::ZERO
        }
        .extend(f64::from(vertex.tangent.w) * sign)
            * bary[i];
        uv += vertex.uv.as_dvec4() * bary[i];
        color += vertex.color.as_dvec4() * bary[i];
    }
    let mut shading = if normal.length_squared() > 0.0 {
        normal.normalize()
    } else {
        geometric
    };
    if instance.flags & 8 != 0 && geometric.dot(ray.direction.as_dvec3()) > 0.0 {
        shading = -shading;
    }
    let albedo =
        (instance.albedo.as_dvec3() * color.truncate()).extend(f64::from(instance.flags & 3));
    let emission = (instance.emission.as_dvec3()
        * if instance.flags & 16 != 0 {
            color.truncate()
        } else {
            DVec3::ONE
        })
    .extend(0.0);
    for (offset, expected, name) in [
        (12, geometric.extend(0.0), "geometric normal"),
        (16, normal.extend(0.0), "interpolated normal"),
        (20, tangent, "tangent"),
        (24, uv, "UV0/UV1"),
        (28, color, "vertex color"),
        (32, shading.extend(0.0), "shading normal"),
        (36, albedo, "material albedo"),
        (40, emission, "emission and reserved zero"),
    ] {
        let actual = DVec4::from_array(std::array::from_fn(|i| {
            f64::from(f32::from_bits(words[offset + i]))
        }));
        if !actual.is_finite() || (actual - expected).abs().max_element() > 2e-4 {
            return Err(format!("{name}: GPU {actual:?}, independent {expected:?}"));
        }
    }
    Ok(())
}

fn trace_gpu(
    data: &SceneData,
    rays: &[Ray],
    device: &bevy::render::renderer::RenderDevice,
    queue: &bevy::render::renderer::RenderQueue,
    pipeline: &ComputePipeline,
    refit_pipeline: &ComputePipeline,
) -> Vec<u8> {
    trace_gpu_for_layers(
        data,
        rays,
        device,
        queue,
        pipeline,
        refit_pipeline,
        &RenderLayers::default(),
    )
}

fn trace_gpu_for_layers(
    data: &SceneData,
    rays: &[Ray],
    device: &bevy::render::renderer::RenderDevice,
    queue: &bevy::render::renderer::RenderQueue,
    pipeline: &ComputePipeline,
    refit_pipeline: &ComputePipeline,
    layers: &RenderLayers,
) -> Vec<u8> {
    trace_gpu_for_layers_with_targets(
        data,
        (rays, &[]),
        device,
        queue,
        pipeline,
        refit_pipeline,
        layers,
    )
}

fn trace_gpu_for_layers_with_targets(
    data: &SceneData,
    (rays, emissive_targets): (&[Ray], &[Option<Vec3>]),
    device: &bevy::render::renderer::RenderDevice,
    queue: &bevy::render::renderer::RenderQueue,
    pipeline: &ComputePipeline,
    refit_pipeline: &ComputePipeline,
    layers: &RenderLayers,
) -> Vec<u8> {
    assert!(
        emissive_targets.is_empty() || emissive_targets.len() == rays.len(),
        "emissive endpoints must match the ray count"
    );
    let (lights, conventional, selector) = view_lights(data, layers);
    let mut lights = StorageBuffer::from(lights);
    lights.write_buffer(device, queue);
    let mut params = UniformBuffer::from(CheckParameters {
        frame: UVec4::new(0, 0, data.instances.len() as u32, conventional),
        tuning: Vec4::new(0.0, 0.0, 0.0001, 0.0),
        size: UVec4::new(rays.len() as u32, 0, 0, 0),
        layers: selector,
    });
    params.write_buffer(device, queue);
    let (_atlas, atlas_view) = crate::render::upload_material_atlas(device, queue, &data.textures);
    let triangles = crate::render::upload_geometry(device, queue, &data.geometry_words);
    let mut words: Vec<_> = data
        .acceleration_prefix
        .iter()
        .chain(data.blas_words.iter())
        .chain(data.instance_words.iter())
        .copied()
        .collect();
    // Poison the uploaded TLAS bounds: this run can pass only if GPU refit
    // rebuilds them from current instance transforms and immutable BLAS bounds.
    for word in &mut words[1..1 + data.nodes.len() * 2] {
        *word = UVec4::new(
            100_000.0f32.to_bits(),
            100_000.0f32.to_bits(),
            100_000.0f32.to_bits(),
            word.w,
        );
    }
    let mut scene = StorageBuffer::from(words);
    scene.write_buffer(device, queue);
    let (order, dispatches) = crate::render::refit_dispatches(&data.nodes);
    let mut order = StorageBuffer::from(order);
    order.write_buffer(device, queue);
    let offsets: Vec<_> = dispatches.iter().map(|value| (value.x, value.y)).collect();
    crate::render::refit_tlas(
        device,
        queue,
        refit_pipeline,
        &scene,
        &order,
        &offsets,
        None,
    );
    let values: Vec<_> = rays
        .iter()
        .enumerate()
        .flat_map(|(index, ray)| {
            let emitter_target = emissive_targets
                .get(index)
                .and_then(|target| *target)
                .map_or(Vec4::ZERO, |target| target.extend(1.0));
            [
                ray.origin.extend(ray.limit),
                ray.direction.extend(u32::from(ray.any_hit) as f32),
                emitter_target,
            ]
        })
        .collect();
    let mut input = StorageBuffer::from(values);
    input.write_buffer(device, queue);
    let output = device.create_buffer(&BufferDescriptor {
        label: Some("traversal_check_output"),
        size: rays.len() as u64 * TRACKED_RESULT_SLOTS as u64 * 16,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let layout = pipeline.get_bind_group_layout(0).into();
    let group = device.create_bind_group(
        "traversal_check",
        &layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: params.binding().unwrap(),
            },
            BindGroupEntry {
                binding: 1,
                resource: triangles.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: scene.binding().unwrap(),
            },
            BindGroupEntry {
                binding: 3,
                resource: input.binding().unwrap(),
            },
            BindGroupEntry {
                binding: 4,
                resource: output.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: BindingResource::TextureView(&atlas_view),
            },
            BindGroupEntry {
                binding: 8,
                resource: lights.binding().unwrap(),
            },
        ],
    );
    dispatch_readback(
        device,
        queue,
        pipeline,
        &group,
        &output,
        rays.len() as u64 * TRACKED_RESULT_SLOTS as u64 * 16,
        (rays.len() as u32).div_ceil(64),
    )
}

fn check_gpu(
    data: &SceneData,
    rays: &[Ray],
    device: &bevy::render::renderer::RenderDevice,
    queue: &bevy::render::renderer::RenderQueue,
    pipeline: &ComputePipeline,
    refit_pipeline: &ComputePipeline,
    expected_errors: u32,
) {
    let (closest, shadow) = check_gpu_for_layers(
        data,
        rays,
        device,
        queue,
        pipeline,
        refit_pipeline,
        expected_errors,
        &RenderLayers::default(),
    );
    assert!(
        closest > 500 && shadow > 500,
        "fixture must exercise many actual hits"
    );
}

#[allow(clippy::too_many_arguments)]
fn check_gpu_for_layers(
    data: &SceneData,
    rays: &[Ray],
    device: &bevy::render::renderer::RenderDevice,
    queue: &bevy::render::renderer::RenderQueue,
    pipeline: &ComputePipeline,
    refit_pipeline: &ComputePipeline,
    expected_errors: u32,
    layers: &RenderLayers,
) -> (usize, usize) {
    let mapped = trace_gpu_for_layers(data, rays, device, queue, pipeline, refit_pipeline, layers);
    let mut failures = Vec::new();
    let mut closest = 0;
    let mut shadow = 0;
    let mut worst_relative = 0.0f64;
    for (index, (ray, record)) in rays
        .iter()
        .zip(mapped.as_chunks::<176>().0.iter())
        .enumerate()
    {
        let words: Vec<_> = record
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| u32::from_le_bytes(*v))
            .collect();
        let distance = f64::from(f32::from_bits(words[0]));
        let expected = brute_force(data, *ray, layers);
        let actual = (distance, words[1], words[2]);
        let hit = words[1] != u32::MAX;
        let tolerance = 0.0002 * distance.abs().max(1.0);
        let accelerated_valid = hit == (words[5] != u32::MAX)
            && words[7] == 0
            && (ray.any_hit
                || (distance - f64::from(f32::from_bits(words[4]))).abs()
                    <= 1e-5 * distance.abs().max(1.0))
            && (words[3] & !expected_errors) == 0
            && (!hit || words[3] == expected_errors);
        if hit && words[11] != 0 {
            failures.push(format!(
                "ray {index}: texture LOD mismatch in slots {:#x}",
                words[11]
            ));
        }
        let valid = accelerated_valid
            && if !hit {
                expected.is_empty()
            } else if ray.any_hit {
                shadow += 1;
                expected.iter().any(|&(t, triangle, instance)| {
                    triangle == words[1]
                        && instance == words[2]
                        && (distance - t).abs() <= tolerance
                })
            } else if let Some(&(t, _, _)) = expected.first() {
                closest += 1;
                worst_relative = worst_relative.max((distance - t).abs() / t.abs().max(1.0));
                (distance - t).abs() <= tolerance
                    && expected.iter().any(|&(candidate, triangle, instance)| {
                        (candidate - t).abs() <= tolerance
                            && triangle == words[1]
                            && instance == words[2]
                    })
            } else {
                false
            };
        if !valid && failures.len() < 12 {
            failures.push(format!(
                "ray {index} {ray:?}: GPU {actual:?}, GPU brute {:?}, CPU {:?}",
                (f32::from_bits(words[4]), words[5], words[6]),
                expected.first()
            ));
        }
        if valid
            && hit
            && failures.len() < 12
            && let Err(error) = check_attributes(data, *ray, &words)
        {
            failures.push(format!("ray {index}: {error}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    println!(
        "{} rays, {layers:?}: {closest} closest hits, {shadow} shadow hits; max normalized distance error {worst_relative:.8}",
        rays.len()
    );
    (closest, shadow)
}

fn forced_deep_scene(data: &SceneData, blas: bool) -> SceneData {
    let mut deep = data.clone();
    let chain = |nodes: &mut Vec<BvhNode>, min, max, first, count| {
        let root = nodes.len() as u32;
        // Lay out a valid binary tree whose first-child path is 64 internal
        // nodes deep. Each sibling is a leaf, so DFS accumulates one pending
        // node per level and exercises the 49-entry stack overflow fallback.
        const INTERNALS: usize = 64;
        const NODE_COUNT: usize = INTERNALS * 2 + 1;
        for i in 0..NODE_COUNT {
            let internal = i == 0 || (i % 2 == 1 && i < INTERNALS * 2 - 1);
            nodes.push(BvhNode {
                min,
                max,
                first: if internal {
                    root + if i == 0 { 1 } else { i as u32 + 2 }
                } else {
                    first
                },
                count: if internal { 0 } else { count },
            });
        }
        root
    };
    if blas {
        let mut nodes = Vec::new();
        let mut roots = HashMap::new();
        let mut instances = deep.instances.as_ref().clone();
        for instance in &mut instances {
            let bounds = deep.blas_nodes[instance.root as usize];
            instance.root = *roots.entry(instance.first).or_insert_with(|| {
                chain(
                    &mut nodes,
                    bounds.min,
                    bounds.max,
                    instance.first,
                    instance.count,
                )
            });
        }
        deep.blas_nodes = Arc::new(nodes);
        deep.instances = Arc::new(instances);
    } else {
        let bounds = deep.nodes[0];
        let mut nodes = Vec::new();
        chain(
            &mut nodes,
            bounds.min,
            bounds.max,
            0,
            deep.instances.len() as u32,
        );
        deep.nodes = Arc::new(nodes);
    }
    deep.blas_words = Arc::new(node_words(&deep.blas_nodes, deep.nodes.len() as u32));
    let base = 1 + deep.nodes.len() * 2 + deep.blas_words.len();
    let mut prefix = vec![UVec4::new(
        (base + deep.instances.len() * INSTANCE_WORDS + deep.material_words.len()) as u32,
        base as u32,
        deep.instances.len() as u32,
        INSTANCE_WORDS as u32,
    )];
    prefix.extend(node_words(&deep.nodes, 0));
    deep.acceleration_prefix = Arc::new(prefix);
    deep.instance_words = Arc::new(pack_instances(
        &deep.instances,
        deep.nodes.len(),
        base,
        &deep.material_words,
        &deep.textures.words,
    ));
    deep
}
fn forced_cyclic_blas_scene(data: &SceneData, instance_index: usize) -> SceneData {
    let mut malformed = data.clone();
    let mut nodes = malformed.blas_nodes.as_ref().clone();
    let root = malformed.instances[instance_index].root as usize;
    let bounds = nodes[root];
    // The ray intersects this node, follows its left child back to itself, and
    // misses the right child. No accepted blocker is found before the step cap.
    nodes[root] = BvhNode {
        min: bounds.min,
        max: bounds.max,
        first: root as u32,
        count: 0,
    };
    nodes[root + 1] = BvhNode {
        min: Vec3::splat(100.0),
        max: Vec3::splat(101.0),
        first: 0,
        count: 1,
    };
    malformed.blas_words = Arc::new(node_words(&nodes, malformed.nodes.len() as u32));
    malformed.blas_nodes = Arc::new(nodes);
    malformed
}

fn forced_oversized_blas_leaf(data: &SceneData, instance_index: usize) -> SceneData {
    let mut malformed = data.clone();
    let mut nodes = malformed.blas_nodes.as_ref().clone();
    let instance = &malformed.instances[instance_index];
    let root = instance.root as usize;
    let bounds = nodes[root];
    // This requests an enormous range beyond this instance's real primitives;
    // the shader must clip the scan but cannot claim that a miss was complete.
    nodes[root] = BvhNode {
        min: bounds.min,
        max: bounds.max,
        first: instance.first,
        count: u32::MAX - instance.first,
    };
    malformed.blas_words = Arc::new(node_words(&nodes, malformed.nodes.len() as u32));
    malformed.blas_nodes = Arc::new(nodes);
    malformed
}

fn check_masked_traversal(
    device: &bevy::render::renderer::RenderDevice,
    queue: &bevy::render::renderer::RenderQueue,
    pipeline: &ComputePipeline,
    refit_pipeline: &ComputePipeline,
) {
    use bevy::{
        asset::RenderAssetUsages,
        image::{ImageAddressMode, ImageSampler, ImageSamplerDescriptor},
        math::Affine2,
        mesh::UvChannel,
    };
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .init_resource::<Assets<Mesh>>()
        .init_resource::<Assets<StandardMaterial>>()
        .init_resource::<Assets<Image>>()
        .init_resource::<GiScene>()
        .add_systems(Update, update_scene);
    app.world_mut()
        .spawn((Camera::default(), RadianceCascades::default()));
    let alphas = [0u8, 80, 180, 255];
    let rgba = |phase| {
        (0..12)
            .flat_map(|i| {
                [
                    64,
                    128,
                    192,
                    if phase == 0 {
                        alphas[(i % 4 + i / 4) % 4]
                    } else {
                        255 - alphas[(i % 4 + i / 4) % 4]
                    },
                ]
            })
            .collect::<Vec<_>>()
    };
    let mut image = Image::new(
        Extent3d {
            width: 4,
            height: 3,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        rgba(0),
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::default(),
    );
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::MirrorRepeat,
        ..ImageSamplerDescriptor::nearest()
    });
    let texture = app.world_mut().resource_mut::<Assets<Image>>().add(image);
    let mut normal_image = Image::new(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        vec![180, 100, 240, 255],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::default(),
    );
    normal_image.sampler = ImageSampler::nearest();
    let normal_texture = app
        .world_mut()
        .resource_mut::<Assets<Image>>()
        .add(normal_image);
    let affine =
        Affine2::from_scale_angle_translation(Vec2::new(2.0, 1.5), 0.0, Vec2::new(-0.3, 0.2));
    let front_material = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial {
            base_color: Color::linear_rgba(0.75, 0.5, 0.25, 0.8),
            base_color_texture: Some(texture.clone()),
            base_color_channel: UvChannel::Uv1,
            emissive: LinearRgba::new(0.6, 0.3, 0.1, 1.0),
            emissive_texture: Some(texture.clone()),
            uv_transform: affine,
            alpha_mode: AlphaMode::Mask(0.3),
            normal_map_texture: Some(normal_texture),
            flip_normal_map_y: true,
            cull_mode: None,
            double_sided: true,
            ..default()
        });
    let back_material = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial {
            base_color: Color::linear_rgb(0.1, 0.2, 0.3),
            cull_mode: None,
            unlit: true,
            ..default()
        });
    let mut front_mesh: Mesh = Rectangle::new(2.0, 2.0).into();
    front_mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0, 0.0]; 4]);
    front_mesh.insert_attribute(
        Mesh::ATTRIBUTE_UV_1,
        vec![[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]],
    );
    front_mesh.insert_attribute(
        Mesh::ATTRIBUTE_COLOR,
        vec![
            [0.8, 0.6, 0.4, 1.0],
            [0.8, 0.6, 0.4, 0.35],
            [0.8, 0.6, 0.4, 0.35],
            [0.8, 0.6, 0.4, 1.0],
        ],
    );
    front_mesh.insert_attribute(Mesh::ATTRIBUTE_TANGENT, vec![[1.0, 0.0, 0.0, 1.0]; 4]);
    let front_mesh = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(front_mesh);
    let back_mesh = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(Rectangle::new(3.0, 3.0));
    let front = app
        .world_mut()
        .spawn((
            Mesh3d(front_mesh.clone()),
            MeshMaterial3d(front_material.clone()),
            RadianceCascadesGeometry,
            GlobalTransform::from(Transform::from_scale(Vec3::new(-1.2, 0.7, 1.0))),
        ))
        .id();
    let back = app
        .world_mut()
        .spawn((
            Mesh3d(back_mesh),
            MeshMaterial3d(back_material),
            RadianceCascadesGeometry,
            GlobalTransform::from(Transform::from_xyz(0.0, 0.0, -1.0)),
        ))
        .id();
    let mut rays = Vec::new();
    for y in 0..31 {
        for x in 0..37 {
            let local = Vec2::new((x as f32 + 0.27) / 37.0, (y as f32 + 0.41) / 31.0) * 1.96
                - Vec2::splat(0.98);
            let origin = Vec3::new(-1.2 * local.x, 0.7 * local.y, 2.0);
            for (any_hit, limit) in [(false, 5.0), (true, 2.5), (true, 5.0)] {
                rays.push(Ray {
                    origin,
                    direction: -Vec3::Z,
                    limit,
                    any_hit,
                });
            }
        }
    }
    wait_for_scene_instances(&mut app, 2);
    let mut front_hits = 0;
    let mut back_hits = 0;
    let mut misses = 0;
    for phase in 0..3 {
        if phase == 1 {
            app.world_mut()
                .resource_mut::<Assets<Image>>()
                .get_mut(&texture)
                .unwrap()
                .data = Some(rgba(1));
            app.world_mut()
                .entity_mut(front)
                .insert(GlobalTransform::from(
                    Transform::from_xyz(0.0, 0.0, -0.25).with_scale(Vec3::new(-1.2, 0.7, 1.0)),
                ));
        } else if phase == 2 {
            app.world_mut()
                .resource_mut::<Assets<StandardMaterial>>()
                .get_mut(&front_material)
                .unwrap()
                .alpha_mode = AlphaMode::Opaque;
        }
        app.update();
        assert!(app.world().resource::<GiScene>().ready);
        let data = app.world().resource::<GiScene>().data.clone();
        assert_eq!(
            data.instances.len(),
            2,
            "both fixture panels must be resident"
        );
        for errors in 0..3 {
            let data = if errors == 0 {
                data.as_ref().clone()
            } else {
                forced_deep_scene(&data, errors == 1)
            };
            // Exercise the expensive stack-overflow fallback at representative
            // accepted and rejected alpha samples. The full ray grid remains
            // covered for the normal topology above; dispatching all of it
            // through a deliberately 48-deep tree can trigger a GPU watchdog.
            let query_indices: Vec<usize> = if errors == 0 {
                (0..rays.len()).collect()
            } else {
                vec![0, 3, 15]
            };
            let query_rays: Vec<_> = query_indices.iter().map(|index| rays[*index]).collect();
            let records = trace_gpu(&data, &query_rays, device, queue, pipeline, refit_pipeline);
            for (ray_index, (ray, record)) in query_rays
                .iter()
                .zip(records.as_chunks::<176>().0.iter())
                .enumerate()
            {
                let words: Vec<_> = record
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| u32::from_le_bytes(*v))
                    .collect();
                let u = (ray.origin.x / -1.2 + 1.0) * 0.5;
                let v = (ray.origin.y / 0.7 + 1.0) * 0.5;
                let uv = affine.transform_point2(Vec2::new(u, v));
                let tx = (uv.x.rem_euclid(1.0) * 4.0).floor() as usize;
                let mirrored = uv.y.rem_euclid(2.0);
                let ty = (mirrored.min(2.0 - mirrored) * 3.0).floor().min(2.0) as usize;
                let alpha = if phase == 0 {
                    alphas[(tx + ty) % 4]
                } else {
                    255 - alphas[(tx + ty) % 4]
                };
                let accepted =
                    phase == 2 || f32::from(alpha) / 255.0 * 0.8 * (0.35 + 0.65 * u) >= 0.3;
                let hit = words[1] != u32::MAX;
                assert_eq!(
                    hit,
                    accepted || ray.limit > 3.0,
                    "phase {phase} {ray:?} alpha {alpha}"
                );
                assert_eq!(
                    words[3],
                    errors,
                    "traversal overflow diagnostic mismatch: phase={phase}, topology={errors} (1=BLAS, 2=TLAS), ray_index={}, ray={ray:?}, hit={hit}, triangle={}, instance={}, brute_triangle={}, brute_instance={}, instance_flags={:#010x}",
                    query_indices[ray_index],
                    words[1],
                    words[2],
                    words[5],
                    words[6],
                    data.instances
                        .get(words[2] as usize)
                        .map_or(0, |instance| instance.flags),
                );
                assert_eq!(
                    hit,
                    words[5] != u32::MAX,
                    "acceleration/brute mask mismatch: phase={phase}, topology={errors}, ray_index={}, ray={ray:?}, acceleration_hit={hit}, acceleration_triangle={}, acceleration_instance={}, brute_triangle={}, brute_instance={}, acceleration_errors={}, brute_errors={}",
                    query_indices[ray_index],
                    words[1],
                    words[2],
                    words[5],
                    words[6],
                    words[3],
                    words[7],
                );
                if !hit {
                    if errors == 0 {
                        misses += 1;
                    }
                    continue;
                }
                assert_eq!(
                    words[11], 0,
                    "texture LOD mismatch in slots {:#x}: phase={phase}, ray={ray:?}",
                    words[11],
                );
                let entity = data.instances[words[2] as usize].entity;
                let is_front = entity == front;
                assert!(
                    is_front && accepted || entity == back && ray.limit > 3.0,
                    "phase {phase} {ray:?}: front={is_front}, accepted={accepted}"
                );
                if !ray.any_hit {
                    assert_eq!(
                        is_front, accepted,
                        "closest hit must continue after a masked rejection"
                    );
                }
                let distance = f32::from_bits(words[0]);
                assert!(
                    (distance
                        - if is_front {
                            if phase == 0 { 2.0 } else { 2.25 }
                        } else {
                            3.0
                        })
                    .abs()
                        < 1e-5
                );
                let emission = Vec3::new(
                    f32::from_bits(words[40]),
                    f32::from_bits(words[41]),
                    f32::from_bits(words[42]),
                );
                let expected = if is_front {
                    Vec3::new(0.75, 0.5, 0.25)
                        * Vec3::new(0.8, 0.6, 0.4)
                        * Vec3::new(64.0, 128.0, 192.0)
                        / 255.0
                } else {
                    Vec3::new(0.1, 0.2, 0.3)
                };
                assert!(
                    (emission - expected).abs().max_element() < 2e-5,
                    "texture/color emission {emission:?} vs {expected:?}"
                );
                if is_front {
                    if errors == 0 {
                        front_hits += 1;
                    }
                    let mapped = Vec3::new(180.0, 100.0, 240.0) / 255.0 * 2.0 - Vec3::ONE;
                    let expected = Vec3::new(mapped.x, mapped.y, -mapped.z).normalize();
                    let normal = Vec3::new(
                        f32::from_bits(words[32]),
                        f32::from_bits(words[33]),
                        f32::from_bits(words[34]),
                    );
                    assert!(
                        (normal - expected).abs().max_element() < 2e-5,
                        "mirrored double-sided normal map {normal:?} vs {expected:?}"
                    );
                } else if errors == 0 {
                    back_hits += 1;
                }
            }
        }
    }
    assert!(front_hits > 500 && back_hits > 500 && misses > 500);
    // Blended surfaces remain in the acceleration structure. Fully transparent
    // texels must transmit, opaque texels must block, and fractional coverage
    // must agree statistically for both closest-hit and shadow rays.
    app.world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .get_mut(&front_material)
        .unwrap()
        .alpha_mode = AlphaMode::Blend;
    app.world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .get_mut(&front_material)
        .unwrap()
        .base_color = Color::WHITE;
    app.world_mut()
        .resource_mut::<Assets<Mesh>>()
        .get_mut(&front_mesh)
        .unwrap()
        .insert_attribute(Mesh::ATTRIBUTE_COLOR, vec![[1.0; 4]; 4]);
    app.update();
    let data = app.world().resource::<GiScene>().data.clone();
    assert_eq!(
        data.instances.len(),
        2,
        "blended panel must remain resident"
    );
    let records = trace_gpu(&data, &rays, device, queue, pipeline, refit_pipeline);
    let mut observed = [0usize; 256];
    let mut totals = [0usize; 256];
    for (ray, record) in rays.iter().zip(records.as_chunks::<176>().0) {
        if ray.any_hit && ray.limit > 3.0 {
            continue;
        }
        let words: Vec<_> = record
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| u32::from_le_bytes(*v))
            .collect();
        assert_eq!(
            words[7], 0,
            "shadow and closest-hit alpha coverage diverged: {ray:?}"
        );
        let u = (ray.origin.x / -1.2 + 1.0) * 0.5;
        let v = (ray.origin.y / 0.7 + 1.0) * 0.5;
        let uv = affine.transform_point2(Vec2::new(u, v));
        let tx = (uv.x.rem_euclid(1.0) * 4.0).floor() as usize;
        let mirrored = uv.y.rem_euclid(2.0);
        let ty = (mirrored.min(2.0 - mirrored) * 3.0).floor().min(2.0) as usize;
        let alpha = usize::from(255 - alphas[(tx + ty) % 4]);
        let front_hit = words[1] != u32::MAX && data.instances[words[2] as usize].entity == front;
        totals[alpha] += 1;
        observed[alpha] += usize::from(front_hit);
        if alpha == 0 || alpha == 255 {
            assert_eq!(front_hit, alpha == 255, "blend endpoint: {ray:?}");
        }
    }
    for alpha in [75usize, 175] {
        assert!(totals[alpha] > 100);
        let fraction = observed[alpha] as f32 / totals[alpha] as f32;
        assert!(
            (fraction - alpha as f32 / 255.0).abs() < 0.06,
            "alpha {alpha}: observed {fraction}, {} samples",
            totals[alpha]
        );
    }
    println!(
        "{} masked/opaque queries through image edits and motion: {front_hits} front, {back_hits} back, {misses} rejected shadows",
        rays.len() * 9
    );
}

fn wait_for_scene_instances(app: &mut App, expected: usize) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while app.world().resource::<GiScene>().data.instances.len() < expected
        && std::time::Instant::now() < deadline
    {
        app.update();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert_eq!(
        app.world().resource::<GiScene>().data.instances.len(),
        expected,
        "all scene fixture meshes must be resident"
    );
}

fn wait_for_blas_builds(app: &mut App, expected: u64) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while app.world().resource::<GiScene>().blas_builds < expected
        && std::time::Instant::now() < deadline
    {
        app.update();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(app.world().resource::<GiScene>().blas_builds >= expected);
}

#[test]
#[ignore = "requires slangc and a desktop GPU; WGPU_BACKEND selects the backend"]
fn gpu_traversal_matches_brute_force_through_motion_and_asset_edits() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .init_resource::<Assets<Mesh>>()
        .init_resource::<Assets<StandardMaterial>>()
        .init_resource::<GiScene>()
        .add_systems(Update, update_scene);
    app.world_mut()
        .spawn((Camera::default(), RadianceCascades::default()));
    let cube = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(Cuboid::default());
    let mut panel_mesh: Mesh = Rectangle::new(2.0, 2.0).into();
    let mut bare_mesh = panel_mesh.clone();
    bare_mesh.remove_attribute(Mesh::ATTRIBUTE_NORMAL);
    bare_mesh.remove_attribute(Mesh::ATTRIBUTE_UV_0);
    let bare = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(bare_mesh);
    panel_mesh.insert_attribute(
        Mesh::ATTRIBUTE_NORMAL,
        vec![
            [0.3e38, 0.1e38, 1.0e38],
            [-0.2e-30, 0.4e-30, 1.0e-30],
            [0.1, -0.3, 1.0],
            [-0.4, -0.2, 1.0],
        ],
    );
    panel_mesh.insert_attribute(
        Mesh::ATTRIBUTE_TANGENT,
        vec![
            [3.0e38, 0.6e38, 0.3e38, -1.0],
            [1.0e-30, 0.0, 0.2e-30, -1.0],
            [1.0, -0.3, 0.0, -1.0],
            [1.0, 0.1, -0.2, -1.0],
        ],
    );
    panel_mesh.insert_attribute(
        Mesh::ATTRIBUTE_UV_1,
        vec![[0.2, 0.3], [-1.0, 2.0], [3.0, -2.0], [0.8, 0.9]],
    );
    panel_mesh.insert_attribute(
        Mesh::ATTRIBUTE_COLOR,
        vec![
            [0.2, 0.4, 0.8, 0.5],
            [0.8, 0.4, 0.2, 1.0],
            [0.3, 0.7, 0.9, 0.2],
            [0.9, 0.1, 0.3, 0.8],
        ],
    );
    let panel = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(panel_mesh);
    let materials: Vec<_> = [None, Some(Face::Back), Some(Face::Front)]
        .into_iter()
        .map(|cull_mode| {
            app.world_mut()
                .resource_mut::<Assets<StandardMaterial>>()
                .add(StandardMaterial {
                    cull_mode,
                    double_sided: cull_mode.is_none(),
                    base_color: Color::linear_rgb(0.8, 0.6, 0.4),
                    emissive: LinearRgba::new(0.1, 0.2, 0.3, 1.0),
                    unlit: cull_mode.is_none(),
                    ..default()
                })
        })
        .collect();
    let entities: Vec<_> = (0..24)
        .map(|i| {
            let mesh = if i == 23 {
                bare.clone()
            } else if i % 2 == 0 {
                cube.clone()
            } else {
                panel.clone()
            };
            let scale = if i == 0 {
                Vec3::new(-0.001, 1000.0, 1.0)
            } else {
                Vec3::new(if i % 2 == 0 { -1.2 } else { 1.2 }, 0.6, 2.0)
            };
            app.world_mut()
                .spawn((
                    Mesh3d(mesh),
                    MeshMaterial3d(materials[i as usize % 3].clone()),
                    RadianceCascadesGeometry,
                    GlobalTransform::from(if i == 23 {
                        Transform::from_xyz(-16.0, 0.0, -20.0)
                    } else {
                        Transform::from_xyz(
                            (i % 6) as f32 * 4.0 - 10.0,
                            (i / 6) as f32 * 5.0 - 7.5,
                            0.0,
                        )
                        .with_rotation(if i == 0 {
                            Quat::IDENTITY
                        } else {
                            Quat::from_euler(EulerRot::XYZ, i as f32 * 0.03, i as f32 * 0.07, 0.2)
                        })
                        .with_scale(scale)
                    }),
                ))
                .id()
        })
        .collect();
    app.update();
    let options = WgpuSettings::default();
    let resources = block_on(initialize_renderer(
        options.backends.unwrap_or(Backends::PRIMARY),
        None,
        &options,
    ));
    println!("Traversal GPU checks initialized the render adapter.");
    let device = &resources.0;
    let queue = &resources.1;
    check_texture_sampling(device, queue);
    let pipeline = traversal_pipeline(device, "checkTraversal");
    let shader = SlangCompiler::default()
        .compile_bundle(
            "acceleration.slang",
            &[(
                "acceleration.slang",
                include_str!("../shaders/acceleration.slang"),
            )],
            &SlangSettings {
                target: SlangTarget::Wgsl,
                entry_points: vec!["refitTlas".into()],
                optimization: Some(2),
                ..default()
            },
        )
        .unwrap();
    let bevy::shader::Source::Wgsl(source) = shader.source else {
        panic!("expected WGSL");
    };
    let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
        label: Some("refit_test"),
        source: ShaderSource::Wgsl(source),
    });
    let layout = crate::render::refit_layout();
    let layout = device.create_bind_group_layout("refit_test", &layout.entries);
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("refit_test"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let refit_pipeline = device.create_compute_pipeline(&RawComputePipelineDescriptor {
        label: Some("refit_test"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("refitTlas"),
        compilation_options: default(),
        cache: None,
    });
    check_masked_traversal(device, queue, &pipeline, &refit_pipeline);
    wait_for_scene_instances(&mut app, entities.len());
    let data = app.world().resource::<GiScene>().data.clone();
    let rays = rays_for(&data);
    check_gpu(&data, &rays, device, queue, &pipeline, &refit_pipeline, 0);
    let triangles = data.triangles.clone();
    app.world_mut()
        .entity_mut(entities[0])
        .insert(GlobalTransform::from(
            Transform::from_xyz(7.0, 8.0, 9.0)
                .with_scale(Vec3::new(-2.0, 0.5, 1.0))
                .with_rotation(Quat::from_rotation_y(1.0)),
        ));
    app.update();
    let data = app.world().resource::<GiScene>().data.clone();
    assert!(Arc::ptr_eq(&triangles, &data.triangles));
    check_gpu(
        &data,
        &rays_for(&data),
        device,
        queue,
        &pipeline,
        &refit_pipeline,
        0,
    );
    app.world_mut()
        .resource_mut::<Assets<Mesh>>()
        .insert(cube.id(), Cuboid::new(1.5, 0.75, 1.0).into())
        .unwrap();
    let expected_blas_builds = app.world().resource::<GiScene>().blas_builds + 1;
    app.update();
    wait_for_blas_builds(&mut app, expected_blas_builds);
    let data = app.world().resource::<GiScene>().data.clone();
    assert!(!Arc::ptr_eq(&triangles, &data.triangles));
    check_gpu(
        &data,
        &rays_for(&data),
        device,
        queue,
        &pipeline,
        &refit_pipeline,
        0,
    );
    // Deliberately exceed the builder's enforced depth. The diagnostic flags
    // must report the overflow and fallback must preserve every occluder/hit.
    for blas in [true, false] {
        let deep = forced_deep_scene(&data, blas);
        check_gpu(
            &deep,
            &rays_for(&deep),
            device,
            queue,
            &pipeline,
            &refit_pipeline,
            if blas { 1 } else { 2 },
        );
    }
    // All views share these BLAS/TLAS buffers. Match native Bevy masks while
    // retaining offscreen members, including masks beyond one machine word.
    let masks = [
        RenderLayers::default(),
        RenderLayers::none(),
        RenderLayers::from_layers(&[31, 32]),
        RenderLayers::from_layers(&[63, 64, 129]),
        RenderLayers::from_layers(&[0, 257]),
    ];
    for (index, entity) in entities.iter().enumerate() {
        app.world_mut()
            .entity_mut(*entity)
            .insert(masks[index % masks.len()].clone());
    }
    app.update();
    let data = app.world().resource::<GiScene>().data.clone();
    let rays = rays_for(&data);
    let camera_masks: Vec<_> = masks
        .iter()
        .cloned()
        .chain([31, 32, 63, 64, 129, 257].map(|layer| RenderLayers::none().with(layer)))
        .chain([RenderLayers::from_layers(&[0, 31, 32, 63, 64, 129, 257])])
        .collect();
    for layers in &camera_masks {
        for topology in [0, 1, 2] {
            let data = match topology {
                0 => data.as_ref().clone(),
                1 => forced_deep_scene(&data, true),
                _ => forced_deep_scene(&data, false),
            };
            let counts = check_gpu_for_layers(
                &data,
                &rays,
                device,
                queue,
                &pipeline,
                &refit_pipeline,
                topology,
                layers,
            );
            if *layers == RenderLayers::none() {
                assert_eq!(counts, (0, 0));
            } else {
                assert!(counts.0 > 100 && counts.1 > 100);
            }
        }
    }
}

#[test]
#[ignore = "requires slangc and a desktop GPU; no demo scene is launched"]
fn gpu_tracked_traversal_bounds_primary_and_nee_dependencies() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .init_resource::<Assets<Mesh>>()
        .init_resource::<Assets<StandardMaterial>>()
        .init_resource::<GiScene>()
        .add_systems(Update, update_scene);
    app.world_mut()
        .spawn((Camera::default(), RadianceCascades::default()));

    let mesh = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(Rectangle::new(2.0, 2.0));
    let cycle_mesh = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(Cuboid::default());
    let material = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial {
            cull_mode: None,
            double_sided: true,
            ..default()
        });
    let primary_panel = app
        .world_mut()
        .spawn((
            Mesh3d(mesh.clone()),
            MeshMaterial3d(material.clone()),
            RadianceCascadesGeometry,
            InheritedVisibility::VISIBLE,
            GlobalTransform::from(Transform::from_xyz(0.0, 0.0, 1.0)),
        ))
        .id();
    // Visit this nearer shadow-panel before the real blocker. Its distinct
    // BLAS loops independently while the farther panel still proves blocked.
    let cycle_panel = app
        .world_mut()
        .spawn((
            Mesh3d(cycle_mesh),
            MeshMaterial3d(material.clone()),
            RadianceCascadesGeometry,
            InheritedVisibility::VISIBLE,
            GlobalTransform::from(
                Transform::from_xyz(2.5, 1.5, 2.0)
                    .with_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2)),
            ),
        ))
        .id();
    // This separated panel expands the scene root without covering the center
    // primary ray or the unblocked shadow ray at y=0.
    app.world_mut().spawn((
        Mesh3d(mesh),
        MeshMaterial3d(material),
        RadianceCascadesGeometry,
        InheritedVisibility::VISIBLE,
        GlobalTransform::from(
            Transform::from_xyz(5.0, 1.5, 2.0)
                .with_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2)),
        ),
    ));
    wait_for_blas_builds(&mut app, 2);
    wait_for_scene_instances(&mut app, 3);
    let data = app.world().resource::<GiScene>().data.as_ref().clone();
    let primary_instance = data.instance_indices[&primary_panel] as usize;
    let cycle_instance = data.instance_indices[&cycle_panel] as usize;

    let options = WgpuSettings::default();
    let resources = block_on(initialize_renderer(
        options.backends.unwrap_or(Backends::PRIMARY),
        None,
        &options,
    ));
    let device = &resources.0;
    let queue = &resources.1;
    let tracked = traversal_pipeline(device, "checkTrackedTraversal");
    let shader = SlangCompiler::default()
        .compile_bundle(
            "acceleration.slang",
            &[(
                "acceleration.slang",
                include_str!("../shaders/acceleration.slang"),
            )],
            &SlangSettings {
                target: SlangTarget::Wgsl,
                entry_points: vec!["refitTlas".into()],
                optimization: Some(2),
                ..default()
            },
        )
        .unwrap();
    let bevy::shader::Source::Wgsl(source) = shader.source else {
        panic!("expected WGSL");
    };
    let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
        label: Some("tracked_traversal_refit"),
        source: ShaderSource::Wgsl(source),
    });
    let layout = crate::render::refit_layout();
    let layout = device.create_bind_group_layout("tracked_traversal_refit", &layout.entries);
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("tracked_traversal_refit"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let refit = device.create_compute_pipeline(&RawComputePipelineDescriptor {
        label: Some("tracked_traversal_refit"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("refitTlas"),
        compilation_options: default(),
        cache: None,
    });
    let rays = [
        // The primary receiver is hit at z=1; the unrelated blocker is outside
        // that first-hit dependency.
        Ray {
            origin: Vec3::ZERO,
            direction: Vec3::Z,
            limit: 10.0,
            any_hit: false,
        },
        // A shadow ray through the separated panel is blocked.
        Ray {
            origin: Vec3::new(0.0, 1.5, 2.0),
            direction: Vec3::X,
            limit: 10.0,
            any_hit: false,
        },
        // The same scene-root segment at y=0 is clear. A newly inserted
        // blocker at (3,0,2) lies in this tracked interval.
        Ray {
            origin: Vec3::new(0.0, 0.0, 2.0),
            direction: Vec3::X,
            limit: 10.0,
            any_hit: false,
        },
        // A ray clipped out by the scene root is measured empty, not unknown.
        Ray {
            origin: Vec3::new(0.0, 4.0, 2.0),
            direction: Vec3::X,
            limit: 10.0,
            any_hit: false,
        },
    ];
    let bytes = trace_gpu_for_layers(
        &data,
        &rays,
        device,
        queue,
        &tracked,
        &refit,
        &RenderLayers::default(),
    );
    let word = |ray: usize, slot: usize, channel: usize| {
        let index = (ray * TRACKED_RESULT_SLOTS + slot) * 16 + channel * 4;
        u32::from_le_bytes(bytes[index..index + 4].try_into().unwrap())
    };
    let scalar = |ray: usize, slot: usize, channel: usize| f32::from_bits(word(ray, slot, channel));
    let contains = |ray: usize, point: Vec3| {
        (0..3).all(|axis| {
            let minimum = scalar(ray, 1, axis);
            let maximum = scalar(ray, 2, axis);
            point[axis] >= minimum - 1e-5 && point[axis] <= maximum + 1e-5
        })
    };

    for ray in 0..rays.len() {
        assert_eq!(
            [scalar(ray, 11, 0), scalar(ray, 11, 1), scalar(ray, 11, 2)],
            [2.0, 3.0, 5.0],
            "directional conventional light contribution belongs to BACKGROUND"
        );
        assert_eq!(
            [scalar(ray, 12, 0), scalar(ray, 12, 1), scalar(ray, 12, 2)],
            [0.0; 3],
            "conventional lights do not add radiance to EMISSIVE"
        );
        assert_eq!(
            [scalar(ray, 13, 0), scalar(ray, 13, 1), scalar(ray, 13, 2)],
            [24.0, 30.0, 36.0],
            "finite point and spot contributions belong to PUNCTUAL"
        );
        assert_eq!(
            [
                scalar(ray, 11, 0) + scalar(ray, 13, 0),
                scalar(ray, 11, 1) + scalar(ray, 13, 1),
                scalar(ray, 11, 2) + scalar(ray, 13, 2),
            ],
            [26.0, 33.0, 41.0],
            "directional, point, and spot contributions are each counted once"
        );
    }

    assert_ne!(word(0, 0, 1), u32::MAX, "primary ray must hit its receiver");
    assert_eq!(
        word(0, 1, 3),
        3,
        "primary hit dependency is bounded and known"
    );
    assert!(
        scalar(0, 2, 2) < 1.01,
        "primary dependency stops at its first hit"
    );
    assert!(
        !contains(0, Vec3::new(5.0, 1.5, 2.0)),
        "far panel is outside the primary dependency"
    );
    assert_eq!(
        word(0, 6, 3),
        1.0f32.to_bits(),
        "stored dependency marks the primary observation known"
    );
    assert_eq!(
        word(0, 7, 3),
        1.0f32.to_bits(),
        "stored primary dependency has bounds"
    );
    assert_eq!(word(0, 8, 3), 0, "unknown union clears the known marker");
    assert_eq!(word(0, 9, 3), 0, "unknown union cannot retain stale bounds");

    assert_eq!(word(1, 3, 0), 1, "the NEE shadow ray must be blocked");
    assert_eq!(word(1, 3, 1), 1, "the NEE query must be supported");
    assert_eq!(
        word(1, 3, 2),
        3,
        "blocked shadow dependency is bounded and known"
    );

    assert_eq!(word(2, 3, 0), 0, "the separated NEE ray remains unblocked");
    assert_eq!(word(2, 3, 1), 1, "the clear NEE query must be supported");
    assert_eq!(
        word(2, 3, 2),
        3,
        "clear in-root ray retains a measured segment"
    );
    assert_eq!(
        word(2, 0, 1),
        u32::MAX,
        "the in-root primary ray misses geometry"
    );
    assert_eq!(
        word(2, 1, 3),
        3,
        "in-root primary miss remains bounded and known"
    );
    assert!(
        contains(2, Vec3::new(3.0, 0.0, 2.0)),
        "dependency must cover a possible new blocker on the ray"
    );
    assert!(
        !contains(2, Vec3::splat(100.0)),
        "disjoint geometry must stay outside the dependency"
    );

    assert_eq!(word(3, 1, 3), 1, "root-clipped primary miss is known empty");
    assert_eq!(
        word(3, 6, 3),
        1.0f32.to_bits(),
        "known-empty dependency survives Entry serialization"
    );
    assert_eq!(word(3, 7, 3), 0, "known-empty dependency has no bounds");
    assert_eq!(word(3, 3, 0), 0, "root-clipped shadow ray is unblocked");
    assert_eq!(word(3, 3, 1), 1, "root-clipped shadow query is supported");
    assert_eq!(
        word(3, 3, 2),
        1,
        "root-clipped shadow dependency is known empty"
    );

    // Keep this endpoint-bias case isolated from the cyclic-BLAS blocker scene.
    let mut endpoint_app = App::new();
    endpoint_app
        .add_plugins(MinimalPlugins)
        .init_resource::<Assets<Mesh>>()
        .init_resource::<Assets<StandardMaterial>>()
        .init_resource::<GiScene>()
        .add_systems(Update, update_scene);
    endpoint_app
        .world_mut()
        .spawn((Camera::default(), RadianceCascades::default()));
    let endpoint_mesh = endpoint_app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(Rectangle::new(2.0, 2.0));
    let endpoint_material = endpoint_app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial {
            cull_mode: None,
            double_sided: true,
            emissive: bevy::color::LinearRgba::new(2.0, 1.0, 0.5, 1.0),
            ..default()
        });
    let root_material = endpoint_app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial {
            cull_mode: None,
            double_sided: true,
            ..default()
        });
    endpoint_app.world_mut().spawn((
        Mesh3d(endpoint_mesh.clone()),
        MeshMaterial3d(endpoint_material),
        RadianceCascadesGeometry,
        InheritedVisibility::VISIBLE,
        GlobalTransform::from(Transform::from_xyz(0.0, 0.0, 1.0)),
    ));
    endpoint_app.world_mut().spawn((
        Mesh3d(endpoint_mesh),
        MeshMaterial3d(root_material),
        RadianceCascadesGeometry,
        InheritedVisibility::VISIBLE,
        GlobalTransform::from(
            Transform::from_xyz(5.0, 1.5, 0.5)
                .with_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2)),
        ),
    ));
    wait_for_blas_builds(&mut endpoint_app, 1);
    wait_for_scene_instances(&mut endpoint_app, 2);
    let endpoint_data = endpoint_app
        .world()
        .resource::<GiScene>()
        .data
        .as_ref()
        .clone();
    let source = endpoint_data
        .emissive_lights
        .first()
        .expect("endpoint fixture includes an emissive source");
    let sample_uv = Vec2::new(0.25, 0.5);
    let root = sample_uv.x.sqrt();
    let emitter_target = source.position.truncate()
        + source.edge1.truncate() * (root * (1.0 - sample_uv.y))
        + source.edge2.truncate() * (root * sample_uv.y);
    let target_distance = emitter_target.length();
    let target_bias = (0.0001f32 * 0.25).min(target_distance * 0.001);
    let endpoint_rays = [Ray {
        origin: Vec3::ZERO,
        direction: emitter_target / target_distance,
        limit: target_distance - target_bias,
        any_hit: false,
    }];
    let endpoint_bytes = trace_gpu_for_layers_with_targets(
        &endpoint_data,
        (&endpoint_rays, &[Some(emitter_target)]),
        device,
        queue,
        &tracked,
        &refit,
        &RenderLayers::default(),
    );
    let endpoint_word = |slot: usize, channel: usize| {
        let index = slot * 16 + channel * 4;
        u32::from_le_bytes(endpoint_bytes[index..index + 4].try_into().unwrap())
    };
    assert_eq!(
        endpoint_word(3, 0),
        0,
        "biased NEE shadow query stops before the source"
    );
    assert_eq!(
        endpoint_word(3, 1),
        1,
        "biased NEE shadow query remains supported"
    );
    assert_eq!(
        endpoint_word(10, 0),
        0,
        "the unbiased target endpoint lies outside the biased visibility segment"
    );
    assert_eq!(
        endpoint_word(10, 1),
        1,
        "tracked visibility dependency includes the sampled emitter target"
    );
    let cyclic = forced_cyclic_blas_scene(&data, cycle_instance);
    let exhausted_rays = [
        Ray {
            origin: Vec3::new(0.0, 1.5, 2.0),
            direction: Vec3::X,
            limit: 3.0,
            any_hit: false,
        },
        Ray {
            origin: Vec3::new(0.0, 1.5, 2.0),
            direction: Vec3::X,
            limit: 10.0,
            any_hit: false,
        },
    ];
    let exhausted_bytes = trace_gpu_for_layers(
        &cyclic,
        &exhausted_rays,
        device,
        queue,
        &tracked,
        &refit,
        &RenderLayers::default(),
    );
    let exhausted_word = |ray: usize, slot: usize, channel: usize| {
        let index = (ray * TRACKED_RESULT_SLOTS + slot) * 16 + channel * 4;
        u32::from_le_bytes(exhausted_bytes[index..index + 4].try_into().unwrap())
    };
    assert_eq!(
        exhausted_word(0, 0, 3) & 4,
        4,
        "a first-hit ray through an exhausted BLAS is unsupported"
    );
    assert_eq!(
        exhausted_word(0, 3, 0),
        0,
        "an exhausted clear ray is not blocked"
    );
    assert_eq!(
        exhausted_word(0, 3, 1),
        0,
        "step-budget exhaustion is not a supported clear result"
    );
    assert_eq!(
        exhausted_word(0, 3, 2),
        0,
        "an exhausted shadow query leaves its tracked dependency unknown"
    );
    assert_eq!(
        exhausted_word(1, 3, 0),
        1,
        "a later accepted blocker still proves the ray blocked"
    );
    assert_eq!(
        exhausted_word(1, 3, 1),
        1,
        "a proven blocker remains supported after an earlier branch exhausts"
    );

    let oversized = forced_oversized_blas_leaf(&data, primary_instance);
    let leaf_ray = [Ray {
        // Travel parallel to the primary panel so its clipped primitive scan
        // finds no blocker, despite intersecting the leaf bounds.
        origin: Vec3::new(0.0, 0.0, 1.0),
        direction: Vec3::X,
        limit: 10.0,
        any_hit: false,
    }];
    let leaf_bytes = trace_gpu_for_layers(
        &oversized,
        &leaf_ray,
        device,
        queue,
        &tracked,
        &refit,
        &RenderLayers::default(),
    );
    let leaf_word = |slot: usize, channel: usize| {
        let index = slot * 16 + channel * 4;
        u32::from_le_bytes(leaf_bytes[index..index + 4].try_into().unwrap())
    };
    assert_eq!(
        leaf_word(0, 1),
        u32::MAX,
        "the overlarge leaf fixture misses its actual primitives"
    );
    assert_eq!(
        leaf_word(0, 3) & 4,
        4,
        "a clipped closest-hit leaf cannot publish a supported miss"
    );
    assert_eq!(
        leaf_word(3, 0),
        0,
        "a clipped shadow leaf miss remains unblocked"
    );
    assert_eq!(
        leaf_word(3, 1),
        0,
        "a clipped shadow leaf miss is unsupported"
    );
    assert_eq!(
        leaf_word(3, 2),
        0,
        "a clipped shadow leaf miss keeps dependency bounds unknown"
    );
}

#[test]
#[ignore = "requires slangc and a desktop GPU; no scene or renderer app is launched"]
fn gpu_linked_sort_matches_stdlib_sort() {
    let options = WgpuSettings::default();
    let resources = block_on(initialize_renderer(
        options.backends.unwrap_or(Backends::PRIMARY),
        None,
        &options,
    ));
    let device = &resources.0;
    let queue = &resources.1;
    let lengths = [0, 1, 2, 3, 5, 17, 63, 64, 129, 257];
    let mut lists = Vec::new();
    let mut nodes = Vec::new();
    let mut expected = Vec::new();
    for length in lengths {
        let base = nodes.len() as u32;
        lists.push(UVec2::new(
            if length == 0 { u32::MAX } else { base },
            length,
        ));
        for i in 0..length {
            let key = ((i.wrapping_mul(1664525).wrapping_add(1013904223) >> 16) % 13) as i32 - 6;
            nodes.push(UVec4::new(
                key as u32,
                if i + 1 == length {
                    u32::MAX
                } else {
                    base + i + 1
                },
                0,
                0,
            ));
        }
        let mut order = (base..base + length).collect::<Vec<_>>();
        order.sort_by_key(|index| (nodes[*index as usize].x as i32, *index));
        expected.push(order);
    }
    let mut input = StorageBuffer::from(lists);
    input.write_buffer(device, queue);
    let mut links = StorageBuffer::from(nodes);
    links.write_buffer(device, queue);
    let bytes = lengths.len() as u64 * 512 * 4;
    let output = device.create_buffer(&BufferDescriptor {
        label: Some("linked_sort_check"),
        size: bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let pipeline = traversal_pipeline(device, "checkLinkedSort");
    let group = device.create_bind_group(
        "linked_sort_check",
        &pipeline.get_bind_group_layout(0).into(),
        &[
            BindGroupEntry {
                binding: 0,
                resource: input.binding().unwrap(),
            },
            BindGroupEntry {
                binding: 1,
                resource: links.binding().unwrap(),
            },
            BindGroupEntry {
                binding: 2,
                resource: output.as_entire_binding(),
            },
        ],
    );
    let result = dispatch_readback(device, queue, &pipeline, &group, &output, bytes, 1);
    let indices = result
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect::<Vec<_>>();
    for (case, order) in expected.iter().enumerate() {
        assert_eq!(&indices[case * 512..case * 512 + order.len()], order);
        assert_eq!(
            indices[case * 512 + order.len()],
            u32::MAX,
            "list must terminate"
        );
    }
}

#[test]
#[ignore = "requires slangc and a desktop GPU; no demo scene is launched"]
fn gpu_shared_edges_and_origin_offsets_are_scale_aware() {
    let options = WgpuSettings::default();
    let resources = block_on(initialize_renderer(
        options.backends.unwrap_or(Backends::PRIMARY),
        None,
        &options,
    ));
    let device = &resources.0;
    let queue = &resources.1;
    let mut queries = Vec::new();
    let mut expected = Vec::new();
    for scale in [1e-12f32, 1e-6, 1.0, 1e6, 1e12] {
        for sign in [-1.0, 1.0] {
            for upper in [false, true] {
                let a = Vec3::new(-scale, -scale, 0.0);
                let diagonal = Vec3::new(scale, scale, 0.0);
                let b = if upper {
                    diagonal
                } else {
                    Vec3::new(scale, -scale, 0.0)
                };
                let c = if upper {
                    Vec3::new(-scale, scale, 0.0)
                } else {
                    diagonal
                };
                let origin = Vec3::new(0.0, 0.0, scale * sign);
                let shorter_limit = if upper {
                    scale * if sign < 0.0 { 1.0 } else { 2.0 }
                } else {
                    scale * 0.5
                };
                queries.extend([
                    a.extend(0.0),
                    b.extend(0.0),
                    c.extend(0.0),
                    origin.extend(scale * 2.0),
                    Vec3::new(0.0, 0.0, -sign).extend(shorter_limit),
                    Vec3::new(0.0, 0.0, sign).extend(0.0),
                ]);
                expected.push((scale, origin, sign, upper));
            }
        }
    }
    let mut input = StorageBuffer::from(queries);
    input.write_buffer(device, queue);
    let bytes = expected.len() as u64 * 32;
    let output = device.create_buffer(&BufferDescriptor {
        label: Some("ray_geometry_check"),
        size: bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let pipeline = traversal_pipeline(device, "checkRayGeometry");
    let group = device.create_bind_group(
        "ray_geometry_check",
        &pipeline.get_bind_group_layout(0).into(),
        &[
            BindGroupEntry {
                binding: 3,
                resource: input.binding().unwrap(),
            },
            BindGroupEntry {
                binding: 4,
                resource: output.as_entire_binding(),
            },
        ],
    );
    let bytes = dispatch_readback(device, queue, &pipeline, &group, &output, bytes, 1);
    let floats = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect::<Vec<_>>();
    for (case, (scale, origin, sign, should_intersect)) in expected.iter().enumerate() {
        let result = &floats[case * 8..case * 8 + 8];
        assert!(
            (result[0] / scale - 1.0).abs() < 1e-5,
            "shared edge missed at scale {scale}: {result:?}"
        );
        assert!(result[1] >= 0.0 && result[2] >= 0.0 && result[1] + result[2] <= 1.0);
        assert_eq!(result[3], if *should_intersect { 7.0 } else { 1.0 });
        assert!(result[4..7].iter().all(|v| v.is_finite()));
        assert!(
            (result[6] - origin.z) * sign > 0.0,
            "offset must move to the normal side at scale {scale}"
        );
    }
}

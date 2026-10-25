//! GPU regressions for the production sparse Split resolve helpers.
use bevy::math::{IVec3, Vec2};
use super::*;

const DIRECTORY_CAPACITY: u32 = 64;
const CACHE_BUCKETS: u32 = 16;
const ENTRY_WORDS: usize = 208 / 4;
const PROBE_WORDS: usize = crate::render::PROBE_SIZE as usize / 4;
const PROBE_LOD_MASK: u32 = 31 << 18;

#[derive(Clone)]
struct Probe {
    key: [i32; 4],
    radiance: [f32; 3],
    directions: u32,
}

fn bin_direction(index: u32) -> Vec3 {
    let x = index % 8;
    let y = index / 8;
    let phi = (x as f32 + 0.5) * std::f32::consts::TAU / 8.0;
    let vertical = (y as f32 + 0.5) * 0.5 - 1.0;
    let radius = (1.0 - vertical * vertical).max(0.0).sqrt();
    Vec3::new(radius * phi.cos(), vertical, radius * phi.sin())
}

fn decode_octahedral(uv: Vec2) -> Vec3 {
    let p = uv * 2.0 - Vec2::ONE;
    let mut normal = Vec3::new(p.x, p.y, 1.0 - p.x.abs() - p.y.abs());
    let fold = (-normal.z).clamp(0.0, 1.0);
    normal.x += if normal.x >= 0.0 { -fold } else { fold };
    normal.y += if normal.y >= 0.0 { -fold } else { fold };
    normal.normalize()
}

fn encode_octahedral(normal: Vec3) -> Vec2 {
    let mut normal = normal.normalize();
    normal /= normal.x.abs() + normal.y.abs() + normal.z.abs();
    let mut p = Vec2::new(normal.x, normal.y);
    if normal.z < 0.0 {
        p = Vec2::new(1.0 - p.y.abs(), 1.0 - p.x.abs())
            * Vec2::new(if p.x >= 0.0 { 1.0 } else { -1.0 }, if p.y >= 0.0 { 1.0 } else { -1.0 });
    }
    p * 0.5 + Vec2::splat(0.5)
}

fn wrap_octahedral_texel(mut pixel: [i32; 2]) -> [i32; 2] {
    if pixel[0] < 0 {
        pixel = [-pixel[0] - 1, 5 - pixel[1]];
    } else if pixel[0] >= 6 {
        pixel = [11 - pixel[0], 5 - pixel[1]];
    }
    if pixel[1] < 0 {
        pixel = [5 - pixel[0], -pixel[1] - 1];
    } else if pixel[1] >= 6 {
        pixel = [5 - pixel[0], 11 - pixel[1]];
    }
    pixel
}

fn reference_probe_table(probe: &Probe) -> [[f32; 4]; 36] {
    std::array::from_fn(|texel| {
        let normal = decode_octahedral(Vec2::new(
            (texel % 6) as f32 / 6.0 + 0.5 / 6.0,
            (texel / 6) as f32 / 6.0 + 0.5 / 6.0,
        ));
        let mut radiance_sum = [0.0; 3];
        let mut nominal_cosine = 0.0;
        let mut observed_cosine = 0.0;
        for direction in 0..32 {
            let cosine = normal.dot(bin_direction(direction)).max(0.0);
            nominal_cosine += cosine;
            if probe.directions & (1 << direction) == 0 {
                continue;
            }
            observed_cosine += cosine;
            for (sum, component) in radiance_sum.iter_mut().zip(probe.radiance) {
                *sum += component * cosine;
            }
        }
        if observed_cosine <= 1e-6 || nominal_cosine <= 1e-6 {
            return [0.0; 4];
        }
        let scale = nominal_cosine * (4.0 / 32.0) / observed_cosine;
        [
            radiance_sum[0] * scale,
            radiance_sum[1] * scale,
            radiance_sum[2] * scale,
            (observed_cosine / nominal_cosine).clamp(0.0, 1.0),
        ]
    })
}

fn reference_angular_lookup(table: &[[f32; 4]; 36], normal: Vec3) -> ([f32; 3], f32) {
    let grid = encode_octahedral(normal) * 6.0 - Vec2::splat(0.5);
    let base = [grid.x.floor() as i32, grid.y.floor() as i32];
    let fraction = Vec2::new(grid.x - grid.x.floor(), grid.y - grid.y.floor());
    let mut sum = [0.0; 3];
    let mut observed = 0.0;
    for corner in 0..4 {
        let bit = [(corner & 1) as i32, (corner >> 1) as i32];
        let axis = [
            if bit[0] == 0 { 1.0 - fraction.x } else { fraction.x },
            if bit[1] == 0 { 1.0 - fraction.y } else { fraction.y },
        ];
        let bilinear = axis[0] * axis[1];
        let pixel = wrap_octahedral_texel([base[0] + bit[0], base[1] + bit[1]]);
        let sample = table[(pixel[1] * 6 + pixel[0]) as usize];
        let weight = bilinear * sample[3].clamp(0.0, 1.0);
        for (sum, component) in sum.iter_mut().zip(&sample[..3]) {
            *sum += component * weight;
        }
        observed += weight;
    }
    if observed <= 1e-6 {
        ([0.0; 3], 0.0)
    } else {
        (sum.map(|component| component / observed), observed.clamp(0.0, 1.0))
    }
}

fn reference_diffuse_at_lod(
    probes: &[Probe],
    position: Vec3,
    normal: Vec3,
    lod: u32,
) -> ([f32; 3], f32) {
    let spacing = 2.0f32.powi(lod as i32);
    let grid = position / spacing - Vec3::splat(0.5);
    let base = grid.floor().as_ivec3();
    let fraction = grid - grid.floor();
    let mut sum = [0.0; 3];
    let mut observed = 0.0;
    let mut potential = 0.0;
    for corner in 0..8 {
        let bit = IVec3::new(
            (corner & 1) as i32,
            ((corner >> 1) & 1) as i32,
            ((corner >> 2) & 1) as i32,
        );
        let axis = Vec3::new(
            if bit.x == 0 { 1.0 - fraction.x } else { fraction.x },
            if bit.y == 0 { 1.0 - fraction.y } else { fraction.y },
            if bit.z == 0 { 1.0 - fraction.z } else { fraction.z },
        );
        let spatial = axis.x * axis.y * axis.z;
        if spatial <= 1e-6 {
            continue;
        }
        potential += spatial;
        let key = [
            base.x + bit.x,
            base.y + bit.y,
            base.z + bit.z,
            (lod << 18) as i32,
        ];
        let Some(probe) = probes.iter().find(|probe| probe.key == key) else {
            continue;
        };
        let table = reference_probe_table(probe);
        let (irradiance, coverage) = reference_angular_lookup(&table, normal);
        if coverage <= 0.0 {
            continue;
        }
        let weight = spatial * coverage;
        for (sum, component) in sum.iter_mut().zip(irradiance) {
            *sum += component * weight;
        }
        observed += weight;
    }
    if observed <= 1e-6 || potential <= 0.0 {
        ([0.0; 3], 0.0)
    } else {
        (sum.map(|component| component / observed), (observed / potential).clamp(0.0, 1.0))
    }
}

fn reference_split_diffuse(
    probes: &[Probe],
    position: Vec3,
    normal: Vec3,
    perspective: bool,
) -> ([f32; 3], f32) {
    let lod_scale = if perspective {
        position.abs().max_element() / 1.0
    } else {
        1.0
    };
    let low_lod = lod_scale.max(1.0).log2().floor().clamp(0.0, 4.0) as u32;
    let mut high_lod = low_lod;
    let mut blend = 0.0;
    if low_lod < 4 {
        let next_start = 2.0f32.powi((low_lod + 1) as i32);
        let overlap_start = 0.9 * next_start;
        if lod_scale >= overlap_start && lod_scale < next_start {
            high_lod = low_lod + 1;
            blend = ((lod_scale - overlap_start) / (next_start - overlap_start)).clamp(0.0, 1.0);
        }
    }
    let (low, low_coverage) = reference_diffuse_at_lod(probes, position, normal, low_lod);
    let low_weight = low_coverage * (1.0 - blend);
    let mut sum = low.map(|component| component * low_weight);
    let mut total = low_weight;
    if high_lod != low_lod {
        let (high, high_coverage) = reference_diffuse_at_lod(probes, position, normal, high_lod);
        let high_weight = high_coverage * blend;
        for (sum, component) in sum.iter_mut().zip(high) {
            *sum += component * high_weight;
        }
        total += high_weight;
    }
    if total <= 1e-6 {
        ([0.0; 3], 0.0)
    } else {
        (sum.map(|component| component / total), total.clamp(0.0, 1.0))
    }
}

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start = source.find(start).expect("production shader section start");
    let tail = &source[start..];
    &tail[..tail.find(end).expect("production shader section end")]
}

fn resolve_pipeline(device: &bevy::render::renderer::RenderDevice) -> ComputePipeline {
    let split = include_str!("../shaders/split.slang");
    let resolve = include_str!("../shaders/world_resolve.slang");
    let mut source = String::from(
        r#"
#include "common.slang"
[[vk::binding(0, 0)]] ConstantBuffer<Parameters> params;
[[vk::binding(6, 0)]] RWStructuredBuffer<Atomic<uint>> heads;
[[vk::binding(7, 0)]] RWStructuredBuffer<Entry> entries;
[[vk::binding(8, 0)]] StructuredBuffer<uint> previousHeads;
[[vk::binding(9, 0)]] RWStructuredBuffer<Entry> previousEntries;
[[vk::binding(12, 0)]] RWStructuredBuffer<DirectionalProbe> directionalProbes;
[[vk::binding(13, 0)]] RWStructuredBuffer<float4> results;
[[vk::binding(14, 0)]] StructuredBuffer<float4> queryPositions;
[[vk::binding(15, 0)]] StructuredBuffer<float4> queryNormals;
"#,
    );
    for production in [
        section(
            split,
            "bool dependencyIntersectsDirty(",
            "float splitIntervalEnd(",
        ),
        section(
            split,
            "uint probeDirectoryCapacity(",
            "uint probeDirectoryBase(",
        ),
        section(
            split,
            "uint probeDirectoryBase(",
            "uint probeDirectoryHead(",
        ),
        section(split, "uint probeDirectoryHead(", "uint probeCountHead("),
        section(
            split,
            "uint probeDirectoryBucket(",
            "uint probeDescriptorIndex(",
        ),
        section(
            split,
            "uint findIrradianceProbe(",
            "float4 interpolateMergedChildren(",
        ),
        section(resolve, "int2 wrapIrradianceTexel(", "[shader"),
    ] {
        source.push_str(production);
        source.push('\n');
    }
    source.push_str(
        r#"
[shader("compute")]
[numthreads(64, 1, 1)]
void resolveCheck(uint3 id : SV_DispatchThreadID) {
    uint count, stride;
    queryPositions.GetDimensions(count, stride);
    if (id.x >= count) return;
    float coverage;
    float3 value = resolveSplitDiffuse(queryPositions[id.x].xyz,
        normalize(queryNormals[id.x].xyz), coverage);
    bool supported = coverage > 1e-6 && all(isfinite(value));
    results[id.x] = float4(value, rawGiSupportAlpha(true, supported, coverage));
}
"#,
    );
    let shader = SlangCompiler::default()
        .compile_bundle(
            "world_resolve_check.slang",
            &[
                ("common.slang", include_str!("../shaders/common.slang")),
                ("world_resolve_check.slang", &source),
            ],
            &SlangSettings {
                target: SlangTarget::Wgsl,
                entry_points: vec!["resolveCheck".to_owned()],
                optimization: Some(2),
                ..default()
            },
        )
        .unwrap();
    let bevy::shader::Source::Wgsl(source) = shader.source else {
        panic!("expected WGSL from the Split world-resolve regression shader");
    };
    let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
        label: Some("world_resolve_regression"),
        source: ShaderSource::Wgsl(source),
    });
    device.create_compute_pipeline(&RawComputePipelineDescriptor {
        label: Some("world_resolve_regression"),
        layout: None,
        module: &module,
        entry_point: Some("resolveCheck"),
        compilation_options: default(),
        cache: None,
    })
}

fn insert_probes(
    probes: &mut Vec<Probe>,
    position: Vec3,
    lod: u32,
    radiance: [f32; 3],
    directions: u32,
    corners: u8,
) {
    let spacing = 2.0f32.powi(lod as i32);
    let grid = (position / spacing - Vec3::splat(0.5)).floor();
    let base = [grid.x as i32, grid.y as i32, grid.z as i32];
    for corner in 0..8 {
        if corners & (1 << corner) == 0 {
            continue;
        }
        let key = [
            base[0] + (corner & 1),
            base[1] + ((corner >> 1) & 1),
            base[2] + ((corner >> 2) & 1),
            (lod << 18) as i32,
        ];
        if let Some(probe) = probes.iter_mut().find(|probe| probe.key == key) {
            probe.radiance = radiance;
            probe.directions = directions;
        } else {
            probes.push(Probe {
                key,
                radiance,
                directions,
            });
        }
    }
}

fn words_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn run_resolve(
    device: &bevy::render::renderer::RenderDevice,
    queue: &bevy::render::renderer::RenderQueue,
    pipeline: &ComputePipeline,
    parameters: TransportParameters,
    probes: &[Probe],
    positions: &[Vec3],
) -> Vec<[f32; 4]> {
    let normals = vec![Vec3::Y; positions.len()];
    run_resolve_with_normals(device, queue, pipeline, parameters, probes, positions, &normals)
}

fn run_resolve_with_normals(
    device: &bevy::render::renderer::RenderDevice,
    queue: &bevy::render::renderer::RenderQueue,
    pipeline: &ComputePipeline,
    mut parameters: TransportParameters,
    probes: &[Probe],
    positions: &[Vec3],
    normals: &[Vec3],
) -> Vec<[f32; 4]> {
    assert_eq!(positions.len(), normals.len());
    parameters.size = UVec4::new(1, 1, 16, CACHE_BUCKETS);
    parameters.viewport = UVec4::new(0, 0, 4, 2);
    parameters.tuning = Vec4::new(1.0, 1.6, 0.0, 1.0);
    parameters.owner_sequence.x = 1;
    parameters.frame.x = 0;
    let mut parameters = UniformBuffer::from(parameters);
    parameters.write_buffer(device, queue);

    let mut heads = vec![u32::MAX; (2 * CACHE_BUCKETS + 36 + 8 * DIRECTORY_CAPACITY) as usize];
    let mut entries = vec![0u32; probes.len().max(1) * ENTRY_WORDS];
    let mut directional = vec![0u32; (2 * DIRECTORY_CAPACITY as usize) * PROBE_WORDS];
    let base = 2 * CACHE_BUCKETS + 36;
    for (index, probe) in probes.iter().enumerate() {
        for (word, key) in probe.key.iter().enumerate() {
            entries[index * ENTRY_WORDS + word] = *key as u32;
        }
        let key = [
            probe.key[0] as u32,
            probe.key[1] as u32,
            probe.key[2] as u32,
            probe.key[3] as u32 & (PROBE_LOD_MASK | (3 << 12)),
        ];
        let mut hash = key[0].wrapping_mul(0x8da6_b343)
            ^ key[1].wrapping_mul(0xd816_3841)
            ^ key[2].wrapping_mul(0xcb1a_b31f)
            ^ key[3].wrapping_mul(0x1656_67b1);
        hash ^= hash >> 16;
        hash = hash.wrapping_mul(0x7feb_352d);
        hash ^= hash >> 15;
        let mut slot = (hash & (DIRECTORY_CAPACITY - 1)) as usize;
        while heads[base as usize + slot] != u32::MAX {
            slot = (slot + 1) & (DIRECTORY_CAPACITY as usize - 1);
        }
        heads[base as usize + slot] = index as u32;
        for (texel, value) in reference_probe_table(probe).iter().enumerate() {
            let sample = slot * PROBE_WORDS + texel * 4;
            for (component, value) in value.iter().enumerate() {
                directional[sample + component] = value.to_bits();
            }
        }
    }
    let heads = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("world_resolve_heads"),
        contents: &words_bytes(&heads),
        usage: BufferUsages::STORAGE,
    });
    let entries = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("world_resolve_entries"),
        contents: &words_bytes(&entries),
        usage: BufferUsages::STORAGE,
    });
    let previous_heads = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("world_resolve_previous_heads"),
        contents: &words_bytes(&vec![u32::MAX; heads.size() as usize / 4]),
        usage: BufferUsages::STORAGE,
    });
    let previous_entries = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("world_resolve_previous_entries"),
        contents: &words_bytes(&vec![0; ENTRY_WORDS]),
        usage: BufferUsages::STORAGE,
    });
    let directional = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("world_resolve_directional_probes"),
        contents: &words_bytes(&directional),
        usage: BufferUsages::STORAGE,
    });
    let query_bytes: Vec<u8> = positions
        .iter()
        .flat_map(|position| {
            position
                .extend(1.0)
                .to_array()
                .into_iter()
                .flat_map(f32::to_le_bytes)
        })
        .collect();
    let queries = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("world_resolve_queries"),
        contents: &query_bytes,
        usage: BufferUsages::STORAGE,
    });
    let normal_bytes: Vec<u8> = normals
        .iter()
        .flat_map(|normal| normal.extend(0.0).to_array().into_iter().flat_map(f32::to_le_bytes))
        .collect();
    let query_normals = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("world_resolve_query_normals"),
        contents: &normal_bytes,
        usage: BufferUsages::STORAGE,
    });
    let output_bytes = (positions.len() * 16) as u64;
    let output = device.create_buffer(&BufferDescriptor {
        label: Some("world_resolve_results"),
        size: output_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let group = device.create_bind_group(
        "world_resolve_check",
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
                binding: 8,
                resource: previous_heads.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 9,
                resource: previous_entries.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 12,
                resource: directional.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 13,
                resource: output.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 14,
                resource: queries.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 15,
                resource: query_normals.as_entire_binding(),
            },
        ],
    );
    let bytes = crate::scene::traversal_tests::dispatch_readback(
        device,
        queue,
        pipeline,
        &group,
        &output,
        output_bytes,
        1,
    );
    (0..positions.len())
        .map(|index| output_vec4(&bytes, index))
        .collect()
}

fn orthographic_parameters() -> TransportParameters {
    let mut parameters = temporal_test_parameters(0);
    parameters.clip_from_world = bevy::math::proj::orthographic(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0);
    parameters.camera_position = Vec4::ZERO;
    parameters
}

fn perspective_parameters() -> TransportParameters {
    let mut parameters = temporal_test_parameters(0);
    parameters.clip_from_world =
        bevy::math::proj::perspective_infinite_reverse(std::f32::consts::FRAC_PI_2, 1.0, 0.1);
    parameters.camera_position = Vec4::ZERO;
    parameters
}

fn close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() <= expected.abs().max(1.0) * 2e-4,
        "actual {actual}, expected {expected}"
    );
}

#[test]
#[ignore = "requires slangc and a desktop GPU; dispatches production Split resolve helpers"]
fn gpu_world_resolve_normalizes_sparse_support_and_isolates_lods() {
    let options = WgpuSettings::default();
    let resources = block_on(initialize_renderer(
        options.backends.unwrap_or(Backends::PRIMARY),
        None,
        &options,
    ));
    let device = &resources.0;
    let queue = &resources.1;
    let pipeline = resolve_pipeline(device);

    let directions = 0x33u32 << 16 | 0x33u32 << 24;
    let mut probes = Vec::new();
    insert_probes(&mut probes, Vec3::ZERO, 0, [2.0; 3], u32::MAX, u8::MAX);
    insert_probes(
        &mut probes,
        Vec3::new(4.0, 0.0, 0.0),
        0,
        [2.0; 3],
        u32::MAX,
        0x33,
    );
    insert_probes(
        &mut probes,
        Vec3::new(8.0, 0.0, 0.0),
        0,
        [2.0; 3],
        directions,
        u8::MAX,
    );
    insert_probes(
        &mut probes,
        Vec3::new(16.0, 0.0, 0.0),
        0,
        [0.0; 3],
        u32::MAX,
        u8::MAX,
    );
    let result = run_resolve(
        device,
        queue,
        &pipeline,
        orthographic_parameters(),
        &probes,
        &[
            Vec3::ZERO,
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(8.0, 0.0, 0.0),
            Vec3::new(16.0, 0.0, 0.0),
            Vec3::new(20.0, 0.0, 0.0),
        ],
    );
    let positions = [
        Vec3::ZERO,
        Vec3::new(4.0, 0.0, 0.0),
        Vec3::new(8.0, 0.0, 0.0),
        Vec3::new(16.0, 0.0, 0.0),
        Vec3::new(20.0, 0.0, 0.0),
    ];
    for (actual, position) in result.iter().zip(positions) {
        let (expected, coverage) = reference_split_diffuse(&probes, position, Vec3::Y, false);
        for (actual, expected) in actual.iter().zip(expected) {
            close(*actual, expected);
        }
        close(
            actual[3],
            if coverage > 1e-6 {
                1.0 + 0.25 * coverage
            } else {
                -1.0
            },
        );
    }
    assert!(result[3][..3].iter().all(|component| *component == 0.0));
    assert!(result[3][3] > 1.0, "supported black remains supported");
    assert_eq!(result[4][3], -1.0, "missing probes remain unsupported");

    let overlap = Vec3::new(0.0, 0.0, -1.9);
    for (lod, radiance) in [(0, [2.0; 3]), (1, [6.0; 3])] {
        let mut field = Vec::new();
        insert_probes(&mut field, overlap, lod, radiance, u32::MAX, u8::MAX);
        let result = run_resolve(
            device,
            queue,
            &pipeline,
            perspective_parameters(),
            &field,
            &[overlap],
        )[0];
        let (expected, coverage) = reference_split_diffuse(&field, overlap, Vec3::Y, true);
        for (actual, expected) in result.iter().zip(expected) {
            close(*actual, expected);
        }
        close(result[3], 1.0 + 0.25 * coverage);
    }
    let mut both_lods = Vec::new();
    insert_probes(&mut both_lods, overlap, 0, [2.0; 3], u32::MAX, u8::MAX);
    insert_probes(&mut both_lods, overlap, 1, [6.0; 3], u32::MAX, u8::MAX);
    let result = run_resolve(
        device,
        queue,
        &pipeline,
        perspective_parameters(),
        &both_lods,
        &[overlap],
    )[0];
    let (expected, coverage) = reference_split_diffuse(&both_lods, overlap, Vec3::Y, true);
    for (actual, expected) in result.iter().zip(expected) {
        close(*actual, expected);
    }
    close(result[3], 1.0 + 0.25 * coverage);

    let mut angular_field = Vec::new();
    insert_probes(
        &mut angular_field,
        Vec3::ZERO,
        0,
        [2.0, 1.0, 3.0],
        0xA55A_C33C,
        u8::MAX,
    );
    let normals = [
        Vec3::Y,
        Vec3::new(0.3, 0.2, 0.9).normalize(),
        Vec3::new(0.001, 0.001, -1.0).normalize(),
        Vec3::new(-0.001, -0.001, -1.0).normalize(),
    ];
    let angular = run_resolve_with_normals(
        device,
        queue,
        &pipeline,
        orthographic_parameters(),
        &angular_field,
        &[Vec3::ZERO; 4],
        &normals,
    );
    let table = reference_probe_table(&angular_field[0]);
    for (actual, normal) in angular.iter().zip(normals) {
        let (expected, coverage) = reference_angular_lookup(&table, normal);
        for (actual, expected) in actual.iter().zip(expected) {
            close(*actual, expected);
        }
        close(actual[3], 1.0 + 0.25 * coverage);
        assert!(coverage > 0.0, "the fixture must retain angular support");
    }
    for (a, b) in angular[2].iter().zip(angular[3].iter()).take(3) {
        assert!(
            (a - b).abs() < 0.02,
            "octahedral south-pole seam must be continuous: {:?} vs {:?}",
            angular[2],
            angular[3]
        );
    }

    let mut zero_coverage = Vec::new();
    insert_probes(&mut zero_coverage, Vec3::ZERO, 0, [100.0; 3], 0, u8::MAX);
    let zero = run_resolve(
        device,
        queue,
        &pipeline,
        orthographic_parameters(),
        &zero_coverage,
        &[Vec3::ZERO],
    )[0];
    assert_eq!(zero, [0.0, 0.0, 0.0, -1.0]);
}

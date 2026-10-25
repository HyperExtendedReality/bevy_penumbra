//! GPU regressions for the production sparse Split resolve helpers.
use super::*;

const DIRECTORY_CAPACITY: u32 = 64;
const CACHE_BUCKETS: u32 = 16;
const ENTRY_WORDS: usize = 208 / 4;
const PROBE_WORDS: usize = 544 / 4;
const PROBE_LOD_MASK: u32 = 31 << 18;

#[derive(Clone)]
struct Probe {
    key: [i32; 4],
    radiance: [f32; 3],
    directions: u32,
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
        section(resolve, "float3 diffuseProbeIrradiance(", "[shader"),
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
    float3 value = resolveSplitDiffuse(queryPositions[id.x].xyz, float3(0, 1, 0), coverage);
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
    mut parameters: TransportParameters,
    probes: &[Probe],
    positions: &[Vec3],
) -> Vec<[f32; 4]> {
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
        for direction in 0..32 {
            let sample = slot * PROBE_WORDS + direction * 4;
            for (component, value) in probe.radiance.iter().enumerate() {
                directional[sample + component] = value.to_bits();
            }
            directional[sample + 3] = if probe.directions & (1 << direction) != 0 {
                1.0f32.to_bits()
            } else {
                0.0f32.to_bits()
            };
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
    for &component in result[0].iter().take(3) {
        close(component, 2.0);
    }
    close(result[0][3], 1.25);
    for &component in result[1].iter().take(3) {
        close(component, 2.0);
    }
    close(result[1][3], 1.125);
    for &component in result[2].iter().take(3) {
        close(component, 2.0);
    }
    close(result[2][3], 1.125);
    for &component in result[3].iter().take(3) {
        close(component, 0.0);
    }
    close(result[3][3], 1.25); // Valid black is supported.
    for &component in result[4].iter().take(3) {
        close(component, 0.0);
    }
    close(result[4][3], -1.0); // Missing probes are unsupported.

    let overlap = Vec3::new(0.0, 0.0, -1.9);
    for (lod, radiance, expected) in [(0, [2.0; 3], 2.0), (1, [6.0; 3], 6.0)] {
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
        for &component in result.iter().take(3) {
            close(component, expected);
        }
        close(result[3], 1.125); // The other LOD is absent, so support is one half.
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
    for &component in result.iter().take(3) {
        close(component, 4.0);
    }
    close(result[3], 1.25);
}

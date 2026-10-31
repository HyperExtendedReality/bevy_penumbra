//! Optional hardware ray-query traversal.
//!
//! `traceDeferredHardware` and `traceVisibilityHardware` replace the software
//! BVH walk with wgpu ray queries. The structures here mirror the live GPU
//! scene: one BLAS per instance triangle range, built from the positions in the
//! scene geometry buffer, and a TLAS whose custom data is the packed instance
//! index. Non-opaque BLAS candidates are re-tested with the software triangle
//! tests, keeping the software hit semantics; hardware-opaque instances (see
//! `hardware_opaque` in scene.rs) need no test and commit in hardware.

use std::collections::HashMap;

use bevy::{
    prelude::*,
    render::{
        render_resource::{Buffer, BufferId, BufferInitDescriptor, BufferUsages},
        renderer::{RenderDevice, RenderQueue},
    },
};

use super::{GpuScene, INSTANCE_FLAG_ACTIVE_DEFORMATION, INSTANCE_WORDS_PER_RECORD, Pipelines};
use crate::{RadianceCascades, scene::GiScene, scene::VERTEX_WORDS};

/// TLAS custom data holds the instance index in 24 bits.
const MAX_HARDWARE_INSTANCES: usize = 1 << 24;

struct HardwareBlas {
    blas: wgpu::Blas,
    size: wgpu::BlasTriangleGeometrySizeDescriptor,
    /// Deformed ranges are rewritten on the GPU every frame and rebuilt with them.
    deformed: bool,
}

#[derive(Resource)]
pub(super) struct HardwareTraversal {
    tlas: wgpu::Tlas,
    capacity: u32,
    /// Geometry buffer and mesh revision that `indices` and `blas` describe.
    source: Option<(BufferId, u64)>,
    indices: Option<Buffer>,
    /// Keyed by the instance's (first triangle, triangle count, hardware opaque).
    blas: HashMap<(u32, u32, bool), HardwareBlas>,
    /// Instance words the TLAS was last built from.
    instances: Vec<UVec4>,
    /// Scene revision the TLAS describes; `None` until it matches the live scene.
    built: Option<u64>,
}

impl HardwareTraversal {
    pub(super) fn tlas(&self) -> &wgpu::Tlas {
        &self.tlas
    }

    /// Whether the TLAS describes the scene revision the GI dispatch will use.
    pub(super) fn ready_for(&self, revision: u64) -> bool {
        self.built == Some(revision)
    }
}

fn create_tlas(device: &RenderDevice, queue: &RenderQueue, capacity: u32) -> wgpu::Tlas {
    let tlas = device
        .wgpu_device()
        .create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("rc_hardware_tlas"),
            max_instances: capacity,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        });
    // Bind groups reference the TLAS before the scene is ready; an empty build
    // keeps every use valid until real instances arrive.
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rc_hardware_tlas_init"),
    });
    encoder.build_acceleration_structures(std::iter::empty(), [&tlas]);
    queue.submit([encoder.finish()]);
    tlas
}

/// Keeps a TLAS large enough for the scene before bind groups are created.
pub(super) fn prepare_hardware_traversal(
    mut commands: Commands,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    pipelines: Option<Res<Pipelines>>,
    scene: Res<GiScene>,
    hardware: Option<ResMut<HardwareTraversal>>,
) {
    if !pipelines.is_some_and(|pipelines| pipelines.ray_query) {
        return;
    }
    let needed = (scene.data.instances.len().min(MAX_HARDWARE_INSTANCES) as u32).max(1);
    match hardware {
        Some(hardware) if hardware.capacity >= needed => {}
        Some(mut hardware) => {
            let capacity = needed.next_power_of_two();
            hardware.tlas = create_tlas(&device, &queue, capacity);
            hardware.capacity = capacity;
            hardware.built = None;
        }
        None => {
            let capacity = needed.next_power_of_two();
            commands.insert_resource(HardwareTraversal {
                tlas: create_tlas(&device, &queue, capacity),
                capacity,
                source: None,
                indices: None,
                blas: HashMap::new(),
                instances: Vec::new(),
                built: None,
            });
        }
    }
}

/// Builds BLASes for the live geometry and the TLAS for the live instances.
/// Runs after `refit_scene`, so deformed positions are already written.
pub(super) fn build_hardware_traversal(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    scene: Res<GiScene>,
    gpu_scene: Option<Res<GpuScene>>,
    views: Query<&RadianceCascades>,
    hardware: Option<ResMut<HardwareTraversal>>,
) {
    let Some(mut hardware) = hardware else {
        return;
    };
    // Nothing is built until a view asks for hardware traversal.
    if !views.iter().any(|settings| {
        super::hardware_traversal_requested(*settings) && settings.radiance_transport_active()
    }) {
        hardware.built = None;
        return;
    }
    let Some(gpu_scene) = gpu_scene else {
        hardware.built = None;
        return;
    };
    let data = &scene.data;
    let count = data.instances.len();
    // Paged geometry lives in textures, which cannot feed a BLAS.
    let Some(geometry) = gpu_scene.geometry.as_ref().filter(|_| {
        scene.ready
            && gpu_scene.ready_for_revision(scene.revision)
            && !gpu_scene.geometry_paged
            && count <= hardware.capacity as usize
            && data.instance_words.len() >= count * INSTANCE_WORDS_PER_RECORD
            && !data.geometry_words.is_empty()
    }) else {
        hardware.built = None;
        return;
    };
    let header = data.geometry_words[0];
    let source = (geometry.id(), gpu_scene.mesh_revision);
    if hardware.source != Some(source) {
        let indices: Vec<u8> = data
            .geometry_words
            .iter()
            .skip(header.x as usize)
            .take(data.triangles.len())
            .flat_map(|word| [word.x, word.y, word.z])
            .flat_map(u32::to_le_bytes)
            .collect();
        hardware.indices = Some(device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("rc_hardware_indices"),
            contents: &indices,
            usage: BufferUsages::BLAS_INPUT,
        }));
        hardware.blas.clear();
        hardware.source = Some(source);
        hardware.built = None;
    }
    let records = &data.instance_words[..count * INSTANCE_WORDS_PER_RECORD];
    let (instance_records, _) = records.as_chunks::<INSTANCE_WORDS_PER_RECORD>();
    let mut builds = Vec::new();
    for record in instance_records {
        let (first, triangles) = (record[8].y, record[8].z);
        if triangles == 0 {
            continue;
        }
        let deformed = record[6].w & INSTANCE_FLAG_ACTIVE_DEFORMATION != 0;
        let opaque = record[6].z != 0;
        let key = (first, triangles, opaque);
        match hardware.blas.entry(key) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                let size = wgpu::BlasTriangleGeometrySizeDescriptor {
                    vertex_format: wgpu::VertexFormat::Float32x3,
                    vertex_count: header.w,
                    index_format: Some(wgpu::IndexFormat::Uint32),
                    index_count: Some(triangles * 3),
                    // Non-opaque triangles reach the shader's acceptance test;
                    // hardware-opaque instances need none.
                    flags: if opaque {
                        wgpu::AccelerationStructureGeometryFlags::OPAQUE
                    } else {
                        wgpu::AccelerationStructureGeometryFlags::NO_DUPLICATE_ANY_HIT_INVOCATION
                    },
                };
                let blas = device.wgpu_device().create_blas(
                    &wgpu::CreateBlasDescriptor {
                        label: Some("rc_hardware_blas"),
                        flags: if deformed {
                            wgpu::AccelerationStructureFlags::PREFER_FAST_BUILD
                        } else {
                            wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE
                        },
                        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                    },
                    wgpu::BlasGeometrySizeDescriptors::Triangles {
                        descriptors: vec![size.clone()],
                    },
                );
                entry.insert(HardwareBlas {
                    blas,
                    size,
                    deformed,
                });
                builds.push(key);
            }
            std::collections::hash_map::Entry::Occupied(_) => {
                if deformed && !builds.contains(&key) {
                    builds.push(key);
                }
            }
        }
    }
    if builds.is_empty()
        && hardware.built == Some(scene.revision)
        && hardware.instances.as_slice() == records
    {
        return;
    }
    let hardware = &mut *hardware;
    let Some(indices) = hardware.indices.as_ref() else {
        return;
    };
    let mut instances = instance_records.iter();
    for slot in 0..hardware.capacity as usize {
        hardware.tlas[slot] = instances.next().and_then(|record| {
            let opaque = record[6].z != 0;
            let blas = hardware.blas.get(&(record[8].y, record[8].z, opaque))?;
            // World-from-local rows, 3×4 row-major as the TLAS expects.
            let mut transform = [0.0; 12];
            for (row, word) in record[3..6].iter().enumerate() {
                for (column, value) in word.to_array().into_iter().enumerate() {
                    transform[row * 4 + column] = f32::from_bits(value);
                }
            }
            // Mask bits mirror `hardwareRayMask` in traversal.slang.
            Some(wgpu::TlasInstance::new(
                &blas.blas,
                transform,
                slot as u32,
                if opaque { 0x01 } else { 0x02 },
            ))
        });
    }
    let first_vertex = header.y / VERTEX_WORDS;
    let entries: Vec<_> = builds
        .iter()
        .filter_map(|key| hardware.blas.get(key).map(|blas| (key, blas)))
        .map(|((first, ..), blas)| wgpu::BlasBuildEntry {
            blas: &blas.blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &blas.size,
                vertex_buffer: geometry,
                first_vertex,
                vertex_stride: u64::from(VERTEX_WORDS) * 16,
                index_buffer: Some(indices),
                first_index: Some(first * 3),
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        })
        .collect();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rc_hardware_build"),
    });
    encoder.build_acceleration_structures(entries.iter(), [&hardware.tlas]);
    queue.submit([encoder.finish()]);
    debug!(
        instances = count,
        blas_builds = builds.len(),
        deformed = hardware.blas.values().filter(|blas| blas.deformed).count(),
        "GI hardware traversal structures built"
    );
    hardware.instances = records.to_vec();
    hardware.built = Some(scene.revision);
}

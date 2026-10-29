use std::{
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
    sync::Arc,
};

use bevy::{
    asset::AssetEvent,
    camera::visibility::RenderLayers,
    ecs::message::{MessageCursor, Messages},
    mesh::{Indices, VertexAttributeValues},
    pbr::Lightmap,
    prelude::*,
    render::{
        RenderApp,
        extract_resource::ExtractResource,
        render_resource::{Face, ShaderType},
    },
    tasks::{AsyncComputeTaskPool, Task, block_on, poll_once},
};

use crate::material::{
    MATERIAL_WORDS, TextureAtlas, TextureCache, TextureOptions, material_support, material_words,
    texture_handles,
};
use crate::{RadianceCascades, RadianceCascadesGeometry};

// CPU build/oracle snapshot. GPU traversal loads indexed positions/attributes.
#[derive(Clone, Copy, Default)]
pub(crate) struct Triangle {
    pub a: Vec4,
    pub b: Vec4,
    pub c: Vec4,
    pub albedo: Vec4,
    pub emission: Vec4,
    pub indices: UVec3,
}

#[derive(Clone, Copy)]
pub(crate) struct MeshVertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub tangent: Vec4,
    pub uv: Vec4,
    pub color: Vec4,
    pub flags: u32,
}

impl MeshVertex {
    fn words(self) -> [UVec4; 5] {
        [
            vec_word(self.position, self.flags),
            vec_word(self.normal, 0),
            UVec4::from_array(self.tangent.to_array().map(f32::to_bits)),
            UVec4::from_array(self.uv.to_array().map(f32::to_bits)),
            UVec4::from_array(self.color.to_array().map(f32::to_bits)),
        ]
    }
}

impl Triangle {
    fn bounds(&self) -> (Vec3, Vec3) {
        (
            self.a
                .truncate()
                .min(self.b.truncate())
                .min(self.c.truncate()),
            self.a
                .truncate()
                .max(self.b.truncate())
                .max(self.c.truncate()),
        )
    }
}

#[derive(Clone, Copy, Default, ShaderType)]
pub(crate) struct BvhNode {
    pub min: Vec3,
    pub first: u32,
    pub max: Vec3,
    // count > 0: leaf; count == 0: first and first+1 are child node indices.
    pub count: u32,
}

#[derive(Clone, Copy, Default, PartialEq, ShaderType)]
pub(crate) struct GiLight {
    // w == 0: direction to directional light; w > 0: point/spot position + range.
    pub position: Vec4,
    pub color: Vec4,
    // Spot forward.xyz and cone scale.w; zero scale denotes a point light.
    // color.w contains the spot cone offset; edge2.w is punctual source radius.
    pub direction: Vec4,
    // Emissive triangle edges; edge1.w is its discrete sampling probability.
    pub edge1: Vec4,
    pub edge2: Vec4,
    // Emissive source's instance/triangle. After the actual lights, per-view
    // mask records use all four words; punctual entries do not use this field.
    pub source: UVec4,
}

impl GiLight {
    fn punctual(
        transform: &GlobalTransform,
        color: Color,
        intensity: f32,
        range: f32,
        radius: f32,
    ) -> Option<Self> {
        let position = transform.translation().extend(range);
        let color = (color.to_linear().to_vec4().truncate() * intensity
            / (4.0 * std::f32::consts::PI))
            .extend(0.0);
        (position.is_finite()
            && color.is_finite()
            && intensity > 0.0
            && range > 0.0
            && radius.is_finite()
            && radius >= 0.0)
            .then_some(Self {
                position,
                color,
                edge2: Vec4::new(0.0, 0.0, 0.0, radius),
                ..default()
            })
    }
}

#[derive(Clone, Default)]
pub(crate) struct SceneData {
    pub deformation: Arc<crate::deformation::DeformationData>,
    pub triangles: Arc<Vec<Triangle>>,
    pub vertices: Arc<Vec<MeshVertex>>,
    pub geometry_words: Arc<Vec<UVec4>>,
    #[cfg(test)]
    pub material_words: Arc<Vec<UVec4>>,
    pub textures: Arc<TextureAtlas>,
    // TLAS nodes index instances; immutable BLAS nodes index local triangles.
    pub nodes: Arc<Vec<BvhNode>>,
    pub blas_nodes: Arc<Vec<BvhNode>>,
    pub instances: Arc<Vec<SceneInstance>>,
    pub acceleration_prefix: Arc<Vec<UVec4>>,
    pub blas_words: Arc<Vec<UVec4>>,
    pub instance_words: Arc<Vec<UVec4>>,
    /// Entity-to-index mapping for the packed GPU instance table. The index is
    /// used by raster receivers to enter the owning BLAS without a TLAS walk.
    pub instance_indices: Arc<HashMap<Entity, u32>>,
    // ponytail: 64-bit signatures keep residency metadata small; retain full
    // geometry/material keys only if hash collisions become a practical risk.
    pub instance_residency: Arc<HashMap<Entity, u64>>,
    /// Shared by per-view last-encoded snapshots without copying light data.
    pub lights: Arc<Vec<GiLight>>,
    pub light_layers: Arc<Vec<RenderLayers>>,
    pub emissive_lights: Arc<Vec<GiLight>>,
}

#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub(crate) struct GiScene {
    pub data: Arc<SceneData>,
    pub pending_entities: Arc<HashSet<Entity>>,
    pub revision: u64,
    pub light_revision: u64,
    pub geometry_revision: u64,
    pub mesh_revision: u64,
    pub topology_revision: u64,
    pub blas_builds: u64,
    pub tlas_builds: u64,
    pub tlas_refits: u64,
    pub ready: bool,
}

const INSTANCE_WORDS: usize = 9;
const INSTANCE_FLAG_UNLIT: u32 = 1 << 4;
const INSTANCE_FLAG_BAKED_RECEIVER: u32 = 1 << 5;

#[derive(Clone)]
pub(crate) struct SceneInstance {
    entity: Entity,
    world_from_local: bevy::math::Affine3A,
    local_from_world: bevy::math::Affine3A,
    min: Vec3,
    max: Vec3,
    root: u32,
    first: u32,
    count: u32,
    pub albedo: Vec3,
    emission: Vec3,
    // Bits 0..1: cull mode; bit 2: mirrored; bit 3: double-sided; bit 4: unlit;
    // bit 5: baked-lightmap receiver; bit 6: active deformation.
    flags: u32,
    material: u32,
    layers: RenderLayers,
}

fn layer_words(layers: &RenderLayers) -> Vec<u32> {
    let mut words: Vec<_> = layers
        .bits()
        .iter()
        .flat_map(|bits| [*bits as u32, (*bits >> 32) as u32])
        .collect();
    while words.last() == Some(&0) {
        words.pop();
    }
    words
}

pub(crate) fn view_lights(data: &SceneData, layers: &RenderLayers) -> (Vec<GiLight>, u32, UVec4) {
    let mut lights: Vec<_> = data
        .lights
        .iter()
        .zip(data.light_layers.iter())
        .filter(|(_, light_layers)| layers.intersects(light_layers))
        .map(|(light, _)| *light)
        .collect();
    let conventional = lights.len() as u32;
    // ponytail: a global emitter CDF wastes samples on hidden emitters. Keep
    // them as null events so NEE and continuation-hit PDFs agree; normalize a
    // per-view CDF only if layer-heavy scenes make this variance costly.
    lights.extend(data.emissive_lights.iter().copied());
    let words = layer_words(layers);
    let selector = UVec4::new(lights.len() as u32, words.len() as u32, 0, 0);
    for group in words.chunks(4) {
        let mut source = [0; 4];
        source[..group.len()].copy_from_slice(group);
        lights.push(GiLight {
            source: UVec4::from_array(source),
            ..default()
        });
    }
    if lights.is_empty() {
        lights.push(GiLight::default());
    }
    (lights, conventional, selector)
}

impl SceneInstance {
    pub(crate) fn world_bounds(&self) -> Option<(Vec3, Vec3)> {
        (self.min.is_finite() && self.max.is_finite() && self.min.cmple(self.max).all())
            .then_some((self.min, self.max))
    }

    #[cfg(test)]
    pub(crate) fn test_with_world_bounds(min: Vec3, max: Vec3) -> Self {
        Self {
            entity: Entity::PLACEHOLDER,
            world_from_local: bevy::math::Affine3A::IDENTITY,
            local_from_world: bevy::math::Affine3A::IDENTITY,
            min,
            max,
            root: 0,
            first: 0,
            count: 0,
            albedo: Vec3::ZERO,
            emission: Vec3::ZERO,
            flags: 0,
            material: 0,
            layers: RenderLayers::default(),
        }
    }

    fn triangle(&self, local: Triangle) -> Triangle {
        Triangle {
            a: self
                .world_from_local
                .transform_point3(local.a.truncate())
                .extend(0.0),
            b: self
                .world_from_local
                .transform_point3(local.b.truncate())
                .extend(0.0),
            c: self
                .world_from_local
                .transform_point3(local.c.truncate())
                .extend(0.0),
            albedo: self.albedo.extend((self.flags & 3) as f32),
            emission: self.emission.extend(0.0),
            indices: local.indices,
        }
    }
}

struct MeshBlas {
    deformation: crate::deformation::MeshDeformation,
    signature: u64,
    geometry_signature: u64,
    vertices: Vec<MeshVertex>,
    triangles: Vec<Triangle>,
    nodes: Vec<BvhNode>,
}

struct BlasBuildOutput {
    signature: u64,
    geometry_signature: u64,
    deformation: crate::deformation::MeshDeformation,
    vertices: Vec<MeshVertex>,
    triangles: Vec<Triangle>,
    nodes: Vec<BvhNode>,
}

struct PendingBlas {
    signature: u64,
    geometry_signature: u64,
    task: Task<BlasBuildOutput>,
}

#[derive(Clone, Copy)]
struct MeshSpan {
    vertex: u32,
    root: u32,
    first: u32,
    count: u32,
}

#[derive(Default)]
pub(crate) struct MeshCache {
    meshes: HashMap<AssetId<Mesh>, MeshBlas>,
    pending_blas: HashMap<AssetId<Mesh>, PendingBlas>,
    needs_validation: HashSet<AssetId<Mesh>>,
    support: HashMap<AssetId<Mesh>, Result<(), &'static str>>,
    packed_keys: Vec<(AssetId<Mesh>, u64)>,
    packed_geometry_keys: Vec<(AssetId<Mesh>, u64)>,
    triangles: Arc<Vec<Triangle>>,
    vertices: Arc<Vec<MeshVertex>>,
    geometry_words: Arc<Vec<UVec4>>,
    nodes: Arc<Vec<BvhNode>>,
    spans: HashMap<AssetId<Mesh>, MeshSpan>,
    animated_spans: HashMap<Entity, MeshSpan>,
    animated_keys: Vec<(Entity, AssetId<Mesh>, u64)>,
    tlas_build_cost: f64,
    tlas_rebuild: Option<Task<(Vec<BvhNode>, Vec<Entity>)>>,
}

fn geometry_signature(mesh: &Mesh, positions: &[[f32; 3]]) -> u64 {
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    positions.len().hash(&mut hash);
    for position in positions {
        for value in position {
            value.to_bits().hash(&mut hash);
        }
    }
    match mesh.indices() {
        Some(Indices::U16(indices)) => indices.hash(&mut hash),
        Some(Indices::U32(indices)) => indices.hash(&mut hash),
        None => 0usize.hash(&mut hash),
    }
    hash.finish()
}

#[cfg(not(target_arch = "wasm32"))]
const BLAS_CACHE_MAGIC: &[u8; 8] = b"PENBLAS1";
#[cfg(not(target_arch = "wasm32"))]
const BLAS_HEADER_BYTES: usize = 24;
#[cfg(not(target_arch = "wasm32"))]
const BLAS_CHECKSUM_BYTES: usize = 8;
#[cfg(not(target_arch = "wasm32"))]
const BLAS_NODE_BYTES: usize = 32;
#[cfg(not(target_arch = "wasm32"))]
const BLAS_TRIANGLE_BYTES: usize = 92;

#[cfg(not(target_arch = "wasm32"))]
fn blas_cache_dir() -> std::path::PathBuf {
    std::env::temp_dir().join("bevy_penumbra").join("blas")
}

#[cfg(not(target_arch = "wasm32"))]
fn blas_checksum(nodes: &[BvhNode], triangles: &[Triangle]) -> u64 {
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    for node in nodes {
        node.min.to_array().map(f32::to_bits).hash(&mut hash);
        node.first.hash(&mut hash);
        node.max.to_array().map(f32::to_bits).hash(&mut hash);
        node.count.hash(&mut hash);
    }
    for triangle in triangles {
        for value in [
            triangle.a,
            triangle.b,
            triangle.c,
            triangle.albedo,
            triangle.emission,
        ] {
            value.to_array().map(f32::to_bits).hash(&mut hash);
        }
        triangle.indices.to_array().hash(&mut hash);
    }
    hash.finish()
}

#[cfg(not(target_arch = "wasm32"))]
fn read_u32(reader: &mut impl std::io::Read) -> Option<u32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes).ok()?;
    Some(u32::from_le_bytes(bytes))
}

#[cfg(not(target_arch = "wasm32"))]
fn read_u64(reader: &mut impl std::io::Read) -> Option<u64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes).ok()?;
    Some(u64::from_le_bytes(bytes))
}

#[cfg(not(target_arch = "wasm32"))]
fn read_vec3(reader: &mut impl std::io::Read) -> Option<Vec3> {
    Some(Vec3::new(
        f32::from_bits(read_u32(reader)?),
        f32::from_bits(read_u32(reader)?),
        f32::from_bits(read_u32(reader)?),
    ))
}

#[cfg(not(target_arch = "wasm32"))]
fn read_vec4(reader: &mut impl std::io::Read) -> Option<Vec4> {
    Some(Vec4::new(
        f32::from_bits(read_u32(reader)?),
        f32::from_bits(read_u32(reader)?),
        f32::from_bits(read_u32(reader)?),
        f32::from_bits(read_u32(reader)?),
    ))
}

#[cfg(not(target_arch = "wasm32"))]
fn write_vec3(writer: &mut impl std::io::Write, value: Vec3) -> std::io::Result<()> {
    for component in value.to_array() {
        writer.write_all(&component.to_bits().to_le_bytes())?;
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn write_vec4(writer: &mut impl std::io::Write, value: Vec4) -> std::io::Result<()> {
    for component in value.to_array() {
        writer.write_all(&component.to_bits().to_le_bytes())?;
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn valid_blas(nodes: &[BvhNode], triangles: &[Triangle]) -> bool {
    !nodes.is_empty()
        && nodes.iter().enumerate().all(|(index, node)| {
            node.min.is_finite()
                && node.max.is_finite()
                && node.min.cmple(node.max).all()
                && if node.count == 0 {
                    node.first as usize > index
                        && (node.first as usize)
                            .checked_add(1)
                            .is_some_and(|child| child < nodes.len())
                } else {
                    (node.first as usize)
                        .checked_add(node.count as usize)
                        .is_some_and(|end| end <= triangles.len())
                }
        })
}

#[cfg(not(target_arch = "wasm32"))]
fn load_blas_cache(
    cache_dir: &std::path::Path,
    geometry_signature: u64,
    expected_triangles: usize,
) -> Option<(Vec<BvhNode>, Vec<Triangle>)> {
    use std::io::Read;

    let path = cache_dir.join(format!("{geometry_signature:016x}.bvh"));
    let file = std::fs::File::open(path).ok()?;
    let file_len = file.metadata().ok()?.len();
    let mut file = std::io::BufReader::new(file);
    let mut magic = [0; BLAS_CACHE_MAGIC.len()];
    file.read_exact(&mut magic).ok()?;
    let stored_signature = read_u64(&mut file)?;
    let node_count = read_u32(&mut file)? as usize;
    let triangle_count = read_u32(&mut file)? as usize;
    let expected_triangles = u32::try_from(expected_triangles).ok()? as usize;
    let max_nodes = expected_triangles.checked_mul(2)?.saturating_sub(1);
    let expected_len = BLAS_HEADER_BYTES
        .checked_add(node_count.checked_mul(BLAS_NODE_BYTES)?)?
        .checked_add(triangle_count.checked_mul(BLAS_TRIANGLE_BYTES)?)?
        .checked_add(BLAS_CHECKSUM_BYTES)?;
    if &magic != BLAS_CACHE_MAGIC
        || stored_signature != geometry_signature
        || triangle_count != expected_triangles
        || node_count == 0
        || node_count > max_nodes
        || expected_len as u64 != file_len
    {
        return None;
    }

    let mut nodes = Vec::with_capacity(node_count);
    for _ in 0..node_count {
        nodes.push(BvhNode {
            min: read_vec3(&mut file)?,
            first: read_u32(&mut file)?,
            max: read_vec3(&mut file)?,
            count: read_u32(&mut file)?,
        });
    }
    let mut triangles = Vec::with_capacity(triangle_count);
    for _ in 0..triangle_count {
        triangles.push(Triangle {
            a: read_vec4(&mut file)?,
            b: read_vec4(&mut file)?,
            c: read_vec4(&mut file)?,
            albedo: read_vec4(&mut file)?,
            emission: read_vec4(&mut file)?,
            indices: UVec3::new(
                read_u32(&mut file)?,
                read_u32(&mut file)?,
                read_u32(&mut file)?,
            ),
        });
    }
    let checksum = read_u64(&mut file)?;
    (checksum == blas_checksum(&nodes, &triangles) && valid_blas(&nodes, &triangles))
        .then_some((nodes, triangles))
}

#[cfg(not(target_arch = "wasm32"))]
fn save_blas_cache(
    cache_dir: &std::path::Path,
    geometry_signature: u64,
    nodes: &[BvhNode],
    triangles: &[Triangle],
) -> std::io::Result<()> {
    use std::io::Write;

    let (Ok(node_count), Ok(triangle_count)) =
        (u32::try_from(nodes.len()), u32::try_from(triangles.len()))
    else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "BLAS cache record exceeds the supported count",
        ));
    };
    let directory = cache_dir;
    let path = directory.join(format!("{geometry_signature:016x}.bvh"));
    let write = || -> std::io::Result<()> {
        std::fs::create_dir_all(directory)?;
        let mut file = tempfile::NamedTempFile::new_in(directory)?;
        file.write_all(BLAS_CACHE_MAGIC)?;
        file.write_all(&geometry_signature.to_le_bytes())?;
        file.write_all(&node_count.to_le_bytes())?;
        file.write_all(&triangle_count.to_le_bytes())?;
        {
            let mut writer = std::io::BufWriter::new(&mut file);
            for node in nodes {
                write_vec3(&mut writer, node.min)?;
                writer.write_all(&node.first.to_le_bytes())?;
                write_vec3(&mut writer, node.max)?;
                writer.write_all(&node.count.to_le_bytes())?;
            }
            for triangle in triangles {
                for value in [
                    triangle.a,
                    triangle.b,
                    triangle.c,
                    triangle.albedo,
                    triangle.emission,
                ] {
                    write_vec4(&mut writer, value)?;
                }
                for index in triangle.indices.to_array() {
                    writer.write_all(&index.to_le_bytes())?;
                }
            }
            writer.write_all(&blas_checksum(nodes, triangles).to_le_bytes())?;
            writer.flush()?;
        }
        file.persist(&path).map_err(|error| error.error)?;
        Ok(())
    };
    write()
}

fn mesh_signature(mesh: &Mesh, geometry: u64) -> u64 {
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    geometry.hash(&mut hash);
    for attribute in [
        Mesh::ATTRIBUTE_NORMAL,
        Mesh::ATTRIBUTE_TANGENT,
        Mesh::ATTRIBUTE_UV_0,
        Mesh::ATTRIBUTE_UV_1,
        Mesh::ATTRIBUTE_COLOR,
    ] {
        mesh.attribute(attribute)
            .map(VertexAttributeValues::get_bytes)
            .hash(&mut hash);
    }
    hash.finish()
}

fn mesh_vertices(mesh: &Mesh, positions: &[[f32; 3]]) -> Result<Vec<MeshVertex>, &'static str> {
    let normals = match mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
        Some(VertexAttributeValues::Float32x3(values)) if values.len() == positions.len() => {
            Some(values)
        }
        None => None,
        _ => return Err("normal attributes must be Float32x3 with one value per vertex"),
    };
    let tangents = match mesh.attribute(Mesh::ATTRIBUTE_TANGENT) {
        Some(VertexAttributeValues::Float32x4(values)) if values.len() == positions.len() => {
            Some(values)
        }
        None => None,
        _ => return Err("tangent attributes must be Float32x4 with one value per vertex"),
    };
    let mut uvs = [None; 2];
    for (i, attribute) in [Mesh::ATTRIBUTE_UV_0, Mesh::ATTRIBUTE_UV_1]
        .into_iter()
        .enumerate()
    {
        uvs[i] = match mesh.attribute(attribute) {
            Some(VertexAttributeValues::Float32x2(values)) if values.len() == positions.len() => {
                Some(values)
            }
            None => None,
            _ => return Err("UV attributes must be Float32x2 with one value per vertex"),
        };
    }
    let colors = match mesh.attribute(Mesh::ATTRIBUTE_COLOR) {
        Some(VertexAttributeValues::Float32x4(values)) if values.len() == positions.len() => {
            Some(values)
        }
        None => None,
        _ => return Err("color attributes must be Float32x4 with one value per vertex"),
    };
    let flags = u32::from(normals.is_some())
        | (u32::from(tangents.is_some()) << 1)
        | (u32::from(uvs[0].is_some()) << 2)
        | (u32::from(uvs[1].is_some()) << 3)
        | (u32::from(colors.is_some()) << 4);
    positions
        .iter()
        .enumerate()
        .map(|(i, position)| {
            let uv0 = uvs[0].map_or(Vec2::ZERO, |v| Vec2::from_array(v[i]));
            let uv1 = uvs[1].map_or(Vec2::ZERO, |v| Vec2::from_array(v[i]));
            let vertex = MeshVertex {
                position: Vec3::from_array(*position),
                normal: normals.map_or(Vec3::ZERO, |v| Vec3::from_array(v[i])),
                tangent: tangents.map_or(Vec4::ZERO, |v| Vec4::from_array(v[i])),
                uv: Vec4::new(uv0.x, uv0.y, uv1.x, uv1.y),
                color: colors.map_or(Vec4::ONE, |v| Vec4::from_array(v[i])),
                flags,
            };
            if !vertex.position.is_finite()
                || !vertex.normal.is_finite()
                || !vertex.tangent.is_finite()
                || !vertex.uv.is_finite()
                || !vertex.color.is_finite()
            {
                return Err("mesh vertex attributes must be finite");
            }
            Ok(vertex)
        })
        .collect()
}

fn mesh_support(mesh: &Mesh) -> Result<(), &'static str> {
    if mesh.primitive_topology() != bevy::render::render_resource::PrimitiveTopology::TriangleList {
        return Err("GI requires TriangleList meshes");
    }
    let Some(VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute(Mesh::ATTRIBUTE_POSITION)
    else {
        return Err("GI requires Float32x3 positions");
    };
    // High-bit group IDs distinguish unmatched raster samples from mesh components.
    if positions.len() > 0x8000_0000usize {
        return Err("GI mesh vertex indices must fit in 31 bits");
    }
    validate_mesh_vertices(mesh, positions)?;
    validate_mesh_deformation(mesh, positions.len())?;
    if !has_usable_triangle(mesh, positions) {
        return Err("GI mesh contains no usable triangles");
    }
    Ok(())
}

pub(crate) fn validate_mesh_vertices(
    mesh: &Mesh,
    positions: &[[f32; 3]],
) -> Result<(), &'static str> {
    let normals = match mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
        Some(VertexAttributeValues::Float32x3(values)) if values.len() == positions.len() => {
            Some(values)
        }
        None => None,
        _ => return Err("normal attributes must be Float32x3 with one value per vertex"),
    };
    let tangents = match mesh.attribute(Mesh::ATTRIBUTE_TANGENT) {
        Some(VertexAttributeValues::Float32x4(values)) if values.len() == positions.len() => {
            Some(values)
        }
        None => None,
        _ => return Err("tangent attributes must be Float32x4 with one value per vertex"),
    };
    let mut uvs = [None; 2];
    for (i, attribute) in [Mesh::ATTRIBUTE_UV_0, Mesh::ATTRIBUTE_UV_1]
        .into_iter()
        .enumerate()
    {
        uvs[i] = match mesh.attribute(attribute) {
            Some(VertexAttributeValues::Float32x2(values)) if values.len() == positions.len() => {
                Some(values)
            }
            None => None,
            _ => return Err("UV attributes must be Float32x2 with one value per vertex"),
        };
    }
    let colors = match mesh.attribute(Mesh::ATTRIBUTE_COLOR) {
        Some(VertexAttributeValues::Float32x4(values)) if values.len() == positions.len() => {
            Some(values)
        }
        None => None,
        _ => return Err("color attributes must be Float32x4 with one value per vertex"),
    };
    for (i, position) in positions.iter().enumerate() {
        if !Vec3::from_array(*position).is_finite()
            || normals.is_some_and(|values| !Vec3::from_array(values[i]).is_finite())
            || tangents.is_some_and(|values| !Vec4::from_array(values[i]).is_finite())
            || uvs
                .iter()
                .flatten()
                .any(|values| !Vec2::from_array(values[i]).is_finite())
            || colors.is_some_and(|values| !Vec4::from_array(values[i]).is_finite())
        {
            return Err("mesh vertex attributes must be finite");
        }
    }
    Ok(())
}

fn validate_mesh_deformation(mesh: &Mesh, vertex_count: usize) -> Result<(), &'static str> {
    match (
        mesh.attribute(Mesh::ATTRIBUTE_JOINT_INDEX),
        mesh.attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT),
    ) {
        (
            Some(VertexAttributeValues::Uint16x4(indices)),
            Some(VertexAttributeValues::Float32x4(weights)),
        ) if indices.len() == vertex_count && weights.len() == vertex_count => {
            if weights.iter().any(|weights| {
                let sum = weights.iter().sum::<f32>();
                weights
                    .iter()
                    .any(|weight| !weight.is_finite() || *weight < 0.0)
                    || !sum.is_finite()
                    || sum <= 0.0
            }) {
                return Err(
                    "skin vertex weights must be finite, nonnegative and have a finite nonzero sum",
                );
            }
        }
        (None, None) => {}
        _ => return Err("skin requires matching Uint16x4 joint indices and Float32x4 weights"),
    }
    if let Some(targets) = mesh.get_morph_targets() {
        if vertex_count == 0 || targets.len() % vertex_count != 0 {
            return Err("morph target vertex counts do not match the mesh");
        }
        if targets.iter().any(|target| {
            !target.position.is_finite()
                || !target.normal.is_finite()
                || !target.tangent.is_finite()
        }) {
            return Err("morph displacements must be finite");
        }
    }
    Ok(())
}

fn has_usable_triangle(mesh: &Mesh, positions: &[[f32; 3]]) -> bool {
    let indices = mesh.indices();
    let index_count = indices.map_or(positions.len(), Indices::len);
    (0..index_count / 3).any(|triangle| {
        let indices = [0, 1, 2].map(|corner| {
            let index = triangle * 3 + corner;
            match indices {
                Some(Indices::U16(values)) => values[index] as usize,
                Some(Indices::U32(values)) => values[index] as usize,
                None => index,
            }
        });
        let [Some(a), Some(b), Some(c)] = indices.map(|index| positions.get(index)) else {
            return false;
        };
        let [a, b, c] = [a, b, c].map(|position| Vec3::from_array(*position));
        let cross = (b - a).cross(c - a);
        a.is_finite()
            && b.is_finite()
            && c.is_finite()
            && cross.is_finite()
            && (cross != Vec3::ZERO
                || mesh.get_morph_targets().is_some()
                || mesh.attribute(Mesh::ATTRIBUTE_JOINT_INDEX).is_some())
    })
}

/// Words per packed vertex. The vertex region starts on a multiple of this
/// stride so hardware acceleration structures can address positions as
/// `first_vertex * stride` bytes into the same buffer.
pub(crate) const VERTEX_WORDS: u32 = 5;

fn geometry_words(triangles: &[Triangle], vertices: &[MeshVertex]) -> Vec<UVec4> {
    let vertex_base = (1 + triangles.len() as u32).next_multiple_of(VERTEX_WORDS);
    let mut words =
        Vec::with_capacity(vertex_base as usize + vertices.len() * VERTEX_WORDS as usize);
    words.push(UVec4::new(
        1,
        vertex_base,
        VERTEX_WORDS,
        vertices.len() as u32,
    ));
    words.extend(triangles.iter().map(|triangle| triangle.indices.extend(0)));
    words.resize(vertex_base as usize, UVec4::ZERO);
    words.extend(vertices.iter().flat_map(|vertex| vertex.words()));
    words
}

fn local_triangles(
    vertices: &[MeshVertex],
    indices: Option<&Indices>,
    preserve_degenerate: bool,
) -> Vec<Triangle> {
    let index_count = match indices {
        Some(Indices::U16(values)) => values.len(),
        Some(Indices::U32(values)) => values.len(),
        None => vertices.len(),
    };
    let mut triangles = Vec::with_capacity(index_count / 3);
    let mut append = |indices: [usize; 3]| {
        let [Some(a), Some(b), Some(c)] = indices.map(|index| vertices.get(index)) else {
            return;
        };
        let [a, b, c] = [a.position, b.position, c.position];
        let cross = (b - a).cross(c - a);
        if !a.is_finite()
            || !b.is_finite()
            || !c.is_finite()
            || !cross.is_finite()
            || (cross == Vec3::ZERO && !preserve_degenerate)
        {
            return;
        }
        triangles.push(Triangle {
            a: a.extend(0.0),
            b: b.extend(0.0),
            c: c.extend(0.0),
            indices: UVec3::new(indices[0] as u32, indices[1] as u32, indices[2] as u32),
            ..default()
        });
    };
    match indices {
        Some(Indices::U16(indices)) => {
            for chunk in indices.as_chunks::<3>().0 {
                append([chunk[0] as usize, chunk[1] as usize, chunk[2] as usize]);
            }
        }
        Some(Indices::U32(indices)) => {
            for chunk in indices.as_chunks::<3>().0 {
                append([chunk[0] as usize, chunk[1] as usize, chunk[2] as usize]);
            }
        }
        None => {
            for triangle in 0..vertices.len() / 3 {
                let first = triangle * 3;
                append([first, first + 1, first + 2]);
            }
        }
    }
    triangles
}

impl MeshCache {
    fn pack(&mut self, mut keys: Vec<AssetId<Mesh>>, animated: Vec<(Entity, AssetId<Mesh>)>) {
        keys.sort();
        keys.dedup();
        self.meshes.retain(|id, _| keys.binary_search(id).is_ok());
        self.needs_validation
            .retain(|id| self.meshes.contains_key(id) || self.pending_blas.contains_key(id));
        if keys
            .iter()
            .map(|&id| (id, self.meshes[&id].signature))
            .eq(self.packed_keys.iter().copied())
            && animated
                .iter()
                .map(|&(entity, id)| (entity, id, self.meshes[&id].signature))
                .eq(self.animated_keys.iter().copied())
        {
            return;
        }
        let keys: Vec<_> = keys
            .into_iter()
            .map(|id| (id, self.meshes[&id].signature))
            .collect();
        let animated_keys: Vec<_> = animated
            .iter()
            .map(|&(entity, id)| (entity, id, self.meshes[&id].signature))
            .collect();
        let geometry_keys: Vec<_> = keys
            .iter()
            .map(|&(id, _)| (id, self.meshes[&id].geometry_signature))
            .collect();
        if geometry_keys == self.packed_geometry_keys && animated_keys == self.animated_keys {
            // Attribute-only changes do not alter triangle indices, spans or BLAS
            // nodes. Repack vertices without expanding and copying those arrays.
            let mut vertices = Vec::with_capacity(self.vertices.len());
            for &(id, _) in &keys {
                vertices.extend_from_slice(&self.meshes[&id].vertices);
            }
            for &(_, id) in &animated {
                vertices.extend_from_slice(&self.meshes[&id].vertices);
            }
            self.geometry_words = Arc::new(geometry_words(&self.triangles, &vertices));
            self.vertices = Arc::new(vertices);
            self.packed_keys = keys;
            return;
        }
        // Known counts avoid geometric capacity growth and simultaneous large
        // old/new allocations when packing multi-million-triangle assets.
        let mut triangles = Vec::with_capacity(
            keys.iter()
                .map(|(id, _)| self.meshes[id].triangles.len())
                .sum::<usize>()
                + animated
                    .iter()
                    .map(|(_, id)| self.meshes[id].triangles.len())
                    .sum::<usize>(),
        );
        let mut vertices = Vec::with_capacity(
            keys.iter()
                .map(|(id, _)| self.meshes[id].vertices.len())
                .sum::<usize>()
                + animated
                    .iter()
                    .map(|(_, id)| self.meshes[id].vertices.len())
                    .sum::<usize>(),
        );
        let mut nodes = Vec::with_capacity(
            keys.iter()
                .map(|(id, _)| self.meshes[id].nodes.len())
                .sum::<usize>()
                + animated
                    .iter()
                    .map(|(_, id)| self.meshes[id].nodes.len())
                    .sum::<usize>(),
        );
        self.spans.clear();
        self.animated_spans.clear();
        for &(id, _) in &keys {
            let mesh = &self.meshes[&id];
            let span = MeshSpan {
                vertex: vertices.len() as u32,
                root: nodes.len() as u32,
                first: triangles.len() as u32,
                count: mesh.triangles.len() as u32,
            };
            self.spans.insert(id, span);
            let vertex_offset = UVec3::splat(vertices.len() as u32);
            triangles.extend(mesh.triangles.iter().map(|triangle| Triangle {
                indices: triangle.indices + vertex_offset,
                ..*triangle
            }));
            vertices.extend_from_slice(&mesh.vertices);
            nodes.extend(mesh.nodes.iter().map(|node| BvhNode {
                first: node.first
                    + if node.count == 0 {
                        span.root
                    } else {
                        span.first
                    },
                ..*node
            }));
        }
        for &(entity, id) in &animated {
            let mesh = &self.meshes[&id];
            let span = MeshSpan {
                vertex: vertices.len() as u32,
                root: nodes.len() as u32,
                first: triangles.len() as u32,
                count: mesh.triangles.len() as u32,
            };
            self.animated_spans.insert(entity, span);
            triangles.extend(mesh.triangles.iter().map(|triangle| Triangle {
                indices: triangle.indices + UVec3::splat(span.vertex),
                ..*triangle
            }));
            vertices.extend_from_slice(&mesh.vertices);
            nodes.extend(mesh.nodes.iter().map(|node| BvhNode {
                first: node.first
                    + if node.count == 0 {
                        span.root
                    } else {
                        span.first
                    },
                ..*node
            }));
        }
        self.geometry_words = Arc::new(geometry_words(&triangles, &vertices));
        self.vertices = Arc::new(vertices);
        if geometry_keys != self.packed_geometry_keys || animated_keys != self.animated_keys {
            self.triangles = Arc::new(triangles);
            self.nodes = Arc::new(nodes);
            self.packed_geometry_keys = geometry_keys;
        }
        self.packed_keys = keys;
        self.animated_keys = animated_keys;
    }
}

fn vec_word(value: Vec3, metadata: u32) -> UVec4 {
    UVec4::new(
        value.x.to_bits(),
        value.y.to_bits(),
        value.z.to_bits(),
        metadata,
    )
}

fn node_words(nodes: &[BvhNode], child_offset: u32) -> Vec<UVec4> {
    nodes
        .iter()
        .flat_map(|node| {
            [
                vec_word(
                    node.min,
                    node.first + if node.count == 0 { child_offset } else { 0 },
                ),
                vec_word(node.max, node.count),
            ]
        })
        .collect()
}

fn refit(nodes: &mut [BvhNode], instances: &[SceneInstance]) {
    for i in (0..nodes.len()).rev() {
        let node = nodes[i];
        let (min, max) = if node.count > 0 {
            instances[node.first as usize..(node.first + node.count) as usize]
                .iter()
                .fold(
                    (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
                    |(min, max), instance| (min.min(instance.min), max.max(instance.max)),
                )
        } else {
            let a = nodes[node.first as usize];
            let b = nodes[node.first as usize + 1];
            (a.min.min(b.min), a.max.max(b.max))
        };
        nodes[i].min = min;
        nodes[i].max = max;
    }
}

fn pack_instances(
    instances: &[SceneInstance],
    tlas_count: usize,
    base: usize,
    materials: &[UVec4],
    textures: &[UVec4],
) -> Vec<UVec4> {
    let mut words = Vec::with_capacity(instances.len() * INSTANCE_WORDS);
    let material_base = base + instances.len() * INSTANCE_WORDS;
    let layer_base = material_base + materials.len() + textures.len();
    let mut layer_masks = Vec::new();
    for instance in instances {
        for transform in [instance.local_from_world, instance.world_from_local] {
            let matrix = Mat4::from(transform).transpose();
            for row in [matrix.x_axis, matrix.y_axis, matrix.z_axis] {
                words.push(UVec4::from_array(row.to_array().map(f32::to_bits)));
            }
        }
        let mask = layer_words(&instance.layers);
        // Former constant-color slots are unused by textured material shading.
        // Keep the nine-word instance stride and flags in w.
        words.push(UVec4::new(
            (layer_base + layer_masks.len()) as u32,
            mask.len() as u32,
            0,
            instance.flags,
        ));
        for group in mask.chunks(4) {
            let mut word = [0; 4];
            word[..group.len()].copy_from_slice(group);
            layer_masks.push(UVec4::from_array(word));
        }
        let identity = instance.entity.to_bits();
        words.push(UVec4::new(
            identity as u32,
            (identity >> 32) as u32,
            0,
            (material_base + instance.material as usize * MATERIAL_WORDS) as u32,
        ));
        words.push(UVec4::new(
            tlas_count as u32 + instance.root,
            instance.first,
            instance.count,
            0,
        ));
    }
    words.extend_from_slice(materials);
    words.extend_from_slice(textures);
    words.extend(layer_masks);
    words
}

fn changed_instance_transform(
    scene: &GiScene,
    entity: Entity,
    transform: &GlobalTransform,
) -> bool {
    scene
        .data
        .instance_indices
        .get(&entity)
        .and_then(|index| scene.data.instances.get(*index as usize))
        .is_none_or(|instance| {
            instance.entity != entity || instance.world_from_local != transform.affine()
        })
}

// Bevy system parameters and query tuples describe the scene inputs.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn update_scene(
    mut scene: ResMut<GiScene>,
    meshes: Res<Assets<Mesh>>,
    materials: Res<Assets<StandardMaterial>>,
    images: Option<Res<Assets<Image>>>,
    texture_options: Option<Res<TextureOptions>>,
    geometry: Query<
        (
            Entity,
            Ref<Mesh3d>,
            Ref<MeshMaterial3d<StandardMaterial>>,
            Ref<GlobalTransform>,
            Option<Ref<InheritedVisibility>>,
            Option<Ref<RenderLayers>>,
            Option<Ref<Lightmap>>,
        ),
        With<RadianceCascadesGeometry>,
    >,
    directional: Query<(
        &DirectionalLight,
        &GlobalTransform,
        Option<&InheritedVisibility>,
        Option<&RenderLayers>,
    )>,
    points: Query<(
        &PointLight,
        &GlobalTransform,
        Option<&InheritedVisibility>,
        Option<&RenderLayers>,
    )>,
    spots: Query<(
        &SpotLight,
        &GlobalTransform,
        Option<&InheritedVisibility>,
        Option<&RenderLayers>,
    )>,
    mut previous_geometry: Local<HashSet<Entity>>,
    mut removed_visibility: RemovedComponents<InheritedVisibility>,
    mut removed_layers: RemovedComponents<RenderLayers>,
    mut removed_lightmaps: RemovedComponents<Lightmap>,
    mut mesh_cache: Local<MeshCache>,
    mut texture_cache: Local<TextureCache>,
    (
        mut light_scratch,
        mut light_layers_scratch,
        (deformation_inputs, mut previous_poses),
        (unsupported_materials, mut warned_unsupported_materials, mut pending_entities),
        (mesh_events, mut mesh_event_cursor),
        (image_events, mut image_event_cursor),
        (material_events, mut material_event_cursor),
        (cameras, mut suspended),
    ): (
        Local<Vec<GiLight>>,
        Local<Vec<RenderLayers>>,
        (
            crate::deformation::DeformationInputs,
            Local<HashMap<Entity, Result<crate::deformation::DeformationPose, &'static str>>>,
        ),
        (
            Query<
                Entity,
                (
                    With<RadianceCascadesGeometry>,
                    With<Mesh3d>,
                    Without<MeshMaterial3d<StandardMaterial>>,
                ),
            >,
            Local<HashSet<Entity>>,
            Local<HashSet<Entity>>,
        ),
        (
            Option<Res<Messages<AssetEvent<Mesh>>>>,
            Local<MessageCursor<AssetEvent<Mesh>>>,
        ),
        (
            Option<Res<Messages<AssetEvent<Image>>>>,
            Local<MessageCursor<AssetEvent<Image>>>,
        ),
        (
            Option<Res<Messages<AssetEvent<StandardMaterial>>>>,
            Local<MessageCursor<AssetEvent<StandardMaterial>>>,
        ),
        (Query<(&Camera, &RadianceCascades)>, Local<bool>),
    ),
) {
    if !cameras
        .iter()
        .any(|(camera, settings)| camera.is_active && settings.sanitized().intensity > 0.0)
    {
        *suspended = true;
        if scene.ready {
            scene.ready = false;
            scene.revision = scene.revision.wrapping_add(1);
        }
        return;
    }
    // Messages and removals can expire while suspended. Revalidate all cached
    // assets on resume instead of relying on the surviving event window.
    let resumed = std::mem::take(&mut *suspended);
    if resumed {
        mesh_cache.support.clear();
        let cache = &mut *mesh_cache;
        cache.needs_validation.extend(
            cache
                .meshes
                .keys()
                .chain(cache.pending_blas.keys())
                .copied(),
        );
    }
    warned_unsupported_materials.retain(|entity| unsupported_materials.contains(*entity));
    for entity in &unsupported_materials {
        if warned_unsupported_materials.insert(entity) {
            warn!(
                ?entity,
                "GI requires StandardMaterial meshes; custom material shaders are unsupported"
            );
        }
    }
    let poses = deformation_inputs.collect();
    let mut pose_dirty_entities: HashSet<_> = poses
        .iter()
        .filter_map(|(entity, pose)| (previous_poses.get(entity) != Some(pose)).then_some(*entity))
        .collect();
    pose_dirty_entities.extend(
        previous_poses
            .keys()
            .filter(|entity| !poses.contains_key(entity))
            .copied(),
    );
    let poses_changed = !pose_dirty_entities.is_empty();
    *previous_poses = poses;
    let poses = &*previous_poses;
    let mut rebuilt = mesh_cache
        .tlas_rebuild
        .as_mut()
        .and_then(|task| block_on(poll_once(task)));
    if rebuilt.is_some() {
        mesh_cache.tlas_rebuild = None;
    }
    let completed_blas: Vec<_> = mesh_cache
        .pending_blas
        .iter_mut()
        .filter_map(|(&id, pending)| {
            block_on(poll_once(&mut pending.task)).map(|output| (id, output))
        })
        .collect();
    let mut rebuilt_blas = HashSet::new();
    for (id, output) in completed_blas {
        let current = mesh_cache.pending_blas.remove(&id).is_some_and(|pending| {
            pending.signature == output.signature
                && pending.geometry_signature == output.geometry_signature
        });
        if current {
            mesh_cache.meshes.insert(
                id,
                MeshBlas {
                    deformation: output.deformation,
                    signature: output.signature,
                    geometry_signature: output.geometry_signature,
                    vertices: output.vertices,
                    triangles: output.triangles,
                    nodes: output.nodes,
                },
            );
            scene.blas_builds = scene.blas_builds.wrapping_add(1);
            rebuilt_blas.insert(id);
        }
    }
    let visibility_removed = removed_visibility
        .read()
        .filter(|entity| previous_geometry.contains(entity))
        .count()
        > 0;
    let layers_removed = removed_layers
        .read()
        .filter(|entity| previous_geometry.contains(entity))
        .count()
        > 0;
    let lightmaps_removed = removed_lightmaps
        .read()
        .any(|entity| previous_geometry.contains(&entity));
    let images_resource_changed = images.as_ref().is_some_and(|images| images.is_changed());
    let changed_images: Option<HashSet<_>> = image_events.as_ref().map(|events| {
        image_event_cursor
            .read(events)
            .filter_map(|event| match event {
                AssetEvent::Added { id }
                | AssetEvent::Modified { id }
                | AssetEvent::Removed { id }
                | AssetEvent::LoadedWithDependencies { id } => Some(*id),
                AssetEvent::Unused { .. } => None,
            })
            .collect()
    });
    let images_changed = resumed
        || changed_images.as_ref().map_or_else(
            || images_resource_changed,
            |changed| {
                !changed.is_empty()
                    && geometry.iter().any(|(_, _, material, _, visible, ..)| {
                        if visible.as_ref().is_some_and(|visible| !visible.get()) {
                            return false;
                        }
                        materials
                            .get(&material.0)
                            .filter(|material| material_support(material).is_ok())
                            .is_some_and(|material| {
                                texture_handles(material)
                                    .into_iter()
                                    .flatten()
                                    .any(|handle| changed.contains(&handle.id()))
                            })
                    })
            },
        );
    let texture_options_changed = texture_options
        .as_ref()
        .is_some_and(|options| options.is_changed());
    let materials_resource_changed = materials.is_changed();
    let changed_materials: Option<HashSet<_>> = material_events.as_ref().map(|events| {
        material_event_cursor
            .read(events)
            .filter_map(|event| match event {
                AssetEvent::Added { id }
                | AssetEvent::Modified { id }
                | AssetEvent::Removed { id }
                | AssetEvent::LoadedWithDependencies { id } => Some(*id),
                AssetEvent::Unused { .. } => None,
            })
            .collect()
    });
    let materials_changed = changed_materials.as_ref().map_or_else(
        || materials_resource_changed,
        |changed| {
            !changed.is_empty()
                && geometry.iter().any(|(_, _, material, _, visible, ..)| {
                    !visible.as_ref().is_some_and(|visible| !visible.get())
                        && changed.contains(&material.0.id())
                })
        },
    );
    let changed_meshes: Option<HashSet<_>> = mesh_events.as_ref().map(|events| {
        mesh_event_cursor
            .read(events)
            .filter_map(|event| match event {
                AssetEvent::Added { id }
                | AssetEvent::Modified { id }
                | AssetEvent::Removed { id }
                | AssetEvent::LoadedWithDependencies { id } => Some(*id),
                AssetEvent::Unused { .. } => None,
            })
            .collect()
    });
    let mesh_assets_changed = changed_meshes.as_ref().map_or_else(
        || meshes.is_changed(),
        |changed| changed.is_empty() && meshes.is_changed(),
    );
    if let Some(changed) = &changed_meshes {
        if changed.is_empty() && meshes.is_changed() {
            mesh_cache.support.clear();
        } else {
            for id in changed {
                mesh_cache.support.remove(id);
            }
        }
    } else if meshes.is_changed() {
        mesh_cache.support.clear();
    }
    pending_entities.clear();
    let mut dirty = !scene.ready
        || poses_changed
        || images_changed
        || texture_options_changed
        || mesh_assets_changed
        || materials_changed
        || visibility_removed
        || layers_removed
        || lightmaps_removed
        || rebuilt.is_some()
        || !rebuilt_blas.is_empty();
    let mut geometry_count = 0;
    let mut changed_geometry = Vec::new();
    for (entity, mesh, material, transform, visible, layers, lightmap) in &geometry {
        geometry_count += 1;
        if !texture_cache.permanent_failure
            && !visible.as_ref().is_some_and(|visible| !visible.get())
            && let Some(material_asset) = materials.get(&material.0)
            && let Some(mesh_asset) = meshes.get(&mesh.0)
            && mesh_cache
                .support
                .entry(mesh.0.id())
                .or_insert_with(|| mesh_support(mesh_asset))
                .is_ok()
            && crate::material::material_mesh_layout_support(material_asset, mesh_asset).is_ok()
        {
            pending_entities.insert(entity);
        }
        let is_new = !previous_geometry.contains(&entity);
        let mesh_asset_changed = changed_meshes
            .as_ref()
            .is_some_and(|changed| changed.contains(&mesh.0.id()));
        let mesh_changed = mesh.is_changed();
        let material_changed = material.is_changed();
        let transform_tick_changed = transform.is_changed();
        let transform_value_changed =
            transform_tick_changed && changed_instance_transform(&scene, entity, &transform);
        let visibility_changed = visible.as_ref().is_some_and(|visible| visible.is_changed());
        let layers_changed = layers.as_ref().is_some_and(|layers| layers.is_changed());
        let lightmap_changed = lightmap
            .as_ref()
            .is_some_and(|lightmap| lightmap.is_changed());
        let entity_dirty = is_new
            || mesh_asset_changed
            || mesh_changed
            || material_changed
            || transform_value_changed
            || visibility_changed
            || layers_changed
            || lightmap_changed;
        if entity_dirty {
            dirty = true;
            let mut changed = Vec::new();
            if is_new {
                changed.push("new");
            }
            if mesh_asset_changed {
                changed.push("mesh_asset_event");
            }
            if mesh_changed {
                changed.push("mesh_handle");
            }
            if material_changed {
                changed.push("material_handle");
            }
            if transform_tick_changed {
                changed.push("transform_tick");
            }
            if transform_value_changed {
                changed.push("transform_value");
            }
            if visibility_changed {
                changed.push("visibility");
            }
            if layers_changed {
                changed.push("layers");
            }
            if lightmap_changed {
                changed.push("lightmap");
            }
            changed_geometry.push((entity, changed));
        }
    }
    if scene.pending_entities.as_ref() != &*pending_entities {
        scene.pending_entities = Arc::new(pending_entities.clone());
    }
    // Removing a required input drops the entity from the query. Membership
    // catches every such removal without rebuilding for unrelated despawns.
    dirty |= geometry_count != previous_geometry.len();
    if dirty {
        debug!(
            images_changed,
            images_resource_changed,
            materials_changed,
            materials_resource_changed,
            mesh_assets_changed,
            poses_changed,
            texture_options_changed,
            visibility_removed,
            layers_removed,
            lightmaps_removed,
            tlas_rebuilt = rebuilt.is_some(),
            rebuilt_blas = ?rebuilt_blas,
            pose_dirty_entities = ?pose_dirty_entities,
            changed_geometry = ?changed_geometry,
            geometry_count,
            previous_geometry_count = previous_geometry.len(),
            "GI scene rebuild requested"
        );
        previous_geometry.clear();
        previous_geometry.extend(geometry.iter().map(|(entity, ..)| entity));
    }

    let mut lights = std::mem::take(&mut *light_scratch);
    lights.clear();
    let mut light_layers = std::mem::take(&mut *light_layers_scratch);
    light_layers.clear();
    for (light, transform, visible, layers) in &directional {
        if visible.is_some_and(|v| !v.get())
            || !light.illuminance.is_finite()
            || light.illuminance <= 0.0
        {
            continue;
        }
        let gpu_light = GiLight {
            position: (-transform.forward()).extend(0.0),
            color: (light.color.to_linear().to_vec4().truncate() * light.illuminance).extend(0.0),
            ..default()
        };
        if gpu_light.position.is_finite() && gpu_light.color.is_finite() {
            lights.push(gpu_light);
            light_layers.push(layers.cloned().unwrap_or_default());
        }
    }
    for (light, transform, visible, layers) in &points {
        if visible.is_some_and(|v| !v.get()) {
            continue;
        }
        if let Some(light) = GiLight::punctual(
            transform,
            light.color,
            light.intensity,
            light.range,
            light.radius,
        ) {
            lights.push(light);
            light_layers.push(layers.cloned().unwrap_or_default());
        }
    }
    for (light, transform, visible, layers) in &spots {
        if visible.is_some_and(|v| !v.get())
            || !light.inner_angle.is_finite()
            || !light.outer_angle.is_finite()
            || light.inner_angle < 0.0
            || light.inner_angle > light.outer_angle
            || light.outer_angle >= std::f32::consts::FRAC_PI_2
        {
            continue;
        }
        if let Some(mut gpu_light) = GiLight::punctual(
            transform,
            light.color,
            light.intensity,
            light.range,
            light.radius,
        ) {
            let cos_outer = light.outer_angle.cos();
            let scale = 1.0 / (light.inner_angle.cos() - cos_outer).max(1e-4);
            gpu_light.direction = transform.forward().extend(scale);
            gpu_light.color.w = -cos_outer * scale;
            if gpu_light.direction.is_finite() {
                lights.push(gpu_light);
                light_layers.push(layers.cloned().unwrap_or_default());
            }
        }
    }
    if !dirty
        && scene.data.lights.as_ref() == &lights
        && scene.data.light_layers.as_ref() == &light_layers
    {
        *light_scratch = lights;
        *light_layers_scratch = light_layers;
        return;
    }
    if !dirty {
        scene.data = Arc::new(SceneData {
            lights: Arc::new(lights),
            light_layers: Arc::new(light_layers),
            ..scene.data.as_ref().clone()
        });
        scene.revision = scene.revision.wrapping_add(1);
        scene.light_revision = scene.light_revision.wrapping_add(1);
        return;
    }

    let mut descriptors = Vec::new();
    let texture_ids = geometry
        .iter()
        .filter(|(_, _, _, _, visible, _, _)| {
            !visible.as_ref().is_some_and(|visible| !visible.get())
        })
        .filter_map(|(_, _, material, ..)| materials.get(&material.0))
        .filter(|material| material_support(material).is_ok())
        .flat_map(|material| {
            texture_handles(material)
                .into_iter()
                .flatten()
                .map(Handle::id)
        })
        .collect();
    let defaults = TextureOptions::default();
    if let Err(reason) = texture_cache.update(
        texture_ids,
        images.as_deref(),
        texture_options.as_deref().unwrap_or(&defaults),
        images_changed || texture_options_changed,
    ) {
        if scene.ready {
            scene.revision = scene.revision.wrapping_add(1);
        }
        scene.ready = false;
        warn_once!("GI scene pending texture residency: {reason}");
        *light_scratch = lights;
        *light_layers_scratch = light_layers;
        return;
    }
    let mut material_indices = HashMap::new();
    let mut packed_materials = Vec::new();
    let mut checked = HashSet::new();
    for (entity, mesh_handle, material_handle, transform, visible, layers, lightmap) in &geometry {
        if visible.is_some_and(|v| !v.get()) {
            continue;
        }
        let (Some(mesh), Some(material)) = (
            meshes.get(&mesh_handle.0),
            materials.get(&material_handle.0),
        ) else {
            continue;
        };
        if let Err(reason) = crate::material::material_mesh_layout_support(material, mesh) {
            warn!(?entity, "GI material rejected: {reason}");
            continue;
        }
        if mesh.primitive_topology()
            != bevy::render::render_resource::PrimitiveTopology::TriangleList
        {
            warn!(?entity, "GI requires TriangleList meshes");
            continue;
        }
        let Some(VertexAttributeValues::Float32x3(positions)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            warn!(
                ?entity,
                "GI requires CPU-readable Float32x3 positions; retain MAIN_WORLD mesh usage"
            );
            continue;
        };
        let id = mesh_handle.0.id();
        let mesh_changed = changed_meshes.as_ref().map_or_else(
            || meshes.is_changed(),
            |changed| changed.contains(&id) || (changed.is_empty() && meshes.is_changed()),
        );
        // ponytail: recheck all used assets after residency failure, since mesh
        // events may have been consumed before the failed texture update.
        if checked.insert(id)
            && (mesh_changed
                || mesh_cache.needs_validation.contains(&id)
                || !scene.ready
                || !mesh_cache.meshes.contains_key(&id)
                || rebuilt_blas.contains(&id))
        {
            mesh_cache.needs_validation.remove(&id);
            let vertices = match mesh_vertices(mesh, positions) {
                Ok(vertices) => vertices,
                Err(reason) => {
                    warn!(?entity, ?id, "GI mesh rejected: {reason}");
                    mesh_cache.meshes.remove(&id);
                    mesh_cache.pending_blas.remove(&id);
                    continue;
                }
            };
            let geometry_signature = geometry_signature(mesh, positions);
            let deformation = match crate::deformation::MeshDeformation::new(mesh, positions.len())
            {
                Ok(data) => data,
                Err(reason) => {
                    warn!(?entity, "GI deformation rejected: {reason}");
                    mesh_cache.meshes.remove(&id);
                    mesh_cache.pending_blas.remove(&id);
                    continue;
                }
            };
            let mut hash = std::collections::hash_map::DefaultHasher::new();
            mesh_signature(mesh, geometry_signature).hash(&mut hash);
            deformation.hash(&mut hash);
            let signature = hash.finish();
            if mesh_cache
                .meshes
                .get(&id)
                .is_none_or(|cached| cached.signature != signature)
            {
                if mesh_cache
                    .meshes
                    .get(&id)
                    .is_some_and(|cached| cached.geometry_signature == geometry_signature)
                {
                    mesh_cache.pending_blas.remove(&id);
                    let cached = mesh_cache.meshes.get_mut(&id).expect("checked cached BLAS");
                    cached.signature = signature;
                    cached.vertices = vertices;
                    cached.deformation = deformation;
                } else {
                    mesh_cache.meshes.remove(&id);
                    let pending_matches = mesh_cache.pending_blas.get(&id).is_some_and(|pending| {
                        pending.signature == signature
                            && pending.geometry_signature == geometry_signature
                    });
                    if !pending_matches {
                        mesh_cache.pending_blas.remove(&id);
                        // Keep triangle expansion off the main scene-update path.
                        // Copy only the compact source index array; the validated
                        // vertex data is moved into the worker and returned with
                        // its BLAS output.
                        let indices = mesh.indices().cloned();
                        let preserve_degenerate = mesh.get_morph_targets().is_some()
                            || mesh.attribute(Mesh::ATTRIBUTE_JOINT_INDEX).is_some();
                        let task = AsyncComputeTaskPool::get().spawn(async move {
                            let mut triangles =
                                local_triangles(&vertices, indices.as_ref(), preserve_degenerate);
                            #[cfg(not(target_arch = "wasm32"))]
                            let (nodes, triangles) = {
                                let cache_dir = blas_cache_dir();
                                if let Some(cached) =
                                    load_blas_cache(&cache_dir, geometry_signature, triangles.len())
                                {
                                    cached
                                } else {
                                    let nodes = build_bvh(&mut triangles);
                                    let _ = save_blas_cache(
                                        &cache_dir,
                                        geometry_signature,
                                        &nodes,
                                        &triangles,
                                    );
                                    (nodes, triangles)
                                }
                            };
                            #[cfg(target_arch = "wasm32")]
                            let (nodes, triangles) = {
                                let nodes = build_bvh(&mut triangles);
                                (nodes, triangles)
                            };
                            BlasBuildOutput {
                                signature,
                                geometry_signature,
                                deformation,
                                vertices,
                                triangles,
                                nodes,
                            }
                        });
                        mesh_cache.pending_blas.insert(
                            id,
                            PendingBlas {
                                signature,
                                geometry_signature,
                                task,
                            },
                        );
                    }
                }
            }
        }
        if mesh_cache
            .meshes
            .get(&id)
            .is_none_or(|mesh| mesh.nodes.is_empty())
        {
            continue;
        }
        let albedo = if material.unlit {
            Vec3::ZERO
        } else {
            material
                .base_color
                .to_linear()
                .to_vec4()
                .truncate()
                .clamp(Vec3::ZERO, Vec3::ONE)
                * (1.0 - material.metallic.clamp(0.0, 1.0))
        };
        let emission = if material.unlit {
            material.base_color.to_linear().to_vec4().truncate()
        } else {
            material.emissive.to_vec4().truncate()
        };
        if !albedo.is_finite() || !emission.is_finite() {
            warn!(?entity, "GI ignores non-finite material radiance");
            continue;
        }
        let (material_index, material_residency) = if let Some(&entry) =
            material_indices.get(&material_handle.0.id())
        {
            entry
        } else {
            let words = match material_words(material, &texture_cache.atlas, images.as_deref()) {
                Ok(words) => words,
                Err(reason) => {
                    warn!(?entity, "GI material rejected: {reason}");
                    continue;
                }
            };
            let index = (packed_materials.len() / MATERIAL_WORDS) as u32;
            let mut hash = std::collections::hash_map::DefaultHasher::new();
            words.hash(&mut hash);
            let residency = hash.finish();
            material_indices.insert(material_handle.0.id(), (index, residency));
            packed_materials.extend(words);
            (index, residency)
        };
        let world = transform.affine();
        let inverse = world.inverse();
        if !world.is_finite() || !inverse.is_finite() {
            warn!(
                ?entity,
                "GI requires a finite invertible instance transform"
            );
            continue;
        }
        let flags = match material.cull_mode {
            Some(Face::Back) => 1,
            Some(Face::Front) => 2,
            None => 0,
        } | if world.matrix3.determinant() < 0.0 {
            4
        } else {
            0
        } | (u32::from(material.double_sided) << 3)
            | (u32::from(material.unlit) * INSTANCE_FLAG_UNLIT)
            | (u32::from(lightmap.is_some()) * INSTANCE_FLAG_BAKED_RECEIVER);
        let flags = flags | (u32::from(poses.contains_key(&entity)) << 6);
        let mut residency = std::collections::hash_map::DefaultHasher::new();
        mesh_cache.meshes[&id].signature.hash(&mut residency);
        material_residency.hash(&mut residency);
        flags.hash(&mut residency);
        layer_words(layers.as_deref().unwrap_or(&RenderLayers::default())).hash(&mut residency);
        descriptors.push((
            entity,
            id,
            world,
            inverse,
            albedo,
            emission.max(Vec3::ZERO),
            flags,
            material_index,
            layers.as_deref().cloned().unwrap_or_default(),
            residency.finish(),
        ));
    }
    descriptors.sort_by_key(|descriptor| descriptor.0.to_bits());
    let animated = descriptors
        .iter()
        .filter(|descriptor| poses.contains_key(&descriptor.0))
        .map(|descriptor| (descriptor.0, descriptor.1))
        .collect();
    mesh_cache.pack(
        descriptors.iter().map(|descriptor| descriptor.1).collect(),
        animated,
    );
    let instance_meshes: HashMap<_, _> = descriptors
        .iter()
        .map(|descriptor| (descriptor.0, descriptor.1))
        .collect();
    let mut instances = Vec::new();
    let mut instance_residency = HashMap::new();
    for (entity, id, world, inverse, albedo, emission, flags, material, layers, residency) in
        descriptors
    {
        let span = mesh_cache
            .animated_spans
            .get(&entity)
            .copied()
            .unwrap_or(mesh_cache.spans[&id]);
        let mut bounds = mesh_cache.nodes[span.root as usize];
        if let Some(pose) = poses.get(&entity) {
            let pose = match pose {
                Ok(pose) => pose,
                Err(reason) => {
                    warn!(?entity, "GI deformation pending: {reason}");
                    continue;
                }
            };
            let data = &mesh_cache.meshes[&id].deformation;
            if let Err(reason) = data.validate_pose(pose) {
                warn!(?entity, "GI deformation rejected: {reason}");
                continue;
            }
            (bounds.min, bounds.max) = data.conservative_bounds(bounds.min, bounds.max, pose);
        }
        let center = world.transform_point3(bounds.min * 0.5 + bounds.max * 0.5);
        let local_extent = bounds.max * 0.5 - bounds.min * 0.5;
        let extent: Vec3 = (world.matrix3.x_axis.abs() * local_extent.x
            + world.matrix3.y_axis.abs() * local_extent.y
            + world.matrix3.z_axis.abs() * local_extent.z)
            .into();
        let absolute = bounds.min.abs().max(bounds.max.abs());
        let error: Vec3 = ((world.translation.abs()
            + world.matrix3.x_axis.abs() * absolute.x
            + world.matrix3.y_axis.abs() * absolute.y
            + world.matrix3.z_axis.abs() * absolute.z)
            * (8.0 * f32::EPSILON))
            .into();
        let min = center - extent - error;
        let max = center + extent + error;
        if !min.is_finite() || !max.is_finite() {
            warn!(
                ?entity,
                "GI instance bounds exceed local f32 precision; rebase or rescale the scene"
            );
            continue;
        }
        instances.push(SceneInstance {
            entity,
            world_from_local: world,
            local_from_world: inverse,
            min,
            max,
            root: span.root,
            first: span.first,
            count: span.count,
            albedo,
            emission,
            flags,
            material,
            layers,
        });
        instance_residency.insert(entity, residency);
    }
    // Emitter ordering follows stable entity/mesh order, independently of TLAS
    // partition/refit. Only emitting triangles require world-space expansion.
    let mut emitters = Vec::new();
    let mut emitter_keys = Vec::new();
    for instance in &instances {
        if instance.emission.max_element() <= 0.0 {
            continue;
        }
        for i in 0..instance.count {
            let mut local = mesh_cache.triangles[(instance.first + i) as usize];
            if let Some(Ok(pose)) = poses.get(&instance.entity) {
                let id = instance_meshes[&instance.entity];
                let span = mesh_cache.animated_spans[&instance.entity];
                let data = &mesh_cache.meshes[&id].deformation;
                let points = local.indices.to_array().map(|index| {
                    data.deform_point(
                        (index - span.vertex) as usize,
                        mesh_cache.vertices[index as usize].position,
                        pose,
                    )
                });
                [local.a, local.b, local.c] = points.map(|point| point.extend(0.0));
            }
            let mut triangle = instance.triangle(local);
            if instance.flags & INSTANCE_FLAG_UNLIT != 0 {
                // A power estimate may be approximate, but all potentially
                // emitting vertex-colored regions retain positive probability.
                let color = local
                    .indices
                    .to_array()
                    .map(|index| {
                        mesh_cache.vertices[index as usize]
                            .color
                            .truncate()
                            .max(Vec3::ZERO)
                    })
                    .into_iter()
                    .sum::<Vec3>()
                    / 3.0;
                triangle.emission *= color.extend(1.0);
            }
            triangle.emission.w = emitter_mask_coverage(
                instance.material,
                local.indices,
                &mesh_cache.vertices,
                &packed_materials,
                &texture_cache.atlas,
            );
            emitters.push(triangle);
            emitter_keys.push((instance.entity, i, instance.first + i));
        }
    }
    let mut emissive_lights = match build_emissive_lights(&mut emitters) {
        Ok(lights) => lights,
        Err(reason) => {
            scene.ready = false;
            bevy::log::warn_once!("GI scene update skipped: {reason}");
            return;
        }
    };
    let mut by_entity: HashMap<_, _> = instances
        .iter()
        .map(|instance| (instance.entity, instance.clone()))
        .collect();
    let nodes = if rebuilt.as_ref().is_some_and(|(_, order)| {
        order.len() == instances.len() && order.iter().all(|entity| by_entity.contains_key(entity))
    }) {
        let (mut nodes, order) = rebuilt.take().expect("checked completed rebuild");
        instances = order
            .iter()
            .map(|entity| by_entity.remove(entity).expect("checked membership"))
            .collect();
        // The worker used a snapshot. Refit to this frame's transforms before
        // committing, so motion during the build never publishes stale bounds.
        refit(&mut nodes, &instances);
        mesh_cache.tlas_build_cost = tree_cost(&nodes);
        scene.tlas_builds = scene.tlas_builds.wrapping_add(1);
        scene.topology_revision = scene.topology_revision.wrapping_add(1);
        nodes
    } else if instances.len() == scene.data.instances.len()
        && scene
            .data
            .instances
            .iter()
            .all(|instance| by_entity.contains_key(&instance.entity))
    {
        instances = scene
            .data
            .instances
            .iter()
            .map(|instance| {
                by_entity
                    .remove(&instance.entity)
                    .expect("checked membership")
            })
            .collect();
        let mut nodes = scene.data.nodes.as_ref().clone();
        refit(&mut nodes, &instances);
        scene.tlas_refits = scene.tlas_refits.wrapping_add(1);
        nodes
    } else {
        mesh_cache.tlas_rebuild = None;
        let nodes = build_tree(&mut instances, |instance| (instance.min, instance.max));
        mesh_cache.tlas_build_cost = tree_cost(&nodes);
        scene.tlas_builds = scene.tlas_builds.wrapping_add(1);
        scene.topology_revision = scene.topology_revision.wrapping_add(1);
        nodes
    };
    let instance_index_map: HashMap<_, _> = instances
        .iter()
        .enumerate()
        .map(|(i, instance)| (instance.entity, i as u32))
        .collect();
    let instance_indices = if scene.data.instance_indices.as_ref() == &instance_index_map {
        scene.data.instance_indices.clone()
    } else {
        Arc::new(instance_index_map)
    };
    for (entity, residency) in &mut instance_residency {
        let mut indexed_residency = std::collections::hash_map::DefaultHasher::new();
        residency.hash(&mut indexed_residency);
        instance_indices[entity].hash(&mut indexed_residency);
        *residency = indexed_residency.finish();
    }
    let instance_residency = if scene.data.instance_residency.as_ref() == &instance_residency {
        scene.data.instance_residency.clone()
    } else {
        Arc::new(instance_residency)
    };
    for light in &mut emissive_lights {
        let (entity, _, triangle) = emitter_keys[light.source.y as usize];
        light.source = UVec4::new(instance_indices[&entity], triangle, 0, 0);
    }
    let emissive_lights = Arc::new(emissive_lights);
    let cost = tree_cost(&nodes);
    if mesh_cache.tlas_rebuild.is_none()
        && mesh_cache.tlas_build_cost > 0.0
        && cost > mesh_cache.tlas_build_cost * 1.5
    {
        let mut snapshot = instances.clone();
        mesh_cache.tlas_rebuild = Some(AsyncComputeTaskPool::get().spawn(async move {
            let nodes = build_tree(&mut snapshot, |instance| (instance.min, instance.max));
            let order = snapshot.iter().map(|instance| instance.entity).collect();
            (nodes, order)
        }));
    }
    let blas_words = if nodes.len() == scene.data.nodes.len()
        && Arc::ptr_eq(&mesh_cache.nodes, &scene.data.blas_nodes)
    {
        scene.data.blas_words.clone()
    } else {
        Arc::new(node_words(&mesh_cache.nodes, nodes.len() as u32))
    };
    let base = 1 + nodes.len() * 2 + blas_words.len();
    let mut prefix = vec![UVec4::new(
        (base + instances.len() * INSTANCE_WORDS + packed_materials.len()) as u32,
        base as u32,
        instances.len() as u32,
        INSTANCE_WORDS as u32,
    )];
    prefix.extend(node_words(&nodes, 0));
    let instance_words = Arc::new(pack_instances(
        &instances,
        nodes.len(),
        base,
        &packed_materials,
        &texture_cache.atlas.words,
    ));
    let mut deformation = crate::deformation::DeformationData::default();
    let mut streams = vec![UVec4::ZERO];
    let mut jobs = Vec::new();
    let mut vertex_map = Vec::new();
    let mut pose_stream = Vec::new();
    let mut refit_levels: Vec<Vec<u32>> = Vec::new();
    for instance in &instances {
        let Some(Ok(pose)) = poses.get(&instance.entity) else {
            continue;
        };
        let id = instance_meshes[&instance.entity];
        let source = mesh_cache.spans[&id];
        let destination = mesh_cache.animated_spans[&instance.entity];
        let mesh = &mesh_cache.meshes[&id];
        let influence = streams.len() as u32;
        streams.extend_from_slice(&mesh.deformation.influences);
        let morph = streams.len() as u32;
        streams.extend_from_slice(&mesh.deformation.morphs);
        let (matrices, weights) = crate::deformation::pose_words(pose, &mut pose_stream);
        let job = jobs.len() as u32 / 3;
        jobs.extend([
            UVec4::new(
                source.vertex,
                destination.vertex,
                mesh.vertices.len() as u32,
                influence,
            ),
            UVec4::new(morph, mesh.deformation.target_count, matrices, weights),
            UVec4::new(pose.matrices.len() as u32, 0, 0, 0),
        ]);
        let overlay_vertex_base = vertex_map.len() as u32;
        vertex_map.extend(
            (0..mesh.vertices.len() as u32)
                .map(|vertex| UVec4::new(job, vertex, overlay_vertex_base + vertex, 0)),
        );
        let mut stack = vec![(destination.root, 0usize)];
        while let Some((index, depth)) = stack.pop() {
            if refit_levels.len() <= depth {
                refit_levels.resize_with(depth + 1, Vec::new);
            }
            refit_levels[depth].push(index + nodes.len() as u32);
            let node = mesh_cache.nodes[index as usize];
            if node.count == 0 {
                stack.extend([(node.first, depth + 1), (node.first + 1, depth + 1)]);
            }
        }
    }
    let has_deformation_jobs = !jobs.is_empty();
    if has_deformation_jobs {
        let job_base = streams.len() as u32;
        streams.extend(jobs);
        let map_base = streams.len() as u32;
        deformation.vertices = vertex_map.len() as u32;
        streams.extend(vertex_map);
        streams[0] = UVec4::new((map_base - job_base) / 3, job_base, map_base, 0);
        deformation.words = if *scene.data.deformation.words == streams {
            scene.data.deformation.words.clone()
        } else {
            Arc::new(streams)
        };
        deformation.poses = Arc::new(if pose_stream.is_empty() {
            vec![UVec4::ZERO]
        } else {
            pose_stream
        });
        for level in refit_levels.into_iter().rev() {
            deformation
                .refit_levels
                .push((deformation.refit_nodes.len() as u32, level.len() as u32));
            deformation.refit_nodes.extend(level);
        }
    }
    if !has_deformation_jobs && scene.data.deformation.words.is_empty() {
        deformation.words = scene.data.deformation.words.clone();
    }
    // ponytail: structural deformation edits restage base pages with the overlay;
    // split their revisions only if those rare uploads become costly.
    if !Arc::ptr_eq(&mesh_cache.geometry_words, &scene.data.geometry_words)
        || !Arc::ptr_eq(&deformation.words, &scene.data.deformation.words)
    {
        scene.mesh_revision = scene.mesh_revision.wrapping_add(1);
    }
    let lighting_changed = scene.data.lights.as_ref() != &lights
        || scene.data.light_layers.as_ref() != &light_layers
        || scene.data.emissive_lights.as_ref() != emissive_lights.as_ref();
    scene.data = Arc::new(SceneData {
        deformation: Arc::new(deformation),
        triangles: mesh_cache.triangles.clone(),
        vertices: mesh_cache.vertices.clone(),
        geometry_words: mesh_cache.geometry_words.clone(),
        #[cfg(test)]
        material_words: Arc::new(packed_materials),
        textures: texture_cache.atlas.clone(),
        nodes: Arc::new(nodes),
        blas_nodes: mesh_cache.nodes.clone(),
        instances: Arc::new(instances),
        acceleration_prefix: Arc::new(prefix),
        blas_words,
        instance_words,
        instance_indices,
        instance_residency,
        lights: Arc::new(lights),
        light_layers: Arc::new(light_layers),
        emissive_lights,
    });
    scene.revision = scene.revision.wrapping_add(1);
    if lighting_changed {
        scene.light_revision = scene.light_revision.wrapping_add(1);
    }
    scene.geometry_revision = scene.geometry_revision.wrapping_add(1);
    scene.ready = true;
    debug!(
        revision = scene.revision,
        triangles = scene.data.triangles.len(),
        vertices = scene.data.vertices.len(),
        "GI scene updated"
    );
}

fn emitter_mask_coverage(
    material: u32,
    indices: UVec3,
    vertices: &[MeshVertex],
    materials: &[UVec4],
    atlas: &TextureAtlas,
) -> f32 {
    let start = material as usize * MATERIAL_WORDS;
    let Some(words) = materials.get(start..start + MATERIAL_WORDS) else {
        return 1.0;
    };
    let cutoff = f32::from_bits(words[1].w);
    if cutoff == -1.0 {
        return 1.0;
    }
    let [Some(a), Some(b), Some(c)] = indices
        .to_array()
        .map(|index| vertices.get(index as usize).copied())
    else {
        return 1.0;
    };
    let uv1 = words[7].y & 1 != 0;
    let uv_x = Vec3::new(
        f32::from_bits(words[4].x),
        f32::from_bits(words[4].y),
        f32::from_bits(words[4].z),
    );
    let uv_y = Vec3::new(
        f32::from_bits(words[5].x),
        f32::from_bits(words[5].y),
        f32::from_bits(words[5].z),
    );
    let uv = |vertex: MeshVertex| {
        if uv1 {
            Vec2::new(vertex.uv.z, vertex.uv.w)
        } else {
            Vec2::new(vertex.uv.x, vertex.uv.y)
        }
    };
    let uvs = [uv(a), uv(b), uv(c)];
    let alphas = [a.color.w, b.color.w, c.color.w];
    const SAMPLES: [[f32; 3]; 4] = [
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [0.6, 0.2, 0.2],
        [0.2, 0.6, 0.2],
        [0.2, 0.2, 0.6],
    ];
    let base_alpha = f32::from_bits(words[0].w);
    let texture = words[6].x;
    let coverage: f32 = SAMPLES
        .into_iter()
        .map(|bary| {
            let uv = uvs[0] * bary[0] + uvs[1] * bary[1] + uvs[2] * bary[2];
            let alpha = (alphas[0] * bary[0] + alphas[1] * bary[1] + alphas[2] * bary[2])
                * base_alpha
                * atlas.sample_alpha(
                    texture,
                    Vec2::new(
                        uv_x.x * uv.x + uv_x.y * uv.y + uv_x.z,
                        uv_y.x * uv.x + uv_y.y * uv.y + uv_y.z,
                    ),
                );
            if cutoff == -2.0 {
                alpha.clamp(0.0, 1.0)
            } else if alpha >= cutoff {
                1.0
            } else {
                0.0
            }
        })
        .sum();
    coverage / SAMPLES.len() as f32
}

fn build_emissive_lights(triangles: &mut [Triangle]) -> Result<Vec<GiLight>, &'static str> {
    // Match the exact discrete support of randomFloat in the shader.
    const CDF_STEPS: u32 = 1 << 24;
    let mut lights = Vec::new();
    let mut weights = Vec::new();
    for (index, triangle) in triangles.iter_mut().enumerate() {
        // ponytail: four LOD-0 samples can miss tiny mask islands; a 1/64 CDF
        // floor preserves their sampling probability, with more samples only if
        // measured emitter variance warrants the added scene-build cost.
        let coverage = triangle.emission.w.clamp(1.0 / 64.0, 1.0);
        triangle.emission.w = 0.0;
        let emission = triangle.emission.truncate().max(Vec3::ZERO);
        let edge1 = (triangle.b - triangle.a).truncate();
        let edge2 = (triangle.c - triangle.a).truncate();
        let cross = edge1.cross(edge2);
        let area = cross.length() * 0.5;
        let power = f64::from(area)
            * f64::from(emission.dot(Vec3::new(0.2126, 0.7152, 0.0722)))
            * f64::from(coverage);
        if !power.is_finite() || power <= 0.0 {
            continue;
        }
        let cull = triangle.albedo.w as u32 & 3;
        lights.push(GiLight {
            position: triangle.a.with_w(if cull == 0 { -2.0 } else { -1.0 }),
            color: emission.extend(area),
            direction: (cross.normalize() * if cull == 2 { -1.0 } else { 1.0 }).extend(0.0),
            edge1: edge1.extend(0.0),
            edge2: edge2.extend(0.0),
            source: UVec4::new(0, index as u32, 0, 0),
        });
        weights.push(power);
    }
    if lights.len() > CDF_STEPS as usize {
        return Err("emissive triangle count exceeds the 24-bit sampling grid");
    }
    let total: f64 = weights.iter().sum();
    if !lights.is_empty() && (!total.is_finite() || total <= 0.0) {
        return Err("emissive triangle power total is not finite and positive");
    }
    let mut cumulative_power = 0.0;
    let mut previous_boundary = 0;
    let light_count = lights.len();
    for (offset, (light, weight)) in lights.iter_mut().zip(weights).enumerate() {
        cumulative_power += weight;
        let boundary = if offset + 1 == light_count {
            CDF_STEPS
        } else {
            let ideal = (cumulative_power / total * f64::from(CDF_STEPS)).round() as u32;
            let max_boundary = CDF_STEPS - (light_count - offset - 1) as u32;
            ideal.clamp(previous_boundary + 1, max_boundary)
        };
        let probability = (boundary - previous_boundary) as f32 / CDF_STEPS as f32;
        debug_assert!(boundary > previous_boundary);
        light.direction.w = boundary as f32 / CDF_STEPS as f32;
        light.edge1.w = probability;
        // The secondary NEE sampler uses this CDF boundary and discrete PDF.
        previous_boundary = boundary;
    }
    debug_assert!(lights.is_empty() || previous_boundary == CDF_STEPS);
    Ok(lights)
}

fn build_bvh(triangles: &mut [Triangle]) -> Vec<BvhNode> {
    build_tree(triangles, Triangle::bounds)
}

fn bounds_area(min: Vec3, max: Vec3) -> f64 {
    let size = max.as_dvec3() - min.as_dvec3();
    2.0 * (size.x * size.y + size.y * size.z + size.z * size.x)
}

fn tree_cost(nodes: &[BvhNode]) -> f64 {
    let Some(root) = nodes.first() else {
        return 0.0;
    };
    let area = bounds_area(root.min, root.max);
    if area <= 0.0 {
        return 0.0;
    }
    nodes
        .iter()
        .map(|node| bounds_area(node.min, node.max) * f64::from(node.count.max(1)))
        .sum::<f64>()
        / area
}

fn build_tree<T>(elements: &mut [T], bounds: impl Fn(&T) -> (Vec3, Vec3)) -> Vec<BvhNode> {
    // Binned SAH, with a median fallback for coincident centroids. Depth is capped
    // at 48, bounding traversal's depth-first stacks to 49 pending nodes.
    if elements.is_empty() {
        return Vec::new();
    }
    fn build<T>(
        nodes: &mut Vec<BvhNode>,
        index: usize,
        elements: &mut [T],
        first: usize,
        depth: u32,
        bounds: &impl Fn(&T) -> (Vec3, Vec3),
    ) {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for element in elements.iter() {
            let (a, b) = bounds(element);
            min = min.min(a);
            max = max.max(b);
        }
        nodes[index] = BvhNode {
            min,
            max,
            first: first as u32,
            count: elements.len() as u32,
        };
        if elements.len() <= 4 || depth == 48 {
            return;
        }
        const BINS: usize = 12;
        #[derive(Clone, Copy)]
        struct Bin {
            min: Vec3,
            max: Vec3,
            count: usize,
        }
        impl Bin {
            const EMPTY: Self = Self {
                min: Vec3::splat(f32::INFINITY),
                max: Vec3::splat(f32::NEG_INFINITY),
                count: 0,
            };
            fn join(self, other: Self) -> Self {
                Self {
                    min: self.min.min(other.min),
                    max: self.max.max(other.max),
                    count: self.count + other.count,
                }
            }
            fn cost(self) -> f64 {
                if self.count == 0 {
                    0.0
                } else {
                    bounds_area(self.min, self.max) * self.count as f64
                }
            }
        }
        let mut center_min = Vec3::splat(f32::INFINITY);
        let mut center_max = Vec3::splat(f32::NEG_INFINITY);
        for element in elements.iter() {
            let (a, b) = bounds(element);
            let center = a * 0.5 + b * 0.5;
            center_min = center_min.min(center);
            center_max = center_max.max(center);
        }
        let mut best_cost = f64::INFINITY;
        let mut split = None;
        for axis in 0..3 {
            let span = f64::from(center_max[axis]) - f64::from(center_min[axis]);
            if span <= 0.0 {
                continue;
            }
            let bin_index = |center: f32| {
                (((f64::from(center) - f64::from(center_min[axis])) / span * BINS as f64) as usize)
                    .min(BINS - 1)
            };
            let mut bins = [Bin::EMPTY; BINS];
            for element in elements.iter() {
                let (a, b) = bounds(element);
                let index = bin_index(a[axis] * 0.5 + b[axis] * 0.5);
                bins[index] = bins[index].join(Bin {
                    min: a,
                    max: b,
                    count: 1,
                });
            }
            let mut right = [Bin::EMPTY; BINS];
            let mut sum = Bin::EMPTY;
            for i in (0..BINS).rev() {
                sum = sum.join(bins[i]);
                right[i] = sum;
            }
            let mut left = Bin::EMPTY;
            for i in 0..BINS - 1 {
                left = left.join(bins[i]);
                let cost = left.cost() + right[i + 1].cost();
                if left.count > 0 && right[i + 1].count > 0 && cost < best_cost {
                    best_cost = cost;
                    split = Some((axis, i, span));
                }
            }
        }
        let midpoint = if let Some((axis, last_bin, span)) = split {
            let mut left = 0;
            let mut right = elements.len();
            while left < right {
                let (a, b) = bounds(&elements[left]);
                let bin = (((f64::from(a[axis] * 0.5 + b[axis] * 0.5)
                    - f64::from(center_min[axis]))
                    / span
                    * BINS as f64) as usize)
                    .min(BINS - 1);
                if bin <= last_bin {
                    left += 1;
                } else {
                    right -= 1;
                    elements.swap(left, right);
                }
            }
            left
        } else {
            let size = center_max - center_min;
            let axis = if size.x >= size.y && size.x >= size.z {
                0
            } else if size.y >= size.z {
                1
            } else {
                2
            };
            let midpoint = elements.len() / 2;
            elements.select_nth_unstable_by(midpoint, |a, b| {
                let a = bounds(a);
                let b = bounds(b);
                (a.0[axis] * 0.5 + a.1[axis] * 0.5).total_cmp(&(b.0[axis] * 0.5 + b.1[axis] * 0.5))
            });
            midpoint
        };
        let children = nodes.len();
        nodes.extend([BvhNode::default(); 2]);
        nodes[index].first = children as u32;
        nodes[index].count = 0;
        let (left, right) = elements.split_at_mut(midpoint);
        build(nodes, children, left, first, depth + 1, bounds);
        build(
            nodes,
            children + 1,
            right,
            first + midpoint,
            depth + 1,
            bounds,
        );
    }
    let mut nodes = vec![BvhNode::default()];
    build(&mut nodes, 0, elements, 0, 0, &bounds);
    nodes
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::asset::RenderAssetUsages;
    use bevy::mesh::morph::MeshMorphWeights;
    use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

    fn rgba8_image(color: [u8; 4]) -> Image {
        Image::new(
            Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            color.to_vec(),
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::default(),
        )
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

    #[derive(Resource, Default)]
    struct GlobalTransformChangeCount(usize);

    fn count_changed_global_transforms(
        changed: Query<Entity, Changed<GlobalTransform>>,
        mut count: ResMut<GlobalTransformChangeCount>,
    ) {
        count.0 = changed.iter().count();
    }

    #[test]
    fn equal_transform_rewrites_do_not_rebuild_scene_but_motion_updates_emitters() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(bevy::transform::TransformPlugin)
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<GiScene>()
            .init_resource::<GlobalTransformChangeCount>()
            .add_message::<AssetEvent<Mesh>>()
            .add_message::<AssetEvent<Image>>()
            .add_message::<AssetEvent<StandardMaterial>>()
            .add_systems(
                PostUpdate,
                (count_changed_global_transforms, update_scene)
                    .chain()
                    .after(bevy::transform::TransformSystems::Propagate),
            );
        app.world_mut()
            .spawn((Camera::default(), RadianceCascades::default()));
        let mesh = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(Cuboid::default());
        let material = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial {
                base_color: Color::BLACK,
                emissive: LinearRgba::WHITE,
                ..default()
            });
        let entity = app
            .world_mut()
            .spawn((
                Mesh3d(mesh),
                MeshMaterial3d(material),
                RadianceCascadesGeometry,
                Transform::IDENTITY,
                GlobalTransform::IDENTITY,
            ))
            .id();
        wait_for_blas_builds(&mut app, 1);
        app.update();
        let (revision, geometry_revision, emitter_positions) = {
            let scene = app.world().resource::<GiScene>();
            (
                scene.revision,
                scene.geometry_revision,
                scene
                    .data
                    .emissive_lights
                    .iter()
                    .map(|light| light.position.truncate())
                    .collect::<Vec<_>>(),
            )
        };
        assert!(!emitter_positions.is_empty());

        app.world_mut()
            .entity_mut(entity)
            .insert(Transform::IDENTITY);
        app.update();
        assert!(
            app.world().resource::<GlobalTransformChangeCount>().0 > 0,
            "equal Transform rewrite should propagate a Changed GlobalTransform"
        );
        let scene = app.world().resource::<GiScene>();
        assert_eq!(scene.revision, revision);
        assert_eq!(scene.geometry_revision, geometry_revision);

        let translation = Vec3::new(3.0, 2.0, -1.0);
        app.world_mut()
            .entity_mut(entity)
            .insert(Transform::from_translation(translation));
        app.update();
        assert!(app.world().resource::<GlobalTransformChangeCount>().0 > 0);
        let scene = app.world().resource::<GiScene>();
        assert!(scene.revision > revision);
        assert!(scene.geometry_revision > geometry_revision);
        let instance = scene
            .data
            .instances
            .iter()
            .find(|instance| instance.entity == entity)
            .unwrap();
        assert_eq!(
            instance.world_from_local.translation.to_array(),
            translation.to_array()
        );
        assert_eq!(scene.data.emissive_lights.len(), emitter_positions.len());
        for (before, after) in emitter_positions
            .iter()
            .zip(scene.data.emissive_lights.iter())
        {
            assert_eq!(after.position.truncate() - *before, translation);
        }
    }

    #[test]
    fn next_event_cdf_includes_all_supported_emissive_triangles() {
        let triangle = |x: f32, flags: u32| Triangle {
            a: Vec3::new(x, 0.0, 0.0).extend(0.0),
            b: Vec3::new(x + 1.0, 0.0, 0.0).extend(0.0),
            c: Vec3::new(x, 1.0, 0.0).extend(0.0),
            albedo: Vec3::ONE.extend(flags as f32),
            emission: Vec3::ONE.extend(1.0),
            ..default()
        };
        let mut triangles = [triangle(0.0, 0), triangle(2.0, 1 << 4)];

        let lights = build_emissive_lights(&mut triangles).unwrap();

        assert_eq!(lights.len(), 2);
        assert_eq!(lights[0].source.y, 0);
        assert_eq!(lights[1].source.y, 1);
        assert!(triangles.iter().all(|triangle| triangle.emission.w == 0.0));
        assert!((lights.iter().map(|light| light.edge1.w).sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(lights[0].direction.w, 0.5);
        assert_eq!(lights[1].direction.w, 1.0);
    }

    #[test]
    fn mesh_support_rejects_non_finite_positions() {
        let mut mesh = Mesh::from(Cuboid::default());
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vec![[f32::NAN, 0.0, 0.0]; 24]);
        assert_eq!(
            mesh_support(&mesh),
            Err("mesh vertex attributes must be finite")
        );
    }

    #[test]
    fn local_triangles_preserve_indexed_and_nonindexed_geometry() {
        let positions = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 1.0),
        ];
        let vertices: Vec<_> = positions
            .iter()
            .map(|&position| MeshVertex {
                position,
                normal: Vec3::ZERO,
                tangent: Vec4::ZERO,
                uv: Vec4::ZERO,
                color: Vec4::ONE,
                flags: 0,
            })
            .collect();
        let indexed_u16 = Indices::U16(vec![2, 0, 1, 5, 3, 4, 1]);
        let indexed_u32 = Indices::U32(vec![2, 0, 1, 5, 3, 4, 1]);
        let indexed = local_triangles(&vertices, Some(&indexed_u16), false);
        let indexed_32 = local_triangles(&vertices, Some(&indexed_u32), false);
        let nonindexed = local_triangles(&vertices, None, false);

        assert_eq!(
            indexed
                .iter()
                .map(|triangle| triangle.indices.to_array())
                .collect::<Vec<_>>(),
            vec![[2, 0, 1], [5, 3, 4]]
        );
        assert_eq!(
            indexed_32
                .iter()
                .map(|triangle| triangle.indices.to_array())
                .collect::<Vec<_>>(),
            vec![[2, 0, 1], [5, 3, 4]]
        );
        assert_eq!(
            nonindexed
                .iter()
                .map(|triangle| triangle.indices.to_array())
                .collect::<Vec<_>>(),
            vec![[0, 1, 2], [3, 4, 5]]
        );
        for (triangle, expected) in indexed.iter().zip([[2, 0, 1], [5, 3, 4]]) {
            assert_eq!(triangle.a.truncate(), positions[expected[0]]);
            assert_eq!(triangle.b.truncate(), positions[expected[1]]);
            assert_eq!(triangle.c.truncate(), positions[expected[2]]);
        }
        for (triangle, expected) in nonindexed.iter().zip([[0, 1, 2], [3, 4, 5]]) {
            assert_eq!(triangle.a.truncate(), positions[expected[0]]);
            assert_eq!(triangle.b.truncate(), positions[expected[1]]);
            assert_eq!(triangle.c.truncate(), positions[expected[2]]);
        }
        assert_eq!(
            indexed_32
                .iter()
                .map(|triangle| (triangle.indices, triangle.a, triangle.b, triangle.c))
                .collect::<Vec<_>>(),
            indexed
                .iter()
                .map(|triangle| (triangle.indices, triangle.a, triangle.b, triangle.c))
                .collect::<Vec<_>>()
        );
        let degenerate = vec![vertices[0]; 3];
        assert!(local_triangles(&degenerate, None, false).is_empty());
        assert_eq!(local_triangles(&degenerate, None, true).len(), 1);
    }

    #[test]
    fn unrelated_image_edits_do_not_rebuild_untextured_scene() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<Assets<Image>>()
            .init_resource::<GiScene>()
            .add_message::<AssetEvent<Image>>()
            .add_message::<AssetEvent<StandardMaterial>>()
            .add_systems(Update, update_scene);
        app.world_mut()
            .spawn((Camera::default(), RadianceCascades::default()));
        let mesh = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(Cuboid::default());
        let material = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial::default());
        let unrelated_image = app
            .world_mut()
            .resource_mut::<Assets<Image>>()
            .add(rgba8_image([255, 0, 0, 255]));
        let _entity = app
            .world_mut()
            .spawn((
                Mesh3d(mesh),
                MeshMaterial3d(material.clone()),
                RadianceCascadesGeometry,
                GlobalTransform::IDENTITY,
            ))
            .id();
        wait_for_blas_builds(&mut app, 1);
        app.update();
        let (revision, geometry_revision) = {
            let scene = app.world().resource::<GiScene>();
            (scene.revision, scene.geometry_revision)
        };

        // Text layout can mutably borrow Assets<Image> without changing any
        // image asset or emitting an AssetEvent. This must not invalidate GI.
        app.world_mut()
            .resource_mut::<Assets<Image>>()
            .set_changed();
        app.update();
        let scene = app.world().resource::<GiScene>();
        assert_eq!(scene.revision, revision);
        assert_eq!(scene.geometry_revision, geometry_revision);

        app.world_mut()
            .resource_mut::<Assets<Image>>()
            .get_mut(&unrelated_image)
            .unwrap()
            .data
            .as_mut()
            .unwrap()[0] = 17;
        app.world_mut()
            .resource_mut::<Messages<AssetEvent<Image>>>()
            .write(AssetEvent::Modified {
                id: unrelated_image.id(),
            });
        app.update();

        let scene = app.world().resource::<GiScene>();
        assert_eq!(scene.revision, revision);
        assert_eq!(scene.geometry_revision, geometry_revision);
    }

    #[test]
    fn referenced_image_edits_rebuild_scene_textures() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<Assets<Image>>()
            .init_resource::<GiScene>()
            .add_message::<AssetEvent<Image>>()
            .add_message::<AssetEvent<StandardMaterial>>()
            .add_systems(Update, update_scene);
        app.world_mut()
            .spawn((Camera::default(), RadianceCascades::default()));
        let mesh = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(Cuboid::default());
        let image = app
            .world_mut()
            .resource_mut::<Assets<Image>>()
            .add(rgba8_image([255, 0, 0, 255]));
        let material = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial {
                base_color_texture: Some(image.clone()),
                ..default()
            });
        let entity = app
            .world_mut()
            .spawn((
                Mesh3d(mesh),
                MeshMaterial3d(material.clone()),
                RadianceCascadesGeometry,
                GlobalTransform::IDENTITY,
            ))
            .id();
        wait_for_blas_builds(&mut app, 1);
        app.update();
        let (revision, geometry_revision, textures) = {
            let scene = app.world().resource::<GiScene>();
            (
                scene.revision,
                scene.geometry_revision,
                scene.data.textures.clone(),
            )
        };

        // A global resource borrow with no per-asset event should be ignored
        // even when the scene does reference a GI texture.
        app.world_mut()
            .resource_mut::<Assets<Image>>()
            .set_changed();
        app.update();
        let scene = app.world().resource::<GiScene>();
        assert_eq!(scene.revision, revision);
        assert_eq!(scene.geometry_revision, geometry_revision);
        assert!(Arc::ptr_eq(&scene.data.textures, &textures));

        app.world_mut()
            .resource_mut::<Assets<Image>>()
            .get_mut_untracked(image.id())
            .unwrap()
            .data
            .as_mut()
            .unwrap()[0] = 0;
        app.world_mut()
            .resource_mut::<Messages<AssetEvent<Image>>>()
            .write(AssetEvent::Modified { id: image.id() });
        app.update();

        let scene = app.world().resource::<GiScene>();
        assert!(scene.revision > revision);
        assert!(scene.geometry_revision > geometry_revision);
        assert!(!Arc::ptr_eq(&scene.data.textures, &textures));

        let revision = scene.revision;
        let geometry_revision = scene.geometry_revision;
        app.world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .get_mut_untracked(material.id())
            .unwrap()
            .base_color = Color::BLACK;
        app.world_mut()
            .resource_mut::<Messages<AssetEvent<StandardMaterial>>>()
            .write(AssetEvent::Modified { id: material.id() });
        app.update();
        let scene = app.world().resource::<GiScene>();
        assert!(scene.revision > revision);
        assert!(scene.geometry_revision > geometry_revision);
        let instance = scene
            .data
            .instances
            .iter()
            .find(|instance| instance.entity == entity)
            .expect("edited material's instance stays in the scene");
        assert_eq!(instance.albedo, Vec3::ZERO);
    }

    #[test]
    fn suspended_scene_revalidates_expired_asset_events_and_motion() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<Assets<Image>>()
            .init_resource::<GiScene>()
            .add_message::<AssetEvent<Mesh>>()
            .add_message::<AssetEvent<Image>>()
            .add_message::<AssetEvent<StandardMaterial>>()
            .add_systems(Update, update_scene);
        let first = app
            .world_mut()
            .spawn((Camera::default(), RadianceCascades::default()))
            .id();
        let second = app
            .world_mut()
            .spawn((Camera::default(), RadianceCascades::default()))
            .id();
        let mesh = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(Cuboid::default());
        let image = app
            .world_mut()
            .resource_mut::<Assets<Image>>()
            .add(rgba8_image([255, 0, 0, 255]));
        let material = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial {
                base_color_texture: Some(image.clone()),
                ..default()
            });
        let entity = app
            .world_mut()
            .spawn((
                Mesh3d(mesh.clone()),
                MeshMaterial3d(material.clone()),
                RadianceCascadesGeometry,
                GlobalTransform::IDENTITY,
            ))
            .id();
        let removed = app
            .world_mut()
            .spawn((
                Mesh3d(mesh.clone()),
                MeshMaterial3d(material.clone()),
                RadianceCascadesGeometry,
                GlobalTransform::IDENTITY,
            ))
            .id();
        wait_for_blas_builds(&mut app, 1);
        app.world_mut()
            .entity_mut(first)
            .get_mut::<Camera>()
            .unwrap()
            .is_active = false;
        app.update();
        assert!(app.world().resource::<GiScene>().ready);

        app.world_mut()
            .entity_mut(second)
            .get_mut::<RadianceCascades>()
            .unwrap()
            .intensity = 0.0;
        app.update();
        let (revision, textures) = {
            let scene = app.world().resource::<GiScene>();
            assert!(!scene.ready);
            (scene.revision, scene.data.textures.clone())
        };
        *app.world_mut()
            .resource_mut::<Assets<Mesh>>()
            .get_mut_untracked(mesh.id())
            .unwrap() = Cuboid::new(2.0, 2.0, 2.0).into();
        app.world_mut()
            .resource_mut::<Assets<Image>>()
            .get_mut_untracked(image.id())
            .unwrap()
            .data
            .as_mut()
            .unwrap()[0] = 0;
        app.world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .get_mut_untracked(material.id())
            .unwrap()
            .base_color = Color::BLACK;
        app.world_mut()
            .resource_mut::<Messages<AssetEvent<Mesh>>>()
            .write(AssetEvent::Modified { id: mesh.id() });
        app.world_mut()
            .resource_mut::<Messages<AssetEvent<Image>>>()
            .write(AssetEvent::Modified { id: image.id() });
        app.world_mut()
            .resource_mut::<Messages<AssetEvent<StandardMaterial>>>()
            .write(AssetEvent::Modified { id: material.id() });
        app.world_mut()
            .entity_mut(removed)
            .remove::<RadianceCascadesGeometry>();
        *app.world_mut()
            .entity_mut(entity)
            .get_mut::<GlobalTransform>()
            .unwrap() = GlobalTransform::from(Transform::from_translation(Vec3::X * 5.0));
        for _ in 0..5 {
            app.update();
        }
        assert_eq!(app.world().resource::<GiScene>().revision, revision);

        app.world_mut()
            .entity_mut(first)
            .get_mut::<Camera>()
            .unwrap()
            .is_active = true;
        wait_for_blas_builds(&mut app, 2);
        let scene = app.world().resource::<GiScene>();
        assert!(scene.ready);
        assert!(!Arc::ptr_eq(&scene.data.textures, &textures));
        assert!(!scene.data.instance_indices.contains_key(&removed));
        let instance = scene
            .data
            .instances
            .iter()
            .find(|item| item.entity == entity)
            .unwrap();
        assert_eq!(instance.albedo, Vec3::ZERO);
        assert!(instance.min.x < 4.01 && instance.max.x > 5.99);
    }

    #[test]
    fn pending_blas_survives_a_period_without_gi_cameras() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<GiScene>()
            .add_systems(Update, update_scene);
        let mesh = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(Cuboid::default());
        let material = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial::default());
        let entity = app
            .world_mut()
            .spawn((
                Mesh3d(mesh),
                MeshMaterial3d(material),
                RadianceCascadesGeometry,
                GlobalTransform::IDENTITY,
            ))
            .id();
        app.update();
        assert_eq!(app.world().resource::<GiScene>().blas_builds, 0);
        let camera = app
            .world_mut()
            .spawn((Camera::default(), RadianceCascades::default()))
            .id();
        app.update();
        app.world_mut().despawn(camera);
        app.update();
        assert!(!app.world().resource::<GiScene>().ready);
        let camera = app
            .world_mut()
            .spawn((Camera::default(), RadianceCascades::default()))
            .id();
        wait_for_blas_builds(&mut app, 1);
        let scene = app.world().resource::<GiScene>();
        assert!(scene.ready);
        assert!(scene.data.instance_indices.contains_key(&entity));
        assert!(app.world().get::<Camera>(camera).is_some());
    }

    #[test]
    fn reintroduced_mesh_does_not_publish_a_stale_pending_blas() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<GiScene>()
            .add_message::<AssetEvent<Mesh>>()
            .add_systems(Update, update_scene);
        let camera = app
            .world_mut()
            .spawn((Camera::default(), RadianceCascades::default()))
            .id();
        let mesh = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(Cuboid::default());
        let material = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial::default());
        let entity = app
            .world_mut()
            .spawn((
                Mesh3d(mesh.clone()),
                MeshMaterial3d(material),
                RadianceCascadesGeometry,
                GlobalTransform::IDENTITY,
            ))
            .id();
        // The first update starts a BLAS task, without polling its output.
        app.update();
        app.world_mut().despawn(camera);
        app.update();
        app.world_mut()
            .entity_mut(entity)
            .remove::<RadianceCascadesGeometry>();
        *app.world_mut()
            .resource_mut::<Assets<Mesh>>()
            .get_mut_untracked(mesh.id())
            .unwrap() = Cuboid::new(4.0, 4.0, 4.0).into();
        app.world_mut()
            .resource_mut::<Messages<AssetEvent<Mesh>>>()
            .write(AssetEvent::Modified { id: mesh.id() });
        for _ in 0..5 {
            app.update();
        }
        app.world_mut()
            .spawn((Camera::default(), RadianceCascades::default()));
        app.update();
        app.world_mut()
            .entity_mut(entity)
            .insert(RadianceCascadesGeometry);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            app.update();
            let scene = app.world().resource::<GiScene>();
            if let Some(instance) = scene
                .data
                .instances
                .iter()
                .find(|item| item.entity == entity)
            {
                assert!(instance.min.x < -1.99 && instance.max.x > 1.99);
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "updated BLAS did not finish"
            );
            std::thread::yield_now();
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn blas_disk_cache_round_trips_and_rejects_corruption() {
        let directory = tempfile::tempdir().unwrap();
        let triangles = [Triangle {
            a: Vec4::ZERO,
            b: Vec4::X,
            c: Vec4::Y,
            ..default()
        }];
        let nodes = [BvhNode {
            min: Vec3::ZERO,
            first: 0,
            max: Vec3::ONE,
            count: 1,
        }];
        let signature = 42;
        save_blas_cache(directory.path(), signature, &nodes, &triangles).unwrap();
        let (loaded_nodes, loaded_triangles) =
            load_blas_cache(directory.path(), signature, triangles.len()).unwrap();
        assert_eq!(loaded_nodes[0].first, nodes[0].first);
        assert_eq!(loaded_nodes[0].count, nodes[0].count);
        assert_eq!(loaded_triangles[0].a, triangles[0].a);
        assert_eq!(loaded_triangles[0].indices, triangles[0].indices);

        let path = directory.path().join(format!("{signature:016x}.bvh"));
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[BLAS_HEADER_BYTES] ^= 1;
        std::fs::write(path, bytes).unwrap();
        assert!(load_blas_cache(directory.path(), signature, triangles.len()).is_none());
    }

    #[test]
    fn lightmaps_exclude_receivers_without_removing_occluders() {
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
            .add(Cuboid::default());
        let material = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial::default());
        let entity = app
            .world_mut()
            .spawn((
                Mesh3d(mesh),
                MeshMaterial3d(material),
                RadianceCascadesGeometry,
            ))
            .id();
        wait_for_blas_builds(&mut app, 1);
        let triangles = app.world().resource::<GiScene>().data.triangles.clone();
        app.world_mut()
            .entity_mut(entity)
            .insert(Lightmap::default());
        app.update();
        let scene = app.world().resource::<GiScene>();
        assert_eq!(scene.data.instances.len(), 1);
        assert_eq!(scene.data.instances[0].flags & 32, 32);
        assert!(Arc::ptr_eq(&triangles, &scene.data.triangles));
        app.world_mut().entity_mut(entity).remove::<Lightmap>();
        app.update();
        assert_eq!(
            app.world().resource::<GiScene>().data.instances[0].flags & 32,
            0
        );
        assert!(
            GiLight::punctual(
                &GlobalTransform::IDENTITY,
                Color::WHITE,
                1.0,
                10.0,
                f32::NAN
            )
            .is_none()
        );
        assert_eq!(
            GiLight::punctual(&GlobalTransform::IDENTITY, Color::WHITE, 1.0, 10.0, 0.25)
                .unwrap()
                .edge2
                .w,
            0.25
        );
    }

    #[test]
    fn layer_edits_and_removal_preserve_shared_geometry_and_filter_view_lights() {
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
            .add(Cuboid::default());
        let material = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial {
                emissive: LinearRgba::WHITE,
                ..default()
            });
        let geometry = app
            .world_mut()
            .spawn((
                Mesh3d(mesh),
                MeshMaterial3d(material),
                RadianceCascadesGeometry,
            ))
            .id();
        let point = app
            .world_mut()
            .spawn((PointLight::default(), InheritedVisibility::VISIBLE))
            .id();
        app.world_mut().spawn((
            DirectionalLight::default(),
            InheritedVisibility::VISIBLE,
            RenderLayers::none().with(129),
        ));
        app.world_mut().spawn((
            SpotLight::default(),
            InheritedVisibility::VISIBLE,
            RenderLayers::none().with(257),
        ));
        wait_for_blas_builds(&mut app, 1);
        let original = app.world().resource::<GiScene>().data.clone();
        assert_eq!(original.lights.len(), 3);
        assert_eq!(original.emissive_lights.len(), 12);
        let masks = [
            RenderLayers::default(),
            RenderLayers::none(),
            RenderLayers::from_layers(&[31, 32, 63, 64, 129]),
            RenderLayers::from_layers(&[0, 257]),
        ];
        for mask in &masks {
            app.world_mut().entity_mut(geometry).insert(mask.clone());
            app.world_mut().entity_mut(point).insert(mask.clone());
            app.update();
            let scene = app.world().resource::<GiScene>();
            assert_eq!(scene.blas_builds, 1);
            assert!(Arc::ptr_eq(
                &scene.data.geometry_words,
                &original.geometry_words
            ));
            assert_eq!(scene.data.instances[0].layers, *mask);
            for view_mask in &masks {
                let (lights, count, selector) = view_lights(&scene.data, view_mask);
                let expected: Vec<_> = scene
                    .data
                    .lights
                    .iter()
                    .zip(scene.data.light_layers.iter())
                    .filter(|(_, layers)| view_mask.intersects(layers))
                    .map(|(light, _)| *light)
                    .collect();
                assert_eq!(count as usize, expected.len());
                assert!(lights[..count as usize] == expected);
                assert!(
                    lights[count as usize..selector.x as usize] == **scene.data.emissive_lights
                );
                let instance_mask = scene.data.instance_words[6];
                let base = scene.data.acceleration_prefix.len() + scene.data.blas_words.len();
                let packed_intersection = (0..instance_mask.y.min(selector.y)).any(|i| {
                    let entity_bits = scene.data.instance_words
                        [instance_mask.x as usize - base + i as usize / 4][i as usize % 4];
                    let camera_bits =
                        lights[selector.x as usize + i as usize / 4].source[i as usize % 4];
                    entity_bits & camera_bits != 0
                });
                assert_eq!(
                    packed_intersection,
                    view_mask.intersects(&scene.data.instances[0].layers)
                );
            }
        }
        app.world_mut()
            .entity_mut(geometry)
            .remove::<RenderLayers>();
        app.world_mut().entity_mut(point).remove::<RenderLayers>();
        app.update();
        let scene = app.world().resource::<GiScene>();
        assert_eq!(scene.data.instances[0].layers, RenderLayers::default());
        assert_eq!(view_lights(&scene.data, &RenderLayers::default()).1, 1);
        assert_eq!(scene.blas_builds, 1);
        assert!(Arc::ptr_eq(
            &scene.data.geometry_words,
            &original.geometry_words
        ));
    }

    #[test]
    fn indexed_attributes_update_without_rebuilding_geometry() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<GiScene>()
            .add_systems(Update, update_scene);
        app.world_mut()
            .spawn((Camera::default(), RadianceCascades::default()));
        let mut indexed: Mesh = Rectangle::new(2.0, 2.0).into();
        indexed.insert_indices(Indices::U16(vec![0, 1, 2, 0, 2, 3]));
        indexed.insert_attribute(Mesh::ATTRIBUTE_UV_1, vec![[0.25, 0.75]; 4]);
        indexed.insert_attribute(Mesh::ATTRIBUTE_TANGENT, vec![[1.0, 0.0, 0.0, -1.0]; 4]);
        indexed.insert_attribute(Mesh::ATTRIBUTE_COLOR, vec![[0.2, 0.4, 0.8, 0.5]; 4]);
        let indexed = app.world_mut().resource_mut::<Assets<Mesh>>().add(indexed);
        let nonindexed = Mesh::new(
            bevy::render::render_resource::PrimitiveTopology::TriangleList,
            bevy::asset::RenderAssetUsages::default(),
        )
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vec![[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
        );
        let nonindexed = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(nonindexed);
        let material = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial::default());
        for mesh in [indexed.clone(), indexed.clone(), nonindexed] {
            app.world_mut().spawn((
                Mesh3d(mesh),
                MeshMaterial3d(material.clone()),
                RadianceCascadesGeometry,
                GlobalTransform::IDENTITY,
            ));
        }
        wait_for_blas_builds(&mut app, 2);
        let scene = app.world().resource::<GiScene>();
        assert_eq!(scene.data.vertices.len(), 7);
        assert_eq!(scene.data.triangles.len(), 3);
        assert_eq!(scene.blas_builds, 2);
        // Header and three triangles, padded so vertices start on a vertex stride.
        assert_eq!(scene.data.geometry_words.len(), 5 + 7 * 5);
        assert_eq!(scene.data.geometry_words[0].y, 5);
        for triangle in scene.data.triangles.iter() {
            for (index, position) in triangle
                .indices
                .to_array()
                .into_iter()
                .zip([triangle.a, triangle.b, triangle.c])
            {
                assert_eq!(
                    scene.data.vertices[index as usize].position,
                    position.truncate()
                );
            }
        }
        let triangles = scene.data.triangles.clone();
        let blas = scene.data.blas_nodes.clone();
        let words = scene.data.geometry_words.clone();
        let revision = scene.mesh_revision;
        app.world_mut()
            .resource_mut::<Assets<Mesh>>()
            .get_mut(&indexed)
            .unwrap()
            .insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 0.6, 0.8]; 4]);
        app.update();
        let scene = app.world().resource::<GiScene>();
        assert_eq!(scene.blas_builds, 2);
        assert_eq!(scene.mesh_revision, revision + 1);
        assert!(Arc::ptr_eq(&triangles, &scene.data.triangles));
        assert!(Arc::ptr_eq(&blas, &scene.data.blas_nodes));
        assert!(!Arc::ptr_eq(&words, &scene.data.geometry_words));
        assert!(
            scene
                .data
                .vertices
                .iter()
                .filter(|v| v.flags != 0)
                .all(|v| v.normal == Vec3::new(0.0, 0.6, 0.8) && v.flags == 31)
        );
        // A malformed shared asset is diagnosed and rejected consistently for
        // every instance, rather than indexing a short attribute or stale cache.
        app.world_mut()
            .resource_mut::<Assets<Mesh>>()
            .get_mut(&indexed)
            .unwrap()
            .insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 1.0, 0.0]; 3]);
        app.update();
        assert_eq!(app.world().resource::<GiScene>().data.instances.len(), 1);
        app.world_mut()
            .resource_mut::<Assets<Mesh>>()
            .get_mut(&indexed)
            .unwrap()
            .insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 0.6, 0.8]; 4]);
        wait_for_blas_builds(&mut app, 3);
        assert_eq!(app.world().resource::<GiScene>().data.instances.len(), 3);
        app.world_mut()
            .resource_mut::<Assets<Mesh>>()
            .get_mut(&indexed)
            .unwrap()
            .insert_attribute(Mesh::ATTRIBUTE_UV_1, vec![[f32::NAN, 0.0]; 4]);
        app.update();
        assert_eq!(app.world().resource::<GiScene>().data.instances.len(), 1);
    }

    #[test]
    fn degraded_tlas_rebuilds_without_losing_current_motion() {
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
            .add(Cuboid::default());
        let material = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial::default());
        let entities: Vec<_> = (0..64)
            .map(|i| {
                app.world_mut()
                    .spawn((
                        Mesh3d(mesh.clone()),
                        MeshMaterial3d(material.clone()),
                        RadianceCascadesGeometry,
                        GlobalTransform::from_translation(Vec3::new(i as f32 * 2.0, 0.0, 0.0)),
                    ))
                    .id()
            })
            .collect();
        wait_for_blas_builds(&mut app, 1);
        let geometry = app.world().resource::<GiScene>().data.triangles.clone();
        let built_cost = tree_cost(&app.world().resource::<GiScene>().data.nodes);
        for (i, &entity) in entities.iter().enumerate() {
            app.world_mut()
                .entity_mut(entity)
                .insert(GlobalTransform::from_translation(Vec3::new(
                    ((i * 33) % 64) as f32 * 2.0,
                    0.0,
                    0.0,
                )));
        }
        app.update();
        let degraded_cost = tree_cost(&app.world().resource::<GiScene>().data.nodes);
        assert!(degraded_cost > built_cost * 1.5);
        assert_eq!(app.world().resource::<GiScene>().tlas_builds, 1);
        // Move while the worker holds its older snapshot. The committed topology
        // must be refitted to this new transform before it reaches the GPU.
        app.world_mut()
            .entity_mut(entities[0])
            .insert(GlobalTransform::from_translation(Vec3::new(
                129.0, 2.0, 0.0,
            )));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
        while app.world().resource::<GiScene>().tlas_builds == 1
            && std::time::Instant::now() < deadline
        {
            app.update();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let scene = app.world().resource::<GiScene>();
        assert_eq!(scene.tlas_builds, 2);
        assert_eq!(scene.blas_builds, 1);
        assert!(Arc::ptr_eq(&geometry, &scene.data.triangles));
        assert!(tree_cost(&scene.data.nodes) < degraded_cost);
        let moved = scene
            .data
            .instances
            .iter()
            .find(|instance| instance.entity == entities[0])
            .unwrap();
        assert_eq!(
            moved.world_from_local.translation,
            Vec3::new(129.0, 2.0, 0.0).into()
        );
        for node in scene.data.nodes.iter() {
            if node.count > 0 {
                for instance in
                    &scene.data.instances[node.first as usize..(node.first + node.count) as usize]
                {
                    assert!(
                        instance.min.cmpge(node.min).all() && instance.max.cmple(node.max).all()
                    );
                }
            }
        }
        println!(
            "TLAS SAH cost: built={built_cost:.4}, degraded={degraded_cost:.4}, rebuilt={:.4}",
            tree_cost(&scene.data.nodes)
        );
    }

    #[test]
    fn sah_build_is_bounded_at_world_workload_size() {
        let mut triangles: Vec<_> = (0..250_000)
            .map(|i| {
                let x = if i >= 249_900 {
                    1000.0 + (i - 249_900) as f32 * 0.01
                } else {
                    (i % 500) as f32 * 0.01
                };
                let y = (i / 500) as f32 * 0.01;
                Triangle {
                    a: Vec4::new(x, y, 0.0, 0.0),
                    b: Vec4::new(x + 0.003, y, 0.0, 0.0),
                    c: Vec4::new(x, y + 0.003, 0.0, 0.0),
                    ..default()
                }
            })
            .collect();
        let started = std::time::Instant::now();
        let nodes = build_bvh(&mut triangles);
        let elapsed = started.elapsed();
        let mut stack = vec![(0usize, 0)];
        let mut depth = 0;
        let mut count = 0;
        while let Some((index, level)) = stack.pop() {
            depth = depth.max(level);
            let node = nodes[index];
            if node.count > 0 {
                count += node.count as usize;
            } else {
                stack.extend([
                    (node.first as usize, level + 1),
                    (node.first as usize + 1, level + 1),
                ]);
            }
        }
        assert_eq!(count, triangles.len());
        assert!(depth <= 48);
        assert!(
            tree_cost(&nodes) < 3.0,
            "SAH should isolate the tiny distant cluster"
        );
        println!(
            "250k-triangle SAH: {:.3} ms, {} nodes, depth {depth}, normalized cost {:.4}",
            elapsed.as_secs_f64() * 1000.0,
            nodes.len(),
            tree_cost(&nodes)
        );
    }

    #[test]
    fn shared_meshes_survive_motion_material_changes_and_asset_edits() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<GiScene>()
            .add_message::<AssetEvent<Mesh>>()
            .add_systems(Update, update_scene);
        app.world_mut()
            .spawn((Camera::default(), RadianceCascades::default()));
        let mesh = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(Cuboid::default());
        let unrelated_mesh = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(Cuboid::default());
        let material = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial::default());
        let entities: Vec<_> = (0..20)
            .map(|i| {
                app.world_mut()
                    .spawn((
                        Mesh3d(mesh.clone()),
                        MeshMaterial3d(material.clone()),
                        RadianceCascadesGeometry,
                        GlobalTransform::from(
                            Transform::from_xyz(i as f32 * 2.0, 1.0, 0.0)
                                .with_rotation(Quat::from_rotation_y(i as f32 * 0.1))
                                .with_scale(Vec3::new(
                                    if i % 2 == 0 { -2.0 } else { 2.0 },
                                    0.5,
                                    3.0,
                                )),
                        ),
                    ))
                    .id()
            })
            .collect();
        wait_for_blas_builds(&mut app, 1);
        let scene = app.world().resource::<GiScene>();
        assert_eq!(scene.blas_builds, 1);
        assert_eq!(scene.data.triangles.len(), 12);
        assert_eq!(scene.data.instances.len(), 20);
        for (index, instance) in scene.data.instances.iter().enumerate() {
            assert_eq!(
                scene.data.instance_indices.get(&instance.entity),
                Some(&(index as u32))
            );
        }
        let triangles = scene.data.triangles.clone();
        let blas = scene.data.blas_nodes.clone();
        let mesh_revision = scene.mesh_revision;
        let topology: Vec<_> = scene
            .data
            .nodes
            .iter()
            .map(|node| (node.first, node.count))
            .collect();
        app.world_mut()
            .entity_mut(entities[0])
            .insert(GlobalTransform::from_translation(Vec3::new(50.0, 4.0, 1.0)));
        app.update();
        let scene = app.world().resource::<GiScene>();
        assert_eq!(scene.blas_builds, 1);
        assert_eq!(scene.mesh_revision, mesh_revision);
        assert!(Arc::ptr_eq(&scene.data.triangles, &triangles));
        assert!(Arc::ptr_eq(&scene.data.blas_nodes, &blas));
        assert_eq!(
            topology,
            scene
                .data
                .nodes
                .iter()
                .map(|node| (node.first, node.count))
                .collect::<Vec<_>>()
        );
        for instance in scene.data.instances.iter() {
            for local in &scene.data.triangles
                [instance.first as usize..(instance.first + instance.count) as usize]
            {
                let world = instance.triangle(*local);
                for point in [world.a, world.b, world.c] {
                    assert!(
                        point.truncate().cmpge(instance.min).all()
                            && point.truncate().cmple(instance.max).all()
                    );
                }
            }
        }
        app.world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .get_mut(&material)
            .unwrap()
            .base_color = Color::BLACK;
        app.update();
        assert_eq!(app.world().resource::<GiScene>().blas_builds, 1);
        assert!(
            app.world()
                .resource::<GiScene>()
                .data
                .instances
                .iter()
                .all(|instance| instance.albedo == Vec3::ZERO)
        );
        app.world_mut()
            .resource_mut::<Assets<Mesh>>()
            .insert(mesh.id(), Cuboid::new(4.0, 1.0, 1.0).into())
            .unwrap();
        app.world_mut()
            .resource_mut::<Messages<AssetEvent<Mesh>>>()
            .write(AssetEvent::Modified {
                id: unrelated_mesh.id(),
            });
        app.update();
        assert_eq!(app.world().resource::<GiScene>().blas_builds, 1);
        assert_eq!(
            app.world().resource::<GiScene>().mesh_revision,
            mesh_revision
        );
        app.world_mut()
            .resource_mut::<Messages<AssetEvent<Mesh>>>()
            .write(AssetEvent::Modified { id: mesh.id() });
        wait_for_blas_builds(&mut app, 2);
        assert_eq!(app.world().resource::<GiScene>().blas_builds, 2);
        assert!(app.world().resource::<GiScene>().mesh_revision > mesh_revision);
        assert!(!Arc::ptr_eq(
            &app.world().resource::<GiScene>().data.triangles,
            &triangles
        ));
        for transform in [
            Transform::from_scale(Vec3::ZERO),
            Transform::from_translation(Vec3::splat(f32::MAX)),
        ] {
            app.world_mut()
                .entity_mut(entities[0])
                .insert(GlobalTransform::from(transform));
            app.update();
            let scene = app.world().resource::<GiScene>();
            assert_eq!(scene.blas_builds, 2);
            assert_eq!(scene.data.instances.len(), 19);
            assert!(
                scene
                    .data
                    .nodes
                    .iter()
                    .all(|node| node.min.is_finite() && node.max.is_finite())
            );
        }
        app.world_mut()
            .entity_mut(entities[0])
            .insert(GlobalTransform::IDENTITY);
        // The mesh event is consumed while a material texture is unavailable.
        app.world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .get_mut(&material)
            .unwrap()
            .base_color_texture = Some(Handle::default());
        app.world_mut()
            .resource_mut::<Assets<Mesh>>()
            .insert(mesh.id(), Cuboid::new(6.0, 1.0, 1.0).into())
            .unwrap();
        app.world_mut()
            .resource_mut::<Messages<AssetEvent<Mesh>>>()
            .write(AssetEvent::Modified { id: mesh.id() });
        app.update();
        assert!(!app.world().resource::<GiScene>().ready);
        assert_eq!(app.world().resource::<GiScene>().blas_builds, 2);
        app.world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .get_mut(&material)
            .unwrap()
            .base_color_texture = None;
        wait_for_blas_builds(&mut app, 3);
        let scene = app.world().resource::<GiScene>();
        assert!(scene.ready);
        assert_eq!(scene.blas_builds, 3);
        assert_eq!(scene.data.instances.len(), 20);
        assert!(
            scene
                .data
                .triangles
                .iter()
                .any(|triangle| triangle.a.x == 3.0)
        );

        // A rejected edit must also reject every sibling using the cached mesh.
        {
            let mut assets = app.world_mut().resource_mut::<Assets<Mesh>>();
            let mut source = assets.get_mut(&mesh).unwrap();
            let count = source.count_vertices();
            source.insert_attribute(
                Mesh::ATTRIBUTE_JOINT_WEIGHT,
                vec![[1.0f32, 0.0, 0.0, 0.0]; count],
            );
        }
        app.world_mut()
            .resource_mut::<Messages<AssetEvent<Mesh>>>()
            .write(AssetEvent::Modified { id: mesh.id() });
        app.update();
        assert!(app.world().resource::<GiScene>().data.instances.is_empty());
    }

    #[test]
    fn emissive_distribution_preserves_area_and_power() {
        let mut triangles = [
            Triangle {
                a: Vec4::ZERO,
                b: Vec3::X.extend(0.0),
                c: Vec3::Y.extend(0.0),
                emission: (Vec3::X * 2.0).extend(0.0),
                albedo: Vec4::new(0.0, 0.0, 0.0, 1.0),
                ..default()
            },
            Triangle {
                a: Vec4::ZERO,
                b: (Vec3::X * 2.0).extend(0.0),
                c: Vec3::Y.extend(0.0),
                emission: Vec3::Y.extend(0.0),
                albedo: Vec4::new(0.0, 0.0, 0.0, 1.0),
                ..default()
            },
            Triangle::default(),
        ];
        let lights = build_emissive_lights(&mut triangles).unwrap();
        assert_eq!(lights.len(), 2);
        assert_eq!(lights[0].color.w, 0.5);
        assert_eq!(lights[1].color.w, 1.0);
        assert_eq!(lights[0].direction.truncate(), Vec3::Z);
        assert!((lights[0].edge1.w - 0.2126 / (0.2126 + 0.7152)).abs() < 1e-6);
        assert!((lights.iter().map(|light| light.edge1.w).sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(lights[0].direction.w, lights[0].edge1.w);
        assert_eq!(lights[1].direction.w, 1.0);
        assert!(lights.iter().all(|light| light.position.w == -1.0));
        assert!(triangles.iter().all(|triangle| triangle.emission.w == 0.0));
    }

    #[test]
    fn morph_pose_updates_emissive_geometry_and_sampling_pdf() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<GiScene>()
            .add_systems(Update, update_scene);
        app.world_mut()
            .spawn((Camera::default(), RadianceCascades::default()));

        let mut morphs = vec![bevy::mesh::morph::MorphAttributes::default(); 6];
        morphs[1].position = Vec3::X * 2.0;
        let mesh = Mesh::new(
            bevy::render::render_resource::PrimitiveTopology::TriangleList,
            bevy::asset::RenderAssetUsages::default(),
        )
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
            ],
        )
        .with_morph_targets(morphs);
        let mesh = app.world_mut().resource_mut::<Assets<Mesh>>().add(mesh);
        let material = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial {
                base_color: Color::BLACK,
                emissive: LinearRgba::WHITE,
                ..default()
            });
        let animated = app
            .world_mut()
            .spawn((
                Mesh3d(mesh.clone()),
                MeshMaterial3d(material.clone()),
                MeshMorphWeights::Value { weights: vec![0.0] },
                RadianceCascadesGeometry,
                GlobalTransform::IDENTITY,
            ))
            .id();
        app.world_mut().spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material),
            RadianceCascadesGeometry,
            GlobalTransform::IDENTITY,
        ));

        wait_for_blas_builds(&mut app, 1);
        let emissive_lights_for = |scene: &GiScene, entity: Entity| {
            let instance = scene.data.instance_indices[&entity];
            let mut lights: Vec<_> = scene
                .data
                .emissive_lights
                .iter()
                .filter(|light| light.source.x == instance)
                .map(|light| {
                    (
                        light.source.y,
                        light.color.w,
                        light.edge1.w,
                        light.direction.w,
                    )
                })
                .collect();
            lights.sort_by_key(|light| light.0);
            lights
        };
        let assert_emissive_cdf = |lights: &[GiLight]| {
            let mut previous = 0.0;
            for light in lights {
                assert!(light.direction.w > previous);
                assert!((light.direction.w - previous - light.edge1.w).abs() < 1e-6);
                previous = light.direction.w;
            }
            assert!((previous - 1.0).abs() < 1e-6);
        };
        let scene = app.world().resource::<GiScene>();
        let revision = scene.revision;
        assert_eq!(scene.data.emissive_lights.len(), 4);
        assert_emissive_cdf(&scene.data.emissive_lights);
        assert!(scene.data.emissive_lights.iter().all(|light| {
            (light.color.w - 0.5).abs() < 1e-6 && (light.edge1.w - 0.25).abs() < 1e-6
        }));
        let initial_animated_lights = emissive_lights_for(scene, animated);
        assert_eq!(initial_animated_lights.len(), 2);
        assert!(
            initial_animated_lights
                .iter()
                .all(|light| { (light.1 - 0.5).abs() < 1e-6 && (light.2 - 0.25).abs() < 1e-6 })
        );

        let mut weights = app
            .world_mut()
            .get_mut::<MeshMorphWeights>(animated)
            .unwrap();
        *weights = MeshMorphWeights::Value { weights: vec![1.0] };
        app.update();

        let scene = app.world().resource::<GiScene>();
        let lights = &scene.data.emissive_lights;
        assert!(scene.revision > revision);
        assert_eq!(lights.len(), 4);
        assert_emissive_cdf(lights);
        let large = lights
            .iter()
            .find(|light| (light.color.w - 1.5).abs() < 1e-6)
            .expect("morphed emitter area");
        assert!((large.edge1.w - 0.5).abs() < 1e-6);
        assert!(
            lights
                .iter()
                .filter(|light| (light.color.w - 0.5).abs() < 1e-6)
                .all(|light| (light.edge1.w - 1.0 / 6.0).abs() < 1e-6)
        );
        let animated_lights = emissive_lights_for(scene, animated);
        assert_eq!(animated_lights.len(), 2);
        let large_animated = animated_lights
            .iter()
            .find(|light| (light.1 - 1.5).abs() < 1e-6)
            .expect("morphed emitter belongs to animated instance");
        assert!((large_animated.2 - 0.5).abs() < 1e-6);
        let small_animated = animated_lights
            .iter()
            .find(|light| (light.1 - 0.5).abs() < 1e-6)
            .expect("unchanged emitter belongs to animated instance");
        assert!((small_animated.2 - 1.0 / 6.0).abs() < 1e-6);
        let pdf_sum = lights.iter().map(|light| light.edge1.w).sum::<f32>();
        assert!((pdf_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn scene_tracks_edits_and_removal_but_reuses_static_geometry() {
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
            .add(Cuboid::default());
        let material = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial::default());
        let entity = app
            .world_mut()
            .spawn((
                Mesh3d(mesh.clone()),
                MeshMaterial3d(material.clone()),
                RadianceCascadesGeometry,
                InheritedVisibility::VISIBLE,
            ))
            .id();
        wait_for_blas_builds(&mut app, 1);
        let revision = app.world().resource::<GiScene>().revision;
        assert_eq!(app.world().resource::<GiScene>().data.triangles.len(), 12);
        assert!(
            app.world()
                .resource::<GiScene>()
                .pending_entities
                .contains(&entity)
        );
        app.update();
        assert_eq!(app.world().resource::<GiScene>().revision, revision);
        let triangles = app.world().resource::<GiScene>().data.triangles.clone();
        let nodes = app.world().resource::<GiScene>().data.nodes.clone();
        let geometry_revision = app.world().resource::<GiScene>().geometry_revision;
        let light = app
            .world_mut()
            .spawn((PointLight::default(), InheritedVisibility::VISIBLE))
            .id();
        app.update();
        app.world_mut()
            .get_mut::<PointLight>(light)
            .unwrap()
            .intensity *= 2.0;
        app.update();
        let scene = app.world().resource::<GiScene>();
        assert!(scene.revision > revision);
        assert_eq!(scene.geometry_revision, geometry_revision);
        assert!(Arc::ptr_eq(&scene.data.triangles, &triangles));
        assert!(Arc::ptr_eq(&scene.data.nodes, &nodes));
        app.world_mut()
            .entity_mut(entity)
            .insert(GlobalTransform::from_translation(Vec3::X * 4.0));
        app.update();
        assert!(app.world().resource::<GiScene>().data.nodes[0].min.x > 3.0);
        app.world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .get_mut(&material)
            .unwrap()
            .emissive = LinearRgba::RED;
        app.update();
        assert_eq!(
            app.world().resource::<GiScene>().data.instances[0]
                .emission
                .x,
            1.0
        );
        let blue = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial {
                emissive: LinearRgba::BLUE,
                ..default()
            });
        let other = app
            .world_mut()
            .spawn((
                Mesh3d(mesh.clone()),
                MeshMaterial3d(blue),
                RadianceCascadesGeometry,
            ))
            .id();
        app.update();
        let emitter_colors = |app: &App| {
            app.world()
                .resource::<GiScene>()
                .data
                .emissive_lights
                .iter()
                .map(|light| light.color)
                .collect::<Vec<_>>()
        };
        let colors = emitter_colors(&app);
        assert_eq!(colors.len(), 24);
        app.world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .get_mut(&material)
            .unwrap()
            .emissive = LinearRgba::BLACK;
        app.update();
        assert_eq!(
            app.world().resource::<GiScene>().data.emissive_lights.len(),
            12,
            "only positive emissive power participates in hit-surface NEE"
        );
        app.world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .get_mut(&material)
            .unwrap()
            .emissive = LinearRgba::RED;
        app.update();
        assert_eq!(
            app.world().resource::<GiScene>().data.emissive_lights.len(),
            24,
            "all emitting meshes participate without a sampling classification"
        );
        // Crossing another emitter changes BVH partitions, but must not shuffle
        // the sampling CDF and turn previously stable samples into new lights.
        app.world_mut()
            .entity_mut(entity)
            .insert(GlobalTransform::from_translation(Vec3::NEG_X * 4.0));
        app.update();
        assert_eq!(emitter_colors(&app), colors);
        app.world_mut().entity_mut(other).despawn();
        app.update();
        app.world_mut()
            .entity_mut(entity)
            .remove::<RadianceCascadesGeometry>();
        app.update();
        assert!(app.world().resource::<GiScene>().data.triangles.is_empty());
        assert!(
            app.world()
                .resource::<GiScene>()
                .pending_entities
                .is_empty()
        );
        app.world_mut()
            .entity_mut(entity)
            .insert(RadianceCascadesGeometry);
        wait_for_blas_builds(&mut app, 2);
        assert_eq!(app.world().resource::<GiScene>().data.triangles.len(), 12);
        assert!(
            app.world()
                .resource::<GiScene>()
                .pending_entities
                .contains(&entity)
        );
        app.world_mut().entity_mut(entity).remove::<Mesh3d>();
        app.update();
        assert!(app.world().resource::<GiScene>().data.triangles.is_empty());
        app.world_mut().entity_mut(entity).insert(Mesh3d(mesh));
        wait_for_blas_builds(&mut app, 3);
        assert_eq!(app.world().resource::<GiScene>().data.triangles.len(), 12);
        app.world_mut()
            .entity_mut(entity)
            .remove::<MeshMaterial3d<StandardMaterial>>();
        app.update();
        assert!(app.world().resource::<GiScene>().data.triangles.is_empty());
        app.world_mut()
            .entity_mut(entity)
            .insert(MeshMaterial3d(material));
        wait_for_blas_builds(&mut app, 4);
        assert_eq!(app.world().resource::<GiScene>().data.triangles.len(), 12);
        app.world_mut()
            .entity_mut(entity)
            .remove::<GlobalTransform>();
        app.update();
        assert!(app.world().resource::<GiScene>().data.triangles.is_empty());
        app.world_mut()
            .entity_mut(entity)
            .insert(GlobalTransform::default());
        wait_for_blas_builds(&mut app, 5);
        assert_eq!(app.world().resource::<GiScene>().data.triangles.len(), 12);
        app.world_mut()
            .entity_mut(entity)
            .insert(InheritedVisibility::HIDDEN);
        app.update();
        assert!(app.world().resource::<GiScene>().data.triangles.is_empty());
        assert!(
            app.world()
                .resource::<GiScene>()
                .pending_entities
                .is_empty()
        );
        app.world_mut()
            .entity_mut(entity)
            .remove::<InheritedVisibility>();
        wait_for_blas_builds(&mut app, 6);
        assert_eq!(app.world().resource::<GiScene>().data.triangles.len(), 12);
        assert!(
            app.world()
                .resource::<GiScene>()
                .pending_entities
                .contains(&entity)
        );
        app.world_mut().entity_mut(entity).despawn();
        app.update();
        assert!(app.world().resource::<GiScene>().data.triangles.is_empty());
        assert!(
            app.world()
                .resource::<GiScene>()
                .pending_entities
                .is_empty()
        );
    }

    #[test]
    fn lights_track_motion_visibility_and_removal_without_rebuilding_geometry() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<GiScene>()
            .add_systems(Update, update_scene);
        app.world_mut()
            .spawn((Camera::default(), RadianceCascades::default()));
        let point = app
            .world_mut()
            .spawn((
                PointLight {
                    intensity: 4.0 * std::f32::consts::PI,
                    ..default()
                },
                InheritedVisibility::VISIBLE,
            ))
            .id();
        let spot = app
            .world_mut()
            .spawn((
                SpotLight {
                    intensity: 4.0 * std::f32::consts::PI,
                    inner_angle: 0.3,
                    outer_angle: 0.6,
                    ..default()
                },
                InheritedVisibility::VISIBLE,
            ))
            .id();
        let directional = app
            .world_mut()
            .spawn((
                DirectionalLight {
                    illuminance: 2.0,
                    ..default()
                },
                InheritedVisibility::VISIBLE,
            ))
            .id();
        app.update();
        let scene = app.world().resource::<GiScene>();
        let revision = scene.revision;
        let light_revision = scene.light_revision;
        let geometry_revision = scene.geometry_revision;
        let nodes = scene.data.nodes.clone();
        assert_eq!(scene.data.lights.len(), 3);
        assert_eq!(scene.data.lights[0].color.x, 2.0);
        assert_eq!(scene.data.lights[1].color.x, 1.0);
        let light = scene.data.lights[2];
        assert_eq!(light.color.x, 1.0);
        assert_eq!(light.direction.truncate(), Vec3::NEG_Z);
        let cone = |cosine: f32| {
            (cosine * light.direction.w + light.color.w)
                .clamp(0.0, 1.0)
                .powi(2)
        };
        assert_eq!(cone(1.0), 1.0);
        assert!(cone(0.45_f32.cos()) > 0.0 && cone(0.45_f32.cos()) < 1.0);
        assert!(cone(0.6_f32.cos()) < 1e-8);
        assert_eq!(cone(-1.0), 0.0);
        app.update();
        assert_eq!(app.world().resource::<GiScene>().revision, revision);
        assert_eq!(
            app.world().resource::<GiScene>().light_revision,
            light_revision
        );
        for entity in [point, spot, directional] {
            let before = app.world().resource::<GiScene>().revision;
            let before_lights = app.world().resource::<GiScene>().light_revision;
            app.world_mut()
                .entity_mut(entity)
                .insert(GlobalTransform::from(
                    Transform::from_xyz(2.0, 3.0, 4.0).with_rotation(Quat::from_rotation_y(0.8)),
                ));
            app.update();
            assert!(app.world().resource::<GiScene>().revision > before);
            assert!(app.world().resource::<GiScene>().light_revision > before_lights);
            assert_eq!(
                app.world().resource::<GiScene>().geometry_revision,
                geometry_revision
            );
            assert!(Arc::ptr_eq(
                &app.world().resource::<GiScene>().data.nodes,
                &nodes
            ));
            app.world_mut()
                .entity_mut(entity)
                .insert(InheritedVisibility::HIDDEN);
            let before_hidden = app.world().resource::<GiScene>().light_revision;
            app.update();
            assert_eq!(app.world().resource::<GiScene>().data.lights.len(), 2);
            assert!(app.world().resource::<GiScene>().light_revision > before_hidden);
            let before_visible = app.world().resource::<GiScene>().light_revision;
            app.world_mut()
                .entity_mut(entity)
                .insert(InheritedVisibility::VISIBLE);
            app.update();
            assert_eq!(app.world().resource::<GiScene>().data.lights.len(), 3);
            assert!(app.world().resource::<GiScene>().light_revision > before_visible);
        }
        let before_point_edit = app.world().resource::<GiScene>().light_revision;
        {
            let mut light = app.world_mut().get_mut::<PointLight>(point).unwrap();
            light.intensity *= 2.0;
            light.color = Color::linear_rgb(0.2, 0.5, 1.0);
            light.range *= 0.5;
        }
        app.update();
        assert!(app.world().resource::<GiScene>().light_revision > before_point_edit);
        let before_spot_edit = app.world().resource::<GiScene>().light_revision;
        {
            let mut light = app.world_mut().get_mut::<SpotLight>(spot).unwrap();
            light.intensity *= 2.0;
            light.inner_angle = 0.15;
            light.outer_angle = 0.4;
        }
        app.update();
        assert!(app.world().resource::<GiScene>().light_revision > before_spot_edit);
        let before_sun_edit = app.world().resource::<GiScene>().light_revision;
        {
            let mut light = app
                .world_mut()
                .get_mut::<DirectionalLight>(directional)
                .unwrap();
            light.illuminance *= 2.0;
            light.color = Color::linear_rgb(1.0, 0.85, 0.65);
        }
        app.update();
        assert!(app.world().resource::<GiScene>().light_revision > before_sun_edit);
        app.world_mut()
            .get_mut::<SpotLight>(spot)
            .unwrap()
            .outer_angle = f32::NAN;
        app.world_mut().entity_mut(point).despawn();
        app.update();
        assert_eq!(app.world().resource::<GiScene>().data.lights.len(), 1);
        assert_eq!(
            app.world().resource::<GiScene>().geometry_revision,
            geometry_revision
        );
        assert!(Arc::ptr_eq(
            &app.world().resource::<GiScene>().data.nodes,
            &nodes
        ));
        app.world_mut().spawn(PointLight {
            color: Color::linear_rgb(f32::INFINITY, 0.0, 0.0),
            ..default()
        });
        app.world_mut()
            .get_mut::<DirectionalLight>(directional)
            .unwrap()
            .illuminance = f32::INFINITY;
        app.update();
        assert!(app.world().resource::<GiScene>().data.lights.is_empty());
        for entity in [spot, directional] {
            app.world_mut().entity_mut(entity).despawn();
        }
        app.update();
        assert!(app.world().resource::<GiScene>().data.lights.is_empty());
        assert_eq!(
            app.world().resource::<GiScene>().geometry_revision,
            geometry_revision
        );
        assert!(Arc::ptr_eq(
            &app.world().resource::<GiScene>().data.nodes,
            &nodes
        ));
    }

    #[test]
    fn bvh_covers_every_triangle_once_and_contains_children() {
        assert!(build_bvh(&mut []).is_empty());
        let mut triangles: Vec<_> = (0..100)
            .map(|i| Triangle {
                a: Vec4::new(i as f32, 0.0, 0.0, 0.0),
                b: Vec4::new(i as f32 + 0.5, 0.0, 0.0, 0.0),
                c: Vec4::new(i as f32, 1.0, 0.0, 0.0),
                ..default()
            })
            .collect();
        let nodes = build_bvh(&mut triangles);
        let mut visits = vec![0; triangles.len()];
        for node in &nodes {
            if node.count > 0 {
                for i in node.first..node.first + node.count {
                    visits[i as usize] += 1;
                    let (min, max) = triangles[i as usize].bounds();
                    assert!(min.cmpge(node.min).all() && max.cmple(node.max).all());
                }
            } else {
                for child in &nodes[node.first as usize..node.first as usize + 2] {
                    assert!(child.min.cmpge(node.min).all() && child.max.cmple(node.max).all());
                }
            }
        }
        assert!(visits.iter().all(|&n| n == 1));
    }
}

#[cfg(test)]
#[path = "traversal_tests.rs"]
pub(crate) mod traversal_tests;

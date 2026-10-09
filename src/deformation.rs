//! CPU snapshots of immutable morph/skin streams and current animation controls.
use crate::RadianceCascadesGeometry;
use bevy::{
    ecs::system::SystemParam,
    mesh::{
        VertexAttributeValues,
        morph::{MeshMorphWeights, MorphWeights},
        skinning::{SkinnedMesh, SkinnedMeshInverseBindposes},
    },
    prelude::*,
};
use std::{collections::HashMap, sync::Arc};

#[derive(SystemParam)]
#[allow(clippy::type_complexity)]
pub(crate) struct DeformationInputs<'w, 's> {
    animated: Query<
        'w,
        's,
        (
            Entity,
            &'static Mesh3d,
            &'static GlobalTransform,
            Option<&'static SkinnedMesh>,
            Option<&'static MeshMorphWeights>,
        ),
        With<RadianceCascadesGeometry>,
    >,
    joints: Query<'w, 's, &'static GlobalTransform>,
    weights: Query<'w, 's, &'static MorphWeights>,
    bindposes: Option<Res<'w, Assets<SkinnedMeshInverseBindposes>>>,
}

#[derive(Clone, Default, PartialEq)]
pub(crate) struct DeformationPose {
    pub matrices: Vec<Mat4>,
    pub weights: Vec<f32>,
}
impl DeformationInputs<'_, '_> {
    pub fn collect(&self) -> HashMap<Entity, Result<DeformationPose, &'static str>> {
        self.animated
            .iter()
            .filter_map(|(entity, _, world, skin, morph)| {
                if skin.is_none() && morph.is_none() {
                    return None;
                }
                let result = (|| {
                    let mut pose = DeformationPose::default();
                    if let Some(skin) = skin {
                        let inverse = self
                            .bindposes
                            .as_ref()
                            .and_then(|assets| assets.get(&skin.inverse_bindposes))
                            .ok_or("skin inverse bindposes are not resident")?;
                        if inverse.len() < skin.joints.len() {
                            return Err("skin bindpose count is smaller than its joint count");
                        }
                        let local_from_world = world.to_matrix().inverse();
                        if !local_from_world.is_finite() {
                            return Err("skinned instance transform is not invertible");
                        }
                        for (entity, inverse) in skin.joints.iter().zip(inverse.iter()) {
                            let joint = self
                                .joints
                                .get(*entity)
                                .map_err(|_| "skin joint is missing a GlobalTransform")?;
                            let matrix = local_from_world * joint.to_matrix() * *inverse;
                            if !matrix.is_finite() {
                                return Err("skin matrices must be finite");
                            }
                            pose.matrices.push(matrix);
                        }
                        if pose.matrices.is_empty() {
                            return Err("skin must contain at least one joint");
                        }
                    }
                    if let Some(morph) = morph {
                        let weights = match morph {
                            MeshMorphWeights::Value { weights } => weights.as_slice(),
                            MeshMorphWeights::Reference(entity) => self
                                .weights
                                .get(*entity)
                                .map_err(|_| "referenced morph weights are not resident")?
                                .weights(),
                        };
                        if weights.iter().any(|w| !w.is_finite()) {
                            return Err("morph weights must be finite");
                        }
                        pose.weights.extend_from_slice(weights);
                    }
                    Ok(pose)
                })();
                Some((entity, result))
            })
            .collect()
    }
}

#[derive(Clone, Default)]
pub(crate) struct MeshDeformation {
    pub influences: Vec<UVec4>,
    pub morphs: Vec<UVec4>,
    morph_bounds: Vec<(Vec3, Vec3)>,
    skin_weight_bounds: (f32, f32),
    pub vertex_count: u32,
    pub target_count: u32,
    pub max_joint: u32,
}
impl MeshDeformation {
    pub fn hash(&self, hash: &mut impl std::hash::Hasher) {
        use std::hash::Hash;
        self.vertex_count.hash(hash);
        self.target_count.hash(hash);
        self.influences.hash(hash);
        self.morphs.hash(hash);
    }

    /// CPU evaluation is only for the emissive triangle distribution; ordinary
    /// geometry deformation and BLAS refits remain GPU work.
    pub fn deform_point(&self, vertex: usize, mut point: Vec3, pose: &DeformationPose) -> Vec3 {
        for (target, weight) in pose.weights.iter().enumerate() {
            let word = self.morphs[(target * self.vertex_count as usize + vertex) * 3];
            point += Vec3::new(
                f32::from_bits(word.x),
                f32::from_bits(word.y),
                f32::from_bits(word.z),
            ) * *weight;
        }
        if pose.matrices.is_empty() {
            return point;
        }
        let indices = self.influences[vertex * 2].to_array();
        let weights = self.influences[vertex * 2 + 1]
            .to_array()
            .map(f32::from_bits);
        indices
            .into_iter()
            .zip(weights)
            .map(|(joint, weight)| pose.matrices[joint as usize].transform_point3(point) * weight)
            .sum()
    }

    /// A morph-expanded box transformed by each joint conservatively contains
    /// every convex blend when vertex skin weights sum to one.
    pub fn conservative_bounds(
        &self,
        minimum: Vec3,
        maximum: Vec3,
        pose: &DeformationPose,
    ) -> (Vec3, Vec3) {
        let mut lower = minimum;
        let mut upper = maximum;
        for (target, weight) in pose.weights.iter().enumerate() {
            let (minimum, maximum) = self.morph_bounds[target];
            let a = minimum * *weight;
            let b = maximum * *weight;
            lower += a.min(b);
            upper += a.max(b);
        }
        if pose.matrices.is_empty() {
            return (lower, upper);
        }
        let mut a = Vec3::splat(f32::INFINITY);
        let mut b = Vec3::splat(f32::NEG_INFINITY);
        // Include weight-sum variation because Bevy deliberately does not
        // normalize imported weights before linear blend skinning.
        let (min_sum, max_sum) = self.skin_weight_bounds;
        for matrix in &pose.matrices {
            for x in [lower.x, upper.x] {
                for y in [lower.y, upper.y] {
                    for z in [lower.z, upper.z] {
                        let point = matrix.transform_point3(Vec3::new(x, y, z));
                        a = a.min(point * min_sum).min(point * max_sum);
                        b = b.max(point * min_sum).max(point * max_sum);
                    }
                }
            }
        }
        (a, b)
    }
    pub fn new(mesh: &Mesh, vertex_count: usize) -> Result<Self, &'static str> {
        let mut data = Self {
            vertex_count: vertex_count as u32,
            skin_weight_bounds: (f32::INFINITY, 0.0),
            ..default()
        };
        match (
            mesh.attribute(Mesh::ATTRIBUTE_JOINT_INDEX),
            mesh.attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT),
        ) {
            (
                Some(VertexAttributeValues::Uint16x4(indices)),
                Some(VertexAttributeValues::Float32x4(weights)),
            ) if indices.len() == vertex_count && weights.len() == vertex_count => {
                for (indices, weights) in indices.iter().zip(weights) {
                    let sum = weights.iter().sum::<f32>();
                    if weights.iter().any(|w| !w.is_finite() || *w < 0.0)
                        || !sum.is_finite()
                        || sum <= 0.0
                    {
                        return Err(
                            "skin vertex weights must be finite, nonnegative and have a finite nonzero sum",
                        );
                    }
                    data.max_joint = data
                        .max_joint
                        .max(u32::from(*indices.iter().max().unwrap_or(&0)));
                    data.skin_weight_bounds.0 = data.skin_weight_bounds.0.min(sum);
                    data.skin_weight_bounds.1 = data.skin_weight_bounds.1.max(sum);
                    data.influences
                        .push(UVec4::from_array(indices.map(u32::from)));
                    data.influences
                        .push(UVec4::from_array(weights.map(f32::to_bits)));
                }
            }
            (None, None) => {}
            _ => return Err("skin requires matching Uint16x4 joint indices and Float32x4 weights"),
        }
        if let Some(targets) = mesh.get_morph_targets() {
            if vertex_count == 0 || targets.len() % vertex_count != 0 {
                return Err("morph target vertex counts do not match the mesh");
            }
            data.target_count = (targets.len() / vertex_count) as u32;
            for target in targets.chunks_exact(vertex_count) {
                let mut minimum = Vec3::splat(f32::INFINITY);
                let mut maximum = Vec3::splat(f32::NEG_INFINITY);
                for vertex in target {
                    minimum = minimum.min(vertex.position);
                    maximum = maximum.max(vertex.position);
                    for value in [vertex.position, vertex.normal, vertex.tangent] {
                        if !value.is_finite() {
                            return Err("morph displacements must be finite");
                        }
                        data.morphs.push(UVec4::from_array(
                            value.extend(0.0).to_array().map(f32::to_bits),
                        ));
                    }
                }
                data.morph_bounds.push((minimum, maximum));
            }
        }
        Ok(data)
    }
    pub fn validate_pose(&self, pose: &DeformationPose) -> Result<(), &'static str> {
        if !pose.matrices.is_empty()
            && (self.influences.is_empty() || self.max_joint >= pose.matrices.len() as u32)
        {
            return Err("skin vertex references an unavailable joint");
        }
        if pose.weights.len() as u32 != self.target_count {
            return Err("morph weight count does not match resident targets");
        }
        Ok(())
    }
}

#[derive(Clone, Default)]
pub(crate) struct DeformationData {
    pub words: Arc<Vec<UVec4>>,
    pub poses: Arc<Vec<UVec4>>,
    pub vertices: u32,
    pub refit_nodes: Vec<u32>,
    /// Deepest nodes first; pairs are scalar-node-index offsets and counts.
    pub refit_levels: Vec<(u32, u32)>,
}

pub(crate) fn pose_words(pose: &DeformationPose, words: &mut Vec<UVec4>) -> (u32, u32) {
    let matrices = words.len() as u32;
    for matrix in &pose.matrices {
        for row in matrix.transpose().to_cols_array_2d() {
            words.push(UVec4::from_array(row.map(f32::to_bits)));
        }
    }
    let weights = words.len() as u32;
    for group in pose.weights.chunks(4) {
        let mut values = [0; 4];
        for (i, value) in group.iter().enumerate() {
            values[i] = value.to_bits();
        }
        words.push(UVec4::from_array(values));
    }
    (matrices, weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn morph_precedes_skin_and_bounds_cover_weight_sums() {
        let mesh = MeshDeformation {
            influences: vec![
                UVec4::new(0, 1, 0, 0),
                UVec4::from_array([0.5f32, 1.0, 0.0, 0.0].map(f32::to_bits)),
            ],
            morphs: vec![
                UVec4::from_array([1.0f32, 0.0, 0.0, 0.0].map(f32::to_bits)),
                UVec4::ZERO,
                UVec4::ZERO,
            ],
            morph_bounds: vec![(Vec3::X, Vec3::X)],
            skin_weight_bounds: (1.5, 1.5),
            vertex_count: 1,
            target_count: 1,
            max_joint: 1,
        };
        let pose = DeformationPose {
            matrices: vec![
                Mat4::from_scale(Vec3::splat(2.0)),
                Mat4::from_translation(Vec3::Y),
            ],
            weights: vec![-0.5],
        };
        mesh.validate_pose(&pose).unwrap();
        let point = mesh.deform_point(0, Vec3::X, &pose);
        assert_eq!(point, Vec3::new(1.0, 1.0, 0.0));
        let (minimum, maximum) = mesh.conservative_bounds(Vec3::ZERO, Vec3::ONE, &pose);
        assert!(point.cmpge(minimum).all() && point.cmple(maximum).all());
        let mut words = Vec::new();
        let (matrices, weights) = pose_words(&pose, &mut words);
        assert_eq!((matrices, weights), (0, 8));
        assert_eq!(f32::from_bits(words[weights as usize].x), -0.5);
    }

    #[test]
    fn cached_skin_bounds_cover_unnormalized_weights() {
        let mut source = Mesh::new(
            bevy::mesh::PrimitiveTopology::TriangleList,
            bevy::asset::RenderAssetUsages::default(),
        );
        source.insert_attribute(
            Mesh::ATTRIBUTE_JOINT_INDEX,
            VertexAttributeValues::Uint16x4(vec![[0; 4]; 2]),
        );
        source.insert_attribute(
            Mesh::ATTRIBUTE_JOINT_WEIGHT,
            vec![[0.25, 0.0, 0.0, 0.0], [1.5, 0.0, 0.0, 0.0]],
        );
        let mesh = MeshDeformation::new(&source, 2).unwrap();
        let pose = DeformationPose {
            matrices: vec![Mat4::from_translation(Vec3::new(-2.0, 1.0, 0.0))],
            ..default()
        };
        assert_eq!(
            mesh.conservative_bounds(Vec3::ZERO, Vec3::ONE, &pose),
            (Vec3::new(-3.0, 0.25, 0.0), Vec3::new(-0.25, 3.0, 1.5)),
        );
    }

    #[test]
    fn cached_morph_bounds_scale_negative_weights() {
        let mesh = MeshDeformation {
            morph_bounds: vec![(Vec3::new(-2.0, -1.0, 0.0), Vec3::new(1.0, 3.0, 0.5))],
            vertex_count: 1,
            target_count: 1,
            ..default()
        };
        let pose = DeformationPose {
            weights: vec![-0.5],
            ..default()
        };
        assert_eq!(
            mesh.conservative_bounds(Vec3::ZERO, Vec3::ONE, &pose),
            (Vec3::new(-0.5, -1.5, -0.25), Vec3::new(2.0, 1.5, 1.0),),
        );
    }
}

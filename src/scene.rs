use std::{collections::HashSet, sync::Arc};

use bevy::{
    mesh::{Indices, VertexAttributeValues},
    prelude::*,
    render::{extract_resource::ExtractResource, render_resource::ShaderType},
};

use crate::RadianceCascadesGeometry;

#[derive(Clone, Copy, Default, ShaderType)]
pub(crate) struct Triangle {
    pub a: Vec4,
    pub b: Vec4,
    pub c: Vec4,
    pub albedo: Vec4,
    pub emission: Vec4,
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
    fn center(&self) -> Vec3 {
        (self.a + self.b + self.c).truncate() / 3.0
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
    // color.w contains the spot cone offset (radius only affects Bevy specular).
    pub direction: Vec4,
    // Emissive triangle edges; edge1.w is its discrete sampling probability.
    pub edge1: Vec4,
    pub edge2: Vec4,
}

impl GiLight {
    fn punctual(
        transform: &GlobalTransform,
        color: Color,
        intensity: f32,
        range: f32,
    ) -> Option<Self> {
        let position = transform.translation().extend(range);
        let color = (color.to_linear().to_vec4().truncate() * intensity
            / (4.0 * std::f32::consts::PI))
            .extend(0.0);
        (position.is_finite() && color.is_finite() && intensity > 0.0 && range > 0.0).then_some(
            Self {
                position,
                color,
                ..default()
            },
        )
    }
}

#[derive(Default)]
pub(crate) struct SceneData {
    pub triangles: Arc<Vec<Triangle>>,
    pub nodes: Arc<Vec<BvhNode>>,
    pub lights: Vec<GiLight>,
    pub emissive_lights: Arc<Vec<GiLight>>,
}

#[derive(Resource, Clone, Default, ExtractResource)]
pub(crate) struct GiScene {
    pub data: Arc<SceneData>,
    pub revision: u64,
    pub geometry_revision: u64,
}

// Bevy system parameters and query tuples describe the scene inputs.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn update_scene(
    mut scene: ResMut<GiScene>,
    meshes: Res<Assets<Mesh>>,
    materials: Res<Assets<StandardMaterial>>,
    geometry: Query<
        (
            Entity,
            Ref<Mesh3d>,
            Ref<MeshMaterial3d<StandardMaterial>>,
            Ref<GlobalTransform>,
            Option<Ref<InheritedVisibility>>,
        ),
        With<RadianceCascadesGeometry>,
    >,
    directional: Query<(
        &DirectionalLight,
        &GlobalTransform,
        Option<&InheritedVisibility>,
    )>,
    points: Query<(&PointLight, &GlobalTransform, Option<&InheritedVisibility>)>,
    spots: Query<(&SpotLight, &GlobalTransform, Option<&InheritedVisibility>)>,
    mut previous_geometry: Local<HashSet<Entity>>,
    mut removed_visibility: RemovedComponents<InheritedVisibility>,
) {
    let visibility_removed = removed_visibility
        .read()
        .filter(|entity| previous_geometry.contains(entity))
        .count()
        > 0;
    let mut dirty = meshes.is_changed() || materials.is_changed() || visibility_removed;
    let mut geometry_count = 0;
    for (entity, mesh, material, transform, visible) in &geometry {
        geometry_count += 1;
        dirty |= !previous_geometry.contains(&entity)
            || mesh.is_changed()
            || material.is_changed()
            || transform.is_changed()
            || visible.as_ref().is_some_and(|v| v.is_changed());
    }
    // Removing a required input drops the entity from the query. Membership
    // catches every such removal without rebuilding for unrelated despawns.
    dirty |= geometry_count != previous_geometry.len();
    if dirty {
        previous_geometry.clear();
        previous_geometry.extend(geometry.iter().map(|(entity, ..)| entity));
    }

    let mut lights = Vec::new();
    for (light, transform, visible) in &directional {
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
        }
    }
    for (light, transform, visible) in &points {
        if visible.is_some_and(|v| !v.get()) {
            continue;
        }
        if let Some(light) = GiLight::punctual(transform, light.color, light.intensity, light.range)
        {
            lights.push(light);
        }
    }
    for (light, transform, visible) in &spots {
        if visible.is_some_and(|v| !v.get())
            || !light.inner_angle.is_finite()
            || !light.outer_angle.is_finite()
            || light.inner_angle < 0.0
            || light.inner_angle > light.outer_angle
            || light.outer_angle >= std::f32::consts::FRAC_PI_2
        {
            continue;
        }
        if let Some(mut gpu_light) =
            GiLight::punctual(transform, light.color, light.intensity, light.range)
        {
            let cos_outer = light.outer_angle.cos();
            let scale = 1.0 / (light.inner_angle.cos() - cos_outer).max(1e-4);
            gpu_light.direction = transform.forward().extend(scale);
            gpu_light.color.w = -cos_outer * scale;
            if gpu_light.direction.is_finite() {
                lights.push(gpu_light);
            }
        }
    }
    if !dirty && scene.data.lights == lights {
        return;
    }
    if !dirty {
        scene.data = Arc::new(SceneData {
            triangles: scene.data.triangles.clone(),
            nodes: scene.data.nodes.clone(),
            lights,
            emissive_lights: scene.data.emissive_lights.clone(),
        });
        scene.revision = scene.revision.wrapping_add(1);
        return;
    }

    // ponytail: rebuild the median-split BVH on scene edits; refit it when
    // large moving scenes make CPU rebuilds a measured bottleneck.
    let mut triangles = Vec::new();
    for (entity, mesh_handle, material_handle, transform, visible) in &geometry {
        if visible.is_some_and(|v| !v.get()) {
            continue;
        }
        let (Some(mesh), Some(material)) = (
            meshes.get(&mesh_handle.0),
            materials.get(&material_handle.0),
        ) else {
            continue;
        };
        if material.alpha_mode != AlphaMode::Opaque {
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
        let indices: Vec<usize> = match mesh.indices() {
            Some(Indices::U16(v)) => v.iter().map(|&i| i as usize).collect(),
            Some(Indices::U32(v)) => v.iter().map(|&i| i as usize).collect(),
            None => (0..positions.len()).collect(),
        };
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
        for indices in indices.as_chunks::<3>().0 {
            let [Some(a), Some(b), Some(c)] =
                [indices[0], indices[1], indices[2]].map(|i| positions.get(i))
            else {
                continue;
            };
            let [a, b, c] = [a, b, c].map(|p| transform.transform_point(Vec3::from_array(*p)));
            if !a.is_finite()
                || !b.is_finite()
                || !c.is_finite()
                || (b - a).cross(c - a).length_squared() < 1e-16
            {
                continue;
            }
            triangles.push(Triangle {
                a: a.extend(0.0),
                b: b.extend(0.0),
                c: c.extend(0.0),
                albedo: albedo.extend(0.0),
                emission: emission.extend(0.0),
            });
        }
    }
    // BVH partitioning changes triangle order during motion. Keep the emitter
    // CDF in extraction order so existing samples do not jump between lights.
    let emissive_lights = Arc::new(build_emissive_lights(&mut triangles));
    let nodes = build_bvh(&mut triangles);
    scene.data = Arc::new(SceneData {
        triangles: Arc::new(triangles),
        nodes: Arc::new(nodes),
        lights,
        emissive_lights,
    });
    scene.revision = scene.revision.wrapping_add(1);
    scene.geometry_revision = scene.geometry_revision.wrapping_add(1);
    debug!(
        revision = scene.revision,
        triangles = scene.data.triangles.len(),
        "GI scene updated"
    );
}

fn build_emissive_lights(triangles: &mut [Triangle]) -> Vec<GiLight> {
    let mut lights = Vec::new();
    let mut weights = Vec::new();
    for (index, triangle) in triangles.iter_mut().enumerate() {
        triangle.emission.w = 0.0;
        let emission = triangle.emission.truncate().max(Vec3::ZERO);
        let edge1 = (triangle.b - triangle.a).truncate();
        let edge2 = (triangle.c - triangle.a).truncate();
        let cross = edge1.cross(edge2);
        let area = cross.length() * 0.5;
        let power = f64::from(area) * f64::from(emission.dot(Vec3::new(0.2126, 0.7152, 0.0722)));
        if !power.is_finite() || power <= 0.0 {
            continue;
        }
        lights.push(GiLight {
            position: triangle.a.with_w(-1.0),
            color: emission.extend(area),
            direction: cross.normalize().extend(0.0),
            edge1: edge1.extend(0.0),
            edge2: edge2.extend(0.0),
        });
        weights.push((index, power));
    }
    let total: f64 = weights.iter().map(|(_, power)| power).sum();
    let mut cdf = 0.0;
    for (light, (index, weight)) in lights.iter_mut().zip(weights) {
        let probability = weight / total;
        cdf += probability;
        light.direction.w = cdf.min(1.0) as f32;
        light.edge1.w = probability as f32;
        // The hit-side MIS PDF must match the emitter CDF after BVH sorting.
        triangles[index].emission.w = probability as f32;
    }
    if let Some(last) = lights.last_mut() {
        last.direction.w = 1.0;
    }
    lights
}

fn build_bvh(triangles: &mut [Triangle]) -> Vec<BvhNode> {
    if triangles.is_empty() {
        return Vec::new();
    }
    fn build(nodes: &mut Vec<BvhNode>, index: usize, triangles: &mut [Triangle], first: usize) {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for triangle in triangles.iter() {
            let bounds = triangle.bounds();
            min = min.min(bounds.0);
            max = max.max(bounds.1);
        }
        nodes[index] = BvhNode {
            min,
            max,
            first: first as u32,
            count: triangles.len() as u32,
        };
        if triangles.len() <= 4 {
            return;
        }
        let size = max - min;
        let axis = if size.x >= size.y && size.x >= size.z {
            0
        } else if size.y >= size.z {
            1
        } else {
            2
        };
        let midpoint = triangles.len() / 2;
        triangles.select_nth_unstable_by(midpoint, |a, b| {
            a.center()[axis].total_cmp(&b.center()[axis])
        });
        let children = nodes.len();
        nodes.extend([BvhNode::default(); 2]);
        nodes[index].first = children as u32;
        nodes[index].count = 0;
        let (left, right) = triangles.split_at_mut(midpoint);
        build(nodes, children, left, first);
        build(nodes, children + 1, right, first + midpoint);
    }
    let mut nodes = vec![BvhNode::default()];
    build(&mut nodes, 0, triangles, 0);
    nodes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emissive_distribution_preserves_area_and_power() {
        let mut triangles = [
            Triangle {
                a: Vec4::ZERO,
                b: Vec3::X.extend(0.0),
                c: Vec3::Y.extend(0.0),
                emission: (Vec3::X * 2.0).extend(0.0),
                ..default()
            },
            Triangle {
                a: Vec4::ZERO,
                b: (Vec3::X * 2.0).extend(0.0),
                c: Vec3::Y.extend(0.0),
                emission: Vec3::Y.extend(0.0),
                ..default()
            },
            Triangle::default(),
        ];
        let lights = build_emissive_lights(&mut triangles);
        assert_eq!(lights.len(), 2);
        assert_eq!(lights[0].color.w, 0.5);
        assert_eq!(lights[1].color.w, 1.0);
        assert_eq!(lights[0].direction.truncate(), Vec3::Z);
        assert!((lights[0].edge1.w - 0.2126 / (0.2126 + 0.7152)).abs() < 1e-6);
        assert!((lights.iter().map(|light| light.edge1.w).sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(lights[0].direction.w, lights[0].edge1.w);
        assert_eq!(lights[1].direction.w, 1.0);
        assert!(lights.iter().all(|light| light.position.w == -1.0));
        for (triangle, light) in triangles.iter().zip(&lights) {
            assert_eq!(triangle.emission.w, light.edge1.w);
        }
        assert_eq!(triangles[2].emission.w, 0.0);
    }

    #[test]
    fn scene_tracks_edits_and_removal_but_reuses_static_geometry() {
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
                Mesh3d(mesh.clone()),
                MeshMaterial3d(material.clone()),
                RadianceCascadesGeometry,
                InheritedVisibility::VISIBLE,
            ))
            .id();
        app.update();
        let revision = app.world().resource::<GiScene>().revision;
        assert_eq!(app.world().resource::<GiScene>().data.triangles.len(), 12);
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
            app.world().resource::<GiScene>().data.triangles[0]
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
        app.world_mut()
            .entity_mut(entity)
            .insert(RadianceCascadesGeometry);
        app.update();
        assert_eq!(app.world().resource::<GiScene>().data.triangles.len(), 12);
        app.world_mut().entity_mut(entity).remove::<Mesh3d>();
        app.update();
        assert!(app.world().resource::<GiScene>().data.triangles.is_empty());
        app.world_mut().entity_mut(entity).insert(Mesh3d(mesh));
        app.update();
        assert_eq!(app.world().resource::<GiScene>().data.triangles.len(), 12);
        app.world_mut()
            .entity_mut(entity)
            .remove::<MeshMaterial3d<StandardMaterial>>();
        app.update();
        assert!(app.world().resource::<GiScene>().data.triangles.is_empty());
        app.world_mut()
            .entity_mut(entity)
            .insert(MeshMaterial3d(material));
        app.update();
        assert_eq!(app.world().resource::<GiScene>().data.triangles.len(), 12);
        app.world_mut()
            .entity_mut(entity)
            .remove::<GlobalTransform>();
        app.update();
        assert!(app.world().resource::<GiScene>().data.triangles.is_empty());
        app.world_mut()
            .entity_mut(entity)
            .insert(GlobalTransform::default());
        app.update();
        assert_eq!(app.world().resource::<GiScene>().data.triangles.len(), 12);
        app.world_mut()
            .entity_mut(entity)
            .insert(InheritedVisibility::HIDDEN);
        app.update();
        assert!(app.world().resource::<GiScene>().data.triangles.is_empty());
        app.world_mut()
            .entity_mut(entity)
            .remove::<InheritedVisibility>();
        app.update();
        assert_eq!(app.world().resource::<GiScene>().data.triangles.len(), 12);
        app.world_mut().entity_mut(entity).despawn();
        app.update();
        assert!(app.world().resource::<GiScene>().data.triangles.is_empty());
    }

    #[test]
    fn lights_track_motion_visibility_and_removal_without_rebuilding_geometry() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<GiScene>()
            .add_systems(Update, update_scene);
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
        for entity in [point, spot, directional] {
            let before = app.world().resource::<GiScene>().revision;
            app.world_mut()
                .entity_mut(entity)
                .insert(GlobalTransform::from(
                    Transform::from_xyz(2.0, 3.0, 4.0).with_rotation(Quat::from_rotation_y(0.8)),
                ));
            app.update();
            assert!(app.world().resource::<GiScene>().revision > before);
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
            app.update();
            assert_eq!(app.world().resource::<GiScene>().data.lights.len(), 2);
            app.world_mut()
                .entity_mut(entity)
                .insert(InheritedVisibility::VISIBLE);
            app.update();
            assert_eq!(app.world().resource::<GiScene>().data.lights.len(), 3);
        }
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

//! PXTGI: real-time global illumination traced against the physics colliders.
//!
//! A port of the experimental PXTGI from the godot_physx module
//! (`modules/godot_physx/nodes/pxtgi_3d.cpp`, `physx_gi_effect.cpp`). Each
//! visible surface sends a few rays per frame. A ray first follows what is on
//! screen, taking the light that point had in last frame's image; where it
//! leaves the screen or passes behind what the screen shows it continues
//! against the [`bevy_box3d::Collider`]s (boxes, spheres, capsules, convex
//! hulls and triangle meshes), which stand in for the geometry. Hits are lit by
//! the sun, point and spot lights (shadowed by the colliders) and by a hashed
//! world-space radiance cache that gives multiple bounces; rays leaving the
//! scene see the camera's `EnvironmentMapLight`. The result is accumulated
//! over frames, filtered along depth and normals, upsampled and added onto the
//! lit deferred image times each pixel's albedo. Smooth surfaces also get
//! traced reflections, which take over from Bevy's environment-map reflection
//! where they find something.
//!
//! Add [`PxtgiPlugin`] after `DefaultPlugins`, put [`Pxtgi`] on a 3D camera and
//! give bodies a `Collider`. A body's colour comes from [`PxtgiLook`], else its
//! own `MeshMaterial3d<StandardMaterial>`, else its first child's; otherwise it
//! bounces as neutral grey. Emissive `StandardMaterial` meshes outside any
//! collider body give off light as their bounding box. Independent of
//! [`crate::RadianceCascadesPlugin`]; use one or the other on a camera.
//!
//! Not ported: reflection probes (Bevy's light probes work differently). The
//! diffuse GI is added to Bevy's own ambient light rather than replacing it,
//! so keep `AmbientLight` low on PXTGI cameras.

use std::{
    collections::{HashMap, HashSet, hash_map::DefaultHasher},
    f32::consts::PI,
    hash::{Hash, Hasher},
    sync::Arc,
};

use bevy::{
    camera::{Hdr, primitives::Aabb},
    core_pipeline::{
        Core3dSystems, FullscreenShader,
        core_3d::{main_opaque_pass_3d, main_transparent_pass_3d},
        prepass::{DeferredPrepass, DepthPrepass, ViewPrepassTextures},
        schedule::Core3d,
    },
    math::Affine3A,
    prelude::*,
    render::{
        Extract, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
        camera::ExtractedCamera,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
        sync_world::RenderEntity,
        texture::GpuImage,
        view::{ExtractedView, ViewTarget},
    },
    shader::Shader,
    tasks::{AsyncComputeTaskPool, Task, block_on, poll_once},
    transform::TransformSystems,
};
use bevy_box3d::Collider;
use bevy_slang::{SlangCompiler, SlangLoaderError, SlangSettings, SlangTarget};

/// Radiance-cache slots shared by every PXTGI view (80 bytes each).
const CACHE_SLOTS: u32 = 1 << 17;
// The cache pass walks the slots as a 512-wide grid, a quarter of them a frame.
const CACHE_ROW: u32 = 512;
const PRIMITIVE_VEC4S: usize = 6;
const MAX_LIGHTS: usize = 8;
const MAX_EMISSIVES: usize = 16;
// Frames the cache follows quickly after the light sources change.
const FRESH_FRAMES: u32 = 30;
// Refits before the tree is rebuilt: moving bodies loosen its boxes.
const REFITS_PER_BUILD: u32 = 120;

const SOURCE: &str = include_str!("../shaders/pxtgi.slang");
const ENTRY_POINTS: [&str; 11] = [
    "cache_update",
    "trace",
    "temporal",
    "atrous1",
    "atrous2",
    "atrous4",
    "reflect_trace",
    "reflect_temporal",
    "reflect_filter1",
    "reflect_filter2",
    "composite",
];

/// Adds PXTGI to cameras with [`Pxtgi`]. Requires Bevy's render plugins and
/// [`bevy_box3d::Box3dPlugin`] (or colliders placed some other way).
///
/// The shaders are Slang, compiled to WGSL by `slangc` (`SLANGC` or on `PATH`)
/// in the background at startup; PXTGI starts drawing once that finishes.
#[derive(Clone, Debug, Default)]
pub struct PxtgiPlugin {
    /// Slang compiler configuration.
    pub compiler: SlangCompiler,
}

fn compile_shader(compiler: &SlangCompiler) -> Result<Shader, SlangLoaderError> {
    let mut shader = compiler.compile_bundle(
        "pxtgi.slang",
        &[("pxtgi.slang", SOURCE)],
        &SlangSettings {
            target: SlangTarget::Wgsl,
            entry_points: ENTRY_POINTS.map(str::to_owned).to_vec(),
            optimization: Some(2),
            ..default()
        },
    )?;
    shader.validate_shader = bevy::shader::ValidateShader::Enabled;
    Ok(shader)
}

/// The background compilation filling the shader handle the pipelines use.
#[derive(Resource)]
struct PxtgiCompilation {
    compiler: SlangCompiler,
    handle: Handle<Shader>,
    task: Option<Task<Result<Shader, SlangLoaderError>>>,
    started: bool,
}

fn compile_pxtgi_shader(mut compilation: ResMut<PxtgiCompilation>, mut shaders: ResMut<Assets<Shader>>) {
    if !compilation.started {
        compilation.started = true;
        let compiler = compilation.compiler.clone();
        compilation.task = Some(AsyncComputeTaskPool::get().spawn(async move { compile_shader(&compiler) }));
    }
    let Some(task) = compilation.task.as_mut() else {
        return;
    };
    let Some(result) = block_on(poll_once(task)) else {
        return;
    };
    compilation.task = None;
    match result {
        Ok(shader) => {
            if let Err(error) = shaders.insert(compilation.handle.id(), shader) {
                error!(%error, "PXTGI shader handle expired");
            }
        }
        Err(error) => error!(%error, "PXTGI shader compilation failed; PXTGI stays off"),
    }
}

/// Which reflections PXTGI traces on surfaces smoother than
/// [`Pxtgi::reflection_roughness_cutoff`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PxtgiReflections {
    /// Leave reflections to Bevy (environment map, screen-space reflections).
    Off,
    /// Trace them on screen only.
    ScreenSpace,
    /// Trace them on screen, then against the colliders, so what is behind
    /// the camera or off screen is reflected too.
    #[default]
    Traced,
}

/// Per-camera PXTGI settings. Distances are in metres.
///
/// Requires an HDR, single-sample deferred 3D camera with a perspective
/// projection; required components are inserted automatically.
#[derive(Component, Clone, Copy, Debug, PartialEq, ExtractComponent)]
#[extract_app(RenderApp)]
#[require(Camera3d, Hdr, DepthPrepass, DeferredPrepass, Msaa::Off)]
pub struct Pxtgi {
    /// Strength of the bounce light: 1 is physically based; zero turns it off.
    pub intensity: f32,
    /// How far rays follow what is on screen before continuing against the
    /// colliders.
    pub max_distance: f32,
    /// How far rays go against the colliders. Light from surfaces further
    /// away is not bounced.
    pub world_ray_distance: f32,
    /// Rays each traced texel sends per frame, 1..=4. More settle faster and
    /// flicker less, at about proportional cost.
    pub rays_per_texel: u32,
    /// Light bounces more than once through the radiance cache. Off, every ray
    /// that leaves the screen lights what it hits from scratch (one bounce).
    pub multi_bounce: bool,
    /// Traced reflections on smooth surfaces.
    pub reflections: PxtgiReflections,
    /// Surfaces with a perceptual roughness below this get traced
    /// reflections, faded into Bevy's own over the last quarter of it. Traced
    /// reflections cost per pixel; rougher surfaces look the same from the
    /// environment map.
    pub reflection_roughness_cutoff: f32,
    /// Radiance (nits) a ray leaving the scene sees when the camera has no
    /// `EnvironmentMapLight`.
    pub sky: LinearRgba,
    /// Shows one stage instead of the lit scene: 0 off, 1 last frame's image
    /// reprojected, 2 the bounce light alone, 3 colliders only (no screen
    /// march), 4 the fraction of rays that hit, 5 the traced reflections.
    pub debug: u32,
}

impl Default for Pxtgi {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            max_distance: 8.0,
            world_ray_distance: 30.0,
            rays_per_texel: 4,
            multi_bounce: true,
            reflections: PxtgiReflections::Traced,
            reflection_roughness_cutoff: 0.5,
            sky: LinearRgba::BLACK,
            debug: 0,
        }
    }
}

/// Overrides the albedo and emitted radiance PXTGI gives a collider's body,
/// for bodies whose material isn't a `StandardMaterial`.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct PxtgiLook {
    /// Diffuse albedo.
    pub albedo: LinearRgba,
    /// Emitted radiance, as `StandardMaterial::emissive`.
    pub emission: LinearRgba,
}

#[derive(Clone, Copy, Default)]
struct Light {
    position: Vec3,
    inv_range2: f32,
    // Linear colour times radiant intensity (candela).
    color: Vec3,
    direction: Vec3,
    cone_scale: f32,
    cone_offset: f32,
}

#[derive(Clone, Copy)]
struct Emissive {
    center: Vec3,
    axes: [Vec3; 3],
    color: Vec3,
}

/// The traced scene as the GPU takes it.
#[derive(Clone, Default)]
struct Geometry {
    version: u64,
    // Records in BVH leaf order, PRIMITIVE_VEC4S each.
    primitives: Vec<[f32; 4]>,
    nodes: Vec<[f32; 4]>,
    // Hull planes and mesh trees and triangles, shared by the records.
    shapes: Arc<Vec<[f32; 4]>>,
    shapes_version: u64,
}

#[derive(Default)]
struct SceneData {
    geometry: Arc<Geometry>,
    // Toward the sun, and linear colour times illuminance (lux).
    sun: Option<(Vec3, Vec3)>,
    omnis: Vec<Light>,
    spots: Vec<Light>,
    emissives: Vec<Emissive>,
    sources_changed: bool,
}

#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
struct PxtgiScene(Arc<SceneData>);

#[derive(Resource)]
struct PxtgiShader(Handle<Shader>);

/// The camera's environment map, as PXTGI's sky.
#[derive(Component)]
struct PxtgiEnvironment {
    specular: AssetId<Image>,
    intensity: f32,
    rotation: Quat,
}

impl Plugin for PxtgiPlugin {
    fn build(&self, app: &mut App) {
        // Pipelines are queued with this handle and wait until compilation fills it.
        let shader = app.world().resource::<Assets<Shader>>().reserve_handle();
        app.init_resource::<PxtgiScene>()
            .insert_resource(PxtgiCompilation {
                compiler: self.compiler.clone(),
                handle: shader.clone(),
                task: None,
                started: false,
            })
            .add_plugins((
                ExtractComponentPlugin::<Pxtgi>::default(),
                ExtractResourcePlugin::<PxtgiScene>::default(),
            ))
            .add_systems(Update, compile_pxtgi_shader)
            .add_systems(PostUpdate, update_scene.after(TransformSystems::Propagate));
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .insert_resource(PxtgiShader(shader))
            .add_systems(ExtractSchedule, extract_environment)
            .add_systems(RenderStartup, init_pipelines)
            .add_systems(
                Render,
                (
                    prepare_scene.in_set(RenderSystems::PrepareResources),
                    prepare_views.in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(
                Core3d,
                render
                    .in_set(Core3dSystems::MainPass)
                    .after(main_opaque_pass_3d)
                    .before(main_transparent_pass_3d),
            );
    }
}

// ---------------------------------------------------------------- main world

/// One traced box, sphere, capsule, hull or mesh: a collider, or an emissive
/// mesh's bounding box.
#[derive(Clone)]
struct Item {
    entity: Entity,
    // Colliders ignore scale (Box3D sizes them in world units); mesh boxes keep it.
    scaled: bool,
    // The shape's local origin in its entity's space (a mesh box's centre).
    offset: Vec3,
    to_world: Affine3A,
    // Kind, then extents or geometry offsets as the shader reads them.
    shape: [f32; 4],
    lower: Vec3,
    upper: Vec3,
    albedo: LinearRgba,
    emission: LinearRgba,
}

impl Item {
    fn place(&mut self, transform: &GlobalTransform) {
        let base = if self.scaled {
            transform.affine()
        } else {
            let (_, rotation, translation) = transform.to_scale_rotation_translation();
            Affine3A::from_rotation_translation(rotation, translation)
        };
        self.to_world = base * Affine3A::from_translation(self.offset);
    }

    fn world_bounds(&self) -> (Vec3, Vec3) {
        let center = self.to_world.transform_point3((self.lower + self.upper) * 0.5);
        let half = (self.upper - self.lower) * 0.5;
        let m = self.to_world.matrix3;
        let extent = Vec3::from(m.x_axis.abs() * half.x + m.y_axis.abs() * half.y + m.z_axis.abs() * half.z);
        (center - extent, center + extent)
    }

    fn rows(&self) -> [[f32; 4]; 3] {
        let to_local = self.to_world.inverse();
        let rows = Mat3::from(to_local.matrix3).transpose();
        std::array::from_fn(|r| rows.col(r).extend(to_local.translation[r]).to_array())
    }

    fn record(&self) -> [[f32; 4]; PRIMITIVE_VEC4S] {
        let [r0, r1, r2] = self.rows();
        let (a, e) = (self.albedo, self.emission);
        [r0, r1, r2, self.shape, [a.red, a.green, a.blue, 1.0], [e.red, e.green, e.blue, 0.0]]
    }

    fn emissive(&self) -> Option<Emissive> {
        let color = Vec3::new(self.emission.red, self.emission.green, self.emission.blue);
        (color.max_element() > 0.0).then(|| {
            let half = (self.upper - self.lower) * 0.5;
            let m = self.to_world.matrix3;
            Emissive {
                center: self.to_world.transform_point3((self.lower + self.upper) * 0.5),
                axes: [
                    Vec3::from(m.x_axis) * half.x,
                    Vec3::from(m.y_axis) * half.y,
                    Vec3::from(m.z_axis) * half.z,
                ],
                color,
            }
        })
    }
}

/// Hull planes and mesh trees, appended once per shared shape.
#[derive(Default)]
struct ShapeStore {
    data: Vec<[f32; 4]>,
    // Shape identity (its data's address) -> its two offsets in `data`.
    index: HashMap<usize, (u32, u32)>,
    version: u64,
}

impl ShapeStore {
    /// The shader's shape vec4 and local bounds of a collider, appending its
    /// hull or mesh data the first time it is seen.
    fn shape(&mut self, collider: &Collider) -> ([f32; 4], Vec3, Vec3) {
        let bits = |(a, b): (u32, u32)| [f32::from_bits(a), f32::from_bits(b)];
        match collider {
            Collider::Cuboid { half_extents } => {
                ([0.0, half_extents.x, half_extents.y, half_extents.z], -*half_extents, *half_extents)
            }
            Collider::Sphere { radius } => ([1.0, *radius, 0.0, 0.0], Vec3::splat(-radius), Vec3::splat(*radius)),
            Collider::Capsule { radius, half_height } => {
                let half = Vec3::new(*radius, half_height + radius, *radius);
                ([2.0, *radius, half.y, 0.0], -half, half)
            }
            Collider::ConvexHull(hull) => {
                let planes = hull.planes();
                let at = *self.index.entry(planes.as_ptr() as usize).or_insert_with(|| {
                    self.version += 1;
                    let first = self.data.len() as u32;
                    self.data.extend_from_slice(planes);
                    (first, planes.len() as u32)
                });
                let [a, b] = bits(at);
                let (lower, upper) = hull.bounds();
                ([4.0, a, b, 0.0], lower, upper)
            }
            Collider::TriMesh(mesh) => {
                let at = *self.index.entry(mesh.nodes().as_ptr() as usize).or_insert_with(|| {
                    self.version += 1;
                    let nodes = self.data.len() as u32;
                    self.data.extend_from_slice(mesh.nodes());
                    let triangles = self.data.len() as u32;
                    self.data.extend_from_slice(mesh.triangles());
                    (nodes, triangles)
                });
                let [a, b] = bits(at);
                let (lower, upper) = mesh.bounds();
                ([5.0, a, b, 0.0], lower, upper)
            }
        }
    }
}

fn shape_key(collider: &Collider) -> Option<usize> {
    match collider {
        Collider::ConvexHull(hull) => Some(hull.planes().as_ptr() as usize),
        Collider::TriMesh(mesh) => Some(mesh.nodes().as_ptr() as usize),
        _ => None,
    }
}

#[derive(Default)]
struct SceneState {
    geometry: Arc<Geometry>,
    store: ShapeStore,
    // Items in record (leaf) order, their world bounds, and where each entity's is.
    items: Vec<Item>,
    bounds: Vec<(Vec3, Vec3)>,
    slots: HashMap<Entity, usize>,
    refits: u32,
    sources_hash: u64,
    fresh_frames: u32,
}

type LightQueries<'w, 's> = (
    Query<'w, 's, (&'static DirectionalLight, &'static GlobalTransform, &'static InheritedVisibility)>,
    Query<'w, 's, (&'static PointLight, &'static GlobalTransform, &'static InheritedVisibility)>,
    Query<'w, 's, (&'static SpotLight, &'static GlobalTransform, &'static InheritedVisibility)>,
);

type ProxyQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static GlobalTransform,
        &'static Aabb,
        &'static MeshMaterial3d<StandardMaterial>,
        Option<&'static ChildOf>,
        &'static InheritedVisibility,
    ),
    (With<Mesh3d>, Without<Collider>),
>;

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn update_scene(
    mut scene: ResMut<PxtgiScene>,
    mut state: Local<SceneState>,
    cameras: Query<&GlobalTransform, With<Pxtgi>>,
    colliders: Query<(
        Entity,
        &Collider,
        &GlobalTransform,
        Option<&PxtgiLook>,
        Option<&MeshMaterial3d<StandardMaterial>>,
        Option<&Children>,
    )>,
    proxies: ProxyQuery,
    has_collider: Query<(), With<Collider>>,
    restyled: (
        Query<
            (),
            (
                With<Collider>,
                Or<(Changed<Collider>, Changed<PxtgiLook>, Changed<MeshMaterial3d<StandardMaterial>>)>,
            ),
        >,
        Query<
            (),
            (
                With<Mesh3d>,
                Without<Collider>,
                Or<(
                    Changed<Mesh3d>,
                    Changed<MeshMaterial3d<StandardMaterial>>,
                    Changed<Aabb>,
                    Changed<InheritedVisibility>,
                )>,
            ),
        >,
    ),
    moved: Query<(Entity, &GlobalTransform), (Or<(With<Collider>, With<Mesh3d>)>, Changed<GlobalTransform>)>,
    mut removed: (RemovedComponents<Collider>, RemovedComponents<Mesh3d>),
    child_materials: Query<&MeshMaterial3d<StandardMaterial>>,
    materials: Res<Assets<StandardMaterial>>,
    mut material_events: MessageReader<AssetEvent<StandardMaterial>>,
    lights: LightQueries,
) {
    let removed = removed.0.read().count() + removed.1.read().count() > 0;
    let material_changed = material_events.read().count() > 0;
    // shortcut: lights and emissives are chosen for the first PXTGI camera;
    // per-view selection if split screen ever needs it.
    let Some(eye) = cameras.iter().next().map(GlobalTransform::translation) else {
        return;
    };
    let state = &mut *state;
    let any_moved = !moved.is_empty();
    let rebuild = removed
        || material_changed
        || !restyled.0.is_empty()
        || !restyled.1.is_empty()
        || state.geometry.version == 0
        || (any_moved && state.refits >= REFITS_PER_BUILD);
    if rebuild {
        // Shared shape data no live collider uses is dropped once it is most of the store.
        let live: HashSet<usize> = colliders.iter().filter_map(|c| shape_key(c.1)).collect();
        if state.store.index.len() > 2 * live.len().max(8) {
            let version = state.store.version + 1;
            state.store = ShapeStore { version, ..default() };
        }
        let mut items = Vec::new();
        for (entity, collider, transform, look, material, children) in &colliders {
            let look = look.map(|look| (look.albedo, look.emission)).or_else(|| {
                material
                    .or_else(|| children?.iter().find_map(|child| child_materials.get(child).ok()))
                    .and_then(|material| materials.get(&material.0))
                    .map(|material| (material.base_color.to_linear(), material.emissive))
            });
            let (albedo, emission) = look.unwrap_or((LinearRgba::rgb(0.5, 0.5, 0.5), LinearRgba::BLACK));
            let (shape, lower, upper) = state.store.shape(collider);
            let mut item = Item {
                entity,
                scaled: false,
                offset: Vec3::ZERO,
                to_world: Affine3A::IDENTITY,
                shape,
                lower,
                upper,
                albedo,
                emission,
            };
            item.place(transform);
            items.push(item);
        }
        // Emissive meshes outside any collider body stand in as their box.
        for (entity, transform, aabb, material, child_of, visible) in &proxies {
            let Some(material) = materials.get(&material.0) else {
                continue;
            };
            let emission = material.emissive;
            if !visible.get()
                || emission.red + emission.green + emission.blue <= 0.0
                || child_of.is_some_and(|c| has_collider.contains(c.parent()))
            {
                continue;
            }
            let half = Vec3::from(aabb.half_extents).max(Vec3::splat(1e-3));
            let mut item = Item {
                entity,
                scaled: true,
                offset: Vec3::from(aabb.center),
                to_world: Affine3A::IDENTITY,
                shape: [0.0, half.x, half.y, half.z],
                lower: -half,
                upper: half,
                albedo: material.base_color.to_linear(),
                emission,
            };
            item.place(transform);
            items.push(item);
        }
        let geometry = build_geometry(&mut items, state.geometry.version + 1);
        state.bounds = items.iter().map(Item::world_bounds).collect();
        state.slots = items.iter().enumerate().map(|(slot, item)| (item.entity, slot)).collect();
        state.items = items;
        state.refits = 0;
        let shapes_changed = state.store.version != state.geometry.shapes_version;
        state.geometry = Arc::new(Geometry {
            shapes: if shapes_changed {
                Arc::new(state.store.data.clone())
            } else {
                state.geometry.shapes.clone()
            },
            shapes_version: state.store.version,
            ..geometry
        });
    } else if any_moved {
        // Only poses changed: rewrite the moved records and refit the tree.
        let geometry = Arc::make_mut(&mut state.geometry);
        let mut refit = false;
        for (entity, transform) in &moved {
            let Some(&slot) = state.slots.get(&entity) else {
                continue;
            };
            let item = &mut state.items[slot];
            item.place(transform);
            let at = slot * PRIMITIVE_VEC4S;
            geometry.primitives[at..at + 3].copy_from_slice(&item.rows());
            state.bounds[slot] = item.world_bounds();
            refit = true;
        }
        if refit {
            refit_nodes(&mut geometry.nodes, &state.bounds);
            geometry.version += 1;
            state.refits += 1;
        }
    }

    let (suns, points, spot_lights) = &lights;
    let sun = suns
        .iter()
        .filter(|(light, _, visible)| visible.get() && light.illuminance > 0.0)
        .max_by(|a, b| a.0.illuminance.total_cmp(&b.0.illuminance))
        .map(|(light, transform, _)| {
            (
                *transform.back(),
                Vec3::from_slice(&light.color.to_linear().to_f32_array()[..3]) * light.illuminance,
            )
        });
    // Of the lights, those whose reach comes nearest the camera.
    let nearest = |mut lights: Vec<(f32, Light)>| {
        lights.sort_by(|a, b| a.0.total_cmp(&b.0));
        lights.into_iter().take(MAX_LIGHTS).map(|(_, light)| light).collect::<Vec<_>>()
    };
    let light = |color: Color, lumens: f32, range: f32, transform: &GlobalTransform| Light {
        position: transform.translation(),
        inv_range2: 1.0 / (range * range).max(1e-8),
        // Bevy's point and spot lights both spread their lumens over 4 pi.
        color: Vec3::from_slice(&color.to_linear().to_f32_array()[..3]) * lumens / (4.0 * PI),
        direction: *transform.forward(),
        ..default()
    };
    let omnis = nearest(
        points
            .iter()
            .filter(|(light, _, visible)| visible.get() && light.intensity > 0.0)
            .map(|(point, transform, _)| {
                (
                    transform.translation().distance(eye) - point.range,
                    light(point.color, point.intensity, point.range, transform),
                )
            })
            .collect(),
    );
    let spots = nearest(
        spot_lights
            .iter()
            .filter(|(light, _, visible)| visible.get() && light.intensity > 0.0)
            .map(|(spot, transform, _)| {
                let (cos_outer, cos_inner) = (spot.outer_angle.cos(), spot.inner_angle.cos());
                let cone_scale = 1.0 / (cos_inner - cos_outer).max(1e-4);
                let light = Light {
                    cone_scale,
                    cone_offset: -cos_outer * cone_scale,
                    ..light(spot.color, spot.intensity, spot.range, transform)
                };
                (transform.translation().distance(eye) - spot.range, light)
            })
            .collect(),
    );
    let mut emissives: Vec<_> = state.items.iter().filter_map(Item::emissive).collect();
    emissives.sort_by(|a, b| a.center.distance_squared(eye).total_cmp(&b.center.distance_squared(eye)));
    emissives.truncate(MAX_EMISSIVES);

    // What gives off light, not where: a change lets the cache follow quickly.
    let mut hasher = DefaultHasher::new();
    let mut hash_vec = |v: Vec3| v.to_array().map(f32::to_bits).hash(&mut hasher);
    if let Some((_, color)) = sun {
        hash_vec(color);
    }
    omnis.iter().chain(&spots).for_each(|light| hash_vec(light.color));
    emissives.iter().for_each(|emissive| hash_vec(emissive.color));
    let hash = hasher.finish();
    if hash != state.sources_hash {
        state.sources_hash = hash;
        state.fresh_frames = FRESH_FRAMES;
    }
    state.fresh_frames = state.fresh_frames.saturating_sub(1);

    scene.0 = Arc::new(SceneData {
        geometry: state.geometry.clone(),
        sun,
        omnis,
        spots,
        emissives,
        sources_changed: state.fresh_frames > 0,
    });
}

struct BuildItem {
    index: usize,
    lo: Vec3,
    hi: Vec3,
    center: Vec3,
}

/// Builds the tree over `items` and reorders them into its leaf order.
fn build_geometry(items: &mut Vec<Item>, version: u64) -> Geometry {
    let mut build: Vec<_> = items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let (lo, hi) = item.world_bounds();
            BuildItem {
                index,
                lo,
                hi,
                center: (lo + hi) * 0.5,
            }
        })
        .collect();
    let mut geometry = Geometry {
        version,
        ..default()
    };
    let mut order = Vec::with_capacity(items.len());
    if !build.is_empty() {
        build_bvh(&mut build, &mut geometry.nodes, &mut order, 0);
    }
    let mut taken: Vec<Option<Item>> = items.drain(..).map(Some).collect();
    items.extend(order.into_iter().map(|index| taken[index].take().expect("leaf order is a permutation")));
    geometry.primitives = items.iter().flat_map(Item::record).collect();
    geometry
}

fn surface_area(lo: Vec3, hi: Vec3) -> f32 {
    let d = (hi - lo).max(Vec3::ZERO);
    d.x * d.y + d.y * d.z + d.z * d.x
}

// Binned SAH: the split (12 bins an axis) with the least summed area times
// count. None when every centre falls in one bin.
fn sah_split(items: &mut [BuildItem], c_lo: Vec3, c_hi: Vec3) -> Option<usize> {
    const BINS: usize = 12;
    let mut best: Option<(f32, usize, usize)> = None;
    for axis in 0..3 {
        let extent = c_hi[axis] - c_lo[axis];
        if extent <= 1e-6 {
            continue;
        }
        let scale = BINS as f32 / extent;
        let bin = |item: &BuildItem| (((item.center[axis] - c_lo[axis]) * scale) as usize).min(BINS - 1);
        let mut count = [0usize; BINS];
        let mut lo = [Vec3::INFINITY; BINS];
        let mut hi = [Vec3::NEG_INFINITY; BINS];
        for item in items.iter() {
            let b = bin(item);
            count[b] += 1;
            lo[b] = lo[b].min(item.lo);
            hi[b] = hi[b].max(item.hi);
        }
        // Cost of the left side up to and including each bin.
        let mut left_cost = [0.0; BINS];
        let (mut acc_lo, mut acc_hi, mut n) = (Vec3::INFINITY, Vec3::NEG_INFINITY, 0);
        for b in 0..BINS {
            (acc_lo, acc_hi, n) = (acc_lo.min(lo[b]), acc_hi.max(hi[b]), n + count[b]);
            left_cost[b] = surface_area(acc_lo, acc_hi) * n as f32;
        }
        let (mut acc_lo, mut acc_hi, mut n) = (Vec3::INFINITY, Vec3::NEG_INFINITY, 0);
        for b in (1..BINS).rev() {
            (acc_lo, acc_hi, n) = (acc_lo.min(lo[b]), acc_hi.max(hi[b]), n + count[b]);
            if n == 0 || n == items.len() {
                continue;
            }
            let cost = left_cost[b - 1] + surface_area(acc_lo, acc_hi) * n as f32;
            if best.is_none_or(|(c, ..)| cost < c) {
                best = Some((cost, axis, b));
            }
        }
    }
    let (_, axis, split) = best?;
    let scale = BINS as f32 / (c_hi[axis] - c_lo[axis]);
    let mut mid = 0;
    for i in 0..items.len() {
        if ((((items[i].center[axis] - c_lo[axis]) * scale) as usize).min(BINS - 1)) < split {
            items.swap(i, mid);
            mid += 1;
        }
    }
    (mid > 0 && mid < items.len()).then_some(mid)
}

// Depth first, the left child right after its parent; leaves of up to four.
// Never deeper than 15 levels: the shader's walk keeps a 16-entry stack.
fn build_bvh(items: &mut [BuildItem], nodes: &mut Vec<[f32; 4]>, order: &mut Vec<usize>, depth: u32) -> usize {
    let (mut lo, mut hi) = (Vec3::INFINITY, Vec3::NEG_INFINITY);
    let (mut c_lo, mut c_hi) = (Vec3::INFINITY, Vec3::NEG_INFINITY);
    for item in items.iter() {
        (lo, hi) = (lo.min(item.lo), hi.max(item.hi));
        (c_lo, c_hi) = (c_lo.min(item.center), c_hi.max(item.center));
    }
    let node = nodes.len() / 2;
    nodes.push(lo.extend(0.0).to_array());
    nodes.push(hi.extend(0.0).to_array());
    if items.len() <= 4 || depth >= 15 {
        nodes[node * 2][3] = f32::from_bits(order.len() as u32);
        nodes[node * 2 + 1][3] = f32::from_bits(items.len() as u32);
        order.extend(items.iter().map(|item| item.index));
        return node;
    }
    let mid = sah_split(items, c_lo, c_hi).unwrap_or_else(|| {
        // No SAH split (all centres in one bin): the median along the widest axis.
        let axis = (c_hi - c_lo).max_position();
        let mid = items.len() / 2;
        items.select_nth_unstable_by(mid, |a, b| a.center[axis].total_cmp(&b.center[axis]));
        mid
    });
    let (left, right) = items.split_at_mut(mid);
    build_bvh(left, nodes, order, depth + 1);
    let right = build_bvh(right, nodes, order, depth + 1);
    // An inner node's hi.w stays 0 (its count); lo.w is the right child.
    nodes[node * 2][3] = f32::from_bits(right as u32);
    node
}

// Children follow their parents, so walking the nodes backwards refits each
// after both of its children.
fn refit_nodes(nodes: &mut [[f32; 4]], bounds: &[(Vec3, Vec3)]) {
    let union = |a: (Vec3, Vec3), b: (Vec3, Vec3)| (a.0.min(b.0), a.1.max(b.1));
    let node_bounds = |nodes: &[[f32; 4]], n: usize| {
        (Vec3::from_slice(&nodes[n * 2]), Vec3::from_slice(&nodes[n * 2 + 1]))
    };
    for n in (0..nodes.len() / 2).rev() {
        let link = nodes[n * 2][3].to_bits() as usize;
        let count = nodes[n * 2 + 1][3].to_bits() as usize;
        let (lo, hi) = if count == 0 {
            union(node_bounds(nodes, n + 1), node_bounds(nodes, link))
        } else {
            bounds[link..link + count]
                .iter()
                .fold((Vec3::INFINITY, Vec3::NEG_INFINITY), |a, b| union(a, *b))
        };
        nodes[n * 2][..3].copy_from_slice(&lo.to_array());
        nodes[n * 2 + 1][..3].copy_from_slice(&hi.to_array());
    }
}

// ---------------------------------------------------------------- render world

fn extract_environment(
    mut commands: Commands,
    cameras: Extract<Query<(RenderEntity, Option<&EnvironmentMapLight>), With<Pxtgi>>>,
) {
    for (entity, environment) in &cameras {
        let mut entity = commands.entity(entity);
        match environment {
            Some(environment) => entity.insert(PxtgiEnvironment {
                specular: environment.specular_map.id(),
                intensity: environment.intensity,
                rotation: environment.rotation,
            }),
            None => entity.remove::<PxtgiEnvironment>(),
        };
    }
}

#[derive(Resource)]
struct PxtgiPipelines {
    scene_layout: BindGroupLayoutDescriptor,
    io_layout: BindGroupLayoutDescriptor,
    composite_layout: BindGroupLayoutDescriptor,
    cache: CachedComputePipelineId,
    // trace, temporal, the three a-trous passes.
    diffuse: [CachedComputePipelineId; 5],
    // reflect_trace, reflect_temporal, two filter passes.
    reflect: [CachedComputePipelineId; 4],
    composite: CachedRenderPipelineId,
    composite_debug: CachedRenderPipelineId,
    sampler: Sampler,
    // Bound in place of an environment map the camera doesn't have.
    no_environment: TextureView,
}

#[derive(Resource)]
struct PxtgiGpu {
    primitives: Buffer,
    nodes: Buffer,
    shapes: Buffer,
    cache_keys: Buffer,
    cells: Buffer,
    version: u64,
    shapes_version: u64,
    primitive_count: u32,
    node_count: u32,
}

struct Target {
    texture: Texture,
    view: TextureView,
}

/// One view's GI targets and history: each camera keeps its own.
#[derive(Component)]
struct PxtgiView {
    size: UVec2,
    trace_size: UVec2,
    reflect_size: UVec2,
    trace: Target,
    accum: [Target; 2],
    accum_depth: [Target; 2],
    moments: [Target; 2],
    variance: Target,
    filtered: [Target; 2],
    filtered_variance: [Target; 2],
    reflect_trace: Target,
    reflect_accum: [Target; 2],
    reflect_depth: [Target; 2],
    reflect_filtered: [Target; 2],
    prev_color: Target,
    frame: Buffer,
    current: usize,
    history: bool,
    prev_clip_from_world: Mat4,
}

#[derive(Component)]
struct PxtgiBindGroups {
    scene: BindGroup,
    // trace, temporal, the three a-trous passes.
    diffuse: [BindGroup; 5],
    // reflect_trace, reflect_temporal, two filter passes.
    reflect: [BindGroup; 4],
    composite: BindGroup,
}

fn storage_buffer(device: &RenderDevice, label: &'static str, bytes: u64) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: bytes.max(16),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn bytes(data: &[[f32; 4]]) -> Vec<u8> {
    data.iter().flatten().flat_map(|v| v.to_le_bytes()).collect()
}

fn init_pipelines(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    fullscreen: Res<FullscreenShader>,
    shader: Res<PxtgiShader>,
) {
    let both = ShaderStages::COMPUTE | ShaderStages::FRAGMENT;
    let scene_layout = BindGroupLayoutDescriptor::new(
        "pxtgi_scene",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer_sized(false, None).visibility(both),
                texture_depth_2d().visibility(both),
                texture_2d(TextureSampleType::Uint).visibility(both),
                texture_2d(TextureSampleType::Float { filterable: true }).visibility(both),
                sampler(SamplerBindingType::Filtering).visibility(both),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                texture_cube(TextureSampleType::Float { filterable: true }).visibility(both),
            ),
        ),
    );
    let unfiltered = || texture_2d(TextureSampleType::Float { filterable: false });
    let io_layout = BindGroupLayoutDescriptor::new(
        "pxtgi_io",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                unfiltered(),
                unfiltered(),
                unfiltered(),
                unfiltered(),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
            ),
        ),
    );
    let composite_layout = BindGroupLayoutDescriptor::new(
        "pxtgi_composite",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::FRAGMENT,
            (
                (8, unfiltered()),
                (9, unfiltered()),
                (10, unfiltered()),
                (11, unfiltered()),
            ),
        ),
    );
    let compute = |entry: &'static str| {
        cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(format!("pxtgi_{entry}").into()),
            layout: vec![scene_layout.clone(), io_layout.clone()],
            shader: shader.0.clone(),
            entry_point: Some(entry.into()),
            ..default()
        })
    };
    let composite = |blend: Option<BlendState>| {
        cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("pxtgi_composite".into()),
            layout: vec![scene_layout.clone(), composite_layout.clone()],
            vertex: fullscreen.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: shader.0.clone(),
                entry_point: Some("composite".into()),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba16Float,
                    blend,
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            ..default()
        })
    };
    let additive = BlendState {
        color: BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::One,
            operation: BlendOperation::Add,
        },
        alpha: BlendComponent {
            src_factor: BlendFactor::Zero,
            dst_factor: BlendFactor::One,
            operation: BlendOperation::Add,
        },
    };
    let no_environment = device
        .create_texture(&TextureDescriptor {
            label: Some("pxtgi_no_environment"),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&TextureViewDescriptor {
            dimension: Some(TextureViewDimension::Cube),
            ..default()
        });
    commands.insert_resource(PxtgiPipelines {
        cache: compute("cache_update"),
        diffuse: ["trace", "temporal", "atrous1", "atrous2", "atrous4"].map(compute),
        reflect: ["reflect_trace", "reflect_temporal", "reflect_filter1", "reflect_filter2"].map(compute),
        composite: composite(Some(additive)),
        composite_debug: composite(None),
        sampler: device.create_sampler(&SamplerDescriptor {
            label: Some("pxtgi_linear"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            ..default()
        }),
        no_environment,
        scene_layout,
        io_layout,
        composite_layout,
    });
    commands.insert_resource(PxtgiGpu {
        primitives: storage_buffer(&device, "pxtgi_primitives", 0),
        nodes: storage_buffer(&device, "pxtgi_nodes", 0),
        shapes: storage_buffer(&device, "pxtgi_shapes", 0),
        // Zeroed at creation: every slot starts empty.
        cache_keys: storage_buffer(&device, "pxtgi_cache_keys", u64::from(CACHE_SLOTS) * 4),
        cells: storage_buffer(&device, "pxtgi_cache_cells", u64::from(CACHE_SLOTS) * 80),
        version: 0,
        shapes_version: 0,
        primitive_count: 0,
        node_count: 0,
    });
}

fn prepare_scene(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    scene: Res<PxtgiScene>,
    gpu: Option<ResMut<PxtgiGpu>>,
) {
    let Some(mut gpu) = gpu else {
        return;
    };
    let geometry = &scene.0.geometry;
    let gpu = &mut *gpu;
    let upload = |buffer: &mut Buffer, label, data: &[[f32; 4]]| {
        let data = bytes(data);
        if buffer.size() < data.len() as u64 {
            *buffer = storage_buffer(&device, label, (data.len() as u64).next_power_of_two());
        }
        queue.write_buffer(buffer, 0, &data);
    };
    if geometry.shapes_version != gpu.shapes_version {
        upload(&mut gpu.shapes, "pxtgi_shapes", &geometry.shapes);
        gpu.shapes_version = geometry.shapes_version;
    }
    if geometry.version != gpu.version {
        upload(&mut gpu.primitives, "pxtgi_primitives", &geometry.primitives);
        upload(&mut gpu.nodes, "pxtgi_nodes", &geometry.nodes);
        gpu.primitive_count = (geometry.primitives.len() / PRIMITIVE_VEC4S) as u32;
        gpu.node_count = (geometry.nodes.len() / 2) as u32;
        gpu.version = geometry.version;
    }
}

fn target(device: &RenderDevice, label: &'static str, size: UVec2, format: TextureFormat, usage: TextureUsages) -> Target {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor::default());
    Target { texture, view }
}

impl PxtgiView {
    fn new(device: &RenderDevice, size: UVec2, clip_from_world: Mat4) -> Self {
        // Diffuse traced at a quarter of the resolution, reflections at half, as in Godot.
        let trace_size = (size / 4).max(UVec2::ONE);
        let reflect_size = (size / 2).max(UVec2::ONE);
        let usage = TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
        let light = |label, size| target(device, label, size, TextureFormat::Rgba16Float, usage);
        let scalar = |label, size| target(device, label, size, TextureFormat::R32Float, usage);
        let pair = |make: &dyn Fn(&'static str, UVec2) -> Target, label, size| [make(label, size), make(label, size)];
        Self {
            size,
            trace_size,
            reflect_size,
            trace: light("pxtgi_trace", trace_size),
            accum: pair(&light, "pxtgi_accum", trace_size),
            accum_depth: pair(&scalar, "pxtgi_accum_depth", trace_size),
            moments: pair(&light, "pxtgi_moments", trace_size),
            variance: scalar("pxtgi_variance", trace_size),
            filtered: pair(&light, "pxtgi_filtered", trace_size),
            filtered_variance: pair(&scalar, "pxtgi_filtered_variance", trace_size),
            reflect_trace: light("pxtgi_reflect_trace", reflect_size),
            reflect_accum: pair(&light, "pxtgi_reflect_accum", reflect_size),
            reflect_depth: pair(&scalar, "pxtgi_reflect_depth", reflect_size),
            reflect_filtered: pair(&light, "pxtgi_reflect_filtered", reflect_size),
            prev_color: target(
                device,
                "pxtgi_prev_color",
                size,
                TextureFormat::Rgba16Float,
                TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            ),
            frame: device.create_buffer(&BufferDescriptor {
                label: Some("pxtgi_frame"),
                size: (FRAME_VEC4S * 16) as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            current: 0,
            history: false,
            prev_clip_from_world: clip_from_world,
        }
    }
}

// Matrices (12), scalars (9), reflections and environment (6), point and spot
// lights (40), emissives (80).
const FRAME_VEC4S: usize = 12 + 9 + 6 + MAX_LIGHTS * 5 + MAX_EMISSIVES * 5;

/// The camera's environment map as the shader takes it: strength (exposed),
/// roughest mip, and the world-to-map rotation.
struct Environment {
    strength: f32,
    roughest_mip: f32,
    rotation: Quat,
}

#[allow(clippy::too_many_arguments)]
fn frame_uniform(
    settings: &Pxtgi,
    scene: &SceneData,
    gpu: &PxtgiGpu,
    clip_from_world: Mat4,
    eye: Vec3,
    near: f32,
    view: &PxtgiView,
    exposure: f32,
    environment: Option<&Environment>,
    frame: u32,
) -> Vec<[f32; 4]> {
    let bits = |v: [u32; 4]| v.map(f32::from_bits);
    let flag = |on: bool| f32::from(u8::from(on));
    // Light as the GI holds it: irradiance / pi, exposed like the lit image.
    let lit = exposure / PI;
    let mut u = Vec::with_capacity(FRAME_VEC4S);
    for m in [clip_from_world, clip_from_world.inverse(), view.prev_clip_from_world] {
        u.extend(m.to_cols_array_2d());
    }
    u.push(eye.extend(near).to_array());
    u.push(bits([view.size.x, view.size.y, view.trace_size.x, view.trace_size.y]));
    u.push([
        (frame % 4096) as f32,
        settings.max_distance.max(0.1),
        settings.intensity.max(0.0),
        settings.debug as f32,
    ]);
    let (sun_direction, sun_color) = scene.sun.unwrap_or_default();
    u.push(sun_direction.extend(flag(scene.sun.is_some())).to_array());
    u.push((sun_color * lit).extend(exposure).to_array());
    let sky = settings.sky;
    u.push((Vec3::new(sky.red, sky.green, sky.blue) * exposure).extend(settings.world_ray_distance.max(0.1)).to_array());
    u.push(bits([gpu.primitive_count, gpu.node_count, scene.omnis.len() as u32, scene.spots.len() as u32]));
    u.push(bits([
        frame,
        CACHE_SLOTS - 1,
        u32::from(settings.multi_bounce),
        settings.rays_per_texel.clamp(1, 4),
    ]));
    u.push([scene.emissives.len() as f32, flag(scene.sources_changed), flag(view.history), 0.0]);
    let mode = match settings.reflections {
        PxtgiReflections::Off => 0.0,
        PxtgiReflections::ScreenSpace => 1.0,
        PxtgiReflections::Traced => 2.0,
    };
    u.push([mode, settings.reflection_roughness_cutoff.clamp(0.0, 1.0), 0.0, 0.0]);
    u.push(bits([view.reflect_size.x, view.reflect_size.y, 0, 0]));
    let (strength, roughest_mip, rotation) =
        environment.map_or((0.0, 0.0, Quat::IDENTITY), |e| (e.strength, e.roughest_mip, e.rotation));
    u.push([strength, roughest_mip, flag(environment.is_some()), 0.0]);
    // Bevy samples its environment map with the inverse of the light's rotation.
    let rows = Mat3::from_quat(rotation.inverse()).transpose();
    u.extend((0..3).map(|r| rows.col(r).extend(0.0).to_array()));
    let pad = |u: &mut Vec<[f32; 4]>, values: &mut dyn Iterator<Item = [f32; 4]>, len: usize| {
        let start = u.len();
        u.extend(values.take(len));
        u.resize(start + len, [0.0; 4]);
    };
    let omnis = &scene.omnis;
    pad(&mut u, &mut omnis.iter().map(|l| l.position.extend(l.inv_range2).to_array()), MAX_LIGHTS);
    pad(&mut u, &mut omnis.iter().map(|l| (l.color * lit).extend(0.0).to_array()), MAX_LIGHTS);
    let spots = &scene.spots;
    pad(&mut u, &mut spots.iter().map(|l| l.position.extend(l.inv_range2).to_array()), MAX_LIGHTS);
    pad(&mut u, &mut spots.iter().map(|l| l.direction.extend(l.cone_scale).to_array()), MAX_LIGHTS);
    pad(&mut u, &mut spots.iter().map(|l| (l.color * lit).extend(l.cone_offset).to_array()), MAX_LIGHTS);
    let emissives = &scene.emissives;
    pad(&mut u, &mut emissives.iter().map(|e| e.center.extend(0.0).to_array()), MAX_EMISSIVES);
    for axis in 0..3 {
        pad(&mut u, &mut emissives.iter().map(|e| e.axes[axis].extend(0.0).to_array()), MAX_EMISSIVES);
    }
    pad(&mut u, &mut emissives.iter().map(|e| (e.color * exposure).extend(0.0).to_array()), MAX_EMISSIVES);
    debug_assert_eq!(u.len(), FRAME_VEC4S);
    u
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn prepare_views(
    mut commands: Commands,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    cache: Res<PipelineCache>,
    pipelines: Option<Res<PxtgiPipelines>>,
    gpu: Option<Res<PxtgiGpu>>,
    scene: Res<PxtgiScene>,
    images: Res<RenderAssets<GpuImage>>,
    mut frame: Local<u32>,
    mut views: Query<(
        Entity,
        &ExtractedView,
        &ExtractedCamera,
        &ViewTarget,
        &ViewPrepassTextures,
        &Pxtgi,
        Option<&PxtgiEnvironment>,
        Option<&mut PxtgiView>,
    )>,
) {
    let (Some(pipelines), Some(gpu)) = (pipelines, gpu) else {
        return;
    };
    *frame = frame.wrapping_add(1);
    for (entity, view, camera, target, prepass, settings, environment, existing) in &mut views {
        let (Some(size), Some(depth), Some(gbuffer)) =
            (camera.physical_viewport_size, prepass.depth_only_view(), prepass.deferred_view())
        else {
            commands.entity(entity).remove::<PxtgiBindGroups>();
            continue;
        };
        let main_size = target.main_texture().size();
        // shortcut: the history copy assumes the viewport covers the whole
        // target; offset viewports (split screen) would copy the wrong region.
        if view.clip_from_view.w_axis.w == 1.0
            || settings.intensity <= 0.0
            || target.main_texture_format() != TextureFormat::Rgba16Float
            || UVec2::new(main_size.width, main_size.height) != size
        {
            warn_once!("PXTGI needs an HDR perspective camera whose viewport fills its target");
            commands.entity(entity).remove::<PxtgiBindGroups>();
            continue;
        }
        // A loaded cube map only; until then (or for a 2D one) the sky colour.
        let environment = environment.and_then(|environment| {
            let image = images.get(environment.specular)?;
            let cube = image
                .texture_view_descriptor
                .as_ref()
                .is_some_and(|d| d.dimension == Some(TextureViewDimension::Cube));
            cube.then(|| {
                let settings = Environment {
                    strength: environment.intensity * camera.exposure,
                    roughest_mip: image.texture_descriptor.mip_level_count.saturating_sub(1) as f32,
                    rotation: environment.rotation,
                };
                (settings, &image.texture_view)
            })
        });
        let clip_from_world = view
            .clip_from_world
            .unwrap_or_else(|| view.clip_from_view * view.world_from_view.to_matrix().inverse());
        let mut fresh = None;
        let state = match existing {
            Some(state) if state.size == size => state.into_inner(),
            _ => fresh.insert(PxtgiView::new(&device, size, clip_from_world)),
        };
        let prev = state.current;
        state.current = 1 - prev;
        let cur = state.current;
        let uniform = frame_uniform(
            settings,
            &scene.0,
            &gpu,
            clip_from_world,
            view.world_from_view.translation(),
            // Infinite reverse-Z perspective keeps the near plane here.
            view.clip_from_view.w_axis.z,
            state,
            camera.exposure,
            environment.as_ref().map(|(settings, _)| settings),
            *frame,
        );
        queue.write_buffer(&state.frame, 0, &bytes(&uniform));
        state.prev_clip_from_world = clip_from_world;
        state.history = true;

        let scene_group = device.create_bind_group(
            "pxtgi_scene",
            &cache.get_bind_group_layout(&pipelines.scene_layout),
            &BindGroupEntries::sequential((
                state.frame.as_entire_binding(),
                depth,
                gbuffer,
                &state.prev_color.view,
                &pipelines.sampler,
                gpu.primitives.as_entire_binding(),
                gpu.nodes.as_entire_binding(),
                gpu.cache_keys.as_entire_binding(),
                gpu.cells.as_entire_binding(),
                gpu.shapes.as_entire_binding(),
                environment.map_or(&pipelines.no_environment, |(_, view)| view),
            )),
        );
        let io_layout = cache.get_bind_group_layout(&pipelines.io_layout);
        let io = |ins: [&Target; 4], outs: [&Target; 4]| {
            device.create_bind_group(
                "pxtgi_io",
                &io_layout,
                &BindGroupEntries::sequential((
                    &ins[0].view,
                    &ins[1].view,
                    &ins[2].view,
                    &ins[3].view,
                    &outs[0].view,
                    &outs[1].view,
                    &outs[2].view,
                    &outs[3].view,
                )),
            )
        };
        let s = &*state;
        // Unused slots hold textures the pass neither reads nor writes, so no
        // texture is bound for reading and writing in one dispatch.
        let atrous = |light: &Target, variance: &Target, i: usize| {
            io(
                [light, variance, &s.accum_depth[cur], &s.trace],
                [&s.filtered[i], &s.filtered_variance[i], &s.moments[prev], &s.accum_depth[prev]],
            )
        };
        let diffuse = [
            io(
                [&s.accum[prev], &s.accum_depth[prev], &s.moments[prev], &s.filtered_variance[1]],
                [&s.trace, &s.variance, &s.filtered[0], &s.filtered_variance[0]],
            ),
            io(
                [&s.trace, &s.accum[prev], &s.accum_depth[prev], &s.moments[prev]],
                [&s.accum[cur], &s.accum_depth[cur], &s.moments[cur], &s.variance],
            ),
            atrous(&s.accum[cur], &s.variance, 0),
            atrous(&s.filtered[0], &s.filtered_variance[0], 1),
            atrous(&s.filtered[1], &s.filtered_variance[1], 0),
        ];
        // The diffuse passes are done by now: their intermediates fill the unused slots.
        let spare = [&s.variance, &s.filtered[1], &s.filtered_variance[1]];
        let reflect = [
            io(
                [&s.accum[prev], &s.accum_depth[prev], &s.moments[prev], &s.trace],
                [&s.reflect_trace, spare[0], spare[1], spare[2]],
            ),
            io(
                [&s.reflect_trace, &s.reflect_accum[prev], &s.reflect_depth[prev], &s.trace],
                [&s.reflect_accum[cur], &s.reflect_depth[cur], spare[1], spare[2]],
            ),
            io(
                [&s.reflect_accum[cur], &s.reflect_depth[cur], &s.trace, &s.trace],
                [&s.reflect_filtered[0], spare[0], spare[1], spare[2]],
            ),
            io(
                [&s.reflect_filtered[0], &s.reflect_depth[cur], &s.trace, &s.trace],
                [&s.reflect_filtered[1], spare[0], spare[1], spare[2]],
            ),
        ];
        let composite = device.create_bind_group(
            "pxtgi_composite",
            &cache.get_bind_group_layout(&pipelines.composite_layout),
            &BindGroupEntries::with_indices((
                (8, &s.filtered[0].view),
                (9, &s.accum_depth[cur].view),
                (10, &s.reflect_filtered[1].view),
                (11, &s.reflect_depth[cur].view),
            )),
        );
        let mut entity_commands = commands.entity(entity);
        entity_commands.insert(PxtgiBindGroups {
            scene: scene_group,
            diffuse,
            reflect,
            composite,
        });
        if let Some(state) = fresh {
            entity_commands.insert(state);
        }
    }
}

fn render(
    view: ViewQuery<(&ViewTarget, &Pxtgi, &PxtgiView, &PxtgiBindGroups)>,
    pipelines: Res<PxtgiPipelines>,
    gpu: Res<PxtgiGpu>,
    cache: Res<PipelineCache>,
    mut context: RenderContext,
) {
    let (target, settings, state, groups) = view.into_inner();
    let compute = |id| cache.get_compute_pipeline(id);
    let diffuse: Option<Vec<_>> = pipelines.diffuse.iter().map(|&id| compute(id)).collect();
    let reflect: Option<Vec<_>> = pipelines.reflect.iter().map(|&id| compute(id)).collect();
    let composite = cache.get_render_pipeline(if matches!(settings.debug, 1 | 2 | 4 | 5) {
        pipelines.composite_debug
    } else {
        pipelines.composite
    });
    let (Some(cache_update), Some(diffuse), Some(reflect), Some(composite)) =
        (compute(pipelines.cache), diffuse, reflect, composite)
    else {
        return;
    };
    let groups_for = |size: UVec2| (size.x.div_ceil(8), size.y.div_ceil(8));
    let encoder = context.command_encoder();
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("pxtgi"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &groups.scene, &[]);
        if settings.multi_bounce && gpu.primitive_count > 0 {
            pass.set_pipeline(cache_update);
            // Any io group: the cache pass reads none of it.
            pass.set_bind_group(1, &groups.diffuse[0], &[]);
            pass.dispatch_workgroups(CACHE_ROW / 8, CACHE_SLOTS / CACHE_ROW / 4 / 8, 1);
        }
        let (x, y) = groups_for(state.trace_size);
        for (pipeline, group) in diffuse.into_iter().zip(&groups.diffuse) {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(1, group, &[]);
            pass.dispatch_workgroups(x, y, 1);
        }
        if settings.reflections != PxtgiReflections::Off {
            let (x, y) = groups_for(state.reflect_size);
            for (pipeline, group) in reflect.into_iter().zip(&groups.reflect) {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(1, group, &[]);
                pass.dispatch_workgroups(x, y, 1);
            }
        }
    }
    {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("pxtgi_composite"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: target.main_texture_view(),
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                },
            })],
            ..default()
        });
        pass.set_pipeline(composite);
        pass.set_bind_group(0, &groups.scene, &[]);
        pass.set_bind_group(1, &groups.composite, &[]);
        pass.draw(0..3, 0..1);
    }
    // Next frame's screen rays read this frame's lit image.
    encoder.copy_texture_to_texture(
        target.main_texture().as_image_copy(),
        state.prev_color.texture.as_image_copy(),
        Extent3d {
            width: state.size.x,
            height: state.size.y,
            depth_or_array_layers: 1,
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(entity: u32, collider: &Collider, at: Vec3, store: &mut ShapeStore) -> Item {
        let (shape, lower, upper) = store.shape(collider);
        let mut item = Item {
            entity: Entity::from_raw_u32(entity).unwrap(),
            scaled: false,
            offset: Vec3::ZERO,
            to_world: Affine3A::IDENTITY,
            shape,
            lower,
            upper,
            albedo: LinearRgba::WHITE,
            emission: LinearRgba::BLACK,
        };
        item.place(&GlobalTransform::from(Transform::from_translation(at)));
        item
    }

    // Every record sits in exactly one leaf, inside that leaf's box.
    fn assert_tree_covers(geometry: &Geometry, items: &[Item]) {
        let mut seen = 0;
        for node in geometry.nodes.chunks(2) {
            let count = node[1][3].to_bits() as usize;
            if count == 0 {
                continue;
            }
            let first = node[0][3].to_bits() as usize;
            let (lo, hi) = (Vec3::from_slice(&node[0]), Vec3::from_slice(&node[1]));
            for item in &items[first..first + count] {
                let (a, b) = item.world_bounds();
                assert!(a.cmpge(lo - 1e-4).all() && b.cmple(hi + 1e-4).all());
            }
            seen += count;
        }
        assert_eq!(seen, items.len());
    }

    #[test]
    fn tree_covers_every_collider_before_and_after_a_refit() {
        let mut store = ShapeStore::default();
        let shapes = [
            Collider::Cuboid { half_extents: Vec3::splat(0.5) },
            Collider::Sphere { radius: 0.4 },
            Collider::Capsule { radius: 0.2, half_height: 0.3 },
        ];
        let mut items: Vec<_> = (0..40)
            .map(|i| item(i + 1, &shapes[i as usize % 3], Vec3::new(i as f32, (i % 5) as f32, 0.0), &mut store))
            .collect();
        let mut geometry = build_geometry(&mut items, 1);
        assert_eq!(geometry.primitives.len(), 40 * PRIMITIVE_VEC4S);
        assert_tree_covers(&geometry, &items);
        // Move one body far away and refit: its leaf and every ancestor grow.
        items[7].place(&GlobalTransform::from(Transform::from_xyz(0.0, 100.0, 0.0)));
        let bounds: Vec<_> = items.iter().map(Item::world_bounds).collect();
        refit_nodes(&mut geometry.nodes, &bounds);
        assert_tree_covers(&geometry, &items);
        assert!(geometry.nodes[1][1] >= 100.0, "root reaches the moved body");
    }

    #[test]
    #[ignore = "requires slangc (or SLANGC) on PATH"]
    fn slang_compiles_to_valid_wgsl_with_every_entry_point() {
        let shader = compile_shader(&SlangCompiler::default()).expect("pxtgi.slang");
        let bevy::shader::Source::Wgsl(source) = shader.source else {
            panic!("expected Slang's WGSL output");
        };
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty())
            .validate(&module)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
        for entry in ENTRY_POINTS {
            assert!(module.entry_points.iter().any(|point| point.name == entry), "{entry}");
        }
    }

    #[test]
    fn hulls_and_meshes_are_stored_once_and_offset_into_the_shape_buffer() {
        let corners: Vec<_> = (0..8)
            .map(|i| Vec3::new((i & 1) as f32, ((i >> 1) & 1) as f32, (i >> 2) as f32) - 0.5)
            .collect();
        let hull = Collider::ConvexHull(bevy_box3d::ConvexHull::new(&corners).unwrap());
        let mesh = Collider::TriMesh(
            bevy_box3d::TriMesh::new(&corners[..4], &[[0, 1, 3], [0, 3, 2]]).unwrap(),
        );
        let mut store = ShapeStore::default();
        let (a, ..) = store.shape(&hull);
        let (b, ..) = store.shape(&hull.clone());
        assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));
        assert_eq!(a[2].to_bits(), 6, "a cube has six planes");
        let (m, ..) = store.shape(&mesh);
        assert_eq!(m[0], 5.0);
        assert_eq!(m[1].to_bits(), 6, "the mesh follows the hull's planes");
        assert_eq!(store.index.len(), 2);
    }
}

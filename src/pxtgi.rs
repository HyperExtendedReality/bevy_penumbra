//! PXTGI ([`crate::GiMethod::Pxtgi`]): global illumination traced against the
//! physics colliders.
//!
//! A port of PXTGI from the godot_physx module
//! (`modules/godot_physx/nodes/pxtgi_3d.cpp`, `physx_gi_effect.cpp`). Each
//! visible surface sends a few rays per frame. A ray first follows what is on
//! screen, taking the light that point had in last frame's image; where it
//! leaves the screen or passes behind what the screen shows it continues
//! against the [`bevy_box3d::Collider`]s (boxes, spheres, capsules, convex
//! hulls and triangle meshes), which stand in for the geometry. Hits are lit by
//! the camera's ambient light, the sun, point and spot lights (shadowed by the
//! colliders) and by a hashed world-space radiance cache that gives multiple
//! bounces; rays leaving the scene see the camera's `EnvironmentMapLight`,
//! else its `Skybox`, else [`crate::RadianceCascades::environment`]. The
//! result is accumulated over frames, filtered along depth and normals,
//! upsampled, and replaces Bevy's ambient light (and environment-map diffuse)
//! by the fraction of rays that hit, times each pixel's albedo. Smooth
//! surfaces also get traced reflections, which take over from Bevy's own
//! reflection where they find something; Bevy reflection probes take over
//! inside their boxes, or serve as capture sources in
//! [`crate::PxtgiReflections::TracedAndProbes`].
//!
//! Installed by [`crate::RadianceCascadesPlugin`]; runs on cameras whose
//! [`crate::RadianceCascades::method`] is `Pxtgi`. A body's colour comes from
//! [`PxtgiLook`] on it, else its own `MeshMaterial3d<StandardMaterial>`, else
//! its child meshes' (albedo from the first, emission from the brightest; a
//! child's own `PxtgiLook` wins); otherwise it bounces as neutral grey. A
//! material's albedo is its base colour times its base-colour texture's mean,
//! and its emission its emissive colour times its emissive texture's mean.
//! Emissive `StandardMaterial` meshes outside any collider body give off light
//! as their bounding box.

use std::{
    collections::{HashMap, HashSet, hash_map::DefaultHasher},
    f32::consts::PI,
    hash::{Hash, Hasher},
    sync::Arc,
};

use bevy::{
    camera::primitives::Aabb,
    core_pipeline::{
        Core3dSystems, FullscreenShader,
        core_3d::{main_opaque_pass_3d, main_transparent_pass_3d},
        prepass::ViewPrepassTextures,
        schedule::Core3d,
    },
    image::Image,
    light::{EnvironmentMapLight, LightProbe, ParallaxCorrection, Skybox},
    math::Affine3A,
    prelude::*,
    render::{
        Extract, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
        camera::ExtractedCamera,
        diagnostic::RecordDiagnostics,
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
use bevy_box3d::{Collider, RigidBody};
use bevy_slang::{SlangCompiler, SlangLoaderError, SlangSettings, SlangTarget};

use crate::{
    GiMethod, PxtgiDebug, PxtgiReflections, PxtgiSettings, RadianceCascades,
    RadianceCascadesRawOutput, RadianceCascadesStatus,
};

/// Radiance-cache slots shared by every PXTGI view (80 bytes each).
const CACHE_SLOTS: u32 = 1 << 17;
// The cache pass walks the slots as a 512-wide grid, a quarter of them a frame.
const CACHE_ROW: u32 = 512;
const PRIMITIVE_VEC4S: usize = 6;
const MAX_LIGHTS: usize = 8;
const MAX_EMISSIVES: usize = 16;
// Frames the cache follows quickly after the light sources change.
const FRESH_FRAMES: u32 = 60;
// Frames of motion after which the tree is rebuilt: a refit keeps the last
// build's grouping, and as a pile tumbles its boxes loosen and every ray
// walks more of the tree (a quarter second at 60 Hz, as Godot's).
const MOVES_PER_BUILD: u32 = 15;
// Frames a pending rescan (bodies added, removed or reshaped) waits, so a
// stream of spawned debris doesn't rebuild the tree every frame.
const RESCAN_FRAMES: u32 = 10;
// Reflection probes PXTGI knows of, the nearest to the camera.
const MAX_PROBES: usize = 8;

const SOURCE: &str = include_str!("../shaders/pxtgi.slang");
const ENTRY_POINTS: [&str; 12] = [
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
    "raw_output",
];

/// Installs PXTGI; called by [`crate::RadianceCascadesPlugin`]. The shader is
/// Slang, compiled to WGSL in the background at startup; PXTGI starts drawing
/// once that finishes.
pub(crate) fn install(app: &mut App, compiler: SlangCompiler, status: RadianceCascadesStatus) {
    if app.get_sub_app(RenderApp).is_none() {
        return;
    }
    // Pipelines are queued with this handle and wait until compilation fills it.
    let shader = app.world().resource::<Assets<Shader>>().reserve_handle();
    app.init_resource::<PxtgiScene>()
        .insert_resource(PxtgiCompilation {
            compiler,
            handle: shader.clone(),
            task: None,
            started: false,
            status,
        })
        .add_plugins(ExtractResourcePlugin::<PxtgiScene>::default())
        .add_systems(Update, compile_pxtgi_shader)
        .add_systems(
            PostUpdate,
            update_scene
                .after(TransformSystems::Propagate)
                .after(bevy::asset::AssetEventSystems),
        );
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .insert_resource(PxtgiShader(shader))
        .add_systems(ExtractSchedule, extract_view_sources)
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
    status: RadianceCascadesStatus,
}

fn compile_pxtgi_shader(
    mut compilation: ResMut<PxtgiCompilation>,
    mut shaders: ResMut<Assets<Shader>>,
    cameras: Query<&RadianceCascades>,
) {
    // Compiled the first time a view asks for PXTGI.
    if !compilation.started {
        if !cameras.iter().any(|settings| settings.pxtgi_active()) {
            return;
        }
        compilation.started = true;
        let compiler = compilation.compiler.clone();
        compilation.task =
            Some(AsyncComputeTaskPool::get().spawn(async move { compile_shader(&compiler) }));
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
        Err(error) => {
            compilation
                .status
                .record_failure(format!("PXTGI shader compilation failed: {error}"));
            error!(%error, "PXTGI shader compilation failed; PXTGI stays off");
        }
    }
}

/// Overrides the albedo and emitted radiance PXTGI gives a collider's body,
/// or a mesh under one: for bodies whose material isn't a `StandardMaterial`,
/// or whose colour comes from per-instance data.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct PxtgiLook {
    /// Diffuse albedo.
    pub albedo: LinearRgba,
    /// Emitted radiance, as `StandardMaterial::emissive`.
    pub emission: LinearRgba,
}

/// Which light a Bevy reflection probe (a `LightProbe` with an
/// `EnvironmentMapLight`) takes over from PXTGI inside its core (its box less
/// its falloff and a quarter metre). Without this component a probe takes
/// over both, as a Godot `ReflectionProbe` does by default.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct PxtgiProbe {
    /// Leave the diffuse light inside the probe to Bevy's probe.
    pub overrides_ambient: bool,
    /// Leave reflections inside the probe to Bevy's probe.
    pub overrides_reflections: bool,
}

impl Default for PxtgiProbe {
    fn default() -> Self {
        Self {
            overrides_ambient: true,
            overrides_reflections: true,
        }
    }
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
    // Bevy's `emissive_exposure_weight`.
    exposure_weight: f32,
}

/// One Bevy reflection probe: a `LightProbe` with an `EnvironmentMapLight`.
#[derive(Clone)]
struct Probe {
    // World to the probe's unit cube [-0.5, 0.5] (its transform, scale and all).
    to_local: Affine3A,
    // The unit cube's axes in world space.
    to_world_dir: Mat3,
    // Parallax-correction half extents in the unit cube, when Bevy box-projects it.
    parallax: Option<Vec3>,
    // Where it takes over from PXTGI, in the unit cube.
    core: Vec3,
    falloff: Vec3,
    volume: f32,
    capture: Vec3,
    specular: AssetId<Image>,
    intensity: f32,
    rotation: Quat,
    overrides: u32,
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
    probes: Vec<Probe>,
    sources_changed: bool,
    // Bumped whenever the light sources change; the geometry's version
    // stands for the scene's revision in `RadianceCascadesStatus`.
    lighting_revision: u64,
}

#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
struct PxtgiScene(Arc<SceneData>);

#[derive(Resource)]
struct PxtgiShader(Handle<Shader>);

/// A cube map PXTGI samples: its specular map, strength (cd/m²) and rotation.
#[derive(Clone, Copy)]
struct CubeSource {
    specular: AssetId<Image>,
    intensity: f32,
    rotation: Quat,
}

/// What a PXTGI camera's rays see beyond the colliders and its ambient light.
#[derive(Component)]
struct PxtgiViewSources {
    // The environment map, else the skybox (which lights nothing in Bevy).
    background: Option<CubeSource>,
    // Whether that background is an `EnvironmentMapLight`, lighting the scene.
    environment_lights: bool,
    // The camera's ambient light, linear radiance (cd/m²), not yet exposed.
    ambient: Vec3,
}

// ---------------------------------------------------------------- main world

/// What a body bounces and gives off: its albedo, emitted radiance and Bevy's
/// `emissive_exposure_weight` for that emission.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Look {
    albedo: LinearRgba,
    emission: LinearRgba,
    exposure_weight: f32,
}

/// A body PXTGI can't read a material for bounces as neutral grey.
const GREY: Look = Look {
    albedo: LinearRgba::rgb(0.5, 0.5, 0.5),
    emission: LinearRgba::BLACK,
    exposure_weight: 0.0,
};

fn luminance(c: LinearRgba) -> f32 {
    0.2126 * c.red + 0.7152 * c.green + 0.0722 * c.blue
}

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
    look: Look,
    // A static body: a reflection probe's capture may have seen it.
    fixed: bool,
}

impl Item {
    fn placement(&self, transform: &GlobalTransform) -> Affine3A {
        let base = if self.scaled {
            transform.affine()
        } else {
            let (_, rotation, translation) = transform.to_scale_rotation_translation();
            Affine3A::from_rotation_translation(rotation, translation)
        };
        base * Affine3A::from_translation(self.offset)
    }

    fn place(&mut self, transform: &GlobalTransform) {
        self.to_world = self.placement(transform);
    }

    fn world_bounds(&self) -> (Vec3, Vec3) {
        let center = self
            .to_world
            .transform_point3((self.lower + self.upper) * 0.5);
        let half = (self.upper - self.lower) * 0.5;
        let m = self.to_world.matrix3;
        let extent =
            Vec3::from(m.x_axis.abs() * half.x + m.y_axis.abs() * half.y + m.z_axis.abs() * half.z);
        (center - extent, center + extent)
    }

    fn rows(&self) -> [[f32; 4]; 3] {
        let to_local = self.to_world.inverse();
        let rows = Mat3::from(to_local.matrix3).transpose();
        std::array::from_fn(|r| rows.col(r).extend(to_local.translation[r]).to_array())
    }

    /// The albedo (w: emission's exposure weight) and emission (w: static) rows.
    fn look_rows(&self) -> [[f32; 4]; 2] {
        let (a, e) = (self.look.albedo, self.look.emission);
        [
            [a.red, a.green, a.blue, self.look.exposure_weight],
            [e.red, e.green, e.blue, f32::from(u8::from(self.fixed))],
        ]
    }

    fn record(&self) -> [[f32; 4]; PRIMITIVE_VEC4S] {
        let [r0, r1, r2] = self.rows();
        let [albedo, emission] = self.look_rows();
        [r0, r1, r2, self.shape, albedo, emission]
    }

    fn emissive(&self) -> Option<Emissive> {
        let e = self.look.emission;
        let color = Vec3::new(e.red, e.green, e.blue);
        (color.max_element() > 0.0).then(|| {
            let half = (self.upper - self.lower) * 0.5;
            let m = self.to_world.matrix3;
            Emissive {
                center: self
                    .to_world
                    .transform_point3((self.lower + self.upper) * 0.5),
                axes: [
                    Vec3::from(m.x_axis) * half.x,
                    Vec3::from(m.y_axis) * half.y,
                    Vec3::from(m.z_axis) * half.z,
                ],
                color,
                exposure_weight: self.look.exposure_weight,
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
            Collider::Cuboid { half_extents } => (
                [0.0, half_extents.x, half_extents.y, half_extents.z],
                -*half_extents,
                *half_extents,
            ),
            Collider::Sphere { radius } => (
                [1.0, *radius, 0.0, 0.0],
                Vec3::splat(-radius),
                Vec3::splat(*radius),
            ),
            Collider::Capsule {
                radius,
                half_height,
            } => {
                let half = Vec3::new(*radius, half_height + radius, *radius);
                ([2.0, *radius, half.y, 0.0], -half, half)
            }
            Collider::ConvexHull(hull) => {
                let planes = hull.planes();
                let at = *self
                    .index
                    .entry(planes.as_ptr() as usize)
                    .or_insert_with(|| {
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
                let at = *self
                    .index
                    .entry(mesh.nodes().as_ptr() as usize)
                    .or_insert_with(|| {
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

/// Reads looks from `StandardMaterial`s, keeping each texture's mean colour.
struct LookReader<'a> {
    materials: &'a Assets<StandardMaterial>,
    images: &'a Assets<Image>,
    means: &'a mut HashMap<AssetId<Image>, Vec3>,
    // Textures not readable on the CPU yet: their looks are read again once they load.
    missing: &'a mut HashSet<AssetId<Image>>,
}

impl LookReader<'_> {
    /// A texture's mean linear colour over a 32 x 32 grid of texels, as Godot
    /// takes it; white while it isn't readable (not loaded, compressed, or
    /// kept on the GPU only).
    fn mean(&mut self, texture: &Handle<Image>) -> Vec3 {
        let id = texture.id();
        if let Some(mean) = self.means.get(&id) {
            return *mean;
        }
        let Some(image) = self.images.get(id) else {
            self.missing.insert(id);
            return Vec3::ONE;
        };
        let (w, h) = (image.width().max(1), image.height().max(1));
        let mut sum = Vec3::ZERO;
        for y in 0..32 {
            for x in 0..32 {
                let Ok(color) = image.get_color_at((x * 2 + 1) * w / 64, (y * 2 + 1) * h / 64)
                else {
                    // Unreadable formats keep the material's own colour.
                    self.means.insert(id, Vec3::ONE);
                    return Vec3::ONE;
                };
                let c = color.to_linear();
                sum += Vec3::new(c.red, c.green, c.blue);
            }
        }
        let mean = sum / 1024.0;
        self.means.insert(id, mean);
        mean
    }

    /// A material's look: its base colour times its base-colour texture's
    /// mean, its emission times its emissive texture's mean.
    fn material(&mut self, id: AssetId<StandardMaterial>) -> Option<Look> {
        let material = self.materials.get(id)?;
        let base = material.base_color.to_linear();
        let albedo_mean = material
            .base_color_texture
            .as_ref()
            .map_or(Vec3::ONE, |t| self.mean(t));
        let emission_mean = material
            .emissive_texture
            .as_ref()
            .map_or(Vec3::ONE, |t| self.mean(t));
        let e = material.emissive;
        Some(Look {
            albedo: LinearRgba::rgb(
                base.red * albedo_mean.x,
                base.green * albedo_mean.y,
                base.blue * albedo_mean.z,
            ),
            emission: LinearRgba::rgb(
                e.red * emission_mean.x,
                e.green * emission_mean.y,
                e.blue * emission_mean.z,
            ),
            exposure_weight: material.emissive_exposure_weight,
        })
    }
}

#[allow(clippy::type_complexity)]
/// A body's look, as Godot's `body_look`: its own [`PxtgiLook`], else its own
/// material, else its child meshes' (the first one's albedo, the brightest
/// one's emission, a child's own `PxtgiLook` before its material); grey when
/// none is readable. Also the materials it was read from.
fn body_look(
    reader: &mut LookReader,
    own: Option<&PxtgiLook>,
    material: Option<&MeshMaterial3d<StandardMaterial>>,
    children: Option<&Children>,
    child_looks: &Query<
        (
            Option<&PxtgiLook>,
            Option<&MeshMaterial3d<StandardMaterial>>,
        ),
        Without<Collider>,
    >,
) -> (Look, Vec<AssetId<StandardMaterial>>) {
    let override_look = |look: &PxtgiLook| Look {
        albedo: look.albedo,
        emission: look.emission,
        exposure_weight: 0.0,
    };
    if let Some(look) = own {
        return (override_look(look), Vec::new());
    }
    if let Some(material) = material {
        let id = material.id();
        return (reader.material(id).unwrap_or(GREY), vec![id]);
    }
    let mut used = Vec::new();
    let mut found: Option<Look> = None;
    for child in children.into_iter().flatten() {
        let Ok((look, material)) = child_looks.get(*child) else {
            continue;
        };
        let look = match (look, material) {
            (Some(look), _) => override_look(look),
            (None, Some(material)) => {
                used.push(material.id());
                let Some(look) = reader.material(material.id()) else {
                    continue;
                };
                look
            }
            (None, None) => continue,
        };
        found = Some(match found {
            None => look,
            Some(first) if luminance(look.emission) > luminance(first.emission) => Look {
                emission: look.emission,
                exposure_weight: look.exposure_weight,
                ..first
            },
            Some(first) => first,
        });
    }
    (found.unwrap_or(GREY), used)
}

#[derive(Default)]
struct SceneState {
    geometry: Arc<Geometry>,
    store: ShapeStore,
    // Items in record (leaf) order, their world bounds, and where each entity's is.
    items: Vec<Item>,
    bounds: Vec<(Vec3, Vec3)>,
    slots: HashMap<Entity, usize>,
    // Which items read their look from each material.
    material_users: HashMap<AssetId<StandardMaterial>, Vec<Entity>>,
    texture_means: HashMap<AssetId<Image>, Vec3>,
    missing_textures: HashSet<AssetId<Image>>,
    // A camera ran PXTGI last frame: the scene was kept up to date.
    active: bool,
    // Bodies came, went or were reshaped since the last build.
    pending_rescan: bool,
    frames_since_rescan: u32,
    moves_since_build: u32,
    sources_hash: u64,
    fresh_frames: u32,
    lighting_revision: u64,
}

type LightQueries<'w, 's> = (
    Query<
        'w,
        's,
        (
            &'static DirectionalLight,
            &'static GlobalTransform,
            &'static InheritedVisibility,
        ),
    >,
    Query<
        'w,
        's,
        (
            &'static PointLight,
            &'static GlobalTransform,
            &'static InheritedVisibility,
        ),
    >,
    Query<
        'w,
        's,
        (
            &'static SpotLight,
            &'static GlobalTransform,
            &'static InheritedVisibility,
        ),
    >,
);

type ProbeQuery<'w, 's> = Query<
    'w,
    's,
    (
        &'static LightProbe,
        &'static EnvironmentMapLight,
        &'static GlobalTransform,
        Option<&'static ParallaxCorrection>,
        Option<&'static PxtgiProbe>,
        &'static InheritedVisibility,
    ),
>;

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

type ColliderQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static Collider,
        &'static GlobalTransform,
        Option<&'static RigidBody>,
        Option<&'static PxtgiLook>,
        Option<&'static MeshMaterial3d<StandardMaterial>>,
        Option<&'static Children>,
    ),
>;

/// Bodies whose shape or presence changed, and those whose look may have.
type ChangeQueries<'w, 's> = (
    Query<'w, 's, (), (With<Collider>, Changed<Collider>)>,
    Query<
        'w,
        's,
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
    Query<
        'w,
        's,
        Entity,
        (
            Or<(With<Collider>, With<ChildOf>)>,
            Or<(
                Changed<PxtgiLook>,
                Changed<MeshMaterial3d<StandardMaterial>>,
                Changed<RigidBody>,
            )>,
        ),
    >,
);

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn update_scene(
    mut scene: ResMut<PxtgiScene>,
    mut state: Local<SceneState>,
    cameras: Query<(&Camera, &GlobalTransform, &RadianceCascades)>,
    colliders: ColliderQuery,
    proxies: ProxyQuery,
    has_collider: Query<(), With<Collider>>,
    parents: Query<&ChildOf>,
    child_looks: Query<
        (
            Option<&PxtgiLook>,
            Option<&MeshMaterial3d<StandardMaterial>>,
        ),
        Without<Collider>,
    >,
    changes: ChangeQueries,
    moved: Query<
        (Entity, &GlobalTransform),
        (Or<(With<Collider>, With<Mesh3d>)>, Changed<GlobalTransform>),
    >,
    mut removed: (RemovedComponents<Collider>, RemovedComponents<Mesh3d>),
    materials: Res<Assets<StandardMaterial>>,
    images: Res<Assets<Image>>,
    mut events: (
        MessageReader<AssetEvent<StandardMaterial>>,
        MessageReader<AssetEvent<Image>>,
    ),
    lights: LightQueries,
    probes: ProbeQuery,
) {
    let removed = removed.0.read().count() + removed.1.read().count() > 0;
    let mut material_changed: HashSet<AssetId<StandardMaterial>> = HashSet::new();
    for event in events.0.read() {
        if let AssetEvent::Modified { id }
        | AssetEvent::LoadedWithDependencies { id }
        | AssetEvent::Added { id } = event
        {
            material_changed.insert(*id);
        }
    }
    let mut textures_changed = false;
    for event in events.1.read() {
        if let AssetEvent::Modified { id }
        | AssetEvent::LoadedWithDependencies { id }
        | AssetEvent::Added { id } = event
        {
            let known = state.texture_means.remove(id).is_some();
            textures_changed |= known || state.missing_textures.remove(id);
        }
    }
    // shortcut: lights, emissives and probes are chosen for the first PXTGI
    // camera; per-view selection if split screen ever needs it.
    let Some(eye) = cameras
        .iter()
        .find(|(camera, _, settings)| camera.is_active && settings.pxtgi_active())
        .map(|(_, transform, _)| transform.translation())
    else {
        // Nothing is kept up to date while no camera runs PXTGI.
        state.active = false;
        return;
    };
    let state = &mut *state;
    let woke = !state.active;
    state.active = true;

    // Bodies that came, went or were reshaped: found again at most every
    // RESCAN_FRAMES frames.
    state.pending_rescan |= removed || !changes.0.is_empty() || !changes.1.is_empty();
    if state.pending_rescan {
        state.frames_since_rescan += 1;
    }
    let tracked_moves: Vec<_> = moved
        .iter()
        .filter_map(|(entity, transform)| {
            let item = &state.items[*state.slots.get(&entity)?];
            let placement = item.placement(transform);
            (placement != item.to_world).then_some((entity, placement))
        })
        .collect();
    if !tracked_moves.is_empty() {
        state.moves_since_build += 1;
    }
    let rebuild = woke
        || state.geometry.version == 0
        || (state.pending_rescan && state.frames_since_rescan >= RESCAN_FRAMES)
        || state.moves_since_build > MOVES_PER_BUILD;
    let mut reader = LookReader {
        materials: &materials,
        images: &images,
        means: &mut state.texture_means,
        missing: &mut state.missing_textures,
    };
    if rebuild {
        // Shared shape data no live collider uses is dropped once it is most of the store.
        let live: HashSet<usize> = colliders.iter().filter_map(|c| shape_key(c.1)).collect();
        if state.store.index.len() > 2 * live.len().max(8) {
            let version = state.store.version + 1;
            state.store = ShapeStore {
                version,
                ..default()
            };
        }
        let mut items = Vec::new();
        let mut material_users: HashMap<AssetId<StandardMaterial>, Vec<Entity>> = HashMap::new();
        for (entity, collider, transform, body, own, material, children) in &colliders {
            let (look, used) = body_look(&mut reader, own, material, children, &child_looks);
            for id in used {
                material_users.entry(id).or_default().push(entity);
            }
            let (shape, lower, upper) = state.store.shape(collider);
            let mut item = Item {
                entity,
                scaled: false,
                offset: Vec3::ZERO,
                to_world: Affine3A::IDENTITY,
                shape,
                lower,
                upper,
                look,
                fixed: matches!(body, None | Some(RigidBody::Static)),
            };
            item.place(transform);
            items.push(item);
        }
        // Emissive meshes outside any collider body stand in as their box.
        for (entity, transform, aabb, material, child_of, visible) in &proxies {
            let Some(look) = reader.material(material.id()) else {
                continue;
            };
            let emission = look.emission;
            if !visible.get()
                || emission.red + emission.green + emission.blue <= 0.0
                || child_of.is_some_and(|c| has_collider.contains(c.parent()))
            {
                continue;
            }
            material_users
                .entry(material.id())
                .or_default()
                .push(entity);
            let half = Vec3::from(aabb.half_extents).max(Vec3::splat(1e-3));
            let mut item = Item {
                entity,
                scaled: true,
                offset: Vec3::from(aabb.center),
                to_world: Affine3A::IDENTITY,
                shape: [0.0, half.x, half.y, half.z],
                lower: -half,
                upper: half,
                look,
                fixed: false,
            };
            item.place(transform);
            items.push(item);
        }
        let geometry = build_geometry(&mut items, state.geometry.version + 1);
        state.bounds = items.iter().map(Item::world_bounds).collect();
        state.slots = items
            .iter()
            .enumerate()
            .map(|(slot, item)| (item.entity, slot))
            .collect();
        state.items = items;
        state.material_users = material_users;
        state.pending_rescan = false;
        state.frames_since_rescan = 0;
        state.moves_since_build = 0;
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
    } else {
        // Only poses or looks changed: rewrite those records, refitting the tree for poses.
        // A changed body is restyled; a changed mesh under a body restyles the body.
        let slots = &state.slots;
        let tracked = |entity: Entity| slots.contains_key(&entity).then_some(entity);
        let mut restyle: HashSet<Entity> = changes
            .2
            .iter()
            .filter_map(|entity| {
                tracked(entity).or_else(|| tracked(parents.get(entity).ok()?.parent()))
            })
            .collect();
        if textures_changed {
            restyle.extend(state.items.iter().map(|item| item.entity));
        } else {
            for id in &material_changed {
                restyle.extend(state.material_users.get(id).into_iter().flatten().copied());
            }
        }
        let mut geometry_changed = false;
        for entity in restyle {
            let slot = state.slots[&entity];
            let look = if let Ok((_, _, _, body, own, material, children)) = colliders.get(entity) {
                state.items[slot].fixed = matches!(body, None | Some(RigidBody::Static));
                body_look(&mut reader, own, material, children, &child_looks).0
            } else if let Ok((_, _, _, material, _, _)) = proxies.get(entity) {
                reader.material(material.id()).unwrap_or(GREY)
            } else {
                continue;
            };
            let item = &mut state.items[slot];
            // An emitter switched on or off changes the sampled emissives: found again.
            if (luminance(look.emission) > 0.0) != (luminance(item.look.emission) > 0.0) {
                state.pending_rescan = true;
            }
            item.look = look;
            let at = slot * PRIMITIVE_VEC4S;
            Arc::make_mut(&mut state.geometry).primitives[at + 4..at + 6]
                .copy_from_slice(&item.look_rows());
            geometry_changed = true;
        }
        let mut refit = false;
        for (entity, transform) in &tracked_moves {
            let slot = state.slots[entity];
            let item = &mut state.items[slot];
            item.to_world = *transform;
            let at = slot * PRIMITIVE_VEC4S;
            Arc::make_mut(&mut state.geometry).primitives[at..at + 3].copy_from_slice(&item.rows());
            state.bounds[slot] = item.world_bounds();
            refit = true;
        }
        if refit {
            refit_nodes(&mut Arc::make_mut(&mut state.geometry).nodes, &state.bounds);
        }
        if refit || geometry_changed {
            Arc::make_mut(&mut state.geometry).version += 1;
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
    // Of the lights, those whose reach comes nearest the camera (all those
    // whose range holds it alike).
    let nearest = |mut lights: Vec<(f32, Light)>| {
        lights.sort_by(|a, b| a.0.total_cmp(&b.0));
        lights
            .into_iter()
            .take(MAX_LIGHTS)
            .map(|(_, light)| light)
            .collect::<Vec<_>>()
    };
    let gap = |transform: &GlobalTransform, range: f32| {
        (transform.translation().distance(eye) - range).max(0.0)
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
                    gap(transform, point.range),
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
                (gap(transform, spot.range), light)
            })
            .collect(),
    );
    let mut emissives: Vec<_> = state.items.iter().filter_map(Item::emissive).collect();
    emissives.sort_by(|a, b| {
        a.center
            .distance_squared(eye)
            .total_cmp(&b.center.distance_squared(eye))
    });
    emissives.truncate(MAX_EMISSIVES);

    // Reflection probes, nearest the camera first.
    let mut probes: Vec<(f32, Probe)> = probes
        .iter()
        .filter(|(_, environment, _, _, _, visible)| visible.get() && environment.intensity > 0.0)
        .map(|(probe, environment, transform, parallax, overrides, _)| {
            let affine = transform.affine();
            let (scale, _, translation) = transform.to_scale_rotation_translation();
            let size = scale.abs().max(Vec3::splat(1e-4));
            let falloff = probe.falloff.clamp(Vec3::ZERO, Vec3::ONE);
            // Godot's core: the box less the blend distance and a quarter metre.
            let core = Vec3::splat(0.5) * (Vec3::ONE - falloff) - Vec3::splat(0.25) / size;
            let overrides = overrides.copied().unwrap_or_default();
            let flags = if core.min_element() > 0.0 {
                u32::from(overrides.overrides_ambient)
                    | (u32::from(overrides.overrides_reflections) << 1)
            } else {
                0
            };
            let parallax = match parallax.copied().unwrap_or_default() {
                ParallaxCorrection::None => None,
                ParallaxCorrection::Auto => Some(Vec3::splat(0.5)),
                ParallaxCorrection::Custom(half) => Some(half),
            };
            let local_eye = affine.inverse().transform_point3(eye);
            let outside = (local_eye.abs() - Vec3::splat(0.5)).max(Vec3::ZERO) * size;
            (
                outside.length(),
                Probe {
                    to_local: affine.inverse(),
                    to_world_dir: Mat3::from(affine.matrix3),
                    parallax,
                    core: core.max(Vec3::ZERO),
                    falloff,
                    volume: size.x * size.y * size.z,
                    capture: translation,
                    specular: environment.specular_map.id(),
                    intensity: environment.intensity,
                    rotation: environment.rotation,
                    overrides: flags,
                },
            )
        })
        .collect();
    probes.sort_by(|a, b| a.0.total_cmp(&b.0));
    let probes: Vec<Probe> = probes
        .into_iter()
        .take(MAX_PROBES)
        .map(|(_, probe)| probe)
        .collect();

    // Changes to position, direction, range or emission change transport just
    // as much as colour changes do.
    let mut hasher = DefaultHasher::new();
    (sun.is_some(), omnis.len(), spots.len(), emissives.len()).hash(&mut hasher);
    let mut hash_values = |values: &[f32]| {
        for value in values {
            value.to_bits().hash(&mut hasher);
        }
    };
    if let Some((direction, color)) = sun {
        hash_values(&direction.to_array());
        hash_values(&color.to_array());
    }
    for light in omnis.iter().chain(&spots) {
        hash_values(&light.position.to_array());
        hash_values(&light.color.to_array());
        hash_values(&light.direction.to_array());
        hash_values(&[light.inv_range2, light.cone_scale, light.cone_offset]);
    }
    for emissive in &emissives {
        hash_values(&emissive.center.to_array());
        for axis in emissive.axes {
            hash_values(&axis.to_array());
        }
        hash_values(&emissive.color.to_array());
        hash_values(&[emissive.exposure_weight]);
    }
    let hash = hasher.finish();
    if hash != state.sources_hash {
        state.sources_hash = hash;
        state.fresh_frames = FRESH_FRAMES;
        state.lighting_revision += 1;
    } else {
        state.fresh_frames = state.fresh_frames.saturating_sub(1);
    }
    let sources_changed = state.fresh_frames > 0;

    scene.0 = Arc::new(SceneData {
        geometry: state.geometry.clone(),
        sun,
        omnis,
        spots,
        emissives,
        probes,
        sources_changed,
        lighting_revision: state.lighting_revision,
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
    items.extend(
        order
            .into_iter()
            .map(|index| taken[index].take().expect("leaf order is a permutation")),
    );
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
        let bin =
            |item: &BuildItem| (((item.center[axis] - c_lo[axis]) * scale) as usize).min(BINS - 1);
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
fn build_bvh(
    items: &mut [BuildItem],
    nodes: &mut Vec<[f32; 4]>,
    order: &mut Vec<usize>,
    depth: u32,
) -> usize {
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
        (
            Vec3::from_slice(&nodes[n * 2]),
            Vec3::from_slice(&nodes[n * 2 + 1]),
        )
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

#[allow(clippy::type_complexity)]
/// Each PXTGI camera's background (environment map, else skybox) and ambient
/// light (its own `AmbientLight`, else the global one).
fn extract_view_sources(
    mut commands: Commands,
    cameras: Extract<
        Query<(
            RenderEntity,
            &RadianceCascades,
            Option<&EnvironmentMapLight>,
            Option<&Skybox>,
            Option<&AmbientLight>,
        )>,
    >,
    global_ambient: Extract<Option<Res<GlobalAmbientLight>>>,
) {
    for (entity, settings, environment, skybox, ambient) in &cameras {
        let mut entity = commands.entity(entity);
        if settings.method != GiMethod::Pxtgi {
            entity.remove::<PxtgiViewSources>();
            continue;
        }
        let environment_source = environment.map(|environment| CubeSource {
            specular: environment.specular_map.id(),
            intensity: environment.intensity,
            rotation: environment.rotation,
        });
        let skybox_source = skybox.and_then(|skybox| {
            Some(CubeSource {
                specular: skybox.image.as_ref()?.id(),
                intensity: skybox.brightness,
                // Bevy rotates the skybox by its rotation; sampled with the inverse.
                rotation: skybox.rotation,
            })
        });
        let (color, brightness) = match (ambient, global_ambient.as_deref()) {
            (Some(ambient), _) => (ambient.color, ambient.brightness),
            (None, Some(global)) => (global.color, global.brightness),
            (None, None) => (Color::BLACK, 0.0),
        };
        let linear = color.to_linear();
        entity.insert(PxtgiViewSources {
            background: environment_source.or(skybox_source),
            environment_lights: environment_source.is_some(),
            ambient: Vec3::new(linear.red, linear.green, linear.blue) * brightness,
        });
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
    // Writes the raw diffuse estimate for `RadianceCascadesRawOutput`.
    raw: CachedComputePipelineId,
    raw_layout: BindGroupLayoutDescriptor,
    sampler: Sampler,
    // Bound in place of an environment map or probe capture that is missing.
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
    // The raw diffuse estimate at a `RadianceCascadesRawOutput` image's size.
    raw: Option<Target>,
    frame: Buffer,
    current: usize,
    history: bool,
    prev_clip_from_world: Mat4,
    history_epoch: u32,
    history_reflections: PxtgiReflections,
    history_debug: bool,
}

#[derive(Component)]
struct PxtgiBindGroups {
    scene: BindGroup,
    clip_from_world: Mat4,
    // trace, temporal, the three a-trous passes.
    diffuse: [BindGroup; 5],
    // reflect_trace, reflect_temporal, two filter passes.
    reflect: [BindGroup; 4],
    composite: BindGroup,
    // The raw output pass's inputs and target, and the image it is copied to.
    raw: Option<(BindGroup, AssetId<Image>)>,
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
    data.iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect()
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
                texture_cube(TextureSampleType::Float { filterable: true }).visibility(both),
                texture_cube(TextureSampleType::Float { filterable: true }).visibility(both),
                texture_cube(TextureSampleType::Float { filterable: true }).visibility(both),
                texture_cube(TextureSampleType::Float { filterable: true }).visibility(both),
                texture_cube(TextureSampleType::Float { filterable: true }).visibility(both),
                texture_cube(TextureSampleType::Float { filterable: true }).visibility(both),
                texture_cube(TextureSampleType::Float { filterable: true }).visibility(both),
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
    let raw_layout = BindGroupLayoutDescriptor::new(
        "pxtgi_raw",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                (8, unfiltered()),
                (9, unfiltered()),
                (
                    12,
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                ),
            ),
        ),
    );
    let raw = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("pxtgi_raw_output".into()),
        layout: vec![scene_layout.clone(), raw_layout.clone()],
        shader: shader.0.clone(),
        entry_point: Some("raw_output".into()),
        ..default()
    });
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
        reflect: [
            "reflect_trace",
            "reflect_temporal",
            "reflect_filter1",
            "reflect_filter2",
        ]
        .map(compute),
        composite: composite(Some(additive)),
        composite_debug: composite(None),
        raw,
        raw_layout,
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
        upload(
            &mut gpu.primitives,
            "pxtgi_primitives",
            &geometry.primitives,
        );
        upload(&mut gpu.nodes, "pxtgi_nodes", &geometry.nodes);
        gpu.primitive_count = (geometry.primitives.len() / PRIMITIVE_VEC4S) as u32;
        gpu.node_count = (geometry.nodes.len() / 2) as u32;
        gpu.version = geometry.version;
    }
}

fn target(
    device: &RenderDevice,
    label: &'static str,
    size: UVec2,
    format: TextureFormat,
    usage: TextureUsages,
) -> Target {
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
        let pair = |make: &dyn Fn(&'static str, UVec2) -> Target, label, size| {
            [make(label, size), make(label, size)]
        };
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
            raw: None,
            current: 0,
            history: false,
            prev_clip_from_world: clip_from_world,
            history_epoch: 0,
            history_reflections: PxtgiReflections::Off,
            history_debug: false,
        }
    }
}

// Matrices (12), scalars (9), reflections and environment (6), point and spot
// lights (40), emissives (80), ambient and probe count (2), probes (13 each).
const FRAME_VEC4S: usize = 12 + 9 + 6 + MAX_LIGHTS * 5 + MAX_EMISSIVES * 5 + 2 + MAX_PROBES * 13;

/// The camera's background cube as the shader takes it: strength (exposed),
/// roughest mip, the map's rotation, and whether it is an environment map
/// lighting the scene (a skybox only shows).
struct Environment {
    strength: f32,
    roughest_mip: f32,
    rotation: Quat,
    lights: bool,
}

/// The shader's number for a debug view (1 to 5 as Godot's).
fn debug_index(debug: PxtgiDebug) -> f32 {
    match debug {
        PxtgiDebug::Off => 0.0,
        PxtgiDebug::LastFrame => 1.0,
        PxtgiDebug::RoundTrip => 2.0,
        PxtgiDebug::WorldOnly => 3.0,
        PxtgiDebug::WorldHits => 4.0,
        PxtgiDebug::WorldPosition => 5.0,
        PxtgiDebug::BounceLight => 6.0,
        PxtgiDebug::Reflections => 7.0,
    }
}

/// Debug views that replace the image rather than adding to it.
fn debug_replaces_image(debug: PxtgiDebug) -> bool {
    !matches!(debug, PxtgiDebug::Off | PxtgiDebug::WorldOnly)
}

#[allow(clippy::too_many_arguments)]
fn frame_uniform(
    settings: &RadianceCascades,
    scene: &SceneData,
    gpu: &PxtgiGpu,
    clip_from_world: Mat4,
    eye: Vec3,
    near: f32,
    view: &PxtgiView,
    exposure: f32,
    environment: Option<&Environment>,
    ambient: Vec3,
    // The probes whose captures are bound, and each capture's roughest mip.
    probes: &[(&Probe, f32)],
    frame: u32,
) -> Vec<[f32; 4]> {
    let px: &PxtgiSettings = &settings.pxtgi;
    let bits = |v: [u32; 4]| v.map(f32::from_bits);
    let flag = |on: bool| f32::from(u8::from(on));
    // Light as the GI holds it: irradiance / pi, exposed like the lit image.
    let lit = exposure / PI;
    let mut u = Vec::with_capacity(FRAME_VEC4S);
    for m in [
        clip_from_world,
        clip_from_world.inverse(),
        view.prev_clip_from_world,
    ] {
        u.extend(m.to_cols_array_2d());
    }
    u.push(eye.extend(near).to_array());
    u.push(bits([
        view.size.x,
        view.size.y,
        view.trace_size.x,
        view.trace_size.y,
    ]));
    u.push([
        (frame % 4096) as f32,
        px.max_distance,
        settings.intensity,
        debug_index(px.debug),
    ]);
    let (sun_direction, sun_color) = scene.sun.unwrap_or_default();
    u.push(sun_direction.extend(flag(scene.sun.is_some())).to_array());
    u.push((sun_color * lit).extend(exposure).to_array());
    let sky = settings.environment;
    u.push(
        (Vec3::new(sky.red, sky.green, sky.blue) * exposure)
            .extend(px.world_ray_distance)
            .to_array(),
    );
    u.push(bits([
        gpu.primitive_count,
        gpu.node_count,
        scene.omnis.len() as u32,
        scene.spots.len() as u32,
    ]));
    u.push(bits([
        frame,
        CACHE_SLOTS - 1,
        u32::from(px.multi_bounce),
        px.rays_per_texel,
    ]));
    let history = view.history
        && view.history_epoch == settings.history_epoch
        && !view.history_debug
        && !debug_replaces_image(px.debug);
    let reflection_history = history
        && px.reflections != PxtgiReflections::Off
        && px.reflections == view.history_reflections;
    u.push([
        scene.emissives.len() as f32,
        flag(scene.sources_changed),
        flag(history),
        flag(reflection_history),
    ]);
    let mode = match px.reflections {
        PxtgiReflections::Off => 0.0,
        PxtgiReflections::ScreenSpace => 1.0,
        PxtgiReflections::Traced => 2.0,
        PxtgiReflections::TracedAndProbes => 3.0,
    };
    u.push([mode, px.reflection_roughness_cutoff, 0.0, 0.0]);
    u.push(bits([view.reflect_size.x, view.reflect_size.y, 0, 0]));
    let (strength, roughest_mip, rotation, lights) = environment
        .map_or((0.0, 0.0, Quat::IDENTITY, false), |e| {
            (e.strength, e.roughest_mip, e.rotation, e.lights)
        });
    u.push([
        strength,
        roughest_mip,
        flag(environment.is_some()),
        flag(lights),
    ]);
    // Bevy samples its cube maps with the inverse of their rotation.
    let rotation_rows = |rotation: Quat| {
        let rows = Mat3::from_quat(rotation.inverse()).transpose();
        [0, 1, 2].map(|r| rows.col(r).extend(0.0).to_array())
    };
    u.extend(rotation_rows(rotation));
    let pad = |u: &mut Vec<[f32; 4]>, values: &mut dyn Iterator<Item = [f32; 4]>, len: usize| {
        let start = u.len();
        u.extend(values.take(len));
        u.resize(start + len, [0.0; 4]);
    };
    let omnis = &scene.omnis;
    pad(
        &mut u,
        &mut omnis
            .iter()
            .map(|l| l.position.extend(l.inv_range2).to_array()),
        MAX_LIGHTS,
    );
    pad(
        &mut u,
        &mut omnis.iter().map(|l| (l.color * lit).extend(0.0).to_array()),
        MAX_LIGHTS,
    );
    let spots = &scene.spots;
    pad(
        &mut u,
        &mut spots
            .iter()
            .map(|l| l.position.extend(l.inv_range2).to_array()),
        MAX_LIGHTS,
    );
    pad(
        &mut u,
        &mut spots
            .iter()
            .map(|l| l.direction.extend(l.cone_scale).to_array()),
        MAX_LIGHTS,
    );
    pad(
        &mut u,
        &mut spots
            .iter()
            .map(|l| (l.color * lit).extend(l.cone_offset).to_array()),
        MAX_LIGHTS,
    );
    let emissives = &scene.emissives;
    pad(
        &mut u,
        &mut emissives.iter().map(|e| e.center.extend(0.0).to_array()),
        MAX_EMISSIVES,
    );
    for axis in 0..3 {
        pad(
            &mut u,
            &mut emissives
                .iter()
                .map(|e| e.axes[axis].extend(0.0).to_array()),
            MAX_EMISSIVES,
        );
    }
    // Bevy weights each emission by the exposure by its exposure weight.
    pad(
        &mut u,
        &mut emissives.iter().map(|e| {
            (e.color * (1.0 + (exposure - 1.0) * e.exposure_weight))
                .extend(0.0)
                .to_array()
        }),
        MAX_EMISSIVES,
    );
    u.push((ambient * exposure).extend(0.0).to_array());
    let sources = px.reflections == PxtgiReflections::TracedAndProbes && !probes.is_empty();
    u.push(bits([probes.len() as u32, u32::from(sources), 0, 0]));
    let affine_rows = |a: &Affine3A| {
        let rows = Mat3::from(a.matrix3).transpose();
        [0, 1, 2].map(|r| rows.col(r).extend(a.translation[r]).to_array())
    };
    let mat_rows = |m: &Mat3| {
        let rows = m.transpose();
        [0, 1, 2].map(|r| rows.col(r).extend(0.0).to_array())
    };
    for row in 0..3 {
        pad(
            &mut u,
            &mut probes.iter().map(|(p, _)| affine_rows(&p.to_local)[row]),
            MAX_PROBES,
        );
    }
    for row in 0..3 {
        pad(
            &mut u,
            &mut probes.iter().map(|(p, _)| mat_rows(&p.to_world_dir)[row]),
            MAX_PROBES,
        );
    }
    pad(
        &mut u,
        &mut probes.iter().map(|(p, _)| {
            let flags = p.overrides | if p.parallax.is_some() { 4 } else { 0 };
            p.parallax
                .unwrap_or_default()
                .extend(flags as f32)
                .to_array()
        }),
        MAX_PROBES,
    );
    pad(
        &mut u,
        &mut probes
            .iter()
            .map(|(p, _)| p.core.extend(p.volume).to_array()),
        MAX_PROBES,
    );
    pad(
        &mut u,
        &mut probes.iter().map(|(p, _)| p.falloff.extend(0.0).to_array()),
        MAX_PROBES,
    );
    pad(
        &mut u,
        &mut probes
            .iter()
            .map(|(p, _)| p.capture.extend(p.intensity * exposure).to_array()),
        MAX_PROBES,
    );
    for row in 0..3 {
        pad(
            &mut u,
            &mut probes.iter().map(|(p, mip)| {
                let mut r = rotation_rows(p.rotation)[row];
                if row == 0 {
                    r[3] = *mip;
                }
                r
            }),
            MAX_PROBES,
        );
    }
    debug_assert_eq!(u.len(), FRAME_VEC4S);
    u
}

/// A loaded cube map, and its roughest mip.
fn cube_image(images: &RenderAssets<GpuImage>, id: AssetId<Image>) -> Option<(&GpuImage, f32)> {
    let image = images.get(id)?;
    let cube = image
        .texture_view_descriptor
        .as_ref()
        .is_some_and(|d| d.dimension == Some(TextureViewDimension::Cube));
    cube.then(|| {
        (
            image,
            image.texture_descriptor.mip_level_count.saturating_sub(1) as f32,
        )
    })
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
        &RadianceCascades,
        Option<&PxtgiViewSources>,
        Option<&RadianceCascadesRawOutput>,
        Option<&mut PxtgiView>,
    )>,
) {
    let (Some(pipelines), Some(gpu)) = (pipelines, gpu) else {
        return;
    };
    *frame = frame.wrapping_add(1);
    // The probes whose captures have loaded, bound in this order.
    let probes: Vec<(&Probe, &GpuImage, f32)> = scene
        .0
        .probes
        .iter()
        .filter_map(|probe| {
            cube_image(&images, probe.specular).map(|(image, mip)| (probe, image, mip))
        })
        .collect();
    for (entity, view, camera, target, prepass, settings, sources, raw_output, existing) in
        &mut views
    {
        let settings = settings.sanitized();
        if !settings.pxtgi_active() {
            // The view runs another method, or none: its targets go too.
            commands
                .entity(entity)
                .remove::<(PxtgiBindGroups, PxtgiView)>();
            continue;
        }
        let (Some(size), Some(depth), Some(gbuffer)) = (
            camera.physical_viewport_size,
            prepass.depth_only_view(),
            prepass.deferred_view(),
        ) else {
            commands.entity(entity).remove::<PxtgiBindGroups>();
            continue;
        };
        let main_size = target.main_texture().size();
        // shortcut: the history copy assumes the viewport covers the whole
        // target; offset viewports (split screen) would copy the wrong region.
        if view.clip_from_view.w_axis.w == 1.0
            || target.main_texture_format() != TextureFormat::Rgba16Float
            || UVec2::new(main_size.width, main_size.height) != size
        {
            warn_once!("PXTGI needs an HDR perspective camera whose viewport fills its target");
            commands.entity(entity).remove::<PxtgiBindGroups>();
            continue;
        }
        // A loaded cube map only; until then the sky colour.
        let environment = sources.and_then(|sources| {
            let background = sources.background?;
            let (image, roughest_mip) = cube_image(&images, background.specular)?;
            let settings = Environment {
                strength: background.intensity * camera.exposure,
                roughest_mip,
                rotation: background.rotation,
                lights: sources.environment_lights,
            };
            Some((settings, &image.texture_view))
        });
        let ambient = sources.map_or(Vec3::ZERO, |sources| sources.ambient);
        let clip_from_world = view
            .clip_from_world
            .unwrap_or_else(|| view.clip_from_view * view.world_from_view.to_matrix().inverse());
        let mut fresh = None;
        let state = match existing {
            Some(state) if state.size == size => state.into_inner(),
            _ => fresh.insert(PxtgiView::new(&device, size, clip_from_world)),
        };
        let prev = state.current;
        let cur = 1 - prev;
        let probe_settings: Vec<(&Probe, f32)> = probes
            .iter()
            .map(|(probe, _, mip)| (*probe, *mip))
            .collect();
        let uniform = frame_uniform(
            &settings,
            &scene.0,
            &gpu,
            clip_from_world,
            view.world_from_view.translation(),
            // Infinite reverse-Z perspective keeps the near plane here.
            view.clip_from_view.w_axis.z,
            state,
            camera.exposure,
            environment.as_ref().map(|(settings, _)| settings),
            ambient,
            &probe_settings,
            *frame,
        );
        queue.write_buffer(&state.frame, 0, &bytes(&uniform));

        let probe_map = |i: usize| {
            probes
                .get(i)
                .map_or(&pipelines.no_environment, |(_, image, _)| {
                    &image.texture_view
                })
        };
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
                probe_map(0),
                probe_map(1),
                probe_map(2),
                probe_map(3),
                probe_map(4),
                probe_map(5),
                probe_map(6),
                probe_map(7),
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
                [
                    &s.filtered[i],
                    &s.filtered_variance[i],
                    &s.moments[prev],
                    &s.accum_depth[prev],
                ],
            )
        };
        let diffuse = [
            io(
                [
                    &s.accum[prev],
                    &s.accum_depth[prev],
                    &s.moments[prev],
                    &s.filtered_variance[1],
                ],
                [
                    &s.trace,
                    &s.variance,
                    &s.filtered[0],
                    &s.filtered_variance[0],
                ],
            ),
            io(
                [
                    &s.trace,
                    &s.accum[prev],
                    &s.accum_depth[prev],
                    &s.moments[prev],
                ],
                [
                    &s.accum[cur],
                    &s.accum_depth[cur],
                    &s.moments[cur],
                    &s.variance,
                ],
            ),
            atrous(&s.accum[cur], &s.variance, 0),
            atrous(&s.filtered[0], &s.filtered_variance[0], 1),
            atrous(&s.filtered[1], &s.filtered_variance[1], 0),
        ];
        // The diffuse passes are done by now: their intermediates fill the unused slots.
        let spare = [&s.variance, &s.filtered[1], &s.filtered_variance[1]];
        let reflect = [
            io(
                [
                    &s.accum[prev],
                    &s.accum_depth[prev],
                    &s.moments[prev],
                    &s.trace,
                ],
                [&s.reflect_trace, spare[0], spare[1], spare[2]],
            ),
            io(
                [
                    &s.reflect_trace,
                    &s.reflect_accum[prev],
                    &s.reflect_depth[prev],
                    &s.trace,
                ],
                [
                    &s.reflect_accum[cur],
                    &s.reflect_depth[cur],
                    spare[1],
                    spare[2],
                ],
            ),
            io(
                [
                    &s.reflect_accum[cur],
                    &s.reflect_depth[cur],
                    &s.trace,
                    &s.trace,
                ],
                [&s.reflect_filtered[0], spare[0], spare[1], spare[2]],
            ),
            io(
                [
                    &s.reflect_filtered[0],
                    &s.reflect_depth[cur],
                    &s.trace,
                    &s.trace,
                ],
                [&s.reflect_filtered[1], spare[0], spare[1], spare[2]],
            ),
        ];
        // The raw output, into a target the size of the application's image.
        let raw_image = raw_output.and_then(|output| {
            let image = images.get(&output.0)?;
            let usable = image.texture_descriptor.format == TextureFormat::Rgba32Float
                && image
                    .texture_descriptor
                    .usage
                    .contains(TextureUsages::COPY_DST);
            if !usable {
                warn_once!("PXTGI raw output needs an Rgba32Float image with COPY_DST usage");
            }
            usable.then(|| (output.0.id(), image.texture_descriptor.size))
        });
        match raw_image {
            Some((_, size)) if s.raw.as_ref().is_none_or(|raw| raw.texture.size() != size) => {
                let target = self::target(
                    &device,
                    "pxtgi_raw_output",
                    UVec2::new(size.width, size.height),
                    TextureFormat::Rgba32Float,
                    TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
                );
                state.raw = Some(target);
            }
            None => state.raw = None,
            _ => {}
        }
        let s = &*state;
        let raw = raw_image.zip(s.raw.as_ref()).map(|((id, _), raw)| {
            let group = device.create_bind_group(
                "pxtgi_raw",
                &cache.get_bind_group_layout(&pipelines.raw_layout),
                &BindGroupEntries::with_indices((
                    (8, &s.filtered[0].view),
                    (9, &s.accum_depth[cur].view),
                    (12, &raw.view),
                )),
            );
            (group, id)
        });
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
            clip_from_world,
            scene: scene_group,
            diffuse,
            reflect,
            composite,
            raw,
        });
        if let Some(state) = fresh {
            entity_commands.insert(state);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn render(
    view: ViewQuery<(
        &ViewTarget,
        &RadianceCascades,
        &mut PxtgiView,
        &PxtgiBindGroups,
    )>,
    pipelines: Res<PxtgiPipelines>,
    gpu: Res<PxtgiGpu>,
    cache: Res<PipelineCache>,
    status: Option<Res<RadianceCascadesStatus>>,
    scene: Res<PxtgiScene>,
    images: Res<RenderAssets<GpuImage>>,
    mut frame_id: Local<u64>,
    mut context: RenderContext,
) {
    let (target, settings, mut state, groups) = view.into_inner();
    let settings = settings.sanitized();
    if !settings.pxtgi_active() {
        return;
    }
    let px = settings.pxtgi;
    let compute = |id| cache.get_compute_pipeline(id);
    let diffuse: Option<Vec<_>> = pipelines.diffuse.iter().map(|&id| compute(id)).collect();
    let reflect: Option<Vec<_>> = pipelines.reflect.iter().map(|&id| compute(id)).collect();
    let composite = cache.get_render_pipeline(if debug_replaces_image(px.debug) {
        pipelines.composite_debug
    } else {
        pipelines.composite
    });
    let (Some(cache_update), Some(diffuse), Some(reflect), Some(composite)) =
        (compute(pipelines.cache), diffuse, reflect, composite)
    else {
        return;
    };
    let diagnostics = context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let total = diagnostics.time_span(context.command_encoder(), "pxtgi");
    let groups_for = |size: UVec2| (size.x.div_ceil(8), size.y.div_ceil(8));
    // One compute pass per stage, so each gets its own GPU timing.
    let dispatch = |context: &mut RenderContext,
                    name: &'static str,
                    pipeline: &ComputePipeline,
                    group: &BindGroup,
                    (x, y): (u32, u32)| {
        let span = diagnostics.time_span(context.command_encoder(), name);
        let mut pass = context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some(name),
                timestamp_writes: None,
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &groups.scene, &[]);
        pass.set_bind_group(1, group, &[]);
        pass.dispatch_workgroups(x, y, 1);
        drop(pass);
        span.end(context.command_encoder());
    };
    if px.multi_bounce && gpu.primitive_count > 0 {
        // Any io group: the cache pass reads none of it.
        dispatch(
            &mut context,
            "cache",
            cache_update,
            &groups.diffuse[0],
            (CACHE_ROW / 8, CACHE_SLOTS / CACHE_ROW / 4 / 8),
        );
    }
    let names = ["trace", "temporal", "atrous1", "atrous2", "atrous4"];
    for ((pipeline, group), name) in diffuse.into_iter().zip(&groups.diffuse).zip(names) {
        dispatch(
            &mut context,
            name,
            pipeline,
            group,
            groups_for(state.trace_size),
        );
    }
    if px.reflections != PxtgiReflections::Off {
        let names = [
            "reflect_trace",
            "reflect_temporal",
            "reflect_filter1",
            "reflect_filter2",
        ];
        for ((pipeline, group), name) in reflect.into_iter().zip(&groups.reflect).zip(names) {
            dispatch(
                &mut context,
                name,
                pipeline,
                group,
                groups_for(state.reflect_size),
            );
        }
    }
    {
        let span = diagnostics.time_span(context.command_encoder(), "composite");
        let mut pass = context
            .command_encoder()
            .begin_render_pass(&RenderPassDescriptor {
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
        drop(pass);
        span.end(context.command_encoder());
    }
    // Next frame's screen rays read this frame's lit image.
    context.command_encoder().copy_texture_to_texture(
        target.main_texture().as_image_copy(),
        state.prev_color.texture.as_image_copy(),
        Extent3d {
            width: state.size.x,
            height: state.size.y,
            depth_or_array_layers: 1,
        },
    );
    // The raw diffuse estimate, copied into the application's image.
    let raw_copied = match (&groups.raw, &state.raw, compute(pipelines.raw)) {
        (Some((group, image)), Some(raw), Some(pipeline)) => match images.get(*image) {
            Some(destination) => {
                let size = raw.texture.size();
                dispatch(
                    &mut context,
                    "raw_output",
                    pipeline,
                    group,
                    (size.width.div_ceil(8), size.height.div_ceil(8)),
                );
                context.command_encoder().copy_texture_to_texture(
                    raw.texture.as_image_copy(),
                    destination.texture.as_image_copy(),
                    size,
                );
                true
            }
            None => false,
        },
        _ => false,
    };
    total.end(context.command_encoder());
    state.current = 1 - state.current;
    state.prev_clip_from_world = groups.clip_from_world;
    state.history_epoch = settings.history_epoch;
    state.history_reflections = px.reflections;
    state.history_debug = debug_replaces_image(px.debug);
    state.history = true;
    *frame_id += 1;
    if let Some(status) = status {
        status.set_active_method(GiMethod::Pxtgi);
        status.set_hardware_ray_queries_active(false);
        let version = scene.0.geometry.version;
        let dispatch = status.record_dispatch(crate::RadianceCascadesRevisions {
            scene: version,
            geometry: version,
            lighting: scene.0.lighting_revision,
        });
        if groups.raw.is_some() {
            status.record_raw_frame(raw_copied.then(|| crate::RadianceCascadesRawFrame {
                frame_id: *frame_id,
                dispatch,
                submitted_at: std::time::Instant::now(),
            }));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_ignores_equal_poses_and_tracks_collider_and_light_motion() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(bevy::transform::TransformPlugin)
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<Assets<Image>>()
            .init_resource::<PxtgiScene>()
            .add_message::<AssetEvent<StandardMaterial>>()
            .add_message::<AssetEvent<Image>>()
            .add_systems(PostUpdate, update_scene.after(TransformSystems::Propagate));
        app.world_mut().spawn((
            Camera::default(),
            GlobalTransform::IDENTITY,
            RadianceCascades {
                method: GiMethod::Pxtgi,
                ..default()
            },
        ));
        let collider = app
            .world_mut()
            .spawn((
                Collider::Cuboid {
                    half_extents: Vec3::splat(0.5),
                },
                Transform::IDENTITY,
                GlobalTransform::IDENTITY,
            ))
            .id();
        let light = app
            .world_mut()
            .spawn((
                PointLight::default(),
                Transform::IDENTITY,
                GlobalTransform::IDENTITY,
                InheritedVisibility::VISIBLE,
            ))
            .id();
        app.update();
        app.update();
        let scene = app.world().resource::<PxtgiScene>().0.clone();
        let geometry_version = scene.geometry.version;
        let lighting_revision = scene.lighting_revision;
        app.world_mut()
            .entity_mut(collider)
            .insert(Transform::IDENTITY);
        app.update();
        let unchanged = &app.world().resource::<PxtgiScene>().0;
        assert_eq!(unchanged.geometry.version, geometry_version);
        assert!(Arc::ptr_eq(&unchanged.geometry, &scene.geometry));
        app.world_mut()
            .entity_mut(collider)
            .insert(Transform::from_xyz(2.0, 0.0, 0.0));
        app.world_mut()
            .entity_mut(light)
            .insert(Transform::from_xyz(3.0, 0.0, 0.0));
        app.update();
        let moved = &app.world().resource::<PxtgiScene>().0;
        assert!(moved.geometry.version > geometry_version);
        assert!(moved.lighting_revision > lighting_revision);
        assert!(moved.geometry.nodes[0][0] >= 1.5);
        assert_eq!(moved.omnis[0].position, Vec3::new(3.0, 0.0, 0.0));
    }

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
            look: GREY,
            fixed: true,
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
            Collider::Cuboid {
                half_extents: Vec3::splat(0.5),
            },
            Collider::Sphere { radius: 0.4 },
            Collider::Capsule {
                radius: 0.2,
                half_height: 0.3,
            },
        ];
        let mut items: Vec<_> = (0..40)
            .map(|i| {
                item(
                    i + 1,
                    &shapes[i as usize % 3],
                    Vec3::new(i as f32, (i % 5) as f32, 0.0),
                    &mut store,
                )
            })
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
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
        for entry in ENTRY_POINTS {
            assert!(
                module.entry_points.iter().any(|point| point.name == entry),
                "{entry}"
            );
        }
        // The frame uniform the CPU writes must be exactly the shader's `Frame`.
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(module.to_ctx()).unwrap();
        let frame = module
            .global_variables
            .iter()
            .find(|(_, global)| global.space == naga::AddressSpace::Uniform)
            .expect("the frame uniform");
        assert_eq!(layouter[frame.1.ty].size as usize, FRAME_VEC4S * 16);
    }

    #[test]
    fn body_look_prefers_its_override_then_its_material_then_its_children() {
        use bevy::ecs::system::SystemState;
        let mut materials = Assets::<StandardMaterial>::default();
        let mut images = Assets::<Image>::default();
        // A 2x2 texture whose mean is a mid red: it scales the base colour.
        let texture = images.add(Image::new_fill(
            Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            &[255, 0, 0, 255],
            TextureFormat::Rgba8Unorm,
            bevy::asset::RenderAssetUsages::default(),
        ));
        let textured = materials.add(StandardMaterial {
            base_color: Color::linear_rgb(0.5, 0.5, 0.5),
            base_color_texture: Some(texture),
            ..default()
        });
        let dim = materials.add(StandardMaterial {
            base_color: Color::linear_rgb(0.2, 0.3, 0.4),
            emissive: LinearRgba::rgb(1.0, 0.0, 0.0),
            ..default()
        });
        let bright = materials.add(StandardMaterial {
            base_color: Color::linear_rgb(0.9, 0.9, 0.9),
            emissive: LinearRgba::rgb(0.0, 8.0, 0.0),
            emissive_exposure_weight: 0.5,
            ..default()
        });
        let mut world = World::new();
        let first = world.spawn(MeshMaterial3d(dim.clone())).id();
        let second = world.spawn(MeshMaterial3d(bright)).id();
        let parent = world.spawn_empty().add_children(&[first, second]).id();
        let mut state: SystemState<(
            Query<&Children>,
            Query<
                (
                    Option<&PxtgiLook>,
                    Option<&MeshMaterial3d<StandardMaterial>>,
                ),
                Without<Collider>,
            >,
        )> = SystemState::new(&mut world);
        let (children, child_looks) = state.get(&world).unwrap();
        let (mut means, mut missing) = (HashMap::new(), HashSet::new());
        let mut reader = LookReader {
            materials: &materials,
            images: &images,
            means: &mut means,
            missing: &mut missing,
        };
        // Children: the first one's albedo, the brightest one's emission.
        let (look, used) = body_look(
            &mut reader,
            None,
            None,
            children.get(parent).ok(),
            &child_looks,
        );
        assert_eq!(look.albedo, LinearRgba::rgb(0.2, 0.3, 0.4));
        assert_eq!(look.emission, LinearRgba::rgb(0.0, 8.0, 0.0));
        assert_eq!(look.exposure_weight, 0.5);
        assert_eq!(used.len(), 2);
        // Its own material wins over its children, its texture's mean included.
        let own = MeshMaterial3d(textured);
        let (look, _) = body_look(
            &mut reader,
            None,
            Some(&own),
            children.get(parent).ok(),
            &child_looks,
        );
        assert!((look.albedo.red - 0.5).abs() < 1e-4 && look.albedo.green.abs() < 1e-4);
        // Its own PxtgiLook wins over everything.
        let override_look = PxtgiLook {
            albedo: LinearRgba::rgb(0.1, 0.1, 0.1),
            emission: LinearRgba::BLACK,
        };
        let (look, used) = body_look(
            &mut reader,
            Some(&override_look),
            Some(&own),
            None,
            &child_looks,
        );
        assert_eq!(look.albedo, override_look.albedo);
        assert!(used.is_empty());
        // Nothing readable: neutral grey.
        assert_eq!(
            body_look(&mut reader, None, None, None, &child_looks).0,
            GREY
        );
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

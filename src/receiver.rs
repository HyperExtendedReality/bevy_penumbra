//! Raster receiver identities, using the native prepass geometry and material pipeline.

use crate::{RadianceCascades, RadianceCascadesGeometry};
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::{
    core_pipeline::{Core3dSystems, schedule::Core3d},
    ecs::{
        query::ROQueryItem,
        system::{
            SystemParamItem,
            lifetimeless::{Read, SRes},
        },
    },
    pbr::{
        DrawMesh, Lightmap, MeshPipeline, RenderMeshInstances, SetMaterialBindGroup,
        SetMeshBindGroup, SetPrepassViewBindGroup, SpecializedPrepassMaterialPipelineCache,
    },
    prelude::*,
    render::{
        Extract, ExtractSchedule, Render, RenderApp, RenderDebugFlags, RenderSystems,
        camera::ExtractedCamera,
        extract_component::{ComponentUniforms, DynamicUniformIndex, UniformComponentPlugin},
        mesh::{RenderMesh, RenderMeshBufferInfo},
        render_asset::RenderAssets,
        render_phase::*,
        render_resource::{binding_types::uniform_buffer, *},
        renderer::{RenderContext, RenderDevice, ViewQuery},
        sync_world::{MainEntity, RenderEntity},
        view::{ExtractedView, RenderVisibleEntities, ViewDepthStencilTexture},
    },
    shader::ShaderDefVal,
};
use std::{
    collections::{HashMap, HashSet},
    ops::Range,
};

#[derive(Component, Clone, Copy, ShaderType)]
struct ReceiverIdentity {
    value: UVec4,
}

fn identity(
    entity: Entity,
    geometry: bool,
    baked: bool,
    material: Option<&StandardMaterial>,
    scene_instance: Option<u32>,
) -> ReceiverIdentity {
    let bits = entity.to_bits();
    ReceiverIdentity {
        value: UVec4::new(
            bits as u32,
            (bits >> 32) as u32,
            u32::from(geometry && !baked && material.is_some_and(|m| !m.unlit)),
            scene_instance.unwrap_or(u32::MAX),
        ),
    }
}

/// Full target-sized identity texture. xy stores Entity::to_bits; z stores
/// eligibility and a 2×15-bit octahedral geometric normal.
#[derive(Component, Clone)]
pub(crate) struct ReceiverTexture {
    _texture: Texture,
    pub view: TextureView,
    size: UVec2,
    samples: u32,
    pub ready: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl ReceiverTexture {
    pub fn bytes(&self) -> u64 {
        u64::from(self.size.x) * u64::from(self.size.y) * u64::from(self.samples) * 16
    }
    pub fn matches(&self, depth: &ViewDepthStencilTexture) -> bool {
        self.size == UVec2::new(depth.texture().width(), depth.texture().height())
            && self.samples == depth.texture().sample_count()
    }
    pub fn new(device: &RenderDevice, depth: &ViewDepthStencilTexture) -> Self {
        let size = UVec2::new(depth.texture().width(), depth.texture().height());
        let samples = depth.texture().sample_count();
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("gi_receiver_identity"),
            size: Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: samples,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Uint,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        Self {
            _texture: texture,
            view,
            size,
            samples,
            ready: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }
}

#[derive(Resource)]
pub(crate) struct ReceiverPipelines {
    shader: Handle<Shader>,
    layout: BindGroupLayoutDescriptor,
    cached: HashMap<CachedRenderPipelineId, CachedRenderPipelineId>,
    incomplete_views: HashSet<bevy::render::view::RetainedViewEntity>,
}
#[derive(Resource)]
struct IdentityBindGroup(BindGroup);

/// Actual allocated capacity, including native dynamic-uniform alignment.
#[derive(Resource, Default)]
pub(crate) struct ReceiverIdentityMemory {
    pub bytes: u64,
}

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PrepareReceiverTextures;

pub(crate) struct ReceiverPhase {
    entity: (Entity, MainEntity),
    pipeline: CachedRenderPipelineId,
    draw: DrawFunctionId,
    range: Range<u32>,
    extra: PhaseItemExtraIndex,
    indexed: bool,
}
impl PhaseItem for ReceiverPhase {
    // ponytail: one identity uniform per mesh draw; batch with a sidecar identity
    // stream only if this extra pass becomes a measured CPU bottleneck.
    const AUTOMATIC_BATCHING: bool = false;
    fn entity(&self) -> Entity {
        self.entity.0
    }
    fn main_entity(&self) -> MainEntity {
        self.entity.1
    }
    fn draw_function(&self) -> DrawFunctionId {
        self.draw
    }
    fn batch_range(&self) -> &Range<u32> {
        &self.range
    }
    fn batch_range_mut(&mut self) -> &mut Range<u32> {
        &mut self.range
    }
    fn extra_index(&self) -> PhaseItemExtraIndex {
        self.extra.clone()
    }
    fn batch_range_and_extra_index_mut(&mut self) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
        (&mut self.range, &mut self.extra)
    }
}
impl CachedRenderPipelinePhaseItem for ReceiverPhase {
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.pipeline
    }
}
impl SortedPhaseItem for ReceiverPhase {
    type SortKey = u64;
    fn sort_key(&self) -> u64 {
        self.entity.0.to_bits()
    }
    fn recalculate_sort_keys(
        _: &mut indexmap::IndexMap<(Entity, MainEntity), Self, bevy::ecs::entity::EntityHash>,
        _: &ExtractedView,
    ) {
    }
    fn indexed(&self) -> bool {
        self.indexed
    }
}

struct SetIdentity;
impl RenderCommand<ReceiverPhase> for SetIdentity {
    type Param = SRes<IdentityBindGroup>;
    type ViewQuery = ();
    type ItemQuery = Read<DynamicUniformIndex<ReceiverIdentity>>;
    fn render<'w>(
        _: &ReceiverPhase,
        _: (),
        identity: Option<ROQueryItem<Self::ItemQuery>>,
        group: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(identity) = identity else {
            return RenderCommandResult::Skip;
        };
        pass.set_bind_group(1, &group.into_inner().0, &[identity.index()]);
        RenderCommandResult::Success
    }
}
type DrawReceiver = (
    SetItemPipeline,
    SetPrepassViewBindGroup<0>,
    SetIdentity,
    SetMeshBindGroup<2>,
    SetMaterialBindGroup<3>,
    DrawMesh,
);

pub(crate) fn install(app: &mut App) {
    if app.get_sub_app(RenderApp).is_none() {
        return;
    }
    let shader = app
        .world_mut()
        .resource_mut::<Assets<Shader>>()
        .add(Shader::from_wesl(
            include_str!("../shaders/receiver.wesl"),
            "penumbra_receiver.wesl",
        ));
    app.add_plugins((
        UniformComponentPlugin::<ReceiverIdentity>::default(),
        SortedRenderPhasePlugin::<ReceiverPhase, MeshPipeline>::new(RenderDebugFlags::default()),
    ));
    let render_app = app.sub_app_mut(RenderApp);
    render_app
        .insert_resource(ReceiverPipelines {
            shader,
            layout: BindGroupLayoutDescriptor::new(
                "gi_receiver_identity",
                &BindGroupLayoutEntries::single(
                    ShaderStages::FRAGMENT,
                    uniform_buffer::<ReceiverIdentity>(true),
                ),
            ),
            cached: HashMap::new(),
            incomplete_views: HashSet::new(),
        })
        .init_resource::<DrawFunctions<ReceiverPhase>>()
        .init_resource::<ReceiverIdentityMemory>()
        .add_render_command::<ReceiverPhase, DrawReceiver>()
        .add_systems(ExtractSchedule, extract)
        .add_systems(
            Render,
            (
                queue
                    .in_set(RenderSystems::QueueMeshes)
                    .after(RenderSystems::PrepareMeshes),
                sort_phase_system::<ReceiverPhase>.in_set(RenderSystems::PhaseSort),
                prepare
                    .in_set(RenderSystems::PrepareBindGroups)
                    .in_set(PrepareReceiverTextures),
            ),
        )
        .add_systems(
            Core3d,
            draw.in_set(Core3dSystems::MainPass)
                .before(bevy::pbr::deferred::deferred_lighting),
        );
}

#[allow(clippy::type_complexity)]
fn extract(
    mut commands: Commands,
    scene: Extract<Res<crate::scene::GiScene>>,
    meshes: Extract<
        Query<
            (
                Entity,
                RenderEntity,
                Option<&MeshMaterial3d<StandardMaterial>>,
                Has<RadianceCascadesGeometry>,
                Has<Lightmap>,
            ),
            With<Mesh3d>,
        >,
    >,
    materials: Extract<Res<Assets<StandardMaterial>>>,
    cameras: Extract<Query<(Entity, &Camera, &RadianceCascades)>>,
    mut phases: ResMut<ViewSortedRenderPhases<ReceiverPhase>>,
    (identities, mut live, mut was_active): (
        Query<Entity, With<ReceiverIdentity>>,
        Local<HashSet<bevy::render::view::RetainedViewEntity>>,
        Local<bool>,
    ),
) {
    let active = cameras
        .iter()
        .any(|(_, camera, settings)| camera.is_active && settings.sanitized().intensity > 0.0);
    if !active {
        if *was_active {
            // Clean up once on the transition, including identities whose
            // meshes were removed while GI was active.
            for entity in &identities {
                commands.entity(entity).remove::<ReceiverIdentity>();
            }
            phases.retain(|_, _| false);
        }
        *was_active = false;
        return;
    }
    *was_active = true;
    for (entity, render_entity, material, geometry, lightmap) in &meshes {
        let scene_instance = scene.data.instance_indices.get(&entity).copied();
        // This pass reuses StandardMaterial's alpha-discard bindings. Other
        // material implementations keep the cleared, excluded identity.
        let identity = identity(
            entity,
            geometry
                && scene.data.instance_residency.contains_key(&entity)
                && scene_instance.is_some(),
            lightmap,
            material.and_then(|m| materials.get(&m.0)),
            scene_instance,
        );
        if identity.value.z == 0 {
            commands.entity(render_entity).remove::<ReceiverIdentity>();
            continue;
        }
        commands.entity(render_entity).insert(identity);
    }
    live.clear();
    for (entity, camera, settings) in &cameras {
        if !camera.is_active || settings.sanitized().intensity <= 0.0 {
            continue;
        }
        let view = bevy::render::view::RetainedViewEntity::new(entity.into(), None, 0);
        phases.prepare_for_new_frame(view);
        live.insert(view);
    }
    phases.retain(|view, _| live.contains(view));
}

#[allow(clippy::too_many_arguments)]
fn queue(
    views: Query<(&ExtractedView, &RenderVisibleEntities), With<RadianceCascades>>,
    native: Res<SpecializedPrepassMaterialPipelineCache>,
    cache: Res<PipelineCache>,
    mut pipelines: ResMut<ReceiverPipelines>,
    mesh_instances: Res<RenderMeshInstances>,
    meshes: Res<RenderAssets<RenderMesh>>,
    identities: Query<(Entity, &MainEntity), With<ReceiverIdentity>>,
    draws: Res<DrawFunctions<ReceiverPhase>>,
    mut phases: ResMut<ViewSortedRenderPhases<ReceiverPhase>>,
    mut identity_entities: Local<HashMap<MainEntity, Entity>>,
) {
    let draw = draws.read().id::<DrawReceiver>();
    pipelines.incomplete_views.clear();
    // Native mesh visibility uses PLACEHOLDER render entities. Resolve the
    // synced entity that owns the dynamic identity uniform from MainEntity.
    identity_entities.clear();
    identity_entities.extend(identities.iter().map(|(entity, main)| (*main, entity)));
    for (view, visible) in &views {
        let (Some(phase), Some(native)) = (
            phases.get_mut(&view.retained_view_entity),
            native.get(&view.retained_view_entity),
        ) else {
            continue;
        };
        let Some(visible) = visible.get::<Mesh3d>() else {
            continue;
        };
        for (_, main) in visible.iter_visible() {
            let Some(&entity) = identity_entities.get(main) else {
                continue;
            };
            let Some((_, source, _)) = native.get(main) else {
                continue;
            };
            let pipeline = if let Some(pipeline) = pipelines.cached.get(source) {
                *pipeline
            } else {
                // Newly queued native IDs have no descriptor until Bevy drains
                // its asynchronous pipeline queue. Retry on a later frame.
                if cache.get_render_pipeline(*source).is_none() {
                    pipelines.incomplete_views.insert(view.retained_view_entity);
                    continue;
                }
                let mut descriptor = cache.get_render_pipeline_descriptor(*source).clone();
                // The receiver material comes from the native deferred G-buffer.
                // Forward-only meshes leave the identity attachment cleared.
                if !descriptor.vertex.shader_defs.iter().any(|definition| {
                    matches!(definition, ShaderDefVal::Bool(name, true) if name == "DEFERRED_PREPASS")
                }) {
                    continue;
                }
                descriptor.label = Some("gi_receiver_raster".into());
                descriptor.layout[1] = pipelines.layout.clone();
                let Some(fragment) = descriptor.fragment.as_mut() else {
                    continue;
                };
                fragment.shader = pipelines.shader.clone();
                fragment.entry_point = Some("fragment".into());
                fragment.targets = vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba32Uint,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })];
                if let Some(depth) = descriptor.depth_stencil.as_mut() {
                    depth.depth_write_enabled = Some(false);
                    depth.depth_compare = Some(CompareFunction::Equal);
                }
                // The native material prepass vertex shader remains unchanged,
                // including deformation, raster bias and instance addressing.
                let pipeline = cache.queue_render_pipeline(descriptor);
                pipelines.cached.insert(*source, pipeline);
                pipeline
            };
            // A newly visible material may still be specializing. Keep the
            // already compiled receivers active; this draw's pixels remain
            // cleared/ineligible until its own identity pipeline is ready.
            if cache.get_render_pipeline(pipeline).is_none() {
                pipelines.incomplete_views.insert(view.retained_view_entity);
                continue;
            }
            let Some(mesh_id) = mesh_instances.mesh_asset_id(*main) else {
                continue;
            };
            let Some(mesh) = meshes.get(mesh_id) else {
                continue;
            };
            phase.add_transient(ReceiverPhase {
                entity: (entity, *main),
                pipeline,
                draw,
                range: 0..1,
                extra: PhaseItemExtraIndex::None,
                indexed: matches!(mesh.buffer_info, RenderMeshBufferInfo::Indexed { .. }),
            });
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    pipelines: Res<ReceiverPipelines>,
    mut uniforms: ResMut<ComponentUniforms<ReceiverIdentity>>,
    identities: Query<(), With<ReceiverIdentity>>,
    textures: Query<&ReceiverTexture>,
    mut memory: ResMut<ReceiverIdentityMemory>,
) {
    // Readiness belongs to this frame's encoded identity pass, even when draw
    // is skipped because a required view component or specialization is absent.
    for texture in &textures {
        texture
            .ready
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }
    if identities.is_empty() {
        *uniforms = ComponentUniforms::default();
        memory.bytes = 0;
        commands.remove_resource::<IdentityBindGroup>();
        return;
    }
    memory.bytes = uniforms
        .uniforms()
        .buffer()
        .map_or(0, |buffer| buffer.size());
    if let Some(binding) = uniforms.uniforms().binding() {
        commands.insert_resource(IdentityBindGroup(device.create_bind_group(
            "gi_receiver_identity",
            &cache.get_bind_group_layout(&pipelines.layout),
            &BindGroupEntries::single(binding),
        )));
    }
}

pub(crate) fn draw(
    world: &World,
    view: ViewQuery<(
        &ExtractedCamera,
        &ExtractedView,
        &ViewDepthStencilTexture,
        &ReceiverTexture,
    )>,
    phases: Res<ViewSortedRenderPhases<ReceiverPhase>>,
    pipelines: Res<PipelineCache>,
    receivers: Res<ReceiverPipelines>,
    mut ctx: RenderContext,
    mut last_phase_items: Local<HashMap<bevy::render::view::RetainedViewEntity, usize>>,
) {
    let entity = view.entity();
    let (camera, view, depth, identity) = view.into_inner();
    identity
        .ready
        .store(false, std::sync::atomic::Ordering::Relaxed);
    let Some(phase) = phases.get(&view.retained_view_entity) else {
        return;
    };
    let item_count = phase.items.len();
    if last_phase_items.insert(view.retained_view_entity, item_count) != Some(item_count) {
        info!(
            view = ?view.retained_view_entity,
            receivers = item_count,
            incomplete = receivers.incomplete_views.contains(&view.retained_view_entity),
            identity_uniforms = world.contains_resource::<IdentityBindGroup>(),
            "GI receiver raster phase"
        );
    }
    // Eligibility is per pixel. One pending specialization must not black out
    // the complete view while its other receivers have valid identities.
    let pipelines_ready = !phase.items.is_empty()
        && phase
            .items
            .values()
            .all(|item| pipelines.get_render_pipeline(item.pipeline).is_some())
        && (phase.items.is_empty() || world.contains_resource::<IdentityBindGroup>());
    let attachments = [Some(RenderPassColorAttachment {
        view: &identity.view,
        depth_slice: None,
        resolve_target: None,
        ops: Operations {
            load: LoadOp::Clear(LinearRgba::NONE.into()),
            store: StoreOp::Store,
        },
    })];
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let span = diagnostics.time_span(ctx.command_encoder(), "gi_receiver_identity");
    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("gi_receiver_identity"),
        color_attachments: &attachments,
        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
            view: &depth.attachment.texture.default_view,
            depth_ops: Some(Operations {
                load: LoadOp::Load,
                store: StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    if let Some(viewport) = camera.viewport.as_ref() {
        pass.set_camera_viewport(viewport);
    }
    if !pipelines_ready {
        drop(pass);
        span.end(ctx.command_encoder());
        return;
    }
    if let Err(error) = phase.render(&mut pass, world, entity) {
        error!("GI receiver identities: {error}");
        drop(pass);
        span.end(ctx.command_encoder());
        return;
    }
    identity
        .ready
        .store(true, std::sync::atomic::Ordering::Relaxed);
    drop(pass);
    span.end(ctx.command_encoder());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identities_keep_generation_and_exclude_baked_unlit_receivers() {
        let mut world = World::new();
        let entity = world.spawn(RadianceCascadesGeometry).id();
        assert!(
            world
                .entity(entity)
                .contains::<bevy::render::sync_world::SyncToRenderWorld>()
        );
        let lit = StandardMaterial::default();
        let first = identity(entity, true, false, Some(&lit), Some(17)).value;
        assert_eq!(
            u64::from(first.x) | (u64::from(first.y) << 32),
            entity.to_bits()
        );
        assert_eq!(first.z, 1);
        assert_eq!(first.w, 17);
        assert_eq!(
            identity(entity, true, true, Some(&lit), Some(17)).value.z,
            0
        );
        assert_eq!(
            identity(entity, false, false, Some(&lit), Some(17)).value.z,
            0
        );
        assert_eq!(identity(entity, true, false, None, Some(17)).value.z, 0);
        let unlit = StandardMaterial {
            unlit: true,
            ..default()
        };
        assert_eq!(
            identity(entity, true, false, Some(&unlit), Some(17))
                .value
                .z,
            0
        );
        world.despawn(entity);
        let replacement = world.spawn_empty().id();
        let replacement = identity(replacement, true, false, Some(&lit), Some(23)).value;
        assert_ne!(first.xy(), replacement.xy());
    }

    #[test]
    fn packed_geometric_normal_shader_is_valid_wgsl() {
        // A Windows checkout may convert the shader to CRLF.
        let source = include_str!("../shaders/receiver.wesl").replace("\r\n", "\n");
        let start = source
            .find("fn pack_geometric_normal(")
            .expect("normal packer");
        let end = start + source[start..].find("\n}\n").expect("normal packer end") + 2;
        let source = format!(
            "{}\n{}",
            &source[start..end],
            concat!(
                "@fragment fn fragment(@location(0) position: vec3<f32>) -> @location(0) u32 {\n",
                "    return pack_geometric_normal(position);\n",
                "}",
            ),
        );
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("normal packer must validate as a fragment shader");
    }
}

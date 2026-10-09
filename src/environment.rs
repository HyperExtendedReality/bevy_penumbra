//! Validated latitude-longitude environment radiance sources.
use bevy::{
    asset::AssetEvent,
    ecs::message::MessageCursor,
    prelude::*,
    render::{
        RenderApp,
        extract_component::ExtractComponent,
        extract_resource::ExtractResource,
        render_resource::{TextureDimension, TextureFormat},
    },
};
use std::collections::{HashMap, HashSet};

/// Unfiltered HDR latitude-longitude radiance used by Split Radiance Cascades.
/// The north pole is +Y, the horizontal seam is -X, and increasing U points
/// toward +Z. Rotation follows Bevy's `EnvironmentMapLight` convention.
/// The image must retain MAIN_WORLD pixels for validation and RENDER_WORLD data
/// for cascade radiance lookup. Prefiltered diffuse/specular cubemaps are not
/// source radiance.
#[derive(Component, Clone, Debug, PartialEq, ExtractComponent)]
#[extract_app(RenderApp)]
pub struct RadianceCascadesEnvironment {
    /// CPU-readable, single-layer Rgba16Float or Rgba32Float HDR image.
    pub image: Handle<Image>,
    /// World-space orientation of the environment.
    pub rotation: Quat,
    /// Nonnegative multiplier of source radiance.
    pub intensity: f32,
}

impl Default for RadianceCascadesEnvironment {
    fn default() -> Self {
        Self {
            image: Handle::default(),
            rotation: Quat::IDENTITY,
            intensity: 1.0,
        }
    }
}

impl RadianceCascadesEnvironment {
    pub(crate) fn validated(&self) -> Result<(Quat, f32), &'static str> {
        if !self.rotation.is_finite()
            || self.rotation.length_squared() <= 0.0
            || !self.intensity.is_finite()
            || self.intensity < 0.0
        {
            return Err(
                "GI environment requires a finite nonzero rotation and nonnegative finite intensity",
            );
        }
        Ok((self.rotation.normalize().inverse(), self.intensity))
    }
}

#[derive(Clone)]
pub(crate) struct EnvironmentTable {
    pub width: u32,
    pub height: u32,
    pub revision: u64,
}

#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub(crate) struct EnvironmentTables {
    /// Presence means this source image passed validation and is ready for use.
    pub tables: HashMap<AssetId<Image>, EnvironmentTable>,
    pub errors: HashMap<AssetId<Image>, String>,
    pub revision: u64,
}

fn validate_image(image: &Image, revision: u64) -> Result<EnvironmentTable, String> {
    let descriptor = &image.texture_descriptor;
    let width = descriptor.size.width;
    let height = descriptor.size.height;
    if descriptor.dimension != TextureDimension::D2
        || descriptor.size.depth_or_array_layers != 1
        || descriptor.sample_count != 1
        || width == 0
        || height == 0
        || !matches!(
            descriptor.format,
            TextureFormat::Rgba16Float | TextureFormat::Rgba32Float
        )
        || image.texture_view_descriptor.as_ref().is_some_and(|view| {
            view.base_mip_level != 0
                || view.base_array_layer != 0
                || view.dimension.is_some_and(|dimension| {
                    dimension != bevy::render::render_resource::TextureViewDimension::D2
                })
                || view.array_layer_count.is_some_and(|count| count != 1)
                || view
                    .format
                    .is_some_and(|format| format != descriptor.format)
        })
    {
        return Err("GI environment requires an unfiltered, single-layer Rgba16Float/Rgba32Float 2D view at mip zero".into());
    }

    let stride = descriptor
        .format
        .block_copy_size(None)
        .ok_or("unsupported HDR format")? as usize;
    let pixel_count = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|count| usize::try_from(count).ok())
        .ok_or("GI environment dimensions exceed addressable memory")?;
    let required_bytes = pixel_count
        .checked_mul(stride)
        .ok_or("GI environment pixel data size overflows addressable memory")?;
    if image
        .data
        .as_ref()
        .is_none_or(|data| data.len() < required_bytes)
    {
        return Err("GI environment requires complete MAIN_WORLD HDR pixel data".into());
    }

    for y in 0..height {
        for x in 0..width {
            let color = image
                .get_color_at(x, y)
                .map_err(|error| format!("GI HDR decode: {error}"))?
                .to_linear()
                .to_vec4();
            if !color.is_finite() || color.truncate().min_element() < 0.0 {
                return Err(
                    "GI environment pixels must contain finite nonnegative radiance".into(),
                );
            }
        }
    }

    Ok(EnvironmentTable {
        width,
        height,
        revision,
    })
}
#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use bevy::asset::RenderAssetUsages;
    use bevy::render::render_resource::Extent3d;

    #[test]
    fn environment_updates_only_for_its_source_asset_events() {
        let mut app = App::new();
        app.init_resource::<Assets<Image>>()
            .init_resource::<EnvironmentTables>()
            .add_message::<AssetEvent<Image>>()
            .add_systems(Update, update_environment_tables);
        let image = Image::new(
            Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            [1.0f32; 4].into_iter().flat_map(f32::to_le_bytes).collect(),
            TextureFormat::Rgba32Float,
            RenderAssetUsages::MAIN_WORLD,
        );
        let source = app
            .world_mut()
            .resource_mut::<Assets<Image>>()
            .add(image.clone());
        let other = app.world_mut().resource_mut::<Assets<Image>>().add(image);
        app.world_mut().spawn(RadianceCascadesEnvironment {
            image: source.clone(),
            ..default()
        });
        app.update();
        let tables = app.world().resource::<EnvironmentTables>();
        let revision = tables.revision;
        assert_eq!(tables.tables[&source.id()].width, 1);
        assert_eq!(tables.tables[&source.id()].height, 1);
        // Untracked source edits must not be detected by another image's event.
        app.world_mut()
            .resource_mut::<Assets<Image>>()
            .get_mut_untracked(source.id())
            .unwrap()
            .data
            .as_mut()
            .unwrap()[..4]
            .copy_from_slice(&2.0f32.to_le_bytes());
        app.world_mut()
            .resource_mut::<Messages<AssetEvent<Image>>>()
            .write(AssetEvent::Modified { id: other.id() });
        app.update();
        assert_eq!(
            app.world().resource::<EnvironmentTables>().revision,
            revision
        );
        app.world_mut()
            .resource_mut::<Messages<AssetEvent<Image>>>()
            .write(AssetEvent::Modified { id: source.id() });
        app.update();
        assert_eq!(
            app.world().resource::<EnvironmentTables>().revision,
            revision + 1
        );
    }
}

pub(crate) fn update_environment_tables(
    environments: Query<&RadianceCascadesEnvironment>,
    images: Option<Res<Assets<Image>>>,
    events: Option<Res<Messages<AssetEvent<Image>>>>,
    mut cursor: Local<MessageCursor<AssetEvent<Image>>>,
    mut tables: ResMut<EnvironmentTables>,
) {
    let Some(images) = images else {
        return;
    };
    let changed: Option<HashSet<_>> = events.as_ref().map(|events| {
        cursor
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
    let used: HashSet<_> = environments
        .iter()
        .map(|environment| environment.image.id())
        .collect();
    tables.tables.retain(|id, _| used.contains(id));
    tables.errors.retain(|id, _| used.contains(id));
    for id in used {
        if (tables.tables.contains_key(&id) || tables.errors.contains_key(&id))
            && changed
                .as_ref()
                .map_or(!images.is_changed(), |changed| !changed.contains(&id))
        {
            continue;
        }
        let Some(image) = images.get(id) else {
            tables.tables.remove(&id);
            tables.errors.remove(&id);
            continue;
        };
        let revision = tables.revision.wrapping_add(1);
        match validate_image(image, revision) {
            Ok(table) => {
                tables.tables.insert(id, table);
                tables.errors.remove(&id);
                tables.revision = revision;
            }
            Err(reason) => {
                tables.tables.remove(&id);
                if tables.errors.get(&id) != Some(&reason) {
                    warn!(?id, "{reason}");
                }
                tables.errors.insert(id, reason);
            }
        }
    }
}

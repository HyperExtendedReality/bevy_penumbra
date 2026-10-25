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
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

const ALIAS_RECORDS_PER_LIGHT: usize = 6;
const UNIFORM_SPHERE_MIX: f64 = 0.01;

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
    /// Per-texel alias threshold, alias index, and discrete PMF.
    pub aliases: Arc<Vec<UVec4>>,
}

impl EnvironmentTable {
    /// Append packed alias rows after the view's ordinary lights and layer masks.
    /// Returns the starting offset in `GiLight` records.
    pub(crate) fn append_alias_lights(&self, lights: &mut Vec<crate::scene::GiLight>) -> u32 {
        let offset = lights.len() as u32;
        let light_count = self.aliases.len().div_ceil(ALIAS_RECORDS_PER_LIGHT);
        lights.reserve(light_count);
        for records in self.aliases.chunks(ALIAS_RECORDS_PER_LIGHT) {
            let row = |index: usize| records.get(index).copied().unwrap_or(UVec4::ZERO);
            let floats = |word: UVec4| Vec4::from_array(word.to_array().map(f32::from_bits));
            lights.push(crate::scene::GiLight {
                position: floats(row(0)),
                color: floats(row(1)),
                direction: floats(row(2)),
                edge1: floats(row(3)),
                edge2: floats(row(4)),
                source: row(5),
            });
        }
        offset
    }
}

#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub(crate) struct EnvironmentTables {
    /// Presence means this source image passed validation and is ready for use.
    pub tables: HashMap<AssetId<Image>, EnvironmentTable>,
    pub errors: HashMap<AssetId<Image>, String>,
    pub revision: u64,
}

pub(crate) fn validate_image(image: &Image, revision: u64) -> Result<EnvironmentTable, String> {
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
    if pixel_count > u32::MAX as usize
        || pixel_count
            .checked_mul(std::mem::size_of::<UVec4>())
            .is_none()
    {
        return Err("GI environment texel count exceeds the alias table index range".into());
    }
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

    let mut radiance_weights = Vec::new();
    radiance_weights
        .try_reserve_exact(pixel_count)
        .map_err(|error| format!("GI environment alias allocation failed: {error}"))?;
    let mut solid_angle_total = 0.0f64;
    for y in 0..height {
        let theta0 = std::f64::consts::PI * f64::from(y) / f64::from(height);
        let theta1 = std::f64::consts::PI * f64::from(y + 1) / f64::from(height);
        let row_solid_angle =
            (std::f64::consts::TAU / f64::from(width)) * (theta0.cos() - theta1.cos());
        solid_angle_total += row_solid_angle * f64::from(width);
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
            let rgb = color.truncate().as_dvec3();
            let luminance = rgb.dot(bevy::math::DVec3::new(0.2126, 0.7152, 0.0722));
            radiance_weights.push(luminance * row_solid_angle);
        }
    }

    let radiance_total: f64 = radiance_weights.iter().sum();
    if !solid_angle_total.is_finite()
        || solid_angle_total <= 0.0
        || !radiance_total.is_finite()
        || radiance_total < 0.0
    {
        return Err("GI environment sampling weights are not finite and positive".into());
    }
    for (index, radiance_weight) in radiance_weights.iter_mut().enumerate() {
        let row = index / width as usize;
        let theta0 = std::f64::consts::PI * row as f64 / f64::from(height);
        let theta1 = std::f64::consts::PI * (row + 1) as f64 / f64::from(height);
        let solid_angle =
            (std::f64::consts::TAU / f64::from(width)) * (theta0.cos() - theta1.cos());
        let uniform_solid_angle = solid_angle / solid_angle_total;
        let probability = if radiance_total > 0.0 {
            (1.0 - UNIFORM_SPHERE_MIX) * (*radiance_weight / radiance_total)
                + UNIFORM_SPHERE_MIX * uniform_solid_angle
        } else {
            uniform_solid_angle
        };
        *radiance_weight = probability;
    }
    let aliases = build_alias_records(radiance_weights)?;

    Ok(EnvironmentTable {
        width,
        height,
        revision,
        aliases: Arc::new(aliases),
    })
}

fn build_alias_records(mut weights: Vec<f64>) -> Result<Vec<UVec4>, String> {
    if weights.is_empty() || weights.len() > u32::MAX as usize {
        return Err("GI environment texel count exceeds the alias table index range".into());
    }
    let total: f64 = weights.iter().sum();
    if !total.is_finite()
        || total <= 0.0
        || weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight <= 0.0)
    {
        return Err("GI environment alias weights must be finite and positive".into());
    }

    let count = weights.len();
    let mut records = Vec::new();
    records
        .try_reserve_exact(count)
        .map_err(|error| format!("GI environment alias allocation failed: {error}"))?;
    for (index, weight) in weights.iter_mut().enumerate() {
        let pmf = (*weight / total) as f32;
        if !pmf.is_finite() || pmf <= 0.0 {
            return Err("GI environment alias probabilities cannot be represented as f32".into());
        }
        records.push(UVec4::new(1.0f32.to_bits(), index as u32, pmf.to_bits(), 0));
        *weight = (*weight / total) * count as f64;
    }

    let mut worklist = Vec::new();
    worklist
        .try_reserve_exact(count)
        .map_err(|error| format!("GI environment alias allocation failed: {error}"))?;
    worklist.resize(count, 0u32);
    let mut small_end = 0;
    let mut large_start = count;
    for (index, &probability) in weights.iter().enumerate() {
        if probability < 1.0 {
            worklist[small_end] = index as u32;
            small_end += 1;
        } else {
            large_start -= 1;
            worklist[large_start] = index as u32;
        }
    }
    while small_end > 0 && large_start < count {
        small_end -= 1;
        let small = worklist[small_end] as usize;
        let large = worklist[large_start] as usize;
        large_start += 1;

        records[small].x = (weights[small] as f32).clamp(0.0, 1.0).to_bits();
        records[small].y = large as u32;
        weights[large] = (weights[large] + weights[small]) - 1.0;
        if weights[large] < 1.0 {
            worklist[small_end] = large as u32;
            small_end += 1;
        } else {
            large_start -= 1;
            worklist[large_start] = large as u32;
        }
    }
    Ok(records)
}
#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use bevy::asset::RenderAssetUsages;
    use bevy::render::render_resource::Extent3d;

    fn hdr_image(width: u32, height: u32, pixels: &[[f32; 4]]) -> Image {
        Image::new(
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            pixels
                .iter()
                .flat_map(|pixel| pixel.iter().flat_map(|channel| channel.to_le_bytes()))
                .collect(),
            TextureFormat::Rgba32Float,
            RenderAssetUsages::MAIN_WORLD,
        )
    }

    fn reconstructed_pmfs(records: &[UVec4]) -> Vec<f64> {
        let mut pmfs = vec![0.0; records.len()];
        let bucket = 1.0 / records.len() as f64;
        for (index, record) in records.iter().enumerate() {
            let threshold = f64::from(f32::from_bits(record.x));
            pmfs[index] += threshold * bucket;
            pmfs[record.y as usize] += (1.0 - threshold) * bucket;
        }
        pmfs
    }

    fn row_solid_angle(width: u32, height: u32, row: u32) -> f64 {
        (std::f64::consts::TAU / f64::from(width))
            * ((std::f64::consts::PI * f64::from(row) / f64::from(height)).cos()
                - (std::f64::consts::PI * f64::from(row + 1) / f64::from(height)).cos())
    }

    #[test]
    fn environment_alias_pmf_is_spherical_and_packed_in_light_rows() {
        let width = 2;
        let height = 4;
        let pixels = vec![[1.0, 1.0, 1.0, 1.0]; (width * height) as usize];
        let table = validate_image(&hdr_image(width, height, &pixels), 1).unwrap();
        let reconstructed = reconstructed_pmfs(&table.aliases);
        let solid_angle_total = (0..height)
            .map(|row| row_solid_angle(width, height, row) * f64::from(width))
            .sum::<f64>();

        assert!((solid_angle_total - 4.0 * std::f64::consts::PI).abs() < 1e-12);
        assert!((reconstructed.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        for row in 0..height {
            let omega = row_solid_angle(width, height, row);
            for column in 0..width {
                let index = (row * width + column) as usize;
                let stored_pmf = f64::from(f32::from_bits(table.aliases[index].z));
                assert!((reconstructed[index] - stored_pmf).abs() < 1e-6);
                assert!((stored_pmf / omega - 1.0 / solid_angle_total).abs() < 1e-8);
            }
        }
        assert!(f32::from_bits(table.aliases[0].z) < f32::from_bits(table.aliases[2].z));

        let mut lights = vec![crate::scene::GiLight::default()];
        let offset = table.append_alias_lights(&mut lights);
        assert_eq!(offset, 1);
        let first = lights[1];
        assert_eq!(
            first.position.to_array().map(f32::to_bits),
            table.aliases[0].to_array()
        );
        assert_eq!(
            first.color.to_array().map(f32::to_bits),
            table.aliases[1].to_array()
        );
        assert_eq!(
            first.direction.to_array().map(f32::to_bits),
            table.aliases[2].to_array()
        );
        assert_eq!(
            first.edge1.to_array().map(f32::to_bits),
            table.aliases[3].to_array()
        );
        assert_eq!(
            first.edge2.to_array().map(f32::to_bits),
            table.aliases[4].to_array()
        );
        assert_eq!(first.source, table.aliases[5]);
        assert_eq!(
            lights[2].position.to_array().map(f32::to_bits),
            table.aliases[6].to_array()
        );
        assert_eq!(
            lights.len(),
            1 + table.aliases.len().div_ceil(ALIAS_RECORDS_PER_LIGHT)
        );
    }

    #[test]
    fn environment_alias_sampling_keeps_dark_texels_in_support() {
        let width = 2;
        let height = 4;
        let mut pixels = vec![[0.0, 0.0, 0.0, 1.0]; (width * height) as usize];
        pixels[0] = [100.0, 100.0, 100.0, 1.0];
        let bright = validate_image(&hdr_image(width, height, &pixels), 1).unwrap();
        let bright_pmfs = reconstructed_pmfs(&bright.aliases);
        assert!(bright_pmfs[0] > bright_pmfs[1]);
        assert!(bright_pmfs.iter().all(|pmf| *pmf > 0.0));
        for (record, reconstructed) in bright.aliases.iter().zip(bright_pmfs) {
            let stored = f64::from(f32::from_bits(record.z));
            assert!((stored - reconstructed).abs() < 1e-6);
        }

        let black = validate_image(
            &hdr_image(
                width,
                height,
                &vec![[0.0, 0.0, 0.0, 1.0]; (width * height) as usize],
            ),
            2,
        )
        .unwrap();
        let black_pmfs = reconstructed_pmfs(&black.aliases);
        let total_solid_angle = (0..height)
            .map(|row| row_solid_angle(width, height, row) * f64::from(width))
            .sum::<f64>();
        for row in 0..height {
            let expected = row_solid_angle(width, height, row) / total_solid_angle;
            for column in 0..width {
                let index = (row * width + column) as usize;
                assert!((black_pmfs[index] - expected).abs() < 1e-6);
            }
        }
    }

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

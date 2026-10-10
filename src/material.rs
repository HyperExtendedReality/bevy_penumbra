//! Portable paged texture storage shared by Split Radiance Cascades.
use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
    sync::Arc,
};

use bevy::{
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    mesh::VertexAttributeValues,
    prelude::*,
    render::render_resource::{TextureDimension, TextureFormat},
};

pub(crate) const NO_TEXTURE: u32 = u32::MAX;
pub(crate) const MATERIAL_WORDS: usize = 8;

pub(crate) fn material_support(material: &StandardMaterial) -> Result<(), &'static str> {
    if !matches!(material.alpha_mode, AlphaMode::Opaque | AlphaMode::Mask(_)) {
        return Err("Split Radiance Cascades supports only opaque and alpha-mask materials");
    }
    if material.depth_bias != 0.0 {
        return Err("depth bias changes raster receiver depth without changing cascade surfaces");
    }
    if material.depth_map.is_some() {
        return Err("parallax depth maps require displaced cascade surface coordinates");
    }
    if material.unlit {
        return Ok(());
    }
    if material.diffuse_transmission != 0.0 || material.specular_transmission != 0.0 {
        return Err("transmission is not supported by diffuse Split Radiance Cascades");
    }
    // These optional Bevy fields may be enabled through feature unification.
    // Cascade transport is diffuse; transmission still changes visibility.
    for field in [
        "diffuse_transmission_texture",
        "specular_transmission_texture",
        "thickness_texture",
    ] {
        if Struct::field(material, field)
            .and_then(|value| value.try_downcast_ref::<Option<Handle<Image>>>())
            .is_some_and(Option::is_some)
        {
            return Err(
                "transmission textures are not supported by diffuse Split Radiance Cascades",
            );
        }
    }
    Ok(())
}

pub(crate) fn material_requires_tangents(material: &StandardMaterial) -> bool {
    !material.unlit && material.normal_map_texture.is_some()
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn material_mesh_support(
    material: &StandardMaterial,
    mesh: &Mesh,
) -> Result<(), &'static str> {
    material_mesh_layout_support(material, mesh)?;
    if let Some(VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute(Mesh::ATTRIBUTE_POSITION)
    {
        crate::scene::validate_mesh_vertices(mesh, positions)?;
    }
    Ok(())
}

pub(crate) fn material_mesh_layout_support(
    material: &StandardMaterial,
    mesh: &Mesh,
) -> Result<(), &'static str> {
    material_support(material)?;
    if mesh.primitive_topology() != bevy::render::render_resource::PrimitiveTopology::TriangleList {
        return Err("GI requires TriangleList meshes");
    }
    let vertex_count = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        Some(VertexAttributeValues::Float32x3(positions)) if !positions.is_empty() => {
            positions.len()
        }
        _ => return Err("GI requires Float32x3 positions"),
    };
    if material_requires_tangents(material) {
        if !matches!(
            mesh.attribute(Mesh::ATTRIBUTE_NORMAL),
            Some(VertexAttributeValues::Float32x3(normals)) if normals.len() == vertex_count
        ) {
            return Err("normal-mapped materials require Float32x3 normals for every vertex");
        }
        if !matches!(
            mesh.attribute(Mesh::ATTRIBUTE_TANGENT),
            Some(VertexAttributeValues::Float32x4(tangents)) if tangents.len() == vertex_count
        ) {
            return Err("normal-mapped materials require Float32x4 tangents for every vertex");
        }
        let uv = match material.normal_map_channel {
            bevy::mesh::UvChannel::Uv0 => Mesh::ATTRIBUTE_UV_0,
            bevy::mesh::UvChannel::Uv1 => Mesh::ATTRIBUTE_UV_1,
        };
        if !matches!(
            mesh.attribute(uv),
            Some(VertexAttributeValues::Float32x2(uvs)) if uvs.len() == vertex_count
        ) {
            return Err("normal-mapped materials require their selected Float32x2 UV channel");
        }
    }
    Ok(())
}

pub(crate) fn texture_handles(material: &StandardMaterial) -> [Option<&Handle<Image>>; 5] {
    if material.unlit {
        return [material.base_color_texture.as_ref(), None, None, None, None];
    }
    [
        material.base_color_texture.as_ref(),
        material.emissive_texture.as_ref(),
        material.metallic_roughness_texture.as_ref(),
        material.normal_map_texture.as_ref(),
        // Artistic AO is read from Bevy's G-buffer only at final composition.
        // It must not consume GI-atlas space or gate cascade residency.
        None,
    ]
}

pub(crate) fn material_words(
    material: &StandardMaterial,
    atlas: &TextureAtlas,
    images: Option<&Assets<Image>>,
) -> Result<[UVec4; MATERIAL_WORDS], String> {
    use bevy::mesh::UvChannel;
    material_support(material)?;
    let base = material.base_color.to_linear().to_vec4();
    let cutoff = match material.alpha_mode {
        AlphaMode::Opaque => -1.0,
        AlphaMode::Mask(cutoff) => cutoff,
        _ => {
            return Err(
                "Split Radiance Cascades supports only opaque and alpha-mask materials".into(),
            );
        }
    };
    let emission = if material.unlit {
        Vec4::ZERO
    } else {
        material.emissive.to_vec4()
    }
    .with_w(cutoff);
    let properties = if material.unlit {
        Vec4::ZERO
    } else {
        Vec4::new(
            material.metallic,
            material.perceptual_roughness,
            material.reflectance,
            material.emissive_exposure_weight,
        )
    };
    let tint = if material.unlit {
        Vec4::ONE
    } else {
        material.specular_tint.to_linear().to_vec4()
    };
    let transform = material.uv_transform;
    let uv_x = Vec4::new(
        transform.matrix2.x_axis.x,
        transform.matrix2.y_axis.x,
        transform.translation.x,
        0.0,
    );
    let uv_y = Vec4::new(
        transform.matrix2.x_axis.y,
        transform.matrix2.y_axis.y,
        transform.translation.y,
        0.0,
    );
    if [base, emission, properties, tint, uv_x, uv_y]
        .iter()
        .any(|v| !v.is_finite())
    {
        return Err("material values and UV transforms must be finite".into());
    }
    let mut textures = [NO_TEXTURE; 5];
    for (i, handle) in texture_handles(material).into_iter().enumerate() {
        if let Some(handle) = handle {
            textures[i] = *atlas
                .textures
                .get(&handle.id())
                .ok_or("material texture is not in the committed atlas")?;
        }
    }
    let channels = [
        &material.base_color_channel,
        &material.emissive_channel,
        &material.metallic_roughness_channel,
        &material.normal_map_channel,
        &material.occlusion_channel,
    ]
    .into_iter()
    .enumerate()
    .fold(0u32, |bits, (i, channel)| {
        bits | (u32::from(channel == &UvChannel::Uv1) << i)
    });
    let two_component = material
        .normal_map_texture
        .as_ref()
        .and_then(|handle| images.and_then(|images| images.get(handle.id())))
        .is_some_and(|image| {
            matches!(
                image.texture_descriptor.format,
                TextureFormat::Rg8Unorm
                    | TextureFormat::Rg16Unorm
                    | TextureFormat::Bc5RgUnorm
                    | TextureFormat::EacRg11Unorm
            )
        });
    let normal_flags = u32::from(material.flip_normal_map_y) | (u32::from(two_component) << 1);
    Ok([
        UVec4::from_array(base.to_array().map(f32::to_bits)),
        UVec4::from_array(emission.to_array().map(f32::to_bits)),
        UVec4::from_array(properties.to_array().map(f32::to_bits)),
        UVec4::from_array(tint.to_array().map(f32::to_bits)),
        UVec4::from_array(uv_x.to_array().map(f32::to_bits)),
        UVec4::from_array(uv_y.to_array().map(f32::to_bits)),
        UVec4::new(textures[0], textures[1], textures[2], textures[3]),
        UVec4::new(textures[4], channels, normal_flags, 0),
    ])
}

#[derive(Clone, Default)]
pub(crate) struct TextureAtlas {
    pub side: u32,
    pub pages: Vec<Vec<u8>>,
    pub words: Vec<UVec4>,
    pub textures: HashMap<AssetId<Image>, u32>,
}

#[derive(Resource, Clone)]
pub(crate) struct TextureOptions {
    pub sampler: ImageSamplerDescriptor,
    pub budget: usize,
}

impl Default for TextureOptions {
    fn default() -> Self {
        Self {
            sampler: ImageSamplerDescriptor::linear(),
            budget: 256 * 1024 * 1024,
        }
    }
}

#[derive(Default)]
pub(crate) struct TextureCache {
    keys: Vec<(AssetId<Image>, u64)>,
    pub atlas: Arc<TextureAtlas>,
    failed: Option<(Vec<AssetId<Image>>, String)>,
    /// All textures are resident but the atlas still can't be built
    /// (budget, format, missing pixel data), so waiting won't help.
    pub permanent_failure: bool,
}

fn address_mode(mode: ImageAddressMode) -> Result<u32, &'static str> {
    match mode {
        ImageAddressMode::ClampToEdge => Ok(0),
        ImageAddressMode::Repeat => Ok(1),
        ImageAddressMode::MirrorRepeat => Ok(2),
        ImageAddressMode::ClampToBorder => {
            Err("clamp-to-border textures require a border-color sampling path")
        }
    }
}

fn sampler(image: &Image, options: &TextureOptions) -> Result<(u32, Vec2), &'static str> {
    let descriptor = match &image.sampler {
        ImageSampler::Default => &options.sampler,
        ImageSampler::Descriptor(descriptor) => descriptor,
    };
    if descriptor.compare.is_some() || descriptor.anisotropy_clamp > 1 {
        return Err(
            "comparison and anisotropic samplers are not supported by cascade texture footprints",
        );
    }
    let limits = Vec2::new(descriptor.lod_min_clamp, descriptor.lod_max_clamp);
    if !limits.is_finite() || limits.x < 0.0 || limits.y < limits.x {
        return Err("invalid sampler LOD limits");
    }
    let flags = address_mode(descriptor.address_mode_u)?
        | (address_mode(descriptor.address_mode_v)? << 2)
        | (u32::from(descriptor.mag_filter == ImageFilterMode::Linear) << 4)
        | (u32::from(descriptor.min_filter == ImageFilterMode::Linear) << 5)
        | (u32::from(descriptor.mipmap_filter == ImageFilterMode::Linear) << 6);
    Ok((flags, limits))
}

fn texel_index(index: i32, length: u32, mode: u32) -> u32 {
    let length = length as i32;
    match mode {
        1 => index.rem_euclid(length) as u32,
        2 => {
            let index = index.rem_euclid(length * 2);
            if index < length {
                index as u32
            } else {
                (length * 2 - 1 - index) as u32
            }
        }
        _ => index.clamp(0, length - 1) as u32,
    }
}

struct Mip {
    size: UVec2,
    pixels: Vec<Vec4>,
}

fn decode(image: &Image) -> Result<Vec<Mip>, String> {
    let descriptor = &image.texture_descriptor;
    if descriptor.dimension != TextureDimension::D2
        || descriptor.size.depth_or_array_layers != 1
        || descriptor.sample_count != 1
        || descriptor.size.width == 0
        || descriptor.size.height == 0
    {
        return Err("GI material textures require nonempty, single-sample 2D images".into());
    }
    let data = image
        .data
        .as_ref()
        .ok_or("GI textures currently require MAIN_WORLD pixel data")?;
    let format = image
        .texture_view_descriptor
        .as_ref()
        .and_then(|view| view.format)
        .unwrap_or(descriptor.format);
    if image
        .texture_view_descriptor
        .as_ref()
        .is_some_and(|view| view.base_mip_level != 0 || view.base_array_layer != 0)
    {
        return Err("restricted texture views need their view-relative mip upload path".into());
    }
    let (block_width, block_height) = format.block_dimensions();
    if block_width != 1 || block_height != 1 {
        return Err(format!(
            "compressed texture {format:?} needs the GPU atlas upload path"
        ));
    }
    // Integer texture views are not compatible with StandardMaterial's float bindings.
    let bytes_per_pixel = format
        .block_copy_size(None)
        .ok_or("unsupported texture format")? as usize;
    if format.block_copy_size(None) != descriptor.format.block_copy_size(None) {
        return Err("incompatible source and texture-view pixel formats".into());
    }
    if matches!(
        format,
        TextureFormat::R8Uint
            | TextureFormat::R16Uint
            | TextureFormat::R32Uint
            | TextureFormat::Rg8Uint
            | TextureFormat::Rg16Uint
            | TextureFormat::Rg32Uint
            | TextureFormat::Rgba8Uint
            | TextureFormat::Rgba16Uint
            | TextureFormat::Rgba32Uint
    ) {
        return Err("integer texture views are not StandardMaterial float textures".into());
    }
    let mut mips = Vec::new();
    let mut offset = 0usize;
    let mut size = UVec2::new(descriptor.size.width, descriptor.size.height);
    let full_mip_count = 32 - size.max_element().leading_zeros();
    if descriptor.mip_level_count == 0 || descriptor.mip_level_count > full_mip_count {
        return Err("invalid source mip count".into());
    }
    let view_mips = image
        .texture_view_descriptor
        .as_ref()
        .and_then(|view| view.mip_level_count);
    if view_mips.is_some_and(|count| count == 0 || count > descriptor.mip_level_count) {
        return Err("invalid texture-view mip count".into());
    }
    let target_mips = view_mips.unwrap_or(full_mip_count);
    let source_mips = descriptor.mip_level_count.min(target_mips);
    for level in 0..source_mips {
        let bytes = (size.x as usize)
            .checked_mul(size.y as usize)
            .and_then(|count| count.checked_mul(bytes_per_pixel))
            .ok_or("texture extent overflows")?;
        let end = offset
            .checked_add(bytes)
            .ok_or("mip byte range overflows")?;
        let pixels = data.get(offset..end).ok_or("truncated source mip data")?;
        let mut extent = descriptor.size;
        extent.width = size.x;
        extent.height = size.y;
        let mut view = Image::new_uninit(extent, TextureDimension::D2, format, image.asset_usage);
        view.data = Some(pixels.to_vec());
        view.texture_descriptor.format = format;
        view.texture_descriptor.size.width = size.x;
        view.texture_descriptor.size.height = size.y;
        view.texture_descriptor.mip_level_count = 1;
        let mut decoded = Vec::with_capacity(size.x as usize * size.y as usize);
        for y in 0..size.y {
            for x in 0..size.x {
                let mut color = view
                    .get_color_at(x, y)
                    .map_err(|error| format!("texture decode: {error}"))?
                    .to_linear()
                    .to_vec4();
                // Bevy's Color helper expands R-only formats as grayscale; GPU texture
                // loads use (R,0,0,1), so retain actual channel semantics for materials.
                if matches!(
                    format,
                    TextureFormat::R8Unorm
                        | TextureFormat::R16Unorm
                        | TextureFormat::R16Float
                        | TextureFormat::R32Float
                ) {
                    color.y = 0.0;
                    color.z = 0.0;
                }
                if !color.is_finite() {
                    return Err("nonfinite texture texel".into());
                }
                decoded.push(color);
            }
        }
        mips.push(Mip {
            size,
            pixels: decoded,
        });
        offset = end;
        if level + 1 < source_mips {
            size = (size / 2).max(UVec2::ONE);
        }
    }
    // Preserve supplied mips. Generate missing levels by area averaging in linear
    // space; odd extents include their last row/column rather than discarding it.
    while mips.len() < target_mips as usize {
        let source = mips.last().expect("validated nonzero source mip count");
        let size = (source.size / 2).max(UVec2::ONE);
        let mut pixels = Vec::with_capacity(size.x as usize * size.y as usize);
        for y in 0..size.y {
            for x in 0..size.x {
                let low = Vec2::new(x as f32, y as f32) * source.size.as_vec2() / size.as_vec2();
                let high = Vec2::new((x + 1) as f32, (y + 1) as f32) * source.size.as_vec2()
                    / size.as_vec2();
                let mut color = Vec4::ZERO;
                for sy in low.y.floor() as u32..high.y.ceil() as u32 {
                    for sx in low.x.floor() as u32..high.x.ceil() as u32 {
                        let weight = (high.x.min((sx + 1) as f32) - low.x.max(sx as f32))
                            * (high.y.min((sy + 1) as f32) - low.y.max(sy as f32));
                        color += source.pixels[(sy * source.size.x + sx) as usize] * weight;
                    }
                }
                pixels.push(color / ((high.x - low.x) * (high.y - low.y)));
            }
        }
        mips.push(Mip { size, pixels });
    }
    Ok(mips)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::render::render_resource::Extent3d;
    use bevy::{
        asset::RenderAssetUsages,
        image::{ImageAddressMode, ImageSampler},
    };

    #[test]
    fn normal_maps_require_complete_vertex_frame_for_lit_materials() {
        let texture = Handle::<Image>::default();
        let mesh = Mesh::from(Cuboid::default());
        let line_mesh = Mesh::new(
            bevy::render::render_resource::PrimitiveTopology::LineList,
            RenderAssetUsages::default(),
        );
        assert!(material_mesh_support(&StandardMaterial::default(), &line_mesh).is_err());
        assert!(!material_requires_tangents(&StandardMaterial::default()));
        assert!(material_requires_tangents(&StandardMaterial {
            normal_map_texture: Some(texture.clone()),
            ..default()
        }));
        assert!(
            material_mesh_support(
                &StandardMaterial {
                    normal_map_texture: Some(texture.clone()),
                    ..default()
                },
                &mesh
            )
            .is_err()
        );
        let mut valid = Mesh::from(Rectangle::default());
        valid.insert_attribute(Mesh::ATTRIBUTE_TANGENT, vec![[1.0, 0.0, 0.0, 1.0]; 4]);
        assert!(
            material_mesh_support(
                &StandardMaterial {
                    normal_map_texture: Some(texture.clone()),
                    ..default()
                },
                &valid
            )
            .is_ok()
        );
        let mut missing_uv1 = valid.clone();
        missing_uv1.remove_attribute(Mesh::ATTRIBUTE_UV_1);
        assert!(
            material_mesh_support(
                &StandardMaterial {
                    normal_map_texture: Some(texture.clone()),
                    normal_map_channel: bevy::mesh::UvChannel::Uv1,
                    ..default()
                },
                &missing_uv1
            )
            .is_err()
        );
        let mut missing_normals = valid.clone();
        missing_normals.remove_attribute(Mesh::ATTRIBUTE_NORMAL);
        assert!(
            material_mesh_support(
                &StandardMaterial {
                    normal_map_texture: Some(texture.clone()),
                    ..default()
                },
                &missing_normals
            )
            .is_err()
        );
        let mut incomplete = Mesh::from(Rectangle::default());
        incomplete.insert_attribute(Mesh::ATTRIBUTE_TANGENT, vec![[1.0, 0.0, 0.0, 1.0]]);
        assert!(
            material_mesh_support(
                &StandardMaterial {
                    normal_map_texture: Some(texture.clone()),
                    ..default()
                },
                &incomplete
            )
            .is_err()
        );
        assert!(!material_requires_tangents(&StandardMaterial {
            normal_map_texture: Some(texture),
            unlit: true,
            ..default()
        }));
        assert!(
            material_mesh_support(
                &StandardMaterial {
                    normal_map_texture: Some(Handle::default()),
                    unlit: true,
                    ..default()
                },
                &mesh
            )
            .is_ok()
        );
    }

    #[test]
    fn artistic_occlusion_does_not_require_gi_texture_residency() {
        let mut material = StandardMaterial {
            occlusion_texture: Some(Handle::default()),
            ..default()
        };
        let atlas = TextureAtlas::default();
        assert!(material_words(&material, &atlas, None).is_ok());
        // GI input textures still require committed atlas residency.
        material.base_color_texture = Some(Handle::default());
        assert!(material_words(&material, &atlas, None).is_err());
    }

    #[test]
    fn diffuse_material_support_accepts_approximated_lobes() {
        let mut material = StandardMaterial::default();
        material.clearcoat = 0.5;
        material.anisotropy_strength = 0.5;
        assert!(material_support(&material).is_ok());
        material.specular_tint = Color::linear_rgb(0.2, 0.6, 0.9);
        assert!(material_support(&material).is_ok());

        for field in [
            "specular_texture",
            "specular_tint_texture",
            "clearcoat_texture",
            "clearcoat_roughness_texture",
            "clearcoat_normal_texture",
            "anisotropy_texture",
        ] {
            let set = if let Some(texture) = Struct::field_mut(&mut material, field)
                .and_then(|value| value.try_downcast_mut::<Option<Handle<Image>>>())
            {
                *texture = Some(Handle::default());
                true
            } else {
                false
            };
            if set {
                assert!(material_support(&material).is_ok(), "{field}");
                if let Some(texture) = Struct::field_mut(&mut material, field)
                    .and_then(|value| value.try_downcast_mut::<Option<Handle<Image>>>())
                {
                    *texture = None;
                }
            }
        }

        for field in [
            "diffuse_transmission_texture",
            "specular_transmission_texture",
            "thickness_texture",
        ] {
            let set = if let Some(texture) = Struct::field_mut(&mut material, field)
                .and_then(|value| value.try_downcast_mut::<Option<Handle<Image>>>())
            {
                *texture = Some(Handle::default());
                true
            } else {
                false
            };
            if set {
                assert!(material_support(&material).is_err(), "{field}");
                if let Some(texture) = Struct::field_mut(&mut material, field)
                    .and_then(|value| value.try_downcast_mut::<Option<Handle<Image>>>())
                {
                    *texture = None;
                }
            }
        }

        material = StandardMaterial::default();
        material.diffuse_transmission = 0.5;
        assert!(material_support(&material).is_err());
        material.diffuse_transmission = 0.0;
        material.depth_bias = 0.1;
        assert!(material_support(&material).is_err());
        material.depth_bias = 0.0;
        material.depth_map = Some(Handle::default());
        assert!(material_support(&material).is_err());
        material.depth_map = None;
        material.alpha_mode = AlphaMode::Blend;
        assert!(material_support(&material).is_err());
    }

    fn rgba32(size: UVec2, pixels: &[Vec4]) -> Image {
        Image::new(
            Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            pixels
                .iter()
                .flat_map(|v| v.to_array())
                .flat_map(f32::to_le_bytes)
                .collect(),
            TextureFormat::Rgba32Float,
            RenderAssetUsages::default(),
        )
    }

    fn pixel(atlas: &TextureAtlas, location: UVec3) -> Vec4 {
        let offset = ((location.y * atlas.side + location.x) * 16) as usize;
        Vec4::from_array(std::array::from_fn(|i| {
            f32::from_le_bytes(
                atlas.pages[location.z as usize][offset + i * 4..offset + i * 4 + 4]
                    .try_into()
                    .unwrap(),
            )
        }))
    }

    #[test]
    fn atlas_preserves_source_mips_tiles_wrap_gutters_and_atomic_failure() {
        let mut images = Assets::<Image>::default();
        let size = UVec2::new(530, 17);
        let pixels: Vec<_> = (0..size.x * size.y)
            .map(|i| {
                Vec4::new(
                    (i % size.x) as f32 + 0.125,
                    (i / size.x) as f32 + 0.25,
                    -0.5,
                    1.0,
                )
            })
            .collect();
        let mut image = rgba32(size, &pixels);
        // An authored second mip must survive rather than be regenerated.
        image.texture_descriptor.mip_level_count = 2;
        let authored = Vec4::new(0.25, 0.5, 0.75, 0.5);
        image.data.as_mut().unwrap().extend(
            (0..265 * 8)
                .flat_map(|_| authored.to_array())
                .flat_map(f32::to_le_bytes),
        );
        image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
            address_mode_u: ImageAddressMode::Repeat,
            address_mode_v: ImageAddressMode::MirrorRepeat,
            ..ImageSamplerDescriptor::linear()
        });
        let handle = images.add(image);
        let mut cache = TextureCache::default();
        let options = TextureOptions::default();
        cache
            .update(vec![handle.id()], Some(&images), &options, true)
            .unwrap();
        let atlas = cache.atlas.clone();
        assert_eq!(atlas.side, 512);
        let texture = atlas.textures[&handle.id()];
        let header = atlas.words[texture as usize];
        assert_eq!(header.y, 10);
        assert_eq!(header.z & 15, 1 | (2 << 2));
        let mip = atlas.words[header.x as usize];
        assert_eq!(mip.z, 2);
        for y in 0..size.y {
            for x in 0..size.x {
                let tile = x / 510;
                let rect = atlas.words[(mip.w + tile) as usize];
                assert_eq!(
                    pixel(
                        &atlas,
                        UVec3::new(rect.x + 1 + x % 510, rect.y + 1 + y, rect.z)
                    ),
                    pixels[(y * size.x + x) as usize]
                );
            }
        }
        let first = atlas.words[mip.w as usize];
        assert_eq!(
            pixel(&atlas, UVec3::new(first.x, first.y + 1, first.z)),
            pixels[529]
        );
        assert_eq!(
            pixel(&atlas, UVec3::new(first.x + 1, first.y, first.z)),
            pixels[0]
        );
        let second = atlas.words[(mip.w + 1) as usize];
        assert_eq!(
            pixel(&atlas, UVec3::new(second.x, second.y + 1, second.z)),
            pixels[509]
        );
        for level in 1..header.y {
            let mip = atlas.words[(header.x + level) as usize];
            let rect = atlas.words[mip.w as usize];
            assert!(
                (pixel(&atlas, UVec3::new(rect.x + 1, rect.y + 1, rect.z)) - authored)
                    .abs()
                    .max_element()
                    < 1e-6
            );
        }
        cache
            .update(vec![handle.id()], Some(&images), &options, false)
            .unwrap();
        assert!(Arc::ptr_eq(&atlas, &cache.atlas));
        let too_small = TextureOptions {
            budget: 1,
            ..options.clone()
        };
        assert!(
            cache
                .update(vec![handle.id()], Some(&images), &too_small, true)
                .is_err()
        );
        assert!(Arc::ptr_eq(&atlas, &cache.atlas));
        images.get_mut(&handle).unwrap().data = None;
        assert!(
            cache
                .update(vec![handle.id()], Some(&images), &options, true)
                .is_err()
        );
        assert!(Arc::ptr_eq(&atlas, &cache.atlas));
        cache.update(vec![], Some(&images), &options, true).unwrap();
        assert!(cache.atlas.pages.is_empty() && cache.atlas.textures.is_empty());

        let odd = rgba32(UVec2::new(3, 1), &[Vec4::ZERO, Vec4::ONE, Vec4::splat(2.0)]);
        assert_eq!(decode(&odd).unwrap()[1].pixels, vec![Vec4::ONE]);
        let mut restricted = odd.clone();
        restricted.texture_view_descriptor =
            Some(bevy::render::render_resource::TextureViewDescriptor {
                mip_level_count: Some(1),
                ..default()
            });
        assert_eq!(
            decode(&restricted).unwrap().len(),
            1,
            "explicit views restrict the mip chain"
        );
        restricted
            .texture_view_descriptor
            .as_mut()
            .unwrap()
            .mip_level_count = Some(2);
        assert!(
            decode(&restricted).is_err(),
            "view cannot expose absent source mips"
        );
        let gray = Image::new(
            Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            vec![128],
            TextureFormat::R8Unorm,
            RenderAssetUsages::default(),
        );
        assert_eq!(
            decode(&gray).unwrap()[0].pixels,
            vec![Vec4::new(128.0 / 255.0, 0.0, 0.0, 1.0)]
        );
        let srgb = Image::new(
            Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            vec![128, 64, 32, 127],
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::default(),
        );
        let expected = Color::srgba(128.0 / 255.0, 64.0 / 255.0, 32.0 / 255.0, 127.0 / 255.0)
            .to_linear()
            .to_vec4();
        assert_eq!(decode(&srgb).unwrap()[0].pixels, vec![expected]);
    }

    #[test]
    fn alpha_sampling_tracks_mask_texture_edits() {
        let mut images = Assets::<Image>::default();
        let mut image = rgba32(
            UVec2::new(2, 1),
            &[Vec4::new(1.0, 1.0, 1.0, 0.0), Vec4::new(1.0, 1.0, 1.0, 1.0)],
        );
        let handle = images.add(image.clone());
        let mut cache = TextureCache::default();
        let options = TextureOptions::default();
        cache
            .update(vec![handle.id()], Some(&images), &options, true)
            .unwrap();
        let texture = cache.atlas.textures[&handle.id()];
        assert_eq!(cache.atlas.sample_alpha(texture, Vec2::new(0.25, 0.5)), 0.0);
        assert_eq!(cache.atlas.sample_alpha(texture, Vec2::new(0.75, 0.5)), 1.0);

        image = rgba32(UVec2::new(2, 1), &[Vec4::ONE, Vec4::ONE]);
        *images.get_mut(&handle).unwrap() = image;
        cache
            .update(vec![handle.id()], Some(&images), &options, true)
            .unwrap();
        assert_eq!(cache.atlas.sample_alpha(texture, Vec2::new(0.25, 0.5)), 1.0);
    }
}

#[derive(Default)]
struct Shelf {
    x: u32,
    y: u32,
    height: u32,
}

impl TextureAtlas {
    pub(crate) fn sample_alpha(&self, texture: u32, uv: Vec2) -> f32 {
        if texture == NO_TEXTURE {
            return 1.0;
        }
        let Some(header) = self.words.get(texture as usize).copied() else {
            return 1.0;
        };
        let Some(mip) = self.words.get(header.x as usize).copied() else {
            return 1.0;
        };
        if mip.x == 0 || mip.y == 0 || self.side <= 2 {
            return 1.0;
        }
        let flags = header.z;
        let uv = Vec2::new(
            texture_uv(uv.x, flags & 3),
            texture_uv(uv.y, (flags >> 2) & 3),
        );
        let position = uv * Vec2::new(mip.x as f32, mip.y as f32);
        if flags & (1 << 4) == 0 {
            return self.alpha_texel(mip, position.floor().as_ivec2(), flags);
        }
        let position = position - Vec2::splat(0.5);
        let low = position.floor().as_ivec2();
        let weight = position.fract();
        let a = self.alpha_texel(mip, low, flags);
        let b = self.alpha_texel(mip, low + IVec2::X, flags);
        let c = self.alpha_texel(mip, low + IVec2::Y, flags);
        let d = self.alpha_texel(mip, low + IVec2::ONE, flags);
        (a + (b - a) * weight.x) * (1.0 - weight.y) + (c + (d - c) * weight.x) * weight.y
    }

    fn alpha_texel(&self, mip: UVec4, texel: IVec2, flags: u32) -> f32 {
        let Some(tile_side) = self.side.checked_sub(2).filter(|side| *side > 0) else {
            return 1.0;
        };
        let x = texel_index(texel.x, mip.x, flags & 3);
        let y = texel_index(texel.y, mip.y, (flags >> 2) & 3);
        let tile = UVec2::new(x / tile_side, y / tile_side);
        let rectangle_index = mip.w as usize + tile.y as usize * mip.z as usize + tile.x as usize;
        let Some(rectangle) = self.words.get(rectangle_index).copied() else {
            return 1.0;
        };
        let pixel = UVec2::new(
            rectangle.x + 1 + x % tile_side,
            rectangle.y + 1 + y % tile_side,
        );
        let offset = ((pixel.y * self.side + pixel.x) * 16 + 12) as usize;
        self.pages
            .get(rectangle.z as usize)
            .and_then(|page| page.get(offset..offset + 4))
            .and_then(|bytes| bytes.try_into().ok())
            .map(f32::from_le_bytes)
            .filter(|alpha| alpha.is_finite())
            .unwrap_or(1.0)
    }

    fn reserve(
        &mut self,
        shelves: &mut Vec<Shelf>,
        size: UVec2,
        budget: usize,
    ) -> Result<UVec3, String> {
        for (page, shelf) in shelves.iter_mut().enumerate() {
            if shelf.x + size.x > self.side {
                shelf.y += shelf.height;
                shelf.x = 0;
                shelf.height = 0;
            }
            if shelf.y + size.y <= self.side {
                let location = UVec3::new(shelf.x, shelf.y, page as u32);
                shelf.x += size.x;
                shelf.height = shelf.height.max(size.y);
                return Ok(location);
            }
        }
        let bytes = self.side as usize * self.side as usize * 16;
        if (self.pages.len() + 1)
            .checked_mul(bytes)
            .is_none_or(|total| total > budget)
        {
            return Err(format!(
                "texture atlas residency exceeds {budget} bytes; no texture was downsampled or dropped"
            ));
        }
        let location = UVec3::new(0, 0, self.pages.len() as u32);
        self.pages.push(vec![0; bytes]);
        shelves.push(Shelf {
            x: size.x,
            y: 0,
            height: size.y,
        });
        Ok(location)
    }

    fn insert(
        &mut self,
        id: AssetId<Image>,
        mips: &[Mip],
        flags: u32,
        limits: Vec2,
        shelves: &mut Vec<Shelf>,
        budget: usize,
    ) -> Result<(), String> {
        let texture = self.words.len() as u32;
        self.textures.insert(id, texture);
        // Texture: mip table/count/sampler flags, then source size/LOD limits.
        self.words
            .push(UVec4::new(texture + 2, mips.len() as u32, flags, 0));
        self.words.push(UVec4::new(
            mips[0].size.x,
            mips[0].size.y,
            limits.x.to_bits(),
            limits.y.to_bits(),
        ));
        let mip_base = self.words.len();
        self.words.resize(mip_base + mips.len(), UVec4::ZERO);
        let tile_side = self.side - 2;
        for (level, mip) in mips.iter().enumerate() {
            let tiles = UVec2::new(
                mip.size.x.div_ceil(tile_side),
                mip.size.y.div_ceil(tile_side),
            );
            let tile_base = self.words.len() as u32;
            self.words[mip_base + level] = UVec4::new(mip.size.x, mip.size.y, tiles.x, tile_base);
            for ty in 0..tiles.y {
                for tx in 0..tiles.x {
                    let start = UVec2::new(tx, ty) * tile_side;
                    let extent = (mip.size - start).min(UVec2::splat(tile_side));
                    let location = self.reserve(shelves, extent + UVec2::splat(2), budget)?;
                    self.words.push(location.extend(0));
                    // One texel of source-correct gutter on every tile, including
                    // wrapped texture edges. Each mip has its own packed rectangles.
                    for y in 0..extent.y + 2 {
                        for x in 0..extent.x + 2 {
                            let sx =
                                texel_index(start.x as i32 + x as i32 - 1, mip.size.x, flags & 3);
                            let sy = texel_index(
                                start.y as i32 + y as i32 - 1,
                                mip.size.y,
                                (flags >> 2) & 3,
                            );
                            let color = mip.pixels[(sy * mip.size.x + sx) as usize];
                            let offset =
                                (((location.y + y) * self.side + location.x + x) * 16) as usize;
                            for (channel, value) in color.to_array().into_iter().enumerate() {
                                self.pages[location.z as usize]
                                    [offset + channel * 4..offset + channel * 4 + 4]
                                    .copy_from_slice(&value.to_le_bytes());
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

fn texture_uv(uv: f32, mode: u32) -> f32 {
    match mode {
        1 => uv - uv.floor(),
        2 => {
            let value = (uv * 0.5 - (uv * 0.5).floor()) * 2.0;
            if value <= 1.0 { value } else { 2.0 - value }
        }
        _ => uv.clamp(0.0, 1.0),
    }
}

impl TextureCache {
    pub fn update(
        &mut self,
        mut ids: Vec<AssetId<Image>>,
        images: Option<&Assets<Image>>,
        options: &TextureOptions,
        changed: bool,
    ) -> Result<(), String> {
        ids.sort();
        ids.dedup();
        if !changed
            && let Some((failed, reason)) = &self.failed
            && failed == &ids
        {
            return Err(reason.clone());
        }
        match self.rebuild(ids.clone(), images, options, changed) {
            Ok(()) => {
                self.failed = None;
                self.permanent_failure = false;
                Ok(())
            }
            Err(reason) => {
                self.permanent_failure = ids
                    .iter()
                    .all(|id| images.is_some_and(|images| images.get(*id).is_some()));
                self.failed = Some((ids, reason.clone()));
                Err(reason)
            }
        }
    }

    fn rebuild(
        &mut self,
        ids: Vec<AssetId<Image>>,
        images: Option<&Assets<Image>>,
        options: &TextureOptions,
        changed: bool,
    ) -> Result<(), String> {
        if !changed && self.keys.iter().map(|&(id, _)| id).eq(ids.iter().copied()) {
            return Ok(());
        }
        let mut keys = Vec::new();
        for &id in &ids {
            let image = images
                .and_then(|images| images.get(id))
                .ok_or_else(|| format!("texture {id:?} is not resident"))?;
            let mut hash = std::collections::hash_map::DefaultHasher::new();
            image.data.hash(&mut hash);
            format!(
                "{:?}{:?}{:?}",
                image.texture_descriptor, image.texture_view_descriptor, image.sampler
            )
            .hash(&mut hash);
            format!("{:?}", options.sampler).hash(&mut hash);
            options.budget.hash(&mut hash);
            keys.push((id, hash.finish()));
        }
        if keys == self.keys {
            return Ok(());
        }
        let mut decoded_bytes = 0usize;
        for &id in &ids {
            let image = images
                .and_then(|images| images.get(id))
                .expect("checked resident images");
            let mut width = image.texture_descriptor.size.width as usize;
            let mut height = image.texture_descriptor.size.height as usize;
            if width == 0 || height == 0 {
                return Err("empty texture extent".into());
            }
            let full_mips = width.max(height).ilog2() + 1;
            let mip_count = image
                .texture_view_descriptor
                .as_ref()
                .and_then(|view| view.mip_level_count)
                .unwrap_or(full_mips);
            if mip_count == 0 || mip_count > full_mips {
                return Err("invalid texture-view mip count".into());
            }
            for _ in 0..mip_count {
                decoded_bytes = width
                    .checked_mul(height)
                    .and_then(|pixels| pixels.checked_mul(16))
                    .and_then(|bytes| decoded_bytes.checked_add(bytes))
                    .ok_or("texture byte extent overflows")?;
                if decoded_bytes > options.budget {
                    return Err("decoded texture residency exceeds atlas budget".into());
                }
                width = (width / 2).max(1);
                height = (height / 2).max(1);
            }
        }
        let max_dimension = ids
            .iter()
            .filter_map(|id| images.and_then(|images| images.get(*id)))
            .map(|image| {
                image
                    .texture_descriptor
                    .size
                    .width
                    .max(image.texture_descriptor.size.height)
            })
            .max()
            .unwrap_or(1);
        let mut atlas = TextureAtlas {
            side: max_dimension
                .saturating_add(2)
                .min(512)
                .next_power_of_two()
                .max(16),
            ..default()
        };
        let mut shelves = Vec::new();
        // shortcut: area-first shelves can still fragment, upgrade to bin packing if failures persist.
        let mut atlas_ids = ids;
        atlas_ids.sort_by_key(|&id| {
            let size = images
                .and_then(|images| images.get(id))
                .expect("checked resident images")
                .texture_descriptor
                .size;
            std::cmp::Reverse(u64::from(size.width) * u64::from(size.height))
        });
        for &id in &atlas_ids {
            let image = images
                .and_then(|images| images.get(id))
                .expect("checked resident images");
            let (flags, limits) = sampler(image, options).map_err(str::to_owned)?;
            let mips = decode(image)?;
            atlas.insert(id, &mips, flags, limits, &mut shelves, options.budget)?;
        }
        // Commit only a complete atlas; failed changes leave the previous one intact.
        self.atlas = Arc::new(atlas);
        self.keys = keys;
        Ok(())
    }
}

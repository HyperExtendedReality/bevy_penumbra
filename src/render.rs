use std::{
    any::TypeId,
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
    },
};

use bevy::{
    camera::visibility::RenderLayers,
    core_pipeline::{
        Core3dSystems,
        core_3d::{main_opaque_pass_3d, main_transparent_pass_3d},
        prepass::ViewPrepassTextures,
        schedule::Core3d,
    },
    pbr::{RenderGpuCulledEntities, deferred::deferred_lighting, screen_space_reflections},
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        camera::ExtractedCamera,
        diagnostic::{DiagnosticsRecorder, RecordDiagnostics},
        render_asset::RenderAssets,
        render_resource::{binding_types::*, *},
        renderer::{
            RenderAdapter, RenderAdapterInfo, RenderContext, RenderDevice, RenderGraph,
            RenderGraphSystems, RenderQueue, ViewQuery,
        },
        sync_world::MainEntity,
        texture::GpuImage,
        view::{
            ExtractedView, RenderShadowMapVisibleEntities, ViewTarget,
            visibility::{RenderVisibleEntities, RenderVisibleEntitiesClass},
        },
    },
    shader::Shader,
    tasks::{AsyncComputeTaskPool, Task, block_on, poll_once},
};
use bevy_slang::{SlangCompiler, SlangLoaderError, SlangSettings, SlangTarget};

mod hardware;

use crate::{
    RadianceCascades, RadianceCascadesRawOutput, RadianceCascadesRevisions, RadianceCascadesStatus,
    scene::{BvhNode, GiLight, GiScene, view_lights},
};

const LEVELS: usize = 4;
// Byte stride of `Entry` in common.slang.
const ENTRY_SIZE: u64 = 272;
const SOURCE_BACKGROUND: u32 = 1;
const SOURCE_EMISSIVE: u32 = 2;
const SOURCE_PUNCTUAL: u32 = 4;
// Current work reserves two groups per ray; persistent history keeps four
// intervals per bucket to reduce cache churn in large scenes.
const CACHE_WORK_GROUPS: u64 = 2;
const CACHE_HISTORY_PER_BUCKET: u64 = 4;
const SPARSE_HISTORY_RESET_FLAG: u32 = 1 << 31;
const SPARSE_UPDATES_DISABLED_FLAG: u32 = 1 << 30;
// Without indirect execution, the trace covers the widest receiver spacing
// (`TRACE_MAX_LANE_STRIDE` in common.slang); surplus lanes exit immediately.
const TRACE_MAX_LANE_STRIDE: u32 = 8;
const COLD_RANK_PHASES: [usize; 3] = [1, 2, 3];
// Three source-separated irradiance fields, dependency bounds, and a
// secondary-cache key and generation header.
pub(crate) const PROBE_SIZE: u64 = 1792;
const PACK_PROBE_GROUPS_PER_ROW: u64 = 256;
const CACHE_COMMIT: usize = 14;
const COMPUTE_ENTRIES: [&str; 29] = [
    "clear",
    "seed",
    "parents",
    "rank",
    "prefix",
    "generateDirections",
    "retain",
    "trace",
    "reduce",
    "merge",
    "seedProbes",
    "packProbes",
    "temporal",
    "denoise",
    "commitCache",
    "remapProbeHeads",
    "worldResolve",
    "traceDeferred",
    "traceRecord",
    "traceVisibility",
    "traceResolve",
    "traceDeferredHardware",
    "traceVisibilityHardware",
    "sunShadowsHardware",
    "rtgiSample",
    "rtgiResolve",
    "rtgiCacheRecord",
    "rtgiCacheVisibility",
    "rtgiCacheShade",
];
const SLANG_OPTIMIZATION_LEVEL: u8 = 2;
const HISTORY_DIAGNOSTIC_WORDS: u64 = 26;

fn probe_directory_capacity(samples: u64) -> u64 {
    samples.saturating_mul(4).max(1).next_power_of_two()
}

fn pack_probe_dispatch_groups(samples: u64) -> Option<(u32, u32)> {
    let descriptors = samples.checked_mul(2)?;
    if descriptors == 0 || descriptors > u64::from(u32::MAX) {
        return None;
    }
    let groups_x = descriptors.min(PACK_PROBE_GROUPS_PER_ROW);
    let groups_y =
        descriptors.checked_add(PACK_PROBE_GROUPS_PER_ROW - 1)? / PACK_PROBE_GROUPS_PER_ROW;
    Some((u32::try_from(groups_x).ok()?, u32::try_from(groups_y).ok()?))
}

fn cold_priority_enabled(history_frames: u32, history_was_valid: bool) -> bool {
    // Split always enables persistent world intervals in intervalHistory.y.
    history_frames > 0 && !history_was_valid
}

fn cold_rank_phases(history_frames: u32, history_was_valid: bool) -> &'static [usize] {
    if cold_priority_enabled(history_frames, history_was_valid) {
        &COLD_RANK_PHASES
    } else {
        &[]
    }
}

// Mirror DEFERRED_* in common.slang: deferred primary trace scratch.
const DEFERRED_TRACE_RECEIVERS: u64 = 8192;
const DEFERRED_VISIBILITY_QUERIES: u64 = 4;
const DEFERRED_SLOT_WORDS: u64 = 32;
const DEFERRED_QUERY_WORDS: u64 = 12;

/// Words after the trace sample list: query count and dispatch arguments,
/// the query list, per-variant hit slots, and the queries themselves.
fn deferred_trace_words(samples: u64) -> u64 {
    let slots = samples.min(DEFERRED_TRACE_RECEIVERS) * 2;
    let queries = slots * DEFERRED_VISIBILITY_QUERIES;
    4 + queries + slots * DEFERRED_SLOT_WORDS + queries * DEFERRED_QUERY_WORDS
}

fn cache_head_bytes(buckets: u64, samples: u64) -> u64 {
    // Cache heads/counters plus per-cascade probe indices and ray counts.
    let counters_and_diagnostics = 10 + HISTORY_DIAGNOSTIC_WORDS;
    let probe_directory = LEVELS as u64 * probe_directory_capacity(samples);
    (buckets * 2
        + counters_and_diagnostics
        + probe_directory * 2
        + 1
        + samples
        + 4
        + deferred_trace_words(samples))
        * 4
}

fn irradiance_probe_bytes(samples: u64, multi_bounce: bool) -> u64 {
    probe_directory_capacity(samples) * (if multi_bounce { 4 } else { 2 }) * PROBE_SIZE
}

fn exposure_invalidates_world_history(exposure_changed: bool) -> bool {
    exposure_changed
}

fn resolve_scale(settings: RadianceCascades) -> u32 {
    settings.sampling.resolve_scale.clamp(1, 8)
}

fn split_lod_target_pixels(settings: RadianceCascades) -> u32 {
    resolve_scale(settings).max(1)
}

fn resolve_size(viewport: UVec2, settings: RadianceCascades) -> UVec2 {
    let scale = resolve_scale(settings);
    UVec2::new(viewport.x.div_ceil(scale), viewport.y.div_ceil(scale)).max(UVec2::ONE)
}

fn cache_work_slots(samples: u64) -> u64 {
    samples * LEVELS as u64 * CACHE_WORK_GROUPS
}

fn cache_entry_bytes(samples: u64, buckets: u64) -> u64 {
    (cache_work_slots(samples) + buckets * CACHE_HISTORY_PER_BUCKET) * ENTRY_SIZE
}

fn cache_bucket_count(samples: u64, settings: RadianceCascades) -> u64 {
    let mut buckets = (samples * 2).next_power_of_two().max(1024);
    let capacity =
        u64::from(cache_query_limits(settings.sampling.max_cache_entries, settings.multi_bounce).0);
    if capacity > 0 {
        let entries_per_bucket = CACHE_HISTORY_PER_BUCKET;
        let maximum_buckets = (capacity / entries_per_bucket).max(1);
        let rounded_buckets = 1u64 << (u64::BITS - 1 - maximum_buckets.leading_zeros());
        // A world cache must keep its capacity when the screen grid shrinks.
        buckets = rounded_buckets;
    }
    buckets
}

fn cache_history_bytes(buckets: u64) -> u64 {
    buckets * CACHE_HISTORY_PER_BUCKET * ENTRY_SIZE
}

fn history_settings_changed(previous: RadianceCascades, next: RadianceCascades) -> bool {
    // Composite intensity leaves stored lighting unchanged. Environment changes
    // are radiometric: reprojection remains valid while world radiance refreshes.
    // Hardware and software traversal produce the same hits.
    // Other settings can change support or receiver interpretation.
    let mut comparable = previous;
    comparable.intensity = next.intensity;
    comparable.environment = next.environment;
    comparable.hardware_ray_queries = next.hardware_ray_queries;
    comparable.ray_traced_sun_shadows = next.ray_traced_sun_shadows;
    comparable != next
}

fn world_cache_settings_changed(mut previous: RadianceCascades, next: RadianceCascades) -> bool {
    // Composition gain, ambient radiance, resolve scale and the traversal
    // backend do not alter cached transport.
    previous.intensity = next.intensity;
    previous.environment = next.environment;
    previous.sampling.resolve_scale = next.sampling.resolve_scale;
    previous.hardware_ray_queries = next.hardware_ray_queries;
    previous.ray_traced_sun_shadows = next.ray_traced_sun_shadows;
    previous != next
}

fn selected_update_budget(settings: RadianceCascades) -> u32 {
    let ceiling = settings.max_rays.max(1);
    let requested = if settings.sampling.max_updates_per_frame == 0 {
        ceiling
    } else {
        settings.sampling.max_updates_per_frame
    };
    requested.min(ceiling).max(1)
}

fn runtime_update_budget(settings: RadianceCascades, runtime_cap: Option<u32>) -> u32 {
    selected_update_budget(settings).min(runtime_cap.unwrap_or(u32::MAX).max(1))
}

fn query_affordable_update_budget(
    settings: RadianceCascades,
    requested: u32,
    conventional_lights: u32,
    emissive_lights: u32,
    has_environment_map: bool,
) -> u32 {
    let query_limit = settings.sampling.max_traversal_queries;
    if query_limit == 0 || requested == 0 {
        return requested;
    }

    // Constant skies now use visibility-tested next-event sampling just like
    // environment maps, so their shadow queries need the same reserved work.
    let has_environment =
        has_environment_map || settings.environment.to_vec4().truncate().max_element() > 0.0;
    let samples = settings.emissive_samples.max(1);
    // Leave a separate tranche for merge before the final gather reserve.
    let (admission_limit, _) = traversal_query_caps(query_limit);
    let outgoing_samples = (samples / 2).max(1);
    let outgoing_nee = outgoing_samples
        .saturating_mul(u32::from(emissive_lights > 0))
        .saturating_add(outgoing_samples.saturating_mul(u32::from(has_environment)))
        .saturating_add(conventional_lights);
    // Nominal Split update cost: probe seed, cascade interval rays, and direct
    // next-event sampling at the first hit. Query admission remains the hard
    // safety cap when geometry needs more than this estimate.
    let queries_per_update = outgoing_nee.saturating_add(3).max(1);
    let affordable = admission_limit / queries_per_update;
    requested.min(affordable)
}

fn traversal_query_caps(query_limit: u32) -> (u32, u32) {
    if query_limit == 0 {
        return (0, 0);
    }

    // Trace/NEE stop at half; merge can reach two thirds; gather can reach
    // the hard cap. Sharing trace's ceiling starved cold interval merges.
    let ordinary_limit = (query_limit / 2).max(1);
    (ordinary_limit, query_limit)
}

fn cache_query_limits(limit: u32, multi_bounce: bool) -> (u32, u32) {
    if !multi_bounce || limit == 0 {
        (limit, limit)
    } else {
        (limit.div_ceil(2), limit / 2)
    }
}

fn view_budget_max_rays(viewport: UVec2, settings: RadianceCascades, budget: u64) -> u32 {
    let mut low = 1;
    let mut high = if settings.sampling.max_receiver_cells == 0 {
        131_072
    } else {
        settings.sampling.max_receiver_cells
    };
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        let mut candidate = settings;
        candidate.sampling.max_receiver_cells = middle;
        if view_bytes(
            resolution(viewport, candidate),
            resolve_size(viewport, candidate),
            candidate,
        ) <= budget
        {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    low
}

fn view_fits_device(
    size: UVec2,
    shading_size: UVec2,
    viewport_size: UVec2,
    settings: RadianceCascades,
    limits: &WgpuLimits,
) -> bool {
    let pixels = u64::from(size.x) * u64::from(size.y);
    let buckets = cache_bucket_count(pixels, settings);
    let probe_directory = probe_directory_capacity(pixels);
    let binding_limit = limits
        .max_storage_buffer_binding_size
        .min(limits.max_buffer_size);
    let workgroups = u64::from(limits.max_compute_workgroups_per_dimension);
    let pack_probe_groups_fit = pack_probe_dispatch_groups(pixels)
        .is_some_and(|(x, y)| u64::from(x) <= workgroups && u64::from(y) <= workgroups);
    size.max_element() <= limits.max_texture_dimension_2d
        && shading_size.max_element() <= limits.max_texture_dimension_2d
        && viewport_size.max_element() <= limits.max_texture_dimension_2d
        && u64::from(size.x.div_ceil(8)) <= workgroups
        && u64::from(size.y.div_ceil(8)) <= workgroups
        && u64::from(shading_size.x.div_ceil(8)) <= workgroups
        && u64::from(shading_size.y.div_ceil(8)) <= workgroups
        && u64::from(viewport_size.x.div_ceil(8)) <= workgroups
        && u64::from(viewport_size.y.div_ceil(8)) <= workgroups
        && pixels.div_ceil(64) <= workgroups
        && buckets.max(probe_directory * LEVELS as u64).div_ceil(64) <= workgroups
        && pack_probe_groups_fit
        && cache_entry_bytes(pixels, buckets) <= binding_limit
        && cache_history_bytes(buckets) <= binding_limit
        && cache_head_bytes(buckets, pixels) <= binding_limit
        && irradiance_probe_bytes(pixels, settings.multi_bounce) <= binding_limit
}

fn view_device_max_rays(viewport: UVec2, settings: RadianceCascades, limits: &WgpuLimits) -> u32 {
    let mut low = 1;
    let mut high = if settings.sampling.max_receiver_cells == 0 {
        131_072
    } else {
        settings.sampling.max_receiver_cells
    };
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        let mut candidate = settings;
        candidate.sampling.max_receiver_cells = middle;
        if view_fits_device(
            resolution(viewport, candidate),
            resolve_size(viewport, candidate),
            viewport,
            candidate,
            limits,
        ) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    low
}

fn settings_match_budgeted_request(
    mut effective: RadianceCascades,
    previous_request: RadianceCascades,
    requested: RadianceCascades,
) -> bool {
    // Adaptation can change these fields, but a caller editing one of them must
    // still get a new allocation decision instead of silently keeping old work.
    if previous_request != requested {
        return false;
    }
    // Budget adaptation only changes the sparse-grid limit and resolve scale.
    // Comparing the normalized settings lets a live view keep its selected
    // effective scale without oscillating against its own retained allocations.
    effective.max_rays = requested.max_rays;
    effective.sampling.max_receiver_cells = requested.sampling.max_receiver_cells;
    effective.sampling.resolve_scale = requested.sampling.resolve_scale;
    effective == requested
}

fn adapt_view_settings(
    viewport: UVec2,
    requested: RadianceCascades,
    available_view_bytes: u64,
    limits: &WgpuLimits,
) -> Option<RadianceCascades> {
    let first_scale = resolve_scale(requested).clamp(1, 8);
    let mut best: Option<(u64, u32, RadianceCascades)> = None;

    for scale in first_scale..=8 {
        let mut candidate = requested;
        candidate.sampling.resolve_scale = scale;
        candidate.sampling.max_receiver_cells =
            view_budget_max_rays(viewport, candidate, available_view_bytes);
        candidate.sampling.max_receiver_cells = candidate
            .sampling
            .max_receiver_cells
            .min(view_device_max_rays(viewport, candidate, limits));

        let size = resolution(viewport, candidate);
        let shading_size = resolve_size(viewport, candidate);
        let bytes = view_bytes(size, shading_size, candidate);
        if bytes > available_view_bytes
            || !view_fits_device(size, shading_size, viewport, candidate, limits)
        {
            continue;
        }
        let cells = u64::from(size.x) * u64::from(size.y);
        let candidate_scale = resolve_scale(candidate);
        if best.as_ref().is_none_or(|(best_cells, best_scale, _)| {
            cells > *best_cells || (cells == *best_cells && candidate_scale < *best_scale)
        }) {
            best = Some((cells, candidate_scale, candidate));
        }
    }
    best.map(|(_, _, settings)| settings)
}

// Hardware ray-query variants of `traceDeferred` and `traceVisibility`.
const HARDWARE_TRACE_DEFERRED: usize = 21;
const HARDWARE_TRACE_VISIBILITY: usize = 22;
/// Ray-traced sun shadow mask; see `RadianceCascades::ray_traced_sun_shadows`.
const SUN_SHADOWS_HARDWARE: usize = 23;
/// Directional lights a sun shadow mask covers; mirrors common.slang.
const SUN_SHADOW_LIGHTS: usize = 1;
/// Ray-traced GI (`GiMethod::RayTraced`): path sampling with temporal reuse,
/// then spatial reuse and resolve.
const RTGI_SAMPLE: usize = 24;
const RTGI_RESOLVE: usize = 25;
/// World radiance cache shading: record shadow rays, trace them in hardware,
/// then replay the shading with the answers.
const RTGI_CACHE_RECORD: usize = 26;
const RTGI_CACHE_VISIBILITY: usize = 27;
const RTGI_CACHE_SHADE: usize = 28;
/// Bytes of one `RtgiReservoir` in cascades.slang.
const RTGI_RESERVOIR_BYTES: u64 = 64;
// Mirror RTGI_CACHE_* in cascades.slang.
const RTGI_CACHE_CAPACITY: u64 = 1 << 18;
const RTGI_CACHE_CELL_BYTES: u64 = 80;
const RTGI_CACHE_QUEUE: u64 = 8192;
const RTGI_CACHE_REFRESH: u64 = 8192;
const HARDWARE_TRAVERSAL_PIPELINES: u64 = (1 << HARDWARE_TRACE_DEFERRED)
    | (1 << HARDWARE_TRACE_VISIBILITY)
    | (1 << SUN_SHADOWS_HARDWARE)
    | (1 << RTGI_SAMPLE)
    | (1 << RTGI_RESOLVE)
    | (1 << RTGI_CACHE_VISIBILITY);
const RAY_QUERY_DEFINE: &str = "PENUMBRA_HARDWARE_RAY_QUERY";

fn hardware_traversal_entry(entry: &str) -> bool {
    matches!(
        entry,
        "traceDeferredHardware"
            | "traceVisibilityHardware"
            | "sunShadowsHardware"
            | "rtgiSample"
            | "rtgiResolve"
            | "rtgiCacheVisibility"
    )
}

/// Pipelines a view cannot render without. Hardware traversal is optional:
/// software traversal produces the same transport while it is unavailable.
fn required_pipelines(settings: RadianceCascades) -> u64 {
    let mut required = ((1u64 << COMPUTE_ENTRIES.len()) - 1) & !HARDWARE_TRAVERSAL_PIPELINES;
    if settings.spatial_filter_steps == 0 {
        required &= !(1 << 13);
    }
    required
}

/// Pipelines to build for a view: the required set plus the hardware
/// traversal variants when the view requests them.
fn wanted_pipelines(settings: RadianceCascades) -> u64 {
    required_pipelines(settings)
        | if settings.hardware_ray_queries {
            HARDWARE_TRAVERSAL_PIPELINES
        } else {
            0
        }
}

const MODULES: [(&str, &[&str]); 4] = [
    ("cascades.slang", &COMPUTE_ENTRIES),
    ("composite.slang", &["vertex", "fragment", "compositeAnchors", "sunShadowDepth"]),
    ("acceleration.slang", &["refitTlas"]),
    ("deformation.slang", &["deformVertices", "refitBlas"]),
];
const SOURCES: &[(&str, &str)] = &[
    (
        "deformation.slang",
        include_str!("../shaders/deformation.slang"),
    ),
    ("common.slang", include_str!("../shaders/common.slang")),
    (
        "world_resolve.slang",
        include_str!("../shaders/world_resolve.slang"),
    ),
    ("bsdf.slang", include_str!("../shaders/bsdf.slang")),
    (
        "environment.slang",
        include_str!("../shaders/environment.slang"),
    ),
    (
        "traversal.slang",
        include_str!("../shaders/traversal.slang"),
    ),
    ("material.slang", include_str!("../shaders/material.slang")),
    ("split.slang", include_str!("../shaders/split.slang")),
    (
        "linked_merge_sort.slang",
        include_str!("../shaders/linked_merge_sort.slang"),
    ),
    (
        "receiver_metadata.slang",
        include_str!("../shaders/receiver_metadata.slang"),
    ),
    ("cascades.slang", include_str!("../shaders/cascades.slang")),
    (
        "acceleration.slang",
        include_str!("../shaders/acceleration.slang"),
    ),
    (
        "composite.slang",
        include_str!("../shaders/composite.slang"),
    ),
];

fn shader_modules() -> impl Iterator<Item = (&'static str, &'static [&'static str])> {
    MODULES.into_iter().flat_map(|(file, entries)| {
        // Keep vertex/fragment together. Compute modules contain only their
        // entry's call graph, avoiding unused functions in driver compilation.
        entries
            .chunks(if file == "composite.slang" {
                entries.len()
            } else {
                1
            })
            .map(move |entries| (file, entries))
    })
}

fn compile(
    compiler: &SlangCompiler,
    file: &str,
    entries: &[&str],
) -> Result<Shader, SlangLoaderError> {
    let mut shader = compiler.compile_bundle(
        file,
        SOURCES,
        &SlangSettings {
            target: SlangTarget::Wgsl,
            entry_points: entries.iter().map(|entry| (*entry).to_owned()).collect(),
            // Only the hardware traversal entries reference the WGSL
            // ray-query helpers; the define keeps every other module free of them.
            defines: entries
                .iter()
                .any(|entry| hardware_traversal_entry(entry))
                .then(|| RAY_QUERY_DEFINE.to_owned())
                .into_iter()
                .collect(),
            optimization: Some(SLANG_OPTIMIZATION_LEVEL),
            ..default()
        },
    )?;
    shader.validate_shader = bevy::shader::ValidateShader::Enabled;
    Ok(shader)
}

fn compiler_version(compiler: &SlangCompiler) -> Result<String, String> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let output = std::process::Command::new(&compiler.executable)
            .arg("-version")
            .output()
            .map_err(|error| {
                format!("could not run {:?} -version: {error}", compiler.executable)
            })?;
        if !output.status.success() {
            let diagnostics = format!(
                "{}{}{}",
                String::from_utf8_lossy(&output.stdout).trim(),
                if output.stdout.is_empty() || output.stderr.is_empty() {
                    ""
                } else {
                    "\n"
                },
                String::from_utf8_lossy(&output.stderr).trim(),
            );
            return Err(format!(
                "{:?} -version exited with {}: {diagnostics}",
                compiler.executable, output.status
            ));
        }
        let version = format!(
            "{}{}{}",
            String::from_utf8_lossy(&output.stdout).trim(),
            if output.stdout.is_empty() || output.stderr.is_empty() {
                ""
            } else {
                "\n"
            },
            String::from_utf8_lossy(&output.stderr).trim(),
        );
        if version.is_empty() {
            Err(format!(
                "{:?} -version returned no version",
                compiler.executable
            ))
        } else {
            Ok(version)
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = compiler;
        Err("Slang compiler version detection is unavailable on WebAssembly".into())
    }
}

fn shader_cache_key(
    compiler: &SlangCompiler,
    compiler_version: Option<&str>,
    file: &str,
    entries: &[&str],
    sources: &[(&str, &str)],
) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    2u32.hash(&mut hasher);
    compiler.executable.hash(&mut hasher);
    compiler.source_root.hash(&mut hasher);
    compiler.include_paths.hash(&mut hasher);
    compiler_version.hash(&mut hasher);
    file.hash(&mut hasher);
    entries.hash(&mut hasher);
    sources.hash(&mut hasher);
    "wgsl".hash(&mut hasher);
    SLANG_OPTIMIZATION_LEVEL.hash(&mut hasher);
    hasher.finish()
}

fn shader_cache_dir() -> std::path::PathBuf {
    if let Some(path) = std::env::var_os("BEVY_PENUMBRA_SHADER_CACHE_DIR") {
        return path.into();
    }

    #[cfg(windows)]
    if let Some(path) = std::env::var_os("LOCALAPPDATA") {
        return std::path::PathBuf::from(path)
            .join("bevy_penumbra")
            .join("shaders");
    }

    #[cfg(target_os = "macos")]
    if let Some(path) = std::env::var_os("HOME") {
        return std::path::PathBuf::from(path)
            .join("Library")
            .join("Caches")
            .join("bevy_penumbra")
            .join("shaders");
    }

    #[cfg(all(not(windows), not(target_os = "macos")))]
    if let Some(path) = std::env::var_os("XDG_CACHE_HOME") {
        return std::path::PathBuf::from(path)
            .join("bevy_penumbra")
            .join("shaders");
    }

    #[cfg(all(not(windows), not(target_os = "macos")))]
    if let Some(path) = std::env::var_os("HOME") {
        return std::path::PathBuf::from(path)
            .join(".cache")
            .join("bevy_penumbra")
            .join("shaders");
    }

    std::env::temp_dir().join("bevy_penumbra").join("shaders")
}

fn cached_compile(
    compiler: &SlangCompiler,
    file: &str,
    entries: &[&str],
    cache_dir: &std::path::Path,
    compiler_version: Option<&str>,
) -> Result<Shader, SlangLoaderError> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (cache_dir, compiler_version);
        return compile(compiler, file, entries);
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::io::Write;

        let key = shader_cache_key(compiler, compiler_version, file, entries, SOURCES);
        let cache_path = cache_dir.join(format!("{key:016x}.wgsl"));
        let cached = std::fs::read_to_string(&cache_path).ok().map(|wgsl| {
            let mut shader = Shader::from_wgsl(wgsl, file.to_owned());
            shader.validate_shader = bevy::shader::ValidateShader::Enabled;
            shader
        });
        if let Some(shader) = cached {
            debug!(file, entries = ?entries, path = %cache_path.display(), "GI shader cache hit");
            return Ok(shader);
        }

        debug!(file, entries = ?entries, path = %cache_path.display(), "GI shader cache miss");
        let shader = compile(compiler, file, entries)?;
        let bytes = match &shader.source {
            bevy::shader::Source::Wgsl(wgsl) => Some(wgsl.as_bytes()),
            _ => None,
        };
        if let Some(bytes) = bytes
            && std::fs::create_dir_all(cache_dir).is_ok()
            && let Ok(mut cache_file) = tempfile::NamedTempFile::new_in(cache_dir)
            && cache_file.write_all(bytes).is_ok()
            && let Err(error) = cache_file.persist(&cache_path)
        {
            debug!(%error, path = %cache_path.display(), "GI shader cache write failed");
        }
        Ok(shader)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod shader_cache_tests {
    use super::{SOURCES, SlangCompiler, cached_compile, shader_cache_key};

    #[test]
    fn shader_cache_key_tracks_source_and_compile_inputs() {
        let source = [("main.slang", "void main() {}")];
        let compiler = SlangCompiler::default();
        let key = shader_cache_key(&compiler, None, "main.slang", &["main"], &source);
        assert_ne!(
            key,
            shader_cache_key(&compiler, None, "other.slang", &["main"], &source)
        );
        assert_ne!(
            key,
            shader_cache_key(&compiler, None, "main.slang", &["other"], &source)
        );
        assert_ne!(
            key,
            shader_cache_key(&compiler, Some("2026.19"), "main.slang", &["main"], &source,)
        );
        assert_ne!(
            key,
            shader_cache_key(
                &compiler,
                None,
                "main.slang",
                &["main"],
                &[("main.slang", "void main() { 1; }")]
            ),
        );
    }

    #[test]
    fn shader_cache_hit_skips_the_external_compiler() {
        let directory = tempfile::tempdir().unwrap();
        let file = "main.slang";
        let entries = ["main"];
        let compiler = SlangCompiler::new("missing-slangc");
        let key = shader_cache_key(&compiler, None, file, &entries, SOURCES);
        let wgsl = "@compute @workgroup_size(1) fn main() {}";
        std::fs::write(directory.path().join(format!("{key:016x}.wgsl")), wgsl).unwrap();
        let shader = cached_compile(&compiler, file, &entries, directory.path(), None).unwrap();
        assert!(matches!(
            shader.validate_shader,
            bevy::shader::ValidateShader::Enabled
        ));
        assert!(matches!(
            shader.source,
            bevy::shader::Source::Wgsl(ref source) if source.as_ref() == wgsl
        ));
    }
}

#[derive(Clone, Copy, Default, ShaderType)]
struct Parameters {
    world_from_clip: Mat4,
    clip_from_world: Mat4,
    previous_clip_from_world: Mat4,
    previous_world_from_clip: Mat4,
    camera_position: Vec4,
    viewport: UVec4,
    size: UVec4,
    tuning: Vec4,
    environment: Vec4,
    history: Vec4,
    interval_history: Vec4,
    frame: UVec4,
    light_counts: UVec4,
    stability: Vec4,
    layers: UVec4,
    environment_info: UVec4,
    environment_rotation: Vec4,
    environment_scale: Vec4,
    geometry_pages: UVec4,
    scene_pages: UVec4,
    scene_overlay: UVec4,
    dirty_min: Vec4,
    dirty_max: Vec4,
    source_history: UVec4,
    sampling: UVec4,
    owner_sequence: UVec4,
    cache_time: UVec4,
    emissive_dirty_min: Vec4,
    emissive_dirty_max: Vec4,
    light_dirty_min: Vec4,
    light_dirty_max: Vec4,
}

#[derive(Resource)]
struct Pipelines {
    compute_layout: BindGroupLayoutDescriptor,
    composite_layout: BindGroupLayoutDescriptor,
    compute: [Option<ComputePipeline>; COMPUTE_ENTRIES.len()],
    started: u64,
    jobs: Vec<PipelineJob>,
    driver_cache: Option<wgpu::PipelineCache>,
    driver_spirv: bool,
    /// The device supports hardware ray queries on the SPIR-V path; the
    /// compute layout then carries the TLAS binding.
    ray_query: bool,
    indirect_execution: bool,
    driver_cache_path: Option<std::path::PathBuf>,
    driver_cache_save: Option<std::thread::JoinHandle<Result<(), String>>>,
    driver_cache_dirty: bool,
    composite: CachedRenderPipelineId,
    sun_shadow_depth: CachedRenderPipelineId,
    composite_anchors: CachedComputePipelineId,
    refit: CachedComputePipelineId,
    deform: [CachedComputePipelineId; 2],
}

type PipelineJob = (
    usize,
    std::thread::JoinHandle<(Result<ComputePipeline, String>, std::time::Duration)>,
);

impl Pipelines {
    fn pipeline(&self, index: usize) -> Option<&ComputePipeline> {
        self.compute[index].as_ref()
    }
}

type ComputeShaderSources = Arc<Mutex<[Option<Arc<str>>; COMPUTE_ENTRIES.len()]>>;

type VulkanComputeSource = (Vec<u32>, (u32, u32, u32));

/// Logical binding of the hardware TLAS in the compute layout.
pub(crate) const TLAS_BINDING: u32 = 41;

/// Prepends the WGSL ray-query loops that the Slang hardware traversal
/// (`traversal.slang`, `PENUMBRA_HARDWARE_RAY_QUERY`) calls through
/// `__intrinsic_asm`. Slang's WGSL target has no ray-query types, and naga
/// requires each query to be a local of the function that drives it, so each
/// loop calls back into the Slang candidate test emitted in `source`.
/// Hardware-opaque instances (see `hardware_opaque` in scene.rs) commit
/// without that test; every other triangle reaches it.
fn ray_query_source(source: &str) -> Result<String, String> {
    // Slang suffixes each function name with `_N`.
    let callback = |name: &str| {
        let start = source.find(&format!("fn {name}_"))? + 3;
        let end = start + source[start..].find('(')?;
        Some(source[start..end].to_owned())
    };
    let closest = callback("hardwareClosestCandidate");
    let occluded = callback("hardwareOccludedCandidate");
    if closest.is_none() && occluded.is_none() {
        return Err("hardware traversal module has no candidate callback".into());
    }
    let mut prelude = format!(
        "enable wgpu_ray_query;\n\
         @group(0) @binding({TLAS_BINDING}) var penumbraTlas: acceleration_structure;\n",
    );
    if let Some(closest) = closest {
        prelude.push_str(&format!(
            "fn penumbraRayQueryClosest(origin: vec3<f32>, direction: vec3<f32>, limit: f32, any_hit: bool, mask: u32) -> bool {{
    var query: ray_query;
    let flags = select(RAY_FLAG_NONE, RAY_FLAG_TERMINATE_ON_FIRST_HIT, any_hit);
    rayQueryInitialize(&query, penumbraTlas, RayDesc(flags, mask, 0.0, limit, origin, direction));
    while (rayQueryProceed(&query)) {{
        let candidate = rayQueryGetCandidateIntersection(&query);
        if ({closest}(candidate.instance_custom_data, candidate.primitive_index)) {{
            rayQueryConfirmIntersection(&query);
            if (any_hit) {{ rayQueryTerminate(&query); }}
        }}
    }}
    // A hardware-opaque hit commits without the candidate test. Run it once
    // here so its hit record matches the shader's triangle test; a hit already
    // confirmed above is not closer than itself and is kept unchanged.
    let committed = rayQueryGetCommittedIntersection(&query);
    if (committed.kind != RAY_QUERY_INTERSECTION_NONE) {{
        {closest}(committed.instance_custom_data, committed.primitive_index);
    }}
    return true;
}}
"
        ));
    }
    if let Some(occluded) = occluded {
        prelude.push_str(&format!(
            "fn penumbraRayQueryOccluded(origin: vec3<f32>, direction: vec3<f32>, limit: f32, mask: u32) -> u32 {{
    var query: ray_query;
    rayQueryInitialize(&query, penumbraTlas, RayDesc(RAY_FLAG_TERMINATE_ON_FIRST_HIT, mask, 0.000025, limit, origin, direction));
    while (rayQueryProceed(&query)) {{
        let candidate = rayQueryGetCandidateIntersection(&query);
        if ({occluded}(candidate.instance_custom_data, candidate.primitive_index)) {{
            rayQueryTerminate(&query);
            return 1u;
        }}
    }}
    // A hardware-opaque hit ends the query without reaching the loop.
    return select(0u, 1u, rayQueryGetCommittedIntersection(&query).kind != RAY_QUERY_INTERSECTION_NONE);
}}
"
        ));
    }
    Ok(prelude + source)
}

pub(crate) fn vulkan_compute_source(
    source: &str,
    entry: &str,
    layout_entries: &[BindGroupLayoutEntry],
) -> Result<VulkanComputeSource, String> {
    let hardware = hardware_traversal_entry(entry);
    let prefixed;
    let source = if hardware {
        prefixed = ray_query_source(source)?;
        prefixed.as_str()
    } else {
        source
    };
    let module =
        naga::front::wgsl::parse_str(source).map_err(|error| error.emit_to_string(source))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        if hardware {
            naga::valid::Capabilities::RAY_QUERY
        } else {
            naga::valid::Capabilities::empty()
        },
    )
    .validate(&module)
    .map_err(|error| error.to_string())?;
    let point = module
        .entry_points
        .iter()
        .find(|point| point.name == entry)
        .ok_or_else(|| format!("missing compute entry {entry}"))?;
    let size = point.workgroup_size;
    let mut options = naga::back::spv::Options {
        fake_missing_bindings: false,
        bounds_check_policies: naga::proc::BoundsCheckPolicies {
            index: naga::proc::BoundsCheckPolicy::ReadZeroSkipWrite,
            buffer: naga::proc::BoundsCheckPolicy::Restrict,
            image_load: naga::proc::BoundsCheckPolicy::ReadZeroSkipWrite,
            binding_array: naga::proc::BoundsCheckPolicy::Restrict,
        },
        ..default()
    };
    for (physical, binding) in layout_entries.iter().enumerate() {
        options.binding_map.insert(
            naga::ResourceBinding {
                group: 0,
                binding: binding.binding,
            },
            naga::back::spv::BindingInfo {
                descriptor_set: 0,
                binding: physical as u32,
                ..default()
            },
        );
    }
    let mut words = naga::back::spv::write_vec(
        &module,
        &info,
        &options,
        Some(&naga::back::spv::PipelineOptions {
            shader_stage: naga::ShaderStage::Compute,
            entry_point: entry.into(),
        }),
    )
    .map_err(|error| error.to_string())?;
    // Naga has no function-control option. Kernels that only traverse (closest
    // hit and shadow queries) leave inlining to the driver: forced calls there
    // spill the traversal stack on every step, costing about 3.6 ms and 3 ms
    // at 1080p Sponza. Kernels that include hit shading keep helper calls,
    // because inlining their call graph takes the NVIDIA compiler minutes.
    let traversal_only = hardware || matches!(entry, "traceDeferred" | "traceVisibility");
    let mut offset = 5;
    while offset < words.len() {
        let count = (words[offset] >> 16) as usize;
        if words[offset] & 0xffff == 54 {
            let control = &mut words[offset + 3];
            *control &= !3;
            if !traversal_only {
                *control |= 2;
            }
        }
        offset += count.max(1);
    }
    Ok((words, (size[0], size[1], size[2])))
}

fn shared_pipeline_pending(
    state: &CachedPipelineState,
    entry: &str,
    status: &RadianceCascadesStatus,
) -> bool {
    match state {
        CachedPipelineState::Ok(..) => false,
        CachedPipelineState::Err(error)
            if matches!(
                error,
                bevy::shader::ShaderCacheError::ProcessShaderError(..)
                    | bevy::shader::ShaderCacheError::CreateShaderModule(..)
            ) =>
        {
            status.record_failure(format!("GI shared pipeline {entry} failed: {error}"));
            false
        }
        _ => true,
    }
}

fn compute_pipeline(
    device: RenderDevice,
    layouts: [BindGroupLayout; 1],
    source: Arc<str>,
    entry: &'static str,
    pipeline_cache: Option<wgpu::PipelineCache>,
    spirv: bool,
    ray_query: bool,
) -> Result<ComputePipeline, String> {
    if hardware_traversal_entry(entry) && !(spirv && ray_query) {
        return Err("hardware traversal needs Vulkan SPIR-V passthrough and ray queries".into());
    }
    let device = device.wgpu_device();
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    // Populated Split history loses the device on the tested NVIDIA driver
    // with the custom no-inline SPIR-V merge. Keep merge on wgpu's standard
    // WGSL path; the driver cache amortizes its longer first compilation.
    let module = if spirv && entry != "merge" {
        let (words, workgroup_size) =
            vulkan_compute_source(&source, entry, &compute_layout(ray_query).entries)?;
        // Naga validates our embedded WGSL and emits bounds-checked SPIR-V for the exact layouts above.
        unsafe {
            device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: Some(entry),
                spirv: Some(std::borrow::Cow::Owned(words)),
                entry_points: std::borrow::Cow::Owned(vec![wgpu::PassthroughShaderEntryPoint {
                    name: std::borrow::Cow::Borrowed(entry),
                    workgroup_size,
                }]),
                ..default()
            })
        }
    } else {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(entry),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(&source)),
        })
    };
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("penumbra_compute"),
        bind_group_layouts: &[Some(&layouts[0])],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(&layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: default(),
        cache: pipeline_cache.as_ref(),
    });
    match block_on(validation.pop()) {
        Some(error) => Err(error.to_string()),
        None => Ok(pipeline.into()),
    }
}

#[derive(Resource)]
struct PipelineShaders {
    handles: Vec<Handle<Shader>>,
    max_concurrent: usize,
    loaded: Arc<AtomicU64>,
    sources: ComputeShaderSources,
    status: RadianceCascadesStatus,
}

#[derive(Resource)]
struct ShaderCompilation {
    compiler: SlangCompiler,
    version: String,
    handles: Vec<Handle<Shader>>,
    loaded: Arc<AtomicU64>,
    sources: ComputeShaderSources,
    pending_loaded: u64,
    started: u64,
    jobs: Vec<Task<(usize, Result<Shader, SlangLoaderError>)>>,
    status: RadianceCascadesStatus,
}

fn required_shader_modules(settings: impl Iterator<Item = RadianceCascades>) -> u64 {
    let mut required = 0;
    for settings in settings {
        if settings.intensity <= 0.0 {
            continue;
        }
        required |= wanted_pipelines(settings);
    }
    if required != 0 {
        required |= 15 << COMPUTE_ENTRIES.len(); // composition, TLAS refit and deformation
    }
    required
}

fn compile_view_shaders(
    mut compilation: ResMut<ShaderCompilation>,
    mut shaders: ResMut<Assets<Shader>>,
    views: Query<(&Camera, &RadianceCascades)>,
) {
    let compilation = &mut *compilation;
    // Allow Bevy's extraction to see the shared raster/refit shaders first.
    if compilation.pending_loaded != 0 {
        compilation
            .loaded
            .fetch_or(compilation.pending_loaded, Ordering::Release);
        compilation.pending_loaded = 0;
    }
    compilation.jobs.retain_mut(|job| {
        let Some((index, result)) = block_on(poll_once(job)) else {
            return true;
        };
        match result {
            Ok(shader) => {
                if index < COMPUTE_ENTRIES.len() {
                    let bevy::shader::Source::Wgsl(ref source) = shader.source else {
                        unreachable!("Slang produces WGSL for compute stages");
                    };
                    compilation.sources.lock().unwrap()[index] = Some(Arc::from(source.as_ref()));
                    compilation.pending_loaded |= 1 << index;
                } else if let Err(error) = shaders.insert(compilation.handles[index].id(), shader) {
                    error!(%error, "GI shader handle expired");
                } else {
                    compilation.pending_loaded |= 1 << index;
                }
            }
            Err(error) => {
                compilation
                    .status
                    .record_failure(format!("GI shader stage {index} compilation failed: {error}"));
                error!(%error, index, "GI shader compilation failed; native lighting remains available");
            }
        }
        false
    });
    let required = required_shader_modules(
        views
            .iter()
            .filter(|(camera, _)| camera.is_active)
            .map(|(_, settings)| settings.sanitized()),
    );
    let modules: Vec<_> = shader_modules().collect();
    // Start the long ray-tracing kernels first, so their pipeline creation can
    // overlap the smaller stages instead of extending the end of startup.
    for index in [
        17,
        18,
        20,
        7,
        12,
        15,
        COMPUTE_ENTRIES.len(),
        COMPUTE_ENTRIES.len() + 1,
    ]
    .into_iter()
    .chain(0..modules.len())
    {
        if compilation.jobs.len() >= 4 {
            break;
        }
        if required & (1 << index) == 0 || compilation.started & (1 << index) != 0 {
            continue;
        }
        compilation.started |= 1 << index;
        let compiler = compilation.compiler.clone();
        let version = compilation.version.clone();
        let (file, entries) = modules[index];
        compilation
            .jobs
            .push(AsyncComputeTaskPool::get().spawn(async move {
                let started = std::time::Instant::now();
                let cache_dir = shader_cache_dir();
                let result = cached_compile(&compiler, file, entries, &cache_dir, Some(&version));
                info!(file, ?entries, elapsed = ?started.elapsed(), success = result.is_ok(), "GI Slang stage ready");
                (index, result)
            }));
    }
    compilation
        .status
        .set_shader_compile_jobs(compilation.jobs.len());
}

#[derive(Component, Clone, Copy)]
struct GiLightingState(bool);

// Current preparation is checked after the receiver pass is encoded.
#[derive(Component)]
struct GiPreparedView {
    settings: RadianceCascades,
    target_size: Option<UVec2>,
    viewport: Option<URect>,
    layers: RenderLayers,
    environment: Option<crate::RadianceCascadesEnvironment>,
    environment_revision: u64,
    view_budget: Option<u64>,
    pipelines_ready: bool,
    receiver_ready: Arc<AtomicBool>,
    scene_history: SceneHistorySnapshot,
}

fn camera_descriptor(camera: Option<&ExtractedCamera>) -> (Option<UVec2>, Option<URect>) {
    camera.map_or((None, None), |camera| {
        let viewport = camera
            .viewport
            .as_ref()
            .map(|viewport| {
                URect::from_corners(
                    viewport.physical_position,
                    viewport.physical_position + viewport.physical_size,
                )
            })
            .or_else(|| {
                camera
                    .physical_viewport_size
                    .map(|size| URect::from_corners(UVec2::ZERO, size))
            });
        (camera.physical_target_size, viewport)
    })
}

impl GiPreparedView {
    fn matches(
        &self,
        settings: RadianceCascades,
        camera: Option<&ExtractedCamera>,
        layers: Option<&RenderLayers>,
        environment: Option<&crate::RadianceCascadesEnvironment>,
        environment_revision: u64,
        view_budget: Option<u64>,
    ) -> bool {
        let (target_size, viewport) = camera_descriptor(camera);
        self.settings == settings
            && self.target_size == target_size
            && self.viewport == viewport
            && self.layers == layers.cloned().unwrap_or_default()
            && self.environment.as_ref() == environment
            && self.environment_revision == environment_revision
            && self.view_budget == view_budget
            && self.pipelines_ready
            && self.receiver_ready.load(Ordering::Relaxed)
    }
}

// Keep native preparation intact. Ownership is decided after this frame's
// identity draw and GPU refits, before any native lighting pass consumes uniforms.
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
fn finalize_view_lighting(
    view: ViewQuery<(
        Option<&RadianceCascades>,
        Option<&ViewBuffers>,
        Option<&ViewBindings>,
        Option<&crate::receiver::ReceiverTexture>,
        Option<&GiPreparedView>,
        &ExtractedCamera,
        Option<&RenderLayers>,
        Option<&crate::RadianceCascadesEnvironment>,
    )>,
    mut commands: Commands,
    scene: Res<GiScene>,
    gpu_scene: Option<Res<GpuScene>>,
    pipelines: Res<Pipelines>,
    cache: Res<PipelineCache>,
    environments: Option<Res<crate::environment::EnvironmentTables>>,
    budget: Option<Res<SceneBudget>>,
    mut startup: Local<HashMap<Entity, String>>,
) {
    let entity = view.entity();
    let mut report = |status: &str| {
        if startup
            .get(&entity)
            .is_none_or(|previous| previous != status)
        {
            info!(?entity, status, "Split Radiance Cascades startup");
            startup.insert(entity, status.to_owned());
        }
    };
    let (settings, buffers, bindings, receiver, prepared, camera, layers, environment) =
        view.into_inner();
    let Some(settings) = settings
        .map(|settings| settings.sanitized())
        .filter(|settings| settings.intensity > 0.0)
    else {
        commands.entity(entity).remove::<GiLightingState>();
        report("disabled by camera intensity");
        return;
    };
    commands.entity(entity).insert(GiLightingState(false));
    let required = required_pipelines(settings);
    let pending: Vec<_> = (0..COMPUTE_ENTRIES.len())
        .filter_map(|i| {
            (required & (1 << i) != 0 && pipelines.pipeline(i).is_none())
                .then_some(COMPUTE_ENTRIES[i])
        })
        .collect();
    if !pending.is_empty() {
        report(&format!(
            "waiting for GPU compute pipelines: {}",
            pending.join(", ")
        ));
        return;
    }
    let Some(buffers) = buffers else {
        report("waiting for view buffers");
        return;
    };
    let Some(_bindings) = bindings else {
        report("waiting for view bindings");
        return;
    };
    let Some(receiver) = receiver else {
        report("waiting for receiver texture");
        return;
    };
    let Some(prepared) = prepared else {
        report("waiting for prepared scene snapshot");
        return;
    };
    if !scene.ready
        || gpu_scene
            .as_deref()
            .is_none_or(|gpu| !gpu.ready_for_revision(scene.revision))
        || !receiver.ready.load(Ordering::Relaxed)
    {
        report("waiting for scene residency or receiver raster pass");
        return;
    }
    let environment_revision = environment
        .and_then(|source| {
            environments
                .as_ref()?
                .tables
                .get(&source.image.id())
                .map(|table| table.revision)
        })
        .unwrap_or(0);
    if !prepared.matches(
        settings,
        Some(camera),
        layers,
        environment,
        environment_revision,
        budget.as_ref().map(|budget| budget.view_bytes),
    ) {
        report("waiting for current view preparation");
        return;
    }
    if prepared.scene_history.revision != scene.revision
        || prepared.scene_history.geometry_revision != scene.geometry_revision
    {
        report("waiting for current scene snapshot");
        return;
    }
    let required = required_pipelines(buffers.settings);
    if (0..COMPUTE_ENTRIES.len())
        .any(|i| required & (1 << i) != 0 && pipelines.pipeline(i).is_none())
        || cache.get_render_pipeline(pipelines.composite).is_none()
        || cache.get_compute_pipeline(pipelines.composite_anchors).is_none()
    {
        report("waiting for composite pipeline");
        return;
    }
    commands.entity(entity).insert(GiLightingState(true));
    report("active");
}
#[allow(clippy::type_complexity)]
fn cleanup_view_allocations(
    mut commands: Commands,
    views: Query<
        (Entity, Option<&RadianceCascades>, Has<ExtractedCamera>),
        Or<(
            With<ViewBuffers>,
            With<ViewBindings>,
            With<GiPreparedView>,
            With<crate::receiver::ReceiverTexture>,
        )>,
    >,
) {
    for (entity, settings, active) in &views {
        if !active || settings.is_none_or(|settings| settings.sanitized().intensity <= 0.0) {
            commands.entity(entity).remove::<(
                ViewBuffers,
                ViewBindings,
                GiPreparedView,
                crate::receiver::ReceiverTexture,
            )>();
        }
    }
}

pub(crate) fn install(
    app: &mut App,
    compiler: SlangCompiler,
    max_concurrent_pipeline_compiles: usize,
    status: RadianceCascadesStatus,
) {
    if app.get_sub_app(RenderApp).is_none() {
        return;
    }
    let compiler_version = match compiler_version(&compiler) {
        Ok(version) => version,
        Err(error) => {
            status.record_failure(format!("GI compiler initialization failed: {error}"));
            error!("Radiance Cascades disabled: {error}");
            return;
        }
    };
    let handles: Vec<_> = shader_modules()
        .map(|_| app.world().resource::<Assets<Shader>>().reserve_handle())
        .collect();
    let loaded = Arc::new(AtomicU64::new(0));
    let sources = Arc::new(Mutex::new(std::array::from_fn(|_| None)));
    app.insert_resource(ShaderCompilation {
        compiler,
        version: compiler_version,
        handles: handles.clone(),
        loaded: loaded.clone(),
        sources: sources.clone(),
        pending_loaded: 0,
        started: 0,
        jobs: Vec::new(),
        status: status.clone(),
    })
    .add_systems(Update, compile_view_shaders);
    crate::receiver::install(app);
    app.sub_app_mut(RenderApp)
        .insert_resource(PipelineShaders {
            handles,
            max_concurrent: max_concurrent_pipeline_compiles.max(1),
            loaded,
            sources,
            status,
        })
        .init_resource::<MemoryHighWater>()
        .init_resource::<RasterResidencyGate>()
        .add_systems(
            Render,
            cleanup_view_allocations
                .after(RenderSystems::ExtractCommands)
                .before(RenderSystems::CreateViews),
        )
        .init_resource::<SunShadowOwnership>()
        .add_systems(
            Render,
            own_sun_shadows
                .after(RenderSystems::ExtractCommands)
                .before(RenderSystems::CreateViews),
        )
        .add_systems(RenderStartup, init_pipelines)
        .add_systems(
            Render,
            hardware::prepare_hardware_traversal.in_set(RenderSystems::PrepareResources),
        )
        .add_systems(
            Render,
            prepare
                .in_set(RenderSystems::PrepareBindGroups)
                .after(crate::receiver::PrepareReceiverTextures),
        )
        .add_systems(
            Render,
            update_raster_residency
                .after(RenderSystems::PrepareMeshes)
                .before(RenderSystems::CreateViews),
        )
        .add_systems(
            Render,
            filter_raster_visibility
                .in_set(RenderSystems::Queue)
                .before(RenderSystems::QueueMeshes),
        )
        .add_systems(
            RenderGraph,
            refit_scene
                .after(RenderGraphSystems::Begin)
                .before(RenderGraphSystems::Render),
        )
        .add_systems(
            RenderGraph,
            hardware::build_hardware_traversal
                .after(refit_scene)
                .before(RenderGraphSystems::Render),
        )
        .add_systems(
            Core3d,
            finalize_view_lighting
                .in_set(Core3dSystems::MainPass)
                .after(crate::receiver::draw)
                .before(deferred_lighting)
                .before(screen_space_reflections)
                .before(main_opaque_pass_3d),
        )
        .add_systems(
            Core3d,
            trace_sun_shadows
                .in_set(Core3dSystems::MainPass)
                .after(crate::receiver::draw)
                .before(deferred_lighting)
                .before(screen_space_reflections)
                .before(main_opaque_pass_3d),
        )
        .add_systems(
            Core3d,
            render
                .in_set(Core3dSystems::MainPass)
                .after(crate::receiver::draw)
                .after(main_opaque_pass_3d)
                .before(main_transparent_pass_3d),
        );
}

/// The Split compute layout. `ray_query` appends the hardware TLAS binding;
/// it is last, so every other physical binding index is unchanged.
fn compute_layout(ray_query: bool) -> BindGroupLayoutDescriptor {
    let mut layout = BindGroupLayoutDescriptor::new(
        "split_rc_compute",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<Parameters>(false),
                storage_buffer_read_only::<Vec<UVec4>>(false),
                storage_buffer_read_only::<Vec<UVec4>>(false),
                storage_buffer_read_only::<Vec<GiLight>>(false),
                texture_2d(TextureSampleType::Depth),
                texture_2d(TextureSampleType::Float { filterable: false }),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_sized(false, None),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                texture_2d(TextureSampleType::Float { filterable: false }),
                storage_buffer_sized(false, None),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d_array(TextureSampleType::Float { filterable: false }),
                storage_buffer_read_only_sized(false, None),
                texture_2d_array(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rg32Float, StorageTextureAccess::WriteOnly),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Uint),
                texture_2d(TextureSampleType::Uint),
                texture_2d_array(TextureSampleType::Uint),
                texture_2d_array(TextureSampleType::Uint),
                texture_2d(TextureSampleType::Float { filterable: false }),
            ),
        ),
    );
    layout.entries.push(
        texture_2d(TextureSampleType::Float { filterable: false }).build(37, ShaderStages::COMPUTE),
    );
    layout.entries.push(
        texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly)
            .build(38, ShaderStages::COMPUTE),
    );
    for binding in [39, 40] {
        layout.entries.push(
            texture_2d(TextureSampleType::Float { filterable: false })
                .build(binding, ShaderStages::COMPUTE),
        );
    }
    // Current and previous ray-traced GI reservoirs.
    layout
        .entries
        .push(storage_buffer_sized(false, None).build(42, ShaderStages::COMPUTE));
    layout
        .entries
        .push(storage_buffer_read_only_sized(false, None).build(43, ShaderStages::COMPUTE));
    // Ray-traced GI world radiance cache: keys, cells, shading queue.
    for binding in [44, 45, 46] {
        layout
            .entries
            .push(storage_buffer_sized(false, None).build(binding, ShaderStages::COMPUTE));
    }
    layout
        .entries
        .retain(|entry| !matches!(entry.binding, 5 | 17 | 18 | 19 | 21 | 24 | 30));
    if ray_query {
        layout.entries.push(BindGroupLayoutEntry {
            binding: TLAS_BINDING,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::AccelerationStructure {
                vertex_return: false,
            },
            count: None,
        });
    }
    // The SPIR-V passthrough numbers bindings by their position here, which
    // must match wgpu's binding order, so the TLAS cannot trail 42 and up.
    layout.entries.sort_by_key(|entry| entry.binding);
    layout
}

fn init_pipelines(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    shaders: Res<PipelineShaders>,
    adapter: Res<RenderAdapterInfo>,
    render_adapter: Res<RenderAdapter>,
) {
    let driver_spirv = adapter.backend == wgpu::Backend::Vulkan
        && device
            .features()
            .contains(wgpu::Features::PASSTHROUGH_SHADERS);
    // Hardware traversal runs only on the SPIR-V path: Slang's WGSL has no ray
    // queries, and the WGSL prelude is added before naga emits SPIR-V.
    let ray_query = driver_spirv
        && device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
    shaders
        .status
        .set_shader_backend(if driver_spirv { 2 } else { 1 });
    shaders.status.set_hardware_ray_queries_supported(ray_query);
    info!(backend = ?shaders.status.shader_backend(), ray_query, "GI shader backend selected");
    let handles = &shaders.handles;
    let driver_cache_path = device
        .features()
        .contains(wgpu::Features::PIPELINE_CACHE)
        .then(|| wgpu::util::pipeline_cache_key(&adapter))
        .flatten()
        .map(|key| shader_cache_dir().join(format!("{key}.bin")));
    let driver_cache = driver_cache_path.as_ref().map(|path| {
        let bytes = std::fs::read(path).ok();
        info!(hit = bytes.is_some(), path = %path.display(), "GI device pipeline cache loaded");
        // Only data previously produced by wgpu is accepted; fallback handles
        // stale driver/device versions or a truncated cache.
        unsafe {
            device
                .wgpu_device()
                .create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                    label: Some("split_rc_driver_cache"),
                    data: bytes.as_deref(),
                    fallback: true,
                })
        }
    });
    let compute_layout = compute_layout(ray_query);
    let mut composite_layout = BindGroupLayoutDescriptor::new(
        "split_rc_composite",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                uniform_buffer::<Parameters>(false),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Depth),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Uint),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Uint),
                texture_2d(TextureSampleType::Float { filterable: false }),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Float { filterable: false }),
                // Not bound: the anchor outputs of compositeAnchors.
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Float { filterable: false }),
            ),
        ),
    );
    composite_layout
        .entries
        .retain(|entry| !matches!(entry.binding, 1 | 4 | 14 | 15));
    let composite_pipeline = |label: &'static str, entry: &'static str| {
        cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some(label.into()),
        layout: vec![composite_layout.clone()],
        vertex: VertexState {
            shader: handles[COMPUTE_ENTRIES.len()].clone(),
            entry_point: Some("vertex".into()),
            ..default()
        },
        fragment: Some(FragmentState {
            shader: handles[COMPUTE_ENTRIES.len()].clone(),
            entry_point: Some(entry.into()),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba16Float,
                blend: Some(BlendState {
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
                }),
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        ..default()
    })
    };
    let composite = composite_pipeline("split_rc_composite", "fragment");
    // Writes traced sun visibility as depth into Bevy's directional shadow texture.
    let sun_shadow_depth = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("split_rc_sun_shadow_depth".into()),
        layout: vec![composite_layout.clone()],
        vertex: VertexState {
            shader: handles[COMPUTE_ENTRIES.len()].clone(),
            entry_point: Some("vertex".into()),
            ..default()
        },
        fragment: Some(FragmentState {
            shader: handles[COMPUTE_ENTRIES.len()].clone(),
            entry_point: Some("sunShadowDepth".into()),
            targets: Vec::new(),
            ..default()
        }),
        depth_stencil: Some(DepthStencilState {
            format: bevy::pbr::CORE_3D_SHADOW_MAP_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(CompareFunction::Always),
            stencil: default(),
            bias: default(),
        }),
        ..default()
    });
    let composite_anchors = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("split_rc_composite_anchors".into()),
        layout: vec![composite_anchor_layout()],
        shader: handles[COMPUTE_ENTRIES.len()].clone(),
        entry_point: Some("compositeAnchors".into()),
        ..default()
    });
    let refit = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("rc_tlas_refit".into()),
        layout: vec![refit_layout()],
        shader: handles[COMPUTE_ENTRIES.len() + 1].clone(),
        entry_point: Some("refitTlas".into()),
        ..default()
    });
    commands.insert_resource(Pipelines {
        compute_layout,
        composite_layout,
        compute: std::array::from_fn(|_| None),
        started: 0,
        jobs: Vec::new(),
        driver_cache,
        driver_spirv,
        ray_query,
        indirect_execution: render_adapter
            .get_downlevel_capabilities()
            .flags
            .contains(wgpu::DownlevelFlags::INDIRECT_EXECUTION),
        driver_cache_path,
        driver_cache_save: None,
        driver_cache_dirty: false,
        composite,
        sun_shadow_depth,
        composite_anchors,
        refit,
        deform: ["deformVertices", "refitBlas"].map(|entry| {
            cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(format!("rc_{entry}").into()),
                layout: vec![deformation_layout()],
                shader: handles[COMPUTE_ENTRIES.len() + 2 + usize::from(entry == "refitBlas")]
                    .clone(),
                entry_point: Some(entry.into()),
                ..default()
            })
        }),
    });
}

#[derive(Resource, Default)]
struct GpuScene {
    pending_geometry: Option<PendingGeometry>,
    pending_atlas: Option<PendingAtlas>,
    pending_acceleration: Option<PendingAcceleration>,
    pending_streams: Option<PendingWords>,
    pending_deform_order: Option<PendingWords>,
    pending_deform_order_revision: Option<(u64, u64)>,
    pending_refit_order: Option<PendingWords>,
    pending_refit_revision: Option<u64>,
    pending_poses: Option<PendingWords>,
    pending_dynamic: Option<PendingDynamic>,
    pending_material_tail: Option<PendingMaterialTail>,
    revision: u64,
    geometry_revision: u64,
    dynamic_revision: u64,
    mesh_revision: u64,
    topology_revision: u64,
    instance_residency: Arc<HashMap<Entity, u64>>,
    acceleration_words: usize,
    prefix_words: usize,
    blas_words: Arc<Vec<UVec4>>,
    material_words: Vec<UVec4>,
    geometry: Option<Buffer>,
    geometry_pages: Option<Texture>,
    geometry_pages_view: Option<TextureView>,
    geometry_pages_input_view: Option<TextureView>,
    geometry_pages_output_view: Option<TextureView>,
    geometry_page_placeholder: Option<Texture>,
    geometry_page_placeholder_view: Option<TextureView>,
    geometry_output_placeholder: Option<Texture>,
    geometry_output_placeholder_view: Option<TextureView>,
    geometry_paged: bool,
    geometry_page_side: u32,
    geometry_bytes: u64,
    scene_texture: Option<Texture>,
    scene_texture_view: Option<TextureView>,
    scene_page_placeholder: Option<Texture>,
    scene_page_placeholder_view: Option<TextureView>,
    scene_paged: bool,
    scene_page_config: UVec4,
    scene_overlay_config: UVec4,
    scene_range_map: Vec<UVec4>,
    scene_instance_overlay_base: usize,
    scene_bytes: u64,
    nodes: Option<Buffer>,
    dynamic_baseline: Option<DynamicBaseline>,
    refit_nodes: Option<Buffer>,
    refit_params: DynamicUniformBuffer<SceneDispatch>,
    refit_offsets: Vec<(u32, u32)>,
    pending_refit: bool,
    textures: Arc<crate::material::TextureAtlas>,
    atlas: Option<Texture>,
    atlas_view: Option<TextureView>,
    sampled_fallback: Option<Texture>,
    sampled_fallback_view: Option<TextureView>,
    deformation_words: Arc<Vec<UVec4>>,
    deformation: Option<Buffer>,
    poses: Option<Buffer>,
    pose_words: Arc<Vec<UVec4>>,
    deform_order: Option<Buffer>,
    deform_order_revision: Option<(u64, u64)>,
    deform_params: DynamicUniformBuffer<SceneDispatch>,
    deform_offsets: Vec<(u32, u32)>,
    deform_vertex_offset: u32,
    pending_deformation: bool,
}

impl GpuScene {
    fn abandon_pending_material_tail(&mut self) {
        if let Some(pending) = self.pending_material_tail.take() {
            self.material_words = pending.applied_words;
        }
    }
}

#[derive(Resource, Default)]
struct RasterResidencyGate {
    gated: HashSet<MainEntity>,
    released: HashSet<MainEntity>,
    scene_keys: Option<Arc<HashMap<Entity, u64>>>,
    gpu_keys: Option<Arc<HashMap<Entity, u64>>>,
    pending_entities: Option<Arc<HashSet<Entity>>>,
    scene_textures: Option<Arc<crate::material::TextureAtlas>>,
    gpu_textures: Option<Arc<crate::material::TextureAtlas>>,
    scene_ready: bool,
}

impl GpuScene {
    // `refit_scene` runs in RenderGraph::Begin and clears these only after
    // queue submission; RenderGraph::Render is ordered after it on the same queue.
    fn ready_for_revision(&self, revision: u64) -> bool {
        self.revision == revision && !self.pending_deformation && !self.pending_refit
    }

    fn scene_page_view(&self) -> Option<&TextureView> {
        if self.scene_paged {
            self.scene_texture_view.as_ref()
        } else {
            self.scene_page_placeholder_view.as_ref()
        }
    }
}

fn update_raster_residency(
    scene: Res<GiScene>,
    gpu_scene: Option<Res<GpuScene>>,
    mut gate: ResMut<RasterResidencyGate>,
    mut dirty: Option<ResMut<bevy::render::camera::DirtySpecializations>>,
) {
    let current = &scene.data.instance_residency;
    let committed = gpu_scene.as_deref();
    let current_keys = Some(current.clone());
    let committed_keys = committed.map(|gpu| gpu.instance_residency.clone());
    let pending_entities = Some(scene.pending_entities.clone());
    let current_textures = Some(scene.data.textures.clone());
    let committed_textures = committed.map(|gpu| gpu.textures.clone());
    if gate.scene_ready == scene.ready
        && same_arc(gate.scene_keys.as_ref(), current_keys.as_ref())
        && same_arc(gate.gpu_keys.as_ref(), committed_keys.as_ref())
        && same_arc(gate.pending_entities.as_ref(), pending_entities.as_ref())
        && same_arc(gate.scene_textures.as_ref(), current_textures.as_ref())
        && same_arc(gate.gpu_textures.as_ref(), committed_textures.as_ref())
    {
        gate.released.clear();
        return;
    }
    let mut gated = HashSet::new();
    if let Some(gpu_scene) =
        committed.filter(|gpu| scene.ready && gpu.ready_for_revision(scene.revision))
    {
        // ponytail: a shared atlas replacement gates every GI instance; split
        // atlas generations only if asset edits make this brief stall visible.
        let atlas_changed = !Arc::ptr_eq(&gpu_scene.textures, &scene.data.textures);
        for (&entity, &key) in current.iter() {
            if atlas_changed || gpu_scene.instance_residency.get(&entity) != Some(&key) {
                gated.insert(MainEntity::from(entity));
            }
        }
        for entity in gpu_scene.instance_residency.keys() {
            if !current.contains_key(entity) {
                gated.insert(MainEntity::from(*entity));
            }
        }
        for entity in scene.pending_entities.iter() {
            if !current.contains_key(entity) {
                gated.insert(MainEntity::from(*entity));
            }
        }
    }
    gate.released = gate.gated.difference(&gated).copied().collect();
    if let Some(dirty) = dirty.as_deref_mut() {
        dirty
            .changed_renderables
            .extend(gate.released.iter().copied());
    }
    gate.gated = gated;
    gate.scene_keys = current_keys;
    gate.gpu_keys = committed_keys;
    gate.pending_entities = pending_entities;
    gate.scene_textures = current_textures;
    gate.gpu_textures = committed_textures;
    gate.scene_ready = scene.ready;
}

fn same_arc<T>(left: Option<&Arc<T>>, right: Option<&Arc<T>>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => Arc::ptr_eq(left, right),
        (None, None) => true,
        _ => false,
    }
}

fn filter_raster_visibility(
    gate: Res<RasterResidencyGate>,
    mut cameras: Query<
        (
            Option<&RenderLayers>,
            &mut RenderVisibleEntities,
            &RadianceCascades,
        ),
        With<ExtractedView>,
    >,
    mut lights: Query<(Option<&RenderLayers>, &mut RenderShadowMapVisibleEntities)>,
    gpu_entities: Res<RenderGpuCulledEntities>,
) {
    if gate.gated.is_empty() && gate.released.is_empty() {
        return;
    }
    let mut active_camera_layers = Vec::new();
    for (layers, mut visible, settings) in &mut cameras {
        if settings.sanitized().intensity > 0.0 {
            let layers = layers.cloned().unwrap_or_default();
            active_camera_layers.push(layers.clone());
            filter_raster_entities(&mut visible, Some(&layers), &gate, &gpu_entities);
        }
    }
    for (layers, mut shadow_views) in &mut lights {
        let light_layers = layers.cloned().unwrap_or_default();
        if !active_camera_layers
            .iter()
            .any(|camera_layers| camera_layers.intersects(&light_layers))
        {
            continue;
        }
        for (_, visible) in shadow_views.subviews.values_mut() {
            filter_raster_entities(visible, Some(&light_layers), &gate, &gpu_entities);
        }
    }
}

fn filter_raster_entities(
    visible: &mut RenderVisibleEntities,
    layers: Option<&RenderLayers>,
    gate: &RasterResidencyGate,
    gpu_entities: &RenderGpuCulledEntities,
) {
    let meshes = visible.classes.entry(TypeId::of::<Mesh3d>()).or_default();
    filter_raster_class(meshes, layers, gate, gpu_entities);
}

fn filter_raster_class(
    meshes: &mut RenderVisibleEntitiesClass,
    layers: Option<&RenderLayers>,
    gate: &RasterResidencyGate,
    gpu_entities: &RenderGpuCulledEntities,
) {
    let added = &mut meshes.added_entities;
    let removed = &mut meshes.removed_entities;
    meshes.entities_cpu_culling.retain(|pair| {
        if !gate.gated.contains(&pair.1) {
            return true;
        }
        if let Ok(index) = added.binary_search_by_key(&pair.1, |(_, entity)| *entity) {
            added.remove(index);
        } else {
            removed.push(*pair);
        }
        false
    });

    let gated_gpu: Vec<_> = meshes
        .entities_gpu_culling
        .keys()
        .copied()
        .filter(|entity| gate.gated.contains(entity))
        .collect();
    for entity in gated_gpu {
        meshes.entities_gpu_culling.remove(&entity);
        if let Ok(index) = added.binary_search_by_key(&entity, |(_, main)| *main) {
            added.remove(index);
        } else {
            removed.push((Entity::PLACEHOLDER, entity));
        }
    }

    let mut restored = false;
    for entity in &gate.released {
        if meshes.entities_gpu_culling.contains_key(entity) {
            continue;
        }
        let Some(entity_layers) = gpu_entities.entities.get(entity) else {
            continue;
        };
        if layers.is_some_and(|view_layers| !view_layers.intersects(entity_layers)) {
            continue;
        }
        meshes
            .entities_gpu_culling
            .insert(*entity, Entity::PLACEHOLDER);
        added.push((Entity::PLACEHOLDER, *entity));
        restored = true;
    }
    if restored {
        added.sort_unstable_by_key(|(_, entity)| *entity);
    }
    meshes
        .removed_entities
        .sort_unstable_by_key(|(_, entity)| *entity);
}

struct PendingGeometry {
    buffer: Buffer,
    overlay_words: Arc<Vec<UVec4>>,
    overlay_cursor: usize,
    texture: Option<Texture>,
    view: Option<TextureView>,
    input_view: Option<TextureView>,
    output_view: Option<TextureView>,
    paged: bool,
    page_side: u32,
    allocated_bytes: u64,
    words: Arc<Vec<UVec4>>,
    revision: u64,
    cursor: usize,
}

#[derive(Clone, Copy)]
struct GeometryPages {
    side: u32,
    base_layers: u32,
    output_layers: u32,
    layers: u32,
    output_base_word: u32,
    bytes: u64,
}

fn geometry_pages(
    words: usize,
    output_vertices: usize,
    limits: &WgpuLimits,
) -> Option<GeometryPages> {
    if u32::try_from(words.max(1)).is_err() {
        return None;
    }
    let side = limits.max_texture_dimension_2d.clamp(1, 1024);
    let page_words = u64::from(side).pow(2);
    let base_layers = (words.max(1) as u64).div_ceil(page_words);
    let output_words = u64::try_from(output_vertices).ok()?.checked_mul(5)?;
    let output_layers = output_words.div_ceil(page_words);
    let layers = base_layers.checked_add(output_layers)?;
    let output_base_word = base_layers.checked_mul(page_words)?;
    let output_end = output_base_word.checked_add(output_words)?;
    if layers > u64::from(limits.max_texture_array_layers) {
        return None;
    }
    if output_base_word > u64::from(u32::MAX) || output_end > u64::from(u32::MAX) + 1 {
        return None;
    }
    Some(GeometryPages {
        side,
        base_layers: base_layers as u32,
        output_layers: output_layers as u32,
        layers: layers as u32,
        output_base_word: output_base_word as u32,
        bytes: layers.checked_mul(page_words)?.checked_mul(16)?,
    })
}

fn geometry_overlay_layout(
    deformation: &crate::deformation::DeformationData,
    output_base_word: u32,
) -> Option<Arc<Vec<UVec4>>> {
    if deformation.vertices == 0 {
        return Some(Arc::new(Vec::new()));
    }
    let stream_header = deformation.words.first()?;
    let job_count = stream_header.x as usize;
    let job_base = stream_header.y as usize;
    let map_base = stream_header.z as usize;
    if job_count == 0
        || job_base.checked_add(job_count.checked_mul(3)?)? != map_base
        || map_base.checked_add(deformation.vertices as usize)? > deformation.words.len()
    {
        return None;
    }
    let output_base = 1usize.checked_add(job_count)?;
    let mut words = Vec::with_capacity(output_base);
    words.push(UVec4::new(
        u32::try_from(job_count).ok()?,
        1,
        output_base_word,
        deformation.vertices,
    ));
    let mut output_vertex = 0u32;
    let mut previous_end = 0u32;
    for job in 0..job_count {
        let mapping = *deformation
            .words
            .get(job_base.checked_add(job.checked_mul(3)?)?)?;
        if mapping.y < previous_end || mapping.z == 0 {
            return None;
        }
        words.push(UVec4::new(mapping.y, mapping.z, output_vertex, 0));
        output_vertex = output_vertex.checked_add(mapping.z)?;
        previous_end = mapping.y.checked_add(mapping.z)?;
    }
    if output_vertex != deformation.vertices {
        return None;
    }
    Some(Arc::new(words))
}

#[derive(Clone)]
struct ScenePageLayout {
    side: u32,
    layers: u32,
    bytes: u64,
    page_config: UVec4,
    overlay_config: UVec4,
    range_map: Vec<UVec4>,
    instance_overlay_base: usize,
}

/// Streams the compacted static scene pages from the existing acceleration
/// arrays. Keeping source ranges avoids a second full-scene CPU allocation
/// while an oversized scene is staged for the GPU.
#[derive(Clone)]
struct ScenePageUpload {
    blas_words: Arc<Vec<UVec4>>,
    instance_words: Arc<Vec<UVec4>>,
    skipped_blas_ranges: Vec<(usize, usize)>,
    dynamic_instance_start: usize,
    static_blas_words: usize,
    word_count: usize,
    cursor: usize,
    blas_cursor: usize,
    range_cursor: usize,
    instance_cursor: usize,
}

impl ScenePageUpload {
    fn new(
        blas_words: Arc<Vec<UVec4>>,
        instance_words: Arc<Vec<UVec4>>,
        dynamic_nodes: &[(usize, usize)],
        dynamic_instance_start: usize,
        static_blas_words: usize,
        word_count: usize,
    ) -> Self {
        let skipped_blas_ranges = dynamic_nodes
            .iter()
            .map(|&(start, count)| (start * 2, (start + count) * 2))
            .collect();
        debug_assert_eq!(
            (static_blas_words + instance_words.len().saturating_sub(dynamic_instance_start))
                .max(1),
            word_count.max(1)
        );
        Self {
            blas_words,
            instance_words,
            skipped_blas_ranges,
            dynamic_instance_start,
            static_blas_words,
            word_count: word_count.max(1),
            cursor: 0,
            blas_cursor: 0,
            range_cursor: 0,
            instance_cursor: 0,
        }
    }

    fn empty() -> Self {
        Self::new(Arc::new(Vec::new()), Arc::new(Vec::new()), &[], 0, 0, 0)
    }

    fn append_next_bytes(&mut self, output: &mut Vec<u8>, max_words: usize) -> usize {
        let count = max_words.min(self.word_count.saturating_sub(self.cursor));
        if count == 0 {
            return 0;
        }

        let available_blas_words = self.static_blas_words;
        let available_instance_words = self
            .instance_words
            .len()
            .saturating_sub(self.dynamic_instance_start);
        if available_blas_words + available_instance_words == 0 {
            let zero = UVec4::ZERO;
            append_words_bytes(output, std::slice::from_ref(&zero));
            self.cursor += 1;
            return 1;
        }

        let mut remaining = count;
        while remaining > 0 {
            while let Some(&(skip_start, skip_end)) =
                self.skipped_blas_ranges.get(self.range_cursor)
            {
                if self.blas_cursor < skip_start {
                    break;
                }
                if self.blas_cursor < skip_end {
                    self.blas_cursor = skip_end;
                }
                self.range_cursor += 1;
            }

            if self.blas_cursor < self.blas_words.len() {
                let segment_end = self
                    .skipped_blas_ranges
                    .get(self.range_cursor)
                    .map_or(self.blas_words.len(), |&(start, _)| start);
                let take = remaining.min(segment_end.saturating_sub(self.blas_cursor));
                debug_assert!(take > 0);
                append_words_bytes(
                    output,
                    &self.blas_words[self.blas_cursor..self.blas_cursor + take],
                );
                self.blas_cursor += take;
                remaining -= take;
            } else {
                let start = self.dynamic_instance_start + self.instance_cursor;
                let take = remaining.min(self.instance_words.len().saturating_sub(start));
                debug_assert!(take > 0);
                append_words_bytes(output, &self.instance_words[start..start + take]);
                self.instance_cursor += take;
                remaining -= take;
            }
        }
        self.cursor += count;
        count
    }
}

fn scene_page_upload_complete(paged: bool, page_upload: &ScenePageUpload) -> bool {
    !paged || page_upload.cursor == page_upload.word_count
}

#[cfg(test)]
mod scene_page_upload_tests {
    use super::*;

    #[test]
    fn streams_compacted_blas_and_instance_words_in_bounded_chunks() {
        let blas_words = Arc::new((0..8).map(UVec4::splat).collect::<Vec<_>>());
        let instance_words = Arc::new((8..12).map(UVec4::splat).collect::<Vec<_>>());
        let mut upload = ScenePageUpload::new(blas_words, instance_words, &[(1, 1)], 2, 6, 8);
        let mut bytes = Vec::new();
        for chunk in [1, 4, 1, 2] {
            assert_eq!(upload.append_next_bytes(&mut bytes, chunk), chunk);
        }
        assert_eq!(upload.cursor, upload.word_count);

        let expected = [0, 1, 4, 5, 6, 7, 10, 11]
            .into_iter()
            .map(UVec4::splat)
            .collect::<Vec<_>>();
        assert_eq!(bytes, words_bytes(&expected));
    }

    #[test]
    fn unpaged_scene_does_not_wait_for_empty_page_upload() {
        let empty = ScenePageUpload::empty();
        assert!(scene_page_upload_complete(false, &empty));
        assert!(!scene_page_upload_complete(true, &empty));
    }
}

type ScenePageRanges = (Vec<(usize, usize)>, Vec<UVec4>);

fn scene_page_ranges(data: &crate::scene::SceneData) -> Option<ScenePageRanges> {
    let prefix_words = data.acceleration_prefix.len();
    let tlas_nodes = u32::try_from(data.nodes.len()).ok()?;
    let blas_nodes = data.blas_words.len() / 2;
    let mut animated_nodes: Vec<_> = data
        .deformation
        .refit_nodes
        .iter()
        .filter_map(|&node| node.checked_sub(tlas_nodes))
        .filter(|&node| (node as usize) < blas_nodes)
        .map(|node| node as usize)
        .collect();
    animated_nodes.sort_unstable();
    animated_nodes.dedup();

    let mut ranges = Vec::<(usize, usize)>::new();
    for node in animated_nodes {
        if let Some((start, count)) = ranges.last_mut()
            && start.checked_add(*count) == Some(node)
        {
            *count = count.checked_add(1)?;
        } else {
            ranges.push((node, 1));
        }
    }

    let node_overlay_base = prefix_words.checked_add(ranges.len())?;
    let mut mutable_before = 0usize;
    let mut range_map = Vec::with_capacity(ranges.len());
    for &(start, count) in &ranges {
        let word_offset = node_overlay_base.checked_add(mutable_before.checked_mul(2)?)?;
        range_map.push(UVec4::new(
            u32::try_from(start).ok()?,
            u32::try_from(count).ok()?,
            u32::try_from(word_offset).ok()?,
            u32::try_from(mutable_before).ok()?,
        ));
        mutable_before = mutable_before.checked_add(count)?;
    }
    Some((ranges, range_map))
}

fn scene_page_data(
    data: &crate::scene::SceneData,
    material_start: usize,
    limits: &WgpuLimits,
    memory_budget: u64,
) -> Option<(ScenePageLayout, ScenePageUpload, Vec<UVec4>)> {
    let prefix_words = data.acceleration_prefix.len();
    let instance_base = prefix_words.checked_add(data.blas_words.len())?;
    let dynamic_instance_words = material_start;
    let (ranges, range_map) = scene_page_ranges(data)?;
    let mutable_nodes = ranges
        .iter()
        .try_fold(0usize, |total, (_, count)| total.checked_add(*count))?;
    let mutable_node_words = mutable_nodes.checked_mul(2)?;
    let static_blas_words = data.blas_words.len().checked_sub(mutable_node_words)?;
    let static_instance_words = data
        .instance_words
        .len()
        .checked_sub(dynamic_instance_words)?;
    let page_word_count = static_blas_words.checked_add(static_instance_words)?.max(1);
    let overlay_words = prefix_words
        .checked_add(ranges.len())?
        .checked_add(mutable_node_words)?
        .checked_add(dynamic_instance_words)?;
    let overlay_bytes = (overlay_words.max(1) as u64).saturating_mul(16);
    if overlay_bytes > limits.max_storage_buffer_binding_size
        || u32::try_from(prefix_words).is_err()
        || u32::try_from(instance_base).is_err()
        || u32::try_from(dynamic_instance_words).is_err()
        || u32::try_from(static_blas_words).is_err()
        || u32::try_from(page_word_count).is_err()
        || u32::try_from(mutable_node_words).is_err()
        || u32::try_from(ranges.len()).is_err()
        || u32::try_from(overlay_words).is_err()
    {
        return None;
    }
    let side = limits.max_texture_dimension_2d.clamp(1, 1024);
    let page_capacity = u64::from(side).pow(2);
    let layers = (page_word_count as u64).div_ceil(page_capacity);
    if layers > u64::from(limits.max_texture_array_layers) {
        return None;
    }
    let page_bytes = layers.saturating_mul(page_capacity).saturating_mul(16);
    if page_bytes.saturating_add(overlay_bytes) > memory_budget {
        return None;
    }
    let mut overlay = Vec::with_capacity(overlay_words.max(1));
    overlay.extend_from_slice(&data.acceleration_prefix);
    overlay.extend(range_map.iter().copied());
    for &(start, count) in &ranges {
        overlay.extend_from_slice(&data.blas_words[start * 2..(start + count) * 2]);
    }
    let instance_overlay_base = overlay.len();
    overlay.extend_from_slice(&data.instance_words[..dynamic_instance_words]);
    let pages = ScenePageUpload::new(
        data.blas_words.clone(),
        data.instance_words.clone(),
        &ranges,
        dynamic_instance_words,
        static_blas_words,
        page_word_count,
    );
    Some((
        ScenePageLayout {
            side,
            layers: layers as u32,
            bytes: page_bytes,
            page_config: UVec4::new(1, side, prefix_words as u32, ranges.len() as u32),
            overlay_config: UVec4::new(
                instance_base as u32,
                dynamic_instance_words as u32,
                static_blas_words as u32,
                mutable_node_words as u32,
            ),
            range_map,
            instance_overlay_base,
        },
        pages,
        overlay,
    ))
}

impl PendingGeometry {
    fn new(
        device: &RenderDevice,
        words: Arc<Vec<UVec4>>,
        revision: u64,
        page: Option<GeometryPages>,
        overlay_words: Arc<Vec<UVec4>>,
    ) -> Self {
        let (
            buffer_bytes,
            texture,
            view,
            input_view,
            output_view,
            paged,
            page_side,
            allocated_bytes,
        ) = if let Some(page) = page {
            let texture = device.create_texture(&TextureDescriptor {
                label: Some("rc_geometry_pages"),
                size: Extent3d {
                    width: page.side,
                    height: page.side,
                    depth_or_array_layers: page.layers,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba32Uint,
                usage: TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_DST
                    | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&TextureViewDescriptor {
                dimension: Some(TextureViewDimension::D2Array),
                ..default()
            });
            let input_view = texture.create_view(&TextureViewDescriptor {
                dimension: Some(TextureViewDimension::D2Array),
                base_array_layer: 0,
                array_layer_count: Some(page.base_layers),
                ..default()
            });
            let output_view = (page.output_layers > 0).then(|| {
                texture.create_view(&TextureViewDescriptor {
                    dimension: Some(TextureViewDimension::D2Array),
                    base_array_layer: page.base_layers,
                    array_layer_count: Some(page.output_layers),
                    ..default()
                })
            });
            let overlay_bytes = (overlay_words.len().max(1) as u64).saturating_mul(16);
            (
                overlay_bytes,
                Some(texture),
                Some(view),
                Some(input_view),
                output_view,
                true,
                page.side,
                page.bytes.saturating_add(overlay_bytes),
            )
        } else {
            let bytes = (words.len().max(1) * 16) as u64;
            (bytes, None, None, None, None, false, 0, bytes)
        };
        Self {
            buffer: device.create_buffer(&BufferDescriptor {
                label: Some(if paged {
                    "rc_geometry_placeholder"
                } else {
                    "rc_pending_geometry"
                }),
                size: buffer_bytes,
                // Hardware ray queries build their BLASes from these positions.
                usage: BufferUsages::STORAGE
                    | BufferUsages::COPY_DST
                    | if !paged
                        && device
                            .features()
                            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
                    {
                        BufferUsages::BLAS_INPUT
                    } else {
                        BufferUsages::empty()
                    },
                mapped_at_creation: false,
            }),
            allocated_bytes,
            overlay_words,
            overlay_cursor: 0,
            texture,
            view,
            input_view,
            output_view,
            paged,
            page_side,
            words,
            revision,
            cursor: 0,
        }
    }

    fn upload(&mut self, queue: &RenderQueue, remaining: &mut usize) -> bool {
        if let Some(texture) = &self.texture {
            upload_texture_words(
                queue,
                texture,
                self.page_side,
                &self.words,
                &mut self.cursor,
                remaining,
            );
        } else {
            upload_words(
                queue,
                &self.buffer,
                &self.words,
                &mut self.cursor,
                0,
                remaining,
            );
        }
        if self.paged {
            upload_words(
                queue,
                &self.buffer,
                &self.overlay_words,
                &mut self.overlay_cursor,
                0,
                remaining,
            );
        }
        self.cursor == self.words.len() && self.overlay_cursor == self.overlay_words.len()
    }
}

struct PendingWords {
    buffer: Buffer,
    words: Arc<Vec<UVec4>>,
    cursor: usize,
}

impl PendingWords {
    fn new(device: &RenderDevice, label: &'static str, words: Arc<Vec<UVec4>>) -> Self {
        Self {
            buffer: device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: (words.len().max(1) * 16) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            words,
            cursor: 0,
        }
    }

    fn upload(&mut self, queue: &RenderQueue, remaining: &mut usize) -> bool {
        upload_words(
            queue,
            &self.buffer,
            &self.words,
            &mut self.cursor,
            0,
            remaining,
        );
        self.cursor == self.words.len()
    }
}

fn order_words(order: &[u32]) -> Arc<Vec<UVec4>> {
    Arc::new(
        order
            .chunks(4)
            .map(|chunk| {
                let mut word = [0; 4];
                word[..chunk.len()].copy_from_slice(chunk);
                UVec4::from_array(word)
            })
            .collect(),
    )
}

struct PendingAcceleration {
    buffer: Buffer,
    texture: Option<Texture>,
    view: Option<TextureView>,
    page_words: ScenePageUpload,
    paged: bool,
    allocated_bytes: u64,
    words: Vec<UVec4>,
    blas_words: Arc<Vec<UVec4>>,
    material_words: Vec<UVec4>,
    topology_revision: u64,
    prefix_words: usize,
    dynamic_baseline: DynamicBaseline,
    dynamic_revision: u64,
    cursor: usize,
    page_config: UVec4,
    overlay_config: UVec4,
    range_map: Vec<UVec4>,
    instance_overlay_base: usize,
}

impl PendingAcceleration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        device: &RenderDevice,
        words: Vec<UVec4>,
        page_words: ScenePageUpload,
        page_layout: Option<ScenePageLayout>,
        blas_words: Arc<Vec<UVec4>>,
        material_words: Vec<UVec4>,
        topology_revision: u64,
        prefix_words: usize,
        full_instance_base: usize,
        dynamic_instance_words: usize,
        dynamic_revision: u64,
    ) -> Self {
        let paged = page_layout.is_some();
        let (
            texture,
            view,
            allocated_bytes,
            page_config,
            overlay_config,
            range_map,
            instance_overlay_base,
        ) = if let Some(layout) = page_layout {
            let texture = device.create_texture(&TextureDescriptor {
                label: Some("rc_scene_pages"),
                size: Extent3d {
                    width: layout.side,
                    height: layout.side,
                    depth_or_array_layers: layout.layers,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba32Uint,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&TextureViewDescriptor {
                dimension: Some(TextureViewDimension::D2Array),
                ..default()
            });
            (
                Some(texture),
                Some(view),
                layout.bytes.saturating_add(words.len().max(1) as u64 * 16),
                layout.page_config,
                layout.overlay_config,
                layout.range_map,
                layout.instance_overlay_base,
            )
        } else {
            (
                None,
                None,
                words.len().max(1) as u64 * 16,
                UVec4::ZERO,
                UVec4::ZERO,
                Vec::new(),
                full_instance_base,
            )
        };
        let dynamic_baseline = DynamicBaseline::from_words(
            &words,
            prefix_words,
            instance_overlay_base,
            dynamic_instance_words,
        )
        .expect("dynamic acceleration ranges must fit the overlay buffer");
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some(if paged {
                "rc_pending_scene_overlay"
            } else {
                "rc_pending_acceleration"
            }),
            size: words.len().max(1) as u64 * 16,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            texture,
            view,
            page_words,
            paged,
            allocated_bytes,
            words,
            blas_words,
            material_words,
            topology_revision,
            prefix_words,
            dynamic_baseline,
            dynamic_revision,
            cursor: 0,
            page_config,
            overlay_config,
            range_map,
            instance_overlay_base,
        }
    }

    fn upload(&mut self, queue: &RenderQueue, remaining: &mut usize) -> bool {
        upload_words(
            queue,
            &self.buffer,
            &self.words,
            &mut self.cursor,
            0,
            remaining,
        );
        if let Some(texture) = &self.texture {
            upload_scene_page_words(
                queue,
                texture,
                self.page_config.y,
                &mut self.page_words,
                remaining,
            );
        }
        self.cursor == self.words.len() && scene_page_upload_complete(self.paged, &self.page_words)
    }
}

#[derive(Clone, Copy, Default, ShaderType)]
struct SceneDispatch {
    dispatch: UVec4,
    scene_pages: UVec4,
    scene_overlay: UVec4,
}

struct PendingDynamic {
    revision: u64,
    prefix: Arc<Vec<UVec4>>,
    instances: Arc<Vec<UVec4>>,
    instance_words: usize,
    instance_overlay_base: usize,
}

#[derive(Clone, Default)]
struct DynamicBaseline {
    prefix: Vec<UVec4>,
    instances: Vec<UVec4>,
    instance_overlay_base: usize,
}

struct PendingMaterialTail {
    applied_words: Vec<UVec4>,
    target_words: Vec<UVec4>,
}

#[derive(Clone, Copy)]
enum MaterialTailDestination<'a> {
    Buffer {
        buffer: &'a Buffer,
        word_base: usize,
    },
    Texture {
        texture: &'a Texture,
        side: u32,
        word_base: usize,
    },
}

fn changed_word_ranges(applied: &[UVec4], target: &[UVec4]) -> Option<Vec<std::ops::Range<usize>>> {
    if applied.len() != target.len() {
        return None;
    }
    let mut ranges = Vec::new();
    let mut cursor = 0;
    while cursor < applied.len() {
        if applied[cursor] == target[cursor] {
            cursor += 1;
            continue;
        }
        let start = cursor;
        cursor += 1;
        while cursor < applied.len() && applied[cursor] != target[cursor] {
            cursor += 1;
        }
        ranges.push(start..cursor);
    }
    Some(ranges)
}

fn material_tail_patch_layout_matches(
    previous_tail: &[UVec4],
    previous_instances: &[UVec4],
    current_tail: &[UVec4],
    current_instances: &[UVec4],
    instance_start: usize,
    instance_count: usize,
    instance_base: usize,
) -> bool {
    if previous_tail.len() != current_tail.len()
        || previous_instances.len() != instance_start
        || current_instances.len() != instance_start
        || instance_start != instance_count.saturating_mul(INSTANCE_WORDS_PER_RECORD)
    {
        return false;
    }
    let Some(first_material_word) = instance_base.checked_add(instance_start) else {
        return false;
    };
    let Ok(first_material) = u32::try_from(first_material_word) else {
        return false;
    };
    let Ok(tail_words) = u32::try_from(current_tail.len()) else {
        return false;
    };
    let Some(material_end) = first_material.checked_add(tail_words) else {
        return false;
    };
    let material_stride = crate::material::MATERIAL_WORDS as u32;
    for index in 0..instance_count {
        let Some(address_index) = index
            .checked_mul(INSTANCE_WORDS_PER_RECORD)
            .and_then(|offset| offset.checked_add(7))
        else {
            return false;
        };
        let (Some(previous), Some(current)) = (
            previous_instances.get(address_index),
            current_instances.get(address_index),
        ) else {
            return false;
        };
        let address = current.w;
        let Some(address_end) = address.checked_add(material_stride) else {
            return false;
        };
        if previous.w != address
            || address < first_material
            || address_end > material_end
            || !(address - first_material).is_multiple_of(material_stride)
        {
            return false;
        }
    }
    true
}

fn material_tail_layout_reusable(
    previous_tail: &[UVec4],
    previous_instances: Option<&[UVec4]>,
    current_tail: &[UVec4],
    current_instances: &[UVec4],
    instance_start: usize,
    instance_count: usize,
    instance_base: usize,
) -> bool {
    previous_tail == current_tail
        || previous_instances.is_some_and(|previous_instances| {
            material_tail_patch_layout_matches(
                previous_tail,
                previous_instances,
                current_tail,
                current_instances,
                instance_start,
                instance_count,
                instance_base,
            )
        })
}

impl PendingMaterialTail {
    fn new(applied_words: Vec<UVec4>, target_words: &[UVec4]) -> Option<Self> {
        (applied_words.len() == target_words.len()).then(|| Self {
            applied_words,
            target_words: target_words.to_vec(),
        })
    }

    fn upload(
        &mut self,
        queue: &RenderQueue,
        destination: MaterialTailDestination<'_>,
        target_words: &[UVec4],
        remaining: &mut usize,
    ) -> bool {
        if self.applied_words.len() != target_words.len() {
            return false;
        }
        if self.target_words != target_words {
            self.target_words.clone_from_slice(target_words);
        }
        let Some(ranges) = changed_word_ranges(&self.applied_words, &self.target_words) else {
            return false;
        };
        for range in ranges {
            let mut start = range.start;
            while start < range.end && *remaining >= 16 {
                let count = (range.end - start).min(*remaining / 16).min(65_536);
                let (destination_start, count) = match destination {
                    MaterialTailDestination::Buffer { buffer, word_base } => {
                        let Some(word_start) = word_base.checked_add(start) else {
                            return false;
                        };
                        let Some(offset) = word_start
                            .checked_mul(16)
                            .and_then(|offset| u64::try_from(offset).ok())
                        else {
                            return false;
                        };
                        queue.write_buffer(
                            buffer,
                            offset,
                            &words_bytes(&self.target_words[start..start + count]),
                        );
                        (start, count)
                    }
                    MaterialTailDestination::Texture {
                        texture,
                        side,
                        word_base,
                    } => {
                        let side = side as usize;
                        let Some(side_words) = side.checked_mul(side).filter(|_| side > 0) else {
                            return false;
                        };
                        let Some(page_word) = word_base.checked_add(start) else {
                            return false;
                        };
                        let within_layer = page_word % side_words;
                        let row = within_layer / side;
                        let column = within_layer % side;
                        let count = count.min(side - column);
                        queue.write_texture(
                            TexelCopyTextureInfo {
                                texture,
                                mip_level: 0,
                                origin: Origin3d {
                                    x: column as u32,
                                    y: row as u32,
                                    z: (page_word / side_words) as u32,
                                },
                                aspect: TextureAspect::All,
                            },
                            &words_bytes(&self.target_words[start..start + count]),
                            TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: Some((count * 16) as u32),
                                rows_per_image: Some(1),
                            },
                            Extent3d {
                                width: count as u32,
                                height: 1,
                                depth_or_array_layers: 1,
                            },
                        );
                        (start, count)
                    }
                };
                let end = destination_start + count;
                self.applied_words[destination_start..end]
                    .copy_from_slice(&self.target_words[destination_start..end]);
                *remaining -= count * 16;
                start = end;
            }
            if start < range.end {
                return false;
            }
        }
        self.applied_words == self.target_words
    }
}

impl DynamicBaseline {
    fn from_words(
        words: &[UVec4],
        prefix_words: usize,
        instance_overlay_base: usize,
        instance_words: usize,
    ) -> Option<Self> {
        let prefix = words.get(..prefix_words)?.to_vec();
        let instance_end = instance_overlay_base.checked_add(instance_words)?;
        let instances = words.get(instance_overlay_base..instance_end)?.to_vec();
        Some(Self {
            prefix,
            instances,
            instance_overlay_base,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DynamicUploadSource {
    Prefix,
    Instances,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DynamicUploadRange {
    source: DynamicUploadSource,
    source_start: usize,
    buffer_start: usize,
    words: usize,
}

fn append_changed_dynamic_ranges(
    ranges: &mut Vec<DynamicUploadRange>,
    source: DynamicUploadSource,
    desired: &[UVec4],
    baseline: &[UVec4],
    buffer_base: usize,
) {
    assert_eq!(desired.len(), baseline.len());
    let mut cursor = 0;
    while cursor < desired.len() {
        if desired[cursor] == baseline[cursor] {
            cursor += 1;
            continue;
        }
        let start = cursor;
        cursor += 1;
        while cursor < desired.len() && desired[cursor] != baseline[cursor] {
            cursor += 1;
        }
        ranges.push(DynamicUploadRange {
            source,
            source_start: start,
            buffer_start: buffer_base + start,
            words: cursor - start,
        });
    }
}

fn dynamic_upload_ranges(
    baseline: &DynamicBaseline,
    prefix: &[UVec4],
    instances: &[UVec4],
    instance_words: usize,
) -> Vec<DynamicUploadRange> {
    assert!(instance_words <= instances.len());
    let mut ranges = Vec::new();
    append_changed_dynamic_ranges(
        &mut ranges,
        DynamicUploadSource::Prefix,
        prefix,
        &baseline.prefix,
        0,
    );
    append_changed_dynamic_ranges(
        &mut ranges,
        DynamicUploadSource::Instances,
        &instances[..instance_words],
        &baseline.instances,
        baseline.instance_overlay_base,
    );
    ranges
}

fn budget_dynamic_upload_ranges(
    ranges: &[DynamicUploadRange],
    budget_bytes: usize,
    max_words_per_write: usize,
) -> Vec<DynamicUploadRange> {
    let mut remaining_words = budget_bytes / 16;
    let max_words_per_write = max_words_per_write.max(1);
    let mut uploads = Vec::new();
    for range in ranges {
        let mut consumed = 0;
        while consumed < range.words && remaining_words > 0 {
            let words = (range.words - consumed)
                .min(remaining_words)
                .min(max_words_per_write);
            uploads.push(DynamicUploadRange {
                source: range.source,
                source_start: range.source_start + consumed,
                buffer_start: range.buffer_start + consumed,
                words,
            });
            consumed += words;
            remaining_words -= words;
        }
        if remaining_words == 0 {
            break;
        }
    }
    uploads
}

impl PendingDynamic {
    fn retarget(
        &mut self,
        revision: u64,
        prefix: Arc<Vec<UVec4>>,
        instances: Arc<Vec<UVec4>>,
        instance_words: usize,
        instance_overlay_base: usize,
    ) {
        self.revision = revision;
        self.prefix = prefix;
        self.instances = instances;
        self.instance_words = instance_words;
        self.instance_overlay_base = instance_overlay_base;
    }

    fn matches_layout(&self, baseline: &DynamicBaseline) -> bool {
        self.prefix.len() == baseline.prefix.len()
            && self.instance_words == baseline.instances.len()
            && self.instance_overlay_base == baseline.instance_overlay_base
    }

    fn upload(
        &mut self,
        queue: &RenderQueue,
        buffer: &Buffer,
        baseline: &mut DynamicBaseline,
        remaining: &mut usize,
    ) -> bool {
        assert_eq!(self.instance_overlay_base, baseline.instance_overlay_base);
        let ranges =
            dynamic_upload_ranges(baseline, &self.prefix, &self.instances, self.instance_words);
        let uploads = budget_dynamic_upload_ranges(&ranges, *remaining, 65536);
        for upload in uploads {
            let end = upload.source_start + upload.words;
            let source = match upload.source {
                DynamicUploadSource::Prefix => &self.prefix[upload.source_start..end],
                DynamicUploadSource::Instances => &self.instances[upload.source_start..end],
            };
            queue.write_buffer(
                buffer,
                (upload.buffer_start * 16) as u64,
                &words_bytes(source),
            );
            let target = match upload.source {
                DynamicUploadSource::Prefix => &mut baseline.prefix,
                DynamicUploadSource::Instances => &mut baseline.instances,
            };
            target[upload.source_start..end].copy_from_slice(source);
            *remaining -= upload.words * 16;
        }
        dynamic_upload_ranges(baseline, &self.prefix, &self.instances, self.instance_words)
            .is_empty()
    }
}

struct PendingAtlas {
    texture: Texture,
    view: TextureView,
    atlas: Arc<crate::material::TextureAtlas>,
    layer: usize,
    cursor: usize,
}

#[derive(Clone)]
struct PendingLights {
    buffer: Buffer,
    source: Option<(bevy::asset::AssetId<Image>, u64)>,
    layers: RenderLayers,
    prefix: Vec<GiLight>,
    prefix_cursor: usize,
    replacement: bool,
    scene_revision: u64,
    geometry_revision: u64,
    light_revision: u64,
}

#[derive(Resource, Clone, bevy::render::extract_resource::ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
pub(crate) struct SceneBudget {
    pub live_bytes: u64,
    pub view_bytes: u64,
    pub upload_bytes: usize,
}

#[derive(Resource, Default)]
struct MemoryHighWater(u64);

fn upload_range(cursor: usize, length: usize, budget: usize) -> std::ops::Range<usize> {
    cursor..cursor.saturating_add(budget / 16).min(length)
}

fn upload_words(
    queue: &RenderQueue,
    buffer: &Buffer,
    words: &[UVec4],
    cursor: &mut usize,
    offset_words: usize,
    remaining: &mut usize,
) {
    let range = upload_range(*cursor, words.len(), *remaining);
    for (index, chunk) in words[range.clone()].chunks(65536).enumerate() {
        queue.write_buffer(
            buffer,
            ((offset_words + range.start + index * 65536) * 16) as u64,
            &words_bytes(chunk),
        );
    }
    *remaining -= range.len() * 16;
    *cursor = range.end;
}

fn upload_texture_pixels(
    queue: &RenderQueue,
    texture: &Texture,
    side: u32,
    base_layer: u32,
    source: &[u8],
    cursor: &mut usize,
    remaining: &mut usize,
) {
    let row_bytes = side as usize * 16;
    let layer_bytes = row_bytes * side as usize;
    while *cursor < source.len() && *remaining >= 16 {
        let within_layer = *cursor % layer_bytes;
        let row = within_layer / row_bytes;
        let column = (within_layer % row_bytes) / 16;
        let (width, rows) = if column == 0 && *remaining >= row_bytes {
            (
                side,
                ((*remaining / row_bytes).min(side as usize - row)) as u32,
            )
        } else {
            (((*remaining / 16).min(side as usize - column)) as u32, 1)
        };
        let bytes = width as usize * rows as usize * 16;
        queue.write_texture(
            TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: Origin3d {
                    x: column as u32,
                    y: row as u32,
                    z: base_layer + (*cursor / layer_bytes) as u32,
                },
                aspect: TextureAspect::All,
            },
            &source[*cursor..*cursor + bytes],
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 16),
                rows_per_image: Some(rows),
            },
            Extent3d {
                width,
                height: rows,
                depth_or_array_layers: 1,
            },
        );
        *cursor += bytes;
        *remaining -= bytes;
    }
}

fn upload_texture_words(
    queue: &RenderQueue,
    texture: &Texture,
    side: u32,
    words: &[UVec4],
    cursor: &mut usize,
    remaining: &mut usize,
) {
    let side_words = side as usize * side as usize;
    while *cursor < words.len() && *remaining >= 16 {
        let within_layer = *cursor % side_words;
        let row = within_layer / side as usize;
        let column = within_layer % side as usize;
        let count = (words.len() - *cursor).min(*remaining / 16).min(65536);
        let (width, rows) = if column == 0 && count >= side as usize {
            let rows = (count / side as usize).min(side as usize - row);
            (side as usize, rows)
        } else {
            (count.min(side as usize - column), 1)
        };
        let word_count = width * rows;
        let end = *cursor + word_count;
        queue.write_texture(
            TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: Origin3d {
                    x: column as u32,
                    y: row as u32,
                    z: (*cursor / side_words) as u32,
                },
                aspect: TextureAspect::All,
            },
            &words_bytes(&words[*cursor..end]),
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some((width * 16) as u32),
                rows_per_image: Some(rows as u32),
            },
            Extent3d {
                width: width as u32,
                height: rows as u32,
                depth_or_array_layers: 1,
            },
        );
        *cursor = end;
        *remaining -= word_count * 16;
    }
}

fn upload_scene_page_words(
    queue: &RenderQueue,
    texture: &Texture,
    side: u32,
    source: &mut ScenePageUpload,
    remaining: &mut usize,
) {
    let side_words = side as usize * side as usize;
    while source.cursor < source.word_count && *remaining >= 16 {
        let within_layer = source.cursor % side_words;
        let row = within_layer / side as usize;
        let column = within_layer % side as usize;
        let count = (source.word_count - source.cursor)
            .min(*remaining / 16)
            .min(65536);
        let (width, rows) = if column == 0 && count >= side as usize {
            let rows = (count / side as usize).min(side as usize - row);
            (side as usize, rows)
        } else {
            (count.min(side as usize - column), 1)
        };
        let word_count = width * rows;
        let mut bytes = Vec::with_capacity(word_count * 16);
        let written_words = source.append_next_bytes(&mut bytes, word_count);
        debug_assert_eq!(written_words, word_count);
        queue.write_texture(
            TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: Origin3d {
                    x: column as u32,
                    y: row as u32,
                    z: ((source.cursor - written_words) / side_words) as u32,
                },
                aspect: TextureAspect::All,
            },
            &bytes,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some((width * 16) as u32),
                rows_per_image: Some(rows as u32),
            },
            Extent3d {
                width: width as u32,
                height: rows as u32,
                depth_or_array_layers: 1,
            },
        );
        *remaining -= word_count * 16;
    }
}

pub(crate) fn deformation_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "rc_deformation",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<SceneDispatch>(true),
                storage_buffer::<Vec<UVec4>>(false),
                storage_buffer::<Vec<UVec4>>(false),
                storage_buffer_read_only::<Vec<UVec4>>(false),
                storage_buffer_read_only::<Vec<UVec4>>(false),
                storage_buffer_read_only::<Vec<u32>>(false),
                texture_2d_array(TextureSampleType::Uint),
                texture_2d_array(TextureSampleType::Uint),
                texture_storage_2d_array(
                    TextureFormat::Rgba32Uint,
                    StorageTextureAccess::ReadWrite,
                ),
            ),
        ),
    )
}

fn allocate_material_atlas(
    device: &RenderDevice,
    atlas: &crate::material::TextureAtlas,
) -> (Texture, TextureView) {
    let side = atlas.side.max(1);
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("gi_material_atlas"),
        size: Extent3d {
            width: side,
            height: side,
            depth_or_array_layers: (atlas.pages.len() as u32).max(1),
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    (texture, view)
}

#[cfg(test)]
pub(crate) fn upload_material_atlas(
    device: &RenderDevice,
    queue: &RenderQueue,
    atlas: &crate::material::TextureAtlas,
) -> (Texture, TextureView) {
    let (texture, view) = allocate_material_atlas(device, atlas);
    let side = atlas.side.max(1);
    for (layer, page) in atlas.pages.iter().enumerate() {
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d {
                    x: 0,
                    y: 0,
                    z: layer as u32,
                },
                aspect: TextureAspect::All,
            },
            page,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(side * 16),
                rows_per_image: Some(side),
            },
            Extent3d {
                width: side,
                height: side,
                depth_or_array_layers: 1,
            },
        );
    }
    (texture, view)
}

/// Display irradiance, G-buffer, display receiver metadata and raster identity
/// in; anchor position and geometric normal out. Bindings match composite.slang.
fn composite_anchor_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "split_rc_composite_anchors",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                (0, uniform_buffer::<Parameters>(false)),
                (2, texture_2d(TextureSampleType::Float { filterable: false })),
                (5, texture_2d(TextureSampleType::Uint)),
                (6, texture_2d(TextureSampleType::Float { filterable: false })),
                (7, texture_2d(TextureSampleType::Uint)),
                (
                    14,
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                ),
                (
                    15,
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                ),
            ),
        ),
    )
}

pub(crate) fn refit_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "rc_tlas_refit",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<SceneDispatch>(true),
                storage_buffer::<Vec<UVec4>>(false),
                storage_buffer_read_only::<Vec<u32>>(false),
                texture_2d_array(TextureSampleType::Uint),
            ),
        ),
    )
}

pub(crate) fn refit_dispatches(nodes: &[BvhNode]) -> (Vec<u32>, Vec<UVec4>) {
    let mut levels: Vec<Vec<u32>> = Vec::new();
    let mut stack = if nodes.is_empty() {
        vec![]
    } else {
        vec![(0u32, 0usize)]
    };
    while let Some((index, depth)) = stack.pop() {
        if levels.len() <= depth {
            levels.resize_with(depth + 1, Vec::new);
        }
        levels[depth].push(index);
        let node = nodes[index as usize];
        if node.count == 0 {
            stack.extend([(node.first, depth + 1), (node.first + 1, depth + 1)]);
        }
    }
    let mut order = Vec::new();
    let mut dispatches = Vec::new();
    for level in levels.iter().rev() {
        dispatches.push(UVec4::new(order.len() as u32, level.len() as u32, 0, 0));
        order.extend(level);
    }
    (order, dispatches)
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn refit_tlas(
    device: &RenderDevice,
    queue: &RenderQueue,
    pipeline: &ComputePipeline,
    nodes: &StorageBuffer<Vec<UVec4>>,
    order: &StorageBuffer<Vec<u32>>,
    offsets: &[(u32, u32)],
    diagnostics: Option<&DiagnosticsRecorder>,
) {
    if let Some(nodes) = nodes.buffer() {
        let mut scene_params = DynamicUniformBuffer::new_with_alignment(
            device.limits().min_uniform_buffer_offset_alignment as u64,
        );
        let scene_offsets: Vec<_> = offsets
            .iter()
            .map(|&(offset, count)| {
                (
                    scene_params.push(&SceneDispatch {
                        dispatch: UVec4::new(offset, count, 0, 0),
                        scene_pages: UVec4::ZERO,
                        scene_overlay: UVec4::ZERO,
                    }),
                    count,
                )
            })
            .collect();
        scene_params.write_buffer(device, queue);
        let page_placeholder = device.create_texture(&TextureDescriptor {
            label: Some("rc_test_scene_page_placeholder"),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Uint,
            usage: TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let page_view = page_placeholder.create_view(&TextureViewDescriptor {
            dimension: Some(TextureViewDimension::D2Array),
            ..default()
        });
        refit_tlas_buffer(
            device,
            queue,
            pipeline,
            &scene_params,
            nodes,
            order.buffer().expect("test refit order uploaded"),
            &scene_offsets,
            diagnostics,
            &page_view,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn refit_tlas_buffer(
    device: &RenderDevice,
    queue: &RenderQueue,
    pipeline: &ComputePipeline,
    params: &DynamicUniformBuffer<SceneDispatch>,
    nodes: &Buffer,
    order: &Buffer,
    offsets: &[(u32, u32)],
    diagnostics: Option<&DiagnosticsRecorder>,
    scene_page_view: &TextureView,
) {
    if offsets.is_empty() {
        return;
    }
    let (Some(params), Some(order)) = (params.binding(), Some(order.as_entire_binding())) else {
        return;
    };
    let layout = pipeline.get_bind_group_layout(0).into();
    let group = device.create_bind_group(
        "rc_tlas_refit",
        &layout,
        &BindGroupEntries::sequential((params, nodes.as_entire_binding(), order, scene_page_view)),
    );
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
    let span = diagnostics.time_span(&mut encoder, "rc_tlas_refit");
    for &(offset, count) in offsets {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("rc_tlas_refit_level"),
            ..default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[offset]);
        pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
    }
    span.end(&mut encoder);
    queue.submit([encoder.finish()]);
}

fn append_words_bytes(bytes: &mut Vec<u8>, words: &[UVec4]) {
    for word in words {
        for value in word.to_array() {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
}

fn words_bytes(words: &[UVec4]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 16);
    append_words_bytes(&mut bytes, words);
    bytes
}

fn lights_bytes(lights: &[GiLight]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(lights.len() * 96);
    for light in lights {
        for field in [
            light.position,
            light.color,
            light.direction,
            light.edge1,
            light.edge2,
        ] {
            for value in field.to_array() {
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        for value in light.source.to_array() {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

#[cfg(test)]
pub(crate) fn upload_geometry(
    device: &RenderDevice,
    queue: &RenderQueue,
    words: &[UVec4],
) -> Buffer {
    let words = if words.is_empty() {
        &[UVec4::ZERO]
    } else {
        words
    };
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("rc_geometry"),
        size: words.len() as u64 * 16,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // Serialize at most 1 MiB at once; StorageBuffer<Vec<_>> kept both a cloned
    // typed scene and a full serialized copy alive for these immutable words.
    for (index, chunk) in words.chunks(65536).enumerate() {
        queue.write_buffer(&buffer, index as u64 * 65536 * 16, &words_bytes(chunk));
    }
    buffer
}

#[allow(clippy::too_many_arguments)]
fn refit_scene(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    cache: Res<PipelineCache>,
    pipelines: Res<Pipelines>,
    scene: Res<GiScene>,
    gpu_scene: Option<ResMut<GpuScene>>,
    diagnostics: Option<Res<DiagnosticsRecorder>>,
) {
    let Some(mut gpu_scene) = gpu_scene else {
        return;
    };
    if gpu_scene.pending_deformation {
        let (Some(deform), Some(refit)) = (
            cache.get_compute_pipeline(pipelines.deform[0]),
            cache.get_compute_pipeline(pipelines.deform[1]),
        ) else {
            return;
        };
        let (
            Some(params),
            Some(geometry),
            Some(nodes),
            Some(streams),
            Some(poses),
            Some(order),
            Some(scene_pages),
            Some(geometry_pages_input),
            Some(geometry_pages_full),
            Some(geometry_output),
            Some(geometry_placeholder),
        ) = (
            gpu_scene.deform_params.binding(),
            gpu_scene.geometry.as_ref(),
            gpu_scene.nodes.as_ref(),
            gpu_scene.deformation.as_ref(),
            gpu_scene.poses.as_ref(),
            gpu_scene.deform_order.as_ref(),
            gpu_scene.scene_page_view(),
            if gpu_scene.geometry_paged {
                gpu_scene.geometry_pages_input_view.as_ref()
            } else {
                gpu_scene.geometry_page_placeholder_view.as_ref()
            },
            if gpu_scene.geometry_paged {
                gpu_scene.geometry_pages_view.as_ref()
            } else {
                gpu_scene.geometry_page_placeholder_view.as_ref()
            },
            if gpu_scene.geometry_paged {
                gpu_scene.geometry_pages_output_view.as_ref()
            } else {
                gpu_scene.geometry_output_placeholder_view.as_ref()
            },
            gpu_scene.geometry_output_placeholder_view.as_ref(),
        )
        else {
            return;
        };
        let deform_group = device.create_bind_group(
            "rc_deform_vertices",
            &deform.get_bind_group_layout(0).into(),
            &BindGroupEntries::sequential((
                params.clone(),
                geometry.as_entire_binding(),
                nodes.as_entire_binding(),
                streams.as_entire_binding(),
                poses.as_entire_binding(),
                order.as_entire_binding(),
                scene_pages,
                geometry_pages_input,
                geometry_output,
            )),
        );
        let refit_group = device.create_bind_group(
            "rc_refit_blas",
            &refit.get_bind_group_layout(0).into(),
            &BindGroupEntries::sequential((
                params,
                geometry.as_entire_binding(),
                nodes.as_entire_binding(),
                streams.as_entire_binding(),
                poses.as_entire_binding(),
                order.as_entire_binding(),
                scene_pages,
                geometry_pages_full,
                geometry_placeholder,
            )),
        );
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
        let recorder = diagnostics.as_deref();
        let span = recorder.time_span(&mut encoder, "rc_deformation");
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("rc_morph_skin"),
                ..default()
            });
            pass.set_pipeline(deform);
            pass.set_bind_group(0, &deform_group, &[gpu_scene.deform_vertex_offset]);
            pass.dispatch_workgroups(scene.data.deformation.vertices.div_ceil(64), 1, 1);
        }
        for &(offset, count) in &gpu_scene.deform_offsets {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("rc_blas_refit"),
                ..default()
            });
            pass.set_pipeline(refit);
            pass.set_bind_group(0, &refit_group, &[offset]);
            pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
        }
        span.end(&mut encoder);
        queue.submit([encoder.finish()]);
        gpu_scene.pending_deformation = false;
    }
    if !gpu_scene.pending_refit {
        return;
    }
    if let Some(pipeline) = cache.get_compute_pipeline(pipelines.refit) {
        refit_tlas_buffer(
            &device,
            &queue,
            pipeline,
            &gpu_scene.refit_params,
            gpu_scene.nodes.as_ref().expect("published acceleration"),
            gpu_scene
                .refit_nodes
                .as_ref()
                .expect("published refit order"),
            &gpu_scene.refit_offsets,
            diagnostics.as_deref(),
            gpu_scene
                .scene_page_view()
                .expect("published scene page placeholder"),
        );
        debug!(
            nodes = scene.data.nodes.len(),
            levels = gpu_scene.refit_offsets.len(),
            "GI TLAS refitted on GPU"
        );
        gpu_scene.pending_refit = false;
    }
}

#[derive(Clone)]
struct CacheBuffers {
    heads: Buffer,
}

struct SecondaryCache {
    caches: [CacheBuffers; 2],
    entries: Buffer,
    history_entries: Buffer,
    hit_textures: [Texture; 2],
    hit_views: [TextureView; 2],
    uniforms: [UniformBuffer<Parameters>; LEVELS],
    clear_uniforms: [UniformBuffer<Parameters>; 2],
    valid: AtomicBool,
}

const INSTANCE_WORDS_PER_RECORD: usize = 9;
const INSTANCE_FLAG_ACTIVE_DEFORMATION: u32 = 1 << 6;

#[derive(Clone, Default)]
struct SceneHistorySnapshot {
    valid: bool,
    revision: u64,
    geometry_revision: u64,
    mesh_revision: u64,
    topology_revision: u64,
    instance_count: usize,
    instance_material_range: Option<(u32, u32)>,
    nodes: Arc<Vec<BvhNode>>,
    instances: Arc<Vec<crate::scene::SceneInstance>>,
    instance_words: Arc<Vec<UVec4>>,
    deformation_poses: Arc<Vec<UVec4>>,
    textures: Arc<crate::material::TextureAtlas>,
    lights: Arc<Vec<GiLight>>,
    light_layers: Arc<Vec<RenderLayers>>,
    emissive_lights: Arc<Vec<GiLight>>,
    view_layers: RenderLayers,
    ambient_environment: LinearRgba,
    environment: Option<crate::RadianceCascadesEnvironment>,
    environment_revision: u64,
}

impl SceneHistorySnapshot {
    fn capture(
        scene: &GiScene,
        view_layers: &RenderLayers,
        ambient_environment: LinearRgba,
        environment: Option<&crate::RadianceCascadesEnvironment>,
        environment_revision: u64,
    ) -> Self {
        let data = &scene.data;
        Self {
            valid: true,
            revision: scene.revision,
            geometry_revision: scene.geometry_revision,
            mesh_revision: scene.mesh_revision,
            topology_revision: scene.topology_revision,
            instance_count: data.instances.len(),
            instance_material_range: data.acceleration_prefix.first().and_then(|prefix| {
                let words_len = u32::try_from(data.instance_words.len()).ok()?;
                (usize::try_from(prefix.z).ok()? == data.instances.len()
                    && usize::try_from(prefix.w).ok()? == INSTANCE_WORDS_PER_RECORD
                    && prefix.y <= prefix.x
                    && prefix.x - prefix.y <= words_len)
                    .then_some((prefix.y, prefix.x))
            }),
            nodes: data.nodes.clone(),
            instances: data.instances.clone(),
            instance_words: data.instance_words.clone(),
            deformation_poses: data.deformation.poses.clone(),
            textures: data.textures.clone(),
            lights: data.lights.clone(),
            light_layers: data.light_layers.clone(),
            emissive_lights: data.emissive_lights.clone(),
            view_layers: view_layers.clone(),
            ambient_environment,
            environment: environment.cloned(),
            environment_revision,
        }
    }

    fn instance_record(&self, index: usize) -> Option<&[UVec4]> {
        let start = index.checked_mul(INSTANCE_WORDS_PER_RECORD)?;
        self.instance_words
            .get(start..start + INSTANCE_WORDS_PER_RECORD)
    }
}

fn material_emission_word_offsets(
    snapshot: &SceneHistorySnapshot,
    instance_tail: usize,
) -> Option<Vec<usize>> {
    let (base, material_end) = snapshot.instance_material_range?;
    let material_words = u32::try_from(crate::material::MATERIAL_WORDS).ok()?;
    let first_material = base.checked_add(u32::try_from(instance_tail).ok()?)?;
    let material_end_local = usize::try_from(material_end.checked_sub(base)?).ok()?;
    if material_end_local > snapshot.instance_words.len() {
        return None;
    }
    let mut starts = Vec::with_capacity(snapshot.instance_count);
    for index in 0..snapshot.instance_count {
        let address = snapshot.instance_record(index)?[7].w;
        let record_end = address.checked_add(material_words)?;
        if address < first_material || record_end > material_end {
            return None;
        }
        let local_address = usize::try_from(address.checked_sub(base)?).ok()?;
        let material_end = local_address.checked_add(crate::material::MATERIAL_WORDS)?;
        if local_address < instance_tail || material_end > material_end_local {
            return None;
        }
        starts.push(local_address);
    }
    starts.sort_unstable();
    starts.dedup();
    let first = *starts.first()?;
    if first.checked_add(starts.len().checked_mul(crate::material::MATERIAL_WORDS)?)?
        != material_end_local
        || starts
            .windows(2)
            .any(|pair| pair[0].checked_add(crate::material::MATERIAL_WORDS) != Some(pair[1]))
    {
        return None;
    }
    Some(starts.into_iter().map(|start| start + 1).collect())
}

fn instance_tail_matches_except_emission(
    previous: &SceneHistorySnapshot,
    current: &SceneHistorySnapshot,
    instance_tail: usize,
) -> bool {
    let (Some(previous_tail), Some(current_tail)) = (
        previous.instance_words.get(instance_tail..),
        current.instance_words.get(instance_tail..),
    ) else {
        return false;
    };
    if previous_tail.len() != current_tail.len() {
        return false;
    }
    if previous_tail == current_tail {
        return true;
    }
    if previous.instance_material_range != current.instance_material_range {
        return false;
    }
    let (Some(previous_emission_words), Some(current_emission_words)) = (
        material_emission_word_offsets(previous, instance_tail),
        material_emission_word_offsets(current, instance_tail),
    ) else {
        return false;
    };
    if previous_emission_words != current_emission_words {
        return false;
    }
    previous_tail
        .iter()
        .zip(current_tail)
        .enumerate()
        .all(|(offset, (previous, current))| {
            previous == current
                || (previous_emission_words
                    .binary_search(&(instance_tail + offset))
                    .is_ok()
                    && previous.w == current.w)
        })
}

fn scene_changed_since_history(
    previous: &SceneHistorySnapshot,
    current: &SceneHistorySnapshot,
) -> bool {
    previous.valid
        && (previous.revision != current.revision
            || previous.geometry_revision != current.geometry_revision
            || previous.mesh_revision != current.mesh_revision
            || previous.topology_revision != current.topology_revision)
}

fn rigid_instance_transform(previous: &[UVec4], current: &[UVec4]) -> bool {
    let world_matrix = |record: &[UVec4]| {
        let row = |index: usize| Vec4::from_array(record[index].to_array().map(f32::from_bits));
        Mat4::from_cols(row(3), row(4), row(5), Vec4::W).transpose()
    };
    let old = world_matrix(previous);
    let new = world_matrix(current);
    if !old.is_finite() || !new.is_finite() {
        return false;
    }
    let delta = new * old.inverse();
    if !delta.is_finite() {
        return false;
    }
    let axes = [
        delta.x_axis.truncate(),
        delta.y_axis.truncate(),
        delta.z_axis.truncate(),
    ];
    // Numerical tolerance for testing a composed rigid transform, not a cache
    // visibility tolerance. Unknown/non-rigid transforms use global invalidation.
    axes.iter()
        .all(|axis| (axis.length_squared() - 1.0).abs() <= 1e-4)
        && axes[0].dot(axes[1]).abs() <= 1e-4
        && axes[0].dot(axes[2]).abs() <= 1e-4
        && axes[1].dot(axes[2]).abs() <= 1e-4
        && axes[0].cross(axes[1]).dot(axes[2]) > 0.0
}

fn bounded_geometry_history_reuse(
    previous: &SceneHistorySnapshot,
    current: &SceneHistorySnapshot,
) -> bool {
    if !previous.valid
        || !current.valid
        || previous.geometry_revision == current.geometry_revision
        || previous.mesh_revision != current.mesh_revision
        || previous.topology_revision != current.topology_revision
        || previous.instance_count == 0
        || previous.instance_count != current.instance_count
        || previous.instance_material_range != current.instance_material_range
        || previous.deformation_poses != current.deformation_poses
        || !Arc::ptr_eq(&previous.textures, &current.textures)
        || scene_bounds(previous) != scene_bounds(current)
        || scene_bounds(current)
            .is_none_or(|(min, max)| !min.is_finite() || !max.is_finite() || min.cmpgt(max).any())
    {
        return false;
    }
    let Some(tail) = current
        .instance_count
        .checked_mul(INSTANCE_WORDS_PER_RECORD)
    else {
        return false;
    };
    if !instance_tail_matches_except_emission(previous, current, tail) {
        return false;
    }
    for index in 0..current.instance_count {
        let (Some(old), Some(new)) = (
            previous.instance_record(index),
            current.instance_record(index),
        ) else {
            return false;
        };
        // Identity/order, material addresses, flags, layers and BLAS metadata
        // must be unchanged. Packed material/texture data is checked above.
        if old[6..] != new[6..] {
            return false;
        }
        if old[..6] != new[..6]
            && (old[3..6] == new[3..6]
                || old[6].w & (INSTANCE_FLAG_ACTIVE_DEFORMATION | (1 << 7)) != 0
                || !rigid_instance_transform(old, new))
        {
            return false;
        }
    }
    true
}

fn light_only_scene_snapshot_unchanged(
    previous: &SceneHistorySnapshot,
    current: &SceneHistorySnapshot,
) -> bool {
    previous.valid
        && current.valid
        && previous.geometry_revision == current.geometry_revision
        && previous.mesh_revision == current.mesh_revision
        && previous.topology_revision == current.topology_revision
        && previous.instance_count == current.instance_count
        && previous.instance_material_range == current.instance_material_range
        && Arc::ptr_eq(&previous.nodes, &current.nodes)
        && Arc::ptr_eq(&previous.deformation_poses, &current.deformation_poses)
        && Arc::ptr_eq(&previous.textures, &current.textures)
        && previous
            .instance_count
            .checked_mul(INSTANCE_WORDS_PER_RECORD)
            .is_some_and(|instance_tail| {
                instance_tail_matches_except_emission(previous, current, instance_tail)
                    && (0..previous.instance_count).all(|index| {
                        matches!(
                            (previous.instance_record(index), current.instance_record(index)),
                            (Some(previous), Some(current)) if previous == current
                        )
                    })
            })
}

fn include_bounds(bounds: &mut Option<(Vec3, Vec3)>, min: Vec3, max: Vec3) -> bool {
    if !min.is_finite() || !max.is_finite() || min.cmpgt(max).any() {
        return false;
    }
    *bounds = Some(bounds.map_or((min, max), |(current_min, current_max)| {
        (current_min.min(min), current_max.max(max))
    }));
    true
}

fn include_instance_leaf_bounds(
    snapshot: &SceneHistorySnapshot,
    index: usize,
    bounds: &mut Option<(Vec3, Vec3)>,
) -> bool {
    let mut included = false;
    for node in snapshot.nodes.iter().filter(|node| {
        node.count > 0
            && (index >= node.first as usize)
            && (index as u64) < u64::from(node.first) + u64::from(node.count)
    }) {
        included |= include_bounds(bounds, node.min, node.max);
    }
    included
}

fn include_instance_dirty_bounds(
    snapshot: &SceneHistorySnapshot,
    index: usize,
    bounds: &mut Option<(Vec3, Vec3)>,
) -> bool {
    // An index beyond this snapshot's instance count represents an instance
    // that did not exist in this scene state (for example, a removed object).
    // There is no pose on this side to invalidate, so do not fall back to the
    // whole scene root.
    if index >= snapshot.instance_count {
        return false;
    }
    if let Some((min, max)) = snapshot
        .instances
        .get(index)
        .and_then(crate::scene::SceneInstance::world_bounds)
    {
        return include_bounds(bounds, min, max);
    }
    if include_instance_leaf_bounds(snapshot, index, bounds) {
        return true;
    }
    scene_bounds(snapshot).is_some_and(|(min, max)| include_bounds(bounds, min, max))
}

fn instance_is_deformed(snapshot: &SceneHistorySnapshot, index: usize) -> bool {
    let flags_word = index
        .checked_mul(INSTANCE_WORDS_PER_RECORD)
        .and_then(|offset| snapshot.instance_words.get(offset + 6));
    flags_word.is_some_and(|word| word.w & INSTANCE_FLAG_ACTIVE_DEFORMATION != 0)
}

fn scene_bounds(snapshot: &SceneHistorySnapshot) -> Option<(Vec3, Vec3)> {
    snapshot.nodes.first().map(|root| (root.min, root.max))
}

fn dirty_scene_bounds(
    previous: &SceneHistorySnapshot,
    current: &SceneHistorySnapshot,
) -> (Vec4, Vec4) {
    if !previous.valid {
        return (Vec4::ZERO, Vec4::ZERO);
    }
    let mut bounds = None;
    let count = previous.instance_count.max(current.instance_count);
    for index in 0..count {
        if previous.instance_record(index) != current.instance_record(index) {
            include_instance_dirty_bounds(previous, index, &mut bounds);
            include_instance_dirty_bounds(current, index, &mut bounds);
        }
    }

    if previous.deformation_poses.as_slice() != current.deformation_poses.as_slice() {
        for index in 0..previous.instance_count {
            if instance_is_deformed(previous, index) {
                include_instance_dirty_bounds(previous, index, &mut bounds);
            }
        }
        for index in 0..current.instance_count {
            if instance_is_deformed(current, index) {
                include_instance_dirty_bounds(current, index, &mut bounds);
            }
        }
    }

    // If geometry changed without a corresponding packed-instance delta (for
    // example, a changed BLAS bound), invalidate the old and new scene extents.
    // A strict material-emission-only proof has no geometry bounds to invalidate.
    if bounds.is_none()
        && previous.geometry_revision != current.geometry_revision
        && !bounded_geometry_history_reuse(previous, current)
    {
        if let Some((min, max)) = scene_bounds(previous) {
            include_bounds(&mut bounds, min, max);
        }
        if let Some((min, max)) = scene_bounds(current) {
            include_bounds(&mut bounds, min, max);
        }
    }

    bounds.map_or((Vec4::ZERO, Vec4::ZERO), |(min, max)| {
        (min.extend(1.0), max.extend(0.0))
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum PunctualLightHistoryChange {
    Unchanged,
    Local { minimum: Vec3, maximum: Vec3 },
    Global,
}

// Every lighting change disables unrestricted reuse; local bounds let the
// shader recover reuse only for samples whose complete dependencies miss them.
fn punctual_light_history_invalidates_transport(change: PunctualLightHistoryChange) -> bool {
    !matches!(change, PunctualLightHistoryChange::Unchanged)
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum EmissiveLightHistoryChange {
    Unchanged,
    Local { minimum: Vec3, maximum: Vec3 },
    Global,
}

fn shared_vec_eq<T: PartialEq>(previous: &Arc<Vec<T>>, current: &Arc<Vec<T>>) -> bool {
    Arc::ptr_eq(previous, current) || previous.as_ref() == current.as_ref()
}

fn emissive_source_is_valid(light: GiLight, instance_count: usize) -> bool {
    (light.source.x as usize) < instance_count
        && light.source.z == 0
        && light.source.w == 0
        && light.position.is_finite()
        && light.position.w < 0.0
        && light.color.is_finite()
        && light.color.truncate().cmpge(Vec3::ZERO).all()
        && light.color.w > 0.0
        && light.direction.is_finite()
        && light.direction.w > 0.0
        && light.direction.w <= 1.0
        && light.edge1.is_finite()
        && light.edge1.w > 0.0
        && light.edge1.w <= 1.0
        && light.edge2.is_finite()
        && light.edge2.w == 0.0
}

// The emitter vector is entity-ordered; use packed entity identity, not either
// the emitter array slot or the transient TLAS instance index, as its merge key.
fn emissive_source_key(snapshot: &SceneHistorySnapshot, light: GiLight) -> Option<(u64, u32)> {
    let identity = snapshot.instance_record(light.source.x as usize)?.get(7)?;
    Some((
        u64::from(identity.x) | (u64::from(identity.y) << 32),
        light.source.y,
    ))
}

// Scene emission is stable entity/triangle order, so valid histories can be
// matched in one allocation-free pass after this ordering check.
fn emissive_sources_are_ordered(snapshot: &SceneHistorySnapshot) -> bool {
    let mut previous = None;
    for light in snapshot.emissive_lights.iter() {
        if !emissive_source_is_valid(*light, snapshot.instance_count) {
            return false;
        }
        let Some(key) = emissive_source_key(snapshot, *light) else {
            return false;
        };
        if previous.is_some_and(|previous| previous >= key) {
            return false;
        }
        previous = Some(key);
    }
    true
}

fn emissive_source_geometry_matches(previous: GiLight, current: GiLight) -> bool {
    previous.position == current.position
        && previous.color.w == current.color.w
        && previous.direction.truncate() == current.direction.truncate()
        && previous.edge1.truncate() == current.edge1.truncate()
        && previous.edge2 == current.edge2
}

fn emissive_source_instance_unchanged(
    previous: &SceneHistorySnapshot,
    current: &SceneHistorySnapshot,
    index: usize,
) -> bool {
    let (Some(previous), Some(current)) = (
        previous.instance_record(index),
        current.instance_record(index),
    ) else {
        return false;
    };
    previous[..6] == current[..6]
}

fn include_emissive_source_bounds(bounds: &mut Option<(Vec3, Vec3)>, light: GiLight) -> bool {
    let a = light.position.truncate();
    let b = a + light.edge1.truncate();
    let c = a + light.edge2.truncate();
    if !a.is_finite() || !b.is_finite() || !c.is_finite() {
        return false;
    }
    let extent =
        Vec3::splat(16.0 * f32::EPSILON * a.abs().max(b.abs()).max(c.abs()).max_element().max(1.0));
    include_bounds(bounds, a.min(b).min(c) - extent, a.max(b).max(c) + extent)
}

fn emissive_light_history_change(
    previous: &SceneHistorySnapshot,
    current: &SceneHistorySnapshot,
    bounded_geometry_reuse: bool,
) -> EmissiveLightHistoryChange {
    if shared_vec_eq(&previous.emissive_lights, &current.emissive_lights) {
        return EmissiveLightHistoryChange::Unchanged;
    }
    if !previous.valid
        || !current.valid
        || previous.revision == current.revision
        || !bounded_geometry_reuse
        || !emissive_sources_are_ordered(previous)
        || !emissive_sources_are_ordered(current)
    {
        return EmissiveLightHistoryChange::Global;
    }

    // CDF/PDF fields may shift when a source is removed. Cached samples retain
    // their original proposal weights; compare source geometry and RGB only.
    let mut previous_index = 0;
    let mut current_index = 0;
    let mut bounds = None;
    let mut decreased = false;
    while let Some(old) = previous.emissive_lights.get(previous_index).copied() {
        let Some(new) = current.emissive_lights.get(current_index).copied() else {
            if old.color.truncate().max_element() <= 0.0
                || !emissive_source_instance_unchanged(previous, current, old.source.x as usize)
                || !include_emissive_source_bounds(&mut bounds, old)
            {
                return EmissiveLightHistoryChange::Global;
            }
            decreased = true;
            previous_index += 1;
            continue;
        };
        let (Some(old_key), Some(new_key)) = (
            emissive_source_key(previous, old),
            emissive_source_key(current, new),
        ) else {
            return EmissiveLightHistoryChange::Global;
        };
        if old_key < new_key {
            if old.color.truncate().max_element() <= 0.0
                || !emissive_source_instance_unchanged(previous, current, old.source.x as usize)
                || !include_emissive_source_bounds(&mut bounds, old)
            {
                return EmissiveLightHistoryChange::Global;
            }
            decreased = true;
            previous_index += 1;
            continue;
        }
        if old_key != new_key || !emissive_source_geometry_matches(old, new) {
            return EmissiveLightHistoryChange::Global;
        }
        let previous_rgb = old.color.truncate();
        let current_rgb = new.color.truncate();
        if !current_rgb.cmple(previous_rgb).all() {
            return EmissiveLightHistoryChange::Global;
        }
        if current_rgb != previous_rgb {
            if !include_emissive_source_bounds(&mut bounds, old)
                || !include_emissive_source_bounds(&mut bounds, new)
            {
                return EmissiveLightHistoryChange::Global;
            }
            decreased = true;
        }
        previous_index += 1;
        current_index += 1;
    }
    if current_index != current.emissive_lights.len() || !decreased {
        return EmissiveLightHistoryChange::Global;
    }
    bounds.map_or(EmissiveLightHistoryChange::Global, |(minimum, maximum)| {
        EmissiveLightHistoryChange::Local { minimum, maximum }
    })
}

fn emissive_light_history_invalidates_transport(change: EmissiveLightHistoryChange) -> bool {
    !matches!(change, EmissiveLightHistoryChange::Unchanged)
}

fn local_lighting_metadata_allowed(global_view_change: bool, local_geometry_proven: bool) -> bool {
    !global_view_change && local_geometry_proven
}

fn punctual_dirty_bounds(
    change: PunctualLightHistoryChange,
    global_view_change: bool,
    local_geometry_proven: bool,
) -> (Vec4, Vec4) {
    match change {
        PunctualLightHistoryChange::Local { minimum, maximum }
            if local_lighting_metadata_allowed(global_view_change, local_geometry_proven) =>
        {
            (minimum.extend(1.0), maximum.extend(0.0))
        }
        _ => (Vec4::ZERO, Vec4::ZERO),
    }
}

fn emissive_dirty_bounds(
    change: EmissiveLightHistoryChange,
    global_view_change: bool,
    local_geometry_proven: bool,
) -> (Vec4, Vec4) {
    match change {
        EmissiveLightHistoryChange::Local { minimum, maximum }
            if local_lighting_metadata_allowed(global_view_change, local_geometry_proven) =>
        {
            (minimum.extend(1.0), maximum.extend(0.0))
        }
        _ => (Vec4::ZERO, Vec4::ZERO),
    }
}

fn include_punctual_influence_bounds(bounds: &mut Option<(Vec3, Vec3)>, light: GiLight) -> bool {
    if !light.position.is_finite()
        || light.position.w <= 0.0
        || !light.color.is_finite()
        || !light.direction.is_finite()
        || !light.edge1.is_finite()
        || !light.edge2.is_finite()
        || light.edge1 != Vec4::ZERO
        || light.edge2.truncate() != Vec3::ZERO
        || light.edge2.w < 0.0
        || light.source != UVec4::ZERO
    {
        return false;
    }

    // Expand by a few float ULPs so a dependency AABB that touches the light's
    // finite range cannot fall just outside the host's influence bound.
    let epsilon = 16.0
        * f32::EPSILON
        * light
            .position
            .truncate()
            .abs()
            .max_element()
            .max(light.position.w)
            .max(1.0);
    let radius = light.position.w + epsilon;
    let extent = Vec3::splat(radius);
    let minimum = light.position.truncate() - extent;
    let maximum = light.position.truncate() + extent;
    if !minimum.is_finite() || !maximum.is_finite() {
        return false;
    }
    *bounds = Some(
        bounds.map_or((minimum, maximum), |(current_min, current_max)| {
            (current_min.min(minimum), current_max.max(maximum))
        }),
    );
    true
}

fn punctual_light_history_change(
    previous: &SceneHistorySnapshot,
    current: &SceneHistorySnapshot,
) -> PunctualLightHistoryChange {
    if shared_vec_eq(&previous.lights, &current.lights)
        && shared_vec_eq(&previous.light_layers, &current.light_layers)
    {
        return PunctualLightHistoryChange::Unchanged;
    }
    if !previous.valid || !current.valid || previous.revision == current.revision {
        return PunctualLightHistoryChange::Global;
    }

    let mut bounds = None;
    for index in 0..previous.lights.len().max(current.lights.len()) {
        let old = previous.lights.get(index).copied();
        let new = current.lights.get(index).copied();
        if old == new && previous.light_layers.get(index) == current.light_layers.get(index) {
            continue;
        }
        for light in [old, new].into_iter().flatten() {
            if light.position.w == 0.0 {
                continue;
            }
            if !light.position.w.is_finite()
                || light.position.w < 0.0
                || !include_punctual_influence_bounds(&mut bounds, light)
            {
                return PunctualLightHistoryChange::Global;
            }
        }
    }

    bounds.map_or(
        PunctualLightHistoryChange::Unchanged,
        |(minimum, maximum)| PunctualLightHistoryChange::Local { minimum, maximum },
    )
}
fn instance_emission_words_changed(
    previous: &SceneHistorySnapshot,
    current: &SceneHistorySnapshot,
) -> bool {
    if !previous.valid || !current.valid {
        return true;
    }
    if Arc::ptr_eq(&previous.instance_words, &current.instance_words) {
        return false;
    }
    if previous.instance_count == 0 {
        return false;
    }
    let Some(instance_tail) = previous
        .instance_count
        .checked_mul(INSTANCE_WORDS_PER_RECORD)
    else {
        return true;
    };
    let (Some(previous_emission_words), Some(current_emission_words)) = (
        material_emission_word_offsets(previous, instance_tail),
        material_emission_word_offsets(current, instance_tail),
    ) else {
        return true;
    };
    previous_emission_words != current_emission_words
        || previous_emission_words.iter().any(|&offset| {
            previous.instance_words.get(offset) != current.instance_words.get(offset)
        })
}

fn active_source_mask(snapshot: &SceneHistorySnapshot) -> u32 {
    let mut active = if snapshot.emissive_lights.is_empty() {
        0
    } else {
        SOURCE_EMISSIVE
    };
    if snapshot
        .lights
        .iter()
        .zip(snapshot.light_layers.iter())
        .any(|(light, layers)| {
            snapshot.view_layers.intersects(layers)
                && light.position.w.is_finite()
                && light.position.w > 0.0
        })
    {
        active |= SOURCE_PUNCTUAL;
    }
    active
}

fn conventional_source_mask(light: GiLight) -> u32 {
    if !light.position.is_finite()
        || !light.color.is_finite()
        || !light.direction.is_finite()
        || !light.edge1.is_finite()
        || !light.edge2.is_finite()
    {
        SOURCE_BACKGROUND | SOURCE_PUNCTUAL
    } else if light.position.w == 0.0 {
        SOURCE_BACKGROUND
    } else if light.position.w > 0.0 {
        SOURCE_PUNCTUAL
    } else {
        SOURCE_BACKGROUND | SOURCE_PUNCTUAL
    }
}

fn conventional_source_changes(
    previous: &SceneHistorySnapshot,
    current: &SceneHistorySnapshot,
) -> u32 {
    if shared_vec_eq(&previous.lights, &current.lights)
        && shared_vec_eq(&previous.light_layers, &current.light_layers)
        && previous.view_layers == current.view_layers
    {
        return 0;
    }
    let mut changed = 0;
    for index in 0..previous.lights.len().max(current.lights.len()) {
        let old = previous.lights.get(index).copied();
        let new = current.lights.get(index).copied();
        let old_visible = old.is_some()
            && previous
                .light_layers
                .get(index)
                .is_some_and(|layers| previous.view_layers.intersects(layers));
        let new_visible = new.is_some()
            && current
                .light_layers
                .get(index)
                .is_some_and(|layers| current.view_layers.intersects(layers));
        if old == new && old_visible == new_visible {
            continue;
        }
        if old_visible {
            changed |= conventional_source_mask(old.expect("visible light has a source"));
        }
        if new_visible {
            changed |= conventional_source_mask(new.expect("visible light has a source"));
        }
    }
    changed
}
fn gradual_rgb(old: Vec3, new: Vec3) -> bool {
    if !old.is_finite() || !new.is_finite() || old.min_element() < 0.0 || new.min_element() < 0.0 {
        return false;
    }
    let peak = old.max_element().max(new.max_element());
    (old - new).abs().max_element() <= peak * 0.05
        && (old.max_element() > 0.0 || new.max_element() == 0.0)
        && (new.max_element() > 0.0 || old.max_element() == 0.0)
}

// Small per-frame changes can borrow radiance; abrupt changes still invalidate it.
fn gradual_directional_motion(
    previous: &SceneHistorySnapshot,
    current: &SceneHistorySnapshot,
) -> bool {
    if !previous.valid
        || !current.valid
        || previous.view_layers != current.view_layers
        || previous.ambient_environment.alpha != current.ambient_environment.alpha
        || !gradual_rgb(
            previous.ambient_environment.to_vec4().truncate(),
            current.ambient_environment.to_vec4().truncate(),
        )
        || previous.environment != current.environment
        || previous.environment_revision != current.environment_revision
        || previous.lights.len() != current.lights.len()
        || previous.light_layers != current.light_layers
        || previous.lights.len() != previous.light_layers.len()
    {
        return false;
    }
    let mut moved = false;
    for (old, new) in previous.lights.iter().zip(current.lights.iter()) {
        if old == new || (old.position.w > 0.0 && new.position.w > 0.0) {
            continue;
        }
        if old.position.w != 0.0
            || new.position.w != 0.0
            || old.color.w != new.color.w
            || !gradual_rgb(old.color.truncate(), new.color.truncate())
            || old.direction != new.direction
            || old.edge1 != new.edge1
            || old.edge2 != new.edge2
            || old.source != new.source
        {
            return false;
        }
        let a = old.position.truncate();
        let b = new.position.truncate();
        if !a.is_finite()
            || !b.is_finite()
            || (a.length_squared() - 1.0).abs() > 0.001
            || (b.length_squared() - 1.0).abs() > 0.001
            || a.dot(b) < 0.996_194_7
        {
            return false;
        }
        moved = true;
    }
    moved
}

fn gradual_punctual_motion(
    previous: &SceneHistorySnapshot,
    current: &SceneHistorySnapshot,
) -> bool {
    if !previous.valid
        || !current.valid
        || previous.view_layers != current.view_layers
        || previous.light_layers != current.light_layers
        || previous.lights.len() != current.lights.len()
        || previous.lights.len() != previous.light_layers.len()
    {
        return false;
    }
    let mut moved = false;
    for (old, new) in previous.lights.iter().zip(current.lights.iter()) {
        if old == new || (old.position.w == 0.0 && new.position.w == 0.0) {
            continue;
        }
        let a = old.direction.truncate();
        let b = new.direction.truncate();
        if !old.position.is_finite()
            || !new.position.is_finite()
            || old.position.w <= 0.0
            || old.position.w != new.position.w
            || (old.position.truncate() - new.position.truncate()).length() > old.position.w * 0.01
            || old.color.w != new.color.w
            || !gradual_rgb(old.color.truncate(), new.color.truncate())
            || !a.is_finite()
            || !b.is_finite()
            || old.direction.w != new.direction.w
            || (a != b
                && ((a.length_squared() - 1.0).abs() > 0.001
                    || (b.length_squared() - 1.0).abs() > 0.001
                    || a.dot(b) < 0.996_194_7))
            || old.edge1 != new.edge1
            || old.edge2 != new.edge2
            || old.source != new.source
        {
            return false;
        }
        moved = true;
    }
    moved
}

fn source_history_fields(
    previous: &SceneHistorySnapshot,
    current: &SceneHistorySnapshot,
    punctual_change: PunctualLightHistoryChange,
    emissive_change: EmissiveLightHistoryChange,
) -> UVec4 {
    let active = active_source_mask(current);
    let previous_active = active_source_mask(previous);
    let mut changed = active ^ previous_active;
    if !previous.valid || !current.valid {
        changed |= SOURCE_BACKGROUND | active | previous_active;
    } else {
        if previous.view_layers != current.view_layers {
            changed |= SOURCE_BACKGROUND | active | previous_active;
        }
        if previous.ambient_environment != current.ambient_environment
            || previous.environment != current.environment
            || previous.environment_revision != current.environment_revision
        {
            changed |= SOURCE_BACKGROUND;
        }
        if emissive_light_history_invalidates_transport(emissive_change)
            || instance_emission_words_changed(previous, current)
        {
            changed |= SOURCE_EMISSIVE;
        }
        changed |= conventional_source_changes(previous, current);
        if punctual_light_history_invalidates_transport(punctual_change) {
            changed |= SOURCE_PUNCTUAL;
        }
    }
    UVec4::new(
        active,
        changed,
        u32::from(gradual_directional_motion(previous, current))
            | (u32::from(gradual_punctual_motion(previous, current)) << 2),
        0,
    )
}

#[derive(Component)]
struct ViewBuffers {
    /// Sparse ray/probe work grid.
    size: UVec2,
    /// Resolve-sized receiver shading and screen-history grid.
    shading_size: UVec2,
    /// Full viewport extent at allocation time, even when resolve is reduced.
    viewport_size: UVec2,
    buckets: u32,
    caches: [CacheBuffers; 2],
    trace_indirect: Buffer,
    entries: Buffer,
    history_entries: Buffer,
    irradiance_probes: Buffer,
    secondary: Option<SecondaryCache>,
    irradiance_textures: [Texture; 4],
    irradiance_views: [TextureView; 4],
    _background_textures: [Texture; 3],
    background_views: [TextureView; 3],
    _metadata: Vec<Texture>,
    // Slots 0/1 hold temporal moments, slot 2 holds sparse receiver metadata,
    // and slots 3/4 ping-pong packed full-viewport receiver history.
    metadata_views: [TextureView; 5],
    _denoise_receiver_history_output: Texture,
    denoise_receiver_history_output_view: TextureView,
    /// Composite anchor position/compatibility and geometric normal.
    _composite_anchor_textures: [Texture; 2],
    composite_anchor_views: [TextureView; 2],
    /// Ray-traced GI reservoirs by frame parity; one placeholder entry each
    /// unless the view uses `GiMethod::RayTraced`.
    rtgi_reservoirs: [Buffer; 2],
    /// Ray-traced GI world radiance cache keys, cells and shading queue.
    rtgi_cache: [Buffer; 3],
    /// Packed hard sun-shadow bits; see `sun_shadow_mask_size`.
    _sun_shadow_mask: Texture,
    sun_shadow_mask_view: TextureView,
    uniforms: [UniformBuffer<Parameters>; LEVELS],
    clear_uniforms: [UniformBuffer<Parameters>; 2],
    /// Reuse bind groups while their exact resources/layout slices remain stable.
    /// The two parity slots bound how long superseded GPU resources stay retained.
    binding_cache: [Option<CachedViewBindings>; 2],
    rendered_frame: Mutex<RenderedFrame>,
    cache_clock: std::time::Instant,
    scene_history: Mutex<SceneHistorySnapshot>,
    // Frame of the last gradual directional change, plus one; zero means none.
    directional_motion_epoch: AtomicU32,
    punctual_motion_epoch: AtomicU32,
    /// Per-view phase for transport candidate epochs; advances only after a
    /// successful render update, independently of the wrapping GPU frame index.
    candidate_schedule: Mutex<CandidateSchedule>,
    viewport: UVec4,
    frame: AtomicU32,
    history_valid: AtomicBool,
    screen_history_valid: AtomicBool,
    /// Raw-output requests enable next-frame debug histograms without changing the ABI.
    history_diagnostics_requested: AtomicBool,
    revision: u64,
    settings: RadianceCascades,
    allocation_request: RadianceCascades,
    mesh_revision: u64,
    geometry_revision: u64,
    render_layers: RenderLayers,
    lights: Option<Buffer>,
    light_data: Vec<GiLight>,
    light_source: Option<(bevy::asset::AssetId<Image>, u64)>,
    light_revision: u64,
    pending_lights: Option<PendingLights>,
    conventional_lights: u32,
    layer_selector: UVec4,
    environment: Option<crate::RadianceCascadesEnvironment>,
    environment_revision: u64,
    environment_info: UVec4,
}

#[derive(Clone, Copy, Default)]
struct RenderedFrame {
    clip_from_world: Mat4,
    world_from_clip: Mat4,
    /// Unjittered camera projection used to invalidate sparse receiver fields.
    stable_clip_from_world: Mat4,
    exposure: f32,
}

#[derive(Clone, Default)]
struct CandidateSchedule {
    width: u32,
    height: u32,
    candidate_count: u32,
    cursor: u32,
    pending_fair_advance: u32,
}

impl CandidateSchedule {
    fn phase_for(
        &mut self,
        width: u32,
        height: u32,
        candidate_count: u32,
        budget: u32,
    ) -> (u32, u32) {
        if (self.width, self.height, self.candidate_count) != (width, height, candidate_count) {
            self.width = width;
            self.height = height;
            self.candidate_count = candidate_count;
            self.cursor = 0;
        }
        if candidate_count > 0 {
            self.cursor %= candidate_count;
        } else {
            self.cursor = 0;
        }

        let fair_quota = budget.min(candidate_count);
        self.pending_fair_advance = fair_quota;
        (self.cursor, fair_quota)
    }

    fn advance_after_successful_update(&mut self, consume_fair_cursor: bool) {
        if consume_fair_cursor && self.candidate_count > 0 {
            self.cursor = ((u64::from(self.cursor) + u64::from(self.pending_fair_advance))
                % u64::from(self.candidate_count)) as u32;
        }
    }
}

fn candidate_field_indices(frame: u32) -> (usize, usize) {
    ((frame & 1) as usize, (frame.wrapping_add(1) & 1) as usize)
}

#[derive(Component, Clone)]
struct ViewBindings {
    compute: [BindGroup; LEVELS],
    clear: [BindGroup; 2],
    trace: Option<BindGroup>,
    secondary: Option<SecondaryBindings>,
    secondary_enabled: bool,
    denoise: [BindGroup; LEVELS],
    temporal: BindGroup,
    world_resolve: BindGroup,
    composite_anchors: BindGroup,
    composite: BindGroup,
    /// Writes the sun shadow mask; present with hardware ray queries.
    sun_shadows: Option<BindGroup>,
    frame: RenderedFrame,
    scene_history: SceneHistorySnapshot,
}

#[derive(Clone)]
struct SecondaryBindings {
    compute: [BindGroup; LEVELS],
    clear: [BindGroup; 2],
}

#[derive(Clone)]
struct CachedViewBindings {
    key: u64,
    bindings: ViewBindings,
}

fn hash_buffer<H: Hasher>(buffer: &Buffer, hasher: &mut H) {
    buffer.id().hash(hasher);
    buffer.size().hash(hasher);
}

fn hash_view<H: Hasher>(view: &TextureView, hasher: &mut H) {
    view.id().hash(hasher);
}

fn composite_display_index(spatial_filter_steps: u32, parity: usize) -> usize {
    if spatial_filter_steps == 0 {
        parity
    } else {
        2 + spatial_filter_steps as usize % 2
    }
}

fn composite_receiver_history_source<'a, T>(
    spatial_filter_steps: u32,
    temporal: &'a T,
    filtered: &'a T,
) -> &'a T {
    if spatial_filter_steps == 0 {
        temporal
    } else {
        filtered
    }
}

fn view_binding_key(
    buffers: &ViewBuffers,
    parity: usize,
    display_index: usize,
    external_buffers: [&Buffer; 3],
    external_views: &[&TextureView],
) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    parity.hash(&mut hasher);
    display_index.hash(&mut hasher);
    for buffer in external_buffers {
        hash_buffer(buffer, &mut hasher);
        0u64.hash(&mut hasher);
        buffer.size().hash(&mut hasher);
    }
    for view in external_views {
        hash_view(view, &mut hasher);
    }
    for buffer in [
        &buffers.caches[parity].heads,
        &buffers.caches[1 - parity].heads,
        &buffers.entries,
        &buffers.history_entries,
        &buffers.irradiance_probes,
        &buffers.rtgi_reservoirs[0],
        &buffers.rtgi_reservoirs[1],
        &buffers.rtgi_cache[0],
        &buffers.rtgi_cache[1],
        &buffers.rtgi_cache[2],
    ] {
        hash_buffer(buffer, &mut hasher);
    }
    if let Some(secondary) = &buffers.secondary {
        for buffer in [
            &secondary.caches[parity].heads,
            &secondary.caches[1 - parity].heads,
            &secondary.entries,
            &secondary.history_entries,
        ] {
            hash_buffer(buffer, &mut hasher);
        }
        for view in &secondary.hit_views {
            hash_view(view, &mut hasher);
        }
        for uniform in secondary.uniforms.iter().chain(&secondary.clear_uniforms) {
            if let Some(buffer) = uniform.buffer() {
                hash_buffer(buffer, &mut hasher);
            }
        }
    }
    for uniform in buffers.uniforms.iter().chain(&buffers.clear_uniforms) {
        if let Some(buffer) = uniform.buffer() {
            hash_buffer(buffer, &mut hasher);
            0u64.hash(&mut hasher);
            std::mem::size_of::<Parameters>().hash(&mut hasher);
        }
    }
    for view in buffers
        .irradiance_views
        .iter()
        .chain(&buffers.metadata_views)
        .chain(&buffers.background_views)
    {
        hash_view(view, &mut hasher);
    }
    hash_view(&buffers.denoise_receiver_history_output_view, &mut hasher);
    for view in &buffers.composite_anchor_views {
        hash_view(view, &mut hasher);
    }
    hash_view(&buffers.sun_shadow_mask_view, &mut hasher);
    hasher.finish()
}

fn resolution(viewport: UVec2, settings: RadianceCascades) -> UVec2 {
    let size_at = |divisor: u32| {
        UVec2::new(
            viewport.x.div_ceil(divisor).max(1),
            viewport.y.div_ceil(divisor).max(1),
        )
    };
    let max_receivers = u64::from(if settings.sampling.max_receiver_cells == 0 {
        131_072
    } else {
        settings.sampling.max_receiver_cells
    });
    let divisor = if settings.sampling.grid_divisor != 0 {
        settings.sampling.grid_divisor
    } else {
        settings.resolution_divisor
    };
    let mut low = divisor.max(1);
    let mut high = viewport.max_element().max(low);
    while low < high {
        let middle = low + (high - low) / 2;
        let size = size_at(middle);
        let receivers = u64::from(size.x) * u64::from(size.y);
        if receivers > max_receivers {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    size_at(low)
}

fn view_change_invalidates_screen_history(
    settings_history_changed: bool,
    viewport_changed: bool,
) -> bool {
    settings_history_changed || viewport_changed
}

fn camera_matrix_changed(previous: Mat4, current: Mat4) -> bool {
    previous
        .to_cols_array()
        .into_iter()
        .zip(current.to_cols_array())
        .any(|(previous, current)| previous != current)
}

fn sparse_history_reset_requested(
    has_rendered_history: bool,
    geometry_or_layers_changed: bool,
    settings_history_changed: bool,
    viewport_changed: bool,
    camera_changed: bool,
) -> bool {
    !has_rendered_history
        || geometry_or_layers_changed
        || settings_history_changed
        || viewport_changed
        || camera_changed
}

fn sparse_candidate_phase_word(candidate_phase: u32, reset: bool, updates_disabled: bool) -> u32 {
    (candidate_phase & 0x03ff_ffff)
        | if reset { SPARSE_HISTORY_RESET_FLAG } else { 0 }
        | if updates_disabled {
            SPARSE_UPDATES_DISABLED_FLAG
        } else {
            0
        }
}

fn reset_cache_history(
    layers_changed: bool,
    exposure_changed: bool,
    meshes_changed: bool,
    settings_changed: bool,
    viewport_changed: bool,
) -> bool {
    layers_changed || exposure_changed || meshes_changed || settings_changed || viewport_changed
}

fn discard_obsolete_scene(live: u64, candidate: u64, budget: u64, obsolete: bool) -> bool {
    obsolete && candidate <= budget && live > budget
}

fn clear_pass_parameters(mut params: Parameters, clear_index: usize) -> Parameters {
    params.owner_sequence.z = u32::from(clear_index == 0);
    params
}

/// Whether this frame's shadow-mapped directional light takes ray-traced
/// visibility instead of Bevy's cascades. Decided before Bevy prepares lights.
#[derive(Resource, Default)]
struct SunShadowOwnership(bool);

/// Hands the shadowed sun to the patched Bevy lookup: no cascades, a marked
/// depth bias, and a directional shadow texture large enough to hold the
/// screen-space visibility in layer 0.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn own_sun_shadows(
    mut ownership: ResMut<SunShadowOwnership>,
    patch: Option<Res<crate::sun_shadow_patch::SunShadowPatch>>,
    scene: Res<GiScene>,
    gpu_scene: Option<Res<GpuScene>>,
    hardware: Option<Res<hardware::HardwareTraversal>>,
    pipelines: Option<Res<Pipelines>>,
    cache: Res<PipelineCache>,
    cameras: Query<
        (
            &ExtractedView,
            Option<&RadianceCascades>,
            Option<&ViewBindings>,
            Has<ViewBuffers>,
            Has<crate::receiver::ReceiverTexture>,
        ),
        With<Camera3d>,
    >,
    mut lights: Query<&mut bevy::pbr::ExtractedDirectionalLight>,
    punctual: Query<&bevy::pbr::ExtractedPointLight>,
    mut shadow_map: ResMut<bevy::light::DirectionalLightShadowMap>,
    status: Option<Res<RadianceCascadesStatus>>,
) {
    let ready = patch.is_some_and(|patch| patch.applied())
        && scene.ready
        && gpu_scene.is_some_and(|gpu| gpu.ready_for_revision(scene.revision))
        && hardware.is_some_and(|hardware| hardware.ready_for(scene.revision))
        && pipelines.is_some_and(|pipelines| {
            pipelines.pipeline(SUN_SHADOWS_HARDWARE).is_some()
                && cache.get_render_pipeline(pipelines.sun_shadow_depth).is_some()
        });
    // A camera on Bevy's shadow maps would lose them, so every 3D camera must
    // trace the sun. Its bindings come from the previous frame's prepare.
    // shortcut: a view whose bindings are dropped later this frame shows one
    // unshadowed frame; move the decision after prepare if that becomes visible.
    let views_ready = !cameras.is_empty()
        && cameras.iter().all(|(_, settings, bindings, buffers, receiver)| {
            settings.is_some_and(|settings| {
                let settings = settings.sanitized();
                settings.ray_traced_sun_shadows && settings.hardware_ray_queries
            }) && bindings.is_some_and(|bindings| bindings.sun_shadows.is_some())
                && buffers
                && receiver
        });
    // Without cascades, shadowed spot lights would take layer 0 as well.
    let spot_shadows = punctual
        .iter()
        .any(|light| light.shadow_maps_enabled && light.spot_light_angles.is_some());
    let shadowed = lights.iter().filter(|light| light.shadow_maps_enabled).count();
    ownership.0 = ready && views_ready && !spot_shadows && shadowed <= SUN_SHADOW_LIGHTS;
    if let Some(status) = status {
        status.set_ray_traced_sun_shadows(ownership.0);
    }
    if !ownership.0 {
        return;
    }
    for mut light in &mut lights {
        if light.shadow_maps_enabled {
            light.shadow_maps_enabled = false;
            light.shadow_depth_bias = crate::sun_shadow_patch::RAY_TRACED_SHADOW_BIAS;
        }
    }
    let extent = cameras
        .iter()
        .map(|(view, ..)| (view.viewport.xy() + view.viewport.zw()).max_element())
        .max()
        .unwrap_or(1) as usize;
    shadow_map.size = shadow_map.size.max(extent);
}

/// Traces the owned sun's visibility, then writes it as depth into layer 0 of
/// Bevy's directional shadow texture before any Bevy lighting reads it.
fn trace_sun_shadows(
    view: ViewQuery<(&ViewBuffers, &ViewBindings, &bevy::pbr::ViewShadowBindings)>,
    ownership: Res<SunShadowOwnership>,
    pipelines: Res<Pipelines>,
    cache: Res<PipelineCache>,
    mut context: RenderContext,
) {
    let (buffers, bindings, shadows) = view.into_inner();
    let (true, Some(group), Some(mask), Some(depth)) = (
        ownership.0,
        bindings.sun_shadows.as_ref(),
        pipelines.pipeline(SUN_SHADOWS_HARDWARE),
        cache.get_render_pipeline(pipelines.sun_shadow_depth),
    ) else {
        return;
    };
    let diagnostics = context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let span = diagnostics.time_span(context.command_encoder(), "sun_shadows");
    let mut pass = context
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("split_rc_sun_shadow_mask"),
            ..default()
        });
    pass.set_pipeline(mask);
    pass.set_bind_group(0, group, &[]);
    pass.dispatch_workgroups(
        buffers.viewport.z.div_ceil(8),
        buffers.viewport.w.div_ceil(8),
        1,
    );
    drop(pass);
    let layer = shadows
        .directional_light_depth_texture
        .create_view(&TextureViewDescriptor {
            label: Some("split_rc_sun_shadow_layer"),
            dimension: Some(TextureViewDimension::D2),
            aspect: TextureAspect::DepthOnly,
            base_array_layer: 0,
            array_layer_count: Some(1),
            ..default()
        });
    let mut pass = context
        .command_encoder()
        .begin_render_pass(&RenderPassDescriptor {
            label: Some("split_rc_sun_shadow_depth"),
            color_attachments: &[],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &layer,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(1.0),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..default()
        });
    let viewport = buffers.viewport.as_vec4();
    pass.set_viewport(viewport.x, viewport.y, viewport.z, viewport.w, 0.0, 1.0);
    pass.set_pipeline(depth);
    pass.set_bind_group(0, &bindings.composite, &[]);
    pass.draw(0..3, 0..1);
    drop(pass);
    span.end(context.command_encoder());
}

/// One reservoir per resolve sample in ray-traced mode, else a placeholder.
fn rtgi_reservoir_bytes(shading_size: UVec2, settings: RadianceCascades) -> u64 {
    if settings.method == crate::GiMethod::RayTraced {
        u64::from(shading_size.x) * u64::from(shading_size.y) * RTGI_RESERVOIR_BYTES
    } else {
        RTGI_RESERVOIR_BYTES
    }
}

/// Cache keys, cells and the two-parity shading queue in ray-traced mode;
/// one word each otherwise. Buffers start zeroed, so every key is empty.
fn rtgi_cache_bytes(settings: RadianceCascades) -> [u64; 3] {
    if settings.method == crate::GiMethod::RayTraced {
        [
            RTGI_CACHE_CAPACITY * 4,
            RTGI_CACHE_CAPACITY * RTGI_CACHE_CELL_BYTES,
            // Counts, two parity queues, then one work slot per update lane.
            (2 + 2 * RTGI_CACHE_QUEUE + RTGI_CACHE_QUEUE + RTGI_CACHE_REFRESH) * 4,
        ]
    } else {
        [16, RTGI_CACHE_CELL_BYTES, 16]
    }
}

/// One texel per 4x8-pixel tile; an 8x8 dispatch group writes two tiles.
fn sun_shadow_mask_size(viewport: UVec2) -> UVec2 {
    UVec2::new(viewport.x.div_ceil(8) * 2, viewport.y.div_ceil(8)).max(UVec2::ONE)
}

fn view_bytes(size: UVec2, shading_size: UVec2, settings: RadianceCascades) -> u64 {
    let pixels = u64::from(size.x) * u64::from(size.y);
    let shading_pixels = u64::from(shading_size.x) * u64::from(shading_size.y);
    let buckets = cache_bucket_count(pixels, settings);
    // Four diffuse, five temporal metadata, one post-denoise receiver, three
    // independent background current/history images, two composite anchors,
    // and the sun shadow mask over the resolve-covered viewport.
    let mask = sun_shadow_mask_size(shading_size * resolve_scale(settings));
    let image_bytes = shading_pixels * 15 * 16
        + u64::from(mask.x) * u64::from(mask.y) * 16
        + 2 * rtgi_reservoir_bytes(shading_size, settings)
        + rtgi_cache_bytes(settings).iter().sum::<u64>();
    image_bytes
        + 12
        + (LEVELS as u64 + 2) * std::mem::size_of::<Parameters>() as u64
        + irradiance_probe_bytes(pixels, settings.multi_bounce)
        + cache_head_bytes(buckets, pixels) * 2
        + cache_entry_bytes(pixels, buckets)
        + cache_history_bytes(buckets)
        + if settings.multi_bounce {
            pixels * 32
                + cache_head_bytes(buckets, pixels) * 2
                + cache_entry_bytes(pixels, buckets)
                + cache_history_bytes(buckets)
                + (LEVELS as u64 + 2) * std::mem::size_of::<Parameters>() as u64
        } else {
            0
        }
}

impl ViewBuffers {
    fn light_bytes(&self) -> u64 {
        self.lights.as_ref().map_or(0, |buffer| buffer.size())
            + self
                .pending_lights
                .as_ref()
                .filter(|pending| pending.replacement)
                .map_or(0, |pending| pending.buffer.size())
    }

    fn changed_light_ranges(
        previous: &[GiLight],
        next: &[GiLight],
    ) -> Option<Vec<std::ops::Range<usize>>> {
        if previous.len() != next.len() {
            return None;
        }
        let mut ranges = Vec::new();
        let mut index = 0;
        while index < next.len() {
            if previous[index] == next[index] {
                index += 1;
                continue;
            }
            let start = index;
            while index < next.len() && previous[index] != next[index] {
                index += 1;
            }
            ranges.push(start..index);
        }
        Some(ranges)
    }

    fn write_light_ranges(
        queue: &RenderQueue,
        buffer: &Buffer,
        next: &[GiLight],
        ranges: &[std::ops::Range<usize>],
    ) {
        for range in ranges {
            queue.write_buffer(
                buffer,
                range.start as u64 * 96,
                &lights_bytes(&next[range.clone()]),
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_lights(
        &mut self,
        device: &RenderDevice,
        queue: &RenderQueue,
        prefix: Vec<GiLight>,
        layers: &RenderLayers,
        source: Option<(bevy::asset::AssetId<Image>, u64)>,
        scene_revision: u64,
        geometry_revision: u64,
        light_revision: u64,
        allocated: &mut u64,
        memory_budget: u64,
        upload_remaining: &mut usize,
    ) -> bool {
        let bytes = (prefix.len().max(1) as u64).saturating_mul(96);
        if self.pending_lights.as_ref().is_some_and(|pending| {
            pending.source != source
                || pending.layers != *layers
                || pending.prefix.len() != prefix.len()
        }) {
            let pending = self.pending_lights.take().unwrap();
            if pending.replacement {
                *allocated = allocated.saturating_sub(pending.buffer.size());
            }
        }
        if bytes > device.limits().max_storage_buffer_binding_size {
            warn_once!("GI view lights/layers exceed this device's storage buffer limit");
            return false;
        }
        let active_buffer = self.lights.as_ref().filter(|buffer| {
            self.pending_lights.is_none()
                && self.light_source == source
                && self.render_layers == *layers
                && buffer.size() == bytes
        });
        if let Some(buffer) = active_buffer {
            let Some(ranges) = Self::changed_light_ranges(&self.light_data, &prefix) else {
                return false;
            };
            let changed_bytes = ranges.iter().map(|range| range.len() * 96).sum::<usize>();
            if changed_bytes > *upload_remaining {
                return false;
            }
            Self::write_light_ranges(queue, buffer, &prefix, &ranges);
            *upload_remaining -= changed_bytes;
            self.light_data = prefix;
            self.light_revision = light_revision;
            return true;
        }
        if self.pending_lights.is_none() {
            let reuse = self.light_source == source
                && self.render_layers == *layers
                && self
                    .lights
                    .as_ref()
                    .is_some_and(|buffer| buffer.size() == bytes);
            if !reuse && allocated.saturating_add(bytes) > memory_budget {
                warn_once!("GI camera/light replacement exceeds the view GPU memory budget");
                return false;
            }
            let buffer = if reuse {
                self.lights.as_ref().unwrap().clone()
            } else {
                *allocated = allocated.saturating_add(bytes);
                device.create_buffer(&BufferDescriptor {
                    label: Some("gi_view_lights"),
                    size: bytes,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            };
            self.pending_lights = Some(PendingLights {
                buffer,
                source,
                layers: layers.clone(),
                prefix,
                prefix_cursor: 0,
                replacement: !reuse,
                scene_revision,
                geometry_revision,
                light_revision,
            });
        } else if let Some(pending) = self.pending_lights.as_mut()
            && pending.prefix != prefix
        {
            let staged = pending.prefix_cursor.min(prefix.len());
            let Some(ranges) =
                Self::changed_light_ranges(&pending.prefix[..staged], &prefix[..staged])
            else {
                return false;
            };
            let changed_bytes = ranges.iter().map(|range| range.len() * 96).sum::<usize>();
            if changed_bytes > *upload_remaining {
                return false;
            }
            Self::write_light_ranges(queue, &pending.buffer, &prefix, &ranges);
            *upload_remaining -= changed_bytes;
            pending.prefix = prefix;
        }
        let pending = self.pending_lights.as_mut().unwrap();
        pending.scene_revision = scene_revision;
        pending.geometry_revision = geometry_revision;
        pending.light_revision = light_revision;
        // A prefix larger than the per-frame budget must remain stable across
        // several frames; do not publish a mix of different light snapshots.
        while pending.prefix_cursor < pending.prefix.len() && *upload_remaining >= 96 {
            let end = (pending.prefix_cursor + (*upload_remaining).min(1024 * 1024) / 96)
                .min(pending.prefix.len());
            queue.write_buffer(
                &pending.buffer,
                pending.prefix_cursor as u64 * 96,
                &lights_bytes(&pending.prefix[pending.prefix_cursor..end]),
            );
            *upload_remaining -= (end - pending.prefix_cursor) * 96;
            pending.prefix_cursor = end;
        }
        if pending.prefix_cursor < pending.prefix.len() {
            return false;
        }
        debug_assert_eq!(pending.scene_revision, scene_revision);
        debug_assert_eq!(pending.geometry_revision, geometry_revision);
        let pending = self.pending_lights.take().unwrap();
        self.lights = Some(pending.buffer);
        self.light_data = pending.prefix;
        self.light_source = source;
        self.light_revision = pending.light_revision;
        true
    }

    fn can_preserve_world_cache(
        previous: &Self,
        size: UVec2,
        viewport: UVec4,
        settings: RadianceCascades,
    ) -> bool {
        let samples = u64::from(size.x) * u64::from(size.y);
        let buckets = cache_bucket_count(samples, settings) as u32;
        previous.viewport == viewport
            && previous.size == size
            && previous.buckets == buckets
            && previous
                .caches
                .iter()
                .all(|cache| cache.heads.size() == cache_head_bytes(u64::from(buckets), samples))
            && previous.entries.size() == cache_entry_bytes(samples, u64::from(buckets))
            && previous.history_entries.size() == cache_history_bytes(u64::from(buckets))
            && previous.irradiance_probes.size()
                == irradiance_probe_bytes(samples, settings.multi_bounce)
            && !world_cache_settings_changed(previous.settings, settings)
    }

    fn reusable_world_cache_bytes(
        previous: &Self,
        size: UVec2,
        viewport: UVec4,
        settings: RadianceCascades,
    ) -> Option<u64> {
        let samples = u64::from(size.x) * u64::from(size.y);
        Self::can_preserve_world_cache(previous, size, viewport, settings).then(|| {
            previous
                .caches
                .iter()
                .map(|cache| cache.heads.size())
                .sum::<u64>()
                + previous.entries.size()
                + previous.history_entries.size()
                + previous.irradiance_probes.size()
                + previous.secondary.as_ref().map_or(0, |secondary| {
                    secondary
                        .caches
                        .iter()
                        .map(|cache| cache.heads.size())
                        .sum::<u64>()
                        + secondary.entries.size()
                        + secondary.history_entries.size()
                        + samples * 32
                })
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        device: &RenderDevice,
        size: UVec2,
        shading_size: UVec2,
        viewport_size: UVec2,
        settings: RadianceCascades,
        revision: u64,
        previous: Option<&Self>,
        viewport: UVec4,
    ) -> Self {
        let samples = size.x * size.y;
        let cache_samples = samples;
        let buckets = cache_bucket_count(u64::from(cache_samples), settings) as u32;
        let previous = previous
            .filter(|previous| Self::can_preserve_world_cache(previous, size, viewport, settings));
        let caches = previous.map_or_else(
            || {
                std::array::from_fn(|_| CacheBuffers {
                    heads: device.create_buffer(&BufferDescriptor {
                        label: Some("split_rc_hash_heads"),
                        size: cache_head_bytes(u64::from(buckets), u64::from(cache_samples)),
                        usage: BufferUsages::STORAGE
                            | BufferUsages::COPY_DST
                            | BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    }),
                })
            },
            |previous| previous.caches.clone(),
        );
        let entries = previous.map_or_else(
            || {
                device.create_buffer(&BufferDescriptor {
                    label: Some("split_rc_interval_work"),
                    size: cache_entry_bytes(u64::from(cache_samples), u64::from(buckets)),
                    usage: BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            },
            |previous| previous.entries.clone(),
        );
        let history_entries = previous.map_or_else(
            || {
                device.create_buffer(&BufferDescriptor {
                    label: Some("split_rc_interval_history"),
                    size: cache_history_bytes(u64::from(buckets)),
                    usage: BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            },
            |previous| previous.history_entries.clone(),
        );
        let irradiance_probes = previous.map_or_else(
            || {
                device.create_buffer(&BufferDescriptor {
                    label: Some("split_rc_directional_probes"),
                    size: irradiance_probe_bytes(u64::from(cache_samples), settings.multi_bounce),
                    usage: BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            },
            |previous| previous.irradiance_probes.clone(),
        );
        let make_texture = |extent: UVec2| {
            device.create_texture(&TextureDescriptor {
                label: Some("split_rc_irradiance"),
                size: Extent3d {
                    width: extent.x,
                    height: extent.y,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba32Float,
                usage: TextureUsages::STORAGE_BINDING
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_SRC
                    | TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let secondary = settings.multi_bounce.then(|| {
            let previous = previous.and_then(|previous| previous.secondary.as_ref());
            let caches = previous.map_or_else(
                || {
                    std::array::from_fn(|_| CacheBuffers {
                        heads: device.create_buffer(&BufferDescriptor {
                            label: Some("split_rc_secondary_heads"),
                            size: cache_head_bytes(u64::from(buckets), u64::from(cache_samples)),
                            usage: BufferUsages::STORAGE
                                | BufferUsages::COPY_SRC
                                | BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        }),
                    })
                },
                |previous| previous.caches.clone(),
            );
            let entries = previous.map_or_else(
                || {
                    device.create_buffer(&BufferDescriptor {
                        label: Some("split_rc_secondary_work"),
                        size: cache_entry_bytes(u64::from(cache_samples), u64::from(buckets)),
                        usage: BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    })
                },
                |previous| previous.entries.clone(),
            );
            let history_entries = previous.map_or_else(
                || {
                    device.create_buffer(&BufferDescriptor {
                        label: Some("split_rc_secondary_history"),
                        size: cache_history_bytes(u64::from(buckets)),
                        usage: BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    })
                },
                |previous| previous.history_entries.clone(),
            );
            let hit_textures = previous.map_or_else(
                || std::array::from_fn(|_| make_texture(size)),
                |previous| previous.hit_textures.clone(),
            );
            let hit_views = std::array::from_fn(|i| {
                hit_textures[i].create_view(&TextureViewDescriptor::default())
            });
            SecondaryCache {
                caches,
                entries,
                history_entries,
                hit_textures,
                hit_views,
                uniforms: std::array::from_fn(|_| UniformBuffer::default()),
                clear_uniforms: std::array::from_fn(|_| UniformBuffer::default()),
                valid: AtomicBool::new(
                    previous.is_some_and(|cache| cache.valid.load(Ordering::Relaxed)),
                ),
            }
        });
        let irradiance: [Texture; 4] = std::array::from_fn(|_| make_texture(shading_size));
        let irradiance_views =
            std::array::from_fn(|i| irradiance[i].create_view(&TextureViewDescriptor::default()));
        let background_textures: [Texture; 3] = std::array::from_fn(|_| make_texture(shading_size));
        let background_views = std::array::from_fn(|i| {
            background_textures[i].create_view(&TextureViewDescriptor::default())
        });
        let metadata: Vec<Texture> = (0..5).map(|_| make_texture(shading_size)).collect();
        let metadata_views: [TextureView; 5] =
            std::array::from_fn(|i| metadata[i].create_view(&TextureViewDescriptor::default()));
        let denoise_receiver_history_output = device.create_texture(&TextureDescriptor {
            label: Some("gi_denoise_receiver_history_output"),
            size: Extent3d {
                width: shading_size.x,
                height: shading_size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let denoise_receiver_history_output_view =
            denoise_receiver_history_output.create_view(&TextureViewDescriptor::default());
        let composite_anchor_textures: [Texture; 2] =
            std::array::from_fn(|_| make_texture(shading_size));
        let composite_anchor_views = std::array::from_fn(|i| {
            composite_anchor_textures[i].create_view(&TextureViewDescriptor::default())
        });
        let rtgi_reservoirs = std::array::from_fn(|_| {
            device.create_buffer(&BufferDescriptor {
                label: Some("split_rc_rtgi_reservoirs"),
                size: rtgi_reservoir_bytes(shading_size, settings),
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        });
        let rtgi_cache = rtgi_cache_bytes(settings).map(|size| {
            device.create_buffer(&BufferDescriptor {
                label: Some("split_rc_rtgi_cache"),
                size,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        });
        let sun_shadow_mask = make_texture(sun_shadow_mask_size(viewport_size));
        let sun_shadow_mask_view = sun_shadow_mask.create_view(&TextureViewDescriptor::default());
        Self {
            size,
            shading_size,
            viewport_size,
            buckets,
            caches,
            trace_indirect: device.create_buffer(&BufferDescriptor {
                label: Some("split_rc_trace_indirect"),
                // Receiver trace arguments, then deferred visibility arguments.
                size: 24,
                usage: BufferUsages::COPY_DST | BufferUsages::INDIRECT,
                mapped_at_creation: false,
            }),
            entries,
            history_entries,
            irradiance_probes,
            secondary,
            irradiance_textures: irradiance,
            irradiance_views,
            _background_textures: background_textures,
            background_views,
            _metadata: metadata,
            metadata_views,
            _denoise_receiver_history_output: denoise_receiver_history_output,
            denoise_receiver_history_output_view,
            _composite_anchor_textures: composite_anchor_textures,
            composite_anchor_views,
            rtgi_reservoirs,
            rtgi_cache,
            _sun_shadow_mask: sun_shadow_mask,
            sun_shadow_mask_view,
            uniforms: std::array::from_fn(|_| UniformBuffer::default()),
            clear_uniforms: std::array::from_fn(|_| UniformBuffer::default()),
            binding_cache: std::array::from_fn(|_| None),
            rendered_frame: Mutex::new(RenderedFrame::default()),
            cache_clock: std::time::Instant::now(),
            scene_history: Mutex::new(SceneHistorySnapshot::default()),
            directional_motion_epoch: AtomicU32::new(0),
            punctual_motion_epoch: AtomicU32::new(0),
            candidate_schedule: Mutex::new(CandidateSchedule::default()),
            viewport: UVec4::ZERO,
            frame: AtomicU32::new(0),
            history_valid: AtomicBool::new(false),
            screen_history_valid: AtomicBool::new(false),
            history_diagnostics_requested: AtomicBool::new(false),
            revision,
            settings,
            allocation_request: settings,
            mesh_revision: 0,
            geometry_revision: 0,
            render_layers: RenderLayers::default(),
            lights: None,
            light_data: Vec::new(),
            light_source: None,
            light_revision: 0,
            pending_lights: None,
            conventional_lights: 0,
            layer_selector: UVec4::ZERO,
            environment: None,
            environment_revision: 0,
            environment_info: UVec4::ZERO,
        }
    }

    fn preserve_world_cache_from(&mut self, previous: &Self, viewport: UVec4) -> bool {
        if !Self::can_preserve_world_cache(previous, self.size, viewport, self.settings) {
            return false;
        }

        self.caches = previous.caches.clone();
        self.entries = previous.entries.clone();
        self.history_entries = previous.history_entries.clone();
        self.irradiance_probes = previous.irradiance_probes.clone();
        self.rendered_frame = Mutex::new(
            *previous
                .rendered_frame
                .lock()
                .unwrap_or_else(|error| error.into_inner()),
        );
        self.cache_clock = previous.cache_clock;
        self.scene_history = Mutex::new(
            previous
                .scene_history
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .clone(),
        );
        self.candidate_schedule = Mutex::new(
            previous
                .candidate_schedule
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .clone(),
        );
        self.viewport = previous.viewport;
        self.frame
            .store(previous.frame.load(Ordering::Relaxed), Ordering::Relaxed);
        self.history_valid.store(
            previous.history_valid.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        self.directional_motion_epoch.store(
            previous.directional_motion_epoch.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        self.punctual_motion_epoch.store(
            previous.punctual_motion_epoch.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        self.history_diagnostics_requested.store(
            previous
                .history_diagnostics_requested
                .load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        self.revision = previous.revision;
        self.mesh_revision = previous.mesh_revision;
        self.geometry_revision = previous.geometry_revision;
        self.render_layers = previous.render_layers.clone();
        self.lights = previous.lights.clone();
        self.light_data = previous.light_data.clone();
        self.light_source = previous.light_source;
        self.light_revision = previous.light_revision;
        self.pending_lights = previous.pending_lights.clone();
        self.conventional_lights = previous.conventional_lights;
        self.layer_selector = previous.layer_selector;
        self.environment = previous.environment.clone();
        self.environment_revision = previous.environment_revision;
        self.environment_info = previous.environment_info;
        true
    }
}

// Bevy system parameters and query tuples describe the render-world inputs.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn prepare(
    mut commands: Commands,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    cache: Res<PipelineCache>,
    mut pipelines: ResMut<Pipelines>,
    shaders: Res<PipelineShaders>,
    mut last_scene_upload_report: Local<Option<std::time::Instant>>,
    image_resources: (
        Res<RenderAssets<GpuImage>>,
        Res<bevy::pbr::DfgLut>,
        Option<Res<hardware::HardwareTraversal>>,
    ),
    scene: Res<GiScene>,
    mut gpu_scene: Option<ResMut<GpuScene>>,
    environments: Res<crate::environment::EnvironmentTables>,
    budget: Res<SceneBudget>,
    diagnostics: Option<Res<DiagnosticsRecorder>>,
    mut residency: ParamSet<(
        Res<crate::receiver::ReceiverIdentityMemory>,
        Res<RasterResidencyGate>,
    )>,
    mut memory_high_water: ResMut<MemoryHighWater>,
    mut view_queries: ParamSet<(
        Query<
            (
                Option<&ViewBuffers>,
                Option<&crate::receiver::ReceiverTexture>,
            ),
            Or<(With<ViewBuffers>, With<crate::receiver::ReceiverTexture>)>,
        >,
        Query<(
            Entity,
            &ExtractedView,
            &ExtractedCamera,
            &ViewTarget,
            &ViewPrepassTextures,
            (
                &RadianceCascades,
                Option<&crate::RadianceCascadesRuntimeBudget>,
            ),
            Option<&RenderLayers>,
            Option<&crate::RadianceCascadesEnvironment>,
            &bevy::render::view::ViewDepthStencilTexture,
            Option<&crate::receiver::ReceiverTexture>,
            Option<&bevy::pbr::ScreenSpaceAmbientOcclusionResources>,
            Option<&bevy::pbr::SsaoPipelineId>,
            Option<&bevy::render::camera::TemporalJitter>,
            Option<&bevy::camera::MainPassResolutionOverride>,
            Option<&mut ViewBuffers>,
        )>,
    )>,
) {
    let (images, dfg_lut, hardware) = image_resources;
    // With ray queries every compute bind group carries the TLAS binding.
    let hardware_tlas = if pipelines.ray_query {
        let Some(hardware) = hardware.as_deref() else {
            return;
        };
        Some(hardware.tlas())
    } else {
        None
    };
    let Some(dfg_image) = images.get(&dfg_lut.texture) else {
        for (entity, ..) in &mut view_queries.p1() {
            commands
                .entity(entity)
                .remove::<(ViewBindings, GiPreparedView)>();
        }
        return;
    };
    let dfg_view = &dfg_image.texture_view;
    // Count every retained allocation, even when native view/prepass resources
    // are temporarily absent and that camera cannot be prepared this frame.
    let mut allocated_views = view_queries
        .p0()
        .iter()
        .map(|(buffers, receiver)| {
            receiver
                .map_or(0, |receiver| receiver.bytes())
                .saturating_add(buffers.map_or(0, |buffers| {
                    view_bytes(buffers.size, buffers.shading_size, buffers.settings)
                        .saturating_add(buffers.light_bytes())
                }))
        })
        .sum::<u64>()
        .saturating_add(residency.p0().bytes);
    let mut views = view_queries.p1();
    let required = views
        .iter()
        .fold(0u64, |mask, (_, _, _, _, _, (settings, _), ..)| {
            let settings = settings.sanitized();
            mask | if settings.intensity > 0.0 {
                required_pipelines(settings)
            } else {
                0
            }
        });
    // Hardware traversal pipelines are built only where the device has them.
    let wanted = views
        .iter()
        .fold(0u64, |mask, (_, _, _, _, _, (settings, _), ..)| {
            let settings = settings.sanitized();
            mask | if settings.intensity > 0.0 {
                wanted_pipelines(settings)
            } else {
                0
            }
        })
        & if pipelines.ray_query {
            u64::MAX
        } else {
            !HARDWARE_TRAVERSAL_PIPELINES
        };
    let mut job = 0;
    let mut cache_changed = false;
    while job < pipelines.jobs.len() {
        if !pipelines.jobs[job].1.is_finished() {
            job += 1;
            continue;
        }
        let (index, worker) = pipelines.jobs.swap_remove(job);
        let (result, elapsed) = worker.join().unwrap_or_else(|_| {
            (
                Err("pipeline worker panicked".into()),
                std::time::Duration::ZERO,
            )
        });
        {
            match result {
                Ok(pipeline) => {
                    info!(
                        entry = COMPUTE_ENTRIES[index],
                        ?elapsed,
                        "GI compute pipeline ready"
                    );
                    pipelines.compute[index] = Some(pipeline);
                    cache_changed = true;
                }
                Err(error) => {
                    shaders.status.record_failure(format!(
                        "GI compute pipeline {} failed: {error}",
                        COMPUTE_ENTRIES[index]
                    ));
                    error!(entry = COMPUTE_ENTRIES[index], ?elapsed, %error, "GI compute pipeline failed")
                }
            }
        }
    }
    if cache_changed {
        pipelines.driver_cache_dirty = true;
    }
    if pipelines
        .driver_cache_save
        .as_ref()
        .is_some_and(std::thread::JoinHandle::is_finished)
    {
        let worker = pipelines.driver_cache_save.take().unwrap();
        match worker.join() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                pipelines.driver_cache_path = None;
                warn!(%error, "Could not save GI device pipeline cache; disk saves disabled for this run");
            }
            Err(_) => {
                pipelines.driver_cache_path = None;
                warn!("GI device pipeline cache worker panicked; disk saves disabled for this run");
            }
        }
    }
    if pipelines.driver_cache_dirty
        && pipelines.driver_cache_save.is_none()
        && pipelines.jobs.is_empty()
        && wanted & !pipelines.started == 0
        && let (Some(cache), Some(path)) = (&pipelines.driver_cache, &pipelines.driver_cache_path)
    {
        let cache = cache.clone();
        let path = path.clone();
        match std::thread::Builder::new()
            .name("penumbra-pipeline-cache-save".into())
            .spawn(move || {
                let Some(data) = cache.get_data() else {
                    return Ok(());
                };
                let result = (|| -> std::io::Result<()> {
                    let parent = path.parent().unwrap_or_else(|| std::path::Path::new("."));
                    std::fs::create_dir_all(parent)?;
                    let mut temp = tempfile::NamedTempFile::new_in(parent)?;
                    std::io::Write::write_all(&mut temp, &data)?;
                    temp.persist(&path).map_err(|error| error.error)?;
                    Ok(())
                })();
                result.map_err(|error| error.to_string())
            }) {
            Ok(worker) => {
                pipelines.driver_cache_save = Some(worker);
                pipelines.driver_cache_dirty = false;
            }
            Err(error) => {
                pipelines.driver_cache_path = None;
                warn!(%error, "Could not start GI device pipeline cache save; disk saves disabled for this run");
            }
        }
    }
    // Larger batches can compete with Bevy's frame rendering and consume
    // substantial driver memory, especially for traversal kernels.
    let max_concurrent = shaders.max_concurrent.max(1);
    let mut compiling = pipelines.jobs.len();
    // Trace and gather dominate cold Vulkan compilation. Queue them before the
    // short kernels so they can compile while the rest of the frame pipeline is
    // being prepared.
    for i in [17, 18, 20, 7, 12, 16]
        .into_iter()
        .chain((0..COMPUTE_ENTRIES.len()).filter(|i| !matches!(i, 17 | 18 | 20 | 7 | 12 | 16)))
    {
        let entry = &COMPUTE_ENTRIES[i];
        if compiling >= max_concurrent || pipelines.driver_cache_save.is_some() {
            break;
        }
        if wanted & (1 << i) != 0
            && pipelines.compute[i].is_none()
            && pipelines.started & (1 << i) == 0
            && shaders.loaded.load(Ordering::Acquire) & (1 << i) != 0
        {
            let Some(source) = shaders.sources.lock().unwrap()[i].clone() else {
                continue;
            };
            let device = device.clone();
            let layouts = [cache
                .get_bind_group_layout(&pipelines.compute_layout)
                .clone()];
            let entry = *entry;
            let pipeline_cache = pipelines.driver_cache.clone();
            let spirv = pipelines.driver_spirv;
            let ray_query = pipelines.ray_query;
            pipelines.started |= 1 << i;
            let queued_at = std::time::Instant::now();
            match std::thread::Builder::new()
                .name(format!("penumbra-{entry}"))
                .spawn(move || {
                    info!(entry, queued_ms = ?queued_at.elapsed(), "GI pipeline worker started");
                    let start = std::time::Instant::now();
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        compute_pipeline(
                            device,
                            layouts,
                            source,
                            entry,
                            pipeline_cache,
                            spirv,
                            ray_query,
                        )
                    }))
                    .unwrap_or_else(|_| Err("wgpu rejected the compute pipeline".into()));
                    (result, start.elapsed())
                }) {
                Ok(worker) => pipelines.jobs.push((i, worker)),
                Err(error) => {
                    shaders.status.record_failure(format!(
                        "GI compute pipeline {} worker failed to start: {error}",
                        COMPUTE_ENTRIES[i]
                    ));
                    error!(entry, %error, "GI compute worker failed to start");
                }
            }
            compiling += 1;
        }
    }
    let shared_pipeline_jobs = if required != 0 {
        [
            (
                "composite",
                cache.get_render_pipeline_state(pipelines.composite),
            ),
            (
                "compositeAnchors",
                cache.get_compute_pipeline_state(pipelines.composite_anchors),
            ),
            (
                "refitTlas",
                cache.get_compute_pipeline_state(pipelines.refit),
            ),
            (
                "deformVertices",
                cache.get_compute_pipeline_state(pipelines.deform[0]),
            ),
            (
                "refitBlas",
                cache.get_compute_pipeline_state(pipelines.deform[1]),
            ),
        ]
        .into_iter()
        .filter(|(entry, state)| shared_pipeline_pending(state, entry, &shaders.status))
        .count()
    } else {
        0
    };
    let pipeline_jobs = pipelines.jobs.len() + shared_pipeline_jobs;
    shaders.status.set_pipeline_compile_jobs(pipeline_jobs);
    if !scene.ready
        || (scene.data.deformation.vertices > 0
            && pipelines
                .deform
                .iter()
                .any(|id| cache.get_compute_pipeline(*id).is_none()))
        || scene.data.textures.side > device.limits().max_texture_dimension_2d
        || scene.data.textures.pages.len() > device.limits().max_texture_array_layers as usize
    {
        if required != 0
            && last_scene_upload_report
                .is_none_or(|last| last.elapsed() >= std::time::Duration::from_secs(5))
        {
            info!(
                scene_ready = scene.ready,
                revision = scene.revision,
                instances = scene.data.instances.len(),
                triangles = scene.data.triangles.len(),
                texture_side = scene.data.textures.side,
                texture_pages = scene.data.textures.pages.len(),
                "GI waiting for prepared software scene or compatible scene textures"
            );
            *last_scene_upload_report = Some(std::time::Instant::now());
        }
        for (entity, .., buffers) in &mut views {
            if let Some(buffers) = buffers {
                buffers.screen_history_valid.store(false, Ordering::Relaxed);
            }
            commands
                .entity(entity)
                .remove::<(ViewBindings, GiPreparedView)>();
        }
        if scene.ready {
            warn_once!("GI texture atlas exceeds this device's texture dimensions/layer limits");
        }
        return;
    }
    // shortcut: ray-traced GI's five buffers share the cascades' layout, so
    // every method needs 13; give RTGI its own layout to get back to WebGPU's 8.
    if device.limits().max_storage_buffers_per_shader_stage < 13 {
        warn_once!("Radiance Cascades needs thirteen storage buffers per compute stage");
        return;
    }
    // The shared layouts contain thirteen storage buffers, fourteen sampled images
    // in group zero plus two in group one, and three storage images plus one.
    if device.limits().max_sampled_textures_per_shader_stage < 16 {
        warn_once!("Radiance Cascades needs sixteen sampled textures per compute stage");
        return;
    }
    if device.limits().max_storage_textures_per_shader_stage < 4 {
        warn_once!("Radiance Cascades needs four storage textures per compute stage");
        return;
    }
    let data = &scene.data;
    let geometry_buffer_bytes = data.geometry_words.len().max(1) as u64 * 16;
    let geometry_page_layout =
        if geometry_buffer_bytes > device.limits().max_storage_buffer_binding_size {
            geometry_pages(
                data.geometry_words.len(),
                data.deformation.vertices as usize,
                &device.limits(),
            )
        } else {
            None
        };
    let geometry_overlay_plan = if let Some(page) = geometry_page_layout {
        geometry_overlay_layout(&data.deformation, page.output_base_word)
    } else {
        Some(Arc::new(Vec::new()))
    };
    let geometry_overlay_bytes = geometry_overlay_plan.as_ref().map_or(u64::MAX, |words| {
        (words.len().max(1) as u64).saturating_mul(16)
    });
    let instance_base = data.acceleration_prefix.len() + data.blas_words.len();
    let acceleration_words = instance_base + data.instance_words.len();
    let material_start = data.instance_words[..data.instances.len() * 9]
        .as_chunks::<9>()
        .0
        .iter()
        .map(|instance| (instance[7].w as usize).saturating_sub(instance_base))
        .min()
        .unwrap_or(data.instance_words.len());
    let material_words = &data.instance_words[material_start..];
    let scene_buffer_bytes = acceleration_words.max(1) as u64 * 16;
    let material_tail_reusable = gpu_scene.as_ref().is_some_and(|old| {
        material_tail_layout_reusable(
            &old.material_words,
            old.dynamic_baseline
                .as_ref()
                .map(|baseline| baseline.instances.as_slice()),
            material_words,
            &data.instance_words[..material_start],
            material_start,
            data.instances.len(),
            instance_base,
        )
    });
    let scene_paged = scene_buffer_bytes > device.limits().max_storage_buffer_binding_size;
    let reusable_scene_range_map = if scene_paged {
        scene_page_ranges(data).map(|(_, range_map)| range_map)
    } else {
        None
    };
    let reuse_scene_pages = scene_paged
        && gpu_scene.as_ref().is_some_and(|old| {
            old.scene_paged
                && old.scene_texture.is_some()
                && old.scene_texture_view.is_some()
                && old.nodes.is_some()
                && old.acceleration_words == acceleration_words
                && old.prefix_words == data.acceleration_prefix.len()
                && old.topology_revision == scene.topology_revision
                && Arc::ptr_eq(&old.blas_words, &data.blas_words)
                && old.scene_overlay_config.x == instance_base as u32
                && old.scene_overlay_config.y == material_start as u32
                && reusable_scene_range_map
                    .as_ref()
                    .is_some_and(|range_map| range_map == &old.scene_range_map)
                && material_tail_reusable
                && (data.deformation.vertices == 0
                    || Arc::ptr_eq(&old.deformation_words, &data.deformation.words))
        });
    let scene_page_data = if scene_paged && !reuse_scene_pages {
        scene_page_data(data, material_start, &device.limits(), budget.live_bytes)
    } else {
        None
    };
    let scene_page_config = scene_page_data.as_ref().map_or_else(
        || {
            gpu_scene
                .as_ref()
                .filter(|_| reuse_scene_pages)
                .map_or(UVec4::ZERO, |old| old.scene_page_config)
        },
        |(layout, _, _)| layout.page_config,
    );
    let scene_overlay_config = scene_page_data.as_ref().map_or_else(
        || {
            gpu_scene
                .as_ref()
                .filter(|_| reuse_scene_pages)
                .map_or(UVec4::ZERO, |old| old.scene_overlay_config)
        },
        |(layout, _, _)| layout.overlay_config,
    );
    let scene_range_map = scene_page_data.as_ref().map_or_else(
        || {
            gpu_scene
                .as_ref()
                .filter(|_| reuse_scene_pages)
                .map_or_else(Vec::new, |old| old.scene_range_map.clone())
        },
        |(layout, _, _)| layout.range_map.clone(),
    );
    let scene_allocation_bytes = scene_page_data.as_ref().map_or_else(
        || {
            gpu_scene
                .as_ref()
                .filter(|_| reuse_scene_pages)
                .map_or(scene_buffer_bytes, |old| old.scene_bytes)
        },
        |(layout, _, overlay)| {
            layout
                .bytes
                .saturating_add(overlay.len().max(1) as u64 * 16)
        },
    );
    let scene_overlay_word_count = scene_page_data.as_ref().map_or_else(
        || {
            gpu_scene
                .as_ref()
                .filter(|_| reuse_scene_pages)
                .and_then(|old| old.nodes.as_ref())
                .map_or(acceleration_words.max(1), |buffer| {
                    (buffer.size() / 16) as usize
                })
        },
        |(_, _, overlay)| overlay.len().max(1),
    );
    let instance_overlay_base = scene_page_data.as_ref().map_or_else(
        || {
            gpu_scene
                .as_ref()
                .filter(|_| reuse_scene_pages)
                .map_or(instance_base, |old| old.scene_instance_overlay_base)
        },
        |(layout, _, _)| layout.instance_overlay_base,
    );
    let scene_paging_failed = scene_paged && !reuse_scene_pages && scene_page_data.is_none();
    if (geometry_buffer_bytes > device.limits().max_storage_buffer_binding_size
        && geometry_page_layout.is_none())
        || geometry_overlay_bytes > device.limits().max_storage_buffer_binding_size
        || scene.data.deformation.words.len() as u64 * 16
            > device.limits().max_storage_buffer_binding_size
        || scene.data.deformation.poses.len() as u64 * 16
            > device.limits().max_storage_buffer_binding_size
        || scene.data.deformation.refit_nodes.len() as u64 * 4
            > device.limits().max_storage_buffer_binding_size
        || scene_paging_failed
    {
        warn_once!("GI scene exceeds this device's storage buffer limit");
        for (entity, .., buffers) in &mut views {
            if let Some(buffers) = buffers {
                buffers.screen_history_valid.store(false, Ordering::Relaxed);
            }
            commands
                .entity(entity)
                .remove::<(ViewBindings, GiPreparedView)>();
        }
        return;
    }
    // ponytail: keep the per-job descriptor map in one storage binding; page it if
    // exceptionally large job counts exceed the device's binding limit.
    let geometry_overlay_words = geometry_overlay_plan.expect("validated deformation overlay");
    let mut new_gpu_scene = None;
    let (refit_order, refit_dispatch_values) = refit_dispatches(&data.nodes);
    let uniform_alignment = u64::from(device.limits().min_uniform_buffer_offset_alignment.max(16));
    let uniform_stride = 48u64.div_ceil(uniform_alignment) * uniform_alignment;
    let mut metadata_upload_bytes = 0usize;
    if gpu_scene.as_ref().is_none_or(|old| {
        old.topology_revision != scene.topology_revision
            || old.refit_nodes.is_none()
            || old.scene_page_config != scene_page_config
            || old.scene_overlay_config != scene_overlay_config
            || old.scene_range_map != scene_range_map
    }) {
        metadata_upload_bytes = metadata_upload_bytes.saturating_add(
            (refit_dispatch_values.len() as u64)
                .saturating_mul(uniform_stride)
                .min(usize::MAX as u64) as usize,
        );
    }
    if data.deformation.vertices > 0
        && gpu_scene.as_ref().is_none_or(|old| {
            !Arc::ptr_eq(&old.deformation_words, &data.deformation.words)
                || old.deform_order.is_none()
                || old.deform_order_revision != Some((scene.topology_revision, scene.mesh_revision))
                || old.scene_page_config != scene_page_config
                || old.scene_overlay_config != scene_overlay_config
                || old.scene_range_map != scene_range_map
        })
    {
        metadata_upload_bytes = metadata_upload_bytes.saturating_add(
            ((1 + data.deformation.refit_levels.len()) as u64)
                .saturating_mul(uniform_stride)
                .min(usize::MAX as u64) as usize,
        );
    }
    let frame_upload_budget = budget.upload_bytes.max(16);
    if metadata_upload_bytes > frame_upload_budget {
        warn_once!("GI upload budget is too small for scene refit parameter tables");
        for (entity, .., buffers) in &mut views {
            if let Some(buffers) = buffers {
                buffers.screen_history_valid.store(false, Ordering::Relaxed);
            }
            commands
                .entity(entity)
                .remove::<(ViewBindings, GiPreparedView)>();
        }
        return;
    }
    let mut upload_remaining = frame_upload_budget;
    let replace_acceleration = gpu_scene.as_ref().is_none_or(|old| {
        old.nodes.is_none()
            || old.acceleration_words != acceleration_words
            || old.prefix_words != data.acceleration_prefix.len()
            || old.dynamic_baseline.as_ref().is_none_or(|baseline| {
                baseline.instances.len() != material_start
                    || baseline.instance_overlay_base != instance_overlay_base
            })
            || old.scene_paged != scene_paged
            || old.scene_page_config != scene_page_config
            || old.scene_overlay_config != scene_overlay_config
            || old.scene_range_map != scene_range_map
            || !Arc::ptr_eq(&old.blas_words, &data.blas_words)
            || !material_tail_reusable
            || old.topology_revision != scene.topology_revision
    });
    let atlas_bytes = |atlas: &crate::material::TextureAtlas| {
        u64::from(atlas.side.max(1)).pow(2) * atlas.pages.len().max(1) as u64 * 16
    };
    let order_bytes = |count: usize| count.div_ceil(4).max(1) as u64 * 16;
    let geometry_allocation_bytes = geometry_page_layout
        .map_or(geometry_buffer_bytes, |page| {
            page.bytes.saturating_add(geometry_overlay_bytes)
        })
        .saturating_add(32); // sampled and storage placeholder views
    let mut live = geometry_allocation_bytes
        .saturating_add(scene_allocation_bytes)
        .saturating_add(16); // permanently retained scene-page placeholder view
    live = live
        .saturating_add(order_bytes(refit_order.len()))
        .saturating_add(atlas_bytes(&data.textures))
        .saturating_add(16);
    if data.deformation.vertices > 0 {
        live = live
            .saturating_add(
                (data.deformation.words.len().max(1) + data.deformation.poses.len().max(1)) as u64
                    * 16,
            )
            .saturating_add(order_bytes(data.deformation.refit_nodes.len()));
    }
    let candidate_bytes = live;
    if let Some(old) = gpu_scene.as_ref() {
        let deform_order_revision = (scene.topology_revision, scene.mesh_revision);
        if old.mesh_revision != scene.mesh_revision {
            live = live.saturating_add(old.geometry_bytes);
        }
        if replace_acceleration {
            live = live.saturating_add(old.scene_bytes);
        }
        let deformation_streams_changed =
            !Arc::ptr_eq(&old.deformation_words, &data.deformation.words)
                || old.deformation.is_none();
        if deformation_streams_changed {
            live = live.saturating_add(old.deformation.as_ref().map_or(0, |buffer| buffer.size()));
        }
        if deformation_streams_changed
            || old.deform_order.is_none()
            || old.deform_order_revision != Some(deform_order_revision)
        {
            live = live.saturating_add(old.deform_order.as_ref().map_or(0, |buffer| buffer.size()));
        }
        if old.pose_words != data.deformation.poses {
            live = live.saturating_add(old.poses.as_ref().map_or(0, |buffer| buffer.size()));
        }
        if old.topology_revision != scene.topology_revision {
            live = live.saturating_add(old.refit_nodes.as_ref().map_or(0, |buffer| buffer.size()));
        }
        if !Arc::ptr_eq(&old.textures, &data.textures) {
            live = live.saturating_add(atlas_bytes(&old.textures));
        }
        if let Some(pending) = &old.pending_geometry
            && (pending.revision != scene.mesh_revision
                || (old.mesh_revision == scene.mesh_revision && old.geometry.is_some()))
        {
            live = live.saturating_add(pending.allocated_bytes);
        }
        if let Some(pending) = &old.pending_atlas
            && (!Arc::ptr_eq(&pending.atlas, &data.textures)
                || (Arc::ptr_eq(&old.textures, &data.textures) && old.atlas.is_some()))
        {
            live = live.saturating_add(atlas_bytes(&pending.atlas));
        }
        if let Some(pending) = &old.pending_acceleration
            && (!replace_acceleration
                || pending.topology_revision != scene.topology_revision
                || pending.prefix_words != data.acceleration_prefix.len()
                || pending.words.len() != scene_overlay_word_count
                || pending.paged != scene_paged
                || pending.page_config != scene_page_config
                || pending.overlay_config != scene_overlay_config
                || pending.range_map != scene_range_map
                || !Arc::ptr_eq(&pending.blas_words, &data.blas_words)
                || pending.material_words != material_words)
        {
            live = live.saturating_add(pending.allocated_bytes);
        }
        if let Some(pending) = &old.pending_streams
            && (!Arc::ptr_eq(&pending.words, &data.deformation.words)
                || (Arc::ptr_eq(&old.deformation_words, &data.deformation.words)
                    && old.deformation.is_some()))
        {
            live = live.saturating_add(pending.buffer.size());
        }
        if let Some(pending) = &old.pending_deform_order
            && old.pending_deform_order_revision != Some(deform_order_revision)
        {
            live = live.saturating_add(pending.buffer.size());
        }
        if let Some(pending) = &old.pending_poses
            && (pending.words != data.deformation.poses
                || (old.pose_words == data.deformation.poses && old.poses.is_some()))
        {
            live = live.saturating_add(pending.buffer.size());
        }
        if let Some(pending) = &old.pending_refit_order
            && (old.pending_refit_revision != Some(scene.topology_revision)
                || (old.topology_revision == scene.topology_revision && old.refit_nodes.is_some()))
        {
            live = live.saturating_add(pending.buffer.size());
        }
    }
    if live > budget.live_bytes {
        warn_once!(
            live_bytes = live,
            candidate_bytes,
            budget_bytes = budget.live_bytes,
            "GI shared scene/replacement exceeds its GPU memory budget"
        );
        if candidate_bytes > budget.live_bytes {
            let failure = format!(
                "GI scene revision {} requires {candidate_bytes} standalone GPU bytes, above the {} byte scene limit",
                scene.revision, budget.live_bytes
            );
            shaders.status.record_failure(failure.clone());
            error!(%failure, "GI scene snapshot cannot fit its GPU memory budget");
        }
        if let Some(old) = gpu_scene.as_mut() {
            // An obsolete snapshot cannot render this revision and must not block a replacement that fits alone.
            if discard_obsolete_scene(
                live,
                candidate_bytes,
                budget.live_bytes,
                old.revision != scene.revision,
            ) {
                **old = GpuScene::default();
            }
            old.pending_geometry = None;
            old.pending_atlas = None;
            old.pending_acceleration = None;
            old.pending_streams = None;
            old.pending_deform_order = None;
            old.pending_deform_order_revision = None;
            old.pending_refit_order = None;
            old.pending_refit_revision = None;
            old.pending_poses = None;
            old.pending_deformation = false;
            old.pending_refit = false;
        }
        for (entity, .., buffers) in &mut views {
            if let Some(buffers) = buffers {
                buffers.screen_history_valid.store(false, Ordering::Relaxed);
            }
            commands
                .entity(entity)
                .remove::<(ViewBindings, GiPreparedView)>();
        }
        return;
    }
    // Keep the published geometry intact until its replacement is complete.
    // GI bindings are removed while raster geometry can already be newer.
    let mut staged = gpu_scene
        .as_mut()
        .map_or_else(GpuScene::default, |state| std::mem::take(&mut **state));
    // Cancel superseded work even when the desired atlas is already published.
    if staged
        .pending_atlas
        .as_ref()
        .is_some_and(|pending| !Arc::ptr_eq(&pending.atlas, &data.textures))
    {
        staged.pending_atlas = None;
    }
    if staged.atlas.is_some() && Arc::ptr_eq(&staged.textures, &data.textures) {
        staged.pending_atlas = None;
    }
    if staged.atlas.is_none() || !Arc::ptr_eq(&staged.textures, &data.textures) {
        if staged
            .pending_atlas
            .as_ref()
            .is_none_or(|pending| !Arc::ptr_eq(&pending.atlas, &data.textures))
        {
            staged.pending_atlas = None;
            let (texture, view) = allocate_material_atlas(&device, &data.textures);
            staged.pending_atlas = Some(PendingAtlas {
                texture,
                view,
                atlas: data.textures.clone(),
                layer: 0,
                cursor: 0,
            });
        }
        let pending = staged.pending_atlas.as_mut().expect("atlas staged");
        let side = pending.atlas.side.max(1);
        while pending.layer < pending.atlas.pages.len() {
            let page = &pending.atlas.pages[pending.layer];
            upload_texture_pixels(
                &queue,
                &pending.texture,
                side,
                pending.layer as u32,
                page,
                &mut pending.cursor,
                &mut upload_remaining,
            );
            if pending.cursor < page.len() {
                break;
            }
            pending.cursor = 0;
            pending.layer += 1;
        }
    }
    if staged.mesh_revision != scene.mesh_revision || staged.geometry.is_none() {
        if staged
            .pending_geometry
            .as_ref()
            .is_none_or(|pending| pending.revision != scene.mesh_revision)
        {
            staged.pending_geometry = None;
            staged.pending_geometry = Some(PendingGeometry::new(
                &device,
                scene.data.geometry_words.clone(),
                scene.mesh_revision,
                geometry_page_layout,
                geometry_overlay_words.clone(),
            ));
        }
        let pending = staged
            .pending_geometry
            .as_mut()
            .expect("geometry upload staged");
        pending.upload(&queue, &mut upload_remaining);
    } else {
        staged.pending_geometry = None;
    }
    // Snapshot immutable topology/material data. Motion does not restart this
    // stream; current instance transforms and TLAS bounds are patched at commit.
    if staged.pending_acceleration.as_ref().is_some_and(|pending| {
        pending.topology_revision != scene.topology_revision
            || pending.prefix_words != data.acceleration_prefix.len()
            || pending.words.len() != scene_overlay_word_count
            || pending.paged != scene_paged
            || pending.page_config != scene_page_config
            || pending.overlay_config != scene_overlay_config
            || pending.range_map != scene_range_map
            || !Arc::ptr_eq(&pending.blas_words, &data.blas_words)
            || pending.material_words != material_words
    }) {
        staged.pending_acceleration = None;
        staged.pending_dynamic = None;
        staged.pending_refit_order = None;
        staged.pending_refit_revision = None;
    }
    if !replace_acceleration {
        staged.pending_acceleration = None;
        staged.pending_dynamic = None;
    }
    if replace_acceleration || staged.pending_acceleration.is_some() {
        if staged.pending_acceleration.is_none() {
            staged.pending_dynamic = None;
            let (words, page_words, layout) =
                if let Some((layout, page_words, overlay)) = scene_page_data.clone() {
                    (overlay, page_words, Some(layout))
                } else {
                    let words = data
                        .acceleration_prefix
                        .iter()
                        .chain(data.blas_words.iter())
                        .chain(data.instance_words.iter())
                        .copied()
                        .collect();
                    (words, ScenePageUpload::empty(), None)
                };
            staged.pending_acceleration = Some(PendingAcceleration::new(
                &device,
                words,
                page_words,
                layout,
                data.blas_words.clone(),
                material_words.to_vec(),
                scene.topology_revision,
                data.acceleration_prefix.len(),
                instance_base,
                material_start,
                scene.geometry_revision,
            ));
        }
        let pending = staged
            .pending_acceleration
            .as_mut()
            .expect("acceleration staged");
        pending.upload(&queue, &mut upload_remaining);
    }
    let dynamic_baseline = staged
        .pending_acceleration
        .as_ref()
        .map(|pending| &pending.dynamic_baseline)
        .or(staged.dynamic_baseline.as_ref());
    if let Some(baseline) = dynamic_baseline {
        if staged
            .pending_dynamic
            .as_ref()
            .is_none_or(|pending| !pending.matches_layout(baseline))
        {
            staged.pending_dynamic = Some(PendingDynamic {
                revision: scene.geometry_revision,
                prefix: data.acceleration_prefix.clone(),
                instances: data.instance_words.clone(),
                instance_words: material_start,
                instance_overlay_base: baseline.instance_overlay_base,
            });
        } else if let Some(pending) = staged.pending_dynamic.as_mut() {
            pending.retarget(
                scene.geometry_revision,
                data.acceleration_prefix.clone(),
                data.instance_words.clone(),
                material_start,
                baseline.instance_overlay_base,
            );
        }
    } else {
        staged.pending_dynamic = None;
    }
    if data.deformation.vertices > 0 {
        let streams_changed = !Arc::ptr_eq(&staged.deformation_words, &data.deformation.words);
        if !streams_changed {
            staged.pending_streams = None;
        } else if staged
            .pending_streams
            .as_ref()
            .is_none_or(|pending| !Arc::ptr_eq(&pending.words, &data.deformation.words))
        {
            staged.pending_streams = Some(PendingWords::new(
                &device,
                "rc_deformation_streams",
                data.deformation.words.clone(),
            ));
        }
        let deform_order_revision = (scene.topology_revision, scene.mesh_revision);
        let deform_order_changed =
            streams_changed || staged.deform_order_revision != Some(deform_order_revision);
        if deform_order_changed
            && staged.pending_deform_order_revision != Some(deform_order_revision)
        {
            staged.pending_deform_order = Some(PendingWords::new(
                &device,
                "rc_deform_order",
                order_words(&data.deformation.refit_nodes),
            ));
            staged.pending_deform_order_revision = Some(deform_order_revision);
        } else if !deform_order_changed {
            staged.pending_deform_order = None;
            staged.pending_deform_order_revision = None;
        }
        if staged.pose_words == data.deformation.poses {
            staged.pending_poses = None;
        } else if staged
            .pending_poses
            .as_ref()
            .is_none_or(|pending| pending.words != data.deformation.poses)
        {
            let reusable = staged.pending_poses.as_ref().is_some_and(|pending| {
                pending.buffer.size() >= (data.deformation.poses.len().max(1) * 16) as u64
            });
            if reusable {
                let pending = staged.pending_poses.as_mut().expect("pending pose buffer");
                pending.words = data.deformation.poses.clone();
                pending.cursor = 0;
            } else {
                staged.pending_poses = Some(PendingWords::new(
                    &device,
                    "rc_deformation_poses",
                    data.deformation.poses.clone(),
                ));
            }
        }
    } else {
        staged.pending_streams = None;
        staged.pending_deform_order = None;
        staged.pending_deform_order_revision = None;
        staged.pending_poses = None;
    }
    if staged.refit_nodes.is_some() && staged.topology_revision == scene.topology_revision {
        staged.pending_refit_order = None;
        staged.pending_refit_revision = None;
    } else if staged.pending_refit_order.is_none()
        || staged.pending_refit_revision != Some(scene.topology_revision)
    {
        staged.pending_refit_order = Some(PendingWords::new(
            &device,
            "rc_tlas_refit_order",
            order_words(&refit_order),
        ));
        staged.pending_refit_revision = Some(scene.topology_revision);
    }
    let mut pending_words = false;
    for pending in [
        &mut staged.pending_streams,
        &mut staged.pending_deform_order,
        &mut staged.pending_refit_order,
        &mut staged.pending_poses,
    ]
    .into_iter()
    .flatten()
    {
        pending_words |= !pending.upload(&queue, &mut upload_remaining);
    }
    let acceleration_ready = staged.pending_acceleration.as_ref().is_none_or(|pending| {
        pending.cursor == pending.words.len()
            && scene_page_upload_complete(pending.paged, &pending.page_words)
    });
    if acceleration_ready {
        let uploaded_revision = if let Some(dynamic) = staged.pending_dynamic.as_mut() {
            if let Some(acceleration) = staged.pending_acceleration.as_mut() {
                dynamic
                    .upload(
                        &queue,
                        &acceleration.buffer,
                        &mut acceleration.dynamic_baseline,
                        &mut upload_remaining,
                    )
                    .then_some(dynamic.revision)
            } else if let (Some(buffer), Some(baseline)) =
                (staged.nodes.as_ref(), staged.dynamic_baseline.as_mut())
            {
                dynamic
                    .upload(&queue, buffer, baseline, &mut upload_remaining)
                    .then_some(dynamic.revision)
            } else {
                None
            }
        } else {
            staged
                .pending_acceleration
                .as_ref()
                .filter(|pending| pending.dynamic_revision == scene.geometry_revision)
                .map(|_| scene.geometry_revision)
        };
        if let Some(revision) = uploaded_revision {
            if let Some(pending) = staged.pending_acceleration.as_mut() {
                pending.dynamic_revision = revision;
            } else {
                staged.dynamic_revision = revision;
            }
            staged.pending_dynamic = None;
        }
    }
    if replace_acceleration {
        staged.abandon_pending_material_tail();
    } else if staged.material_words != material_words || staged.pending_material_tail.is_some() {
        let mut pending = staged
            .pending_material_tail
            .take()
            .or_else(|| PendingMaterialTail::new(staged.material_words.clone(), material_words));
        if let Some(mut pending) = pending.take() {
            let destination = if staged.scene_paged {
                staged
                    .scene_texture
                    .as_ref()
                    .map(|texture| MaterialTailDestination::Texture {
                        texture,
                        side: staged.scene_page_config.y,
                        word_base: staged.scene_overlay_config.z as usize,
                    })
            } else {
                staged
                    .nodes
                    .as_ref()
                    .map(|buffer| MaterialTailDestination::Buffer {
                        buffer,
                        word_base: staged
                            .prefix_words
                            .saturating_add(staged.blas_words.len())
                            .saturating_add(material_start),
                    })
            };
            let complete = destination.is_some_and(|destination| {
                pending.upload(&queue, destination, material_words, &mut upload_remaining)
            });
            if complete {
                staged.material_words = pending.applied_words;
            } else {
                staged.pending_material_tail = Some(pending);
            }
        }
    }
    if pending_words
        || staged.pending_geometry.as_ref().is_some_and(|pending| {
            pending.cursor < pending.words.len()
                || pending.overlay_cursor < pending.overlay_words.len()
        })
        || staged
            .pending_atlas
            .as_ref()
            .is_some_and(|pending| pending.layer < pending.atlas.pages.len())
        || staged.pending_acceleration.as_ref().is_some_and(|pending| {
            pending.cursor < pending.words.len()
                || !scene_page_upload_complete(pending.paged, &pending.page_words)
        })
        || staged.pending_dynamic.is_some()
        || staged.pending_material_tail.is_some()
    {
        if last_scene_upload_report
            .is_none_or(|last| last.elapsed() >= std::time::Duration::from_secs(5))
        {
            let remaining_geometry_bytes = staged.pending_geometry.as_ref().map_or(0, |pending| {
                pending
                    .words
                    .len()
                    .saturating_sub(pending.cursor)
                    .saturating_add(
                        pending
                            .overlay_words
                            .len()
                            .saturating_sub(pending.overlay_cursor),
                    ) as u64
                    * 16
            });
            let remaining_acceleration_bytes =
                staged.pending_acceleration.as_ref().map_or(0, |pending| {
                    let remaining_page_words = if pending.paged {
                        pending
                            .page_words
                            .word_count
                            .saturating_sub(pending.page_words.cursor)
                    } else {
                        0
                    };
                    pending
                        .words
                        .len()
                        .saturating_sub(pending.cursor)
                        .saturating_add(remaining_page_words) as u64
                        * 16
                });
            info!(
                pending_words,
                remaining_geometry_bytes,
                remaining_acceleration_bytes,
                pending_atlas_layer = staged
                    .pending_atlas
                    .as_ref()
                    .map_or(0, |pending| pending.layer),
                pending_atlas_pages = staged
                    .pending_atlas
                    .as_ref()
                    .map_or(0, |pending| pending.atlas.pages.len()),
                pending_dynamic = staged.pending_dynamic.is_some(),
                pending_material_tail = staged.pending_material_tail.is_some(),
                upload_budget_bytes = frame_upload_budget,
                upload_bytes_left = upload_remaining,
                live_scene_bytes = live,
                scene_budget_bytes = budget.live_bytes,
                "GI scene GPU upload pending"
            );
            *last_scene_upload_report = Some(std::time::Instant::now());
        }
        staged.pending_deformation = false;
        staged.pending_refit = false;
        for (entity, .., buffers) in &mut views {
            if let Some(buffers) = buffers {
                buffers.screen_history_valid.store(false, Ordering::Relaxed);
            }
            commands
                .entity(entity)
                .remove::<(ViewBindings, GiPreparedView)>();
        }
        if let Some(state) = gpu_scene.as_mut() {
            **state = staged;
        } else {
            commands.insert_resource(staged);
        }
        return;
    }
    if upload_remaining < metadata_upload_bytes {
        for (entity, .., buffers) in &mut views {
            if let Some(buffers) = buffers {
                buffers.screen_history_valid.store(false, Ordering::Relaxed);
            }
            commands
                .entity(entity)
                .remove::<(ViewBindings, GiPreparedView)>();
        }
        if let Some(state) = gpu_scene.as_mut() {
            **state = staged;
        } else {
            commands.insert_resource(staged);
        }
        return;
    }
    upload_remaining -= metadata_upload_bytes;
    if let Some(state) = gpu_scene.as_mut() {
        **state = staged;
    } else {
        new_gpu_scene = Some(staged);
    }
    if gpu_scene
        .as_ref()
        .is_none_or(|s| s.revision != scene.revision)
    {
        let upload_meshes = gpu_scene
            .as_ref()
            .is_none_or(|s| s.mesh_revision != scene.mesh_revision || s.geometry.is_none());
        let mut next = gpu_scene.as_mut().map_or_else(
            || new_gpu_scene.take().unwrap_or_default(),
            |s| std::mem::take(&mut **s),
        );
        if next.sampled_fallback_view.is_none() {
            let texture = device.create_texture(&TextureDescriptor {
                label: Some("gi_sampled_fallback"),
                size: Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba32Float,
                usage: TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            next.sampled_fallback_view =
                Some(texture.create_view(&TextureViewDescriptor::default()));
            next.sampled_fallback = Some(texture);
        }
        if next.geometry_page_placeholder_view.is_none() {
            let texture = device.create_texture(&TextureDescriptor {
                label: Some("gi_geometry_page_placeholder"),
                size: Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba32Uint,
                usage: TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            next.geometry_page_placeholder_view =
                Some(texture.create_view(&TextureViewDescriptor {
                    dimension: Some(TextureViewDimension::D2Array),
                    ..default()
                }));
            next.geometry_page_placeholder = Some(texture);
        }
        if next.geometry_output_placeholder_view.is_none() {
            let texture = device.create_texture(&TextureDescriptor {
                label: Some("gi_geometry_output_placeholder"),
                size: Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba32Uint,
                usage: TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            next.geometry_output_placeholder_view =
                Some(texture.create_view(&TextureViewDescriptor {
                    dimension: Some(TextureViewDimension::D2Array),
                    ..default()
                }));
            next.geometry_output_placeholder = Some(texture);
        }
        if next.scene_page_placeholder_view.is_none() {
            let texture = device.create_texture(&TextureDescriptor {
                label: Some("gi_scene_page_placeholder"),
                size: Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba32Uint,
                usage: TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            next.scene_page_placeholder_view = Some(texture.create_view(&TextureViewDescriptor {
                dimension: Some(TextureViewDimension::D2Array),
                ..default()
            }));
            next.scene_page_placeholder = Some(texture);
        }
        if next.atlas_view.is_none() || !Arc::ptr_eq(&next.textures, &scene.data.textures) {
            let pending = next.pending_atlas.take().expect("completed staged atlas");
            next.atlas = Some(pending.texture);
            next.atlas_view = Some(pending.view);
            next.textures = pending.atlas;
        }
        if upload_meshes {
            let pending = next
                .pending_geometry
                .take()
                .expect("completed staged geometry");
            next.geometry = Some(pending.buffer);
            next.geometry_pages = pending.texture;
            next.geometry_pages_view = pending.view;
            next.geometry_pages_input_view = pending.input_view;
            next.geometry_pages_output_view = pending.output_view;
            next.geometry_paged = pending.paged;
            next.geometry_page_side = pending.page_side;
            next.geometry_bytes = pending.allocated_bytes;
            next.mesh_revision = scene.mesh_revision;
        }
        let data = &scene.data;
        let streams_changed = !Arc::ptr_eq(&next.deformation_words, &data.deformation.words)
            || next.deformation.is_none();
        let deform_order_revision = (scene.topology_revision, scene.mesh_revision);
        let deform_order_changed = streams_changed
            || next.deform_order.is_none()
            || next.deform_order_revision != Some(deform_order_revision);
        if data.deformation.vertices > 0
            && (upload_meshes
                || next.pose_words != data.deformation.poses
                || streams_changed
                || deform_order_changed
                || next.scene_page_config != scene_page_config
                || next.scene_overlay_config != scene_overlay_config
                || next.scene_range_map != scene_range_map)
        {
            if streams_changed {
                next.deformation = Some(
                    next.pending_streams
                        .take()
                        .expect("completed deformation streams")
                        .buffer,
                );
                next.deformation_words = data.deformation.words.clone();
            }
            if deform_order_changed {
                next.deform_order = Some(
                    next.pending_deform_order
                        .take()
                        .expect("completed deformation order")
                        .buffer,
                );
                next.deform_order_revision = Some(deform_order_revision);
                next.pending_deform_order_revision = None;
            }
            if streams_changed
                || deform_order_changed
                || next.scene_page_config != scene_page_config
                || next.scene_overlay_config != scene_overlay_config
                || next.scene_range_map != scene_range_map
            {
                next.deform_params = DynamicUniformBuffer::new_with_alignment(
                    device.limits().min_uniform_buffer_offset_alignment as u64,
                );
                next.deform_vertex_offset = next.deform_params.push(&SceneDispatch {
                    dispatch: UVec4::new(
                        0,
                        data.deformation.vertices,
                        if next.geometry_paged { 2 } else { 0 },
                        next.geometry_page_side,
                    ),
                    scene_pages: scene_page_config,
                    scene_overlay: scene_overlay_config,
                });
                next.deform_offsets = data
                    .deformation
                    .refit_levels
                    .iter()
                    .map(|&(start, count)| {
                        (
                            next.deform_params.push(&SceneDispatch {
                                dispatch: UVec4::new(
                                    start,
                                    count,
                                    if next.geometry_paged { 2 } else { 0 },
                                    next.geometry_page_side,
                                ),
                                scene_pages: scene_page_config,
                                scene_overlay: scene_overlay_config,
                            }),
                            count,
                        )
                    })
                    .collect();
                next.deform_params.write_buffer(&device, &queue);
            }
            if let Some(pending) = next.pending_poses.take() {
                next.poses = Some(pending.buffer);
                next.pose_words = pending.words;
            }
            next.pending_deformation = true;
        } else {
            next.pending_deformation = false;
        }
        let instance_words = &data.instance_words;
        let geometry_changed = next.geometry_revision != scene.geometry_revision;
        let scene_access_changed = next.scene_page_config != scene_page_config
            || next.scene_overlay_config != scene_overlay_config
            || next.scene_range_map != scene_range_map;
        let topology_changed = next.topology_revision != scene.topology_revision
            || next.refit_nodes.is_none()
            || scene_access_changed;
        if topology_changed {
            if let Some(pending) = next.pending_refit_order.take() {
                next.refit_nodes = Some(pending.buffer);
            }
            next.refit_params = DynamicUniformBuffer::new_with_alignment(
                device.limits().min_uniform_buffer_offset_alignment as u64,
            );
            next.refit_offsets = refit_dispatch_values
                .iter()
                .map(|&dispatch| {
                    (
                        next.refit_params.push(&SceneDispatch {
                            dispatch,
                            scene_pages: scene_page_config,
                            scene_overlay: scene_overlay_config,
                        }),
                        dispatch.y,
                    )
                })
                .collect();
            next.refit_params.write_buffer(&device, &queue);
            next.pending_refit_revision = None;
            next.topology_revision = scene.topology_revision;
        }
        let words = data.acceleration_prefix.len() + data.blas_words.len() + instance_words.len();
        let full_upload = next.pending_acceleration.is_some();
        if full_upload {
            let pending = next
                .pending_acceleration
                .take()
                .expect("completed acceleration");
            next.dynamic_revision = pending.dynamic_revision;
            next.dynamic_baseline = Some(pending.dynamic_baseline);
            next.nodes = Some(pending.buffer);
            next.scene_texture = pending.texture;
            next.scene_texture_view = pending.view;
            next.scene_paged = pending.paged;
            next.scene_page_config = pending.page_config;
            next.scene_overlay_config = pending.overlay_config;
            next.scene_range_map = pending.range_map;
            next.scene_instance_overlay_base = pending.instance_overlay_base;
            next.scene_bytes = pending.allocated_bytes;
            next.acceleration_words = words;
            next.prefix_words = data.acceleration_prefix.len();
            next.blas_words = data.blas_words.clone();
            next.material_words = pending.material_words;
        }
        if geometry_changed {
            next.pending_refit = true;
        }
        next.geometry_revision = scene.geometry_revision;
        next.revision = scene.revision;
        next.instance_residency = scene.data.instance_residency.clone();
        if let Some(gpu_scene) = gpu_scene.as_mut() {
            **gpu_scene = next;
        } else {
            new_gpu_scene = Some(next);
        }
    }
    let Some(gpu_scene_ref) = new_gpu_scene.as_ref().or(gpu_scene.as_deref()) else {
        return;
    };
    let (Some(triangles), Some(nodes)) = (
        gpu_scene_ref
            .geometry
            .as_ref()
            .map(|buffer| buffer.as_entire_binding()),
        gpu_scene_ref
            .nodes
            .as_ref()
            .map(|buffer| buffer.as_entire_binding()),
    ) else {
        return;
    };
    let Some(atlas_view) = gpu_scene_ref.atlas_view.as_ref() else {
        return;
    };
    let Some(geometry_page_view) = (if gpu_scene_ref.geometry_paged {
        gpu_scene_ref.geometry_pages_view.as_ref()
    } else {
        gpu_scene_ref.geometry_page_placeholder_view.as_ref()
    }) else {
        return;
    };
    let Some(scene_page_view) = gpu_scene_ref.scene_page_view() else {
        return;
    };
    let Some(fallback_view) = gpu_scene_ref.sampled_fallback_view.as_ref() else {
        return;
    };
    for (
        entity,
        view,
        camera,
        target,
        prepass,
        (settings, runtime_budget),
        render_layers,
        environment,
        view_depth,
        receiver,
        ssao,
        ssao_pipeline,
        temporal_jitter,
        resolution_override,
        existing,
    ) in &mut views
    {
        if !residency.p1().gated.is_empty() {
            if let Some(buffers) = existing.as_ref() {
                buffers.screen_history_valid.store(false, Ordering::Relaxed);
            }
            commands
                .entity(entity)
                .remove::<(ViewBindings, GiPreparedView)>();
            continue;
        }
        let requested_settings = settings.sanitized();
        let mut settings = requested_settings;
        let view_budget_bytes = budget.view_bytes;
        let allocation_request = settings;
        // Native SSAO textures/dispatch use viewport-local dimensions. Its
        // depth/normal sampling is only correct for a full target at origin zero.
        let full_target = view.viewport.xy() == UVec2::ZERO
            && view.viewport.zw()
                == UVec2::new(view_depth.texture().width(), view_depth.texture().height());
        let ssao_valid = full_target
            && ssao.is_some()
            && ssao_pipeline
                .is_some_and(|pipeline| cache.get_compute_pipeline(pipeline.0).is_some());
        if settings.artistic_occlusion && ssao.is_some() && !full_target {
            warn_once!(
                "GI material AO remains active, but native SSAO requires a full-target viewport at origin zero"
            );
        }
        let (environment_table, environment_view, environment_rotation, environment_intensity) =
            if let Some(environment) = environment {
                let Ok((rotation, intensity)) = environment.validated() else {
                    warn_once!("GI environment rotation/intensity is invalid");
                    if let Some(buffers) = existing.as_ref() {
                        buffers.screen_history_valid.store(false, Ordering::Relaxed);
                    }
                    commands
                        .entity(entity)
                        .remove::<(ViewBindings, GiPreparedView)>();
                    continue;
                };
                let (Some(table), Some(image)) = (
                    environments.tables.get(&environment.image.id()),
                    images.get(&environment.image),
                ) else {
                    if let Some(buffers) = existing.as_ref() {
                        buffers.screen_history_valid.store(false, Ordering::Relaxed);
                    }
                    commands
                        .entity(entity)
                        .remove::<(ViewBindings, GiPreparedView)>();
                    continue;
                };
                if image.size_2d() != UVec2::new(table.width, table.height) {
                    if let Some(buffers) = existing.as_ref() {
                        buffers.screen_history_valid.store(false, Ordering::Relaxed);
                    }
                    commands
                        .entity(entity)
                        .remove::<(ViewBindings, GiPreparedView)>();
                    continue;
                }
                (
                    Some(table),
                    &image.texture_view,
                    Vec4::from_array(rotation.to_array()),
                    intensity,
                )
            } else {
                (None, fallback_view, Vec4::W, 1.0)
            };
        if settings.intensity <= 0.0 {
            if let Some(buffers) = existing {
                buffers.screen_history_valid.store(false, Ordering::Relaxed);
            }
            commands
                .entity(entity)
                .remove::<(ViewBindings, GiPreparedView)>();
            continue;
        }
        let (Some(depth), Some(gbuffer)) = (prepass.depth_only_view(), prepass.deferred_view())
        else {
            warn_once!("GI view is waiting for the depth or deferred prepass texture");
            if let Some(buffers) = existing {
                buffers.screen_history_valid.store(false, Ordering::Relaxed);
            }
            commands
                .entity(entity)
                .remove::<(ViewBindings, GiPreparedView)>();
            continue;
        };
        if target.main_texture_format() != TextureFormat::Rgba16Float
            || view.viewport.z == 0
            || view.viewport.w == 0
        {
            warn_once!(
                format = ?target.main_texture_format(),
                viewport = ?view.viewport,
                "GI view requires a non-empty viewport and an RGBA16Float main target"
            );
            if let Some(buffers) = existing {
                buffers.screen_history_valid.store(false, Ordering::Relaxed);
            }
            commands
                .entity(entity)
                .remove::<(ViewBindings, GiPreparedView)>();
            continue;
        }
        let viewport = UVec2::new(view.viewport.z, view.viewport.w);
        let receiver_bytes = u64::from(view_depth.texture().width())
            .saturating_mul(u64::from(view_depth.texture().height()))
            .saturating_mul(u64::from(view_depth.texture().sample_count()))
            .saturating_mul(16);
        let retained_view_bytes = existing.as_ref().map_or(0, |buffers| {
            view_bytes(buffers.size, buffers.shading_size, buffers.settings)
                .saturating_add(buffers.light_bytes())
        });
        let retained_receiver_bytes = receiver.map_or(0, |receiver| receiver.bytes());
        let reuse_effective_settings = existing.as_ref().is_some_and(|buffers| {
            receiver.is_some_and(|receiver| receiver.matches(view_depth))
                && buffers.viewport_size == viewport
                && buffers.size == resolution(viewport, buffers.settings)
                && buffers.shading_size == resolve_size(viewport, buffers.settings)
                && settings_match_budgeted_request(
                    buffers.settings,
                    buffers.allocation_request,
                    allocation_request,
                )
        });
        if reuse_effective_settings {
            settings = existing.as_ref().expect("checked existing view").settings;
        } else {
            let other_live_bytes = allocated_views
                .saturating_sub(retained_view_bytes)
                .saturating_sub(retained_receiver_bytes);
            let available_view_bytes = view_budget_bytes
                .saturating_sub(other_live_bytes)
                .saturating_sub(receiver_bytes);
            let Some(adapted) = adapt_view_settings(
                viewport,
                allocation_request,
                available_view_bytes,
                &device.limits(),
            ) else {
                let mut minimum = allocation_request;
                minimum.sampling.resolve_scale = 8;
                minimum.sampling.max_receiver_cells = 1;
                let minimum_size = resolution(viewport, minimum);
                let minimum_view_bytes =
                    view_bytes(minimum_size, resolve_size(viewport, minimum), minimum);
                warn_once!(
                    viewport = ?viewport,
                    requested_resolve_scale = resolve_scale(allocation_request),
                    receiver_bytes,
                    other_live_bytes,
                    minimum_view_bytes,
                    view_budget_bytes,
                    "GI view cannot fit the GPU memory/device budget through resolve scale 8; lower the viewport, reduce other GI views, or raise view_gpu_budget_bytes"
                );
                if let Some(buffers) = existing.as_ref() {
                    buffers.screen_history_valid.store(false, Ordering::Relaxed);
                }
                commands.entity(entity).remove::<(
                    ViewBuffers,
                    ViewBindings,
                    GiPreparedView,
                    crate::receiver::ReceiverTexture,
                )>();
                continue;
            };
            settings = adapted;
            let requested_size = resolution(viewport, allocation_request);
            let effective_size = resolution(viewport, settings);
            if resolve_scale(settings) != resolve_scale(allocation_request)
                || effective_size != requested_size
            {
                info!(
                    viewport = ?viewport,
                    requested_resolve_scale = resolve_scale(allocation_request),
                    effective_resolve_scale = resolve_scale(settings),
                    requested_grid = ?requested_size,
                    effective_grid = ?effective_size,
                    receiver_bytes,
                    other_live_bytes,
                    view_bytes = view_bytes(
                        effective_size,
                        resolve_size(viewport, settings),
                        settings,
                    ),
                    view_budget_bytes,
                    "GI view resolution adapted to fit GPU memory/device limits"
                );
            }
        }
        let shading_size = resolve_size(viewport, settings);
        let size = resolution(viewport, settings);
        if !view_fits_device(size, shading_size, viewport, settings, &device.limits()) {
            warn_once!("GI minimum cascade allocation exceeds this device's limits");
            if let Some(buffers) = existing.as_ref() {
                buffers.screen_history_valid.store(false, Ordering::Relaxed);
                buffers.history_valid.store(false, Ordering::Relaxed);
            }
            commands
                .entity(entity)
                .remove::<(ViewBindings, GiPreparedView)>();
            continue;
        }
        let mut created_receiver = None;
        if receiver.is_none_or(|receiver| !receiver.matches(view_depth)) {
            let bytes = receiver_bytes;
            if allocated_views.saturating_add(bytes) > view_budget_bytes {
                warn_once!(
                    viewport = ?viewport,
                    retained_bytes = allocated_views,
                    receiver_bytes = bytes,
                    view_budget_bytes,
                    "GI receiver identity texture/resize overlap exceeds the view GPU memory budget; wait for the old view allocation to retire or lower the viewport"
                );
                if let Some(buffers) = existing.as_ref() {
                    buffers.screen_history_valid.store(false, Ordering::Relaxed);
                }
                commands.entity(entity).remove::<(
                    ViewBuffers,
                    ViewBindings,
                    GiPreparedView,
                    crate::receiver::ReceiverTexture,
                )>();
                continue;
            }
            allocated_views = allocated_views.saturating_add(bytes);
            created_receiver = Some(crate::receiver::ReceiverTexture::new(&device, view_depth));
        }
        let receiver = created_receiver
            .as_ref()
            .or(receiver)
            .expect("admitted receiver texture");
        let mut created = None;
        let mut existing = existing;
        if existing.as_ref().is_none_or(|v| {
            v.size != size
                || v.shading_size != shading_size
                || v.viewport_size != viewport
                || v.settings.max_rays != settings.max_rays
                || v.settings.multi_bounce != settings.multi_bounce
                || v.settings.method != settings.method
                || v.settings.sampling != settings.sampling
                || v.buckets
                    != cache_bucket_count(u64::from(size.x) * u64::from(size.y), settings) as u32
        }) {
            let reusable_world_cache = existing
                .as_ref()
                .and_then(|previous| {
                    ViewBuffers::reusable_world_cache_bytes(previous, size, view.viewport, settings)
                })
                .unwrap_or(0);
            let candidate =
                view_bytes(size, shading_size, settings).saturating_sub(reusable_world_cache);
            if allocated_views.saturating_add(candidate) > view_budget_bytes {
                warn_once!(
                    viewport = ?viewport,
                    resolve_scale = resolve_scale(settings),
                    grid = ?size,
                    retained_bytes = allocated_views,
                    candidate_view_bytes = candidate,
                    view_budget_bytes,
                    "GI camera allocation/resize overlap exceeds the view GPU memory budget; old resources must retire before the adapted allocation can be created"
                );
                if let Some(buffers) = existing.as_ref() {
                    buffers.screen_history_valid.store(false, Ordering::Relaxed);
                }
                commands.entity(entity).remove::<(
                    ViewBuffers,
                    ViewBindings,
                    GiPreparedView,
                    crate::receiver::ReceiverTexture,
                )>();
                continue;
            }
            allocated_views = allocated_views.saturating_add(candidate);
            let mut replacement = ViewBuffers::new(
                &device,
                size,
                shading_size,
                viewport,
                settings,
                scene.revision,
                existing.as_deref(),
                view.viewport,
            );
            if let Some(previous) = existing.as_deref() {
                replacement.preserve_world_cache_from(previous, view.viewport);
            }
            created = Some(replacement);
        }
        let buffers_created = created.is_some();
        let Some(buffers) = created.as_mut().or(existing.as_deref_mut()) else {
            continue;
        };
        if allocated_views > view_budget_bytes {
            warn_once!("GI live camera/light allocations exceed the view GPU memory budget");
            buffers.screen_history_valid.store(false, Ordering::Relaxed);
            commands.entity(entity).remove::<(
                ViewBuffers,
                ViewBindings,
                GiPreparedView,
                crate::receiver::ReceiverTexture,
            )>();
            continue;
        }
        let rendered = *buffers
            .rendered_frame
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let default_layers = RenderLayers::default();
        let render_layers = render_layers.unwrap_or(&default_layers);
        let environment_revision = environment_table.map_or(0, |table| table.revision);
        let scene_history = SceneHistorySnapshot::capture(
            &scene,
            render_layers,
            settings.environment,
            environment,
            environment_revision,
        );
        let previous_scene_history = buffers
            .scene_history
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone();
        let (dirty_min, mut dirty_max) =
            dirty_scene_bounds(&previous_scene_history, &scene_history);
        let scene_changed_since_history =
            scene_changed_since_history(&previous_scene_history, &scene_history);
        // Keep compatible screen history across scene revisions even when the
        // CPU skipped intermediate revisions. dirty_scene_bounds compares the
        // last rendered snapshot directly with the current scene, so the shader
        // can reject only receivers inside the accumulated old/new dirty bounds.
        // Clearing the whole image here turns every sparse update phase into a
        // visible grid whenever motion vectors are unavailable.
        // Cached paths may include emissive radiance pre-compensated for Bevy's
        // exposure weight. Compare with the last encoded frame so skipped renders
        // retain the cache reset; screen history is managed independently below.
        let exposure_changed = rendered.exposure != camera.exposure;
        let scene_changed = buffers.revision != scene.revision;
        let meshes_changed = previous_scene_history.valid
            && previous_scene_history.mesh_revision != scene_history.mesh_revision;
        let geometry_changed = previous_scene_history.valid
            && previous_scene_history.geometry_revision != scene_history.geometry_revision;
        let layers_changed = previous_scene_history.valid
            && previous_scene_history.view_layers != scene_history.view_layers;
        let environment_changed = previous_scene_history.valid
            && (previous_scene_history.environment != scene_history.environment
                || previous_scene_history.environment_revision
                    != scene_history.environment_revision);
        let settings_changed = buffers.settings != settings;
        let settings_history_changed = history_settings_changed(buffers.settings, settings);
        let world_cache_settings_changed = world_cache_settings_changed(buffers.settings, settings);
        let viewport_changed = buffers.viewport != view.viewport;
        let punctual_light_change =
            punctual_light_history_change(&previous_scene_history, &scene_history);
        let bounded_geometry_reuse = !layers_changed
            && !exposure_changed
            && !meshes_changed
            && !world_cache_settings_changed
            && !viewport_changed
            && bounded_geometry_history_reuse(&previous_scene_history, &scene_history);
        let local_light_geometry_proven = if geometry_changed {
            bounded_geometry_reuse
        } else {
            !scene_changed_since_history
                || light_only_scene_snapshot_unchanged(&previous_scene_history, &scene_history)
        };
        let emissive_light_change = emissive_light_history_change(
            &previous_scene_history,
            &scene_history,
            local_light_geometry_proven,
        );
        let global_view_change = layers_changed
            || exposure_changed
            || settings_history_changed
            || world_cache_settings_changed
            || meshes_changed
            || viewport_changed;
        let (light_dirty_min, light_dirty_max) = punctual_dirty_bounds(
            punctual_light_change,
            global_view_change,
            local_light_geometry_proven,
        );
        let (emissive_dirty_min, emissive_dirty_max) = emissive_dirty_bounds(
            emissive_light_change,
            global_view_change,
            local_light_geometry_proven,
        );
        let mut source_history = source_history_fields(
            &previous_scene_history,
            &scene_history,
            punctual_light_change,
            emissive_light_change,
        );
        // Geometry motion still enables native pixel motion vectors via
        // stability.y. Only validated dependency bounds permit world reuse.
        dirty_max.w = f32::from(bounded_geometry_reuse);
        let reset_cache = reset_cache_history(
            layers_changed,
            exposure_invalidates_world_history(exposure_changed),
            // Unknown geometry changes retain global invalidation. The bounded
            // rigid case uses each interval/probe's complete visibility dependencies.
            meshes_changed || (geometry_changed && !bounded_geometry_reuse),
            world_cache_settings_changed,
            viewport_changed,
        );
        if scene_changed || scene_changed_since_history {
            debug!(
                entity = ?entity,
                scene_changed,
                scene_changed_since_history,
                geometry_changed,
                meshes_changed,
                bounded_geometry_reuse,
                reset_cache,
                punctual_light_change = ?punctual_light_change,
                emissive_light_change = ?emissive_light_change,
                source_active = source_history.x,
                source_changed = source_history.y,
                dirty_min = ?dirty_min,
                dirty_max = ?dirty_max,
                light_dirty_min = ?light_dirty_min,
                light_dirty_max = ?light_dirty_max,
                emissive_dirty_min = ?emissive_dirty_min,
                emissive_dirty_max = ?emissive_dirty_max,
                previous_geometry_revision = previous_scene_history.geometry_revision,
                current_geometry_revision = scene_history.geometry_revision,
                previous_mesh_revision = previous_scene_history.mesh_revision,
                current_mesh_revision = scene_history.mesh_revision,
                "GI scene history decision"
            );
        }
        let reset_screen = exposure_changed
            || view_change_invalidates_screen_history(settings_history_changed, viewport_changed);
        buffers.allocation_request = allocation_request;
        let environment_source_changed =
            buffers.environment.as_ref().map(|source| source.image.id())
                != environment.map(|source| source.image.id())
                || buffers.environment_revision != environment_revision;
        if scene_changed
            || layers_changed
            || environment_source_changed
            || buffers.lights.is_none()
            || buffers.pending_lights.is_some()
        {
            let (mut lights, conventional, selector) = view_lights(&scene.data, render_layers);
            let alias_records =
                environment_table.map_or(0, |table| table.aliases.len().div_ceil(6));
            let light_records = lights.len().saturating_add(alias_records).max(1);
            let light_bytes = (light_records as u64).saturating_mul(96);
            let source = environment.map(|source| (source.image.id(), environment_revision));
            let pending_matches = buffers.pending_lights.as_ref().is_some_and(|pending| {
                pending.source == source
                    && pending.layers == *render_layers
                    && pending.buffer.size() == light_bytes
            });
            let active_matches = buffers.light_source == source
                && buffers.render_layers == *render_layers
                && buffers
                    .lights
                    .as_ref()
                    .is_some_and(|buffer| buffer.size() == light_bytes);
            if light_bytes > device.limits().max_storage_buffer_binding_size
                || (!pending_matches
                    && !active_matches
                    && allocated_views.saturating_add(light_bytes) > view_budget_bytes)
            {
                warn_once!("GI environment aliases/lights exceed view binding or memory budget");
                buffers.screen_history_valid.store(false, Ordering::Relaxed);
                commands
                    .entity(entity)
                    .remove::<(ViewBindings, GiPreparedView)>();
                continue;
            }
            let alias_offset = environment_table
                .map(|table| table.append_alias_lights(&mut lights))
                .unwrap_or(0);
            let environment_info = environment_table.map_or(UVec4::ZERO, |table| {
                UVec4::new(alias_offset, table.width, table.height, 1)
            });
            if !buffers.prepare_lights(
                &device,
                &queue,
                lights,
                render_layers,
                source,
                scene.revision,
                scene.geometry_revision,
                scene.light_revision,
                &mut allocated_views,
                view_budget_bytes,
                &mut upload_remaining,
            ) {
                buffers.screen_history_valid.store(false, Ordering::Relaxed);
                commands
                    .entity(entity)
                    .remove::<(ViewBindings, GiPreparedView)>();
                // Preserve newly admitted buffers and their upload cursor while
                // the rest of this camera remains disabled until commit.
                if let Some(created) = created.take() {
                    commands.entity(entity).insert(created);
                }
                if let Some(receiver) = created_receiver.take() {
                    commands.entity(entity).insert(receiver);
                }
                continue;
            }
            buffers.environment_info = environment_info;
            buffers.conventional_lights = conventional;
            buffers.layer_selector = selector;
        }
        if scene_changed
            || layers_changed
            || environment_changed
            || exposure_changed
            || meshes_changed
            || geometry_changed
            || buffers.settings != settings
            || buffers.viewport != view.viewport
        {
            if reset_screen {
                buffers.screen_history_valid.store(false, Ordering::Relaxed);
            }
            if reset_cache {
                buffers.history_valid.store(false, Ordering::Relaxed);
            }
            buffers.revision = scene.revision;
            buffers.mesh_revision = scene.mesh_revision;
            buffers.geometry_revision = scene.geometry_revision;
            buffers.settings = settings;
            buffers.viewport = view.viewport;
            buffers.render_layers = render_layers.clone();
            buffers.environment = environment.cloned();
            buffers.environment_revision = environment_revision;
        }
        let Some(lights) = buffers
            .lights
            .as_ref()
            .map(|buffer| buffer.as_entire_binding())
        else {
            continue;
        };
        if buffers.pending_lights.is_none()
            && buffers.light_revision == scene.light_revision
            && gpu_scene
                .as_ref()
                .is_some_and(|gpu| gpu.ready_for_revision(scene.revision))
        {
            shaders
                .status
                .record_submitted_revisions(RadianceCascadesRevisions {
                    scene: scene.revision,
                    geometry: scene.geometry_revision,
                    lighting: scene.light_revision,
                });
        }
        let frame = buffers.frame.load(Ordering::Relaxed);
        let motion_epoch = if source_history.z & 1 != 0 {
            frame.wrapping_add(1).max(1)
        } else {
            buffers.directional_motion_epoch.load(Ordering::Relaxed)
        };
        if motion_epoch != 0 {
            source_history.z |= 2;
            source_history.w = motion_epoch.wrapping_sub(1);
        }
        let punctual_motion_epoch = if source_history.z & 4 != 0 {
            frame.wrapping_add(1).max(1)
        } else {
            buffers.punctual_motion_epoch.load(Ordering::Relaxed)
        };
        if punctual_motion_epoch != 0 {
            source_history.z |= 8;
        }
        let candidate_count = size.x * size.y;
        let (shared_primary_query_limit, secondary_query_limit) = cache_query_limits(
            settings.sampling.max_traversal_queries,
            settings.multi_bounce,
        );
        let (shared_primary_cache_limit, secondary_cache_limit) =
            cache_query_limits(settings.sampling.max_cache_entries, settings.multi_bounce);
        let requested_update_budget = runtime_update_budget(
            settings,
            runtime_budget.map(|budget| budget.max_updates_per_frame),
        );
        let secondary_enabled = settings.multi_bounce
            && requested_update_budget >= 2
            && (settings.sampling.max_traversal_queries == 0 || secondary_query_limit > 0)
            && (settings.sampling.max_cache_entries == 0 || secondary_cache_limit > 0);
        let primary_query_limit = if secondary_enabled {
            shared_primary_query_limit
        } else {
            settings.sampling.max_traversal_queries
        };
        let primary_cache_limit = if secondary_enabled {
            shared_primary_cache_limit
        } else {
            settings.sampling.max_cache_entries
        };
        let primary_requested = if secondary_enabled {
            requested_update_budget / 2
        } else {
            requested_update_budget
        };
        let secondary_update_budget = requested_update_budget - primary_requested;
        let mut primary_settings = settings;
        primary_settings.sampling.max_traversal_queries = primary_query_limit;
        let update_budget = query_affordable_update_budget(
            primary_settings,
            primary_requested,
            buffers.conventional_lights,
            scene.data.emissive_lights.len() as u32,
            buffers.environment_info.w != 0,
        );
        let current = &buffers.caches[(frame % 2) as usize];
        let previous = &buffers.caches[(frame.wrapping_add(1) % 2) as usize];
        let irradiance = &buffers.irradiance_views[(frame % 2) as usize];
        let previous_irradiance = &buffers.irradiance_views[(frame.wrapping_add(1) % 2) as usize];
        let metadata = &buffers.metadata_views[(frame % 2) as usize];
        let previous_metadata = &buffers.metadata_views[(frame.wrapping_add(1) % 2) as usize];
        // Match Bevy's raster projection, including TAA's subpixel jitter.
        // Reconstructing jittered depth with the unjittered projection shifts
        // software receivers and rejects otherwise valid temporal history.
        let view_from_world = Mat4::from(view.world_from_view.affine().inverse());
        // Exclude subpixel TAA jitter from sparse-field invalidation; it is a
        // temporal sampling offset, not a camera change.
        let stable_clip_from_world = view.clip_from_view * view_from_world;
        let clip_from_world = if let Some(jitter) = temporal_jitter {
            let mut projection = view.clip_from_view;
            let viewport_size = resolution_override
                .map_or(view.viewport.zw().as_vec2(), |resolution| {
                    resolution.0.as_vec2()
                });
            jitter.jitter_projection(&mut projection, viewport_size);
            projection * view_from_world
        } else {
            view.clip_from_world
                .unwrap_or(view.clip_from_view * view_from_world)
        };
        let has_rendered_history = frame != 0;
        let scene_or_lights_changed = scene_changed_since_history
            || scene_changed
            || layers_changed
            || environment_changed
            || environment_source_changed
            || exposure_changed
            || meshes_changed
            || geometry_changed;
        let camera_changed =
            camera_matrix_changed(rendered.stable_clip_from_world, stable_clip_from_world);
        if camera_changed || frame % 60 == 0 {
            debug!(
                frame,
                camera_changed,
                scene_changed_since_history,
                scene_changed,
                layers_changed,
                environment_changed,
                exposure_changed,
                settings_changed,
                screen_history_valid = buffers.screen_history_valid.load(Ordering::Relaxed),
                world_history_valid = buffers.history_valid.load(Ordering::Relaxed),
                "Split history validity"
            );
        }
        let geometry_or_layers_changed = geometry_changed || meshes_changed || layers_changed;
        let (candidate_phase, fair_quota) = buffers
            .candidate_schedule
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .phase_for(size.x, size.y, candidate_count, update_budget);
        let sparse_history_reset = sparse_history_reset_requested(
            has_rendered_history,
            geometry_or_layers_changed,
            settings_history_changed,
            viewport_changed,
            camera_changed,
        );
        if sparse_history_reset {
            debug!(
                view = ?entity,
                frame,
                has_rendered_history,
                scene_or_lights_changed,
                scene_changed_since_history,
                scene_changed,
                layers_changed,
                environment_changed,
                environment_source_changed,
                exposure_changed,
                meshes_changed,
                geometry_changed,
                settings_changed,
                viewport_changed,
                camera_changed,
                scene_revision = scene.revision,
                geometry_revision = scene.geometry_revision,
                previous_scene_revision = previous_scene_history.revision,
                previous_geometry_revision = previous_scene_history.geometry_revision,
                "Reset sparse GI history"
            );
        }
        let params = Parameters {
            world_from_clip: clip_from_world.inverse(),
            clip_from_world,
            previous_clip_from_world: rendered.clip_from_world,
            previous_world_from_clip: rendered.world_from_clip,
            camera_position: view.world_from_view.translation().extend(1.0),
            viewport: view.viewport,
            size: UVec4::new(size.x, size.y, candidate_count, buffers.buckets),
            tuning: Vec4::new(
                settings.probe_spacing,
                settings.interval_length,
                settings.normal_bias,
                settings.intensity,
            ),
            environment: settings.environment.to_vec4().with_w(camera.exposure),
            history: Vec4::new(
                settings.temporal_blend,
                0.0,
                if buffers.history_valid.load(Ordering::Relaxed) {
                    1.0
                } else {
                    0.0
                },
                settings.history_frames as f32,
            ),
            interval_history: Vec4::new(
                settings.split_transmittance_blend,
                1.0,
                // Fractional .25 requests diffuse debug histograms. The request
                // follows render by one prepare frame.
                if buffers
                    .history_diagnostics_requested
                    .load(Ordering::Relaxed)
                {
                    0.25
                } else {
                    0.0
                },
                traversal_query_caps(primary_query_limit).0 as f32,
            ),
            frame: UVec4::new(
                frame,
                0,
                scene.data.instances.len() as u32,
                buffers.conventional_lights,
            ),
            light_counts: UVec4::new(
                buffers.conventional_lights,
                scene.data.emissive_lights.len() as u32,
                settings.emissive_samples,
                u32::from(secondary_enabled && settings.cached_hit_sky),
            ),
            stability: Vec4::new(
                if buffers.screen_history_valid.load(Ordering::Relaxed) {
                    1.0
                } else {
                    0.0
                },
                if scene_changed_since_history
                    || scene_changed
                    || layers_changed
                    || environment_changed
                    || exposure_changed
                    || world_cache_settings_changed
                {
                    1.0
                } else {
                    0.0
                },
                f32::from(prepass.motion_vectors_view().is_some()),
                1.0,
            ),
            layers: UVec4::new(
                buffers.layer_selector.x,
                buffers.layer_selector.y,
                u32::from(settings.artistic_occlusion) | (u32::from(ssao_valid) << 1),
                // Hardware ray mask; hardware-opaque instances only carry layer 0.
                if render_layers.intersects(&RenderLayers::layer(0)) {
                    0
                } else {
                    0xfe
                },
            ),
            environment_info: buffers.environment_info,
            environment_rotation,
            environment_scale: Vec4::new(
                environment_intensity,
                f32::from(secondary_enabled),
                0.0,
                f32::from(
                    secondary_enabled
                        && buffers.history_valid.load(Ordering::Relaxed)
                        && buffers
                            .secondary
                            .as_ref()
                            .is_some_and(|cache| cache.valid.load(Ordering::Relaxed))
                        && !reset_cache
                        && !geometry_changed,
                ),
            ),
            geometry_pages: UVec4::new(
                if gpu_scene_ref.geometry_paged {
                    if scene.data.deformation.vertices > 0 {
                        2
                    } else {
                        1
                    }
                } else {
                    0
                },
                gpu_scene_ref.geometry_page_side,
                0,
                0,
            ),
            scene_pages: gpu_scene_ref.scene_page_config,
            scene_overlay: gpu_scene_ref.scene_overlay_config,
            dirty_min,
            dirty_max,
            source_history,
            sampling: UVec4::new(
                update_budget,
                primary_query_limit,
                primary_cache_limit,
                resolve_scale(settings),
            ),
            owner_sequence: UVec4::new(
                split_lod_target_pixels(settings),
                0,
                fair_quota,
                sparse_candidate_phase_word(
                    candidate_phase,
                    sparse_history_reset,
                    update_budget == 0,
                ),
            ),
            cache_time: UVec4::new(
                buffers.cache_clock.elapsed().as_millis() as u32,
                punctual_motion_epoch.wrapping_sub(1),
                0,
                0,
            ),
            emissive_dirty_min,
            emissive_dirty_max,
            light_dirty_min,
            light_dirty_max,
        };
        for (level, uniform) in buffers.uniforms.iter_mut().enumerate() {
            let mut level_params = params;
            level_params.frame.y = (level % LEVELS) as u32;
            uniform.set(level_params);
            uniform.write_buffer(&device, &queue);
        }
        for (clear_index, uniform) in buffers.clear_uniforms.iter_mut().enumerate() {
            // Reset root-ray offsets before surface seeding, but preserve them
            // during the second cache-head clear.
            let clear_params = clear_pass_parameters(params, clear_index);
            uniform.set(clear_params);
            uniform.write_buffer(&device, &queue);
        }
        if let Some(secondary) = buffers.secondary.as_mut() {
            let mut secondary_params = params;
            secondary_params.environment_scale.z = 1.0;
            if !secondary.valid.load(Ordering::Relaxed) {
                secondary_params.history.z = 0.0;
            }
            secondary_params.history.w = secondary_params.history.w.max(1.0);
            secondary_params.sampling.x = secondary_update_budget;
            secondary_params.sampling.y = secondary_query_limit;
            secondary_params.sampling.z = secondary_cache_limit;
            secondary_params.owner_sequence.w &= !SPARSE_UPDATES_DISABLED_FLAG;
            secondary_params.interval_history.w =
                traversal_query_caps(secondary_query_limit).0 as f32;
            for (level, uniform) in secondary.uniforms.iter_mut().enumerate() {
                secondary_params.frame.y = level as u32;
                uniform.set(secondary_params);
                uniform.write_buffer(&device, &queue);
            }
            secondary_params.frame.y = 0;
            for (index, uniform) in secondary.clear_uniforms.iter_mut().enumerate() {
                uniform.set(clear_pass_parameters(secondary_params, index));
                uniform.write_buffer(&device, &queue);
            }
        }
        let ssao_view = ssao.map_or(fallback_view, |ssao| {
            &ssao.screen_space_ambient_occlusion_texture.default_view
        });
        let parity = (frame & 1) as usize;
        let display_index = composite_display_index(settings.spatial_filter_steps, parity);
        let binding_key = view_binding_key(
            buffers,
            parity,
            display_index,
            [
                gpu_scene_ref.geometry.as_ref().unwrap(),
                gpu_scene_ref.nodes.as_ref().unwrap(),
                buffers.lights.as_ref().unwrap(),
            ],
            &[
                depth,
                gbuffer,
                atlas_view,
                fallback_view,
                prepass.motion_vectors_view().unwrap_or(fallback_view),
                &receiver.view,
                environment_view,
                geometry_page_view,
                scene_page_view,
                ssao_view,
                dfg_view,
            ],
        );
        if !buffers_created
            && created_receiver.is_none()
            && let Some(mut bindings) = buffers.binding_cache[parity]
                .as_ref()
                .filter(|cached| cached.key == binding_key)
                .map(|cached| cached.bindings.clone())
        {
            bindings.frame = RenderedFrame {
                clip_from_world,
                world_from_clip: params.world_from_clip,
                stable_clip_from_world,
                exposure: camera.exposure,
            };
            bindings.secondary_enabled = secondary_enabled;
            bindings.scene_history = scene_history.clone();
            buffers.binding_cache[parity] = Some(CachedViewBindings {
                key: binding_key,
                bindings: bindings.clone(),
            });
            let required = required_pipelines(settings);
            let pipelines_ready = (0..COMPUTE_ENTRIES.len())
                .all(|i| required & (1 << i) == 0 || pipelines.pipeline(i).is_some())
                && cache.get_render_pipeline(pipelines.composite).is_some()
                && cache.get_compute_pipeline(pipelines.composite_anchors).is_some();
            let (target_size, viewport) = camera_descriptor(Some(camera));
            commands.entity(entity).insert(GiPreparedView {
                settings: requested_settings,
                target_size,
                viewport,
                layers: render_layers.clone(),
                environment: environment.cloned(),
                environment_revision,
                view_budget: Some(budget.view_bytes),
                pipelines_ready,
                receiver_ready: receiver.ready.clone(),
                scene_history,
            });
            commands.entity(entity).insert(bindings);
            continue;
        }
        let uniforms = &buffers.uniforms;
        // The current frame writes its candidate fields into frame parity;
        // the opposite parity contains the complete prior estimate.
        let (candidate_output_index, candidate_input_index) = candidate_field_indices(frame);
        // Temporal reconstruction keeps receiver history separate from scratch storage.
        let current_receiver_history = &buffers.metadata_views[3 + candidate_output_index];
        let previous_receiver_history = &buffers.metadata_views[3 + candidate_input_index];
        let current_background = &buffers.background_views[0];
        let next_background = &buffers.background_views[1 + candidate_output_index];
        let previous_background = &buffers.background_views[1 + candidate_input_index];
        let bind_compute = |uniform: &UniformBuffer<Parameters>,
                            output: &TextureView,
                            input: &TextureView,
                            sample_input: &TextureView,
                            metadata_output: &TextureView,
                            metadata_input: &TextureView,
                            receiver_metadata: &TextureView,
                            receiver_history_output: &TextureView,
                            receiver_history_input: &TextureView,
                            background_output: &TextureView,
                            background_sample: &TextureView,
                            secondary: Option<&SecondaryCache>,
                            feedback: bool| {
            let current_heads =
                secondary.map_or(&current.heads, |cache| &cache.caches[parity].heads);
            let work_entries = secondary.map_or(&buffers.entries, |cache| &cache.entries);
            let history_cache = if feedback {
                buffers.secondary.as_ref()
            } else {
                secondary
            };
            let previous_heads =
                history_cache.map_or(&previous.heads, |cache| &cache.caches[1 - parity].heads);
            let history_entries =
                history_cache.map_or(&buffers.history_entries, |cache| &cache.history_entries);
            let mut entries = BindGroupEntries::sequential((
                uniform,
                triangles.clone(),
                nodes.clone(),
                lights.clone(),
                depth,
                fallback_view,
                current_heads.as_entire_binding(),
                work_entries.as_entire_binding(),
                previous_heads.as_entire_binding(),
                history_entries.as_entire_binding(),
                output,
                input,
                buffers.irradiance_probes.as_entire_binding(),
                sample_input,
                metadata_output,
                metadata_input,
                atlas_view,
                buffers.irradiance_probes.as_entire_binding(),
                atlas_view,
                fallback_view,
                prepass.motion_vectors_view().unwrap_or(fallback_view),
                receiver_metadata,
                environment_view,
                receiver_history_output,
                fallback_view,
                receiver_history_input,
                &receiver.view,
                gbuffer,
                geometry_page_view,
                scene_page_view,
            ))
            .iter()
            .filter(|entry| !matches!(entry.binding, 5 | 17 | 18 | 19 | 21 | 24 | 30))
            .cloned()
            .collect::<Vec<_>>();
            entries.push(BindGroupEntry {
                binding: 37,
                resource: BindingResource::TextureView(dfg_view),
            });
            for (binding, view) in [
                (38, background_output),
                (39, background_sample),
                (40, previous_background),
            ] {
                entries.push(BindGroupEntry {
                    binding,
                    resource: BindingResource::TextureView(view),
                });
            }
            for (binding, reservoirs) in [
                (42, &buffers.rtgi_reservoirs[parity]),
                (43, &buffers.rtgi_reservoirs[1 - parity]),
                (44, &buffers.rtgi_cache[0]),
                (45, &buffers.rtgi_cache[1]),
                (46, &buffers.rtgi_cache[2]),
            ] {
                entries.push(BindGroupEntry {
                    binding,
                    resource: reservoirs.as_entire_binding(),
                });
            }
            if let Some(tlas) = hardware_tlas {
                entries.push(BindGroupEntry {
                    binding: TLAS_BINDING,
                    resource: tlas.as_binding(),
                });
            }
            device.create_bind_group(
                "split_rc_compute",
                &cache.get_bind_group_layout(&pipelines.compute_layout),
                &entries,
            )
        };
        let compute = std::array::from_fn(|level| {
            bind_compute(
                &uniforms[level],
                irradiance,
                previous_irradiance,
                previous_irradiance,
                metadata,
                previous_metadata,
                &buffers.metadata_views[2],
                current_receiver_history,
                previous_receiver_history,
                next_background,
                current_background,
                None,
                false,
            )
        });
        let clear = std::array::from_fn(|index| {
            bind_compute(
                &buffers.clear_uniforms[index],
                irradiance,
                previous_irradiance,
                previous_irradiance,
                metadata,
                previous_metadata,
                &buffers.metadata_views[2],
                current_receiver_history,
                previous_receiver_history,
                next_background,
                current_background,
                None,
                false,
            )
        });
        let temporal = bind_compute(
            &uniforms[0],
            &buffers.irradiance_views[2],
            previous_irradiance,
            &buffers.irradiance_views[3],
            metadata,
            previous_metadata,
            &buffers.metadata_views[2],
            current_receiver_history,
            previous_receiver_history,
            next_background,
            current_background,
            None,
            false,
        );
        let world_resolve = bind_compute(
            &uniforms[0],
            &buffers.irradiance_views[3],
            previous_irradiance,
            previous_irradiance,
            metadata,
            previous_metadata,
            &buffers.metadata_views[2],
            current_receiver_history,
            previous_receiver_history,
            current_background,
            next_background,
            None,
            false,
        );
        let denoise = std::array::from_fn(|level| {
            // Temporal reconstruction writes the packed receiver history to the current ring slot.
            // It also writes the accumulated irradiance to view 2.
            // Start filtering from that result and ping-pong into view 3 first;
            // reading the raw sample here discards temporal accumulation.
            let input = &buffers.irradiance_views[2 + level % 2];
            let output = &buffers.irradiance_views[2 + (level + 1) % 2];
            bind_compute(
                &uniforms[level],
                output,
                input,
                metadata,
                previous_metadata,
                metadata,
                &buffers.metadata_views[2],
                &buffers.denoise_receiver_history_output_view,
                // Filter taps follow temporal reconstruction in this frame.
                current_receiver_history,
                next_background,
                current_background,
                None,
                false,
            )
        });
        let trace = buffers.secondary.as_ref().map(|secondary| {
            bind_compute(
                &uniforms[0],
                &secondary.hit_views[parity],
                &secondary.hit_views[1 - parity],
                previous_irradiance,
                metadata,
                previous_metadata,
                &buffers.metadata_views[2],
                current_receiver_history,
                previous_receiver_history,
                next_background,
                current_background,
                None,
                true,
            )
        });
        let secondary = buffers.secondary.as_ref().map(|secondary| {
            let bind = |uniform| {
                bind_compute(
                    uniform,
                    &secondary.hit_views[parity],
                    &secondary.hit_views[1 - parity],
                    previous_irradiance,
                    metadata,
                    previous_metadata,
                    &buffers.metadata_views[2],
                    current_receiver_history,
                    previous_receiver_history,
                    next_background,
                    current_background,
                    Some(secondary),
                    true,
                )
            };
            SecondaryBindings {
                compute: std::array::from_fn(|level| bind(&secondary.uniforms[level])),
                clear: std::array::from_fn(|index| bind(&secondary.clear_uniforms[index])),
            }
        });
        // Raw ping-pong irradiance is temporal history. Filtering only the
        // display output avoids recursively diffusing lighting across surfaces.
        let display = &buffers.irradiance_views[display_index];
        let display_receiver_history = composite_receiver_history_source(
            settings.spatial_filter_steps,
            current_receiver_history,
            &buffers.denoise_receiver_history_output_view,
        );
        let sun_shadows = hardware_tlas.is_some().then(|| {
            bind_compute(
                &uniforms[0],
                &buffers.sun_shadow_mask_view,
                previous_irradiance,
                previous_irradiance,
                metadata,
                previous_metadata,
                &buffers.metadata_views[2],
                current_receiver_history,
                previous_receiver_history,
                next_background,
                current_background,
                None,
                false,
            )
        });
        let composite_anchors = device.create_bind_group(
            "split_rc_composite_anchors",
            &cache.get_bind_group_layout(&composite_anchor_layout()),
            &BindGroupEntries::with_indices((
                (0, &uniforms[0]),
                (2, display),
                (5, gbuffer),
                (6, display_receiver_history),
                (7, &receiver.view),
                (14, &buffers.composite_anchor_views[0]),
                (15, &buffers.composite_anchor_views[1]),
            )),
        );
        let composite_entries = BindGroupEntries::sequential((
            &uniforms[0],
            display,
            display,
            depth,
            fallback_view,
            gbuffer,
            display_receiver_history,
            &receiver.view,
            ssao_view,
            current.heads.as_entire_binding(),
            buffers.entries.as_entire_binding(),
            buffers.irradiance_probes.as_entire_binding(),
            &buffers.composite_anchor_views[0],
            &buffers.composite_anchor_views[1],
            &buffers.composite_anchor_views[0],
            &buffers.composite_anchor_views[1],
            &buffers.sun_shadow_mask_view,
        ))
        .iter()
        .filter(|entry| !matches!(entry.binding, 1 | 4 | 14 | 15))
        .cloned()
        .collect::<Vec<_>>();
        let composite = device.create_bind_group(
            "split_rc_composite",
            &cache.get_bind_group_layout(&pipelines.composite_layout),
            &composite_entries,
        );
        let required = required_pipelines(settings);
        let pipelines_ready = (0..COMPUTE_ENTRIES.len())
            .all(|i| required & (1 << i) == 0 || pipelines.pipeline(i).is_some())
            && cache.get_render_pipeline(pipelines.composite).is_some()
            && cache.get_compute_pipeline(pipelines.composite_anchors).is_some();
        let (target_size, viewport) = camera_descriptor(Some(camera));
        commands.entity(entity).insert(GiPreparedView {
            settings: requested_settings,
            target_size,
            viewport,
            layers: render_layers.clone(),
            environment: environment.cloned(),
            environment_revision,
            view_budget: Some(budget.view_bytes),
            pipelines_ready,
            receiver_ready: receiver.ready.clone(),
            scene_history: scene_history.clone(),
        });
        let bindings = ViewBindings {
            compute,
            clear,
            trace,
            secondary,
            secondary_enabled,
            denoise,
            temporal,
            world_resolve,
            composite_anchors,
            composite,
            sun_shadows,
            frame: RenderedFrame {
                clip_from_world,
                world_from_clip: params.world_from_clip,
                stable_clip_from_world,
                exposure: camera.exposure,
            },
            scene_history,
        };
        buffers.binding_cache[parity] = Some(CachedViewBindings {
            key: binding_key,
            bindings: bindings.clone(),
        });
        commands.entity(entity).insert(bindings);
        if let Some(created) = created {
            commands.entity(entity).insert(created);
        }
        if let Some(receiver) = created_receiver {
            commands.entity(entity).insert(receiver);
        }
    }
    let total_live_bytes = live.saturating_add(allocated_views);
    memory_high_water.0 = memory_high_water.0.max(total_live_bytes);
    if let Some(diagnostics) = diagnostics
        && upload_remaining >= 20
    {
        // Count this tiny readback source against the same transfer allowance.
        upload_remaining -= 20;
        let values = [
            live.min(u64::from(u32::MAX)) as u32,
            candidate_bytes.min(u64::from(u32::MAX)) as u32,
            frame_upload_budget
                .saturating_sub(upload_remaining)
                .min(u32::MAX as usize) as u32,
            allocated_views.min(u64::from(u32::MAX)) as u32,
            memory_high_water.0.min(u64::from(u32::MAX)) as u32,
        ];
        let mut bytes = [0u8; 20];
        for (index, value) in values.into_iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        let counters = device.create_buffer(&BufferDescriptor {
            label: Some("gi_scene_upload_diagnostics"),
            size: 20,
            usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        queue.write_buffer(&counters, 0, &bytes);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
        diagnostics.record_u32(
            &mut encoder,
            &counters.slice(0..4),
            "penumbra/scene_live_bytes",
        );
        diagnostics.record_u32(
            &mut encoder,
            &counters.slice(4..8),
            "penumbra/scene_candidate_bytes",
        );
        diagnostics.record_u32(
            &mut encoder,
            &counters.slice(8..12),
            "penumbra/upload_bytes_used",
        );
        diagnostics.record_u32(
            &mut encoder,
            &counters.slice(12..16),
            "penumbra/view_live_bytes",
        );
        diagnostics.record_u32(
            &mut encoder,
            &counters.slice(16..20),
            "penumbra/memory_high_water_bytes",
        );
        queue.submit([encoder.finish()]);
    }
    if let Some(gpu_scene) = new_gpu_scene {
        commands.insert_resource(gpu_scene);
    }
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn render(
    view: ViewQuery<(
        &ViewTarget,
        &RadianceCascades,
        &ViewBuffers,
        &ViewBindings,
        &crate::receiver::ReceiverTexture,
        Option<&RadianceCascadesRawOutput>,
        Option<&GiLightingState>,
    )>,
    scene: Res<GiScene>,
    gpu_scene: Option<Res<GpuScene>>,
    shaders: Res<PipelineShaders>,
    pipelines: Res<Pipelines>,
    cache: Res<PipelineCache>,
    images: Res<RenderAssets<GpuImage>>,
    mut context: RenderContext,
    mut pending_pipelines: Local<u64>,
    hardware: Option<Res<hardware::HardwareTraversal>>,
) {
    let diagnostic_view = view.entity();
    let (target, settings, buffers, bindings, receiver, raw_output, lighting) = view.into_inner();
    let history_was_valid = buffers.history_valid.load(Ordering::Relaxed);
    let cold_priority = cold_priority_enabled(settings.history_frames, history_was_valid);
    buffers
        .history_diagnostics_requested
        .store(raw_output.is_some(), Ordering::Relaxed);
    if settings.intensity <= 0.0 {
        if raw_output.is_some() {
            shaders.status.record_raw_frame(None);
        }
        return;
    }
    // The current identity pass and committed scene decide lighting ownership
    // before native lighting draws, so fallback and GI never overlap.
    if !lighting.is_some_and(|lighting| lighting.0) {
        buffers.screen_history_valid.store(false, Ordering::Relaxed);
        if raw_output.is_some() {
            shaders.status.record_raw_frame(None);
        }
        return;
    }
    // Do not dispatch retained bindings after staging or a residency failure.
    if !scene.ready
        || gpu_scene
            .as_ref()
            .is_none_or(|gpu| !gpu.ready_for_revision(scene.revision))
    {
        buffers.screen_history_valid.store(false, Ordering::Relaxed);
        if raw_output.is_some() {
            shaders.status.record_raw_frame(None);
        }
        return;
    }
    if !receiver.ready.load(Ordering::Relaxed) {
        buffers.screen_history_valid.store(false, Ordering::Relaxed);
        if raw_output.is_some() {
            shaders.status.record_raw_frame(None);
        }
        return;
    }
    let compute: [Option<&ComputePipeline>; COMPUTE_ENTRIES.len()] =
        std::array::from_fn(|i| pipelines.pipeline(i));
    let required = required_pipelines(buffers.settings);
    let missing = compute
        .iter()
        .enumerate()
        .fold(0u64, |mask, (i, pipeline)| {
            mask | if pipeline.is_none() {
                required & (1 << i)
            } else {
                0
            }
        });
    if missing != 0 {
        buffers.screen_history_valid.store(false, Ordering::Relaxed);
        if *pending_pipelines != missing {
            debug!(pending = ?COMPUTE_ENTRIES.iter().enumerate().filter_map(|(i,name)|
                (missing & (1 << i) != 0).then_some(name)).collect::<Vec<_>>(), "GI waiting for shader pipelines");
            *pending_pipelines = missing;
        }
        if raw_output.is_some() {
            shaders.status.record_raw_frame(None);
        }
        return;
    }
    *pending_pipelines = 0;
    let (Some(composite), Some(composite_anchors)) = (
        cache.get_render_pipeline(pipelines.composite),
        cache.get_compute_pipeline(pipelines.composite_anchors),
    ) else {
        buffers.screen_history_valid.store(false, Ordering::Relaxed);
        if raw_output.is_some() {
            shaders.status.record_raw_frame(None);
        }
        return;
    };
    if buffers.frame.load(Ordering::Relaxed) == 0 {
        info!(
            ?diagnostic_view,
            rays = buffers.size.x * buffers.size.y,
            cascades = LEVELS,
            "Radiance Cascades dispatching",
        );
    }
    let diagnostics = context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let total_span = diagnostics.time_span(context.command_encoder(), "split_rc");
    let mut clear_dispatch_index = 0usize;
    let mut secondary_clear_dispatch_index = 0usize;
    let skip_passes: Vec<String> = std::env::var("PENUMBRA_SKIP_PASSES")
        .map(|value| {
            value
                .split(',')
                .map(|item| item.trim().to_owned())
                .collect()
        })
        .unwrap_or_default();
    let mut dispatch = |name: &'static str,
                        pipeline: &ComputePipeline,
                        level: usize,
                        x: u32,
                        y: u32| {
        if skip_passes.iter().any(|skipped| skipped.as_str() == name) {
            return;
        }
        // Indirect-argument slot: receiver traces use bytes 0..12, deferred
        // visibility queries 12..24. Each slot is copied before the first
        // pass that consumes it, after the pass that writes it.
        let indirect_offset = match name {
            "trace" | "secondary/trace" | "traceRecord" | "traceResolve" | "traceOverflow" => {
                Some(0)
            }
            "traceVisibility" => Some(12),
            _ => None,
        };
        if name == "traceVisibility" && pipelines.indirect_execution {
            let heads = &buffers.caches[(buffers.frame.load(Ordering::Relaxed) & 1) as usize].heads;
            let samples = u64::from(buffers.size.x) * u64::from(buffers.size.y);
            // Deferred scratch starts with the query count, then its arguments.
            let args_offset = heads.size() - deferred_trace_words(samples) * 4 + 4;
            context.command_encoder().copy_buffer_to_buffer(
                heads,
                args_offset,
                &buffers.trace_indirect,
                12,
                12,
            );
        }
        if matches!(name, "trace" | "secondary/trace") && pipelines.indirect_execution {
            let parity = (buffers.frame.load(Ordering::Relaxed) & 1) as usize;
            let heads = if name == "secondary/trace" {
                &buffers
                    .secondary
                    .as_ref()
                    .expect("secondary cache ready")
                    .caches[parity]
                    .heads
            } else {
                &buffers.caches[parity].heads
            };
            // Count, three dispatch arguments, and sample IDs precede the
            // deferred trace scratch at the end of the heads buffer.
            let samples = u64::from(buffers.size.x) * u64::from(buffers.size.y);
            let args_offset = heads.size() - (samples + 3 + deferred_trace_words(samples)) * 4;
            let copy_name = if name == "trace" {
                "traceArguments"
            } else {
                "secondary/traceArguments"
            };
            let copy_span = diagnostics.time_span(context.command_encoder(), copy_name);
            context.command_encoder().copy_buffer_to_buffer(
                heads,
                args_offset,
                &buffers.trace_indirect,
                0,
                12,
            );
            copy_span.end(context.command_encoder());
        }
        let span = diagnostics.time_span(context.command_encoder(), name);
        let mut pass = context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("split_rc"),
                ..default()
            });
        pass.set_pipeline(pipeline);
        let bind_group = match name.strip_prefix("secondary/") {
            Some(stage) => {
                let secondary = bindings
                    .secondary
                    .as_ref()
                    .expect("secondary bindings ready");
                if stage == "clear" {
                    let index = secondary_clear_dispatch_index.min(secondary.clear.len() - 1);
                    secondary_clear_dispatch_index += 1;
                    &secondary.clear[index]
                } else {
                    &secondary.compute[level]
                }
            }
            None => match name {
                "clear" => {
                    let index = clear_dispatch_index.min(bindings.clear.len() - 1);
                    clear_dispatch_index += 1;
                    &bindings.clear[index]
                }
                "trace" | "generateDirections" | "traceRecord" | "traceVisibility"
                | "traceResolve" | "traceOverflow" => {
                    bindings.trace.as_ref().unwrap_or(&bindings.compute[level])
                }
                // Ray-traced GI writes worldResolve's outputs.
                "worldResolve" | "rtgiSample" | "rtgiResolve" | "rtgiCacheRecord"
                | "rtgiCacheVisibility" | "rtgiCacheShade" => {
                    &bindings.world_resolve
                }
                _ => &bindings.compute[level],
            },
        };
        pass.set_bind_group(0, bind_group, &[]);
        if let Some(offset) = indirect_offset.filter(|_| pipelines.indirect_execution) {
            pass.dispatch_workgroups_indirect(&buffers.trace_indirect, offset);
        } else {
            pass.dispatch_workgroups(x, y, 1);
        }
        drop(pass);
        if let Some(diagnostics) = diagnostics.filter(|_| name == "packProbes") {
            let heads = &buffers.caches[(buffers.frame.load(Ordering::Relaxed) & 1) as usize].heads;
            let retained = u64::from(buffers.buckets) * 4;
            let rejected = (u64::from(buffers.buckets) * 2 + 2) * 4;
            diagnostics.record_u32(
                context.command_encoder(),
                &heads.slice(retained..retained + 4),
                format!("penumbra/view_{diagnostic_view}/retention_demand"),
            );
            let retention_dropped = (u64::from(buffers.buckets) * 2 + 1) * 4;
            diagnostics.record_u32(
                context.command_encoder(),
                &heads.slice(retention_dropped..retention_dropped + 4),
                format!("penumbra/view_{diagnostic_view}/retention_dropped"),
            );
            let bucket_collisions = (u64::from(buffers.buckets) * 2 + 3) * 4;
            diagnostics.record_u32(
                context.command_encoder(),
                &heads.slice(bucket_collisions..bucket_collisions + 4),
                format!("penumbra/view_{diagnostic_view}/previous_bucket_collisions"),
            );
            diagnostics.record_u32(
                context.command_encoder(),
                &heads.slice(rejected..rejected + 4),
                format!("penumbra/view_{diagnostic_view}/irradiance_cache_rejections"),
            );
        } else if let Some(diagnostics) = diagnostics.filter(|_| name == "commitCache") {
            let heads = &buffers.caches[(buffers.frame.load(Ordering::Relaxed) & 1) as usize].heads;
            let dropped = (u64::from(buffers.buckets) * 2 + 4) * 4;
            diagnostics.record_u32(
                context.command_encoder(),
                &heads.slice(dropped..dropped + 4),
                format!("penumbra/view_{diagnostic_view}/persistent_cache_dropped"),
            );
        } else if let Some(diagnostics) = diagnostics.filter(|_| name == "worldResolve") {
            let heads = &buffers.caches[(buffers.frame.load(Ordering::Relaxed) & 1) as usize].heads;
            for (offset, label) in [
                (5, "transport_query_attempts"),
                (6, "transport_queries_admitted"),
                (7, "selected_receiver_updates"),
                (8, "receiver_candidates"),
                (9, "directional_probe_count"),
            ] {
                let byte_offset = (u64::from(buffers.buckets) * 2 + offset) * 4;
                diagnostics.record_u32(
                    context.command_encoder(),
                    &heads.slice(byte_offset..byte_offset + 4),
                    format!("penumbra/view_{diagnostic_view}/{label}"),
                );
            }
        }
        span.end(context.command_encoder());
    };
    let image_x = buffers.size.x.div_ceil(8);
    let image_y = buffers.size.y.div_ceil(8);
    let shading_x = buffers.shading_size.x.div_ceil(8);
    let shading_y = buffers.shading_size.y.div_ceil(8);
    let bucket_groups = buffers.buckets.div_ceil(64);
    let samples = u64::from(buffers.size.x) * u64::from(buffers.size.y);
    let probe_directory = probe_directory_capacity(samples) as u32;
    let probe_directory_groups = probe_directory.div_ceil(64);
    let (pack_probe_groups_x, pack_probe_groups_y) = pack_probe_dispatch_groups(samples)
        .expect("the validated receiver count must fit the pack-probe dispatch grid");
    let clear_groups = buffers
        .buckets
        .max(probe_directory.saturating_mul(LEVELS as u32))
        .div_ceil(64);
    let ray_groups = (buffers.size.x * buffers.size.y).div_ceil(64);
    let trace_groups = ray_groups * TRACE_MAX_LANE_STRIDE;
    // Ray-traced GI replaces every cascade pass with path sampling and ReSTIR
    // GI reuse; the outputs feed the same temporal filter and composition.
    // Without hardware traversal the view falls back to the cascades.
    let ray_traced = settings.method == crate::GiMethod::RayTraced
        && settings.hardware_ray_queries
        && hardware
            .as_ref()
            .is_some_and(|hardware| hardware.ready_for(scene.revision))
        && compute[RTGI_SAMPLE].is_some()
        && compute[RTGI_RESOLVE].is_some()
        && compute[RTGI_CACHE_RECORD].is_some()
        && compute[RTGI_CACHE_VISIBILITY].is_some()
        && compute[RTGI_CACHE_SHADE].is_some();
    // The multi-bounce secondary cache belongs to the cascades.
    let secondary_enabled = bindings.secondary_enabled && !ray_traced;
    shaders.status.set_active_method(if ray_traced {
        crate::GiMethod::RayTraced
    } else {
        crate::GiMethod::SplitRadianceCascades
    });
    if ray_traced {
        shaders.status.set_hardware_ray_queries_active(true);
        // Empties the cascades' probe directory, so the composite's world-probe
        // fallback finds nothing instead of scanning stale probes, and resets
        // the per-frame counters.
        // shortcut: sub-resolve facets then get no GI; give them an RTGI fallback if it shows.
        dispatch(
            "clear",
            compute[0].expect("cascade pipeline ready"),
            0,
            clear_groups,
            1,
        );
        for (name, stage) in [("rtgiSample", RTGI_SAMPLE), ("rtgiResolve", RTGI_RESOLVE)] {
            dispatch(name, compute[stage].expect("checked"), 0, shading_x, shading_y);
        }
        // Shades cells queued by this frame's gather; the next gather reads them.
        let work_groups = (RTGI_CACHE_QUEUE + RTGI_CACHE_REFRESH).div_ceil(64) as u32;
        for (name, stage, groups) in [
            ("rtgiCacheRecord", RTGI_CACHE_RECORD, work_groups),
            (
                "rtgiCacheVisibility",
                RTGI_CACHE_VISIBILITY,
                work_groups * DEFERRED_VISIBILITY_QUERIES as u32,
            ),
            ("rtgiCacheShade", RTGI_CACHE_SHADE, work_groups),
        ] {
            dispatch(name, compute[stage].expect("checked"), 0, groups, 1);
        }
    } else {
        dispatch(
            "clear",
            compute[0].expect("cascade pipeline ready"),
            0,
            clear_groups,
            1,
        );
        dispatch(
            "seed",
            compute[1].expect("cascade pipeline ready"),
            0,
            image_x,
            image_y,
        );
        dispatch(
            "rank",
            compute[3].expect("cascade pipeline ready"),
            0,
            ray_groups,
            1,
        );
        for &phase in cold_rank_phases(settings.history_frames, history_was_valid) {
            dispatch(
                "rank",
                compute[3].expect("cascade pipeline ready"),
                phase,
                ray_groups,
                1,
            );
        }
        for level in 1..LEVELS {
            dispatch(
                "parents",
                compute[2].expect("cascade pipeline ready"),
                level,
                probe_directory_groups,
                1,
            );
        }
        for level in (0..LEVELS).rev() {
            dispatch(
                "prefix",
                compute[4].expect("cascade pipeline ready"),
                level,
                probe_directory_groups,
                1,
            );
        }
        dispatch(
            "generateDirections",
            compute[5].expect("cascade pipeline ready"),
            0,
            image_x,
            image_y,
        );
        dispatch(
            "clear",
            compute[0].expect("cascade pipeline ready"),
            0,
            clear_groups,
            1,
        );
        // Primary transport runs in passes that each hold at most one hit-shading
        // call: closest hits, recorded visibility queries, the queries in
        // parallel, then shading and deposit. The immediate kernel takes only
        // receivers beyond the deferred scratch capacity.
        let deferred_query_lanes = (u64::from(buffers.size.x) * u64::from(buffers.size.y))
            .min(DEFERRED_TRACE_RECEIVERS)
            * 2
            * DEFERRED_VISIBILITY_QUERIES
            * u64::from(TRACE_MAX_LANE_STRIDE);
        // Hardware traversal replaces only the closest-hit and visibility kernels,
        // and only once the TLAS describes this scene revision.
        let hardware_traversal = settings.hardware_ray_queries
            && hardware.is_some_and(|hardware| hardware.ready_for(scene.revision))
            && compute[HARDWARE_TRACE_DEFERRED].is_some()
            && compute[HARDWARE_TRACE_VISIBILITY].is_some();
        shaders
            .status
            .set_hardware_ray_queries_active(hardware_traversal);
        let (closest_hit_stage, visibility_stage) = if hardware_traversal {
            (HARDWARE_TRACE_DEFERRED, HARDWARE_TRACE_VISIBILITY)
        } else {
            (17, 19)
        };
        for (name, stage, groups) in [
            ("trace", closest_hit_stage, trace_groups),
            ("traceRecord", 18, trace_groups),
            (
                "traceVisibility",
                visibility_stage,
                deferred_query_lanes.div_ceil(64) as u32,
            ),
            ("traceResolve", 20, trace_groups),
            ("traceOverflow", 7, trace_groups),
        ] {
            dispatch(
                name,
                compute[stage].expect("radiance cascade trace pipeline ready"),
                0,
                groups,
                1,
            );
        }
        if secondary_enabled {
            for (name, stage, x, y) in [
                ("secondary/clear", 0, clear_groups, 1),
                ("secondary/seed", 1, image_x, image_y),
                ("secondary/rank", 3, ray_groups, 1),
            ] {
                dispatch(
                    name,
                    compute[stage].expect("secondary pipeline ready"),
                    0,
                    x,
                    y,
                );
            }
            for &phase in cold_rank_phases(settings.history_frames.max(1), history_was_valid) {
                dispatch(
                    "secondary/rank",
                    compute[3].expect("secondary pipeline ready"),
                    phase,
                    ray_groups,
                    1,
                );
            }
            for level in 1..LEVELS {
                dispatch(
                    "secondary/parents",
                    compute[2].expect("secondary pipeline ready"),
                    level,
                    probe_directory_groups,
                    1,
                );
            }
            for level in (0..LEVELS).rev() {
                dispatch(
                    "secondary/prefix",
                    compute[4].expect("secondary pipeline ready"),
                    level,
                    probe_directory_groups,
                    1,
                );
            }
            for (name, stage, x, y) in [
                ("secondary/generateDirections", 5, image_x, image_y),
                ("secondary/clear", 0, clear_groups, 1),
                ("secondary/trace", 7, trace_groups, 1),
                ("secondary/retain", 6, bucket_groups, 1),
                ("secondary/reduce", 8, bucket_groups, 1),
            ] {
                dispatch(
                    name,
                    compute[stage].expect("secondary pipeline ready"),
                    0,
                    x,
                    y,
                );
            }
            let secondary_work = cache_work_slots(samples);
            let secondary_entries =
                secondary_work + CACHE_HISTORY_PER_BUCKET * u64::from(buffers.buckets);
            for level in (0..LEVELS).rev() {
                dispatch(
                    "secondary/merge",
                    compute[9].expect("secondary pipeline ready"),
                    level,
                    secondary_entries.div_ceil(64 * LEVELS as u64) as u32,
                    LEVELS as u32,
                );
            }
            for (name, stage, x, y) in [
                (
                    "secondary/seedProbes",
                    10,
                    secondary_work.div_ceil(64 * LEVELS as u64) as u32,
                    1,
                ),
                (
                    "secondary/packProbes",
                    11,
                    pack_probe_groups_x,
                    pack_probe_groups_y,
                ),
            ] {
                dispatch(
                    name,
                    compute[stage].expect("secondary pipeline ready"),
                    0,
                    x,
                    y,
                );
            }
        }
        dispatch(
            "retain",
            compute[6].expect("cascade pipeline ready"),
            0,
            bucket_groups,
            1,
        );
        dispatch(
            "reduce",
            compute[8].expect("cascade pipeline ready"),
            0,
            bucket_groups,
            1,
        );
        let current_capacity = cache_work_slots(samples);
        let entry_count = current_capacity + CACHE_HISTORY_PER_BUCKET * u64::from(buffers.buckets);
        for level in (0..LEVELS).rev() {
            dispatch(
                "merge",
                compute[9].expect("cascade pipeline ready"),
                level,
                entry_count.div_ceil(64 * LEVELS as u64) as u32,
                LEVELS as u32,
            );
        }
        for (name, stage, x, y) in [
            (
                "seedProbes",
                10,
                current_capacity.div_ceil(64 * LEVELS as u64) as u32,
                1,
            ),
            ("packProbes", 11, pack_probe_groups_x, pack_probe_groups_y),
        ] {
            dispatch(
                name,
                compute[stage].expect("directional probe pipeline ready"),
                0,
                x,
                y,
            );
        }
        dispatch(
            "worldResolve",
            compute[16].expect("world field resolve pipeline ready"),
            0,
            shading_x,
            shading_y,
        );
    }
    let frame = buffers.frame.load(Ordering::Relaxed);
    // Copy the geometry-aware sparse reconstruction before temporal clamping,
    // filtering, albedo, AO, intensity and exposure.
    let raw_output_copied = if let Some(output) =
        raw_output.and_then(|output| images.get(&output.0))
    {
        let descriptor = &output.texture_descriptor;
        if descriptor.format == TextureFormat::Rgba32Float
            && output.size_2d() == buffers.shading_size
            && descriptor.size.depth_or_array_layers == 1
            && descriptor.sample_count == 1
            && descriptor.usage.contains(TextureUsages::COPY_DST)
        {
            context.command_encoder().copy_texture_to_texture(
                buffers.irradiance_textures[3].as_image_copy(),
                output.texture.as_image_copy(),
                Extent3d {
                    width: buffers.shading_size.x,
                    height: buffers.shading_size.y,
                    depth_or_array_layers: 1,
                },
            );
            true
        } else {
            warn_once!(
                "RadianceCascadesRawOutput requires a COPY_DST Rgba32Float image matching the active GI resolve size"
            );
            false
        }
    } else {
        false
    };
    let temporal_span = diagnostics.time_span(context.command_encoder(), "temporal");
    let mut pass = context
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("split_rc_temporal"),
            ..default()
        });
    pass.set_pipeline(compute[12].expect("temporal pipeline ready"));
    pass.set_bind_group(0, &bindings.temporal, &[]);
    pass.dispatch_workgroups(shading_x, shading_y, 1);
    drop(pass);

    // Only temporal results become history; spatially filtered display values
    // never feed back and spread lighting recursively across contacts.
    context.command_encoder().copy_texture_to_texture(
        buffers.irradiance_textures[2].as_image_copy(),
        buffers.irradiance_textures[(frame & 1) as usize].as_image_copy(),
        Extent3d {
            width: buffers.shading_size.x,
            height: buffers.shading_size.y,
            depth_or_array_layers: 1,
        },
    );
    temporal_span.end(context.command_encoder());
    if let Some(diagnostics) = diagnostics.filter(|_| raw_output.is_some()) {
        let heads = &buffers.caches[(frame & 1) as usize].heads;
        // Confidence is fractional observed kernel mass, not a frame count.
        const CONFIDENCE: [&str; 8] = [
            "none",
            "up_to_one",
            "1_to_4",
            "4_to_8",
            "8_to_16",
            "16_to_32",
            "32_to_64",
            "64_plus",
        ];
        const LUMINANCE: [&str; 9] = [
            "unsupported",
            "invalid",
            "black",
            "below_1",
            "1_10",
            "10_100",
            "100_1000",
            "1000_10000",
            "10000_plus",
        ];
        for word in 0..HISTORY_DIAGNOSTIC_WORDS {
            let (category, bin) = if word < 8 {
                ("history_length", CONFIDENCE[word as usize])
            } else if word < 17 {
                ("raw_luminance", LUMINANCE[(word - 8) as usize])
            } else {
                ("temporal_luminance", LUMINANCE[(word - 17) as usize])
            };
            let offset = (u64::from(buffers.buckets) * 2 + 10 + word) * 4;
            diagnostics.record_u32(
                context.command_encoder(),
                &heads.slice(offset..offset + 4),
                format!("penumbra/view_{diagnostic_view}/diffuse/{category}/{bin}"),
            );
        }
    }
    let steps = buffers.settings.spatial_filter_steps;
    for level in 0..steps as usize {
        let span = diagnostics.time_span(
            context.command_encoder(),
            ["denoise_0", "denoise_1", "denoise_2", "denoise_3"][level],
        );
        let mut pass = context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("split_rc_denoise"),
                ..default()
            });
        pass.set_pipeline(compute[13].expect("denoise pipeline ready"));
        pass.set_bind_group(0, &bindings.denoise[level], &[]);
        pass.dispatch_workgroups(shading_x, shading_y, 1);
        drop(pass);

        span.end(context.command_encoder());
    }

    // Add diffuse light to the current HDR attachment without swapping the view target.
    let span = diagnostics.time_span(context.command_encoder(), "compositeAnchors");
    let mut pass = context
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("split_rc_composite_anchors"),
            ..default()
        });
    pass.set_pipeline(composite_anchors);
    pass.set_bind_group(0, &bindings.composite_anchors, &[]);
    pass.dispatch_workgroups(shading_x, shading_y, 1);
    drop(pass);
    span.end(context.command_encoder());

    let composition_span = diagnostics.time_span(context.command_encoder(), "composition");
    let mut pass = context
        .command_encoder()
        .begin_render_pass(&RenderPassDescriptor {
            label: Some("split_rc_composite"),
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
    pass.set_bind_group(0, &bindings.composite, &[]);
    pass.draw(0..3, 0..1);
    drop(pass);
    composition_span.end(context.command_encoder());
    if required & (1 << CACHE_COMMIT) != 0 && !ray_traced {
        // Publish once, after composition has finished reading the live table.
        // Publication changes heads from working-entry to history-entry offsets.
        for (name, stage, groups) in [
            ("commitCache", CACHE_COMMIT, bucket_groups),
            ("remapProbeHeads", 15, probe_directory_groups),
        ] {
            if skip_passes.iter().any(|skipped| skipped == name) {
                continue;
            }
            let span = diagnostics.time_span(context.command_encoder(), name);
            let mut pass = context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some(name),
                    ..default()
                });
            pass.set_pipeline(compute[stage].expect("cache retention pipeline ready"));
            pass.set_bind_group(0, &bindings.compute[0], &[]);
            pass.dispatch_workgroups(groups, 1, 1);
            drop(pass);
            if name == "commitCache" {
                let heads = &buffers.caches[(frame & 1) as usize].heads;
                let dropped = (u64::from(buffers.buckets) * 2 + 4) * 4;
                diagnostics.record_u32(
                    context.command_encoder(),
                    &heads.slice(dropped..dropped + 4),
                    format!("penumbra/view_{diagnostic_view}/persistent_cache_dropped"),
                );
            }
            span.end(context.command_encoder());
        }
    }
    if secondary_enabled {
        let secondary = bindings
            .secondary
            .as_ref()
            .expect("secondary bindings ready");
        for (name, stage, groups) in [
            ("secondary/commitCache", CACHE_COMMIT, bucket_groups),
            ("secondary/remapProbeHeads", 15, probe_directory_groups),
        ] {
            if skip_passes.iter().any(|skipped| skipped == name) {
                continue;
            }
            let span = diagnostics.time_span(context.command_encoder(), name);
            let mut pass = context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some(name),
                    ..default()
                });
            pass.set_pipeline(compute[stage].expect("secondary retention pipeline ready"));
            pass.set_bind_group(0, &secondary.compute[0], &[]);
            pass.dispatch_workgroups(groups, 1, 1);
            drop(pass);
            span.end(context.command_encoder());
        }
    }
    total_span.end(context.command_encoder());
    let dispatch =
        if buffers.pending_lights.is_none() && buffers.light_revision == scene.light_revision {
            Some(shaders.status.record_dispatch(RadianceCascadesRevisions {
                scene: scene.revision,
                geometry: scene.geometry_revision,
                lighting: scene.light_revision,
            }))
        } else {
            None
        };
    if raw_output.is_some() {
        let raw_frame = dispatch.filter(|_| raw_output_copied).map(|dispatch| {
            crate::RadianceCascadesRawFrame {
                frame_id: u64::from(frame),
                dispatch,
                submitted_at: std::time::Instant::now(),
            }
        });
        shaders.status.record_raw_frame(raw_frame);
    }
    *buffers
        .rendered_frame
        .lock()
        .unwrap_or_else(|error| error.into_inner()) = bindings.frame;
    // Publish only after this view's GI work was encoded. Skipped frames must
    // keep the old snapshot so their geometry changes cannot escape invalidation.
    let mut previous_scene = buffers
        .scene_history
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if gradual_directional_motion(&previous_scene, &bindings.scene_history) {
        buffers
            .directional_motion_epoch
            .store(frame.wrapping_add(1).max(1), Ordering::Relaxed);
    }
    if gradual_punctual_motion(&previous_scene, &bindings.scene_history) {
        buffers
            .punctual_motion_epoch
            .store(frame.wrapping_add(1).max(1), Ordering::Relaxed);
    }
    *previous_scene = bindings.scene_history.clone();
    if buffers.frame.fetch_add(1, Ordering::Relaxed) == 0 {
        info!("First radiance-cascades frame submitted");
    }
    buffers.history_valid.store(true, Ordering::Relaxed);
    if let Some(secondary) = &buffers.secondary {
        secondary.valid.store(secondary_enabled, Ordering::Relaxed);
    }
    buffers.screen_history_valid.store(true, Ordering::Relaxed);
    buffers
        .candidate_schedule
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .advance_after_successful_update(!cold_priority);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_bounce_accounts_for_cache_memory_and_shares_query_limit() {
        let size = UVec2::new(16, 8);
        let shading = UVec2::new(32, 16);
        let primary = RadianceCascades::default();
        let secondary = RadianceCascades {
            multi_bounce: true,
            ..primary
        };
        let samples = u64::from(size.x) * u64::from(size.y);
        let buckets = cache_bucket_count(samples, secondary);
        let extra = irradiance_probe_bytes(samples, true) - irradiance_probe_bytes(samples, false)
            + cache_head_bytes(buckets, samples) * 2
            + cache_entry_bytes(samples, buckets)
            + cache_history_bytes(buckets)
            + samples * 32
            + (LEVELS as u64 + 2) * std::mem::size_of::<Parameters>() as u64;
        assert_eq!(
            view_bytes(size, shading, secondary) - view_bytes(size, shading, primary),
            extra
        );
        assert_eq!(cache_query_limits(0, true), (0, 0));
        assert_eq!(cache_query_limits(1, true), (1, 0));
        assert_eq!(cache_query_limits(11, true), (6, 5));
        assert_eq!(cache_query_limits(11, false), (11, 11));
        for cap in [1, 2, 3, 7, 8, 9, 16, 100] {
            let mut settings = secondary;
            settings.sampling.max_cache_entries = cap;
            let buckets = cache_bucket_count(samples, settings);
            let (primary_limit, secondary_limit) = cache_query_limits(cap, true);
            let quota = |limit: u32| (u64::from(limit) / buckets).clamp(1, 4);
            let retained = buckets
                * (quota(primary_limit)
                    + if secondary_limit > 0 {
                        quota(secondary_limit)
                    } else {
                        0
                    });
            assert!(retained <= u64::from(cap), "cap {cap}: retained {retained}");
        }
    }

    #[test]
    fn exposure_changes_invalidate_world_history() {
        assert!(exposure_invalidates_world_history(true));
        assert!(!exposure_invalidates_world_history(false));
    }

    #[test]
    fn light_upload_ranges_skip_unchanged_receiver_and_environment_records() {
        let previous = vec![GiLight::default(); 6];
        let mut next = previous.clone();
        next[0].position.x = 4.0;
        next[4].color.y = 2.0;
        assert_eq!(
            ViewBuffers::changed_light_ranges(&previous, &next),
            Some(vec![0..1, 4..5])
        );
        assert_eq!(
            ViewBuffers::changed_light_ranges(&previous, &previous),
            Some(Vec::new())
        );
        assert_eq!(
            ViewBuffers::changed_light_ranges(&previous, &next[..5]),
            None
        );
    }

    #[test]
    fn split_world_cache_capacity_does_not_follow_screen_resolution() {
        let split = RadianceCascades::low_end();
        for samples in [1, 300, 14_400, 131_072] {
            assert_eq!(cache_bucket_count(samples, split), 16_384);
        }
        let irregular_cap = RadianceCascades {
            sampling: crate::SamplingSettings {
                max_cache_entries: 65_535,
                ..split.sampling
            },
            ..split
        };
        assert_eq!(cache_bucket_count(300, irregular_cap), 8_192);
        let derived = RadianceCascades {
            sampling: crate::SamplingSettings {
                max_cache_entries: 0,
                ..split.sampling
            },
            ..split
        };
        assert_eq!(cache_bucket_count(300, derived), 1_024);
    }

    #[test]
    fn compute_layout_is_in_binding_order() {
        // The SPIR-V passthrough binding map relies on it; see compute_layout.
        for ray_query in [false, true] {
            assert!(compute_layout(ray_query)
                .entries
                .is_sorted_by_key(|entry| entry.binding));
        }
    }

    #[test]
    fn compute_layout_fits_thirteen_storage_buffers() {
        let count = compute_layout(true)
            .entries
            .iter()
            .filter(|entry| {
                matches!(
                    entry.ty,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { .. },
                        ..
                    }
                )
            })
            .count();
        assert_eq!(count, 13);
    }

    #[test]
    fn split_diffuse_output_tracks_the_resolve_extent() {
        let viewport = UVec2::new(512, 512);
        let settings = RadianceCascades {
            sampling: crate::SamplingSettings {
                resolve_scale: 4,
                ..default()
            },
            ..default()
        };
        assert_eq!(resolve_size(viewport, settings), UVec2::new(128, 128));
    }

    #[test]
    fn cascade_allocations_fit_storage_limits_after_resolution_reduction() {
        let limits = WgpuLimits {
            max_storage_buffer_binding_size: 4 * 1024 * 1024,
            ..default()
        };
        let viewport = UVec2::new(1920, 1080);
        let mut settings = RadianceCascades {
            max_rays: 131_072,
            resolution_divisor: 1,
            ..default()
        };
        assert!(!view_fits_device(
            resolution(viewport, settings),
            viewport,
            viewport,
            settings,
            &limits
        ));
        settings.sampling.max_receiver_cells = view_device_max_rays(viewport, settings, &limits);
        assert!(view_fits_device(
            resolution(viewport, settings),
            viewport,
            viewport,
            settings,
            &limits
        ));
        // Several cell caps produce the same integer screen grid. Admission
        // limits the allocated receivers, not the upper end of that plateau.
        let size = resolution(viewport, settings);
        let pixels = u64::from(size.x) * u64::from(size.y);
        let shading_pixels = u64::from(viewport.x) * u64::from(viewport.y);
        let buckets = cache_bucket_count(pixels, settings);
        let binding_limit = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        assert!(pixels <= u64::from(settings.sampling.max_receiver_cells));
        assert!(view_bytes(size, viewport, settings) >= (shading_pixels * 3 + 6) * 16);
        assert!(cache_entry_bytes(pixels, buckets) <= binding_limit);
        assert!(cache_history_bytes(buckets) <= binding_limit);
        assert!(cache_head_bytes(buckets, pixels) <= binding_limit);
        assert!(irradiance_probe_bytes(pixels, false) <= binding_limit);
        let insufficient = WgpuLimits {
            max_storage_buffer_binding_size: 1024,
            ..limits
        };
        assert!(!view_fits_device(
            UVec2::ONE,
            UVec2::ONE,
            UVec2::ONE,
            settings,
            &insufficient
        ));
    }

    #[test]
    fn obsolete_scene_cannot_deadlock_a_replacement_that_fits() {
        assert!(discard_obsolete_scene(247, 163, 215, true));
        assert!(!discard_obsolete_scene(247, 216, 215, true));
        assert!(!discard_obsolete_scene(215, 163, 215, true));
        assert!(!discard_obsolete_scene(247, 163, 215, false));
    }

    #[test]
    #[ignore = "requires slangc and a desktop GPU; no demo scene is launched"]
    fn split_cache_clear_resets_only_the_root_ray_offset() {
        use bevy::render::{
            renderer::initialize_renderer,
            settings::{Backends, WgpuSettings},
        };
        let options = WgpuSettings::default();
        let resources = bevy::tasks::block_on(initialize_renderer(
            options.backends.unwrap_or(Backends::PRIMARY),
            None,
            &options,
        ));
        let device = &resources.0;
        let queue = &resources.1;
        let shader = compile(&SlangCompiler::default(), "cascades.slang", &["clear"])
            .expect("compile production cache clear");
        let bevy::shader::Source::Wgsl(source) = shader.source else {
            panic!("expected WGSL");
        };
        let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
            label: Some("split_cache_clear_check"),
            source: ShaderSource::Wgsl(source),
        });
        let pipeline = device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("split_cache_clear_check"),
            layout: None,
            module: &module,
            entry_point: Some("clear"),
            compilation_options: default(),
            cache: None,
        });
        const BUCKETS: u32 = 1024;
        let bytes = cache_head_bytes(u64::from(BUCKETS), 1);
        let heads = device.create_buffer(&BufferDescriptor {
            label: Some("split_clear_heads"),
            size: bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let irradiance = device.create_texture(&TextureDescriptor {
            label: Some("split_clear_irradiance"),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let irradiance_view = irradiance.create_view(&TextureViewDescriptor::default());
        queue.write_buffer(&heads, 0, &vec![0xabu8; bytes as usize]);
        let mut params = UniformBuffer::from(Parameters {
            size: UVec4::new(1, 1, 1, BUCKETS),
            owner_sequence: UVec4::new(1, 1, 1, SPARSE_HISTORY_RESET_FLAG),
            interval_history: Vec4::new(0.0, 0.0, 0.25, 0.0),
            ..default()
        });
        params.write_buffer(device, queue);
        let group = device.create_bind_group(
            "split_clear_check",
            &pipeline.get_bind_group_layout(0).into(),
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: params.binding().unwrap(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: heads.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 10,
                    resource: BindingResource::TextureView(&irradiance_view),
                },
            ],
        );
        let output = crate::scene::traversal_tests::dispatch_readback(
            device,
            queue,
            &pipeline,
            &group,
            &heads,
            bytes,
            BUCKETS.div_ceil(64),
        );
        let words: Vec<_> = output
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| u32::from_le_bytes(*bytes))
            .collect();
        // Both full hash tables must be empty, including the last irradiance slot.
        assert!(
            words[..BUCKETS as usize]
                .iter()
                .all(|&word| word == u32::MAX)
        );
        assert_eq!(words[BUCKETS as usize], 0);
        assert!(
            words[BUCKETS as usize + 1..=2 * BUCKETS as usize]
                .iter()
                .all(|&word| word == u32::MAX)
        );
        let counters_start = 2 * BUCKETS as usize + 1;
        let candidate_count_offset = 2 * BUCKETS as usize + 8;
        let directories_start = 2 * BUCKETS as usize + 10 + HISTORY_DIAGNOSTIC_WORDS as usize;
        let directory_words = LEVELS * probe_directory_capacity(1) as usize;
        let directory_counts_start = directories_start + directory_words;
        assert!(
            words[counters_start..candidate_count_offset]
                .iter()
                .all(|&word| word == 0)
        );
        assert_eq!(words[candidate_count_offset], 1);
        assert!(
            words[candidate_count_offset + 1..directories_start]
                .iter()
                .all(|&word| word == 0)
        );
        assert!(
            words[directories_start..directory_counts_start]
                .iter()
                .all(|&word| word == u32::MAX)
        );
        assert!(
            words[directory_counts_start..directory_counts_start + directory_words]
                .iter()
                .all(|&word| word == 0)
        );

        // After z=1 reset, z=0 preserves the root-ray offset.
        let attempts_offset = u64::from(2 * BUCKETS + 5) * 4;
        let admitted_offset = u64::from(2 * BUCKETS + 6) * 4;
        queue.write_buffer(&heads, attempts_offset, &17u32.to_le_bytes());
        queue.write_buffer(&heads, admitted_offset, &11u32.to_le_bytes());
        let sentinel = 0x1234_5678u32;
        queue.write_buffer(&heads, bytes - 4, &sentinel.to_le_bytes());
        params.get_mut().owner_sequence.z = 0;
        params.write_buffer(device, queue);
        let counter_output = crate::scene::traversal_tests::dispatch_readback(
            device,
            queue,
            &pipeline,
            &group,
            &heads,
            bytes,
            BUCKETS.div_ceil(64),
        );
        let counter_words: Vec<_> = counter_output
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| u32::from_le_bytes(*bytes))
            .collect();
        assert_eq!(counter_words[2 * BUCKETS as usize + 5], 17);
        assert_eq!(counter_words[2 * BUCKETS as usize + 6], 11);
        assert_eq!(counter_words[bytes as usize / 4 - 1], sentinel);
    }

    #[test]
    #[ignore = "requires slangc and a desktop GPU; no demo scene is launched"]
    fn split_cache_commit_bounds_entries_and_remaps_probe_heads() {
        use bevy::render::{
            renderer::initialize_renderer,
            settings::{Backends, WgpuSettings},
        };
        const BUCKETS: u32 = 2;
        const FRAME: u32 = 7;
        const ENTRY_WORDS: usize = ENTRY_SIZE as usize / 4;
        let options = WgpuSettings::default();
        let resources = bevy::tasks::block_on(initialize_renderer(
            options.backends.unwrap_or(Backends::PRIMARY),
            None,
            &options,
        ));
        let device = &resources.0;
        let queue = &resources.1;
        let make_pipeline = |entry| {
            let shader = compile(&SlangCompiler::default(), "cascades.slang", &[entry]).unwrap();
            let bevy::shader::Source::Wgsl(source) = shader.source else {
                panic!("expected WGSL");
            };
            let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
                label: Some(entry),
                source: ShaderSource::Wgsl(source),
            });
            device.create_compute_pipeline(&RawComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: default(),
                cache: None,
            })
        };
        let commit = make_pipeline("commitCache");
        let remap = make_pipeline("remapProbeHeads");
        let work_bytes = cache_entry_bytes(1, u64::from(BUCKETS));
        let history_bytes = cache_history_bytes(u64::from(BUCKETS));
        let heads_bytes = cache_head_bytes(u64::from(BUCKETS), 1);
        let directory_base = (2 * u64::from(BUCKETS) + 10 + HISTORY_DIAGNOSTIC_WORDS) as usize;
        let directory_counts_base = directory_base + LEVELS * probe_directory_capacity(1) as usize;
        let run_case = |name: &str,
                        frame: u32,
                        last_samples: [u32; 3],
                        interval_history_enabled: bool,
                        history_frames: f32,
                        sample_quota: u32,
                        expected_keys: [u32; 2],
                        retained_count: usize,
                        expected_probe_heads: [u32; 2],
                        expected_overflow: u32,
                        expected_source_slots: [u32; 3]| {
            let heads = device.create_buffer(&BufferDescriptor {
                label: Some("split_commit_heads"),
                size: heads_bytes,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let work = device.create_buffer(&BufferDescriptor {
                label: Some("split_commit_work"),
                size: work_bytes,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let history = device.create_buffer(&BufferDescriptor {
                label: Some("split_commit_history"),
                size: history_bytes,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut head_words = vec![u32::MAX; (heads_bytes / 4) as usize];
            head_words[BUCKETS as usize] = 0;
            head_words[2 * BUCKETS as usize + 1..directory_base].fill(0);
            head_words[directory_counts_base..].fill(0);
            head_words[0] = 0;
            // The first level-zero directory slots point at descriptors 2 and 0.
            // remapProbeHeads follows their commit remap markers in place.
            head_words[directory_base] = 2;
            head_words[directory_base + 1] = 0;
            let head_data: Vec<_> = head_words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect();
            queue.write_buffer(&heads, 0, &head_data);

            let mut entries = vec![0u32; (work_bytes / 4) as usize];
            for (index, last_sample) in last_samples.into_iter().enumerate() {
                let base = index * ENTRY_WORDS;
                entries[base] = 10 + index as u32;
                entries[base + 4] = 0x1000 + index as u32;
                entries[base + 5] = 0x2000 + index as u32;
                entries[base + 16] = if index == 2 {
                    u32::MAX
                } else {
                    index as u32 + 1
                };
                entries[base + 17] = frame | 0x8000_0000;
                entries[base + 18] = last_sample;
            }
            let entry_data: Vec<_> = entries.iter().flat_map(|word| word.to_le_bytes()).collect();
            queue.write_buffer(&work, 0, &entry_data);
            let empty_history = vec![u32::MAX; (history_bytes / 4) as usize];
            let empty_history_data: Vec<_> = empty_history
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect();
            queue.write_buffer(&history, 0, &empty_history_data);

            let mut uniform = UniformBuffer::from(Parameters {
                size: UVec4::new(1, 1, 1, BUCKETS),
                history: Vec4::new(0.9, 0.0, 1.0, history_frames),
                interval_history: Vec4::new(
                    0.9,
                    if interval_history_enabled { 1.0 } else { 0.0 },
                    0.0,
                    0.0,
                ),
                frame: UVec4::new(frame, 0, 0, 0),
                sampling: UVec4::new(0, 0, sample_quota, 0),
                ..default()
            });
            uniform.write_buffer(device, queue);
            let make_group = |pipeline: &ComputePipeline, include_history: bool| {
                let mut bindings = vec![
                    BindGroupEntry {
                        binding: 0,
                        resource: uniform.binding().unwrap(),
                    },
                    BindGroupEntry {
                        binding: 6,
                        resource: heads.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 7,
                        resource: work.as_entire_binding(),
                    },
                ];
                if include_history {
                    bindings.push(BindGroupEntry {
                        binding: 9,
                        resource: history.as_entire_binding(),
                    });
                }
                device.create_bind_group(
                    "split_cache_commit_check",
                    &pipeline.get_bind_group_layout(0).into(),
                    &bindings,
                )
            };
            let commit_group = make_group(&commit, true);
            let remap_group = make_group(&remap, true);
            let readback_bytes = heads_bytes + history_bytes + work_bytes;
            let readback = device.create_buffer(&BufferDescriptor {
                label: Some("split_commit_readback"),
                size: readback_bytes,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
            for (pipeline, group) in [(&commit, &commit_group), (&remap, &remap_group)] {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&heads, 0, &readback, 0, heads_bytes);
            encoder.copy_buffer_to_buffer(&history, 0, &readback, heads_bytes, history_bytes);
            encoder.copy_buffer_to_buffer(
                &work,
                0,
                &readback,
                heads_bytes + history_bytes,
                work_bytes,
            );
            let submission = queue.submit([encoder.finish()]);
            let (sender, receiver) = std::sync::mpsc::channel();
            readback
                .slice(..)
                .map_async(MapMode::Read, move |result| sender.send(result).unwrap());
            device
                .poll(PollType::Wait {
                    submission_index: Some(submission),
                    timeout: Some(std::time::Duration::from_secs(30)),
                })
                .unwrap();
            receiver
                .recv_timeout(std::time::Duration::from_secs(1))
                .unwrap()
                .unwrap();
            let output = readback
                .slice(..)
                .get_mapped_range()
                .expect("readback buffer mapping should be valid")
                .to_vec();
            readback.unmap();
            let words: Vec<u32> = output
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| u32::from_le_bytes(*word))
                .collect();
            let heads_out = &words[..(heads_bytes / 4) as usize];
            let history_out =
                &words[(heads_bytes / 4) as usize..((heads_bytes + history_bytes) / 4) as usize];
            let work_out = &words[((heads_bytes + history_bytes) / 4) as usize..];

            let expected_head = if retained_count == 0 {
                u32::MAX
            } else {
                retained_count as u32 - 1
            };
            assert_eq!(heads_out[0], expected_head, "{name}: retained chain head");
            assert_eq!(
                &heads_out[directory_base..directory_base + 2],
                &expected_probe_heads[..],
                "{name}: source probe heads remap to retained slots"
            );
            assert_eq!(
                heads_out[(2 * BUCKETS + 4) as usize],
                expected_overflow,
                "{name}: over-quota eligible entries"
            );
            assert_eq!(
                work_out[22], expected_source_slots[0],
                "{name}: key 10 source slot"
            );
            assert_eq!(
                work_out[ENTRY_WORDS + 22],
                expected_source_slots[1],
                "{name}: key 11 source slot"
            );
            assert_eq!(
                work_out[2 * ENTRY_WORDS + 22],
                expected_source_slots[2],
                "{name}: key 12 source slot"
            );
            for (retained, &expected_key) in expected_keys.iter().take(retained_count).enumerate() {
                let base = retained * ENTRY_WORDS;
                let source = (expected_key - 10) as usize;
                assert_eq!(
                    history_out[base], expected_key,
                    "{name}: retained key order"
                );
                assert_eq!(
                    history_out[base + 4],
                    0x1000 + source as u32,
                    "{name}: radiance payload preserved"
                );
                assert_eq!(
                    history_out[base + 5],
                    0x2000 + source as u32,
                    "{name}: interval payload preserved"
                );
                assert_eq!(
                    history_out[base + 16],
                    if retained == 0 {
                        u32::MAX
                    } else {
                        retained as u32 - 1
                    },
                    "{name}: retained history links"
                );
            }
            for retained in retained_count..2 {
                assert_eq!(
                    history_out[retained * ENTRY_WORDS],
                    u32::MAX,
                    "{name}: unused history slot is empty"
                );
            }
        };

        // Ordinary Split keeps its finite history window and key-order behavior.
        run_case(
            "finite ttl",
            FRAME,
            [FRAME, FRAME, FRAME - 5],
            false,
            4.0,
            0,
            [10, 11],
            2,
            [u32::MAX - 1, 0],
            0,
            [0, 1, u32::MAX],
        );
        // Persistent Split retains recent world keys instead of expiring them
        // because camera visibility did not schedule a fresh sample.
        run_case(
            "persistent world intervals",
            1_000,
            [3, 5, 997],
            true,
            4.0,
            BUCKETS * 2,
            [12, 11],
            2,
            [0, u32::MAX - 1],
            1,
            [u32::MAX, 0x8000_0001, 0x8000_0000],
        );
        run_case(
            "finite ttl with matching ages",
            1_000,
            [3, 5, 997],
            false,
            4.0,
            BUCKETS * 2,
            [12, u32::MAX],
            1,
            [0, u32::MAX - 1],
            0,
            [u32::MAX, u32::MAX, 0],
        );
        // Even persistent mode needs a positive history horizon; zero is the
        // explicit switch that disables retention and must leave no stale links.
        run_case(
            "zero history horizon",
            1_000,
            [3, 5, 997],
            true,
            0.0,
            BUCKETS * 2,
            [u32::MAX; 2],
            0,
            [u32::MAX - 1; 2],
            0,
            [u32::MAX; 3],
        );
        // Dispatch the production remap against post-commit storage: the old
        // working representative is gone, but an exact sibling may survive.
        for (name, sibling_present, expected) in [
            ("surviving sibling direction", true, 0),
            ("all directions evicted", false, u32::MAX - 1),
        ] {
            let storage = |label, words: &[u32]| {
                let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
                device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some(label),
                    contents: &bytes,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                })
            };
            let mut head_words = vec![u32::MAX; (heads_bytes / 4) as usize];
            if sibling_present {
                head_words[..BUCKETS as usize].fill(0);
            }
            head_words[directory_counts_base..].fill(0);
            head_words[directory_base] = 0;
            let mut representative = vec![0u32; ENTRY_WORDS];
            representative[0] = 10;
            representative[3] = 1 << 18; // LOD 1, level 0, direction 0.
            representative[22] = u32::MAX;
            let mut sibling = representative.clone();
            sibling[3] |= 7; // A different direction in the same grid cell and LOD.
            sibling[16] = u32::MAX;
            sibling[22] = if sibling_present { 0 } else { u32::MAX };
            let heads = storage("remap_sibling_heads", &head_words);
            let work = storage("remap_sibling_work", &representative);
            let history = storage("remap_sibling_history", &sibling);
            let mut uniform = UniformBuffer::from(Parameters {
                size: UVec4::new(1, 1, 1, BUCKETS),
                interval_history: Vec4::new(0.98, 1.0, 0.0, 0.0),
                ..default()
            });
            uniform.write_buffer(device, queue);
            let group = device.create_bind_group(
                "remap_sibling_check",
                &remap.get_bind_group_layout(0).into(),
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: uniform.binding().unwrap(),
                    },
                    BindGroupEntry {
                        binding: 6,
                        resource: heads.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 7,
                        resource: work.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 9,
                        resource: history.as_entire_binding(),
                    },
                ],
            );
            let bytes = crate::scene::traversal_tests::dispatch_readback(
                device,
                queue,
                &remap,
                &group,
                &heads,
                heads_bytes,
                1,
            );
            let offset = directory_base * 4;
            assert_eq!(
                u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()),
                expected,
                "{name}: remapped directory representative"
            );
        }
    }

    #[test]
    fn gi_scene_waits_for_matching_deformation_and_tlas_refit() {
        let mut scene = GpuScene {
            revision: 7,
            ..default()
        };
        assert!(scene.ready_for_revision(7));
        assert!(!scene.ready_for_revision(8));
        scene.pending_deformation = true;
        assert!(!scene.ready_for_revision(7));
        scene.pending_deformation = false;
        scene.pending_refit = true;
        assert!(!scene.ready_for_revision(7));
    }

    #[test]
    fn view_changes_reset_screen_history() {
        assert!(!view_change_invalidates_screen_history(false, false));
        assert!(view_change_invalidates_screen_history(true, false));
        assert!(view_change_invalidates_screen_history(false, true));
    }

    #[test]
    fn radiometric_settings_keep_screen_history_but_layout_and_epoch_changes_reset_it() {
        let base = RadianceCascades::low_end();
        let resolve_scale = RadianceCascades {
            sampling: crate::SamplingSettings {
                resolve_scale: 8,
                ..base.sampling
            },
            ..base
        };
        assert!(!world_cache_settings_changed(base, resolve_scale));
        assert!(history_settings_changed(base, resolve_scale));

        let lighting = RadianceCascades {
            intensity: base.intensity * 0.5,
            ..base
        };
        assert!(!history_settings_changed(base, lighting));
        assert!(!world_cache_settings_changed(base, lighting));
        assert!(!view_change_invalidates_screen_history(
            history_settings_changed(base, lighting),
            false,
        ));
        assert!(!sparse_history_reset_requested(
            true,
            false,
            history_settings_changed(base, lighting),
            false,
            false,
        ));
        let ambient = RadianceCascades {
            environment: LinearRgba::new(0.2, 0.3, 0.4, 1.0),
            ..base
        };
        assert!(!history_settings_changed(base, ambient));
        assert!(!world_cache_settings_changed(base, ambient));

        // Hardware and software traversal produce the same hits.
        let hardware = RadianceCascades {
            hardware_ray_queries: !base.hardware_ray_queries,
            ..base
        };
        assert!(!history_settings_changed(base, hardware));
        assert!(!world_cache_settings_changed(base, hardware));

        // Sun shadow ownership changes direct light, not cached GI transport.
        let sun = RadianceCascades {
            ray_traced_sun_shadows: !base.ray_traced_sun_shadows,
            ..base
        };
        assert!(!history_settings_changed(base, sun));
        assert!(!world_cache_settings_changed(base, sun));

        let layout = RadianceCascades {
            probe_spacing: base.probe_spacing * 2.0,
            ..base
        };
        assert!(history_settings_changed(base, layout));
        assert!(view_change_invalidates_screen_history(
            history_settings_changed(base, layout),
            false,
        ));

        let epoch = RadianceCascades {
            history_epoch: base.history_epoch.wrapping_add(1),
            ..base
        };
        assert!(view_change_invalidates_screen_history(
            history_settings_changed(base, epoch),
            false,
        ));
    }

    #[test]
    fn cache_history_resets_only_for_global_view_changes() {
        assert!(!reset_cache_history(false, false, false, false, false));
        for changed in 0..5 {
            let mut changes = [false; 5];
            changes[changed] = true;
            assert!(reset_cache_history(
                changes[0], changes[1], changes[2], changes[3], changes[4]
            ));
        }
    }

    fn history_snapshot(
        revision: u64,
        geometry_revision: u64,
        instance_count: usize,
        nodes: Vec<BvhNode>,
        instance_words: Vec<UVec4>,
        deformation_poses: Vec<UVec4>,
    ) -> SceneHistorySnapshot {
        SceneHistorySnapshot {
            valid: true,
            revision,
            geometry_revision,
            mesh_revision: 0,
            topology_revision: 0,
            instance_count,
            instance_material_range: None,
            nodes: Arc::new(nodes),
            instances: Arc::new(Vec::new()),
            instance_words: Arc::new(instance_words),
            deformation_poses: Arc::new(deformation_poses),
            textures: Arc::new(crate::material::TextureAtlas::default()),
            lights: Arc::new(Vec::new()),
            light_layers: Arc::new(Vec::new()),
            emissive_lights: Arc::new(Vec::new()),
            view_layers: RenderLayers::default(),
            ambient_environment: LinearRgba::default(),
            environment: None,
            environment_revision: 0,
        }
    }

    fn history_emissive_light(
        source: UVec4,
        position: Vec3,
        color: Vec3,
        cdf: f32,
        probability: f32,
    ) -> GiLight {
        GiLight {
            position: position.extend(-1.0),
            color: color.extend(0.5),
            direction: Vec3::Z.extend(cdf),
            edge1: Vec3::X.extend(probability),
            edge2: Vec3::Y.extend(0.0),
            source,
        }
    }

    fn packed_history_instance(world_from_local: Mat4, flags: u32) -> Vec<UVec4> {
        // Match scene::pack_instances: records contain the first three rows of
        // the transposed world matrix after the corresponding local transform.
        let packed = world_from_local.transpose();
        let mut words = vec![UVec4::ZERO; INSTANCE_WORDS_PER_RECORD + 1];
        for (index, row) in [packed.x_axis, packed.y_axis, packed.z_axis]
            .into_iter()
            .enumerate()
        {
            words[3 + index] = UVec4::from_array(row.to_array().map(f32::to_bits));
        }
        words[6].w = flags;
        // A word after all instance records stands in for material/scene tail
        // data and is intentionally preserved by transform-only updates.
        words[INSTANCE_WORDS_PER_RECORD] = UVec4::splat(0x1234_5678);
        words
    }

    #[test]
    fn bounded_geometry_history_reuses_rigid_motion_with_strict_geometry_metadata() {
        fn moved(previous: &SceneHistorySnapshot, world_from_local: Mat4) -> SceneHistorySnapshot {
            let mut current = previous.clone();
            current.revision += 1;
            current.geometry_revision += 1;
            let mut words = current.instance_words.as_ref().clone();
            let packed = world_from_local.transpose();
            for (index, row) in [packed.x_axis, packed.y_axis, packed.z_axis]
                .into_iter()
                .enumerate()
            {
                words[3 + index] = UVec4::from_array(row.to_array().map(f32::to_bits));
            }
            current.instance_words = Arc::new(words);
            current
        }

        let root = BvhNode {
            min: Vec3::splat(-10.0),
            max: Vec3::splat(10.0),
            first: 0,
            count: 1,
        };
        let previous = history_snapshot(
            1,
            10,
            1,
            vec![root],
            packed_history_instance(Mat4::IDENTITY, 0),
            vec![],
        );
        let translated = moved(&previous, Mat4::from_translation(Vec3::new(0.5, 0.0, 0.0)));
        assert!(bounded_geometry_history_reuse(&previous, &translated));

        let mut changed = translated.clone();
        let mut nodes = changed.nodes.as_ref().clone();
        nodes[0].max.x += 1.0;
        changed.nodes = Arc::new(nodes);
        assert!(
            !bounded_geometry_history_reuse(&previous, &changed),
            "root bounds changed"
        );

        let mut changed = translated.clone();
        let mut words = changed.instance_words.as_ref().clone();
        *words.last_mut().unwrap() = UVec4::ZERO;
        changed.instance_words = Arc::new(words);
        assert!(
            !bounded_geometry_history_reuse(&previous, &changed),
            "material tail changed"
        );

        let mut changed = translated.clone();
        changed.textures = Arc::new((*previous.textures).clone());
        assert!(
            !bounded_geometry_history_reuse(&previous, &changed),
            "texture atlas Arc changed"
        );

        let mut changed = translated.clone();
        Arc::make_mut(&mut changed.lights).push(GiLight {
            position: Vec4::X,
            ..default()
        });
        assert!(
            bounded_geometry_history_reuse(&previous, &changed),
            "light changes affect radiance history, not geometry reuse"
        );

        let mut changed = translated.clone();
        Arc::make_mut(&mut changed.light_layers).push(RenderLayers::none().with(3));
        assert!(
            bounded_geometry_history_reuse(&previous, &changed),
            "light layer changes affect radiance history, not geometry reuse"
        );

        let mut changed = translated.clone();
        changed.mesh_revision += 1;
        assert!(
            !bounded_geometry_history_reuse(&previous, &changed),
            "mesh revision changed"
        );

        let mut changed = translated.clone();
        changed.topology_revision += 1;
        assert!(
            !bounded_geometry_history_reuse(&previous, &changed),
            "topology revision changed"
        );

        let mut changed = translated.clone();
        changed.deformation_poses = Arc::new(vec![UVec4::ONE]);
        assert!(
            !bounded_geometry_history_reuse(&previous, &changed),
            "deformation pose changed"
        );

        let mut emitter_previous = previous.clone();
        emitter_previous.instance_count = 2;
        emitter_previous.nodes = Arc::new(vec![
            BvhNode {
                min: Vec3::splat(-10.0),
                max: Vec3::splat(10.0),
                first: 0,
                count: 0,
            },
            BvhNode {
                min: Vec3::splat(-1.0),
                max: Vec3::ONE,
                first: 0,
                count: 1,
            },
            BvhNode {
                min: Vec3::splat(3.0),
                max: Vec3::splat(4.0),
                first: 1,
                count: 1,
            },
        ]);
        let record = packed_history_instance(Mat4::IDENTITY, 0);
        let mut words = vec![UVec4::ZERO; INSTANCE_WORDS_PER_RECORD * 2 + 1];
        words[..INSTANCE_WORDS_PER_RECORD].copy_from_slice(&record[..INSTANCE_WORDS_PER_RECORD]);
        words[INSTANCE_WORDS_PER_RECORD..INSTANCE_WORDS_PER_RECORD * 2]
            .copy_from_slice(&record[..INSTANCE_WORDS_PER_RECORD]);
        words[INSTANCE_WORDS_PER_RECORD * 2] = UVec4::splat(0x1234_5678);
        emitter_previous.instance_words = Arc::new(words);
        emitter_previous.emissive_lights = Arc::new(vec![history_emissive_light(
            UVec4::new(0, 7, 0, 0),
            Vec3::ZERO,
            Vec3::ONE,
            1.0,
            1.0,
        )]);
        let mut emitter_moved = moved(
            &emitter_previous,
            Mat4::from_translation(Vec3::new(0.5, 0.0, 0.0)),
        );
        let mut emissive_lights = emitter_moved.emissive_lights.as_ref().clone();
        emissive_lights[0].position.x += 0.5;
        emitter_moved.emissive_lights = Arc::new(emissive_lights);
        let mut nodes = emitter_moved.nodes.as_ref().clone();
        nodes[1].min.x += 0.5;
        nodes[1].max.x += 0.5;
        emitter_moved.nodes = Arc::new(nodes);
        let bounded = bounded_geometry_history_reuse(&emitter_previous, &emitter_moved);
        assert!(
            bounded,
            "rigid emissive motion keeps geometry history reusable outside the dirty bounds"
        );
        assert_eq!(
            punctual_light_history_change(&emitter_previous, &emitter_moved),
            PunctualLightHistoryChange::Unchanged
        );
        let emitter_change =
            emissive_light_history_change(&emitter_previous, &emitter_moved, bounded);
        assert_eq!(emitter_change, EmissiveLightHistoryChange::Global);
        let (dirty_min, dirty_max) = dirty_scene_bounds(&emitter_previous, &emitter_moved);
        assert_eq!(dirty_min, Vec3::splat(-1.0).extend(1.0));
        assert_eq!(dirty_max, Vec3::new(1.5, 1.0, 1.0).extend(0.0));

        let scaled = moved(&previous, Mat4::from_scale(Vec3::new(2.0, 1.0, 1.0)));
        assert!(
            !bounded_geometry_history_reuse(&previous, &scaled),
            "scale changed"
        );

        let sheared = moved(
            &previous,
            Mat4::from_cols(Vec4::X, Vec4::new(0.5, 1.0, 0.0, 0.0), Vec4::Z, Vec4::W),
        );
        assert!(
            !bounded_geometry_history_reuse(&previous, &sheared),
            "shear changed"
        );
    }

    #[test]
    fn bounded_geometry_history_allows_only_emissive_rgb_material_edits() {
        let base = 100u32;
        let material0 = base + 18;
        let material1 = base + 26;
        let mut previous = history_snapshot(
            10,
            20,
            2,
            vec![
                BvhNode {
                    min: Vec3::splat(-10.0),
                    max: Vec3::splat(10.0),
                    first: 0,
                    count: 0,
                },
                BvhNode {
                    min: Vec3::splat(-1.0),
                    max: Vec3::ONE,
                    first: 0,
                    count: 1,
                },
                BvhNode {
                    min: Vec3::splat(3.0),
                    max: Vec3::splat(4.0),
                    first: 1,
                    count: 1,
                },
            ],
            vec![UVec4::ZERO; 2 * INSTANCE_WORDS_PER_RECORD + 2 * crate::material::MATERIAL_WORDS],
            vec![],
        );
        previous.instance_material_range = Some((base, base + 34));
        let mut words = previous.instance_words.as_ref().clone();
        let identity = packed_history_instance(Mat4::IDENTITY, 0);
        words[..INSTANCE_WORDS_PER_RECORD].copy_from_slice(&identity[..INSTANCE_WORDS_PER_RECORD]);
        words[INSTANCE_WORDS_PER_RECORD..INSTANCE_WORDS_PER_RECORD * 2]
            .copy_from_slice(&identity[..INSTANCE_WORDS_PER_RECORD]);
        words[7] = UVec4::new(1, 0, 0, material0);
        words[8] = UVec4::new(0, 0, 1, base + 18);
        words[INSTANCE_WORDS_PER_RECORD + 7] = UVec4::new(2, 0, 0, material1);
        let emission_word = |color: Vec3| {
            UVec4::from_array(
                Vec4::new(color.x, color.y, color.z, -1.0)
                    .to_array()
                    .map(f32::to_bits),
            )
        };
        words[18] = UVec4::from_array(Vec4::ONE.to_array().map(f32::to_bits));
        words[19] = emission_word(Vec3::new(14.0, 4.0, 0.6));
        words[26] = UVec4::from_array(Vec4::ONE.to_array().map(f32::to_bits));
        words[27] = emission_word(Vec3::ZERO);
        previous.instance_words = Arc::new(words);
        previous.emissive_lights = Arc::new(vec![history_emissive_light(
            UVec4::new(0, 7, 0, 0),
            Vec3::ZERO,
            Vec3::new(14.0, 4.0, 0.6),
            1.0,
            1.0,
        )]);

        let mut emission_only = previous.clone();
        emission_only.revision += 1;
        emission_only.geometry_revision += 1;
        let mut words = emission_only.instance_words.as_ref().clone();
        words[19] = emission_word(Vec3::new(3.0, 10.0, 18.0));
        emission_only.instance_words = Arc::new(words);
        let mut emitters = emission_only.emissive_lights.as_ref().clone();
        emitters[0].color = Vec3::new(3.0, 10.0, 18.0).extend(0.5);
        emission_only.emissive_lights = Arc::new(emitters);
        assert!(bounded_geometry_history_reuse(&previous, &emission_only));
        assert_eq!(
            dirty_scene_bounds(&previous, &emission_only),
            (Vec4::ZERO, Vec4::ZERO),
            "a proven emission-only change has no geometry bounds to invalidate"
        );
        assert_eq!(
            punctual_light_history_change(&previous, &emission_only),
            PunctualLightHistoryChange::Unchanged
        );
        let emission_change = emissive_light_history_change(
            &previous,
            &emission_only,
            bounded_geometry_history_reuse(&previous, &emission_only),
        );
        assert_eq!(emission_change, EmissiveLightHistoryChange::Global);

        let mut emission_and_punctual_motion = emission_only.clone();
        emission_and_punctual_motion.revision += 1;
        emission_and_punctual_motion.geometry_revision += 1;
        let mut words = emission_and_punctual_motion.instance_words.as_ref().clone();
        let blocker = Mat4::from_translation(Vec3::new(0.5, 0.0, 0.0)).transpose();
        for (index, row) in [blocker.x_axis, blocker.y_axis, blocker.z_axis]
            .into_iter()
            .enumerate()
        {
            words[INSTANCE_WORDS_PER_RECORD + 3 + index] =
                UVec4::from_array(row.to_array().map(f32::to_bits));
        }
        emission_and_punctual_motion.instance_words = Arc::new(words);
        let mut nodes = emission_and_punctual_motion.nodes.as_ref().clone();
        nodes[2].min += Vec3::X * 0.5;
        nodes[2].max += Vec3::X * 0.5;
        emission_and_punctual_motion.nodes = Arc::new(nodes);
        Arc::make_mut(&mut emission_and_punctual_motion.lights).push(GiLight {
            position: Vec3::new(1.0, 2.0, 0.0).extend(8.0),
            color: Vec3::ONE.extend(0.0),
            ..default()
        });
        Arc::make_mut(&mut emission_and_punctual_motion.light_layers).push(RenderLayers::default());
        assert!(bounded_geometry_history_reuse(
            &previous,
            &emission_and_punctual_motion
        ));
        let punctual_change =
            punctual_light_history_change(&previous, &emission_and_punctual_motion);
        let emissive_change = emissive_light_history_change(
            &previous,
            &emission_and_punctual_motion,
            bounded_geometry_history_reuse(&previous, &emission_and_punctual_motion),
        );
        assert_eq!(emissive_change, EmissiveLightHistoryChange::Global);
        assert!(local_lighting_metadata_allowed(false, true));
        let (punctual_dirty_min, punctual_dirty_max) =
            punctual_dirty_bounds(punctual_change, false, true);
        assert!(punctual_dirty_min.w > 0.5 && punctual_dirty_max.w < 0.5);
        assert_eq!(
            emissive_dirty_bounds(emissive_change, false, true),
            (Vec4::ZERO, Vec4::ZERO),
            "a global emissive update must not suppress independent punctual bounds"
        );
        let (dirty_min, dirty_max) = dirty_scene_bounds(&previous, &emission_and_punctual_motion);
        assert_eq!(dirty_min, Vec3::new(3.0, 3.0, 3.0).extend(1.0));
        assert_eq!(dirty_max, Vec3::new(4.5, 4.0, 4.0).extend(0.0));

        let mut current = previous.clone();
        current.revision += 1;
        current.geometry_revision += 1;
        let mut words = current.instance_words.as_ref().clone();
        let blocker = Mat4::from_translation(Vec3::new(0.5, 0.0, 0.0)).transpose();
        for (index, row) in [blocker.x_axis, blocker.y_axis, blocker.z_axis]
            .into_iter()
            .enumerate()
        {
            words[INSTANCE_WORDS_PER_RECORD + 3 + index] =
                UVec4::from_array(row.to_array().map(f32::to_bits));
        }
        words[19] = emission_word(Vec3::new(3.0, 10.0, 18.0));
        current.instance_words = Arc::new(words);
        let mut nodes = current.nodes.as_ref().clone();
        nodes[2].min.x += 0.5;
        nodes[2].max.x += 0.5;
        current.nodes = Arc::new(nodes);
        let mut emitters = current.emissive_lights.as_ref().clone();
        emitters[0].color = Vec3::new(3.0, 10.0, 18.0).extend(0.5);
        current.emissive_lights = Arc::new(emitters);

        assert_eq!(
            material_emission_word_offsets(&previous, 2 * INSTANCE_WORDS_PER_RECORD),
            Some(vec![19, 27]),
            "material addresses must resolve through the checked acceleration base"
        );
        assert!(bounded_geometry_history_reuse(&previous, &current));
        assert_eq!(
            punctual_light_history_change(&previous, &current),
            PunctualLightHistoryChange::Unchanged
        );
        let emissive_change = emissive_light_history_change(
            &previous,
            &current,
            bounded_geometry_history_reuse(&previous, &current),
        );
        assert_eq!(emissive_change, EmissiveLightHistoryChange::Global);
        assert_eq!(
            dirty_scene_bounds(&previous, &current),
            (
                Vec3::splat(3.0).extend(1.0),
                Vec3::new(4.5, 4.0, 4.0).extend(0.0)
            ),
            "the moved blocker keeps its bounded geometry invalidation"
        );

        let mut cutoff_changed = current.clone();
        let mut words = cutoff_changed.instance_words.as_ref().clone();
        words[19].w = 0.5f32.to_bits();
        cutoff_changed.instance_words = Arc::new(words);
        assert!(!bounded_geometry_history_reuse(&previous, &cutoff_changed));

        let mut base_alpha_changed = current.clone();
        let mut words = base_alpha_changed.instance_words.as_ref().clone();
        words[18].w ^= 1;
        base_alpha_changed.instance_words = Arc::new(words);
        assert!(!bounded_geometry_history_reuse(
            &previous,
            &base_alpha_changed
        ));

        let mut uv_changed = current.clone();
        let mut words = uv_changed.instance_words.as_ref().clone();
        words[22].x ^= 1;
        uv_changed.instance_words = Arc::new(words);
        assert!(!bounded_geometry_history_reuse(&previous, &uv_changed));

        let mut material_reassigned = current.clone();
        let mut words = material_reassigned.instance_words.as_ref().clone();
        words[INSTANCE_WORDS_PER_RECORD + 7].w = material0;
        material_reassigned.instance_words = Arc::new(words);
        assert!(!bounded_geometry_history_reuse(
            &previous,
            &material_reassigned
        ));

        let mut cull_changed = current.clone();
        let mut words = cull_changed.instance_words.as_ref().clone();
        words[6].w ^= 1;
        cull_changed.instance_words = Arc::new(words);
        assert!(!bounded_geometry_history_reuse(&previous, &cull_changed));
    }

    #[test]
    fn dirty_bounds_cover_both_sides_of_moved_instances() {
        let node = |min: Vec3, max: Vec3, first| BvhNode {
            min,
            max,
            first,
            count: 1,
        };
        let mut before_words = vec![UVec4::ZERO; 18];
        before_words[7] = UVec4::new(1, 0, 0, 0);
        before_words[16] = UVec4::new(2, 0, 0, 0);
        let mut after_words = before_words.clone();
        after_words[9] = UVec4::new(1.0f32.to_bits(), 0, 0, 0);
        let before = history_snapshot(
            4,
            2,
            2,
            vec![
                node(Vec3::ZERO, Vec3::ONE, 0),
                node(Vec3::splat(10.0), Vec3::splat(11.0), 1),
            ],
            before_words,
            vec![],
        );
        let after = history_snapshot(
            8,
            7,
            2,
            vec![
                node(Vec3::ZERO, Vec3::ONE, 0),
                node(Vec3::splat(20.0), Vec3::splat(21.0), 1),
            ],
            after_words,
            vec![],
        );
        let (dirty_min, dirty_max) = dirty_scene_bounds(&before, &after);
        assert_eq!(dirty_min, Vec3::splat(10.0).extend(1.0));
        assert_eq!(dirty_max, Vec3::splat(21.0).extend(0.0));
    }

    #[test]
    fn light_only_scene_revision_keeps_cached_spatial_history() {
        let snapshot = history_snapshot(
            20,
            9,
            1,
            vec![BvhNode {
                min: Vec3::ZERO,
                max: Vec3::ONE,
                first: 0,
                count: 1,
            }],
            vec![UVec4::ZERO; INSTANCE_WORDS_PER_RECORD],
            vec![],
        );
        let light_only = history_snapshot(
            21,
            9,
            1,
            snapshot.nodes.as_ref().clone(),
            snapshot.instance_words.as_ref().clone(),
            vec![],
        );
        assert_eq!(
            dirty_scene_bounds(&snapshot, &light_only),
            (Vec4::ZERO, Vec4::ZERO)
        );
        assert!(!reset_cache_history(false, false, false, false, false));
    }

    #[test]
    fn finite_punctual_light_changes_bound_old_and_new_influence() {
        let punctual = |position: Vec3, range: f32| GiLight {
            position: position.extend(range),
            color: Vec3::ONE.extend(0.0),
            direction: Vec4::new(0.0, 0.0, -1.0, 0.5),
            edge2: Vec4::new(0.0, 0.0, 0.0, 0.25),
            ..GiLight::default()
        };
        let old_light = punctual(Vec3::new(-4.0, 2.0, 1.0), 3.0);
        let new_light = punctual(Vec3::new(7.0, -1.0, 5.0), 5.0);
        let mut previous = history_snapshot(30, 7, 0, vec![], vec![], vec![]);
        Arc::make_mut(&mut previous.lights).push(old_light);
        previous.light_layers = Arc::new(vec![RenderLayers::default()]);
        let mut current = previous.clone();
        current.revision += 1;
        Arc::make_mut(&mut current.lights)[0] = new_light;

        let PunctualLightHistoryChange::Local { minimum, maximum } =
            punctual_light_history_change(&previous, &current)
        else {
            panic!("finite point/spot changes should have local influence bounds");
        };
        let old_minimum = old_light.position.truncate() - Vec3::splat(old_light.position.w);
        let old_maximum = old_light.position.truncate() + Vec3::splat(old_light.position.w);
        let new_minimum = new_light.position.truncate() - Vec3::splat(new_light.position.w);
        let new_maximum = new_light.position.truncate() + Vec3::splat(new_light.position.w);
        assert!(minimum.cmple(old_minimum).all());
        assert!(maximum.cmpge(old_maximum).all());
        assert!(minimum.cmple(new_minimum).all());
        assert!(maximum.cmpge(new_maximum).all());

        let mut added = history_snapshot(31, 7, 0, vec![], vec![], vec![]);
        Arc::make_mut(&mut added.lights).push(new_light);
        added.light_layers = Arc::new(vec![RenderLayers::default()]);
        assert!(matches!(
            punctual_light_history_change(&previous, &added),
            PunctualLightHistoryChange::Local { .. }
        ));
        let mut removed = added.clone();
        removed.revision += 1;
        Arc::make_mut(&mut removed.lights).clear();
        Arc::make_mut(&mut removed.light_layers).clear();
        assert!(matches!(
            punctual_light_history_change(&added, &removed),
            PunctualLightHistoryChange::Local { .. }
        ));

        let mut directional = current.clone();
        directional.revision += 1;
        Arc::make_mut(&mut directional.lights)[0].position.w = 0.0;
        assert!(matches!(
            punctual_light_history_change(&current, &directional),
            PunctualLightHistoryChange::Local { .. }
        ));

        let mut emissive = history_snapshot(32, 7, 0, vec![], vec![], vec![]);
        Arc::make_mut(&mut emissive.lights).push(GiLight {
            position: Vec3::ZERO.extend(-1.0),
            edge1: Vec4::X,
            source: UVec4::new(1, 2, 0, 0),
            ..GiLight::default()
        });
        assert_eq!(
            punctual_light_history_change(&previous, &emissive),
            PunctualLightHistoryChange::Global
        );

        let mut changed_layers = current.clone();
        changed_layers.revision += 1;
        Arc::make_mut(&mut changed_layers.light_layers)[0] = RenderLayers::layer(1);
        let PunctualLightHistoryChange::Local { minimum, maximum } =
            punctual_light_history_change(&current, &changed_layers)
        else {
            panic!("finite light-layer changes stay within the light's influence");
        };
        assert!(minimum.cmple(new_minimum).all());
        assert!(maximum.cmpge(new_maximum).all());

        let mut invalid_history = current.clone();
        invalid_history.valid = false;
        invalid_history.revision += 1;
        Arc::make_mut(&mut invalid_history.lights)[0].color.x *= 2.0;
        assert_eq!(
            punctual_light_history_change(&invalid_history, &current),
            PunctualLightHistoryChange::Global
        );
    }

    #[test]
    fn skipped_emissive_and_punctual_changes_use_last_encoded_bounds() {
        let instance_count = 3;
        let instance_tail = instance_count * INSTANCE_WORDS_PER_RECORD;
        let material_words = instance_count * crate::material::MATERIAL_WORDS;
        let base = 100u32;
        let emission_word = |color: Vec3| {
            UVec4::from_array(
                Vec4::new(color.x, color.y, color.z, -1.0)
                    .to_array()
                    .map(f32::to_bits),
            )
        };
        let source_colors = [
            Vec3::new(4.0, 2.0, 1.0),
            Vec3::splat(2.0),
            Vec3::new(4.0, 2.0, 1.0),
        ];
        let mut submitted = history_snapshot(
            40,
            7,
            instance_count,
            vec![BvhNode {
                min: Vec3::splat(-200.0),
                max: Vec3::splat(200.0),
                first: 0,
                count: 0,
            }],
            vec![UVec4::ZERO; instance_tail + material_words],
            vec![],
        );
        submitted.instance_material_range = Some((
            base,
            base + u32::try_from(instance_tail + material_words).unwrap(),
        ));
        let mut words = submitted.instance_words.as_ref().clone();
        let identity = packed_history_instance(Mat4::IDENTITY, 0);
        for (index, &color) in source_colors.iter().enumerate() {
            let record = index * INSTANCE_WORDS_PER_RECORD;
            words[record..record + INSTANCE_WORDS_PER_RECORD]
                .copy_from_slice(&identity[..INSTANCE_WORDS_PER_RECORD]);
            let material_address = base
                + u32::try_from(instance_tail + index * crate::material::MATERIAL_WORDS).unwrap();
            words[record + 7] = UVec4::new(index as u32 + 1, 0, 0, material_address);
            words[record + 8] = UVec4::new(0, 0, 1, 0);
            let material = instance_tail + index * crate::material::MATERIAL_WORDS;
            words[material] = UVec4::from_array(Vec4::ONE.to_array().map(f32::to_bits));
            words[material + 1] = emission_word(color);
        }
        submitted.instance_words = Arc::new(words);
        submitted.emissive_lights = Arc::new(vec![
            history_emissive_light(
                UVec4::new(0, 7, 0, 0),
                Vec3::new(-4.0, 0.0, 0.0),
                source_colors[0],
                0.25,
                0.25,
            ),
            history_emissive_light(
                UVec4::new(1, 3, 0, 0),
                Vec3::new(100.0, 0.0, 0.0),
                source_colors[1],
                0.5,
                0.25,
            ),
            history_emissive_light(
                UVec4::new(2, 5, 0, 0),
                Vec3::new(4.0, 0.0, 0.0),
                source_colors[2],
                1.0,
                0.5,
            ),
        ]);
        let punctual = |x: f32, range: f32| GiLight {
            position: Vec3::new(x, 2.0, 1.0).extend(range),
            color: Vec3::ONE.extend(0.0),
            direction: Vec4::new(0.0, 0.0, -1.0, 0.5),
            edge2: Vec4::new(0.0, 0.0, 0.0, 0.25),
            ..GiLight::default()
        };
        submitted.lights = Arc::new(vec![punctual(-10.0, 2.0)]);
        submitted.light_layers = Arc::new(vec![RenderLayers::default()]);

        let mut current = submitted.clone();
        current.revision += 2;
        current.geometry_revision += 1;
        let mut words = current.instance_words.as_ref().clone();
        words[instance_tail + 1] = emission_word(Vec3::ZERO);
        words[instance_tail + 2 * crate::material::MATERIAL_WORDS + 1] =
            emission_word(Vec3::new(2.0, 1.0, 0.5));
        current.instance_words = Arc::new(words);
        current.emissive_lights = Arc::new(vec![
            history_emissive_light(
                UVec4::new(1, 3, 0, 0),
                Vec3::new(100.0, 0.0, 0.0),
                source_colors[1],
                1.0 / 3.0,
                1.0 / 3.0,
            ),
            history_emissive_light(
                UVec4::new(2, 5, 0, 0),
                Vec3::new(4.0, 0.0, 0.0),
                Vec3::new(2.0, 1.0, 0.5),
                1.0,
                2.0 / 3.0,
            ),
        ]);
        Arc::make_mut(&mut current.lights)[0] = punctual(10.0, 3.0);

        let bounded = bounded_geometry_history_reuse(&submitted, &current);
        assert!(
            bounded,
            "strict emission-only proof permits local source reuse"
        );
        let emissive_change = emissive_light_history_change(&submitted, &current, bounded);
        let EmissiveLightHistoryChange::Local { minimum, maximum } = emissive_change else {
            panic!("blackening/decreasing unchanged emitters should have local source bounds");
        };
        assert!(minimum.x <= -4.0 && maximum.x >= 5.0);
        assert!(
            maximum.x < 100.0,
            "an unchanged distant emitter is not dirty"
        );

        let punctual_change = punctual_light_history_change(&submitted, &current);
        let PunctualLightHistoryChange::Local {
            minimum: light_minimum,
            maximum: light_maximum,
        } = punctual_change
        else {
            panic!("finite punctual motion should retain old and current influence bounds");
        };
        assert!(light_minimum.x <= -12.0);
        assert!(light_maximum.x >= 13.0);
        assert!(local_lighting_metadata_allowed(false, bounded));
        assert!(punctual_dirty_bounds(punctual_change, false, bounded).0.w > 0.5);
        assert!(emissive_dirty_bounds(emissive_change, false, bounded).0.w > 0.5);

        let mut emission_only = current.clone();
        emission_only.lights = submitted.lights.clone();
        let emission_only_bounded = bounded_geometry_history_reuse(&submitted, &emission_only);
        let emission_only_change =
            emissive_light_history_change(&submitted, &emission_only, emission_only_bounded);
        assert!(matches!(
            punctual_light_history_change(&submitted, &emission_only),
            PunctualLightHistoryChange::Unchanged
        ));
        assert!(matches!(
            emission_only_change,
            EmissiveLightHistoryChange::Local { .. }
        ));
        assert!(local_lighting_metadata_allowed(
            false,
            emission_only_bounded
        ));
        assert!(!local_lighting_metadata_allowed(true, bounded));

        let mut unknown_punctual = current.clone();
        unknown_punctual.revision += 1;
        Arc::make_mut(&mut unknown_punctual.lights)[0].position.w = -1.0;
        let global_punctual = punctual_light_history_change(&submitted, &unknown_punctual);
        assert_eq!(global_punctual, PunctualLightHistoryChange::Global);
        let geometry_proven = bounded_geometry_history_reuse(&submitted, &unknown_punctual);
        let still_local_emission =
            emissive_light_history_change(&submitted, &unknown_punctual, geometry_proven);
        assert!(matches!(
            still_local_emission,
            EmissiveLightHistoryChange::Local { .. }
        ));
        assert!(
            emissive_dirty_bounds(still_local_emission, false, geometry_proven)
                .0
                .w
                > 0.5
        );
        assert_eq!(
            punctual_dirty_bounds(global_punctual, false, geometry_proven),
            (Vec4::ZERO, Vec4::ZERO)
        );

        let mut moved = current.clone();
        moved.revision += 1;
        moved.geometry_revision += 1;
        let transform = Mat4::from_translation(Vec3::new(0.5, 0.0, 0.0)).transpose();
        let mut words = moved.instance_words.as_ref().clone();
        for (offset, row) in [transform.x_axis, transform.y_axis, transform.z_axis]
            .into_iter()
            .enumerate()
        {
            words[INSTANCE_WORDS_PER_RECORD + 3 + offset] =
                UVec4::from_array(row.to_array().map(f32::to_bits));
        }
        moved.instance_words = Arc::new(words);
        let mut emitters = moved.emissive_lights.as_ref().clone();
        emitters[0].position.x += 0.5;
        moved.emissive_lights = Arc::new(emitters);
        let moved_bounded = bounded_geometry_history_reuse(&current, &moved);
        assert!(moved_bounded);
        let moved_change = emissive_light_history_change(&current, &moved, moved_bounded);
        assert_eq!(moved_change, EmissiveLightHistoryChange::Global);
        assert_eq!(
            emissive_dirty_bounds(moved_change, false, moved_bounded),
            (Vec4::ZERO, Vec4::ZERO)
        );
        let mut added = emission_only.clone();
        added.revision += 1;
        added.geometry_revision += 1;
        let mut words = added.instance_words.as_ref().clone();
        words[instance_tail + 1] = emission_word(source_colors[0]);
        added.instance_words = Arc::new(words);
        let mut emitters = added.emissive_lights.as_ref().clone();
        emitters.insert(
            0,
            history_emissive_light(
                UVec4::new(0, 7, 0, 0),
                Vec3::new(-4.0, 0.0, 0.0),
                source_colors[0],
                0.25,
                0.25,
            ),
        );
        added.emissive_lights = Arc::new(emitters);
        let added_bounded = bounded_geometry_history_reuse(&emission_only, &added);
        assert!(added_bounded);
        let added_change = emissive_light_history_change(&emission_only, &added, added_bounded);
        assert_eq!(added_change, EmissiveLightHistoryChange::Global);
        assert_eq!(
            emissive_dirty_bounds(added_change, false, added_bounded),
            (Vec4::ZERO, Vec4::ZERO)
        );
    }

    #[test]
    fn gradual_sun_motion_is_distinct_from_extinction_and_other_sources() {
        let mut previous = history_snapshot(1, 1, 0, vec![], vec![], vec![]);
        previous.lights = Arc::new(vec![
            GiLight {
                position: Vec3::Y.extend(0.0),
                color: Vec3::ONE.extend(0.0),
                ..default()
            },
            GiLight {
                position: Vec3::ZERO.extend(5.0),
                color: Vec3::ONE.extend(0.0),
                ..default()
            },
        ]);
        previous.light_layers = Arc::new(vec![RenderLayers::default(); 2]);
        for frame in 1..=60 {
            let mut current = previous.clone();
            current.revision += 1;
            let angle = frame as f32 * 0.01;
            Arc::make_mut(&mut current.lights)[0].position =
                Vec3::new(angle.sin(), angle.cos(), 0.0).extend(0.0);
            let fields = source_history_fields(
                &previous,
                &current,
                punctual_light_history_change(&previous, &current),
                EmissiveLightHistoryChange::Unchanged,
            );
            assert_eq!(fields.y, SOURCE_BACKGROUND);
            assert_eq!(fields.z, 1);
            assert_eq!(fields.x, SOURCE_PUNCTUAL);
            previous = current;
        }
        let stopped = source_history_fields(
            &previous,
            &previous,
            PunctualLightHistoryChange::Unchanged,
            EmissiveLightHistoryChange::Unchanged,
        );
        assert_eq!(stopped.y, 0);
        assert_eq!(stopped.z, 0);
        let mut removed = previous.clone();
        removed.revision += 1;
        Arc::make_mut(&mut removed.lights).remove(0);
        Arc::make_mut(&mut removed.light_layers).remove(0);
        let removal = source_history_fields(
            &previous,
            &removed,
            punctual_light_history_change(&previous, &removed),
            EmissiveLightHistoryChange::Unchanged,
        );
        assert_ne!(removal.y & SOURCE_BACKGROUND, 0);
        assert_eq!(removal.z, 0);
        let mut gradual = previous.clone();
        Arc::make_mut(&mut gradual.lights)[0].position =
            Vec3::new(0.57, 0.82, 0.0).normalize().extend(0.0);
        Arc::make_mut(&mut gradual.lights)[0].color = Vec3::splat(1.02).extend(0.0);
        gradual.ambient_environment = LinearRgba::new(0.01, 0.01, 0.01, 1.0);
        assert!(!gradual_directional_motion(&previous, &gradual));
        previous.ambient_environment = LinearRgba::new(0.01, 0.01, 0.01, 1.0);
        gradual.ambient_environment = LinearRgba::new(0.0102, 0.0102, 0.0102, 1.0);
        assert!(gradual_directional_motion(&previous, &gradual));
        Arc::make_mut(&mut gradual.lights)[0].color = Vec3::ZERO.extend(0.0);
        assert!(!gradual_directional_motion(&previous, &gradual));
        let mut changed_environment = previous.clone();
        changed_environment.environment_revision += 1;
        assert!(!gradual_directional_motion(&previous, &changed_environment));
        let mut step = previous.clone();
        Arc::make_mut(&mut step.lights)[0].position = Vec3::NEG_Y.extend(0.0);
        assert!(!gradual_directional_motion(&previous, &step));
    }

    #[test]
    fn source_history_masks_track_active_offscreen_emission_and_source_classes() {
        let mut offscreen_emitter = history_snapshot(1, 1, 1, vec![], vec![], vec![]);
        offscreen_emitter.view_layers = RenderLayers::layer(1);
        offscreen_emitter.emissive_lights = Arc::new(vec![history_emissive_light(
            UVec4::new(0, 0, 0, 0),
            Vec3::splat(10_000.0),
            Vec3::ONE,
            1.0,
            1.0,
        )]);
        let initial = source_history_fields(
            &SceneHistorySnapshot::default(),
            &offscreen_emitter,
            PunctualLightHistoryChange::Unchanged,
            EmissiveLightHistoryChange::Global,
        );
        assert_eq!(initial.x, SOURCE_EMISSIVE);
        assert_eq!(
            initial.y & (SOURCE_BACKGROUND | SOURCE_EMISSIVE | SOURCE_PUNCTUAL),
            SOURCE_BACKGROUND | SOURCE_EMISSIVE
        );

        let mut removed_emitter = offscreen_emitter.clone();
        removed_emitter.revision += 1;
        removed_emitter.emissive_lights = Arc::new(Vec::new());
        let removed = source_history_fields(
            &offscreen_emitter,
            &removed_emitter,
            PunctualLightHistoryChange::Unchanged,
            EmissiveLightHistoryChange::Global,
        );
        assert_eq!(removed.x, 0);
        assert_eq!(removed.y & SOURCE_EMISSIVE, SOURCE_EMISSIVE);
        assert_eq!(removed.y & SOURCE_BACKGROUND, 0);

        let mut directional_before = history_snapshot(2, 1, 0, vec![], vec![], vec![]);
        directional_before.lights = Arc::new(vec![GiLight {
            position: Vec3::Y.extend(0.0),
            color: Vec3::ONE.extend(0.0),
            ..default()
        }]);
        directional_before.light_layers = Arc::new(vec![RenderLayers::default()]);
        let mut directional_after = directional_before.clone();
        directional_after.revision += 1;
        Arc::make_mut(&mut directional_after.lights)[0].color.x *= 2.0;
        let directional_change =
            punctual_light_history_change(&directional_before, &directional_after);
        assert_eq!(directional_change, PunctualLightHistoryChange::Unchanged);
        let directional_history = source_history_fields(
            &directional_before,
            &directional_after,
            directional_change,
            EmissiveLightHistoryChange::Unchanged,
        );
        assert_eq!(directional_history.y & SOURCE_BACKGROUND, SOURCE_BACKGROUND);
        assert_eq!(directional_history.y & SOURCE_PUNCTUAL, 0);

        let mut punctual_before = history_snapshot(3, 1, 0, vec![], vec![], vec![]);
        punctual_before.lights = Arc::new(vec![GiLight {
            position: Vec3::ZERO.extend(4.0),
            color: Vec3::ONE.extend(0.0),
            ..default()
        }]);
        punctual_before.light_layers = Arc::new(vec![RenderLayers::default()]);
        let mut punctual_after = punctual_before.clone();
        punctual_after.revision += 1;
        Arc::make_mut(&mut punctual_after.lights)[0].color.x *= 2.0;
        let punctual_change = punctual_light_history_change(&punctual_before, &punctual_after);
        let punctual_history = source_history_fields(
            &punctual_before,
            &punctual_after,
            punctual_change,
            EmissiveLightHistoryChange::Unchanged,
        );
        assert_eq!(punctual_history.x, SOURCE_PUNCTUAL);
        assert_eq!(punctual_history.y & SOURCE_PUNCTUAL, SOURCE_PUNCTUAL);
        assert_eq!(punctual_history.y & SOURCE_BACKGROUND, 0);
        let mut mixed_after = punctual_after.clone();
        mixed_after.emissive_lights = Arc::new(vec![history_emissive_light(
            UVec4::ZERO,
            Vec3::splat(50.0),
            Vec3::ONE,
            1.0,
            1.0,
        )]);
        let mixed_history = source_history_fields(
            &punctual_before,
            &mixed_after,
            punctual_light_history_change(&punctual_before, &mixed_after),
            EmissiveLightHistoryChange::Global,
        );
        assert_eq!(mixed_history.x, SOURCE_EMISSIVE | SOURCE_PUNCTUAL);
        assert_eq!(
            mixed_history.y & (SOURCE_BACKGROUND | SOURCE_EMISSIVE | SOURCE_PUNCTUAL),
            SOURCE_EMISSIVE | SOURCE_PUNCTUAL
        );
        let background_before = history_snapshot(4, 1, 0, vec![], vec![], vec![]);
        let mut background_after = background_before.clone();
        background_after.environment_revision = 1;
        let background_history = source_history_fields(
            &background_before,
            &background_after,
            PunctualLightHistoryChange::Unchanged,
            EmissiveLightHistoryChange::Unchanged,
        );
        assert_eq!(background_history.x, 0);
        assert_eq!(background_history.y, SOURCE_BACKGROUND);
        let mut ambient_after = background_before.clone();
        ambient_after.ambient_environment = LinearRgba::new(0.2, 0.3, 0.4, 1.0);
        let ambient_history = source_history_fields(
            &background_before,
            &ambient_after,
            PunctualLightHistoryChange::Unchanged,
            EmissiveLightHistoryChange::Unchanged,
        );
        assert_eq!(ambient_history.x, 0);
        assert_eq!(ambient_history.y, SOURCE_BACKGROUND);
    }

    #[test]
    fn skipped_light_revisions_compare_against_the_last_encoded_snapshot() {
        let punctual = |x: f32| GiLight {
            position: Vec3::new(x, 2.0, 1.0).extend(2.0),
            color: Vec3::ONE.extend(0.0),
            ..GiLight::default()
        };
        let mut submitted = history_snapshot(40, 7, 0, vec![], vec![], vec![]);
        Arc::make_mut(&mut submitted.lights).push(punctual(-10.0));
        submitted.light_layers = Arc::new(vec![RenderLayers::default()]);

        // These scene revisions were prepared while the view had no encoded GI
        // dispatch. Its committed snapshot must remain `submitted` until the
        // final current state is actually encoded.
        let mut skipped = submitted.clone();
        skipped.revision += 1;
        Arc::make_mut(&mut skipped.lights)[0] = punctual(0.0);
        let mut current = skipped.clone();
        current.revision += 1;
        Arc::make_mut(&mut current.lights)[0] = punctual(10.0);

        assert!(scene_changed_since_history(&submitted, &skipped));
        assert!(scene_changed_since_history(&submitted, &current));
        assert!(light_only_scene_snapshot_unchanged(&submitted, &current));
        let PunctualLightHistoryChange::Local { minimum, maximum } =
            punctual_light_history_change(&submitted, &current)
        else {
            panic!(
                "finite light movement should retain local dirty bounds across skipped revisions"
            );
        };
        assert!(
            minimum.x <= -12.0,
            "old submitted light influence omitted: {minimum:?}"
        );
        assert!(
            maximum.x >= 12.0,
            "current light influence omitted: {maximum:?}"
        );
        let encoded_source_history = source_history_fields(
            &submitted,
            &current,
            punctual_light_history_change(&submitted, &current),
            EmissiveLightHistoryChange::Unchanged,
        );
        assert_eq!(encoded_source_history.x, SOURCE_PUNCTUAL);
        assert_eq!(encoded_source_history.y & SOURCE_PUNCTUAL, SOURCE_PUNCTUAL);
        assert_eq!(encoded_source_history.y & SOURCE_BACKGROUND, 0);

        // The new history becomes the baseline only after it is encoded.
        assert!(!scene_changed_since_history(&current, &current));
        assert_eq!(
            punctual_light_history_change(&current, &current),
            PunctualLightHistoryChange::Unchanged
        );
    }

    #[test]
    fn punctual_light_changes_do_not_weaken_rigid_geometry_checks() {
        let root = BvhNode {
            min: Vec3::splat(-10.0),
            max: Vec3::splat(10.0),
            first: 0,
            count: 1,
        };
        let previous = history_snapshot(
            40,
            9,
            1,
            vec![root],
            packed_history_instance(Mat4::IDENTITY, 0),
            vec![],
        );
        let mut light_only = previous.clone();
        light_only.revision += 1;
        Arc::make_mut(&mut light_only.lights).push(GiLight {
            position: Vec3::new(0.0, 2.0, 0.0).extend(4.0),
            color: Vec3::ONE.extend(0.0),
            ..GiLight::default()
        });
        assert!(light_only_scene_snapshot_unchanged(&previous, &light_only));
        let mut texture_changed_light = light_only.clone();
        texture_changed_light.textures = Arc::new(crate::material::TextureAtlas::default());
        assert!(!light_only_scene_snapshot_unchanged(
            &previous,
            &texture_changed_light
        ));

        let mut current = previous.clone();
        current.revision += 1;
        current.geometry_revision += 1;
        let moved = Mat4::from_translation(Vec3::new(0.5, 0.0, 0.0)).transpose();
        let mut words = current.instance_words.as_ref().clone();
        for (index, row) in [moved.x_axis, moved.y_axis, moved.z_axis]
            .into_iter()
            .enumerate()
        {
            words[3 + index] = UVec4::from_array(row.to_array().map(f32::to_bits));
        }
        current.instance_words = Arc::new(words);
        Arc::make_mut(&mut current.lights).push(GiLight {
            position: Vec3::new(0.0, 2.0, 0.0).extend(4.0),
            color: Vec3::ONE.extend(0.0),
            ..GiLight::default()
        });

        assert!(bounded_geometry_history_reuse(&previous, &current));
        assert!(matches!(
            punctual_light_history_change(&previous, &current),
            PunctualLightHistoryChange::Local { .. }
        ));

        let mut material_changed = current.clone();
        material_changed.textures = Arc::new(crate::material::TextureAtlas::default());
        assert!(!bounded_geometry_history_reuse(
            &previous,
            &material_changed
        ));
    }

    #[test]
    fn dirty_bounds_cover_removed_and_pose_changed_deformed_instances() {
        let mut before_words = vec![UVec4::ZERO; 18];
        before_words[15].w = INSTANCE_FLAG_ACTIVE_DEFORMATION;
        let before = history_snapshot(
            10,
            7,
            2,
            vec![
                BvhNode {
                    min: Vec3::ZERO,
                    max: Vec3::ONE,
                    first: 0,
                    count: 1,
                },
                BvhNode {
                    min: Vec3::splat(100.0),
                    max: Vec3::splat(101.0),
                    first: 1,
                    count: 1,
                },
            ],
            before_words,
            vec![UVec4::ZERO],
        );
        let after = history_snapshot(
            11,
            8,
            1,
            vec![BvhNode {
                min: Vec3::ZERO,
                max: Vec3::ONE,
                first: 0,
                count: 1,
            }],
            vec![UVec4::ZERO; INSTANCE_WORDS_PER_RECORD],
            vec![UVec4::ONE],
        );
        let (dirty_min, dirty_max) = dirty_scene_bounds(&before, &after);
        assert_eq!(dirty_min, Vec3::splat(100.0).extend(1.0));
        assert_eq!(dirty_max, Vec3::splat(101.0).extend(0.0));
    }

    #[test]
    fn dirty_bounds_use_exact_instance_bounds_inside_a_shared_tlas_leaf() {
        let previous_mover_min = Vec3::new(-6.0, 0.0, 1.0);
        let previous_mover_max = Vec3::new(-5.0, 1.0, 2.0);
        let current_mover_min = Vec3::new(-4.0, 0.0, 1.0);
        let current_mover_max = Vec3::new(-3.0, 1.0, 2.0);
        let neighbor_min = Vec3::new(20.0, 5.0, 6.0);
        let neighbor_max = Vec3::new(22.0, 7.0, 8.0);
        let leaf = BvhNode {
            min: Vec3::new(-10.0, -1.0, 0.0),
            max: Vec3::new(25.0, 10.0, 10.0),
            first: 0,
            count: 2,
        };
        let mut previous = history_snapshot(
            1,
            10,
            2,
            vec![leaf],
            vec![UVec4::ZERO; INSTANCE_WORDS_PER_RECORD * 2 + 1],
            vec![],
        );
        previous.instances = Arc::new(vec![
            crate::scene::SceneInstance::test_with_world_bounds(
                previous_mover_min,
                previous_mover_max,
            ),
            crate::scene::SceneInstance::test_with_world_bounds(neighbor_min, neighbor_max),
        ]);

        let mut current = previous.clone();
        current.revision += 1;
        current.geometry_revision += 1;
        let mut instance_words = current.instance_words.as_ref().clone();
        instance_words[3].x = 1;
        current.instance_words = Arc::new(instance_words);
        current.instances = Arc::new(vec![
            crate::scene::SceneInstance::test_with_world_bounds(
                current_mover_min,
                current_mover_max,
            ),
            crate::scene::SceneInstance::test_with_world_bounds(neighbor_min, neighbor_max),
        ]);

        let (dirty_min, dirty_max) = dirty_scene_bounds(&previous, &current);
        assert_eq!(dirty_min, previous_mover_min.extend(1.0));
        assert_eq!(dirty_max, current_mover_max.extend(0.0));
        assert!(
            dirty_max.x < neighbor_min.x,
            "unchanged neighbor must stay outside dirty bounds"
        );
    }

    #[test]
    fn geometry_overlay_packs_sparse_deformation_ranges() {
        let mut words = vec![UVec4::ZERO; 12];
        words[0] = UVec4::new(2, 1, 7, 0);
        words[1] = UVec4::new(0, 3, 2, 0);
        words[4] = UVec4::new(2, 9, 3, 0);
        let deformation = crate::deformation::DeformationData {
            words: Arc::new(words),
            vertices: 5,
            ..default()
        };
        let words = geometry_overlay_layout(&deformation, 64).unwrap();
        assert_eq!(
            words.as_slice(),
            &[
                UVec4::new(2, 1, 64, 5),
                UVec4::new(3, 2, 0, 0),
                UVec4::new(9, 3, 2, 0),
            ]
        );

        let mut overlapping = deformation;
        Arc::make_mut(&mut overlapping.words)[4].y = 4;
        assert!(geometry_overlay_layout(&overlapping, 64).is_none());
    }

    #[test]
    fn paged_geometry_reserves_output_layers_after_static_pages() {
        let limits = WgpuLimits {
            max_texture_dimension_2d: 4,
            max_texture_array_layers: 8,
            ..default()
        };
        let pages = geometry_pages(17, 3, &limits).unwrap();
        assert_eq!(pages.base_layers, 2);
        assert_eq!(pages.output_layers, 1);
        assert_eq!(pages.layers, 3);
        assert_eq!(pages.output_base_word, 32);
        assert_eq!(pages.bytes, 768);

        let limits = WgpuLimits {
            max_texture_dimension_2d: 4,
            max_texture_array_layers: 2,
            ..default()
        };
        assert!(geometry_pages(17, 3, &limits).is_none());
    }

    #[test]
    fn pack_probe_dispatch_covers_canonical_descriptors_in_a_bounded_2d_grid() {
        for samples in [1u64, 2, 127, 128, 129, 32_767, 32_768, 32_769, 131_072] {
            let descriptors = samples * 2;
            let (groups_x, groups_y) =
                pack_probe_dispatch_groups(samples).expect("bounded probe dispatch");
            assert_eq!(
                u64::from(groups_x),
                descriptors.min(PACK_PROBE_GROUPS_PER_ROW)
            );
            assert_eq!(
                u64::from(groups_y),
                descriptors.div_ceil(PACK_PROBE_GROUPS_PER_ROW)
            );
            assert!(groups_x > 0 && u64::from(groups_x) <= PACK_PROBE_GROUPS_PER_ROW);
            assert!(u64::from(groups_x) * u64::from(groups_y) >= descriptors);

            let mut visited = vec![false; descriptors as usize];
            for group_y in 0..u64::from(groups_y) {
                for group_x in 0..u64::from(groups_x) {
                    let ordinal = group_x + PACK_PROBE_GROUPS_PER_ROW * group_y;
                    if ordinal >= descriptors {
                        continue;
                    }
                    assert!(!visited[ordinal as usize], "descriptor {ordinal} repeated");
                    visited[ordinal as usize] = true;
                }
            }
            assert!(visited.into_iter().all(|covered| covered));
        }

        assert_eq!(pack_probe_dispatch_groups(0), None);
        assert_eq!(
            pack_probe_dispatch_groups(u64::from(u32::MAX) / 2 + 1),
            None
        );
    }

    #[test]
    fn composite_uses_post_denoise_receiver_metadata_only_when_filtering() {
        let temporal = "temporal receiver history";
        let filtered = "post-denoise receiver metadata";
        assert_eq!(
            composite_receiver_history_source(0, &temporal, &filtered),
            &temporal
        );
        for steps in 1..=4 {
            assert_eq!(
                composite_receiver_history_source(steps, &temporal, &filtered),
                &filtered
            );
        }
    }

    #[test]
    fn split_high_resolution_view_fits_the_approved_sponza_budget_without_reduction() {
        let viewport = UVec2::new(1920, 1080);
        let receiver_bytes = u64::from(viewport.x) * u64::from(viewport.y) * 16;
        let view_budget = 1024 * 1024 * 1024 - receiver_bytes;
        let requested = RadianceCascades {
            resolution_divisor: 4,
            sampling: crate::SamplingSettings {
                grid_divisor: 4,
                max_receiver_cells: 32_768,
                max_updates_per_frame: 8_192,
                max_cache_entries: 262_144,
                max_traversal_queries: 131_072,
                resolve_scale: 4,
            },
            ..default()
        };
        let limits = WgpuLimits {
            max_storage_buffer_binding_size: 512 * 1024 * 1024,
            max_buffer_size: 512 * 1024 * 1024,
            ..WgpuLimits::default()
        };
        let effective = adapt_view_settings(viewport, requested, view_budget, &limits)
            .expect("the accuracy profile's 1 GiB budget should admit the Sponza view");

        assert_eq!(effective.sampling.max_receiver_cells, 32_768);
        assert_eq!(effective.sampling.max_cache_entries, 262_144);
        assert_eq!(effective.sampling.resolve_scale, 4);
        let size = resolution(viewport, effective);
        let shading_size = resolve_size(viewport, effective);
        assert_eq!(size, UVec2::new(240, 135));
        assert!(view_fits_device(
            size,
            shading_size,
            viewport,
            effective,
            &limits
        ));
        let allocated = view_bytes(size, shading_size, effective) + receiver_bytes;
        assert!(
            allocated <= 1024 * 1024 * 1024,
            "allocated {allocated} bytes"
        );
    }

    #[test]
    fn raster_residency_preserves_fallback_and_gates_ready_scene_edits() {
        let mut world = World::new();
        let committed = world.spawn_empty().id();
        let pending = world.spawn_empty().id();
        let textures = Arc::new(crate::material::TextureAtlas::default());
        let mut scene = GiScene::default();
        scene.pending_entities = Arc::new(HashSet::from([committed, pending]));
        scene.data = Arc::new(crate::scene::SceneData {
            textures: textures.clone(),
            instance_residency: Arc::new(HashMap::from([(committed, 1)])),
            ..default()
        });
        let mut app = App::new();
        app.insert_resource(scene)
            .insert_resource(GpuScene {
                textures: textures.clone(),
                instance_residency: Arc::new(HashMap::from([(committed, 1)])),
                ..default()
            })
            .init_resource::<RasterResidencyGate>()
            .add_systems(Update, update_raster_residency);

        app.update();
        let gate = app.world().resource::<RasterResidencyGate>();
        assert!(gate.gated.is_empty());

        app.world_mut().resource_mut::<GiScene>().ready = true;
        app.update();
        assert_eq!(
            app.world().resource::<RasterResidencyGate>().gated,
            HashSet::from([MainEntity::from(pending)])
        );

        let keys = Arc::new(HashMap::from([(committed, 1), (pending, 2)]));
        {
            let mut scene = app.world_mut().resource_mut::<GiScene>();
            scene.ready = true;
            scene.pending_entities = Arc::new(HashSet::from([committed, pending]));
            scene.data = Arc::new(crate::scene::SceneData {
                textures: textures.clone(),
                instance_residency: keys.clone(),
                ..default()
            });
        }
        {
            let mut gpu_scene = app.world_mut().resource_mut::<GpuScene>();
            gpu_scene.textures = textures;
            gpu_scene.instance_residency = keys.clone();
        }

        app.update();
        let gate = app.world().resource::<RasterResidencyGate>();
        assert!(gate.gated.is_empty());
        assert_eq!(gate.released, HashSet::from([MainEntity::from(pending)]));

        let replacement_atlas = Arc::new(crate::material::TextureAtlas::default());
        app.world_mut().resource_mut::<GiScene>().data = Arc::new(crate::scene::SceneData {
            textures: replacement_atlas.clone(),
            instance_residency: keys.clone(),
            ..default()
        });
        app.update();
        assert_eq!(
            app.world().resource::<RasterResidencyGate>().gated,
            HashSet::from([MainEntity::from(committed), MainEntity::from(pending)])
        );

        app.world_mut().resource_mut::<GpuScene>().textures = replacement_atlas;
        app.update();
        let gate = app.world().resource::<RasterResidencyGate>();
        assert!(gate.gated.is_empty());
        assert_eq!(
            gate.released,
            HashSet::from([MainEntity::from(committed), MainEntity::from(pending)])
        );
    }

    #[test]
    fn raster_residency_filters_and_restores_visible_meshes() {
        let mut world = World::new();
        let new_cpu = MainEntity::from(world.spawn_empty().id());
        let old_cpu = MainEntity::from(world.spawn_empty().id());
        let gpu = MainEntity::from(world.spawn_empty().id());
        let mut meshes = RenderVisibleEntitiesClass {
            entities_cpu_culling: vec![
                (Entity::PLACEHOLDER, new_cpu),
                (Entity::PLACEHOLDER, old_cpu),
            ],
            ..default()
        };
        meshes.added_entities.push((Entity::PLACEHOLDER, new_cpu));
        meshes.entities_gpu_culling.insert(gpu, Entity::PLACEHOLDER);

        let mut gate = RasterResidencyGate::default();
        gate.gated.extend([new_cpu, old_cpu, gpu]);
        let mut gpu_entities = RenderGpuCulledEntities::default();
        gpu_entities.entities.insert(gpu, RenderLayers::default());
        filter_raster_class(
            &mut meshes,
            Some(&RenderLayers::default()),
            &gate,
            &gpu_entities,
        );
        assert!(meshes.entities_cpu_culling.is_empty());
        assert!(meshes.entities_gpu_culling.is_empty());
        assert!(meshes.added_entities.is_empty());
        assert_eq!(meshes.removed_entities.len(), 2);
        assert!(
            meshes
                .removed_entities
                .contains(&(Entity::PLACEHOLDER, old_cpu))
        );
        assert!(
            meshes
                .removed_entities
                .contains(&(Entity::PLACEHOLDER, gpu))
        );
        assert!(
            meshes
                .removed_entities
                .windows(2)
                .all(|pair| pair[0].1 <= pair[1].1)
        );

        gate.gated.clear();
        gate.released.insert(gpu);
        meshes.removed_entities.clear();
        filter_raster_class(
            &mut meshes,
            Some(&RenderLayers::none().with(1)),
            &gate,
            &gpu_entities,
        );
        assert!(!meshes.entities_gpu_culling.contains_key(&gpu));
        assert!(meshes.added_entities.is_empty());

        filter_raster_class(
            &mut meshes,
            Some(&RenderLayers::default()),
            &gate,
            &gpu_entities,
        );
        assert_eq!(
            meshes.entities_gpu_culling.get(&gpu),
            Some(&Entity::PLACEHOLDER)
        );
        assert_eq!(meshes.added_entities, vec![(Entity::PLACEHOLDER, gpu)]);
        assert!(meshes.removed_entities.is_empty());
    }

    #[test]
    fn staged_upload_ranges_cover_words_with_bounded_progress() {
        let mut cursor = 0;
        let mut seen = Vec::new();
        while cursor < 23 {
            let range = upload_range(cursor, 23, 80);
            assert!(range.len() <= 5 && !range.is_empty());
            cursor = range.end;
            seen.extend(range);
        }
        assert_eq!(seen, (0..23).collect::<Vec<_>>());
        assert_eq!(upload_range(23, 23, 0), 23..23);
        assert_eq!(upload_range(0, 2, 0), 0..0);
        let alias_bits = 0x7fc01234;
        let light = GiLight {
            position: Vec4::new(f32::from_bits(alias_bits), 2.0, 3.0, 4.0),
            color: Vec4::splat(5.0),
            direction: Vec4::splat(6.0),
            edge1: Vec4::splat(7.0),
            edge2: Vec4::splat(8.0),
            source: UVec4::new(9, 10, 11, 12),
        };
        let bytes = lights_bytes(&[light, light]);
        assert_eq!(bytes.len(), 192);
        assert_eq!(&bytes[..4], &alias_bits.to_le_bytes());
        assert_eq!(&bytes[16..20], &5.0f32.to_bits().to_le_bytes());
        assert_eq!(&bytes[64..68], &8.0f32.to_bits().to_le_bytes());
        assert_eq!(&bytes[80..84], &9u32.to_le_bytes());
        assert_eq!(&bytes[92..96], &12u32.to_le_bytes());
        assert_eq!(&bytes[..96], &bytes[96..]);
    }

    #[test]
    fn sparse_dynamic_ranges_coalesce_and_keep_paged_overlay_offsets() {
        let baseline = DynamicBaseline {
            prefix: vec![UVec4::ZERO; 6],
            instances: (0..4).map(UVec4::splat).collect(),
            instance_overlay_base: 40,
        };
        let mut prefix = baseline.prefix.clone();
        prefix[1] = UVec4::ONE;
        prefix[2] = UVec4::ONE;
        prefix[5] = UVec4::ONE;
        let mut instances = baseline.instances.clone();
        instances[0] = UVec4::ONE;
        instances[3] = UVec4::ONE;

        let ranges = dynamic_upload_ranges(&baseline, &prefix, &instances, instances.len());
        assert_eq!(
            ranges,
            vec![
                DynamicUploadRange {
                    source: DynamicUploadSource::Prefix,
                    source_start: 1,
                    buffer_start: 1,
                    words: 2,
                },
                DynamicUploadRange {
                    source: DynamicUploadSource::Prefix,
                    source_start: 5,
                    buffer_start: 5,
                    words: 1,
                },
                DynamicUploadRange {
                    source: DynamicUploadSource::Instances,
                    source_start: 0,
                    buffer_start: 40,
                    words: 1,
                },
                DynamicUploadRange {
                    source: DynamicUploadSource::Instances,
                    source_start: 3,
                    buffer_start: 43,
                    words: 1,
                },
            ]
        );
        assert_eq!(
            budget_dynamic_upload_ranges(&ranges, 48, 65536),
            ranges[..2].to_vec()
        );
    }

    #[test]
    fn same_layout_material_emission_edits_patch_the_tail_without_replacing_blas_pages() {
        let instance_start = INSTANCE_WORDS_PER_RECORD;
        let instance_base = 200;
        let material_address = (instance_base + instance_start) as u32;
        let previous_instances = {
            let mut words = vec![UVec4::ZERO; instance_start];
            words[7].w = material_address;
            words
        };
        let current_instances = previous_instances.clone();
        let previous_tail = vec![UVec4::ZERO; crate::material::MATERIAL_WORDS];
        let mut current_tail = previous_tail.clone();
        current_tail[1] = UVec4::ONE;

        assert!(material_tail_layout_reusable(
            &previous_tail,
            Some(&previous_instances),
            &current_tail,
            &current_instances,
            instance_start,
            1,
            instance_base,
        ));
        assert_eq!(
            changed_word_ranges(&previous_tail, &current_tail),
            Some(std::iter::once(1..2).collect()),
            "an emission-only edit stages only the changed material word"
        );

        let mut reassigned = current_instances.clone();
        reassigned[7].w += crate::material::MATERIAL_WORDS as u32;
        assert!(!material_tail_layout_reusable(
            &previous_tail,
            Some(&previous_instances),
            &current_tail,
            &reassigned,
            instance_start,
            1,
            instance_base,
        ));
        assert!(!material_tail_layout_reusable(
            &previous_tail,
            Some(&previous_instances),
            &current_tail[..current_tail.len() - 1],
            &current_instances,
            instance_start,
            1,
            instance_base,
        ));
    }

    #[test]
    fn abandoned_partial_material_patch_keeps_the_gpu_shadow_for_reverts() {
        let instance_start = INSTANCE_WORDS_PER_RECORD;
        let instance_base = 200;
        let mut instances = vec![UVec4::ZERO; instance_start];
        instances[7].w = (instance_base + instance_start) as u32;
        let original = vec![UVec4::ZERO; crate::material::MATERIAL_WORDS];
        let mut partially_applied = original.clone();
        partially_applied[1] = UVec4::ONE;
        let replacement_target = vec![UVec4::splat(2); crate::material::MATERIAL_WORDS];
        let mut scene = GpuScene {
            material_words: original.clone(),
            pending_material_tail: Some(PendingMaterialTail {
                applied_words: partially_applied.clone(),
                target_words: replacement_target,
            }),
            ..default()
        };

        scene.abandon_pending_material_tail();

        assert_eq!(scene.material_words, partially_applied);
        assert!(material_tail_layout_reusable(
            &scene.material_words,
            Some(&instances),
            &original,
            &instances,
            instance_start,
            1,
            instance_base,
        ));
        assert_eq!(
            changed_word_ranges(&scene.material_words, &original),
            Some(std::iter::once(1..2).collect()),
            "reverting to the old scene must patch back every word already applied"
        );
    }

    #[test]
    fn sparse_dynamic_uploads_recompute_after_partial_supersession_and_revert() {
        let zero = UVec4::ZERO;
        let one = UVec4::ONE;
        let mut baseline = DynamicBaseline {
            prefix: vec![zero; 4],
            instances: vec![zero; 3],
            instance_overlay_base: 12,
        };
        let mut target_prefix = vec![zero; 4];
        target_prefix[1] = one;
        target_prefix[2] = one;
        let mut target_instances = vec![zero; 3];
        target_instances[0] = one;

        apply_sparse_dynamic_upload(&mut baseline, &target_prefix, &target_instances, 16);
        assert_eq!(baseline.prefix[1], one);
        assert_eq!(baseline.instances[0], zero);

        // The already-queued word is reverted, while other old ranges have not
        // been written yet. Recompute from the applied shadow before every write.
        target_prefix[1] = zero;
        target_instances[1] = one;
        let ranges = dynamic_upload_ranges(
            &baseline,
            &target_prefix,
            &target_instances,
            target_instances.len(),
        );
        assert_eq!(ranges[0].source, DynamicUploadSource::Prefix);
        assert_eq!(ranges[0].source_start, 1);
        assert_eq!(ranges[0].words, 2);
        apply_sparse_dynamic_upload(&mut baseline, &target_prefix, &target_instances, 16);
        assert_eq!(baseline.prefix[1], zero);
        assert_eq!(baseline.prefix[2], zero);

        // A new revision supersedes the still-pending portion and changes the
        // previously written word again. The next plan contains only values
        // that differ from the queue-applied shadow.
        target_prefix[1] = one;
        target_prefix[2] = zero;
        target_instances[0] = one;
        apply_sparse_dynamic_upload(&mut baseline, &target_prefix, &target_instances, 1024);
        assert_eq!(baseline.prefix, target_prefix);
        assert_eq!(baseline.instances, target_instances);
        assert!(
            dynamic_upload_ranges(
                &baseline,
                &target_prefix,
                &target_instances,
                target_instances.len(),
            )
            .is_empty()
        );
    }

    #[test]
    fn sparse_instance_updates_leave_gpu_refit_prefix_words_alone() {
        let mut initialized_buffer = vec![UVec4::ZERO; 10];
        initialized_buffer[1] = UVec4::new(10, 11, 12, 13);
        initialized_buffer[2] = UVec4::new(20, 21, 22, 23);
        initialized_buffer[7] = UVec4::new(70, 71, 72, 73);
        let baseline = DynamicBaseline::from_words(&initialized_buffer, 3, 7, 2)
            .expect("fresh acceleration fully initializes its dynamic ranges");
        let mut instances = baseline.instances.clone();
        instances[1] = UVec4::ONE;

        let ranges =
            dynamic_upload_ranges(&baseline, &baseline.prefix, &instances, instances.len());
        assert_eq!(
            ranges,
            vec![DynamicUploadRange {
                source: DynamicUploadSource::Instances,
                source_start: 1,
                buffer_start: 8,
                words: 1,
            }]
        );
        // An instance-only patch cannot overwrite the TLAS prefix that the GPU
        // refit pass may have updated since the full acceleration upload.
        assert_eq!(baseline.prefix[1..], initialized_buffer[1..3]);
    }

    fn apply_sparse_dynamic_upload(
        baseline: &mut DynamicBaseline,
        prefix: &[UVec4],
        instances: &[UVec4],
        budget_bytes: usize,
    ) {
        let ranges = dynamic_upload_ranges(baseline, prefix, instances, instances.len());
        for upload in budget_dynamic_upload_ranges(&ranges, budget_bytes, 65536) {
            let end = upload.source_start + upload.words;
            let (source, target) = match upload.source {
                DynamicUploadSource::Prefix => (prefix, &mut baseline.prefix),
                DynamicUploadSource::Instances => (instances, &mut baseline.instances),
            };
            target[upload.source_start..end].copy_from_slice(&source[upload.source_start..end]);
        }
    }

    #[test]
    fn resolution_respects_budget_and_zero_viewports() {
        let settings = RadianceCascades {
            sampling: crate::SamplingSettings {
                max_receiver_cells: 8192,
                ..default()
            },
            ..default()
        };
        for viewport in [
            UVec2::ZERO,
            UVec2::ONE,
            UVec2::new(7680, 4320),
            UVec2::new(1, 50000),
            UVec2::splat(u32::MAX),
        ] {
            let size = resolution(viewport, settings);
            assert!(
                size.x > 0 && size.y > 0 && size.x * size.y <= settings.sampling.max_receiver_cells
            );
        }
    }

    #[test]
    fn resolve_scale_uses_ceil_division_through_eight() {
        let base = RadianceCascades::low_end();
        let viewport = UVec2::new(3841, 2161);
        for (scale, expected) in [
            (1, UVec2::new(3841, 2161)),
            (2, UVec2::new(1921, 1081)),
            (4, UVec2::new(961, 541)),
            (8, UVec2::new(481, 271)),
        ] {
            let settings = RadianceCascades {
                sampling: crate::SamplingSettings {
                    resolve_scale: scale,
                    ..base.sampling
                },
                ..base
            };
            assert_eq!(resolve_size(viewport, settings), expected);
            assert_eq!(split_lod_target_pixels(settings), scale);
        }
    }

    #[test]
    fn memory_adaptation_reserves_full_resolution_receiver_at_4k() {
        let viewport = UVec2::new(3840, 2160);
        let requested = RadianceCascades::low_end();
        let view_budget = 512 * 1024 * 1024u64;
        let receiver_bytes = u64::from(viewport.x) * u64::from(viewport.y) * 16;
        let available_view_bytes = view_budget - receiver_bytes;
        let settings = adapt_view_settings(
            viewport,
            requested,
            available_view_bytes,
            &WgpuLimits::default(),
        )
        .expect("4K low-end GI should fit after reducing resolve resolution");

        let size = resolution(viewport, settings);
        let bytes = view_bytes(size, resolve_size(viewport, settings), settings);
        // Adapt sparse buffers and resolve textures around the reserved
        // full-resolution receiver, including the device's binding limits.
        assert!(resolve_scale(settings) >= 3);
        let half_resolution_bytes = view_bytes(
            resolution(viewport, requested),
            resolve_size(viewport, requested),
            requested,
        );
        assert!(half_resolution_bytes > available_view_bytes);
        assert!(settings.sampling.max_receiver_cells > 0);
        assert!(settings.sampling.max_receiver_cells <= requested.sampling.max_receiver_cells);
        assert!(view_fits_device(
            size,
            resolve_size(viewport, settings),
            viewport,
            settings,
            &WgpuLimits::default(),
        ));
        assert_eq!(
            settings.sampling.max_updates_per_frame,
            requested.sampling.max_updates_per_frame
        );
        assert_eq!(
            settings.sampling.max_cache_entries,
            requested.sampling.max_cache_entries
        );
        assert_eq!(
            settings.sampling.max_traversal_queries,
            requested.sampling.max_traversal_queries
        );
        assert!(bytes + receiver_bytes <= view_budget);
    }

    #[test]
    fn memory_adaptation_reports_no_fit_when_receiver_alone_exceeds_budget() {
        let viewport = UVec2::splat(8192);
        let requested = RadianceCascades::low_end();
        let view_budget = 512 * 1024 * 1024u64;
        let receiver_bytes = u64::from(viewport.x) * u64::from(viewport.y) * 16;
        assert!(receiver_bytes > view_budget);
        assert!(
            adapt_view_settings(
                viewport,
                requested,
                view_budget.saturating_sub(receiver_bytes),
                &WgpuLimits::default(),
            )
            .is_none()
        );
    }

    fn reference_candidate_permute(value: u32, mask: u32, seed: u32) -> u32 {
        let mut value = (value ^ seed) & mask;
        value = (value ^ (value >> 16)) & mask;
        value = value.wrapping_mul(0x7feb_352d) & mask;
        value = (value ^ (value >> 15)) & mask;
        value = value.wrapping_mul(0x846c_a68b) & mask;
        (value ^ (value >> 16)) & mask
    }

    fn reference_candidate_rank(sample: u32, width: u32, height: u32, candidate_count: u32) -> u32 {
        if candidate_count <= 1 {
            return 0;
        }
        let mut mask = 0u32;
        while mask < candidate_count - 1 {
            mask = (mask << 1) | 1;
        }
        let seed =
            (width.wrapping_mul(0x9e37_79b9) ^ height.wrapping_mul(0x85eb_ca6b) ^ candidate_count)
                & mask;
        let mut rank = sample;
        loop {
            rank = reference_candidate_permute(rank, mask, seed);
            if rank < candidate_count {
                return rank;
            }
        }
    }

    fn reference_fair_selected(
        sample: u32,
        candidate_count: u32,
        width: u32,
        height: u32,
        cursor: u32,
        quota: u32,
    ) -> bool {
        if candidate_count == 0 || quota == 0 {
            return false;
        }
        let rank = reference_candidate_rank(sample, width, height, candidate_count);
        let distance = if rank >= cursor {
            rank - cursor
        } else {
            candidate_count - cursor + rank
        };
        distance < quota
    }

    #[test]
    fn transport_candidate_schedule_is_bijective_and_exactly_budgeted() {
        for (width, height, budgets) in [
            (13u32, 7, vec![0, 1, 6, 11, 91]),
            (17, 9, vec![1, 13, 17, 153]),
            (7, 13, vec![0, 1, 6, 11, 91]),
            (9, 17, vec![1, 13, 17, 153]),
            (160, 90, vec![1_024]),
        ] {
            let candidate_count = width.checked_mul(height).unwrap();

            let mut seen_ranks = vec![false; candidate_count as usize];
            for sample in 0..candidate_count {
                let rank = reference_candidate_rank(sample, width, height, candidate_count);
                assert!(rank < candidate_count, "{width}x{height}: rank {rank}");
                assert!(!seen_ranks[rank as usize], "duplicate rank {rank}");
                seen_ranks[rank as usize] = true;
            }
            assert!(seen_ranks.into_iter().all(|seen| seen));

            for budget in budgets {
                let mut schedule = CandidateSchedule::default();
                let mut selected_count = vec![0u32; candidate_count as usize];
                let updates_per_frame = if budget == 0 || budget >= candidate_count {
                    candidate_count
                } else {
                    budget
                };
                for _ in 0..candidate_count.div_ceil(updates_per_frame) {
                    let (cursor, fair_quota) =
                        schedule.phase_for(width, height, candidate_count, budget);
                    assert_eq!(fair_quota, budget.min(candidate_count));
                    let mut this_frame = 0;
                    for sample in 0..candidate_count {
                        let selected = if budget == 0 || budget >= candidate_count {
                            true
                        } else {
                            reference_fair_selected(
                                sample,
                                candidate_count,
                                width,
                                height,
                                cursor,
                                fair_quota,
                            )
                        };
                        if selected {
                            selected_count[sample as usize] += 1;
                            this_frame += 1;
                        }
                    }
                    assert_eq!(
                        this_frame, updates_per_frame,
                        "{width}x{height}, budget {budget}"
                    );
                    assert!(budget == 0 || this_frame <= budget);
                    schedule.advance_after_successful_update(true);
                }
                let min = selected_count.iter().min().copied().unwrap();
                let max = selected_count.iter().max().copied().unwrap();
                assert!(max - min <= 1, "{width}x{height}, budget {budget}");
            }
        }
    }

    #[test]
    fn transport_candidate_schedule_wraps_full_budget_across_uneven_tail() {
        const WIDTH: u32 = 160;
        const HEIGHT: u32 = 90;
        const CANDIDATE_COUNT: u32 = WIDTH * HEIGHT;
        const BUDGET: u32 = 512;
        let mut schedule = CandidateSchedule::default();
        let mut selected_count = vec![0u32; CANDIDATE_COUNT as usize];
        let batches = CANDIDATE_COUNT.div_ceil(BUDGET);

        for batch in 0..batches {
            let (cursor, quota) = schedule.phase_for(WIDTH, HEIGHT, CANDIDATE_COUNT, BUDGET);
            assert_eq!(quota, BUDGET, "batch {batch}");
            if batch == 28 {
                assert_eq!(cursor, 14_336);
            }
            let mut selected = 0;
            for sample in 0..CANDIDATE_COUNT {
                if reference_fair_selected(sample, CANDIDATE_COUNT, WIDTH, HEIGHT, cursor, quota) {
                    selected_count[sample as usize] += 1;
                    selected += 1;
                }
            }
            assert_eq!(selected, BUDGET, "batch {batch}");
            schedule.advance_after_successful_update(true);
        }

        let min = selected_count.iter().min().copied().unwrap();
        let max = selected_count.iter().max().copied().unwrap();
        assert_eq!((min, max), (1, 2));
        assert_eq!(
            schedule.phase_for(WIDTH, HEIGHT, CANDIDATE_COUNT, BUDGET),
            (448, BUDGET)
        );
    }

    #[test]
    fn low_end_transport_batches_are_spatially_uniform_each_phase() {
        const WIDTH: u32 = 160;
        const HEIGHT: u32 = 90;
        const BUDGET: u32 = 1_024;
        const TILE: u32 = 8;
        let candidate_count = WIDTH * HEIGHT;
        let tile_columns = WIDTH.div_ceil(TILE);
        let tile_rows = HEIGHT.div_ceil(TILE);
        let mut schedule = CandidateSchedule::default();
        let mut covered = vec![false; candidate_count as usize];
        while covered.iter().any(|covered| !covered) {
            let (cursor, fair_quota) = schedule.phase_for(WIDTH, HEIGHT, candidate_count, BUDGET);
            let mut tile_counts = vec![0u32; (tile_columns * tile_rows) as usize];
            let mut touched_columns = vec![false; tile_columns as usize];
            let mut touched_rows = vec![false; tile_rows as usize];
            let mut selected = 0;
            for sample in 0..candidate_count {
                if !reference_fair_selected(
                    sample,
                    candidate_count,
                    WIDTH,
                    HEIGHT,
                    cursor,
                    fair_quota,
                ) {
                    continue;
                }
                selected += 1;
                covered[sample as usize] = true;
                let x = sample % WIDTH;
                let y = sample / WIDTH;
                let tile_x = x / TILE;
                let tile_y = y / TILE;
                tile_counts[(tile_y * tile_columns + tile_x) as usize] += 1;
                touched_columns[tile_x as usize] = true;
                touched_rows[tile_y as usize] = true;
            }
            assert!(selected <= BUDGET);
            if fair_quota == BUDGET {
                assert_eq!(selected, BUDGET);
                assert!(tile_counts.iter().filter(|count| **count > 0).count() >= 225);
                assert!(touched_columns.into_iter().all(|touched| touched));
                assert!(touched_rows.into_iter().all(|touched| touched));
                assert!(tile_counts.into_iter().max().unwrap() <= 12);
            } else {
                assert_eq!(selected, fair_quota);
            }
            schedule.advance_after_successful_update(true);
        }
    }

    #[test]
    fn transport_candidate_phase_survives_wrapping_frame_counter() {
        let (width, height, candidate_count, budget) = (5, 3, 15, 3);
        let wrapped_frame_values = [u32::MAX - 1, u32::MAX, 0, 1, 2];
        let mut schedule = CandidateSchedule::default();
        for (expected_cursor, frame_x) in [0, 3, 6, 9, 12].into_iter().zip(wrapped_frame_values) {
            let (cursor, fair_quota) = schedule.phase_for(width, height, candidate_count, budget);
            assert_eq!(fair_quota, 3);
            assert_eq!(cursor, expected_cursor, "frame.x={frame_x}");
            // Preparing without a successful render does not consume a phase.
            assert_eq!(
                schedule.phase_for(width, height, candidate_count, budget),
                (cursor, fair_quota)
            );
            schedule.advance_after_successful_update(true);
        }
        assert_eq!(
            schedule.phase_for(width, height, candidate_count, budget).0,
            0
        );
    }

    #[test]
    fn cold_rank_phases_preserve_fair_cursor_and_history_disabled_keeps_rolling() {
        const WIDTH: u32 = 160;
        const HEIGHT: u32 = 90;
        const CANDIDATES: u32 = WIDTH * HEIGHT;
        const BUDGET: u32 = 512;

        assert_eq!(cold_rank_phases(24, false), &COLD_RANK_PHASES);
        assert!(cold_priority_enabled(24, false));
        assert!(cold_rank_phases(24, true).is_empty());
        assert!(!cold_priority_enabled(24, true));
        assert!(cold_rank_phases(0, false).is_empty());
        assert!(!cold_priority_enabled(0, false));

        let mut schedule = CandidateSchedule::default();
        assert_eq!(
            schedule.phase_for(WIDTH, HEIGHT, CANDIDATES, BUDGET),
            (0, BUDGET)
        );
        schedule.advance_after_successful_update(!cold_priority_enabled(24, false));
        assert_eq!(
            schedule.phase_for(WIDTH, HEIGHT, CANDIDATES, BUDGET),
            (0, BUDGET)
        );

        // With history disabled there is no persistent cold-priority mode, so
        // the ordinary fair window advances on every successful update.
        schedule.advance_after_successful_update(!cold_priority_enabled(0, false));
        assert_eq!(
            schedule.phase_for(WIDTH, HEIGHT, CANDIDATES, BUDGET),
            (BUDGET, BUDGET)
        );
    }

    #[test]
    fn candidate_fields_read_previous_parity_and_write_current_parity() {
        assert_eq!(candidate_field_indices(0), (0, 1));
        assert_eq!(candidate_field_indices(1), (1, 0));
        assert_eq!(candidate_field_indices(2), (0, 1));
        assert_eq!(candidate_field_indices(u32::MAX), (1, 0));
    }

    #[test]
    fn sparse_history_reset_tracks_camera_and_invalidations_without_losing_phase() {
        assert!(!sparse_history_reset_requested(
            true, false, false, false, false
        ));
        assert!(sparse_history_reset_requested(
            false, false, false, false, false
        ));
        assert!(sparse_history_reset_requested(
            true, true, false, false, false
        ));
        assert!(sparse_history_reset_requested(
            true, false, true, false, false
        ));
        assert!(sparse_history_reset_requested(
            true, false, false, true, false
        ));

        let camera = Mat4::IDENTITY;
        assert!(!camera_matrix_changed(camera, camera));
        let moved_camera = Mat4::from_translation(Vec3::X);
        assert!(camera_matrix_changed(camera, moved_camera));
        assert!(sparse_history_reset_requested(
            true,
            false,
            false,
            false,
            camera_matrix_changed(camera, moved_camera),
        ));

        let phase = 0x1234_5678;
        let reset_word = sparse_candidate_phase_word(phase, true, false);
        assert_eq!(
            reset_word & !(SPARSE_HISTORY_RESET_FLAG | SPARSE_UPDATES_DISABLED_FLAG),
            phase & 0x03ff_ffff
        );
        assert_ne!(reset_word & SPARSE_HISTORY_RESET_FLAG, 0);
        assert_eq!(
            sparse_candidate_phase_word(phase, false, false),
            phase & 0x03ff_ffff
        );
        let disabled_word = sparse_candidate_phase_word(0, false, true);
        assert_eq!(disabled_word, SPARSE_UPDATES_DISABLED_FLAG);
        assert_eq!(disabled_word & SPARSE_HISTORY_RESET_FLAG, 0);

        let params = Parameters {
            owner_sequence: UVec4::new(1, 1, 1, reset_word),
            ..default()
        };
        assert_eq!(
            clear_pass_parameters(params, 0).owner_sequence,
            UVec4::new(1, 1, 1, reset_word)
        );
        assert_eq!(
            clear_pass_parameters(params, 1).owner_sequence,
            UVec4::new(1, 1, 0, reset_word)
        );
    }

    #[test]
    fn transport_candidate_rank_stays_bounded_at_u32_grid_limit() {
        let width = 65_535u32;
        let height = 65_537u32;
        let candidate_count = width.checked_mul(height).unwrap();
        assert_eq!(candidate_count, u32::MAX);
        for sample in [
            0,
            1,
            width - 1,
            width,
            candidate_count - 2,
            candidate_count - 1,
        ] {
            let rank = reference_candidate_rank(sample, width, height, candidate_count);
            assert!(
                rank < candidate_count,
                "sample {sample} produced rank {rank}"
            );
        }

        let mut schedule = CandidateSchedule::default();
        let (cursor, fair_quota) = schedule.phase_for(width, height, candidate_count, 0x8000_0000);
        assert_eq!(cursor, 0);
        assert_eq!(fair_quota, 0x8000_0000);
        schedule.advance_after_successful_update(true);
        let (cursor, fair_quota) = schedule.phase_for(width, height, candidate_count, 0x8000_0000);
        assert_eq!(cursor, 0x8000_0000);
        assert_eq!(fair_quota, 0x8000_0000);
        schedule.advance_after_successful_update(true);
        assert_eq!(
            schedule.phase_for(width, height, candidate_count, 0x8000_0000),
            (1, 0x8000_0000)
        );
    }

    #[test]
    fn explicit_sampling_separates_candidate_density_from_update_budget() {
        let viewport = UVec2::new(1280, 720);
        let low_end = RadianceCascades::low_end();
        let size = resolution(viewport, low_end);
        assert_eq!(size, UVec2::new(160, 90));
        assert_eq!(size.x * size.y, 14_400);
        assert_eq!(selected_update_budget(low_end), 1_024);
        assert_eq!(runtime_update_budget(low_end, Some(32)), 32);
        assert_eq!(runtime_update_budget(low_end, Some(2_048)), 1_024);
        assert_eq!(runtime_update_budget(low_end, Some(0)), 1);

        let one_update = RadianceCascades {
            max_rays: 1,
            ..low_end
        };
        assert_eq!(resolution(viewport, one_update), size);
        assert_eq!(selected_update_budget(one_update), 1);

        let capped = RadianceCascades {
            sampling: crate::SamplingSettings {
                max_receiver_cells: 256,
                ..low_end.sampling
            },
            ..low_end
        };
        let capped_size = resolution(viewport, capped);
        assert!(capped_size.x * capped_size.y <= 256);
        assert_eq!(selected_update_budget(capped), 1_024);

        // Resolve scale alone changes reconstruction extent, not the
        // candidate grid.
        let half_res_only = RadianceCascades {
            sampling: crate::SamplingSettings {
                resolve_scale: 2,
                ..crate::SamplingSettings::default()
            },
            ..RadianceCascades::default()
        };
        assert_eq!(
            resolution(viewport, half_res_only),
            resolution(viewport, RadianceCascades::default())
        );
        assert_eq!(resolve_size(viewport, half_res_only), UVec2::new(640, 360));
        assert_eq!(resolve_size(viewport, low_end), UVec2::new(640, 360));
    }

    #[test]
    fn split_traversal_budget_limits_ray_work() {
        let low_end = RadianceCascades::low_end();
        let default_request = selected_update_budget(low_end);
        let (split_early_limit, split_hard_limit) = traversal_query_caps(16_384);
        assert_eq!(split_early_limit, 8_192);
        assert_eq!(split_hard_limit, 16_384);
        assert_eq!(
            query_affordable_update_budget(low_end, default_request, 0, 1, false),
            default_request,
            "the low-end batch must also reserve queries for interval merging"
        );

        let dense = RadianceCascades {
            max_rays: 20_000,
            sampling: crate::SamplingSettings {
                max_updates_per_frame: 20_000,
                ..low_end.sampling
            },
            ..low_end
        };
        let dense_request = selected_update_budget(dense);
        let dense_budget = query_affordable_update_budget(dense, dense_request, 0, 1, false);
        assert_eq!(dense_budget, 1_638); // seed, transport and first-hit NEE; no primary NEE
        assert!(dense_budget < dense_request);
        assert!(dense_budget * 5 <= split_early_limit);
        assert!((dense_budget + 1) * 5 > split_early_limit);

        let hdr_and_punctual = query_affordable_update_budget(dense, dense_request, 3, 2, true);
        assert_eq!(hdr_and_punctual, 819); // first-hit NEE + seed and interval transport
        assert!(hdr_and_punctual * 10 <= split_early_limit);
        assert!((hdr_and_punctual + 1) * 10 > split_early_limit);
        let constant_sky = RadianceCascades {
            environment: LinearRgba::new(500.0, 700.0, 1000.0, 1.0),
            ..dense
        };
        assert_eq!(
            query_affordable_update_budget(constant_sky, dense_request, 3, 2, false),
            hdr_and_punctual,
            "constant-sky visibility must reserve the same query work as an environment map"
        );

        let unbounded = RadianceCascades {
            sampling: crate::SamplingSettings {
                max_traversal_queries: 0,
                ..dense.sampling
            },
            ..dense
        };
        assert_eq!(
            query_affordable_update_budget(unbounded, dense_request, 0, 1, false),
            dense_request
        );

        let tiny_cap = RadianceCascades {
            sampling: crate::SamplingSettings {
                max_traversal_queries: 8,
                ..low_end.sampling
            },
            ..low_end
        };
        assert_eq!(
            query_affordable_update_budget(tiny_cap, selected_update_budget(tiny_cap), 0, 1, false),
            0,
            "never select a receiver when the hard cap cannot finish seed + direct NEE"
        );
        let mut schedule = CandidateSchedule::default();
        assert_eq!(schedule.phase_for(8, 8, 64, 0).0, 0);
        let disabled = sparse_candidate_phase_word(0, false, true);
        assert_eq!(
            disabled & SPARSE_UPDATES_DISABLED_FLAG,
            SPARSE_UPDATES_DISABLED_FLAG
        );
        assert_eq!(disabled & SPARSE_HISTORY_RESET_FLAG, 0);
    }

    #[test]
    fn adapted_view_reuse_preserves_limits_only_for_the_same_request() {
        let requested = RadianceCascades::low_end();
        let mut effective = requested;
        effective.max_rays = 128;
        effective.sampling.max_receiver_cells = 512;
        effective.sampling.resolve_scale = 4;
        assert!(settings_match_budgeted_request(
            effective, requested, requested
        ));

        // All three fields were previously normalized out of the comparison,
        // causing live edits to be replaced by the retained effective settings.
        for edited in [
            RadianceCascades {
                max_rays: 64,
                ..requested
            },
            RadianceCascades {
                sampling: crate::SamplingSettings {
                    max_receiver_cells: 256,
                    ..requested.sampling
                },
                ..requested
            },
            RadianceCascades {
                sampling: crate::SamplingSettings {
                    resolve_scale: 1,
                    ..requested.sampling
                },
                ..requested
            },
        ] {
            assert!(!settings_match_budgeted_request(
                effective, requested, edited
            ));
        }
    }

    #[test]
    fn vulkan_translation_validates_bindings() {
        let source = "
            @group(0) @binding(23) var output: texture_storage_2d<rgba32float, write>;
            @group(0) @binding(27) var input: texture_2d<u32>;
            fn value(x: u32) -> f32 { return f32(x) * 0.5; }
            @compute @workgroup_size(8, 8, 1) fn check() {
                textureStore(output, vec2<i32>(0), vec4<f32>(value(textureLoad(input, vec2<i32>(0), 0).x)));
            }
        ";
        let layout = compute_layout(false);
        let (words, size) = vulkan_compute_source(source, "check", &layout.entries).unwrap();
        assert_eq!(size, (8, 8, 1));
        let mut bindings = Vec::new();
        let mut offset = 5;
        while offset < words.len() {
            let count = (words[offset] >> 16) as usize;
            match words[offset] & 0xffff {
                71 if count == 4 && words[offset + 2] == 33 => bindings.push(words[offset + 3]),
                _ => (),
            }
            offset += count;
        }
        bindings.sort_unstable();
        let physical_binding = |logical| {
            layout
                .entries
                .iter()
                .position(|entry| entry.binding == logical)
                .expect("source binding exists in the host layout") as u32
        };
        assert_eq!(bindings, [physical_binding(23), physical_binding(27)]);
        assert!(
            vulkan_compute_source(source, "missing", &layout.entries)
                .unwrap_err()
                .contains("missing")
        );
        assert!(vulkan_compute_source("invalid WGSL", "check", &layout.entries).is_err());
    }

    #[test]
    fn shared_pipeline_readiness_distinguishes_missing_shaders_from_failure() {
        let status = RadianceCascadesStatus::default();
        assert!(shared_pipeline_pending(
            &CachedPipelineState::Queued,
            "composite",
            &status
        ));
        assert!(shared_pipeline_pending(
            &CachedPipelineState::Err(bevy::shader::ShaderCacheError::ShaderImportNotYetAvailable),
            "composite",
            &status,
        ));
        assert!(status.failures().is_empty());
        assert!(!shared_pipeline_pending(
            &CachedPipelineState::Err(bevy::shader::ShaderCacheError::ProcessShaderError(
                "broken shader".into()
            )),
            "composite",
            &status,
        ));
        assert!(status.failures()[0].contains("composite failed"));
        assert!(status.failures()[0].contains("broken shader"));
    }

    #[test]
    #[ignore = "requires slangc and a desktop GPU; creates production compute pipelines"]
    fn production_compute_pipelines_validate_sparse_bindings() {
        use bevy::render::{
            renderer::initialize_renderer,
            settings::{Backends, WgpuSettings},
        };
        let options = WgpuSettings::default();
        let resources = block_on(initialize_renderer(
            options.backends.unwrap_or(Backends::PRIMARY),
            None,
            &options,
        ));
        let device = &resources.0;
        let view = ViewBuffers::new(
            device,
            UVec2::splat(4),
            UVec2::splat(4),
            UVec2::splat(4),
            RadianceCascades::default(),
            0,
            None,
            UVec4::new(0, 0, 4, 4),
        );
        let texture_bytes: u64 = view
            .irradiance_textures
            .iter()
            .chain(&view._metadata)
            .chain(&view._background_textures)
            .chain(std::iter::once(&view._denoise_receiver_history_output))
            .chain(&view._composite_anchor_textures)
            .chain(std::iter::once(&view._sun_shadow_mask))
            .map(|texture| u64::from(texture.width()) * u64::from(texture.height()) * 16)
            .sum();
        let buffer_bytes: u64 = view
            .caches
            .iter()
            .map(|cache| cache.heads.size())
            .sum::<u64>()
            + view.entries.size()
            + view.history_entries.size()
            + view.irradiance_probes.size()
            + view.rtgi_reservoirs.iter().map(|buffer| buffer.size()).sum::<u64>()
            + view.rtgi_cache.iter().map(|buffer| buffer.size()).sum::<u64>()
            + (LEVELS as u64 + 2) * std::mem::size_of::<Parameters>() as u64;
        assert!(
            view_bytes(view.size, view.shading_size, view.settings) >= texture_bytes + buffer_bytes,
            "view budget must cover all actual GI textures, buffers, and uniforms"
        );
        let external_buffers = [&view.caches[0].heads, &view.caches[1].heads, &view.entries];
        let unfiltered_key = view_binding_key(
            &view,
            0,
            composite_display_index(0, 0),
            external_buffers,
            &[],
        );
        let filtered_key = view_binding_key(
            &view,
            0,
            composite_display_index(1, 0),
            external_buffers,
            &[],
        );
        assert_ne!(
            unfiltered_key, filtered_key,
            "changing the composite filter output must invalidate cached view bindings"
        );
        let spirv = resources.2.backend == wgpu::Backend::Vulkan
            && device
                .features()
                .contains(wgpu::Features::PASSTHROUGH_SHADERS);
        let ray_query = spirv
            && device
                .features()
                .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
        let layout = compute_layout(ray_query);
        let layouts =
            [device.create_bind_group_layout(Some(layout.label.as_ref()), &layout.entries)];
        for entry in COMPUTE_ENTRIES {
            if hardware_traversal_entry(entry) && !ray_query {
                continue;
            }
            let shader = compile(&SlangCompiler::default(), "cascades.slang", &[entry]).unwrap();
            let bevy::shader::Source::Wgsl(source) = shader.source else {
                panic!("expected validated WGSL");
            };
            compute_pipeline(
                device.clone(),
                layouts.clone(),
                Arc::from(source.as_ref()),
                entry,
                None,
                spirv,
                ray_query,
            )
            .unwrap_or_else(|error| panic!("{entry}: {error}"));
        }
        let error = compute_pipeline(
            device.clone(),
            layouts,
            Arc::from("@compute @workgroup_size(1) fn main() {}"),
            "missing",
            None,
            spirv,
            ray_query,
        )
        .unwrap_err();
        assert!(error.contains("missing"), "{error}");
    }

    #[test]
    #[ignore = "requires slangc (or SLANGC) on PATH"]
    fn compile_all_embedded_slang_stages() {
        for (file, entries) in shader_modules() {
            let shader = compile(&SlangCompiler::default(), file, entries).expect(file);
            let bevy::shader::Source::Wgsl(source) = shader.source else {
                panic!("expected Slang's WGSL output")
            };
            // Hardware traversal modules call the ray-query helpers that the
            // SPIR-V path prepends, exactly as production compiles them.
            let hardware = entries.iter().any(|entry| hardware_traversal_entry(entry));
            let source = if hardware {
                ray_query_source(&source).unwrap()
            } else {
                source.to_string()
            };
            let module = naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|error| panic!("{file}: {}", error.emit_to_string(&source)));
            assert_eq!(module.entry_points.len(), entries.len(), "{file} entries");
            for entry in entries {
                assert!(module.entry_points.iter().any(|point| point.name == *entry));
            }
            let info = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                if hardware {
                    naga::valid::Capabilities::RAY_QUERY
                } else {
                    naga::valid::Capabilities::empty()
                },
            )
            .validate(&module)
            .unwrap_or_else(|error| panic!("{file}: {error:?}"));
            // Parsing succeeds for some control flows the Vulkan writer cannot
            // lower. Exercise that backend before waiting for GPU pipelines.
            for entry in &module.entry_points {
                eprintln!("Validating {file} {} SPIR-V", entry.name);
                naga::back::spv::write_vec(
                    &module,
                    &info,
                    &naga::back::spv::Options::default(),
                    Some(&naga::back::spv::PipelineOptions {
                        shader_stage: entry.stage,
                        entry_point: entry.name.clone(),
                    }),
                )
                .unwrap_or_else(|error| panic!("{file} {} SPIR-V: {error:?}", entry.name));
            }
            if file == "cascades.slang" && entries.contains(&"packProbes") {
                let (_, entry) = module
                    .types
                    .iter()
                    .find(|(_, ty)| {
                        ty.name
                            .as_deref()
                            .is_some_and(|name| name.starts_with("Entry"))
                    })
                    .expect("radiance cache entry type");
                let naga::TypeInner::Struct { span, .. } = &entry.inner else {
                    panic!("cache entry must be a structure");
                };
                assert_eq!(
                    u64::from(*span),
                    ENTRY_SIZE,
                    "cache-entry allocation must match shader ABI"
                );
                let (_, probe) = module
                    .types
                    .iter()
                    .find(|(_, ty)| {
                        ty.name
                            .as_deref()
                            .is_some_and(|name| name.starts_with("DirectionalProbe"))
                    })
                    .expect("split irradiance probe type");
                let naga::TypeInner::Struct { span, .. } = &probe.inner else {
                    panic!("irradiance probe must be a structure");
                };
                assert_eq!(
                    u64::from(*span),
                    PROBE_SIZE,
                    "probe allocation must match shader ABI"
                );
            }
            if matches!(file, "cascades.slang") {
                // Validate the actual production layout, not only WGSL syntax.
                // An inserted atlas binding previously shifted every later
                // resource while standalone traversal tests still passed.
                let layout = compute_layout(hardware);
                for (_, global) in module.global_variables.iter() {
                    let Some(binding) = &global.binding else {
                        continue;
                    };
                    assert_eq!(binding.group, 0, "unexpected production bind group");
                    let expected = layout
                        .entries
                        .iter()
                        .find(|entry| entry.binding == binding.binding)
                        .unwrap_or_else(|| {
                            panic!("missing production binding {}", binding.binding)
                        });
                    match &module.types[global.ty].inner {
                        naga::TypeInner::Image { dim, arrayed, .. } => {
                            assert_eq!(*dim, naga::ImageDimension::D2);
                            let dimension = match expected.ty {
                                BindingType::Texture { view_dimension, .. }
                                | BindingType::StorageTexture { view_dimension, .. } => {
                                    view_dimension
                                }
                                _ => panic!(
                                    "binding {}: shader image requires texture layout",
                                    binding.binding
                                ),
                            };
                            assert_eq!(
                                dimension,
                                if *arrayed {
                                    TextureViewDimension::D2Array
                                } else {
                                    TextureViewDimension::D2
                                },
                                "production binding {} dimension",
                                binding.binding
                            );
                        }
                        naga::TypeInner::AccelerationStructure { .. } => assert!(
                            matches!(expected.ty, BindingType::AccelerationStructure { .. }),
                            "binding {}: shader TLAS requires acceleration-structure layout",
                            binding.binding
                        ),
                        _ => assert!(
                            matches!(expected.ty, BindingType::Buffer { .. }),
                            "binding {}: shader buffer requires buffer layout",
                            binding.binding
                        ),
                    }
                }
            }
        }
    }
}

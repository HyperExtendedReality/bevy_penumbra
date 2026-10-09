use std::{
    any::TypeId,
    collections::{HashMap, HashSet},
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
    pbr::{
        GpuLights, LightMeta, LightProbesBuffer, LightProbesUniform, RenderGpuCulledEntities,
        ViewLightProbesUniformOffset, ViewLightsUniformOffset, deferred::deferred_lighting,
        screen_space_reflections,
    },
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        camera::ExtractedCamera,
        diagnostic::{DiagnosticsRecorder, RecordDiagnostics},
        render_asset::RenderAssets,
        render_resource::{binding_types::*, *},
        renderer::{
            RenderAdapterInfo, RenderContext, RenderDevice, RenderGraph, RenderGraphSystems,
            RenderQueue, ViewQuery,
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

use crate::{
    RadianceCascades, RadianceCascadesRawOutput,
    scene::{BvhNode, GiLight, GiScene, view_lights},
};

const LEVELS: usize = 4;
// Byte stride of `Entry` in common.slang.
const ENTRY_SIZE: u64 = 96;
// currentCapacity() reserves two groups per ray layer; history is bucket-bounded.
const CACHE_WORK_GROUPS: u64 = 2;
const CACHE_HISTORY_PER_BUCKET: u64 = 2;
const PROBE_SIZE: u64 = 592;
const CACHE_COMMIT: usize = 16;
const PROBE_HEAD_REMAP: usize = 17;
const COMPUTE_ENTRIES: [&str; 18] = [
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
    "seedIrradiance",
    "prefilter",
    "gather",
    "temporal",
    "denoise",
    "traceSecondary",
    "commitCache",
    "remapProbeHeads",
];
const SLANG_OPTIMIZATION_LEVEL: u8 = 2;

fn cache_head_bytes(buckets: u64) -> u64 {
    // Bucket heads, retention count, irradiance heads, then four diagnostics.
    (buckets * 2 + 5) * 4
}

fn cache_work_slots(samples: u64, layers: u64) -> u64 {
    samples * layers * LEVELS as u64 * CACHE_WORK_GROUPS
}

fn cache_entry_bytes(samples: u64, layers: u64, buckets: u64) -> u64 {
    (cache_work_slots(samples, layers) + buckets * CACHE_HISTORY_PER_BUCKET) * ENTRY_SIZE
}

fn cache_history_bytes(buckets: u64) -> u64 {
    buckets * CACHE_HISTORY_PER_BUCKET * ENTRY_SIZE
}

fn view_budget_max_rays(viewport: UVec2, settings: RadianceCascades, budget: u64) -> u32 {
    let mut low = 1;
    let mut high = settings.max_rays.max(1);
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        let mut candidate = settings;
        candidate.max_rays = middle;
        if view_bytes(resolution(viewport, candidate), candidate) <= budget {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    low
}

fn view_fits_device(size: UVec2, settings: RadianceCascades, limits: &WgpuLimits) -> bool {
    let pixels = u64::from(size.x) * u64::from(size.y);
    let buckets = (pixels * 2).next_power_of_two().max(1024);
    let layers = 1 + u64::from(settings.bounce_feedback > 0.0);
    let binding_limit =
        u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size);
    let workgroups = u64::from(limits.max_compute_workgroups_per_dimension);
    size.max_element() <= limits.max_texture_dimension_2d
        && u64::from(size.x.div_ceil(8)) <= workgroups
        && u64::from(size.y.div_ceil(8)) <= workgroups
        && buckets.div_ceil(64) <= workgroups
        && cache_entry_bytes(pixels, layers, buckets) <= binding_limit
        && cache_history_bytes(buckets) <= binding_limit
        && cache_head_bytes(buckets) <= binding_limit
        && buckets * PROBE_SIZE * 2 <= binding_limit
}

fn view_device_max_rays(viewport: UVec2, settings: RadianceCascades, limits: &WgpuLimits) -> u32 {
    let mut low = 1;
    let mut high = settings.max_rays.max(1);
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        let candidate = RadianceCascades {
            max_rays: middle,
            ..settings
        };
        if view_fits_device(resolution(viewport, candidate), candidate, limits) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    low
}

fn required_pipelines(settings: RadianceCascades) -> u64 {
    let mut required = (1u64 << COMPUTE_ENTRIES.len()) - 1;
    if settings.bounce_feedback <= 0.0 {
        required &= !(1 << 15);
    }
    if settings.spatial_filter_steps == 0 {
        required &= !(1 << 14);
    }
    required
}

const MODULES: [(&str, &[&str]); 4] = [
    ("cascades.slang", &COMPUTE_ENTRIES),
    ("composite.slang", &["vertex", "fragment"]),
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

#[cfg(test)]
fn compile(
    compiler: &SlangCompiler,
    file: &str,
    entries: &[&str],
) -> Result<Shader, SlangLoaderError> {
    compile_backend(compiler, file, entries, false)
}

fn compile_backend(
    compiler: &SlangCompiler,
    file: &str,
    entries: &[&str],
    spirv: bool,
) -> Result<Shader, SlangLoaderError> {
    // Bevy's compositor shares a vertex/fragment WGSL module. Slang SPIR-V
    // outputs are per stage, so passthrough is used only for compute modules.
    let spirv =
        spirv && entries.len() == 1 && !matches!(file, "acceleration.slang" | "deformation.slang");
    let mut shader = compiler.compile_bundle(
        file,
        SOURCES,
        &SlangSettings {
            target: if spirv {
                SlangTarget::SpirV
            } else {
                SlangTarget::Wgsl
            },
            profile: spirv.then(|| "glsl_460+spirv_1_5".into()),
            entry_points: entries.iter().map(|entry| (*entry).to_owned()).collect(),
            optimization: Some(if spirv { 0 } else { SLANG_OPTIMIZATION_LEVEL }),
            ..default()
        },
    )?;
    // Native SPIR-V is trusted, embedded program code and is passed directly
    // to the driver. Keep wgpu's checks on the portable WGSL path.
    shader.validate_shader = if spirv {
        bevy::shader::ValidateShader::Disabled
    } else {
        bevy::shader::ValidateShader::Enabled
    };
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

fn shader_cache_target_key(key: u64, spirv: bool) -> u64 {
    if !spirv {
        return key;
    }
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut hasher);
    "spirv-1.5-O0-column-major".hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
fn cached_compile(
    compiler: &SlangCompiler,
    file: &str,
    entries: &[&str],
    cache_dir: &std::path::Path,
    compiler_version: Option<&str>,
) -> Result<Shader, SlangLoaderError> {
    cached_compile_backend(compiler, file, entries, cache_dir, compiler_version, false)
}

fn cached_compile_backend(
    compiler: &SlangCompiler,
    file: &str,
    entries: &[&str],
    cache_dir: &std::path::Path,
    compiler_version: Option<&str>,
    spirv: bool,
) -> Result<Shader, SlangLoaderError> {
    let spirv =
        spirv && entries.len() == 1 && !matches!(file, "acceleration.slang" | "deformation.slang");
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (cache_dir, compiler_version);
        return compile_backend(compiler, file, entries, false);
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::io::Write;

        let key = shader_cache_target_key(
            shader_cache_key(compiler, compiler_version, file, entries, SOURCES),
            spirv,
        );
        let extension = if spirv { "spv" } else { "wgsl" };
        let cache_path = cache_dir.join(format!("{key:016x}.{extension}"));
        let cached = std::fs::read(&cache_path).ok().and_then(|bytes| {
            if spirv {
                (bytes.len() >= 20
                    && bytes.len() % 4 == 0
                    && bytes[..4] == [0x03, 0x02, 0x23, 0x07])
                .then(|| Shader::from_spirv(bytes, file.to_owned()))
            } else {
                String::from_utf8(bytes).ok().map(|wgsl| {
                    let mut shader = Shader::from_wgsl(wgsl, file.to_owned());
                    shader.validate_shader = bevy::shader::ValidateShader::Enabled;
                    shader
                })
            }
        });
        if let Some(shader) = cached {
            debug!(file, entries = ?entries, path = %cache_path.display(), "GI shader cache hit");
            return Ok(shader);
        }

        debug!(file, entries = ?entries, path = %cache_path.display(), "GI shader cache miss");
        let shader = compile_backend(compiler, file, entries, spirv)?;
        let bytes = match &shader.source {
            bevy::shader::Source::Wgsl(wgsl) => Some(wgsl.as_bytes()),
            bevy::shader::Source::SpirV(bytes) => Some(bytes.as_ref()),
            _ => None,
        };
        if let Some(bytes) = bytes {
            if std::fs::create_dir_all(cache_dir).is_ok()
                && let Ok(mut cache_file) = tempfile::NamedTempFile::new_in(cache_dir)
                && cache_file.write_all(bytes).is_ok()
            {
                if let Err(error) = cache_file.persist(&cache_path) {
                    debug!(%error, path = %cache_path.display(), "GI shader cache write failed");
                }
            }
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

    #[test]
    fn native_shader_cache_is_separate_and_skips_the_compiler() {
        let directory = tempfile::tempdir().unwrap();
        let compiler = SlangCompiler::new("missing-slangc");
        let base = shader_cache_key(&compiler, None, "main.slang", &["main"], SOURCES);
        let native = super::shader_cache_target_key(base, true);
        assert_ne!(base, native);
        let bytes: Vec<_> = [0x0723_0203u32, 0x0001_0500, 0, 1, 0]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        let path = directory.path().join(format!("{native:016x}.spv"));
        std::fs::write(&path, &bytes).unwrap();
        let shader = super::cached_compile_backend(
            &compiler,
            "main.slang",
            &["main"],
            directory.path(),
            None,
            true,
        )
        .unwrap();
        assert!(
            matches!(shader.source, bevy::shader::Source::SpirV(ref data) if data.as_ref() == bytes)
        );
        std::fs::write(&path, [0u8; 20]).unwrap();
        assert!(
            super::cached_compile_backend(
                &compiler,
                "main.slang",
                &["main"],
                directory.path(),
                None,
                true,
            )
            .is_err()
        );
    }
}

#[derive(Clone, Copy, Default, ShaderType)]
struct Parameters {
    world_from_clip: Mat4,
    previous_clip_from_world: Mat4,
    previous_world_from_clip: Mat4,
    camera_position: Vec4,
    viewport: UVec4,
    size: UVec4,
    tuning: Vec4,
    environment: Vec4,
    history: Vec4,
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
}

#[derive(Resource)]
struct Pipelines {
    compute_layout: BindGroupLayoutDescriptor,
    surface_group_layout: BindGroupLayoutDescriptor,
    composite_layout: BindGroupLayoutDescriptor,
    compute: [Option<CachedComputePipelineId>; COMPUTE_ENTRIES.len()],
    native_compute: [Option<ComputePipeline>; COMPUTE_ENTRIES.len()],
    native_started: u64,
    native_jobs: Vec<Task<(usize, Result<ComputePipeline, String>, std::time::Duration)>>,
    native_cache: Option<wgpu::PipelineCache>,
    native_cache_path: Option<std::path::PathBuf>,
    timing: ComputePipelineTiming,
    composite: CachedRenderPipelineId,
    refit: CachedComputePipelineId,
    deform: [CachedComputePipelineId; 2],
}

impl Pipelines {
    fn pipeline<'a>(
        &'a self,
        index: usize,
        cache: &'a PipelineCache,
    ) -> Option<&'a ComputePipeline> {
        self.native_compute[index]
            .as_ref()
            .or_else(|| self.compute[index].and_then(|id| cache.get_compute_pipeline(id)))
    }
}

type NativeShaderBytes = Arc<Mutex<[Option<Arc<Vec<u8>>>; COMPUTE_ENTRIES.len()]>>;

// Slang supplies the executable SPIR-V, but wgpu also needs its entry-point
// metadata. Bevy 0.20's passthrough wrapper leaves that metadata empty.
fn spirv_workgroup(bytes: &[u8]) -> Result<(Vec<u32>, (u32, u32, u32)), String> {
    if bytes.len() < 20 || bytes.len() % 4 != 0 {
        return Err("invalid SPIR-V byte length".into());
    }
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect();
    if words[0] != 0x07230203 {
        return Err("invalid SPIR-V magic".into());
    }
    let mut offset = 5;
    while offset < words.len() {
        let count = (words[offset] >> 16) as usize;
        if count == 0 || offset + count > words.len() {
            return Err("invalid SPIR-V instruction".into());
        }
        // OpExecutionMode, LocalSize. Each cached module has one compute entry.
        if words[offset] & 0xffff == 16 && count == 6 && words[offset + 2] == 17 {
            let size = (words[offset + 3], words[offset + 4], words[offset + 5]);
            return Ok((words, size));
        }
        offset += count;
    }
    Err("SPIR-V compute entry has no static LocalSize".into())
}

async fn native_compute_pipeline(
    device: RenderDevice,
    layouts: [BindGroupLayout; 2],
    bytes: Arc<Vec<u8>>,
    entry: &'static str,
    pipeline_cache: Option<wgpu::PipelineCache>,
) -> Result<ComputePipeline, String> {
    let (words, workgroup_size) = spirv_workgroup(&bytes)?;
    let device = device.wgpu_device();
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    // Only our bundled, Slang-compiled sources enter this path. Their explicit
    // layouts match the bindings below; arbitrary user shaders remain WGSL.
    let module = unsafe {
        device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
            label: Some(entry),
            spirv: Some(std::borrow::Cow::Owned(words)),
            entry_points: std::borrow::Cow::Owned(vec![wgpu::PassthroughShaderEntryPoint {
                name: std::borrow::Cow::Borrowed(entry),
                workgroup_size,
            }]),
            ..default()
        })
    };
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("split_rc_native"),
        bind_group_layouts: &[Some(&layouts[0]), Some(&layouts[1])],
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
    if let Some(error) = scope.pop().await {
        return Err(error.to_string());
    }
    Ok(pipeline.into())
}

#[derive(Default)]
struct ComputePipelineTiming {
    current: Option<(usize, CachedComputePipelineId, std::time::Instant)>,
    creating: bool,
    warned: bool,
}

#[derive(Resource)]
struct PipelineShaders {
    handles: Vec<Handle<Shader>>,
    loaded: Arc<AtomicU64>,
    backend: Arc<AtomicU32>,
    native: NativeShaderBytes,
}

#[derive(Resource)]
struct ShaderCompilation {
    compiler: SlangCompiler,
    version: String,
    handles: Vec<Handle<Shader>>,
    loaded: Arc<AtomicU64>,
    backend: Arc<AtomicU32>,
    native: NativeShaderBytes,
    pending_loaded: u64,
    started: u64,
    jobs: Vec<Task<(usize, Result<Shader, SlangLoaderError>)>>,
}

fn required_shader_modules(settings: impl Iterator<Item = RadianceCascades>) -> u64 {
    let mut required = 0;
    for settings in settings {
        if settings.intensity <= 0.0 {
            continue;
        }
        required |= required_pipelines(settings);
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
    // RenderStartup chooses from the actual device features. Never send Slang
    // SPIR-V through Naga's SPIR-V reader on a backend without passthrough.
    let backend = compilation.backend.load(Ordering::Acquire);
    if backend == 0 {
        return;
    }
    // Give Bevy's render-asset extraction and ShaderCache one frame to see each
    // inserted shader before the render-world pipeline queue is allowed to use it.
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
                if index < COMPUTE_ENTRIES.len()
                    && let bevy::shader::Source::SpirV(ref bytes) = shader.source
                {
                    compilation.native.lock().unwrap()[index] = Some(Arc::new(bytes.to_vec()));
                }
                if let Err(error) = shaders.insert(compilation.handles[index].id(), shader) {
                    error!(%error, "GI shader handle expired");
                } else {
                    compilation.pending_loaded |= 1 << index;
                }
            }
            Err(error) => error!(%error, index, "GI shader compilation failed; native lighting remains available"),
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
    for index in [7, 12, 15, COMPUTE_ENTRIES.len(), COMPUTE_ENTRIES.len() + 1]
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
                let cache_dir = shader_cache_dir();
                (
                    index,
                    cached_compile_backend(
                        &compiler,
                        file,
                        entries,
                        &cache_dir,
                        Some(&version),
                        backend == 2,
                    ),
                )
            }));
    }
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

const NATIVE_AMBIENT_OFFSET: u64 = <GpuLights as ShaderType>::METADATA.offset(1);
const NATIVE_PROBE_COUNTS_OFFSET: u64 = <LightProbesUniform as ShaderType>::METADATA.offset(3);

fn native_lighting_patch_ranges(
    light_offset: u32,
    light_size: u64,
    probe_offset: u32,
    probe_size: u64,
) -> Option<(u64, u64)> {
    let ambient = u64::from(light_offset).checked_add(NATIVE_AMBIENT_OFFSET)?;
    let probes = u64::from(probe_offset).checked_add(NATIVE_PROBE_COUNTS_OFFSET)?;
    (ambient.checked_add(16)? <= light_size && probes.checked_add(4)? <= probe_size)
        .then_some((ambient, probes))
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
        Option<&ViewLightsUniformOffset>,
        Option<&ViewLightProbesUniformOffset>,
        &ExtractedCamera,
        Option<&RenderLayers>,
        Option<&crate::RadianceCascadesEnvironment>,
    )>,
    mut commands: Commands,
    scene: Res<GiScene>,
    gpu_scene: Option<Res<GpuScene>>,
    pipelines: Res<Pipelines>,
    cache: Res<PipelineCache>,
    queue: Res<RenderQueue>,
    lights: Option<Res<LightMeta>>,
    probes: Option<Res<LightProbesBuffer>>,
    environments: Option<Res<crate::environment::EnvironmentTables>>,
    budget: Option<Res<SceneBudget>>,
    mut startup: Local<HashMap<Entity, &'static str>>,
) {
    let entity = view.entity();
    let mut report = |status: &'static str| {
        if startup.insert(entity, status) != Some(status) {
            info!(?entity, status, "Split Radiance Cascades startup");
        }
    };
    let (
        settings,
        buffers,
        bindings,
        receiver,
        prepared,
        light_offset,
        probe_offset,
        camera,
        layers,
        environment,
    ) = view.into_inner();
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
            (required & (1 << i) != 0 && pipelines.pipeline(i, &cache).is_none())
                .then_some(COMPUTE_ENTRIES[i])
        })
        .collect();
    if !pending.is_empty() {
        report("waiting for GPU compute pipelines");
        return;
    }
    let (
        Some(buffers),
        Some(_bindings),
        Some(receiver),
        Some(prepared),
        Some(light_offset),
        Some(probe_offset),
    ) = (
        buffers,
        bindings,
        receiver,
        prepared,
        light_offset,
        probe_offset,
    )
    else {
        report("waiting for view buffers, bindings, receiver or native lighting uniforms");
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
    let required = required_pipelines(buffers.settings);
    if (0..COMPUTE_ENTRIES.len())
        .any(|i| required & (1 << i) != 0 && pipelines.pipeline(i, &cache).is_none())
        || cache.get_render_pipeline(pipelines.composite).is_none()
    {
        report("waiting for composite pipeline");
        return;
    }
    let (Some(light_buffer), Some(probe_buffer)) = (
        lights
            .as_ref()
            .and_then(|lights| lights.view_gpu_lights.buffer()),
        probes.as_ref().and_then(|probes| probes.buffer()),
    ) else {
        report("waiting for native lighting buffers");
        return;
    };
    let Some((ambient_offset, probes_offset)) = native_lighting_patch_ranges(
        light_offset.offset,
        light_buffer.size(),
        **probe_offset,
        probe_buffer.size(),
    ) else {
        report("native lighting buffer ranges are invalid");
        return;
    };
    // Only indirect fields change: all punctual lights and shadow bindings remain native.
    queue.write_buffer(light_buffer, ambient_offset, &[0u8; 16]);
    queue.write_buffer(probe_buffer, probes_offset, &[0u8; 4]);
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

pub(crate) fn install(app: &mut App, compiler: SlangCompiler) {
    if app.get_sub_app(RenderApp).is_none() {
        return;
    }
    let compiler_version = match compiler_version(&compiler) {
        Ok(version) => version,
        Err(error) => {
            error!("Radiance Cascades disabled: {error}");
            return;
        }
    };
    let handles: Vec<_> = shader_modules()
        .map(|_| app.world().resource::<Assets<Shader>>().reserve_handle())
        .collect();
    let loaded = Arc::new(AtomicU64::new(0));
    let backend = Arc::new(AtomicU32::new(0));
    let native = Arc::new(Mutex::new(std::array::from_fn(|_| None)));
    app.insert_resource(ShaderCompilation {
        compiler,
        version: compiler_version,
        handles: handles.clone(),
        loaded: loaded.clone(),
        backend: backend.clone(),
        native: native.clone(),
        pending_loaded: 0,
        started: 0,
        jobs: Vec::new(),
    })
    .add_systems(Update, compile_view_shaders);
    crate::receiver::install(app);
    app.sub_app_mut(RenderApp)
        .insert_resource(PipelineShaders {
            handles,
            loaded,
            backend,
            native,
        })
        .init_resource::<MemoryHighWater>()
        .init_resource::<RasterResidencyGate>()
        .add_systems(
            Render,
            cleanup_view_allocations
                .after(RenderSystems::ExtractCommands)
                .before(RenderSystems::CreateViews),
        )
        .add_systems(RenderStartup, init_pipelines)
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
            render
                .in_set(Core3dSystems::MainPass)
                .after(crate::receiver::draw)
                .after(main_opaque_pass_3d)
                .before(main_transparent_pass_3d),
        );
}

fn compute_layout() -> BindGroupLayoutDescriptor {
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
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Uint),
                texture_2d(TextureSampleType::Uint),
                texture_2d_array(TextureSampleType::Uint),
                texture_2d_array(TextureSampleType::Uint),
            ),
        ),
    );
    layout
        .entries
        .retain(|entry| !matches!(entry.binding, 17 | 18 | 19 | 24));
    layout
}

fn surface_group_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "split_rc_surface_groups",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::R32Uint, StorageTextureAccess::WriteOnly),
                texture_2d(TextureSampleType::Uint),
                texture_2d(TextureSampleType::Uint),
            ),
        ),
    )
}

fn init_pipelines(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    shaders: Res<PipelineShaders>,
    adapter: Res<RenderAdapterInfo>,
) {
    let spirv = device
        .features()
        .contains(wgpu::Features::PASSTHROUGH_SHADERS);
    shaders
        .backend
        .store(if spirv { 2 } else { 1 }, Ordering::Release);
    info!(
        backend = if spirv {
            "Slang SPIR-V passthrough"
        } else {
            "Slang WGSL"
        },
        "GI shader backend selected"
    );
    let handles = &shaders.handles;
    let native_cache_path = device
        .features()
        .contains(wgpu::Features::PIPELINE_CACHE)
        .then(|| wgpu::util::pipeline_cache_key(&adapter))
        .flatten()
        .map(|key| shader_cache_dir().join(format!("{key}.bin")));
    let native_cache = native_cache_path.as_ref().map(|path| {
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
    let compute_layout = compute_layout();
    let surface_group_layout = surface_group_layout();
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
                texture_2d(TextureSampleType::Uint),
            ),
        ),
    );
    composite_layout.entries.retain(|entry| entry.binding != 6);
    let composite = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("split_rc_composite".into()),
        layout: vec![composite_layout.clone()],
        vertex: VertexState {
            shader: handles[COMPUTE_ENTRIES.len()].clone(),
            entry_point: Some("vertex".into()),
            ..default()
        },
        fragment: Some(FragmentState {
            shader: handles[COMPUTE_ENTRIES.len()].clone(),
            entry_point: Some("fragment".into()),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba16Float,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
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
        surface_group_layout,
        composite_layout,
        compute: [None; COMPUTE_ENTRIES.len()],
        native_compute: std::array::from_fn(|_| None),
        native_started: 0,
        native_jobs: Vec::new(),
        native_cache,
        native_cache_path,
        timing: default(),
        composite,
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
    let side = limits.max_texture_dimension_2d.min(1024).max(1);
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

fn scene_page_ranges(data: &crate::scene::SceneData) -> Option<(Vec<(usize, usize)>, Vec<UVec4>)> {
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
) -> Option<(ScenePageLayout, Arc<Vec<UVec4>>, Vec<UVec4>)> {
    let prefix_words = data.acceleration_prefix.len();
    let instance_base = prefix_words.checked_add(data.blas_words.len())?;
    let dynamic_instance_words = material_start;
    let (ranges, range_map) = scene_page_ranges(data)?;
    let blas_nodes = data.blas_words.len() / 2;
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
    let side = limits.max_texture_dimension_2d.min(1024).max(1);
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
    let mut pages = Vec::with_capacity(page_word_count);
    let mut range_index = 0;
    for node in 0..blas_nodes {
        if ranges
            .get(range_index)
            .is_some_and(|&(start, count)| (start..start + count).contains(&node))
        {
            if node + 1 == ranges[range_index].0 + ranges[range_index].1 {
                range_index += 1;
            }
            continue;
        }
        pages.extend_from_slice(&data.blas_words[node * 2..node * 2 + 2]);
    }
    pages.extend_from_slice(&data.instance_words[dynamic_instance_words..]);
    if pages.is_empty() {
        pages.push(UVec4::ZERO);
    }
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
        Arc::new(pages),
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
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
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
    page_words: Arc<Vec<UVec4>>,
    page_cursor: usize,
    paged: bool,
    allocated_bytes: u64,
    words: Vec<UVec4>,
    blas_words: Arc<Vec<UVec4>>,
    material_words: Vec<UVec4>,
    topology_revision: u64,
    prefix_words: usize,
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
        page_words: Arc<Vec<UVec4>>,
        page_layout: Option<ScenePageLayout>,
        blas_words: Arc<Vec<UVec4>>,
        material_words: Vec<UVec4>,
        topology_revision: u64,
        prefix_words: usize,
        full_instance_base: usize,
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
            page_cursor: 0,
            paged,
            allocated_bytes,
            words,
            blas_words,
            material_words,
            topology_revision,
            prefix_words,
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
            upload_texture_words(
                queue,
                texture,
                self.page_config.y,
                &self.page_words,
                &mut self.page_cursor,
                remaining,
            );
        }
        self.cursor == self.words.len() && self.page_cursor == self.page_words.len()
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
    cursor: usize,
}

impl PendingDynamic {
    fn upload(&mut self, queue: &RenderQueue, buffer: &Buffer, remaining: &mut usize) -> bool {
        let total = self.prefix.len() + self.instance_words;
        while self.cursor < total && *remaining >= 16 {
            let count = (total - self.cursor).min(*remaining / 16).min(65536);
            if self.cursor < self.prefix.len() {
                let end = (self.cursor + count).min(self.prefix.len());
                let words = &self.prefix[self.cursor..end];
                queue.write_buffer(buffer, (self.cursor * 16) as u64, &words_bytes(words));
                let uploaded = words.len() * 16;
                self.cursor = end;
                *remaining -= uploaded;
                continue;
            }
            let offset = self.cursor - self.prefix.len();
            let words = &self.instances[offset..offset + count];
            queue.write_buffer(
                buffer,
                ((self.instance_overlay_base + offset) * 16) as u64,
                &words_bytes(words),
            );
            let uploaded = words.len() * 16;
            self.cursor += words.len();
            *remaining -= uploaded;
        }
        self.cursor == total
    }
}

struct PendingAtlas {
    texture: Texture,
    view: TextureView,
    atlas: Arc<crate::material::TextureAtlas>,
    layer: usize,
    cursor: usize,
}

struct PendingLights {
    buffer: Buffer,
    source: Option<(bevy::asset::AssetId<Image>, u64)>,
    layers: RenderLayers,
    prefix: Vec<GiLight>,
    prefix_cursor: usize,
    replacement: bool,
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

fn words_bytes(words: &[UVec4]) -> Vec<u8> {
    words
        .iter()
        .flat_map(|word| word.to_array().into_iter().flat_map(u32::to_le_bytes))
        .collect()
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

struct CacheBuffers {
    heads: Buffer,
}

#[derive(Component)]
struct ViewBuffers {
    size: UVec2,
    ray_layers: u32,
    buckets: u32,
    caches: [CacheBuffers; 2],
    entries: Buffer,
    history_entries: Buffer,
    irradiance_probes: Buffer,
    irradiance_textures: [Texture; 4],
    irradiance_views: [TextureView; 4],
    _metadata: Vec<Texture>,
    metadata_views: [TextureView; 5],
    _surface_group_textures: [Texture; 3],
    surface_group_views: [TextureView; 3],
    _unused_outputs: [Texture; 1],
    unused_output_views: [TextureView; 1],
    rendered_frame: Mutex<RenderedFrame>,
    viewport: UVec4,
    frame: AtomicU32,
    history_valid: AtomicBool,
    screen_history_valid: AtomicBool,
    revision: u64,
    settings: RadianceCascades,
    mesh_revision: u64,
    geometry_revision: u64,
    render_layers: RenderLayers,
    lights: Option<Buffer>,
    light_source: Option<(bevy::asset::AssetId<Image>, u64)>,
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
    exposure: f32,
}

#[derive(Component)]
struct ViewBindings {
    compute: [BindGroup; LEVELS],
    surface_groups: [BindGroup; 2],
    denoise: [BindGroup; LEVELS],
    temporal: BindGroup,
    composite: [(TextureViewId, BindGroup); 2],
    frame: RenderedFrame,
}

fn resolution(viewport: UVec2, settings: RadianceCascades) -> UVec2 {
    let size_at = |divisor: u32| {
        UVec2::new(
            viewport.x.div_ceil(divisor).max(1),
            viewport.y.div_ceil(divisor).max(1),
        )
    };
    let max_rays = u64::from(settings.max_rays.max(1));
    let max_receivers = max_rays;
    let mut low = settings.resolution_divisor.max(1);
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

fn invalidate_screen_history_for_scene_change(
    previous_revision: u64,
    revision: u64,
    geometry_changed: bool,
    has_motion_vectors: bool,
) -> bool {
    !has_motion_vectors && (revision.wrapping_sub(previous_revision) != 1 || geometry_changed)
}

fn discard_obsolete_scene(live: u64, candidate: u64, budget: u64, obsolete: bool) -> bool {
    obsolete && candidate <= budget && live > budget
}

fn view_bytes(size: UVec2, settings: RadianceCascades) -> u64 {
    let pixels = u64::from(size.x) * u64::from(size.y);
    let buckets = (pixels * 2).next_power_of_two().max(1024);
    let layers = 1 + u64::from(settings.bounce_feedback > 0.0);
    // Four irradiance images, five metadata images, three R32Uint identity images,
    // one 1x1 storage placeholder and four cascade/filter uniforms.
    pixels * (9 * 16 + 3 * 4)
        + 16
        + LEVELS as u64 * std::mem::size_of::<Parameters>() as u64
        + buckets * PROBE_SIZE * 2
        + cache_head_bytes(buckets) * 2
        + cache_entry_bytes(pixels, layers, buckets)
        + cache_history_bytes(buckets)
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

    #[allow(clippy::too_many_arguments)]
    fn prepare_lights(
        &mut self,
        device: &RenderDevice,
        queue: &RenderQueue,
        prefix: Vec<GiLight>,
        layers: &RenderLayers,
        source: Option<(bevy::asset::AssetId<Image>, u64)>,
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
            });
        } else if let Some(pending) = self.pending_lights.as_mut()
            && pending.prefix != prefix
        {
            // Restart staging when moving emitters change the lighting snapshot.
            pending.prefix = prefix;
            pending.prefix_cursor = 0;
        }
        let pending = self.pending_lights.as_mut().unwrap();
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
        let pending = self.pending_lights.take().unwrap();
        self.lights = Some(pending.buffer);
        self.light_source = source;
        true
    }

    fn new(device: &RenderDevice, size: UVec2, settings: RadianceCascades, revision: u64) -> Self {
        let samples = size.x * size.y;
        let ray_layers = 1 + u32::from(settings.bounce_feedback > 0.0);
        let cache_samples = samples;
        let buckets = (samples * 2).next_power_of_two().max(1024);
        let caches = std::array::from_fn(|_| CacheBuffers {
            heads: device.create_buffer(&BufferDescriptor {
                label: Some("split_rc_hash_heads"),
                size: cache_head_bytes(u64::from(buckets)),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
        });
        let entries = device.create_buffer(&BufferDescriptor {
            label: Some("split_rc_interval_work"),
            size: cache_entry_bytes(
                u64::from(cache_samples),
                u64::from(ray_layers),
                u64::from(buckets),
            ),
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let history_entries = device.create_buffer(&BufferDescriptor {
            label: Some("split_rc_interval_history"),
            size: cache_history_bytes(u64::from(buckets)),
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let irradiance_probes = device.create_buffer(&BufferDescriptor {
            label: Some("split_rc_irradiance_probes"),
            size: u64::from(buckets) * PROBE_SIZE * 2,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let make_texture = || {
            device.create_texture(&TextureDescriptor {
                label: Some("split_rc_irradiance"),
                size: Extent3d {
                    width: size.x,
                    height: size.y,
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
        let irradiance: [Texture; 4] = std::array::from_fn(|_| make_texture());
        let unused_outputs: [Texture; 1] = std::array::from_fn(|_| {
            device.create_texture(&TextureDescriptor {
                label: Some("gi_unused_storage_output"),
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
            })
        });
        let unused_output_views = std::array::from_fn(|i| {
            unused_outputs[i].create_view(&TextureViewDescriptor::default())
        });
        let irradiance_views =
            std::array::from_fn(|i| irradiance[i].create_view(&TextureViewDescriptor::default()));
        let metadata: Vec<Texture> = (0..5).map(|_| make_texture()).collect();
        let metadata_views: [TextureView; 5] =
            std::array::from_fn(|i| metadata[i].create_view(&TextureViewDescriptor::default()));
        let group_size = size;
        let surface_group_textures: [Texture; 3] = std::array::from_fn(|_| {
            device.create_texture(&TextureDescriptor {
                label: Some("split_rc_surface_groups"),
                size: Extent3d {
                    width: group_size.x,
                    height: group_size.y,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::R32Uint,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        });
        let surface_group_views = std::array::from_fn(|i| {
            surface_group_textures[i].create_view(&TextureViewDescriptor::default())
        });
        Self {
            size,
            ray_layers,
            buckets,
            caches,
            entries,
            history_entries,
            irradiance_probes,
            irradiance_textures: irradiance,
            irradiance_views,
            _metadata: metadata,
            metadata_views,
            _surface_group_textures: surface_group_textures,
            surface_group_views,
            _unused_outputs: unused_outputs,
            unused_output_views,
            rendered_frame: Mutex::new(RenderedFrame::default()),
            viewport: UVec4::ZERO,
            frame: AtomicU32::new(0),
            history_valid: AtomicBool::new(false),
            screen_history_valid: AtomicBool::new(false),
            revision,
            settings,
            mesh_revision: 0,
            geometry_revision: 0,
            render_layers: RenderLayers::default(),
            lights: None,
            light_source: None,
            pending_lights: None,
            conventional_lights: 0,
            layer_selector: UVec4::ZERO,
            environment: None,
            environment_revision: 0,
            environment_info: UVec4::ZERO,
        }
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
    images: Res<RenderAssets<GpuImage>>,
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
            &RadianceCascades,
            Option<&RenderLayers>,
            Option<&crate::RadianceCascadesEnvironment>,
            &bevy::render::view::ViewDepthStencilTexture,
            Option<&crate::receiver::ReceiverTexture>,
            Option<&bevy::pbr::ScreenSpaceAmbientOcclusionResources>,
            Option<&bevy::pbr::SsaoPipelineId>,
            Option<&mut ViewBuffers>,
        )>,
    )>,
) {
    // Count every retained allocation, even when native view/prepass resources
    // are temporarily absent and that camera cannot be prepared this frame.
    let mut allocated_views = view_queries
        .p0()
        .iter()
        .map(|(buffers, receiver)| {
            receiver
                .map_or(0, |receiver| receiver.bytes())
                .saturating_add(buffers.map_or(0, |buffers| {
                    view_bytes(buffers.size, buffers.settings).saturating_add(buffers.light_bytes())
                }))
        })
        .sum::<u64>()
        .saturating_add(residency.p0().bytes);
    let mut views = view_queries.p1();
    let required = views
        .iter()
        .fold(0u64, |mask, (_, _, _, _, _, settings, ..)| {
            let settings = settings.sanitized();
            mask | if settings.intensity > 0.0 {
                required_pipelines(settings)
            } else {
                0
            }
        });
    if let Some((index, id, queued_at)) = pipelines.timing.current {
        match cache.get_compute_pipeline_state(id) {
            CachedPipelineState::Creating(..) => {
                if !pipelines.timing.creating {
                    info!(
                        entry = COMPUTE_ENTRIES[index],
                        queued_ms = ?queued_at.elapsed(),
                        "GI compute pipeline creation started",
                    );
                    pipelines.timing.creating = true;
                }
                if !pipelines.timing.warned
                    && queued_at.elapsed() >= std::time::Duration::from_secs(30)
                {
                    warn!(
                        entry = COMPUTE_ENTRIES[index],
                        elapsed = ?queued_at.elapsed(),
                        "GI compute pipeline remains in device creation",
                    );
                    pipelines.timing.warned = true;
                }
            }
            CachedPipelineState::Ok(..) => {
                info!(
                    entry = COMPUTE_ENTRIES[index],
                    elapsed = ?queued_at.elapsed(),
                    "GI compute pipeline ready",
                );
                pipelines.timing.current = None;
                pipelines.timing.creating = false;
                pipelines.timing.warned = false;
            }
            CachedPipelineState::Err(error) => {
                error!(
                    entry = COMPUTE_ENTRIES[index],
                    elapsed = ?queued_at.elapsed(),
                    %error,
                    "GI compute pipeline failed",
                );
                pipelines.timing.current = None;
                pipelines.timing.creating = false;
                pipelines.timing.warned = false;
            }
            CachedPipelineState::Queued => {}
        }
    }
    let mut job = 0;
    let mut cache_changed = false;
    while job < pipelines.native_jobs.len() {
        if let Some((index, result, elapsed)) = block_on(poll_once(&mut pipelines.native_jobs[job]))
        {
            let _ = pipelines.native_jobs.swap_remove(job);
            match result {
                Ok(pipeline) => {
                    info!(
                        entry = COMPUTE_ENTRIES[index],
                        ?elapsed,
                        "GI native compute pipeline ready"
                    );
                    pipelines.native_compute[index] = Some(pipeline);
                    cache_changed = true;
                }
                Err(error) => {
                    error!(entry = COMPUTE_ENTRIES[index], ?elapsed, %error, "GI native compute pipeline failed")
                }
            }
        } else {
            job += 1;
        }
    }
    if cache_changed {
        if let (Some(cache), Some(path)) = (&pipelines.native_cache, &pipelines.native_cache_path) {
            if let Some(data) = cache.get_data() {
                let path = path.clone();
                AsyncComputeTaskPool::get()
                    .spawn(async move {
                        let result = (|| -> std::io::Result<()> {
                            let parent = path.parent().unwrap();
                            std::fs::create_dir_all(parent)?;
                            let mut temp = tempfile::NamedTempFile::new_in(parent)?;
                            std::io::Write::write_all(&mut temp, &data)?;
                            temp.persist(&path).map_err(|error| error.error)?;
                            Ok(())
                        })();
                        if let Err(error) = result {
                            warn!(%error, "Could not save GI device pipeline cache");
                        }
                    })
                    .detach();
            }
        }
    }
    // Keep two device pipeline creations in flight; larger batches compete
    // with Bevy's frame rendering on the same async compute pool.
    let mut compiling = pipelines
        .compute
        .iter()
        .filter(|id| {
            id.is_some_and(|id| {
                matches!(
                    cache.get_compute_pipeline_state(id),
                    CachedPipelineState::Queued | CachedPipelineState::Creating(..)
                )
            })
        })
        .count()
        + pipelines.native_jobs.len();
    // Trace and gather dominate cold Vulkan compilation. Queue them before the
    // short kernels so they can compile while the rest of the frame pipeline is
    // being prepared.
    for i in [7, 12, 15]
        .into_iter()
        .chain((0..COMPUTE_ENTRIES.len()).filter(|i| !matches!(i, 7 | 12 | 15)))
    {
        let entry = &COMPUTE_ENTRIES[i];
        if compiling >= 2 {
            break;
        }
        if required & (1 << i) != 0
            && pipelines.compute[i].is_none()
            && pipelines.native_started & (1 << i) == 0
            && shaders.loaded.load(Ordering::Acquire) & (1 << i) != 0
        {
            if shaders.backend.load(Ordering::Acquire) == 2 {
                let Some(bytes) = shaders.native.lock().unwrap()[i].clone() else {
                    continue;
                };
                let device = device.clone();
                let layouts = [
                    cache
                        .get_bind_group_layout(&pipelines.compute_layout)
                        .clone(),
                    cache
                        .get_bind_group_layout(&pipelines.surface_group_layout)
                        .clone(),
                ];
                let entry = *entry;
                let pipeline_cache = pipelines.native_cache.clone();
                pipelines.native_started |= 1 << i;
                info!(entry, "GI native compute pipeline creation started");
                pipelines
                    .native_jobs
                    .push(AsyncComputeTaskPool::get().spawn(async move {
                        // wgpu's scopes are thread-local. Keep creation and scope
                        // pop on a private thread, isolated from Bevy's movable
                        // pipeline-cache tasks on the async pool.
                        std::thread::spawn(move || {
                            let start = std::time::Instant::now();
                            let result = block_on(native_compute_pipeline(
                                device, layouts, bytes, entry, pipeline_cache,
                            ));
                            (i, result, start.elapsed())
                        }).join().unwrap_or_else(|_| (
                            i, Err("native pipeline worker panicked".into()), std::time::Duration::ZERO,
                        ))
                    }));
                compiling += 1;
                continue;
            }
            let first_layout = pipelines.compute_layout.clone();
            let id = cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(format!("split_rc_{entry}").into()),
                layout: vec![first_layout, pipelines.surface_group_layout.clone()],
                shader: shaders.handles[i].clone(),
                entry_point: Some((*entry).into()),
                ..default()
            });
            pipelines.compute[i] = Some(id);
            if pipelines.timing.current.is_none() {
                pipelines.timing.current = Some((i, id, std::time::Instant::now()));
                pipelines.timing.creating = false;
                pipelines.timing.warned = false;
            }
            compiling += 1;
            debug!(entry = *entry, "Queued GI compute pipeline compilation");
        }
    }
    if !scene.ready
        || (scene.data.deformation.vertices > 0
            && pipelines
                .deform
                .iter()
                .any(|id| cache.get_compute_pipeline(*id).is_none()))
        || scene.data.textures.side > device.limits().max_texture_dimension_2d
        || scene.data.textures.pages.len() > device.limits().max_texture_array_layers as usize
    {
        for (entity, .., buffers) in &mut views {
            if let Some(buffers) = buffers {
                buffers.screen_history_valid.store(false, Ordering::Relaxed);
                buffers.history_valid.store(false, Ordering::Relaxed);
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
    if device.limits().max_storage_buffers_per_shader_stage < 8 {
        warn_once!("Radiance Cascades needs eight storage buffers per compute stage");
        return;
    }
    // The shared layouts contain eight storage buffers, fourteen sampled images
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
                && old.material_words == material_words
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
                buffers.history_valid.store(false, Ordering::Relaxed);
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
                buffers.history_valid.store(false, Ordering::Relaxed);
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
            || old.scene_paged != scene_paged
            || old.scene_page_config != scene_page_config
            || old.scene_overlay_config != scene_overlay_config
            || old.scene_range_map != scene_range_map
            || !Arc::ptr_eq(&old.blas_words, &data.blas_words)
            || old.material_words != material_words
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
                buffers.history_valid.store(false, Ordering::Relaxed);
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
    // Emissive probability records precede materials and vary with motion.
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
        staged.pending_refit_order = None;
        staged.pending_refit_revision = None;
    }
    if !replace_acceleration {
        staged.pending_acceleration = None;
    }
    if replace_acceleration || staged.pending_acceleration.is_some() {
        if staged.pending_acceleration.is_none() {
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
                    (words, Arc::new(Vec::new()), None)
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
                scene.geometry_revision,
            ));
        }
        let pending = staged
            .pending_acceleration
            .as_mut()
            .expect("acceleration staged");
        pending.upload(&queue, &mut upload_remaining);
    }
    let pending_revision = staged
        .pending_acceleration
        .as_ref()
        .map_or(staged.dynamic_revision, |pending| pending.dynamic_revision);
    let partial_dynamic = staged
        .pending_dynamic
        .as_ref()
        .is_some_and(|pending| pending.cursor > 0);
    let acceleration_incomplete = staged.pending_acceleration.as_ref().is_some_and(|pending| {
        pending.cursor < pending.words.len() || pending.page_cursor < pending.page_words.len()
    });
    if pending_revision == scene.geometry_revision {
        if partial_dynamic && !acceleration_incomplete {
            staged.pending_dynamic = Some(PendingDynamic {
                revision: scene.geometry_revision,
                prefix: data.acceleration_prefix.clone(),
                instances: data.instance_words.clone(),
                instance_words: material_start,
                instance_overlay_base,
                cursor: 0,
            });
        } else {
            staged.pending_dynamic = None;
        }
    } else if staged.pending_dynamic.as_ref().is_none_or(|pending| {
        pending.revision != scene.geometry_revision
            || pending.instance_overlay_base != instance_overlay_base
    }) {
        staged.pending_dynamic = Some(PendingDynamic {
            revision: scene.geometry_revision,
            prefix: data.acceleration_prefix.clone(),
            instances: data.instance_words.clone(),
            instance_words: material_start,
            instance_overlay_base,
            cursor: 0,
        });
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
        pending.cursor == pending.words.len() && pending.page_cursor == pending.page_words.len()
    });
    if acceleration_ready {
        let target = staged
            .pending_acceleration
            .as_ref()
            .map(|pending| &pending.buffer)
            .or(staged.nodes.as_ref());
        let uploaded_revision =
            if let (Some(pending), Some(buffer)) = (staged.pending_dynamic.as_mut(), target) {
                pending
                    .upload(&queue, buffer, &mut upload_remaining)
                    .then_some(pending.revision)
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
            pending.cursor < pending.words.len() || pending.page_cursor < pending.page_words.len()
        })
        || staged.pending_dynamic.is_some()
    {
        staged.pending_deformation = false;
        staged.pending_refit = false;
        for (entity, .., buffers) in &mut views {
            if let Some(buffers) = buffers {
                buffers.screen_history_valid.store(false, Ordering::Relaxed);
                buffers.history_valid.store(false, Ordering::Relaxed);
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
                buffers.history_valid.store(false, Ordering::Relaxed);
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
        settings,
        render_layers,
        environment,
        view_depth,
        receiver,
        ssao,
        ssao_pipeline,
        existing,
    ) in &mut views
    {
        if !residency.p1().gated.is_empty() {
            if let Some(buffers) = existing.as_ref() {
                buffers.history_valid.store(false, Ordering::Relaxed);
                buffers.screen_history_valid.store(false, Ordering::Relaxed);
            }
            commands
                .entity(entity)
                .remove::<(ViewBindings, GiPreparedView)>();
            continue;
        }
        let requested_settings = settings.sanitized();
        let mut settings = requested_settings;
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
                buffers.history_valid.store(false, Ordering::Relaxed);
                buffers.screen_history_valid.store(false, Ordering::Relaxed);
            }
            commands
                .entity(entity)
                .remove::<(ViewBindings, GiPreparedView)>();
            continue;
        }
        let layers = 1 + u32::from(settings.bounce_feedback > 0.0);
        let (Some(depth), Some(normal), Some(gbuffer)) = (
            prepass.depth_only_view(),
            prepass.normal_view(),
            prepass.deferred_view(),
        ) else {
            if let Some(buffers) = existing {
                buffers.history_valid.store(false, Ordering::Relaxed);
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
            if let Some(buffers) = existing {
                buffers.history_valid.store(false, Ordering::Relaxed);
                buffers.screen_history_valid.store(false, Ordering::Relaxed);
            }
            commands
                .entity(entity)
                .remove::<(ViewBindings, GiPreparedView)>();
            continue;
        }
        let viewport = UVec2::new(view.viewport.z, view.viewport.w);
        settings.max_rays = view_budget_max_rays(viewport, settings, budget.view_bytes);
        settings.max_rays = view_device_max_rays(viewport, settings, &device.limits());
        let size = resolution(viewport, settings);
        if !view_fits_device(size, settings, &device.limits()) {
            warn_once!("GI minimum cascade allocation exceeds this device's limits");
            commands
                .entity(entity)
                .remove::<(ViewBindings, GiPreparedView)>();
            continue;
        }
        let mut created_receiver = None;
        if receiver.is_none_or(|receiver| !receiver.matches(view_depth)) {
            let bytes = u64::from(view_depth.texture().width())
                * u64::from(view_depth.texture().height())
                * u64::from(view_depth.texture().sample_count())
                * 16;
            if allocated_views.saturating_add(bytes) > budget.view_bytes {
                warn_once!(
                    "GI receiver identity texture/resize overlap exceeds the view GPU memory budget"
                );
                if let Some(buffers) = existing.as_ref() {
                    buffers.history_valid.store(false, Ordering::Relaxed);
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
            v.size != size || v.ray_layers != layers || v.settings.max_rays != settings.max_rays
        }) {
            let candidate = view_bytes(size, settings);
            if allocated_views.saturating_add(candidate) > budget.view_bytes {
                warn_once!(
                    "GI camera allocations/resize overlap exceed the view GPU memory budget"
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
            created = Some(ViewBuffers::new(&device, size, settings, scene.revision));
        }
        let Some(buffers) = created.as_mut().or(existing.as_deref_mut()) else {
            continue;
        };
        if allocated_views > budget.view_bytes {
            warn_once!("GI live camera/light allocations exceed the view GPU memory budget");
            buffers.history_valid.store(false, Ordering::Relaxed);
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
        // Exposure participates in emissive_exposure_weight at ray hits.
        // Compare with the last encoded frame so skipped renders retain the reset.
        let exposure_changed = rendered.exposure != camera.exposure;
        let scene_changed = buffers.revision != scene.revision;
        let meshes_changed = buffers.mesh_revision != scene.mesh_revision;
        let geometry_changed = buffers.geometry_revision != scene.geometry_revision;
        let default_layers = RenderLayers::default();
        let render_layers = render_layers.unwrap_or(&default_layers);
        let layers_changed = buffers.render_layers != *render_layers;
        let environment_revision = environment_table.map_or(0, |table| table.revision);
        let environment_changed = buffers.environment.as_ref() != environment
            || buffers.environment_revision != environment_revision;
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
            let (lights, conventional, selector) = view_lights(&scene.data, render_layers);
            let environment_info = environment_table.map_or(UVec4::ZERO, |table| {
                UVec4::new(0, table.width, table.height, 1)
            });
            let source = environment.map(|source| (source.image.id(), environment_revision));
            if !buffers.prepare_lights(
                &device,
                &queue,
                lights,
                render_layers,
                source,
                &mut allocated_views,
                budget.view_bytes,
                &mut upload_remaining,
            ) {
                buffers.history_valid.store(false, Ordering::Relaxed);
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
            if scene_changed
                && invalidate_screen_history_for_scene_change(
                    buffers.revision,
                    scene.revision,
                    geometry_changed,
                    prepass.motion_vectors_view().is_some(),
                )
            {
                // Without motion vectors, camera-only reprojection cannot
                // follow geometry edits or identify edits missed by this view.
                buffers.screen_history_valid.store(false, Ordering::Relaxed);
            }
            if layers_changed
                || environment_changed
                || exposure_changed
                || meshes_changed
                || buffers.settings != settings
                || buffers.viewport != view.viewport
            {
                buffers.screen_history_valid.store(false, Ordering::Relaxed);
            }
            // Keep cache parity and the R2 sequence advancing during motion.
            // Only history validity resets; restarting samples repeats noise.
            buffers.history_valid.store(false, Ordering::Relaxed);
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
        let frame = buffers.frame.load(Ordering::Relaxed);

        let current = &buffers.caches[(frame % 2) as usize];
        let previous = &buffers.caches[(frame.wrapping_add(1) % 2) as usize];
        let irradiance = &buffers.irradiance_views[(frame % 2) as usize];
        let previous_irradiance = &buffers.irradiance_views[(frame.wrapping_add(1) % 2) as usize];
        let metadata = &buffers.metadata_views[(frame % 2) as usize];
        let previous_metadata = &buffers.metadata_views[(frame.wrapping_add(1) % 2) as usize];
        let clip_from_world = view.clip_from_world.unwrap_or_else(|| {
            view.clip_from_view * Mat4::from(view.world_from_view.affine().inverse())
        });
        let params = Parameters {
            world_from_clip: clip_from_world.inverse(),
            previous_clip_from_world: rendered.clip_from_world,
            previous_world_from_clip: rendered.world_from_clip,
            camera_position: view.world_from_view.translation().extend(1.0),
            viewport: view.viewport,
            size: UVec4::new(size.x, size.y, size.x * size.y, buffers.buckets),
            tuning: Vec4::new(
                settings.probe_spacing,
                settings.interval_length,
                settings.normal_bias,
                settings.intensity,
            ),
            environment: settings.environment.to_vec4().with_w(camera.exposure),
            history: Vec4::new(
                settings.temporal_blend,
                settings.bounce_feedback,
                if buffers.history_valid.load(Ordering::Relaxed) {
                    1.0
                } else {
                    0.0
                },
                settings.history_frames as f32,
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
                0,
            ),
            stability: Vec4::new(
                if buffers.screen_history_valid.load(Ordering::Relaxed) {
                    1.0
                } else {
                    0.0
                },
                if scene_changed || layers_changed {
                    1.0
                } else {
                    0.0
                },
                f32::from(prepass.motion_vectors_view().is_some()),
                0.0,
            ),
            layers: UVec4::new(
                buffers.layer_selector.x,
                buffers.layer_selector.y,
                u32::from(settings.artistic_occlusion) | (u32::from(ssao_valid) << 1),
                1,
            ),
            environment_info: buffers.environment_info,
            environment_rotation,
            environment_scale: Vec4::new(environment_intensity, 0.0, 0.0, 0.0),
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
        };
        let uniforms: [UniformBuffer<Parameters>; LEVELS] = std::array::from_fn(|level| {
            let mut p = params;
            p.frame.y = (level % LEVELS) as u32;
            let mut buffer = UniformBuffer::from(p);
            buffer.write_buffer(&device, &queue);
            buffer
        });
        // Temporal reconstruction keeps receiver history separate from scratch storage.
        let bind_compute = |level: usize,
                            output: &TextureView,
                            input: &TextureView,
                            sample_input: &TextureView,
                            metadata_output: &TextureView,
                            metadata_input: &TextureView,
                            receiver_metadata: &TextureView,
                            split_receiver_history: bool| {
            device.create_bind_group(
                "split_rc_compute",
                &cache.get_bind_group_layout(&pipelines.compute_layout),
                &BindGroupEntries::sequential((
                    &uniforms[level],
                    triangles.clone(),
                    nodes.clone(),
                    lights.clone(),
                    depth,
                    normal,
                    current.heads.as_entire_binding(),
                    buffers.entries.as_entire_binding(),
                    previous.heads.as_entire_binding(),
                    buffers.history_entries.as_entire_binding(),
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
                    prepass.motion_vectors_view().unwrap_or(normal),
                    receiver_metadata,
                    environment_view,
                    if split_receiver_history {
                        &buffers.metadata_views[3 + (frame & 1) as usize]
                    } else {
                        &buffers.unused_output_views[0]
                    },
                    &buffers.unused_output_views[0],
                    if split_receiver_history {
                        &buffers.metadata_views[3 + ((frame + 1) & 1) as usize]
                    } else {
                        &buffers.metadata_views[3]
                    },
                    &receiver.view,
                    gbuffer,
                    geometry_page_view,
                    scene_page_view,
                ))
                .iter()
                .filter(|entry| !matches!(entry.binding, 17 | 18 | 19 | 24))
                .cloned()
                .collect::<Vec<_>>(),
            )
        };
        let surface_group_layout = cache.get_bind_group_layout(&pipelines.surface_group_layout);
        let surface_groups = [
            device.create_bind_group(
                "split_rc_surface_groups",
                &surface_group_layout,
                &BindGroupEntries::sequential((
                    &buffers.surface_group_views[2],
                    &buffers.surface_group_views[((frame + 1) & 1) as usize],
                    &buffers.surface_group_views[(frame & 1) as usize],
                )),
            ),
            device.create_bind_group(
                "split_rc_surface_group_seed",
                &surface_group_layout,
                &BindGroupEntries::sequential((
                    &buffers.surface_group_views[((frame + 1) & 1) as usize],
                    &buffers.surface_group_views[(frame & 1) as usize],
                    &buffers.surface_group_views[(frame & 1) as usize],
                )),
            ),
        ];
        let compute = std::array::from_fn(|level| {
            bind_compute(
                level,
                irradiance,
                previous_irradiance,
                previous_irradiance,
                metadata,
                previous_metadata,
                &buffers.metadata_views[2],
                false,
            )
        });
        let temporal = bind_compute(
            0,
            &buffers.irradiance_views[2],
            previous_irradiance,
            irradiance,
            metadata,
            previous_metadata,
            &buffers.metadata_views[2],
            true,
        );
        let denoise = std::array::from_fn(|level| {
            let output = &buffers.irradiance_views[2 + level % 2];
            let input = if level == 0 {
                irradiance
            } else {
                &buffers.irradiance_views[2 + (level - 1) % 2]
            };
            bind_compute(
                level,
                output,
                input,
                metadata,
                previous_metadata,
                metadata,
                &buffers.metadata_views[2],
                false,
            )
        });
        // Raw ping-pong irradiance is temporal history. Filtering only the
        // display output avoids recursively diffusing lighting across surfaces.
        let display = if settings.spatial_filter_steps == 0 {
            irradiance
        } else {
            &buffers.irradiance_views[2 + (settings.spatial_filter_steps as usize - 1) % 2]
        };
        let composite =
            [target.main_texture_view(), target.main_texture_other_view()].map(|source| {
                (
                    source.id(),
                    device.create_bind_group(
                        "split_rc_composite",
                        &cache.get_bind_group_layout(&pipelines.composite_layout),
                        &BindGroupEntries::sequential((
                            &uniforms[0],
                            source,
                            display,
                            depth,
                            normal,
                            gbuffer,
                            display,
                            &receiver.view,
                            ssao.map_or(fallback_view, |ssao| {
                                &ssao.screen_space_ambient_occlusion_texture.default_view
                            }),
                            &buffers.surface_group_views[((frame + 1) & 1) as usize],
                        ))
                        .iter()
                        .filter(|entry| entry.binding != 6)
                        .cloned()
                        .collect::<Vec<_>>(),
                    ),
                )
            });
        let required = required_pipelines(settings);
        let pipelines_ready = (0..COMPUTE_ENTRIES.len())
            .all(|i| required & (1 << i) == 0 || pipelines.pipeline(i, &cache).is_some())
            && cache.get_render_pipeline(pipelines.composite).is_some();
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
        });
        commands.entity(entity).insert(ViewBindings {
            compute,
            surface_groups,
            denoise,
            temporal,
            composite,
            frame: RenderedFrame {
                clip_from_world,
                world_from_clip: params.world_from_clip,
                exposure: camera.exposure,
            },
        });
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

#[allow(clippy::type_complexity)]
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
    pipelines: Res<Pipelines>,
    cache: Res<PipelineCache>,
    images: Res<RenderAssets<GpuImage>>,
    mut context: RenderContext,
    mut pending_pipelines: Local<u64>,
) {
    let diagnostic_view = view.entity();
    let (target, settings, buffers, bindings, receiver, raw_output, lighting) = view.into_inner();
    if settings.intensity <= 0.0 {
        return;
    }
    // The current identity pass and committed scene decide lighting ownership
    // before native lighting draws, so fallback and GI never overlap.
    if !lighting.is_some_and(|lighting| lighting.0) {
        buffers.history_valid.store(false, Ordering::Relaxed);
        buffers.screen_history_valid.store(false, Ordering::Relaxed);
        return;
    }
    // Do not dispatch retained bindings after staging or a residency failure.
    if !scene.ready
        || gpu_scene
            .as_ref()
            .is_none_or(|gpu| !gpu.ready_for_revision(scene.revision))
    {
        buffers.history_valid.store(false, Ordering::Relaxed);
        buffers.screen_history_valid.store(false, Ordering::Relaxed);
        return;
    }
    if !receiver.ready.load(Ordering::Relaxed) {
        buffers.screen_history_valid.store(false, Ordering::Relaxed);
        return;
    }
    let compute: [Option<&ComputePipeline>; COMPUTE_ENTRIES.len()] =
        std::array::from_fn(|i| pipelines.pipeline(i, &cache));
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
        return;
    }
    *pending_pipelines = 0;
    let Some(composite) = cache.get_render_pipeline(pipelines.composite) else {
        buffers.screen_history_valid.store(false, Ordering::Relaxed);
        return;
    };
    if buffers.frame.load(Ordering::Relaxed) == 0 {
        info!(
            ?diagnostic_view,
            rays = buffers.size.x * buffers.size.y,
            cascades = LEVELS,
            bounce_feedback = buffers.settings.bounce_feedback,
            "Split Radiance Cascades dispatching",
        );
    }
    let diagnostics = context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let total_span = diagnostics.time_span(context.command_encoder(), "split_rc");
    let mut dispatch =
        |name: &'static str, pipeline: &ComputePipeline, level: usize, x: u32, y: u32| {
            let span = diagnostics.time_span(context.command_encoder(), name);
            let mut pass = context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("split_rc"),
                    ..default()
                });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bindings.compute[level], &[]);
            pass.set_bind_group(
                1,
                if name == "seed" {
                    &bindings.surface_groups[1]
                } else {
                    &bindings.surface_groups[0]
                },
                &[],
            );
            pass.dispatch_workgroups(x, y, 1);
            drop(pass);
            if let Some(diagnostics) = diagnostics.filter(|_| name == "prefilter") {
                let heads =
                    &buffers.caches[(buffers.frame.load(Ordering::Relaxed) & 1) as usize].heads;
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
                let heads =
                    &buffers.caches[(buffers.frame.load(Ordering::Relaxed) & 1) as usize].heads;
                let dropped = (u64::from(buffers.buckets) * 2 + 4) * 4;
                diagnostics.record_u32(
                    context.command_encoder(),
                    &heads.slice(dropped..dropped + 4),
                    format!("penumbra/view_{diagnostic_view}/persistent_cache_dropped"),
                );
            }
            span.end(context.command_encoder());
        };
    let image_x = buffers.size.x.div_ceil(8);
    let image_y = buffers.size.y.div_ceil(8);
    let bucket_groups = buffers.buckets.div_ceil(64);
    let ray_groups = (buffers.size.x * buffers.size.y).div_ceil(64);
    dispatch(
        "clear",
        compute[0].expect("cascade pipeline ready"),
        0,
        bucket_groups,
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
        bucket_groups,
        1,
    );
    for level in 1..LEVELS {
        dispatch(
            "parents",
            compute[2].expect("cascade pipeline ready"),
            level,
            ray_groups,
            1,
        );
        dispatch(
            "rank",
            compute[3].expect("cascade pipeline ready"),
            level,
            bucket_groups,
            1,
        );
    }
    for level in (0..LEVELS).rev() {
        dispatch(
            "prefix",
            compute[4].expect("cascade pipeline ready"),
            level,
            bucket_groups,
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
        bucket_groups,
        1,
    );
    dispatch(
        "trace",
        compute[7].expect("cascade pipeline ready"),
        0,
        image_x,
        image_y,
    );
    if buffers.settings.bounce_feedback > 0.0 {
        dispatch(
            "traceSecondary",
            compute[15].expect("secondary pipeline ready"),
            0,
            image_x,
            image_y,
        );
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
    let entry_count = cache_work_slots(
        u64::from(buffers.size.x) * u64::from(buffers.size.y),
        u64::from(buffers.ray_layers),
    ) + 2 * u64::from(buffers.buckets);
    for level in (0..LEVELS).rev() {
        dispatch(
            "merge",
            compute[9].expect("cascade pipeline ready"),
            level,
            entry_count.div_ceil(64 * LEVELS as u64) as u32,
            LEVELS as u32,
        );
    }
    dispatch(
        "seedIrradiance",
        compute[10].expect("cascade pipeline ready"),
        0,
        bucket_groups,
        1,
    );
    dispatch(
        "prefilter",
        compute[11].expect("cascade pipeline ready"),
        0,
        bucket_groups,
        1,
    );
    dispatch(
        "gather",
        compute[12].expect("cascade pipeline ready"),
        0,
        image_x,
        image_y,
    );
    dispatch(
        "commitCache",
        compute[CACHE_COMMIT].expect("cascade pipeline ready"),
        0,
        bucket_groups,
        1,
    );
    dispatch(
        "remapProbeHeads",
        compute[PROBE_HEAD_REMAP].expect("cascade pipeline ready"),
        0,
        bucket_groups,
        1,
    );
    let frame = buffers.frame.load(Ordering::Relaxed);
    // Copy before temporal clamping, filtering, albedo, AO, intensity and exposure.
    if let Some(output) = raw_output.and_then(|output| images.get(&output.0)) {
        let descriptor = &output.texture_descriptor;
        if descriptor.format == TextureFormat::Rgba32Float
            && output.size_2d() == buffers.size
            && descriptor.size.depth_or_array_layers == 1
            && descriptor.sample_count == 1
            && descriptor.usage.contains(TextureUsages::COPY_DST)
        {
            context.command_encoder().copy_texture_to_texture(
                buffers.irradiance_textures[(frame & 1) as usize].as_image_copy(),
                output.texture.as_image_copy(),
                Extent3d {
                    width: buffers.size.x,
                    height: buffers.size.y,
                    depth_or_array_layers: 1,
                },
            );
        } else {
            warn_once!(
                "RadianceCascadesRawOutput requires a COPY_DST Rgba32Float image matching the GI dimensions"
            );
        }
    }
    {
        let temporal_span = diagnostics.time_span(context.command_encoder(), "temporal");
        let mut pass = context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("split_rc_temporal"),
                ..default()
            });
        pass.set_pipeline(compute[13].expect("temporal pipeline ready"));
        pass.set_bind_group(0, &bindings.temporal, &[]);
        pass.set_bind_group(1, &bindings.surface_groups[0], &[]);
        pass.dispatch_workgroups(buffers.size.x.div_ceil(8), buffers.size.y.div_ceil(8), 1);
        drop(pass);

        // Only temporal results become history; spatially filtered display values
        // never feed back and spread lighting recursively across contacts.
        context.command_encoder().copy_texture_to_texture(
            buffers.irradiance_textures[2].as_image_copy(),
            buffers.irradiance_textures[(frame & 1) as usize].as_image_copy(),
            Extent3d {
                width: buffers.size.x,
                height: buffers.size.y,
                depth_or_array_layers: 1,
            },
        );
        temporal_span.end(context.command_encoder());
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
        pass.set_pipeline(compute[14].expect("denoise pipeline ready"));
        pass.set_bind_group(0, &bindings.denoise[level], &[]);
        pass.set_bind_group(1, &bindings.surface_groups[0], &[]);
        pass.dispatch_workgroups(buffers.size.x.div_ceil(8), buffers.size.y.div_ceil(8), 1);
        drop(pass);

        span.end(context.command_encoder());
    }

    let post = target.post_process_write();
    let group = if bindings.composite[0].0 == post.source.id() {
        &bindings.composite[0].1
    } else {
        &bindings.composite[1].1
    };
    let composition_span = diagnostics.time_span(context.command_encoder(), "composition");
    let mut pass = context
        .command_encoder()
        .begin_render_pass(&RenderPassDescriptor {
            label: Some("split_rc_composite"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: post.destination,
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
    pass.set_bind_group(0, group, &[]);
    pass.draw(0..3, 0..1);
    drop(pass);
    composition_span.end(context.command_encoder());
    total_span.end(context.command_encoder());
    *buffers
        .rendered_frame
        .lock()
        .unwrap_or_else(|error| error.into_inner()) = bindings.frame;
    if buffers.frame.fetch_add(1, Ordering::Relaxed) == 0 {
        info!("First Split Radiance Cascades frame submitted");
    }
    buffers.history_valid.store(true, Ordering::Relaxed);
    buffers.screen_history_valid.store(true, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn src_pipeline_requirements_track_secondary_cache_and_filter() {
        let settings = RadianceCascades {
            bounce_feedback: 0.0,
            spatial_filter_steps: 0,
            ..default()
        };
        let required = required_pipelines(settings);
        assert_eq!(required & (1 << 15), 0);
        assert_eq!(required & (1 << 14), 0);
        for index in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17] {
            assert_ne!(required & (1 << index), 0);
        }
        let full = required_pipelines(RadianceCascades {
            bounce_feedback: 0.5,
            spatial_filter_steps: 1,
            ..settings
        });
        assert_eq!(full, (1 << COMPUTE_ENTRIES.len()) - 1);
    }

    #[test]
    fn cascade_allocations_fit_storage_limits_after_resolution_reduction() {
        let limits = WgpuLimits {
            max_storage_buffer_binding_size: 2 * 1024 * 1024,
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
            settings,
            &limits
        ));
        settings.max_rays = view_device_max_rays(viewport, settings, &limits);
        assert!(view_fits_device(
            resolution(viewport, settings),
            settings,
            &limits
        ));
        // Several ray budgets produce the same integer screen grid. Admission
        // limits the allocated receivers, not the upper end of that plateau.
        let size = resolution(viewport, settings);
        assert!(size.x * size.y <= 512);
        let insufficient = WgpuLimits {
            max_storage_buffer_binding_size: 1024,
            ..limits
        };
        assert!(!view_fits_device(UVec2::ONE, settings, &insufficient));
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
    fn split_cache_clear_keeps_irradiance_heads_separate_from_counters() {
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
        let bytes = cache_head_bytes(u64::from(BUCKETS));
        let heads = device.create_buffer(&BufferDescriptor {
            label: Some("split_clear_heads"),
            size: bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&heads, 0, &vec![0xabu8; bytes as usize]);
        let mut params = UniformBuffer::from(Parameters {
            size: UVec4::new(0, 0, 0, BUCKETS),
            ..default()
        });
        params.write_buffer(&device, &queue);
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
            ],
        );
        let output = crate::scene::traversal_tests::dispatch_readback(
            &device,
            &queue,
            &pipeline,
            &group,
            &heads,
            bytes,
            BUCKETS.div_ceil(64),
        );
        let words: Vec<_> = output
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(words.len(), 2 * BUCKETS as usize + 5);
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
        assert_eq!(words[2 * BUCKETS as usize + 1..], [0; 4]);
    }

    #[test]
    #[ignore = "requires slangc and a desktop GPU; no demo scene is launched"]
    fn split_prefilter_matches_exact_directional_gather_and_oct_seams() {
        use bevy::render::{
            renderer::initialize_renderer,
            settings::{Backends, WgpuSettings},
        };

        fn decode_oct(uv: Vec2) -> Vec3 {
            let p = uv * 2.0 - Vec2::ONE;
            let mut n = Vec3::new(p.x, p.y, 1.0 - p.x.abs() - p.y.abs());
            let t = (-n.z).clamp(0.0, 1.0);
            n.x += if n.x >= 0.0 { -t } else { t };
            n.y += if n.y >= 0.0 { -t } else { t };
            n.normalize()
        }

        fn bin_direction(index: usize) -> Vec3 {
            let uv = Vec2::new(
                ((index % 8) as f32 + 0.5) / 8.0,
                ((index / 8) as f32 + 0.5) / 4.0,
            );
            let phi = uv.x * std::f32::consts::TAU;
            let y = uv.y * 2.0 - 1.0;
            let radius = (1.0 - y * y).max(0.0).sqrt();
            Vec3::new(radius * phi.cos(), y, radius * phi.sin())
        }

        fn surface_face(normal: Vec3) -> usize {
            let axis = if normal.x.abs() >= normal.y.abs() && normal.x.abs() >= normal.z.abs() {
                0
            } else if normal.y.abs() >= normal.z.abs() {
                1
            } else {
                2
            };
            axis * 2 + usize::from(normal[axis] < 0.0)
        }

        fn hash_key(key: [u32; 4], buckets: usize) -> usize {
            let mut hash = key[0].wrapping_mul(0x8da6_b343)
                ^ key[1].wrapping_mul(0xd816_3841)
                ^ key[2].wrapping_mul(0xcb1a_b31f)
                ^ key[3].wrapping_mul(0x1656_67b1);
            hash ^= hash >> 16;
            hash = hash.wrapping_mul(0x7feb_352d);
            hash ^= hash >> 15;
            hash as usize & (buckets - 1)
        }

        fn words_buffer(
            device: &RenderDevice,
            label: &str,
            words: &[u32],
            usage: BufferUsages,
        ) -> Buffer {
            let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
            device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some(label),
                contents: &bytes,
                usage,
            })
        }

        const BUCKETS: usize = 128;
        const DIRECTIONS: usize = 32;
        const ENTRY_WORDS: usize = 24;
        const PROBE_WORDS: usize = 148;
        const IDENTITY: [u32; 2] = [17, 29];
        let options = WgpuSettings::default();
        let resources = bevy::tasks::block_on(initialize_renderer(
            options.backends.unwrap_or(Backends::PRIMARY),
            None,
            &options,
        ));
        let device = &resources.0;
        let queue = &resources.1;

        let mut face_texels = [None; 6];
        for y in 0..6 {
            for x in 0..6 {
                let normal = decode_oct(Vec2::new((x as f32 + 0.5) / 6.0, (y as f32 + 0.5) / 6.0));
                let face = surface_face(normal);
                face_texels[face].get_or_insert((x, y, normal));
            }
        }
        let face_texels = face_texels.map(|texel| texel.expect("all six octahedral faces sampled"));
        let case_count = face_texels.len() * 4;
        let mut entries = vec![[0u32; ENTRY_WORDS]; case_count * (DIRECTIONS + 1)];
        let mut heads = vec![u32::MAX; BUCKETS * 2 + 5];
        let mut output_slots = Vec::with_capacity(case_count);
        let mut expected: Vec<[Vec3; 36]> = Vec::with_capacity(case_count);

        for (case, (face, lod)) in (0..6)
            .flat_map(|face| (0..4).map(move |lod| (face, lod)))
            .enumerate()
        {
            let mode = (face as u32) << 15 | (lod as u32) << 18;
            let probe_index = case as u32;
            let mut probe_slot = hash_key([0, 0, 0, mode], BUCKETS);
            while heads[BUCKETS + 1 + probe_slot] != u32::MAX {
                probe_slot = (probe_slot + 1) & (BUCKETS - 1);
            }
            heads[BUCKETS + 1 + probe_slot] = probe_index;
            output_slots.push(probe_slot);

            entries[case][0..4].copy_from_slice(&[0, 0, 0, mode]);
            entries[case][20..22].copy_from_slice(&IDENTITY);
            let cell_size = (1u32 << lod) as f32;
            let origin = Vec3::splat(cell_size * 0.5);
            let mut values = [Vec3::ZERO; DIRECTIONS];
            for direction in 0..DIRECTIONS {
                let index = case_count + case * DIRECTIONS + direction;
                let record = &mut entries[index];
                record[0..4].copy_from_slice(&[0, 0, 0, mode | direction as u32]);
                record[8..12].copy_from_slice(&[
                    (0.1 + ((direction * 7 + face * 3 + lod) % 17) as f32 * 0.031).to_bits(),
                    (0.2 + ((direction * 11 + face + lod * 5) % 13) as f32 * 0.027).to_bits(),
                    (0.05 + ((direction * 3 + face * 5 + lod * 2) % 19) as f32 * 0.019).to_bits(),
                    1.0f32.to_bits(),
                ]);
                record[12..16].copy_from_slice(&[
                    origin.x.to_bits(),
                    origin.y.to_bits(),
                    origin.z.to_bits(),
                    1.0f32.to_bits(),
                ]);
                record[20..22].copy_from_slice(&IDENTITY);
                let bucket = hash_key([0, 0, 0, mode | direction as u32], BUCKETS);
                record[16] = heads[bucket];
                heads[bucket] = index as u32;
                values[direction] = Vec3::new(
                    f32::from_bits(record[8]),
                    f32::from_bits(record[9]),
                    f32::from_bits(record[10]),
                );
            }
            expected.push(std::array::from_fn(|texel| {
                let normal = decode_oct(Vec2::new(
                    ((texel % 6) as f32 + 0.5) / 6.0,
                    ((texel / 6) as f32 + 0.5) / 6.0,
                ));
                let mut sum = Vec3::ZERO;
                let mut total = 0.0;
                for (radiance, direction) in values.iter().zip((0..DIRECTIONS).map(bin_direction)) {
                    let cosine = normal.dot(direction).max(0.0);
                    sum += *radiance * cosine;
                    total += cosine;
                }
                sum / total
            }));
        }

        let entries_words: Vec<_> = entries.into_iter().flatten().collect();
        let entry_buffer = words_buffer(
            device,
            "split_prefilter_entries",
            &entries_words,
            BufferUsages::STORAGE,
        );
        let head_buffer = words_buffer(
            device,
            "split_prefilter_heads",
            &heads,
            BufferUsages::STORAGE,
        );
        let previous_heads = words_buffer(
            device,
            "split_prefilter_previous_heads",
            &vec![u32::MAX; BUCKETS * 2 + 5],
            BufferUsages::STORAGE,
        );
        let previous_entries = words_buffer(
            device,
            "split_prefilter_previous_entries",
            &vec![0; entries_words.len()],
            BufferUsages::STORAGE,
        );
        let probe_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("split_prefilter_output"),
            size: (2 * BUCKETS * PROBE_WORDS * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut params = UniformBuffer::from(Parameters {
            size: UVec4::new(1, 1, 1, BUCKETS as u32),
            tuning: Vec4::new(1.0, 1.0, 0.001, 1.0),
            frame: UVec4::ZERO,
            ..default()
        });
        params.write_buffer(device, queue);

        let shader = compile(&SlangCompiler::default(), "cascades.slang", &["prefilter"])
            .expect("compile production SplitDiffuse prefilter");
        let bevy::shader::Source::Wgsl(source) = shader.source else {
            panic!("expected Slang's WGSL output");
        };
        let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
            label: Some("split_prefilter_acceptance"),
            source: ShaderSource::Wgsl(source),
        });
        let pipeline = device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("split_prefilter_acceptance"),
            layout: None,
            module: &module,
            entry_point: Some("prefilter"),
            compilation_options: default(),
            cache: None,
        });
        let group = device.create_bind_group(
            "split_prefilter_acceptance",
            &pipeline.get_bind_group_layout(0).into(),
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: params.binding().unwrap(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: head_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: entry_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: previous_heads.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: previous_entries.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 12,
                    resource: probe_buffer.as_entire_binding(),
                },
            ],
        );
        let probe_bytes = (2 * BUCKETS * PROBE_WORDS * 4) as u64;
        let bytes = crate::scene::traversal_tests::dispatch_readback(
            device,
            queue,
            &pipeline,
            &group,
            &probe_buffer,
            probe_bytes,
            (BUCKETS as u32).div_ceil(64),
        );
        let output_word = |slot: usize, texel: usize, channel: usize| {
            let word = slot * PROBE_WORDS + texel * 4 + channel;
            f32::from_bits(u32::from_le_bytes(
                bytes[word * 4..word * 4 + 4].try_into().unwrap(),
            ))
        };
        let mut maximum_error = 0.0f32;
        for (case, (slot, targets)) in output_slots.iter().zip(&expected).enumerate() {
            for (texel, target) in targets.iter().enumerate() {
                assert_eq!(
                    output_word(*slot, texel, 3),
                    1.0,
                    "case {case}, texel {texel} has no support"
                );
                let actual = Vec3::new(
                    output_word(*slot, texel, 0),
                    output_word(*slot, texel, 1),
                    output_word(*slot, texel, 2),
                );
                let error = (actual - *target).abs().max_element();
                maximum_error = maximum_error.max(error);
                assert!(
                    error < 2e-5,
                    "prefilter case {case}, texel {texel}: GPU {actual:?}, exact gather {target:?}"
                );
            }
        }
        println!(
            "SplitDiffuse prefilter: {case_count} face/LOD cases, {} texels, max RGB error {maximum_error:.8}",
            case_count * 36
        );

        let check_shader = SlangCompiler::default()
            .compile_bundle(
                "split_prefilter_check.slang",
                &[
                    ("common.slang", include_str!("../shaders/common.slang")),
                    (
                        "split_prefilter_check.slang",
                        include_str!("../shaders/split_prefilter_check.slang"),
                    ),
                ],
                &SlangSettings {
                    target: SlangTarget::SpirV,
                    entry_points: vec!["checkIrradianceOctPad".into()],
                    optimization: Some(2),
                    ..default()
                },
            )
            .expect("compile production octahedral pad helpers");
        let bevy::shader::Source::Wgsl(source) = check_shader.source else {
            panic!("expected Slang's WGSL output");
        };
        let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
            label: Some("split_prefilter_oct_seams"),
            source: ShaderSource::Wgsl(source),
        });
        let pipeline = device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("split_prefilter_oct_seams"),
            layout: None,
            module: &module,
            entry_point: Some("checkIrradianceOctPad"),
            compilation_options: default(),
            cache: None,
        });
        let coordinates: Vec<_> = (0..8)
            .flat_map(|y| (0..8).map(move |x| [x, y, 0, 0]))
            .flatten()
            .collect();
        let coordinate_buffer = words_buffer(
            device,
            "split_prefilter_oct_coordinates",
            &coordinates,
            BufferUsages::STORAGE,
        );
        let seam_output = device.create_buffer(&BufferDescriptor {
            label: Some("split_prefilter_oct_output"),
            size: 64 * 32,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let seam_group = device.create_bind_group(
            "split_prefilter_oct_seams",
            &pipeline.get_bind_group_layout(0).into(),
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: coordinate_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: seam_output.as_entire_binding(),
                },
            ],
        );
        let seam_bytes = crate::scene::traversal_tests::dispatch_readback(
            device,
            queue,
            &pipeline,
            &seam_group,
            &seam_output,
            64 * 32,
            1,
        );
        for (index, (x, y)) in (0..8).flat_map(|y| (0..8).map(move |x| (x, y))).enumerate() {
            let mut px = x - 1;
            let mut py = y - 1;
            if px < 0 {
                px = 0;
                py = 5 - py;
            } else if px > 5 {
                px = 5;
                py = 5 - py;
            }
            if py < 0 {
                py = 0;
                px = 5 - px;
            } else if py > 5 {
                py = 5;
                px = 5 - px;
            }
            let actual: [u32; 4] = std::array::from_fn(|channel| {
                u32::from_le_bytes(
                    seam_bytes[index * 32 + channel * 4..index * 32 + channel * 4 + 4]
                        .try_into()
                        .unwrap(),
                )
            });
            let actual_uv_y = u32::from_le_bytes(
                seam_bytes[index * 32 + 16..index * 32 + 20]
                    .try_into()
                    .unwrap(),
            );
            assert_eq!(
                actual[..3],
                [px as u32, py as u32, (px + py * 6) as u32],
                "oct pad pixel ({x},{y})"
            );
            // GPU division may use a reciprocal followed by multiplication.
            // Compare UVs numerically; texel addresses above remain exact.
            assert!(
                (f32::from_bits(actual[3]) - (px as f32 + 0.5) / 6.0).abs() < 1e-7,
                "oct pad UV x ({x},{y})"
            );
            assert!(
                (f32::from_bits(actual_uv_y) - (py as f32 + 0.5) / 6.0).abs() < 1e-7,
                "oct pad UV y ({x},{y})"
            );
        }
        println!("SplitDiffuse octahedral pad fold: all 64 core/border tap coordinates matched");
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
        let lookup = make_pipeline("checkProbeLookup");
        let work_bytes = cache_entry_bytes(1, 1, u64::from(BUCKETS));
        let history_bytes = cache_history_bytes(u64::from(BUCKETS));
        let heads_bytes = cache_head_bytes(u64::from(BUCKETS));
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
        let probe_heads = device.create_buffer(&BufferDescriptor {
            label: Some("split_commit_probe_heads"),
            size: heads_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut head_words = vec![u32::MAX; (heads_bytes / 4) as usize];
        head_words[BUCKETS as usize] = 0;
        head_words[2 * BUCKETS as usize + 1..].fill(0);
        head_words[0] = 0;
        head_words[BUCKETS as usize + 1] = 2;
        head_words[BUCKETS as usize + 2] = 0;
        let head_data: Vec<_> = head_words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        queue.write_buffer(&heads, 0, &head_data);
        let mut entries = vec![0u32; (work_bytes / 4) as usize];
        for index in 0..3usize {
            let base = index * 24;
            entries[base] = 10 + index as u32;
            entries[base + 16] = if index == 2 {
                u32::MAX
            } else {
                index as u32 + 1
            };
            entries[base + 17] = FRAME | 0x8000_0000;
            entries[base + 18] = FRAME;
        }
        let entry_data: Vec<_> = entries.iter().flat_map(|word| word.to_le_bytes()).collect();
        queue.write_buffer(&work, 0, &entry_data);
        let mut uniform = UniformBuffer::from(Parameters {
            size: UVec4::new(1, 1, 1, BUCKETS),
            history: Vec4::new(0.9, 0.0, 1.0, 4.0),
            frame: UVec4::new(FRAME, 0, 0, 0),
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
        let remap_group = make_group(&remap, false);
        let lookup_group = device.create_bind_group(
            "split_cache_probe_lookup_check",
            &lookup.get_bind_group_layout(0).into(),
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
                    binding: 8,
                    resource: probe_heads.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: history.as_entire_binding(),
                },
            ],
        );
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
        encoder.copy_buffer_to_buffer(&heads, 0, &probe_heads, 0, heads_bytes);
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            pass.set_pipeline(&lookup);
            pass.set_bind_group(0, &lookup_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&heads, 0, &readback, 0, heads_bytes);
        encoder.copy_buffer_to_buffer(&history, 0, &readback, heads_bytes, history_bytes);
        encoder.copy_buffer_to_buffer(&work, 0, &readback, heads_bytes + history_bytes, work_bytes);
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
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect();
        let heads_out = &words[..(heads_bytes / 4) as usize];
        let history_out =
            &words[(heads_bytes / 4) as usize..((heads_bytes + history_bytes) / 4) as usize];
        let work_out = &words[((heads_bytes + history_bytes) / 4) as usize..];
        assert_eq!(heads_out[0], 1);
        assert_eq!(heads_out[BUCKETS as usize + 1], u32::MAX - 1);
        assert_eq!(heads_out[BUCKETS as usize + 2], 0);
        assert_eq!(work_out[23], 1);
        assert_eq!(heads_out[(2 * BUCKETS + 4) as usize], 1);
        assert_eq!(history_out[0], 10);
        assert_eq!(history_out[24], 11);
        assert_eq!(history_out[16], u32::MAX);
        assert_eq!(history_out[24 + 16], 0);
        assert_eq!(work_out[22], 0);
        assert_eq!(work_out[24 + 22], 1);
        assert_eq!(work_out[48 + 22], u32::MAX);
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
    fn screen_history_invalidation_tracks_geometry_motion_and_missed_edits() {
        assert!(!invalidate_screen_history_for_scene_change(
            10, 11, false, false
        ));
        assert!(invalidate_screen_history_for_scene_change(
            10, 11, true, false
        ));
        assert!(invalidate_screen_history_for_scene_change(
            10, 12, false, false
        ));
        assert!(!invalidate_screen_history_for_scene_change(
            10, 11, true, true
        ));
        assert!(!invalidate_screen_history_for_scene_change(
            u64::MAX,
            0,
            false,
            false
        ));
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
    fn split_view_budget_covers_shader_entry_ranges() {
        let samples = 128u64;
        let settings = RadianceCascades {
            bounce_feedback: 0.0,
            ..default()
        };
        let buckets = 1024u64;
        let entries = cache_entry_bytes(samples, 1, buckets);
        assert_eq!(
            entries,
            (cache_work_slots(samples, 1) + 2 * buckets) * ENTRY_SIZE
        );
        assert_eq!(
            cache_entry_bytes(samples, 2, buckets),
            (cache_work_slots(samples, 2) + 2 * buckets) * ENTRY_SIZE
        );
        assert_eq!(
            view_bytes(UVec2::new(samples as u32, 1), settings),
            samples * 9 * 16
                + samples * 12
                + 16
                + LEVELS as u64 * std::mem::size_of::<Parameters>() as u64
                + buckets * PROBE_SIZE * 2
                + cache_head_bytes(buckets) * 2
                + entries
                + cache_history_bytes(buckets)
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
        let mut scene = app.world_mut().resource_mut::<GiScene>();
        scene.ready = true;
        scene.pending_entities = Arc::new(HashSet::from([committed, pending]));
        scene.data = Arc::new(crate::scene::SceneData {
            textures: textures.clone(),
            instance_residency: keys.clone(),
            ..default()
        });
        drop(scene);
        let mut gpu_scene = app.world_mut().resource_mut::<GpuScene>();
        gpu_scene.textures = textures;
        gpu_scene.instance_residency = keys.clone();
        drop(gpu_scene);

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
        let mut meshes = RenderVisibleEntitiesClass::default();
        meshes.entities_cpu_culling = vec![
            (Entity::PLACEHOLDER, new_cpu),
            (Entity::PLACEHOLDER, old_cpu),
        ];
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
    fn resolution_respects_budget_and_zero_viewports() {
        let settings = RadianceCascades {
            max_rays: 8192,
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
            assert!(size.x > 0 && size.y > 0 && size.x * size.y <= settings.max_rays);
        }
    }

    #[test]
    #[ignore = "requires slangc (or SLANGC) on PATH"]
    fn compile_all_embedded_slang_stages() {
        for (file, entries) in shader_modules() {
            let shader = compile(&SlangCompiler::default(), file, entries).expect(file);
            let bevy::shader::Source::Wgsl(source) = shader.source else {
                panic!("expected Slang's WGSL output")
            };
            let module = naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|error| panic!("{file}: {}", error.emit_to_string(&source)));
            assert_eq!(module.entry_points.len(), entries.len(), "{file} entries");
            for entry in entries {
                assert!(module.entry_points.iter().any(|point| point.name == *entry));
            }
            let info = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::empty(),
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
            if file == "cascades.slang" && entries.contains(&"prefilter") {
                let (_, probe) = module
                    .types
                    .iter()
                    .find(|(_, ty)| ty.name.as_deref().is_some_and(|name| name.starts_with("IrradianceProbe")))
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
                let layout = compute_layout();
                let surface_layout = surface_group_layout();
                for (_, global) in module.global_variables.iter() {
                    let Some(binding) = &global.binding else {
                        continue;
                    };
                    let group_layout = match binding.group {
                        0 => &layout,
                        1 => &surface_layout,
                        group => panic!("unexpected production bind group {group}"),
                    };
                    let expected = group_layout
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

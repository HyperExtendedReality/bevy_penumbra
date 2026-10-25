//! Opt-in queue timestamps for acceptance benchmarks, with bounded async readback.
use bevy::{
    diagnostic::{Diagnostic, DiagnosticMeasurement, DiagnosticPath, DiagnosticsStore},
    platform::time::Instant,
    prelude::*,
    render::{
        GpuResourceAppExt, Render, RenderApp, RenderSystems,
        renderer::{RenderDevice, RenderQueue},
    },
};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

/// Measures the GPU queue envelope from before asset preparation through render
/// submission, including uploads and GI deformation/refit. Excludes extraction,
/// CPU presentation wait and work submitted outside the render schedule.
/// This adds two submissions and async readback; enable only for profiling.
pub struct FrameTimingPlugin;

#[derive(Resource, Clone, Default)]
struct Completed(Arc<Mutex<Vec<(Instant, f64)>>>);

#[derive(Resource, Clone, Default)]
struct RenderCpuCompleted(Arc<Mutex<Vec<(Instant, f64)>>>);

#[derive(Resource, Default)]
struct CpuTiming {
    start: Option<Instant>,
    completed: Vec<(Instant, f64)>,
}

#[derive(Resource, Default)]
struct RenderCpuStart(Option<Instant>);

struct Slot {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    busy: Arc<AtomicBool>,
}

#[derive(Resource)]
struct Timing {
    slots: Vec<Slot>,
    active: Option<usize>,
}

impl FromWorld for Timing {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>().wgpu_device();
        let supported = device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
        let slots = (0..if supported { 8 } else { 0 })
            .map(|_| Slot {
                queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("penumbra_frame_timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: 2,
                }),
                resolve: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("penumbra_frame_resolve"),
                    size: 16,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                read: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("penumbra_frame_readback"),
                    size: 16,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                busy: Arc::new(AtomicBool::new(false)),
            })
            .collect();
        Self {
            slots,
            active: None,
        }
    }
}

impl Plugin for FrameTimingPlugin {
    fn build(&self, app: &mut App) {
        let completed = Completed::default();
        let render_cpu_completed = RenderCpuCompleted::default();
        app.insert_resource(completed.clone())
            .insert_resource(render_cpu_completed.clone())
            .init_resource::<CpuTiming>()
            .add_systems(First, begin_cpu)
            .add_systems(Last, end_cpu)
            .add_systems(PreUpdate, (publish, publish_cpu, publish_render_cpu));
        if let Some(render) = app.get_sub_app_mut(RenderApp) {
            render
                .insert_resource(completed)
                .insert_resource(render_cpu_completed)
                .init_gpu_resource::<Timing>()
                .init_resource::<RenderCpuStart>()
                .add_systems(
                    Render,
                    (begin, begin_render_cpu)
                        .after(RenderSystems::ExtractCommands)
                        .before(RenderSystems::PrepareAssets)
                        .before(RenderSystems::PrepareMeshes)
                        .before(RenderSystems::Prepare),
                )
                .add_systems(
                    Render,
                    (end, end_render_cpu)
                        .after(RenderSystems::Render)
                        .before(RenderSystems::Cleanup),
                );
        }
    }
}

fn begin_cpu(mut timing: ResMut<CpuTiming>) {
    timing.start = Some(Instant::now());
}

fn end_cpu(mut timing: ResMut<CpuTiming>) {
    if let Some(start) = timing.start.take() {
        timing
            .completed
            .push((Instant::now(), start.elapsed().as_secs_f64() * 1_000.0));
    }
}

fn begin_render_cpu(mut timing: ResMut<RenderCpuStart>) {
    timing.0 = Some(Instant::now());
}

fn end_render_cpu(mut timing: ResMut<RenderCpuStart>, completed: Res<RenderCpuCompleted>) {
    if let Some(start) = timing.0.take()
        && let Ok(mut rows) = completed.0.lock()
    {
        rows.push((Instant::now(), start.elapsed().as_secs_f64() * 1_000.0));
    }
}

fn begin(world: &mut World) {
    world.resource_scope(|world, mut timing: Mut<Timing>| {
        timing.active = timing
            .slots
            .iter()
            .position(|slot| !slot.busy.swap(true, Ordering::AcqRel));
        let Some(index) = timing.active else {
            return;
        };
        let device = world.resource::<RenderDevice>();
        let mut encoder = device.create_command_encoder(&default());
        encoder.write_timestamp(&timing.slots[index].queries, 0);
        // Submit immediately: later queue writes/refit submissions must follow it.
        world.resource::<RenderQueue>().submit([encoder.finish()]);
    });
}

fn end(world: &mut World) {
    world.resource_scope(|world, mut timing: Mut<Timing>| {
        let Some(index) = timing.active.take() else {
            return;
        };
        let slot = &timing.slots[index];
        let device = world.resource::<RenderDevice>();
        let queue = world.resource::<RenderQueue>();
        let mut encoder = device.create_command_encoder(&default());
        encoder.write_timestamp(&slot.queries, 1);
        encoder.resolve_query_set(&slot.queries, 0..2, &slot.resolve, 0);
        encoder.copy_buffer_to_buffer(&slot.resolve, 0, &slot.read, 0, 16);
        let read = slot.read.clone();
        let busy = slot.busy.clone();
        let completed = world.resource::<Completed>().clone();
        let period = queue.get_timestamp_period();
        encoder.map_buffer_on_submit(&slot.read, wgpu::MapMode::Read, .., move |result| {
            if result.is_ok() {
                let mapped = read
                    .get_mapped_range(..)
                    .expect("timestamp buffer mapping should be valid");
                let begin = u64::from_le_bytes(mapped[..8].try_into().unwrap());
                let end = u64::from_le_bytes(mapped[8..16].try_into().unwrap());
                if let Some(ms) = elapsed_ms(begin, end, period)
                    && let Ok(mut rows) = completed.0.lock()
                {
                    rows.push((Instant::now(), ms));
                }
                drop(mapped);
                read.unmap();
            }
            busy.store(false, Ordering::Release);
        });
        // RenderSystems::Render includes native graph submission and readbacks.
        queue.submit([encoder.finish()]);
    });
}

fn elapsed_ms(begin: u64, end: u64, period_ns: f32) -> Option<f64> {
    if !period_ns.is_finite() || period_ns <= 0.0 {
        return None;
    }
    let result = end.checked_sub(begin)? as f64 * f64::from(period_ns) / 1_000_000.0;
    (result.is_finite() && result >= 0.0).then_some(result)
}

fn publish(completed: Res<Completed>, mut store: ResMut<DiagnosticsStore>) {
    let path = DiagnosticPath::new("render/penumbra_frame/elapsed_gpu");
    if store.get(&path).is_none() {
        store.add(Diagnostic::new(path.clone()).with_suffix("ms"));
    }
    if let Ok(mut rows) = completed.0.lock() {
        for (time, value) in rows.drain(..) {
            store
                .get_mut(&path)
                .unwrap()
                .add_measurement(DiagnosticMeasurement { time, value });
        }
    }
}

fn publish_cpu(mut timing: ResMut<CpuTiming>, mut store: ResMut<DiagnosticsStore>) {
    let path = DiagnosticPath::new("render/penumbra_frame/main_schedule_cpu");
    if store.get(&path).is_none() {
        store.add(Diagnostic::new(path.clone()).with_suffix("ms"));
    }
    for &(time, value) in &timing.completed {
        store
            .get_mut(&path)
            .unwrap()
            .add_measurement(DiagnosticMeasurement { time, value });
    }
    // Keep only measurements from the current frame; the diagnostics store
    // owns its rolling history.
    timing.completed.clear();
}

fn publish_render_cpu(completed: Res<RenderCpuCompleted>, mut store: ResMut<DiagnosticsStore>) {
    let path = DiagnosticPath::new("render/penumbra_frame/render_schedule_cpu");
    if store.get(&path).is_none() {
        store.add(Diagnostic::new(path.clone()).with_suffix("ms"));
    }
    if let Ok(mut rows) = completed.0.lock() {
        for (time, value) in rows.drain(..) {
            store
                .get_mut(&path)
                .unwrap()
                .add_measurement(DiagnosticMeasurement { time, value });
        }
    }
}

#[cfg(test)]
mod checks {
    use super::*;
    #[test]
    fn timestamp_conversion_rejects_invalid_order_and_period() {
        assert_eq!(elapsed_ms(100, 1_000_100, 2.0), Some(2.0));
        assert_eq!(elapsed_ms(2, 1, 1.0), None);
        assert_eq!(elapsed_ms(0, 1, f32::NAN), None);
        assert_eq!(elapsed_ms(0, 0, -1.0), None);
    }
}

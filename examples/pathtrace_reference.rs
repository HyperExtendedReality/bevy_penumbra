//! Small independent CPU path tracer for the exact Cornell scene in
//! `cornell_box.rs`. It writes camera-space scene-linear diffuse radiance in
//! the same `irradiance / pi, before receiver albedo` units as the raw GI
//! capture, plus a primary-hit mask. No Bevy cache, screen history, denoiser,
//! exposure, or production traversal code is used. It is an offline quality
//! reference, not a selectable GI transport. `--bounces` counts diffuse
//! vertices including the primary receiver.

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::path::Path;
use std::time::Instant;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct V3(f32, f32, f32);

impl V3 {
    const ZERO: Self = Self(0.0, 0.0, 0.0);
    const ONE: Self = Self(1.0, 1.0, 1.0);
    const X: Self = Self(1.0, 0.0, 0.0);
    const Y: Self = Self(0.0, 1.0, 0.0);
    const Z: Self = Self(0.0, 0.0, 1.0);

    fn dot(self, rhs: Self) -> f32 {
        self.0 * rhs.0 + self.1 * rhs.1 + self.2 * rhs.2
    }

    fn cross(self, rhs: Self) -> Self {
        Self(
            self.1 * rhs.2 - self.2 * rhs.1,
            self.2 * rhs.0 - self.0 * rhs.2,
            self.0 * rhs.1 - self.1 * rhs.0,
        )
    }

    fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    fn is_finite(self) -> bool {
        self.0.is_finite() && self.1.is_finite() && self.2.is_finite()
    }

    fn normalized(self) -> Self {
        self / self.length().max(1.0e-20)
    }

    fn component_min(self, rhs: Self) -> Self {
        Self(self.0.min(rhs.0), self.1.min(rhs.1), self.2.min(rhs.2))
    }

    fn component_max(self, rhs: Self) -> Self {
        Self(self.0.max(rhs.0), self.1.max(rhs.1), self.2.max(rhs.2))
    }

    fn hadamard(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0, self.1 * rhs.1, self.2 * rhs.2)
    }

    fn max_component(self) -> f32 {
        self.0.max(self.1).max(self.2)
    }

    fn axis(self, axis: usize) -> f32 {
        match axis {
            0 => self.0,
            1 => self.1,
            _ => self.2,
        }
    }

    fn rotate_y(self, angle: f32) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self(
            cos * self.0 + sin * self.2,
            self.1,
            -sin * self.0 + cos * self.2,
        )
    }
}

impl Add for V3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2)
    }
}
impl AddAssign for V3 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl Sub for V3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2)
    }
}
impl Mul<f32> for V3 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Self(self.0 * rhs, self.1 * rhs, self.2 * rhs)
    }
}
impl Div<f32> for V3 {
    type Output = Self;
    fn div(self, rhs: f32) -> Self {
        Self(self.0 / rhs, self.1 / rhs, self.2 / rhs)
    }
}
impl Neg for V3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0, -self.1, -self.2)
    }
}

#[derive(Clone, Copy)]
struct Ray {
    origin: V3,
    direction: V3,
}

#[derive(Clone, Copy, Debug)]
struct Material {
    albedo: V3,
    emission: V3,
    emitter: bool,
}

#[derive(Clone, Copy, Debug)]
struct Triangle {
    a: V3,
    b: V3,
    c: V3,
    normal: V3,
    material: Material,
    min: V3,
    max: V3,
    centroid: V3,
}

impl Triangle {
    fn new(a: V3, b: V3, c: V3, material: Material) -> Self {
        let normal = (b - a).cross(c - a).normalized();
        let min = a.component_min(b).component_min(c);
        let max = a.component_max(b).component_max(c);
        Self {
            a,
            b,
            c,
            normal,
            material,
            min: min - V3(1.0e-5, 1.0e-5, 1.0e-5),
            max: max + V3(1.0e-5, 1.0e-5, 1.0e-5),
            centroid: (a + b + c) / 3.0,
        }
    }

    fn hit(&self, ray: Ray, max_t: f32) -> Option<f32> {
        let edge1 = self.b - self.a;
        let edge2 = self.c - self.a;
        let p = ray.direction.cross(edge2);
        let determinant = edge1.dot(p);
        // Cull the back side. All Cornell surfaces are authored with their
        // visible room-side normal, including the one-sided ceiling emitter.
        if determinant <= 1.0e-8 {
            return None;
        }
        let inverse = 1.0 / determinant;
        let from_a = ray.origin - self.a;
        let u = from_a.dot(p) * inverse;
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let q = from_a.cross(edge1);
        let v = ray.direction.dot(q) * inverse;
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let t = edge2.dot(q) * inverse;
        (t > RAY_EPSILON && t < max_t).then_some(t)
    }
}

#[derive(Clone, Copy, Debug)]
struct Hit {
    t: f32,
    normal: V3,
    material: Material,
    #[cfg(test)]
    triangle: usize,
}

#[derive(Clone, Copy, Debug)]
struct Bounds {
    min: V3,
    max: V3,
}

impl Bounds {
    fn from_triangle(triangle: Triangle) -> Self {
        Self {
            min: triangle.min,
            max: triangle.max,
        }
    }

    fn union(self, rhs: Self) -> Self {
        Self {
            min: self.min.component_min(rhs.min),
            max: self.max.component_max(rhs.max),
        }
    }

    fn hit(self, ray: Ray, max_t: f32) -> bool {
        let mut near = RAY_EPSILON;
        let mut far = max_t;
        for axis in 0..3 {
            let origin = ray.origin.axis(axis);
            let direction = ray.direction.axis(axis);
            let min = self.min.axis(axis);
            let max = self.max.axis(axis);
            if direction.abs() < 1.0e-12 {
                if origin < min || origin > max {
                    return false;
                }
                continue;
            }
            let inverse = 1.0 / direction;
            let mut a = (min - origin) * inverse;
            let mut b = (max - origin) * inverse;
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
            near = near.max(a);
            far = far.min(b);
            if far < near {
                return false;
            }
        }
        far > RAY_EPSILON
    }
}

#[derive(Clone, Copy, Debug)]
enum NodeKind {
    Leaf { start: usize, end: usize },
    Branch { left: usize, right: usize },
}

#[derive(Clone, Copy, Debug)]
struct BvhNode {
    bounds: Bounds,
    kind: NodeKind,
}

#[derive(Clone, Copy)]
struct AreaLight {
    center: V3,
    edge_u: V3,
    edge_v: V3,
    normal: V3,
    radiance: V3,
    area: f32,
}

struct Scene {
    triangles: Vec<Triangle>,
    nodes: Vec<BvhNode>,
    triangle_order: Vec<usize>,
    light: AreaLight,
}

const RAY_EPSILON: f32 = 1.0e-4;
const PI: f32 = std::f32::consts::PI;

impl Scene {
    fn cornell() -> Self {
        let white = Material {
            albedo: srgb(V3(0.8, 0.8, 0.8)),
            emission: V3::ZERO,
            emitter: false,
        };
        let red = Material {
            albedo: srgb(V3(0.85, 0.12, 0.08)),
            ..white
        };
        let green = Material {
            albedo: srgb(V3(0.1, 0.7, 0.2)),
            ..white
        };
        let emitter = Material {
            albedo: V3::ONE,
            emission: V3(30.0, 25.0, 18.0),
            emitter: true,
        };
        let mut triangles = Vec::with_capacity(42);

        // These are the visible inner planes of the six cuboids in the Bevy
        // demo. The open front is intentionally absent.
        add_quad(
            &mut triangles,
            V3(0.0, 0.0, 0.0),
            V3(6.0, 0.0, 0.0),
            V3(0.0, 0.0, -6.0),
            white,
        ); // floor, +Y
        add_quad(
            &mut triangles,
            V3(0.0, 2.0, -3.0),
            V3(6.0, 0.0, 0.0),
            V3(0.0, 4.0, 0.0),
            white,
        ); // back, +Z
        add_quad(
            &mut triangles,
            V3(0.0, 4.0, 0.0),
            V3(6.0, 0.0, 0.0),
            V3(0.0, 0.0, 6.0),
            white,
        ); // ceiling underside, -Y
        add_quad(
            &mut triangles,
            V3(-3.0, 2.0, 0.0),
            V3(0.0, 4.0, 0.0),
            V3(0.0, 0.0, 6.0),
            red,
        ); // left, +X
        add_quad(
            &mut triangles,
            V3(3.0, 2.0, 0.0),
            V3(0.0, 0.0, 6.0),
            V3(0.0, 4.0, 0.0),
            green,
        ); // right, -X

        let light = AreaLight {
            center: V3(0.0, 3.97, -0.8),
            edge_u: V3(2.0, 0.0, 0.0),
            edge_v: V3(0.0, 0.0, 2.0),
            normal: V3(0.0, -1.0, 0.0),
            radiance: emitter.emission,
            area: 4.0,
        };
        add_quad(
            &mut triangles,
            light.center,
            light.edge_u,
            light.edge_v,
            emitter,
        );

        add_box(
            &mut triangles,
            V3(-1.1, 0.75, 0.0),
            V3(1.4, 1.5, 1.4),
            0.3,
            white,
        );
        add_box(
            &mut triangles,
            V3(1.0, 1.4, -1.0),
            V3(1.2, 2.8, 1.2),
            -0.3,
            white,
        );

        let mut scene = Self {
            triangle_order: (0..triangles.len()).collect(),
            nodes: Vec::with_capacity(triangles.len() * 2),
            triangles,
            light,
        };
        build_node(
            &mut scene.nodes,
            &scene.triangles,
            &mut scene.triangle_order,
            0,
            scene.triangles.len(),
        );
        scene
    }

    fn intersect(&self, ray: Ray) -> Option<Hit> {
        let mut best_t = f32::INFINITY;
        let mut best = None;
        let mut stack = [0_usize; 64];
        let mut count = 1;
        stack[0] = 0;
        while count > 0 {
            count -= 1;
            let index = stack[count];
            let node = self.nodes[index];
            if !node.bounds.hit(ray, best_t) {
                continue;
            }
            match node.kind {
                NodeKind::Leaf { start, end } => {
                    for &triangle_index in &self.triangle_order[start..end] {
                        let triangle = self.triangles[triangle_index];
                        if let Some(t) = triangle.hit(ray, best_t) {
                            best_t = t;
                            best = Some(Hit {
                                t,
                                normal: triangle.normal,
                                material: triangle.material,
                                #[cfg(test)]
                                triangle: triangle_index,
                            });
                        }
                    }
                }
                NodeKind::Branch { left, right } => {
                    // Child ordering is only an acceleration hint. Pushing the
                    // farther child first lets the nearer one tighten best_t.
                    let left_hit = self.nodes[left].bounds.hit(ray, best_t);
                    let right_hit = self.nodes[right].bounds.hit(ray, best_t);
                    match (left_hit, right_hit) {
                        (true, true) => {
                            stack[count] = right;
                            stack[count + 1] = left;
                            count += 2;
                        }
                        (true, false) => {
                            stack[count] = left;
                            count += 1;
                        }
                        (false, true) => {
                            stack[count] = right;
                            count += 1;
                        }
                        (false, false) => {}
                    }
                }
            }
        }
        best
    }

    #[cfg(test)]
    fn brute_intersect(&self, ray: Ray) -> Option<Hit> {
        let mut best: Option<Hit> = None;
        for (triangle_index, triangle) in self.triangles.iter().enumerate() {
            let max_t = best.map_or(f32::INFINITY, |hit| hit.t);
            if let Some(t) = triangle.hit(ray, max_t) {
                best = Some(Hit {
                    t,
                    normal: triangle.normal,
                    material: triangle.material,
                    #[cfg(test)]
                    triangle: triangle_index,
                });
            }
        }
        best
    }

    fn visible_to_light(&self, origin: V3, normal: V3, point: V3) -> bool {
        let delta = point - (origin + normal * RAY_EPSILON);
        let distance = delta.length();
        if distance <= RAY_EPSILON {
            return false;
        }
        let ray = Ray {
            origin: origin + normal * RAY_EPSILON,
            direction: delta / distance,
        };
        self.intersect(ray)
            .is_some_and(|hit| hit.material.emitter && hit.t >= distance - 2.0e-3)
    }
}

fn add_quad(triangles: &mut Vec<Triangle>, center: V3, edge_u: V3, edge_v: V3, material: Material) {
    let p00 = center - edge_u * 0.5 - edge_v * 0.5;
    let p10 = center + edge_u * 0.5 - edge_v * 0.5;
    let p01 = center - edge_u * 0.5 + edge_v * 0.5;
    let p11 = center + edge_u * 0.5 + edge_v * 0.5;
    triangles.push(Triangle::new(p00, p10, p01, material));
    triangles.push(Triangle::new(p10, p11, p01, material));
}

fn add_box(triangles: &mut Vec<Triangle>, center: V3, size: V3, yaw: f32, material: Material) {
    let faces = [
        (V3::X, V3::Y, V3::Z, size.1, size.2),
        (-V3::X, V3::Z, V3::Y, size.2, size.1),
        (V3::Y, V3::Z, V3::X, size.2, size.0),
        (-V3::Y, V3::X, V3::Z, size.0, size.2),
        (V3::Z, V3::X, V3::Y, size.0, size.1),
        (-V3::Z, V3::Y, V3::X, size.1, size.0),
    ];
    for (normal, axis_u, axis_v, extent_u, extent_v) in faces {
        let half_extent = size.hadamard(V3(normal.0.abs(), normal.1.abs(), normal.2.abs())) * 0.5;
        let local_center = normal.hadamard(half_extent);
        let face_center = center + local_center.rotate_y(yaw);
        let edge_u = axis_u.rotate_y(yaw) * extent_u;
        let edge_v = axis_v.rotate_y(yaw) * extent_v;
        add_quad(triangles, face_center, edge_u, edge_v, material);
    }
}

fn build_node(
    nodes: &mut Vec<BvhNode>,
    triangles: &[Triangle],
    order: &mut [usize],
    start: usize,
    end: usize,
) -> usize {
    let mut bounds = Bounds::from_triangle(triangles[order[start]]);
    let mut centroid_min = triangles[order[start]].centroid;
    let mut centroid_max = centroid_min;
    for &triangle_index in &order[start + 1..end] {
        let triangle = triangles[triangle_index];
        bounds = bounds.union(Bounds::from_triangle(triangle));
        centroid_min = centroid_min.component_min(triangle.centroid);
        centroid_max = centroid_max.component_max(triangle.centroid);
    }
    let index = nodes.len();
    nodes.push(BvhNode {
        bounds,
        kind: NodeKind::Leaf { start, end },
    });
    if end - start <= 4 {
        return index;
    }
    let extent = centroid_max - centroid_min;
    let axis = if extent.0 >= extent.1 && extent.0 >= extent.2 {
        0
    } else if extent.1 >= extent.2 {
        1
    } else {
        2
    };
    order[start..end].sort_unstable_by(|&a, &b| {
        triangles[a]
            .centroid
            .axis(axis)
            .total_cmp(&triangles[b].centroid.axis(axis))
    });
    let middle = start + (end - start) / 2;
    let left = build_node(nodes, triangles, order, start, middle);
    let right = build_node(nodes, triangles, order, middle, end);
    nodes[index].kind = NodeKind::Branch { left, right };
    index
}

fn srgb(value: V3) -> V3 {
    fn channel(value: f32) -> f32 {
        if value <= 0.04045 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    }
    V3(channel(value.0), channel(value.1), channel(value.2))
}

#[derive(Clone, Copy)]
struct LightSample {
    point: V3,
    pdf_solid_angle: f32,
    cosine_surface: f32,
}

fn sample_light(scene: &Scene, point: V3, normal: V3, u: f32, v: f32) -> Option<LightSample> {
    let light = scene.light;
    let sample = light.center + light.edge_u * (u - 0.5) + light.edge_v * (v - 0.5);
    let delta = sample - point;
    let distance_squared = delta.dot(delta);
    if distance_squared <= 1.0e-12 {
        return None;
    }
    let distance = distance_squared.sqrt();
    let direction = delta / distance;
    let cosine_surface = normal.dot(direction).max(0.0);
    let cosine_light = light.normal.dot(-direction).max(0.0);
    if cosine_surface <= 0.0 || cosine_light <= 0.0 {
        return None;
    }
    Some(LightSample {
        point: sample,
        pdf_solid_angle: distance_squared / (light.area * cosine_light),
        cosine_surface,
    })
}

fn sample_cosine_hemisphere(normal: V3, u: f32, v: f32) -> (V3, f32) {
    let radius = u.sqrt();
    let angle = 2.0 * PI * v;
    let local = V3(radius * angle.cos(), radius * angle.sin(), (1.0 - u).sqrt());
    let tangent = if normal.2.abs() < 0.999 {
        V3::Z.cross(normal).normalized()
    } else {
        V3::X.cross(normal).normalized()
    };
    let bitangent = normal.cross(tangent);
    let direction = (tangent * local.0 + bitangent * local.1 + normal * local.2).normalized();
    (direction, direction.dot(normal).max(0.0) / PI)
}

fn power_heuristic(a: f32, b: f32) -> f32 {
    let aa = a * a;
    aa / (aa + b * b).max(1.0e-30)
}

// Matches shaders/common.slang:disneyDiffuseMultiplier for the Cornell demo's
// perceptual_roughness=1.0 materials. Cosine samples are only a proposal; the
// directional Disney factor weights direct and indirect transport.
fn disney_diffuse_multiplier(normal: V3, light: V3, view: V3, perceptual_roughness: f32) -> f32 {
    let n_dot_l = normal.dot(light).clamp(0.0, 1.0);
    let n_dot_v = normal.dot(view).clamp(0.0, 1.0);
    let half_sum = light + view;
    let half_length_squared = half_sum.dot(half_sum);
    if n_dot_l <= 0.0 || n_dot_v <= 0.0 || half_length_squared <= 1.0e-20 {
        return 0.0;
    }
    let half_vector = half_sum / half_length_squared.sqrt();
    let roughness = perceptual_roughness.clamp(0.0, 1.0).max(0.089).powi(2);
    let l_dot_h = light.dot(half_vector).clamp(0.0, 1.0);
    let f90 = 0.5 + 2.0 * roughness * l_dot_h * l_dot_h;
    let light_scatter = 1.0 + (f90 - 1.0) * (1.0 - n_dot_l).powi(5);
    let view_scatter = 1.0 + (f90 - 1.0) * (1.0 - n_dot_v).powi(5);
    light_scatter * view_scatter
}

struct Sequence {
    sample: u64,
    seed: u64,
    dimension: usize,
}

impl Sequence {
    fn new(sample: u64, pixel: u64) -> Self {
        Self {
            sample,
            seed: mix64(pixel ^ 0x9e37_79b9_7f4a_7c15),
            dimension: 0,
        }
    }

    fn next(&mut self) -> f32 {
        const PRIMES: [u32; 40] = [
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
            89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
        ];
        let dimension = self.dimension;
        self.dimension += 1;
        let base = PRIMES[dimension % PRIMES.len()];
        let scramble = unit_float(mix64(
            self.seed ^ (dimension as u64).wrapping_mul(0xd6e8_feb8_6659_fd93),
        ));
        let value = radical_inverse(self.sample + 1, base);
        (value + scramble).fract().clamp(0.0, 1.0 - f32::EPSILON)
    }
}

fn radical_inverse(mut index: u64, base: u32) -> f32 {
    let inverse = 1.0 / base as f32;
    let mut place = inverse;
    let mut result = 0.0;
    while index > 0 {
        result += (index % u64::from(base)) as f32 * place;
        index /= u64::from(base);
        place *= inverse;
    }
    result
}

fn unit_float(value: u64) -> f32 {
    ((value >> 40) as u32) as f32 / 16_777_216.0
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

struct Camera {
    origin: V3,
    forward: V3,
    right: V3,
    up: V3,
    aspect: f32,
    tan_half_fov: f32,
}

impl Camera {
    fn cornell(width: usize, height: usize) -> Self {
        let origin = V3(0.0, 2.0, 8.5);
        let forward = (V3(0.0, 1.8, -0.5) - origin).normalized();
        let right = forward.cross(V3::Y).normalized();
        let up = right.cross(forward).normalized();
        Self {
            origin,
            forward,
            right,
            up,
            aspect: width as f32 / height as f32,
            tan_half_fov: (std::f32::consts::FRAC_PI_4 * 0.5).tan(),
        }
    }

    fn ray(
        &self,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        jitter_x: f32,
        jitter_y: f32,
    ) -> Ray {
        let screen_x =
            (2.0 * (x as f32 + jitter_x) / width as f32 - 1.0) * self.aspect * self.tan_half_fov;
        let screen_y = (1.0 - 2.0 * (y as f32 + jitter_y) / height as f32) * self.tan_half_fov;
        Ray {
            origin: self.origin,
            direction: (self.forward + self.right * screen_x + self.up * screen_y).normalized(),
        }
    }
}

fn trace_sample(
    scene: &Scene,
    mut ray: Ray,
    sequence: &mut Sequence,
    max_bounces: usize,
) -> (V3, bool) {
    let Some(mut hit) = scene.intersect(ray) else {
        return (V3::ZERO, false);
    };
    let primary_receiver = !hit.material.emitter;
    if !primary_receiver {
        return (hit.material.emission, false);
    }
    let mut radiance = V3::ZERO;
    let mut throughput = V3::ONE;
    let mut previous_bsdf_pdf = 0.0;
    let mut previous_point = V3::ZERO;
    let mut previous_was_bsdf = false;

    for bounce in 0..max_bounces.max(1) {
        let point = ray.origin + ray.direction * hit.t;
        if hit.material.emitter {
            let from_previous = point - previous_point;
            let direction = from_previous.normalized();
            let cosine_light = scene.light.normal.dot(-direction).max(0.0);
            let light_pdf = if previous_was_bsdf && cosine_light > 0.0 {
                from_previous.dot(from_previous) / (scene.light.area * cosine_light)
            } else {
                0.0
            };
            let mis = if previous_was_bsdf {
                power_heuristic(previous_bsdf_pdf, light_pdf)
            } else {
                1.0
            };
            radiance += throughput.hadamard(hit.material.emission) * mis;
            break;
        }

        // Primary albedo is set to one: this is irradiance/pi before receiver
        // albedo, matching RadianceCascadesRawOutput. Later vertices use their
        // exact linearized StandardMaterial base color.
        let rho = if bounce == 0 {
            V3::ONE
        } else {
            hit.material.albedo
        };
        let surface_point = point + hit.normal * RAY_EPSILON;
        let view = (-ray.direction).normalized();

        if let Some(sample) = sample_light(
            scene,
            surface_point,
            hit.normal,
            sequence.next(),
            sequence.next(),
        ) && scene.visible_to_light(surface_point, hit.normal, sample.point)
        {
            let bsdf_pdf = sample.cosine_surface / PI;
            // NEE is the only estimator on the final requested diffuse
            // vertex. MIS would downweight it as if a competing BSDF ray
            // had been sampled, even though that ray is deliberately not
            // traced at this depth.
            let has_bsdf_competitor = bounce + 1 < max_bounces.max(1);
            let mis = if has_bsdf_competitor {
                power_heuristic(sample.pdf_solid_angle, bsdf_pdf)
            } else {
                1.0
            };
            let diffuse = disney_diffuse_multiplier(
                hit.normal,
                (sample.point - surface_point).normalized(),
                view,
                1.0,
            );
            let scale = diffuse * sample.cosine_surface / (PI * sample.pdf_solid_angle) * mis;
            radiance += throughput.hadamard(rho).hadamard(scene.light.radiance) * scale;
        }
        if bounce + 1 >= max_bounces.max(1) {
            break;
        }

        let (direction, pdf) =
            sample_cosine_hemisphere(hit.normal, sequence.next(), sequence.next());
        if pdf <= 0.0 || !pdf.is_finite() {
            break;
        }
        throughput =
            throughput.hadamard(rho) * disney_diffuse_multiplier(hit.normal, direction, view, 1.0);
        if bounce >= 3 {
            let survival = throughput.max_component().clamp(0.05, 0.95);
            if sequence.next() >= survival {
                break;
            }
            throughput = throughput / survival;
        }
        previous_bsdf_pdf = pdf;
        previous_point = surface_point;
        previous_was_bsdf = true;
        ray = Ray {
            origin: surface_point,
            direction,
        };
        let Some(next_hit) = scene.intersect(ray) else {
            break;
        };
        hit = next_hit;
    }
    (radiance, primary_receiver)
}

fn render(width: usize, height: usize, samples: u64, bounces: usize) -> (Vec<V3>, Vec<u8>, f64) {
    let scene = Scene::cornell();
    let camera = Camera::cornell(width, height);
    let mut image = vec![V3::ZERO; width * height];
    let mut mask = vec![0_u8; width * height];
    let started = Instant::now();
    for y in 0..height {
        for x in 0..width {
            let pixel = y as u64 * width as u64 + x as u64;
            let mut accumulated = V3::ZERO;
            let mut covered = false;
            for sample in 0..samples {
                let mut sequence = Sequence::new(sample, pixel);
                // Match the production raster receiver path: one fixed
                // center sample per pixel (MSAA is disabled), so support
                // never mixes a receiver and the open background/emitter
                // across stochastic camera jitter.
                let ray = camera.ray(x, y, width, height, 0.5, 0.5);
                let (value, primary_receiver) = trace_sample(&scene, ray, &mut sequence, bounces);
                accumulated += value;
                covered |= primary_receiver;
            }
            image[pixel as usize] = accumulated / samples as f32;
            mask[pixel as usize] = if covered { 255 } else { 0 };
        }
    }
    debug_assert!(image.iter().all(|pixel| pixel.is_finite()));
    (image, mask, started.elapsed().as_secs_f64())
}

fn write_pfm(path: &Path, width: usize, height: usize, image: &[V3]) -> std::io::Result<()> {
    let mut bytes = format!("PF\n{width} {height}\n-1.0\n").into_bytes();
    for row in image.chunks_exact(width).rev() {
        for pixel in row {
            for channel in [pixel.0, pixel.1, pixel.2] {
                bytes.extend_from_slice(&channel.to_le_bytes());
            }
        }
    }
    std::fs::write(path, bytes)
}

fn write_mask(path: &Path, width: usize, height: usize, mask: &[u8]) -> std::io::Result<()> {
    let mut bytes = format!("P5\n{width} {height}\n255\n").into_bytes();
    bytes.extend_from_slice(mask);
    std::fs::write(path, bytes)
}

fn parse_arg<T: std::str::FromStr>(args: &[String], prefix: &str, fallback: T) -> T {
    args.iter()
        .find_map(|argument| argument.strip_prefix(prefix)?.parse().ok())
        .unwrap_or(fallback)
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let width = parse_arg(&args, "--width=", 160_usize).max(1);
    let height = parse_arg(&args, "--height=", 120_usize).max(1);
    let samples = parse_arg(&args, "--samples=", 256_u64).max(1);
    let bounces = if args.iter().any(|arg| arg == "--deep") {
        8
    } else {
        parse_arg(&args, "--bounces=", 3_usize).max(1)
    };
    let output = args
        .iter()
        .find_map(|arg| arg.strip_prefix("--output=").map(Path::new))
        .unwrap_or_else(|| Path::new("tmp/cornell_reference.pfm"));
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    let (image, mask, elapsed) = render(width, height, samples, bounces);
    write_pfm(output, width, height, &image)?;
    let mask_path = output.with_extension("mask.pgm");
    write_mask(&mask_path, width, height, &mask)?;
    let max_luma = image
        .iter()
        .map(|v| (0.2126 * v.0 + 0.7152 * v.1 + 0.0722 * v.2) as f64)
        .fold(0.0, f64::max);
    let mean_luma = image
        .iter()
        .map(|v| 0.2126 * v.0 as f64 + 0.7152 * v.1 as f64 + 0.0722 * v.2 as f64)
        .sum::<f64>()
        / image.len() as f64;
    let coverage = mask.iter().filter(|&&value| value != 0).count() as f64 / mask.len() as f64;
    let metadata = format!(
        "{{\n  \"algorithm\": \"independent CPU path tracing, area-light NEE with power-heuristic MIS\",\n  \"units\": \"linear RGB diffuse transport divided by pi, before primary receiver albedo\",\n  \"direct_diffuse\": \"Bevy Disney diffuse multiplier, perceptual roughness 1.0\",\n  \"indirect_continuation\": \"cosine-proposed Disney diffuse, albedo weighted at every bounce\",\n  \"camera_sampling\": \"fixed pixel centers; matches MSAA-off raster GI receivers\",\n  \"scene\": \"cornell_box.rs geometry and linearized StandardMaterial base colors\",\n  \"width\": {width},\n  \"height\": {height},\n  \"samples_per_pixel\": {samples},\n  \"diffuse_bounces\": {bounces},\n  \"camera_fov_degrees\": 45.0,\n  \"elapsed_seconds\": {elapsed:.6},\n  \"primary_receiver_coverage\": {coverage:.8},\n  \"mean_luminance\": {mean_luma:.8},\n  \"max_luminance\": {max_luma:.8},\n  \"mask_file\": {:?}\n}}\n",
        mask_path.to_string_lossy()
    );
    std::fs::write(output.with_extension("json"), metadata)?;
    println!(
        "rendered {width}x{height} at {samples} spp / {bounces} diffuse bounces in {elapsed:.2}s; coverage {:.1}% output {}",
        coverage * 100.0,
        output.display()
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("pathtrace_reference: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_hemisphere_sampling_matches_its_proposal_pdf() {
        let normal = V3::Y;
        let mut mean_cosine = 0.0;
        let samples = 4096;
        for index in 0..samples {
            let u = radical_inverse(index + 1, 2);
            let v = radical_inverse(index + 1, 3);
            let (direction, pdf) = sample_cosine_hemisphere(normal, u, v);
            let cosine = normal.dot(direction).max(0.0);
            assert!(pdf.is_finite() && pdf > 0.0);
            assert!((pdf - cosine / PI).abs() < 1.0e-6);
            mean_cosine += cosine;
        }
        assert!((mean_cosine / samples as f32 - 2.0 / 3.0).abs() < 1.0e-3);
    }

    #[test]
    fn bvh_matches_brute_force_on_deterministic_cornell_rays() {
        let scene = Scene::cornell();
        for index in 0..4096_u64 {
            let mut sequence = Sequence::new(index, 0x1234_5678);
            let direction = V3(
                sequence.next() * 2.0 - 1.0,
                sequence.next() * 2.0 - 1.0,
                -sequence.next(),
            )
            .normalized();
            let ray = Ray {
                origin: V3(0.0, 2.0, 8.5),
                direction,
            };
            let accelerated = scene.intersect(ray);
            let brute = scene.brute_intersect(ray);
            assert_eq!(accelerated.is_some(), brute.is_some(), "ray {index}");
            if let (Some(a), Some(b)) = (accelerated, brute) {
                assert_eq!(a.triangle, b.triangle, "ray {index}");
                assert!(
                    (a.t - b.t).abs() < 1.0e-5,
                    "ray {index}: {} vs {}",
                    a.t,
                    b.t
                );
            }
        }
    }

    #[test]
    fn progressive_sequence_is_deterministic_and_advances_between_samples() {
        let mut first = Sequence::new(7, 99);
        let mut repeat = Sequence::new(7, 99);
        let mut next = Sequence::new(8, 99);
        let a = [first.next(), first.next(), first.next(), first.next()];
        let b = [repeat.next(), repeat.next(), repeat.next(), repeat.next()];
        let c = [next.next(), next.next(), next.next(), next.next()];
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a.iter().all(|x| (0.0..1.0).contains(x)));
    }

    #[test]
    fn raw_hdr_emission_is_not_clamped() {
        let scene = Scene::cornell();
        let mut sequence = Sequence::new(0, 0);
        let (value, receiver) = trace_sample(
            &scene,
            Ray {
                origin: V3(0.0, 3.0, -0.8),
                direction: V3::Y,
            },
            &mut sequence,
            1,
        );
        assert!(!receiver);
        assert_eq!(value, scene.light.radiance);
        assert!(value.max_component() > 16.0);
    }

    #[test]
    fn final_vertex_nee_matches_analytic_rectangle_irradiance() {
        let receiver = Material {
            albedo: V3::ONE,
            emission: V3::ZERO,
            emitter: false,
        };
        let light_material = Material {
            albedo: V3::ONE,
            emission: V3(30.0, 25.0, 18.0),
            emitter: true,
        };
        let light = AreaLight {
            center: V3(0.0, 0.5, 0.0),
            edge_u: V3(2.0, 0.0, 0.0),
            edge_v: V3(0.0, 0.0, 2.0),
            normal: V3(0.0, -1.0, 0.0),
            radiance: light_material.emission,
            area: 4.0,
        };
        let mut triangles = Vec::new();
        add_quad(
            &mut triangles,
            V3::ZERO,
            V3(4.0, 0.0, 0.0),
            V3(0.0, 0.0, -4.0),
            receiver,
        );
        add_quad(
            &mut triangles,
            light.center,
            light.edge_u,
            light.edge_v,
            light_material,
        );
        let mut scene = Scene {
            triangle_order: (0..triangles.len()).collect(),
            nodes: Vec::with_capacity(triangles.len() * 2),
            triangles,
            light,
        };
        build_node(
            &mut scene.nodes,
            &scene.triangles,
            &mut scene.triangle_order,
            0,
            scene.triangles.len(),
        );

        let ray = Ray {
            origin: V3(0.0, 1.0, 0.0),
            direction: -V3::Y,
        };
        let sample_count = 32_768_u64;
        let mut estimate = V3::ZERO;
        for sample in 0..sample_count {
            let mut sequence = Sequence::new(sample, 0x4e45_455f_5245_4631);
            let (value, receiver_hit) = trace_sample(&scene, ray, &mut sequence, 1);
            assert!(receiver_hit, "sample {sample} missed the receiver");
            estimate += value;
        }
        estimate = estimate / sample_count as f32;

        // Integrate the production direct diffuse area-light kernel on a dense
        // midpoint grid. This reference does not use the path tracer's light
        // PDF or MIS implementation.
        let receiver_point = V3(0.0, RAY_EPSILON, 0.0);
        let quadrature_resolution = 1024_u32;
        let mut integrated = V3::ZERO;
        for y in 0..quadrature_resolution {
            for x in 0..quadrature_resolution {
                let uv = V3(
                    (x as f32 + 0.5) / quadrature_resolution as f32,
                    0.0,
                    (y as f32 + 0.5) / quadrature_resolution as f32,
                );
                let light_point =
                    light.center + light.edge_u * (uv.0 - 0.5) + light.edge_v * (uv.2 - 0.5);
                let delta = light_point - receiver_point;
                let distance_squared = delta.dot(delta);
                let direction = delta / distance_squared.sqrt();
                let cosine_surface = V3::Y.dot(direction).max(0.0);
                let cosine_light = light.normal.dot(-direction).max(0.0);
                let diffuse = disney_diffuse_multiplier(V3::Y, direction, V3::Y, 1.0);
                integrated += light.radiance
                    * (diffuse * cosine_surface * cosine_light / (PI * distance_squared)
                        * light.area
                        / (quadrature_resolution * quadrature_resolution) as f32);
            }
        }
        let expected = integrated;
        let relative_error =
            |actual: f32, reference: f32| (actual - reference).abs() / reference.abs().max(1.0e-6);
        assert!(
            relative_error(estimate.0, expected.0) < 0.015,
            "red: estimate={}, expected={}",
            estimate.0,
            expected.0
        );
        assert!(
            relative_error(estimate.1, expected.1) < 0.015,
            "green: estimate={}, expected={}",
            estimate.1,
            expected.1
        );
        assert!(
            relative_error(estimate.2, expected.2) < 0.015,
            "blue: estimate={}, expected={}",
            estimate.2,
            expected.2
        );
    }
}

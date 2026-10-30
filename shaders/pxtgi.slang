// PXTGI: bounce light traced on screen, then against the physics colliders.
// Ported from the godot_physx module's PXTGI (physx_gi_effect.cpp) to world
// space: Bevy's prepass depth is reverse-Z infinite perspective and its
// deferred normals are world space. Light is kept as Godot's GI buffer holds
// it: the mean radiance arriving, cosine weighted (irradiance / pi), exposed.
// Compiled to WGSL by slangc at startup (src/pxtgi.rs).

static const float PI = 3.14159265;
static const int STEPS = 24;
static const int MAX_PROBES = 8;

// Debug views, as `PxtgiDebug` numbers them (1-5 as Godot's).
static const int DEBUG_LAST_FRAME = 1;
static const int DEBUG_ROUND_TRIP = 2;
static const int DEBUG_WORLD_ONLY = 3;
static const int DEBUG_WORLD_HITS = 4;
static const int DEBUG_WORLD_POSITION = 5;
static const int DEBUG_BOUNCE_LIGHT = 6;
static const int DEBUG_REFLECTIONS = 7;

// Reflection probe flags (probeBox.w bits).
static const int PROBE_OVERRIDES_AMBIENT = 1;
static const int PROBE_OVERRIDES_REFLECTIONS = 2;
static const int PROBE_PARALLAX = 4;

struct Frame {
    float4x4 clipFromWorld;
    float4x4 worldFromClip;
    float4x4 prevClipFromWorld;
    float4 camera;          // xyz position, w near plane
    uint4 sizes;            // xy frame, zw trace
    float4 misc;            // frame index, screen ray distance, intensity, debug
    float4 sunDirection;    // toward the sun; w 1 when there is one
    float4 sunColor;        // irradiance / pi, exposed; w exposure
    float4 sky;             // radiance a ray leaving the scene sees; w world ray distance
    uint4 counts;           // primitives, nodes, point lights, spot lights
    uint4 cache;            // frame, slot mask, on, rays per texel
    float4 extra;           // emissives, light sources just changed, history valid
    float4 reflection;      // mode (0 off, 1 screen, 2 and the colliders, 3 and probes), roughness cutoff
    uint4 reflectSize;      // xy the half-size reflection textures
    float4 env;             // environment map strength (exposed), roughest mip, present, lights the scene
    float4 envX;            // world-to-environment rotation rows
    float4 envY;
    float4 envZ;
    float4 omniPosition[8]; // w 1 / range^2
    float4 omniColor[8];
    float4 spotPosition[8]; // w 1 / range^2
    float4 spotDirection[8]; // w angular scale
    float4 spotColor[8];     // w angular offset
    float4 emitCenter[16];
    float4 emitX[16]; // half axes
    float4 emitY[16];
    float4 emitZ[16];
    float4 emitColor[16];
    float4 ambient;          // rgb the camera's ambient light (exposed radiance)
    uint4 probes;            // x reflection probes; y 1 when they are sources (Traced + Probes)
    // Per reflection probe (a Bevy LightProbe with an EnvironmentMapLight):
    // world to its unit cube [-0.5, 0.5] (rows, w translation); its unit cube
    // to world directions (rows); parallax half extents in the unit cube (w
    // flags); the core where it overrides PXTGI and its falloff ratios; the
    // capture point (w strength, exposed); the world-to-map rotation rows (w of
    // the first its roughest mip).
    float4 probeX[8];
    float4 probeY[8];
    float4 probeZ[8];
    float4 probeDirX[8];
    float4 probeDirY[8];
    float4 probeDirZ[8];
    float4 probeBox[8];
    float4 probeCore[8];
    float4 probeFalloff[8];
    float4 probeCapture[8];
    float4 probeRotX[8];
    float4 probeRotY[8];
    float4 probeRotZ[8];
};

// One collider: world-to-local rows; shape (x: 0 box, 1 sphere, 2 capsule,
// 3 cylinder along local y, 4 convex hull, 5 triangle mesh; yzw half extents /
// radius / radius and half height; a hull's first plane and plane count, a
// mesh's first tree node and first triangle in the geometry buffer, as int
// bits); its body's albedo and emitted radiance.
struct Primitive {
    float4 r0;
    float4 r1;
    float4 r2;
    float4 shape;
    float4 albedo;
    float4 emission;
};

// Two float4 per node: lo (w: right child, or first primitive of a leaf) and
// hi (w: 0 inner, else leaf count), ints as float bits; left child follows.
struct Node {
    float4 lo;
    float4 hi;
};

// Radiance cache cell: where it was hit (w: frame last used, as bits), normal
// (w: refinements), albedo, bounce light arriving, light leaving.
struct Cell {
    float4 pos;
    float4 normal;
    float4 albedo;
    float4 incoming;
    float4 outgoing;
};

[[vk::binding(0, 0)]] ConstantBuffer<Frame> f;
[[vk::binding(1, 0)]] DepthTexture2D depthTexture;
[[vk::binding(2, 0)]] Texture2D<uint4> gbuffer;
[[vk::binding(3, 0)]] Texture2D<float4> colorTexture; // last frame's lit image
[[vk::binding(4, 0)]] SamplerState linearSampler;
[[vk::binding(5, 0)]] StructuredBuffer<Primitive> prims;
[[vk::binding(6, 0)]] StructuredBuffer<Node> nodes;
[[vk::binding(7, 0)]] RWStructuredBuffer<Atomic<uint>> cacheKeys;
[[vk::binding(8, 0)]] RWStructuredBuffer<Cell> cells;
// Hull planes (normal, offset), mesh tree nodes (two each) and mesh triangles
// (three corners), shared by the colliders that point into it.
[[vk::binding(9, 0)]] StructuredBuffer<float4> geo;
// The camera's specular environment map: what rays leaving the scene see.
[[vk::binding(10, 0)]] TextureCube<float4> envMap;
// The reflection probes' specular maps.
[[vk::binding(11, 0)]] TextureCube<float4> probeMap0;
[[vk::binding(12, 0)]] TextureCube<float4> probeMap1;
[[vk::binding(13, 0)]] TextureCube<float4> probeMap2;
[[vk::binding(14, 0)]] TextureCube<float4> probeMap3;
[[vk::binding(15, 0)]] TextureCube<float4> probeMap4;
[[vk::binding(16, 0)]] TextureCube<float4> probeMap5;
[[vk::binding(17, 0)]] TextureCube<float4> probeMap6;
[[vk::binding(18, 0)]] TextureCube<float4> probeMap7;

// Per-pass inputs and outputs.
[[vk::binding(0, 1)]] Texture2D<float4> in0;
[[vk::binding(1, 1)]] Texture2D<float4> in1;
[[vk::binding(2, 1)]] Texture2D<float4> in2;
[[vk::binding(3, 1)]] Texture2D<float4> in3;
[[vk::binding(4, 1)]] [format("rgba16f")] WTexture2D<float4> out0;
[[vk::binding(5, 1)]] [format("r32f")] WTexture2D<float> out1;
[[vk::binding(6, 1)]] [format("rgba16f")] WTexture2D<float4> out2;
[[vk::binding(7, 1)]] [format("r32f")] WTexture2D<float> out3;
// The composite's inputs: filtered light and its linear depth, and the
// filtered reflections and their depth.
[[vk::binding(8, 1)]] Texture2D<float4> giLight;
[[vk::binding(9, 1)]] Texture2D<float4> giDepth;
[[vk::binding(10, 1)]] Texture2D<float4> reflectLight;
[[vk::binding(11, 1)]] Texture2D<float4> reflectDepth;

// ---------------------------------------------------------------- view helpers

float2 frameSize() { return float2(f.sizes.xy); }
int2 traceSize() { return int2(f.sizes.zw); }
int2 reflectSize() { return int2(f.reflectSize.xy); }

float linearDepth(float d) { return f.camera.w / d; }

// World point at continuous pixel position pc (texel + 0.5) and device depth.
float3 worldFromPixel(float2 pc, float depth) {
    float2 uv = pc / frameSize();
    float4 w = mul(f.worldFromClip, float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0));
    return w.xyz / w.w;
}

// Continuous pixel position and linear depth of world point x under m.
float3 project(float4x4 m, float3 x) {
    float4 c = mul(m, float4(x, 1.0));
    float2 ndc = c.xy / c.w;
    return float3((ndc.x * 0.5 + 0.5) * frameSize().x, (0.5 - ndc.y * 0.5) * frameSize().y, c.w);
}

bool onScreen(float2 pc) { return all(pc >= 0.0) && all(pc < frameSize()); }

float loadDepth(int2 p) { return depthTexture.Load(int3(p, 0)); }
uint4 loadGbuffer(int2 p) { return gbuffer.Load(int3(p, 0)); }

float surfaceDepth(float2 pc) {
    float d = loadDepth(int2(pc));
    return d > 0.0 ? linearDepth(d) : 1e9;
}

int2 framePixel(int2 px, int2 size) {
    return int2((float2(px) + 0.5) * frameSize() / float2(size));
}

float3 decodeNormal(float2 packed) {
    float2 p = packed * 2.0 - 1.0;
    float3 n = float3(p, 1.0 - abs(p.x) - abs(p.y));
    float t = saturate(-n.z);
    n.x += n.x >= 0.0 ? -t : t;
    n.y += n.y >= 0.0 ? -t : t;
    return normalize(n);
}

float3 gbufferNormal(uint4 g) {
    return decodeNormal(float2(float(g.w & 4095u), float((g.w >> 12u) & 4095u)) / 4095.0);
}

float3 gbufferBase(uint4 g) {
    float3 base = float3(float(g.x & 255u), float((g.x >> 8u) & 255u), float((g.x >> 16u) & 255u)) / 255.0;
    return pow(base, 2.2);
}

float gbufferMetallic(uint4 g) { return float((g.z >> 8u) & 255u) / 255.0; }
float3 gbufferAlbedo(uint4 g) { return gbufferBase(g) * (1.0 - gbufferMetallic(g)); }
float gbufferRoughness(uint4 g) { return float(g.x >> 24u) / 255.0; }

// Specular reflectance at normal incidence, as Bevy's PBR takes it.
float3 gbufferF0(uint4 g) {
    float reflectance = float(g.z & 255u) / 255.0;
    float metallic = gbufferMetallic(g);
    return 0.16 * reflectance * reflectance * (1.0 - metallic) + gbufferBase(g) * metallic;
}

// Split-sum environment BRDF, analytic fit (Karis, "Physically Based Shading
// on Mobile").
float3 envBrdf(float3 f0, float roughness, float nv) {
    float4 r = roughness * float4(-1.0, -0.0275, -0.572, 0.022) + float4(1.0, 0.0425, 1.04, -0.04);
    float a004 = min(r.x * r.x, exp2(-9.28 * nv)) * r.x + r.y;
    float2 ab = float2(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + ab.y;
}

// What a ray leaving the scene along d sees: the environment map (blurred to
// `level` of its roughest mip), or the sky colour. Cube maps are left-handed.
float3 backgroundLight(float3 d, float level) {
    if (f.env.z < 0.5) return f.sky.rgb;
    float3 e = float3(dot(f.envX.xyz, d), dot(f.envY.xyz, d), -dot(f.envZ.xyz, d));
    return envMap.SampleLevel(linearSampler, e, level * f.env.y).rgb * f.env.x;
}

// ---------------------------------------------------------------- reflection probes

// World point w in probe i's unit cube [-0.5, 0.5].
float3 probeLocal(int i, float3 w) {
    return float3(dot(f.probeX[i].xyz, w) + f.probeX[i].w, dot(f.probeY[i].xyz, w) + f.probeY[i].w,
        dot(f.probeZ[i].xyz, w) + f.probeZ[i].w);
}

// A unit-cube direction or offset of probe i in world space.
float3 probeWorldDir(int i, float3 l) {
    return float3(dot(f.probeDirX[i].xyz, l), dot(f.probeDirY[i].xyz, l), dot(f.probeDirZ[i].xyz, l));
}

// A world direction in probe i's unit cube.
float3 probeLocalDir(int i, float3 d) {
    return float3(dot(f.probeX[i].xyz, d), dot(f.probeY[i].xyz, d), dot(f.probeZ[i].xyz, d));
}

// Probe i's specular capture along world direction d (left-handed, as the
// environment map), at mip `level`.
float3 probeSample(int i, float3 d, float level) {
    float3 e = float3(dot(f.probeRotX[i].xyz, d), dot(f.probeRotY[i].xyz, d), -dot(f.probeRotZ[i].xyz, d));
    float3 c = 0.0;
    switch (i) {
    case 0: c = probeMap0.SampleLevel(linearSampler, e, level).rgb; break;
    case 1: c = probeMap1.SampleLevel(linearSampler, e, level).rgb; break;
    case 2: c = probeMap2.SampleLevel(linearSampler, e, level).rgb; break;
    case 3: c = probeMap3.SampleLevel(linearSampler, e, level).rgb; break;
    case 4: c = probeMap4.SampleLevel(linearSampler, e, level).rgb; break;
    case 5: c = probeMap5.SampleLevel(linearSampler, e, level).rgb; break;
    case 6: c = probeMap6.SampleLevel(linearSampler, e, level).rgb; break;
    default: c = probeMap7.SampleLevel(linearSampler, e, level).rgb; break;
    }
    return c * f.probeCapture[i].w;
}

// Whether a probe that takes over `what` (ambient or reflections) holds w in
// its core: there PXTGI leaves that light to Bevy's probe.
bool probeOverrides(float3 w, int what) {
    [loop] for (int i = 0; i < int(f.probes.x); i++) {
        if ((int(f.probeBox[i].w) & what) == 0) continue;
        if (all(abs(probeLocal(i, w)) <= f.probeCore[i].xyz)) return true;
    }
    return false;
}

// The smallest probe holding w (-1 for none), its unit-cube position and its
// influence there (falling off over the probe's falloff ratio, as Bevy's).
int probeAt(float3 w, out float3 l, out float weight) {
    int best = -1;
    float bestVolume = 1e30;
    l = 0.0;
    weight = 0.0;
    [loop] for (int i = 0; i < int(f.probes.x); i++) {
        float3 li = probeLocal(i, w);
        if (any(abs(li) > 0.5)) continue;
        // Its world volume, kept in the core's w.
        float volume = f.probeCore[i].w;
        if (volume < bestVolume) {
            best = i;
            bestVolume = volume;
            l = li;
        }
    }
    if (best >= 0) {
        float3 falloff = max(f.probeFalloff[best].xyz, 1e-4);
        float3 b = saturate((0.5 - abs(l)) / (0.5 * falloff));
        weight = b.x * b.y * b.z;
    }
    return best;
}

// Probe i's reflection along world direction r from world point w: the
// direction box-projected onto its parallax box when it has one.
float3 probeReflection(int i, float3 w, float3 l, float3 r, float rough) {
    float3 d = r;
    if ((int(f.probeBox[i].w) & PROBE_PARALLAX) != 0) {
        float3 dl = probeLocalDir(i, r);
        float3 e = f.probeBox[i].xyz;
        float3 inv = safeInv(dl);
        float3 farSide = max((e - l) * inv, (-e - l) * inv);
        float t = min(min(farSide.x, farSide.y), farSide.z);
        d = probeWorldDir(i, l + dl * t);
    }
    return probeSample(i, normalize(d), sqrt(rough) * f.probeRotX[i].w);
}

// What Bevy reflects at w along r: the probe it is in (blended over its
// falloff into the camera's environment map), else the environment map.
float3 bevyReflection(float3 w, float3 r, float rough) {
    float3 env = f.env.w > 0.5 ? backgroundLight(r, rough) : float3(0.0);
    float3 l;
    float weight;
    int i = probeAt(w, l, weight);
    if (i < 0) return env;
    return lerp(env, probeReflection(i, w, l, r, rough), weight);
}

// Off screen, a reflection probe's capture may have seen a static body's
// point itself (textures and all): its light there, when it did.
bool probeSource(float3 x, float3 n, int hit, out float3 seen) {
    seen = 0.0;
    if (prims[hit].emission.w < 0.5) return false;
    [loop] for (int i = 0; i < int(f.probes.x); i++) {
        if (any(abs(probeLocal(i, x)) > 0.5)) continue;
        float3 to = f.probeCapture[i].xyz - x;
        float dist = length(to);
        if (dist < 0.05 || dot(n, to) <= 0.0 || blocked(x + n * 0.02, to / dist, dist - 0.05)) continue;
        seen = probeSample(i, -to / dist, 0.0);
        return true;
    }
    return false;
}

// A texture read at continuous texel position t (texel centres at +0.5),
// blended from the four nearest texels as a linear sampler clamped to the
// edge would: unfilterable formats share the bindings.
float4 bilinear(Texture2D<float4> tex, float2 t, int2 size) {
    float2 c = t - 0.5;
    int2 b = int2(floor(c));
    float2 fr = c - float2(b);
    float4 a = tex.Load(int3(clamp(b, 0, size - 1), 0));
    float4 bx = tex.Load(int3(clamp(b + int2(1, 0), 0, size - 1), 0));
    float4 by = tex.Load(int3(clamp(b + int2(0, 1), 0, size - 1), 0));
    float4 bxy = tex.Load(int3(clamp(b + int2(1, 1), 0, size - 1), 0));
    return lerp(lerp(a, bx, fr.x), lerp(by, bxy, fr.x), fr.y);
}

// Lit surfaces only: Bevy's unlit flag is gbuffer bit 24.
bool receives(float depth, uint4 g) { return depth > 0.0 && (g.w & (1u << 24u)) == 0u; }

float luminance(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

// One very bright hit (a glint on last frame's image) would survive the filter
// as a flickering blob; cap it far above any lit surface.
float3 clampFirefly(float3 c) { return c * min(1.0, 4.0 / max(luminance(c), 1e-4)); }

bool bad(float4 v) { return any(v != v) || any(abs(v) > 1e30); }

float3 safeInv(float3 d) { return 1.0 / select(abs(d) < 1e-20, float3(1e-20), d); }

// A march step found the ray behind the depth buffer: bisect the crossing so
// the hit doesn't glitter from texel to texel and frame to frame.
float refineHit(float3 origin, float3 dir, float inFront, float behind) {
    [loop] for (int i = 0; i < 6; i++) {
        float mid = 0.5 * (inFront + behind);
        float3 q = project(f.clipFromWorld, origin + dir * mid);
        float2 qp = clamp(q.xy, 0.0, frameSize() - 1.0);
        if (q.z > surfaceDepth(qp)) {
            behind = mid;
        } else {
            inFront = mid;
        }
    }
    return behind;
}

void tangentFrame(float3 n, out float3 t, out float3 b) {
    t = normalize(cross(n, abs(n.y) < 0.99 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0)));
    b = cross(n, t);
}

float3 aroundNormal(float3 n, float3 v) {
    float3 t, b;
    tangentFrame(n, t, b);
    return normalize(t * v.x + b * v.y + n * v.z);
}

float3 cosineDir(float3 n, float u, float phi) {
    return aroundNormal(n, float3(sqrt(u) * cos(phi), sqrt(u) * sin(phi), sqrt(max(0.0, 1.0 - u))));
}

// Interleaved gradient noise, stepped along an R2 sequence per frame and ray.
float ign(float2 p) { return frac(52.9829189 * frac(dot(p, float2(0.06711056, 0.00583715)))); }

// The a-trous kernel's taps, by distance from the centre.
float kernelTap(int i) { return i == 0 ? 0.375 : (i == 1 ? 0.25 : 0.0625); }

// ---------------------------------------------------------------- collider trace

// Where the ray enters a box (1e30: misses it, or only beyond best).
float boxEntry(float3 o, float3 inv, float3 lo, float3 hi, float best) {
    float3 t0 = (lo - o) * inv;
    float3 t1 = (hi - o) * inv;
    float3 nearT = min(t0, t1);
    float3 farT = max(t0, t1);
    float enter = max(max(nearT.x, nearT.y), max(nearT.z, 0.0));
    float leave = min(min(farT.x, farT.y), min(farT.z, best));
    return enter <= leave ? enter : 1e30;
}

float meshBoxEntry(float3 o, float3 inv, int at, float best) {
    return boxEntry(o, inv, geo[at].xyz, geo[at + 1].xyz, best);
}

// The nearest triangle of a mesh along a local ray (either side), walking its
// own tree near child first. Box3D's trees stay shallow; a deeper one only
// drops the farthest branches.
bool hitMesh(int nodesAt, int tris, float3 o, float3 d, inout float t, inout float3 n) {
    bool found = false;
    float3 inv = safeInv(d);
    if (meshBoxEntry(o, inv, nodesAt, t) >= 1e30) return false;
    int stack[24];
    int top = 0;
    int node = 0;
    [loop] while (true) {
        float4 lo = geo[nodesAt + node * 2];
        float4 hi = geo[nodesAt + node * 2 + 1];
        int count = asint(hi.w);
        if (count == 0) {
            int left = node + 1;
            int right = asint(lo.w);
            float tl = meshBoxEntry(o, inv, nodesAt + left * 2, t);
            float tr = meshBoxEntry(o, inv, nodesAt + right * 2, t);
            if (tl < 1e30 && tr < 1e30) {
                bool leftFirst = tl <= tr;
                if (top < 24) {
                    stack[top] = leftFirst ? right : left;
                    top++;
                }
                node = leftFirst ? left : right;
                continue;
            } else if (tl < 1e30) {
                node = left;
                continue;
            } else if (tr < 1e30) {
                node = right;
                continue;
            }
        } else {
            int first = asint(lo.w);
            [loop] for (int k = first; k < first + count; k++) {
                float3 a = geo[tris + k * 3].xyz;
                float3 e1 = geo[tris + k * 3 + 1].xyz - a;
                float3 e2 = geo[tris + k * 3 + 2].xyz - a;
                float3 pv = cross(d, e2);
                float det = dot(e1, pv);
                if (abs(det) < 1e-12) continue;
                float invDet = 1.0 / det;
                float3 tv = o - a;
                float u = dot(tv, pv) * invDet;
                if (u < 0.0 || u > 1.0) continue;
                float3 qv = cross(tv, e1);
                float w = dot(d, qv) * invDet;
                if (w < 0.0 || u + w > 1.0) continue;
                float th = dot(e2, qv) * invDet;
                if (th > 0.0 && th < t) {
                    t = th;
                    n = cross(e1, e2);
                    found = true;
                }
            }
        }
        bool next = false;
        [loop] while (top > 0) {
            top--;
            int s = stack[top];
            if (meshBoxEntry(o, inv, nodesAt + s * 2, t) < 1e30) {
                node = s;
                next = true;
                break;
            }
        }
        if (!next) break;
    }
    // Facing the ray: a mesh is hit from either side.
    if (found && dot(n, d) > 0.0) n = -n;
    return found;
}

// The nearest hit of one collider closer than best; its world normal.
bool hitPrimitive(Primitive q, float3 o, float3 d, inout float best, out float3 normal) {
    normal = float3(0.0);
    float3 lo = float3(dot(q.r0.xyz, o) + q.r0.w, dot(q.r1.xyz, o) + q.r1.w, dot(q.r2.xyz, o) + q.r2.w);
    float3 ld = float3(dot(q.r0.xyz, d), dot(q.r1.xyz, d), dot(q.r2.xyz, d));
    float3 n = float3(0.0);
    float t = 0.0;
    int kind = int(q.shape.x + 0.5);
    if (kind == 4) {
        // Convex hull: the ray's span inside every face plane's half space.
        int first = asint(q.shape.y);
        int count = asint(q.shape.z);
        float tIn = -1e30;
        float tOut = best;
        [loop] for (int i = first; i < first + count; i++) {
            float4 plane = geo[i];
            float denom = dot(plane.xyz, ld);
            float dist = dot(plane.xyz, lo) - plane.w;
            if (abs(denom) < 1e-12) {
                if (dist > 0.0) return false;
                continue;
            }
            float th = -dist / denom;
            if (denom < 0.0) {
                if (th > tIn) {
                    tIn = th;
                    n = plane.xyz;
                }
            } else {
                tOut = min(tOut, th);
            }
        }
        if (tIn > tOut || tIn <= 0.0 || tIn >= best) return false;
        t = tIn;
    } else if (kind == 5) {
        t = best;
        if (!hitMesh(asint(q.shape.y), asint(q.shape.z), lo, ld, t, n)) return false;
    } else if (kind == 2 || kind == 3) {
        // Capsule / cylinder along local y: the infinite cylinder, then the caps.
        float r = q.shape.y;
        float halfLen = kind == 2 ? max(q.shape.z - r, 0.0) : q.shape.z;
        float3 pa = float3(0.0, -halfLen, 0.0);
        float3 ba = float3(0.0, 2.0 * halfLen, 0.0);
        float3 oa = lo - pa;
        float baba = dot(ba, ba);
        float bard = dot(ba, ld);
        float baoa = dot(ba, oa);
        float a = baba * dot(ld, ld) - bard * bard;
        float b = baba * dot(oa, ld) - baoa * bard;
        float c = baba * dot(oa, oa) - baoa * baoa - r * r * baba;
        float h = b * b - a * c;
        if (h < 0.0) return false;
        h = sqrt(h);
        t = (-b - h) / a;
        float y = baoa + t * bard;
        if (y > 0.0 && y < baba) {
            n = (oa + ld * t - ba * (y / baba)) / r;
        } else if (kind == 2) {
            float3 oc = y <= 0.0 ? oa : lo - (pa + ba);
            float qa = dot(ld, ld);
            float qb = dot(oc, ld);
            float qc = dot(oc, oc) - r * r;
            float qh = qb * qb - qa * qc;
            if (qh < 0.0) return false;
            t = (-qb - sqrt(qh)) / qa;
            n = oc + ld * t;
        } else {
            t = ((y < 0.0 ? 0.0 : baba) - baoa) / bard;
            if (abs(b + a * t) >= h) return false;
            n = float3(0.0, y < 0.0 ? -1.0 : 1.0, 0.0);
        }
        if (t <= 0.0 || t >= best) return false;
    } else if (kind == 0) {
        float3 inv = safeInv(ld);
        float3 t0 = (-q.shape.yzw - lo) * inv;
        float3 t1 = (q.shape.yzw - lo) * inv;
        float3 loT = min(t0, t1);
        float3 hiT = max(t0, t1);
        t = max(max(loT.x, loT.y), loT.z);
        float outT = min(min(hiT.x, hiT.y), hiT.z);
        if (t > outT || t <= 0.0 || t >= best) return false;
        if (t == loT.x) {
            n = float3(-sign(ld.x), 0.0, 0.0);
        } else if (t == loT.y) {
            n = float3(0.0, -sign(ld.y), 0.0);
        } else {
            n = float3(0.0, 0.0, -sign(ld.z));
        }
    } else {
        float a = dot(ld, ld);
        float b = dot(lo, ld);
        float c = dot(lo, lo) - q.shape.y * q.shape.y;
        float h = b * b - a * c;
        if (h < 0.0) return false;
        t = (-b - sqrt(h)) / a;
        if (t <= 0.0 || t >= best) return false;
        n = lo + ld * t;
    }
    best = t;
    // Local to world for a normal: the transpose of the world-to-local rows.
    normal = normalize(n.x * q.r0.xyz + n.y * q.r1.xyz + n.z * q.r2.xyz);
    return true;
}

// The nearest collider along the ray, near child first; with anyHit the first
// one in the way (shadow rays). Returns its index, or -1.
int walkWorld(float3 o, float3 d, inout float t, out float3 normal, bool anyHit) {
    normal = float3(0.0);
    int hit = -1;
    if (f.counts.y == 0u) return hit;
    float3 inv = safeInv(d);
    if (boxEntry(o, inv, nodes[0].lo.xyz, nodes[0].hi.xyz, t) >= 1e30) return hit;
    int stack[16];
    int top = 0;
    int node = 0;
    [loop] while (true) {
        Node nd = nodes[node];
        int count = asint(nd.hi.w);
        if (count == 0) {
            int left = node + 1;
            int right = asint(nd.lo.w);
            float tl = boxEntry(o, inv, nodes[left].lo.xyz, nodes[left].hi.xyz, t);
            float tr = boxEntry(o, inv, nodes[right].lo.xyz, nodes[right].hi.xyz, t);
            if (tl < 1e30 && tr < 1e30) {
                bool leftFirst = tl <= tr;
                if (top < 16) {
                    stack[top] = leftFirst ? right : left;
                    top++;
                }
                node = leftFirst ? left : right;
                continue;
            } else if (tl < 1e30) {
                node = left;
                continue;
            } else if (tr < 1e30) {
                node = right;
                continue;
            }
        } else {
            int first = asint(nd.lo.w);
            [loop] for (int i = first; i < first + count; i++) {
                float3 n;
                if (hitPrimitive(prims[i], o, d, t, n)) {
                    hit = i;
                    normal = n;
                    if (anyHit) return hit;
                }
            }
        }
        // The next waiting node that could still hold something nearer.
        bool found = false;
        [loop] while (top > 0) {
            top--;
            int s = stack[top];
            if (boxEntry(o, inv, nodes[s].lo.xyz, nodes[s].hi.xyz, t) < 1e30) {
                node = s;
                found = true;
                break;
            }
        }
        if (!found) break;
    }
    return hit;
}

bool blocked(float3 o, float3 d, float reach) {
    float t = reach;
    float3 n;
    return walkWorld(o, d, t, n, true) >= 0;
}

// A ray that hit nothing within reach has left the scene only if it also left
// the box around every collider (a long tunnel doesn't see the sky).
bool escapes(float3 o, float3 d, float reach) {
    if (f.counts.y == 0u) return true;
    float3 inv = safeInv(d);
    float3 farT = max((nodes[0].lo.xyz - o) * inv, (nodes[0].hi.xyz - o) * inv);
    return min(min(farT.x, farT.y), farT.z) <= reach;
}

// ---------------------------------------------------------------- radiance cache

uint hashU(uint x) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

// Cells a quarter metre across near the camera, doubling with each doubling
// of distance past 8 m.
uint cacheLevel(float3 x) {
    float dist = length(x - f.camera.xyz);
    return uint(clamp(floor(log2(max(dist, 1.0) / 8.0)) + 1.0, 0.0, 5.0));
}

uint facing(float3 n) {
    float3 a = abs(n);
    if (a.x > a.y && a.x > a.z) return n.x > 0.0 ? 0u : 1u;
    if (a.y > a.z) return n.y > 0.0 ? 2u : 3u;
    return n.z > 0.0 ? 4u : 5u;
}

uint cellHash(int3 cell, uint side, uint level) {
    return hashU(uint(cell.x) + hashU(uint(cell.y) + hashU(uint(cell.z) + hashU(side * 8u + level))));
}

// The slot of x's cell, claiming an empty one when asked; -1 when the table
// is full there.
int cacheSlot(float3 x, float3 n, bool claim, out bool claimed) {
    claimed = false;
    uint level = cacheLevel(x);
    int3 cell = int3(floor(x / (0.25 * float(1u << level))));
    uint h = cellHash(cell, facing(n), level);
    uint key = hashU(h ^ 0x9e3779b9u) | 1u;
    [loop] for (uint i = 0u; i < 8u; i++) {
        uint slot = (h + i) & f.cache.y;
        uint was = cacheKeys[slot].load();
        if (was == key) return int(slot);
        if (was == 0u) {
            if (!claim) return -1;
            uint previous = cacheKeys[slot].compareExchange(0u, key);
            if (previous == 0u) {
                claimed = true;
                return int(slot);
            }
            if (previous == key) return int(slot);
        }
    }
    return -1;
}

// The bounce light arriving at x (or with leaving, the light leaving it),
// blended from the eight refined cells around it as a texture is filtered: a
// cell's own value would show as a square patch in a reflection.
float3 smoothCache(float3 x, float3 n, bool leaving) {
    uint level = cacheLevel(x);
    float3 c = x / (0.25 * float(1u << level)) - 0.5;
    int3 base = int3(floor(c));
    float3 t = c - float3(base);
    uint side = facing(n);
    float4 sum = 0.0;
    [loop] for (int i = 0; i < 8; i++) {
        int3 o = int3(i & 1, (i >> 1) & 1, i >> 2);
        uint h = cellHash(base + o, side, level);
        uint key = hashU(h ^ 0x9e3779b9u) | 1u;
        [loop] for (uint j = 0u; j < 8u; j++) {
            uint slot = (h + j) & f.cache.y;
            uint was = cacheKeys[slot].load();
            if (was == key) {
                if (cells[slot].normal.w > 0.5) {
                    float3 w3 = lerp(1.0 - t, t, float3(o));
                    float w = w3.x * w3.y * w3.z;
                    float3 light = leaving ? cells[slot].outgoing.rgb : cells[slot].incoming.rgb;
                    sum += float4(light * w, w);
                }
                break;
            }
            if (was == 0u) break;
        }
    }
    return sum.w > 1e-4 ? sum.rgb / sum.w : float3(0.0);
}

// ---------------------------------------------------------------- lighting

// Irradiance / pi at a surface patch straight from the camera's ambient light,
// the sun, point and spot lights, shadowed by the colliders.
float3 directLight(float3 x, float3 n) {
    float3 light = f.ambient.rgb;
    float3 o = x + n * 0.02;
    float lit = dot(n, f.sunDirection.xyz);
    if (lit > 0.0 && f.sunDirection.w > 0.5 && !blocked(o, f.sunDirection.xyz, 1e6)) {
        light += f.sunColor.rgb * lit;
    }
    // Bevy's falloff: inverse square, windowed to zero at the range.
    [loop] for (uint i = 0u; i < f.counts.z; i++) {
        float3 to = f.omniPosition[i].xyz - x;
        float d2 = max(dot(to, to), 1e-4);
        float3 l = to * rsqrt(d2);
        float nl = dot(n, l);
        float w = saturate(1.0 - pow(d2 * f.omniPosition[i].w, 2.0));
        if (nl <= 0.0 || w <= 0.0 || blocked(o, l, sqrt(d2) - 0.05)) continue;
        light += f.omniColor[i].rgb * nl * w * w / d2;
    }
    [loop] for (uint i = 0u; i < f.counts.w; i++) {
        float3 to = f.spotPosition[i].xyz - x;
        float d2 = max(dot(to, to), 1e-4);
        float3 l = to * rsqrt(d2);
        float nl = dot(n, l);
        float w = saturate(1.0 - pow(d2 * f.spotPosition[i].w, 2.0));
        float cone = saturate(dot(-l, f.spotDirection[i].xyz) * f.spotDirection[i].w + f.spotColor[i].w);
        if (nl <= 0.0 || w <= 0.0 || cone <= 0.0 || blocked(o, l, sqrt(d2) - 0.05)) continue;
        light += f.spotColor[i].rgb * nl * w * w * cone * cone / d2;
    }
    return light;
}

// How a hit is seen: by a diffuse ray, a glossy reflection, or a mirror.
static const uint SEEN_DIFFUSE = 0u;
static const uint SEEN_GLOSSY = 1u;
static const uint SEEN_MIRROR = 2u;

// Set by the reflection pass for texels whose off-screen hits may take a
// reflection probe's capture (Traced + Probes).
static bool probeTexture = false;

// The light leaving a collider's surface toward the ray: last frame's image
// where that point was on it; off screen, for a reflection, a probe's capture
// of a static body when it saw the point; otherwise its glow, plus its cache
// cell once refined, or else the direct light on its albedo. A pixel's ray
// claims the cell and keeps it alive. A reflection shows the point itself, so
// it takes the cells around it blended; a mirror's lights the point directly
// and takes only their bounce light (lit directly, a glossy surface's
// scattered rays would flicker between light and shadow).
float3 worldLight(float3 x, float3 n, int hit, bool fromPixels, bool withGlow, uint seenAs) {
    // Bevy weights a material's emission by the exposure by its
    // `emissive_exposure_weight` (albedo.w).
    float3 glowAll = prims[hit].emission.rgb * lerp(1.0, f.sunColor.w, prims[hit].albedo.w);
    float3 s = project(f.clipFromWorld, x);
    if (s.z > f.camera.w && onScreen(s.xy) && abs(surfaceDepth(s.xy) - s.z) < 0.1 + 0.02 * s.z) {
        float3 pp = project(f.prevClipFromWorld, x);
        if (onScreen(pp.xy)) {
            float3 seen = colorTexture.SampleLevel(linearSampler, pp.xy / frameSize(), 0.0).rgb;
            return withGlow ? seen : max(seen - glowAll, 0.0);
        }
    }
    if (seenAs != SEEN_DIFFUSE && probeTexture) {
        float3 seen;
        if (probeSource(x, n, hit, seen)) return seen;
    }
    float3 albedo = prims[hit].albedo.rgb;
    float3 glow = withGlow ? glowAll : float3(0.0);
    if (f.cache.z != 0u) {
        bool claimed;
        int slot = cacheSlot(x, n, fromPixels, claimed);
        if (slot >= 0) {
            if (claimed) {
                cells[slot].pos = float4(x, asfloat(f.cache.x));
                cells[slot].normal = float4(n, 0.0);
                cells[slot].albedo = float4(albedo, 0.0);
                cells[slot].incoming = 0.0;
            } else {
                if (fromPixels) cells[slot].pos.w = asfloat(f.cache.x);
                if (cells[slot].normal.w > 0.5) {
                    if (seenAs == SEEN_DIFFUSE) return glow + cells[slot].outgoing.rgb;
                    if (seenAs == SEEN_GLOSSY) return glow + smoothCache(x, n, true);
                    return glow + albedo * (directLight(x, n) + smoothCache(x, n, false));
                }
            }
        }
    }
    return glow + albedo * directLight(x, n);
}

// The emissive whose box holds world point w (a little grown), or -1.
int emissiveAt(float3 w) {
    [loop] for (int i = 0; i < int(f.extra.x); i++) {
        float3 d = w - f.emitCenter[i].xyz;
        float3 ax = f.emitX[i].xyz;
        float3 ay = f.emitY[i].xyz;
        float3 az = f.emitZ[i].xyz;
        if (abs(dot(d, ax)) <= dot(ax, ax) + 0.03 * length(ax)
            && abs(dot(d, ay)) <= dot(ay, ay) + 0.03 * length(ay)
            && abs(dot(d, az)) <= dot(az, az) + 0.03 * length(az)) {
            return i;
        }
    }
    return -1;
}

// One emissive (picked uniformly), a point on its box's faces toward x
// (picked by area), a shadow ray: its light at x as irradiance / pi.
float3 sampleEmissive(float3 x, float3 n, float2 xi, float pick, int count) {
    int i = min(int(pick * float(count)), count - 1);
    float3 c = f.emitCenter[i].xyz;
    float3 ax = f.emitX[i].xyz;
    float3 ay = f.emitY[i].xyz;
    float3 az = f.emitZ[i].xyz;
    float a0 = length(ay) * length(az);
    float a1 = length(ax) * length(az);
    float a2 = length(ax) * length(ay);
    float total = a0 + a1 + a2;
    if (total <= 0.0) return 0.0;
    float r = xi.x * total;
    float3 face = az;
    float3 u = ax;
    float3 v = ay;
    if (r < a0) {
        face = ax;
        u = ay;
        v = az;
        xi.x = r / a0;
    } else if (r < a0 + a1) {
        face = ay;
        u = ax;
        v = az;
        xi.x = (r - a0) / a1;
    } else {
        xi.x = (r - a0 - a1) / a2;
    }
    float3 faceN = normalize(face);
    if (dot(x - c, faceN) < 0.0) {
        face = -face;
        faceN = -faceN;
    }
    float3 to = c + face + u * (2.0 * xi.x - 1.0) + v * (2.0 * xi.y - 1.0) - x;
    float d2 = max(dot(to, to), 1e-2);
    float dist = sqrt(d2);
    float3 l = to / dist;
    float cosX = dot(n, l);
    float cosE = dot(faceN, -l);
    if (cosX <= 0.0 || cosE <= 0.0 || blocked(x + n * 0.03, l, dist - 0.05)) return 0.0;
    // pdf over the facing faces' area: 1 / (4 * total); one emissive of count.
    float3 e = f.emitColor[i].rgb * cosX * cosE * (4.0 * total) / d2 * float(count) / PI;
    return e * min(1.0, 8.0 / max(luminance(e), 1e-4));
}

// Where the last march's hit was, in world space.
static float3 marchHit;

// A ray marched against the depth buffer in `steps` steps growing to reach;
// on a hit, the light that point had in last frame's image (w 1).
float4 march(float3 origin, float3 dir, float reach, int steps, float jitter) {
    float last = 0.0;
    [loop] for (int s = 0; s < steps; s++) {
        float along = reach * pow((float(s) + jitter) / float(steps), 2.0);
        float inFront = last;
        last = along;
        // How far this step went: a surface thinner than that could be stepped over.
        float stride = reach * 2.0 * (float(s) + jitter + 1.0) / float(steps * steps);
        float3 q = project(f.clipFromWorld, origin + dir * along);
        if (q.z < f.camera.w || !onScreen(q.xy)) break;
        float sd = loadDepth(int2(q.xy));
        if (sd <= 0.0) continue;
        float surface = linearDepth(sd);
        float behind = q.z - surface;
        if (behind > 0.0 && behind < max(0.1 + 0.02 * surface, 1.5 * stride)) {
            float3 hq = project(f.clipFromWorld, origin + dir * refineHit(origin, dir, inFront, along));
            float2 hp = clamp(hq.xy, 0.0, frameSize() - 1.0);
            float hd = loadDepth(int2(hp));
            marchHit = worldFromPixel(floor(hp) + 0.5, hd > 0.0 ? hd : sd);
            float3 pp = project(f.prevClipFromWorld, marchHit);
            if (onScreen(pp.xy)) {
                return float4(colorTexture.SampleLevel(linearSampler, pp.xy / frameSize(), 0.0).rgb, 1.0);
            }
            break;
        }
    }
    return 0.0;
}

// ---------------------------------------------------------------- passes

// A quarter of the cache slots a frame, one ray each: the cell's arriving
// light as a running mean, and the light leaving it.
[shader("compute")]
[numthreads(8, 8, 1)]
void cache_update(uint3 id : SV_DispatchThreadID) {
    uint slot = (id.y * 512u + id.x) * 4u + (f.cache.x & 3u);
    if (slot > f.cache.y || cacheKeys[slot].load() == 0u) return;
    Cell cell = cells[slot];
    if (f.cache.x - asuint(cell.pos.w) > 600u) {
        cacheKeys[slot].store(0u);
        return;
    }
    float3 n = normalize(cell.normal.xyz);
    float u = frac(float(hashU(slot) & 0xffffu) / 65536.0 + float(f.cache.x) * 0.7548776662);
    float phi = 6.2831853 * frac(float(hashU(slot ^ 0x5bd1e995u) & 0xffffu) / 65536.0 + float(f.cache.x) * 0.5698402910);
    float3 dir = cosineDir(n, u, phi);
    float3 origin = cell.pos.xyz + n * 0.03;
    float3 arriving = 0.0;
    float far = f.sky.w;
    float3 wn;
    int hit = walkWorld(origin, dir, far, wn, false);
    if (hit >= 0) {
        arriving = clampFirefly(worldLight(origin + dir * far, wn, hit, false, true, SEEN_DIFFUSE));
    } else if (escapes(origin, dir, f.sky.w)) {
        arriving = backgroundLight(dir, 1.0);
    }
    // The last 32 samples; while the light sources just changed, the last 4.
    float samples = min(cell.normal.w + 1.0, f.extra.y > 0.0 ? 4.0 : 32.0);
    float3 incoming = lerp(cell.incoming.rgb, arriving, 1.0 / samples);
    if (any(incoming != incoming)) incoming = 0.0;
    cells[slot].incoming = float4(incoming, 0.0);
    cells[slot].normal.w = samples;
    cells[slot].outgoing = float4(cell.albedo.rgb * (directLight(cell.pos.xyz, n) + incoming), 0.0);
}

// A few cosine-weighted rays per texel, marched against the depth buffer (a
// hit takes that point's light on last frame's image), continuing against the
// colliders. Writes the hits' light / rays and the fraction that hit.
[shader("compute")]
[numthreads(8, 8, 1)]
void trace(uint3 id : SV_DispatchThreadID) {
    int2 px = int2(id.xy);
    if (any(px >= traceSize())) return;
    int2 fp = framePixel(px, traceSize());
    float depth = loadDepth(fp);
    uint4 g = loadGbuffer(fp);
    if (!receives(depth, g)) {
        out0.Store(px, 0.0);
        return;
    }
    float3 p = worldFromPixel(float2(fp) + 0.5, depth);
    // A reflection probe that takes over the ambient light lights this point.
    if (probeOverrides(p, PROBE_OVERRIDES_AMBIENT)) {
        out0.Store(px, 0.0);
        return;
    }
    float3 n = gbufferNormal(g);
    float2 pf = float2(px);
    float3 noise = float3(ign(pf), ign(pf + float2(47.0, 17.0)), ign(pf + float2(13.0, 101.0)));
    int emissives = int(f.extra.x);
    int rays = clamp(int(f.cache.w), 1, 4);
    // Godot's debug views 3 to 5 show the colliders' light alone: no screen march.
    int debug = int(f.misc.w);
    int steps = debug >= DEBUG_WORLD_ONLY && debug <= DEBUG_WORLD_POSITION ? 0 : STEPS;
    float3 light = 0.0;
    float hits = 0.0;
    float3 origin = p + n * (0.02 + 0.002 * linearDepth(depth));
    [loop] for (int r = 0; r < rays; r++) {
        float index = f.misc.x * float(rays) + float(r);
        float u = frac(noise.x + index * 0.7548776662);
        float phi = 6.2831853 * frac(noise.y + index * 0.5698402910);
        float3 dir = cosineDir(n, u, phi);
        float jitter = frac(noise.z + index * 0.6180339887);
        float4 seen = march(origin, dir, f.misc.y, steps, jitter);
        bool found = seen.w > 0.5;
        if (found) {
            float3 lightSeen = seen.rgb;
            if (emissives > 0) {
                // Sampled directly below: don't count its glow twice.
                int e = emissiveAt(marchHit);
                if (e >= 0) lightSeen = max(lightSeen - f.emitColor[e].rgb, 0.0);
            }
            light += clampFirefly(lightSeen);
            hits += 1.0;
        }
        if (!found && f.counts.x > 0u) {
            // Off the screen, or past what it shows: the colliders.
            float t = f.sky.w;
            float3 wn;
            int hit = walkWorld(origin, dir, t, wn, false);
            if (hit >= 0) {
                light += clampFirefly(worldLight(origin + dir * t, wn, hit, true, emissives == 0, SEEN_DIFFUSE));
                hits += 1.0;
            }
        }
    }
    float4 result = float4(light, hits) / float(rays);
    if (emissives > 0) {
        float2 xi = float2(frac(noise.x + 0.37 + f.misc.x * 0.7548776662), frac(noise.y + 0.71 + f.misc.x * 0.5698402910));
        float pick = frac(noise.z + 0.13 + f.misc.x * 0.6180339887);
        result.rgb += sampleEmissive(p, n, xi, pick, emissives);
    }
    // One bad sample would otherwise live on in the accumulation forever.
    out0.Store(px, bad(result) ? float4(0.0) : result);
}

// This frame's rays blended into what the same surface accumulated before
// (reprojected onto last frame; a different depth there starts over).
// in0 trace, in1 previous light, in2 previous depth, in3 previous moments;
// out0 light, out1 linear depth, out2 moments, out3 variance.
[shader("compute")]
[numthreads(8, 8, 1)]
void temporal(uint3 id : SV_DispatchThreadID) {
    int2 px = int2(id.xy);
    int2 size = traceSize();
    if (any(px >= size)) return;
    int2 fp = framePixel(px, size);
    float depth = loadDepth(fp);
    if (!receives(depth, loadGbuffer(fp))) {
        out0.Store(px, 0.0);
        out1.Store(px, 0.0);
        out2.Store(px, 0.0);
        out3.Store(px, 0.0);
        return;
    }
    float3 p = worldFromPixel(float2(fp) + 0.5, depth);
    float4 now = in0.Load(int3(px, 0));
    float l = luminance(now.rgb);
    float4 result = now;
    float3 m = float3(l, l * l, 1.0); // mean, mean of the square, frames of history
    if (f.extra.z > 0.5) {
        float3 pp = project(f.prevClipFromWorld, p);
        if (onScreen(pp.xy)) {
            // Depth nearest, light and moments bilinear, as Godot samples them.
            float2 at = pp.xy * float2(size) / frameSize();
            int2 tp = clamp(int2(at), 0, size - 1);
            float was = in2.Load(int3(tp, 0)).r;
            if (abs(was - pp.z) < 0.05 * pp.z) {
                float4 wasLight = bilinear(in1, at, size);
                float4 wasM = bilinear(in3, at, size);
                if (!bad(wasLight) && !bad(wasM)) {
                    // A surface just uncovered takes its first frames at full weight.
                    float frames = min(wasM.z + 1.0, 32.0);
                    float alpha = max(1.0 / frames, 0.05);
                    result = lerp(wasLight, now, alpha);
                    m = float3(lerp(wasM.xy, float2(l, l * l), alpha), frames);
                }
            }
        }
    }
    // A young history hasn't seen enough frames to tell: taken as noisy.
    float variance = max(m.y - m.x * m.x, 0.0);
    if (m.z < 4.0) variance = max(variance, 0.25 * m.x * m.x + 0.01);
    out0.Store(px, result);
    out1.Store(px, linearDepth(depth));
    out2.Store(px, float4(m, 0.0));
    out3.Store(px, variance);
}

// One pass of an edge-aware a-trous filter: neighbours on the same plane and
// facing the same way, averaged more where still noisy.
// in0 light, in1 variance, in2 linear depth; out0 light, out1 variance.
void atrous(int2 px, int step) {
    int2 size = traceSize();
    if (any(px >= size)) return;
    float4 c = in0.Load(int3(px, 0));
    float v = in1.Load(int3(px, 0)).r;
    float z = in2.Load(int3(px, 0)).r;
    if (z <= 0.0) {
        out0.Store(px, c);
        out1.Store(px, v);
        return;
    }
    int2 fp = framePixel(px, size);
    float3 pos = worldFromPixel(float2(fp) + 0.5, f.camera.w / z);
    float3 n = gbufferNormal(loadGbuffer(fp));
    float lc = luminance(c.rgb);
    float sigma = 4.0 * sqrt(v) + 1e-3;
    float4 sum = 0.0;
    float weights = 0.0;
    float varSum = 0.0;
    [loop] for (int y = -2; y <= 2; y++) {
        [loop] for (int x = -2; x <= 2; x++) {
            int2 q = px + int2(x, y) * step;
            if (any(q < 0) || any(q >= size)) continue;
            float zq = in2.Load(int3(q, 0)).r;
            if (zq <= 0.0) continue;
            int2 fq = framePixel(q, size);
            float3 nq = gbufferNormal(loadGbuffer(fq));
            float4 cq = in0.Load(int3(q, 0));
            float plane = abs(dot(n, worldFromPixel(float2(fq) + 0.5, f.camera.w / zq) - pos));
            float w = kernelTap(abs(x)) * kernelTap(abs(y));
            w *= exp(-plane / (0.02 + 0.01 * z));
            w *= pow(max(dot(n, nq), 0.0), 32.0);
            w *= exp(-abs(luminance(cq.rgb) - lc) / sigma);
            sum += cq * w;
            weights += w;
            varSum += in1.Load(int3(q, 0)).r * w * w;
        }
    }
    out0.Store(px, weights > 1e-6 ? sum / weights : c);
    out1.Store(px, weights > 1e-6 ? varSum / (weights * weights) : v);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void atrous1(uint3 id : SV_DispatchThreadID) { atrous(int2(id.xy), 1); }

[shader("compute")]
[numthreads(8, 8, 1)]
void atrous2(uint3 id : SV_DispatchThreadID) { atrous(int2(id.xy), 2); }

[shader("compute")]
[numthreads(8, 8, 1)]
void atrous4(uint3 id : SV_DispatchThreadID) { atrous(int2(id.xy), 4); }

// One reflection ray per half-size texel, the view mirrored about a GGX
// microfacet normal (a mirror's rays stay together, a rough surface's spread),
// marched on screen, then against the colliders. Surfaces rougher than the
// cutoff, and rays that find nothing, keep Bevy's own reflection.
// out0: the light found, w 1 when found.
[shader("compute")]
[numthreads(8, 8, 1)]
void reflect_trace(uint3 id : SV_DispatchThreadID) {
    int2 px = int2(id.xy);
    int2 size = reflectSize();
    if (any(px >= size)) return;
    int2 fp = framePixel(px, size);
    float depth = loadDepth(fp);
    uint4 g = loadGbuffer(fp);
    float rough = gbufferRoughness(g);
    if (!receives(depth, g) || f.reflection.x < 0.5 || rough > f.reflection.y) {
        out0.Store(px, 0.0);
        return;
    }
    float3 n = gbufferNormal(g);
    float3 p = worldFromPixel(float2(fp) + 0.5, depth);
    // A reflection probe that takes over reflections shows here instead.
    if (probeOverrides(p, PROBE_OVERRIDES_REFLECTIONS)) {
        out0.Store(px, 0.0);
        return;
    }
    float3 v = normalize(p - f.camera.xyz);
    float2 pf = float2(px);
    float2 xi = float2(frac(ign(pf) + f.misc.x * 0.7548776662), frac(ign(pf + float2(47.0, 17.0)) + f.misc.x * 0.5698402910));
    float a2 = rough * rough * rough * rough;
    float cosT = sqrt((1.0 - xi.x) / (1.0 + (a2 - 1.0) * xi.x));
    float sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
    float phi = 6.2831853 * xi.y;
    float3 h = aroundNormal(n, float3(sinT * cos(phi), sinT * sin(phi), cosT));
    float3 dir = reflect(v, h);
    if (dot(dir, n) <= 0.0) dir = reflect(v, n);
    float3 origin = p + n * (0.02 + 0.002 * linearDepth(depth));
    float jitter = frac(ign(pf + float2(13.0, 101.0)) + f.misc.x * 0.6180339887);
    // Traced + Probes: a share of the texels, all of a smooth surface's and
    // fewer of a rougher one's, take a probe's capture of what they hit.
    probeTexture = f.probes.y != 0u
        && frac(ign(pf + float2(71.0, 29.0)) + f.misc.x * 0.3819660113) < 1.0 - smoothstep(0.15, 0.25, rough);
    float4 result = march(origin, dir, 3.0 * f.misc.y, 32, jitter);
    if (result.w < 0.5 && f.reflection.x > 1.5 && f.counts.x > 0u) {
        float far = 60.0;
        float3 wn;
        int hit = walkWorld(origin, dir, far, wn, false);
        if (hit >= 0) {
            uint seenAs = rough < 0.06 ? SEEN_MIRROR : SEEN_GLOSSY;
            result = float4(worldLight(origin + dir * far, wn, hit, true, true, seenAs), 1.0);
        }
    }
    out0.Store(px, bad(result) ? float4(0.0) : result);
}

// The reflection rays blended into what the same surface accumulated before:
// a mirror keeps a short history so its reflection follows the camera, a
// rough surface a longer one; each side weighted down by its brightness so a
// lone glint does not flare for frames.
// in0 this frame's rays, in1 previous light, in2 previous depth; out0 light, out1 depth.
[shader("compute")]
[numthreads(8, 8, 1)]
void reflect_temporal(uint3 id : SV_DispatchThreadID) {
    int2 px = int2(id.xy);
    int2 size = reflectSize();
    if (any(px >= size)) return;
    int2 fp = framePixel(px, size);
    float depth = loadDepth(fp);
    uint4 g = loadGbuffer(fp);
    if (!receives(depth, g)) {
        out0.Store(px, 0.0);
        out1.Store(px, 0.0);
        return;
    }
    float3 p = worldFromPixel(float2(fp) + 0.5, depth);
    float rough = gbufferRoughness(g);
    float4 now = in0.Load(int3(px, 0));
    float4 result = now;
    if (f.extra.z > 0.5) {
        float3 pp = project(f.prevClipFromWorld, p);
        if (onScreen(pp.xy)) {
            float2 at = pp.xy * float2(size) / frameSize();
            int2 tp = clamp(int2(at), 0, size - 1);
            float was = in2.Load(int3(tp, 0)).r;
            float4 wasLight = bilinear(in1, at, size);
            if (abs(was - pp.z) < 0.05 * pp.z && !bad(wasLight)) {
                float take = lerp(0.45, 0.06, min(sqrt(rough * 2.5), 1.0));
                float wWas = (1.0 - take) / (1.0 + luminance(wasLight.rgb));
                float wNow = take / (1.0 + luminance(now.rgb));
                result = (wasLight * wWas + now * wNow) / (wWas + wNow);
            }
        }
    }
    out0.Store(px, result);
    out1.Store(px, linearDepth(depth));
}

// One pass of an edge-avoiding a-trous filter over the accumulated
// reflections, over a reach growing with roughness; a mirror's left alone.
// in0 light, in1 depth; out0 light.
void reflectFilter(int2 px, int step) {
    int2 size = reflectSize();
    if (any(px >= size)) return;
    float4 c = in0.Load(int3(px, 0));
    float z = in1.Load(int3(px, 0)).r;
    int2 fp = framePixel(px, size);
    uint4 g = loadGbuffer(fp);
    float rough = gbufferRoughness(g);
    float reach = 16.0 * rough;
    if (z <= 0.0 || reach < 0.3 || rough > f.reflection.y) {
        out0.Store(px, c);
        return;
    }
    float3 pos = worldFromPixel(float2(fp) + 0.5, f.camera.w / z);
    float3 n = gbufferNormal(g);
    float4 sum = 0.0;
    float weights = 0.0;
    [loop] for (int y = -2; y <= 2; y++) {
        [loop] for (int x = -2; x <= 2; x++) {
            int2 q = px + int2(x, y) * step;
            if (any(q < 0) || any(q >= size)) continue;
            float zq = in1.Load(int3(q, 0)).r;
            if (zq <= 0.0) continue;
            int2 fq = framePixel(q, size);
            uint4 gq = loadGbuffer(fq);
            float d2 = float(x * x + y * y) * float(step * step);
            float plane = abs(dot(n, worldFromPixel(float2(fq) + 0.5, f.camera.w / zq) - pos));
            float w = kernelTap(abs(x)) * kernelTap(abs(y));
            w *= exp(-d2 / (2.0 * reach * reach));
            w *= exp(-plane / (0.02 + 0.01 * z));
            // A normal map's bumps are one surface, a corner is not.
            w *= pow(max(dot(n, gbufferNormal(gq)), 0.0), 8.0);
            w *= exp(-abs(gbufferRoughness(gq) - rough) / 0.05);
            sum += in0.Load(int3(q, 0)) * w;
            weights += w;
        }
    }
    out0.Store(px, weights > 1e-6 ? sum / weights : c);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void reflect_filter1(uint3 id : SV_DispatchThreadID) { reflectFilter(int2(id.xy), 1); }

[shader("compute")]
[numthreads(8, 8, 1)]
void reflect_filter2(uint3 id : SV_DispatchThreadID) { reflectFilter(int2(id.xy), 2); }

// The half-size reflections on the same surface around this pixel, blurred
// more the rougher it is: rgb the light, a how much of it was found (faded out
// over the last quarter of the cutoff).
float4 resolveReflection(int2 fp, float3 n, float d, float rough) {
    if (f.reflection.x < 0.5 || rough > f.reflection.y) return 0.0;
    if (probeOverrides(worldFromPixel(float2(fp) + 0.5, f.camera.w / d), PROBE_OVERRIDES_REFLECTIONS)) return 0.0;
    int2 size = reflectSize();
    float2 rc = (float2(fp) + 0.5) * float2(size) / frameSize() - 0.5;
    int2 rb = int2(floor(rc));
    float spread = 0.35 + 4.0 * sqrt(rough);
    float4 sum = 0.0;
    float weights = 0.0;
    [loop] for (int y = -1; y <= 2; y++) {
        [loop] for (int x = -1; x <= 2; x++) {
            int2 t = clamp(rb + int2(x, y), 0, size - 1);
            float tz = reflectDepth.Load(int3(t, 0)).r;
            if (tz <= 0.0) continue;
            float3 tn = gbufferNormal(loadGbuffer(framePixel(t, size)));
            float2 o = float2(t) - rc;
            float4 rt = reflectLight.Load(int3(t, 0));
            // A lone bright texel among dim ones is a glint, not the surface.
            float w = exp(-dot(o, o) / (2.0 * spread * spread)) * exp(-abs(tz - d) / (0.03 * d + 0.01))
                * pow(max(dot(tn, n), 0.0), 32.0) / (1.0 + luminance(rt.rgb));
            sum += rt * w;
            weights += w;
        }
    }
    if (weights <= 1e-4) return 0.0;
    float4 r = sum / weights;
    float fade = smoothstep(0.0, 1.0, (f.reflection.y - rough) / max(0.25 * f.reflection.y, 1e-3));
    return float4(r.a > 1e-3 ? r.rgb / r.a : float3(0.0), r.a * fade);
}

// The filtered quarter-size light around full-size pixel fp, on the same
// surface (facing n at linear depth d): rgb premultiplied by the fraction of
// rays that hit, a that fraction.
float4 upsampleGi(int2 fp, float3 n, float d) {
    int2 size = traceSize();
    float2 tc = (float2(fp) + 0.5) * float2(size) / frameSize() - 0.5;
    int2 base = int2(floor(tc));
    float4 sum = 0.0;
    float weights = 0.0;
    [loop] for (int y = -1; y <= 2; y++) {
        [loop] for (int x = -1; x <= 2; x++) {
            int2 t = clamp(base + int2(x, y), 0, size - 1);
            float tz = giDepth.Load(int3(t, 0)).r;
            if (tz <= 0.0) continue;
            float3 tn = gbufferNormal(loadGbuffer(framePixel(t, size)));
            float2 o = float2(t) - tc;
            float w = exp(-dot(o, o) * 0.5) * exp(-abs(tz - d) / (0.03 * d + 0.01)) * pow(max(dot(tn, n), 0.0), 8.0);
            sum += giLight.Load(int3(t, 0)) * w;
            weights += w;
        }
    }
    if (weights <= 1e-4) {
        // A sliver no texel shares (a face seen edge on): the texels at a
        // similar depth, whichever way they face.
        [loop] for (int y = -1; y <= 2; y++) {
            [loop] for (int x = -1; x <= 2; x++) {
                int2 t = clamp(base + int2(x, y), 0, size - 1);
                float tz = giDepth.Load(int3(t, 0)).r;
                if (tz <= 0.0) continue;
                float2 o = float2(t) - tc;
                float w = exp(-dot(o, o) * 0.5) * exp(-abs(tz - d) / (0.1 * d + 0.05));
                sum += giLight.Load(int3(t, 0)) * w;
                weights += w;
            }
        }
    }
    return weights > 1e-4 ? sum / weights : float4(0.0);
}

// Bevy's own diffuse ambient at a surface facing n, before its albedo: the
// camera's ambient light plus, when an environment map lights the scene, its
// irradiance (the specular map's roughest mip stands in for the diffuse map).
float3 bevyAmbient(float3 n) {
    float3 ambient = f.ambient.rgb;
    if (f.env.w > 0.5) ambient += backgroundLight(n, 1.0);
    return ambient;
}

// Up to full size, each pixel taking the filtered texels around it on the same
// surface. As Godot's surface shader mixes its GI buffer over its own ambient
// light by the fraction of rays that hit, this replaces Bevy's ambient light
// by the traced light where rays hit (rgb is premultiplied by that fraction),
// times the albedo: the difference is added onto Bevy's lit image.
[shader("fragment")]
float4 composite(float4 position : SV_Position) : SV_Target {
    int2 fp = int2(position.xy);
    float depth = loadDepth(fp);
    uint4 g = loadGbuffer(fp);
    if (!receives(depth, g)) return 0.0;
    int debug = int(f.misc.w);
    float3 p = worldFromPixel(float2(fp) + 0.5, depth);
    if (debug == DEBUG_LAST_FRAME) {
        float3 pp = project(f.prevClipFromWorld, p);
        return float4(colorTexture.SampleLevel(linearSampler, pp.xy / frameSize(), 0.0).rgb, 1.0);
    }
    if (debug == DEBUG_ROUND_TRIP) {
        // The pixel the reconstructed point projects back to: red where it
        // lands away from this one.
        float err = length(project(f.clipFromWorld, p).xy - (float2(fp) + 0.5));
        return float4(saturate(err), saturate(1.0 - err), 0.0, 1.0);
    }
    if (debug == DEBUG_WORLD_POSITION) {
        // One-metre bands: a floor at y = 0 shows a clean band edge at its surface.
        return float4(frac(p), 1.0);
    }
    float3 n = gbufferNormal(g);
    float d = linearDepth(depth);
    float4 gi = upsampleGi(fp, n, d);
    if (debug == DEBUG_BOUNCE_LIGHT) return float4(gi.rgb * f.misc.z, 1.0);
    if (debug == DEBUG_WORLD_HITS) return float4(gi.aaa, 1.0);
    // Traced reflections take over from Bevy's own reflection (its
    // environment map or reflection probe) where they found something: add
    // the difference, weighted by the split-sum specular response (Bevy's own
    // reflection is already in the image).
    float rough = gbufferRoughness(g);
    float4 reflection = resolveReflection(fp, n, d, rough);
    float3 view = normalize(f.camera.xyz - p);
    float3 mirrored = reflect(-view, n);
    if (f.probes.y != 0u && f.reflection.x > 0.5 && rough <= f.reflection.y
        && !probeOverrides(p, PROBE_OVERRIDES_REFLECTIONS)) {
        // Traced + Probes: where rays found nothing, the probe the point is in
        // fills in, box-projected, faded over its falloff.
        float3 l;
        float weight;
        int i = probeAt(p, l, weight);
        if (i >= 0 && weight > 0.0) {
            float3 r = normalize(lerp(mirrored, n, rough * rough));
            float3 fromProbe = probeReflection(i, p, l, r, rough);
            float a = reflection.a;
            float alpha = a + (1.0 - a) * weight;
            reflection = float4((reflection.rgb * a + fromProbe * (1.0 - a) * weight) / alpha, alpha);
        }
    }
    float3 specular = envBrdf(gbufferF0(g), rough, max(dot(n, view), 1e-4))
        * reflection.a * (reflection.rgb - bevyReflection(p, mirrored, rough));
    if (debug == DEBUG_REFLECTIONS) return float4(reflection.rgb * reflection.a, 1.0);
    float3 diffuse = gi.rgb * f.misc.z - gi.a * bevyAmbient(n);
    return float4(diffuse * gbufferAlbedo(g) + specular, 0.0);
}

// The raw diffuse estimate for `RadianceCascadesRawOutput`, at that image's
// size: rgb the light arriving (cosine weighted, divided by pi) before albedo,
// intensity and exposure; a 0 off receivers, else 1 + 0.25 * the fraction of
// rays that hit.
[[vk::binding(12, 1)]] [format("rgba32f")] WTexture2D<float4> rawOut;

[shader("compute")]
[numthreads(8, 8, 1)]
void raw_output(uint3 id : SV_DispatchThreadID) {
    uint width, height;
    rawOut.GetDimensions(width, height);
    int2 px = int2(id.xy);
    if (any(px >= int2(width, height))) return;
    int2 fp = clamp(int2((float2(px) + 0.5) * frameSize() / float2(width, height)), 0, int2(frameSize()) - 1);
    float depth = loadDepth(fp);
    uint4 g = loadGbuffer(fp);
    if (!receives(depth, g)) {
        rawOut.Store(px, 0.0);
        return;
    }
    float4 gi = upsampleGi(fp, gbufferNormal(g), linearDepth(depth));
    rawOut.Store(px, float4(gi.rgb / max(f.sunColor.w, 1e-20), 1.0 + 0.25 * saturate(gi.a)));
}

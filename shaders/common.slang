// Equal-area direction mapping: Split RC Algorithm 2. +Y is the polar axis.
static const float PI = 3.141592653589793;
static const uint CASCADES = 4;
static const uint BASE_THETA = 4;
static const uint EMPTY = 0xffffffff;
static const uint TOMBSTONE = 0xfffffffe;
// Distance LODs are independent of the four angular/distance cascades.
// Cap scaling at 2^24 (largest interval exponent 30), with five key bits.
static const uint MAX_PROBE_LOD = 24u;
static const uint PROBE_LOD = 31u << 18;

struct Parameters {
    float4x4 worldFromClip;
    float4x4 clipFromWorld;
    float4x4 previousClipFromWorld;
    float4x4 previousWorldFromClip;
    float4 cameraPosition;
    uint4 viewport;       // full-resolution viewport: origin.xy, size.zw
    uint4 size;           // GI width, height, pixel count, hash bucket count
    float4 tuning;        // probe spacing, interval cutoff, normal bias, intensity
    float4 environment;  // RGB radiance, exposure
    float4 history;      // temporal blend, reserved, valid, retention frames
    float4 intervalHistory; // transmittance decay, Split flag, diffuse debug .25, early query cap
    uint4 frame;         // frame index, merge cascade, instance count, light count
    uint4 lightCounts;   // conventional count, emissive count, hit-lighting sample budget, reserved
    float4 stability;    // screen history valid, scene changed, motion vectors, exact Disney diffuse
    uint4 layers;        // light-buffer mask tail, word count, occlusion flags, reserved
    uint4 environmentInfo; // alias-record offset, width, height, enabled
    float4 environmentRotation; // world-to-environment quaternion
    float4 environmentScale; // HDR intensity, reserved, reserved, reserved
    uint4 geometryPages; // geometry mode (0 buffer, 1 pages, 2 pages + deformed overlay), side
    uint4 scenePages; // paged scene texture enable, layer width, prefix words, animated ranges
    uint4 sceneOverlay; // instance base/words, immutable BLAS words, mutable BLAS words
    float4 dirtyMin; // changed scene union bounds.xyz; w > 0 means bounds are valid
    float4 dirtyMax; // w > 0 means the host proved bounded geometry-history reuse safe
    float4 transport; // reserved transport values, lighting-only invalidation
    uint4 sampling; // selected updates, traversal-query budget, retained cache cap, resolve scale
    uint4 ownerSequence; // LOD target pixels, clear-pass flag, fair ray quota, cursor/disabled flag
    uint4 cacheTime; // elapsed milliseconds; wraps after 49.7 days
    uint4 worldReceivers; // reserved
    uint4 worldReceiverState; // reserved
    float4 lightDirtyMin; // finite punctual-light influence union; w > 0 marks local change
    float4 lightDirtyMax;
};

[ForceInline]
uint cacheHistoryQuota(Parameters p) {
    uint quota = 4u;
    if (p.sampling.z > 0u)
        quota = min(quota, max(1u, p.sampling.z / max(p.size.w, 1u)));
    return quota;
}
bool persistentWorldIntervals(Parameters p) {
    return p.intervalHistory.y > 0.5;
}
bool worldGeometryHistoryUnchanged(Parameters p) {
    bool acceptedBoundedGeometry = persistentWorldIntervals(p)
        && p.dirtyMax.w > 0.5;
    return p.stability.y <= 0.5 || acceptedBoundedGeometry;
}
bool worldRadianceUnchanged(Parameters p) {
    return p.transport.w <= 0.5 && worldGeometryHistoryUnchanged(p);
}
bool intervalHistoryAlive(uint lastSample, Parameters p) {
    // Camera discovery must not expire unchanged world lighting with screen history.
    return (p.history.w > 0.0 && persistentWorldIntervals(p))
        || p.frame.x - lastSample <= uint(max(p.history.w, 0.0));
}
[ForceInline]
uint stochasticSampleFrame(Parameters p) {
    return p.frame.x;
}
[ForceInline]
uint giResolveScale(Parameters p) { return max(p.sampling.w, 1u); }
[ForceInline]
uint2 giResolveExtent(Parameters p) {
    uint scale = giResolveScale(p);
    return (p.viewport.zw + scale - 1u) / scale;
}

struct Triangle {
    float4 a, b, c;
    float4 albedo, emission; // emission.w: reserved
};
struct BvhNode {
    float3 minimum; uint first;
    float3 maximum; uint count;
};
struct Light { float4 position, color, direction, edge1, edge2; uint4 source; };

// One bounded node per ray/cascade, rather than a worst-case directional atlas.
// Full keys are compared, so hash collisions never blend unrelated probes.
struct Entry {
    int4 key;
    float4 interval; // J.rgb, beta (paper Eq. 6)
    float4 merged;   // I.rgb, valid
    float4 origin;   // canonical world-grid center.xyz, retained transmittance sample count.w
    uint next, valid, lastSample, samples;
    uint2 identity; // reserved; probe identity is the world-grid key
    uint2 padding; // x: selected-descriptor / retained-cache remap; y: visible-init marker
    // Primary interval segment sphere (center.xyz, radius.w); zero/malformed is unknown.
    float4 normal;
    // Local-light dependency spheres: center.xyz, radius.w > 0; w == -1 is
    // proven lighting-independent zero; zero/malformed values are unknown.
    float4 boundaryInterval; // Union of receiver hits contributing to retained raw J.
    float4 boundaryMerged; // Union of supported I hits, including cascade children.
    float4 dependencyMin; // RAW J/beta dependencies; w: 1 when measured (0 = unknown)
    float4 dependencyMax; // xyz: union maximum, w: 1 when the measured union is nonempty
    float4 mergedDependencyMin; // Full I includes child/visibility dependencies.
    float4 mergedDependencyMax;
};

// A dependency is the union of world-space segments that established a value.
// Keep "measured empty" distinct from unknown: empty bounds are valid only
// while the host's scene-root revision remains unchanged.
struct DependencyContext {
    float3 minimum;
    float3 maximum;
    bool known;
    bool hasBounds;
};

[ForceInline]
DependencyContext dependencyEmpty() {
    DependencyContext context;
    context.minimum = 0.0;
    context.maximum = 0.0;
    context.known = true;
    context.hasBounds = false;
    return context;
}

[ForceInline]
DependencyContext dependencyUnknown() {
    DependencyContext context = dependencyEmpty();
    context.known = false;
    return context;
}

[ForceInline]
void dependencyIncludeBounds(inout DependencyContext context, float3 minimum, float3 maximum) {
    if (!context.known) return;
    if (!all(isfinite(minimum)) || !all(isfinite(maximum)) || any(minimum > maximum)) {
        context = dependencyUnknown();
        return;
    }
    if (!context.hasBounds) {
        context.minimum = minimum;
        context.maximum = maximum;
        context.hasBounds = true;
    } else {
        context.minimum = min(context.minimum, minimum);
        context.maximum = max(context.maximum, maximum);
    }
}

[ForceInline]
void dependencyIncludeSegment(inout DependencyContext context, float3 start, float3 end,
                              float padding) {
    if (!all(isfinite(start)) || !all(isfinite(end)) || !isfinite(padding)) {
        context = dependencyUnknown();
        return;
    }
    float coordinateScale = max(1.0, max(max(abs(start.x), abs(start.y)), abs(start.z)));
    coordinateScale = max(coordinateScale,
        max(max(abs(end.x), abs(end.y)), abs(end.z)));
    // Keep endpoint arithmetic conservative at large world coordinates; this
    // is float-roundoff slack, not a shading or geometric acceptance tolerance.
    float3 radius = max(padding, 0.0) + max(coordinateScale * 1e-6, 1.1920929e-7);
    dependencyIncludeBounds(context, min(start, end) - radius, max(start, end) + radius);
}

[ForceInline]
void dependencyUnion(inout DependencyContext context, DependencyContext other) {
    if (!context.known || !other.known) {
        context = dependencyUnknown();
        return;
    }
    if (other.hasBounds)
        dependencyIncludeBounds(context, other.minimum, other.maximum);
}

[ForceInline]
DependencyContext storedDependency(float4 minimum, float4 maximum) {
    DependencyContext context;
    context.minimum = minimum.xyz;
    context.maximum = maximum.xyz;
    context.known = minimum.w > 0.5;
    context.hasBounds = maximum.w > 0.5;
    if (context.known && context.hasBounds
        && (!all(isfinite(context.minimum)) || !all(isfinite(context.maximum))
            || any(context.minimum > context.maximum)))
        context.known = false;
    return context;
}

[ForceInline]
DependencyContext entryDependency(Entry entry) {
    return storedDependency(entry.dependencyMin, entry.dependencyMax);
}

[ForceInline]
DependencyContext entryMergedDependency(Entry entry) {
    return storedDependency(entry.mergedDependencyMin, entry.mergedDependencyMax);
}

[ForceInline]
void storeEntryDependency(inout Entry entry, DependencyContext context) {
    entry.dependencyMin = float4(context.minimum, context.known ? 1.0 : 0.0);
    entry.dependencyMax = float4(context.maximum, context.hasBounds ? 1.0 : 0.0);
}

[ForceInline]
void storeEntryMergedDependency(inout Entry entry, DependencyContext context) {
    entry.mergedDependencyMin = float4(context.minimum, context.known ? 1.0 : 0.0);
    entry.mergedDependencyMax = float4(context.maximum, context.hasBounds ? 1.0 : 0.0);
}

// Probe-major Split irradiance stores a 6x6 octahedral table. Each texel holds
// conditional diffuse irradiance in rgb and angular coverage in w; aggregate
// dependencies preserve conservative invalidation of packed history.
struct DirectionalProbe {
    float4 irradiance[36];
    float4 dependencyMin; // aggregate dependency for supported raw directional bins
    float4 dependencyMax;
};

[ForceInline]
DependencyContext probeDependency(float4 minimumRecord, float4 maximumRecord) {
    DependencyContext context;
    context.minimum = minimumRecord.xyz;
    context.maximum = maximumRecord.xyz;
    context.known = minimumRecord.w > 0.5;
    context.hasBounds = maximumRecord.w > 0.5;
    if (context.known && context.hasBounds
        && (!all(isfinite(context.minimum)) || !all(isfinite(context.maximum))
            || any(context.minimum > context.maximum)))
        context.known = false;
    return context;
}

[ForceInline]
void storeProbeDependency(inout DirectionalProbe probe, DependencyContext context) {
    probe.dependencyMin = float4(context.minimum, context.known ? 1.0 : 0.0);
    probe.dependencyMax = float4(context.maximum, context.hasBounds ? 1.0 : 0.0);
}
[ForceInline]
float luminance(float3 color) { return dot(color, float3(0.2126, 0.7152, 0.0722)); }
// Independent scalar streams for estimators that need four decorrelated
// uniforms. Offsetting a 2D low-discrepancy pair only translates the same
// dimensions and is not independent for alias selection plus texel UVs.
[ForceInline]
uint transportHash(uint value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

[ForceInline]
float transportRandom(uint seed, uint lane) {
    uint bits = transportHash(seed + 0x9e3779b9u * (lane + 1u));
    return (float(bits >> 8) + 0.5) * (1.0 / 16777216.0);
}
// Disney/Burley scattering factor; the diffuse BSDF is albedo * factor / PI.
// NdotL, NdotV, and LdotH are evaluated for each sampled incident direction.
[ForceInline]
float disneyDiffuseMultiplier(float3 normal, float3 light, float3 view, float perceptualRoughness) {
    float nDotL = saturate(dot(normal, light));
    float nDotV = saturate(dot(normal, view));
    float3 halfSum = light + view;
    float halfLengthSquared = dot(halfSum, halfSum);
    if (nDotL <= 0 || nDotV <= 0 || halfLengthSquared <= 1e-20) return 0;
    float3 halfVector = halfSum * rsqrt(halfLengthSquared);
    // Bevy PBR passes perceptual roughness through perceptualRoughnessToRoughness.
    float roughness = max(saturate(perceptualRoughness), 0.089);
    roughness *= roughness;
    float lDotH = saturate(dot(light, halfVector));
    float f90 = 0.5 + 2.0 * roughness * lDotH * lDotH;
    float lightScatter = 1.0 + (f90 - 1.0) * pow(1.0 - nDotL, 5.0);
    float viewScatter = 1.0 + (f90 - 1.0) * pow(1.0 - nDotV, 5.0);
    return lightScatter * viewScatter;
}

float3 clampOutgoingRadiance(float3 value, Parameters p) {
    if (!all(isfinite(value))) return 0;
    // Transport caches store scene-linear energy. Firefly handling belongs in
    // reconstruction, where raw transport remains available for validation.
    return max(value, 0.0);
}

[ForceInline]
float rawGiSupportAlpha(bool eligible, bool supported, float confidence) {
    return !eligible ? 0.0 : (supported ? 1.0 + 0.25 * confidence : -1.0);
}

bool dirtyContains(float3 position, Parameters p) {
    return p.dirtyMin.w > 0.0 && all(position >= p.dirtyMin.xyz) && all(position <= p.dirtyMax.xyz);
}
float3 deferredBaseColor(uint4 packed) {
    float3 encoded = float3(packed.x & 255u, (packed.x >> 8) & 255u, (packed.x >> 16) & 255u) / 255.0;
    return pow(encoded, float3(2.2));
}
float3 deferredDiffuseAlbedo(uint4 packed) {
    float metallic = float((packed.z >> 8) & 255u) / 255.0;
    return deferredBaseColor(packed) * (1.0 - metallic);
}

float3 deferredShadingNormal(uint4 packed) {
    return decodeNormal(float2(packed.w & 4095u,(packed.w >> 12) & 4095u)/4095.0);
}
// Move along the oriented geometric normal by both the application's world
// bias and a coordinate-relative representable offset (Ray Tracing Gems Ch. 6).
float3 rayOffset(float3 position, float3 normal, float bias) {
    float3 origin = position + normal * bias;
    int3 step = int3(normal * 256.0);
    int3 bits = asint(origin);
    float3 integerOffset = asfloat(bits + int3(origin.x < 0 ? -step.x : step.x,
        origin.y < 0 ? -step.y : step.y, origin.z < 0 ? -step.z : step.z));
    float3 shifted = float3(abs(origin.x) < 0.03125 ? origin.x + normal.x / 65536.0 : integerOffset.x,
        abs(origin.y) < 0.03125 ? origin.y + normal.y / 65536.0 : integerOffset.y,
        abs(origin.z) < 0.03125 ? origin.z + normal.z / 65536.0 : integerOffset.z);
    // At the finite f32 coordinate ceiling no outward offset is representable.
    // Retain the finite origin; applications must rebase before that ceiling.
    return float3(isfinite(shifted.x) ? shifted.x : position.x,
        isfinite(shifted.y) ? shifted.y : position.y,
        isfinite(shifted.z) ? shifted.z : position.z);
}
[ForceInline]
float2 encodeNormal(float3 n) {
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    float2 p = n.xy;
    if (n.z < 0) p = (1 - abs(p.yx)) * float2(p.x >= 0 ? 1 : -1, p.y >= 0 ? 1 : -1);
    return p * 0.5 + 0.5;
}
[ForceInline]
float3 decodeNormal(float2 p) {
    p = p * 2 - 1;
    float3 n = float3(p, 1 - abs(p.x) - abs(p.y));
    float t = saturate(-n.z);
    n.xy += float2(n.x >= 0 ? -t : t, n.y >= 0 ? -t : t);
    return normalize(n);
}

[ForceInline]
float3 decodeDirection(float2 uv) {
    float phi = uv.x * 2.0 * PI;
    float y = uv.y * 2.0 - 1.0;
    float r = sqrt(max(0.0, 1.0 - y * y));
    return float3(r * cos(phi), y, r * sin(phi));
}
[ForceInline]
float2 encodeDirection(float3 direction) {
    return float2(frac(atan2(direction.z, direction.x) / (2.0 * PI) + 1.0),
                  saturate(direction.y * 0.5 + 0.5));
}
[ForceInline]
uint directionIndex(float3 direction, uint level) {
    uint theta = BASE_THETA << level;
    uint2 xy = min(uint2(encodeDirection(direction) * float2(2 * theta, theta)),
                   uint2(2 * theta - 1, theta - 1));
    return xy.x + xy.y * 2 * theta;
}
[ForceInline]
float3 binDirection(uint index, uint level) {
    uint theta = BASE_THETA << level;
    return decodeDirection((float2(index % (2 * theta), index / (2 * theta)) + 0.5)
                           / float2(2 * theta, theta));
}
[ForceInline]
float2 r2(uint index) {
    // Fixed-point phase avoids float sequence quantization after long sessions.
    uint2 phase = index * uint2(3242174889u, 2447445414u);
    return float2(phase >> 8) / 16777216.0;
}
float3 cosineDirection(float3 normal, float2 uv) {
    float3 tangent = normalize(cross(abs(normal.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0), normal));
    float radius = sqrt(uv.x), angle = 2 * PI * uv.y;
    return tangent * (radius * cos(angle)) + cross(normal, tangent) * (radius * sin(angle))
           + normal * sqrt(1 - uv.x);
}
[ForceInline]
float spacing(Parameters p, uint level, uint lod = 0) {
    return p.tuning.x * exp2(float(min(level, CASCADES - 1u) + min(lod, MAX_PROBE_LOD)));
}
[ForceInline]
int4 cacheKey(float3 position, uint direction, uint level, Parameters p, uint lod = 0) {
    lod = min(lod, MAX_PROBE_LOD);
    return int4(int3(floor(position / spacing(p, level, lod))),
                int(direction | (level << 12) | (lod << 18)));
}
[ForceInline]
uint hashKey(int4 key, uint bucketCount) {
    uint4 k = uint4(key);
    uint h = k.x * 0x8da6b343u ^ k.y * 0xd8163841u ^ k.z * 0xcb1ab31fu ^ k.w * 0x165667b1u;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15;
    return h & (bucketCount - 1);
}
[ForceInline]
float3 worldPosition(float2 pixel, float depth, Parameters p) {
    float2 uv = (pixel - float2(p.viewport.xy)) / float2(p.viewport.zw);
    float4 h = mul(p.worldFromClip, float4(uv.x * 2 - 1, 1 - uv.y * 2, depth, 1));
    return h.xyz / h.w;
}
float2 primaryRayCone(float2 pixel, float depth, float3 origin, float3 receiver, Parameters p) {
    float2 pixelStep = float2(p.viewport.zw) / float2(p.size.xy);
    float3 originX = worldPosition(pixel + float2(pixelStep.x, 0), 1, p);
    float3 originY = worldPosition(pixel + float2(0, pixelStep.y), 1, p);
    float3 receiverX = worldPosition(pixel + float2(pixelStep.x, 0), depth, p);
    float3 receiverY = worldPosition(pixel + float2(0, pixelStep.y), depth, p);
    float width = max(length(originX - origin), length(originY - origin));
    float receiverWidth = max(length(receiverX - receiver), length(receiverY - receiver));
    return float2(width, max(receiverWidth - width, 0) / max(length(receiver - origin), 1e-20));
}
[ForceInline]
float3 previousWorldPosition(float2 pixel, float depth, Parameters p) {
    float2 uv = (pixel - float2(p.viewport.xy)) / float2(p.viewport.zw);
    float4 h = mul(p.previousWorldFromClip, float4(uv.x * 2 - 1, 1 - uv.y * 2, depth, 1));
    return h.xyz / h.w;
}
// Invert the integer prepass sample centers, including noninteger resolution ratios.
float2 sparseGridJitter(Parameters p) {
    // Keep partial schedules anchored so a revisit addresses the same raster sample.
    bool partialReceiverBudget = (p.ownerSequence.w & 0x40000000u) != 0u
        || (p.sampling.x > 0u && p.sampling.x < p.size.z);
    if (partialReceiverBudget || p.history.x <= 0.0 || p.history.w <= 0.0) return 0.0;
    float2 scale = float2(p.viewport.zw) / float2(p.size.xy);
    return (r2(p.frame.x * 977u + 487u) - 0.5) * scale;
}
float2 irradianceGrid(float2 pixel, Parameters p) {
    float2 scale = float2(p.viewport.zw) / float2(p.size.xy);
    float2 local = pixel - float2(p.viewport.xy);
    float2 jitter = sparseGridJitter(p);
    int2 base = int2(floor((local - jitter) / scale - 0.5));
    float2 center = floor((float2(base) + 0.5) * scale + jitter) + 0.5;
    base -= int2(local.x < center.x ? 1 : 0, local.y < center.y ? 1 : 0);
    float2 left = floor((float2(base) + 0.5) * scale + jitter) + 0.5;
    float2 right = floor((float2(base) + 1.5) * scale + jitter) + 0.5;
    return float2(base) + saturate((local - left) / (right - left));
}
float surfaceTolerance(Parameters p) { return max(p.tuning.z * 2, p.tuning.x * 0.02); }
// A chord on a smooth surface lies near the average tangent plane of its endpoints.
float surfacePlaneDistance(float3 delta, float3 targetNormal, float3 sampleNormal) {
    float3 averageNormal = targetNormal + sampleNormal;
    float normalLengthSquared = dot(averageNormal, averageNormal);
    if (!(normalLengthSquared > 1e-20) || !isfinite(normalLengthSquared))
        return abs(dot(delta, targetNormal));
    return abs(dot(delta, averageNormal * rsqrt(normalLengthSquared)));
}
uint2 fullPixel(uint2 pixel, Parameters p) {
    float2 jitter = sparseGridJitter(p);
    return p.viewport.xy + min(uint2((float2(pixel) + 0.5) * float2(p.viewport.zw) / float2(p.size.xy) + jitter),
                               p.viewport.zw - 1);
}

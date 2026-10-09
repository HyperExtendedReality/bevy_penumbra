// Equal-area direction mapping: Split RC Algorithm 2. +Y is the polar axis.
static const float PI = 3.141592653589793;
static const uint CASCADES = 4;
static const uint BASE_THETA = 4;
static const uint IRRADIANCE_SIDE = 6;
static const uint EMPTY = 0xffffffff;
static const uint TOMBSTONE = 0xfffffffe;
static const uint SECONDARY = 1u << 14;
static const uint SURFACE = 7u << 15;
static const uint PROBE_LOD = 3u << 18;

struct Parameters {
    float4x4 worldFromClip;
    float4x4 previousClipFromWorld;
    float4x4 previousWorldFromClip;
    float4 cameraPosition;
    uint4 viewport;       // full-resolution viewport: origin.xy, size.zw
    uint4 size;           // GI width, height, pixel count, hash bucket count
    float4 tuning;        // probe spacing, interval cutoff, normal bias, intensity
    float4 environment;  // RGB radiance, exposure
    float4 history;      // temporal blend, bounce feedback, valid, retention frames
    uint4 frame;         // frame index, merge cascade, instance count, light count
    uint4 lightCounts;   // conventional count, emissive count, receiver sample budget, reserved
    float4 stability;    // screen history valid, scene changed, reserved.xy
    uint4 layers;        // light-buffer mask tail, 32-bit word count, reserved.xy
    uint4 environmentInfo; // alias-record offset, width, height, enabled
    float4 environmentRotation; // world-to-environment quaternion
    float4 environmentScale; // HDR intensity, reserved.xyz
    uint4 geometryPages; // geometry mode (0 buffer, 1 pages, 2 pages + deformed overlay), side
    uint4 scenePages; // paged scene texture enable, layer width, prefix words, animated ranges
    uint4 sceneOverlay; // instance base/words, immutable BLAS words, mutable BLAS words
};

struct Triangle {
    float4 a, b, c;
    float4 albedo, emission; // emission.w: discrete light probability for MIS
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
    float4 origin;   // actual ray origin.xyz, decayed interval sample count.w
    uint next, valid, lastSample, samples;
    uint2 identity; // stable entity bits, including its generation
    uint2 padding; // x: retained-cache remap; y: finest-level mesh component
};

// 6x6 octahedral irradiance, as in Chandra; virtual border taps fold onto
// this core grid without allocating duplicate texels. Each texel
// stores independently normalized E/pi and validity.
struct IrradianceProbe { float4 coefficients[36]; float4 origin; };
int2 irradianceOctTexel(int2 pixel) {
    int2 p = pixel - 1;
    int last = int(IRRADIANCE_SIDE) - 1;
    if (p.x < 0) { p.x = 0; p.y = last - p.y; }
    else if (p.x > last) { p.x = last; p.y = last - p.y; }
    if (p.y < 0) { p.y = 0; p.x = last - p.x; }
    else if (p.y > last) { p.y = last; p.x = last - p.x; }
    return p;
}
float2 irradianceOctUv(int2 pixel) { return (float2(irradianceOctTexel(pixel)) + 0.5) / float(IRRADIANCE_SIDE); }
uint irradianceOctIndex(int2 pixel) {
    int2 texel = irradianceOctTexel(pixel);
    return uint(texel.x + texel.y * int(IRRADIANCE_SIDE));
}
float luminance(float3 color) { return dot(color, float3(0.2126, 0.7152, 0.0722)); }
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
float deferredPerceptualRoughness(uint4 packed) { return float(packed.x >> 24)/255.0; }
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
float2 encodeNormal(float3 n) {
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    float2 p = n.xy;
    if (n.z < 0) p = (1 - abs(p.yx)) * float2(p.x >= 0 ? 1 : -1, p.y >= 0 ? 1 : -1);
    return p * 0.5 + 0.5;
}
float3 decodeNormal(float2 p) {
    p = p * 2 - 1;
    float3 n = float3(p, 1 - abs(p.x) - abs(p.y));
    float t = saturate(-n.z);
    n.xy += float2(n.x >= 0 ? -t : t, n.y >= 0 ? -t : t);
    return normalize(n);
}

float3 decodeDirection(float2 uv) {
    float phi = uv.x * 2.0 * PI;
    float y = uv.y * 2.0 - 1.0;
    float r = sqrt(max(0.0, 1.0 - y * y));
    return float3(r * cos(phi), y, r * sin(phi));
}
float2 encodeDirection(float3 direction) {
    return float2(frac(atan2(direction.z, direction.x) / (2.0 * PI) + 1.0),
                  saturate(direction.y * 0.5 + 0.5));
}
uint directionIndex(float3 direction, uint level) {
    uint theta = BASE_THETA << level;
    uint2 xy = min(uint2(encodeDirection(direction) * float2(2 * theta, theta)),
                   uint2(2 * theta - 1, theta - 1));
    return xy.x + xy.y * 2 * theta;
}
float3 binDirection(uint index, uint level) {
    uint theta = BASE_THETA << level;
    return decodeDirection((float2(index % (2 * theta), index / (2 * theta)) + 0.5)
                           / float2(2 * theta, theta));
}
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
float spacing(Parameters p, uint level, bool secondary = false, uint lod = 0) {
    return p.tuning.x * float(1u << (level + lod)) * (secondary ? 4.0 : 1.0);
}
// Opposing faces in a thin wall must not share hemisphere samples or feedback.
// Six normal sectors split opposing faces; component IDs separately split disconnected islands.
uint surfaceKey(float3 normal) {
    float3 n = abs(normal);
    uint axis = n.x >= n.y && n.x >= n.z ? 0 : (n.y >= n.z ? 1 : 2);
    return (axis * 2 + (normal[axis] < 0 ? 1u : 0u)) << 15;
}
int4 cacheKey(float3 position, uint direction, uint level, Parameters p, bool secondary = false, uint surface = 0, uint lod = 0) {
    return int4(int3(floor(position / spacing(p, level, secondary, lod))),
                int(direction | (level << 12) | (secondary ? SECONDARY : 0) | surface | (lod << 18)));
}
uint hashKey(int4 key, uint bucketCount) {
    uint4 k = uint4(key);
    uint h = k.x * 0x8da6b343u ^ k.y * 0xd8163841u ^ k.z * 0xcb1ab31fu ^ k.w * 0x165667b1u;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15;
    return h & (bucketCount - 1);
}
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
float3 previousWorldPosition(float2 pixel, float depth, Parameters p) {
    float2 uv = (pixel - float2(p.viewport.xy)) / float2(p.viewport.zw);
    float4 h = mul(p.previousWorldFromClip, float4(uv.x * 2 - 1, 1 - uv.y * 2, depth, 1));
    return h.xyz / h.w;
}
// Invert the actual integer prepass sample centers, including noninteger
// resolution ratios. Nominal low-res centers shift history by half a full pixel.
float2 irradianceGrid(float2 pixel, Parameters p) {
    float2 scale = float2(p.viewport.zw) / float2(p.size.xy);
    float2 local = pixel - float2(p.viewport.xy);
    int2 base = int2(floor(local / scale - 0.5));
    float2 center = floor((float2(base) + 0.5) * scale) + 0.5;
    base -= int2(local.x < center.x ? 1 : 0, local.y < center.y ? 1 : 0);
    float2 left = floor((float2(base) + 0.5) * scale) + 0.5;
    float2 right = floor((float2(base) + 1.5) * scale) + 0.5;
    return float2(base) + saturate((local - left) / (right - left));
}
float surfaceTolerance(Parameters p) { return max(p.tuning.z * 2, p.tuning.x * 0.02); }
uint2 fullPixel(uint2 pixel, Parameters p) {
    return p.viewport.xy + min(uint2((float2(pixel) + 0.5) * float2(p.viewport.zw) / float2(p.size.xy)),
                               p.viewport.zw - 1);
}

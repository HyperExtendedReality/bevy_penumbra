// Equal-area direction mapping: Split RC Algorithm 2. +Y is the polar axis.
static const float PI = 3.141592653589793;
static const uint CASCADES = 4;
static const uint BASE_THETA = 4;
static const uint EMPTY = 0xffffffff;
static const uint SECONDARY = 1u << 14;
static const uint SURFACE = 7u << 15;

struct Parameters {
    float4x4 worldFromClip;
    float4x4 previousClipFromWorld;
    float4x4 previousWorldFromClip;
    uint4 viewport;       // full-resolution viewport: origin.xy, size.zw
    uint4 size;           // GI width, height, sample count, hash bucket count
    float4 tuning;        // probe spacing, interval cutoff, normal bias, intensity
    float4 environment;  // RGB radiance, exposure
    float4 history;      // temporal blend, bounce feedback, valid, retention frames
    uint4 frame;         // frame index, merge cascade, triangle count, light count
    uint4 lightCounts;   // conventional count, emissive count, receiver sample budget
    float4 stability;    // screen history valid, scene changed, reserved.xy
};

struct Triangle {
    float4 a, b, c;
    float4 albedo, emission; // emission.w: discrete light probability for MIS
};
struct BvhNode {
    float3 minimum; uint first;
    float3 maximum; uint count;
};
struct Light { float4 position, color, direction, edge1, edge2; };

// One bounded node per ray/cascade, rather than a worst-case directional atlas.
// Full keys are compared, so hash collisions never blend unrelated probes.
struct Entry {
    int4 key;
    float4 interval; // J.rgb, beta (paper Eq. 6)
    float4 merged;   // I.rgb, valid
    float4 origin;   // representative actual ray origin; never a centroid in a wall
    uint next, valid, lastSample, samples;
};

// Nonnegative ambient cube: sparse SH coverage division rings near zero and
// creates bright splotches. Each lobe contains an independently normalized E/pi.
struct IrradianceProbe { float4 coefficients[6]; float4 origin; };
float3 lobeDirection(uint i) {
    float3 n = 0; n[i / 2] = (i & 1) == 0 ? 1.0 : -1.0; return n;
}
float luminance(float3 color) { return dot(color, float3(0.2126, 0.7152, 0.0722)); }
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
float spacing(Parameters p, uint level, bool secondary = false) {
    return p.tuning.x * float(1u << level) * (secondary ? 4.0 : 1.0);
}
// Opposing faces in a thin wall must not share hemisphere samples or feedback.
// ponytail: six normal sectors; use surface clusters if curved-surface seams matter.
uint surfaceKey(float3 normal) {
    float3 n = abs(normal);
    uint axis = n.x >= n.y && n.x >= n.z ? 0 : (n.y >= n.z ? 1 : 2);
    return (axis * 2 + (normal[axis] < 0 ? 1u : 0u)) << 15;
}
int4 cacheKey(float3 position, uint direction, uint level, Parameters p, bool secondary = false, uint surface = 0) {
    return int4(int3(floor(position / spacing(p, level, secondary))),
                int(direction | (level << 12) | (secondary ? SECONDARY : 0) | surface));
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

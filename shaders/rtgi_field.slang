// Ray-traced GI's float4 field, shared by cascades.slang (read-write) and
// composite.slang (read-only). The including file declares `params` and
// `rtgiField`. Layout, mirrored by rtgi_field_bytes in render.rs:
//   irradiance volume probes, RTGI_VOLUME_CASCADES x RTGI_VOLUME_PROBES
//   probe ray scratch, RTGI_VOLUME_BATCH x RTGI_VOLUME_RAYS
//   two gather images (SH, before and between the separable denoise passes)
//   two temporal SH histories by frame parity
// All per-pixel images cover giResolveExtent(params).

// idTech 8's camera-centred cascaded irradiance volumes: 16^3 probes per
// cascade, spacing doubling per cascade from RTGI_VOLUME_SPACING.
static const uint RTGI_VOLUME_SIDE = 16u;
static const uint RTGI_VOLUME_PROBES = RTGI_VOLUME_SIDE * RTGI_VOLUME_SIDE * RTGI_VOLUME_SIDE;
static const uint RTGI_VOLUME_CASCADES = 4u;
static const float RTGI_VOLUME_SPACING = 1.0;
// Rays per probe update and probes updated per frame: half a cascade, so
// each probe updates every 2 * RTGI_VOLUME_CASCADES frames.
static const uint RTGI_VOLUME_RAYS = 32u;
static const uint RTGI_VOLUME_BATCH = RTGI_VOLUME_PROBES / 2u;
// Probe record: header (grid coordinate, state), L1 SH of surface radiance
// (r, g, b), L1 SH of sky radiance (r, g, b), then 8x8 octahedral visibility
// texels (mean distance, mean squared distance), two per float4.
static const uint RTGI_VISIBILITY_SIDE = 8u;
static const uint RTGI_PROBE_WORDS = 7u + RTGI_VISIBILITY_SIDE * RTGI_VISIBILITY_SIDE / 2u;
static const uint RTGI_VOLUME_WORDS = RTGI_VOLUME_CASCADES * RTGI_VOLUME_PROBES * RTGI_PROBE_WORDS;
// Per ray: direction and hit distance (negative for back faces, beyond the
// volume for escapes), then the cell slot and key the hit touched.
static const uint RTGI_RAY_WORDS = RTGI_VOLUME_BATCH * RTGI_VOLUME_RAYS * 2u;
// Gather image texel: SH r, g, b, (hit distance, valid, 0, 0), then the
// receiver's (position, 1 when valid) and shading normal for the denoise.
static const uint RTGI_GATHER_WORDS = 6u;
// History texel: SH r, g, b, (position, sample count), (normal, hit
// distance), and the raster receiver identity encoded as float bits.
static const uint RTGI_HISTORY_WORDS = 6u;
// Stable cache-cell snapshot consumed by the bounce dispatch.
static const uint RTGI_SNAPSHOT_WORDS = 8u;

uint rtgiPixelCount() {
    uint2 extent = giResolveExtent(params);
    return extent.x * extent.y;
}
uint rtgiProbeBase(uint cascade, uint probe) {
    return (cascade * RTGI_VOLUME_PROBES + probe) * RTGI_PROBE_WORDS;
}
uint rtgiRayBase(uint lane) {
    return RTGI_VOLUME_WORDS + lane * 2u;
}
uint rtgiGatherBase(uint image, uint pixel) {
    return RTGI_VOLUME_WORDS + RTGI_RAY_WORDS
        + (image * rtgiPixelCount() + pixel) * RTGI_GATHER_WORDS;
}
uint rtgiHistoryBase(uint parity, uint pixel) {
    return RTGI_VOLUME_WORDS + RTGI_RAY_WORDS + 2u * rtgiPixelCount() * RTGI_GATHER_WORDS
        + (parity * rtgiPixelCount() + pixel) * RTGI_HISTORY_WORDS;
}
uint rtgiSnapshotBase(uint slot) {
    return RTGI_VOLUME_WORDS + RTGI_RAY_WORDS
        + 2u * rtgiPixelCount() * (RTGI_GATHER_WORDS + RTGI_HISTORY_WORDS)
        + slot * RTGI_SNAPSHOT_WORDS;
}

// L1 spherical harmonics of radiance per colour channel: (Y00, Y1 x, y, z)
// projection coefficients.
struct RtgiSh {
    float4 r;
    float4 g;
    float4 b;
};
static const float RTGI_SH_Y0 = 0.282095;
static const float RTGI_SH_Y1 = 0.488603;

float4 rtgiShBasis(float3 direction) {
    return float4(RTGI_SH_Y0, RTGI_SH_Y1 * direction);
}
RtgiSh rtgiShZero() {
    RtgiSh sh;
    sh.r = 0.0; sh.g = 0.0; sh.b = 0.0;
    return sh;
}
// Adds radiance arriving from `direction`, weighted (for Monte Carlo
// projection, by the reciprocal sample density).
void rtgiShAdd(inout RtgiSh sh, float3 radiance, float3 direction, float weight) {
    float4 basis = rtgiShBasis(direction) * weight;
    sh.r += radiance.r * basis;
    sh.g += radiance.g * basis;
    sh.b += radiance.b * basis;
}
RtgiSh rtgiShScale(RtgiSh sh, float scale) {
    sh.r *= scale; sh.g *= scale; sh.b *= scale;
    return sh;
}
RtgiSh rtgiShLerp(RtgiSh a, RtgiSh b, float t) {
    a.r = lerp(a.r, b.r, t); a.g = lerp(a.g, b.g, t); a.b = lerp(a.b, b.b, t);
    return a;
}
void rtgiShAccumulate(inout RtgiSh sum, RtgiSh sh, float weight) {
    sum.r += sh.r * weight; sum.g += sh.g * weight; sum.b += sh.b * weight;
}
// Irradiance over pi at a surface with `normal`: the cosine lobe's zonal
// factors are pi and 2pi/3 (Ramamoorthi and Hanrahan 2001).
float3 rtgiShIrradiance(RtgiSh sh, float3 normal) {
    float4 lobe = float4(RTGI_SH_Y0, (2.0 / 3.0) * RTGI_SH_Y1 * normal);
    float3 value = float3(dot(sh.r, lobe), dot(sh.g, lobe), dot(sh.b, lobe));
    return all(isfinite(value)) ? max(value, 0.0) : float3(0.0);
}
RtgiSh rtgiLoadSh(uint base) {
    RtgiSh sh;
    sh.r = rtgiField[base];
    sh.g = rtgiField[base + 1u];
    sh.b = rtgiField[base + 2u];
    return sh;
}

// Octahedral direction mapping onto [0, 1]^2.
float2 rtgiOctEncode(float3 direction) {
    direction /= max(abs(direction.x) + abs(direction.y) + abs(direction.z), 1e-20);
    float2 folded = direction.xy;
    if (direction.z < 0.0) {
        float2 flip = float2(direction.x >= 0.0 ? 1.0 : -1.0, direction.y >= 0.0 ? 1.0 : -1.0);
        folded = (1.0 - abs(direction.yx)) * flip;
    }
    return folded * 0.5 + 0.5;
}
float3 rtgiOctDecode(float2 uv) {
    float2 f = uv * 2.0 - 1.0;
    float3 direction = float3(f, 1.0 - abs(f.x) - abs(f.y));
    if (direction.z < 0.0) {
        float2 flip = float2(direction.x >= 0.0 ? 1.0 : -1.0, direction.y >= 0.0 ? 1.0 : -1.0);
        direction.xy = (1.0 - abs(direction.yx)) * flip;
    }
    return normalize(direction);
}
// Wraps a visibility texel one step outside the map onto its octahedral
// neighbour, as DDGI's probe borders do.
int2 rtgiOctFold(int2 texel) {
    int last = int(RTGI_VISIBILITY_SIDE) - 1;
    if (texel.x < 0) { texel.x = 0; texel.y = last - texel.y; }
    else if (texel.x > last) { texel.x = last; texel.y = last - texel.y; }
    if (texel.y < 0) { texel.y = 0; texel.x = last - texel.x; }
    else if (texel.y > last) { texel.y = last; texel.x = last - texel.x; }
    return texel;
}
float2 rtgiVisibilityTexel(uint probeBase, int2 texel) {
    texel = rtgiOctFold(texel);
    uint index = uint(texel.x) + uint(texel.y) * RTGI_VISIBILITY_SIDE;
    float4 pair = rtgiField[probeBase + 7u + index / 2u];
    return (index & 1u) != 0u ? pair.zw : pair.xy;
}
float2 rtgiVisibility(uint probeBase, float3 direction) {
    float2 texel = rtgiOctEncode(direction) * float(RTGI_VISIBILITY_SIDE) - 0.5;
    int2 base = int2(floor(texel));
    float2 f = texel - float2(base);
    return lerp(lerp(rtgiVisibilityTexel(probeBase, base),
                     rtgiVisibilityTexel(probeBase, base + int2(1, 0)), f.x),
                lerp(rtgiVisibilityTexel(probeBase, base + int2(0, 1)),
                     rtgiVisibilityTexel(probeBase, base + int2(1, 1)), f.x), f.y);
}

float rtgiVolumeCellSize(uint cascade) {
    return RTGI_VOLUME_SPACING * float(1u << cascade);
}
// Grid coordinate of the cascade's first probe; the volume follows the camera.
int3 rtgiVolumeOrigin(uint cascade) {
    return int3(floor(params.cameraPosition.xyz / rtgiVolumeCellSize(cascade)))
        - int(RTGI_VOLUME_SIDE / 2u);
}
// Probes are stored toroidally, so a camera move only replaces the probes
// that left the volume.
uint rtgiVolumeSlot(int3 coordinate) {
    int side = int(RTGI_VOLUME_SIDE);
    uint3 wrapped = uint3(((coordinate % side) + side) % side);
    return wrapped.x + (wrapped.y + wrapped.z * RTGI_VOLUME_SIDE) * RTGI_VOLUME_SIDE;
}
// The grid coordinate a stored probe stands for this frame.
int3 rtgiVolumeCoordinate(uint cascade, uint slot) {
    int side = int(RTGI_VOLUME_SIDE);
    int3 origin = rtgiVolumeOrigin(cascade);
    int3 stored = int3(slot % RTGI_VOLUME_SIDE, (slot / RTGI_VOLUME_SIDE) % RTGI_VOLUME_SIDE,
        slot / (RTGI_VOLUME_SIDE * RTGI_VOLUME_SIDE));
    return origin + (((stored - origin) % side) + side) % side;
}
float3 rtgiVolumeProbePosition(uint cascade, int3 coordinate) {
    return (float3(coordinate) + 0.5) * rtgiVolumeCellSize(cascade);
}

// Irradiance over pi from the irradiance volumes at a surface, split into
// light bounced off surfaces and sky light, with DDGI's backface and
// Chebyshev visibility weights (Majercik et al. 2019). False when no probe
// of the finest cascade covering `position` has been traced yet.
bool rtgiVolumeIrradiance(float3 position, float3 normal, out float3 surface, out float3 sky) {
    surface = 0.0;
    sky = 0.0;
    [loop] for (uint cascade = 0u; cascade < RTGI_VOLUME_CASCADES; ++cascade) {
        float size = rtgiVolumeCellSize(cascade);
        int3 origin = rtgiVolumeOrigin(cascade);
        float3 grid = position / size - 0.5;
        int3 base = int3(floor(grid));
        if (any(base < origin) || any(base + 1 > origin + int(RTGI_VOLUME_SIDE) - 1)) continue;
        float3 alpha = grid - float3(base);
        float3 biased = position + normal * (0.2 * size);
        float3 surfaceSum = 0.0, skySum = 0.0;
        float weightSum = 0.0;
        [loop] for (uint corner = 0u; corner < 8u; ++corner) {
            int3 offset = int3(corner & 1u, (corner >> 1u) & 1u, corner >> 2u);
            int3 coordinate = base + offset;
            uint probeBase = rtgiProbeBase(cascade, rtgiVolumeSlot(coordinate));
            float4 header = rtgiField[probeBase];
            if (!(header.w > 0.0) || any(asint(header.xyz) != coordinate)) continue;
            float3 probe = rtgiVolumeProbePosition(cascade, coordinate);
            float3 toProbe = probe - position;
            float toProbeLength = length(toProbe);
            float3 trilinear = lerp(1.0 - alpha, alpha, float3(offset));
            float weight = 1.0;
            if (toProbeLength > 1e-6) {
                float facing = (dot(toProbe / toProbeLength, normal) + 1.0) * 0.5;
                weight = facing * facing + 0.2;
            }
            float3 fromProbe = biased - probe;
            float distance = length(fromProbe);
            if (distance > 1e-6) {
                float2 moments = rtgiVisibility(probeBase, fromProbe / distance);
                if (distance > moments.x) {
                    float variance = max(moments.y - moments.x * moments.x, 1e-4 * size * size);
                    float excess = distance - moments.x;
                    float chebyshev = variance / (variance + excess * excess);
                    weight *= chebyshev * chebyshev * chebyshev;
                }
            }
            weight = max(weight, 1e-6);
            // Crush tiny weights so a leaking probe cannot dominate.
            if (weight < 0.2) weight *= weight * weight / 0.04;
            weight *= trilinear.x * trilinear.y * trilinear.z;
            surfaceSum += rtgiShIrradiance(rtgiLoadSh(probeBase + 1u), normal) * weight;
            skySum += rtgiShIrradiance(rtgiLoadSh(probeBase + 4u), normal) * weight;
            weightSum += weight;
        }
        if (!(weightSum > 1e-6)) return false;
        surface = surfaceSum / weightSum;
        sky = skySum / weightSum;
        return true;
    }
    return false;
}

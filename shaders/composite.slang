#include "common.slang"
#include "receiver_metadata.slang"

[[vk::binding(0, 0)]] ConstantBuffer<Parameters> params;
[[vk::binding(2, 0)]] Texture2D<float4> irradiance;
[[vk::binding(3, 0)]] DepthTexture2D depthTexture;
[[vk::binding(5, 0)]] Texture2D<uint4> gbuffer;
[[vk::binding(6, 0)]] Texture2D<float4> receiverHistory;
[[vk::binding(7, 0)]] Texture2D<uint4> rasterReceiver;
[[vk::binding(8, 0)]] Texture2D<float> screenSpaceOcclusion;
[[vk::binding(9, 0)]] StructuredBuffer<uint> probeHeads;
[[vk::binding(10, 0)]] StructuredBuffer<Entry> probeEntries;
[[vk::binding(11, 0)]] StructuredBuffer<DirectionalProbe> directionalProbes;
[[vk::binding(12, 0)]] Texture2D<float4> anchorPosition;
[[vk::binding(13, 0)]] Texture2D<float4> anchorNormal;
[[vk::binding(14, 0)]] [format("rgba32f")] WTexture2D<float4> anchorPositionOutput;
[[vk::binding(15, 0)]] [format("rgba32f")] WTexture2D<float4> anchorNormalOutput;
[[vk::binding(16, 0)]] Texture2D<float4> sunShadowMask;
// Ray-traced GI's field: temporal SH and irradiance volumes.
[[vk::binding(17, 0)]] StructuredBuffer<float4> rtgiField;
#include "rtgi_field.slang"

void splitLodBlend(float3 position, out uint lowLod, out uint highLod, out float blend) {
    splitLodBlendForParameters(position, params, lowLod, highLod, blend);
}

uint findIrradianceProbe(int4 key, bool previous) {
    if (previous) return EMPTY;
    uint capacity = probeDirectoryCapacity(params);
    uint slot = probeDirectoryBucket(key, params);
    uint count, stride;
    probeEntries.GetDimensions(count, stride);
    [loop] for (uint attempt = 0u; attempt < capacity; ++attempt) {
        uint index = probeHeads[probeDirectoryHead(0u, slot, params)];
        if (index == EMPTY) return EMPTY;
        if (index != TOMBSTONE) {
            if (index >= count) return EMPTY;
            int4 candidate = probeEntries[index].key;
            if (all(candidate.xyz == key.xyz)
                && (uint(candidate.w) & PROBE_LOD) == (uint(key.w) & PROBE_LOD))
                return slot;
        }
        slot = (slot + 1u) & (capacity - 1u);
    }
    return EMPTY;
}

#define GI_COMPOSITE_RESOLVE
#include "world_resolve.slang"
#undef GI_COMPOSITE_RESOLVE

struct VertexOutput { float4 position : SV_Position; };

[shader("vertex")]
VertexOutput vertex(uint index : SV_VertexID) {
    VertexOutput output;
    output.position = float4(index == 1 ? 3.0 : -1.0,
                             index == 2 ? 3.0 : -1.0, 0.0, 1.0);
    return output;
}

// The anchor-side half of compositeCompatibleCandidate. About scale² full-
// resolution pixels test each display anchor, so its own checks, world
// position and geometric normal are resolved once here. Position w is one
// for a compatible anchor.
[shader("compute")]
[numthreads(8, 8, 1)]
void compositeAnchors(uint3 thread : SV_DispatchThreadID) {
    if (any(thread.xy >= giResolveExtent(params))) return;
    int2 samplePixel = int2(thread.xy);
    float4 anchor = 0.0;
    float3 rasterNormal = 0.0;
    float4 candidate = irradiance.Load(int3(samplePixel, 0));
    float4 receiver = receiverHistory.Load(int3(samplePixel, 0));
    // Alpha remains receiver depth, not a lighting-validity bit. Receiver
    // metadata carries support after temporal resolve and any denoise pass.
    if (candidate.w > 0.0 && isfinite(candidate.w) && all(isfinite(candidate.xyz))
        && receiverHistoryLength(receiver) != 0u
        && receiverScreenHistorySupportMask(receiver) != 0u) {
        uint scale = giResolveScale(params);
        uint2 sampleLocal = min(uint2(samplePixel) * scale + scale / 2u,
                                params.viewport.zw - 1u);
        uint2 full = params.viewport.xy + sampleLocal;
        uint4 sampleRaster = rasterReceiver.Load(int3(full, 0));
        uint4 sampleDeferred = gbuffer.Load(int3(full, 0));
        // A target that matches the metadata identity then matches the raster.
        if ((sampleRaster.z & 0x80000000u) != 0u
            && all(sampleRaster.xy == asuint(receiver.xy))
            && (sampleDeferred.w & (1u << 24u)) == 0u) {
            float3 sampleNormal = receiverNormal(receiver);
            rasterNormal = receiverGeometricNormal(sampleRaster.z,
                deferredShadingNormal(sampleDeferred));
            float3 samplePosition = worldPosition(float2(full) + 0.5, candidate.w, params);
            if (dot(sampleNormal, rasterNormal) > 0.97 && all(isfinite(samplePosition)))
                anchor = float4(samplePosition, 1.0);
        }
    }
    anchorPositionOutput.Store(samplePixel, anchor);
    anchorNormalOutput.Store(samplePixel, float4(rasterNormal, 0.0));
}

bool compositeCompatibleCandidate(int2 samplePixel, uint2 targetIdentity,
                                  float3 targetPosition, float3 targetNormal,
                                  float tolerance, out float3 value) {
    value = 0.0;
    uint2 extent = giResolveExtent(params);
    if (any(samplePixel < 0) || any(samplePixel >= int2(extent))) return false;
    // compositeAnchors resolved the anchor-side checks, position and normal.
    float4 anchor = anchorPosition.Load(int3(samplePixel, 0));
    if (anchor.w == 0.0) return false;
    float4 receiver = receiverHistory.Load(int3(samplePixel, 0));
    if (any(asuint(receiver.xy) != targetIdentity)) return false;
    float3 sampleNormal = receiverNormal(receiver);
    float3 rasterNormal = anchorNormal.Load(int3(samplePixel, 0)).xyz;
    if (dot(targetNormal, sampleNormal) <= 0.97
        || dot(targetNormal, rasterNormal) <= 0.97
        || surfacePlaneDistance(anchor.xyz - targetPosition, targetNormal, rasterNormal) > tolerance)
        return false;
    value = irradiance.Load(int3(samplePixel, 0)).xyz;
    return true;
}

float3 reconstructCompositeGi(uint2 pixel, float depth, uint4 deferred) {
    uint4 targetRaster = rasterReceiver.Load(int3(pixel, 0));
    if (!(depth > 0.0) || !isfinite(depth)
        || (targetRaster.z & 0x80000000u) == 0u
        || (deferred.w & (1u << 24u)) != 0u) return 0.0;
    float3 normal = receiverGeometricNormal(targetRaster.z, deferredShadingNormal(deferred));
    float3 position = worldPosition(float2(pixel) + 0.5, depth, params);
    if (!all(isfinite(position))) return 0.0;

    uint scale = giResolveScale(params);
    uint2 extent = giResolveExtent(params);
    float2 local = float2(pixel - params.viewport.xy);
    float2 grid = (local - float2(scale / 2u)) / float(scale);
    int2 base = int2(floor(grid));
    // Resolve samples use integer raster anchors, including a clamped final
    // partial tile. Invert those actual anchors rather than a uniform grid.
    float2 last = float2(params.viewport.zw - 1u);
    float2 left = min(float2(base * int(scale) + int(scale / 2u)), last);
    float2 right = min(float2((base + 1) * int(scale) + int(scale / 2u)), last);
    float2 fraction = saturate((local - left) / max(right - left, 1.0));
    float tolerance = surfaceTolerance(params);
    float3 sum = 0.0;
    float support = 0.0;
    // Bilinear taps that were evaluated and rejected; the test is deterministic.
    uint rejectedTaps = 0u;
    [unroll] for (uint tap = 0u; tap < 4u; ++tap) {
        int2 bit = int2(tap & 1u, tap >> 1u);
        int2 samplePixel = base + bit;
        float2 axisWeight = lerp(1.0 - fraction, fraction, float2(bit));
        float weight = axisWeight.x * axisWeight.y;
        if (!(weight > 0.0)) continue;
        float3 candidate;
        if (!compositeCompatibleCandidate(samplePixel, targetRaster.xy,
                                          position, normal, tolerance, candidate)) {
            rejectedTaps |= 1u << tap;
            continue;
        }
        sum += candidate * weight;
        support += weight;
    }
    // Bilinear anchors can all land on adjacent facets around a thin receiver.
    // Only when no compatible bilinear support exists, use the nearest supported
    // sample within two resolve cells; a supported black result remains black.
    if (support <= 1e-6) {
        float nearestDistanceSquared = 4.0;
        float3 nearest = 0.0;
        bool found = false;
        [loop] for (int y = -2; y <= 2; ++y) {
            [loop] for (int x = -2; x <= 2; ++x) {
                int2 samplePixel = base + int2(x, y);
                if (any(samplePixel < 0) || any(samplePixel >= int2(extent))) continue;
                uint2 sampleLocal = min(uint2(samplePixel) * scale + scale / 2u,
                                        params.viewport.zw - 1u);
                float2 delta = (float2(sampleLocal) - local) / float(scale);
                float distanceSquared = dot(delta, delta);
                if (distanceSquared > nearestDistanceSquared) continue;
                if (uint(x) <= 1u && uint(y) <= 1u
                    && (rejectedTaps & (1u << (uint(x) + 2u * uint(y)))) != 0u) continue;
                float3 candidate;
                if (!compositeCompatibleCandidate(samplePixel, targetRaster.xy,
                                                  position, normal, tolerance, candidate)) continue;
                nearestDistanceSquared = distanceSquared;
                nearest = candidate;
                found = true;
            }
        }
        if (found) {
            sum = nearest;
            support = 1.0;
        }
    }
    if (support > 1e-6) return sum / support;
    // A facet smaller than the resolve grid can have no compatible anchor at
    // all. Resolve its actual shading normal from the live world probes.
    float3 shadingNormal = deferredShadingNormal(deferred);
    if (!all(isfinite(shadingNormal)) || dot(shadingNormal, shadingNormal) <= 0.01)
        return 0.0;
    float coverage, backgroundCoverage, anyCoverage;
    float3 background;
    float3 fallback = resolveSplitDiffuse(position, normalize(shadingNormal),
        coverage, background, backgroundCoverage, anyCoverage);
    return anyCoverage > 0.0 && all(isfinite(fallback)) ? fallback : 0.0;
}

// Ray-traced GI (idTech 8's bilateral upscale): bilinear taps of the
// temporal SH, weighted by plane distance and geometric-normal similarity,
// evaluated with this pixel's shading normal so normal-map detail survives
// the reduced gather resolution. A pixel no tap supports takes the nearest
// supported sample within two resolve cells, then the irradiance volumes.
float3 reconstructRtgi(uint2 pixel, float depth, uint4 deferred) {
    uint4 targetRaster = rasterReceiver.Load(int3(pixel, 0));
    if (!(depth > 0.0) || !isfinite(depth)
        || (targetRaster.z & 0x80000000u) == 0u
        || (deferred.w & (1u << 24u)) != 0u) return 0.0;
    float3 shading = deferredShadingNormal(deferred);
    if (!all(isfinite(shading)) || dot(shading, shading) <= 0.01) return 0.0;
    shading = normalize(shading);
    float3 normal = receiverGeometricNormal(targetRaster.z, shading);
    float3 position = worldPosition(float2(pixel) + 0.5, depth, params);
    if (!all(isfinite(position))) return 0.0;

    uint scale = giResolveScale(params);
    uint2 extent = giResolveExtent(params);
    float2 local = float2(pixel - params.viewport.xy);
    float2 grid = (local - float2(scale / 2u)) / float(scale);
    int2 base = int2(floor(grid));
    // Resolve samples use integer raster anchors, including a clamped final
    // partial tile. Invert those actual anchors rather than a uniform grid.
    float2 last = float2(params.viewport.zw - 1u);
    float2 left = min(float2(base * int(scale) + int(scale / 2u)), last);
    float2 right = min(float2((base + 1) * int(scale) + int(scale / 2u)), last);
    float2 fraction = saturate((local - left) / max(right - left, 1.0));
    uint parity = params.frame.x & 1u;
    float tolerance = max(length(position - params.cameraPosition.xyz) * 0.01, 0.02);
    RtgiSh sum = rtgiShZero();
    float support = 0.0;
    [unroll] for (uint tap = 0u; tap < 4u; ++tap) {
        int2 bit = int2(tap & 1u, tap >> 1u);
        int2 samplePixel = base + bit;
        if (any(samplePixel < 0) || any(samplePixel >= int2(extent))) continue;
        uint history = rtgiHistoryBase(parity, uint(samplePixel.x) + uint(samplePixel.y) * extent.x);
        float4 anchor = rtgiField[history + 3u];
        if (!(anchor.w > 0.0)) continue;
        float facing = saturate(dot(rtgiField[history + 4u].xyz, normal));
        facing *= facing; facing *= facing;
        float2 axisWeight = lerp(1.0 - fraction, fraction, float2(bit));
        float weight = axisWeight.x * axisWeight.y * facing * facing
            * saturate(1.0 - abs(dot(anchor.xyz - position, normal)) / tolerance);
        if (!(weight > 1e-5)) continue;
        rtgiShAccumulate(sum, rtgiLoadSh(history), weight);
        support += weight;
    }
    if (support > 1e-5) return rtgiShIrradiance(rtgiShScale(sum, 1.0 / support), shading);
    float nearestDistanceSquared = 8.0;
    uint nearest = EMPTY;
    [loop] for (int y = -2; y <= 3; ++y) {
        [loop] for (int x = -2; x <= 3; ++x) {
            int2 samplePixel = base + int2(x, y);
            if (any(samplePixel < 0) || any(samplePixel >= int2(extent))) continue;
            uint history = rtgiHistoryBase(parity, uint(samplePixel.x) + uint(samplePixel.y) * extent.x);
            float4 anchor = rtgiField[history + 3u];
            if (!(anchor.w > 0.0)
                || abs(dot(anchor.xyz - position, normal)) > tolerance
                || dot(rtgiField[history + 4u].xyz, normal) <= 0.0) continue;
            float2 delta = (float2(samplePixel) - grid);
            float distanceSquared = dot(delta, delta);
            if (distanceSquared >= nearestDistanceSquared) continue;
            nearestDistanceSquared = distanceSquared;
            nearest = history;
        }
    }
    if (nearest != EMPTY) return rtgiShIrradiance(rtgiLoadSh(nearest), shading);
    float3 surface, sky;
    // Only bounced light: Bevy's environment light already lights this pixel.
    if (rtgiVolumeIrradiance(position, shading, surface, sky)) return surface;
    return 0.0;
}

float4 composeGi(uint2 pixel, uint4 deferred, float3 diffuseGi) {
    float3 diffuseOcclusion = 1.0;
    if ((params.layers.z & 1u) != 0u) {
        diffuseOcclusion = float((deferred.z >> 16u) & 255u) / 255.0;
        if ((params.layers.z & 2u) != 0u) {
            float ssao = saturate(screenSpaceOcclusion.Load(int3(pixel, 0)));
            float3 baseColor = deferredBaseColor(deferred);
            float3 a = 2.0404 * baseColor - 0.3324;
            float3 b = -4.7951 * baseColor + 0.6417;
            float3 c = 2.7552 * baseColor + 0.6903;
            diffuseOcclusion = min(diffuseOcclusion,
                max(float3(ssao), ((ssao * a + b) * ssao + c) * ssao));
        }
    }
    float3 albedo = deferredDiffuseAlbedo(deferred);
    return float4(diffuseGi * albedo * diffuseOcclusion
        * params.tuning.w * params.environment.w, 0.0);
}

[shader("fragment")]
float4 fragment(VertexOutput input) : SV_Target {
    uint2 pixel = uint2(input.position.xy);
    if (any(pixel < params.viewport.xy)
        || any(pixel >= params.viewport.xy + params.viewport.zw)) return 0.0;
    float depth = depthTexture.Load(int3(pixel, 0));
    uint4 deferred = gbuffer.Load(int3(pixel, 0));
    if (!(depth > 0.0) || (deferred.w & (1u << 24u)) != 0u) return 0.0;
    return composeGi(pixel, deferred, reconstructCompositeGi(pixel, depth, deferred));
}

[shader("fragment")]
float4 rtgiFragment(VertexOutput input) : SV_Target {
    uint2 pixel = uint2(input.position.xy);
    if (any(pixel < params.viewport.xy)
        || any(pixel >= params.viewport.xy + params.viewport.zw)) return 0.0;
    float depth = depthTexture.Load(int3(pixel, 0));
    uint4 deferred = gbuffer.Load(int3(pixel, 0));
    if (!(depth > 0.0) || (deferred.w & (1u << 24u)) != 0u) return 0.0;
    return composeGi(pixel, deferred, reconstructRtgi(pixel, depth, deferred));
}

// Bevy's directional shadow lookup reads this layer at the shaded pixel for a
// light whose shadow is ray traced (see sun_shadow_patch.rs) and uses the
// stored depth as the light's visibility. Each pixel's visibility is a 5x5
// tent filter of sunShadowsHardware's disk-jittered rays, restricted to
// neighbours on the pixel's surface plane, so shadow edges are antialiased
// without bleeding light across depth discontinuities.
bool sunShadowOccludedAt(int2 local) {
    float4 mask = sunShadowMask.Load(int3(sunShadowTexel(uint2(local)), 0));
    return (sunShadowBits(mask, 0u) & (1u << sunShadowBit(uint2(local)))) != 0u;
}

float sunShadowVisibility(uint2 pixel) {
    int2 local = int2(pixel - params.viewport.xy);
    float visibility = sunShadowOccludedAt(local) ? 0.0 : 1.0;
    float depth = depthTexture.Load(int3(pixel, 0));
    if (!(depth > 0.0)) return visibility;
    float3 position = worldPosition(float2(pixel) + 0.5, depth, params);
    float3 normal = receiverGeometricNormal(rasterReceiver.Load(int3(pixel, 0)).z,
        deferredShadingNormal(gbuffer.Load(int3(pixel, 0))));
    // Depth reconstruction error grows with distance; allow a pixel's slope.
    float tolerance = 0.002 * length(position - params.cameraPosition.xyz) + 0.01;
    int2 extent = int2(params.viewport.zw);
    float sum = 0.0, total = 0.0;
    [unroll] for (int y = -2; y <= 2; ++y) {
        [unroll] for (int x = -2; x <= 2; ++x) {
            int2 tap = local + int2(x, y);
            if (any(tap < 0) || any(tap >= extent)) continue;
            uint2 tapPixel = params.viewport.xy + uint2(tap);
            float tapDepth = depthTexture.Load(int3(tapPixel, 0));
            if (!(tapDepth > 0.0)) continue;
            float3 tapPosition = worldPosition(float2(tapPixel) + 0.5, tapDepth, params);
            if (abs(dot(tapPosition - position, normal)) > tolerance) continue;
            float weight = float((3 - abs(x)) * (3 - abs(y)));
            sum += sunShadowOccludedAt(tap) ? 0.0 : weight;
            total += weight;
        }
    }
    return total > 0.0 ? sum / total : visibility;
}

// Slang's WGSL backend maps SV_Depth to frag_depth only on a struct field.
struct SunShadowDepthOutput {
    float visibility : SV_Depth;
};

[shader("fragment")]
SunShadowDepthOutput sunShadowDepth(VertexOutput input) {
    SunShadowDepthOutput output;
    output.visibility = sunShadowVisibility(uint2(input.position.xy));
    return output;
}

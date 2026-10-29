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

bool compositeCompatibleCandidate(int2 samplePixel, uint2 targetIdentity,
                                  float3 targetPosition, float3 targetNormal,
                                  float tolerance, out float3 value) {
    value = 0.0;
    uint2 extent = giResolveExtent(params);
    if (any(samplePixel < 0) || any(samplePixel >= int2(extent))) return false;
    uint scale = giResolveScale(params);
    float4 candidate = irradiance.Load(int3(samplePixel, 0));
    // Alpha remains receiver depth, not a lighting-validity bit. Receiver
    // metadata carries support after temporal resolve and any denoise pass.
    if (!(candidate.w > 0.0) || !isfinite(candidate.w)
        || !all(isfinite(candidate.xyz))) return false;
    float4 receiver = receiverHistory.Load(int3(samplePixel, 0));
    if (receiverHistoryLength(receiver) == 0u
        || receiverScreenHistorySupportMask(receiver) == 0u
        || any(asuint(receiver.xy) != targetIdentity)) return false;
    uint2 sampleLocal = min(uint2(samplePixel) * scale + scale / 2u,
                            params.viewport.zw - 1u);
    uint2 full = params.viewport.xy + sampleLocal;
    uint4 sampleRaster = rasterReceiver.Load(int3(full, 0));
    if ((sampleRaster.z & 0x80000000u) == 0u
        || any(sampleRaster.xy != targetIdentity)) return false;
    uint4 sampleDeferred = gbuffer.Load(int3(full, 0));
    if ((sampleDeferred.w & (1u << 24u)) != 0u) return false;
    float3 sampleNormal = receiverNormal(receiver);
    float3 rasterNormal = receiverGeometricNormal(sampleRaster.z, deferredShadingNormal(sampleDeferred));
    if (dot(targetNormal, sampleNormal) <= 0.97
        || dot(targetNormal, rasterNormal) <= 0.97
        || dot(sampleNormal, rasterNormal) <= 0.97) return false;
    float3 samplePosition = worldPosition(float2(full) + 0.5, candidate.w, params);
    if (!all(isfinite(samplePosition))
        || surfacePlaneDistance(samplePosition - targetPosition, targetNormal, rasterNormal) > tolerance)
        return false;
    value = candidate.xyz;
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
    [unroll] for (uint tap = 0u; tap < 4u; ++tap) {
        int2 bit = int2(tap & 1u, tap >> 1u);
        int2 samplePixel = base + bit;
        float2 axisWeight = lerp(1.0 - fraction, fraction, float2(bit));
        float weight = axisWeight.x * axisWeight.y;
        if (!(weight > 0.0)) continue;
        float3 candidate;
        if (!compositeCompatibleCandidate(samplePixel, targetRaster.xy,
                                          position, normal, tolerance, candidate)) continue;
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

[shader("fragment")]
float4 fragment(VertexOutput input) : SV_Target {
    uint2 pixel = uint2(input.position.xy);
    if (any(pixel < params.viewport.xy)
        || any(pixel >= params.viewport.xy + params.viewport.zw)) return 0.0;
    float depth = depthTexture.Load(int3(pixel, 0));
    uint4 deferred = gbuffer.Load(int3(pixel, 0));
    if (!(depth > 0.0) || (deferred.w & (1u << 24u)) != 0u) return 0.0;

    float3 diffuseGi = reconstructCompositeGi(pixel, depth, deferred);
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

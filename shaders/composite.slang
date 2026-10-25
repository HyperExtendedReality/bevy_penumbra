#include "common.slang"
#include "receiver_metadata.slang"

[[vk::binding(0, 0)]] ConstantBuffer<Parameters> params;
[[vk::binding(2, 0)]] Texture2D<float4> irradiance;
[[vk::binding(3, 0)]] DepthTexture2D depthTexture;
[[vk::binding(5, 0)]] Texture2D<uint4> gbuffer;
[[vk::binding(6, 0)]] Texture2D<float4> receiverHistory;
[[vk::binding(7, 0)]] Texture2D<uint4> rasterReceiver;
[[vk::binding(8, 0)]] Texture2D<float> screenSpaceOcclusion;

struct VertexOutput { float4 position : SV_Position; };

[shader("vertex")]
VertexOutput vertex(uint index : SV_VertexID) {
    VertexOutput output;
    output.position = float4(index == 1 ? 3.0 : -1.0,
                             index == 2 ? 3.0 : -1.0, 0.0, 1.0);
    return output;
}

float3 reconstructCompositeGi(uint2 pixel) {
    float depth = depthTexture.Load(int3(pixel, 0));
    uint4 targetRaster = rasterReceiver.Load(int3(pixel, 0));
    uint4 deferred = gbuffer.Load(int3(pixel, 0));
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
        if (any(samplePixel < 0) || any(samplePixel >= int2(extent))) continue;
        float2 axisWeight = lerp(1.0 - fraction, fraction, float2(bit));
        float weight = axisWeight.x * axisWeight.y;
        if (!(weight > 0.0)) continue;
        float4 candidate = irradiance.Load(int3(samplePixel, 0));
        // Alpha remains receiver depth, not a lighting-validity bit. Receiver
        // metadata carries support after temporal resolve and any denoise pass.
        if (!(candidate.w > 0.0) || !isfinite(candidate.w)
            || !all(isfinite(candidate.xyz))) continue;
        float4 receiver = receiverHistory.Load(int3(samplePixel, 0));
        if (receiverHistoryLength(receiver) == 0u
            || receiverScreenHistorySupportMask(receiver) == 0u
            || any(asuint(receiver.xy) != targetRaster.xy)) continue;
        uint2 sampleLocal = min(uint2(samplePixel) * scale + scale / 2u,
                                params.viewport.zw - 1u);
        uint2 full = params.viewport.xy + sampleLocal;
        uint4 sampleRaster = rasterReceiver.Load(int3(full, 0));
        if ((sampleRaster.z & 0x80000000u) == 0u
            || any(sampleRaster.xy != targetRaster.xy)) continue;
        uint4 sampleDeferred = gbuffer.Load(int3(full, 0));
        if ((sampleDeferred.w & (1u << 24u)) != 0u) continue;
        float3 sampleNormal = receiverNormal(receiver);
        float3 rasterNormal = receiverGeometricNormal(sampleRaster.z, deferredShadingNormal(sampleDeferred));
        if (dot(normal, sampleNormal) <= 0.97
            || dot(normal, rasterNormal) <= 0.97
            || dot(sampleNormal, rasterNormal) <= 0.97) continue;
        float3 samplePosition = worldPosition(float2(full) + 0.5, candidate.w, params);
        if (!all(isfinite(samplePosition))
            || surfacePlaneDistance(samplePosition - position, normal, rasterNormal) > tolerance) continue;
        sum += candidate.xyz * weight;
        support += weight;
    }
    return support > 1e-6 ? sum / support : 0.0;
}

[shader("fragment")]
float4 fragment(VertexOutput input) : SV_Target {
    uint2 pixel = uint2(input.position.xy);
    if (any(pixel < params.viewport.xy)
        || any(pixel >= params.viewport.xy + params.viewport.zw)) return 0.0;
    float depth = depthTexture.Load(int3(pixel, 0));
    uint4 deferred = gbuffer.Load(int3(pixel, 0));
    if (!(depth > 0.0) || (deferred.w & (1u << 24u)) != 0u) return 0.0;

    float3 diffuseGi = reconstructCompositeGi(pixel);
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

#include "common.slang"

[[vk::binding(0, 0)]] ConstantBuffer<Parameters> params;
[[vk::binding(1, 0)]] Texture2D<float4> sourceTexture;
[[vk::binding(2, 0)]] Texture2D<float4> irradiance;
[[vk::binding(3, 0)]] DepthTexture2D depthTexture;
[[vk::binding(4, 0)]] Texture2D<float4> normalTexture;
[[vk::binding(5, 0)]] Texture2D<uint4> gbuffer;
[[vk::binding(7, 0)]] Texture2D<uint4> receiverIdentity;
[[vk::binding(8, 0)]] Texture2D<float> screenSpaceOcclusion;
[[vk::binding(9, 0)]] Texture2D<uint> surfaceGroups;

struct VertexOutput { float4 position : SV_Position; };

[shader("vertex")]
VertexOutput vertex(uint index : SV_VertexID) {
    VertexOutput output;
    output.position = float4(index == 1 ? 3.0 : -1.0, index == 2 ? 3.0 : -1.0, 0.0, 1.0);
    return output;
}

[shader("fragment")]
float4 fragment(VertexOutput input) : SV_Target {
    uint2 pixel = uint2(input.position.xy);
    float4 source = sourceTexture.Load(int3(pixel, 0));
    // Copy the complete source when flipping Bevy's ping-pong target, including
    // pixels outside this camera's viewport (split-screen/shared render targets).
    if (any(pixel < params.viewport.xy) || any(pixel >= params.viewport.xy + params.viewport.zw)) return source;
    float depth = depthTexture.Load(int3(pixel, 0));
    uint4 receiver = receiverIdentity.Load(int3(pixel,0));
    uint4 deferred = gbuffer.Load(int3(pixel, 0));
    if (depth <= 0.0 || receiver.z == 0) return source;
    if (((deferred.x & 0xffffffu) == 0 || ((deferred.z >> 8) & 255u) == 255u)) return source;
    float3 normal = normalize(normalTexture.Load(int3(pixel, 0)).xyz * 2.0 - 1.0);
    float3 shadingNormal = deferredShadingNormal(deferred);
    float3 position = worldPosition(input.position.xy, depth, params);
    float2 grid = irradianceGrid(input.position.xy, params);
    int2 base = int2(floor(grid)); float2 fraction = frac(grid);
    float3 sum = 0.0; float total = 0.0;
    uint selectedSurfaceGroup = 0xffffffffu; float nearestSurface = 3.402823e38;
    if (params.layers.w != 0) for (uint corner = 0; corner < 4; ++corner) {
        uint2 bit = uint2(corner & 1, corner >> 1);
        int2 low = clamp(base + int2(bit), int2(0), int2(params.size.xy) - 1);
        float4 value = irradiance.Load(int3(low, 0));
        if (value.w <= 0.0) continue;
        uint2 full = fullPixel(uint2(low), params);
        uint4 otherReceiver = receiverIdentity.Load(int3(full,0));
        if (otherReceiver.z == 0 || any(otherReceiver.xy != receiver.xy)) continue;
        float3 other = worldPosition(float2(full) + 0.5, value.w, params);
        float3 otherNormal = normalize(normalTexture.Load(int3(full, 0)).xyz * 2.0 - 1.0);
        if (dot(normal, otherNormal) < 0.95 || abs(dot(position - other, normal)) > surfaceTolerance(params)) continue;
        float distance = dot(position - other, position - other);
        if (distance < nearestSurface) {
            nearestSurface = distance;
            selectedSurfaceGroup = surfaceGroups.Load(int3(low,0));
        }
    }
    for (uint corner = 0; corner < 4; ++corner) {
        uint2 bit = uint2(corner & 1, corner >> 1);
        int2 low = clamp(base + int2(bit), int2(0), int2(params.size.xy) - 1);
        float4 value = irradiance.Load(int3(low, 0));
        if (value.w <= 0.0) continue;
        uint2 full = fullPixel(uint2(low), params);
        if (params.layers.w != 0 && surfaceGroups.Load(int3(low,0)) != selectedSurfaceGroup) continue;
        uint4 otherReceiver = receiverIdentity.Load(int3(full,0));
        if (otherReceiver.z == 0 || any(otherReceiver.xy != receiver.xy)) continue;
        float3 other = worldPosition(float2(full) + 0.5, value.w, params);
        float3 otherNormal = normalize(normalTexture.Load(int3(full, 0)).xyz * 2.0 - 1.0);
        if (dot(normal, otherNormal) < 0.95 || abs(dot(position - other, normal)) > surfaceTolerance(params)) continue;
        float2 weights = lerp(1.0 - fraction, fraction, float2(bit));
        float weight = weights.x * weights.y * pow(max(0.0, dot(normal, otherNormal)), 16.0)
                       * exp(-length(position - other) / max(params.tuning.x, 0.001));
        sum += value.xyz * weight;
        total += weight;
    }
    // Bevy 0.19's deferred base color uses gamma 2.2 (pbr_deferred_functions).
    float3 baseColor = deferredBaseColor(deferred);
    float3 diffuseOcclusion = 1.0;
    if ((params.layers.z & 1u) != 0) {
        diffuseOcclusion = float((deferred.z >> 16) & 255u) / 255.0;
        if ((params.layers.z & 2u) != 0) {
            float ssao = saturate(screenSpaceOcclusion.Load(int3(pixel,0)));
            // Bevy 0.19.1 ssao_multibounce and deferred_lighting conventions.
            float3 a = 2.0404 * baseColor - 0.3324;
            float3 b = -4.7951 * baseColor + 0.6417;
            float3 c = 2.7552 * baseColor + 0.6903;
            diffuseOcclusion = min(diffuseOcclusion, max(float3(ssao), ((ssao*a+b)*ssao+c)*ssao));
        }
    }
    float3 albedo = deferredDiffuseAlbedo(deferred);
    float3 gi = total > 0.00001 ? sum / total : float3(0.0);
    gi *= albedo;
    gi *= diffuseOcclusion;
    return float4(source.xyz + gi * params.tuning.w * params.environment.w, source.w);
}

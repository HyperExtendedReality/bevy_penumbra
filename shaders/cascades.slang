#include "common.slang"
#include "receiver_metadata.slang"

[[vk::binding(0, 0)]] ConstantBuffer<Parameters> params;
[[vk::binding(1, 0)]] StructuredBuffer<uint4> geometryWords;
// TLAS, immutable mesh BLASes and instances share one binding. Integer records
// preserve bit patterns; transforms/material values are decoded explicitly.
[[vk::binding(2, 0)]] StructuredBuffer<uint4> sceneWords;
[[vk::binding(3, 0)]] StructuredBuffer<Light> lights;
[[vk::binding(4, 0)]] DepthTexture2D depthTexture;
[[vk::binding(5, 0)]] Texture2D<float4> normalTexture;
[[vk::binding(6, 0)]] RWStructuredBuffer<Atomic<uint>> heads;
[[vk::binding(7, 0)]] RWStructuredBuffer<Entry> entries;
[[vk::binding(8, 0)]] StructuredBuffer<uint> previousHeads;
[[vk::binding(9, 0)]] RWStructuredBuffer<Entry> previousEntries;
[[vk::binding(10, 0)]] [format("rgba32f")] WTexture2D<float4> irradiance;
[[vk::binding(11, 0)]] Texture2D<float4> previousIrradiance;
[[vk::binding(12, 0)]] RWStructuredBuffer<IrradianceProbe> irradianceProbes;
[[vk::binding(13, 0)]] Texture2D<float4> currentIrradiance;
[[vk::binding(14, 0)]] [format("rgba32f")] WTexture2D<float4> historyMetadata;
[[vk::binding(15, 0)]] Texture2D<float4> previousMetadata;
[[vk::binding(16, 0)]] Texture2DArray<float4> materialAtlas;
[[vk::binding(20, 0)]] Texture2D<float4> motionTexture;
[[vk::binding(23, 0)]] [format("rgba32f")] WTexture2D<float4> receiverHistory;
[[vk::binding(25, 0)]] Texture2D<float4> previousReceiverHistory;
[[vk::binding(26, 0)]] Texture2D<uint4> rasterReceiver;
[[vk::binding(27, 0)]] Texture2D<uint4> gbuffer;
[[vk::binding(0, 1)]] [format("r32ui")] WTexture2D<uint> surfaceGroupOutput;
[[vk::binding(1, 1)]] Texture2D<uint> surfaceGroups;
[[vk::binding(2, 1)]] Texture2D<uint> previousSurfaceGroups;

#include "traversal.slang"
#include "environment.slang"
#include "split.slang"

// Next-event estimation of emissive surfaces. Bright, small sources no longer
// depend on a hemisphere ray accidentally reaching the emitter. Power-weighted
// CDF selection is logarithmic in emitter count and uses the existing light buffer.
float3 emissiveIrradiance(float3 position, float3 normal, float3 geometricNormal, uint count) {
    if (params.lightCounts.y == 0) return 0;
    float3 sum = 0;
    // Independent world-space rotations prevent an entire wall sharing the
    // same noisy emitter sample and pulsing together from frame to frame.
    uint seed = hashKey(int4(int3(floor(position / (params.tuning.x * 0.05))), 0), 16777216);
    float2 rotation = r2(seed);
    // Share light selection within a probe for GPU coherence, but give probes
    // independent phases so changes in sampled light counts cannot pulse a wall.
    uint selectionSeed = hashKey(cacheKey(position, 0, 0, params), 65536);
    float phase = frac(r2(params.frame.x).x + r2(selectionSeed).x);
    // Sample the emitter distribution directly. This unbiased estimator needs
    // only visibility rays; mixing in a second material-tested hemisphere trace
    // duplicated the full traversal call graph in every receiver gather shader.
    uint lightSamples = count;
    [loop] for (uint sample = 0; sample < count; ++sample) {
        float2 uv = frac(r2(sample + params.frame.x * count + 1) + rotation);
        float selection = frac((float(sample) + 0.5) / float(lightSamples) + phase);
        uint low = 0, high = params.lightCounts.y - 1;
        [loop] while (low < high) {
            uint middle = (low + high) / 2;
            if (lights[params.lightCounts.x + middle].direction.w < selection) low = middle + 1;
            else high = middle;
        }
        Light light = lights[params.lightCounts.x + low];
        if (!instanceVisible(light.source.x)) continue;
        if (light.edge1.w <= 0 || (light.position.w != -2 && dot(light.direction.xyz, position - light.position.xyz) <= 0)) continue;
        float root = sqrt(uv.x);
        float3 target = light.position.xyz + light.edge1.xyz * (root * (1 - uv.y))
                        + light.edge2.xyz * (root * uv.y);
        float3 toLight = target - position;
        float lightDistance = length(toLight);
        if (lightDistance <= params.tuning.z * 0.25) continue;
        float3 lightDirection = toLight / lightDistance;
        float cosinePdf = max(0.0, dot(normal, lightDirection)) / PI;
        float emitterCosine = max(0.0, sourceCosine(light, lightDirection));
        if (cosinePdf <= 0 || emitterCosine <= 0 || dot(geometricNormal, lightDirection) <= 0) continue;
        float lightPdf = light.edge1.w * lightDistance * lightDistance / (light.color.w * emitterCosine);
        float pdf = lightPdf;
        // Evaluate irradiance at the real surface. Reduce only the visibility
        // ray's offset near the light so bias cannot move it across the emitter.
        float planeDistance = abs(dot(light.direction.xyz, position - light.position.xyz));
        float3 origin = rayOffset(position, geometricNormal, min(params.tuning.z, planeDistance * 0.25));
        float3 delta = target - origin;
        float distance = length(delta);
        if (distance <= params.tuning.z * 0.25) continue;
        float3 direction = delta / distance;
        if (traceOccluded(origin, direction, distance - min(params.tuning.z * 0.25, distance * 0.001))) continue;
        sum += emitterRadiance(light, float3(1-root, root*(1-uv.y), root*uv.y)) * (cosinePdf / max(pdf, 1e-20));
    }
    return sum / float(count);
}
float3 emissiveIrradiance(float3 position, float3 normal, uint count) {
    return emissiveIrradiance(position, normal, normal, count);
}

float3 punctualDiffuse(float3 position, float3 normal, float3 geometricNormal) {
    float3 offset = rayOffset(position, geometricNormal, params.tuning.z);
    float3 diffuse = 0;
    [loop] for (uint i = 0; i < params.frame.w; ++i) {
        Light light = lights[i];
        float3 toLight = light.position.xyz; float distance = 1e20, attenuation = 1.0;
        if (light.position.w > 0) {
            toLight -= position; distance = length(toLight);
            if (distance >= light.position.w || distance < params.tuning.z) continue;
            toLight /= distance;
            float rangeFade = saturate(1.0 - pow(distance / light.position.w, 4.0));
            // Match Bevy's diffuse inverse-square falloff; radius is specular.
            attenuation = rangeFade * rangeFade / max(distance * distance, 0.0001);
            if (light.direction.w > 0) {
                float cone = saturate(dot(light.direction.xyz, -toLight) * light.direction.w + light.color.w);
                attenuation *= cone * cone;
            }
        }
        float cosine = max(0.0, dot(normal, toLight));
        if (cosine > 0 && dot(geometricNormal, toLight) > 0 && attenuation > 0 && !traceOccluded(offset, toLight, distance - params.tuning.z))
            diffuse += light.color.xyz * (attenuation * cosine / PI);
    }
    return diffuse;
}
float3 punctualDiffuse(float3 position, float3 normal) {
    return punctualDiffuse(position, normal, normal);
}

float3 outgoingRadiance(float3 origin, float3 direction, Hit hit) {
    if (hit.triangle == EMPTY) return environmentRadiance(direction);
    Triangle triangle = worldTriangle(hit);
    if (all(triangle.albedo.xyz <= 0)) return 0;
    float3 position = origin + direction * hit.distance;
    HitAttributes attributes = hitAttributes(hit);
    float3 geometricNormal = attributes.geometricNormal;
    if (dot(geometricNormal, direction) > 0) geometricNormal = -geometricNormal;
    float3 normal = hitShadingNormal(hit, attributes, direction);
    if (dot(normal, -direction) <= 0 || dot(normal, geometricNormal) <= 0) return 0;
    float3 offset = rayOffset(position, geometricNormal, params.tuning.z);
    float3 diffuse = emissiveIrradiance(position, normal, geometricNormal, max(1u, params.lightCounts.z / 2))
                   + punctualDiffuse(position, normal, geometricNormal);
    if (params.history.y > 0 && params.history.z > 0)
        diffuse += diffuseGather(offset, normal, true, true, hitInstanceIdentity(hit), triangleSurfaceGroup(hit.triangle)) * params.history.y;
    // Emission is sampled explicitly at every receiver; including it here too
    // would count direct emitter paths twice. Cascades carry scattered radiance.
    return max(0.0, triangle.albedo.xyz * diffuse);
}

float4 splitRasterMetadata(uint2 full) {
    uint4 raster = rasterReceiver.Load(int3(full,0));
    uint4 material = gbuffer.Load(int3(full,0));
    return float4(asfloat(raster.xy),receiverPackedNormal(deferredShadingNormal(material)),
                  receiverPackedHistory(deferredPerceptualRoughness(material),raster.z != 0 ? 1u : 0u));
}
bool splitReceiverCompatible(float4 a, float4 b) {
    return receiverSame(a,b) && dot(receiverNormal(a),receiverNormal(b)) > 0.97 &&
        abs(receiverRoughness(a)-receiverRoughness(b)) < 0.04;
}
[shader("compute")]
[numthreads(8, 8, 1)]
void temporal(uint3 thread : SV_DispatchThreadID) {
    if (any(thread.xy >= params.size.xy)) return;
    int2 pixel = int2(thread.xy);
    float4 center = currentIrradiance.Load(int3(pixel, 0));
    uint2 full = fullPixel(thread.xy, params);
    uint surfaceGroup = surfaceGroups.Load(int3(pixel,0));
    float4 receiver = splitRasterMetadata(full);
    if (center.w <= 0 || receiverHistoryLength(receiver) == 0) {
        irradiance.Store(pixel, 0); historyMetadata.Store(pixel, 0); receiverHistory.Store(pixel,0); return;
    }
    receiverHistory.Store(pixel,receiver);
    float3 normal = normalize(normalTexture.Load(int3(full, 0)).xyz * 2 - 1);
    float3 position = worldPosition(float2(full) + 0.5, center.w, params);
    float3 mean = 0, square = 0, maximum = center.xyz;
    float count = 0;
    [loop] for (int y = -1; y <= 1; ++y) for (int x = -1; x <= 1; ++x) {
        int2 neighbor = pixel + int2(x, y);
        if (any(neighbor < 0) || any(neighbor >= int2(params.size.xy))) continue;
        float4 value = currentIrradiance.Load(int3(neighbor, 0));
        if (value.w <= 0) continue;
        uint2 otherFull = fullPixel(uint2(neighbor), params);
        if (!splitReceiverCompatible(receiver,splitRasterMetadata(otherFull)) ||
            surfaceGroups.Load(int3(neighbor,0)) != surfaceGroup) continue;
        float3 otherNormal = normalize(normalTexture.Load(int3(otherFull, 0)).xyz * 2 - 1);
        float3 otherPosition = worldPosition(float2(otherFull) + 0.5, value.w, params);
        if (dot(normal, otherNormal) < 0.95 || abs(dot(otherPosition - position, normal)) > surfaceTolerance(params)) continue;
        mean += value.xyz; square += value.xyz * value.xyz; count += 1;
        maximum = max(maximum, value.xyz);
    }
    mean /= max(count, 1); square /= max(count, 1);
    float3 variance = max(0.0, square - mean * mean);
    float3 deviation = sqrt(variance);
    float4 history = 0, metadata = 0; float total = 0;
    float2 previousFull = 0; bool projected = false;
    bool hasMotion = params.stability.z > 0;
    if (hasMotion) {
        float2 motion = motionTexture.Load(int3(full,0)).xy;
        previousFull = float2(full)+0.5-motion*float2(params.viewport.zw);
        projected = all(isfinite(motion));
    } else {
        float4 clip = mul(params.previousClipFromWorld, float4(position, 1));
        float3 ndc = clip.xyz / clip.w;
        float2 uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        previousFull = float2(params.viewport.xy)+uv*float2(params.viewport.zw);
        projected = clip.w > 0 && all(isfinite(uv));
    }
    // Unmatched raster samples are isolated; never borrow old history if their per-frame IDs collide.
    if ((surfaceGroup & 0x80000000u) == 0 && projected && params.stability.x > 0 &&
        params.history.x > 0 && all(previousFull >= float2(params.viewport.xy)) &&
        all(previousFull < float2(params.viewport.xy+params.viewport.zw))) {
        float2 grid = irradianceGrid(previousFull, params);
        int2 base = int2(floor(grid)); float2 fraction = frac(grid);
        [loop] for (uint corner = 0; corner < 4; ++corner) {
                int2 bit = int2(corner & 1, corner >> 1), oldPixel = base + bit;
                if (any(oldPixel < 0) || any(oldPixel >= int2(params.size.xy))) continue;
                float4 old = previousIrradiance.Load(int3(oldPixel, 0));
                float4 meta = previousMetadata.Load(int3(oldPixel, 0));
                float4 oldReceiver = previousReceiverHistory.Load(int3(oldPixel,0));
                if (meta.z <= 0 || old.w <= 0 || !all(isfinite(old.xyz)) ||
                    dot(normal, decodeNormal(meta.xy)) < 0.97 ||
                    !splitReceiverCompatible(receiver,oldReceiver) ||
                    surfaceGroups.Load(int3(pixel,0)) != surfaceGroup ||
                    previousSurfaceGroups.Load(int3(oldPixel,0)) != surfaceGroup) continue;
                if (hasMotion) {
                    // Native prepass includes rigid and deformed previous positions.
                    float expected = asfloat(rasterReceiver.Load(int3(full,0)).w);
                    if (!isfinite(expected) || expected <= 0) continue;
                    float tolerance = max(0.000002,abs(expected)*0.0001);
                    [loop] for (uint axis = 0; axis < 2; ++axis) {
                        int2 neighbor = min(oldPixel+(axis == 0 ? int2(1,0) : int2(0,1)),int2(params.size.xy)-1);
                        float4 neighborReceiver = previousReceiverHistory.Load(int3(neighbor,0));
                        float neighborDepth = previousIrradiance.Load(int3(neighbor,0)).w;
                        if (splitReceiverCompatible(oldReceiver,neighborReceiver) && neighborDepth > 0 &&
                            previousSurfaceGroups.Load(int3(neighbor,0)) == surfaceGroup)
                            tolerance += 2*abs(neighborDepth-old.w);
                    }
                    if (abs(expected-old.w) > tolerance) continue;
                } else {
                    float3 oldPosition = previousWorldPosition(float2(fullPixel(uint2(oldPixel), params)) + 0.5, old.w, params);
                    if (abs(dot(oldPosition - position, normal)) > surfaceTolerance(params)) continue;
                }
                float2 w = lerp(1 - fraction, fraction, float2(bit)); float weight = w.x * w.y;
                history += old * weight; metadata += meta * weight; total += weight;
            }
    }
    float weight = 0; float length = 1;
    if (total > 0.00001) {
        history /= total; metadata /= total;
        // Track temporal variance too: a spatially uniform Monte Carlo sample
        // may fluctuate across the entire plane, so neighborhood min/max alone
        // clamps away useful history and preserves global brightness flicker.
        float delta = luminance(history.xyz) - luminance(center.xyz);
        float temporalVariance = max(metadata.w, delta * delta * 0.25);
        float3 radius = 2.5 * max(deviation, float3(sqrt(temporalVariance))) + 0.0001;
        float3 low = max(0.0, mean - radius), high = mean + radius;
        float3 clipped = clamp(history.xyz, low, high);
        if (luminance(maximum) < 0.00001) clipped = 0;
        float change = abs(luminance(clipped) - luminance(center.xyz))
                       / max(max(luminance(clipped), luminance(center.xyz)), 0.001);
        // Noise in an unchanged scene is precisely what accumulation removes.
        // Use reactive blending for scene edits, not every noisy current sample.
        weight = min(params.history.x, metadata.z / (metadata.z + 1)) * (params.stability.y > 0 ? 1 - change * change : 1);
        center.xyz = lerp(center.xyz, clipped, weight);
        length = min(64.0, metadata.z + 1);
        metadata.w = lerp(luminance(variance), metadata.w, weight) + weight * (1 - weight) * delta * delta;
    }
    irradiance.Store(pixel, center);
    historyMetadata.Store(pixel, float4(encodeNormal(normal), length,
                          total > 0.00001 ? metadata.w : luminance(variance)));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void denoise(uint3 thread : SV_DispatchThreadID) {
    if (any(thread.xy >= params.size.xy)) return;
    int2 pixel = int2(thread.xy);
    float4 center = previousIrradiance.Load(int3(pixel, 0));
    if (center.w <= 0) { irradiance.Store(pixel, center); return; }
    uint2 full = fullPixel(thread.xy, params);
    float4 receiver = splitRasterMetadata(full);
    float3 normal = normalize(normalTexture.Load(int3(full, 0)).xyz * 2 - 1);
    float3 position = worldPosition(float2(full) + 0.5, center.w, params);
    float variance = currentIrradiance.Load(int3(pixel, 0)).w;
    float3 sum = 0; float total = 0;
    uint step = 1u << params.frame.y;
    const float kernel[5] = {1.0, 4.0, 6.0, 4.0, 1.0};
    [loop] for (int y = -2; y <= 2; ++y) for (int x = -2; x <= 2; ++x) {
        int2 otherPixel = pixel + int2(x, y) * int(step);
        if (any(otherPixel < 0) || any(otherPixel >= int2(params.size.xy))) continue;
        float4 value = previousIrradiance.Load(int3(otherPixel, 0));
        if (value.w <= 0) continue;
        uint2 otherFull = fullPixel(uint2(otherPixel), params);
        if (!splitReceiverCompatible(receiver,splitRasterMetadata(otherFull)) ||
            surfaceGroups.Load(int3(pixel,0)) != surfaceGroups.Load(int3(otherPixel,0))) continue;
        float3 otherNormal = normalize(normalTexture.Load(int3(otherFull, 0)).xyz * 2 - 1);
        float3 otherPosition = worldPosition(float2(otherFull) + 0.5, value.w, params);
        float planeDistance = abs(dot(otherPosition - position, normal));
        if (dot(normal, otherNormal) < 0.95 || planeDistance > surfaceTolerance(params)) continue;
        float weight = kernel[x + 2] * kernel[y + 2]
            * pow(max(0.0, dot(normal, otherNormal)), 32.0)
            * exp(-planeDistance / max(params.tuning.z * 2, params.tuning.x * 0.05));
        weight *= exp(-abs(luminance(value.xyz) - luminance(center.xyz)) / max(0.02, 3 * sqrt(max(variance, 0.0))));
        sum += value.xyz * weight; total += weight;
    }
    irradiance.Store(pixel, float4(total > 0 ? sum / total : center.xyz, center.w));
}

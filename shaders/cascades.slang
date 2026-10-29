#include "common.slang"
#include "receiver_metadata.slang"

[[vk::binding(0, 0)]] ConstantBuffer<Parameters> params;
[[vk::binding(1, 0)]] StructuredBuffer<uint4> geometryWords;
// TLAS, immutable mesh BLASes and instances share one binding. Integer records
// preserve bit patterns; transforms/material values are decoded explicitly.
[[vk::binding(2, 0)]] StructuredBuffer<uint4> sceneWords;
[[vk::binding(3, 0)]] StructuredBuffer<Light> lights;
[[vk::binding(4, 0)]] DepthTexture2D depthTexture;
[[vk::binding(6, 0)]] RWStructuredBuffer<Atomic<uint>> heads;
[[vk::binding(7, 0)]] RWStructuredBuffer<Entry> entries;
[[vk::binding(8, 0)]] StructuredBuffer<uint> previousHeads;
[[vk::binding(9, 0)]] RWStructuredBuffer<Entry> previousEntries;
[[vk::binding(10, 0)]] [format("rgba32f")] WTexture2D<float4> irradiance;
[[vk::binding(11, 0)]] Texture2D<float4> previousIrradiance;
[[vk::binding(12, 0)]] RWStructuredBuffer<DirectionalProbe> directionalProbes;
[[vk::binding(13, 0)]] Texture2D<float4> currentIrradiance;
[[vk::binding(14, 0)]] [format("rgba32f")] WTexture2D<float4> historyMetadata;
[[vk::binding(15, 0)]] Texture2D<float4> previousMetadata;
[[vk::binding(16, 0)]] Texture2DArray<float4> materialAtlas;
[[vk::binding(20, 0)]] Texture2D<float4> motionTexture;
[[vk::binding(23, 0)]] [format("rgba32f")] WTexture2D<float4> receiverHistory;
[[vk::binding(25, 0)]] Texture2D<float4> previousReceiverHistory;
[[vk::binding(26, 0)]] Texture2D<uint4> rasterReceiver;
[[vk::binding(27, 0)]] Texture2D<uint4> gbuffer;
[[vk::binding(37, 0)]] Texture2D<float4> dfgTexture;
[[vk::binding(38, 0)]] [format("rgba32f")] WTexture2D<float4> backgroundOutput;
[[vk::binding(39, 0)]] Texture2D<float4> currentBackground;
[[vk::binding(40, 0)]] Texture2D<float4> previousBackground;
float2 bsdfDfg(float roughness, float nDotV) {
    uint width, height; dfgTexture.GetDimensions(width, height);
    float2 pixel = float2(saturate(nDotV), max(saturate(roughness), 0.089)) * float2(width, height) - 0.5;
    int2 low = int2(floor(pixel)); float2 f = frac(pixel);
    int2 last = int2(width, height) - 1;
    return lerp(lerp(dfgTexture.Load(int3(clamp(low, 0, last), 0)).xy,
                     dfgTexture.Load(int3(clamp(low + int2(1,0), 0, last), 0)).xy, f.x),
                lerp(dfgTexture.Load(int3(clamp(low + int2(0,1), 0, last), 0)).xy,
                     dfgTexture.Load(int3(clamp(low + 1, 0, last), 0)).xy, f.x), f.y);
}
#include "bsdf.slang"
#include "traversal.slang"
SurfaceBsdf hitSurfaceBsdf(Hit hit, float3 baseColor) {
    float4 properties = asfloat(sceneWord(materialBase(hit) + 2u));
    float4 metalRough = materialSample(hit, 2u);
    SurfaceBsdf material;
    material.baseColor = baseColor;
    material.metallic = saturate(properties.x * metalRough.b);
    material.perceptualRoughness = saturate(properties.y * metalRough.g);
    material.reflectance = properties.z;
    return material;
}
#include "environment.slang"
#include "split.slang"
#include "world_resolve.slang"

// Next-event estimation of emissive surfaces. Bright, small sources no longer
// depend on a hemisphere ray accidentally reaching the emitter. Power-weighted
// CDF selection is logarithmic in emitter count and uses the existing light buffer.
float3 emissiveResponseTracked(float3 position, float3 normal, float3 geometricNormal,
                          float3 view, float roughness, uint count,
                          uint sequenceBase, out bool querySupported,
                          inout DependencyContext dependency, uint admissionLimit,
                          bool fullBsdf, SurfaceBsdf material, float2 ab) {
    querySupported = true;
    if (count == 0 || params.lightCounts.y == 0) return 0;
    float3 sum = 0;
    // Independent world-space rotations prevent an entire wall sharing the
    // same noisy emitter sample and pulsing together from frame to frame.
    uint seed = hashKey(int4(int3(floor(position / (params.tuning.x * 0.05))), 0), 16777216);
    float2 rotation = r2(seed);
    // Share light selection within a probe for GPU coherence, but give probes
    // independent phases so changes in sampled light counts cannot pulse a wall.
    uint selectionSeed = hashKey(cacheKey(position, 0, 0, params), 65536);
    uint sampleFrame = sequenceBase;
    float phase = frac(r2(sampleFrame).x + r2(selectionSeed).x);
    // Sample the emitter distribution directly. This unbiased estimator needs
    // only visibility rays; mixing in a second material-tested hemisphere trace
    // duplicated the full traversal call graph in every receiver gather shader.
    uint lightSamples = count;
    [loop] for (uint sample = 0; sample < count; ++sample) {
        float2 uv = frac(r2(sample + sampleFrame * count + 1) + rotation);
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
        float cosine = max(0.0, dot(normal, lightDirection));
        float emitterCosine = max(0.0, sourceCosine(light, lightDirection));
        if (cosine <= 0 || emitterCosine <= 0 || dot(geometricNormal, lightDirection) <= 0) continue;
        float3 emission = emitterRadiance(light, float3(1-root, root*(1-uv.y), root*uv.y));
        if (!any(emission > 0.0)) {
            // Even a zero sample depends on the selected emitter's material.
            dependencyIncludeVisibilityTarget(dependency, target);
            continue;
        }
        float lightPdf = light.edge1.w * lightDistance * lightDistance / (light.color.w * emitterCosine);
        // Evaluate irradiance at the real surface. Reduce only the visibility
        // ray's offset near the light so bias cannot move it across the emitter.
        float planeDistance = abs(dot(light.direction.xyz, position - light.position.xyz));
        float3 origin = rayOffset(position, geometricNormal, min(params.tuning.z, planeDistance * 0.25));
        float3 delta = target - origin;
        float distance = length(delta);
        if (distance <= params.tuning.z * 0.25) continue;
        float3 direction = delta / distance;
        bool visibilitySupported;
        DependencyContext visibilityDependency;
        bool blocked = traceOccludedTracked(origin, direction,
            distance - min(params.tuning.z * 0.25, distance * 0.001),
            visibilitySupported, visibilityDependency, admissionLimit);
        if (!visibilitySupported) { querySupported = false; return 0.0; }
        // The visibility query deliberately stops before the emitter to avoid
        // self-intersection. Keep its known endpoint in the dependency so
        // emissive-history invalidation covers this sampled source surface.
        dependencyIncludeVisibilityTarget(visibilityDependency, target);
        dependencyUnion(dependency, visibilityDependency);
        if (blocked) continue;
        float3 response = scatteringResponse(normal, view, lightDirection, roughness, fullBsdf, material, ab);
        sum += emission * (response * cosine / max(lightPdf, 1e-20));
    }
    return sum / float(count);
}

void conventionalLightResponseTracked(float3 position, float3 normal, float3 geometricNormal,
                       float3 view, float roughness, out float3 background,
                       out float3 punctual, out bool querySupported,
                       inout DependencyContext dependency, uint admissionLimit,
                       bool fullBsdf, SurfaceBsdf material, float2 ab) {
    querySupported = true;
    background = 0.0;
    punctual = 0.0;
    float3 offset = rayOffset(position, geometricNormal, params.tuning.z);
    [loop] for (uint i = 0; i < params.frame.w; ++i) {
        Light light = lights[i];
        bool finiteLight = conventionalLightIsPunctual(light);
        float3 toLight = light.position.xyz; float distance = 1e20, attenuation = 1.0;
        if (finiteLight) {
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
        if (cosine > 0 && dot(geometricNormal, toLight) > 0 && attenuation > 0) {
            bool visibilitySupported;
            DependencyContext visibilityDependency;
            bool blocked = traceOccludedTracked(offset, toLight, distance - params.tuning.z,
                visibilitySupported, visibilityDependency, admissionLimit);
            dependencyUnion(dependency, visibilityDependency);
            if (!visibilitySupported) { querySupported = false; return; }
            if (!blocked) {
                float3 contribution = light.color.xyz * (attenuation * cosine
                    * scatteringResponse(normal, view, toLight, roughness, fullBsdf, material, ab));
                addConventionalLightRadiance(background, punctual, finiteLight, contribution);
            }
        }
    }
}

float4 splitRasterMetadata(uint2 full);
TransportRadiance clampOutgoingTransportRadiance(TransportRadiance radiance) {
    float3 total = radiance.background + radiance.emissive + radiance.punctual;
    if (!all(isfinite(radiance.background)) || !all(isfinite(radiance.emissive))
        || !all(isfinite(radiance.punctual)) || !all(isfinite(total)))
        return (TransportRadiance)0;
    radiance.background = clampOutgoingRadiance(radiance.background, params);
    radiance.emissive = clampOutgoingRadiance(radiance.emissive, params);
    radiance.punctual = clampOutgoingRadiance(radiance.punctual, params);
    return radiance;
}

TransportRadiance outgoingTransportRadianceTracked(float3 origin, float3 direction, Hit hit,
                               out bool querySupported,
                               inout DependencyContext dependency,
                               uint admissionLimit = 0u) {
    TransportRadiance radiance = (TransportRadiance)0;
    querySupported = (hit.errors & HIT_QUERY_UNSUPPORTED) == 0u;
    if (!querySupported) return radiance;
    // Ray-hit surfaces include direct environment through NEE, or through the
    // secondary cache with cached hit sky. The primary visible receiver's
    // direct environment remains native Bevy lighting, so escaped primary
    // rays contribute supported zero radiance to indirect GI.
    if (hit.triangle == EMPTY) {
        bool escapeSupported;
        radiance.background = outgoingEnvironmentEscape(direction, querySupported,
            escapeSupported);
        querySupported = escapeSupported;
        return radiance;
    }
    float3 baseColor = materialBaseColor(hit).rgb;
    // Secondary rays already account for direct emitters through hit-point NEE.
    if (params.environmentScale.z <= 0.5)
        radiance.emissive = hitEmission(hit, baseColor);
    if ((sceneWord(instanceBase(hit.instance) + 6u).w & 16u) != 0u)
        return clampOutgoingTransportRadiance(radiance);
    float3 position = origin + direction * hit.distance;
    HitAttributes attributes = hitAttributes(hit);
    float3 geometricNormal = attributes.geometricNormal;
    if (dot(geometricNormal, direction) > 0) geometricNormal = -geometricNormal;
    float3 normal = hitShadingNormal(hit, attributes, direction);
    if (dot(normal, -direction) <= 0 || dot(normal, geometricNormal) <= 0)
        return clampOutgoingTransportRadiance(radiance);
    float3 view = normalize(-direction);
    SurfaceBsdf material = hitSurfaceBsdf(hit, baseColor);
    float roughness = isfinite(material.perceptualRoughness)
        ? material.perceptualRoughness : 1.0;
    float2 ab = bsdfDfg(roughness, saturate(dot(normal, view)));
    uint directSamples = max(1u, params.lightCounts.z / 2);
    uint directSequence = stochasticSampleFrame(params);
    // Read the cache first: with cached hit sky, a covered hit takes its sky
    // light from the cache and skips the environment visibility rays. The
    // lookup reads only the previous frame's cache, so deferred recording and
    // replay make the same choice and issue the same query sequence.
    float indirectBackgroundCoverage;
    TransportRadiance indirect = secondaryFeedback(position, normal, dependency,
        indirectBackgroundCoverage);
    bool skyFromCache = cachedHitSky(params) && indirectBackgroundCoverage > 0.0;
    bool emissiveSupported, environmentSupported = true, conventionalSupported;
    float3 emissiveNee = emissiveResponseTracked(position, normal, geometricNormal, view,
        roughness, directSamples, directSequence ^ 0xE11E0001u, emissiveSupported,
        dependency, admissionLimit, true, material, ab);
    float3 environment = 0.0;
    if (!skyFromCache)
        environment = environmentResponseTracked(position, normal, geometricNormal, view,
            roughness, directSamples, directSequence ^ 0xE17A0002u, environmentSupported,
            dependency, admissionLimit, true, material, ab);
    float3 directional, punctual;
    conventionalLightResponseTracked(position, normal, geometricNormal, view, roughness,
        directional, punctual, conventionalSupported, dependency, admissionLimit,
        true, material, ab);
    querySupported = emissiveSupported && environmentSupported && conventionalSupported;
    if (!querySupported) return (TransportRadiance)0;

    radiance.background = environment + directional;
    radiance.emissive += emissiveNee;
    radiance.punctual = punctual;
    float3 diffuse = bsdfDiffuseColor(material) * (1.0 - bsdfSpecularEnergy(material, ab));
    radiance.background += indirect.background * diffuse;
    radiance.emissive += indirect.emissive * diffuse;
    radiance.punctual += indirect.punctual * diffuse;
    return clampOutgoingTransportRadiance(radiance);
}


float4 splitRasterMetadata(uint2 full) {
    uint4 material = gbuffer.Load(int3(full,0));
    uint4 raster = rasterReceiver.Load(int3(full,0));
    // Cache probes are shared in world space and carry no receiver entity ID.
    // Keep the raster identity here for screen-history validation.
    uint2 identity = raster.xy;
    float3 geometricNormal = receiverGeometricNormal(raster.z, deferredShadingNormal(material));
    return float4(asfloat(identity),receiverPackedNormal(geometricNormal),
                  receiverPackedHistory(1u));
}
bool splitReceiverCompatible(float4 a, float4 b) {
    return receiverSame(a,b) && dot(receiverNormal(a),receiverNormal(b)) > 0.97;
}

uint temporalDiagnosticLuminanceBin(float4 value) {
    if (!all(isfinite(value))) return 1u;
    if (value.w <= 0.0) return 0u;
    float light = luminance(max(value.xyz, 0.0));
    if (light <= 1e-6) return 2u;
    if (light < 1.0) return 3u;
    if (light < 10.0) return 4u;
    if (light < 100.0) return 5u;
    if (light < 1000.0) return 6u;
    return light < 10000.0 ? 7u : 8u;
}

void recordTemporalDiagnostics(float4 raw, float4 filtered, float length) {
    // Fractional .25 is reserved for raw-output debug requests. Regular
    // rendering performs no atomics.
    if (!persistentWorldIntervals(params) || frac(params.intervalHistory.z) <= 0.1) return;
    uint confidence = length <= 0.0 ? 0u : (length <= 1.01 ? 1u
        : (length < 4.0 ? 2u : (length < 8.0 ? 3u : (length < 16.0 ? 4u
        : (length < 32.0 ? 5u : (length < 64.0 ? 6u : 7u))))));
    uint base = 2u * params.size.w + 10u;
    heads[base + confidence].add(1u);
    heads[base + 8u + temporalDiagnosticLuminanceBin(raw)].add(1u);
    heads[base + 17u + temporalDiagnosticLuminanceBin(filtered)].add(1u);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void temporal(uint3 thread : SV_DispatchThreadID) {
    uint resolveFactor = giResolveScale(params);
    uint2 resolveExtent = giResolveExtent(params);
    if (any(thread.xy >= resolveExtent)) return;
    int2 pixel = int2(thread.xy);
    float4 center = currentIrradiance.Load(int3(pixel, 0));
    float4 rawForDiagnostics = center;
    uint2 fullLocal = min(thread.xy * resolveFactor + resolveFactor / 2u,
                          params.viewport.zw - 1u);
    uint2 full = params.viewport.xy + fullLocal;
    float4 receiver = splitRasterMetadata(full);
    float depth = depthTexture.Load(int3(full, 0));
    if (depth <= 0 || receiverHistoryLength(receiver) == 0) {
        backgroundOutput.Store(pixel, 0);
        irradiance.Store(pixel, 0); historyMetadata.Store(pixel, 0); receiverHistory.Store(pixel,0);
        recordTemporalDiagnostics(rawForDiagnostics, float4(0.0), 0.0);
        return;
    }
    bool currentFinite = isfinite(center.w) && all(isfinite(center.xyz));
    float4 backgroundSample = currentBackground.Load(int3(pixel, 0));
    bool backgroundSupported = backgroundSample.w > 0.0
        && all(isfinite(backgroundSample));
    float backgroundFreshness = backgroundSupported
        ? saturate(frac(backgroundSample.w) * 4.0) : 0.0;
    uint currentSupportMask = currentFinite && center.w > 0.0
        ? uint(floor(center.w)) & 1u : 0u;
    bool currentSupported = currentSupportMask != 0u && currentFinite;
    // A complete raw sample requires every active source. For display only,
    // preserve independently supported background while a moving source is
    // still being sampled; the raw support marker remains unchanged.
    float currentFreshness = currentFinite && currentSupported
        ? saturate(frac(center.w) * 4.0) : 0.0;
    if (!currentSupported && currentFinite && center.w < 0.0 && backgroundSupported) {
        currentSupported = true;
        currentSupportMask = 1u;
        currentFreshness = backgroundFreshness;
    }
    if (!currentSupported) center.xyz = 0.0;
    // Sparse reconstruction can leave a receiver without a current lighting
    // sample. Raster depth still identifies the surface for history validation.
    center.w = depth;
    uint outputSupportMask = currentSupportMask;
    float3 geometricNormal = receiverNormal(receiver);
    float3 normal = geometricNormal;
    float3 position = worldPosition(float2(full) + 0.5, depth, params);
    float3 mean = 0, square = 0;
    float count = 0;
    uint2 cellStride = persistentWorldIntervals(params) ? uint2(1u) : max(uint2(round(float2(params.viewport.zw)
        / (float2(params.size.xy) * float(resolveFactor)))), uint2(1u));
    [loop] for (int y = -1; y <= 1; ++y) for (int x = -1; x <= 1; ++x) {
        int2 neighbor = pixel + int2(x, y) * int2(cellStride);
        // The irradiance, moments, and receiver histories are stored at resolve
        // extent. Check these coordinates before sampling any resolve-sized image.
        if (any(neighbor < 0) || any(neighbor >= int2(resolveExtent))) continue;
        float4 value = currentIrradiance.Load(int3(neighbor, 0));
        if (value.w <= 0) continue;
        uint neighborSupportMask = uint(floor(value.w)) & 1u;
        if (currentSupportMask != 0u && neighborSupportMask != currentSupportMask) continue;
        // Depth and receiver attachments remain full viewport size. Map the
        // resolve texel center back to the same full-resolution representative
        // used by temporal and spatial reconstruction.
        uint2 otherLocal = min(uint2(neighbor) * resolveFactor + resolveFactor / 2u,
                               params.viewport.zw - 1u);
        uint2 otherFull = params.viewport.xy + otherLocal;
        float otherDepth = depthTexture.Load(int3(otherFull, 0));
        if (otherDepth <= 0) continue;
        float4 otherReceiver = splitRasterMetadata(otherFull);
        if (!splitReceiverCompatible(receiver, otherReceiver)) continue;
        float3 otherNormal = receiverNormal(otherReceiver);
        float3 otherPosition = worldPosition(float2(otherFull) + 0.5, otherDepth, params);
        if (dot(geometricNormal, otherNormal) < 0.95
            || surfacePlaneDistance(otherPosition - position, geometricNormal, otherNormal)
                > surfaceTolerance(params)) continue;
        mean += value.xyz; square += value.xyz * value.xyz; count += 1;
    }
    mean /= max(count, 1); square /= max(count, 1);
    float3 variance = max(0.0, square - mean * mean);
    float3 deviation = sqrt(variance);
    float4 history = 0, metadata = 0; float total = 0;
    float4 backgroundHistory = 0.0;
    float backgroundWeight = 0.0;
    uint historySupportMask = 0u;
    float2 previousFull = 0; bool projected = false;
    // Static world points reproject exactly without relying on raster motion history.
    bool hasMotion = params.stability.z > 0
        && (!persistentWorldIntervals(params) || params.stability.y > 0.5);
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
    // Gradual sun motion reuses validated coverage as a bounded temporal
    // estimate; abrupt changes and other source groups remain hard invalidations.
    bool gradualSun = (params.sourceHistory.z & 1u) != 0u;
    uint hardSourceChanges = params.sourceHistory.y
        & ~(gradualSun ? BACKGROUND_SOURCE : 0u);
    bool motionHistoryExpired = (params.sourceHistory.z & 2u) != 0u
        && params.frame.x - params.sourceHistory.w > 120u;
    if (projected && params.stability.x > 0
        && (hardSourceChanges & BACKGROUND_SOURCE) == 0u &&
        !dirtyContains(position, params) &&
        params.history.x > 0 && params.history.w > 0 &&
        all(previousFull >= float2(params.viewport.xy)) &&
        all(previousFull < float2(params.viewport.xy+params.viewport.zw))) {
        // Invert the actual resolve anchors, including the clamped last tile.
        float2 local = previousFull - float2(params.viewport.xy) - 0.5;
        float2 grid = (local - float2(resolveFactor / 2u)) / float(resolveFactor);
        int2 base = int2(floor(grid));
        float2 last = float2(params.viewport.zw - 1u);
        float2 left = min(float2(base * int(resolveFactor) + int(resolveFactor / 2u)), last);
        float2 right = min(float2((base + 1) * int(resolveFactor) + int(resolveFactor / 2u)), last);
        float2 fraction = saturate((local - left) / max(right - left, 1.0));
        [loop] for (uint corner = 0; corner < 4; ++corner) {
                int2 bit = int2(corner & 1, corner >> 1), oldPixel = base + bit;
                if (any(oldPixel < 0) || any(oldPixel >= int2(resolveExtent))) continue;
                float4 old = previousIrradiance.Load(int3(oldPixel, 0));
                float4 meta = previousMetadata.Load(int3(oldPixel, 0));
                float4 oldReceiver = previousReceiverHistory.Load(int3(oldPixel,0));
                if (meta.z <= 0 || old.w <= 0 || !all(isfinite(old.xyz)) ||
                    dot(normal, decodeNormal(meta.xy)) < 0.95 ||
                    !splitReceiverCompatible(receiver,oldReceiver)) continue;
                uint2 oldLocal = min(uint2(oldPixel) * resolveFactor + resolveFactor / 2u,
                                     params.viewport.zw - 1u);
                uint2 oldFull = params.viewport.xy + oldLocal;
                float2 oldPixelCenter = float2(oldFull) + 0.5;
                float3 oldPosition = previousWorldPosition(oldPixelCenter, old.w, params);
                if (dirtyContains(oldPosition, params)) continue;

                if (hasMotion) {
                    // Compare depths in the same camera space. Current depth
                    // changes when the camera moves even on a static surface.
                    float4 previousClip = mul(params.previousClipFromWorld, float4(position, 1));
                    float expected = previousClip.z / previousClip.w;
                    if (!isfinite(expected) || expected <= 0) continue;
                    float tolerance = max(0.000002,abs(expected)*0.0001);
                    [loop] for (uint axis = 0; axis < 2; ++axis) {
                        int2 neighbor = min(oldPixel+(axis == 0 ? int2(1,0) : int2(0,1)),int2(resolveExtent)-1);
                        float4 neighborReceiver = previousReceiverHistory.Load(int3(neighbor,0));
                        float neighborDepth = previousIrradiance.Load(int3(neighbor,0)).w;
                        if (splitReceiverCompatible(oldReceiver,neighborReceiver) && neighborDepth > 0)
                            tolerance += 2*abs(neighborDepth-old.w);
                    }
                    if (abs(expected-old.w) > tolerance) continue;
                } else {
                    if (surfacePlaneDistance(oldPosition - position, geometricNormal,
                        receiverNormal(oldReceiver)) > surfaceTolerance(params)) continue;
                }
                float2 w = lerp(1 - fraction, fraction, float2(bit));
                float weight = w.x * w.y;
                if (weight <= 0.0) continue;
                uint oldSupportMask = receiverScreenHistorySupportMask(oldReceiver);
                if (oldSupportMask == 0u) continue;
                float4 oldBackground = previousBackground.Load(int3(oldPixel, 0));
                if (params.dirtyMin.w <= 0.0 && oldBackground.w > 0.0
                    && !(motionHistoryExpired && backgroundFreshness <= 0.00001)
                    && all(isfinite(oldBackground))) {
                    backgroundHistory += oldBackground * weight;
                    backgroundWeight += weight;
                }
                if (hardSourceChanges != 0u
                    || (motionHistoryExpired && currentFreshness <= 0.00001)) continue;
                historySupportMask = 1u;
                history += old * weight; metadata += meta * weight; total += weight;
            }
    }
    float historyCapacity = max(params.history.w, 0.0);
    float3 stableBackground = backgroundSupported ? backgroundSample.xyz : 0.0;
    float backgroundLength = backgroundSupported
        ? (persistentWorldIntervals(params) ? backgroundFreshness : 1.0) : 0.0;
    if (backgroundWeight > 0.00001) {
        backgroundHistory /= backgroundWeight;
        float oldLength = min(backgroundHistory.w, historyCapacity);
        if (backgroundSupported && backgroundFreshness > 0.00001) {
            float weight = min(gradualSun ? min(params.history.x, 0.9) : params.history.x,
                               oldLength / (oldLength + 1.0));
            if (persistentWorldIntervals(params))
                weight = weight / max(weight + backgroundFreshness * (1.0 - weight), 1e-8);
            stableBackground = lerp(stableBackground, backgroundHistory.xyz, weight);
            backgroundLength = min(historyCapacity, oldLength + backgroundFreshness);
        } else {
            // Do not fade energy, but bound unsupported history during a
            // continuously moving sun so it cannot persist indefinitely.
            backgroundLength = gradualSun
                ? max(0.0, oldLength - max(historyCapacity, 1.0) / 120.0) : oldLength;
            stableBackground = backgroundLength > 0.0 ? backgroundHistory.xyz : 0.0;
        }
    }
    backgroundOutput.Store(pixel, float4(stableBackground, backgroundLength));
    if (hardSourceChanges != 0u && backgroundLength > 0.0) {
        center.xyz += stableBackground - (backgroundSupported ? backgroundSample.xyz : 0.0);
        if (!currentSupported && currentFinite) {
            currentSupported = true;
            currentSupportMask = 1u;
            outputSupportMask = 1u;
            currentFreshness = min(backgroundLength, 1.0);
        }
    }
    // Length zero marks missing lighting support for downstream spatial passes.
    // A supported current sample remains valid even with temporal history disabled.
    float retainedLength = currentSupported
        ? (persistentWorldIntervals(params) ? currentFreshness : 1.0) : 0.0;
    float outputVariance = currentSupported ? luminance(variance) : 0.0;
    if (total > 0.00001) {
        history /= total; metadata /= total;
        outputSupportMask = historySupportMask;
        float oldLength = min(metadata.z, historyCapacity);
        if (gradualSun && currentFreshness <= 0.00001) {
            oldLength = max(0.0, oldLength - max(historyCapacity, 1.0) / 120.0);
            if (oldLength <= 0.0) {
                irradiance.Store(pixel, 0.0);
                historyMetadata.Store(pixel, 0.0);
                receiverHistory.Store(pixel, receiverWithScreenHistorySupportMask(receiver, 0u));
                recordTemporalDiagnostics(rawForDiagnostics, 0.0, 0.0);
                return;
            }
        }
        if (currentSupported) {
            if (currentFreshness <= 0.00001) {
                // A retained candidate with negligible support is not a new
                // observation and cannot replace validated screen history.
                center.xyz = history.xyz;
                retainedLength = oldLength;
                outputVariance = metadata.w;
                irradiance.Store(pixel, center);
                historyMetadata.Store(pixel, float4(encodeNormal(normal), retainedLength, outputVariance));
                receiverHistory.Store(pixel, receiverWithScreenHistorySupportMask(
                    receiver, outputSupportMask));
                recordTemporalDiagnostics(rawForDiagnostics, center, retainedLength);
                return;
            }
            bool completeObservation = !persistentWorldIntervals(params) || currentFreshness >= 1.0;
            float3 clipped = history.xyz;
            // Clamp only when independent same-surface observations exist.
            // Partial angular observations cannot bound unobserved directions
            // in a validated complete history, even with multiple raster taps.
            if (count >= 2.0 && completeObservation) {
                float deltaToCurrent = luminance(history.xyz) - luminance(center.xyz);
                float temporalVariance = max(metadata.w, deltaToCurrent * deltaToCurrent * 0.25);
                float3 radius = 2.5 * max(deviation, float3(sqrt(temporalVariance))) + 0.0001;
                float3 low = max(0.0, mean - radius), high = mean + radius;
                clipped = clamp(history.xyz, low, high);
            }
            float historyLuminance = luminance(history.xyz);
            float currentLuminance = luminance(center.xyz);
            float delta = historyLuminance - currentLuminance;
            float change = abs(luminance(clipped) - currentLuminance)
                           / max(max(luminance(clipped), luminance(center.xyz)), 0.001);
            float reactiveWeight = !gradualSun && params.stability.y > 0 && completeObservation
                ? 1.0 - change * change : 1.0;
            float referenceLuminance = max(max(historyLuminance, currentLuminance), 0.001);
            float relativeHistoryNoise = sqrt(max(metadata.w, 0.0)) / referenceLuminance;
            // A global scene-change flag also covers moving geometry. Only
            // discard this receiver's accumulated history when its own signal
            // differs by more than 3 sigma of retained temporal noise. The 5%
            // floor filters rounding/TAA jitter; the 25% cap keeps clear light
            // steps responsive even when their prior estimate was noisy.
            float radiometricThreshold = clamp(3.0 * relativeHistoryNoise, 0.05, 0.25);
            bool radiometricStep = currentFreshness > 0.0
                && completeObservation
                && !gradualSun && params.stability.y > 0.5
                && change > radiometricThreshold;
            // A radiometric step changes the target distribution. Let the
            // refreshed estimate drive every transport path instead of
            // blending the pre-change screen estimate a second time.
            if (radiometricStep) oldLength = 0.0;
            float weight = min(gradualSun ? min(params.history.x, 0.9) : params.history.x,
                               oldLength / (oldLength + 1.0));
            if (persistentWorldIntervals(params)) {
                // Convert the full-observation blend to equivalent prior mass.
                // q=1 preserves the original blend; partial coverage has only
                // fractional authority, even when the blend cap dominates.
                weight = weight / max(weight + currentFreshness * (1.0 - weight), 1e-8);
            }
            weight *= reactiveWeight;
            // Sparse candidate means and screen history refer to the same
            // receiver, but neighboring candidate updates can still move the
            // reconstructed pixel estimate. Keep bounded receiver history to
            // suppress that resampling noise; radiometric steps above clear its
            // confidence before the new lighting is accumulated.
            if (radiometricStep) weight = 0.0;
            center.xyz = lerp(center.xyz, clipped, weight);
            retainedLength = min(historyCapacity, oldLength + currentFreshness);
            outputVariance = lerp(luminance(variance), metadata.w, weight)
                           + weight * (1.0 - weight) * delta * delta;
        } else {
            // Missing current support is not a black observation. Keep
            // geometrically validated history without growing its confidence.
            // Missing support is not black. Keep validated history exactly
            // until a supported current estimate arrives; a synthetic fade
            // would make sparse light updates appear as slowly moving spots.
            center.xyz = history.xyz;
            retainedLength = oldLength;
            outputVariance = metadata.w;
        }
    }
    // Alpha carries both the representative depth and a lighting-support
    // marker into denoise and composite. A raster surface with no current
    // estimate and no validated history must not look like supported black GI.
    center.w = currentSupported || retainedLength > 0.0 ? depth : 0.0;
    irradiance.Store(pixel, center);
    historyMetadata.Store(pixel, float4(encodeNormal(normal), retainedLength, outputVariance));
    receiverHistory.Store(pixel, receiverWithScreenHistorySupportMask(
        receiver, outputSupportMask));
    recordTemporalDiagnostics(rawForDiagnostics, center, retainedLength);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void denoise(uint3 thread : SV_DispatchThreadID) {
    uint resolveFactor = giResolveScale(params);
    uint2 resolveExtent = giResolveExtent(params);
    if (any(thread.xy >= resolveExtent)) return;
    int2 pixel = int2(thread.xy);
    float4 center = previousIrradiance.Load(int3(pixel, 0));
    bool centerHasLighting = center.w > 0.0 && isfinite(center.w)
        && all(isfinite(center.xyz));
    uint2 fullLocal = min(thread.xy * resolveFactor + resolveFactor / 2u,
                          params.viewport.zw - 1u);
    uint2 full = params.viewport.xy + fullLocal;
    float depth = centerHasLighting ? center.w : depthTexture.Load(int3(full, 0));
    if (depth <= 0.0 || !isfinite(depth)) {
        irradiance.Store(pixel, 0.0); receiverHistory.Store(pixel, 0.0); return;
    }
    // Temporal already wrote this pixel's packed raster identity and geometric
    // normal. Reuse that metadata for all filter taps instead
    // of decoding the G-buffer and receiver attachment repeatedly.
    float4 receiver = previousReceiverHistory.Load(int3(pixel, 0));
    uint centerSupportMask = receiverScreenHistorySupportMask(receiver);
    float3 geometricNormal = receiverNormal(receiver);
    float3 position = worldPosition(float2(full) + 0.5, depth, params);
    // For denoise, binding 13 is the temporal metadata output: z is the
    // retained-support length and w is its accumulated variance.
    float4 moments = currentIrradiance.Load(int3(pixel, 0));
    float3 normal = geometricNormal;
    bool centerSupported = moments.z > 0.0;
    float variance = max(moments.w, 0.0);
    float3 sum = 0; float total = 0;
    uint2 cellStride = persistentWorldIntervals(params) ? uint2(1u) : max(uint2(round(float2(params.viewport.zw)
        / (float2(params.size.xy) * float(resolveFactor)))), uint2(1u));
    uint2 baseStride = max(uint2(round(float2(cellStride) * 0.25)), uint2(1u));
    uint2 step = params.frame.y == 0
        ? uint2(1u)
        : baseStride * (1u << params.frame.y);
    // Progress through distinct scales (1, 2, 4 at the default density).
    // Repeating stride one wasted a pass on already-filtered texels and left
    // the sparse-grid noise outside the final filter's footprint.
    const float kernel[5] = {1.0, 4.0, 6.0, 4.0, 1.0};
    // The four far corners contribute only 4/256 of the original separable
    // binomial kernel. Omitting them retains its axial and diagonal support
    // while avoiding four texture/geometry samples per pixel per pass.
    [loop] for (int y = -2; y <= 2; ++y) {
        int xRadius = abs(y) == 2 ? 1 : 2;
        [loop] for (int x = -xRadius; x <= xRadius; ++x) {
            int2 otherPixel = pixel + int2(x, y) * int2(step);
            if (any(otherPixel < 0) || any(otherPixel >= int2(resolveExtent))) continue;
            float4 value = previousIrradiance.Load(int3(otherPixel, 0));
            if (value.w <= 0) continue;
            float4 otherMoments = currentIrradiance.Load(int3(otherPixel, 0));
            if (otherMoments.z <= 0.0) continue;
            float4 otherReceiver = previousReceiverHistory.Load(int3(otherPixel, 0));
            if (!splitReceiverCompatible(receiver, otherReceiver)) continue;
            uint otherSupportMask = receiverScreenHistorySupportMask(otherReceiver);
            if (centerSupportMask != 0u && otherSupportMask != centerSupportMask) continue;
            if (centerSupportMask == 0u && otherSupportMask == 0u) continue;
            float3 otherNormal = receiverNormal(otherReceiver);
            uint2 otherLocal = min(uint2(otherPixel) * resolveFactor + resolveFactor / 2u,
                                   params.viewport.zw - 1u);
            uint2 otherFull = params.viewport.xy + otherLocal;
            float3 otherPosition = worldPosition(float2(otherFull) + 0.5, value.w, params);
            if (dot(geometricNormal, otherNormal) < 0.95) continue;
            float planeDistance = surfacePlaneDistance(otherPosition - position, geometricNormal, otherNormal);
            if (planeDistance > surfaceTolerance(params)) continue;
            float weight = kernel[x + 2] * kernel[y + 2]
                * pow(max(0.0, dot(normal, otherNormal)), 32.0)
                * exp(-planeDistance / max(params.tuning.z * 2, params.tuning.x * 0.05));
            if (centerSupported) {
                float colorScale = max(0.02, 2.0 * sqrt(max(variance + otherMoments.w, 0.0)));
                weight *= exp(-abs(luminance(value.xyz) - luminance(center.xyz)) / colorScale);
            }
            sum += value.xyz * weight; total += weight;
        }
    }
    bool filteredSupport = total > 0.00001;
    float3 filtered = filteredSupport ? sum / total : center.xyz;
    // Missing current support may be reconstructed from compatible neighbors.
    // Preserve zero alpha only when neither this pixel nor the filter has light.
    float outputDepth = centerHasLighting || centerSupported || filteredSupport ? depth : 0.0;
    irradiance.Store(pixel, float4(filtered, outputDepth));
    receiverHistory.Store(pixel, receiverWithScreenHistorySupportMask(
        receiver, outputDepth > 0.0 ? 1u : 0u));
}

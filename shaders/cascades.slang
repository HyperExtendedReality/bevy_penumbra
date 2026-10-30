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

// Ray-traced GI cache shading evaluates direct light only and reads back the
// surface's diffuse reflectance: the cell keeps indirect light as a separate
// estimator, multiplied by that reflectance when the cell is read (GI-1.2).
static bool rtgiDirectOnly = false;
static float3 rtgiDiffuseReflectance = 0.0;

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
    float indirectBackgroundCoverage = 0.0;
    TransportRadiance indirect = (TransportRadiance)0;
    if (!rtgiDirectOnly)
        indirect = secondaryFeedback(position, normal, dependency, indirectBackgroundCoverage);
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
    rtgiDiffuseReflectance = diffuse;
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

#ifdef PENUMBRA_HARDWARE_RAY_QUERY
// One shadow ray per pixel for each directional light Bevy would shadow-map,
// aimed at a point on the sun's disk. The point follows a fixed per-pixel
// pattern, so neighbouring pixels sample the whole disk and sunShadowDepth's
// filter integrates them into an antialiased penumbra; the pattern is
// screen-stable, so a still frame does not shimmer. An 8x8 group covers two
// mask tiles; each lane records its per-light occlusion, then one lane per
// tile packs and stores the tile.
groupshared uint sunShadowOccluded[64];
// Bevy's default `DirectionalLight::sun_disk_angular_size`, in radians.
static const float SUN_DISK_ANGULAR_SIZE = 0.00930842;

float3 sunDiskDirection(float3 direction, uint2 pixel) {
    float3 tangent = normalize(abs(direction.y) < 0.999
        ? cross(float3(0.0, 1.0, 0.0), direction)
        : cross(float3(1.0, 0.0, 0.0), direction));
    float3 bitangent = cross(direction, tangent);
    // Interleaved gradient noise (Jimenez 2014) decorrelates neighbours.
    float2 u = float2(
        frac(52.9829189 * frac(dot(float2(pixel), float2(0.06711056, 0.00583715)))),
        frac(52.9829189 * frac(dot(float2(pixel) + float2(47.0, 17.0),
            float2(0.06711056, 0.00583715)))));
    float radius = tan(0.5 * SUN_DISK_ANGULAR_SIZE) * sqrt(u.x);
    float phi = 2.0 * PI * u.y;
    return normalize(direction + (tangent * cos(phi) + bitangent * sin(phi)) * radius);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void sunShadowsHardware(uint3 thread : SV_DispatchThreadID, uint3 local : SV_GroupThreadID,
                        uint3 group : SV_GroupID) {
    uint tile = local.x / 4u;
    uint occluded = 0u; // bit k: light k blocked
    occlusionIgnoresCulling = true;
    if (all(thread.xy < params.viewport.zw) && params.frame.z != 0u) {
        uint2 pixel = params.viewport.xy + thread.xy;
        float depth = depthTexture.Load(int3(pixel, 0));
        uint4 deferred = gbuffer.Load(int3(pixel, 0));
        // Bevy shadows lit receivers only: not unlit, shadow-receiver flag set.
        if (depth > 0.0 && (deferred.w & (1u << 24u)) == 0u && (deferred.w & (1u << 26u)) != 0u) {
            float3 shadingNormal = deferredShadingNormal(deferred);
            float3 position = worldPosition(float2(pixel) + 0.5, depth, params);
            float3 normal = receiverGeometricNormal(rasterReceiver.Load(int3(pixel, 0)).z,
                                                    shadingNormal);
            // The visible side of the surface is the one facing the camera.
            if (dot(normal, params.cameraPosition.xyz - position) < 0.0) normal = -normal;
            // Reconstructed depth loses precision with distance.
            float bias = params.tuning.z
                * (1.0 + 0.05 * length(params.cameraPosition.xyz - position));
            float3 origin = rayOffset(position, normal, bias);
            uint4 geometryHeader = geometryWord(0);
            uint selected = 0u;
            for (uint index = 0u; index < params.lightCounts.x
                    && selected < SUN_SHADOW_LIGHTS; ++index) {
                Light light = lights[index];
                if (!sunShadowLight(light)) continue;
                float3 direction = normalize(light.position.xyz);
                float3 ray = sunDiskDirection(direction, pixel);
                // A normal map can light a pixel whose triangle faces away from
                // the sun. Its ray would re-enter the receiver's own triangle,
                // speckling sunlit stone with self-shadow; Bevy's shading
                // treats that surface as lit, so step the ray past the plane.
                float grazing = dot(normal, ray);
                float3 rayOrigin = grazing < 0.1
                    ? origin + ray * min(0.1, 4.0 * bias / max(abs(grazing), 0.04))
                    : origin;
                // Bevy's light is zero where the shading normal faces away;
                // record it as occluded so the filter keeps the terminator dark.
                if (dot(shadingNormal, direction) <= 0.0
                    || hardwareOccluded(rayOrigin, ray, 1.0e7, geometryHeader))
                    occluded |= 1u << selected;
                ++selected;
            }
        }
    }
    sunShadowOccluded[local.x + 8u * local.y] = occluded;
    GroupMemoryBarrierWithGroupSync();
    if (local.y == 0u && (local.x & 3u) == 0u) {
        uint bits = 0u;
        for (uint y = 0u; y < 8u; ++y) {
            [unroll] for (uint x = 0u; x < 4u; ++x) {
                if ((sunShadowOccluded[tile * 4u + x + 8u * y] & 1u) != 0u)
                    bits |= 1u << sunShadowBit(uint2(x, y));
            }
        }
        irradiance.Store(int2(group.x * 2u + tile, group.y),
            float4(bits & 0xffffu, bits >> 16u, 0.0, 0.0));
    }
}
#endif

// Ray-traced GI (GiMethod::RayTraced), after idTech 8 (Sousa 2025, "Fast as
// Hell") and AMD GI-1.2 (Boisse et al. 2022; Tjernell 2025):
//  - Final gather (rtgiSample): one cosine-weighted, unshaded hardware ray per
//    resolve sample. Its hit takes radiance from the first cache level that
//    has it: last frame's lit image when the camera sees the hit, then the
//    spatial-hash world radiance cache, then the irradiance volumes.
//  - ReSTIR GI (Ouyang et al. 2021) reuses the gathered paths over time and
//    space (rtgiSample, rtgiResolve); the chosen path is projected onto L1 SH
//    per resolve sample, so each one is a small irradiance probe.
//  - World radiance cache: every gather, bounce and probe ray hit claims its
//    cell and requests shading for a new or stale cell. A bounded number of
//    cells are shaded per frame (rtgiCacheRecord/Visibility/Shade) and each
//    traces one bounce ray (rtgiCacheBounce). Direct and indirect light are
//    separate estimators; the indirect one is multiplied by the cell's diffuse
//    reflectance when the cell is read (GI-1.2 multibounce).
//  - Irradiance volumes (rtgiVolumeTrace/Update): camera-centred cascaded
//    probes whose rays read the world cache; the last cache level.
//  - Denoise: separable bilateral passes over normal, plane and hit distance
//    (rtgiDenoiseX/Y), a temporal filter with a per-texel sample count
//    (rtgiTemporal), then composite.slang's rtgiFragment upsamples, evaluating
//    the SH with each pixel's own normal.

struct RtgiReservoir {
    float4 sample;   // xyz: path vertex; w: unbiased contribution weight W
    float4 normal;   // xyz: vertex geometric normal, zero for escaped rays; w: M
    float4 radiance; // rgb: radiance leaving the vertex; w: unused
    float4 receiver; // xyz: receiver position; w: 1 when the receiver is valid
};
// One world radiance cache cell. Mirrors RTGI_CACHE_CELL_BYTES in render.rs.
struct RtgiCacheCell {
    float4 direct;      // rgb: mean outgoing direct radiance; w: sample count
    float4 indirect;    // rgb: mean incoming bounce radiance; w: sample count
    float4 reflectance; // rgb: diffuse reflectance at the shading point
    float4 origin;      // xyz: recorded ray origin; w: hit distance
    float4 direction;   // xyz: recorded ray direction; w: hit footprint
    uint4 hit;          // x: triangle; y: instance; z: barycentric yz (2 x unorm16); w: last used frame
    float4 dependencyMin; // xyz: retained path bounds; w: dependency known
    float4 dependencyMax; // xyz: retained path bounds; w: bounds present
};
[[vk::binding(42, 0)]] RWStructuredBuffer<RtgiReservoir> rtgiReservoirs;
[[vk::binding(43, 0)]] StructuredBuffer<RtgiReservoir> rtgiPreviousReservoirs;
// Cell keys, shading-request stamps, last-use stamps and published-hit
// flags, then queue counts and entries. Mirrors rtgi_cache_bytes in render.rs.
[[vk::binding(44, 0)]] RWStructuredBuffer<Atomic<uint>> rtgiCacheWords;
[[vk::binding(45, 0)]] RWStructuredBuffer<RtgiCacheCell> rtgiCacheCells;
[[vk::binding(46, 0)]] RWStructuredBuffer<float4> rtgiField;
// Last frame's composed HDR image over the viewport, exposure applied.
[[vk::binding(47, 0)]] Texture2D<float4> rtgiLitHistory;
#include "rtgi_field.slang"

// Mirror RTGI_CACHE_* in render.rs.
static const uint RTGI_CACHE_CAPACITY = 1u << 18;
// Cells shaded per frame: the cascades' deferred visibility scratch, idle in
// this method, holds 2 x DEFERRED_TRACE_RECEIVERS lanes. A quarter of them
// sweep the table to evict and refresh cells; the rest drain the queue.
static const uint RTGI_CACHE_WORK = 16384u;
static const uint RTGI_CACHE_QUEUE = RTGI_CACHE_WORK - RTGI_CACHE_WORK / 4u;
static const uint RTGI_CACHE_RESHADE = 4u;    // frames before a touched cell is shaded again
static const uint RTGI_CACHE_LIFETIME = 240u; // frames without a ray hit before eviction
static const float RTGI_CACHE_CELL = 0.25;    // metres at the first distance LOD
static const float RTGI_CACHE_LOD_DISTANCE = 10.0;
static const float RTGI_CACHE_MIN_BLEND = 0.1;
static const float RTGI_VOLUME_MIN_BLEND = 0.15;

static const float RTGI_SPATIAL_RADIUS = 12.0; // resolve pixels
static const float RTGI_ESCAPE_DISTANCE = 1.0e4;
static const int RTGI_DENOISE_RADIUS = 3;      // taps each side per axis

uint rtgiStampHead(uint slot) { return RTGI_CACHE_CAPACITY + slot; }
uint rtgiLastUseHead(uint slot) { return 2u * RTGI_CACHE_CAPACITY + slot; }
uint rtgiValidHitHead(uint slot) { return 3u * RTGI_CACHE_CAPACITY + slot; }
uint rtgiQueueCountHead(uint parity) { return 4u * RTGI_CACHE_CAPACITY + parity; }
uint rtgiQueueHead(uint parity, uint index) {
    return 4u * RTGI_CACHE_CAPACITY + 2u + parity * RTGI_CACHE_QUEUE + index;
}
uint rtgiWorkHead(uint lane) {
    return 4u * RTGI_CACHE_CAPACITY + 2u + 2u * RTGI_CACHE_QUEUE + lane;
}

uint rtgiHash(uint x) {
    x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u;
    return x;
}
float rtgiRandom(inout uint state) {
    state = rtgiHash(state);
    return float(state >> 8u) * (1.0 / 16777216.0);
}
float rtgiLuminance(float3 value) {
    return dot(value, float3(0.2126, 0.7152, 0.0722));
}
float3 rtgiTangent(float3 normal) {
    return normalize(abs(normal.y) < 0.999
        ? cross(float3(0.0, 1.0, 0.0), normal)
        : cross(float3(1.0, 0.0, 0.0), normal));
}
float3 rtgiCosineDirection(float3 normal, float2 u) {
    float3 tangent = rtgiTangent(normal);
    float3 bitangent = cross(normal, tangent);
    float phi = 2.0 * PI * u.y;
    return normalize(tangent * (cos(phi) * sqrt(u.x))
        + bitangent * (sin(phi) * sqrt(u.x)) + normal * sqrt(1.0 - u.x));
}

// Cell key: position quantized by a distance-dependent cell size (idTech 8)
// and the dominant axis of the normal, so opposite faces of a thin wall never
// share radiance. Returns the home slot; `key` is never zero.
float rtgiCacheLod(float3 position) {
    float distance = length(position - params.cameraPosition.xyz);
    return exp2(floor(log2(1.0 + distance / RTGI_CACHE_LOD_DISTANCE)));
}

uint rtgiCacheHome(float3 position, float3 normal, out uint key) {
    float lod = rtgiCacheLod(position);
    int3 cell = int3(floor(position / (RTGI_CACHE_CELL * lod)));
    float3 magnitude = abs(normal);
    uint axis = magnitude.x >= magnitude.y && magnitude.x >= magnitude.z ? 0u
        : (magnitude.y >= magnitude.z ? 1u : 2u);
    uint face = axis * 2u + (normal[axis] < 0.0 ? 1u : 0u);
    uint hash = rtgiHash(uint(lod) + rtgiHash(uint(cell.z) + rtgiHash(uint(cell.y)
        + rtgiHash(uint(cell.x) + rtgiHash(face)))));
    key = rtgiHash(hash ^ 0x9e3779b9u) | 1u;
    return hash % RTGI_CACHE_CAPACITY;
}

// Finds a cell, or claims an empty one when `insert` is set, by linear
// probing (Gautron 2020). EMPTY when the probe sequence is full.
static const uint RTGI_CACHE_PROBES = 8u;
uint rtgiCacheFind(float3 position, float3 normal, bool insert, out bool inserted) {
    inserted = false;
    uint key;
    uint slot = rtgiCacheHome(position, normal, key);
    uint probe = 0u;
    [loop] for (uint attempt = 0u; probe < RTGI_CACHE_PROBES && attempt < 4u * RTGI_CACHE_PROBES;
                ++attempt) {
        uint stored = rtgiCacheWords[slot].load();
        if (stored == key) return slot;
        if (stored == 0u && insert) {
            uint previous = rtgiCacheWords[slot].compareExchange(0u, key);
            if (previous == key) return slot;
            if (previous == 0u) {
                // WGSL's compare-exchange may fail spuriously: the slot is ours
                // only when it now holds our key; still empty means retry it.
                uint now = rtgiCacheWords[slot].load();
                if (now == key) {
                    inserted = true;
                    return slot;
                }
                if (now == 0u) continue;
            }
        }
        slot = (slot + 1u) % RTGI_CACHE_CAPACITY;
        ++probe;
    }
    return EMPTY;
}

void rtgiCacheStoreHit(uint slot, float3 origin, float3 direction, Hit hit) {
    uint2 barycentric = uint2(round(saturate(hit.barycentric.yz) * 65535.0));
    rtgiCacheCells[slot].origin = float4(origin, hit.distance);
    rtgiCacheCells[slot].direction = float4(direction, hit.footprint);
    rtgiCacheCells[slot].hit = uint4(hit.triangle, hit.instance,
        barycentric.x | (barycentric.y << 16u), params.frame.x);
}

Hit rtgiCellHit(RtgiCacheCell cell) {
    Hit hit;
    hit.distance = cell.origin.w;
    hit.triangle = cell.hit.x;
    hit.instance = cell.hit.y;
    hit.errors = 0u;
    float2 yz = float2(cell.hit.z & 0xffffu, cell.hit.z >> 16u) / 65535.0;
    hit.barycentric = float3(1.0 - yz.x - yz.y, yz);
    hit.footprint = cell.direction.w;
    return hit;
}

// Outgoing radiance of a shaded cell: direct light plus the indirect
// estimator through the cell's diffuse reflectance (GI-1.2's resolve).
float3 rtgiCellRadiance(RtgiCacheCell cell) {
    float3 radiance = cell.direct.rgb + cell.reflectance.rgb * cell.indirect.rgb;
    return all(isfinite(radiance)) ? max(radiance, 0.0) : float3(0.0);
}

DependencyContext rtgiCellDependency(RtgiCacheCell cell) {
    return storedDependency(cell.dependencyMin, cell.dependencyMax);
}

void rtgiStoreCellDependency(uint slot, DependencyContext dependency) {
    rtgiCacheCells[slot].dependencyMin = float4(dependency.minimum, dependency.known ? 1.0 : 0.0);
    rtgiCacheCells[slot].dependencyMax = float4(dependency.maximum, dependency.hasBounds ? 1.0 : 0.0);
}

// Finds the world cache cell at a ray hit and keeps it alive. A missing cell
// is claimed. A new cell, or one not shaded for RTGI_CACHE_RESHADE frames,
// takes this hit as its shading point and joins `parity`'s shading queue
// (idTech 8: every ray hit triggers a cache update). Only the lane that wins
// the request writes the hit, so a shading point is never torn between rays.
// The lookup point is jittered by up to half a cell in the surface plane
// (GI-1.0, SHaRC), so cell edges average out instead of showing as blocks.
// EMPTY when the probe sequence is full.
uint rtgiCacheTouch(float3 origin, float3 direction, Hit hit, float3 position,
                    float3 normal, uint parity, inout uint state) {
    float3 tangent = rtgiTangent(normal);
    float3 bitangent = cross(normal, tangent);
    float jitter = RTGI_CACHE_CELL * rtgiCacheLod(position);
    float3 lookup = position + (tangent * (rtgiRandom(state) - 0.5)
        + bitangent * (rtgiRandom(state) - 0.5)) * jitter;
    bool inserted;
    uint slot = rtgiCacheFind(lookup, normal, true, inserted);
    if (slot == EMPTY) return EMPTY;
    uint stamp = params.frame.x + 1u;
    // Keys can become visible before their inserting lane runs again. Cells
    // are initialized before eviction publishes an empty key, so whichever
    // lane wins the atomic request is the sole writer of the hit record.
    rtgiCacheWords[rtgiLastUseHead(slot)].max(stamp);
    uint last = rtgiCacheWords[rtgiStampHead(slot)].load();
    bool request = false;
    if (last == 0u || stamp - last >= RTGI_CACHE_RESHADE)
        request = rtgiCacheWords[rtgiStampHead(slot)].exchange(stamp) != stamp;
    if (request) {
        uint work = min(deferredTraceCapacity(params) * 2u, RTGI_CACHE_WORK);
        uint queueLanes = work - work / 4u;
        uint queued = rtgiCacheWords[rtgiQueueCountHead(parity)].add(1u);
        if (queued < min(RTGI_CACHE_QUEUE, queueLanes)) {
            rtgiCacheStoreHit(slot, origin, direction, hit);
            // The record dispatch runs after this pass. Never refresh an
            // occupied slot whose request overflowed before recording a hit.
            rtgiCacheWords[rtgiValidHitHead(slot)].store(1u);
            rtgiCacheWords[rtgiQueueHead(parity, queued)].store(slot);
        } else {
            // An unadmitted request must remain eligible in a later frame.
            rtgiCacheWords[rtgiStampHead(slot)].compareExchange(stamp, 0u);
        }
    }
    return slot;
}

// idTech 8's first cache level: a gather hit the camera also sees takes its
// radiance from last frame's lit image, which carries every bounce at full
// shading detail. False when the hit is off screen, faces away from the
// camera or is hidden in the current depth buffer.
// shortcut: the image is view-dependent, so glossy hits bring their camera-facing specular; mask by roughness if that shows.
bool rtgiScreenRadiance(float3 position, float3 normal, out float3 radiance) {
    radiance = 0.0;
    if (params.rayTraced.w == 0u
        || dot(normal, params.cameraPosition.xyz - position) <= 0.0) return false;
    float4 clip = mul(params.clipFromWorld, float4(position, 1.0));
    float4 previous = mul(params.previousClipFromWorld, float4(position, 1.0));
    if (!(clip.w > 0.0) || !(previous.w > 0.0)) return false;
    float2 uv = float2(clip.x / clip.w * 0.5 + 0.5, 0.5 - clip.y / clip.w * 0.5);
    float2 previousUv = float2(previous.x / previous.w * 0.5 + 0.5,
        0.5 - previous.y / previous.w * 0.5);
    if (any(uv <= 0.0) || any(uv >= 1.0) || any(previousUv <= 0.0) || any(previousUv >= 1.0))
        return false;
    uint2 pixel = params.viewport.xy + min(uint2(uv * float2(params.viewport.zw)),
        params.viewport.zw - 1u);
    float depth = depthTexture.Load(int3(pixel, 0));
    if (!(depth > 0.0)) return false;
    float3 seen = worldPosition(float2(pixel) + 0.5, depth, params);
    float distance = length(position - params.cameraPosition.xyz);
    if (!(length(seen - position) < 0.01 * distance + 0.02)) return false;
    uint2 previousPixel = min(uint2(previousUv * float2(params.viewport.zw)),
        params.viewport.zw - 1u);
    float3 lit = rtgiLitHistory.Load(int3(previousPixel, 0)).rgb
        / max(params.environment.w, 1e-20);
    if (!all(isfinite(lit))) return false;
    radiance = max(lit, 0.0);
    return true;
}

// The last cache level: outgoing diffuse radiance at a hit from the
// irradiance volumes, before its world cache cell has been shaded.
bool rtgiVolumeRadiance(Hit hit, float3 position, float3 normal, float3 direction,
                        out float3 radiance) {
    radiance = 0.0;
    float3 surface, sky;
    if (!rtgiVolumeIrradiance(position, normal, surface, sky)) return false;
    float3 baseColor = materialBaseColor(hit).rgb;
    SurfaceBsdf material = hitSurfaceBsdf(hit, baseColor);
    float2 ab = bsdfDfg(material.perceptualRoughness, saturate(dot(normal, -direction)));
    float3 diffuse = bsdfDiffuseColor(material) * (1.0 - bsdfSpecularEnergy(material, ab));
    radiance = diffuse * (surface + sky) + hitEmission(hit, baseColor);
    return all(isfinite(radiance));
}

// Radiance leaving a ray hit toward the ray origin, from the first cache
// level that has it: the screen (when `screen`), the world cache, then the
// irradiance volumes. False when none does yet: unknown, not black.
bool rtgiHitRadiance(float3 origin, float3 direction, Hit hit, uint parity, bool screen,
                     inout uint state, out float3 radiance) {
    radiance = 0.0;
    float3 position = origin + direction * hit.distance;
    float3 normal = hitGeometricNormal(hit);
    if (dot(normal, direction) > 0.0) normal = -normal;
    // The cache is touched even for an on-screen hit, keeping its cell alive
    // and shaded for when the hit leaves the screen.
    uint slot = rtgiCacheTouch(origin, direction, hit, position, normal, parity, state);
    if (screen && rtgiScreenRadiance(position, normal, radiance)) return true;
    if (slot != EMPTY) {
        float4 direct = rtgiCacheCells[slot].direct;
        if (direct.w > 0.0) {
            float3 indirect = rtgiCacheCells[slot].indirect.rgb;
            float3 reflectance = rtgiCacheCells[slot].reflectance.rgb;
            radiance = max(direct.rgb + reflectance * indirect, 0.0);
            if (!all(isfinite(radiance))) return false;
            return true;
        }
    }
    return rtgiVolumeRadiance(hit, position, normal, direction, radiance);
}

// Cache cells are shaded like the cascades' deferred primary trace: hit
// shading never shares a kernel with a ray-query loop, which hung the driver.
// rtgiCacheRecord shades with visibility recorded, rtgiCacheVisibility traces
// the recorded shadow rays in hardware, and rtgiCacheShade replays the
// shading with those answers. One lane per work slot; work slots reuse the
// deferred visibility scratch of the cascade trace.
uint rtgiCacheWorkLanes() {
    return min(deferredTraceCapacity(params) * 2u, RTGI_CACHE_WORK);
}

// A bounded rigid-geometry change only invalidates cells whose retained
// shading or bounce paths could have crossed the changed bounds. Queue
// counters are cleared before the gather so no request from the old scene can
// shade a hit that was replaced in this frame.
[shader("compute")]
[numthreads(64, 1, 1)]
void rtgiCacheInvalidate(uint3 thread : SV_DispatchThreadID) {
    if (params.dirtyMin.w <= 0.0 || params.dirtyMax.w <= 0.0) return;
    uint slot = thread.x;
    if (slot == 0u) {
        rtgiCacheWords[rtgiQueueCountHead(0u)].store(0u);
        rtgiCacheWords[rtgiQueueCountHead(1u)].store(0u);
    }
    if (slot >= RTGI_CACHE_CAPACITY || rtgiCacheWords[slot].load() == 0u) return;
    RtgiCacheCell cell = rtgiCacheCells[slot];
    bool validHit = rtgiCacheWords[rtgiValidHitHead(slot)].load() != 0u;
    float3 point = cell.origin.xyz + cell.direction.xyz * cell.origin.w;
    float padding = max(params.tuning.z * 2.0, 1e-5);
    bool movedHit = validHit && (!all(isfinite(point))
        || (all(point >= params.dirtyMin.xyz - padding)
            && all(point <= params.dirtyMax.xyz + padding)));
    bool retained = cell.direct.w > 0.0 || cell.indirect.w > 0.0;
    if (!movedHit && (!retained || !dependencyIntersectsDirty(rtgiCellDependency(cell)))) return;
    rtgiCacheCells[slot].direct = 0.0;
    rtgiCacheCells[slot].indirect = 0.0;
    rtgiStoreCellDependency(slot, dependencyEmpty());
    rtgiCacheWords[rtgiStampHead(slot)].store(0u);
    if (movedHit) rtgiCacheWords[rtgiValidHitHead(slot)].store(0u);
}

// Picks this lane's cell: a queued request, or a slot of the rotating
// refresh window, evicting cells no ray has reached for a while.
uint rtgiCacheSelect(uint lane) {
    uint parity = params.frame.x & 1u;
    // This frame's queue is being drained; the next frame queues into the other.
    if (lane == 0u) rtgiCacheWords[rtgiQueueCountHead(1u - parity)].store(0u);
    uint work = rtgiCacheWorkLanes();
    if (lane >= work) return EMPTY;
    uint queueLanes = work - work / 4u;
    if (lane < queueLanes) {
        uint queued = min(rtgiCacheWords[rtgiQueueCountHead(parity)].load(), queueLanes);
        return lane < queued ? rtgiCacheWords[rtgiQueueHead(parity, lane)].load() : EMPTY;
    }
    uint refresh = work - queueLanes;
    uint slot = (params.frame.x * refresh + lane - queueLanes) % RTGI_CACHE_CAPACITY;
    if (rtgiCacheWords[slot].load() == 0u) return EMPTY;
    if ((params.frame.x + 1u) - rtgiCacheWords[rtgiLastUseHead(slot)].load()
        > RTGI_CACHE_LIFETIME) {
        rtgiCacheCells[slot].direct = 0.0;
        rtgiCacheCells[slot].indirect = 0.0;
        rtgiCacheCells[slot].reflectance = 0.0;
        rtgiCacheWords[rtgiStampHead(slot)].store(0u);
        rtgiCacheWords[rtgiLastUseHead(slot)].store(0u);
        rtgiCacheWords[rtgiValidHitHead(slot)].store(0u);
        rtgiCacheWords[slot].store(0u);
        return EMPTY;
    }
    if (rtgiCacheWords[rtgiValidHitHead(slot)].load() == 0u) return EMPTY;
    // A recent request is queued or was just shaded; shading it here as well
    // would race with its queue lane.
    uint stamp = params.frame.x + 1u;
    if (stamp - rtgiCacheWords[rtgiStampHead(slot)].load() < RTGI_CACHE_RESHADE) return EMPTY;
    rtgiCacheWords[rtgiStampHead(slot)].store(stamp);
    return slot;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rtgiCacheRecord(uint3 thread : SV_DispatchThreadID) {
    uint lane = thread.x;
    if (lane >= RTGI_CACHE_WORK) return;
    uint slot = rtgiCacheSelect(lane);
    rtgiCacheWords[rtgiWorkHead(lane)].store(slot);
    if (slot == EMPTY || slot >= RTGI_CACHE_CAPACITY) return;
    transportAdmissionBypass = true;
    visibilityMode = VISIBILITY_RECORD;
    visibilitySlot = lane;
    visibilityQueries = 0u;
    rtgiDirectOnly = true;
    RtgiCacheCell cell = rtgiCacheCells[slot];
    bool supported;
    DependencyContext dependency = dependencyEmpty();
    outgoingTransportRadianceTracked(cell.origin.xyz, cell.direction.xyz, rtgiCellHit(cell),
        supported, dependency);
    uint queries = min(visibilityQueries, DEFERRED_VISIBILITY_QUERIES);
    if (queries == 0u) return;
    uint base = heads[deferredQueryCountHead(params)].add(queries);
    for (uint i = 0u; i < queries; ++i)
        heads[deferredQueryListHead(params) + base + i].store(lane * DEFERRED_VISIBILITY_QUERIES + i);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rtgiCacheShade(uint3 thread : SV_DispatchThreadID) {
    uint lane = thread.x;
    if (lane >= RTGI_CACHE_WORK) return;
    uint slot = rtgiCacheWords[rtgiWorkHead(lane)].load();
    if (slot == EMPTY || slot >= RTGI_CACHE_CAPACITY) return;
    transportAdmissionBypass = true;
    visibilityMode = VISIBILITY_REPLAY;
    visibilitySlot = lane;
    visibilityQueries = 0u;
    rtgiDirectOnly = true;
    rtgiDiffuseReflectance = 0.0;
    RtgiCacheCell cell = rtgiCacheCells[slot];
    bool supported;
    DependencyContext dependency = dependencyEmpty();
    TransportRadiance radiance = outgoingTransportRadianceTracked(cell.origin.xyz,
        cell.direction.xyz, rtgiCellHit(cell), supported, dependency);
    if (!supported) return;
    // Each retained average includes previous shading points and shadow
    // queries; keep their union until the estimator is invalidated.
    if (cell.direct.w > 0.0 || cell.indirect.w > 0.0)
        dependencyUnion(dependency, rtgiCellDependency(cell));
    float3 shadingPoint = cell.origin.xyz + cell.direction.xyz * cell.origin.w;
    dependencyIncludeSegment(dependency, shadingPoint, shadingPoint, params.tuning.z);
    rtgiStoreCellDependency(slot, dependency);
    float3 direct = radiance.background + radiance.emissive + radiance.punctual;
    // Hysteresis: average the first samples, then keep a floor on the blend
    // so lighting changes reach the cache within about ten updates.
    float count = cell.direct.w + 1.0;
    float blend = max(1.0 / count, RTGI_CACHE_MIN_BLEND);
    rtgiCacheCells[slot].direct = float4(lerp(cell.direct.rgb, direct, blend),
        min(count, 1.0 / RTGI_CACHE_MIN_BLEND));
    rtgiCacheCells[slot].reflectance = float4(rtgiDiffuseReflectance, 0.0);
}
// Snapshot the cache after shading and before any bounce invocation writes
// either a new hit record or indirect radiance.
[shader("compute")]
[numthreads(64, 1, 1)]
void rtgiCacheSnapshot(uint3 thread : SV_DispatchThreadID) {
    uint slot = thread.x;
    if (slot >= RTGI_CACHE_CAPACITY) return;
    RtgiCacheCell cell = rtgiCacheCells[slot];
    uint base = rtgiSnapshotBase(slot);
    rtgiField[base] = cell.direct;
    rtgiField[base + 1u] = cell.indirect;
    rtgiField[base + 2u] = cell.reflectance;
    rtgiField[base + 3u] = cell.origin;
    rtgiField[base + 4u] = cell.direction;
    rtgiField[base + 5u] = asfloat(cell.hit);
    rtgiField[base + 6u] = cell.dependencyMin;
    rtgiField[base + 7u] = cell.dependencyMax;
}

RtgiCacheCell rtgiSnapshotCell(uint slot) {
    uint base = rtgiSnapshotBase(slot);
    RtgiCacheCell cell;
    cell.direct = rtgiField[base];
    cell.indirect = rtgiField[base + 1u];
    cell.reflectance = rtgiField[base + 2u];
    cell.origin = rtgiField[base + 3u];
    cell.direction = rtgiField[base + 4u];
    cell.hit = asuint(rtgiField[base + 5u]);
    cell.dependencyMin = rtgiField[base + 6u];
    cell.dependencyMax = rtgiField[base + 7u];
    return cell;
}

// Final gather noise: interleaved gradient noise (Jimenez 2014) decorrelates
// neighbouring samples and the R2 sequence (Roberts 2018) steps it each
// frame, a cheap stand-in for idTech 8's blue noise.
float2 rtgiGatherNoise(uint2 pixel) {
    float2 offset = float2(
        frac(52.9829189 * frac(dot(float2(pixel), float2(0.06711056, 0.00583715)))),
        frac(52.9829189 * frac(dot(float2(pixel) + float2(47.0, 17.0),
            float2(0.06711056, 0.00583715)))));
    return frac(r2(params.frame.x) + offset);
}

struct RtgiReceiver {
    bool valid;
    float3 position;
    float3 normal;
    float3 geometricNormal;
};
// The receiver and eligibility of worldResolve.
RtgiReceiver rtgiReceiver(uint2 pixel) {
    RtgiReceiver receiver = (RtgiReceiver)0;
    uint scale = giResolveScale(params);
    uint2 full = params.viewport.xy + min(pixel * scale + scale / 2u, params.viewport.zw - 1u);
    uint4 deferred = gbuffer.Load(int3(full, 0));
    uint4 raster = rasterReceiver.Load(int3(full, 0));
    float depth = depthTexture.Load(int3(full, 0));
    if (!(depth > 0.0) || !isfinite(depth) || (raster.z & 0x80000000u) == 0u
        || (deferred.w & (1u << 24u)) != 0u) return receiver;
    float3 normal = deferredShadingNormal(deferred);
    if (!all(isfinite(normal)) || dot(normal, normal) <= 0.01) return receiver;
    receiver.normal = normalize(normal);
    receiver.geometricNormal = receiverGeometricNormal(raster.z, receiver.normal);
    receiver.position = worldPosition(float2(full) + 0.5, depth, params);
    receiver.valid = all(isfinite(receiver.position));
    return receiver;
}

// Plane-distance tolerance for reusing another receiver's samples.
float rtgiPlaneTolerance(RtgiReceiver receiver) {
    return max(length(receiver.position - params.cameraPosition.xyz) * 0.0025, 0.001);
}

// The previous frame's full-resolution position of a resolve sample.
bool rtgiReproject(uint2 pixel, RtgiReceiver receiver, out float2 previousFull) {
    uint scale = giResolveScale(params);
    uint2 full = params.viewport.xy + min(pixel * scale + scale / 2u, params.viewport.zw - 1u);
    if (params.stability.z > 0.5) {
        float2 motion = motionTexture.Load(int3(full, 0)).xy;
        previousFull = float2(full) + 0.5 - motion * float2(params.viewport.zw);
        return all(isfinite(motion));
    }
    float4 clip = mul(params.previousClipFromWorld, float4(receiver.position, 1));
    float3 ndc = clip.xyz / clip.w;
    float2 uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    previousFull = float2(params.viewport.xy) + uv * float2(params.viewport.zw);
    return clip.w > 0 && all(isfinite(uv));
}

// idTech 8's spatial denoise: separable bilateral Gaussian over the gather
// SH, weighted by normal, plane distance and hit-distance similarity, so
// contact shadows from short rays are not smeared across open areas.
float rtgiHitWeight(float a, float b) {
    a = clamp(a, 1e-3, 100.0);
    b = clamp(b, 1e-3, 100.0);
    return sqrt(min(a, b) / max(a, b));
}
void rtgiDenoise(uint2 pixel, int2 axis, uint source, uint target) {
    uint2 extent = giResolveExtent(params);
    if (any(pixel >= extent)) return;
    uint index = pixel.x + pixel.y * extent.x;
    uint input = rtgiGatherBase(source, index);
    uint output = rtgiGatherBase(target, index);
    // rtgiResolve stored each receiver's geometry, so taps need no G-buffer reads.
    float4 position = rtgiField[input + 4u];
    float4 normal = rtgiField[input + 5u];
    rtgiField[output + 4u] = position;
    rtgiField[output + 5u] = normal;
    float4 centre = rtgiField[input + 3u];
    bool centreValid = centre.y > 0.5;
    float tolerance = 4.0 * max(length(position.xyz - params.cameraPosition.xyz) * 0.0025, 0.001);
    RtgiSh sum = rtgiShZero();
    float weightSum = 0.0, hitSum = 0.0;
    if (position.w > 0.5) {
        [loop] for (int k = -RTGI_DENOISE_RADIUS; k <= RTGI_DENOISE_RADIUS; ++k) {
            int2 tap = int2(pixel) + axis * k;
            if (any(tap < 0) || any(tap >= int2(extent))) continue;
            uint base = rtgiGatherBase(source, uint(tap.x) + uint(tap.y) * extent.x);
            float4 meta = rtgiField[base + 3u];
            if (!(meta.y > 0.5)) continue;
            float weight = exp(-0.125 * float(k * k));
            if (k != 0) {
                float facing = saturate(dot(rtgiField[base + 5u].xyz, normal.xyz));
                facing *= facing; facing *= facing; facing *= facing;
                weight *= facing * facing
                    * saturate(1.0 - abs(dot(rtgiField[base + 4u].xyz - position.xyz, normal.xyz))
                        / tolerance);
                if (centreValid) weight *= rtgiHitWeight(meta.x, centre.x);
            }
            if (!(weight > 1e-4)) continue;
            rtgiShAccumulate(sum, rtgiLoadSh(base), weight);
            hitSum += meta.x * weight;
            weightSum += weight;
        }
    }
    if (!(weightSum > 1e-4)) {
        [unroll] for (uint word = 0u; word < 4u; ++word)
            rtgiField[output + word] = 0.0;
        return;
    }
    RtgiSh filtered = rtgiShScale(sum, 1.0 / weightSum);
    rtgiField[output] = filtered.r;
    rtgiField[output + 1u] = filtered.g;
    rtgiField[output + 2u] = filtered.b;
    rtgiField[output + 3u] = float4(hitSum / weightSum, 1.0, 0.0, 0.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void rtgiDenoiseX(uint3 thread : SV_DispatchThreadID) {
    rtgiDenoise(thread.xy, int2(1, 0), 0u, 1u);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void rtgiDenoiseY(uint3 thread : SV_DispatchThreadID) {
    rtgiDenoise(thread.xy, int2(0, 1), 1u, 0u);
}

// idTech 8's temporal filter: the reprojected history blends with the new
// SH at 1/n, where n counts the texel's samples up to `history_frames`.
// Texels with no new sample keep their history rather than turning black.
[shader("compute")]
[numthreads(8, 8, 1)]
void rtgiTemporal(uint3 thread : SV_DispatchThreadID) {
    uint2 extent = giResolveExtent(params);
    if (any(thread.xy >= extent)) return;
    uint index = thread.x + thread.y * extent.x;
    uint parity = params.frame.x & 1u;
    uint output = rtgiHistoryBase(parity, index);
    RtgiReceiver receiver = rtgiReceiver(thread.xy);
    if (!receiver.valid) {
        [unroll] for (uint word = 0u; word < RTGI_HISTORY_WORDS; ++word)
            rtgiField[output + word] = 0.0;
        return;
    }
    uint receiverScale = giResolveScale(params);
    uint2 full = params.viewport.xy + min(thread.xy * receiverScale + receiverScale / 2u,
        params.viewport.zw - 1u);
    uint2 identity = rasterReceiver.Load(int3(full, 0)).xy;
    uint currentBase = rtgiGatherBase(0u, index);
    float4 meta = rtgiField[currentBase + 3u];
    bool current = meta.y > 0.5;

    RtgiSh history = rtgiShZero();
    float historyCount = 0.0, historyHit = 0.0, historyWeight = 0.0;
    float2 previousFull;
    if (params.rayTraced.w != 0u && params.history.w > 0.0
        && rtgiReproject(thread.xy, receiver, previousFull)) {
        uint scale = giResolveScale(params);
        float2 grid = (previousFull - float2(params.viewport.xy)
            - (float(scale / 2u) + 0.5)) / float(scale);
        int2 base = int2(floor(grid));
        float2 last = float2(params.viewport.zw - 1u);
        float2 left = min(float2(base * int(scale) + int(scale / 2u)), last);
        float2 right = min(float2((base + 1) * int(scale) + int(scale / 2u)), last);
        float2 fraction = saturate((previousFull - float2(params.viewport.xy) - 0.5 - left)
            / max(right - left, 1.0));
        float tolerance = 2.0 * rtgiPlaneTolerance(receiver);
        [unroll] for (uint tap = 0u; tap < 4u; ++tap) {
            int2 bit = int2(tap & 1u, tap >> 1u);
            int2 previousPixel = base + bit;
            if (any(previousPixel < 0) || any(previousPixel >= int2(extent))) continue;
            uint previousBase = rtgiHistoryBase(1u - parity,
                uint(previousPixel.x) + uint(previousPixel.y) * extent.x);
            float4 position = rtgiField[previousBase + 3u];
            float4 normal = rtgiField[previousBase + 4u];
            if (!(position.w > 0.0) || any(asuint(rtgiField[previousBase + 5u].xy) != identity)
                || dot(normal.xyz, receiver.geometricNormal) < 0.9
                || abs(dot(position.xyz - receiver.position, receiver.geometricNormal)) > tolerance)
                continue;
            float2 axisWeight = lerp(1.0 - fraction, fraction, float2(bit));
            float weight = axisWeight.x * axisWeight.y;
            if (!(weight > 1e-4)) continue;
            rtgiShAccumulate(history, rtgiLoadSh(previousBase), weight);
            historyCount += position.w * weight;
            historyHit += normal.w * weight;
            historyWeight += weight;
        }
        if (historyWeight > 1e-4) {
            history = rtgiShScale(history, 1.0 / historyWeight);
            historyCount /= historyWeight;
            historyHit /= historyWeight;
        } else {
            historyCount = 0.0;
        }
    }
    float limit = clamp(params.history.w, 1.0, 64.0);
    // A light source changed: follow it within a few frames.
    if (params.sourceHistory.y != 0u) historyCount = min(historyCount, 4.0);
    historyCount = min(historyCount, limit);
    RtgiSh result = history;
    float count = historyCount, hit = historyHit;
    if (current) {
        count = min(historyCount + 1.0, limit);
        float blend = 1.0 / max(count, 1.0);
        result = rtgiShLerp(history, rtgiLoadSh(currentBase), blend);
        hit = lerp(historyHit, meta.x, blend);
    }
    rtgiField[output] = result.r;
    rtgiField[output + 1u] = result.g;
    rtgiField[output + 2u] = result.b;
    rtgiField[output + 3u] = float4(receiver.position, count);
    rtgiField[output + 4u] = float4(receiver.geometricNormal, hit);
    rtgiField[output + 5u] = float4(asfloat(identity), 0.0, 0.0);
}

// Irradiance volume probes updated this frame: half of one cascade.
void rtgiVolumeBatchProbe(uint lane, out uint cascade, out uint slot) {
    uint phase = params.frame.x % (2u * RTGI_VOLUME_CASCADES);
    cascade = phase / 2u;
    slot = (phase & 1u) * RTGI_VOLUME_BATCH + lane;
}

// Rays of a probe update: a spherical Fibonacci set under a random rotation
// per update (DDGI).
float3 rtgiVolumeRayDirection(uint ray) {
    float z = 1.0 - (2.0 * float(ray) + 1.0) / float(RTGI_VOLUME_RAYS);
    float phi = 2.0 * PI * frac(float(ray) * 0.618033989);
    float radius = sqrt(max(1.0 - z * z, 0.0));
    float3 direction = float3(radius * cos(phi), radius * sin(phi), z);
    uint state = rtgiHash(params.frame.x * 7919u + 13u);
    float u1 = rtgiRandom(state), u2 = 2.0 * PI * rtgiRandom(state), u3 = 2.0 * PI * rtgiRandom(state);
    float4 q = float4(sqrt(1.0 - u1) * sin(u2), sqrt(1.0 - u1) * cos(u2),
        sqrt(u1) * sin(u3), sqrt(u1) * cos(u3));
    float3 t = 2.0 * cross(q.xyz, direction);
    return normalize(direction + q.w * t + cross(q.xyz, t));
}

// Ray kinds in the probe ray scratch.
static const uint RTGI_RAY_UNSUPPORTED = 0u;
static const uint RTGI_RAY_SURFACE = 1u;
static const uint RTGI_RAY_SKY = 2u;
static const uint RTGI_RAY_UNSHADED = 3u; // a surface whose radiance is unknown yet

groupshared float4 rtgiProbeRays[RTGI_VOLUME_RAYS];     // direction, distance
groupshared float4 rtgiProbeRadiance[RTGI_VOLUME_RAYS]; // radiance, kind

// Integrates the rays rtgiVolumeTrace stored into one probe, after this
// frame's cache shading: L1 SH of surface and sky radiance, and DDGI-style
// distance moments for visibility. One workgroup per probe.
[shader("compute")]
[numthreads(64, 1, 1)]
void rtgiVolumeUpdate(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint probeLane = group.x;
    uint t = local.x;
    uint cascade, slot;
    rtgiVolumeBatchProbe(probeLane, cascade, slot);
    int3 coordinate = rtgiVolumeCoordinate(cascade, slot);
    uint base = rtgiProbeBase(cascade, slot);
    float maxDistance = rtgiVolumeCellSize(cascade) * float(RTGI_VOLUME_SIDE);
    if (t < RTGI_VOLUME_RAYS) {
        uint rayBase = rtgiRayBase(probeLane * RTGI_VOLUME_RAYS + t);
        float4 ray = rtgiField[rayBase];
        float4 info = rtgiField[rayBase + 1u];
        uint kind = uint(info.z);
        float3 radiance = 0.0;
        if (kind == RTGI_RAY_SURFACE) {
            uint cellSlot = asuint(info.x);
            kind = RTGI_RAY_UNSHADED;
            // The key check drops a cell evicted and reused since the trace.
            if (cellSlot < RTGI_CACHE_CAPACITY
                && rtgiCacheWords[cellSlot].load() == asuint(info.y)) {
                RtgiCacheCell cell = rtgiCacheCells[cellSlot];
                if (cell.direct.w > 0.0) {
                    radiance = rtgiCellRadiance(cell);
                    kind = RTGI_RAY_SURFACE;
                }
            }
        } else if (kind == RTGI_RAY_SKY) {
            radiance = environmentRadiance(ray.xyz);
            if (!all(isfinite(radiance))) radiance = 0.0;
        }
        rtgiProbeRays[t] = float4(ray.xyz, clamp(ray.w, 0.0, maxDistance));
        rtgiProbeRadiance[t] = float4(max(radiance, 0.0), float(kind));
    }
    GroupMemoryBarrierWithGroupSync();
    float4 header = rtgiField[base];
    bool reset = !(header.w > 0.0) || any(asint(header.xyz) != coordinate);
    float count = reset ? 0.0 : header.w;
    float blend = max(1.0 / (count + 1.0), RTGI_VOLUME_MIN_BLEND);
    uint known = 0u;
    [loop] for (uint r = 0u; r < RTGI_VOLUME_RAYS; ++r) {
        uint kind = uint(rtgiProbeRadiance[r].w);
        known += uint(kind == RTGI_RAY_SURFACE || kind == RTGI_RAY_SKY);
    }
    // Every lane has read the header before lane 0 rewrites it.
    GroupMemoryBarrierWithGroupSync();
    // Missing surface radiance is not a sample of black or sky. Renormalizing
    // only known directions overrepresents sky during cache warmup; preserve
    // the previous probe until this entire ray batch is supported.
    if (known != RTGI_VOLUME_RAYS) return;
    if (t < 6u) {
        // Lanes 0-2: surface SH r, g, b; lanes 3-5: sky SH r, g, b.
        uint source = t < 3u ? RTGI_RAY_SURFACE : RTGI_RAY_SKY;
        uint channel = t % 3u;
        float4 sum = 0.0;
        [loop] for (uint r = 0u; r < RTGI_VOLUME_RAYS; ++r) {
            float4 radiance = rtgiProbeRadiance[r];
            if (uint(radiance.w) != source) continue;
            sum += radiance[channel] * rtgiShBasis(rtgiProbeRays[r].xyz);
        }
        sum *= 4.0 * PI / float(RTGI_VOLUME_RAYS);
        uint word = base + 1u + t;
        rtgiField[word] = reset ? sum : lerp(rtgiField[word], sum, blend);
    }
    if (t < RTGI_VISIBILITY_SIDE * RTGI_VISIBILITY_SIDE / 2u) {
        uint word = base + 7u + t;
        float4 moments = rtgiField[word];
        [unroll] for (uint pair = 0u; pair < 2u; ++pair) {
            uint texel = 2u * t + pair;
            float3 direction = rtgiOctDecode((float2(texel % RTGI_VISIBILITY_SIDE,
                texel / RTGI_VISIBILITY_SIDE) + 0.5) / float(RTGI_VISIBILITY_SIDE));
            float weightSum = 0.0;
            float2 sum = 0.0;
            [loop] for (uint r = 0u; r < RTGI_VOLUME_RAYS; ++r) {
                if (uint(rtgiProbeRadiance[r].w) == RTGI_RAY_UNSUPPORTED) continue;
                float4 ray = rtgiProbeRays[r];
                float weight = saturate(dot(direction, ray.xyz));
                weight *= weight; weight *= weight; weight *= weight;
                sum += weight * float2(ray.w, ray.w * ray.w);
                weightSum += weight;
            }
            float2 previous = pair == 0u ? moments.xy : moments.zw;
            float2 next = previous;
            if (weightSum > 1e-4)
                next = reset ? sum / weightSum : lerp(previous, sum / weightSum, blend);
            else if (reset)
                next = float2(maxDistance, maxDistance * maxDistance);
            if (pair == 0u) moments.xy = next; else moments.zw = next;
        }
        rtgiField[word] = moments;
    }
    if (t == 0u)
        rtgiField[base] = float4(asfloat(coordinate), min(count + 1.0, 1.0 / RTGI_VOLUME_MIN_BLEND));
}

#ifdef PENUMBRA_HARDWARE_RAY_QUERY
// Traces the shadow rays rtgiCacheRecord stored.
[shader("compute")]
[numthreads(64, 1, 1)]
void rtgiCacheVisibility(uint3 thread : SV_DispatchThreadID) {
    if (thread.x >= heads[deferredQueryCountHead(params)].load()) return;
    transportAdmissionBypass = true;
    traceVisibilityQuery(heads[deferredQueryListHead(params) + thread.x].load());
}

// Multi-bounce (GI-1.2): traces one cosine-weighted ray from each cell shaded
// this frame and accumulates the radiance found into the cell's indirect
// estimator. The cell that ray hits carries its own indirect light, so light
// gains a bounce per cache update.
[shader("compute")]
[numthreads(64, 1, 1)]
void rtgiCacheBounce(uint3 thread : SV_DispatchThreadID) {
    uint lane = thread.x;
    if (lane >= RTGI_CACHE_WORK) return;
    uint slot = rtgiCacheWords[rtgiWorkHead(lane)].load();
    if (slot == EMPTY || slot >= RTGI_CACHE_CAPACITY) return;
    if (params.rayTraced.x == 0u) {
        rtgiCacheCells[slot].indirect = 0.0;
        return;
    }
    transportAdmissionBypass = true;
    RtgiCacheCell cell = rtgiSnapshotCell(slot);
    float3 position = cell.origin.xyz + cell.direction.xyz * cell.origin.w;
    float3 normal = hitGeometricNormal(rtgiCellHit(cell));
    if (dot(normal, cell.direction.xyz) > 0.0) normal = -normal;
    uint state = rtgiHash(slot * 6271u ^ rtgiHash(params.frame.x * 9781u + 3u));
    float3 direction = rtgiCosineDirection(normal, float2(rtgiRandom(state), rtgiRandom(state)));
    float3 origin = rayOffset(position, normal, params.tuning.z);
    DependencyContext traceDependency;
    float dependencyStart, dependencyDistance, dependencyPadding;
    Hit hit = traceRayTracked(origin, direction, 1e20, traceDependency,
        dependencyStart, dependencyDistance, dependencyPadding);
    if ((hit.errors & HIT_QUERY_UNSUPPORTED) != 0u) return;
    DependencyContext dependency = rtgiCellDependency(cell);
    dependencyUnion(dependency, traceDependency);
    dependencyIncludeSegment(dependency, position, position, params.tuning.z);
    // Escaped rays add nothing: cell shading takes the sky through NEE.
    float3 incoming = 0.0;
    // Requests go on the next frame's queue: this frame's is already drained.
    if (hit.triangle != EMPTY) {
        float3 hitPosition = origin + direction * hit.distance;
        float3 hitNormal = hitGeometricNormal(hit);
        if (dot(hitNormal, direction) > 0.0) hitNormal = -hitNormal;
        uint target = rtgiCacheTouch(origin, direction, hit, hitPosition, hitNormal,
            1u - (params.frame.x & 1u), state);
        RtgiCacheCell targetCell = (RtgiCacheCell)0;
        if (target != EMPTY) targetCell = rtgiSnapshotCell(target);
        if (target != EMPTY && targetCell.direct.w > 0.0) {
            incoming = rtgiCellRadiance(targetCell);
            dependencyUnion(dependency, rtgiCellDependency(targetCell));
        } else {
            if (!rtgiVolumeRadiance(hit, hitPosition, hitNormal, direction, incoming)) return;
            // Volume samples do not yet carry transitive geometry bounds.
            dependency = dependencyUnknown();
        }
    }
    float count = cell.indirect.w + 1.0;
    float blend = max(1.0 / count, RTGI_CACHE_MIN_BLEND);
    rtgiCacheCells[slot].indirect = float4(lerp(cell.indirect.rgb, incoming, blend),
        min(count, 1.0 / RTGI_CACHE_MIN_BLEND));
    rtgiStoreCellDependency(slot, dependency);
}

// idTech 8's world sampling: each probe of this frame's batch traces
// RTGI_VOLUME_RAYS rays. Hits claim world cache cells and request their
// shading; rtgiVolumeUpdate reads the cells once they are shaded.
[shader("compute")]
[numthreads(64, 1, 1)]
void rtgiVolumeTrace(uint3 thread : SV_DispatchThreadID) {
    uint lane = thread.x;
    if (lane >= RTGI_VOLUME_BATCH * RTGI_VOLUME_RAYS) return;
    uint probeLane = lane / RTGI_VOLUME_RAYS;
    uint cascade, slot;
    rtgiVolumeBatchProbe(probeLane, cascade, slot);
    float3 origin = rtgiVolumeProbePosition(cascade, rtgiVolumeCoordinate(cascade, slot));
    float3 direction = rtgiVolumeRayDirection(lane % RTGI_VOLUME_RAYS);
    transportAdmissionBypass = true;
    DependencyContext traceDependency;
    float dependencyStart, dependencyDistance, dependencyPadding;
    Hit hit = traceRayTracked(origin, direction, 1e20, traceDependency,
        dependencyStart, dependencyDistance, dependencyPadding);
    float4 record = float4(direction, RTGI_ESCAPE_DISTANCE);
    float4 info = float4(asfloat(EMPTY), 0.0, float(RTGI_RAY_SKY), 0.0);
    if ((hit.errors & HIT_QUERY_UNSUPPORTED) != 0u) {
        info.z = float(RTGI_RAY_UNSUPPORTED);
    } else if (hit.triangle != EMPTY) {
        record.w = hit.distance;
        float3 position = origin + direction * hit.distance;
        float3 normal = hitGeometricNormal(hit);
        if (dot(normal, direction) > 0.0) normal = -normal;
        uint state = rtgiHash(lane * 9781u ^ rtgiHash(params.frame.x * 6271u + 5u));
        uint cellSlot = rtgiCacheTouch(origin, direction, hit, position, normal,
            params.frame.x & 1u, state);
        info = float4(asfloat(cellSlot),
            asfloat(cellSlot == EMPTY ? 0u : rtgiCacheWords[cellSlot].load()),
            float(cellSlot == EMPTY ? RTGI_RAY_UNSHADED : RTGI_RAY_SURFACE), 0.0);
    }
    uint rayBase = rtgiRayBase(lane);
    rtgiField[rayBase] = record;
    rtgiField[rayBase + 1u] = info;
}

// Target function: luminance of the cosine-weighted incoming radiance. The
// constant 1/pi cancels in every resampling ratio.
float rtgiTarget(RtgiReservoir reservoir, RtgiReceiver receiver) {
    float3 toSample = reservoir.sample.xyz - receiver.position;
    float lengthSquared = dot(toSample, toSample);
    if (!(lengthSquared > 1e-12)) return 0.0;
    float3 direction = toSample * rsqrt(lengthSquared);
    float cosine = dot(receiver.normal, direction);
    if (cosine <= 0.0 || dot(receiver.geometricNormal, direction) <= 0.0) return 0.0;
    return rtgiLuminance(reservoir.radiance.rgb) * cosine;
}

// Reconnection Jacobian from `source` to `receiver` (ReSTIR GI, Eq. 11).
// Escaped vertices are directions and need none.
float rtgiJacobian(RtgiReservoir reservoir, float3 source, float3 receiver) {
    if (dot(reservoir.normal.xyz, reservoir.normal.xyz) <= 0.0) return 1.0;
    float3 toSource = source - reservoir.sample.xyz;
    float3 toReceiver = receiver - reservoir.sample.xyz;
    float sourceDistance = dot(toSource, toSource);
    float receiverDistance = dot(toReceiver, toReceiver);
    float sourceCosine = abs(dot(reservoir.normal.xyz, toSource * rsqrt(max(sourceDistance, 1e-12))));
    float receiverCosine = abs(dot(reservoir.normal.xyz, toReceiver * rsqrt(max(receiverDistance, 1e-12))));
    if (!(sourceCosine > 1e-4) || !(receiverDistance > 1e-8)) return 0.0;
    // shortcut: clamped against grazing-angle fireflies at a small bias; replace with MIS-weighted reuse if it shows.
    return min(receiverCosine / sourceCosine * sourceDistance / receiverDistance, 10.0);
}

// Weighted reservoir streaming; `weight` already includes the candidate's M.
// Returns true when the candidate was selected.
bool rtgiMerge(inout RtgiReservoir reservoir, inout float weightSum,
               RtgiReservoir candidate, float weight, float count, inout uint state) {
    reservoir.normal.w += count;
    if (!(weight > 0.0) || !isfinite(weight)) return false;
    weightSum += weight;
    if (rtgiRandom(state) * weightSum >= weight) return false;
    reservoir.sample.xyz = candidate.sample.xyz;
    reservoir.normal.xyz = candidate.normal.xyz;
    reservoir.radiance = candidate.radiance;
    return true;
}

// Final gather: one unshaded hardware ray per resolve sample, radiance from
// the cache levels, then temporal reservoir reuse.
[shader("compute")]
[numthreads(8, 8, 1)]
void rtgiSample(uint3 thread : SV_DispatchThreadID) {
    uint2 extent = giResolveExtent(params);
    if (any(thread.xy >= extent)) return;
    uint index = thread.x + thread.y * extent.x;
    RtgiReceiver receiver = rtgiReceiver(thread.xy);
    RtgiReservoir reservoir = (RtgiReservoir)0;
    if (!receiver.valid) {
        rtgiReservoirs[index] = reservoir;
        return;
    }
    transportAdmissionBypass = true;
    uint state = rtgiHash(index * 9781u ^ rtgiHash(params.frame.x * 6271u + 1u));

    // Initial candidate: one cosine-weighted path vertex.
    float3 direction = rtgiCosineDirection(receiver.normal, rtgiGatherNoise(thread.xy));
    RtgiReservoir candidate = (RtgiReservoir)0;
    float candidateWeight = 0.0;
    float candidateCount = 1.0;
    if (dot(direction, receiver.geometricNormal) > 0.0) {
        float3 origin = rayOffset(receiver.position, receiver.geometricNormal, params.tuning.z);
        DependencyContext traceDependency;
        float dependencyStart, dependencyDistance, dependencyPadding;
        Hit hit = traceRayTracked(origin, direction, 1e20, traceDependency,
            dependencyStart, dependencyDistance, dependencyPadding);
        if ((hit.errors & HIT_QUERY_UNSUPPORTED) == 0u) {
            if (hit.triangle == EMPTY) {
                // Bevy's environment light lights the receiver directly, so an
                // escaped primary ray carries supported zero, as in the cascades.
                candidate.sample.xyz = origin + direction * RTGI_ESCAPE_DISTANCE;
            } else {
                float3 normal = hitGeometricNormal(hit);
                if (dot(normal, direction) > 0.0) normal = -normal;
                candidate.sample.xyz = origin + direction * hit.distance;
                candidate.normal.xyz = normal;
                float3 radiance;
                // A hit no cache level knows yet is unknown rather than black;
                // counting it would darken every newly seen surface for a frame.
                if (rtgiHitRadiance(origin, direction, hit, params.frame.x & 1u, true, state,
                        radiance))
                    candidate.radiance.rgb = radiance;
                else
                    candidateCount = 0.0;
            }
            // Source pdf cos/pi, target luminance*cos: weight pi*luminance.
            candidateWeight = PI * rtgiLuminance(candidate.radiance.rgb);
        } else {
            candidateCount = 0.0;
        }
    }
    float weightSum = 0.0;
    if (!rtgiMerge(reservoir, weightSum, candidate, candidateWeight, candidateCount, state))
        reservoir.sample.xyz = receiver.position + receiver.normal;

    // Temporal reuse of the reprojected previous reservoir.
    float2 previousFull;
    if (params.rayTraced.w != 0u && params.history.w > 0.0 && params.sourceHistory.y == 0u
        && rtgiReproject(thread.xy, receiver, previousFull)) {
        int2 previousPixel = int2(floor((previousFull - float2(params.viewport.xy))
            / float(giResolveScale(params))));
        if (all(previousPixel >= 0) && all(previousPixel < int2(extent))) {
            RtgiReservoir previous =
                rtgiPreviousReservoirs[previousPixel.x + previousPixel.y * extent.x];
            float3 offset = previous.receiver.xyz - receiver.position;
            bool similar = previous.receiver.w > 0.5 && previous.normal.w > 0.0
                && abs(dot(offset, receiver.normal)) < rtgiPlaneTolerance(receiver)
                && dot(offset, offset) < 0.0625;
            if (similar) {
                float count = min(previous.normal.w, float(params.rayTraced.z));
                float weight = rtgiTarget(previous, receiver) * previous.sample.w * count
                    * rtgiJacobian(previous, previous.receiver.xyz, receiver.position);
                float3 delta = previous.sample.xyz - receiver.position;
                float distance = length(delta);
                if (weight > 0.0 && distance > 1e-6) {
                    float3 origin = rayOffset(receiver.position, receiver.geometricNormal, params.tuning.z);
                    float limit = dot(previous.normal.xyz, previous.normal.xyz) > 0.0
                        ? max(distance - params.tuning.z * 2.0, 0.0) : 1e20;
                    if (!traceOccluded(origin, delta / distance, limit))
                        rtgiMerge(reservoir, weightSum, previous, weight, count, state);
                }
            }
        }
    }
    float target = rtgiTarget(reservoir, receiver);
    reservoir.sample.w = target > 0.0 ? weightSum / (reservoir.normal.w * target) : 0.0;
    reservoir.receiver = float4(receiver.position, 1.0);
    rtgiReservoirs[index] = reservoir;
}

// Spatial reservoir reuse, then the raw irradiance output and the gather SH.
[shader("compute")]
[numthreads(8, 8, 1)]
void rtgiResolve(uint3 thread : SV_DispatchThreadID) {
    // rtgiCacheRecord appends this frame's cache shadow rays after this pass.
    if (all(thread.xy == 0u)) heads[deferredQueryCountHead(params)].store(0u);
    uint2 extent = giResolveExtent(params);
    if (any(thread.xy >= extent)) return;
    uint index = thread.x + thread.y * extent.x;
    uint gather = rtgiGatherBase(0u, index);
    RtgiReceiver receiver = rtgiReceiver(thread.xy);
    if (!receiver.valid) {
        irradiance.Store(int2(thread.xy), float4(0.0, 0.0, 0.0, rawGiSupportAlpha(false, false, 0.0)));
        [unroll] for (uint word = 0u; word < RTGI_GATHER_WORDS; ++word)
            rtgiField[gather + word] = 0.0;
        return;
    }
    transportAdmissionBypass = true;
    uint state = rtgiHash(index * 7919u ^ rtgiHash(params.frame.x * 104729u + 7u));
    RtgiReservoir own = rtgiReservoirs[index];
    RtgiReservoir reservoir = own;
    reservoir.normal.w = 0.0;
    float weightSum = 0.0;
    rtgiMerge(reservoir, weightSum, own,
        rtgiTarget(own, receiver) * own.sample.w * own.normal.w, own.normal.w, state);
    bool reused = false;
    [loop] for (uint i = 0u; i < params.rayTraced.y; ++i) {
        float angle = 2.0 * PI * rtgiRandom(state);
        float radius = RTGI_SPATIAL_RADIUS * sqrt(rtgiRandom(state));
        int2 pixel = int2(thread.xy) + int2(round(float2(cos(angle), sin(angle)) * radius));
        if (any(pixel < 0) || any(pixel >= int2(extent)) || all(pixel == int2(thread.xy))) continue;
        RtgiReservoir neighbour = rtgiReservoirs[pixel.x + pixel.y * extent.x];
        if (neighbour.receiver.w < 0.5 || !(neighbour.normal.w > 0.0)) continue;
        RtgiReceiver other = rtgiReceiver(uint2(pixel));
        if (!other.valid || dot(other.normal, receiver.normal) < 0.9
            || abs(dot(other.position - receiver.position, receiver.normal))
                > rtgiPlaneTolerance(receiver)) continue;
        float weight = rtgiTarget(neighbour, receiver) * neighbour.sample.w * neighbour.normal.w
            * rtgiJacobian(neighbour, other.position, receiver.position);
        if (rtgiMerge(reservoir, weightSum, neighbour, weight, neighbour.normal.w, state))
            reused = true;
    }
    float target = rtgiTarget(reservoir, receiver);
    float contribution = target > 0.0 ? weightSum / (reservoir.normal.w * target) : 0.0;
    float3 toSample = reservoir.sample.xyz - receiver.position;
    float sampleDistance = length(toSample);
    float3 direction = sampleDistance > 0.0 ? toSample / sampleDistance : receiver.normal;
    // A neighbour's vertex may be hidden from this receiver: rather than leak
    // light through occluders or drop to black, keep this pixel's own sample.
    if (reused && contribution > 0.0) {
        float3 origin = rayOffset(receiver.position, receiver.geometricNormal, params.tuning.z);
        float limit = dot(reservoir.normal.xyz, reservoir.normal.xyz) > 0.0
            ? max(sampleDistance - params.tuning.z * 2.0, 0.0) : 1e20;
        if (traceOccluded(origin, direction, limit)) {
            reservoir = own;
            contribution = own.sample.w;
            toSample = own.sample.xyz - receiver.position;
            sampleDistance = length(toSample);
            direction = sampleDistance > 0.0 ? toSample / sampleDistance : receiver.normal;
        }
    }
    // Raw output: diffuse irradiance over pi, f(y)·W with f = L·cos/pi.
    float3 diffuse = reservoir.radiance.rgb * (max(dot(receiver.normal, direction), 0.0)
        * contribution / PI);
    // No sample yet (only unknown hits seen): unsupported, so the temporal
    // filter keeps its history instead of blending in black.
    bool supported = all(isfinite(diffuse)) && reservoir.normal.w > 0.0
        && isfinite(contribution);
    if (!supported) diffuse = 0.0;
    irradiance.Store(int2(thread.xy), float4(diffuse, rawGiSupportAlpha(true, supported, 1.0)));
    // f·W estimates any integral over the receiver's hemisphere, so L·Y(w)·W
    // estimates the radiance's SH projection (idTech 8 stores L1 SH per
    // gather texel to keep normal-map detail through the upscale).
    RtgiSh sh = rtgiShZero();
    if (supported) rtgiShAdd(sh, reservoir.radiance.rgb, direction, contribution);
    bool escaped = dot(reservoir.normal.xyz, reservoir.normal.xyz) <= 0.0;
    rtgiField[gather] = sh.r;
    rtgiField[gather + 1u] = sh.g;
    rtgiField[gather + 2u] = sh.b;
    rtgiField[gather + 3u] = float4(escaped ? RTGI_ESCAPE_DISTANCE : sampleDistance,
        supported ? 1.0 : 0.0, 0.0, 0.0);
    rtgiField[gather + 4u] = float4(receiver.position, 1.0);
    rtgiField[gather + 5u] = float4(receiver.normal, 0.0);
}
#endif

int2 wrapIrradianceTexel(int2 pixel) {
    if (pixel.x < 0) pixel = int2(-pixel.x - 1, 5 - pixel.y);
    else if (pixel.x >= 6) pixel = int2(11 - pixel.x, 5 - pixel.y);
    if (pixel.y < 0) pixel = int2(5 - pixel.x, -pixel.y - 1);
    else if (pixel.y >= 6) pixel = int2(5 - pixel.x, 11 - pixel.y);
    return pixel;
}
uint irradianceTexelIndex(int2 pixel) {
    pixel = wrapIrradianceTexel(pixel);
    return uint(pixel.y * 6 + pixel.x);
}


struct DiffuseSources {
    float3 background;
    float3 emissive;
    float3 punctual;
    float3 coverage;
};
// Every spatial corner (and both LODs) uses the same normal. Compute the
// octahedral footprint once per receiver instead of once per probe.
struct IrradianceLookup {
    uint4 texels;
    float4 weights;
};

IrradianceLookup irradianceLookup(float3 normal) {
    IrradianceLookup lookup;
    float2 grid = encodeNormal(normal) * 6.0 - 0.5;
    int2 base = int2(floor(grid));
    float2 fraction = frac(grid);
    float2 complement = 1.0 - fraction;
    lookup.weights = float4(complement.x * complement.y, fraction.x * complement.y,
                            complement.x * fraction.y, fraction.x * fraction.y);
    lookup.texels = uint4(irradianceTexelIndex(base),
                          irradianceTexelIndex(base + int2(1, 0)),
                          irradianceTexelIndex(base + int2(0, 1)),
                          irradianceTexelIndex(base + int2(1, 1)));
    return lookup;
}


float4 probeIrradiance(uint page, uint texel, uint source) {
    if (source == BACKGROUND_SOURCE)
        return directionalProbes[page].irradiance[texel];
    if (source == EMISSIVE_SOURCE)
        return directionalProbes[page].emissiveIrradiance[texel];
    return directionalProbes[page].punctualIrradiance[texel];
}

DiffuseSources diffuseProbePage(uint page, IrradianceLookup lookup) {
    DiffuseSources result = (DiffuseSources)0;
    float3 backgroundSum = 0.0;
    float3 emissiveSum = 0.0;
    float3 punctualSum = 0.0;
    float3 observedWeight = 0.0;
    [unroll] for (uint corner = 0u; corner < 4u; ++corner) {
        float bilinearWeight = lookup.weights[corner];
        if (!(bilinearWeight > 0.0)) continue;
        uint texel = lookup.texels[corner];
        [unroll] for (uint group = 0u; group < 3u; ++group) {
            uint source = 1u << group;
            if (!transportSourceActive(params, source)) continue;
            float4 sample = probeIrradiance(page, texel, source);
            if (!(sample.w > 0.0) || !all(isfinite(sample))) continue;
            float weight = bilinearWeight * saturate(sample.w);
            if (group == 0u) {
                backgroundSum += sample.xyz * weight;
                observedWeight.x += weight;
            } else if (group == 1u) {
                emissiveSum += sample.xyz * weight;
                observedWeight.y += weight;
            } else {
                punctualSum += sample.xyz * weight;
                observedWeight.z += weight;
            }
        }
    }
    if (observedWeight.x > 0.0)
        result.background = backgroundSum / observedWeight.x;
    if (observedWeight.y > 0.0)
        result.emissive = emissiveSum / observedWeight.y;
    if (observedWeight.z > 0.0)
        result.punctual = punctualSum / observedWeight.z;
    result.coverage = saturate(observedWeight);
    if (!transportSourceActive(params, EMISSIVE_SOURCE)) result.coverage.y = 1.0;
    if (!transportSourceActive(params, PUNCTUAL_SOURCE)) result.coverage.z = 1.0;
    return result;
}

DiffuseSources diffuseProbeIrradiance(uint slot, IrradianceLookup lookup) {
    uint page = (params.frame.x & 1u) * probeDirectoryCapacity(params) + slot;
    return diffuseProbePage(page, lookup);
}

DiffuseSources diffuseAtLod(float3 position, IrradianceLookup lookup, uint lod) {
    DiffuseSources result = (DiffuseSources)0;
    float spacingValue = spacing(params, 0u, lod);
    float3 grid = position / spacingValue - 0.5;
    int3 base = int3(floor(grid));
    float3 fraction = frac(grid);
    float3 backgroundSum = 0.0;
    float3 emissiveSum = 0.0;
    float3 punctualSum = 0.0;
    float3 observedWeight = 0.0;
    float potentialWeight = 0.0;
    [loop] for (uint corner = 0u; corner < 8u; ++corner) {
        uint3 bit = uint3(corner & 1u, (corner >> 1u) & 1u, corner >> 2u);
        float3 axis = float3(
            bit.x == 0u ? 1.0 - fraction.x : fraction.x,
            bit.y == 0u ? 1.0 - fraction.y : fraction.y,
            bit.z == 0u ? 1.0 - fraction.z : fraction.z);
        float spatialWeight = axis.x * axis.y * axis.z;
        if (!(spatialWeight > 0.0)) continue;
        potentialWeight += spatialWeight;
        int4 key = int4(base + int3(bit), int(lod << 18u));
        uint slot = findIrradianceProbe(key, false);
        if (slot == EMPTY) continue;
        DiffuseSources probe = diffuseProbeIrradiance(slot, lookup);
        [unroll] for (uint group = 0u; group < 3u; ++group) {
            uint source = 1u << group;
            if (!transportSourceActive(params, source)) continue;
            float support = probe.coverage[group];
            if (!(support > 0.0)) continue;
            float weight = spatialWeight * support;
            if (group == 0u) {
                backgroundSum += probe.background * weight;
                observedWeight.x += weight;
            } else if (group == 1u) {
                emissiveSum += probe.emissive * weight;
                observedWeight.y += weight;
            } else {
                punctualSum += probe.punctual * weight;
                observedWeight.z += weight;
            }
        }
    }
    if (potentialWeight > 0.0) {
        result.coverage = saturate(observedWeight / potentialWeight);
        if (observedWeight.x > 0.0)
            result.background = backgroundSum / observedWeight.x;
        if (observedWeight.y > 0.0)
            result.emissive = emissiveSum / observedWeight.y;
        if (observedWeight.z > 0.0)
            result.punctual = punctualSum / observedWeight.z;
    }
    if (!transportSourceActive(params, EMISSIVE_SOURCE)) result.coverage.y = 1.0;
    if (!transportSourceActive(params, PUNCTUAL_SOURCE)) result.coverage.z = 1.0;
    return result;
}

float3 resolveLodSource(float3 low, float lowCoverage,
                        float3 high, float highCoverage,
                        float lodBlend, out float coverage) {
    float lowWeight = lowCoverage * (1.0 - lodBlend);
    float highWeight = highCoverage * lodBlend;
    float totalWeight = lowWeight + highWeight;
    coverage = saturate(totalWeight);
    return totalWeight > 0.0
        ? (low * lowWeight + high * highWeight) / totalWeight : 0.0;
}

float3 resolveSplitDiffuse(float3 position, float3 normal, out float coverage,
                           out float3 background, out float backgroundCoverage,
                           out float anyCoverage) {
    uint lowLod, highLod;
    float lodBlend;
    splitLodBlend(position, lowLod, highLod, lodBlend);
    IrradianceLookup lookup = irradianceLookup(normal);
    DiffuseSources low = diffuseAtLod(position, lookup, lowLod);
    DiffuseSources high = (DiffuseSources)0;
    if (highLod != lowLod) high = diffuseAtLod(position, lookup, highLod);
    else high.coverage = 1.0;

    float3 resolved = 0.0;
    coverage = 1.0;
    float groupCoverage;
    background = resolveLodSource(low.background, low.coverage.x,
        high.background, high.coverage.x, lodBlend, backgroundCoverage);
    resolved += background;
    coverage = min(coverage, backgroundCoverage);
    anyCoverage = backgroundCoverage;
    if (transportSourceActive(params, EMISSIVE_SOURCE)) {
        resolved += resolveLodSource(low.emissive, low.coverage.y,
            high.emissive, high.coverage.y, lodBlend, groupCoverage);
        coverage = min(coverage, groupCoverage);
        anyCoverage = max(anyCoverage, groupCoverage);
    }
    if (transportSourceActive(params, PUNCTUAL_SOURCE)) {
        resolved += resolveLodSource(low.punctual, low.coverage.z,
            high.punctual, high.coverage.z, lodBlend, groupCoverage);
        coverage = min(coverage, groupCoverage);
        anyCoverage = max(anyCoverage, groupCoverage);
    }
    return resolved;
}

float3 resolveSplitDiffuse(float3 position, float3 normal, out float coverage) {
    float3 background;
    float backgroundCoverage, anyCoverage;
    return resolveSplitDiffuse(position, normal, coverage, background, backgroundCoverage, anyCoverage);
}

#ifndef GI_COMPOSITE_RESOLVE
// Trace bindings 8/9 address the completed secondary cache, while binding 12
// contains its two packed directional pages after the primary pages.
DiffuseSources secondaryAtLod(float3 position, IrradianceLookup lookup, uint lod,
                              inout DependencyContext dependency) {
    DiffuseSources result = (DiffuseSources)0;
    int4 key = cacheKey(position, 0u, 0u, params, lod);
    uint capacity = probeDirectoryCapacity(params);
    uint entryCount, stride;
    previousEntries.GetDimensions(entryCount, stride);
    uint slot = probeDirectoryBucket(key, params);
    [loop] for (uint attempt = 0u; attempt < capacity; ++attempt) {
        uint index = previousHeads[probeDirectoryHead(0u, slot, params)];
        if (index == EMPTY) break;
        if (index != TOMBSTONE && index < entryCount) {
            Entry entry = previousEntries[index];
            if (all(entry.key.xyz == key.xyz)
                && (uint(entry.key.w) & PROBE_LOD) == (uint(key.w) & PROBE_LOD)) {
                uint page = (2u + ((params.frame.x + 1u) & 1u)) * capacity + slot;
                if (directionalProbes[page].cacheGeneration.y == 0u
                    || directionalProbes[page].cacheGeneration.x != params.frame.x - 1u
                    || any(directionalProbes[page].cacheKey != key)) break;
                DependencyContext bounds = probeDependency(
                    directionalProbes[page].dependencyMin, directionalProbes[page].dependencyMax);
                if (dependencyIntersectsDirty(entryMergedDependency(entry))
                    || dependencyIntersectsDirty(bounds)) break;
                result = diffuseProbePage(page, lookup);
                [unroll] for (uint group = 0u; group < 3u; ++group) {
                    uint source = 1u << group;
                    if (!transportSourceActive(params, source) || !(result.coverage[group] > 0.0)
                        || !radianceHistoryEligible(bounds, unknownLightHitBounds(), source)
                        || (source == BACKGROUND_SOURCE && !backgroundMotionHistoryAlive(entry.lastSample)))
                        result.coverage[group] = 0.0;
                }
                if (any(result.coverage > 0.0)) dependencyUnion(dependency, bounds);
                break;
            }
        }
        slot = (slot + 1u) & (capacity - 1u);
    }
    return result;
}

TransportRadiance secondaryFeedback(float3 position, float3 normal,
                                   inout DependencyContext dependency,
                                   out float backgroundCoverage) {
    TransportRadiance result = (TransportRadiance)0;
    backgroundCoverage = 0.0;
    if (params.environmentScale.y <= 0.5 || params.environmentScale.w <= 0.5) return result;
    uint low, high; float blend;
    splitLodBlendForParameters(position, params, low, high, blend);
    low = min(low + 2u, MAX_PROBE_LOD);
    high = min(high + 2u, MAX_PROBE_LOD);
    if (low == high) blend = 0.0;
    IrradianceLookup lookup = irradianceLookup(normal);
    DiffuseSources sample = secondaryAtLod(position, lookup, low, dependency);
    DiffuseSources higher = (DiffuseSources)0;
    if (high != low) higher = secondaryAtLod(position, lookup, high, dependency);
    float coverage;
    result.background = resolveLodSource(sample.background, sample.coverage.x,
        higher.background, higher.coverage.x, blend, coverage);
    backgroundCoverage = coverage;
    if (!(coverage > 0.0) && higher.coverage.x > 0.0) {
        result.background = higher.background;
        backgroundCoverage = higher.coverage.x;
    }
    result.emissive = resolveLodSource(sample.emissive, sample.coverage.y,
        higher.emissive, higher.coverage.y, blend, coverage);
    if (!(coverage > 0.0) && higher.coverage.y > 0.0)
        result.emissive = higher.emissive;
    result.punctual = resolveLodSource(sample.punctual, sample.coverage.z,
        higher.punctual, higher.coverage.z, blend, coverage);
    if (!(coverage > 0.0) && higher.coverage.z > 0.0)
        result.punctual = higher.punctual;
    return result;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void worldResolve(uint3 thread : SV_DispatchThreadID) {
    if (any(thread.xy >= giResolveExtent(params))) return;
    uint scale = giResolveScale(params);
    uint2 full = params.viewport.xy + min(thread.xy * scale + scale / 2u,
                                          params.viewport.zw - 1u);
    uint4 deferred = gbuffer.Load(int3(full, 0));
    uint4 receiver = rasterReceiver.Load(int3(full, 0));
    float depth = depthTexture.Load(int3(full, 0));
    bool eligible = depth > 0.0 && isfinite(depth)
        && (receiver.z & 0x80000000u) != 0u
        && (deferred.w & (1u << 24u)) == 0u;
    float3 diffuse = 0.0;
    float3 background = 0.0;
    float backgroundCoverage = 0.0, anyCoverage = 0.0;
    float coverage = 0.0;
    bool supported = false;
    if (eligible) {
        float3 normal = deferredShadingNormal(deferred);
        if (all(isfinite(normal)) && dot(normal, normal) > 0.01) {
            normal = normalize(normal);
            float3 position = worldPosition(float2(full) + 0.5, depth, params);
            diffuse = resolveSplitDiffuse(position, normal, coverage, background, backgroundCoverage, anyCoverage);
            supported = coverage > 0.0 && all(isfinite(diffuse));
        }
    }
    irradiance.Store(int2(thread.xy), float4(diffuse,
        rawGiSupportAlpha(eligible, supported, coverage)));
    bool backgroundSupported = eligible && backgroundCoverage > 0.0 && all(isfinite(background));
    backgroundOutput.Store(int2(thread.xy), float4(background,
        rawGiSupportAlpha(eligible, backgroundSupported, backgroundCoverage)));
}
#endif

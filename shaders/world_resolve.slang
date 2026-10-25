float3 diffuseProbeIrradiance(uint slot, float3 normal, out float coverage) {
    coverage = 0.0;
    uint page = (params.frame.x & 1u) * probeDirectoryCapacity(params) + slot;
    float3 sum = 0.0;
    float observedCosine = 0.0;
    float nominalCosine = 0.0;
    [loop] for (uint direction = 0u; direction < 32u; ++direction) {
        float3 omega = binDirection(direction, 0u);
        float cosine = max(dot(normal, omega), 0.0);
        nominalCosine += cosine;
        if (cosine <= 0.0) continue;
        float4 sample = directionalProbes[page].radiance[direction];
        if (!(sample.w > 0.0) || !all(isfinite(sample))) continue;
        float weight = cosine * saturate(sample.w);
        sum += sample.xyz * weight;
        observedCosine += weight;
    }
    if (observedCosine <= 1e-6 || nominalCosine <= 1e-6) return 0.0;
    coverage = saturate(observedCosine / nominalCosine);
    // The 32 bins are equal-area: Δω/π = 4/32. Renormalize the observed
    // angular support, then report its coverage separately from valid black.
    return sum / observedCosine * (nominalCosine * (4.0 / 32.0));
}

float3 diffuseAtLod(float3 position, float3 normal, uint lod,
                    out float coverage) {
    float spacingValue = spacing(params, 0u, lod);
    float3 grid = position / spacingValue - 0.5;
    int3 base = int3(floor(grid));
    float3 fraction = frac(grid);
    float3 sum = 0.0;
    float observedWeight = 0.0;
    float potentialWeight = 0.0;
    [loop] for (uint corner = 0u; corner < 8u; ++corner) {
        uint3 bit = uint3(corner & 1u, (corner >> 1u) & 1u, corner >> 2u);
        float3 axis = lerp(1.0 - fraction, fraction, float3(bit));
        float spatialWeight = axis.x * axis.y * axis.z;
        if (spatialWeight <= 1e-6) continue;
        potentialWeight += spatialWeight;
        int4 key = int4(base + int3(bit), int(lod << 18u));
        uint slot = findIrradianceProbe(key, false);
        if (slot == EMPTY) continue;
        float probeCoverage;
        float3 irradianceValue = diffuseProbeIrradiance(slot, normal, probeCoverage);
        if (probeCoverage <= 0.0) continue;
        float weight = spatialWeight * probeCoverage;
        sum += irradianceValue * weight;
        observedWeight += weight;
    }
    coverage = potentialWeight > 0.0
        ? saturate(observedWeight / potentialWeight) : 0.0;
    return observedWeight > 1e-6 ? sum / observedWeight : 0.0;
}

float3 resolveSplitDiffuse(float3 position, float3 normal,
                           out float coverage) {
    uint lowLod, highLod;
    float lodBlend;
    splitLodBlend(position, lowLod, highLod, lodBlend);
    float lowCoverage;
    float3 low = diffuseAtLod(position, normal, lowLod, lowCoverage);
    float lowWeight = lowCoverage * (1.0 - lodBlend);
    float3 sum = low * lowWeight;
    float totalWeight = lowWeight;
    if (highLod != lowLod) {
        float highCoverage;
        float3 high = diffuseAtLod(position, normal, highLod, highCoverage);
        float highWeight = highCoverage * lodBlend;
        sum += high * highWeight;
        totalWeight += highWeight;
    }
    coverage = saturate(totalWeight);
    return totalWeight > 1e-6 ? sum / totalWeight : 0.0;
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
    float coverage = 0.0;
    bool supported = false;
    if (eligible) {
        float3 normal = deferredShadingNormal(deferred);
        if (all(isfinite(normal)) && dot(normal, normal) > 0.01) {
            normal = normalize(normal);
            float3 position = worldPosition(float2(full) + 0.5, depth, params);
            diffuse = resolveSplitDiffuse(position, normal, coverage);
            supported = coverage > 1e-6 && all(isfinite(diffuse));
        }
    }
    irradiance.Store(int2(thread.xy), float4(diffuse,
        rawGiSupportAlpha(eligible, supported, coverage)));
}

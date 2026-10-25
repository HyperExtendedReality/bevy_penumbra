uint gpuSortNext(uint index) { return entries[index].next; }
void gpuSortSetNext(uint index, uint next) { entries[index].next = next; }
bool gpuSortLess(uint left, uint right) {
    int4 a = entries[left].key, b = entries[right].key;
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    if (a.z != b.z) return a.z < b.z;
    if (a.w != b.w) return a.w < b.w;
    return left < right;
}
#include "linked_merge_sort.slang"

bool dependencyIntersectsDirty(DependencyContext dependency) {
    if (params.dirtyMin.w <= 0.0) return false;
    if (!all(isfinite(params.dirtyMin.xyz)) || !all(isfinite(params.dirtyMax.xyz))
        || any(params.dirtyMin.xyz > params.dirtyMax.xyz)) return true;
    // An untracked dependency cannot prove the observation is independent of
    // changed geometry. A measured-empty set remains valid under the root-revision
    // precondition enforced by the host.
    if (!dependency.known) return true;
    if (!dependency.hasBounds) return false;
    return all(dependency.minimum <= params.dirtyMax.xyz)
        && all(dependency.maximum >= params.dirtyMin.xyz);
}

bool localPunctualLightHistoryAvailable() {
    return persistentWorldIntervals(params)
        && params.lightDirtyMin.w > 0.5
        && all(isfinite(params.lightDirtyMin.xyz))
        && all(isfinite(params.lightDirtyMax.xyz))
        && all(params.lightDirtyMin.xyz <= params.lightDirtyMax.xyz);
}

// Raw and merged lighting bounds store a conservative sphere around only the
// surfaces where outgoing radiance was evaluated. Radius > 0 is a measured
// sphere, -1 is proven lighting-independent zero, and all other values are
// unknown (including zero-initialized legacy/reserved storage).
float4 emptyLightHitBounds() { return float4(0.0, 0.0, 0.0, -1.0); }
float4 unknownLightHitBounds() { return 0.0; }

float4 lightHitBoundsAt(float3 position) {
    if (!all(isfinite(position))) return unknownLightHitBounds();
    float scale = max(1.0, max(abs(position.x), max(abs(position.y), abs(position.z))));
    return float4(position, max(scale * 1e-6, 1.1920929e-7));
}

float4 dependencySegmentSphere(float3 start, float3 end, float padding) {
    if (!all(isfinite(start)) || !all(isfinite(end)) || !isfinite(padding))
        return unknownLightHitBounds();
    float3 delta = end - start;
    float3 center = start + delta * 0.5;
    float scale = max(1.0, max(abs(start.x), max(abs(start.y), abs(start.z))));
    scale = max(scale, max(abs(end.x), max(abs(end.y), abs(end.z))));
    float radius = length(delta) * 0.5 + max(padding, 0.0)
        + max(scale * 1e-6, 1.1920929e-7);
    if (!all(isfinite(center)) || !isfinite(radius) || !(radius > 0.0))
        return unknownLightHitBounds();
    return float4(center, radius);
}

float4 unionConservativeSpheres(float4 left, float4 right) {
    bool leftFinite = all(isfinite(left));
    bool rightFinite = all(isfinite(right));
    bool leftEmpty = leftFinite && left.w == -1.0;
    bool rightEmpty = rightFinite && right.w == -1.0;
    bool leftSphere = leftFinite && left.w > 0.0;
    bool rightSphere = rightFinite && right.w > 0.0;
    if (leftEmpty) return rightFinite && (rightEmpty || rightSphere)
        ? right : unknownLightHitBounds();
    if (rightEmpty) return leftSphere ? left : unknownLightHitBounds();
    if (!leftSphere || !rightSphere) return unknownLightHitBounds();

    float3 delta = right.xyz - left.xyz;
    float distance = length(delta);
    if (!isfinite(distance)) return unknownLightHitBounds();
    if (left.w >= distance + right.w) return left;
    if (right.w >= distance + left.w) return right;
    float radius = 0.5 * (distance + left.w + right.w);
    float3 center = left.xyz + delta * ((radius - left.w) / max(distance, 1e-20));
    if (!all(isfinite(center)) || !isfinite(radius) || !(radius > 0.0))
        return unknownLightHitBounds();
    return float4(center, radius);
}

bool dependencySphereIntersectsDirty(float4 bounds) {
    if (params.dirtyMin.w <= 0.0) return false;
    if (!all(isfinite(params.dirtyMin.xyz)) || !all(isfinite(params.dirtyMax.xyz))
        || any(params.dirtyMin.xyz > params.dirtyMax.xyz)) return true;
    if (!all(isfinite(bounds)) || !(bounds.w > 0.0)) return true;
    float3 closest = clamp(bounds.xyz, params.dirtyMin.xyz, params.dirtyMax.xyz);
    float3 delta = bounds.xyz - closest;
    return dot(delta, delta) <= bounds.w * bounds.w;
}

bool betaHistoryEligible(DependencyContext combinedDependency, float4 segmentSphere) {
    bool combinedIntersectsDirty = dependencyIntersectsDirty(combinedDependency);
    // Reserved zero/malformed spheres came from old or incomplete cache entries.
    // Fall back to the combined dependency rather than assuming a short segment.
    if (!all(isfinite(segmentSphere)) || !(segmentSphere.w > 0.0))
        return !combinedIntersectsDirty;
    // The AABB is often much tighter than a sphere around a long ray segment.
    // Either disjoint proof is sufficient; both must overlap to reject beta.
    return !combinedIntersectsDirty || !dependencySphereIntersectsDirty(segmentSphere);
}

bool lightHitBoundsIntersectLocalLightDirty(float4 bounds) {
    if (!localPunctualLightHistoryAvailable()) return false;
    if (all(isfinite(bounds)) && bounds.w == -1.0) return false;
    if (!all(isfinite(bounds)) || !(bounds.w > 0.0)) return true;
    float3 closest = clamp(bounds.xyz, params.lightDirtyMin.xyz, params.lightDirtyMax.xyz);
    float3 delta = bounds.xyz - closest;
    return dot(delta, delta) <= bounds.w * bounds.w;
}

bool radianceHistoryEligible(DependencyContext geometryDependency,
                             float4 lightHitBounds) {
    // Bounded geometry reuse permits the global transport flag to stay open,
    // but it does not make a dependency crossing the changed geometry valid.
    if (dependencyIntersectsDirty(geometryDependency)) return false;
    // Empty intervals and escaped Split rays carry measured zero J; their
    // radiance cannot depend on local lights, emissive sources, or the sky.
    if (all(isfinite(lightHitBounds)) && lightHitBounds.w == -1.0) return true;
    if (worldRadianceUnchanged(params)) return true;
    return localPunctualLightHistoryAvailable()
        && !lightHitBoundsIntersectLocalLightDirty(lightHitBounds);
}

bool radianceHistoryMayBeReused() {
    return worldRadianceUnchanged(params) || localPunctualLightHistoryAvailable();
}

bool dirtyDependency(Entry entry) {
    return dependencyIntersectsDirty(entryMergedDependency(entry));
}

bool dirtyRawCacheEntry(Entry entry) {
    return dependencyIntersectsDirty(entryDependency(entry));
}

bool dirtyCacheEntry(Entry entry) {
    return dirtyDependency(entry);
}

uint mergeVisibilityAdmissionLimit() {
    return params.sampling.y;
}

void splitLodBlend(float3 position, out uint lowLod, out uint highLod,
                   out float blend) {
    // Affine view composition preserves an orthographic projection's zero
    // spatial w row; perspective projections retain a nonzero spatial row.
    float3 perspectiveProjection = float3(params.clipFromWorld[3][0],
        params.clipFromWorld[3][1], params.clipFromWorld[3][2]);
    bool perspective = all(isfinite(perspectiveProjection))
        && dot(perspectiveProjection, perspectiveProjection) > 1e-12;
    float3 verticalProjection = float3(params.clipFromWorld[1][0],
        params.clipFromWorld[1][1], params.clipFromWorld[1][2]);
    float focalPixels = 0.5 * float(params.viewport.w) * length(verticalProjection);
    uint targetPixels = max(params.ownerSequence.x, 1u);
    float lodScale;
    if (perspective) {
        float3 offset = abs(position - params.cameraPosition.xyz);
        float radius = max(offset.x, max(offset.y, offset.z));
        float baseDistance = max(focalPixels * params.tuning.x / float(targetPixels),
                                 params.tuning.x);
        lodScale = radius / baseDistance;
    } else {
        // Orthographic world spacing has the same projected size at any depth.
        lodScale = float(targetPixels) / max(focalPixels * params.tuning.x, 1e-20);
    }
    lowLod = uint(clamp(floor(log2(max(lodScale, 1.0))), 0.0, float(MAX_PROBE_LOD)));
    highLod = lowLod;
    blend = 0.0;
    if (lowLod < MAX_PROBE_LOD) {
        float nextStart = exp2(float(lowLod + 1u));
        float overlapStart = 0.9 * nextStart;
        if (lodScale >= overlapStart && lodScale < nextStart) {
            highLod = lowLod + 1u;
            blend = saturate((lodScale - overlapStart) / (nextStart - overlapStart));
        }
    }
}

uint transportCacheBucket(int4 key, uint bucketCount) {
    return hashKey(key, bucketCount);
}

float splitIntervalEnd(uint level, uint lod) {
    return params.tuning.y * exp2(float(2u * min(level, CASCADES - 1u)
        + min(lod, MAX_PROBE_LOD)));
}

float splitIntervalStart(uint level, uint lod) {
    return level == 0u ? 0.0 : splitIntervalEnd(level - 1u, lod);
}

bool splitIntervalContainsHit(float distance, uint level, uint lod) {
    return distance > splitIntervalStart(level, lod)
        && distance <= splitIntervalEnd(level, lod);
}

uint probeDirectoryCapacity(Parameters p) {
    // Each level can contain at most two LOD descriptors per visible pixel;
    // 4N slots keep its worst-case occupancy at or below one half.
    uint value = max(1u, p.size.z * 4u) - 1u;
    value |= value >> 1u;
    value |= value >> 2u;
    value |= value >> 4u;
    value |= value >> 8u;
    value |= value >> 16u;
    return value + 1u;
}

uint probeDirectoryBase(uint level, Parameters p) {
    // Existing cache heads, counters and one-lobe diagnostics occupy 2B+36
    // words. Four independent directories avoid mixing all 8N nodes in B.
    return 2u * p.size.w + 36u + level * probeDirectoryCapacity(p);
}

uint probeDirectoryHead(uint level, uint slot, Parameters p) {
    return probeDirectoryBase(level, p) + slot;
}

uint probeCountHead(uint level, uint slot, Parameters p) {
    uint directoryCapacity = probeDirectoryCapacity(p);
    // The four ray counts follow the four probe-index directories.
    return probeDirectoryBase(0u, p) + 4u * directoryCapacity
        + level * directoryCapacity + slot;
}

uint probeDirectoryBucket(int4 key, Parameters p) {
    key.w = int(uint(key.w) & (PROBE_LOD | (3u << 12u)));
    return hashKey(key, probeDirectoryCapacity(p));
}

uint probeDescriptorIndex(uint sample, uint variant, uint level) {
    return (variant * params.size.z + sample) * CASCADES + level;
}

uint findProbeNode(int4 key) {
    key.w = int(uint(key.w) & (PROBE_LOD | (3u << 12u)));
    uint level = (uint(key.w) >> 12u) & 3u;
    uint directoryCapacity = probeDirectoryCapacity(params);
    uint slot = probeDirectoryBucket(key, params);
    [loop] for (uint attempt = 0u; attempt < directoryCapacity; ++attempt) {
        uint index = heads[probeDirectoryHead(level, slot, params)].load();
        if (index == EMPTY) return EMPTY;
        if (index != TOMBSTONE) {
            uint descriptorCount = params.size.z * CASCADES * 2u;
            if (index >= descriptorCount) return EMPTY;
            if (all(entries[index].key == key)) return index;
        }
        slot = (slot + 1u) & (directoryCapacity - 1u);
    }
    return EMPTY;
}

uint findProbeNodeSlot(int4 key) {
    key.w = int(uint(key.w) & (PROBE_LOD | (3u << 12u)));
    uint level = (uint(key.w) >> 12u) & 3u;
    uint directoryCapacity = probeDirectoryCapacity(params);
    uint descriptorCount = params.size.z * CASCADES * 2u;
    uint slot = probeDirectoryBucket(key, params);
    [loop] for (uint attempt = 0u; attempt < directoryCapacity; ++attempt) {
        uint index = heads[probeDirectoryHead(level, slot, params)].load();
        if (index == EMPTY) return EMPTY;
        if (index != TOMBSTONE && index < descriptorCount
            && all(entries[index].key == key)) return slot;
        slot = (slot + 1u) & (directoryCapacity - 1u);
    }
    return EMPTY;
}

uint insertProbeDescriptor(int4 key, uint descriptorIndex) {
    key.w = int(uint(key.w) & (PROBE_LOD | (3u << 12u)));
    uint level = (uint(key.w) >> 12u) & 3u;
    uint directoryCapacity = probeDirectoryCapacity(params);
    uint slot = probeDirectoryBucket(key, params);
    [loop] for (uint attempt = 0u; attempt < directoryCapacity; ++attempt) {
        uint head = probeDirectoryHead(level, slot, params);
        // All descriptor keys are published by the preceding seed dispatch;
        // the CAS publishes only an immutable descriptor index.
        uint candidate = heads[head].compareExchange(EMPTY, descriptorIndex);
        if (candidate == EMPTY) {
            // WGSL's compare-exchange may fail spuriously while the word is
            // still EMPTY. A successful publication is visible as our unique
            // descriptor index; otherwise retry this same slot.
            uint observed = heads[head].load();
            if (observed == descriptorIndex) return descriptorIndex;
            if (observed == EMPTY) continue;
            candidate = observed;
        }
        if (candidate != TOMBSTONE) {
            uint descriptorCount = params.size.z * CASCADES * 2u;
            if (candidate >= descriptorCount) return EMPTY;
            if (all(entries[candidate].key == key)) return candidate;
        }
        slot = (slot + 1u) & (directoryCapacity - 1u);
    }
    return EMPTY;
}

uint findEntry(int4 key, bool previous = false) {
    if (previous && params.history.z <= 0.0) return EMPTY;
    uint bucket = transportCacheBucket(key, params.size.w);
    uint index = previous ? previousHeads[bucket] : heads[bucket].load();
    uint count, stride;
    if (previous) previousEntries.GetDimensions(count, stride);
    else entries.GetDimensions(count, stride);
    uint limit = previous ? min(cacheHistoryQuota(params), count) : count;
    [loop] for (uint visited = 0u; index != EMPTY && visited < limit; ++visited) {
        if (index >= count) return EMPTY;
        Entry entry = previous ? previousEntries[index] : entries[index];
        if (entry.next == index) return EMPTY;
        if (all(entry.key == key)) return index;
        index = entry.next;
    }
    return EMPTY;
}

uint findIrradianceProbe(int4 key, bool previous) {
    if (previous && params.history.z <= 0.0) return EMPTY;
    uint directoryCapacity = probeDirectoryCapacity(params);
    uint slot = probeDirectoryBucket(key, params);
    uint head = probeDirectoryHead(0u, slot, params);
    uint index = previous ? previousHeads[head] : heads[head].load();
    uint count, stride;
    if (previous) previousEntries.GetDimensions(count, stride);
    else entries.GetDimensions(count, stride);
    [loop] for (uint attempt = 0u; attempt < directoryCapacity; ++attempt) {
        if (index == EMPTY) return EMPTY;
        if (index != TOMBSTONE) {
            if (index >= count) return EMPTY;
            Entry entry = previous ? previousEntries[index] : entries[index];
            if (all(entry.key.xyz == key.xyz)
                && (uint(entry.key.w) & PROBE_LOD) == (uint(key.w) & PROBE_LOD)) {
                if (previous && dirtyCacheEntry(entry)) return EMPTY;
                if (previous) {
                    uint page = ((params.frame.x + 1u) & 1u) * directoryCapacity + slot;
                    DependencyContext packedDependency = probeDependency(
                        directionalProbes[page].dependencyMin,
                        directionalProbes[page].dependencyMax);
                    // Packed probes have only their aggregate geometry bounds;
                    // unknown hit bounds conservatively reject local-light reuse.
                    if (!radianceHistoryEligible(packedDependency,
                            unknownLightHitBounds())) return EMPTY;
                }
                return slot;
            }
        }
        slot = (slot + 1u) & (directoryCapacity - 1u);
        head = probeDirectoryHead(0u, slot, params);
        index = previous ? previousHeads[head] : heads[head].load();
    }
    return EMPTY;
}

float4 interpolateMergedChildren(int3 base, float3 fraction, uint direction,
                                 uint level, uint lod, bool previous,
                                 inout DependencyContext dependency,
                                 inout float4 lightHitBounds) {
    float3 sum = 0.0;
    float total = 0.0;
    float potential = 0.0;
    [loop] for (uint corner = 0u; corner < 8u; ++corner) {
        uint3 bit = uint3(corner & 1u, (corner >> 1u) & 1u, corner >> 2u);
        float3 axisWeight = lerp(1.0 - fraction, fraction, float3(bit));
        float weight = axisWeight.x * axisWeight.y * axisWeight.z;
        if (weight <= 1e-6) continue;
        potential += weight;
        int4 key = int4(base + int3(bit), int(direction | (level << 12u) | (lod << 18u)));
        uint index = findEntry(key, previous);
        if (index == EMPTY) continue;
        Entry entry = previous ? previousEntries[index] : entries[index];
        if (previous && (!intervalHistoryAlive(entry.lastSample, params)
            || !radianceHistoryEligible(entryMergedDependency(entry), entry.boundaryMerged))) continue;
        if (entry.merged.w <= 0.0 || !all(isfinite(entry.merged))) continue;
        // RGB is the conditional mean over the supported fraction, not a
        // premultiplied value. Carry that fraction through spatial filtering.
        float supportedWeight = weight * saturate(entry.merged.w);
        sum += entry.merged.xyz * supportedWeight;
        total += supportedWeight;
        dependencyUnion(dependency, entryMergedDependency(entry));
        lightHitBounds = unionConservativeSpheres(lightHitBounds, entry.boundaryMerged);
    }
    if (total <= 1e-6) return 0.0;
    return float4(sum / total, saturate(total / max(potential, 1e-8)));
}
[shader("compute")]
[numthreads(64, 1, 1)]
void clear(uint3 thread : SV_DispatchThreadID) {
    if (thread.x < params.size.w) {
        heads[thread.x].store(EMPTY);
        heads[params.size.w + 1u + thread.x].store(EMPTY);
    }
    uint directoryCapacity = probeDirectoryCapacity(params);
    uint directorySlots = 4u * directoryCapacity;
    if (thread.x < directorySlots) {
        heads[probeDirectoryBase(0u, params) + thread.x].store(EMPTY);
        uint level = thread.x / directoryCapacity;
        uint slot = thread.x % directoryCapacity;
        heads[probeCountHead(level, slot, params)].store(0u);
    }
    if (thread.x == 0) {
        heads[params.size.w].store(0);
        // The irradiance heads include index 2*size.w; counters start after it.
        heads[2 * params.size.w + 1].store(0); // dropped retained intervals
        heads[2 * params.size.w + 2].store(0); // rejected irradiance insertions
        heads[2 * params.size.w + 3].store(0); // entries colliding in prior buckets
        heads[2 * params.size.w + 4].store(0); // persistent cache overflow
        if (params.ownerSequence.z != 0u) {
            heads[2 * params.size.w + 5].store(0); // traversal-query attempts
            heads[2 * params.size.w + 6].store(0); // traversal queries admitted
            heads[2 * params.size.w + 7].store(0); // selected receiver updates
            heads[2 * params.size.w + 8].store(params.size.z); // visible pixel candidates
            heads[2 * params.size.w + 9].store(0); // occupied probe slots
            // One diffuse lobe: 8 confidence + 9 raw + 9 temporal bins.
            [loop] for (uint word = 0u; word < 26u; ++word)
                heads[2 * params.size.w + 10u + word].store(0u);
        }
    }
    if (thread.x == 0u && params.ownerSequence.z != 0u)
        rootRayOffset[0].store(0u);
    // Both head clears run before transport. The second one preserves shared
    // query/owner counters; only the first needs to clear sparse output images,
    // which otherwise incurs a redundant full-image write on every frame.
    if (params.ownerSequence.z != 0u) {
        uint clearThreads = max(params.size.w, directorySlots);
        uint workgroupThreads = max(((clearThreads + 63u) / 64u) * 64u, 64u);
        uint resolveCount = giResolveExtent(params).x * giResolveExtent(params).y;
        [loop] for (uint pixel = thread.x; pixel < resolveCount; pixel += workgroupThreads) {
            uint2 coordinate = uint2(pixel % giResolveExtent(params).x,
                                     pixel / giResolveExtent(params).x);
            irradiance.Store(int2(coordinate), 0.0);
        }
    }
    // Entries unreachable from a hash head are never read. Avoid clearing the
    // large interval allocation; trace and retain initialize reachable nodes.
}

uint permuteCandidateDomain(uint value, uint mask, uint seed) {
    // Each xor-shift and odd multiply is invertible modulo 2^k. Mask after
    // every operation so this remains a permutation of the selected power-of-2
    // domain rather than only of all 32-bit values.
    value = (value ^ seed) & mask;
    value = (value ^ (value >> 16)) & mask;
    value = (value * 0x7feb352du) & mask;
    value = (value ^ (value >> 15)) & mask;
    value = (value * 0x846ca68bu) & mask;
    value = (value ^ (value >> 16)) & mask;
    return value;
}

uint inverseXorShiftRight(uint value, uint shift) {
    uint result = value;
    [loop] for (uint offset = shift; offset < 32u; offset += shift)
        result ^= value >> offset;
    return result;
}

uint inversePermuteCandidateDomain(uint value, uint mask, uint seed) {
    value = inverseXorShiftRight(value, 16u) & mask;
    value = (value * 0x43021123u) & mask;
    value = inverseXorShiftRight(value, 15u) & mask;
    value = (value * 0x1d69e2a5u) & mask;
    value = inverseXorShiftRight(value, 16u) & mask;
    return (value ^ seed) & mask;
}

uint transportCandidateRank(uint sample) {
    uint candidateCount = params.size.z;
    if (candidateCount <= 1u) return 0u;
    // Cycle walking restricts this reversible permutation to an arbitrary
    // candidate count while preserving exact no-repeat epochs. The power-of-2
    // domain is at most twice the candidate grid, so rejection is usually rare.
    uint mask = 0u;
    while (mask < candidateCount - 1u) mask = (mask << 1u) | 1u;
    uint seed = (params.size.x * 0x9e3779b9u)
        ^ (params.size.y * 0x85ebca6bu) ^ candidateCount;
    seed &= mask;
    uint rank = permuteCandidateDomain(sample, mask, seed);
    [loop] while (rank >= candidateCount) {
        rank = permuteCandidateDomain(rank, mask, seed);
    }
    return rank;
}

bool transportCandidateSelected(uint sample) {
    uint candidateCount = params.size.z;
    // An explicit per-frame query cap can be too small to complete one
    // receiver's seed + direct-light estimator. In that case the host records
    // a disabled-update bit so budget zero is not mistaken for legacy dense
    // sampling.
    if ((params.ownerSequence.w & 0x40000000u) != 0u) return false;
    uint budget = params.sampling.x;
    if (candidateCount == 0 || budget == 0 || budget >= candidateCount) return true;

    // Fair work is a circular window in the stable rank permutation. The host
    // advances this cursor by the actual fair quota only after a submitted frame.
    uint fairStart = (params.ownerSequence.w & 0x03ffffffu) % candidateCount;
    uint fairQuota = params.ownerSequence.z;
    uint rank = transportCandidateRank(sample);
    bool fairSelected = fairQuota > 0u
        && (rank >= fairStart
            ? rank - fairStart < fairQuota
            : candidateCount - fairStart + rank < fairQuota);

    return fairSelected;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void seed(uint3 thread : SV_DispatchThreadID) {
    if (any(thread.xy >= params.size.xy)) return;
    uint sample = thread.x + thread.y * params.size.x;
    [unroll] for (uint variant = 0u; variant < 2u; ++variant)
        [unroll] for (uint level = 0u; level < CASCADES; ++level) {
            uint descriptor = probeDescriptorIndex(sample, variant, level);
            entries[descriptor] = (Entry)0;
        }

    uint2 full = fullPixel(thread.xy, params);
    float depth = depthTexture.Load(int3(full, 0));
    if (!(depth > 0.0) || !isfinite(depth)) return;
    uint4 deferred = gbuffer.Load(int3(full, 0));
    uint4 raster = rasterReceiver.Load(int3(full, 0));
    float3 shadingNormal = deferredShadingNormal(deferred);
    if (!(dot(shadingNormal, shadingNormal) > 0.01)
        || !all(isfinite(shadingNormal))) return;
    float3 geometricNormal = receiverGeometricNormal(raster.z, shadingNormal);
    float3 surfacePosition = worldPosition(float2(full) + 0.5, depth, params);
    float3 cameraNear = worldPosition(float2(full) + 0.5, 1.0, params);
    if (dot(geometricNormal, cameraNear - surfacePosition) < 0.0)
        geometricNormal = -geometricNormal;

    uint lowLod, highLod;
    float lodBlend;
    splitLodBlend(surfacePosition, lowLod, highLod, lodBlend);
    uint selectedLod = 0u;
    if (highLod != lowLod) {
        uint phaseBits = transportHash(sample ^ (params.frame.x * 0x9e3779b9u));
        float phase = float(phaseBits >> 8u) * (1.0 / 16777216.0);
        selectedLod = phase < lodBlend ? 1u : 0u;
    }
    bool selectedRay = transportCandidateSelected(sample);
    [unroll] for (uint variant = 0u; variant < 2u; ++variant) {
        if (variant == 1u && highLod == lowLod) continue;
        uint lod = variant == 0u ? lowLod : highLod;
        [unroll] for (uint level = 0u; level < CASCADES; ++level) {
            uint index = probeDescriptorIndex(sample, variant, level);
            int4 key = cacheKey(surfacePosition, 0u, level, params, lod);
            Entry entry = (Entry)0;
            entry.key = key;
            entry.origin = float4((float3(key.xyz) + 0.5)
                * spacing(params, level, lod), 1.0);
            entry.valid = 1u;
            entry.next = EMPTY;
            // Each screen update traces once. In an LOD overlap, temporal
            // low-discrepancy assignment balances it between the two fields.
            entry.padding.x = (level == 0u && selectedRay && variant == selectedLod)
                ? 1u : 0u;
            entry.padding.y = 1u;
            entries[index] = entry;
        }
    }
    if (selectedRay) heads[2u * params.size.w + 7u].add(1u);
}

// Deduplicate immutable per-pixel descriptors after seed's dispatch barrier.
// Insertion publishes only a descriptor index; keys are never read while a
// writer is still constructing the record. Same-cell pixels take one bounded
// probe-directory lookup instead of forming a screen-sized linked list.
[shader("compute")]
[numthreads(64, 1, 1)]
void rank(uint3 thread : SV_DispatchThreadID) {
    if (thread.x >= params.size.z) return;
    uint sample = thread.x;
    [unroll] for (uint variant = 0u; variant < 2u; ++variant)
        [unroll] for (uint level = 0u; level < CASCADES; ++level) {
            uint index = probeDescriptorIndex(sample, variant, level);
            Entry descriptor = entries[index];
            if (descriptor.valid == 0u) continue;
            uint canonical = insertProbeDescriptor(descriptor.key, index);
            if (canonical == EMPTY) {
                entries[index].valid = 0u;
                continue;
            }
            entries[index].next = canonical;
            if (level == 0u && descriptor.padding.x != 0u) {
                uint slot = findProbeNodeSlot(descriptor.key);
                if (slot == EMPTY) continue;
                uint localOrdinal = heads[probeCountHead(0u, slot, params)].add(1u);
                entries[index].lastSample = localOrdinal;
            }
        }
}

// Every visible point seeds its nearest probe at each level and active LOD;
// with doubled grid spacing this equals Algorithm 1's nearest-parent chain.
// Counts flow from each unique child to its already initialized same-LOD parent.
[shader("compute")]
[numthreads(64, 1, 1)]
void parents(uint3 thread : SV_DispatchThreadID) {
    uint parentLevel = params.frame.y;
    if (parentLevel == 0u || parentLevel >= CASCADES) return;
    uint parentSlot = thread.x;
    if (parentSlot >= probeDirectoryCapacity(params)) return;
    uint parentIndex = heads[probeDirectoryHead(parentLevel, parentSlot, params)].load();
    if (parentIndex == EMPTY || parentIndex == TOMBSTONE) return;
    Entry parent = entries[parentIndex];
    if (((uint(parent.key.w) >> 12u) & 3u) != parentLevel) return;

    uint childLevel = parentLevel - 1u;
    uint lodBits = uint(parent.key.w) & PROBE_LOD;
    uint rayCount = 0u;
    [unroll] for (uint child = 0u; child < 8u; ++child) {
        int3 bit = int3(child & 1u, (child >> 1u) & 1u, (child >> 2u) & 1u);
        int4 childKey = int4(parent.key.xyz * 2 + bit,
            int((childLevel << 12u) | lodBits));
        uint childSlot = findProbeNodeSlot(childKey);
        if (childSlot != EMPTY)
            rayCount += heads[probeCountHead(childLevel, childSlot, params)].load();
    }
    heads[probeCountHead(parentLevel, parentSlot, params)].store(rayCount);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void prefix(uint3 thread : SV_DispatchThreadID) {
    if (thread.x >= probeDirectoryCapacity(params)) return;
    uint level = params.frame.y;
    uint index = heads[probeDirectoryHead(level, thread.x, params)].load();
    if (index == EMPTY || index == TOMBSTONE) return;
    Entry probe = entries[index];
    int4 key = probe.key;
    level = (uint(key.w) >> 12u) & 3u;
    if (level != params.frame.y) return;
    uint rayCount = heads[probeCountHead(level, thread.x, params)].load();

    uint offset = 0u;
    if (level + 1u == CASCADES) {
        // Root iteration order is unspecified; atomic ranges are disjoint and
        // form one contiguous prefix over the actual selected rays.
        offset = rootRayOffset[0].add(rayCount);
    } else {
        int3 parent = key.xyz >> 1;
        uint lodBits = uint(key.w) & PROBE_LOD;
        uint parentIndex = findProbeNode(int4(parent,
            int(((level + 1u) << 12u) | lodBits)));
        if (parentIndex != EMPTY) offset = entries[parentIndex].valid;
        int3 local = key.xyz - parent * 2;
        uint ordinal = uint(local.x + local.y * 2 + local.z * 4);
        [loop] for (uint sibling = 0u; sibling < ordinal; ++sibling) {
            int3 bit = int3(sibling & 1u, (sibling >> 1u) & 1u,
                            (sibling >> 2u) & 1u);
            int4 childKey = int4(parent * 2 + bit,
                int((level << 12u) | lodBits));
            uint childSlot = findProbeNodeSlot(childKey);
            if (childSlot != EMPTY)
                offset += heads[probeCountHead(level, childSlot, params)].load();
        }
    }
    entries[index].valid = offset;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void generateDirections(uint3 thread : SV_DispatchThreadID) {
    if (any(thread.xy >= params.size.xy)) return;
    uint sample = thread.x + thread.y * params.size.x;
    [unroll] for (uint variant = 0u; variant < 2u; ++variant) {
        uint index = probeDescriptorIndex(sample, variant, 0u);
        if (entries[index].padding.x == 0u || entries[index].next == EMPTY) continue;
        uint canonical = entries[index].next;
        entries[index].lastSample += entries[canonical].valid;
    }
}

uint currentCapacity() {
    // Two LOD variants × four levels × one visible-pixel descriptor.
    return params.size.z * CASCADES * 2u;
}


[shader("compute")]
[numthreads(64, 1, 1)]
void retain(uint3 thread : SV_DispatchThreadID) {
    if (thread.x >= params.size.w || params.history.z == 0
        || (params.history.x == 0 && params.intervalHistory.x == 0)
        || params.history.w == 0) return;
    uint capacity = currentCapacity();
    uint bucketQuota = cacheHistoryQuota(params);
    uint bucketBase = thread.x * bucketQuota;
    uint bucketRetained = 0;
    uint bucketLoad = 0;
    uint previousCount, previousStride;
    previousEntries.GetDimensions(previousCount, previousStride);
    [loop] for (uint index = previousHeads[thread.x]; index != EMPTY && bucketLoad < bucketQuota; index = previousEntries[index].next) {
        if (index >= previousCount) break;
        Entry entry = previousEntries[index];
        if (entry.next == index) break;
        bucketLoad++;
        if (!intervalHistoryAlive(entry.lastSample, params)) continue;
        if (!betaHistoryEligible(entryDependency(entry), entry.normal)) continue;
        if (persistentWorldIntervals(params)
            && !radianceHistoryEligible(entryMergedDependency(entry), entry.boundaryMerged)) {
            // A changed dependency invalidates composed I, not independent clean J/beta.
            entry.merged = 0.0;
            storeEntryMergedDependency(entry, dependencyUnknown());
            entry.boundaryMerged = unknownLightHitBounds();
        }
        heads[params.size.w].add(1);
        if (bucketRetained >= bucketQuota) {
            heads[2 * params.size.w + 1].add(1);
            continue;
        }
        uint slot = capacity + bucketBase + bucketRetained++;
        entry.samples = 0;
        entry.valid = params.frame.x | 0x80000000u;
        entry.next = heads[thread.x].exchange(slot);
        entries[slot] = entry;
    }
    if (bucketLoad > 1) heads[2 * params.size.w + 3].add(bucketLoad - 1);
}

void deposit(uint first, float3 samplePosition, float3 rayOrigin,
             float3 direction, Hit hit, float3 radiance, uint lod,
             DependencyContext rayDependency, float dependencyStart,
             float dependencyDistance, float dependencyPadding,
             DependencyContext outgoingDependency) {
    if ((hit.errors & HIT_QUERY_UNSUPPORTED) != 0u) return;
    [loop] for (uint level = 0u; level < CASCADES; ++level) {
        uint index = first + level;
        float end = splitIntervalEnd(level, lod);
        float intervalStart = splitIntervalStart(level, lod);
        // The last cascade owns the unbounded tail, including supported-zero
        // escapes. Its nominal cutoff only determines the preceding interval.
        bool terminal = level + 1u == CASCADES;
        bool terminates = terminal || splitIntervalContainsHit(hit.distance, level, lod);
        Entry entry = (Entry)0;
        entry.key = cacheKey(samplePosition, directionIndex(direction, level),
                             level, params, lod);
        entry.origin = float4((float3(entry.key.xyz) + 0.5)
            * spacing(params, level, lod), 1.0);
        entry.interval = terminates ? float4(radiance, 0.0) : float4(0.0, 0.0, 0.0, 1.0);
        DependencyContext intervalDependency = rayDependency.known
            ? dependencyEmpty() : dependencyUnknown();
        float intervalEnd = terminal ? dependencyDistance
            : min(dependencyDistance, max(end, 0.0));
        float intervalDependencyStart = max(dependencyStart, intervalStart);
        entry.normal = unknownLightHitBounds();
        if (rayDependency.known && rayDependency.hasBounds
            && intervalEnd >= intervalDependencyStart) {
            float3 segmentStart = rayOrigin + direction * intervalDependencyStart;
            float3 segmentEnd = rayOrigin + direction * intervalEnd;
            dependencyIncludeSegment(intervalDependency,
                segmentStart, segmentEnd, dependencyPadding);
            entry.normal = dependencySegmentSphere(segmentStart, segmentEnd,
                dependencyPadding);
        }
        if (terminates) dependencyUnion(intervalDependency, outgoingDependency);
        storeEntryDependency(entry, intervalDependency);
        storeEntryMergedDependency(entry, intervalDependency);
        float4 lightHitBounds = terminates && hit.triangle != EMPTY
            ? lightHitBoundsAt(rayOrigin + direction * hit.distance)
            : emptyLightHitBounds();
        entry.boundaryInterval = lightHitBounds;
        entry.boundaryMerged = lightHitBounds;
        entry.valid = params.frame.x | 0x80000000u;
        entry.lastSample = params.frame.x;
        entry.samples = 1u;
        entry.next = heads[transportCacheBucket(entry.key, params.size.w)].exchange(index);
        entries[index] = entry;
        // Paper §5 leaves all farther intervals unfilled after the first hit.
        if (terminates) break;
    }
}

void traceTransport(uint3 thread) {
    uint2 pixel = thread.xy;
    if (any(pixel >= params.size.xy)) return;
    uint sample = pixel.x + pixel.y * params.size.x;
    uint2 full = fullPixel(pixel, params);
    float depth = depthTexture.Load(int3(full, 0));
    if (!(depth > 0.0) || !isfinite(depth)) return;
    uint4 deferred = gbuffer.Load(int3(full, 0));
    uint4 raster = rasterReceiver.Load(int3(full, 0));
    float3 shadingNormal = deferredShadingNormal(deferred);
    if (!(dot(shadingNormal, shadingNormal) > 0.01)
        || !all(isfinite(shadingNormal))) return;
    float3 geometricNormal = receiverGeometricNormal(raster.z, shadingNormal);
    float3 surfacePosition = worldPosition(float2(full) + 0.5, depth, params);
    float3 cameraNear = worldPosition(float2(full) + 0.5, 1.0, params);
    if (dot(geometricNormal, cameraNear - surfacePosition) < 0.0)
        geometricNormal = -geometricNormal;

    float2 jitter = r2(transportHash(params.frame.x ^ 0x5a17c9e3u));
    [unroll] for (uint variant = 0u; variant < 2u; ++variant) {
        uint first = probeDescriptorIndex(sample, variant, 0u);
        Entry descriptor = entries[first];
        if (descriptor.padding.x == 0u || descriptor.next == EMPTY) continue;
        uint lod = (uint(descriptor.key.w) & PROBE_LOD) >> 18u;
        uint sequence = descriptor.lastSample;
        float2 uv = frac(0.5 + r2(sequence + 1u) + jitter);
        float3 direction = decodeDirection(uv);
        if (dot(direction, geometricNormal) < 0.0) direction = -direction;
        float3 rayOrigin = rayOffset(surfacePosition, geometricNormal, params.tuning.z);
        DependencyContext rayDependency = dependencyUnknown();
        DependencyContext outgoingDependency = dependencyEmpty();
        float dependencyStart = 0.0, dependencyDistance = 0.0;
        float dependencyPadding = 0.0;
        float coneWidth = params.tuning.x * 0.5;
        Hit hit = traceRayTracked(rayOrigin, direction, 1e20, rayDependency,
            dependencyStart, dependencyDistance, dependencyPadding,
            false, coneWidth, 0.5);
        bool radianceSupported;
        float3 radiance = outgoingRadianceTracked(rayOrigin, direction, hit,
            radianceSupported, outgoingDependency, mergeVisibilityAdmissionLimit());
        if (!radianceSupported) continue;
        deposit(first, surfacePosition, rayOrigin, direction, hit, radiance, lod,
            rayDependency, dependencyStart, dependencyDistance, dependencyPadding,
            outgoingDependency);
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void trace(uint3 thread : SV_DispatchThreadID) {
    traceTransport(thread);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void reduce(uint3 thread : SV_DispatchThreadID) {
    uint bucket = thread.x;
    if (bucket >= params.size.w) return;
    uint source = gpuSortLinkedList(heads[bucket].load());
    uint unique = EMPTY;
    [loop] while (source != EMPTY) {
        Entry firstEntry = entries[source];
        int4 key = firstEntry.key;
        uint cursor = source;
        uint freshCount = 0u;
        uint freshSource = EMPTY;
        float4 intervalSum = 0.0;
        DependencyContext freshDependency = dependencyEmpty();
        float4 freshLightHitBounds = emptyLightHitBounds();
        float4 freshBetaSegmentSphere = emptyLightHitBounds();
        Entry retained = (Entry)0;
        bool hasRetained = false;
        [loop] while (cursor != EMPTY && all(entries[cursor].key == key)) {
            Entry sample = entries[cursor];
            uint next = sample.next;
            if (sample.samples > 0u) {
                if (freshSource == EMPTY) freshSource = cursor;
                intervalSum += sample.interval * float(sample.samples);
                freshCount += sample.samples;
                dependencyUnion(freshDependency, entryDependency(sample));
                freshLightHitBounds = unionConservativeSpheres(
                    freshLightHitBounds, sample.boundaryInterval);
                freshBetaSegmentSphere = unionConservativeSpheres(
                    freshBetaSegmentSphere, sample.normal);
            } else if (!hasRetained && sample.valid == (params.frame.x | 0x80000000u)) {
                retained = sample;
                hasRetained = true;
            }
            if (cursor != source) entries[cursor].valid = 0u;
            cursor = next;
        }
        uint nextGroup = cursor;
        if (freshCount > 0u) {
            Entry aggregate = firstEntry;
            aggregate.interval = intervalSum / float(freshCount);
            aggregate.key = key;
            uint level = (uint(key.w) >> 12u) & 3u;
            uint lod = (uint(key.w) & PROBE_LOD) >> 18u;
            aggregate.origin = float4((float3(key.xyz) + 0.5)
                * spacing(params, level, lod), 1.0);
            aggregate.samples = freshCount;
            aggregate.valid = params.frame.x | 0x80000000u;
            aggregate.lastSample = params.frame.x;
            aggregate.merged = 0.0;
            aggregate.next = unique;
            storeEntryDependency(aggregate, freshDependency);
            storeEntryMergedDependency(aggregate, dependencyUnknown());
            aggregate.boundaryInterval = freshLightHitBounds;
            aggregate.boundaryMerged = unknownLightHitBounds();
            aggregate.normal = freshBetaSegmentSphere;
            entries[source] = aggregate;
            unique = source;
        } else if (hasRetained) {
            retained.next = unique;
            retained.valid = params.frame.x | 0x80000000u;
            retained.samples = 0u;
            entries[source] = retained;
            unique = source;
        } else {
            entries[source].valid = 0u;
        }
        source = nextGroup;
    }
    heads[bucket].store(unique);
}

void mergeEntry(uint index) {
    uint level = params.frame.y;
    int4 key = entries[index].key;
    if (((uint(key.w) >> 12u) & 3u) != level) return;

    float4 interval = entries[index].interval;
    float4 rawLightHitBounds = entries[index].boundaryInterval;
    float4 betaSegmentSphere = entries[index].normal;
    float samples = float(entries[index].samples);
    // A batch of n observations retains blend^n of old mass and contributes
    // 1 + blend + ... + blend^(n-1) new mass. Thus history is bounded by
    // 1/(1-blend), and batches have the same response as repeated observations.
    float radianceBlend = min(saturate(params.history.x), 0.999);
    float transmittanceBlend = min(saturate(params.intervalHistory.x), 0.999);
    float radianceMassLimit = 1.0 / (1.0 - radianceBlend);
    float transmittanceMassLimit = 1.0 / (1.0 - transmittanceBlend);
    float radianceRetention = samples > 0.0 ? pow(radianceBlend, samples) : 1.0;
    float transmittanceRetention = samples > 0.0 ? pow(transmittanceBlend, samples) : 1.0;
    float radianceSamples = samples > 0.0
        ? (1.0 - radianceRetention) * radianceMassLimit : 0.0;
    float transmittanceSamples = samples > 0.0
        ? (1.0 - transmittanceRetention) * transmittanceMassLimit : 0.0;
    DependencyContext rawDependency = entryDependency(entries[index]);
    uint previous = (params.history.x > 0.0 || params.intervalHistory.x > 0.0)
        && params.history.z > 0.0 && params.history.w > 0.0
        ? findEntry(key, true) : EMPTY;
    if (previous != EMPTY) {
        Entry old = previousEntries[previous];
        bool oldAlive = intervalHistoryAlive(old.lastSample, params);
        bool oldRadianceGeometryValid = oldAlive && !dirtyRawCacheEntry(old);
        bool oldBetaGeometryValid = oldAlive
            && betaHistoryEligible(entryDependency(old), old.normal);
        float oldRadianceSamples = asfloat(old.samples);
        float oldTransmittanceSamples = old.origin.w;
        if (!isfinite(oldRadianceSamples) || oldRadianceSamples < 0.0)
            oldRadianceSamples = 0.0;
        if (!isfinite(oldTransmittanceSamples) || oldTransmittanceSamples < 0.0)
            oldTransmittanceSamples = 0.0;
        bool reuseRadiance = oldRadianceGeometryValid && params.history.x > 0.0
            && radianceHistoryEligible(entryDependency(old), old.boundaryInterval);
        bool reuseTransmittance = oldBetaGeometryValid && params.intervalHistory.x > 0.0;
        if (samples > 0.0) {
            // Clamp incoming mass when observations resume, including after a
            // retention-setting change. Unobserved world state never decays.
            oldRadianceSamples = min(oldRadianceSamples, radianceMassLimit)
                * radianceRetention;
            oldTransmittanceSamples = min(oldTransmittanceSamples, transmittanceMassLimit)
                * transmittanceRetention;
            if (reuseRadiance && oldRadianceSamples > 0.0) {
                interval.xyz = (interval.xyz * radianceSamples
                    + old.interval.xyz * oldRadianceSamples)
                    / (radianceSamples + oldRadianceSamples);
                radianceSamples += oldRadianceSamples;
                dependencyUnion(rawDependency, entryDependency(old));
                // A temporally blended J depends on every receiver position
                // represented by both means, not only the newest observations.
                rawLightHitBounds = unionConservativeSpheres(
                    rawLightHitBounds, old.boundaryInterval);
            }
            if (reuseTransmittance && oldTransmittanceSamples > 0.0) {
                interval.w = (interval.w * transmittanceSamples
                    + old.interval.w * oldTransmittanceSamples)
                    / (transmittanceSamples + oldTransmittanceSamples);
                transmittanceSamples += oldTransmittanceSamples;
                // Keep the raw AABB as a second conservative beta proof too;
                // the segment sphere alone can be loose for long rays.
                dependencyUnion(rawDependency, entryDependency(old));
                betaSegmentSphere = unionConservativeSpheres(
                    betaSegmentSphere, old.normal);
            }
        } else {
            if (reuseRadiance && oldRadianceSamples > 0.0) {
                interval.xyz = old.interval.xyz;
                radianceSamples = oldRadianceSamples;
                rawLightHitBounds = old.boundaryInterval;
                dependencyUnion(rawDependency, entryDependency(old));
            } else {
                interval.xyz = 0.0;
                radianceSamples = 0.0;
            }
            if (reuseTransmittance && oldTransmittanceSamples > 0.0) {
                interval.w = old.interval.w;
                transmittanceSamples = oldTransmittanceSamples;
                dependencyUnion(rawDependency, entryDependency(old));
                betaSegmentSphere = old.normal;
            } else {
                interval.w = 1.0;
                transmittanceSamples = 0.0;
            }
        }
    } else if (samples == 0.0) {
        interval.xyz = 0.0;
        interval.w = 1.0;
        radianceSamples = 0.0;
        transmittanceSamples = 0.0;
    }

    entries[index].interval = interval;
    entries[index].boundaryInterval = rawLightHitBounds;
    entries[index].normal = betaSegmentSphere;
    storeEntryDependency(entries[index], rawDependency);
    entries[index].samples = asuint(radianceSamples);
    entries[index].origin.w = transmittanceSamples;

    // Carry one invariant throughout the hierarchy: RGB is a conditional mean
    // and alpha is its supported mass. Raw J is already premultiplied by the
    // opaque fraction; beta without J cannot invent a supported black surface.
    float transmittance = transmittanceSamples > 0.0 ? saturate(interval.w) : 0.0;
    float support = radianceSamples > 0.0 && transmittanceSamples > 0.0
        ? 1.0 - transmittance : 0.0;
    float3 radiance = support > 0.0 ? interval.xyz : 0.0;
    float4 mergedLightHitBounds = support > 0.0
        ? entries[index].boundaryInterval : emptyLightHitBounds();
    DependencyContext mergedDependency = rawDependency;
    if (transmittance > 0.0 && level + 1u < CASCADES) {
        float3 position = entries[index].origin.xyz;
        uint direction = uint(key.w) & 4095u;
        uint theta = BASE_THETA << level;
        uint2 xy = uint2(direction % (2u * theta), direction / (2u * theta));
        float3 sum = 0.0;
        float childSupport = 0.0;
        [unroll] for (uint child = 0u; child < 4u; ++child) {
            uint2 fine = xy * 2u + uint2(child & 1u, child >> 1u);
            uint fineDirection = fine.x + fine.y * (4u * theta);
            uint lod = (uint(key.w) & PROBE_LOD) >> 18u;
            float3 grid = position / spacing(params, level + 1u, lod) - 0.5;
            DependencyContext childDependency = dependencyEmpty();
            float4 childLightHitBounds = emptyLightHitBounds();
            float4 value = interpolateMergedChildren(int3(floor(grid)), frac(grid),
                fineDirection, level + 1u, lod, false, childDependency,
                childLightHitBounds);
            dependencyUnion(mergedDependency, childDependency);
            if (value.w > 0.0) {
                sum += value.xyz * value.w;
                childSupport += value.w;
                mergedLightHitBounds = unionConservativeSpheres(
                    mergedLightHitBounds, childLightHitBounds);
            }
        }
        if (childSupport > 0.0) {
            radiance += transmittance * sum * 0.25;
            support += transmittance * childSupport * 0.25;
        }
    }
    entries[index].merged = support > 0.0
        ? float4(radiance / support, saturate(support)) : 0.0;
    storeEntryMergedDependency(entries[index], mergedDependency);
    entries[index].boundaryMerged = mergedLightHitBounds;
}
[shader("compute")]
[numthreads(64, 1, 1)]
void merge(uint3 thread : SV_DispatchThreadID) {
    uint index = thread.x * CASCADES + thread.y;
    if (index >= currentCapacity() + cacheHistoryQuota(params) * params.size.w) return;
    Entry entry = entries[index];
    if (entry.valid != (params.frame.x | 0x80000000u)
        || !intervalHistoryAlive(entry.lastSample, params)) return;
    // One thread per canonical entry avoids serial ray traversals along a bucket.
    // Retained physical offsets need not agree with the key's cascade level.
    mergeEntry(index);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void seedProbes(uint3 thread : SV_DispatchThreadID) {
    if (thread.y >= CASCADES) return;
    uint index = thread.x * CASCADES + thread.y;
    if (index >= currentCapacity()) return;
    Entry entry = entries[index];
    // Include every initialized visible descriptor, including candidates with
    // zero prefix offsets; retained cache entries are read while packing radiance.
    bool initialized = entry.padding.y != 0u;
    bool traced = entry.valid == (params.frame.x | 0x80000000u);
    if ((!initialized && !traced)
        || ((uint(entry.key.w) >> 12u) & 3u) != 0u) return;
    int4 key = int4(entry.key.xyz, int(uint(entry.key.w) & PROBE_LOD));
    uint slot = probeDirectoryBucket(key, params);
    uint directoryCapacity = probeDirectoryCapacity(params);
    [loop] for (uint attempt = 0u; attempt < directoryCapacity; ++attempt) {
        uint head = probeDirectoryHead(0u, slot, params);
        uint candidate = heads[head].compareExchange(EMPTY, index);
        if (candidate == EMPTY) {
            uint observed = heads[head].load();
            if (observed == index) return;
            if (observed == EMPTY) continue;
            candidate = observed;
        }
        if (candidate != TOMBSTONE && candidate < currentCapacity()) {
            Entry other = entries[candidate];
            if (all(other.key.xyz == key.xyz)
                && (uint(other.key.w) & PROBE_LOD) == uint(key.w)) return;
        }
        slot = (slot + 1u) & (directoryCapacity - 1u);
    }
    heads[2u * params.size.w + 2u].add(1u);
}

groupshared float4 packProbeRadiance[32];
groupshared uint packProbeSource[32];

[shader("compute")]
[numthreads(64, 1, 1)]
void packProbes(uint3 groupId : SV_GroupID, uint lane : SV_GroupIndex) {
    uint ordinal = groupId.x + 256u * groupId.y;
    bool active = ordinal < currentCapacity() / CASCADES;
    uint index = EMPTY;
    uint slot = EMPTY;
    int4 key = 0;
    if (active) {
        index = ordinal * CASCADES;
        key = entries[index].key;
        slot = findProbeNodeSlot(key);
        active = slot != EMPTY
            && heads[probeDirectoryHead(0u, slot, params)].load() == index;
    }

    if (lane < 32u) {
        uint source = EMPTY;
        float4 value = 0.0;
        if (active) {
            uint lodBits = uint(key.w) & PROBE_LOD;
            int4 directionKey = int4(key.xyz, int(lane | lodBits));
            source = findEntry(directionKey, false);
            if (source != EMPTY) value = entries[source].merged;
        }
        packProbeSource[lane] = source;
        packProbeRadiance[lane] = value;
    }
    GroupMemoryBarrierWithGroupSync();
    if (!active) return;

    uint directoryCapacity = probeDirectoryCapacity(params);
    uint output = (params.frame.x & 1u) * directoryCapacity + slot;
    if (lane < 36u) {
        uint x = lane % 6u;
        uint y = lane / 6u;
        float3 normal = decodeNormal((float2(x, y) + 0.5) / 6.0);
        float3 sum = 0.0;
        float nominalCosine = 0.0;
        float observedCosine = 0.0;
        [loop] for (uint direction = 0u; direction < 32u; ++direction) {
            float cosine = max(dot(normal, binDirection(direction, 0u)), 0.0);
            nominalCosine += cosine;
            float4 sample = packProbeRadiance[direction];
            if (cosine <= 0.0 || !(sample.w > 0.0) || !all(isfinite(sample))) continue;
            float weight = cosine * saturate(sample.w);
            sum += sample.xyz * weight;
            observedCosine += weight;
        }
        float4 value = 0.0;
        if (observedCosine > 1e-6 && nominalCosine > 1e-6) {
            value = float4(sum / observedCosine
                * (nominalCosine * (4.0 / 32.0)),
                saturate(observedCosine / nominalCosine));
        }
        directionalProbes[output].irradiance[lane] = value;
    }
    if (lane == 0u) {
        DependencyContext packedDependency = dependencyEmpty();
        [loop] for (uint direction = 0u; direction < 32u; ++direction) {
            uint source = packProbeSource[direction];
            float4 sample = packProbeRadiance[direction];
            if (source != EMPTY && sample.w > 0.0 && all(isfinite(sample)))
                dependencyUnion(packedDependency, entryMergedDependency(entries[source]));
        }
        storeProbeDependency(directionalProbes[output], packedDependency);
        heads[2u * params.size.w + 9u].add(1u);
    }
}
[shader("compute")]
[numthreads(64, 1, 1)]
void commitCache(uint3 thread : SV_DispatchThreadID) {
    uint bucket = thread.x;
    if (bucket >= params.size.w) return;
    uint quota = min(cacheHistoryQuota(params), 4u);
    if (quota == 0u) return;
    uint base = bucket * quota;
    uint selected[4];
    [unroll] for (uint slot = 0; slot < 4; ++slot) selected[slot] = EMPTY;
    uint source = heads[bucket].load();
    [loop] while (source != EMPTY) {
        Entry entry = entries[source];
        uint next = entry.next;
        bool eligible = params.history.w > 0.0
            && entry.valid == (params.frame.x | 0x80000000u)
            && intervalHistoryAlive(entry.lastSample, params);
        entries[source].padding.x = EMPTY;
        if (eligible) {
            // Reduction sorts by key, not age. Retaining its first keys can
            // permanently starve newly discovered probes in a full bucket.
            uint candidate = source;
            [loop] for (uint slot = 0; slot < quota; ++slot) {
                uint incumbent = selected[slot];
                if (incumbent == EMPTY || params.frame.x - entries[candidate].lastSample
                    < params.frame.x - entries[incumbent].lastSample) {
                    selected[slot] = candidate;
                    candidate = incumbent;
                    if (candidate == EMPTY) break;
                }
            }
            if (candidate != EMPTY) heads[2 * params.size.w + 4].add(1);
        }
        source = next;
    }
    uint output = EMPTY;
    [loop] for (uint retained = 0; retained < quota; ++retained) {
        source = selected[retained];
        if (source == EMPTY) break;
        uint slot = base + retained;
        Entry entry = entries[source];
        entry.next = output;
        entry.padding.x = slot | (params.intervalHistory.y > 0.5 ? 0x80000000u : 0u);
        previousEntries[slot] = entry;
        entries[source].padding.x = entry.padding.x;
        output = slot;
    }
    // After commit this head array addresses the retained history entries.
    heads[bucket].store(output);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void remapProbeHeads(uint3 thread : SV_DispatchThreadID) {
    uint slot = thread.x;
    if (slot >= probeDirectoryCapacity(params)) return;
    uint head = probeDirectoryHead(0u, slot, params);
    uint source = heads[head].load();
    if (source != EMPTY) {
        uint encoded = entries[source].padding.x;
        uint mapped = encoded == EMPTY ? EMPTY : encoded & 0x7fffffffu;
        if (mapped == EMPTY) {
            // Commit can evict the directory representative while retaining
            // sibling directions. Main heads now address previousEntries, not
            // the working entries array; preserve the packed probe's slot.
            Entry representative = entries[source];
            uint mode = uint(representative.key.w) & PROBE_LOD;
            uint quota = cacheHistoryQuota(params);
            [loop] for (uint direction = 0u; direction < 32u && mapped == EMPTY; ++direction) {
                int4 key = int4(representative.key.xyz, int(direction | mode));
                uint bucket = transportCacheBucket(key, params.size.w);
                uint committed = heads[bucket].load();
                [loop] for (uint count = 0u; count < quota && committed != EMPTY; ++count) {
                    Entry candidate = previousEntries[committed];
                    if (all(candidate.key == key)) {
                        mapped = committed;
                        break;
                    }
                    committed = candidate.next;
                }
            }
        }
        heads[head].store(mapped == EMPTY ? TOMBSTONE : mapped);
    }
}

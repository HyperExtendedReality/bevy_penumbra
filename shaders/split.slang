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

bool localEmissiveLightHistoryAvailable() {
    return persistentWorldIntervals(params)
        && isfinite(params.emissiveDirtyMin.w)
        && params.emissiveDirtyMin.w > 0.5
        && all(isfinite(params.emissiveDirtyMin.xyz))
        && all(isfinite(params.emissiveDirtyMax.xyz))
        && all(params.emissiveDirtyMin.xyz <= params.emissiveDirtyMax.xyz);
}

bool emissiveDirtyRequested() {
    return !isfinite(params.emissiveDirtyMin.w) || params.emissiveDirtyMin.w > 0.5;
}

bool dependencyIntersectsEmissiveDirty(DependencyContext dependency) {
    if (!emissiveDirtyRequested()) return false;
    if (!all(isfinite(params.emissiveDirtyMin.xyz))
        || !all(isfinite(params.emissiveDirtyMax.xyz))
        || any(params.emissiveDirtyMin.xyz > params.emissiveDirtyMax.xyz)) return true;
    if (!dependency.known) return true;
    if (!dependency.hasBounds) return false;
    return all(dependency.minimum <= params.emissiveDirtyMax.xyz)
        && all(dependency.maximum >= params.emissiveDirtyMin.xyz);
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

bool sameBits(float4 left, float4 right) {
    return all(asuint(left) == asuint(right));
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

[ForceInline]
float4 entryRadianceInterval(Entry entry, uint source) {
    if (source == BACKGROUND_SOURCE) return entry.interval;
    if (source == EMISSIVE_SOURCE) return entry.emissiveInterval;
    return entry.punctualInterval;
}

[ForceInline]
void setEntryRadianceInterval(inout Entry entry, uint source, float4 value) {
    if (source == BACKGROUND_SOURCE) entry.interval = value;
    else if (source == EMISSIVE_SOURCE) entry.emissiveInterval = value;
    else entry.punctualInterval = value;
}

[ForceInline]
void setEntryRadianceMerged(inout Entry entry, uint source, float4 value) {
    if (source == BACKGROUND_SOURCE) entry.merged = value;
    else if (source == EMISSIVE_SOURCE) entry.emissiveMerged = value;
    else entry.punctualMerged = value;
}


bool radianceHistoryEligible(DependencyContext geometryDependency,
                             float4 lightHitBounds, uint source) {
    if (dependencyIntersectsDirty(geometryDependency)
        || !transportSourceActive(params, source)) return false;
    if ((params.sourceHistory.y & source) == 0u) return true;
    if (source == BACKGROUND_SOURCE) return (params.sourceHistory.z & 1u) != 0u;
    if (source == EMISSIVE_SOURCE)
        return localEmissiveLightHistoryAvailable()
            && !dependencyIntersectsEmissiveDirty(geometryDependency);
    if (source == PUNCTUAL_SOURCE)
        return (params.sourceHistory.z & 4u) != 0u || (localPunctualLightHistoryAvailable()
            && !lightHitBoundsIntersectLocalLightDirty(lightHitBounds));
    return false;
}
// After a gradual change even a subsequently sampled entry can contain a
// blend of pre-change radiance. Expire inactive background observations after
// a bounded age; preserve beta and the independent source groups.
bool backgroundMotionHistoryAlive(uint lastSample) {
    return (params.sourceHistory.z & 2u) == 0u
        || params.frame.x - lastSample <= 120u;
}
bool punctualMotionHistoryAlive(uint lastSample) {
    return (params.sourceHistory.z & 8u) == 0u
        || params.frame.x - lastSample <= 120u;
}


bool radianceProbeHistoryEligible(DependencyContext dependency) {
    float4 unknownHitBounds = unknownLightHitBounds();
    return radianceHistoryEligible(dependency, unknownHitBounds, BACKGROUND_SOURCE)
        && (!transportSourceActive(params, EMISSIVE_SOURCE)
            || radianceHistoryEligible(dependency, unknownHitBounds, EMISSIVE_SOURCE))
        && (!transportSourceActive(params, PUNCTUAL_SOURCE)
            || radianceHistoryEligible(dependency, unknownHitBounds, PUNCTUAL_SOURCE));
}

bool dirtyRawCacheEntry(Entry entry) {
    return dependencyIntersectsDirty(entryDependency(entry));
}

uint mergeVisibilityAdmissionLimit() {
    return params.sampling.y;
}

void splitLodBlend(float3 position, out uint lowLod, out uint highLod,
                   out float blend) {
    // Both sparse compute and full-resolution composite use the same LOD.
    splitLodBlendForParameters(position, params, lowLod, highLod, blend);
    if (params.environmentScale.z > 0.5) {
        lowLod = min(lowLod + 2u, MAX_PROBE_LOD);
        highLod = min(highLod + 2u, MAX_PROBE_LOD);
        if (lowLod == highLod) blend = 0.0;
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

uint probeCountHead(uint level, uint slot, Parameters p) {
    uint directoryCapacity = probeDirectoryCapacity(p);
    // Four count-head arrays follow the probe directories. Cold rank borrows
    // levels 1-3 as scratch; parent-count passes replace occupied slots before prefix.
    return probeDirectoryBase(0u, p) + 4u * directoryCapacity
        + level * directoryCapacity + slot;
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
            int4 candidate = entries[index].key;
            candidate.w = int(uint(candidate.w) & (PROBE_LOD | (3u << 12u)));
            if (all(candidate == key)) return index;
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
        if (index != TOMBSTONE && index < descriptorCount) {
            int4 candidate = entries[index].key;
            // A packed probe's canonical entry can be a traced interval: its
            // angular direction is not part of the probe-directory identity.
            candidate.w = int(uint(candidate.w) & (PROBE_LOD | (3u << 12u)));
            if (all(candidate == key)) return slot;
        }
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
        // Hash-chain traversal needs only the link and key, not the 272-byte
        // radiance record. A conditional aggregate load expands to every field.
        uint next = previous ? previousEntries[index].next : entries[index].next;
        if (next == index) return EMPTY;
        int4 candidate = previous ? previousEntries[index].key : entries[index].key;
        if (all(candidate == key)) return index;
        index = next;
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
            int4 candidate = previous ? previousEntries[index].key : entries[index].key;
            if (all(candidate.xyz == key.xyz)
                && (uint(candidate.w) & PROBE_LOD) == (uint(key.w) & PROBE_LOD)) {
                if (previous) {
                    DependencyContext entryBounds = storedDependency(
                        previousEntries[index].mergedDependencyMin,
                        previousEntries[index].mergedDependencyMax);
                    if (dependencyIntersectsDirty(entryBounds)) return EMPTY;
                    uint page = (((params.frame.x + 1u) & 1u)
                        + (params.environmentScale.z > 0.5 ? 2u : 0u))
                        * directoryCapacity + slot;
                    DependencyContext packedDependency = probeDependency(
                        directionalProbes[page].dependencyMin,
                        directionalProbes[page].dependencyMax);
                    // Packed dependency bounds aggregate active source groups.
                    if (!radianceProbeHistoryEligible(packedDependency)) return EMPTY;
                    if (!backgroundMotionHistoryAlive(previousEntries[index].lastSample)
                        || !punctualMotionHistoryAlive(previousEntries[index].lastSample)) return EMPTY;
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

// Interpolates every source in `sources` from one walk over the eight child
// corners. Each hash-chain lookup is shared by the sources instead of being
// repeated per source. Every source keeps its own support, weights and
// light-hit bounds, accumulated in the same corner order. An inactive source
// yields analytic zero, float4(0, 0, 0, 1), and touches no bounds.
void interpolateMergedChildSources(int3 base, float3 fraction, uint direction,
                                   uint level, uint lod, uint sources,
                                   inout DependencyContext dependency,
                                   out float4 background, out float4 emissive,
                                   out float4 punctual,
                                   inout float4 backgroundBounds,
                                   inout float4 emissiveBounds,
                                   inout float4 punctualBounds) {
    uint active = 0u;
    [unroll] for (uint bit = 1u; bit <= PUNCTUAL_SOURCE; bit <<= 1u)
        if ((sources & bit) != 0u && transportSourceActive(params, bit)) active |= bit;
    float3 backgroundSum = 0.0, emissiveSum = 0.0, punctualSum = 0.0;
    float backgroundTotal = 0.0, emissiveTotal = 0.0, punctualTotal = 0.0;
    float potential = 0.0;
    if (active != 0u) {
        [loop] for (uint corner = 0u; corner < 8u; ++corner) {
            uint3 bit = uint3(corner & 1u, (corner >> 1u) & 1u, corner >> 2u);
            float3 axisWeight = float3(
                bit.x == 0u ? 1.0 - fraction.x : fraction.x,
                bit.y == 0u ? 1.0 - fraction.y : fraction.y,
                bit.z == 0u ? 1.0 - fraction.z : fraction.z);
            float weight = axisWeight.x * axisWeight.y * axisWeight.z;
            if (!(weight > 0.0)) continue;
            potential += weight;
            int4 key = int4(base + int3(bit), int(direction | (level << 12u) | (lod << 18u)));
            uint index = findEntry(key, false);
            if (index == EMPTY) continue;
            float4 childLightHitBounds = entries[index].boundaryMerged;
            bool supported = false;
            // The sources usually accumulate the same corners, so their
            // bounds stay bit-identical. Reuse a union already computed from
            // an identical accumulator this corner instead of repeating it.
            float4 backgroundBefore = backgroundBounds;
            float4 emissiveBefore = emissiveBounds;
            bool backgroundUnited = false;
            bool emissiveUnited = false;
            // RGB is conditional per source; support stays independent until packing.
            if ((active & BACKGROUND_SOURCE) != 0u) {
                float4 sample = entries[index].merged;
                if (sample.w > 0.0 && all(isfinite(sample))) {
                    float supportedWeight = weight * saturate(sample.w);
                    backgroundSum += sample.xyz * supportedWeight;
                    backgroundTotal += supportedWeight;
                    backgroundBounds = unionConservativeSpheres(backgroundBounds, childLightHitBounds);
                    backgroundUnited = true;
                    supported = true;
                }
            }
            if ((active & EMISSIVE_SOURCE) != 0u) {
                float4 sample = entries[index].emissiveMerged;
                if (sample.w > 0.0 && all(isfinite(sample))) {
                    float supportedWeight = weight * saturate(sample.w);
                    emissiveSum += sample.xyz * supportedWeight;
                    emissiveTotal += supportedWeight;
                    emissiveBounds = backgroundUnited && sameBits(emissiveBounds, backgroundBefore)
                        ? backgroundBounds
                        : unionConservativeSpheres(emissiveBounds, childLightHitBounds);
                    emissiveUnited = true;
                    supported = true;
                }
            }
            if ((active & PUNCTUAL_SOURCE) != 0u) {
                float4 sample = entries[index].punctualMerged;
                if (sample.w > 0.0 && all(isfinite(sample))) {
                    float supportedWeight = weight * saturate(sample.w);
                    punctualSum += sample.xyz * supportedWeight;
                    punctualTotal += supportedWeight;
                    punctualBounds = backgroundUnited && sameBits(punctualBounds, backgroundBefore)
                        ? backgroundBounds
                        : emissiveUnited && sameBits(punctualBounds, emissiveBefore)
                            ? emissiveBounds
                            : unionConservativeSpheres(punctualBounds, childLightHitBounds);
                    supported = true;
                }
            }
            // Dependency union is an order-independent box union.
            if (supported)
                dependencyUnion(dependency, storedDependency(
                    entries[index].mergedDependencyMin, entries[index].mergedDependencyMax));
        }
    }
    background = (active & BACKGROUND_SOURCE) == 0u ? float4(0.0, 0.0, 0.0, 1.0)
        : (backgroundTotal > 0.0 ? float4(backgroundSum / backgroundTotal,
            saturate(backgroundTotal / max(potential, 1e-8))) : 0.0);
    emissive = (active & EMISSIVE_SOURCE) == 0u ? float4(0.0, 0.0, 0.0, 1.0)
        : (emissiveTotal > 0.0 ? float4(emissiveSum / emissiveTotal,
            saturate(emissiveTotal / max(potential, 1e-8))) : 0.0);
    punctual = (active & PUNCTUAL_SOURCE) == 0u ? float4(0.0, 0.0, 0.0, 1.0)
        : (punctualTotal > 0.0 ? float4(punctualSum / punctualTotal,
            saturate(punctualTotal / max(potential, 1e-8))) : 0.0);
}

float4 interpolateMergedChildren(int3 base, float3 fraction, uint direction,
                                 uint level, uint lod, uint source,
                                 inout DependencyContext dependency,
                                 inout float4 lightHitBounds) {
    float4 background, emissive, punctual;
    float4 backgroundBounds = lightHitBounds, emissiveBounds = lightHitBounds;
    float4 punctualBounds = lightHitBounds;
    interpolateMergedChildSources(base, fraction, direction, level, lod, source,
        dependency, background, emissive, punctual,
        backgroundBounds, emissiveBounds, punctualBounds);
    if (source == BACKGROUND_SOURCE) {
        lightHitBounds = backgroundBounds;
        return background;
    }
    if (source == EMISSIVE_SOURCE) {
        lightHitBounds = emissiveBounds;
        return emissive;
    }
    lightHitBounds = punctualBounds;
    return punctual;
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
    if (thread.x == 0u && params.ownerSequence.z != 0u) {
        heads[rootRayOffsetHead(params)].store(0u);
        heads[traceCountHead(params)].store(0u);
        heads[traceArgsHead(params)].store(0u);
        heads[traceArgsHead(params) + 1u].store(1u);
        heads[traceArgsHead(params) + 2u].store(1u);
        uint headWords, headStride;
        heads.GetDimensions(headWords, headStride);
        if (params.environmentScale.z <= 0.5 && headWords >= deferredTraceEnd(params)) {
            heads[deferredQueryCountHead(params)].store(0u);
            heads[deferredQueryArgsHead(params)].store(0u);
            heads[deferredQueryArgsHead(params) + 1u].store(1u);
            heads[deferredQueryArgsHead(params) + 2u].store(1u);
        }
    }
    // Both head clears run before transport. The second one preserves shared
    // query/owner counters; only the first needs to clear sparse output images,
    // which otherwise incurs a redundant full-image write on every frame.
    if (params.ownerSequence.z != 0u && params.environmentScale.z <= 0.5) {
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

uint candidatePermutationMask(uint candidateCount) {
    uint mask = 0u;
    while (mask < candidateCount - 1u) mask = (mask << 1u) | 1u;
    return mask;
}

uint candidatePermutationSeed(uint mask) {
    return ((params.size.x * 0x9e3779b9u)
        ^ (params.size.y * 0x85ebca6bu) ^ params.size.z) & mask;
}

uint transportCandidateRank(uint sample) {
    uint candidateCount = params.size.z;
    if (candidateCount <= 1u) return 0u;
    // Cycle walking restricts this reversible permutation to an arbitrary
    // candidate count while preserving exact no-repeat epochs. The power-of-2
    // domain is at most twice the candidate grid, so rejection is usually rare.
    uint mask = candidatePermutationMask(candidateCount);
    uint seed = candidatePermutationSeed(mask);
    uint rank = permuteCandidateDomain(sample, mask, seed);
    [loop] while (rank >= candidateCount) {
        rank = permuteCandidateDomain(rank, mask, seed);
    }
    return rank;
}

uint transportCandidateFromRank(uint rank) {
    uint candidateCount = params.size.z;
    if (candidateCount <= 1u) return 0u;
    uint mask = candidatePermutationMask(candidateCount);
    uint seed = candidatePermutationSeed(mask);
    uint sample = inversePermuteCandidateDomain(rank, mask, seed);
    [loop] while (sample >= candidateCount)
        sample = inversePermuteCandidateDomain(sample, mask, seed);
    return sample;
}

uint coldCandidateRank(uint sample) {
    uint candidateCount = params.size.z;
    if (candidateCount <= 1u) return 0u;
    uint mask = candidatePermutationMask(candidateCount);
    uint seed = candidatePermutationSeed(mask)
        ^ (params.frame.x * 0x9e3779b9u);
    seed &= mask;
    uint rank = permuteCandidateDomain(sample, mask, seed);
    [loop] while (rank >= candidateCount)
        rank = permuteCandidateDomain(rank, mask, seed);
    return rank;
}

uint coldCandidateFromRank(uint rank) {
    uint candidateCount = params.size.z;
    if (candidateCount <= 1u) return 0u;
    uint mask = candidatePermutationMask(candidateCount);
    uint seed = candidatePermutationSeed(mask)
        ^ (params.frame.x * 0x9e3779b9u);
    seed &= mask;
    uint sample = inversePermuteCandidateDomain(rank, mask, seed);
    [loop] while (sample >= candidateCount)
        sample = inversePermuteCandidateDomain(sample, mask, seed);
    return sample;
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

bool coldPriorityEnabled(Parameters p) {
    return p.environmentScale.z <= 0.5 && persistentWorldIntervals(p) && p.history.w > 0.0 && p.history.z <= 0.0
        && p.sampling.x > 0u && (p.ownerSequence.w & 0x40000000u) == 0u;
}

uint coldPopulationBin(uint population) {
    uint bin = 0u;
    [loop] while (population > 1u && bin < 31u) {
        population >>= 1u;
        ++bin;
    }
    return bin;
}

bool reserveColdUpdate() {
    uint counter = 2u * params.size.w + 7u;
    uint budget = params.sampling.x;
    uint ticket = heads[counter].add(1u);
    if (ticket < budget) return true;
    // Avoid relying on strong compare-exchange semantics after Slang lowers
    // atomics to WGSL's weak compare-exchange operation.
    heads[counter].add(0xffffffffu);
    return false;
}

// Secondary seeds are previous-frame primary hits, never secondary hits.
bool splitSampleSurface(uint2 pixel, out float3 position, out float3 normal) {
    position = 0.0;
    normal = 0.0;
    if (params.environmentScale.z > 0.5) {
        if (params.environmentScale.w <= 0.5) return false;
        float4 hit = previousIrradiance.Load(int3(pixel, 0));
        if (!(hit.w >= 1.0) || !all(isfinite(hit))) return false;
        uint packed = uint(hit.w - 1.0);
        normal = decodeNormal(float2(packed & 4095u, packed >> 12u) / 4095.0);
        position = hit.xyz;
        return true;
    }
    uint2 full = fullPixel(pixel, params);
    float depth = depthTexture.Load(int3(full, 0));
    if (!(depth > 0.0) || !isfinite(depth)) return false;
    uint4 deferred = gbuffer.Load(int3(full, 0));
    uint4 raster = rasterReceiver.Load(int3(full, 0));
    float3 shadingNormal = deferredShadingNormal(deferred);
    if (!(dot(shadingNormal, shadingNormal) > 0.01) || !all(isfinite(shadingNormal))) return false;
    normal = receiverGeometricNormal(raster.z, shadingNormal);
    position = worldPosition(float2(full) + 0.5, depth, params);
    float3 cameraNear = worldPosition(float2(full) + 0.5, 1.0, params);
    if (dot(normal, cameraNear - position) < 0.0) normal = -normal;
    return all(isfinite(position));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void seed(uint3 thread : SV_DispatchThreadID) {
    if (any(thread.xy >= params.size.xy)) return;
    uint sample = thread.x + thread.y * params.size.x;
    // Every descriptor is written exactly once, either with its seeded entry
    // or zero. Entries are 272 bytes, so clearing and then overwriting them
    // doubled this bandwidth-bound pass.
    float3 surfacePosition, geometricNormal;
    if (!splitSampleSurface(thread.xy, surfacePosition, geometricNormal)) {
        [unroll] for (uint variant = 0u; variant < 2u; ++variant)
            [unroll] for (uint level = 0u; level < CASCADES; ++level)
                entries[probeDescriptorIndex(sample, variant, level)] = (Entry)0;
        return;
    }

    uint lowLod, highLod;
    float lodBlend;
    splitLodBlend(surfacePosition, lowLod, highLod, lodBlend);
    uint selectedLod = 0u;
    if (highLod != lowLod) {
        uint phaseBits = transportHash(sample ^ (params.frame.x * 0x9e3779b9u));
        float phase = float(phaseBits >> 8u) * (1.0 / 16777216.0);
        selectedLod = phase < lodBlend ? 1u : 0u;
    }
    bool coldPriority = coldPriorityEnabled(params);
    // The preceding primary window chose these hits; cap them independently of the current window.
    bool secondary = params.environmentScale.z > 0.5;
    bool selectedRay = secondary
        ? ((params.ownerSequence.w & 0x40000000u) == 0u
            && (params.sampling.x == 0u || reserveColdUpdate()))
        : transportCandidateSelected(sample);
    bool eligibleRay = selectedRay || coldPriority;
    [unroll] for (uint variant = 0u; variant < 2u; ++variant) {
        if (variant == 1u && highLod == lowLod) {
            [unroll] for (uint level = 0u; level < CASCADES; ++level)
                entries[probeDescriptorIndex(sample, variant, level)] = (Entry)0;
            continue;
        }
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
            entry.padding.x = (level == 0u && eligibleRay && variant == selectedLod)
                ? 1u : 0u;
            entry.padding.y = 1u;
            entries[index] = entry;
        }
    }
    if (selectedRay && !coldPriority && (!secondary || params.sampling.x == 0u))
        heads[2u * params.size.w + 7u].add(1u);
}

// Deduplicate immutable per-pixel descriptors after seed's dispatch barrier.
// Insertion publishes only a descriptor index; keys are never read while a
// writer is still constructing the record. Same-cell pixels take one bounded
// probe-directory lookup instead of forming a screen-sized linked list.
void rankCandidate(uint dispatchRank) {
    if (dispatchRank >= params.size.z) return;
    bool coldPriority = coldPriorityEnabled(params);
    if (!coldPriority && params.frame.y != 0u) return;
    uint sample = coldPriority && params.frame.y != 0u
        ? transportCandidateFromRank(dispatchRank) : dispatchRank;
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
            if (level == 0u) {
                uint slot = findProbeNodeSlot(descriptor.key);
                if (slot != EMPTY) {
                    uint countHead = probeCountHead(0u, slot, params);
                    if (coldPriority) {
                        if (params.frame.y == 0u && descriptor.padding.x != 0u) {
                            // A frame-seeded permutation samples a stable receiver
                            // representative without depending on atomic arrival order.
                            uint rank = coldCandidateRank(sample);
                            heads[countHead].max(0xffffffffu - rank);
                            // Count only selected-LOD candidates, weighting cold
                            // admission by projected coverage.
                            heads[probeCountHead(1u, slot, params)].add(1u);
                        } else if (params.frame.y == 1u && index == canonical) {
                            uint population = heads[
                                probeCountHead(1u, slot, params)].load();
                            if (population != 0u) {
                                uint bin = coldPopulationBin(population);
                                heads[probeCountHead(2u, bin, params)].add(1u);
                            }
                        } else if (params.frame.y == 2u && index == canonical) {
                            uint nomineePriority = heads[countHead].load();
                            if (nomineePriority != 0u) {
                                uint nomineeRank = 0xffffffffu - nomineePriority;
                                uint nomineeSample = coldCandidateFromRank(nomineeRank);
                                uint nomineeIndex = probeDescriptorIndex(nomineeSample, 0u, 0u);
                                if (entries[nomineeIndex].valid == 0u
                                    || any(entries[nomineeIndex].key != descriptor.key))
                                    nomineeIndex = probeDescriptorIndex(nomineeSample, 1u, 0u);

                                uint population = heads[
                                    probeCountHead(1u, slot, params)].load();
                                uint bin = coldPopulationBin(population);
                                uint maxBin = coldPopulationBin(params.size.z);
                                uint higherPopulationProbes = 0u;
                                [loop] for (uint higherBin = bin + 1u;
                                            higherBin <= maxBin; ++higherBin)
                                    higherPopulationProbes += heads[
                                        probeCountHead(2u, higherBin, params)].load();

                                uint budget = params.sampling.x;
                                bool admitted = false;
                                if (population != 0u && higherPopulationProbes < budget) {
                                    uint ticket = heads[
                                        probeCountHead(3u, bin, params)].add(1u);
                                    admitted = ticket < budget - higherPopulationProbes;
                                }
                                entries[nomineeIndex].padding.x = admitted ? 2u : 0u;
                                entries[nomineeIndex].lastSample = 0u;
                                heads[countHead].store(admitted ? 1u : 0u);
                                if (admitted) heads[2u * params.size.w + 7u].add(1u);
                            }
                        } else if (params.frame.y == 3u && descriptor.padding.x != 0u) {
                            if (descriptor.padding.x == 2u) {
                                entries[index].padding.x = 1u;
                            } else if (descriptor.padding.x == 1u
                                       && transportCandidateSelected(sample)
                                       && reserveColdUpdate()) {
                                uint localOrdinal = heads[countHead].add(1u);
                                entries[index].lastSample = localOrdinal;
                            } else {
                                entries[index].padding.x = 0u;
                            }
                        }
                    } else if (descriptor.padding.x != 0u) {
                        uint localOrdinal = heads[countHead].add(1u);
                        entries[index].lastSample = localOrdinal;
                    }
                }
            }
        }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rank(uint3 thread : SV_DispatchThreadID) {
    rankCandidate(thread.x);
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
        offset = heads[rootRayOffsetHead(params)].add(rayCount);
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
    if (params.environmentScale.y > 0.5 && params.environmentScale.z <= 0.5)
        irradiance.Store(int2(thread.xy), 0.0);
    bool selected = false;
    [unroll] for (uint variant = 0u; variant < 2u; ++variant) {
        uint index = probeDescriptorIndex(sample, variant, 0u);
        if (entries[index].padding.x == 0u || entries[index].next == EMPTY) continue;
        uint canonical = entries[index].next;
        entries[index].lastSample += entries[canonical].valid;
        selected = true;
    }
    if (selected) {
        uint ticket = heads[traceCountHead(params)].add(1u);
        heads[traceSamplesHead(params) + ticket].store(sample);
        heads[traceArgsHead(params)].max(ticket * traceLaneStride(params) / 64u + 1u);
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
        uint next = previousEntries[index].next;
        if (next == index) break;
        bucketLoad++;
        if (!intervalHistoryAlive(previousEntries[index].lastSample, params)) continue;
        DependencyContext dependency = storedDependency(
            previousEntries[index].dependencyMin, previousEntries[index].dependencyMax);
        if (!betaHistoryEligible(dependency, previousEntries[index].normal)) continue;
        Entry entry = previousEntries[index];
        if (!backgroundMotionHistoryAlive(entry.lastSample)
            || !radianceHistoryEligible(entryDependency(entry),
                entry.boundaryInterval, BACKGROUND_SOURCE))
            entry.interval.xyz = 0.0;
        if (!backgroundMotionHistoryAlive(entry.lastSample)
            || !radianceHistoryEligible(entryMergedDependency(entry),
                entry.boundaryMerged, BACKGROUND_SOURCE))
            entry.merged = 0.0;
        if (!transportSourceActive(params, EMISSIVE_SOURCE)) {
            entry.emissiveInterval = 0.0;
            entry.emissiveMerged = float4(0.0, 0.0, 0.0, 1.0);
        } else {
            if (!radianceHistoryEligible(entryDependency(entry),
                    entry.boundaryInterval, EMISSIVE_SOURCE))
                entry.emissiveInterval = 0.0;
            if (!radianceHistoryEligible(entryMergedDependency(entry),
                    entry.boundaryMerged, EMISSIVE_SOURCE))
                entry.emissiveMerged = 0.0;
        }
        if (!transportSourceActive(params, PUNCTUAL_SOURCE)) {
            entry.punctualInterval = 0.0;
            entry.punctualMerged = float4(0.0, 0.0, 0.0, 1.0);
        } else {
            if (!punctualMotionHistoryAlive(entry.lastSample) || !radianceHistoryEligible(entryDependency(entry),
                    entry.boundaryInterval, PUNCTUAL_SOURCE))
                entry.punctualInterval = 0.0;
            if (!punctualMotionHistoryAlive(entry.lastSample) || !radianceHistoryEligible(entryMergedDependency(entry),
                    entry.boundaryMerged, PUNCTUAL_SOURCE))
                entry.punctualMerged = 0.0;
        }
        uint slot = capacity + bucketBase + bucketRetained++;
        entry.samples = 0;
        entry.valid = params.frame.x | 0x80000000u;
        entry.next = heads[thread.x].exchange(slot);
        entries[slot] = entry;
    }
    // The loop visits at most bucketQuota entries, so every eligible entry fits.
    // Aggregate the diagnostic rather than contending on it for each entry.
    if (bucketRetained != 0u) heads[params.size.w].add(bucketRetained);
    if (bucketLoad > 1) heads[2 * params.size.w + 3].add(bucketLoad - 1);
}

void deposit(uint first, float3 samplePosition, float3 rayOrigin,
             float3 direction, Hit hit, TransportRadiance radiance, uint lod,
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
        entry.interval = float4(terminates ? radiance.background : 0.0,
            terminates ? 0.0 : 1.0);
        entry.emissiveInterval = float4(
            terminates && transportSourceActive(params, EMISSIVE_SOURCE)
                ? radiance.emissive : 0.0,
            transportSourceActive(params, EMISSIVE_SOURCE) ? 1.0 : 0.0);
        entry.punctualInterval = float4(
            terminates && transportSourceActive(params, PUNCTUAL_SOURCE)
                ? radiance.punctual : 0.0,
            transportSourceActive(params, PUNCTUAL_SOURCE) ? 1.0 : 0.0);
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
            ? (params.environmentScale.y > 0.5 ? unknownLightHitBounds()
                : lightHitBoundsAt(rayOrigin + direction * hit.distance))
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

// Immediate traces shade and deposit here. Deferred traces only store the
// closest hit; `deferred` is a literal at each call so the closest-hit kernel
// carries no hit-shading code.
[ForceInline]
void traceTransport(uint3 thread, bool deferred, uint ticket) {
    uint2 pixel = thread.xy;
    if (any(pixel >= params.size.xy)) return;
    uint sample = pixel.x + pixel.y * params.size.x;
    uint firstLow = probeDescriptorIndex(sample, 0u, 0u);
    uint firstHigh = probeDescriptorIndex(sample, 1u, 0u);
    bool lowSelected = entries[firstLow].padding.x != 0u && entries[firstLow].next != EMPTY;
    bool highSelected = entries[firstHigh].padding.x != 0u && entries[firstHigh].next != EMPTY;
    if (!lowSelected && !highSelected) return;
    float3 surfacePosition, geometricNormal;
    if (!splitSampleSurface(pixel, surfacePosition, geometricNormal)) return;

    float2 jitter = r2(transportHash(params.frame.x ^ 0x5a17c9e3u));
    // Not unrolled: each copy would inline the whole traversal, and the driver's
    // compile time grows far faster than the code.
    [loop] for (uint variant = 0u; variant < 2u; ++variant) {
        if (variant == 0u ? !lowSelected : !highSelected) continue;
        uint first = variant == 0u ? firstLow : firstHigh;
        uint lod = (uint(entries[first].key.w) & PROBE_LOD) >> 18u;
        uint sequence = entries[first].lastSample;
        float2 uv = frac(0.5 + r2(sequence + 1u) + jitter);
        float3 direction = decodeDirection(uv);
        if (dot(direction, geometricNormal) < 0.0) direction = -direction;
        float3 rayOrigin = rayOffset(surfacePosition, geometricNormal, params.tuning.z);
        DependencyContext rayDependency = dependencyUnknown();
        float dependencyStart = 0.0, dependencyDistance = 0.0;
        float dependencyPadding = 0.0;
        float coneWidth = params.tuning.x * 0.5;
        Hit hit = traceRayTracked(rayOrigin, direction, 1e20, rayDependency,
            dependencyStart, dependencyDistance, dependencyPadding,
            false, coneWidth, 0.5);
        if (!deferred) {
            finishTransportVariant(pixel, first, lod, surfacePosition, rayOrigin,
                direction, hit, rayDependency, dependencyStart, dependencyDistance,
                dependencyPadding);
            continue;
        }
        storeDeferredSlot(ticket * 2u + variant, surfacePosition, rayOrigin, direction,
            hit, rayDependency, dependencyStart, dependencyDistance, dependencyPadding, 0u);
    }
}

bool loadDeferredSlot(uint slot, out float3 surfacePosition, out float3 rayOrigin,
                      out float3 direction, out Hit hit, out DependencyContext rayDependency) {
    uint word = deferredSlotHead(slot, params);
    [unroll] for (uint i = 0u; i < 3u; ++i) {
        surfacePosition[i] = asfloat(heads[word + i].load());
        rayOrigin[i] = asfloat(heads[word + 3u + i].load());
        direction[i] = asfloat(heads[word + 6u + i].load());
        hit.barycentric[i] = asfloat(heads[word + 13u + i].load());
        rayDependency.minimum[i] = asfloat(heads[word + 17u + i].load());
        rayDependency.maximum[i] = asfloat(heads[word + 20u + i].load());
    }
    hit.distance = asfloat(heads[word + 9u].load());
    hit.triangle = heads[word + 10u].load();
    hit.instance = heads[word + 11u].load();
    hit.errors = heads[word + 12u].load();
    hit.footprint = asfloat(heads[word + 16u].load());
    uint dependencyFlags = heads[word + 23u].load();
    rayDependency.known = (dependencyFlags & 1u) != 0u;
    rayDependency.hasBounds = (dependencyFlags & 2u) != 0u;
    return heads[word + 31u].load() == deferredSlotTag();
}

// Hit shading, optional secondary seeding, and interval deposition for one
// traced receiver variant. Shared by the immediate and deferred traces.
void finishTransportVariant(uint2 pixel, uint first, uint lod, float3 surfacePosition,
                            float3 rayOrigin, float3 direction, Hit hit,
                            DependencyContext rayDependency, float dependencyStart,
                            float dependencyDistance, float dependencyPadding) {
    DependencyContext outgoingDependency = dependencyEmpty();
    bool radianceSupported;
    TransportRadiance radiance = outgoingTransportRadianceTracked(rayOrigin,
        direction, hit, radianceSupported, outgoingDependency,
        mergeVisibilityAdmissionLimit());
    if (!radianceSupported) return;
    if (params.environmentScale.y > 0.5 && params.environmentScale.z <= 0.5
        && hit.triangle != EMPTY && (hit.errors & HIT_QUERY_UNSUPPORTED) == 0u) {
        HitAttributes attributes = hitAttributes(hit);
        float3 hitNormal = attributes.geometricNormal;
        if (dot(hitNormal, direction) > 0.0) hitNormal = -hitNormal;
        float3 hitPosition = rayOrigin + direction * hit.distance;
        float3 hitShading = hitShadingNormal(hit, attributes, direction);
        bool unlit = (sceneWord(instanceBase(hit.instance) + 6u).w & 16u) != 0u;
        if (!unlit && all(isfinite(hitPosition)) && all(isfinite(hitNormal))
            && all(isfinite(hitShading)) && dot(hitNormal, hitNormal) > 0.01
            && dot(hitShading, -direction) > 0.0 && dot(hitShading, hitNormal) > 0.0)
            irradiance.Store(int2(pixel), float4(hitPosition,
                receiverPackedNormal(normalize(hitNormal)) + 1.0));
    }
    deposit(first, surfacePosition, rayOrigin, direction, hit, radiance, lod,
        rayDependency, dependencyStart, dependencyDistance, dependencyPadding,
        outgoingDependency);
}

// The deferred scratch exists only in primary cache heads sized for it.
bool deferredTraceAvailable() {
    if (params.environmentScale.z > 0.5 || params.history.y > 0.5) return false;
    uint words, stride;
    heads.GetDimensions(words, stride);
    return words >= deferredTraceEnd(params);
}

uint deferredSlotTag() { return params.frame.x | 0x80000000u; }

void storeDeferredSlot(uint slot, float3 surfacePosition, float3 rayOrigin, float3 direction,
                       Hit hit, DependencyContext rayDependency, float dependencyStart,
                       float dependencyDistance, float dependencyPadding, uint queries) {
    uint word = deferredSlotHead(slot, params);
    [unroll] for (uint i = 0u; i < 3u; ++i) {
        heads[word + i].store(asuint(surfacePosition[i]));
        heads[word + 3u + i].store(asuint(rayOrigin[i]));
        heads[word + 6u + i].store(asuint(direction[i]));
        heads[word + 13u + i].store(asuint(hit.barycentric[i]));
        heads[word + 17u + i].store(asuint(rayDependency.minimum[i]));
        heads[word + 20u + i].store(asuint(rayDependency.maximum[i]));
    }
    heads[word + 9u].store(asuint(hit.distance));
    heads[word + 10u].store(hit.triangle);
    heads[word + 11u].store(hit.instance);
    heads[word + 12u].store(hit.errors);
    heads[word + 16u].store(asuint(hit.footprint));
    heads[word + 23u].store((rayDependency.known ? 1u : 0u)
        | (rayDependency.hasBounds ? 2u : 0u));
    heads[word + 24u].store(asuint(dependencyStart));
    heads[word + 25u].store(asuint(dependencyDistance));
    heads[word + 26u].store(asuint(dependencyPadding));
    heads[word + 30u].store(queries);
    heads[word + 31u].store(deferredSlotTag());
}

uint traceTicket(uint lane) {
    uint stride = traceLaneStride(params);
    if (lane % stride != 0u) return EMPTY;
    uint ticket = lane / stride;
    return ticket < heads[traceCountHead(params)].load() ? ticket : EMPTY;
}

uint2 ticketPixel(uint ticket) {
    uint sample = heads[traceSamplesHead(params) + ticket].load();
    return uint2(sample % params.size.x, sample / params.size.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void trace(uint3 thread : SV_DispatchThreadID) {
    uint ticket = traceTicket(thread.x);
    if (ticket == EMPTY) return;
    // With deferred scratch, this kernel only takes tickets beyond its capacity.
    if (ticket < deferredTraceCapacity(params) && deferredTraceAvailable()) return;
    traceTransport(uint3(ticketPixel(ticket), 0u), false, ticket);
}

// Deferred primary trace, pass 1: closest hits only.
[ForceInline]
void traceDeferredKernel(uint3 thread) {
    uint ticket = traceTicket(thread.x);
    if (ticket == EMPTY || ticket >= deferredTraceCapacity(params)
        || !deferredTraceAvailable()) return;
    traceTransport(uint3(ticketPixel(ticket), 0u), true, ticket);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void traceDeferred(uint3 thread : SV_DispatchThreadID) {
    traceDeferredKernel(thread);
}

// Pass 2: run hit shading once, recording its visibility queries.
[shader("compute")]
[numthreads(64, 1, 1)]
void traceRecord(uint3 thread : SV_DispatchThreadID) {
    uint ticket = traceTicket(thread.x);
    if (ticket == EMPTY || ticket >= deferredTraceCapacity(params)
        || !deferredTraceAvailable()) return;
    [loop] for (uint variant = 0u; variant < 2u; ++variant) {
        uint slot = ticket * 2u + variant;
        float3 surfacePosition, rayOrigin, direction;
        Hit hit;
        DependencyContext rayDependency;
        if (!loadDeferredSlot(slot, surfacePosition, rayOrigin, direction, hit,
                              rayDependency)) continue;
        visibilityMode = VISIBILITY_RECORD;
        visibilitySlot = slot;
        visibilityQueries = 0u;
        bool recordedSupported;
        DependencyContext recordedDependency = dependencyEmpty();
        outgoingTransportRadianceTracked(rayOrigin, direction, hit, recordedSupported,
            recordedDependency, mergeVisibilityAdmissionLimit());
        visibilityMode = VISIBILITY_TRACE;
        uint queries = min(visibilityQueries, DEFERRED_VISIBILITY_QUERIES);
        if (queries == 0u) continue;
        uint base = heads[deferredQueryCountHead(params)].add(queries);
        [loop] for (uint k = 0u; k < queries; ++k)
            heads[deferredQueryListHead(params) + base + k].store(
                slot * DEFERRED_VISIBILITY_QUERIES + k);
        heads[deferredQueryArgsHead(params)].max(
            (base + queries - 1u) * traceLaneStride(params) / 64u + 1u);
    }
}

// Pass 2: every recorded visibility query in its own lane. This kernel holds
// only shadow traversal, so it runs with far fewer registers than hit shading.
[ForceInline]
void traceVisibilityKernel(uint3 thread) {
    uint stride = traceLaneStride(params);
    if (thread.x % stride != 0u || !deferredTraceAvailable()) return;
    uint index = thread.x / stride;
    if (index >= heads[deferredQueryCountHead(params)].load()) return;
    traceVisibilityQuery(heads[deferredQueryListHead(params) + index].load());
}

[shader("compute")]
[numthreads(64, 1, 1)]
void traceVisibility(uint3 thread : SV_DispatchThreadID) {
    traceVisibilityKernel(thread);
}

#ifdef PENUMBRA_HARDWARE_RAY_QUERY
// The same two traversal-only kernels, compiled with hardware ray queries.
// Hit shading, recording and replay stay on the software kernels.
[shader("compute")]
[numthreads(64, 1, 1)]
void traceDeferredHardware(uint3 thread : SV_DispatchThreadID) {
    traceDeferredKernel(thread);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void traceVisibilityHardware(uint3 thread : SV_DispatchThreadID) {
    traceVisibilityKernel(thread);
}
#endif

// Pass 3: replay hit shading with the stored answers, then deposit.
[shader("compute")]
[numthreads(64, 1, 1)]
void traceResolve(uint3 thread : SV_DispatchThreadID) {
    uint ticket = traceTicket(thread.x);
    if (ticket == EMPTY || ticket >= deferredTraceCapacity(params)
        || !deferredTraceAvailable()) return;
    uint2 pixel = ticketPixel(ticket);
    uint sample = pixel.x + pixel.y * params.size.x;
    [loop] for (uint variant = 0u; variant < 2u; ++variant) {
        uint slot = ticket * 2u + variant;
        float3 surfacePosition, rayOrigin, direction;
        Hit hit;
        DependencyContext rayDependency;
        if (!loadDeferredSlot(slot, surfacePosition, rayOrigin, direction, hit,
                              rayDependency)) continue;
        uint word = deferredSlotHead(slot, params);
        uint first = probeDescriptorIndex(sample, variant, 0u);
        uint lod = (uint(entries[first].key.w) & PROBE_LOD) >> 18u;
        visibilityMode = VISIBILITY_REPLAY;
        visibilitySlot = slot;
        visibilityQueries = 0u;
        finishTransportVariant(pixel, first, lod, surfacePosition, rayOrigin, direction,
            hit, rayDependency, asfloat(heads[word + 24u].load()),
            asfloat(heads[word + 25u].load()), asfloat(heads[word + 26u].load()));
        visibilityMode = VISIBILITY_TRACE;
    }
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
        float4 intervalSum = 0.0;
        float3 emissiveSum = 0.0;
        float emissiveCount = 0.0;
        float3 punctualSum = 0.0;
        float punctualCount = 0.0;
        DependencyContext freshDependency = dependencyEmpty();
        float4 freshLightHitBounds = emptyLightHitBounds();
        float4 freshBetaSegmentSphere = emptyLightHitBounds();
        uint retainedIndex = EMPTY;
        [loop] while (cursor != EMPTY && all(entries[cursor].key == key)) {
            Entry sample = entries[cursor];
            uint next = sample.next;
            if (sample.samples > 0u) {
                intervalSum += sample.interval * float(sample.samples);
                freshCount += sample.samples;
                float sampleEmissiveCount = max(sample.emissiveInterval.w, 0.0);
                if (isfinite(sampleEmissiveCount) && sampleEmissiveCount > 0.0) {
                    emissiveSum += sample.emissiveInterval.xyz * sampleEmissiveCount;
                    emissiveCount += sampleEmissiveCount;
                }
                float samplePunctualCount = max(sample.punctualInterval.w, 0.0);
                if (isfinite(samplePunctualCount) && samplePunctualCount > 0.0) {
                    punctualSum += sample.punctualInterval.xyz * samplePunctualCount;
                    punctualCount += samplePunctualCount;
                }
                dependencyUnion(freshDependency, entryDependency(sample));
                freshLightHitBounds = unionConservativeSpheres(
                    freshLightHitBounds, sample.boundaryInterval);
                freshBetaSegmentSphere = unionConservativeSpheres(
                    freshBetaSegmentSphere, sample.normal);
            } else if (retainedIndex == EMPTY
                       && sample.valid == (params.frame.x | 0x80000000u)) {
                retainedIndex = cursor;
            }
            if (cursor != source) entries[cursor].valid = 0u;
            cursor = next;
        }
        uint nextGroup = cursor;
        if (freshCount > 0u) {
            Entry aggregate = firstEntry;
            aggregate.interval = intervalSum / float(freshCount);
            aggregate.emissiveInterval = emissiveCount > 0.0
                ? float4(emissiveSum / emissiveCount, emissiveCount) : 0.0;
            aggregate.punctualInterval = punctualCount > 0.0
                ? float4(punctualSum / punctualCount, punctualCount) : 0.0;
            aggregate.key = key;
            uint level = (uint(key.w) >> 12u) & 3u;
            uint lod = (uint(key.w) & PROBE_LOD) >> 18u;
            aggregate.origin = float4((float3(key.xyz) + 0.5)
                * spacing(params, level, lod), 1.0);
            aggregate.samples = freshCount;
            aggregate.valid = params.frame.x | 0x80000000u;
            aggregate.lastSample = params.frame.x;
            aggregate.merged = 0.0;
            aggregate.emissiveMerged = 0.0;
            aggregate.punctualMerged = 0.0;
            aggregate.next = unique;
            storeEntryDependency(aggregate, freshDependency);
            storeEntryMergedDependency(aggregate, dependencyUnknown());
            aggregate.boundaryInterval = freshLightHitBounds;
            aggregate.boundaryMerged = unknownLightHitBounds();
            aggregate.normal = freshBetaSegmentSphere;
            entries[source] = aggregate;
            unique = source;
        } else if (retainedIndex != EMPTY) {
            // The walk only cleared this slot's valid word, which is rewritten below.
            Entry retained = entries[retainedIndex];
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

[ForceInline]
float mergeRawRadianceGroup(inout Entry entry, Entry old, bool reuseOld,
                            uint source, float observedCount,
                            float radianceBlend, float radianceMassLimit,
                            inout DependencyContext rawDependency,
                            inout float4 rawLightHitBounds) {
    float4 current = entryRadianceInterval(entry, source);
    float4 prior = entryRadianceInterval(old, source);
    if (!transportSourceActive(params, source)) {
        if (source != BACKGROUND_SOURCE)
            setEntryRadianceInterval(entry, source, 0.0);
        return 0.0;
    }
    float incomingCount = observedCount > 0.0
        ? (source == BACKGROUND_SOURCE ? observedCount : current.w) : 0.0;
    if (!isfinite(incomingCount) || incomingCount < 0.0) incomingCount = 0.0;
    float oldMass = source == BACKGROUND_SOURCE ? asfloat(old.samples) : prior.w;
    if (!isfinite(oldMass) || oldMass < 0.0) oldMass = 0.0;
    float mass = 0.0;
    if (incomingCount > 0.0) {
        float retention = pow(radianceBlend, incomingCount);
        mass = (1.0 - retention) * radianceMassLimit;
        oldMass = min(oldMass, radianceMassLimit) * retention;
        if (reuseOld && oldMass > 0.0) {
            current.xyz = (current.xyz * mass + prior.xyz * oldMass)
                / (mass + oldMass);
            mass += oldMass;
            dependencyUnion(rawDependency, entryDependency(old));
            rawLightHitBounds = unionConservativeSpheres(
                rawLightHitBounds, old.boundaryInterval);
        }
    } else if (reuseOld && oldMass > 0.0) {
        current.xyz = prior.xyz;
        mass = oldMass;
        dependencyUnion(rawDependency, entryDependency(old));
        rawLightHitBounds = unionConservativeSpheres(
            rawLightHitBounds, old.boundaryInterval);
    } else {
        current.xyz = 0.0;
    }
    current.w = source == BACKGROUND_SOURCE ? current.w : mass;
    setEntryRadianceInterval(entry, source, current);
    return mass;
}

struct MergeGroup {
    float3 radiance;
    float support;
    float4 lightHitBounds;
    float3 childSum;
    float childSupport;
    bool active;
};

[ForceInline]
MergeGroup beginMergeGroup(Entry entry, uint source, float transmittance,
                           float transmittanceMass, float radianceMass) {
    MergeGroup group = (MergeGroup)0;
    group.active = transportSourceActive(params, source);
    float4 raw = entryRadianceInterval(entry, source);
    if (radianceMass > 0.0 && transmittanceMass > 0.0) {
        group.support = 1.0 - transmittance;
        if (group.support > 0.0) group.radiance = raw.xyz;
    }
    group.lightHitBounds = group.support > 0.0
        ? entry.boundaryInterval : emptyLightHitBounds();
    return group;
}

[ForceInline]
void addMergeChild(inout MergeGroup group, float4 value, float4 childLightHitBounds) {
    if (!group.active || !(value.w > 0.0)) return;
    group.childSum += value.xyz * value.w;
    group.childSupport += value.w;
    group.lightHitBounds = unionConservativeSpheres(group.lightHitBounds, childLightHitBounds);
}

[ForceInline]
void finishMergeGroup(inout Entry entry, uint source, float transmittance,
                      MergeGroup group, inout float4 mergedLightHitBounds) {
    if (!group.active) {
        setEntryRadianceMerged(entry, source, float4(0.0, 0.0, 0.0, 1.0));
        return;
    }
    float3 radiance = group.radiance;
    float support = group.support;
    if (group.childSupport > 0.0) {
        radiance += transmittance * group.childSum * 0.25;
        support += transmittance * group.childSupport * 0.25;
    }
    float4 merged = support > 0.0
        ? float4(radiance / support, saturate(support)) : 0.0;
    setEntryRadianceMerged(entry, source, merged);
    mergedLightHitBounds = unionConservativeSpheres(
        mergedLightHitBounds, group.lightHitBounds);
}

// Merges all three source groups. The four child directions are interpolated
// once for every active source, so each child probe is found once rather than
// once per source.
void mergeRadianceGroups(inout Entry entry, uint level, float transmittance,
                         float transmittanceMass, float backgroundMass,
                         float emissiveMass, float punctualMass,
                         inout DependencyContext mergedDependency,
                         inout float4 mergedLightHitBounds) {
    MergeGroup background = beginMergeGroup(entry, BACKGROUND_SOURCE, transmittance,
        transmittanceMass, backgroundMass);
    MergeGroup emissive = beginMergeGroup(entry, EMISSIVE_SOURCE, transmittance,
        transmittanceMass, emissiveMass);
    MergeGroup punctual = beginMergeGroup(entry, PUNCTUAL_SOURCE, transmittance,
        transmittanceMass, punctualMass);
    int4 key = entry.key;
    if (transmittance > 0.0 && level + 1u < CASCADES) {
        float3 position = entry.origin.xyz;
        uint direction = uint(key.w) & 4095u;
        uint theta = BASE_THETA << level;
        uint2 xy = uint2(direction % (2u * theta), direction / (2u * theta));
        uint lod = (uint(key.w) & PROBE_LOD) >> 18u;
        float3 grid = position / spacing(params, level + 1u, lod) - 0.5;
        [unroll] for (uint child = 0u; child < 4u; ++child) {
            uint2 fine = xy * 2u + uint2(child & 1u, child >> 1u);
            uint fineDirection = fine.x + fine.y * (4u * theta);
            DependencyContext childDependency = dependencyEmpty();
            float4 backgroundValue, emissiveValue, punctualValue;
            float4 backgroundBounds = emptyLightHitBounds();
            float4 emissiveBounds = emptyLightHitBounds();
            float4 punctualBounds = emptyLightHitBounds();
            interpolateMergedChildSources(int3(floor(grid)), frac(grid),
                fineDirection, level + 1u, lod,
                BACKGROUND_SOURCE | EMISSIVE_SOURCE | PUNCTUAL_SOURCE,
                childDependency, backgroundValue, emissiveValue, punctualValue,
                backgroundBounds, emissiveBounds, punctualBounds);
            dependencyUnion(mergedDependency, childDependency);
            addMergeChild(background, backgroundValue, backgroundBounds);
            addMergeChild(emissive, emissiveValue, emissiveBounds);
            addMergeChild(punctual, punctualValue, punctualBounds);
        }
    }
    finishMergeGroup(entry, BACKGROUND_SOURCE, transmittance, background,
        mergedLightHitBounds);
    finishMergeGroup(entry, EMISSIVE_SOURCE, transmittance, emissive,
        mergedLightHitBounds);
    finishMergeGroup(entry, PUNCTUAL_SOURCE, transmittance, punctual,
        mergedLightHitBounds);
}

void mergeEntry(uint index) {
    uint level = params.frame.y;
    Entry entry = entries[index];
    int4 key = entry.key;
    if (((uint(key.w) >> 12u) & 3u) != level) return;

    float samples = float(entry.samples);
    float radianceBlend = min(saturate(params.history.x), 0.999);
    float transmittanceBlend = min(saturate(params.intervalHistory.x), 0.999);
    float radianceMassLimit = 1.0 / (1.0 - radianceBlend);
    float transmittanceMassLimit = 1.0 / (1.0 - transmittanceBlend);
    float transmittanceRetention = samples > 0.0
        ? pow(transmittanceBlend, samples) : 1.0;
    float transmittanceSamples = samples > 0.0
        ? (1.0 - transmittanceRetention) * transmittanceMassLimit : 0.0;
    DependencyContext rawDependency = entryDependency(entry);
    float4 rawLightHitBounds = entry.boundaryInterval;
    float4 betaSegmentSphere = entry.normal;
    uint previous = (params.history.x > 0.0 || params.intervalHistory.x > 0.0)
        && params.history.z > 0.0 && params.history.w > 0.0
        ? findEntry(key, true) : EMPTY;
    Entry old = (Entry)0;
    bool reuseBeta = false;
    bool reuseBackground = false;
    bool reuseEmissive = false;
    bool reusePunctual = false;
    float oldTransmittanceSamples = 0.0;
    if (previous != EMPTY) {
        old = previousEntries[previous];
        bool oldAlive = intervalHistoryAlive(old.lastSample, params);
        bool oldBetaGeometryValid = oldAlive
            && betaHistoryEligible(entryDependency(old), old.normal);
        oldTransmittanceSamples = old.origin.w;
        if (!isfinite(oldTransmittanceSamples) || oldTransmittanceSamples < 0.0)
            oldTransmittanceSamples = 0.0;
        reuseBeta = oldBetaGeometryValid && params.intervalHistory.x > 0.0;
        DependencyContext oldDependency = entryDependency(old);
        bool oldRadianceGeometryValid = oldAlive && !dirtyRawCacheEntry(old);
        reuseBackground = oldRadianceGeometryValid && params.history.x > 0.0
            && backgroundMotionHistoryAlive(old.lastSample)
            && radianceHistoryEligible(oldDependency, old.boundaryInterval,
                BACKGROUND_SOURCE);
        reuseEmissive = oldRadianceGeometryValid && params.history.x > 0.0
            && radianceHistoryEligible(oldDependency, old.boundaryInterval,
                EMISSIVE_SOURCE);
        reusePunctual = oldRadianceGeometryValid && params.history.x > 0.0
            && punctualMotionHistoryAlive(old.lastSample)
            && radianceHistoryEligible(oldDependency, old.boundaryInterval,
                PUNCTUAL_SOURCE);
    }
    float backgroundBlend = (params.sourceHistory.z & 1u) != 0u
        ? min(radianceBlend, 0.9) : radianceBlend;
    float backgroundMass = mergeRawRadianceGroup(entry, old, reuseBackground,
        BACKGROUND_SOURCE, samples, backgroundBlend, 1.0 / (1.0 - backgroundBlend),
        rawDependency, rawLightHitBounds);
    float emissiveMass = mergeRawRadianceGroup(entry, old, reuseEmissive,
        EMISSIVE_SOURCE, samples, radianceBlend, radianceMassLimit,
        rawDependency, rawLightHitBounds);
    float punctualMass = mergeRawRadianceGroup(entry, old, reusePunctual,
        PUNCTUAL_SOURCE, samples, (params.sourceHistory.z & 4u) != 0u
            ? min(radianceBlend, 0.9) : radianceBlend,
        (params.sourceHistory.z & 4u) != 0u ? min(radianceMassLimit, 10.0) : radianceMassLimit,
        rawDependency, rawLightHitBounds);

    if (samples > 0.0) {
        oldTransmittanceSamples = min(oldTransmittanceSamples,
            transmittanceMassLimit) * transmittanceRetention;
        if (reuseBeta && oldTransmittanceSamples > 0.0) {
            entry.interval.w = (entry.interval.w * transmittanceSamples
                + old.interval.w * oldTransmittanceSamples)
                / (transmittanceSamples + oldTransmittanceSamples);
            transmittanceSamples += oldTransmittanceSamples;
            dependencyUnion(rawDependency, entryDependency(old));
            betaSegmentSphere = unionConservativeSpheres(
                betaSegmentSphere, old.normal);
        }
    } else if (reuseBeta && oldTransmittanceSamples > 0.0) {
        entry.interval.w = old.interval.w;
        transmittanceSamples = oldTransmittanceSamples;
        dependencyUnion(rawDependency, entryDependency(old));
        betaSegmentSphere = old.normal;
    } else {
        entry.interval.w = 1.0;
        transmittanceSamples = 0.0;
    }

    entry.boundaryInterval = rawLightHitBounds;
    entry.normal = betaSegmentSphere;
    storeEntryDependency(entry, rawDependency);
    entry.samples = asuint(backgroundMass);
    entry.origin.w = transmittanceSamples;
    float transmittance = transmittanceSamples > 0.0
        ? saturate(entry.interval.w) : 0.0;
    DependencyContext mergedDependency = rawDependency;
    float4 mergedLightHitBounds = emptyLightHitBounds();
    mergeRadianceGroups(entry, level, transmittance, transmittanceSamples,
        backgroundMass, emissiveMass, punctualMass, mergedDependency,
        mergedLightHitBounds);
    storeEntryMergedDependency(entry, mergedDependency);
    entry.boundaryMerged = mergedLightHitBounds;
    entries[index] = entry;
}
[shader("compute")]
[numthreads(64, 1, 1)]
void merge(uint3 thread : SV_DispatchThreadID) {
    uint index = thread.x * CASCADES + thread.y;
    // Seed, deposit and reduce store level-L keys only at current descriptors with index % CASCADES == L.
    if (index < currentCapacity() && thread.y != params.frame.y) return;
    if (index >= currentCapacity() + cacheHistoryQuota(params) * params.size.w) return;
    if (entries[index].valid != (params.frame.x | 0x80000000u)
        || !intervalHistoryAlive(entries[index].lastSample, params)
        || ((uint(entries[index].key.w) >> 12u) & 3u) != params.frame.y) return;
    // One thread per canonical entry avoids serial ray traversals along a bucket.
    // Retained physical offsets need not agree with the key's cascade level.
    mergeEntry(index);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void seedProbes(uint3 thread : SV_DispatchThreadID) {
    // Only level-0 keys seed probes, and descriptors keep level L at index % CASCADES == L.
    if (thread.y != 0u) return;
    uint index = thread.x * CASCADES;
    if (index >= currentCapacity()) return;
    // Include every initialized visible descriptor, including candidates with
    // zero prefix offsets; retained cache entries are read while packing radiance.
    bool initialized = entries[index].padding.y != 0u;
    bool traced = entries[index].valid == (params.frame.x | 0x80000000u);
    if (!initialized && !traced) return;
    int4 entryKey = entries[index].key;
    if (((uint(entryKey.w) >> 12u) & 3u) != 0u) return;
    int4 key = int4(entryKey.xyz, int(uint(entryKey.w) & PROBE_LOD));
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
            int4 otherKey = entries[candidate].key;
            if (all(otherKey.xyz == key.xyz)
                && (uint(otherKey.w) & PROBE_LOD) == uint(key.w)) return;
        }
        slot = (slot + 1u) & (directoryCapacity - 1u);
    }
    heads[2u * params.size.w + 2u].add(1u);
}

groupshared float4 packProbeBackground[32];
groupshared float4 packProbeEmissive[32];
groupshared float4 packProbePunctual[32];
groupshared uint packProbeSource[32];
groupshared uint packProbeActive;
groupshared uint packProbeSlot;
groupshared int4 packProbeKey;
groupshared float3 packProbeDirection[32];

[shader("compute")]
[numthreads(64, 1, 1)]
void packProbes(uint3 groupId : SV_GroupID, uint lane : SV_GroupIndex) {
    uint ordinal = groupId.x + 256u * groupId.y;
    if (lane == 0u) {
        uint slot = EMPTY;
        int4 key = 0;
        bool active = ordinal < currentCapacity() / CASCADES;
        if (active) {
            uint index = ordinal * CASCADES;
            key = entries[index].key;
            slot = findProbeNodeSlot(key);
            active = slot != EMPTY
                && heads[probeDirectoryHead(0u, slot, params)].load() == index;
        }
        packProbeActive = active ? 1u : 0u;
        packProbeSlot = slot;
        packProbeKey = key;
    }
    GroupMemoryBarrierWithGroupSync();
    // The decision is group-uniform, including at the next barrier.
    if (packProbeActive == 0u) return;

    if (lane < 32u) {
        uint source = EMPTY;
        float4 background = 0.0;
        float4 emissive = transportSourceActive(params, EMISSIVE_SOURCE)
            ? 0.0 : float4(0.0, 0.0, 0.0, 1.0);
        float4 punctual = transportSourceActive(params, PUNCTUAL_SOURCE)
            ? 0.0 : float4(0.0, 0.0, 0.0, 1.0);
        packProbeDirection[lane] = binDirection(lane, 0u);
        {
            uint lodBits = uint(packProbeKey.w) & PROBE_LOD;
            int4 directionKey = int4(packProbeKey.xyz, int(lane | lodBits));
            source = findEntry(directionKey, false);
            if (source != EMPTY) {
                background = entries[source].merged;
                if (transportSourceActive(params, EMISSIVE_SOURCE))
                    emissive = entries[source].emissiveMerged;
                if (transportSourceActive(params, PUNCTUAL_SOURCE))
                    punctual = entries[source].punctualMerged;
            }
        }
        // Validate once per direction, not once per output texel. Unsupported
        // and non-finite values contribute neither energy nor dependency bounds.
        if (!(background.w > 0.0) || !all(isfinite(background))) background = 0.0;
        if (!(emissive.w > 0.0) || !all(isfinite(emissive))) emissive = 0.0;
        if (!(punctual.w > 0.0) || !all(isfinite(punctual))) punctual = 0.0;
        packProbeSource[lane] = source;
        packProbeBackground[lane] = background;
        packProbeEmissive[lane] = emissive;
        packProbePunctual[lane] = punctual;
    }
    GroupMemoryBarrierWithGroupSync();

    uint directoryCapacity = probeDirectoryCapacity(params);
    uint output = ((params.frame.x & 1u)
        + (params.environmentScale.z > 0.5 ? 2u : 0u))
        * directoryCapacity + packProbeSlot;
    if (lane < 36u) {
        uint x = lane % 6u;
        uint y = lane / 6u;
        float3 normal = decodeNormal((float2(x, y) + 0.5) / 6.0);
        float3 backgroundSum = 0.0;
        float3 emissiveSum = 0.0;
        float3 punctualSum = 0.0;
        float nominalCosine = 0.0;
        float3 observedCosine = 0.0;
        [loop] for (uint direction = 0u; direction < 32u; ++direction) {
            float cosine = max(dot(normal, packProbeDirection[direction]), 0.0);
            nominalCosine += cosine;
            if (cosine <= 0.0) continue;
            float4 background = packProbeBackground[direction];
            if (background.w > 0.0) {
                float weight = cosine * saturate(background.w);
                backgroundSum += background.xyz * weight;
                observedCosine.x += weight;
            }
            float4 emissive = packProbeEmissive[direction];
            if (emissive.w > 0.0) {
                float weight = cosine * saturate(emissive.w);
                emissiveSum += emissive.xyz * weight;
                observedCosine.y += weight;
            }
            float4 punctual = packProbePunctual[direction];
            if (punctual.w > 0.0) {
                float weight = cosine * saturate(punctual.w);
                punctualSum += punctual.xyz * weight;
                observedCosine.z += weight;
            }
        }
        float4 backgroundValue = 0.0;
        float4 emissiveValue = 0.0;
        float4 punctualValue = 0.0;
        if (nominalCosine > 0.0) {
            float scale = nominalCosine * (4.0 / 32.0);
            if (observedCosine.x > 0.0)
                backgroundValue = float4(backgroundSum / observedCosine.x * scale,
                    saturate(observedCosine.x / nominalCosine));
            if (observedCosine.y > 0.0)
                emissiveValue = float4(emissiveSum / observedCosine.y * scale,
                    saturate(observedCosine.y / nominalCosine));
            if (observedCosine.z > 0.0)
                punctualValue = float4(punctualSum / observedCosine.z * scale,
                    saturate(observedCosine.z / nominalCosine));
        }
        directionalProbes[output].irradiance[lane] = backgroundValue;
        directionalProbes[output].emissiveIrradiance[lane] = emissiveValue;
        directionalProbes[output].punctualIrradiance[lane] = punctualValue;
    }
    if (lane == 0u) {
        DependencyContext packedDependency = dependencyEmpty();
        [loop] for (uint direction = 0u; direction < 32u; ++direction) {
            uint source = packProbeSource[direction];
            if (source == EMPTY) continue;
            bool supported = packProbeBackground[direction].w > 0.0;
            if (transportSourceActive(params, EMISSIVE_SOURCE))
                supported = supported || packProbeEmissive[direction].w > 0.0;
            if (transportSourceActive(params, PUNCTUAL_SOURCE))
                supported = supported || packProbePunctual[direction].w > 0.0;
            if (supported)
                dependencyUnion(packedDependency,
                    entryMergedDependency(entries[source]));
        }
        storeProbeDependency(directionalProbes[output], packedDependency);
        if (params.environmentScale.z > 0.5) {
            directionalProbes[output].cacheKey = int4(packProbeKey.xyz,
                int(uint(packProbeKey.w) & PROBE_LOD));
            directionalProbes[output].cacheGeneration = uint4(params.frame.x, 1u, 0u, 0u);
        }
        heads[2u * params.size.w + 9u].add(1);
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

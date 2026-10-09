uint gpuSortNext(uint index) { return entries[index].next; }
void gpuSortSetNext(uint index, uint next) { entries[index].next = next; }
bool gpuSortLess(uint left, uint right) {
    int4 a = entries[left].key, b = entries[right].key;
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    if (a.z != b.z) return a.z < b.z;
    if (a.w != b.w) return a.w < b.w;
    uint2 x = entries[left].identity, y = entries[right].identity;
    if (x.x != y.x) return x.x < y.x;
    if (x.y != y.y) return x.y < y.y;
    if (entries[left].padding.y != entries[right].padding.y)
        return entries[left].padding.y < entries[right].padding.y;
    return left < right;
}
#include "linked_merge_sort.slang"

float splitDistanceLod(float3 position) {
    // ponytail: four distance LODs, transition every doubling after 32 base
    // spacings; expose the transition distance if real scenes need tuning.
    return clamp(log2(max(length(position - params.cameraPosition.xyz) / (32 * params.tuning.x), 1.0)), 0.0, 3.0);
}

float4 findMerged(int4 key, bool previous, uint2 identity, uint surfaceGroup, out float3 origin) {
    origin = 0.0;
    if (previous && params.history.z == 0.0) return 0.0;
    uint bucket = (hashKey(key, params.size.w) ^ surfaceGroup) & (params.size.w - 1);
    uint index = previous ? previousHeads[bucket] : heads[bucket].load();
    [loop] while (index != EMPTY) {
        Entry entry = previous ? previousEntries[index] : entries[index];
        if (all(entry.key == key) && all(entry.identity == identity) && entry.padding.y == surfaceGroup) {
            origin = entry.origin.xyz;
            return entry.merged;
        }
        index = entry.next;
    }
    return 0.0;
}
float4 interpolate(float3 position, uint direction, uint level, bool previous, bool secondary, uint surface, uint2 identity, uint surfaceGroup) {
    float3 grid = position / spacing(params, level, secondary, (surface & PROBE_LOD) >> 18) - 0.5;
    int3 base = int3(floor(grid)); float3 fraction = frac(grid);
    float3 sum = 0.0; float total = 0.0;
    [loop] for (uint corner = 0; corner < 8; ++corner) {
        uint3 bit = uint3(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
        float3 weights = lerp(1.0 - fraction, fraction, float3(bit));
        float weight = weights.x * weights.y * weights.z;
        if (weight <= 0.00001) continue;
        int4 key = int4(base + int3(bit), int(direction | (level << 12) | (secondary ? SECONDARY : 0) | surface));
        float3 origin;
        float4 value = findMerged(key, previous, identity, surfaceGroup, origin);
        if (value.w > 0) {
            float3 delta = origin - position;
            float distance = length(delta);
            if (distance > params.tuning.z * 2 && traceOccluded(position, delta / distance, distance - params.tuning.z)) continue;
            sum += value.xyz * weight;
            total += weight;
        }
    }
    return total > 0.00001 ? float4(sum / total, 1.0) : float4(0.0);
}
// Returns irradiance / pi for the Lambertian BRDF, without the surface albedo.
float4 diffuseGatherExact(float3 position, float3 normal, bool previous, bool secondary, uint2 identity, uint surfaceGroup, uint lod) {
    float3 sum = 0.0; float total = 0.0;
    [loop] for (uint direction = 0; direction < 32; ++direction) {
        float cosine = max(0.0, dot(normal, binDirection(direction, 0)));
        if (cosine <= 0.0) continue;
        float4 value = interpolate(position, direction, 0, previous, secondary, surfaceKey(normal) | (lod << 18), identity, surfaceGroup);
        if (value.w > 0) { sum += value.xyz * cosine; total += cosine; }
    }
    return total > 0.0 ? float4(sum / total, 1) : float4(0);
}

uint findIrradianceProbe(int4 key, bool previous, uint2 identity, uint surfaceGroup) {
    if (previous && params.history.z == 0) return EMPTY;
    uint slot = (hashKey(key, params.size.w) ^ surfaceGroup) & (params.size.w - 1);
    [loop] for (uint attempt = 0; attempt < 32; ++attempt) {
        uint index = previous ? previousHeads[params.size.w + 1 + slot] : heads[params.size.w + 1 + slot].load();
        if (index == EMPTY) return EMPTY;
        if (index != TOMBSTONE) {
            int4 candidate = previous ? previousEntries[index].key : entries[index].key;
            uint2 candidateIdentity = previous ? previousEntries[index].identity : entries[index].identity;
            uint candidateSurfaceGroup = previous ? previousEntries[index].padding.y : entries[index].padding.y;
            if (all(candidate.xyz == key.xyz) && (uint(candidate.w) & (SECONDARY | SURFACE | PROBE_LOD)) == uint(key.w) && all(candidateIdentity == identity) && candidateSurfaceGroup == surfaceGroup) return slot;
        }
        slot = (slot + 1) & (params.size.w - 1);
    }
    return EMPTY;
}

float4 diffuseGatherLod(float3 position, float3 normal, bool previous, bool secondary, uint2 identity, uint surfaceGroup, uint lod) {
    if (previous && params.history.z == 0) return 0;
    float3 grid = position / spacing(params, 0, secondary, lod) - 0.5;
    int3 base = int3(floor(grid)); float3 fraction = frac(grid);
    float3 sum = 0; float total = 0;
    uint frame = params.frame.x + (previous ? 1 : 0);
    [loop] for (uint corner = 0; corner < 8; ++corner) {
        int3 bit = int3(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
        float3 weights = lerp(1 - fraction, fraction, float3(bit));
        float weight = weights.x * weights.y * weights.z;
        if (weight <= 0.00001) continue;
        uint probe = findIrradianceProbe(int4(base + bit, int((secondary ? SECONDARY : 0) | surfaceKey(normal) | (lod << 18))), previous, identity, surfaceGroup);
        if (probe == EMPTY) continue;
        uint probeIndex = (frame & 1) * params.size.w + probe;
        // Probes live on actual surface samples. Reject reuse through a wall
        // instead of interpolating from arbitrary cell centers inside solids.
        float3 delta = irradianceProbes[probeIndex].origin.xyz - position;
        float distance = length(delta);
        if (distance > params.tuning.z * 2 && traceOccluded(position, delta / distance, distance - params.tuning.z)) {
            continue;
        }
        float2 oct = encodeNormal(normal) * float(IRRADIANCE_SIDE) + 0.5;
        int2 texel = int2(floor(oct)); float2 blend = frac(oct);
        float4 diffuse = 0;
        [unroll] for (uint tap = 0; tap < 4; ++tap) {
            int2 bit = int2(tap & 1, tap >> 1);
            float2 weights = lerp(1 - blend, blend, float2(bit));
            int2 p = texel + bit;
            diffuse += irradianceProbes[probeIndex].coefficients[irradianceOctIndex(p)] * weights.x * weights.y;
        }
        if (diffuse.w <= 0.00001) continue;
        sum += diffuse.xyz / diffuse.w * weight; total += weight;
    }
    // Sparse interpolation ignores absent probes (paper Section 4). Repeating
    // a 32-direction, eight-corner ray-tested gather here defeated prefiltering
    // and multiplied the driver's inlined traversal graph during cold startup.
    return total > 0.00001 ? float4(sum / total, 1) : float4(0);
}
float3 diffuseGather(float3 position, float3 normal, bool previous, bool secondary, uint2 identity, uint surfaceGroup) {
    float level = splitDistanceLod(position);
    uint low = uint(level), high = min(low + 1, 3u);
    float4 value = 0;
    [loop] for (uint lod = low; lod <= high; ++lod) {
        float weight = low == high ? 1.0 : (lod == low ? 1.0 - frac(level) : frac(level));
        if (weight > 0)
            value += weight * diffuseGatherLod(position, normal, previous, secondary, identity, surfaceGroup, lod);
    }
    return value.w > 0 ? value.xyz / value.w : float3(0);
}
[shader("compute")]
[numthreads(64, 1, 1)]
void clear(uint3 thread : SV_DispatchThreadID) {
    if (thread.x < params.size.w) heads[thread.x].store(EMPTY);
    if (thread.x < params.size.w) heads[params.size.w + 1 + thread.x].store(EMPTY);
    if (thread.x == 0) {
        heads[params.size.w].store(0);
        // The irradiance heads include index 2*size.w; counters start after it.
        heads[2 * params.size.w + 1].store(0); // dropped retained intervals
        heads[2 * params.size.w + 2].store(0); // rejected irradiance insertions
        heads[2 * params.size.w + 3].store(0); // entries colliding in prior buckets
        heads[2 * params.size.w + 4].store(0); // persistent cache overflow
    }
    // Entries unreachable from a hash head are never read. Avoid clearing the
    // large interval allocation; trace and retain initialize reachable nodes.
}

uint findEntry(int4 key, uint2 identity, uint surfaceGroup, bool previous = false) {
    if (previous && params.history.z == 0) return EMPTY;
    uint bucket = (hashKey(key, params.size.w) ^ surfaceGroup) & (params.size.w - 1);
    uint index = previous ? previousHeads[bucket] : heads[bucket].load();
    [loop] while (index != EMPTY) {
        Entry entry = previous ? previousEntries[index] : entries[index];
        if (all(entry.key == key) && all(entry.identity == identity) && entry.padding.y == surfaceGroup) break;
        index = entry.next;
    }
    return index;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void seed(uint3 thread : SV_DispatchThreadID) {
    if (any(thread.xy >= params.size.xy)) return;
    uint sample = thread.x + thread.y * params.size.x;
    surfaceGroupOutput.Store(int2(thread.xy), EMPTY);
    [loop] for (uint level = 0; level < CASCADES; ++level) entries[sample * CASCADES + level].valid = 0;
    entries[sample * CASCADES].samples = 0;
    uint2 full = fullPixel(thread.xy, params);
    float depth = depthTexture.Load(int3(full, 0));
    float3 normal = normalTexture.Load(int3(full, 0)).xyz * 2 - 1;
    uint4 receiver = rasterReceiver.Load(int3(full, 0));
    if (depth <= 0 || receiver.z == 0 || dot(normal, normal) < 0.01) return;
    float2 encodedNormal = float2(float(receiver.z & 0x7fffu), float((receiver.z >> 15) & 0x7fffu))
        / 32767.0;
    float3 geometricNormal = decodeNormal(encodedNormal);
    if (dot(geometricNormal, normal) < 0) geometricNormal = -geometricNormal;
    float3 surfacePosition = worldPosition(float2(full) + 0.5, depth, params);
    float3 near = worldPosition(float2(full) + 0.5, 1, params);
    float distance = length(surfacePosition - near);
    // Keep an unmatched raster sample isolated instead of borrowing another surface's cache.
    uint fallbackOffset = hashKey(int4(int(params.frame.x),0,0,0),0x7fffffffu);
    uint surfaceGroup = 0x80000000u | ((sample + fallbackOffset) % 0x7fffffffu);
    if (distance > 0 && isfinite(distance)) {
        Hit receiverHit = traceRay(near, (surfacePosition - near) / distance, distance + params.tuning.z);
        if (receiverHit.triangle != EMPTY
            && abs(receiverHit.distance - distance) <= max(params.tuning.z, distance * 0.00001)
            && all(hitInstanceIdentity(receiverHit) == receiver.xy))
            surfaceGroup = triangleSurfaceGroup(receiverHit.triangle);
    }
    surfaceGroupOutput.Store(int2(thread.xy), surfaceGroup);
    float3 position = rayOffset(surfacePosition, geometricNormal, params.tuning.z);
    uint index = sample * CASCADES;
    Entry entry = (Entry)0;
    entry.identity = receiver.xy;
    entry.padding.y = surfaceGroup;
    entry.merged = float4(geometricNormal, 0); // receiver normal, consumed before trace deposit
    entry.key = cacheKey(position, 0, 0, params, false, surfaceKey(normal), uint(splitDistanceLod(position))); entry.samples = 1; entry.valid = 1;
    entry.next = heads[(hashKey(entry.key, params.size.w) ^ surfaceGroup) & (params.size.w - 1)].exchange(index);
    entries[index] = entry;
}

// Count parents from unique children, not from every pixel. This bounds the
// serial chain work in coarse probes that cover thousands of screen pixels.
[shader("compute")]
[numthreads(64, 1, 1)]
void parents(uint3 thread : SV_DispatchThreadID) {
    if (thread.x >= params.size.z) return;
    uint index = thread.x * CASCADES + params.frame.y;
    Entry child = entries[index - 1];
    if (child.valid == 0) return;
    Entry entry = (Entry)0;
    entry.identity = child.identity;
    entry.key = int4(child.key.xyz >> 1, int((params.frame.y << 12) | (uint(child.key.w) & (SURFACE | PROBE_LOD))));
    entry.samples = child.samples; entry.valid = 1;
    entry.next = heads[hashKey(entry.key, params.size.w)].exchange(index);
    entries[index] = entry;
}

// Algorithm 3: count rays per probe and assign a local ordinal. The order of
// pixels within one probe is immaterial; its R2 segment is contiguous.
[shader("compute")]
[numthreads(64, 1, 1)]
void rank(uint3 thread : SV_DispatchThreadID) {
    if (thread.x >= params.size.w) return;
    uint source = gpuSortLinkedList(heads[thread.x].load()), unique = EMPTY;
    [loop] while (source != EMPTY) {
        Entry sample = entries[source]; uint next = sample.next;
        uint match = unique != EMPTY && all(entries[unique].key == sample.key) && all(entries[unique].identity == sample.identity) && entries[unique].padding.y == sample.padding.y ? unique : EMPTY;
        if (match == EMPTY) {
            sample.next = unique; sample.lastSample = 0; sample.valid = 1;
            entries[source] = sample; unique = source;
        } else {
            entries[source].lastSample = entries[match].samples;
            entries[match].samples += sample.samples;
            entries[source].valid = 0;
        }
        source = next;
    }
    heads[thread.x].store(unique);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void prefix(uint3 thread : SV_DispatchThreadID) {
    if (thread.x >= params.size.w) return;
    uint level = params.frame.y;
    [loop] for (uint index = heads[thread.x].load(); index != EMPTY; index = entries[index].next) {
        int4 key = entries[index].key;
        if ((uint(key.w) >> 12 & 3u) != level) continue;
        // Independent roots may use independent sequence phases. Descendants
        // share contiguous segments at every cascade, including negative cells.
        uint offset = hashKey(key, 65536);
        if (level + 1 < CASCADES) {
            int3 parent = key.xyz >> 1;
            uint parentIndex = findEntry(int4(parent, int(((level + 1) << 12) | (uint(key.w) & (SURFACE | PROBE_LOD)))), entries[index].identity, 0);
            offset = parentIndex == EMPTY ? 0 : entries[parentIndex].valid;
            int3 local = key.xyz - parent * 2;
            uint ordinal = uint(local.x + local.y * 2 + local.z * 4);
            [loop] for (uint sibling = 0; sibling < ordinal; ++sibling) {
                int3 bit = int3(sibling & 1, (sibling >> 1) & 1, (sibling >> 2) & 1);
                uint child = findEntry(int4(parent * 2 + bit, int((level << 12) | (uint(key.w) & (SURFACE | PROBE_LOD)))), entries[index].identity, level == 0 ? entries[index].padding.y : 0);
                if (child != EMPTY) offset += entries[child].samples;
            }
        }
        entries[index].valid = offset;
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void generateDirections(uint3 thread : SV_DispatchThreadID) {
    if (any(thread.xy >= params.size.xy)) return;
    uint2 full = fullPixel(thread.xy, params);
    if (depthTexture.Load(int3(full, 0)) <= 0) return;
    uint index = (thread.x + thread.y * params.size.x) * CASCADES;
    if (entries[index].samples == 0) return;
    uint probe = findEntry(entries[index].key, entries[index].identity, entries[index].padding.y);
    entries[index].lastSample = probe == EMPTY ? 0 : entries[probe].valid + entries[index].lastSample;
}

uint currentCapacity() {
    return params.size.z * CASCADES * (params.history.y > 0 ? 4 : 2);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void retain(uint3 thread : SV_DispatchThreadID) {
    if (thread.x >= params.size.w || params.history.z == 0 || params.history.x == 0 || params.history.w == 0) return;
    uint capacity = currentCapacity();
    uint historyCapacity = 2 * params.size.w;
    // Previous chains are canonical key/identity order from reduce. Fixed quotas keep
    // overflow deterministic without a prefix-sum pass; hash skew can leave slots unused.
    // shortcut: two slots per bucket waste capacity under hash skew; use prefix compaction when drops matter.
    uint bucketQuota = historyCapacity / params.size.w + uint(thread.x < historyCapacity % params.size.w);
    uint bucketBase = thread.x * (historyCapacity / params.size.w) + min(thread.x, historyCapacity % params.size.w);
    uint bucketRetained = 0;
    uint bucketLoad = 0;
    [loop] for (uint index = previousHeads[thread.x]; index != EMPTY; index = previousEntries[index].next) {
        Entry entry = previousEntries[index];
        bucketLoad++;
        if (params.frame.x - entry.lastSample > uint(params.history.w)) continue;
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

void deposit(uint first, float3 position, float3 normal, float3 direction, Hit hit, float3 radiance, bool secondary, uint2 identity, uint surfaceGroup, uint lod) {
    [loop] for (uint level = 0; level < CASCADES; ++level) {
        uint index = first + level;
        float end = params.tuning.y * float(1u << (2u * level + lod)) * (secondary ? 4.0 : 1.0);
        bool terminates = hit.distance <= end || level == CASCADES - 1;
        Entry entry = (Entry)0;
        entry.identity = identity;
        entry.padding.y = level == 0 ? surfaceGroup : 0;
        entry.key = cacheKey(position, directionIndex(direction, level), level, params, secondary, surfaceKey(normal), lod);
        entry.interval = terminates ? float4(radiance, 0.0) : float4(0, 0, 0, 1);
        entry.merged = 0.0;
        entry.origin = float4(position, 1);
        // An epoch distinguishes live records from untouched physical slots.
        // Reduction clears duplicate records, leaving canonical entries stamped.
        entry.valid = params.frame.x | 0x80000000u;
        entry.lastSample = params.frame.x; entry.samples = 1;
        entry.next = heads[(hashKey(entry.key, params.size.w) ^ entry.padding.y) & (params.size.w - 1)].exchange(index);
        entries[index] = entry;
        // Do not deposit an occluded ray in higher cascades (paper Section 5).
        if (terminates) break;
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void trace(uint3 thread : SV_DispatchThreadID) {
    uint2 pixel = thread.xy;
    if (any(pixel >= params.size.xy)) return;
    uint sample = pixel.x + pixel.y * params.size.x;
    bool receiverValid = entries[sample * CASCADES].samples > 0;
    float3 geometricNormal = entries[sample * CASCADES].merged.xyz;
    uint rank = entries[sample * CASCADES].lastSample;
    [loop] for (uint level = 0; level < CASCADES; ++level) {
        entries[sample * CASCADES + level].valid = 0;
        if (params.history.y > 0) entries[(params.size.z + sample) * CASCADES + level].valid = 0;
    }
    uint2 full = fullPixel(pixel, params);
    float depth = depthTexture.Load(int3(full, 0));
    if (depth <= 0.0 || !receiverValid) return;
    float3 normal = normalTexture.Load(int3(full, 0)).xyz * 2.0 - 1.0;
    if (dot(normal, normal) < 0.01) return;
    normal = normalize(normal);
    float3 surfacePosition = worldPosition(float2(full) + 0.5, depth, params);
    float3 position = rayOffset(surfacePosition, geometricNormal, params.tuning.z);
    // Like Chandra's randomized frame rotation: adding R2(frame) to R2(rank)
    // only shifts the same sequence by one and repeats N-1 of a probe's N rays.
    float2 jitter = r2(hashKey(int4(int(params.frame.x), 29, 968, 0), 0x80000000u));
    float2 uv = frac(0.5 + r2(rank + 1) + jitter);
    float3 direction = decodeDirection(uv);
    if (dot(direction, normal) < 0) direction = -direction;
    // The last cascade covers the remainder of the ray. A finite cutoff
    // mistakes distant occluders for open sky and leaks environment lighting.
    float3 pixelNeighbor = worldPosition(float2(full) + 0.5 + float2(float(params.viewport.z) / float(params.size.x), 0), depth, params);
    float coneWidth = length(pixelNeighbor - surfacePosition);
    Hit hit = traceRay(position, direction, 1e20, false, coneWidth, 0.5);
    float3 radiance = outgoingRadiance(position, direction, hit);
    if (params.history.y > 0 && hit.triangle != EMPTY && any(worldTriangle(hit).albedo.xyz > 0)) {
        HitAttributes attributes = hitAttributes(hit);
        float3 hitNormal = hitShadingNormal(hit, attributes, direction);
        float3 geometricNormal = attributes.geometricNormal;
        if (dot(geometricNormal, direction) > 0) geometricNormal = -geometricNormal;
        float3 secondaryPosition = rayOffset(position + direction * hit.distance, geometricNormal, params.tuning.z);
        // Independent phase: a fixed rotation of the primary R2 sample couples
        // both path vertices and can miss an emitter across a whole moving frame.
        uint secondaryPhase = hashKey(int4(int(sample), int(params.frame.x), int(rank), 0), 0x80000000u);
        float3 secondaryDirection = decodeDirection(frac(0.5 + r2(secondaryPhase)));
        if (dot(secondaryDirection, geometricNormal) < 0) secondaryDirection = -secondaryDirection;
        // Reuse this ray's reserved secondary interval record as a descriptor.
        // The next pass consumes it before deposit replaces it with intervals.
        uint secondary = (params.size.z + sample) * CASCADES;
        entries[secondary].origin = float4(secondaryPosition, hit.footprint);
        entries[secondary].interval = float4(secondaryDirection, 0);
        entries[secondary].merged = float4(hitNormal, 0);
        entries[secondary].valid = 1;
        entries[secondary].identity = hitInstanceIdentity(hit);
        entries[secondary].padding.y = triangleSurfaceGroup(hit.triangle);
        // Emitter sampling at the primary hit already supplies a fresh reflected
        // bounce during motion. A second unfiltered path here adds third-bounce
        // flashes; secondary samples instead feed the next frame's probe cache.
    }
    uint2 identity = entries[sample * CASCADES].identity;
    uint surfaceGroup = entries[sample * CASCADES].padding.y;
    uint lod = uint(splitDistanceLod(position));
    deposit(sample * CASCADES, position, normal, direction, hit, radiance, false, identity, surfaceGroup, lod);
    uint transition = (params.size.z * (params.history.y > 0 ? 2 : 1) + sample) * CASCADES;
    if (lod < 3) deposit(transition, position, normal, direction, hit, radiance, false, identity, surfaceGroup, lod + 1);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void traceSecondary(uint3 thread : SV_DispatchThreadID) {
    if (params.history.y <= 0 || any(thread.xy >= params.size.xy)) return;
    uint first = (params.size.z + thread.x + thread.y * params.size.x) * CASCADES;
    Entry descriptor = entries[first];
    if (descriptor.valid == 0) return;
    entries[first].valid = 0;
    Hit hit = traceRay(descriptor.origin.xyz, descriptor.interval.xyz, 1e20,
                       false, descriptor.origin.w, 0.5);
    float3 radiance = outgoingRadiance(descriptor.origin.xyz, descriptor.interval.xyz, hit);
    uint lod = uint(splitDistanceLod(descriptor.origin.xyz));
    deposit(first, descriptor.origin.xyz, descriptor.merged.xyz, descriptor.interval.xyz, hit, radiance, true, descriptor.identity, descriptor.padding.y, lod);
    uint sample = thread.x + thread.y * params.size.x;
    uint transition = (params.size.z * 3 + sample) * CASCADES;
    if (lod < 3) deposit(transition, descriptor.origin.xyz, descriptor.merged.xyz, descriptor.interval.xyz, hit, radiance, true, descriptor.identity, descriptor.padding.y, lod + 1);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void reduce(uint3 thread : SV_DispatchThreadID) {
    uint bucket = thread.x;
    if (bucket >= params.size.w) return;
    uint source = gpuSortLinkedList(heads[bucket].load()), unique = EMPTY;
    // Sorting makes equal full keys contiguous, so reduction is a linear pass.
    [loop] while (source != EMPTY) {
        Entry sample = entries[source];
        uint next = sample.next;
        uint match = unique != EMPTY && all(entries[unique].key == sample.key) && all(entries[unique].identity == sample.identity) && entries[unique].padding.y == sample.padding.y ? unique : EMPTY;
        if (match == EMPTY) {
            sample.next = unique;
            entries[source] = sample;
            unique = source;
        } else {
            if (sample.samples > 0) {
                float3 center = (float3(sample.key.xyz) + 0.5) * spacing(params, uint(sample.key.w) >> 12 & 3u, (uint(sample.key.w) & SECONDARY) != 0, (uint(sample.key.w) & PROBE_LOD) >> 18);
                if (entries[match].samples == 0) {
                    entries[match].interval = 0; entries[match].origin = sample.origin;
                } else if (length(sample.origin.xyz - center) < length(entries[match].origin.xyz - center)) {
                    entries[match].origin = sample.origin;
                }
                entries[match].interval += sample.interval;
                entries[match].samples += sample.samples;
                entries[match].lastSample = params.frame.x;
            }
            entries[source].valid = 0;
        }
        source = next;
    }
    heads[bucket].store(unique);
    [loop] for (uint index = unique; index != EMPTY; index = entries[index].next) {
        if (entries[index].samples > 0) {
            entries[index].interval /= float(entries[index].samples);
        }
    }
}

void mergeEntry(uint index) {
    // Retained raw intervals must also recompose when coarse dependencies change.
    uint level = params.frame.y;
    int4 key = entries[index].key;
    if ((uint(key.w) >> 12 & 3u) != level) return;
    bool secondary = (uint(key.w) & SECONDARY) != 0;
    float4 interval = entries[index].interval;
    uint previous = params.history.x > 0 ? findEntry(key, entries[index].identity, entries[index].padding.y, true) : EMPTY;
    float samples = float(entries[index].samples);
    if (previous != EMPTY && entries[index].samples > 0) {
        Entry old = previousEntries[previous];
        // Chandra accumulates weighted fluence and decays its sample weight.
        // Accumulate J/beta before merging, so a one-ray update cannot carry
        // the same confidence as a whole probe's fresh samples.
        float oldSamples = old.origin.w * pow(params.history.x, float(params.frame.x - old.lastSample));
        interval = (interval * samples + old.interval * oldSamples) / (samples + oldSamples);
        samples += oldSamples;
    }
    entries[index].interval = interval;
    if (entries[index].samples > 0) entries[index].origin.w = samples;
    float3 radiance = interval.xyz;
    bool known = interval.w < 1.0;
    if (interval.w > 0.0 && level + 1 < CASCADES) {
        float3 position = entries[index].origin.xyz;
        uint direction = uint(key.w) & 4095u;
        uint theta = BASE_THETA << level;
        uint2 xy = uint2(direction % (2 * theta), direction / (2 * theta));
        uint fineDirections[4];
        [loop] for (uint child = 0; child < 4; ++child) {
            uint2 fine = xy * 2 + uint2(child & 1, child >> 1);
            fineDirections[child] = fine.x + fine.y * (4 * theta);
        }
        // Validate each reused angular endpoint, rather than one representative
        // origin whose visibility cannot prove visibility of sibling directions.
        float3 sum = 0.0; float count = 0;
        [loop] for (uint child = 0; child < 4; ++child) {
            float4 value = interpolate(position, fineDirections[child], level + 1, false, secondary, uint(key.w) & (SURFACE | PROBE_LOD), entries[index].identity, 0);
            if (value.w > 0) { sum += value.xyz; count += 1; }
        }
        if (count > 0) { radiance += interval.w * sum / count; known = true; }
    }
    // Separate compute passes provide storage barriers between cascade levels.
    entries[index].merged = float4(radiance, known ? 1.0 : 0.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void merge(uint3 thread : SV_DispatchThreadID) {
    uint index = thread.x * CASCADES + thread.y;
    if (index >= currentCapacity() + 2 * params.size.w) return;
    Entry entry = entries[index];
    if (entry.valid != (params.frame.x | 0x80000000u)
        || params.frame.x - entry.lastSample > uint(params.history.w)) return;
    // One thread per canonical entry avoids serial ray traversals along a bucket.
    // Retained physical offsets need not agree with the key's cascade level.
    mergeEntry(index);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void seedIrradiance(uint3 thread : SV_DispatchThreadID) {
    if (thread.x >= params.size.w) return;
    [loop] for (uint index = heads[thread.x].load(); index != EMPTY; index = entries[index].next) {
        int4 key = entries[index].key;
        if ((uint(key.w) >> 12 & 3u) != 0) continue;
        key.w = int(uint(key.w) & (SECONDARY | SURFACE | PROBE_LOD));
        uint surfaceGroup = entries[index].padding.y;
        uint slot = (hashKey(key, params.size.w) ^ surfaceGroup) & (params.size.w - 1);
        bool inserted = false;
        [loop] for (uint attempt = 0; attempt < 32; ++attempt) {
            uint candidate = heads[params.size.w + 1 + slot].compareExchange(EMPTY, index);
            if (candidate == EMPTY) { inserted = true; break; }
            int4 other = entries[candidate].key;
            if (all(other.xyz == key.xyz) && (uint(other.w) & (SECONDARY | SURFACE | PROBE_LOD)) == uint(key.w) && all(entries[candidate].identity == entries[index].identity) && entries[candidate].padding.y == surfaceGroup) { inserted = true; break; }
            slot = (slot + 1) & (params.size.w - 1);
        }
        if (!inserted) heads[2 * params.size.w + 2].add(1);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void prefilter(uint3 thread : SV_DispatchThreadID) {
    if (thread.x >= params.size.w) return;
    uint index = heads[params.size.w + 1 + thread.x].load();
    if (index == EMPTY) return;
    int4 key = entries[index].key;
    uint mode = uint(key.w) & (SECONDARY | SURFACE | PROBE_LOD);
    IrradianceProbe probe = (IrradianceProbe)0;
    uint output = (params.frame.x & 1) * params.size.w + thread.x;
    // ponytail: direct owned-probe writes avoid FXC local-array failures;
    // use a workgroup tile if bandwidth becomes a measured bottleneck.
    irradianceProbes[output] = probe;
    float3 center = (float3(key.xyz) + 0.5) * spacing(params, 0, (mode & SECONDARY) != 0, (mode & PROBE_LOD) >> 18);
    [loop] for (uint direction = 0; direction < 32; ++direction) {
        float3 origin;
        float4 radiance = findMerged(int4(key.xyz, int(direction | mode)), false, entries[index].identity, entries[index].padding.y, origin);
        if (radiance.w == 0) continue;
        float3 n = binDirection(direction, 0);
        [loop] for (uint i = 0; i < IRRADIANCE_SIDE * IRRADIANCE_SIDE; ++i) {
            float3 receiver = decodeNormal(irradianceOctUv(int2(i % IRRADIANCE_SIDE + 1, i / IRRADIANCE_SIDE + 1)));
            irradianceProbes[output].coefficients[i] += radiance * max(0.0, dot(n, receiver));
        }
        uint source = findEntry(int4(key.xyz, int(direction | mode)), entries[index].identity, entries[index].padding.y);
        if (source != EMPTY && (probe.origin.w == 0 || length(entries[source].origin.xyz - center) < length(probe.origin.xyz - center)))
            probe.origin = entries[source].origin;
    }
    [loop] for (uint i = 0; i < IRRADIANCE_SIDE * IRRADIANCE_SIDE; ++i) {
        float4 coefficient = irradianceProbes[output].coefficients[i];
        if (coefficient.w > 0) irradianceProbes[output].coefficients[i] = float4(coefficient.xyz / coefficient.w, 1);
    }
    irradianceProbes[output].origin = probe.origin;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void gather(uint3 thread : SV_DispatchThreadID) {
    uint2 pixel = thread.xy;
    if (any(pixel >= params.size.xy)) return;
    uint2 full = fullPixel(pixel, params);
    float depth = depthTexture.Load(int3(full, 0));
    float3 normal = normalTexture.Load(int3(full, 0)).xyz * 2.0 - 1.0;
    if (depth <= 0.0 || dot(normal, normal) < 0.01) { irradiance.Store(int2(pixel), float4(0.0)); return; }
    normal = normalize(normal);
    uint receiver = (pixel.x + pixel.y * params.size.x) * CASCADES;
    if (entries[receiver].samples == 0) { irradiance.Store(int2(pixel), 0); return; }
    float3 position = entries[receiver].origin.xyz;
    float3 result = diffuseGather(position, normal, false, false, entries[receiver].identity, entries[receiver].padding.y)
                   + emissiveIrradiance(worldPosition(float2(full) + 0.5, depth, params), normal, params.lightCounts.z);
    irradiance.Store(int2(pixel), float4(result, depth));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void commitCache(uint3 thread : SV_DispatchThreadID) {
    uint bucket = thread.x;
    if (bucket >= params.size.w) return;
    uint capacity = 2 * params.size.w;
    uint quota = capacity / params.size.w;
    uint base = bucket * quota;
    uint retained = 0;
    uint output = EMPTY;
    uint source = heads[bucket].load();
    [loop] while (source != EMPTY) {
        Entry entry = entries[source];
        uint next = entry.next;
        // Unmatched receiver groups are transient; never persist their sample-local IDs.
        if (params.history.w > 0
            && (entry.padding.y & 0x80000000u) == 0
            && entry.valid == (params.frame.x | 0x80000000u)
            && params.frame.x - entry.lastSample <= uint(params.history.w)
            && retained < quota) {
            uint slot = base + retained++;
            entry.next = output;
            entry.padding.x = slot;
            previousEntries[slot] = entry;
            entries[source].padding.x = slot;
            output = slot;
        } else {
            entries[source].padding.x = EMPTY;
            if (params.history.w > 0
                && (entry.padding.y & 0x80000000u) == 0
                && entry.valid == (params.frame.x | 0x80000000u)
                && params.frame.x - entry.lastSample <= uint(params.history.w))
                heads[2 * params.size.w + 4].add(1);
        }
        source = next;
    }
    // After gather this head array becomes the next frame's history table.
    heads[bucket].store(output);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void remapProbeHeads(uint3 thread : SV_DispatchThreadID) {
    uint slot = thread.x;
    if (slot >= params.size.w) return;
    uint head = params.size.w + 1 + slot;
    uint source = heads[head].load();
    if (source != EMPTY) {
        uint mapped = entries[source].padding.x;
        heads[head].store(mapped == EMPTY ? TOMBSTONE : mapped);
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void checkProbeLookup(uint3 thread : SV_DispatchThreadID) {
    if (thread.x == 0)
        entries[0].padding.y = findIrradianceProbe(int4(10, 0, 0, 0), true, uint2(0), 0);
}


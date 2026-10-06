#include "common.slang"

[[vk::binding(0, 0)]] ConstantBuffer<Parameters> params;
[[vk::binding(1, 0)]] StructuredBuffer<Triangle> triangles;
[[vk::binding(2, 0)]] StructuredBuffer<BvhNode> nodes;
[[vk::binding(3, 0)]] StructuredBuffer<Light> lights;
[[vk::binding(4, 0)]] DepthTexture2D depthTexture;
[[vk::binding(5, 0)]] Texture2D<float4> normalTexture;
[[vk::binding(6, 0)]] RWStructuredBuffer<Atomic<uint>> heads;
[[vk::binding(7, 0)]] RWStructuredBuffer<Entry> entries;
[[vk::binding(8, 0)]] StructuredBuffer<uint> previousHeads;
[[vk::binding(9, 0)]] StructuredBuffer<Entry> previousEntries;
[[vk::binding(10, 0)]] [format("rgba32f")] WTexture2D<float4> irradiance;
[[vk::binding(11, 0)]] Texture2D<float4> previousIrradiance;
[[vk::binding(12, 0)]] RWStructuredBuffer<IrradianceProbe> irradianceProbes;
[[vk::binding(13, 0)]] Texture2D<float4> currentIrradiance;
[[vk::binding(14, 0)]] [format("rgba32f")] WTexture2D<float4> historyMetadata;
[[vk::binding(15, 0)]] Texture2D<float4> previousMetadata;

struct Hit { float distance; uint triangle; };

bool intersectBox(float3 origin, float3 direction, BvhNode node, float limit) {
    float near = 0.0, far = limit;
    for (uint axis = 0; axis < 3; ++axis) {
        if (abs(direction[axis]) < 1e-8) {
            if (origin[axis] < node.minimum[axis] || origin[axis] > node.maximum[axis]) return false;
        } else {
            float a = (node.minimum[axis] - origin[axis]) / direction[axis];
            float b = (node.maximum[axis] - origin[axis]) / direction[axis];
            near = max(near, min(a, b)); far = min(far, max(a, b));
        }
    }
    return near <= far;
}
float intersectTriangle(float3 origin, float3 direction, Triangle triangle) {
    float3 e1 = triangle.b.xyz - triangle.a.xyz, e2 = triangle.c.xyz - triangle.a.xyz;
    float3 h = cross(direction, e2);
    float determinant = dot(e1, h);
    if (abs(determinant) < 1e-8) return -1.0;
    float inverse = 1.0 / determinant;
    float3 s = origin - triangle.a.xyz;
    float u = inverse * dot(s, h);
    if (u < 0.0 || u > 1.0) return -1.0;
    float3 q = cross(s, e1);
    float v = inverse * dot(direction, q);
    if (v < 0.0 || u + v > 1.0) return -1.0;
    return inverse * dot(e2, q);
}
Hit traceRay(float3 origin, float3 direction, float limit, bool anyHit = false) {
    Hit hit; hit.distance = limit; hit.triangle = EMPTY;
    if (params.frame.z == 0) return hit;
    // Median splitting guarantees depth <= 32 for a u32 triangle count.
    uint stack[64]; uint count = 1; stack[0] = 0;
    while (count != 0) {
        BvhNode node = nodes[stack[--count]];
        if (!intersectBox(origin, direction, node, hit.distance)) continue;
        if (node.count == 0) {
            stack[count++] = node.first;
            stack[count++] = node.first + 1;
        } else {
            for (uint i = node.first; i < node.first + node.count; ++i) {
                float distance = intersectTriangle(origin, direction, triangles[i]);
                if (distance > params.tuning.z * 0.25 && distance < hit.distance) {
                    hit.distance = distance; hit.triangle = i;
                    if (anyHit) return hit;
                }
            }
        }
    }
    return hit;
}

// Next-event estimation of emissive surfaces. Bright, small sources no longer
// depend on a hemisphere ray accidentally reaching the emitter. Power-weighted
// CDF selection is logarithmic in emitter count and uses the existing light buffer.
float3 emissiveIrradiance(float3 position, float3 normal, uint count) {
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
    uint lightSamples = (count + 1) / 2;
    float lightFraction = float(lightSamples) / float(count);
    for (uint sample = 0; sample < count; ++sample) {
        float2 uv = frac(r2(sample + params.frame.x * count + 1) + rotation);
        if (sample >= lightSamples) {
            float3 direction = cosineDirection(normal, uv);
            Hit hit = traceRay(position, direction, 1e20);
            if (hit.triangle == EMPTY) continue;
            Triangle source = triangles[hit.triangle];
            if (source.emission.w <= 0) continue;
            float3 crossEdges = cross(source.b.xyz - source.a.xyz, source.c.xyz - source.a.xyz);
            float area = length(crossEdges) * 0.5;
            float emitterCosine = dot(normalize(crossEdges), -direction);
            if (emitterCosine <= 0) continue;
            float cosinePdf = max(0.0, dot(normal, direction)) / PI;
            float lightPdf = source.emission.w * hit.distance * hit.distance / (area * emitterCosine);
            float pdf = lightFraction * lightPdf + (1 - lightFraction) * cosinePdf;
            sum += max(source.emission.xyz, 0) * (cosinePdf / max(pdf, 1e-20));
            continue;
        }
        float selection = frac((float(sample) + 0.5) / float(lightSamples) + phase);
        uint low = 0, high = params.lightCounts.y - 1;
        while (low < high) {
            uint middle = (low + high) / 2;
            if (lights[params.lightCounts.x + middle].direction.w < selection) low = middle + 1;
            else high = middle;
        }
        Light light = lights[params.lightCounts.x + low];
        if (light.edge1.w <= 0 || dot(light.direction.xyz, position - light.position.xyz) <= 0) continue;
        float root = sqrt(uv.x);
        float3 target = light.position.xyz + light.edge1.xyz * (root * (1 - uv.y))
                        + light.edge2.xyz * (root * uv.y);
        float3 toLight = target - position;
        float lightDistance = length(toLight);
        if (lightDistance <= params.tuning.z * 0.25) continue;
        float3 lightDirection = toLight / lightDistance;
        float cosinePdf = max(0.0, dot(normal, lightDirection)) / PI;
        float emitterCosine = max(0.0, dot(light.direction.xyz, -lightDirection));
        if (cosinePdf <= 0 || emitterCosine <= 0) continue;
        float lightPdf = light.edge1.w * lightDistance * lightDistance / (light.color.w * emitterCosine);
        // Balance heuristic in solid-angle measure. With both techniques,
        // contributions are bounded by Le/cosineFraction, even for tiny lights.
        float pdf = lightFraction * lightPdf + (1 - lightFraction) * cosinePdf;
        // Evaluate irradiance at the real surface. Reduce only the visibility
        // ray's offset near the light so bias cannot move it across the emitter.
        float planeDistance = dot(light.direction.xyz, position - light.position.xyz);
        float3 origin = position + normal * min(params.tuning.z, planeDistance * 0.25);
        float3 delta = target - origin;
        float distance = length(delta);
        if (distance <= params.tuning.z * 0.25) continue;
        float3 direction = delta / distance;
        if (traceRay(origin, direction, distance - min(params.tuning.z * 0.25, distance * 0.001), true).triangle != EMPTY) continue;
        sum += light.color.xyz * (cosinePdf / max(pdf, 1e-20));
    }
    return sum / float(count);
}

float4 findMerged(int4 key, bool previous, float3 position = float3(0), bool visibility = false) {
    if (previous && params.history.z == 0.0) return 0.0;
    uint bucket = hashKey(key, params.size.w);
    uint index = previous ? previousHeads[bucket] : heads[bucket].load();
    while (index != EMPTY) {
        Entry entry = previous ? previousEntries[index] : entries[index];
        if (all(entry.key == key)) {
            if (visibility && entry.merged.w > 0) {
                float3 delta = entry.origin.xyz - position;
                float distance = length(delta);
                if (distance > params.tuning.z * 2 && traceRay(position, delta / distance, distance - params.tuning.z, true).triangle != EMPTY) return 0;
            }
            return entry.merged;
        }
        index = entry.next;
    }
    return 0.0;
}
float4 interpolate(float3 position, uint direction, uint level, bool previous, bool secondary = false, uint surface = 0, bool visibility = false) {
    float3 grid = position / spacing(params, level, secondary) - 0.5;
    int3 base = int3(floor(grid)); float3 fraction = frac(grid);
    float3 sum = 0.0; float total = 0.0;
    for (uint corner = 0; corner < 8; ++corner) {
        uint3 bit = uint3(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
        float3 weights = lerp(1.0 - fraction, fraction, float3(bit));
        float weight = weights.x * weights.y * weights.z;
        if (weight <= 0.00001) continue;
        int4 key = int4(base + int3(bit), int(direction | (level << 12) | (secondary ? SECONDARY : 0) | surface));
        float4 value = findMerged(key, previous, position, visibility);
        if (value.w > 0) { sum += value.xyz * weight; total += weight; }
    }
    return total > 0.00001 ? float4(sum / total, 1.0) : float4(0.0);
}
// Returns irradiance / pi for the Lambertian BRDF, without the surface albedo.
float3 diffuseGatherExact(float3 position, float3 normal, bool previous, bool secondary = false) {
    float3 sum = 0.0; float total = 0.0;
    for (uint direction = 0; direction < 2 * BASE_THETA * BASE_THETA; ++direction) {
        float cosine = max(0.0, dot(normal, binDirection(direction, 0)));
        if (cosine <= 0.0) continue;
        float4 value = interpolate(position, direction, 0, previous, secondary, surfaceKey(normal), true);
        if (value.w > 0) { sum += value.xyz * cosine; total += cosine; }
    }
    return total > 0.0 ? sum / total : float3(0.0);
}

uint findIrradianceProbe(int4 key, bool previous) {
    if (previous && params.history.z == 0) return EMPTY;
    uint slot = hashKey(key, params.size.w);
    for (uint attempt = 0; attempt < 32; ++attempt) {
        uint index = previous ? previousHeads[params.size.w + 1 + slot] : heads[params.size.w + 1 + slot].load();
        if (index == EMPTY) return EMPTY;
        int4 candidate = previous ? previousEntries[index].key : entries[index].key;
        if (all(candidate.xyz == key.xyz) && (uint(candidate.w) & (SECONDARY | SURFACE)) == uint(key.w)) return slot;
        slot = (slot + 1) & (params.size.w - 1);
    }
    return EMPTY;
}

float3 diffuseGather(float3 position, float3 normal, bool previous, bool secondary = false) {
    if (previous && params.history.z == 0) return 0;
    float3 grid = position / spacing(params, 0, secondary) - 0.5;
    int3 base = int3(floor(grid)); float3 fraction = frac(grid);
    float3 sum = 0; float total = 0; bool occluded = false;
    uint frame = params.frame.x + (previous ? 1 : 0);
    for (uint corner = 0; corner < 8; ++corner) {
        int3 bit = int3(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
        float3 weights = lerp(1 - fraction, fraction, float3(bit));
        float weight = weights.x * weights.y * weights.z;
        if (weight <= 0.00001) continue;
        uint probe = findIrradianceProbe(int4(base + bit, int((secondary ? SECONDARY : 0) | surfaceKey(normal))), previous);
        if (probe == EMPTY) continue;
        IrradianceProbe value = irradianceProbes[(frame & 1) * params.size.w + probe];
        // Probes live on actual surface samples. Reject reuse through a wall
        // instead of interpolating from arbitrary cell centers inside solids.
        float3 delta = value.origin.xyz - position;
        float distance = length(delta);
        if (distance > params.tuning.z * 2 && traceRay(position, delta / distance, distance - params.tuning.z, true).triangle != EMPTY) {
            occluded = true; continue;
        }
        float4 diffuse = 0;
        [unroll] for (uint axis = 0; axis < 3; ++axis) {
            uint lobe = axis * 2 + (normal[axis] < 0 ? 1 : 0);
            diffuse += value.coefficients[lobe] * (normal[axis] * normal[axis]);
        }
        if (diffuse.w <= 0.00001) continue;
        sum += diffuse.xyz / diffuse.w * weight; total += weight;
    }
    // ponytail: bounded linear probing may overflow under sparse, moving views;
    // the exact directional gather keeps those pixels lit without GPU stalls.
    return total > 0.00001 ? sum / total : (occluded ? float3(0) : diffuseGatherExact(position, normal, previous, secondary));
}
float3 outgoingRadiance(float3 origin, float3 direction, Hit hit) {
    if (hit.triangle == EMPTY) return params.environment.xyz;
    Triangle triangle = triangles[hit.triangle];
    if (all(triangle.albedo.xyz <= 0)) return 0;
    float3 position = origin + direction * hit.distance;
    float3 normal = normalize(cross(triangle.b.xyz - triangle.a.xyz, triangle.c.xyz - triangle.a.xyz));
    if (dot(normal, direction) > 0) normal = -normal;
    float3 offset = position + normal * params.tuning.z;
    float3 diffuse = emissiveIrradiance(position, normal, max(1u, params.lightCounts.z / 2));
    for (uint i = 0; i < params.frame.w; ++i) {
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
        if (cosine > 0 && attenuation > 0 && traceRay(offset, toLight, distance - params.tuning.z, true).triangle == EMPTY)
            diffuse += light.color.xyz * (attenuation * cosine / PI);
    }
    if (params.history.y > 0 && params.history.z > 0)
        diffuse += diffuseGather(offset, normal, true, true) * params.history.y;
    // Emission is sampled explicitly at every receiver; including it here too
    // would count direct emitter paths twice. Cascades carry scattered radiance.
    return max(0.0, triangle.albedo.xyz * diffuse);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void clear(uint3 thread : SV_DispatchThreadID) {
    if (thread.x < params.size.w) heads[thread.x].store(EMPTY);
    if (thread.x < params.size.w) heads[params.size.w + 1 + thread.x].store(EMPTY);
    if (thread.x == 0) heads[params.size.w].store(0);
    // Entries unreachable from a hash head are never read. Avoid clearing the
    // large interval allocation; trace and retain initialize reachable nodes.
}

uint findEntry(int4 key) {
    uint index = heads[hashKey(key, params.size.w)].load();
    while (index != EMPTY && !all(entries[index].key == key)) index = entries[index].next;
    return index;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void seed(uint3 thread : SV_DispatchThreadID) {
    if (any(thread.xy >= params.size.xy)) return;
    uint sample = thread.x + thread.y * params.size.x;
    for (uint level = 0; level < CASCADES; ++level) entries[sample * CASCADES + level].valid = 0;
    uint2 full = fullPixel(thread.xy, params);
    float depth = depthTexture.Load(int3(full, 0));
    float3 normal = normalTexture.Load(int3(full, 0)).xyz * 2 - 1;
    if (depth <= 0 || dot(normal, normal) < 0.01) return;
    float3 position = worldPosition(float2(full) + 0.5, depth, params) + normalize(normal) * params.tuning.z;
    uint index = sample * CASCADES;
    Entry entry = (Entry)0;
    entry.key = cacheKey(position, 0, 0, params, false, surfaceKey(normal)); entry.samples = 1; entry.valid = 1;
    entry.next = heads[hashKey(entry.key, params.size.w)].exchange(index);
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
    entry.key = int4(child.key.xyz >> 1, int((params.frame.y << 12) | (uint(child.key.w) & SURFACE)));
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
    uint source = heads[thread.x].load(), unique = EMPTY;
    while (source != EMPTY) {
        Entry sample = entries[source]; uint next = sample.next, match = unique;
        while (match != EMPTY && !all(entries[match].key == sample.key)) match = entries[match].next;
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
    for (uint index = heads[thread.x].load(); index != EMPTY; index = entries[index].next) {
        int4 key = entries[index].key;
        if ((uint(key.w) >> 12 & 3u) != level) continue;
        // Independent roots may use independent sequence phases. Descendants
        // share contiguous segments at every cascade, including negative cells.
        uint offset = hashKey(key, 65536);
        if (level + 1 < CASCADES) {
            int3 parent = key.xyz >> 1;
            uint parentIndex = findEntry(int4(parent, int(((level + 1) << 12) | (uint(key.w) & SURFACE))));
            offset = parentIndex == EMPTY ? 0 : entries[parentIndex].valid;
            int3 local = key.xyz - parent * 2;
            uint ordinal = uint(local.x + local.y * 2 + local.z * 4);
            for (uint sibling = 0; sibling < ordinal; ++sibling) {
                int3 bit = int3(sibling & 1, (sibling >> 1) & 1, (sibling >> 2) & 1);
                uint child = findEntry(int4(parent * 2 + bit, int((level << 12) | (uint(key.w) & SURFACE))));
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
    uint probe = findEntry(entries[index].key);
    entries[index].lastSample = probe == EMPTY ? 0 : entries[probe].valid + entries[index].lastSample;
}

uint currentCapacity() {
    return params.size.z * CASCADES * (params.history.y > 0 ? 2 : 1);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void retain(uint3 thread : SV_DispatchThreadID) {
    if (thread.x >= params.size.w || params.history.z == 0 || params.history.x == 0 || params.history.w == 0) return;
    uint capacity = currentCapacity();
    for (uint index = previousHeads[thread.x]; index != EMPTY; index = previousEntries[index].next) {
        Entry entry = previousEntries[index];
        if (params.frame.x - entry.lastSample > uint(params.history.w)) continue;
        uint slot = heads[params.size.w].add(1);
        if (slot >= capacity) continue;
        slot += capacity;
        entry.samples = 0;
        entry.next = heads[thread.x].exchange(slot);
        entries[slot] = entry;
    }
}

void deposit(uint first, float3 position, float3 normal, float3 direction, Hit hit, float3 radiance, bool secondary) {
    for (uint level = 0; level < CASCADES; ++level) {
        uint index = first + level;
        float end = params.tuning.y * float(1u << (2u * level)) * (secondary ? 4.0 : 1.0);
        bool terminates = hit.distance <= end || level == CASCADES - 1;
        Entry entry;
        entry.key = cacheKey(position, directionIndex(direction, level), level, params, secondary, surfaceKey(normal));
        entry.interval = terminates ? float4(radiance, 0.0) : float4(0, 0, 0, 1);
        entry.merged = 0.0;
        entry.origin = float4(position, 1);
        entry.valid = 1; entry.lastSample = params.frame.x; entry.samples = 1;
        entry.next = heads[hashKey(entry.key, params.size.w)].exchange(index);
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
    uint rank = entries[sample * CASCADES].lastSample;
    for (uint level = 0; level < CASCADES; ++level) {
        entries[sample * CASCADES + level].valid = 0;
        if (params.history.y > 0) entries[(params.size.z + sample) * CASCADES + level].valid = 0;
    }
    uint2 full = fullPixel(pixel, params);
    float depth = depthTexture.Load(int3(full, 0));
    if (depth <= 0.0) return;
    float3 normal = normalTexture.Load(int3(full, 0)).xyz * 2.0 - 1.0;
    if (dot(normal, normal) < 0.01) return;
    normal = normalize(normal);
    float3 position = worldPosition(float2(full) + 0.5, depth, params) + normal * params.tuning.z;
    float2 jitter = r2(params.frame.x);
    float2 uv = frac(0.5 + r2(rank + 1) + jitter);
    float3 direction = decodeDirection(uv);
    if (dot(direction, normal) < 0) direction = -direction;
    // The last cascade covers the remainder of the ray. A finite cutoff
    // mistakes distant occluders for open sky and leaks environment lighting.
    Hit hit = traceRay(position, direction, 1e20);
    float3 radiance = outgoingRadiance(position, direction, hit);
    if (params.history.y > 0 && hit.triangle != EMPTY && any(triangles[hit.triangle].albedo.xyz > 0)) {
        Triangle triangle = triangles[hit.triangle];
        float3 hitNormal = normalize(cross(triangle.b.xyz - triangle.a.xyz, triangle.c.xyz - triangle.a.xyz));
        if (dot(hitNormal, direction) > 0) hitNormal = -hitNormal;
        float3 secondaryPosition = position + direction * hit.distance + hitNormal * params.tuning.z;
        // Independent phase: a fixed rotation of the primary R2 sample couples
        // both path vertices and can miss an emitter across a whole moving frame.
        uint secondaryPhase = hashKey(int4(int(sample), int(params.frame.x), int(rank), 0), 0x80000000u);
        float3 secondaryDirection = decodeDirection(frac(0.5 + r2(secondaryPhase)));
        if (dot(secondaryDirection, hitNormal) < 0) secondaryDirection = -secondaryDirection;
        Hit secondaryHit = traceRay(secondaryPosition, secondaryDirection, 1e20);
        float3 secondaryRadiance = outgoingRadiance(secondaryPosition, secondaryDirection, secondaryHit);
        deposit((params.size.z + sample) * CASCADES, secondaryPosition, hitNormal, secondaryDirection, secondaryHit, secondaryRadiance, true);
        // Emitter sampling at the primary hit already supplies a fresh reflected
        // bounce during motion. A second unfiltered path here adds third-bounce
        // flashes; secondary samples instead feed the next frame's probe cache.
    }
    deposit(sample * CASCADES, position, normal, direction, hit, radiance, false);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void reduce(uint3 thread : SV_DispatchThreadID) {
    uint bucket = thread.x;
    if (bucket >= params.size.w) return;
    uint source = heads[bucket].load(), unique = EMPTY;
    // One owner per bucket: combine equal full keys without float atomics or
    // spinlocks. All later passes traverse only unique directional entries.
    // ponytail: quadratic bucket reduction; use sorted segmented reduction if
    // heavily colliding workloads make this pass dominate GPU measurements.
    while (source != EMPTY) {
        Entry sample = entries[source];
        uint next = sample.next, match = unique;
        while (match != EMPTY && !all(entries[match].key == sample.key)) match = entries[match].next;
        if (match == EMPTY) {
            sample.next = unique;
            entries[source] = sample;
            unique = source;
        } else {
            if (sample.samples > 0) {
                float3 center = (float3(sample.key.xyz) + 0.5) * spacing(params, uint(sample.key.w) >> 12 & 3u, (uint(sample.key.w) & SECONDARY) != 0);
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
    for (uint index = unique; index != EMPTY; index = entries[index].next) {
        if (entries[index].samples > 0) {
            entries[index].interval /= float(entries[index].samples);
        }
    }
}

void mergeEntry(uint index) {
    // Unchanged samples already have merged radiance. Fresh samples update it;
    // scene edits invalidate both caches, so no stale lighting survives edits.
    if (entries[index].samples == 0) return;
    uint level = params.frame.y;
    int4 key = entries[index].key;
    if ((uint(key.w) >> 12 & 3u) != level) return;
    bool secondary = (uint(key.w) & SECONDARY) != 0;
    float4 interval = entries[index].interval;
    float3 radiance = interval.xyz;
    bool known = interval.w < 1.0;
    if (interval.w > 0.0 && level + 1 < CASCADES) {
        float3 position = entries[index].origin.xyz;
        uint direction = uint(key.w) & 4095u;
        uint theta = BASE_THETA << level;
        uint2 xy = uint2(direction % (2 * theta), direction / (2 * theta));
        float3 sum = 0.0; float count = 0;
        for (uint child = 0; child < 4; ++child) {
            uint2 bit = uint2(child & 1, child >> 1);
            uint2 fine = xy * 2 + bit;
            uint fineDirection = fine.x + fine.y * (4 * theta);
            // Reject transport from a coarse surface origin behind geometry;
            // checking only the final irradiance probe leaves this leak baked in.
            float4 value = interpolate(position, fineDirection, level + 1, false, secondary, uint(key.w) & SURFACE, true);
            if (value.w > 0) { sum += value.xyz; count += 1; }
        }
        if (count > 0) { radiance += interval.w * sum / count; known = true; }
    }
    float4 old = findMerged(key, true, entries[index].origin.xyz, true);
    if (old.w > 0) {
        radiance = known ? lerp(radiance, old.xyz, params.history.x) : old.xyz;
        known = true;
    }
    // Separate compute passes provide storage barriers between cascade levels.
    entries[index].merged = float4(radiance, known ? 1.0 : 0.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void merge(uint3 thread : SV_DispatchThreadID) {
    if (thread.x >= currentCapacity() / CASCADES) return;
    uint index = thread.x * CASCADES + params.frame.y;
    if (entries[index].valid == 0) return;
    // Canonical fresh entries can merge independently. Retained entries keep
    // their existing I, so neither serial bucket walks nor old slots are needed.
    mergeEntry(index);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void seedIrradiance(uint3 thread : SV_DispatchThreadID) {
    if (thread.x >= params.size.w) return;
    for (uint index = heads[thread.x].load(); index != EMPTY; index = entries[index].next) {
        int4 key = entries[index].key;
        if ((uint(key.w) >> 12 & 3u) != 0) continue;
        key.w = int(uint(key.w) & (SECONDARY | SURFACE));
        uint slot = hashKey(key, params.size.w);
        for (uint attempt = 0; attempt < 32; ++attempt) {
            uint candidate = heads[params.size.w + 1 + slot].compareExchange(EMPTY, index);
            if (candidate == EMPTY) break;
            int4 other = entries[candidate].key;
            if (all(other.xyz == key.xyz) && (uint(other.w) & (SECONDARY | SURFACE)) == uint(key.w)) break;
            slot = (slot + 1) & (params.size.w - 1);
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void prefilter(uint3 thread : SV_DispatchThreadID) {
    if (thread.x >= params.size.w) return;
    uint index = heads[params.size.w + 1 + thread.x].load();
    if (index == EMPTY) return;
    int4 key = entries[index].key;
    uint mode = uint(key.w) & (SECONDARY | SURFACE);
    IrradianceProbe probe = (IrradianceProbe)0;
    float3 center = (float3(key.xyz) + 0.5) * spacing(params, 0, (mode & SECONDARY) != 0);
    for (uint direction = 0; direction < 32; ++direction) {
        float4 radiance = findMerged(int4(key.xyz, int(direction | mode)), false);
        if (radiance.w == 0) continue;
        float3 n = binDirection(direction, 0);
        [unroll] for (uint i = 0; i < 6; ++i)
            probe.coefficients[i] += radiance * max(0.0, dot(n, lobeDirection(i)));
        uint source = findEntry(int4(key.xyz, int(direction | mode)));
        if (source != EMPTY && (probe.origin.w == 0 || length(entries[source].origin.xyz - center) < length(probe.origin.xyz - center)))
            probe.origin = entries[source].origin;
    }
    [unroll] for (uint i = 0; i < 6; ++i)
        if (probe.coefficients[i].w > 0) probe.coefficients[i] = float4(probe.coefficients[i].xyz / probe.coefficients[i].w, 1);
    irradianceProbes[(params.frame.x & 1) * params.size.w + thread.x] = probe;
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
    float3 position = worldPosition(float2(full) + 0.5, depth, params) + normal * params.tuning.z;
    float3 result = diffuseGather(position, normal, false)
                   + emissiveIrradiance(position - normal * params.tuning.z, normal, params.lightCounts.z);
    irradiance.Store(int2(pixel), float4(result, depth));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void temporal(uint3 thread : SV_DispatchThreadID) {
    if (any(thread.xy >= params.size.xy)) return;
    int2 pixel = int2(thread.xy);
    float4 center = currentIrradiance.Load(int3(pixel, 0));
    if (center.w <= 0) { irradiance.Store(pixel, 0); historyMetadata.Store(pixel, 0); return; }
    uint2 full = fullPixel(thread.xy, params);
    float3 normal = normalize(normalTexture.Load(int3(full, 0)).xyz * 2 - 1);
    float3 position = worldPosition(float2(full) + 0.5, center.w, params);
    float3 mean = 0, square = 0, maximum = center.xyz;
    float count = 0;
    for (int y = -1; y <= 1; ++y) for (int x = -1; x <= 1; ++x) {
        int2 neighbor = pixel + int2(x, y);
        if (any(neighbor < 0) || any(neighbor >= int2(params.size.xy))) continue;
        float4 value = currentIrradiance.Load(int3(neighbor, 0));
        if (value.w <= 0) continue;
        uint2 otherFull = fullPixel(uint2(neighbor), params);
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
    if (params.stability.x > 0 && params.history.x > 0) {
        float4 clip = mul(params.previousClipFromWorld, float4(position, 1));
        float3 ndc = clip.xyz / clip.w;
        float2 uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if (clip.w > 0 && all(uv >= 0) && all(uv < 1)) {
            float2 grid = irradianceGrid(float2(params.viewport.xy) + uv * float2(params.viewport.zw), params);
            int2 base = int2(floor(grid)); float2 fraction = frac(grid);
            for (uint corner = 0; corner < 4; ++corner) {
                int2 bit = int2(corner & 1, corner >> 1), oldPixel = base + bit;
                if (any(oldPixel < 0) || any(oldPixel >= int2(params.size.xy))) continue;
                float4 old = previousIrradiance.Load(int3(oldPixel, 0));
                float4 meta = previousMetadata.Load(int3(oldPixel, 0));
                if (meta.z <= 0 || old.w <= 0 || dot(normal, decodeNormal(meta.xy)) < 0.97) continue;
                float3 oldPosition = previousWorldPosition(float2(fullPixel(uint2(oldPixel), params)) + 0.5, old.w, params);
                if (abs(dot(oldPosition - position, normal)) > surfaceTolerance(params)) continue;
                float2 w = lerp(1 - fraction, fraction, float2(bit)); float weight = w.x * w.y;
                history += old * weight; metadata += meta * weight; total += weight;
            }
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
    float3 normal = normalize(normalTexture.Load(int3(full, 0)).xyz * 2 - 1);
    float3 position = worldPosition(float2(full) + 0.5, center.w, params);
    float variance = currentIrradiance.Load(int3(pixel, 0)).w;
    float3 sum = 0; float total = 0;
    uint step = 1u << params.frame.y;
    const float kernel[5] = {1.0, 4.0, 6.0, 4.0, 1.0};
    for (int y = -2; y <= 2; ++y) for (int x = -2; x <= 2; ++x) {
        int2 otherPixel = pixel + int2(x, y) * int(step);
        if (any(otherPixel < 0) || any(otherPixel >= int2(params.size.xy))) continue;
        float4 value = previousIrradiance.Load(int3(otherPixel, 0));
        if (value.w <= 0) continue;
        uint2 otherFull = fullPixel(uint2(otherPixel), params);
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

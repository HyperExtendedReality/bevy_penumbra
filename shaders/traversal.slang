struct Hit { float distance; uint triangle; uint instance; uint errors; float3 barycentric; float footprint; };

// A malformed or partially resident hierarchy can contain a cycle that never
// grows the traversal stack (a node whose only intersecting child links back to
// an ancestor), and its leaf ranges can cover the whole scene. Both would let a
// single ray run for seconds and trip the GPU watchdog, so every traversal also
// carries a step budget and a bounded leaf scan.
static const uint MAX_TRAVERSAL_STEPS = 4096u;
static const uint MAX_LEAF_TRIANGLES = 256u;
static const uint MAX_LEAF_INSTANCES = 4096u;

// A denied traversal is not a miss: treating it as sky or an unoccluded
// shadow sample would inject energy (or claim false visibility). The ticket
// counter is reset by the cache-clear pass once per frame. A zero budget is
// unbounded while still recording attempts when the buffer is present.
static const uint HIT_QUERY_UNSUPPORTED = 4u;
#ifndef GI_TRAVERSAL_CHECK
bool admitTransportQuery(uint admissionLimit) {
    uint attemptIndex = 2u * params.size.w + 5u;
    uint admittedIndex = 2u * params.size.w + 6u;
    heads[attemptIndex].add(1u);
    if (params.sampling.y == 0u) {
        heads[admittedIndex].add(1u);
        return true;
    }

    uint limit = params.sampling.y;
    if (admissionLimit != 0u) limit = min(limit, admissionLimit);
    // Every receiver starts its first query together; a compare-exchange retry
    // loop on this one counter serializes them. Take a ticket and return it
    // when over the limit: never over-admits, and the final count is exact.
    if (heads[admittedIndex].add(1u) < limit) return true;
    heads[admittedIndex].add(0xffffffffu);
    return false;
}
uint ordinaryTransportAdmissionLimit() {
    return uint(max(params.intervalHistory.w, 0.0));
}
bool admitTransportQuery() {
    return admitTransportQuery(ordinaryTransportAdmissionLimit());
}
#else
bool admitTransportQuery(uint admissionLimit) { return true; }
bool admitTransportQuery() { return admitTransportQuery(0u); }
uint ordinaryTransportAdmissionLimit() { return 0u; }
#endif

#ifndef GI_TRAVERSAL_CHECK
[[vk::binding(28, 0)]] Texture2DArray<uint4> geometryPageTexture;
// Scene and geometry words are addressed by values that ultimately come from
// shader-computed triangles and material records. A single stale or malformed
// index must not become an out-of-range buffer/texture address: that faults the
// GPU (device lost) instead of producing a wrong colour. Every read below is
// therefore clamped to the bound buffer or page array.
uint4 geometryInlineWord(uint index) {
    uint words, stride;
    geometryWords.GetDimensions(words, stride);
    return words == 0u ? (uint4)0 : geometryWords[min(index, words - 1u)];
}
uint4 geometryPageWord(uint index) {
    uint width, height, layers;
    geometryPageTexture.GetDimensions(width, height, layers);
    uint side = params.geometryPages.y;
    if (side == 0u || layers == 0u) return (uint4)0;
    uint pageWords = side * side;
    uint layer = index / pageWords;
    if (layer >= layers) return (uint4)0;
    uint withinPage = index % pageWords;
    return geometryPageTexture.Load(int4(withinPage % side, withinPage / side, layer, 0));
}
uint4 geometryWord(uint index) {
    if (params.geometryPages.x == 0) return geometryInlineWord(index);
    if (params.geometryPages.x != 2) return geometryPageWord(index);
    uint4 header = geometryPageWord(0);
    if (index < header.y || index >= header.y + header.w * header.z)
        return geometryPageWord(index);
    uint vertex = (index - header.y) / header.z;
    uint field = (index - header.y) % header.z;
    uint4 global = geometryInlineWord(0u);
    uint count = global.x, low = 0, high = count;
    [loop] while (low < high) {
        uint middle = (low + high) >> 1;
        uint4 range = geometryInlineWord(global.y + middle);
        if (vertex >= range.x) low = middle + 1;
        else high = middle;
    }
    if (low > 0) {
        uint4 range = geometryInlineWord(global.y + low - 1);
        if (vertex >= range.x && vertex < range.x + range.y)
            return geometryPageWord(global.z + (range.z + vertex - range.x) * 5 + field);
    }
    return geometryPageWord(index);
}
#else
uint4 geometryWord(uint index) { return geometryWords[index]; }
#endif

#ifndef GI_TRAVERSAL_CHECK
[[vk::binding(29, 0)]] Texture2DArray<uint4> scenePageTexture;
uint4 scenePageWord(uint index) {
    uint width, height, layers;
    scenePageTexture.GetDimensions(width, height, layers);
    uint side = params.scenePages.y;
    if (side == 0u || layers == 0u) return (uint4)0;
    uint pageWords = side * side;
    uint layer = index / pageWords;
    if (layer >= layers) return (uint4)0;
    uint withinPage = index % pageWords;
    return scenePageTexture.Load(int4(withinPage % side, withinPage / side, layer, 0));
}
uint4 sceneInlineWord(uint index) {
    uint words, stride;
    sceneWords.GetDimensions(words, stride);
    return words == 0u ? (uint4)0 : sceneWords[min(index, words - 1u)];
}
uint4 sceneWord(uint index) {
    if (params.scenePages.x == 0) return sceneInlineWord(index);
    uint prefixWords = params.scenePages.z;
    uint rangeCount = params.scenePages.w;
    uint instanceBase = params.sceneOverlay.x;
    uint dynamicInstanceWords = params.sceneOverlay.y;
    uint staticBlasWords = params.sceneOverlay.z;
    uint mutableBlasWords = params.sceneOverlay.w;
    if (index < prefixWords) return sceneInlineWord(index);
    uint nodeBase = prefixWords + rangeCount;
    uint instanceOverlayBase = nodeBase + mutableBlasWords;
    if (index >= instanceBase && index < instanceBase + dynamicInstanceWords)
        return sceneInlineWord(instanceOverlayBase + index - instanceBase);
    if (index < instanceBase) {
        uint localWord = index - prefixWords;
        uint low = 0, high = rangeCount;
        [loop] while (low < high) {
            uint middle = (low + high) >> 1;
            uint4 range = sceneInlineWord(prefixWords + middle);
            if (localWord >= range.x * 2) low = middle + 1;
            else high = middle;
        }
        uint mutableBefore = 0;
        if (low > 0) {
            uint4 range = sceneInlineWord(prefixWords + low - 1);
            uint rangeStart = range.x * 2;
            uint rangeEnd = rangeStart + range.y * 2;
            if (localWord < rangeEnd)
                return sceneInlineWord(range.z + localWord - rangeStart);
            mutableBefore = range.w + range.y;
        }
        return scenePageWord(localWord - mutableBefore * 2);
    }
    return scenePageWord(staticBlasWords + index - instanceBase - dynamicInstanceWords);
}
#else
uint4 sceneWord(uint index) { return sceneWords[index]; }
#endif

uint3 triangleIndices(uint triangle, uint4 header) { return geometryWord(header.x + triangle).xyz; }
uint3 triangleIndices(uint triangle) { return triangleIndices(triangle, geometryWord(0)); }
uint vertexBase(uint vertex, uint4 header) { return header.y + vertex * header.z; }
uint vertexBase(uint vertex) { return vertexBase(vertex, geometryWord(0)); }
Triangle localTriangle(uint3 indices, uint4 header) {
    Triangle triangle;
    triangle.a = float4(asfloat(geometryWord(vertexBase(indices.x, header)).xyz), 0);
    triangle.b = float4(asfloat(geometryWord(vertexBase(indices.y, header)).xyz), 0);
    triangle.c = float4(asfloat(geometryWord(vertexBase(indices.z, header)).xyz), 0);
    triangle.albedo = 0; triangle.emission = 0;
    return triangle;
}
Triangle localTriangle(uint index, uint4 header) {
    return localTriangle(triangleIndices(index, header), header);
}
Triangle localTriangle(uint index) { return localTriangle(index, geometryWord(0)); }
float4 hitVertexAttribute(Hit hit, uint attribute, uint3 indices, uint4 header) {
    return asfloat(geometryWord(vertexBase(indices.x, header) + attribute)) * hit.barycentric.x
         + asfloat(geometryWord(vertexBase(indices.y, header) + attribute)) * hit.barycentric.y
         + asfloat(geometryWord(vertexBase(indices.z, header) + attribute)) * hit.barycentric.z;
}
float4 hitVertexAttribute(Hit hit, uint attribute) {
    uint4 header = geometryWord(0);
    return hitVertexAttribute(hit, attribute, triangleIndices(hit.triangle, header), header);
}
float3 unitVector(float3 value, float3 fallback) {
    float scale = max(abs(value.x), max(abs(value.y), abs(value.z)));
    if (scale == 0 || !isfinite(scale)) return fallback;
    // DX12 division can lower to a reciprocal. Keep that reciprocal normal
    // even at f32::MAX, instead of letting flush-to-zero erase the direction.
    value /= max(scale * 0.25, asfloat(0x00800000u));
    return value / length(value);
}

BvhNode sceneNode(uint index) {
    uint4 a = sceneWord(1 + index * 2), b = sceneWord(2 + index * 2);
    BvhNode node; node.minimum = asfloat(a.xyz); node.first = a.w;
    node.maximum = asfloat(b.xyz); node.count = b.w; return node;
}
[ForceInline]
uint instanceBase(uint instance) {
    uint4 header = sceneWord(0);
    return header.y + instance * header.w;
}
uint2 hitInstanceIdentity(Hit hit) { return sceneWord(instanceBase(hit.instance) + 7).xy; }
bool instanceVisibleRecord(uint4 mask) {
    uint count = min(mask.y, params.layers.y);
    uint wholeWords = count / 4u;
    // One scene/page lookup and one light lookup cover four layer bits.
    [loop] for (uint word = 0; word < wholeWords; ++word) {
        uint4 instanceBits = sceneWord(mask.x + word);
        uint4 cameraBits = lights[params.layers.x + word].source;
        if (any((instanceBits & cameraBits) != uint4(0))) return true;
    }
    if (wholeWords * 4u < count) {
        uint4 instanceBits = sceneWord(mask.x + wholeWords);
        uint4 cameraBits = lights[params.layers.x + wholeWords].source;
        [unroll] for (uint i = 0; i < 3u; ++i) {
            if (wholeWords * 4u + i < count && (instanceBits[i] & cameraBits[i]) != 0) return true;
        }
    }
    return false;
}
bool instanceVisible(uint instance) {
    return instanceVisibleRecord(sceneWord(instanceBase(instance) + 6));
}
[ForceInline]
float3 transformPoint(uint base, float3 value, bool direction = false) {
    float4 h = float4(value, direction ? 0 : 1);
    return float3(dot(asfloat(sceneWord(base)), h), dot(asfloat(sceneWord(base+1)), h), dot(asfloat(sceneWord(base+2)), h));
}
[ForceInline]
float3 transformDirection(uint base, float3 value, bool transpose = false) {
    float3 a = asfloat(sceneWord(base)).xyz;
    float3 b = asfloat(sceneWord(base + 1)).xyz;
    float3 c = asfloat(sceneWord(base + 2)).xyz;
    // Only direction matters. Scale operands before multiplying so finite,
    // large/small imported attributes do not overflow under instance scale.
    float3 magnitude = max(abs(a), max(abs(b), abs(c)));
    float scale = max(magnitude.x, max(magnitude.y, magnitude.z));
    if (scale == 0) return 0;
    scale = max(scale * 0.25, asfloat(0x00800000u));
    a /= scale; b /= scale; c /= scale;
    value = unitVector(value, float3(0));
    return transpose ? a * value.x + b * value.y + c * value.z
                     : float3(dot(a, value), dot(b, value), dot(c, value));
}
struct HitAttributes {
    float3 geometricNormal;
    float3 worldNormal; // normalized per vertex, then interpolated (MikkTSpace)
    float4 worldTangent;
};
float3 triangleGeometricNormal(uint base, uint flags, Triangle triangle) {
    return ((flags & 4) != 0 ? -1 : 1) * unitVector(transformDirection(base,
        cross(triangle.b.xyz - triangle.a.xyz, triangle.c.xyz - triangle.a.xyz), true), float3(0, 1, 0));
}
float3 hitGeometricNormal(Hit hit) {
    uint base = instanceBase(hit.instance);
    return triangleGeometricNormal(base, sceneWord(base + 6).w, localTriangle(hit.triangle));
}
HitAttributes hitAttributes(Hit hit) {
    uint base = instanceBase(hit.instance);
    uint4 header = geometryWord(0);
    uint3 indices = triangleIndices(hit.triangle, header);
    Triangle triangle = localTriangle(indices, header);
    uint instanceFlags = sceneWord(base + 6).w;
    float sign = (instanceFlags & 4) != 0 ? -1 : 1;
    HitAttributes attributes;
    attributes.geometricNormal = triangleGeometricNormal(base, instanceFlags, triangle);
    attributes.worldNormal = 0; attributes.worldTangent = 0;
    [loop] for (uint i = 0; i < 3; ++i) {
        uint vertex = vertexBase(indices[i], header);
        uint flags = geometryWord(vertex).w;
        float3 normal = attributes.geometricNormal;
        if ((flags & 1) != 0)
            normal = unitVector(transformDirection(base, asfloat(geometryWord(vertex + 1).xyz), true), normal);
        // Match Bevy: normalize transformed normals/tangents before interpolation;
        // retain the interpolated lengths for the later normal-map TBN transform.
        attributes.worldNormal += normal * hit.barycentric[i];
        float4 tangent = asfloat(geometryWord(vertex + 2));
        attributes.worldTangent += float4(unitVector(transformDirection(base + 3, tangent.xyz), float3(0)),
                                          tangent.w * sign) * hit.barycentric[i];
    }
    return attributes;
}
#include "material.slang"
float3 hitShadingNormal(Hit hit, HitAttributes attributes, float3 incoming) {
    return materialNormal(hit, attributes, incoming);
}
float3 hitEmission(Hit hit, float3 baseColor) {
    uint material = materialBase(hit);
    uint flags = sceneWord(instanceBase(hit.instance)+6).w;
    float3 emission;
    if ((flags & 16) != 0) emission = baseColor;
    else emission = asfloat(sceneWord(material+1).xyz)*materialSample(hit,1).xyz;
#ifndef GI_TRAVERSAL_CHECK
    float exposureWeight = (flags&16) != 0 ? 0 : asfloat(sceneWord(material+2).w);
    // Preserve Bevy's emissive exposure weight before cached GI is pre-exposed.
    float exposure = max(params.environment.w,1e-20);
    emission *= lerp(1/exposure,1.0,exposureWeight);
#endif
    return emission;
}
Triangle worldTriangle(Hit hit) {
    uint base = instanceBase(hit.instance);
    Triangle triangle = localTriangle(hit.triangle);
    triangle.a.xyz = transformPoint(base + 3, triangle.a.xyz);
    triangle.b.xyz = transformPoint(base + 3, triangle.b.xyz);
    triangle.c.xyz = transformPoint(base + 3, triangle.c.xyz);
    float4 color = materialBaseColor(hit);
    uint material = materialBase(hit);
    float metallic = saturate(asfloat(sceneWord(material + 2).x) * materialSample(hit, 2).b);
    // Preserve the low two cull-mode bits for transport shading.
    triangle.albedo = float4(color.xyz * (1 - metallic), float(sceneWord(base + 6).w & 3u));
    if ((sceneWord(base + 6).w & 16) != 0) triangle.albedo.xyz = 0;
    float3 emission = hitEmission(hit,color.xyz);
    triangle.emission = float4(emission, 0);
    return triangle;
}
float3 emitterRadiance(Light light, float3 barycentric) {
    if (!instanceVisible(light.source.x)) return float3(0);
    Hit hit; hit.distance = 0; hit.instance = light.source.x; hit.triangle = light.source.y;
    hit.barycentric = barycentric; hit.errors = 0; hit.footprint = 0;
    float coverage = materialCoverage(hit, materialAlphaCutoff(hit));
    if (!(coverage > 0.0)) return 0.0;
    float3 baseColor = materialBaseColor(hit).rgb;
    return max(hitEmission(hit, baseColor), 0.0) * coverage;
}
float sourceCosine(Triangle triangle, float3 direction) {
    float c = dot(normalize(cross(triangle.b.xyz - triangle.a.xyz, triangle.c.xyz - triangle.a.xyz)), -direction);
    uint cullMode = uint(triangle.albedo.w) & 3u;
    return cullMode == 0 ? abs(c) : (cullMode == 2 ? -c : c);
}
float sourceCosine(Light light, float3 direction) {
    float c = dot(light.direction.xyz, -direction);
    return light.position.w == -2 ? abs(c) : c;
}

[ForceInline]
bool conventionalLightIsPunctual(Light light) {
    // Conventional records use position.w > 0 for finite point/spot sources;
    // directional records encode the direction to the source in position.xyz.
    return light.position.w > 0.0;
}
[ForceInline]
void addConventionalLightRadiance(inout float3 background, inout float3 punctual,
                                  bool finiteLight, float3 contribution) {
    if (finiteLight) punctual += contribution;
    else background += contribution;
}

[ForceInline]
bool boxRayInterval(float3 origin, float3 direction, BvhNode node, float limit,
                    out float entry, out float exit) {
    float near = 0.0, far = limit;
    [unroll] for (uint axis = 0; axis < 3; ++axis) {
        if (direction[axis] == 0) {
            if (origin[axis] < node.minimum[axis] || origin[axis] > node.maximum[axis]) {
                entry = 0.0; exit = 0.0;
                return false;
            }
        } else {
            float a = (node.minimum[axis] - origin[axis]) / direction[axis];
            float b = (node.maximum[axis] - origin[axis]) / direction[axis];
            float axisExit = max(a, b);
            // Conservative far endpoint accounts for slab division roundoff.
            near = max(near, min(a, b)); far = min(far, axisExit + abs(axisExit) * 3.5762791e-7);
        }
    }
    entry = near; exit = far;
    return near <= far;
}
[ForceInline]
float boxEntryDistance(float3 origin, float3 direction, BvhNode node, float limit) {
    float entry, exit;
    return boxRayInterval(origin, direction, node, limit, entry, exit) ? entry : -1.0;
}
[ForceInline]
bool intersectBox(float3 origin, float3 direction, BvhNode node, float limit) {
    return boxEntryDistance(origin, direction, node, limit) >= 0.0;
}
[ForceInline]
float orientedEdge(float2 a, float2 b) {
    // Evaluate a shared edge in a canonical order and apply its sign afterwards.
    // GPU contraction of a*b-c*d can otherwise give opposite directed edges
    // different rounding (one product is fused, the other is rounded first).
    bool reverse = a.x > b.x || (a.x == b.x && a.y > b.y);
    float2 first = reverse ? b : a, second = reverse ? a : b;
    precise float productA = first.x * second.y;
    precise float productB = first.y * second.x;
    precise float edge = productA - productB;
    float error = (abs(productA) + abs(productB)) * 2.3841858e-7;
    if (abs(edge) <= error) edge = 0;
    return reverse ? -edge : edge;
}
struct TriangleRay {
    uint kx, ky, kz;
    float sx, sy;
    bool valid;
};

TriangleRay triangleRay(float3 direction) {
    // The shear and dominant axis depend only on the local ray, not on the
    // triangle. Reuse them for every primitive visited in this instance.
    TriangleRay ray = (TriangleRay)0;
    float3 magnitude = abs(direction);
    ray.kz = magnitude.x > magnitude.y ? (magnitude.x > magnitude.z ? 0 : 2) : (magnitude.y > magnitude.z ? 1 : 2);
    ray.valid = direction[ray.kz] != 0;
    if (!ray.valid) return ray;
    ray.kx = (ray.kz + 1) % 3;
    ray.ky = (ray.kx + 1) % 3;
    if (direction[ray.kz] < 0) { uint swap = ray.kx; ray.kx = ray.ky; ray.ky = swap; }
    ray.sx = direction[ray.kx] / direction[ray.kz];
    ray.sy = direction[ray.ky] / direction[ray.kz];
    return ray;
}

float3 intersectTriangle(float3 origin, float3 direction, TriangleRay ray, Triangle triangle) {
    // Sheared edge functions give adjacent triangles matching edge decisions.
    // No absolute determinant epsilon: local directions retain world t and can
    // be arbitrarily short/long under nonuniform instance scale.
    if (!ray.valid) return float3(-1, 0, 0);
    uint kx = ray.kx, ky = ray.ky, kz = ray.kz;
    float sx = ray.sx, sy = ray.sy;
    float3 a = triangle.a.xyz - origin, b = triangle.b.xyz - origin, c = triangle.c.xyz - origin;
    float2 pa = float2(a[kx] - sx * a[kz], a[ky] - sy * a[kz]);
    float2 pb = float2(b[kx] - sx * b[kz], b[ky] - sy * b[kz]);
    float2 pc = float2(c[kx] - sx * c[kz], c[ky] - sy * c[kz]);
    float largest = max(max(max(abs(pa.x), abs(pa.y)), max(abs(pb.x), abs(pb.y))), max(abs(pc.x), abs(pc.y)));
    if (largest == 0 || !isfinite(largest)) return float3(-1, 0, 0);
    // Exact power-of-two rescaling preserves shared-edge signs while keeping
    // edge products normal for small/large local mesh coordinates. Do not use
    // reciprocal(largest): desktop GPU flush-to-zero can erase that reciprocal.
    uint exponent = (asuint(largest) >> 23) & 255u;
    float scale = asfloat((254u - clamp(exponent, 1u, 253u)) << 23);
    pa *= scale; pb *= scale; pc *= scale;
    float3 edges = float3(orientedEdge(pb, pc), orientedEdge(pc, pa), orientedEdge(pa, pb));
    if (any(edges < 0) && any(edges > 0)) return float3(-1, 0, 0);
    float determinant = edges.x + edges.y + edges.z;
    if (determinant == 0 || !isfinite(determinant)) return float3(-1, 0, 0);
    // ponytail: f32-only edge predicates on portable WGSL; extremely skinny
    // triangles whose edge differences underflow need an exact-predicate backend.
    float3 weights = edges / determinant;
    float distance = dot(weights, float3(a[kz], b[kz], c[kz])) / direction[kz];
    return isfinite(distance) ? float3(distance, weights.yz) : float3(-1, 0, 0);
}

Hit traceTriangle(float3 origin, float3 direction, TriangleRay ray, float3 worldDirection,
                  float coneWidth, float coneSpread, uint triangleIndex, uint instance,
                  uint base, uint flags, uint4 geometryHeader, Hit hit) {
    Triangle triangle = localTriangle(triangleIndex, geometryHeader);
    if ((flags & 3u) != 0) {
        float facing = dot(cross(triangle.b.xyz-triangle.a.xyz, triangle.c.xyz-triangle.a.xyz), direction)
                       * ((flags & 4) != 0 ? -1 : 1);
        if (((flags & 1) != 0 && facing >= 0) || ((flags & 2) != 0 && facing <= 0)) return hit;
    }
    float3 intersection = intersectTriangle(origin, direction, ray, triangle);
    float distance = intersection.x;
    // Reject sub-precision self hits at translated triangle boundaries.
    if (distance > 0.000025 && distance < hit.distance) {
        Hit candidate = hit; candidate.triangle = triangleIndex; candidate.instance = instance;
        candidate.barycentric = float3(1 - intersection.y - intersection.z, intersection.yz);
        candidate.footprint = 0;
        float cutoff = materialAlphaCutoff(candidate);
        if (cutoff != -1.0 && (coneWidth > 0 || coneSpread > 0)) {
            float cosine = abs(dot(triangleGeometricNormal(base, flags, triangle), worldDirection));
            candidate.footprint = (max(coneWidth, 0.0) + max(coneSpread, 0.0) * distance)
                                  / max(cosine, 0.001);
        }
        if (!materialAccepts(candidate, cutoff)) return hit;
        hit.distance = distance; hit.triangle = triangleIndex; hit.instance = instance;
        hit.barycentric = float3(1 - intersection.y - intersection.z, intersection.yz);
        hit.footprint = candidate.footprint;
    }
    return hit;
}
Hit traceInstance(float3 origin, float3 direction, uint instance, Hit hit, bool anyHit,
                  float coneWidth, float coneSpread, uint4 geometryHeader) {
    uint base = instanceBase(instance);
    uint4 instanceRecord = sceneWord(base + 6);
    if (!instanceVisibleRecord(instanceRecord)) return hit;
    // Both transforms use the same three inverse-matrix rows. Loading them
    // once avoids repeating paged-scene/overlay resolution for every instance.
    float4 row0 = asfloat(sceneWord(base));
    float4 row1 = asfloat(sceneWord(base + 1));
    float4 row2 = asfloat(sceneWord(base + 2));
    float4 point = float4(origin, 1);
    float4 vector = float4(direction, 0);
    float3 localOrigin = float3(dot(row0, point), dot(row1, point), dot(row2, point));
    // Keep this unnormalized so triangle/box distances remain world distances.
    float3 localDirection = float3(dot(row0, vector), dot(row1, vector), dot(row2, vector));
    uint flags = instanceRecord.w;
    uint2 stack[49]; uint count = 0;
    uint4 geometry = sceneWord(base + 8);
    uint currentNode = geometry.x;
    BvhNode node = sceneNode(currentNode);
    bool currentNodeReady = intersectBox(localOrigin, localDirection, node, hit.distance);
    TriangleRay ray = (TriangleRay)0;
    if (currentNodeReady) ray = triangleRay(localDirection);
    uint traversalSteps = 0u;
    [loop] while (currentNodeReady || count != 0) {
        if (++traversalSteps > MAX_TRAVERSAL_STEPS) {
            hit.errors |= HIT_QUERY_UNSUPPORTED;
            break;
        }
        if (!currentNodeReady) {
            // The box was already tested; only a closer hit can prune it.
            uint2 deferred = stack[--count];
            if (asfloat(deferred.y) > hit.distance) continue;
            currentNode = deferred.x;
            node = sceneNode(currentNode);
        }
        currentNodeReady = false;
        if (node.count == 0) {
            if (count + 2 > 49) {
                // Unexpected topology cannot silently drop an occluder. Fall
                // back to this instance's primitive range and report the flag.
                hit.errors |= 1;
                node.first = geometry.y; node.count = geometry.z;
                count = 0;
            } else {
                uint leftIndex = node.first;
                uint rightIndex = leftIndex + 1;
                BvhNode leftNode = sceneNode(leftIndex);
                BvhNode rightNode = sceneNode(rightIndex);
                float leftEntry = boxEntryDistance(localOrigin, localDirection, leftNode, hit.distance);
                float rightEntry = boxEntryDistance(localOrigin, localDirection, rightNode, hit.distance);
                if (leftEntry >= 0.0 && rightEntry >= 0.0) {
                    if (leftEntry <= rightEntry) {
                        stack[count++] = uint2(rightIndex, asuint(rightEntry));
                        currentNode = leftIndex;
                        node = leftNode;
                    } else {
                        stack[count++] = uint2(leftIndex, asuint(leftEntry));
                        currentNode = rightIndex;
                        node = rightNode;
                    }
                    currentNodeReady = true;
                } else if (leftEntry >= 0.0) {
                    currentNode = leftIndex;
                    node = leftNode;
                    currentNodeReady = true;
                } else if (rightEntry >= 0.0) {
                    currentNode = rightIndex;
                    node = rightNode;
                    currentNodeReady = true;
                }
                continue;
            }
        }
        // A clipped closest-hit scan cannot prove that a reported hit is
        // nearest, or that a miss is actually empty.
        uint triangleBegin = max(node.first, geometry.y);
        uint triangleLimit = geometry.y + geometry.z;
        uint requestedEnd = node.first + node.count;
        uint triangleEnd = min(requestedEnd, triangleLimit);
        bool truncated = node.first < geometry.y || requestedEnd < node.first
            || requestedEnd > triangleLimit || node.count > MAX_LEAF_TRIANGLES;
        uint boundedEnd = min(triangleEnd, triangleBegin + MAX_LEAF_TRIANGLES);
        truncated = truncated || boundedEnd < triangleEnd;
        if (truncated) hit.errors |= HIT_QUERY_UNSUPPORTED;
        [loop] for (uint i = triangleBegin; i < boundedEnd; ++i) {
            hit = traceTriangle(localOrigin, localDirection, ray, direction, coneWidth, coneSpread,
                                i, instance, base, flags, geometryHeader, hit);
            if (anyHit && hit.triangle != EMPTY) return hit;
        }
    }
    return hit;
}
#ifdef PENUMBRA_HARDWARE_RAY_QUERY
// Hardware ray queries replace only the BVH walk. Slang's WGSL target has no
// ray-query types, and naga requires each query to be a local of the function
// that drives it, so render.rs prepends the two query loops in WGSL
// (`ray_query_source`). Every BLAS is non-opaque: the loops call the candidate
// callbacks below, which re-test each proposed triangle with the software
// tests. Accepted hits keep the software distance, barycentrics, culling,
// layer and alpha-coverage decisions.
static float3 hardwareOrigin;
static float3 hardwareDirection;
static float hardwareLimit;
static float hardwareConeWidth;
static float hardwareConeSpread;
static uint4 hardwareGeometryHeader;
static Hit hardwareHit;

// Runs the closest-hit loop; always returns true.
bool hardwareQueryClosest(float3 origin, float3 direction, float limit, bool anyHit) {
    __target_switch { case wgsl: __intrinsic_asm "penumbraRayQueryClosest($0, $1, $2, $3)"; }
}
// Runs the occlusion loop; returns 1 when blocked and 0 otherwise.
uint hardwareQueryOccluded(float3 origin, float3 direction, float limit) {
    __target_switch { case wgsl: __intrinsic_asm "penumbraRayQueryOccluded($0, $1, $2)"; }
}

// The TLAS custom data is the packed instance index, and each BLAS covers that
// instance's contiguous triangle range in order.
[ForceInline]
uint hardwareTriangle(uint instance, uint primitive, out uint base, out uint4 record,
                      out float3 localOrigin, out float3 localDirection) {
    base = instanceBase(instance);
    record = sceneWord(base + 6);
    float4 row0 = asfloat(sceneWord(base));
    float4 row1 = asfloat(sceneWord(base + 1));
    float4 row2 = asfloat(sceneWord(base + 2));
    float4 point = float4(hardwareOrigin, 1);
    float4 vector = float4(hardwareDirection, 0);
    localOrigin = float3(dot(row0, point), dot(row1, point), dot(row2, point));
    // Unnormalized, as in traceInstance, so local distances stay world distances.
    localDirection = float3(dot(row0, vector), dot(row1, vector), dot(row2, vector));
    return sceneWord(base + 8).y + primitive;
}

// Called by the WGSL closest-hit loop; true confirms the candidate.
[noinline]
bool hardwareClosestCandidate(uint instance, uint primitive) {
    if (instance == EMPTY) return false;
    uint base;
    uint4 record;
    float3 localOrigin, localDirection;
    uint triangle = hardwareTriangle(instance, primitive, base, record,
                                     localOrigin, localDirection);
    if (!instanceVisibleRecord(record)) return false;
    Hit candidate = traceTriangle(localOrigin, localDirection, triangleRay(localDirection),
        hardwareDirection, hardwareConeWidth, hardwareConeSpread, triangle, instance, base,
        record.w, hardwareGeometryHeader, hardwareHit);
    // traceTriangle returns its input unchanged unless it accepts a closer hit.
    if (!(candidate.distance < hardwareHit.distance)) return false;
    hardwareHit = candidate;
    return true;
}

// Called by the WGSL occlusion loop; true ends the query as blocked.
[noinline]
bool hardwareOccludedCandidate(uint instance, uint primitive) {
    if (instance == EMPTY) return false;
    uint base;
    uint4 record;
    float3 localOrigin, localDirection;
    uint triangle = hardwareTriangle(instance, primitive, base, record,
                                     localOrigin, localDirection);
    if (!instanceVisibleRecord(record)) return false;
    return occludedTriangle(localOrigin, localDirection, triangleRay(localDirection),
                            triangle, instance, record.w, hardwareLimit,
                            hardwareGeometryHeader);
}

Hit hardwareClosestHit(float3 origin, float3 direction, Hit hit, bool anyHit,
                       float coneWidth, float coneSpread, uint4 geometryHeader) {
    hardwareOrigin = origin;
    hardwareDirection = direction;
    hardwareConeWidth = coneWidth;
    hardwareConeSpread = coneSpread;
    hardwareGeometryHeader = geometryHeader;
    hardwareHit = hit;
    // The query always returns true. The guarded call keeps the callback in
    // the emitted module, where only the WGSL loop calls it.
    if (!hardwareQueryClosest(origin, direction, hit.distance, anyHit))
        hardwareClosestCandidate(EMPTY, 0u);
    return hardwareHit;
}

bool hardwareOccluded(float3 origin, float3 direction, float limit, uint4 geometryHeader) {
    hardwareOrigin = origin;
    hardwareDirection = direction;
    hardwareLimit = limit;
    hardwareGeometryHeader = geometryHeader;
    uint blocked = hardwareQueryOccluded(origin, direction, limit);
    // Never above one; the guarded call keeps the callback in the module.
    if (blocked > 1u) hardwareOccludedCandidate(EMPTY, 0u);
    return blocked != 0u;
}
#endif
Hit traceRayTracked(float3 origin, float3 direction, float limit,
                    out DependencyContext dependency, out float dependencyStart,
                    out float dependencyDistance, out float dependencyPadding,
                    bool anyHit = false,
                    float coneWidth = 0, float coneSpread = 0, uint admissionLimit = 0u) {
    Hit hit; hit.distance = limit; hit.triangle = EMPTY; hit.instance = EMPTY; hit.errors = 0; hit.barycentric = 0; hit.footprint = 0;
    dependency = dependencyUnknown();
    dependencyStart = 0.0;
    dependencyDistance = 0.0;
    dependencyPadding = 0.0;
    if (params.frame.z == 0) { dependency = dependencyEmpty(); return hit; }
    uint effectiveAdmissionLimit = admissionLimit == 0u
        ? ordinaryTransportAdmissionLimit() : admissionLimit;
    if (!admitTransportQuery(effectiveAdmissionLimit)) { hit.errors |= HIT_QUERY_UNSUPPORTED; return hit; }
    // The near-child walk retains each deferred box's entry distance so a
    // closer hit can prune it without retesting the box.
    uint2 stack[49]; uint count = 0;
    uint currentNode = 0;
    BvhNode node = sceneNode(currentNode);
    float rootEntry, rootExit;
    bool intersectsRoot = boxRayInterval(origin, direction, node, hit.distance,
                                          rootEntry, rootExit);
    dependency = dependencyEmpty();
    if (!intersectsRoot) return hit;
    uint4 geometryHeader = geometryWord(0);
    dependencyStart = max(rootEntry, 0.0);
    dependencyDistance = min(limit, rootExit);
#ifdef PENUMBRA_HARDWARE_RAY_QUERY
    hit = hardwareClosestHit(origin, direction, hit, anyHit, coneWidth, coneSpread,
                             geometryHeader);
#else
    bool currentNodeReady = rootEntry >= 0.0;
    uint traversalSteps = 0u;
    [loop] while (currentNodeReady || count != 0) {
        if (++traversalSteps > MAX_TRAVERSAL_STEPS) {
            hit.errors |= HIT_QUERY_UNSUPPORTED;
            break;
        }
        if (!currentNodeReady) {
            uint2 deferred = stack[--count];
            if (asfloat(deferred.y) > hit.distance) continue;
            currentNode = deferred.x;
            node = sceneNode(currentNode);
        }
        currentNodeReady = false;
        if (node.count == 0) {
            if (count + 2 > 49) {
                hit.errors |= 2;
                node.first = 0; node.count = params.frame.z;
                count = 0;
            } else {
                uint leftIndex = node.first;
                uint rightIndex = leftIndex + 1;
                BvhNode leftNode = sceneNode(leftIndex);
                BvhNode rightNode = sceneNode(rightIndex);
                float leftEntry = boxEntryDistance(origin, direction, leftNode, hit.distance);
                float rightEntry = boxEntryDistance(origin, direction, rightNode, hit.distance);
                if (leftEntry >= 0.0 && rightEntry >= 0.0) {
                    if (leftEntry <= rightEntry) {
                        stack[count++] = uint2(rightIndex, asuint(rightEntry));
                        currentNode = leftIndex;
                        node = leftNode;
                    } else {
                        stack[count++] = uint2(leftIndex, asuint(leftEntry));
                        currentNode = rightIndex;
                        node = rightNode;
                    }
                    currentNodeReady = true;
                } else if (leftEntry >= 0.0) {
                    currentNode = leftIndex;
                    node = leftNode;
                    currentNodeReady = true;
                } else if (rightEntry >= 0.0) {
                    currentNode = rightIndex;
                    node = rightNode;
                    currentNodeReady = true;
                }
                continue;
            }
        }
        uint requestedEnd = node.first + node.count;
        bool truncated = requestedEnd < node.first || node.first > params.frame.z
            || requestedEnd > params.frame.z || node.count > MAX_LEAF_INSTANCES;
        uint instanceBegin = min(node.first, params.frame.z);
        uint instanceEnd = min(requestedEnd, params.frame.z);
        uint boundedEnd = min(instanceEnd, instanceBegin + MAX_LEAF_INSTANCES);
        truncated = truncated || boundedEnd < instanceEnd;
        if (truncated) hit.errors |= HIT_QUERY_UNSUPPORTED;
        [loop] for (uint i = instanceBegin; i < boundedEnd; ++i) {
            hit = traceInstance(origin, direction, i, hit, anyHit, coneWidth, coneSpread, geometryHeader);
            if (anyHit && hit.triangle != EMPTY) {
                // Any accepted hit is a complete proof for an any-hit query,
                // regardless of earlier branches that could not be inspected.
                hit.errors &= ~HIT_QUERY_UNSUPPORTED;
                count = 0;
                break;
            }
        }
    }
#endif
    if (hit.triangle != EMPTY && (coneWidth > 0 || coneSpread > 0)) {
        uint base = instanceBase(hit.instance);
        uint flags = sceneWord(base + 6).w;
        Triangle triangle = localTriangle(hit.triangle);
        float cosine = abs(dot(triangleGeometricNormal(base, flags, triangle), direction));
        hit.footprint = (max(coneWidth, 0.0) + max(coneSpread, 0.0) * hit.distance)
                        / max(cosine, 0.001);
    }
    float dependencyEnd = hit.triangle != EMPTY ? min(hit.distance, rootExit) : rootExit;
    float padding = max(coneWidth, 0.0) + max(coneSpread, 0.0) * max(dependencyEnd, 0.0);
    dependencyDistance = max(dependencyEnd, 0.0);
    dependencyPadding = padding;
    dependencyIncludeSegment(dependency, origin + direction * dependencyStart,
                             origin + direction * dependencyEnd, padding);
    return hit;
}
Hit traceRay(float3 origin, float3 direction, float limit, bool anyHit = false,
             float coneWidth = 0, float coneSpread = 0, uint admissionLimit = 0u) {
    DependencyContext unusedDependency;
    float unusedStart, unusedDistance, unusedPadding;
    return traceRayTracked(origin, direction, limit, unusedDependency,
                           unusedStart, unusedDistance, unusedPadding, anyHit,
                           coneWidth, coneSpread, admissionLimit);
}

bool occludedTriangle(float3 origin, float3 direction, TriangleRay ray, uint triangleIndex,
                      uint instance, uint flags, float limit, uint4 geometryHeader) {
    Triangle triangle = localTriangle(triangleIndex, geometryHeader);
    if ((flags & 3u) != 0) {
        float facing = dot(cross(triangle.b.xyz-triangle.a.xyz,triangle.c.xyz-triangle.a.xyz),direction)
                       * ((flags & 4) != 0 ? -1 : 1);
        if (((flags & 1) != 0 && facing >= 0) || ((flags & 2) != 0 && facing <= 0)) return false;
    }
    float3 intersection = intersectTriangle(origin,direction,ray,triangle);
    if (!(intersection.x > 0.000025 && intersection.x < limit)) return false;
    Hit candidate; candidate.distance = intersection.x; candidate.triangle = triangleIndex;
    candidate.instance = instance; candidate.errors = 0;
    candidate.barycentric = float3(1-intersection.y-intersection.z,intersection.yz);
    candidate.footprint = 0;
    float cutoff = materialAlphaCutoff(candidate);
    if (cutoff == -1.0) return true;
    return materialAccepts(candidate,cutoff);
}

bool occludedInstance(float3 origin, float3 direction, uint instance, float limit,
                      uint4 geometryHeader, out bool querySupported) {
    querySupported = true;
    uint base = instanceBase(instance);
    uint4 instanceRecord = sceneWord(base+6);
    if (!instanceVisibleRecord(instanceRecord)) return false;
    float4 row0 = asfloat(sceneWord(base));
    float4 row1 = asfloat(sceneWord(base + 1));
    float4 row2 = asfloat(sceneWord(base + 2));
    float4 point = float4(origin, 1);
    float4 vector = float4(direction, 0);
    float3 localOrigin = float3(dot(row0, point), dot(row1, point), dot(row2, point));
    float3 localDirection = float3(dot(row0, vector), dot(row1, vector), dot(row2, vector));
    uint flags = instanceRecord.w;
    uint stack[49]; uint count = 0;
    uint4 geometry = sceneWord(base+8);
    uint currentNode = geometry.x;
    BvhNode node = sceneNode(currentNode);
    bool currentNodeReady = intersectBox(localOrigin,localDirection,node,limit);
    TriangleRay ray = (TriangleRay)0;
    if (currentNodeReady) ray = triangleRay(localDirection);
    uint traversalSteps = 0u;
    bool incomplete = false;
    [loop] while (currentNodeReady || count != 0) {
        if (++traversalSteps > MAX_TRAVERSAL_STEPS) {
            querySupported = false;
            return false;
        }
        if (!currentNodeReady) {
            currentNode = stack[--count];
            node = sceneNode(currentNode);
            // The deferred node passed the unchanged ray/limit test at its parent.
        }
        currentNodeReady = false;
        if (node.count == 0) {
            if (count + 2 > 49) {
                node.first = geometry.y; node.count = geometry.z;
                count = 0;
            } else {
                uint leftIndex = node.first;
                uint rightIndex = leftIndex + 1;
                BvhNode leftNode = sceneNode(leftIndex);
                BvhNode rightNode = sceneNode(rightIndex);
                float leftEntry = boxEntryDistance(localOrigin,localDirection,leftNode,limit);
                float rightEntry = boxEntryDistance(localOrigin,localDirection,rightNode,limit);
                if (leftEntry >= 0.0 && rightEntry >= 0.0) {
                    if (leftEntry <= rightEntry) {
                        stack[count++] = rightIndex;
                        currentNode = leftIndex;
                        node = leftNode;
                    } else {
                        stack[count++] = leftIndex;
                        currentNode = rightIndex;
                        node = rightNode;
                    }
                    currentNodeReady = true;
                } else if (leftEntry >= 0.0) {
                    currentNode = leftIndex;
                    node = leftNode;
                    currentNodeReady = true;
                } else if (rightEntry >= 0.0) {
                    currentNode = rightIndex;
                    node = rightNode;
                    currentNodeReady = true;
                }
                continue;
            }
        }
        // Scan only the proven primitive range and preserve incompleteness if
        // an oversized or malformed range was clipped. A real accepted hit is
        // still a complete proof that this shadow ray is blocked.
        uint triangleBegin = max(node.first, geometry.y);
        uint triangleLimit = geometry.y + geometry.z;
        uint requestedEnd = node.first + node.count;
        uint triangleEnd = min(requestedEnd, triangleLimit);
        bool truncated = node.first < geometry.y || requestedEnd < node.first
            || requestedEnd > triangleLimit || node.count > MAX_LEAF_TRIANGLES;
        uint boundedEnd = min(triangleEnd, triangleBegin + MAX_LEAF_TRIANGLES);
        truncated = truncated || boundedEnd < triangleEnd;
        incomplete = incomplete || truncated;
        [loop] for (uint i = triangleBegin; i < boundedEnd; ++i) {
            if (occludedTriangle(localOrigin,localDirection,ray,i,instance,flags,limit,geometryHeader))
                return true;
        }
    }
    querySupported = !incomplete;
    return false;
}

bool traceOccluded(float3 origin, float3 direction, float limit) {
    bool querySupported;
    bool result = traceOccluded(origin, direction, limit, querySupported);
    // Preserve conservative behavior for callers that cannot consume a
    // support bit. Production estimators use the out-parameter overload below.
    return querySupported ? result : true;
}

bool traceOccluded(float3 origin, float3 direction, float limit, out bool querySupported) {
    return traceOccluded(origin, direction, limit, querySupported,
                         ordinaryTransportAdmissionLimit());
}

bool traceOccluded(float3 origin, float3 direction, float limit,
                    out bool querySupported, uint admissionLimit) {
    bool intersectsRoot;
    float rootEntry, rootExit;
    return traceOccluded(origin, direction, limit, querySupported, admissionLimit,
                         intersectsRoot, rootEntry, rootExit);
}

bool traceOccluded(float3 origin, float3 direction, float limit,
                    out bool querySupported, uint admissionLimit,
                    out bool intersectsRoot, out float rootEntry, out float rootExit) {
    intersectsRoot = false;
    rootEntry = 0.0;
    rootExit = 0.0;
    querySupported = true;
    if (params.frame.z == 0) return false;
    if (admissionLimit == 0u) admissionLimit = ordinaryTransportAdmissionLimit();
    if (!admitTransportQuery(admissionLimit)) { querySupported = false; return false; }
    uint stack[49]; uint count = 0;
    uint currentNode = 0;
    BvhNode node = sceneNode(currentNode);
    intersectsRoot = boxRayInterval(origin, direction, node, limit, rootEntry, rootExit);
    bool currentNodeReady = intersectsRoot && rootEntry >= 0.0;
    if (!currentNodeReady) return false;
    uint4 geometryHeader = geometryWord(0);
#ifdef PENUMBRA_HARDWARE_RAY_QUERY
    // Hardware traversal is never truncated, so every admitted query is complete.
    return hardwareOccluded(origin, direction, limit, geometryHeader);
#else
    uint traversalSteps = 0u;
    bool incomplete = false;
    [loop] while (currentNodeReady || count != 0) {
        if (++traversalSteps > MAX_TRAVERSAL_STEPS) {
            querySupported = false;
            return false;
        }
        if (!currentNodeReady) {
            currentNode = stack[--count];
            node = sceneNode(currentNode);
            // The deferred node passed the unchanged ray/limit test at its parent.
        }
        currentNodeReady = false;
        if (node.count == 0) {
            if (count + 2 > 49) {
                node.first = 0; node.count = params.frame.z;
                count = 0;
            } else {
                uint leftIndex = node.first;
                uint rightIndex = leftIndex + 1;
                BvhNode leftNode = sceneNode(leftIndex);
                BvhNode rightNode = sceneNode(rightIndex);
                float leftEntry = boxEntryDistance(origin,direction,leftNode,limit);
                float rightEntry = boxEntryDistance(origin,direction,rightNode,limit);
                if (leftEntry >= 0.0 && rightEntry >= 0.0) {
                    if (leftEntry <= rightEntry) {
                        stack[count++] = rightIndex;
                        currentNode = leftIndex;
                        node = leftNode;
                    } else {
                        stack[count++] = leftIndex;
                        currentNode = rightIndex;
                        node = rightNode;
                    }
                    currentNodeReady = true;
                } else if (leftEntry >= 0.0) {
                    currentNode = leftIndex;
                    node = leftNode;
                    currentNodeReady = true;
                } else if (rightEntry >= 0.0) {
                    currentNode = rightIndex;
                    node = rightNode;
                    currentNodeReady = true;
                }
                continue;
            }
        }
        uint requestedEnd = node.first + node.count;
        bool truncated = requestedEnd < node.first || node.first > params.frame.z
            || requestedEnd > params.frame.z || node.count > MAX_LEAF_INSTANCES;
        uint instanceBegin = min(node.first, params.frame.z);
        uint instanceEnd = min(requestedEnd, params.frame.z);
        uint boundedEnd = min(instanceEnd, instanceBegin + MAX_LEAF_INSTANCES);
        incomplete = incomplete || truncated || boundedEnd < instanceEnd;
        [loop] for (uint instance = instanceBegin; instance < boundedEnd; ++instance) {
            bool instanceSupported;
            bool blocked = occludedInstance(origin,direction,instance,limit,
                                             geometryHeader,instanceSupported);
            incomplete = incomplete || !instanceSupported;
            if (blocked) {
                // Any accepted blocker proves the result even if another branch
                // could not be fully inspected.
                querySupported = true;
                return true;
            }
        }
    }
    querySupported = !incomplete;
    return false;
#endif
}

#ifndef GI_TRAVERSAL_CHECK
// Deferred primary traces run hit shading twice. Recording stores each
// visibility query instead of tracing it and answers "visible"; that answer
// is discarded with the rest of the recording run. Queries depend only on
// the hit and earlier query positions, never on earlier answers, so replay
// issues the same sequence (or a prefix, when a query is unsupported) and
// reads the answers the visibility pass stored. Queries beyond the per-slot
// capacity are traced during replay.
static const uint VISIBILITY_TRACE = 0u;
static const uint VISIBILITY_RECORD = 1u;
static const uint VISIBILITY_REPLAY = 2u;
static uint visibilityMode = VISIBILITY_TRACE;
static uint visibilitySlot = 0u;
static uint visibilityQueries = 0u;
static const uint VISIBILITY_BLOCKED = 1u;
static const uint VISIBILITY_SUPPORTED = 2u;
static const uint VISIBILITY_INTERSECTS_ROOT = 4u;

void storeVisibilityQuery(uint query, float3 origin, float3 direction, float limit,
                          uint admissionLimit) {
    uint word = deferredQueryHead(query, params);
    [unroll] for (uint i = 0u; i < 3u; ++i) {
        heads[word + i].store(asuint(origin[i]));
        heads[word + 3u + i].store(asuint(direction[i]));
    }
    heads[word + 6u].store(asuint(limit));
    heads[word + 7u].store(admissionLimit);
}

// Trace one stored query and keep the root interval its dependency needs.
void traceVisibilityQuery(uint query) {
    uint word = deferredQueryHead(query, params);
    float3 origin, direction;
    [unroll] for (uint i = 0u; i < 3u; ++i) {
        origin[i] = asfloat(heads[word + i].load());
        direction[i] = asfloat(heads[word + 3u + i].load());
    }
    bool supported, intersectsRoot;
    float entry, exit;
    bool blocked = traceOccluded(origin, direction, asfloat(heads[word + 6u].load()),
        supported, heads[word + 7u].load(), intersectsRoot, entry, exit);
    heads[word + 8u].store((blocked ? VISIBILITY_BLOCKED : 0u)
        | (supported ? VISIBILITY_SUPPORTED : 0u)
        | (intersectsRoot ? VISIBILITY_INTERSECTS_ROOT : 0u));
    heads[word + 9u].store(asuint(entry));
    heads[word + 10u].store(asuint(exit));
}
#endif

bool traceOccludedTracked(float3 origin, float3 direction, float limit,
                          out bool querySupported, out DependencyContext dependency,
                          uint admissionLimit = 0u) {
    bool intersectsRoot;
    float entry, exit;
    bool blocked;
#ifndef GI_TRAVERSAL_CHECK
    uint deferredQuery = visibilityQueries;
    if (visibilityMode != VISIBILITY_TRACE) ++visibilityQueries;
    if (visibilityMode == VISIBILITY_RECORD) {
        if (deferredQuery < DEFERRED_VISIBILITY_QUERIES)
            storeVisibilityQuery(visibilitySlot * DEFERRED_VISIBILITY_QUERIES + deferredQuery,
                                 origin, direction, limit, admissionLimit);
        querySupported = true;
        dependency = dependencyEmpty();
        return false;
    }
    bool replayed = false;
    if (visibilityMode == VISIBILITY_REPLAY && deferredQuery < DEFERRED_VISIBILITY_QUERIES) {
        uint word = deferredQueryHead(
            visibilitySlot * DEFERRED_VISIBILITY_QUERIES + deferredQuery, params);
        float3 recordedOrigin, recordedDirection;
        [unroll] for (uint i = 0u; i < 3u; ++i) {
            recordedOrigin[i] = asfloat(heads[word + i].load());
            recordedDirection[i] = asfloat(heads[word + 3u + i].load());
        }
        // The recording ran in another kernel. Use its answer only for the
        // same ray; any divergence is traced here instead.
        if (distance(recordedDirection, direction) <= 1e-4
            && distance(recordedOrigin, origin) <= 1e-4 * max(1.0, length(origin))) {
            // The dependency segment follows the ray that was actually traced.
            origin = recordedOrigin;
            direction = recordedDirection;
            uint flags = heads[word + 8u].load();
            blocked = (flags & VISIBILITY_BLOCKED) != 0u;
            querySupported = (flags & VISIBILITY_SUPPORTED) != 0u;
            intersectsRoot = (flags & VISIBILITY_INTERSECTS_ROOT) != 0u;
            entry = asfloat(heads[word + 9u].load());
            exit = asfloat(heads[word + 10u].load());
            replayed = true;
        }
    }
    if (!replayed)
#endif
    blocked = traceOccluded(origin, direction, limit, querySupported, admissionLimit,
                            intersectsRoot, entry, exit);
    dependency = dependencyUnknown();
    if (!querySupported) return blocked;
    dependency = dependencyEmpty();
    if (!intersectsRoot) return blocked;
    dependencyIncludeSegment(dependency, origin + direction * max(entry, 0.0),
                             origin + direction * exit, 0.0);
    return blocked;
}

[ForceInline]
void dependencyIncludeVisibilityTarget(inout DependencyContext dependency,
                                       float3 target) {
    dependencyIncludeSegment(dependency, target, target, 0.0);
}

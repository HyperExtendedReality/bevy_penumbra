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
// counter is reset by the cache-clear pass once per frame. A zero budget keeps
// legacy settings unbounded while still recording attempts when the buffer is
// present.
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
    uint admitted = heads[admittedIndex].load();
    [loop] while (admitted < limit) {
        uint previous = heads[admittedIndex].compareExchange(admitted, admitted + 1u);
        if (previous == admitted) return true;
        admitted = previous;
    }
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
    uint4 value = geometryPageWord(index);
    if (params.geometryPages.x != 2) return value;
    uint4 header = geometryPageWord(0);
    if (index < header.y || index >= header.y + header.w * header.z) return value;
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
    return value;
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
Triangle localTriangle(uint index, uint4 header) {
    uint3 indices = triangleIndices(index, header);
    Triangle triangle;
    triangle.a = float4(asfloat(geometryWord(vertexBase(indices.x, header)).xyz), 0);
    triangle.b = float4(asfloat(geometryWord(vertexBase(indices.y, header)).xyz), 0);
    triangle.c = float4(asfloat(geometryWord(vertexBase(indices.z, header)).xyz), 0);
    triangle.albedo = 0; triangle.emission = 0;
    return triangle;
}
Triangle localTriangle(uint index) { return localTriangle(index, geometryWord(0)); }
float4 hitVertexAttribute(Hit hit, uint attribute) {
    uint3 indices = triangleIndices(hit.triangle);
    return asfloat(geometryWord(vertexBase(indices.x) + attribute)) * hit.barycentric.x
         + asfloat(geometryWord(vertexBase(indices.y) + attribute)) * hit.barycentric.y
         + asfloat(geometryWord(vertexBase(indices.z) + attribute)) * hit.barycentric.z;
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
uint instanceBase(uint instance) { return sceneWord(0).y + instance * sceneWord(0).w; }
uint2 hitInstanceIdentity(Hit hit) { return sceneWord(instanceBase(hit.instance) + 7).xy; }
bool instanceVisibleRecord(uint4 mask) {
    [loop] for (uint i = 0; i < min(mask.y, params.layers.y); ++i) {
        uint instanceBits = sceneWord(mask.x + i / 4)[i & 3];
        uint cameraBits = lights[params.layers.x + i / 4].source[i & 3];
        if ((instanceBits & cameraBits) != 0) return true;
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
    float4 uv; // UV0.xy, UV1.zw
    float4 color;
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
    uint3 indices = triangleIndices(hit.triangle);
    Triangle triangle = localTriangle(hit.triangle);
    float sign = (sceneWord(base + 6).w & 4) != 0 ? -1 : 1;
    HitAttributes attributes;
    attributes.geometricNormal = triangleGeometricNormal(base, sceneWord(base + 6).w, triangle);
    attributes.worldNormal = 0; attributes.worldTangent = 0;
    [loop] for (uint i = 0; i < 3; ++i) {
        uint vertex = vertexBase(indices[i]);
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
    attributes.uv = hitVertexAttribute(hit, 3);
    attributes.color = hitVertexAttribute(hit, 4);
    return attributes;
}
#include "material.slang"
float3 hitShadingNormal(Hit hit, HitAttributes attributes, float3 incoming) {
    return materialNormal(hit, attributes, incoming);
}
float3 hitEmission(Hit hit, float3 baseColor) {
    uint material = materialBase(hit);
    uint flags = sceneWord(instanceBase(hit.instance)+6).w;
    float3 emission = asfloat(sceneWord(material+1).xyz)*materialSample(hit,1).xyz;
    if ((flags&16) != 0) emission = baseColor;
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
    return materialAccepts(hit, materialAlphaCutoff(hit)) ? max(worldTriangle(hit).emission.xyz, 0) : float3(0);
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
bool boxRayInterval(float3 origin, float3 direction, BvhNode node, float limit,
                    out float entry, out float exit) {
    float near = 0.0, far = limit;
    [loop] for (uint axis = 0; axis < 3; ++axis) {
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
float3 intersectTriangle(float3 origin, float3 direction, Triangle triangle) {
    // Sheared edge functions give adjacent triangles matching edge decisions.
    // No absolute determinant epsilon: local directions retain world t and can
    // be arbitrarily short/long under nonuniform instance scale.
    float3 magnitude = abs(direction);
    uint kz = magnitude.x > magnitude.y ? (magnitude.x > magnitude.z ? 0 : 2) : (magnitude.y > magnitude.z ? 1 : 2);
    if (direction[kz] == 0) return float3(-1, 0, 0);
    uint kx = (kz + 1) % 3, ky = (kx + 1) % 3;
    if (direction[kz] < 0) { uint swap = kx; kx = ky; ky = swap; }
    float3 a = triangle.a.xyz - origin, b = triangle.b.xyz - origin, c = triangle.c.xyz - origin;
    float sx = direction[kx] / direction[kz], sy = direction[ky] / direction[kz];
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

Hit traceTriangle(float3 origin, float3 direction, float3 worldDirection, float coneWidth, float coneSpread,
                  uint triangleIndex, uint instance, uint base, uint flags, uint4 geometryHeader, Hit hit) {
    Triangle triangle = localTriangle(triangleIndex, geometryHeader);
    float facing = dot(cross(triangle.b.xyz-triangle.a.xyz, triangle.c.xyz-triangle.a.xyz), direction)
                   * ((flags & 4) != 0 ? -1 : 1);
    if (((flags & 1) != 0 && facing >= 0) || ((flags & 2) != 0 && facing <= 0)) return hit;
    float3 intersection = intersectTriangle(origin, direction, triangle);
    float distance = intersection.x;
    // Reject sub-precision self hits at translated triangle boundaries.
    if (distance > 0.000025 && distance < hit.distance) {
        Hit candidate = hit; candidate.triangle = triangleIndex; candidate.instance = instance;
        candidate.barycentric = float3(1 - intersection.y - intersection.z, intersection.yz);
        candidate.footprint = 0;
        float cutoff = materialAlphaCutoff(candidate);
        if (cutoff >= 0 && (coneWidth > 0 || coneSpread > 0)) {
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
    float3 localOrigin = transformPoint(base, origin);
    // Keep this unnormalized so triangle/box distances remain world distances.
    float3 localDirection = transformPoint(base, direction, true);
    uint flags = instanceRecord.w;
    uint stack[49]; uint count = 0;
    uint currentNode = sceneWord(base + 8).x;
    BvhNode node = sceneNode(currentNode);
    bool currentNodeReady = intersectBox(localOrigin, localDirection, node, hit.distance);
    uint traversalSteps = 0u;
    [loop] while (currentNodeReady || count != 0) {
        if (++traversalSteps > MAX_TRAVERSAL_STEPS) {
            hit.errors |= HIT_QUERY_UNSUPPORTED;
            break;
        }
        if (!currentNodeReady) {
            currentNode = stack[--count];
            node = sceneNode(currentNode);
            if (!intersectBox(localOrigin, localDirection, node, hit.distance)) continue;
        }
        currentNodeReady = false;
        if (node.count == 0) {
            if (count + 2 > 49) {
                // Unexpected topology cannot silently drop an occluder. Fall
                // back to this instance's primitive range and report the flag.
                hit.errors |= 1;
                uint4 geometry = sceneWord(base + 8);
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
        // Share the leaf scan with overflow fallback to avoid duplicating the
        // full material/intersection call graph in the driver compiler. The
        // range is clamped to this instance's primitives and to a fixed maximum
        // so a malformed node cannot turn one leaf into an unbounded scan.
        uint4 leafGeometry = sceneWord(base + 8);
        uint triangleBegin = max(node.first, leafGeometry.y);
        uint triangleEnd = min(node.first + node.count, leafGeometry.y + leafGeometry.z);
        triangleEnd = min(triangleEnd, triangleBegin + MAX_LEAF_TRIANGLES);
        [loop] for (uint i = triangleBegin; i < triangleEnd; ++i) {
            hit = traceTriangle(localOrigin, localDirection, direction, coneWidth, coneSpread,
                                i, instance, base, flags, geometryHeader, hit);
            if (anyHit && hit.triangle != EMPTY) return hit;
        }
    }
    return hit;
}
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
    uint4 geometryHeader = geometryWord(0);
    // The near-child walk keeps one deferred branch per depth while retaining
    // the original stack capacity and overflow fallback threshold.
    uint stack[49]; uint count = 0;
    uint currentNode = 0;
    BvhNode node = sceneNode(currentNode);
    float rootEntry, rootExit;
    bool intersectsRoot = boxRayInterval(origin, direction, node, hit.distance,
                                          rootEntry, rootExit);
    dependency = dependencyEmpty();
    if (!intersectsRoot) return hit;
    dependencyStart = max(rootEntry, 0.0);
    dependencyDistance = min(limit, rootExit);
    bool currentNodeReady = intersectBox(origin, direction, node, hit.distance);
    uint traversalSteps = 0u;
    [loop] while (currentNodeReady || count != 0) {
        if (++traversalSteps > MAX_TRAVERSAL_STEPS) {
            hit.errors |= HIT_QUERY_UNSUPPORTED;
            break;
        }
        if (!currentNodeReady) {
            currentNode = stack[--count];
            node = sceneNode(currentNode);
            if (!intersectBox(origin, direction, node, hit.distance)) continue;
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
        uint instanceBegin = min(node.first, params.frame.z);
        uint instanceEnd = min(node.first + node.count, params.frame.z);
        instanceEnd = min(instanceEnd, instanceBegin + MAX_LEAF_INSTANCES);
        [loop] for (uint i = instanceBegin; i < instanceEnd; ++i) {
            hit = traceInstance(origin, direction, i, hit, anyHit, coneWidth, coneSpread, geometryHeader);
            if (anyHit && hit.triangle != EMPTY) {
                count = 0;
                break;
            }
        }
    }
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

bool occludedTriangle(float3 origin, float3 direction, uint triangleIndex, uint instance, uint flags,
                      float limit, uint4 geometryHeader) {
    Triangle triangle = localTriangle(triangleIndex, geometryHeader);
    float facing = dot(cross(triangle.b.xyz-triangle.a.xyz,triangle.c.xyz-triangle.a.xyz),direction)
                   * ((flags & 4) != 0 ? -1 : 1);
    if (((flags & 1) != 0 && facing >= 0) || ((flags & 2) != 0 && facing <= 0)) return false;
    float3 intersection = intersectTriangle(origin,direction,triangle);
    if (!(intersection.x > 0.000025 && intersection.x < limit)) return false;
    Hit candidate; candidate.distance = intersection.x; candidate.triangle = triangleIndex;
    candidate.instance = instance; candidate.errors = 0;
    candidate.barycentric = float3(1-intersection.y-intersection.z,intersection.yz);
    candidate.footprint = 0;
    float cutoff = materialAlphaCutoff(candidate);
    if (cutoff < 0) return true;
    return materialAccepts(candidate,cutoff);
}

bool occludedInstance(float3 origin, float3 direction, uint instance, float limit, uint4 geometryHeader) {
    uint base = instanceBase(instance);
    uint4 instanceRecord = sceneWord(base+6);
    if (!instanceVisibleRecord(instanceRecord)) return false;
    float3 localOrigin = transformPoint(base,origin);
    float3 localDirection = transformPoint(base,direction,true);
    uint flags = instanceRecord.w;
    uint stack[49]; uint count = 0;
    uint currentNode = sceneWord(base+8).x;
    BvhNode node = sceneNode(currentNode);
    bool currentNodeReady = intersectBox(localOrigin,localDirection,node,limit);
    uint traversalSteps = 0u;
    [loop] while (currentNodeReady || count != 0) {
        if (++traversalSteps > MAX_TRAVERSAL_STEPS) return true;
        if (!currentNodeReady) {
            currentNode = stack[--count];
            node = sceneNode(currentNode);
            // The deferred node passed the unchanged ray/limit test at its parent.
        }
        currentNodeReady = false;
        if (node.count == 0) {
            if (count + 2 > 49) {
                uint4 geometry = sceneWord(base+8);
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
        // Share the material-tested scan with stack overflow to keep driver
        // compilation bounded. Clamp its range like the primary traversal.
        uint4 leafGeometry = sceneWord(base+8);
        uint triangleBegin = max(node.first, leafGeometry.y);
        uint triangleEnd = min(node.first+node.count, leafGeometry.y+leafGeometry.z);
        triangleEnd = min(triangleEnd, triangleBegin + MAX_LEAF_TRIANGLES);
        [loop] for (uint i = triangleBegin; i < triangleEnd; ++i)
            if (occludedTriangle(localOrigin,localDirection,i,instance,flags,limit,geometryHeader)) return true;
    }
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
    querySupported = true;
    if (params.frame.z == 0) return false;
    if (admissionLimit == 0u) admissionLimit = ordinaryTransportAdmissionLimit();
    if (!admitTransportQuery(admissionLimit)) { querySupported = false; return false; }
    uint4 geometryHeader = geometryWord(0);
    uint stack[49]; uint count = 0;
    uint currentNode = 0;
    BvhNode node = sceneNode(currentNode);
    bool currentNodeReady = intersectBox(origin,direction,node,limit);
    uint traversalSteps = 0u;
    [loop] while (currentNodeReady || count != 0) {
        if (++traversalSteps > MAX_TRAVERSAL_STEPS) return true;
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
        uint occlusionBegin = min(node.first, params.frame.z);
        uint occlusionEnd = min(node.first+node.count, params.frame.z);
        occlusionEnd = min(occlusionEnd, occlusionBegin + MAX_LEAF_INSTANCES);
        [loop] for (uint instance = occlusionBegin; instance < occlusionEnd; ++instance)
            if (occludedInstance(origin,direction,instance,limit,geometryHeader)) return true;
    }
    return false;
}

bool traceOccludedTracked(float3 origin, float3 direction, float limit,
                          out bool querySupported, out DependencyContext dependency,
                          uint admissionLimit = 0u) {
    bool blocked = traceOccluded(origin, direction, limit, querySupported, admissionLimit);
    dependency = dependencyUnknown();
    if (!querySupported) return blocked;
    dependency = dependencyEmpty();
    if (params.frame.z == 0) return blocked;
    float entry, exit;
    if (!boxRayInterval(origin, direction, sceneNode(0), limit, entry, exit)) return blocked;
    dependencyIncludeSegment(dependency, origin + direction * max(entry, 0.0),
                             origin + direction * exit, 0.0);
    return blocked;
}


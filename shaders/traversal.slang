struct Hit { float distance; uint triangle; uint instance; uint errors; float3 barycentric; float footprint; };

#ifndef GI_TRAVERSAL_CHECK
[[vk::binding(28, 0)]] Texture2DArray<uint4> geometryPageTexture;
uint4 geometryPageWord(uint index) {
    uint side = params.geometryPages.y;
    uint pageWords = side * side;
    uint withinPage = index % pageWords;
    return geometryPageTexture.Load(int4(withinPage % side, withinPage / side, index / pageWords, 0));
}
uint4 geometryWord(uint index) {
    if (params.geometryPages.x == 0) return geometryWords[index];
    uint4 value = geometryPageWord(index);
    if (params.geometryPages.x != 2) return value;
    uint4 header = geometryPageWord(0);
    if (index < header.y || index >= header.y + header.w * header.z) return value;
    uint vertex = (index - header.y) / header.z;
    uint field = (index - header.y) % header.z;
    uint count = geometryWords[0].x, low = 0, high = count;
    [loop] while (low < high) {
        uint middle = (low + high) >> 1;
        uint4 range = geometryWords[geometryWords[0].y + middle];
        if (vertex >= range.x) low = middle + 1;
        else high = middle;
    }
    if (low > 0) {
        uint4 range = geometryWords[geometryWords[0].y + low - 1];
        if (vertex >= range.x && vertex < range.x + range.y)
            return geometryPageWord(geometryWords[0].z + (range.z + vertex - range.x) * 5 + field);
    }
    return value;
}
#else
uint4 geometryWord(uint index) { return geometryWords[index]; }
#endif

#ifndef GI_TRAVERSAL_CHECK
[[vk::binding(29, 0)]] Texture2DArray<uint4> scenePageTexture;
uint4 scenePageWord(uint index) {
    uint side = params.scenePages.y;
    uint pageWords = side * side;
    uint withinPage = index % pageWords;
    return scenePageTexture.Load(int4(withinPage % side, withinPage / side, index / pageWords, 0));
}
uint4 sceneWord(uint index) {
    if (params.scenePages.x == 0) return sceneWords[index];
    uint prefixWords = params.scenePages.z;
    uint rangeCount = params.scenePages.w;
    uint instanceBase = params.sceneOverlay.x;
    uint dynamicInstanceWords = params.sceneOverlay.y;
    uint staticBlasWords = params.sceneOverlay.z;
    uint mutableBlasWords = params.sceneOverlay.w;
    if (index < prefixWords) return sceneWords[index];
    uint nodeBase = prefixWords + rangeCount;
    uint instanceOverlayBase = nodeBase + mutableBlasWords;
    if (index >= instanceBase && index < instanceBase + dynamicInstanceWords)
        return sceneWords[instanceOverlayBase + index - instanceBase];
    if (index < instanceBase) {
        uint localWord = index - prefixWords;
        uint low = 0, high = rangeCount;
        [loop] while (low < high) {
            uint middle = (low + high) >> 1;
            uint4 range = sceneWords[prefixWords + middle];
            if (localWord >= range.x * 2) low = middle + 1;
            else high = middle;
        }
        uint mutableBefore = 0;
        if (low > 0) {
            uint4 range = sceneWords[prefixWords + low - 1];
            uint rangeStart = range.x * 2;
            uint rangeEnd = rangeStart + range.y * 2;
            if (localWord < rangeEnd)
                return sceneWords[range.z + localWord - rangeStart];
            mutableBefore = range.w + range.y;
        }
        return scenePageWord(localWord - mutableBefore * 2);
    }
    return scenePageWord(staticBlasWords + index - instanceBase - dynamicInstanceWords);
}
#else
uint4 sceneWord(uint index) { return sceneWords[index]; }
#endif

uint3 triangleIndices(uint triangle) { return geometryWord(geometryWord(0).x + triangle).xyz; }
uint triangleSurfaceGroup(uint triangle) { return geometryWord(geometryWord(0).x + triangle).w; }
uint vertexBase(uint vertex) {
    uint4 header = geometryWord(0);
    return header.y + vertex * header.z;
}
Triangle localTriangle(uint index) {
    uint3 indices = triangleIndices(index);
    Triangle triangle;
    triangle.a = float4(asfloat(geometryWord(vertexBase(indices.x)).xyz), 0);
    triangle.b = float4(asfloat(geometryWord(vertexBase(indices.y)).xyz), 0);
    triangle.c = float4(asfloat(geometryWord(vertexBase(indices.z)).xyz), 0);
    triangle.albedo = 0; triangle.emission = 0;
    return triangle;
}
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
bool instanceVisible(uint instance) {
    uint4 mask = sceneWord(instanceBase(instance) + 6);
    [loop] for (uint i = 0; i < min(mask.y, params.layers.y); ++i) {
        uint instanceBits = sceneWord(mask.x + i / 4)[i & 3];
        uint cameraBits = lights[params.layers.x + i / 4].source[i & 3];
        if ((instanceBits & cameraBits) != 0) return true;
    }
    return false;
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
    emission *= lerp(1/max(params.environment.w,1e-20),1.0,exposureWeight);
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
    triangle.albedo = float4(color.xyz * (1 - metallic), float(sceneWord(base + 6).w & 3));
    if ((sceneWord(base + 6).w & 16) != 0) triangle.albedo.xyz = 0;
    uint4 geometry = sceneWord(base + 8);
    uint local = hit.triangle - geometry.y;
    float probability = geometry.w == 0 ? 0 : asfloat(sceneWord(geometry.w + local / 4)[local & 3]);
    float3 emission = hitEmission(hit,color.xyz);
    triangle.emission = float4(emission, probability);
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
    return triangle.albedo.w == 0 ? abs(c) : (triangle.albedo.w == 2 ? -c : c);
}
float sourceCosine(Light light, float3 direction) {
    float c = dot(light.direction.xyz, -direction);
    return light.position.w == -2 ? abs(c) : c;
}

[ForceInline]
float boxEntryDistance(float3 origin, float3 direction, BvhNode node, float limit) {
    float near = 0.0, far = limit;
    [loop] for (uint axis = 0; axis < 3; ++axis) {
        if (direction[axis] == 0) {
            if (origin[axis] < node.minimum[axis] || origin[axis] > node.maximum[axis]) return -1.0;
        } else {
            float a = (node.minimum[axis] - origin[axis]) / direction[axis];
            float b = (node.maximum[axis] - origin[axis]) / direction[axis];
            float exit = max(a, b);
            // Conservative far endpoint accounts for slab division roundoff.
            near = max(near, min(a, b)); far = min(far, exit + abs(exit) * 3.5762791e-7);
        }
    }
    return near <= far ? near : -1.0;
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
                  uint triangleIndex, uint instance, uint flags, Hit hit) {
    Triangle triangle = localTriangle(triangleIndex);
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
            float cosine = abs(dot(triangleGeometricNormal(instanceBase(instance), flags, triangle), worldDirection));
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
                  float coneWidth, float coneSpread) {
    if (!instanceVisible(instance)) return hit;
    uint base = instanceBase(instance);
    float3 localOrigin = transformPoint(base, origin);
    // Keep this unnormalized so triangle/box distances remain world distances.
    float3 localDirection = transformPoint(base, direction, true);
    uint flags = sceneWord(base + 6).w;
    uint stack[49]; uint count = 1; stack[0] = sceneWord(base + 8).x;
    [loop] while (count != 0) {
        BvhNode node = sceneNode(stack[--count]);
        if (!intersectBox(localOrigin, localDirection, node, hit.distance)) continue;
        if (node.count == 0) {
            if (count + 2 > 49) {
                // Unexpected topology cannot silently drop an occluder. Fall
                // back to this instance's primitive range and report the flag.
                hit.errors |= 1;
                uint4 geometry = sceneWord(base + 8);
                node.first = geometry.y; node.count = geometry.z;
                count = 0;
            } else {
                float leftEntry = boxEntryDistance(localOrigin, localDirection, sceneNode(node.first), hit.distance);
                float rightEntry = boxEntryDistance(localOrigin, localDirection, sceneNode(node.first + 1), hit.distance);
                if (leftEntry >= 0.0 && rightEntry >= 0.0) {
                    if (leftEntry <= rightEntry) {
                        stack[count++] = node.first + 1;
                        stack[count++] = node.first;
                    } else {
                        stack[count++] = node.first;
                        stack[count++] = node.first + 1;
                    }
                } else if (leftEntry >= 0.0) {
                    stack[count++] = node.first;
                } else if (rightEntry >= 0.0) {
                    stack[count++] = node.first + 1;
                }
                continue;
            }
        }
        // Share the leaf scan with overflow fallback to avoid duplicating the
        // full material/intersection call graph in the driver compiler.
        [loop] for (uint i = node.first; i < node.first + node.count; ++i) {
            hit = traceTriangle(localOrigin, localDirection, direction, coneWidth, coneSpread,
                                i, instance, flags, hit);
            if (anyHit && hit.triangle != EMPTY) return hit;
        }
    }
    return hit;
}
Hit traceRay(float3 origin, float3 direction, float limit, bool anyHit = false, float coneWidth = 0, float coneSpread = 0) {
    Hit hit; hit.distance = limit; hit.triangle = EMPTY; hit.instance = EMPTY; hit.errors = 0; hit.barycentric = 0; hit.footprint = 0;
    if (params.frame.z == 0) return hit;
    // A depth-48 binary tree can leave 49 pending nodes on the DFS stack.
    uint stack[49]; uint count = 1; stack[0] = 0;
    [loop] while (count != 0) {
        BvhNode node = sceneNode(stack[--count]);
        if (!intersectBox(origin, direction, node, hit.distance)) continue;
        if (node.count == 0) {
            if (count + 2 > 49) {
                hit.errors |= 2;
                node.first = 0; node.count = params.frame.z;
                count = 0;
            } else {
                float leftEntry = boxEntryDistance(origin, direction, sceneNode(node.first), hit.distance);
                float rightEntry = boxEntryDistance(origin, direction, sceneNode(node.first + 1), hit.distance);
                if (leftEntry >= 0.0 && rightEntry >= 0.0) {
                    if (leftEntry <= rightEntry) {
                        stack[count++] = node.first + 1;
                        stack[count++] = node.first;
                    } else {
                        stack[count++] = node.first;
                        stack[count++] = node.first + 1;
                    }
                } else if (leftEntry >= 0.0) {
                    stack[count++] = node.first;
                } else if (rightEntry >= 0.0) {
                    stack[count++] = node.first + 1;
                }
                continue;
            }
        }
        [loop] for (uint i = node.first; i < node.first + node.count; ++i) {
            hit = traceInstance(origin, direction, i, hit, anyHit, coneWidth, coneSpread);
            if (anyHit && hit.triangle != EMPTY) return hit;
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
    return hit;
}

bool occludedTriangle(float3 origin, float3 direction, uint triangleIndex, uint instance, uint flags, float limit) {
    Triangle triangle = localTriangle(triangleIndex);
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

bool occludedInstance(float3 origin, float3 direction, uint instance, float limit) {
    if (!instanceVisible(instance)) return false;
    uint base = instanceBase(instance);
    float3 localOrigin = transformPoint(base,origin);
    float3 localDirection = transformPoint(base,direction,true);
    uint flags = sceneWord(base+6).w;
    uint stack[49]; uint count = 1; stack[0] = sceneWord(base+8).x;
    [loop] while (count != 0) {
        BvhNode node = sceneNode(stack[--count]);
        if (!intersectBox(localOrigin,localDirection,node,limit)) continue;
        if (node.count == 0) {
            if (count + 2 > 49) {
                uint4 geometry = sceneWord(base+8);
                node.first = geometry.y; node.count = geometry.z;
                count = 0;
            } else {
                stack[count++] = node.first; stack[count++] = node.first+1;
                continue;
            }
        }
        // Share the material-tested scan with stack overflow to keep driver compilation bounded.
        [loop] for (uint i = node.first; i < node.first+node.count; ++i)
            if (occludedTriangle(localOrigin,localDirection,i,instance,flags,limit)) return true;
    }
    return false;
}

bool traceOccluded(float3 origin, float3 direction, float limit) {
    if (params.frame.z == 0) return false;
    uint stack[49]; uint count = 1; stack[0] = 0;
    [loop] while (count != 0) {
        BvhNode node = sceneNode(stack[--count]);
        if (!intersectBox(origin,direction,node,limit)) continue;
        if (node.count == 0) {
            if (count + 2 > 49) {
                node.first = 0; node.count = params.frame.z;
                count = 0;
            } else {
                stack[count++] = node.first; stack[count++] = node.first+1;
                continue;
            }
        }
        [loop] for (uint instance = node.first; instance < node.first+node.count; ++instance)
            if (occludedInstance(origin,direction,instance,limit)) return true;
    }
    return false;
}


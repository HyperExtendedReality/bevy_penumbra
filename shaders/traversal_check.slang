// Test-only entry point: exercise the exact production traversal with known rays.
#include "common.slang"
struct CheckParameters { uint4 frame; float4 tuning; uint4 size; uint4 layers; };
[[vk::binding(0, 0)]] ConstantBuffer<CheckParameters> params;
[[vk::binding(1, 0)]] StructuredBuffer<uint4> geometryWords;
[[vk::binding(2, 0)]] StructuredBuffer<uint4> sceneWords;
[[vk::binding(3, 0)]] StructuredBuffer<float4> rays;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint4> results;
[[vk::binding(5, 0)]] Texture2DArray<float4> materialAtlas;
[[vk::binding(6, 0)]] Texture2D<float4> nativeTexture;
[[vk::binding(7, 0)]] SamplerState nativeSampler;
[[vk::binding(8, 0)]] StructuredBuffer<Light> lights;
#define GI_TRAVERSAL_CHECK
#include "traversal.slang"
[shader("compute")]
[numthreads(64, 1, 1)]
void checkRayGeometry(uint3 id : SV_DispatchThreadID) {
    uint count, stride; rays.GetDimensions(count, stride);
    if (id.x >= count / 6) return;
    uint base = id.x * 6;
    Triangle triangle = (Triangle)0;
    triangle.a = rays[base]; triangle.b = rays[base + 1]; triangle.c = rays[base + 2];
    float3 intersection = intersectTriangle(rays[base + 3].xyz, rays[base + 4].xyz, triangle);
    results[id.x * 2] = asuint(float4(intersection, 0));
    results[id.x * 2 + 1] = asuint(float4(rayOffset(rays[base + 3].xyz, rays[base + 5].xyz, rays[base + 5].w), 0));
}
[shader("compute")]
[numthreads(64, 1, 1)]
void checkTextureSampling(uint3 id : SV_DispatchThreadID) {
    if (id.x >= params.size.x) return;
    float4 query = rays[id.x];
    results[id.x * 2] = asuint(sampleTexture(params.frame.x, query.xy, query.z));
    results[id.x * 2 + 1] = asuint(nativeTexture.SampleLevel(nativeSampler, query.xy, query.z));
}
[shader("compute")]
[numthreads(64, 1, 1)]
void checkTraversal(uint3 id : SV_DispatchThreadID) {
    if (id.x >= params.size.x) return;
    float4 origin = rays[id.x * 2], direction = rays[id.x * 2 + 1];
    Hit hit = traceRay(origin.xyz, direction.xyz, origin.w, direction.w != 0);
    bool occluded = traceOccluded(origin.xyz,direction.xyz,origin.w);
    uint output = id.x * 11;
    results[output] = uint4(asuint(hit.distance), hit.triangle, hit.instance, hit.errors);
    if (hit.triangle != EMPTY) {
        HitAttributes attributes = hitAttributes(hit);
        results[output + 2] = asuint(float4(hit.barycentric, 0));
        results[output + 3] = asuint(float4(attributes.geometricNormal, 0));
        results[output + 4] = asuint(float4(attributes.worldNormal, 0));
        results[output + 5] = asuint(attributes.worldTangent);
        results[output + 6] = asuint(attributes.uv);
        results[output + 7] = asuint(attributes.color);
        results[output + 8] = asuint(float4(hitShadingNormal(hit, attributes, direction.xyz), 0));
        Triangle triangle = worldTriangle(hit);
        results[output + 9] = asuint(triangle.albedo);
        results[output + 10] = asuint(triangle.emission);
    }
    // A GPU brute-force query separates box/traversal failures from numerical
    // intersection differences in the independent f64 CPU comparison.
    Hit brute; brute.distance = origin.w; brute.triangle = EMPTY; brute.instance = EMPTY; brute.errors = 0; brute.barycentric = 0; brute.footprint = 0;
    for (uint instance = 0; instance < params.frame.z; ++instance) {
        if (!instanceVisible(instance)) continue;
        uint base = instanceBase(instance);
        float3 localOrigin = transformPoint(base, origin.xyz);
        float3 localDirection = transformPoint(base, direction.xyz, true);
        uint4 instanceGeometry = sceneWord(base + 8);
        uint4 geometryHeader = geometryWord(0);
        uint flags = sceneWord(base + 6).w;
        for (uint triangle = instanceGeometry.y;
             triangle < instanceGeometry.y + instanceGeometry.z; ++triangle) {
            brute = traceTriangle(localOrigin, localDirection, direction.xyz, 0.0, 0.0,
                                  triangle, instance, base, flags, geometryHeader, brute);
        }
    }
    results[output + 1] = uint4(asuint(brute.distance), brute.triangle, brute.instance,
        uint(occluded != (hit.triangle != EMPTY)));
}

// Focused fixture for geometry-dependency bounds. Keep this on the production
// tracked traversal entry points so a miss, closest hit, and NEE visibility
// query all exercise their real dependency propagation.
[shader("compute")]
[numthreads(64, 1, 1)]
void checkTrackedTraversal(uint3 id : SV_DispatchThreadID) {
    if (id.x >= params.size.x) return;
    float4 origin = rays[id.x * 2], direction = rays[id.x * 2 + 1];
    DependencyContext rayDependency;
    float dependencyStart, dependencyDistance, dependencyPadding;
    Hit hit = traceRayTracked(origin.xyz, direction.xyz, origin.w,
                              rayDependency, dependencyStart, dependencyDistance,
                              dependencyPadding, false, 0.0, 0.0, 0u);
    bool querySupported;
    DependencyContext shadowDependency;
    bool blocked = traceOccludedTracked(origin.xyz, direction.xyz, origin.w,
                                        querySupported, shadowDependency, 0u);
    uint output = id.x * 11u;
    results[output] = uint4(asuint(hit.distance), hit.triangle, hit.instance, hit.errors);
    uint rayFlags = (rayDependency.known ? 1u : 0u)
        | (rayDependency.hasBounds ? 2u : 0u);
    uint shadowFlags = (shadowDependency.known ? 1u : 0u)
        | (shadowDependency.hasBounds ? 2u : 0u);
    results[output + 1u] = uint4(asuint(rayDependency.minimum), rayFlags);
    results[output + 2u] = uint4(asuint(rayDependency.maximum), rayFlags);
    results[output + 3u] = uint4(uint(blocked), uint(querySupported), shadowFlags, 0u);
    results[output + 4u] = uint4(asuint(shadowDependency.minimum), shadowFlags);
    results[output + 5u] = uint4(asuint(shadowDependency.maximum), shadowFlags);
    Entry stored = (Entry)0;
    storeEntryDependency(stored, rayDependency);
    results[output + 6u] = asuint(stored.dependencyMin);
    results[output + 7u] = asuint(stored.dependencyMax);
    DependencyContext unknownUnion = rayDependency;
    dependencyUnion(unknownUnion, dependencyUnknown());
    storeEntryDependency(stored, unknownUnion);
    results[output + 8u] = asuint(stored.dependencyMin);
    results[output + 9u] = asuint(stored.dependencyMax);
}

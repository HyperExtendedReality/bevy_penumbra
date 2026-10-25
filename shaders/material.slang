// Integer indirection and explicit filtering avoid nonuniform descriptor indexing.
uint materialBase(Hit hit) { return sceneWord(instanceBase(hit.instance) + 7).w; }
uint materialTexture(Hit hit, uint slot) {
    uint base = materialBase(hit);
    return slot < 4 ? sceneWord(base + 6)[slot] : sceneWord(base + 7).x;
}
float2 transformUv(uint base, float2 uv) {
    float3 value = float3(uv, 1);
    return float2(dot(asfloat(sceneWord(base + 4).xyz), value), dot(asfloat(sceneWord(base + 5).xyz), value));
}
float2 materialUv(Hit hit, uint slot) {
    float4 uv = hitVertexAttribute(hit, 3);
    uint base = materialBase(hit);
    return transformUv(base, ((sceneWord(base + 7).y >> slot) & 1) != 0 ? uv.zw : uv.xy);
}
int textureIndex(int index, int length, uint mode) {
    // UV normalization bounds taps to [-1,length]. Keep remainder operands
    // nonnegative: negative signed remainder failed on the tested Vulkan backend.
    if (mode == 1) return int(uint(index + length) % uint(length));
    if (mode == 2) {
        index = int(uint(index + length * 2) % uint(length * 2));
        return index < length ? index : length * 2 - 1 - index;
    }
    return clamp(index, 0, length - 1);
}
float textureUv(float uv, uint mode) {
    if (mode == 1) return frac(uv);
    if (mode == 2) { float value = frac(uv * 0.5) * 2; return value <= 1 ? value : 2 - value; }
    return saturate(uv);
}
float4 textureTexel(uint4 mip, int2 texel, uint flags) {
    // A zero-sized mip record means this slot has no usable texture data. Index
    // arithmetic on it would overflow into an out-of-range scene/material read,
    // which can fault the GPU instead of returning a texel.
    if (mip.x == 0 || mip.y == 0) return 1;
    int2 wrapped = int2(textureIndex(texel.x, int(mip.x), flags & 3),
                        textureIndex(texel.y, int(mip.y), (flags >> 2) & 3));
    uint width, height, pages; materialAtlas.GetDimensions(width, height, pages);
    if (width <= 2 || pages == 0) return 1;
    uint tileSide = width - 2;
    uint2 tile = uint2(wrapped) / tileSide;
    uint4 rectangle = sceneWord(sceneWord(0).x + mip.w + tile.y * mip.z + tile.x);
    if (rectangle.z >= pages) return 1;
    uint2 pixel = rectangle.xy + 1 + uint2(wrapped) % tileSide;
    return materialAtlas.Load(int4(int2(pixel), int(rectangle.z), 0));
}
float4 textureMip(uint texture, uint level, float2 uv, bool linear) {
    uint base = sceneWord(0).x;
    uint4 record = sceneWord(base + texture);
    if (record.y == 0) return 1;
    level = min(level, record.y - 1);
    uint4 mip = sceneWord(base + record.x + level);
    uv = float2(textureUv(uv.x, record.z & 3), textureUv(uv.y, (record.z >> 2) & 3));
    float2 position = uv * float2(mip.xy);
    if (!linear) return textureTexel(mip, int2(floor(position)), record.z);
    position -= 0.5;
    int2 low = int2(floor(position));
    float2 weight = frac(position);
    return lerp(lerp(textureTexel(mip, low, record.z), textureTexel(mip, low + int2(1, 0), record.z), weight.x),
                lerp(textureTexel(mip, low + int2(0, 1), record.z), textureTexel(mip, low + 1, record.z), weight.x), weight.y);
}
float4 sampleTexture(uint texture, float2 uv, float lod = 0) {
    if (texture == EMPTY) return 1;
    uint base = sceneWord(0).x;
    uint4 record = sceneWord(base + texture);
    uint4 size = sceneWord(base + texture + 1);
    // Without a mip chain there is no valid level, and a non-finite LOD would
    // convert to an out-of-range unsigned index below. Fall back to the
    // neutral texel both cases would otherwise invent an address for.
    if (record.y == 0) return 1;
    if (!isfinite(lod)) lod = 0;
    float maximum = min(asfloat(size.w), float(record.y - 1));
    lod = clamp(lod, min(asfloat(size.z), maximum), maximum);
    if (!(lod >= 0.0)) lod = 0;
    bool linear = (record.z & (lod > 0 ? 32 : 16)) != 0;
    if ((record.z & 64) == 0) return textureMip(texture, uint(floor(lod + 0.5)), uv, linear);
    uint low = uint(floor(lod)), high = min(low + 1, record.y - 1);
    return lerp(textureMip(texture, low, uv, linear), textureMip(texture, high, uv, linear), frac(lod));
}
float textureLod(Hit hit, uint slot) {
    uint texture = materialTexture(hit, slot);
    if (texture == EMPTY || hit.footprint <= 0) return 0;
    uint3 vertices = triangleIndices(hit.triangle);
    uint base = materialBase(hit);
    uint channel = (sceneWord(base + 7).y >> slot) & 1;
    float2 uv[3]; float3 positions[3];
    for (uint i = 0; i < 3; ++i) {
        uint vertex = vertexBase(vertices[i]);
        float4 coordinates = asfloat(geometryWord(vertex + 3));
        uv[i] = transformUv(base, channel != 0 ? coordinates.zw : coordinates.xy);
        positions[i] = transformPoint(instanceBase(hit.instance) + 3, asfloat(geometryWord(vertex).xyz));
    }
    float3 a = positions[1] - positions[0], b = positions[2] - positions[0];
    float3 crossEdges = cross(a, b);
    float areaSquared = dot(crossEdges, crossEdges);
    if (areaSquared <= 0 || !isfinite(areaSquared)) return 0;
    float3 gradientU = cross(b, crossEdges) / areaSquared;
    float3 gradientV = cross(crossEdges, a) / areaSquared;
    float2 deltaA = uv[1] - uv[0], deltaB = uv[2] - uv[0];
    uint2 size = sceneWord(sceneWord(0).x + texture + 1).xy;
    float texels = hit.footprint * max(length(gradientU * deltaA.x + gradientV * deltaB.x) * float(size.x),
                                      length(gradientU * deltaA.y + gradientV * deltaB.y) * float(size.y));
    return texels > 0 ? max(0.0, log2(texels)) : 0;
}
float4 materialSample(Hit hit, uint slot) {
#ifdef GI_WARMUP
    return 1;
#else
    return sampleTexture(materialTexture(hit, slot), materialUv(hit, slot), textureLod(hit, slot));
#endif
}
float4 materialBaseColor(Hit hit) {
    return asfloat(sceneWord(materialBase(hit))) * hitVertexAttribute(hit, 4) * materialSample(hit, 0);
}
float materialPerceptualRoughness(Hit hit) {
    uint base = materialBase(hit);
    float roughness = asfloat(sceneWord(base + 2).y) * materialSample(hit, 2).g;
    return isfinite(roughness) ? saturate(roughness) : 1.0;
}
float materialAlphaCutoff(Hit hit) {
    return asfloat(sceneWord(materialBase(hit) + 1).w);
}
bool materialAccepts(Hit hit, float cutoff) {
    if (cutoff < 0) return true;
    uint base = materialBase(hit);
    float alpha = asfloat(sceneWord(base)).w * hitVertexAttribute(hit, 4).w;
    return alpha * materialSample(hit, 0).w >= cutoff;
}
float3 materialNormal(Hit hit, HitAttributes attributes, float3 incoming) {
    uint base = materialBase(hit);
    bool back = dot(attributes.geometricNormal, incoming) > 0;
    bool doubleSided = (sceneWord(instanceBase(hit.instance) + 6).w & 8) != 0;
    if (materialTexture(hit, 3) == EMPTY) {
        float3 normal = unitVector(attributes.worldNormal, attributes.geometricNormal);
        return back && doubleSided ? -normal : normal;
    }
    float3 mapped = materialSample(hit, 3).xyz * 2 - 1;
    uint flags = sceneWord(base + 7).z;
    if ((flags & 2) != 0) mapped.z = sqrt(max(0.0, 1 - dot(mapped.xy, mapped.xy)));
    if ((flags & 1) != 0) mapped.y = -mapped.y;
    if (back && doubleSided) mapped = -mapped;
    // MikkTSpace retains interpolated lengths and does not orthogonalize TBN.
    float3 tangent = attributes.worldTangent.xyz;
    float3 bitangent = attributes.worldTangent.w * cross(attributes.worldNormal, tangent);
    return unitVector(tangent * mapped.x + bitangent * mapped.y + attributes.worldNormal * mapped.z,
                      attributes.geometricNormal);
}

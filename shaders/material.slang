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
float4 textureTexel(uint4 mip, int2 texel, uint flags, uint base, uint tileSide, uint pages) {
    // A zero-sized mip record means this slot has no usable texture data. Index
    // arithmetic on it would overflow into an out-of-range scene/material read,
    // which can fault the GPU instead of returning a texel.
    if (mip.x == 0 || mip.y == 0) return 1;
    int2 wrapped = int2(textureIndex(texel.x, int(mip.x), flags & 3),
                        textureIndex(texel.y, int(mip.y), (flags >> 2) & 3));
    if (tileSide == 0 || pages == 0) return 1;
    uint2 tile = uint2(wrapped) / tileSide;
    uint4 rectangle = sceneWord(base + mip.w + tile.y * mip.z + tile.x);
    if (rectangle.z >= pages) return 1;
    uint2 pixel = rectangle.xy + 1 + uint2(wrapped) % tileSide;
    return materialAtlas.Load(int4(int2(pixel), int(rectangle.z), 0));
}
float4 textureMip(uint base, uint4 record, uint level, float2 uv, bool linear,
                  uint tileSide, uint pages) {
    level = min(level, record.y - 1);
    uint4 mip = sceneWord(base + record.x + level);
    uv = float2(textureUv(uv.x, record.z & 3), textureUv(uv.y, (record.z >> 2) & 3));
    float2 position = uv * float2(mip.xy);
    if (!linear) return textureTexel(mip, int2(floor(position)), record.z, base, tileSide, pages);
    position -= 0.5;
    int2 low = int2(floor(position));
    float2 weight = frac(position);
    // Each packed tile carries source-correct one-texel gutters, including
    // wrapped edges and corners. Anchor inside the image, then fetch all four
    // taps from that tile rather than resolving four independent rectangles.
    if (mip.x == 0 || mip.y == 0 || tileSide == 0 || pages == 0) return 1;
    int2 anchor = clamp(low, int2(0, 0), int2(mip.xy) - 1);
    uint2 tile = uint2(anchor) / tileSide;
    uint4 rectangle = sceneWord(base + mip.w + tile.y * mip.z + tile.x);
    if (rectangle.z >= pages) return 1;
    int2 pixel = int2(rectangle.xy) + 1 + int2(uint2(anchor) % tileSide) + low - anchor;
    int page = int(rectangle.z);
    return lerp(lerp(materialAtlas.Load(int4(pixel, page, 0)),
                     materialAtlas.Load(int4(pixel + int2(1, 0), page, 0)), weight.x),
                lerp(materialAtlas.Load(int4(pixel + int2(0, 1), page, 0)),
                     materialAtlas.Load(int4(pixel + 1, page, 0)), weight.x), weight.y);
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
    uint width, height, pages;
    materialAtlas.GetDimensions(width, height, pages);
    uint tileSide = width > 2 ? width - 2 : 0;
    if (!isfinite(lod)) lod = 0;
    float maximum = min(asfloat(size.w), float(record.y - 1));
    lod = clamp(lod, min(asfloat(size.z), maximum), maximum);
    if (!(lod >= 0.0)) lod = 0;
    bool linear = (record.z & (lod > 0 ? 32 : 16)) != 0;
    if ((record.z & 64) == 0)
        return textureMip(base, record, uint(floor(lod + 0.5)), uv, linear, tileSide, pages);
    uint low = uint(floor(lod)), high = min(low + 1, record.y - 1);
    float4 lowSample = textureMip(base, record, low, uv, linear, tileSide, pages);
    // Keep the lerp even when both levels coincide, preserving its floating
    // point behavior while avoiding a duplicate mip/atlas lookup.
    float4 highSample = high == low ? lowSample
        : textureMip(base, record, high, uv, linear, tileSide, pages);
    return lerp(lowSample, highSample, frac(lod));
}
float textureLod(Hit hit, uint slot, uint texture) {
    if (hit.footprint <= 0) return 0;
    uint4 geometryHeader = geometryWord(0);
    uint3 vertices = triangleIndices(hit.triangle, geometryHeader);
    uint base = materialBase(hit);
    uint channel = (sceneWord(base + 7).y >> slot) & 1;
    float3 uvRow0 = asfloat(sceneWord(base + 4).xyz);
    float3 uvRow1 = asfloat(sceneWord(base + 5).xyz);
    uint instance = instanceBase(hit.instance) + 3;
    float4 row0 = asfloat(sceneWord(instance));
    float4 row1 = asfloat(sceneWord(instance + 1));
    float4 row2 = asfloat(sceneWord(instance + 2));
    float2 uv[3]; float3 positions[3];
    for (uint i = 0; i < 3; ++i) {
        uint vertex = vertexBase(vertices[i], geometryHeader);
        float4 coordinates = asfloat(geometryWord(vertex + 3));
        float3 uvValue = float3(channel != 0 ? coordinates.zw : coordinates.xy, 1);
        uv[i] = float2(dot(uvRow0, uvValue), dot(uvRow1, uvValue));
        float4 position = float4(asfloat(geometryWord(vertex).xyz), 1);
        positions[i] = float3(dot(row0, position), dot(row1, position), dot(row2, position));
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
    uint texture = materialTexture(hit, slot);
    if (texture == EMPTY) return 1;
    return sampleTexture(texture, materialUv(hit, slot), textureLod(hit, slot, texture));
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
// Opaque uses -1, while -2 denotes fractional alpha coverage. Other values
// are Bevy's binary mask cutoff.
float materialCoverage(Hit hit, float cutoff) {
    if (cutoff == -1.0) return 1.0;
    uint base = materialBase(hit);
    float alpha = asfloat(sceneWord(base)).w * hitVertexAttribute(hit, 4).w
        * materialSample(hit, 0).w;
    if (cutoff == -2.0) return isfinite(alpha) ? saturate(alpha) : 0.0;
    return alpha >= cutoff ? 1.0 : 0.0;
}
bool materialAccepts(Hit hit, float cutoff) {
    float coverage = materialCoverage(hit, cutoff);
    if (coverage <= 0.0) return false;
    if (coverage >= 1.0) return true;
    // A per-intersection, per-frame Bernoulli sample estimates fractional
    // coverage without forcing an alpha threshold on blended materials.
    int4 key = int4(asint(hit.barycentric.y), asint(hit.barycentric.z),
        int(hit.triangle), int(hit.instance ^ (params.frame.x * 0x9e3779b9u)));
    float sample = float(hashKey(key, 0x1000000u)) * (1.0 / 16777216.0);
    return sample < coverage;
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

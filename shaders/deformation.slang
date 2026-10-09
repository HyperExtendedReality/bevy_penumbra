// Dispatch vertices, then BLAS depths from leaves to roots, then refit TLAS.
struct DeformationParameters { uint4 dispatch; uint4 scenePages; uint4 sceneOverlay; };
[[vk::binding(0, 0)]] ConstantBuffer<DeformationParameters> params;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint4> geometryWords;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint4> sceneWords;
[[vk::binding(3, 0)]] StructuredBuffer<uint4> deformationWords;
[[vk::binding(4, 0)]] StructuredBuffer<uint4> poseWords;
[[vk::binding(5, 0)]] StructuredBuffer<uint> refitNodes;
[[vk::binding(6, 0)]] Texture2DArray<uint4> scenePageTexture;
[[vk::binding(7, 0)]] Texture2DArray<uint4> geometryPageTexture;
[[vk::binding(8, 0)]] RWTexture2DArray<uint4> geometryOutputTexture;
uint4 geometryPageWord(uint index) {
    uint side = params.dispatch.w, pageWords = side * side;
    uint withinPage = index % pageWords;
    return geometryPageTexture.Load(int4(withinPage % side, withinPage / side, index / pageWords, 0));
}
void geometryOutputStore(uint index, uint4 value) {
    uint side = params.dispatch.w, pageWords = side * side;
    uint local = index - geometryWords[0].z;
    uint withinPage = local % pageWords;
    geometryOutputTexture[int3(withinPage % side, withinPage / side, local / pageWords)] = value;
}
uint4 geometryBaseWord(uint index) {
    return params.dispatch.z == 0 ? geometryWords[index] : geometryPageWord(index);
}
uint4 geometryWord(uint index) {
    if (params.dispatch.z == 0) return geometryWords[index];
    uint4 value = geometryPageWord(index);
    if (params.dispatch.z != 2) return value;
    uint4 header = geometryPageWord(0);
    if (index < header.y || index >= header.y + header.w * header.z) return value;
    uint vertex = (index - header.y) / header.z;
    uint field = (index - header.y) % header.z;
    uint count = geometryWords[0].x, low = 0, high = count;
    while (low < high) {
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
uint4 scenePageWord(uint index) {
    uint side = params.scenePages.y, pageWords = side * side;
    uint withinPage = index % pageWords;
    return scenePageTexture.Load(int4(withinPage % side, withinPage / side, index / pageWords, 0));
}
uint4 sceneWord(uint index) {
    if (params.scenePages.x == 0) return sceneWords[index];
    uint prefixWords = params.scenePages.z, rangeCount = params.scenePages.w;
    uint instanceBase = params.sceneOverlay.x, dynamicInstanceWords = params.sceneOverlay.y;
    uint staticBlasWords = params.sceneOverlay.z, mutableBlasWords = params.sceneOverlay.w;
    if (index < prefixWords) return sceneWords[index];
    uint nodeBase = prefixWords + rangeCount;
    uint instanceOverlayBase = nodeBase + mutableBlasWords;
    if (index >= instanceBase && index < instanceBase + dynamicInstanceWords)
        return sceneWords[instanceOverlayBase + index - instanceBase];
    if (index < instanceBase) {
        uint localWord = index - prefixWords, low = 0, high = rangeCount;
        while (low < high) {
            uint middle = (low + high) >> 1;
            uint4 range = sceneWords[prefixWords + middle];
            if (localWord >= range.x * 2) low = middle + 1;
            else high = middle;
        }
        uint mutableBefore = 0;
        if (low > 0) {
            uint4 range = sceneWords[prefixWords + low - 1];
            uint rangeStart = range.x * 2, rangeEnd = rangeStart + range.y * 2;
            if (localWord < rangeEnd) return sceneWords[range.z + localWord - rangeStart];
            mutableBefore = range.w + range.y;
        }
        return scenePageWord(localWord - mutableBefore * 2);
    }
    return scenePageWord(staticBlasWords + index - instanceBase - dynamicInstanceWords);
}
void sceneStore(uint index, uint4 value) {
    if (params.scenePages.x == 0 || index < params.scenePages.z) {
        sceneWords[index] = value;
        return;
    }
    uint prefixWords = params.scenePages.z, rangeCount = params.scenePages.w;
    uint localWord = index - prefixWords, low = 0, high = rangeCount;
    while (low < high) {
        uint middle = (low + high) >> 1;
        uint4 range = sceneWords[prefixWords + middle];
        if (localWord >= range.x * 2) low = middle + 1;
        else high = middle;
    }
    if (low > 0) {
        uint4 range = sceneWords[prefixWords + low - 1];
        uint rangeStart = range.x * 2;
        if (localWord < rangeStart + range.y * 2)
            sceneWords[range.z + localWord - rangeStart] = value;
    }
}
uint vertexBase(uint vertex) {
    uint4 header = geometryWord(0);
    return header.y + vertex * header.z;
}
float3 direction(float3 value) {
    float scale = max(abs(value.x), max(abs(value.y), abs(value.z)));
    if (!isfinite(scale) || scale == 0) return 0;
    value /= max(scale * 0.25, asfloat(0x00800000u));
    return value / length(value);
}
[shader("compute")]
[numthreads(64, 1, 1)]
void deformVertices(uint3 id : SV_DispatchThreadID) {
    if (id.x >= params.dispatch.y) return;
    uint4 mapping = deformationWords[deformationWords[0].z + params.dispatch.x + id.x];
    uint job = deformationWords[0].y + mapping.x * 3;
    uint4 a = deformationWords[job], b = deformationWords[job+1], c = deformationWords[job+2];
    uint local = mapping.y, source = vertexBase(a.x + local), destination = vertexBase(a.y + local);
    uint4 sourcePosition = geometryBaseWord(source);
    uint4 sourceNormal = geometryBaseWord(source+1);
    float3 position = asfloat(sourcePosition.xyz);
    float3 normal = asfloat(sourceNormal.xyz);
    float4 tangent = asfloat(geometryBaseWord(source+2));
    for (uint target = 0; target < b.y; ++target) {
        float weight = asfloat(poseWords[b.w + target/4][target&3]);
        uint morph = b.x + (target*a.z + local)*3;
        position += weight*asfloat(deformationWords[morph].xyz);
        normal += weight*asfloat(deformationWords[morph+1].xyz);
        tangent.xyz += weight*asfloat(deformationWords[morph+2].xyz);
    }
    if (c.x != 0) {
        uint4 joints = deformationWords[a.w + local*2];
        float4 weights = asfloat(deformationWords[a.w + local*2+1]);
        float4 x = 0, y = 0, z = 0;
        for (uint i = 0; i < 4; ++i) {
            uint matrix = b.z + joints[i]*4;
            x += weights[i]*asfloat(poseWords[matrix]);
            y += weights[i]*asfloat(poseWords[matrix+1]);
            z += weights[i]*asfloat(poseWords[matrix+2]);
        }
        position = float3(dot(x,float4(position,1)),dot(y,float4(position,1)),dot(z,float4(position,1)));
        float3 col0 = float3(x.x,y.x,z.x), col1 = float3(x.y,y.y,z.y), col2 = float3(x.z,y.z,z.z);
        float3 nx = cross(col1,col2), ny = cross(col2,col0), nz = cross(col0,col1);
        float determinant = dot(col0,nx);
        normal = determinant != 0 ? direction((nx*normal.x+ny*normal.y+nz*normal.z)/determinant) : float3(0);
        tangent.xyz = direction(float3(dot(x.xyz,tangent.xyz),dot(y.xyz,tangent.xyz),dot(z.xyz,tangent.xyz)));
    }
    uint4 output[5];
    output[0] = uint4(asuint(position),sourcePosition.w);
    output[1] = uint4(asuint(normal),sourceNormal.w);
    output[2] = asuint(tangent);
    output[3] = geometryBaseWord(source+3);
    output[4] = geometryBaseWord(source+4);
    if (params.dispatch.z == 2) {
        uint target = geometryWords[0].z + mapping.z * 5;
        for (uint word = 0; word < 5; ++word) geometryOutputStore(target + word, output[word]);
    } else {
        for (uint word = 0; word < 5; ++word) geometryWords[destination + word] = output[word];
    }
}
[shader("compute")]
[numthreads(64, 1, 1)]
void refitBlas(uint3 id : SV_DispatchThreadID) {
    if (id.x >= params.dispatch.y) return;
    uint node = refitNodes[params.dispatch.x + id.x];
    uint4 lower = sceneWord(1+node*2), upper = sceneWord(2+node*2);
    float3 minimum = float3(3.402823466e+38), maximum = -minimum;
    if (upper.w == 0) {
        uint child = lower.w;
        minimum = min(asfloat(sceneWord(1+child*2).xyz),asfloat(sceneWord(1+(child+1)*2).xyz));
        maximum = max(asfloat(sceneWord(2+child*2).xyz),asfloat(sceneWord(2+(child+1)*2).xyz));
    } else for (uint triangle = lower.w; triangle < lower.w+upper.w; ++triangle) {
        uint3 vertices = geometryWord(geometryWord(0).x + triangle).xyz;
        for (uint i = 0; i < 3; ++i) {
            float3 position = asfloat(geometryWord(vertexBase(vertices[i])).xyz);
            minimum = min(minimum,position); maximum = max(maximum,position);
        }
    }
    sceneStore(1+node*2, uint4(asuint(minimum),lower.w));
    sceneStore(2+node*2, uint4(asuint(maximum),upper.w));
}

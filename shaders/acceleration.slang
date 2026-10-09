// Each depth has a separate pass so child writes precede parent reads.
struct RefitParameters { uint4 dispatch; uint4 scenePages; uint4 sceneOverlay; };
[[vk::binding(0, 0)]] ConstantBuffer<RefitParameters> params;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint4> sceneWords;
[[vk::binding(2, 0)]] StructuredBuffer<uint> refitNodes;
[[vk::binding(3, 0)]] Texture2DArray<uint4> scenePageTexture;
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
[shader("compute")]
[numthreads(64, 1, 1)]
void refitTlas(uint3 id : SV_DispatchThreadID) {
    if (id.x >= params.dispatch.y) return;
    uint node = refitNodes[params.dispatch.x + id.x];
    uint4 lower = sceneWord(1+node*2), upper = sceneWord(2+node*2);
    float3 minimum = float3(3.402823466e+38), maximum = -minimum;
    if (upper.w == 0) {
        uint child = lower.w;
        minimum = min(asfloat(sceneWord(1+child*2).xyz), asfloat(sceneWord(1+(child+1)*2).xyz));
        maximum = max(asfloat(sceneWord(2+child*2).xyz), asfloat(sceneWord(2+(child+1)*2).xyz));
    } else for (uint instance = lower.w; instance < lower.w+upper.w; ++instance) {
        uint base = sceneWord(0).y + instance*sceneWord(0).w;
        uint root = sceneWord(base+8).x;
        float3 a = asfloat(sceneWord(1+root*2).xyz), b = asfloat(sceneWord(2+root*2).xyz);
        float3 center = a*0.5+b*0.5, extent = b*0.5-a*0.5;
        float4 x = asfloat(sceneWord(base+3)), y = asfloat(sceneWord(base+4)), z = asfloat(sceneWord(base+5));
        float3 worldCenter = float3(dot(x,float4(center,1)),dot(y,float4(center,1)),dot(z,float4(center,1)));
        float3 worldExtent = float3(dot(abs(x.xyz),extent),dot(abs(y.xyz),extent),dot(abs(z.xyz),extent));
        float3 absolute = max(abs(a),abs(b));
        float3 error = (float3(abs(x.w),abs(y.w),abs(z.w))
            + float3(dot(abs(x.xyz),absolute),dot(abs(y.xyz),absolute),dot(abs(z.xyz),absolute)))*9.536743164e-7;
        minimum = min(minimum,worldCenter-worldExtent-error);
        maximum = max(maximum,worldCenter+worldExtent+error);
    }
    sceneStore(1+node*2, uint4(asuint(minimum),lower.w));
    sceneStore(2+node*2, uint4(asuint(maximum),upper.w));
}

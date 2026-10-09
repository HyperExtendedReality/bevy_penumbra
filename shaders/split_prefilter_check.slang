#include "common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<uint4> inputPixels;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint4> outputPixels;

[shader("compute")]
[numthreads(64, 1, 1)]
void checkIrradianceOctPad(uint3 thread : SV_DispatchThreadID) {
    if (thread.x >= 64) return;
    int2 pixel = int2(inputPixels[thread.x].xy);
    int2 texel = irradianceOctTexel(pixel);
    float2 uv = irradianceOctUv(pixel);
    outputPixels[thread.x*2] = uint4(
        uint(texel.x), uint(texel.y), irradianceOctIndex(pixel), asuint(uv.x));
    outputPixels[thread.x*2+1] = uint4(asuint(uv.y), 0, 0, 0);
}

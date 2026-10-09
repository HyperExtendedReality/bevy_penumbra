// Numeric packing stays exactly representable in f32 and never creates NaNs.
float receiverPackedNormal(float3 normal) {
    uint2 oct = uint2(round(saturate(encodeNormal(normal))*4095));
    return float(oct.x | (oct.y << 12));
}
float3 receiverNormal(float4 metadata) {
    uint packed = uint(metadata.z);
    return decodeNormal(float2(packed & 4095,packed >> 12)/4095.0);
}
float receiverCurvedSurfaceTolerance(float3 aNormal, float3 bNormal, float lateralDistance, Parameters p) {
    // A tangent plane departs from a curved surface by its sagitta. The normal
    // change bounds that departure; cap the allowance so different nearby
    // surfaces cannot gain an unbounded depth tolerance.
    float alignment = saturate(dot(normalize(aNormal),normalize(bNormal)));
    float halfAngleTangent = sqrt((1.0-alignment)/max(1.0+alignment,0.000001));
    float curvature = min(0.15,halfAngleTangent/(sqrt(1.0+halfAngleTangent*halfAngleTangent)+1.0));
    return surfaceTolerance(p)+max(lateralDistance,0.0)*curvature;
}
float3 receiverGeometricNormal(uint packed, float3 fallback) {
    if ((packed & 0x80000000u) == 0) return normalize(fallback);
    float2 oct = float2(packed & 0x7fffu,(packed >> 15) & 0x7fffu)/32767.0*2.0-1.0;
    float3 normal = float3(oct,1.0-abs(oct.x)-abs(oct.y));
    if (normal.z < 0.0)
        normal.xy = (1.0-abs(normal.yx))*float2(normal.x >= 0.0 ? 1.0 : -1.0,
                                              normal.y >= 0.0 ? 1.0 : -1.0);
    return normalize(normal);
}
float receiverRoughness(float4 metadata) { return float((uint(metadata.w) >> 8) & 255)/255.0; }
uint receiverHistoryLength(float4 metadata) { return uint(metadata.w) & 255; }
float receiverPackedHistory(float roughness, uint length) {
    return float((uint(round(saturate(roughness)*255)) << 8) | min(length,255u));
}
// Eight logarithmic bits cover first-hit distance from zero to 65535 units;
// escape rays saturate. Relative comparisons are stable under quantization.
float receiverFirstDistance(float4 metadata) {
    return exp2(float(uint(metadata.w) >> 16)/16.0)-1;
}
float receiverWithDistance(float packedHistory, float distance) {
    uint encoded = uint(round(clamp(log2(1+max(distance,0.0))*16,0.0,255.0)));
    return float(uint(packedHistory) | (encoded << 16));
}
float receiverWithHistory(float4 metadata, uint length) {
    return float((uint(metadata.w) & 0xffff00u) | min(length,255u));
}
bool receiverSame(float4 a, float4 b) {
    return receiverHistoryLength(a) > 0 && receiverHistoryLength(b) > 0 &&
        all(asuint(a.xy) == asuint(b.xy));
}

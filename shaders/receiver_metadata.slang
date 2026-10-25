// Numeric packing stays exactly representable in f32 and never creates NaNs.
float receiverPackedNormal(float3 normal) {
    uint2 oct = uint2(round(saturate(encodeNormal(normal))*4095));
    return float(oct.x | (oct.y << 12));
}
float3 receiverNormal(float4 metadata) {
    uint packed = uint(metadata.z);
    return decodeNormal(float2(packed & 4095,packed >> 12)/4095.0);
}
float3 receiverGeometricNormal(uint packed, float3 fallback) {
    fallback = normalize(fallback);
    if ((packed & 0x80000000u) == 0) return fallback;
    float2 oct = float2(packed & 0x7fffu,(packed >> 15) & 0x7fffu)/32767.0*2.0-1.0;
    float3 normal = float3(oct,1.0-abs(oct.x)-abs(oct.y));
    if (normal.z < 0.0)
        normal.xy = (1.0-abs(normal.yx))*float2(normal.x >= 0.0 ? 1.0 : -1.0,
                                              normal.y >= 0.0 ? 1.0 : -1.0);
    normal = normalize(normal);
    // Screen-space derivatives can reverse orientation because framebuffer Y
    // points down. Keep the packed geometric normal on the visible shading
    // hemisphere before using it for hit-point direct lighting.
    return dot(normal, fallback) < 0.0 ? -normal : normal;
}
uint receiverHistoryLength(float4 metadata) { return uint(metadata.w) & 255; }
float receiverPackedHistory(uint length) {
    return float(min(length,255u));
}
// Screen-history receiver metadata comes from splitRasterMetadata, whose w
// field only uses the low 16 bits. Bit 16 marks a supported diffuse-GI value;
// zero remains unknown/unobserved. Do not use these helpers for receiver
// records that encode first distance in the high bits.
uint receiverScreenHistorySupportMask(float4 metadata) {
    return (uint(metadata.w) >> 16) & 1u;
}
float4 receiverWithScreenHistorySupportMask(float4 metadata, uint supportMask) {
    uint packed = uint(metadata.w);
    packed = (packed & ~(1u << 16)) | ((supportMask & 1u) << 16);
    return float4(metadata.xyz, float(packed));
}
bool receiverSame(float4 a, float4 b) {
    return receiverHistoryLength(a) > 0 && receiverHistoryLength(b) > 0 &&
        all(asuint(a.xy) == asuint(b.xy));
}

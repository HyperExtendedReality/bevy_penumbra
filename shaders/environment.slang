// Alias records share the light storage binding, after its layer-mask tail.
[[vk::binding(22,0)]] Texture2D<float4> environmentTexture;
float3 environmentRotate(float3 d, float4 q) {
    float3 t = 2*cross(q.xyz,d);
    return d+q.w*t+cross(q.xyz,t);
}
float2 environmentUv(float3 d) {
    d = normalize(environmentRotate(d,params.environmentRotation));
    return float2(frac(atan2(d.z,d.x)/(2*PI)+0.5),atan2(length(d.xz),d.y)/PI);
}
uint4 environmentAlias(uint index) {
    Light record = lights[params.environmentInfo.x+index/6];
    switch (index%6) {
        case 0: return asuint(record.position);
        case 1: return asuint(record.color);
        case 2: return asuint(record.direction);
        case 3: return asuint(record.edge1);
        case 4: return asuint(record.edge2);
        default: return record.source;
    }
}
float3 environmentTexel(int2 pixel) {
    int width = int(params.environmentInfo.y), height = int(params.environmentInfo.z);
    pixel.x = (pixel.x%width+width)%width; pixel.y = clamp(pixel.y,0,height-1);
    return environmentTexture.Load(int3(pixel,0)).xyz;
}
float3 environmentRadiance(float3 direction) {
    if (params.environmentInfo.w == 0) return params.environment.xyz;
    float2 pixel = environmentUv(direction)*float2(params.environmentInfo.yz)-0.5;
    int2 base = int2(floor(pixel)); float2 f = frac(pixel);
    return lerp(lerp(environmentTexel(base),environmentTexel(base+int2(1,0)),f.x),
                lerp(environmentTexel(base+int2(0,1)),environmentTexel(base+int2(1,1)),f.x),f.y)*params.environmentScale.x;
}

// Alias records share the light storage binding, after its layer-mask tail.
float3 scatteringResponse(float3 normal, float3 view, float3 light, float roughness,
                          bool fullBsdf, SurfaceBsdf material, float2 ab) {
    return fullBsdf ? bsdfEvaluate(material, normal, view, light, ab)
        : float3(disneyDiffuseMultiplier(normal, light, view, roughness) / PI);
}
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
float environmentTexelSolidAngle(uint y) {
    float height = float(params.environmentInfo.z);
    float thetaMid = PI * (float(y) + 0.5) / height;
    float thetaHalf = PI * 0.5 / height;
    // Cosine difference in product form remains stable for narrow polar rows.
    return (4.0 * PI / float(params.environmentInfo.y)) * sin(thetaMid) * sin(thetaHalf);
}
float environmentPdf(float3 direction) {
    if (params.environmentInfo.w == 0) return 1.0 / (4.0 * PI);
    float2 uv = environmentUv(direction);
    uint x = min(uint(uv.x * float(params.environmentInfo.y)), params.environmentInfo.y - 1);
    uint y = min(uint(uv.y * float(params.environmentInfo.z)), params.environmentInfo.z - 1);
    uint index = x + y * params.environmentInfo.y;
    float mass = asfloat(environmentAlias(index).z);
    return max(mass, 0.0) / max(environmentTexelSolidAngle(y), 1e-20);
}

// Sample the HDR alias distribution in exact solid angle within the selected
// lat-long texel. The four random values choose the alias branch and texel UV.
bool sampleEnvironment(float4 random, out float3 direction, out float pdf, out float3 radiance) {
    if (params.environmentInfo.w == 0) {
        float y = 1.0 - 2.0 * saturate(random.x);
        float phi = 2.0 * PI * frac(random.y);
        float radius = sqrt(max(0.0, 1.0 - y * y));
        direction = float3(radius * cos(phi), y, radius * sin(phi));
        pdf = 1.0 / (4.0 * PI);
        radiance = environmentRadiance(direction);
        return all(isfinite(direction)) && all(isfinite(radiance));
    }

    uint width = params.environmentInfo.y, height = params.environmentInfo.z;
    uint count = width * height;
    if (count == 0) { direction = 0; pdf = 0; radiance = 0; return false; }
    // Keep closed-interval test inputs off the upper edge. Otherwise a texel
    // coordinate of exactly one can round into the following row/column and
    // make the returned PDF disagree with environmentPdf(direction).
    float columnSample = min(saturate(random.x), 0.99999994);
    uint column = min(uint(columnSample * float(count)), count - 1);
    uint4 columnRecord = environmentAlias(column);
    float aliasSample = min(saturate(random.y), 0.99999994);
    uint index = aliasSample < asfloat(columnRecord.x) ? column : min(columnRecord.y, count - 1);
    uint x = index % width, y = index / width;
    float texelU = min(saturate(random.z), 0.99999994);
    float u = (float(x) + texelU) / float(width);
    float phi = 2.0 * PI * (u - 0.5);
    // Sample uniformly in cos(theta) while keeping precision at both poles.
    // Reflect southern rows into the northern hemisphere before the half-angle
    // calculation: sin(theta/2) rounds to one near pi, collapsing a polar row.
    bool south = y * 2u >= height;
    uint rowFromPole = south ? height - 1u - y : y;
    float theta0 = PI * float(rowFromPole) / float(height);
    float theta1 = PI * float(rowFromPole + 1u) / float(height);
    float halfAngleLow = sin(theta0 * 0.5);
    float halfAngleHigh = sin(theta1 * 0.5);
    float r = min(saturate(random.w), 0.99999994);
    float sinHalfSquared = lerp(halfAngleLow * halfAngleLow, halfAngleHigh * halfAngleHigh, r);
    float cosTheta = (south ? -1.0 : 1.0) * (1.0 - 2.0 * sinHalfSquared);
    float sinTheta = 2.0 * sqrt(max(0.0, sinHalfSquared * (1.0 - sinHalfSquared)));
    float3 localDirection = float3(sinTheta * cos(phi), cosTheta, sinTheta * sin(phi));
    float4 inverseRotation = float4(-params.environmentRotation.xyz, params.environmentRotation.w);
    direction = normalize(environmentRotate(localDirection, inverseRotation));
    pdf = asfloat(environmentAlias(index).z) / max(environmentTexelSolidAngle(y), 1e-20);
    radiance = environmentRadiance(direction);
    return pdf > 0.0 && isfinite(pdf) && all(isfinite(direction)) && all(isfinite(radiance));
}

// A cosine-weighted incident sample's contribution after a visibility query.
// Taking support and occlusion as inputs keeps this production math directly
// testable with a deterministic visibility stub; blocked samples are supported
// black while unavailable traversal remains unsupported.
float3 environmentNeeSampleContribution(float3 radiance,
                                        float3 normal, float3 geometricNormal,
                                        float3 view, float roughness,
                                        float3 direction, float pdf,
                                        bool visibilitySupported, bool blocked,
                                        out bool sampleSupported) {
    sampleSupported = visibilitySupported;
    if (!visibilitySupported || blocked) return 0.0;
    float cosine = max(0.0, dot(normal, direction));
    if (cosine <= 0.0 || dot(geometricNormal, direction) <= 0.0 ||
        pdf <= 0.0 || !isfinite(pdf)) return 0.0;
    float diffuse = disneyDiffuseMultiplier(normal, direction, view, roughness) / PI;
    return radiance * (diffuse * cosine / pdf);
}

// Escaped Split rays are excluded from the indirect field. Direct environment
// lighting at visible receivers belongs to Bevy's native ambient/environment
// lighting; ray-hit surfaces use visibility-tested NEE in the outgoing source.
// Keep support separate so denied traversal is not confused with a measured
// zero-radiance escape.
float3 outgoingEnvironmentEscape(bool traversalSupported, out bool querySupported) {
    querySupported = traversalSupported;
    return 0.0;
}

float3 environmentResponseTracked(float3 position, float3 normal, float3 geometricNormal,
                             float3 view, float roughness, uint count,
                             uint sequenceBase, out bool querySupported,
                             inout DependencyContext dependency, uint admissionLimit,
                             bool fullBsdf, SurfaceBsdf material, float2 ab) {
    querySupported = true;
    if (count == 0 || (params.environmentInfo.w == 0 && all(params.environment.xyz <= 0.0)))
        return 0.0;

    float3 sum = 0.0;
    uint seed = hashKey(int4(int3(floor(position / (params.tuning.x * 0.05))), 37), 16777216);
    uint aliasCount = params.environmentInfo.w == 0 ? 0u : (count + 1u) / 2u;
    uint cosineCount = count - aliasCount;
    uint sampleFrame = sequenceBase;
    float aliasWeight = float(aliasCount) / float(count);
    float cosineWeight = float(cosineCount) / float(count);
    [loop] for (uint sample = 0; sample < count; ++sample) {
        uint sequence = sample + sampleFrame * count;
        uint sampleSeed = transportHash(seed ^ transportHash(sequence + 0x85ebca6bu));
        float4 random = float4(transportRandom(sampleSeed, 0), transportRandom(sampleSeed, 1),
                               transportRandom(sampleSeed, 2), transportRandom(sampleSeed, 3));
        float3 lightDirection, radiance;
        if (sample < aliasCount) {
            float aliasPdf;
            if (!sampleEnvironment(random, lightDirection, aliasPdf, radiance) || aliasPdf <= 0.0)
                continue;
        } else {
            // Constant environments use cosine proposals; the Disney BSDF
            // factor remains in environmentNeeSampleContribution below.
            if (fullBsdf && random.z < 0.5) {
                float proposalPdf;
                lightDirection = bsdfSampleVisibleGgx(random.xy, material, normal, view, proposalPdf);
                if (proposalPdf <= 0.0) continue;
            } else lightDirection = cosineDirection(normal, random.xy);
            radiance = environmentRadiance(lightDirection);
        }

        float cosine = max(0.0, dot(normal, lightDirection));
        if (cosine <= 0.0 || dot(geometricNormal, lightDirection) <= 0.0) continue;
        float aliasPdf = environmentPdf(lightDirection);
        float cosinePdf = cosine / PI;
        float materialPdf = fullBsdf ? 0.5 * (cosinePdf
            + bsdfVisibleGgxPdf(material, normal, view, lightDirection)) : cosinePdf;
        float pdf = aliasWeight * aliasPdf + cosineWeight * materialPdf;
        if (pdf <= 0.0 || !isfinite(pdf)) continue;

        float3 origin = rayOffset(position, geometricNormal, params.tuning.z);
        bool visibilitySupported;
        DependencyContext visibilityDependency;
        bool blocked = traceOccludedTracked(origin, lightDirection, 1e20,
            visibilitySupported, visibilityDependency, admissionLimit);
        dependencyUnion(dependency, visibilityDependency);
        bool sampleSupported;
        float3 contribution;
        if (fullBsdf) {
            sampleSupported = visibilitySupported;
            contribution = blocked || !visibilitySupported ? float3(0.0)
                : radiance * scatteringResponse(normal, view, lightDirection, roughness,
                    true, material, ab) * (cosine / pdf);
        } else contribution = environmentNeeSampleContribution(
                radiance, normal, geometricNormal, view, roughness, lightDirection, pdf,
                visibilitySupported, blocked, sampleSupported);
        if (!sampleSupported) { querySupported = false; return 0.0; }
        sum += contribution;
    }
    return sum / float(count);
}

float3 environmentResponse(float3 position, float3 normal, float3 geometricNormal,
                           float3 view, float roughness, uint count,
                           uint sequenceBase, out bool querySupported,
                           uint admissionLimit, bool fullBsdf, SurfaceBsdf material, float2 ab) {
    DependencyContext ignoredDependency = dependencyEmpty();
    return environmentResponseTracked(position, normal, geometricNormal, view, roughness,
        count, sequenceBase, querySupported, ignoredDependency, admissionLimit,
        fullBsdf, material, ab);
}

float3 environmentIrradiance(float3 position, float3 normal, float3 geometricNormal,
                             float3 view, float roughness, uint count,
                             uint sequenceBase, out bool querySupported,
                             uint admissionLimit = 0u) {
    return environmentResponse(position, normal, geometricNormal, view, roughness,
        count, sequenceBase, querySupported, admissionLimit, false, (SurfaceBsdf)0, float2(0));
}

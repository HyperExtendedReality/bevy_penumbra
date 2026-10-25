// Isotropic opaque-surface BSDF helpers aligned with Bevy 0.20's
// pbr_lighting.wesl. Directions point away from the surface; the returned
// values are BRDFs per steradian (the caller multiplies by NdotL).
static const float BSDF_PI = 3.14159265358979323846;
static const float BSDF_MIN_PERCEPTUAL_ROUGHNESS = 0.089;

struct SurfaceBsdf {
    float3 baseColor;
    float metallic;
    float perceptualRoughness;
    float reflectance;
};

float bsdfPow5(float value) {
    float squared = value * value;
    return squared * squared * value;
}

float3 bsdfUnitOrZero(float3 value) {
    float lengthSquared = dot(value, value);
    if (!all(isfinite(value)) || !isfinite(lengthSquared) || lengthSquared <= 1.0e-20) return 0.0;
    return value * rsqrt(lengthSquared);
}

float3 bsdfBaseColor(SurfaceBsdf material) {
    return all(isfinite(material.baseColor)) ? saturate(material.baseColor) : 0.0;
}

float bsdfMetallic(SurfaceBsdf material) {
    return isfinite(material.metallic) ? saturate(material.metallic) : 0.0;
}

float bsdfPerceptualRoughness(SurfaceBsdf material) {
    float value = isfinite(material.perceptualRoughness)
        ? saturate(material.perceptualRoughness) : 1.0;
    return max(value, BSDF_MIN_PERCEPTUAL_ROUGHNESS);
}

float bsdfAlpha(SurfaceBsdf material) {
    float roughness = bsdfPerceptualRoughness(material);
    return roughness * roughness;
}

float bsdfReflectance(SurfaceBsdf material) {
    return isfinite(material.reflectance) ? saturate(material.reflectance) : 0.5;
}

float3 bsdfDiffuseColor(SurfaceBsdf material) {
    return bsdfBaseColor(material) * (1.0 - bsdfMetallic(material));
}

// Bevy maps dielectric reflectance to F0 = 0.16 * reflectance^2, then uses
// base color as F0 for metals.
float3 bsdfF0(SurfaceBsdf material) {
    float3 dielectricF0 = 0.16 * bsdfReflectance(material) * bsdfReflectance(material);
    return lerp(dielectricF0.xxx, bsdfBaseColor(material), bsdfMetallic(material));
}

// Bevy's isotropic GGX distribution. `alpha` is perceptual roughness squared.
float bsdfD_GGX(float alpha, float NdotH) {
    alpha = isfinite(alpha) ? clamp(alpha, 0.0, 1.0) : 1.0;
    NdotH = isfinite(NdotH) ? saturate(NdotH) : 0.0;
    if (alpha <= 0.0 || NdotH <= 0.0) return 0.0;
    float oneMinusNdotHSquared = 1.0 - NdotH * NdotH;
    float a = NdotH * alpha;
    float denominator = oneMinusNdotHSquared + a * a;
    if (denominator <= 1.0e-20 || !isfinite(denominator)) return 0.0;
    float k = alpha / denominator;
    float value = k * k * (1.0 / BSDF_PI);
    return isfinite(value) ? max(value, 0.0) : 0.0;
}

// Bevy's exact height-correlated Smith GGX visibility term V = G/(4 NoV NoL).
float bsdfV_SmithGGXCorrelated(float alpha, float NdotV, float NdotL) {
    alpha = isfinite(alpha) ? clamp(alpha, 0.0, 1.0) : 1.0;
    NdotV = isfinite(NdotV) ? saturate(NdotV) : 0.0;
    NdotL = isfinite(NdotL) ? saturate(NdotL) : 0.0;
    if (NdotV <= 0.0 || NdotL <= 0.0) return 0.0;
    float a2 = alpha * alpha;
    float lambdaV = NdotL * sqrt(max((NdotV - a2 * NdotV) * NdotV + a2, 0.0));
    float lambdaL = NdotV * sqrt(max((NdotL - a2 * NdotL) * NdotL + a2, 0.0));
    float denominator = lambdaV + lambdaL;
    if (denominator <= 1.0e-20 || !isfinite(denominator)) return 0.0;
    float value = 0.5 / denominator;
    return isfinite(value) ? max(value, 0.0) : 0.0;
}

float3 bsdfFresnelSchlick(float3 f0, float VdotH) {
    if (!all(isfinite(f0)) || !isfinite(VdotH)) return 0.0;
    return f0 + (1.0 - f0) * bsdfPow5(1.0 - saturate(VdotH));
}

// The caller supplies the correlated GGX DFG integral (A,B), preferably from
// Bevy's DFG LUT. Keeping this explicit avoids silently substituting the
// analytic EnvBRDF fit, whose grazing-angle error is large for this BRDF.
float3 bsdfSpecularEnergyCompensation(SurfaceBsdf material, float2 ab) {
    ab = all(isfinite(ab)) ? max(ab, float2(0.00005, 0.00005)) : float2(0.00005, 0.00005);
    float denominator = max(ab.x + ab.y, 0.0001);
    return 1.0 + bsdfF0(material) * (1.0 / denominator - 1.0);
}

// View-dependent reflected energy from caller-supplied correlated DFG A/B,
// after Bevy/Filament-style per-lobe compensation. This remains an approximate
// multiple-scattering model, not an exact Kulla-Conty solution.
float3 bsdfSpecularEnergy(SurfaceBsdf material, float2 ab) {
    ab = all(isfinite(ab)) ? max(ab, float2(0.00005, 0.00005)) : float2(0.00005, 0.00005);
    float3 f0 = bsdfF0(material);
    float denominator = max(ab.x + ab.y, 0.0001);
    float3 compensation = 1.0 + f0 * (1.0 / denominator - 1.0);
    float3 energy = (f0 * ab.x + ab.y) * compensation;
    return all(isfinite(energy)) ? saturate(energy) : 0.0;
}

// Reciprocal of the exact view-conditioned hemispherical integral of the
// Disney/Burley diffuse shape used below. This is a view-dependent energy
// normalization (not a reciprocal BSDF); use it in the direct-hit BSDF
// evaluation so the diffuse lobe integrates to its remaining energy budget.
float bsdfDiffuseNormalization(float perceptualRoughness, float NdotV) {
    float pr = isfinite(perceptualRoughness)
        ? max(saturate(perceptualRoughness), BSDF_MIN_PERCEPTUAL_ROUGHNESS) : 1.0;
    float r = pr * pr;
    NdotV = isfinite(NdotV) ? saturate(NdotV) : 0.0;
    float b = r - 0.5;
    float s = bsdfPow5(1.0 - NdotV);
    float oneMinusNvSquared = 1.0 - NdotV * NdotV;
    float e = 1.0
        + b * s
        + (2.0 / 3.0) * r * s * NdotV
        + (b + b * b * s) / 21.0
        + r * (1.0 + 2.0 * b * s) * NdotV / 84.0
        + r * r * s * ((11.0 / 504.0) * oneMinusNvSquared + (1.0 / 252.0) * NdotV * NdotV);
    return isfinite(e) && e > 1.0e-4 ? 1.0 / e : 0.0;
}

// Isotropic Disney/Burley diffuse BRDF. Diffuse is attenuated by the remaining
// view-conditioned energy after the DFG-compensated specular, and normalized
// so its cosine-weighted hemispherical integral equals that remaining budget.
float3 bsdfDiffuse(SurfaceBsdf material, float3 normal, float3 view, float3 light, float2 ab) {
    float3 n = bsdfUnitOrZero(normal);
    float3 v = bsdfUnitOrZero(view);
    float3 l = bsdfUnitOrZero(light);
    if (dot(n, n) <= 0.0 || dot(v, v) <= 0.0 || dot(l, l) <= 0.0) return 0.0;
    float NdotV = dot(n, v);
    float NdotL = dot(n, l);
    if (!isfinite(NdotV) || !isfinite(NdotL) || NdotV <= 0.0 || NdotL <= 0.0) return 0.0;
    float3 halfSum = l + v;
    float halfLengthSquared = dot(halfSum, halfSum);
    if (!isfinite(halfLengthSquared) || halfLengthSquared <= 1.0e-20) return 0.0;
    float3 h = halfSum * rsqrt(halfLengthSquared);
    float LdotH = saturate(dot(l, h));
    float alpha = bsdfAlpha(material);
    float f90 = 0.5 + 2.0 * alpha * LdotH * LdotH;
    float lightScatter = 1.0 + (f90 - 1.0) * bsdfPow5(1.0 - saturate(NdotL));
    float viewScatter = 1.0 + (f90 - 1.0) * bsdfPow5(1.0 - saturate(NdotV));
    float3 remainingSpecularEnergy = 1.0 - bsdfSpecularEnergy(material, ab);
    float3 scale = lightScatter * viewScatter * (1.0 / BSDF_PI)
        * bsdfDiffuseNormalization(bsdfPerceptualRoughness(material), NdotV)
        * remainingSpecularEnergy;
    float3 value = bsdfDiffuseColor(material) * scale;
    return all(isfinite(value)) ? max(value, 0.0) : 0.0;
}

// Microfacet GGX reflection BRDF per steradian; the caller applies NdotL.
float3 bsdfSpecular(SurfaceBsdf material, float3 normal, float3 view, float3 light, float2 ab) {
    float3 n = bsdfUnitOrZero(normal);
    float3 v = bsdfUnitOrZero(view);
    float3 l = bsdfUnitOrZero(light);
    if (dot(n, n) <= 0.0 || dot(v, v) <= 0.0 || dot(l, l) <= 0.0) return 0.0;
    float NdotV = dot(n, v);
    float NdotL = dot(n, l);
    if (!isfinite(NdotV) || !isfinite(NdotL) || NdotV <= 0.0 || NdotL <= 0.0) return 0.0;
    float3 halfSum = l + v;
    float halfLengthSquared = dot(halfSum, halfSum);
    if (!isfinite(halfLengthSquared) || halfLengthSquared <= 1.0e-20) return 0.0;
    float3 h = halfSum * rsqrt(halfLengthSquared);
    float NdotH = dot(n, h);
    float VdotH = dot(v, h);
    if (NdotH <= 0.0 || VdotH <= 0.0) return 0.0;
    float alpha = bsdfAlpha(material);
    float distribution = bsdfD_GGX(alpha, NdotH);
    float visibility = bsdfV_SmithGGXCorrelated(alpha, NdotV, NdotL);
    float3 fresnel = bsdfFresnelSchlick(bsdfF0(material), VdotH);
    float3 value = distribution * visibility * fresnel
        * bsdfSpecularEnergyCompensation(material, ab);
    return all(isfinite(value)) ? max(value, 0.0) : 0.0;
}

float3 bsdfEvaluate(SurfaceBsdf material, float3 normal, float3 view, float3 light, float2 ab) {
    return bsdfDiffuse(material, normal, view, light, ab)
        + bsdfSpecular(material, normal, view, light, ab);
}

// Bevy 0.20's bounded visible-normal sampler's outgoing-direction PDF. This
// expression already includes the half-vector reflection Jacobian; do not
// divide it by 4*VdotH a second time.
float bsdfGgxVndfPdf(float NdotV, float NdotH, float alpha) {
    NdotV = isfinite(NdotV) ? saturate(NdotV) : 0.0;
    NdotH = isfinite(NdotH) ? saturate(NdotH) : 0.0;
    alpha = isfinite(alpha) ? clamp(alpha, 0.0, 1.0) : 1.0;
    if (NdotV <= 0.0 || NdotH <= 0.0 || alpha <= 0.0) return 0.0;
    float xyLengthSquared = max(1.0 - NdotV * NdotV, 0.0);
    float len2 = alpha * alpha * xyLengthSquared;
    float t = sqrt(max(len2 + NdotV * NdotV, 0.0));
    float s = 1.0 + sqrt(xyLengthSquared);
    float a2 = alpha * alpha;
    float s2 = s * s;
    float denominator = s2 + a2 * NdotV * NdotV;
    if (denominator <= 1.0e-20 || !isfinite(denominator)) return 0.0;
    float k = (1.0 - a2) * s2 / denominator;
    float pdf = bsdfD_GGX(alpha, NdotH) / (2.0 * (k * NdotV + t));
    return isfinite(pdf) ? max(pdf, 0.0) : 0.0;
}

// PDF of the reflected direction from the bounded VNDF distribution. The
// closed-form helper above already accounts for the reflection Jacobian.
float bsdfVisibleGgxPdf(SurfaceBsdf material, float3 normal, float3 view, float3 light) {
    float3 n = bsdfUnitOrZero(normal);
    float3 v = bsdfUnitOrZero(view);
    float3 l = bsdfUnitOrZero(light);
    if (dot(n, n) <= 0.0 || dot(v, v) <= 0.0 || dot(l, l) <= 0.0) return 0.0;
    float NdotV = dot(n, v);
    float NdotL = dot(n, l);
    if (NdotV <= 0.0 || NdotL <= 0.0) return 0.0;
    float3 halfSum = v + l;
    float halfLengthSquared = dot(halfSum, halfSum);
    if (!isfinite(halfLengthSquared) || halfLengthSquared <= 1.0e-20) return 0.0;
    float3 h = halfSum * rsqrt(halfLengthSquared);
    float VdotH = abs(dot(v, h));
    if (!isfinite(VdotH) || VdotH <= 1.0e-7) return 0.0;
    float pdf = bsdfGgxVndfPdf(NdotV, dot(n, h), bsdfAlpha(material));
    return isfinite(pdf) ? max(pdf, 0.0) : 0.0;
}

// Bounded VNDF sampler from Bevy 0.20 (Heitz/AMD, Listing 1). Returns zero and
// pdf=0 for invalid input or when the reflected direction falls below the
// geometric hemisphere; callers may treat that sample as a zero contribution.
float3 bsdfSampleVisibleGgx(float2 xi, SurfaceBsdf material, float3 normal, float3 view,
                           out float pdf) {
    pdf = 0.0;
    if (!all(isfinite(xi))) return 0.0;
    float3 n = bsdfUnitOrZero(normal);
    float3 v = bsdfUnitOrZero(view);
    if (dot(n, n) <= 0.0 || dot(v, v) <= 0.0) return 0.0;
    float wiN = dot(v, n);
    if (!isfinite(wiN) || wiN <= 0.0) return 0.0;
    float alpha = bsdfAlpha(material);
    float3 wiXY = v - n * wiN;
    float3 wiZ = -n * wiN;
    float3 wiStd = -bsdfUnitOrZero(alpha * wiXY + wiZ);
    if (dot(wiStd, wiStd) <= 0.0) return 0.0;
    float wiStdZ = dot(wiStd, n);
    float s = 1.0 + length(wiXY);
    float a = clamp(alpha, 0.0, 1.0);
    float a2 = a * a;
    float s2 = s * s;
    float denominator = s2 + a2 * wiN * wiN;
    if (denominator <= 1.0e-20 || !isfinite(denominator)) return 0.0;
    float k = (1.0 - a2) * s2 / denominator;
    float b = k * wiStdZ;

    float2 u = clamp(xi, float2(0.0, 0.0), float2(1.0 - 1.0e-7, 1.0 - 1.0e-7));
    float z = 1.0 - u.y * (1.0 + b);
    float sinTheta = sqrt(max(0.0, 1.0 - z * z));
    float phi = 2.0 * BSDF_PI * u.x - BSDF_PI;
    float3 cStd = float3(sinTheta * cos(phi), sinTheta * sin(phi), z);

    // Rotate the sample frame so +Z aligns with the supplied normal.
    float3 helper = abs(n.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 tangent = bsdfUnitOrZero(cross(helper, n));
    float3 bitangent = cross(n, tangent);
    float3 c = tangent * cStd.x + bitangent * cStd.y + n * cStd.z;
    float3 wmStd = c + wiStd;
    float3 wmStdZ = n * dot(n, wmStd);
    float3 wmStdXY = wmStdZ - wmStd;
    float3 h = bsdfUnitOrZero(alpha * wmStdXY + wmStdZ);
    if (dot(h, h) <= 0.0 || dot(n, h) <= 0.0) return 0.0;
    float3 light = bsdfUnitOrZero(reflect(-v, h));
    if (dot(light, light) <= 0.0 || dot(n, light) <= 0.0) return 0.0;
    pdf = bsdfVisibleGgxPdf(material, n, v, light);
    return pdf > 0.0 ? light : 0.0;
}

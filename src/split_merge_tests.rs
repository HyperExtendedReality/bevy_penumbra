//! GPU regressions for bounded interval history and conditional merged support.
use super::*;

fn production_section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let source = &source[source.find(start).expect("production section start")..];
    &source[..source.find(end).expect("production section end")]
}

const MERGE_CASES: &str = r#"
Entry observed(float radiance, float beta, uint count, uint level) {
    Entry value = (Entry)0;
    value.key = int4(0, 0, 0, int(level << 12u));
    value.interval = float4(radiance, radiance, radiance, beta);
    value.samples = count;
    value.lastSample = params.frame.x;
    value.next = EMPTY;
    storeEntryDependency(value, dependencyEmpty());
    storeEntryMergedDependency(value, dependencyEmpty());
    value.boundaryInterval = emptyLightHitBounds();
    value.boundaryMerged = emptyLightHitBounds();
    return value;
}
void record(uint slot, Entry value) {
    results[slot] = value.interval;
    results[slot + 1u] = float4(asfloat(value.samples), value.origin.w, value.merged.w, 0.0);
}
void advance(float radiance, float beta, uint count) {
    entries[0] = observed(radiance, beta, count, params.frame.y);
    mergeEntry(0u);
    previousEntries[0] = entries[0];
}
void temporalCases() {
    previousEntries[0] = observed(0.0, 0.0, 0u, params.frame.y);
    previousEntries[0].origin.w = 0.0;
    [loop] for (uint frame = 0u; frame < 512u; ++frame) advance(2.0, 0.0, 1u);
    record(0u, entries[0]);
    advance(0.0, 1.0, 1u);
    record(2u, entries[0]);
    [loop] for (uint frame = 0u; frame < 200u; ++frame) advance(0.0, 1.0, 0u);
    record(4u, entries[0]);
    // The real retain stage must keep beta-only history discoverable.
    heads[0].store(EMPTY);
    retain(uint3(0u, 0u, 0u));
    uint retained = heads[0].load();
    results[6] = float4(retained != EMPTY ? 1.0 : 0.0, 0.0, 0.0, 0.0);
    if (retained != EMPTY) {
        mergeEntry(retained);
        record(7u, entries[retained]);
    }
    // Batched observations consume retention per observation, not per dispatch.
    [loop] for (uint frame = 0u; frame < 512u; ++frame) advance(2.0, 0.0, 1u);
    advance(0.0, 1.0, 4u);
    record(9u, entries[0]);
    // The final cascade owns the tail beyond its nominal finite cutoff.
    Hit hit = (Hit)0;
    hit.distance = 1e20;
    hit.triangle = EMPTY;
    DependencyContext rayDependency = dependencyEmpty();
    dependencyIncludeSegment(rayDependency, float3(0.0), float3(0.0, 0.0, 20000.0), 0.0);
    DependencyContext outgoingDependency = dependencyEmpty();
    heads[0].store(EMPTY);
    deposit(0u, float3(0.5), float3(0.0), float3(0.0, 0.0, 1.0),
        hit, float3(0.0), 0u, rayDependency, 0.0, 20000.0, 0.0, outgoingDependency);
    mergeEntry(3u);
    results[11] = entries[3].merged;
    heads[0].store(EMPTY);
    hit.distance = 20000.0;
    hit.triangle = 0u;
    deposit(0u, float3(0.5), float3(0.0), float3(0.0, 0.0, 1.0),
        hit, float3(4.0, 2.0, 1.0), 0u, rayDependency, 0.0, 20000.0, 0.0, outgoingDependency);
    mergeEntry(3u);
    results[12] = entries[3].merged;
    results[13] = entries[3].dependencyMax;
    entries[3] = (Entry)0;
    hit.errors = HIT_QUERY_UNSUPPORTED;
    deposit(0u, float3(0.5), float3(0.0), float3(0.0, 0.0, 1.0),
        hit, float3(4.0, 2.0, 1.0), 0u, rayDependency, 0.0, 20000.0, 0.0, outgoingDependency);
    results[14] = float4(float(entries[3].samples), float(entries[3].valid), 0.0, 0.0);
}
void supportCases() {
    uint stage = params.frame.w;
    if (stage == 0u) {
        // Eight spatial corners for each of four angular children, all in one
        // collision bucket so production findEntry exercises full key matching.
        [loop] for (uint child = 0u; child < 4u; ++child) {
            uint direction = (child & 1u) + (child >> 1u) * 16u;
            [loop] for (uint corner = 0u; corner < 8u; ++corner) {
                uint index = child * 8u + corner;
                uint3 bit = uint3(corner & 1u, (corner >> 1u) & 1u, corner >> 2u);
                Entry value = observed(0.0, 0.0, 1u, 1u);
                value.key = int4(int3(bit), int(direction | (1u << 12u)));
                value.merged = bit.x == 0u ? float4(2.0, 2.0, 2.0, 1.0)
                    : float4(10.0, 10.0, 10.0, 0.25);
                value.next = index + 1u < 32u ? index + 1u : EMPTY;
                entries[index] = value;
            }
        }
        heads[0].store(0u);
        DependencyContext dependency = dependencyEmpty();
        float4 lightDependency = emptyLightHitBounds();
        results[0] = interpolateMergedChildren(int3(0), float3(0.5, 0.0, 0.0),
            0u, 1u, 0u, false, dependency, lightDependency);
        Entry parent = observed(2.0, 0.5, 1u, 0u);
        parent.origin.xyz = float3(1.0, 0.5, 0.5) * spacing(params, 1u, 0u);
        entries[32] = parent;
        mergeEntry(32u);
        results[1] = entries[32].merged;
    } else if (stage == 1u) {
        // Fully supported data must reduce exactly to Eq. 7: J + beta * mean(I).
        [loop] for (uint index = 0u; index < 32u; ++index)
            entries[index].merged = float4(8.0, 8.0, 8.0, 1.0);
        entries[32] = observed(2.0, 0.5, 1u, 0u);
        entries[32].origin.xyz = float3(1.0, 0.5, 0.5) * spacing(params, 1u, 0u);
        mergeEntry(32u);
        results[2] = entries[32].merged;
    } else if (stage == 2u) {
        // Missing children never turn a measured opaque black interval unsupported.
        [loop] for (uint index = 0u; index < 32u; ++index) entries[index].merged = 0.0;
        entries[32] = observed(0.0, 0.0, 1u, 0u);
        mergeEntry(32u);
        results[3] = entries[32].merged;
    } else if (stage == 3u) {
        // One partially supported angular child must retain its fractional mass.
        entries[0].merged = float4(8.0, 8.0, 8.0, 0.25);
        entries[32] = observed(2.0, 0.5, 1u, 0u);
        entries[32].origin.xyz = float3(0.5) * spacing(params, 1u, 0u);
        mergeEntry(32u);
        results[4] = entries[32].merged;
    } else if (stage == 4u) {
        // A transparent interval may pass partially supported child data.
        previousEntries[0] = observed(7.0, 1.0, 0u, 0u);
        previousEntries[0].origin.w = 4.0;
        entries[32] = observed(0.0, 1.0, 0u, 0u);
        entries[32].origin.xyz = float3(0.5) * spacing(params, 1u, 0u);
        mergeEntry(32u);
        results[5] = entries[32].merged;
    } else if (stage == 5u) {
        // Invalidated J cannot invent support through an opaque interval.
        previousEntries[0].interval.w = 0.0;
        entries[32] = observed(0.0, 1.0, 0u, 0u);
        mergeEntry(32u);
        results[6] = entries[32].merged;
    } else if (stage == 6u) {
        // Different hues make support-weighting errors visible independently of energy.
        entries[0].merged = float4(1.0, 0.0, 0.0, 0.25);
        entries[1].merged = float4(0.0, 0.0, 1.0, 1.0);
        DependencyContext dependency = dependencyEmpty();
        float4 lightDependency = emptyLightHitBounds();
        results[7] = interpolateMergedChildren(int3(0), float3(0.5, 0.0, 0.0),
            0u, 1u, 0u, false, dependency, lightDependency);
    } else if (stage == 7u) {
        [loop] for (uint index = 0u; index < 32u; ++index)
            entries[index].merged = float4(0.0, 0.0, 1.0, 0.25);
        previousEntries[0] = (Entry)0;
        entries[32] = observed(0.0, 0.5, 1u, 0u);
        entries[32].interval.xyz = float3(0.5, 0.0, 0.0);
        entries[32].origin.xyz = float3(0.5) * spacing(params, 1u, 0u);
        mergeEntry(32u);
        results[8] = entries[32].merged;
    }
}
void dependencyGeometryCases() {
    DependencyContext outside = dependencyEmpty();
    dependencyIncludeBounds(outside, float3(5.0), float3(6.0));
    Entry old = observed(10.0, 0.0, 1u, 0u);
    old.samples = asuint(4.0);
    storeEntryDependency(old, outside);
    previousEntries[0] = old;
    entries[0] = observed(2.0, 0.0, 1u, 0u);
    mergeEntry(0u);
    float disjoint = entries[0].merged.x;

    DependencyContext overlapping = dependencyEmpty();
    dependencyIncludeBounds(overlapping, float3(0.25), float3(0.75));
    old = observed(10.0, 0.0, 1u, 0u);
    old.samples = asuint(4.0);
    storeEntryDependency(old, overlapping);
    previousEntries[0] = old;
    entries[0] = observed(2.0, 0.0, 1u, 0u);
    mergeEntry(0u);
    float intersecting = entries[0].merged.x;

    old = observed(10.0, 0.0, 1u, 0u);
    old.samples = asuint(4.0);
    storeEntryDependency(old, dependencyUnknown());
    previousEntries[0] = old;
    entries[0] = observed(2.0, 0.0, 1u, 0u);
    mergeEntry(0u);
    results[13] = float4(disjoint, intersecting, entries[0].merged.x, 1.0);
}
void dependencyLightCases() {
    DependencyContext dependency = dependencyEmpty();
    dependencyIncludeBounds(dependency, float3(0.25), float3(0.75));
    Entry old = observed(10.0, 0.2, 1u, 0u);
    old.samples = asuint(4.0);
    old.origin.w = 4.0;
    old.boundaryInterval = lightHitBoundsAt(float3(0.5));
    old.boundaryMerged = old.boundaryInterval;
    storeEntryDependency(old, dependency);
    previousEntries[0] = old;
    entries[0] = observed(2.0, 0.8, 1u, 0u);
    mergeEntry(0u);
    results[14] = float4(entries[0].interval.x, entries[0].interval.w,
        asfloat(entries[0].samples), entries[0].origin.w);

    // The old ray/shadow path can cross a light's influence AABB even when its
    // outgoing hit is outside it. Only the hit-position sphere gates J reuse.
    DependencyContext crossingPath = dependencyEmpty();
    dependencyIncludeBounds(crossingPath, float3(-5.0, 0.0, 0.0),
        float3(5.0, 0.0, 0.0));
    float4 outsideHit = lightHitBoundsAt(float3(5.0, 0.0, 0.0));
    bool rayCrossesLight = all(crossingPath.minimum <= params.lightDirtyMax.xyz)
        && all(crossingPath.maximum >= params.lightDirtyMin.xyz);
    results[15] = float4(rayCrossesLight ? 1.0 : 0.0,
        lightHitBoundsIntersectLocalLightDirty(outsideHit) ? 1.0 : 0.0,
        radianceHistoryEligible(crossingPath, outsideHit) ? 1.0 : 0.0,
        radianceHistoryEligible(crossingPath, emptyLightHitBounds()) ? 1.0 : 0.0);
    results[16] = float4(radianceHistoryEligible(crossingPath,
        unknownLightHitBounds()) ? 1.0 : 0.0, 0.0, 0.0, 0.0);
}
void recordLightRevision(uint slot, Entry value) {
    results[slot] = value.interval;
    results[slot + 1u] = float4(asfloat(value.samples), value.origin.w,
        value.merged.w, 0.0);
    results[slot + 2u] = value.merged;
}
void retainLightRevisionEntry() {
    heads[0].store(EMPTY);
    retain(uint3(0u, 0u, 0u));
}
void lightRevisionCases() {
    uint retainedIndex = currentCapacity();
    if (params.frame.w == 0u) {
        // Start with established world radiance and visibility whose measured
        // dependency overlaps a moving local light's influence bounds.
        // The seeded beta mass of four exceeds the current blend's limit of two.
        Entry old = observed(10.0, 0.2, 1u, 0u);
        old.samples = asuint(4.0);
        old.origin.w = 4.0;
        old.merged = float4(10.0, 10.0, 10.0, 0.8);
        old.boundaryInterval = lightHitBoundsAt(float3(0.5));
        old.boundaryMerged = old.boundaryInterval;
        DependencyContext dependency = dependencyEmpty();
        dependencyIncludeBounds(dependency, float3(0.25), float3(0.75));
        storeEntryDependency(old, dependency);
        storeEntryMergedDependency(old, dependency);
        previousEntries[0] = old;
        retainLightRevisionEntry();
        mergeEntry(retainedIndex);
        recordLightRevision(0u, entries[retainedIndex]);
    } else if (params.frame.w == 1u) {
        // The host's dirty-light bounds are a one-frame pulse. A beta-only
        // retained entry must not recover the old J on the clean next frame.
        previousEntries[0] = entries[retainedIndex];
        retainLightRevisionEntry();
        mergeEntry(retainedIndex);
        recordLightRevision(3u, entries[retainedIndex]);
    } else if (params.frame.w == 2u) {
        // A fresh observation after the revision repopulates J and support.
        previousEntries[0] = entries[retainedIndex];
        entries[0] = observed(2.0, 0.8, 1u, 0u);
        DependencyContext dependency = dependencyEmpty();
        dependencyIncludeBounds(dependency, float3(0.25), float3(0.75));
        storeEntryDependency(entries[0], dependency);
        storeEntryMergedDependency(entries[0], dependency);
        entries[0].boundaryInterval = lightHitBoundsAt(float3(0.5));
        entries[0].boundaryMerged = entries[0].boundaryInterval;
        mergeEntry(0u);
        recordLightRevision(6u, entries[0]);
    } else if (params.frame.w == 3u) {
        previousEntries[0] = entries[0];
        retainLightRevisionEntry();
        mergeEntry(retainedIndex);
        recordLightRevision(9u, entries[retainedIndex]);
    } else if (params.frame.w == 4u) {
        // A second revision must invalidate the newly measured J as well.
        previousEntries[0] = entries[retainedIndex];
        retainLightRevisionEntry();
        mergeEntry(retainedIndex);
        recordLightRevision(12u, entries[retainedIndex]);
    } else if (params.frame.w == 5u) {
        previousEntries[0] = entries[retainedIndex];
        retainLightRevisionEntry();
        mergeEntry(retainedIndex);
        recordLightRevision(15u, entries[retainedIndex]);
    } else {
        // Reacquisition after the second invalidation should use only the new
        // sample, while retaining the independently measured visibility.
        previousEntries[0] = entries[retainedIndex];
        entries[0] = observed(4.0, 0.4, 1u, 0u);
        DependencyContext dependency = dependencyEmpty();
        dependencyIncludeBounds(dependency, float3(0.25), float3(0.75));
        storeEntryDependency(entries[0], dependency);
        storeEntryMergedDependency(entries[0], dependency);
        entries[0].boundaryInterval = lightHitBoundsAt(float3(0.5));
        entries[0].boundaryMerged = entries[0].boundaryInterval;
        mergeEntry(0u);
        recordLightRevision(18u, entries[0]);
    }
}
void multiHitLightCases() {
    uint retainedIndex = currentCapacity();
    if (params.frame.w == 0u) {
        // Write distinct buffer slots directly. Keeping two Entry locals here
        // aliases their aggregate storage in the test shader on Vulkan.
        entries[0] = observed(2.0, 0.2, 1u, 0u);
        entries[1] = observed(6.0, 0.2, 1u, 0u);
        entries[0].boundaryInterval = lightHitBoundsAt(float3(0.25, 0.0, 0.0));
        entries[0].boundaryMerged = entries[0].boundaryInterval;
        entries[1].boundaryInterval = lightHitBoundsAt(float3(0.75, 0.0, 0.0));
        entries[1].boundaryMerged = entries[1].boundaryInterval;
        entries[0].next = 1u;
        entries[1].next = EMPTY;
        heads[0].store(0u);
        results[30] = entries[0].interval;
        results[31] = entries[1].interval;
    } else if (params.frame.w == 1u) {
        // Production reduction runs in its own pass before canonical merging.
        reduce(uint3(0u, 0u, 0u));
        results[32] = entries[0].interval;
        results[33] = float4(float(entries[0].samples), 0.0, 0.0, 0.0);
    } else if (params.frame.w == 2u) {
        mergeEntry(0u);
        recordLightRevision(21u, entries[0]);
    } else if (params.frame.w == 3u) {
        // The dirty box touches only the second hit. Reduction must preserve
        // the union so the aggregate's old J is invalidated conservatively.
        previousEntries[0] = entries[0];
        retainLightRevisionEntry();
        mergeEntry(retainedIndex);
        recordLightRevision(24u, entries[retainedIndex]);
    } else {
        previousEntries[0] = entries[retainedIndex];
        retainLightRevisionEntry();
        mergeEntry(retainedIndex);
        recordLightRevision(27u, entries[retainedIndex]);
    }
}
void globalLightCases() {
    uint retainedIndex = currentCapacity();
    if (params.frame.w == 0u) {
        // A terminal escape is a measured zero owned by Bevy's direct sky;
        // keep this actual observation through a global radiometric revision.
        Entry old = observed(0.0, 0.0, 1u, 0u);
        old.samples = asuint(4.0);
        old.origin.w = 4.0;
        old.merged = float4(0.0, 0.0, 0.0, 1.0);
        old.boundaryInterval = emptyLightHitBounds();
        old.boundaryMerged = emptyLightHitBounds();
        previousEntries[0] = old;
        retainLightRevisionEntry();
        mergeEntry(retainedIndex);
        recordLightRevision(30u, entries[retainedIndex]);
    } else if (params.frame.w == 1u) {
        previousEntries[0] = entries[retainedIndex];
        retainLightRevisionEntry();
        mergeEntry(retainedIndex);
        recordLightRevision(33u, entries[retainedIndex]);
    } else if (params.frame.w == 2u) {
        // A hit-surface J depends on global radiance and must still be cleared.
        Entry old = observed(10.0, 0.0, 1u, 0u);
        old.samples = asuint(4.0);
        old.origin.w = 4.0;
        old.merged = float4(10.0, 10.0, 10.0, 1.0);
        old.boundaryInterval = lightHitBoundsAt(float3(0.5));
        old.boundaryMerged = old.boundaryInterval;
        previousEntries[0] = old;
        retainLightRevisionEntry();
        mergeEntry(retainedIndex);
        recordLightRevision(36u, entries[retainedIndex]);
    } else {
        previousEntries[0] = entries[retainedIndex];
        retainLightRevisionEntry();
        mergeEntry(retainedIndex);
        recordLightRevision(39u, entries[retainedIndex]);
    }
}
void betaGeometryCases() {
    uint retainedIndex = currentCapacity();
    if (params.frame.w == 0u) {
        Entry old = observed(10.0, 0.2, 1u, 0u);
        old.samples = asuint(4.0);
        old.origin.w = 4.0;
        old.boundaryInterval = lightHitBoundsAt(float3(0.5));
        old.boundaryMerged = old.boundaryInterval;
        old.normal = dependencySegmentSphere(float3(0.0), float3(1.0, 0.0, 0.0), 0.0);
        DependencyContext combined = dependencyEmpty();
        dependencyIncludeBounds(combined, float3(0.0, -0.01, -0.01),
            float3(1.0, 0.01, 0.01));
        dependencyIncludeBounds(combined, float3(5.0, -0.01, -0.01),
            float3(6.0, 0.01, 0.01));
        storeEntryDependency(old, combined);
        storeEntryMergedDependency(old, combined);
        previousEntries[0] = old;

        Entry current = observed(0.0, 0.0, 0u, 0u);
        current.key = old.key;
        entries[0] = current;
        mergeEntry(0u);
        recordLightRevision(30u, entries[0]);
    } else if (params.frame.w == 1u) {
        // Geometry changed only on the outgoing NEE shadow path (x=5..6).
        // The primary interval is clean, so keep beta while rejecting stale J.
        previousEntries[0] = entries[0];
        retainLightRevisionEntry();
        mergeEntry(retainedIndex);
        results[32] = entries[retainedIndex].interval;
        results[33] = float4(asfloat(entries[retainedIndex].samples),
            entries[retainedIndex].origin.w, entries[retainedIndex].merged.w, 0.0);
    } else if (params.frame.w == 2u) {
        // A later clean frame must retain visibility without restoring old J.
        previousEntries[0] = entries[retainedIndex];
        retainLightRevisionEntry();
        mergeEntry(retainedIndex);
        results[34] = entries[retainedIndex].interval;
        results[35] = float4(asfloat(entries[retainedIndex].samples),
            entries[retainedIndex].origin.w, entries[retainedIndex].merged.w, 0.0);
    } else {
        // A primary-segment edit invalidates the measured beta as well.
        Entry old = observed(10.0, 0.2, 1u, 0u);
        old.samples = asuint(4.0);
        old.origin.w = 4.0;
        old.boundaryInterval = lightHitBoundsAt(float3(0.5));
        old.boundaryMerged = old.boundaryInterval;
        old.normal = dependencySegmentSphere(float3(0.0), float3(1.0, 0.0, 0.0), 0.0);
        DependencyContext combined = dependencyEmpty();
        dependencyIncludeBounds(combined, float3(0.0, -0.01, -0.01),
            float3(1.0, 0.01, 0.01));
        dependencyIncludeBounds(combined, float3(5.0, -0.01, -0.01),
            float3(6.0, 0.01, 0.01));
        storeEntryDependency(old, combined);
        storeEntryMergedDependency(old, combined);
        previousEntries[0] = old;
        entries[0] = observed(0.0, 0.0, 0u, 0u);
        entries[0].key = old.key;
        heads[0].store(EMPTY);
        retainLightRevisionEntry();
        results[38] = float4(heads[0].load() != EMPTY ? 1.0 : 0.0, 0.0, 0.0, 0.0);
        mergeEntry(0u);
        results[36] = entries[0].interval;
        results[37] = float4(asfloat(entries[0].samples), entries[0].origin.w,
            entries[0].merged.w, 0.0);
    }
}
void betaGeometryPredicateCases() {
    DependencyContext longRay = dependencyEmpty();
    dependencyIncludeBounds(longRay, float3(-10.0, 0.0, 0.0),
        float3(10.0, 0.0, 0.0));
    float4 longRaySphere = dependencySegmentSphere(
        float3(-10.0, 0.0, 0.0), float3(10.0, 0.0, 0.0), 0.0);
    DependencyContext shadowOnly = dependencyEmpty();
    dependencyIncludeBounds(shadowOnly, float3(0.0, 0.0, 0.0),
        float3(0.0, 10.0, 0.0));
    results[47] = float4(
        betaHistoryEligible(longRay, longRaySphere) ? 1.0 : 0.0,
        dependencyIntersectsDirty(longRay) ? 1.0 : 0.0,
        dependencySphereIntersectsDirty(longRaySphere) ? 1.0 : 0.0,
        betaHistoryEligible(shadowOnly, unknownLightHitBounds()) ? 1.0 : 0.0);
}
void betaBlendGeometryCases() {
    if (params.frame.w == 0u) {
        // A light-only revision rejects old J but can retain old beta from ray A.
        previousEntries[0] = observed(10.0, 0.2, 1u, 0u);
        previousEntries[0].samples = asuint(4.0);
        previousEntries[0].origin.w = 4.0;
        previousEntries[0].boundaryInterval = lightHitBoundsAt(float3(0.5, 0.0, 0.0));
        previousEntries[0].boundaryMerged = previousEntries[0].boundaryInterval;
        previousEntries[0].normal = dependencySegmentSphere(
            float3(0.0), float3(1.0, 0.0, 0.0), 0.0);
        DependencyContext oldDependency = dependencyEmpty();
        dependencyIncludeBounds(oldDependency, float3(0.0), float3(1.0, 0.0, 0.0));
        storeEntryDependency(previousEntries[0], oldDependency);
        storeEntryMergedDependency(previousEntries[0], oldDependency);

        // The fresh observation sees ray B. Its J is new, while beta still
        // contains retained visibility from A and must keep both dependencies.
        entries[0] = observed(4.0, 0.4, 1u, 0u);
        entries[0].boundaryInterval = lightHitBoundsAt(float3(3.5, 0.0, 0.0));
        entries[0].boundaryMerged = entries[0].boundaryInterval;
        entries[0].normal = dependencySegmentSphere(float3(3.0, 0.0, 0.0),
            float3(4.0, 0.0, 0.0), 0.0);
        DependencyContext freshDependency = dependencyEmpty();
        dependencyIncludeBounds(freshDependency, float3(3.0, 0.0, 0.0),
            float3(4.0, 0.0, 0.0));
        storeEntryDependency(entries[0], freshDependency);
        storeEntryMergedDependency(entries[0], freshDependency);
        mergeEntry(0u);
        results[42] = entries[0].interval;
        results[43] = float4(asfloat(entries[0].samples), entries[0].origin.w,
            entries[0].merged.w, 0.0);
        results[44] = float4(entries[0].dependencyMin.x, entries[0].dependencyMax.x,
            entries[0].normal.x, entries[0].normal.w);
    } else if (params.frame.w == 1u) {
        // A later geometry edit touches only A. Neither the clean fresh J AABB
        // nor the larger retained-beta sphere may hide this dependency.
        previousEntries[0] = entries[0];
        heads[0].store(EMPTY);
        retainLightRevisionEntry();
        results[47] = float4(heads[0].load() != EMPTY ? 1.0 : 0.0, 0.0, 0.0, 0.0);
        Entry current = observed(0.0, 0.0, 0u, 0u);
        current.key = previousEntries[0].key;
        entries[0] = current;
        mergeEntry(0u);
        results[45] = entries[0].interval;
        results[46] = float4(asfloat(entries[0].samples), entries[0].origin.w,
            entries[0].merged.w, 0.0);
    }
}
void crossFrameLightCases() {
    uint retainedIndex = currentCapacity();
    if (params.frame.w == 0u) {
        // The current mean blends a previous receiver at x=.25 with this
        // frame's receiver at x=.75. Its light dependency must cover both.
        previousEntries[0] = observed(2.0, 0.2, 1u, 0u);
        previousEntries[0].samples = asuint(2.0);
        previousEntries[0].origin.w = 1.0;
        previousEntries[0].boundaryInterval = lightHitBoundsAt(float3(0.25, 0.0, 0.0));
        previousEntries[0].boundaryMerged = previousEntries[0].boundaryInterval;

        entries[0] = observed(6.0, 0.2, 1u, 0u);
        entries[0].boundaryInterval = lightHitBoundsAt(float3(0.75, 0.0, 0.0));
        entries[0].boundaryMerged = entries[0].boundaryInterval;
        mergeEntry(0u);
        results[42] = float4(entries[0].interval.x, asfloat(entries[0].samples),
            entries[0].boundaryInterval.x, entries[0].boundaryInterval.w);
    } else if (params.frame.w == 1u) {
        // The light changes near only the old sample. That must invalidate the
        // complete mixed mean, even though the newest hit lies outside the box.
        previousEntries[0] = entries[0];
        retainLightRevisionEntry();
        mergeEntry(retainedIndex);
        results[43] = entries[retainedIndex].interval;
        results[44] = float4(asfloat(entries[retainedIndex].samples),
            entries[retainedIndex].origin.w, entries[retainedIndex].merged.w, 0.0);
    } else {
        // The one-frame light-change pulse has expired, but invalidated J must
        // remain absent until a fresh ray observes the new lighting.
        previousEntries[0] = entries[retainedIndex];
        retainLightRevisionEntry();
        mergeEntry(retainedIndex);
        results[45] = entries[retainedIndex].interval;
        results[46] = float4(asfloat(entries[retainedIndex].samples),
            entries[retainedIndex].origin.w, entries[retainedIndex].merged.w, 0.0);
    }
}
uint testChildDirection(uint direction, uint level, uint child) {
    uint theta = BASE_THETA << level;
    uint2 xy = uint2(direction % (2u * theta), direction / (2u * theta));
    uint2 fine = xy * 2u + uint2(child & 1u, child >> 1u);
    return fine.x + fine.y * (4u * theta);
}
Entry cascadeObservation(float radiance, float beta, int3 position,
                         uint direction, uint level) {
    Entry value = observed(radiance, beta, 1u, level);
    value.key = int4(position, int(direction | (level << 12u)));
    value.origin.xyz = (float3(position) + 0.5) * spacing(params, level, 0u);
    return value;
}
void insertCascadeEntry(inout uint next, Entry value) {
    uint index = next++;
    value.next = heads[0].load();
    entries[index] = value;
    heads[0].store(index);
}
void fourCascadeCases() {
    if (params.frame.y == 3u) {
        heads[0].store(EMPTY);
        uint next = 0u;
        uint parentDirections[4] = {0u, 1u, 16u, 17u};
        uint levelTwoDirections[16];
        uint directionCount = 0u;
        [unroll] for (uint parent = 0u; parent < 4u; ++parent)
            [unroll] for (uint child = 0u; child < 4u; ++child)
                levelTwoDirections[directionCount++] =
                    testChildDirection(parentDirections[parent], 1u, child);

        [loop] for (uint parent = 0u; parent < 16u; ++parent) {
            [unroll] for (uint child = 0u; child < 4u; ++child) {
                uint direction = testChildDirection(levelTwoDirections[parent], 2u, child);
                [unroll] for (uint corner = 0u; corner < 8u; ++corner) {
                    int3 bit = int3(corner & 1u, (corner >> 1u) & 1u, corner >> 2u);
                    insertCascadeEntry(next, cascadeObservation(1.0, 0.0,
                        int3(-1) + bit, direction, 3u));
                }
            }
        }
        [unroll] for (uint index = 0u; index < 16u; ++index)
            insertCascadeEntry(next, cascadeObservation(2.0, 0.5, int3(0),
                levelTwoDirections[index], 2u));
        [unroll] for (uint index = 0u; index < 4u; ++index)
            insertCascadeEntry(next, cascadeObservation(4.0, 0.5, int3(1),
                parentDirections[index], 1u));
        insertCascadeEntry(next, cascadeObservation(8.0, 0.25, int3(3), 0u, 0u));
    }
    if (params.frame.y == 3u) {
        [loop] for (uint index = 0u; index < 512u; ++index) mergeEntry(index);
        results[9] = entries[0].merged;
    } else if (params.frame.y == 2u) {
        [loop] for (uint index = 512u; index < 528u; ++index) mergeEntry(index);
        results[10] = entries[512].merged;
    } else if (params.frame.y == 1u) {
        [loop] for (uint index = 528u; index < 532u; ++index) mergeEntry(index);
        results[11] = entries[528].merged;
    } else {
        mergeEntry(532u);
        results[12] = entries[532].merged;
    }
}
[shader("compute")]
[numthreads(1, 1, 1)]
void checkMerge(uint3 id : SV_DispatchThreadID) {
    if (params.frame.z == 0u) temporalCases();
    else if (params.frame.z == 1u) supportCases();
    else if (params.frame.z == 2u) dependencyGeometryCases();
    else if (params.frame.z == 3u) dependencyLightCases();
    else if (params.frame.z == 4u) fourCascadeCases();
    else if (params.frame.z == 5u) lightRevisionCases();
    else if (params.frame.z == 6u) multiHitLightCases();
    else if (params.frame.z == 7u) globalLightCases();
    else if (params.frame.z == 8u) crossFrameLightCases();
    else if (params.frame.z == 9u) betaGeometryCases();
    else if (params.frame.z == 10u) betaGeometryPredicateCases();
    else betaBlendGeometryCases();
}
"#;

#[test]
#[ignore = "requires slangc and a desktop GPU; dispatches production Split retention and merging"]
fn gpu_split_merge_bounds_history_and_preserves_fractional_support() {
    let production = include_str!("../shaders/split.slang");
    let mut source = String::from(
        r#"
#include "common.slang"
[[vk::binding(0, 0)]] ConstantBuffer<Parameters> params;
[[vk::binding(6, 0)]] RWStructuredBuffer<Atomic<uint>> heads;
[[vk::binding(7, 0)]] RWStructuredBuffer<Entry> entries;
[[vk::binding(8, 0)]] StructuredBuffer<uint> previousHeads;
[[vk::binding(9, 0)]] RWStructuredBuffer<Entry> previousEntries;
[[vk::binding(13, 0)]] RWStructuredBuffer<float4> results;
struct Hit {
    float distance;
    uint triangle;
    uint instance;
    uint errors;
    float3 barycentric;
    float footprint;
};
static const uint HIT_QUERY_UNSUPPORTED = 4u;
"#,
    );
    for (start, end) in [
        ("uint gpuSortNext(", "bool dependencyIntersectsDirty("),
        (
            "bool dependencyIntersectsDirty(",
            "uint mergeVisibilityAdmissionLimit(",
        ),
        ("uint transportCacheBucket(", "float splitIntervalEnd("),
        ("float splitIntervalEnd(", "uint probeDirectoryCapacity("),
        ("uint findEntry(", "uint findIrradianceProbe("),
        ("float4 interpolateMergedChildren(", "[shader"),
        ("uint currentCapacity(", "[shader"),
        ("void retain(", "void deposit("),
        ("void reduce(", "void mergeEntry("),
        ("void mergeEntry(", "[shader"),
        ("void deposit(", "void traceTransport("),
    ] {
        source.push_str(production_section(production, start, end));
        source.push('\n');
    }
    source.push_str(MERGE_CASES);
    let shader = SlangCompiler::default()
        .compile_bundle(
            "split_merge_check.slang",
            &[
                ("common.slang", include_str!("../shaders/common.slang")),
                (
                    "linked_merge_sort.slang",
                    include_str!("../shaders/linked_merge_sort.slang"),
                ),
                ("split_merge_check.slang", &source),
            ],
            &SlangSettings {
                target: SlangTarget::Wgsl,
                entry_points: vec!["checkMerge".into()],
                optimization: Some(2),
                ..default()
            },
        )
        .unwrap();
    let bevy::shader::Source::Wgsl(source) = shader.source else {
        panic!("expected WGSL from Split merge regression");
    };
    let options = WgpuSettings::default();
    let resources = block_on(initialize_renderer(
        options.backends.unwrap_or(Backends::PRIMARY),
        None,
        &options,
    ));
    let device = &resources.0;
    let queue = &resources.1;
    let spirv = resources.2.backend == wgpu::Backend::Vulkan
        && device
            .features()
            .contains(wgpu::Features::PASSTHROUGH_SHADERS);
    let layout_entries = BindGroupLayoutEntries::with_indices(
        ShaderStages::COMPUTE,
        (
            (
                0,
                bevy::render::render_resource::binding_types::uniform_buffer::<TransportParameters>(
                    false,
                ),
            ),
            (
                6,
                bevy::render::render_resource::binding_types::storage_buffer_sized(false, None),
            ),
            (
                7,
                bevy::render::render_resource::binding_types::storage_buffer_sized(false, None),
            ),
            (
                8,
                bevy::render::render_resource::binding_types::storage_buffer_read_only_sized(
                    false, None,
                ),
            ),
            (
                9,
                bevy::render::render_resource::binding_types::storage_buffer_sized(false, None),
            ),
            (
                13,
                bevy::render::render_resource::binding_types::storage_buffer_sized(false, None),
            ),
        ),
    );
    let module = if spirv {
        let (words, workgroup_size) =
            crate::render::vulkan_compute_source(source.as_ref(), "checkMerge", &layout_entries)
                .unwrap();
        // SAFETY: Naga validates this fixture's WGSL and lowers it for the exact bind layout.
        unsafe {
            device.wgpu_device().create_shader_module_passthrough(
                wgpu::ShaderModuleDescriptorPassthrough {
                    label: Some("split_merge_regression"),
                    spirv: Some(std::borrow::Cow::Owned(words)),
                    entry_points: std::borrow::Cow::Owned(vec![
                        wgpu::PassthroughShaderEntryPoint {
                            name: std::borrow::Cow::Borrowed("checkMerge"),
                            workgroup_size,
                        },
                    ]),
                    ..default()
                },
            )
        }
    } else {
        device.create_and_validate_shader_module(ShaderModuleDescriptor {
            label: Some("split_merge_regression"),
            source: ShaderSource::Wgsl(source),
        })
    };
    let layout = device.create_bind_group_layout("split_merge_regression", &layout_entries);
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("split_merge_regression"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&RawComputePipelineDescriptor {
        label: Some("split_merge_regression"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("checkMerge"),
        compilation_options: default(),
        cache: None,
    });
    let run = |radiance_blend: f32, beta_blend: f32, case: u32| {
        let mut parameters = temporal_test_parameters(30);
        parameters.size = UVec4::new(1, 1, 1, 1);
        parameters.history = Vec4::new(radiance_blend, 0.0, 1.0, 30.0);
        parameters.interval_history = Vec4::new(beta_blend, 1.0, 0.0, 0.0);
        parameters.frame = UVec4::new(1024, if case == 0 { 3 } else { 0 }, case, 0);
        if case == 2 {
            parameters.dirty_min = Vec4::new(0.0, 0.0, 0.0, 1.0);
            parameters.dirty_max = Vec4::ONE;
        } else if case == 3 {
            parameters.transport.w = 1.0;
            parameters.light_dirty_min = Vec4::new(0.0, 0.0, 0.0, 1.0);
            parameters.light_dirty_max = Vec4::ONE;
        }
        let mut parameters = UniformBuffer::from(parameters);
        parameters.write_buffer(device, queue);
        let make_buffer = |label, bytes: &[u8]| {
            device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            })
        };
        let heads = make_buffer("split_merge_heads", &[0u8; 64 * 4]);
        let entries = make_buffer("split_merge_entries", &[0u8; 1024 * 208]);
        let previous_heads = make_buffer("split_merge_previous_heads", &[0u8; 4]);
        let previous_entries = make_buffer("split_merge_previous_entries", &[0u8; 208]);
        let output = make_buffer("split_merge_results", &[0u8; 48 * 16]);
        let group = device.create_bind_group(
            "split_merge_regression",
            &pipeline.get_bind_group_layout(0).into(),
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: parameters.binding().unwrap(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: heads.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: entries.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: previous_heads.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: previous_entries.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 13,
                    resource: output.as_entire_binding(),
                },
            ],
        );
        if case == 1 {
            // Mirror the production pass boundary: each canonical entry is
            // merged once per dispatch after its inputs have been prepared.
            for stage in 0..=7 {
                parameters.get_mut().frame.w = stage;
                parameters.write_buffer(device, queue);
                let mut encoder =
                    device.create_command_encoder(&CommandEncoderDescriptor::default());
                {
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &group, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
                queue.submit([encoder.finish()]);
            }
            // dispatch_readback performs one final dispatch for all cases.
            parameters.get_mut().frame.w = 8;
            parameters.write_buffer(device, queue);
        } else if case == 4 {
            for level in [3, 2, 1] {
                parameters.get_mut().frame.y = level;
                parameters.write_buffer(device, queue);
                let mut encoder =
                    device.create_command_encoder(&CommandEncoderDescriptor::default());
                {
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &group, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
                queue.submit([encoder.finish()]);
            }
            parameters.get_mut().frame.y = 0;
            parameters.write_buffer(device, queue);
        } else if case == 5 {
            for stage in 0..=6 {
                let dirty_light = stage == 0 || stage == 4;
                parameters.get_mut().frame.y = 0;
                parameters.get_mut().frame.w = stage;
                parameters.get_mut().transport.w = if dirty_light { 1.0 } else { 0.0 };
                parameters.get_mut().light_dirty_min = if dirty_light {
                    Vec4::new(0.25, 0.25, 0.25, 1.0)
                } else {
                    Vec4::ZERO
                };
                parameters.get_mut().light_dirty_max = if dirty_light {
                    Vec4::new(0.75, 0.75, 0.75, 1.0)
                } else {
                    Vec4::ZERO
                };
                parameters.write_buffer(device, queue);
                let mut encoder =
                    device.create_command_encoder(&CommandEncoderDescriptor::default());
                {
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &group, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
                queue.submit([encoder.finish()]);
            }
        } else if case == 6 {
            for stage in 0..=4 {
                let dirty_light = stage == 3;
                parameters.get_mut().frame.y = 0;
                parameters.get_mut().frame.w = stage;
                parameters.get_mut().transport.w = if dirty_light { 1.0 } else { 0.0 };
                parameters.get_mut().light_dirty_min = if dirty_light {
                    Vec4::new(0.70, -0.1, -0.1, 1.0)
                } else {
                    Vec4::ZERO
                };
                parameters.get_mut().light_dirty_max = if dirty_light {
                    Vec4::new(0.80, 0.1, 0.1, 1.0)
                } else {
                    Vec4::ZERO
                };
                parameters.write_buffer(device, queue);
                let mut encoder =
                    device.create_command_encoder(&CommandEncoderDescriptor::default());
                {
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &group, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
                queue.submit([encoder.finish()]);
            }
        } else if case == 7 {
            for stage in 0..=3 {
                parameters.get_mut().frame.y = 0;
                parameters.get_mut().frame.w = stage;
                parameters.get_mut().transport.w = if stage < 3 { 1.0 } else { 0.0 };
                parameters.get_mut().light_dirty_min = Vec4::ZERO;
                parameters.get_mut().light_dirty_max = Vec4::ZERO;
                parameters.write_buffer(device, queue);
                let mut encoder =
                    device.create_command_encoder(&CommandEncoderDescriptor::default());
                {
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &group, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
                queue.submit([encoder.finish()]);
            }
        } else if case == 8 {
            for stage in 0..=2 {
                parameters.get_mut().frame.y = 0;
                parameters.get_mut().frame.w = stage;
                let dirty_light = stage == 1;
                parameters.get_mut().transport.w = if dirty_light { 1.0 } else { 0.0 };
                parameters.get_mut().light_dirty_min = if dirty_light {
                    Vec4::new(0.24, -0.01, -0.01, 1.0)
                } else {
                    Vec4::ZERO
                };
                parameters.get_mut().light_dirty_max = if dirty_light {
                    Vec4::new(0.26, 0.01, 0.01, 1.0)
                } else {
                    Vec4::ZERO
                };
                parameters.write_buffer(device, queue);
                let mut encoder =
                    device.create_command_encoder(&CommandEncoderDescriptor::default());
                {
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &group, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
                queue.submit([encoder.finish()]);
            }
        } else if case == 9 {
            for stage in 0..=3 {
                parameters.get_mut().frame.y = 0;
                parameters.get_mut().frame.w = stage;
                if stage == 1 {
                    parameters.get_mut().dirty_min = Vec4::new(5.0, -0.1, -0.1, 1.0);
                    parameters.get_mut().dirty_max = Vec4::new(6.0, 0.1, 0.1, 1.0);
                } else if stage == 3 {
                    parameters.get_mut().dirty_min = Vec4::new(0.0, -0.1, -0.1, 1.0);
                    parameters.get_mut().dirty_max = Vec4::new(1.0, 0.1, 0.1, 1.0);
                } else {
                    parameters.get_mut().dirty_min = Vec4::ZERO;
                    parameters.get_mut().dirty_max = Vec4::ZERO;
                }
                parameters.write_buffer(device, queue);
                let mut encoder =
                    device.create_command_encoder(&CommandEncoderDescriptor::default());
                {
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &group, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
                queue.submit([encoder.finish()]);
            }
        } else if case == 10 {
            parameters.get_mut().dirty_min = Vec4::new(-0.1, 8.9, -0.1, 1.0);
            parameters.get_mut().dirty_max = Vec4::new(0.1, 9.1, 0.1, 1.0);
            parameters.write_buffer(device, queue);
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            queue.submit([encoder.finish()]);
        } else if case == 11 {
            for stage in 0..=1 {
                parameters.get_mut().frame.y = 0;
                parameters.get_mut().frame.w = stage;
                if stage == 0 {
                    parameters.get_mut().transport.w = 1.0;
                    parameters.get_mut().light_dirty_min = Vec4::new(0.45, -0.1, -0.1, 1.0);
                    parameters.get_mut().light_dirty_max = Vec4::new(0.55, 0.1, 0.1, 1.0);
                    parameters.get_mut().dirty_min = Vec4::ZERO;
                    parameters.get_mut().dirty_max = Vec4::ZERO;
                } else {
                    parameters.get_mut().transport.w = 0.0;
                    parameters.get_mut().light_dirty_min = Vec4::ZERO;
                    parameters.get_mut().light_dirty_max = Vec4::ZERO;
                    parameters.get_mut().dirty_min = Vec4::new(0.0, -0.1, -0.1, 1.0);
                    parameters.get_mut().dirty_max = Vec4::new(1.0, 0.1, 0.1, 1.0);
                }
                parameters.write_buffer(device, queue);
                let mut encoder =
                    device.create_command_encoder(&CommandEncoderDescriptor::default());
                {
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &group, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
                queue.submit([encoder.finish()]);
            }
            // dispatch_readback performs one final dispatch; stage 2 is inert.
            parameters.get_mut().frame.w = 2;
            parameters.write_buffer(device, queue);
        }
        let bytes = crate::scene::traversal_tests::dispatch_readback(
            device,
            queue,
            &pipeline,
            &group,
            &output,
            48 * 16,
            1,
        );
        std::array::from_fn::<_, 48, _>(|index| output_vec4(&bytes, index))
    };
    let close = |actual: f32, expected: f32| {
        assert!(
            (actual - expected).abs() < 2e-4,
            "actual={actual}, expected={expected}"
        );
    };
    for (j, beta) in [(0.5f32, 0.75f32), (0.9, 0.95)] {
        let result = run(j, beta, 0);
        close(result[1][0], 1.0 / (1.0 - j));
        close(result[1][1], 1.0 / (1.0 - beta));
        close(result[2][0], 2.0 * j);
        close(result[2][3], 1.0 - beta);
        assert_eq!(
            result[2], result[4],
            "zero observations must not decay world state"
        );
        assert_eq!(
            result[3], result[5],
            "zero observations must not decay mass or support"
        );
        close(result[9][0], 2.0 * j.powi(4));
        close(result[9][3], 1.0 - beta.powi(4));
        close(result[10][0], 1.0 / (1.0 - j));
        close(result[10][1], 1.0 / (1.0 - beta));
        assert_eq!(
            result[11],
            [0.0, 0.0, 0.0, 1.0],
            "escape is supported zero at the terminal interval"
        );
        assert_eq!(
            result[12],
            [4.0, 2.0, 1.0, 1.0],
            "the final interval owns distant hits"
        );
        assert!(
            result[13][2] >= 20000.0,
            "the terminal ray dependency must cover its distant hit"
        );
        assert_eq!(
            result[14], [0.0; 4],
            "unsupported queries must not deposit a terminal observation"
        );
        close(result[6][0], 1.0);
    }
    let beta_only = run(0.0, 0.9, 0);
    close(beta_only[2][0], 0.0);
    close(beta_only[2][3], 0.1);
    close(beta_only[5][0], 0.0);
    close(beta_only[5][1], 10.0);
    close(beta_only[5][2], 0.0);
    close(beta_only[6][0], 1.0);
    close(beta_only[8][0], 0.0);
    close(beta_only[8][1], 10.0);
    close(beta_only[8][2], 0.0);
    let radiance_only = run(0.9, 0.0, 0);
    close(radiance_only[2][0], 1.8);
    close(radiance_only[2][3], 1.0);
    close(radiance_only[5][0], 10.0);
    close(radiance_only[5][1], 0.0);
    close(radiance_only[5][2], 0.0);
    let disabled = run(0.0, 0.0, 0);
    close(disabled[6][0], 0.0);
    let result = run(0.0, 0.9, 1);
    close(result[0][0], 3.6);
    close(result[0][3], 0.625);
    close(result[1][0], 3.125 / 0.8125);
    close(result[1][3], 0.8125);
    close(result[2][0], 6.0);
    close(result[2][3], 1.0);
    assert_eq!(result[3], [0.0, 0.0, 0.0, 1.0]);
    close(result[4][0], 2.25 / 0.53125);
    close(result[4][3], 0.53125);
    close(result[5][0], 8.0);
    close(result[5][3], 0.0625);
    assert_eq!(result[6], [0.0; 4]);
    close(result[7][0], 0.2);
    close(result[7][2], 0.8);
    close(result[7][3], 0.625);
    close(result[8][0], 0.8);
    close(result[8][2], 0.2);
    close(result[8][3], 0.625);

    let dependency = run(0.5, 0.0, 2);
    close(dependency[13][0], 6.0);
    close(dependency[13][1], 2.0);
    close(dependency[13][2], 2.0);

    let local_light = run(0.5, 0.5, 3);
    close(local_light[14][0], 2.0);
    close(local_light[14][1], 0.5);
    close(local_light[14][2], 1.0);
    close(local_light[14][3], 2.0);
    assert_eq!(local_light[15], [1.0, 0.0, 1.0, 1.0]);
    assert_eq!(local_light[16][0], 0.0, "unknown light bounds reject reuse");

    let cascade = run(0.0, 0.0, 4);
    close(cascade[9][0], 1.0);
    close(cascade[9][3], 1.0);
    close(cascade[10][0], 2.5);
    close(cascade[10][3], 1.0);
    let spatial_support = 0.75f32.powi(3);
    let level_one_support = 0.5 + 0.5 * spatial_support;
    let level_one_radiance = 4.0 + 0.5 * 2.5 * spatial_support;
    let level_one_mean = level_one_radiance / level_one_support;
    close(cascade[11][0], level_one_mean);
    close(cascade[11][3], level_one_support);
    let level_zero_support = 0.75 + 0.25 * spatial_support * level_one_support;
    // Child RGB is a conditional mean, so its fractional support also weights
    // the parent radiance numerator.
    let level_zero_radiance = 8.0 + 0.25 * spatial_support * level_one_radiance;
    close(cascade[12][0], level_zero_radiance / level_zero_support);
    close(cascade[12][3], level_zero_support);

    let light_revisions = run(0.5, 0.5, 5);
    // First light revision clears old radiance but preserves visibility.
    close(light_revisions[0][0], 0.0);
    close(light_revisions[0][3], 0.2);
    close(light_revisions[1][0], 0.0);
    close(light_revisions[1][1], 4.0);
    assert_eq!(
        light_revisions[2], [0.0; 4],
        "stale merged I must be cleared"
    );
    // Once the dirty-light bounds expire, zero radiance mass stays zero.
    close(light_revisions[3][0], 0.0);
    close(light_revisions[4][0], 0.0);
    assert_eq!(light_revisions[5], [0.0; 4]);
    // A new trace contributes without mixing the pre-move radiance.
    close(light_revisions[6][0], 2.0);
    close(light_revisions[6][3], 0.5);
    close(light_revisions[7][0], 1.0);
    close(light_revisions[7][1], 2.0);
    close(light_revisions[7][2], 0.5);
    close(light_revisions[8][0], 4.0);
    close(light_revisions[8][3], 0.5);
    // The clean frame keeps the fresh radiance until a second light revision.
    close(light_revisions[9][0], 2.0);
    close(light_revisions[10][0], 1.0);
    close(light_revisions[10][1], 2.0);
    assert_eq!(light_revisions[11], [4.0, 4.0, 4.0, 0.5]);
    // That second revision clears J, and a later clean frame cannot resurrect
    // it from retained beta or the previously merged field.
    close(light_revisions[12][0], 0.0);
    close(light_revisions[13][0], 0.0);
    close(light_revisions[13][1], 2.0);
    assert_eq!(light_revisions[14], [0.0; 4]);
    close(light_revisions[15][0], 0.0);
    close(light_revisions[16][0], 0.0);
    close(light_revisions[16][1], 2.0);
    assert_eq!(light_revisions[17], [0.0; 4]);
    // A later fresh observation restores support using the new radiance.
    close(light_revisions[18][0], 4.0);
    close(light_revisions[18][3], 0.45);
    close(light_revisions[19][0], 1.0);
    close(light_revisions[19][1], 2.0);
    close(light_revisions[19][2], 0.55);
    close(light_revisions[20][0], 4.0 / 0.55);
    close(light_revisions[20][3], 0.55);

    let multi_hit = run(0.5, 0.5, 6);
    close(multi_hit[30][0], 2.0);
    close(multi_hit[31][0], 6.0);
    close(multi_hit[32][0], 4.0);
    close(multi_hit[32][3], 0.2);
    close(multi_hit[33][0], 2.0);
    close(multi_hit[21][0], 4.0);
    close(multi_hit[21][3], 0.2);
    close(multi_hit[22][0], 1.5);
    close(multi_hit[22][1], 1.5);
    close(multi_hit[22][2], 0.8);
    close(multi_hit[23][0], 5.0);
    close(multi_hit[23][3], 0.8);
    close(multi_hit[24][0], 0.0);
    close(multi_hit[25][0], 0.0);
    close(multi_hit[25][1], 1.5);
    assert_eq!(multi_hit[26], [0.0; 4]);
    close(multi_hit[27][0], 0.0);
    assert_eq!(multi_hit[29], [0.0; 4]);

    let global_light = run(0.5, 0.5, 7);
    close(global_light[30][0], 0.0);
    close(global_light[31][0], 4.0);
    close(global_light[31][1], 4.0);
    close(global_light[31][2], 1.0);
    assert_eq!(global_light[32], [0.0, 0.0, 0.0, 1.0]);
    close(global_light[33][0], 0.0);
    close(global_light[34][0], 4.0);
    close(global_light[34][1], 4.0);
    close(global_light[34][2], 1.0);
    assert_eq!(global_light[35], [0.0, 0.0, 0.0, 1.0]);
    close(global_light[36][0], 0.0);
    close(global_light[37][0], 0.0);
    close(global_light[37][1], 4.0);
    assert_eq!(global_light[38], [0.0; 4]);
    close(global_light[39][0], 0.0);
    close(global_light[40][0], 0.0);
    close(global_light[40][1], 4.0);
    close(global_light[40][2], 0.0);
    assert_eq!(global_light[41], [0.0; 4]);

    let cross_frame = run(0.5, 0.5, 8);
    close(cross_frame[42][0], 4.0);
    close(cross_frame[42][1], 2.0);
    close(cross_frame[42][2], 0.5);
    assert!(
        cross_frame[42][3] >= 0.25,
        "old and fresh hit bounds must be unioned"
    );
    close(cross_frame[43][0], 0.0);
    close(cross_frame[43][3], 0.2);
    close(cross_frame[44][0], 0.0);
    close(cross_frame[44][1], 1.5);
    assert_eq!(cross_frame[45][0], 0.0);
    close(cross_frame[46][0], 0.0);

    let beta_geometry = run(0.5, 0.5, 9);
    close(beta_geometry[30][0], 10.0);
    close(beta_geometry[30][3], 0.2);
    close(beta_geometry[32][0], 0.0);
    close(beta_geometry[32][3], 0.2);
    close(beta_geometry[33][0], 0.0);
    close(beta_geometry[33][1], 4.0);
    close(beta_geometry[34][0], 0.0);
    close(beta_geometry[34][3], 0.2);
    close(beta_geometry[35][0], 0.0);
    close(beta_geometry[35][1], 4.0);
    assert_eq!(beta_geometry[36][0], 0.0);
    close(beta_geometry[36][3], 1.0);
    close(beta_geometry[37][0], 0.0);
    close(beta_geometry[37][1], 0.0);
    assert_eq!(
        beta_geometry[38][0], 0.0,
        "primary geometry changes drop retained beta"
    );

    let long_ray = run(0.5, 0.5, 10);
    assert_eq!(
        long_ray[47],
        [1.0, 0.0, 1.0, 0.0],
        "a clean segment AABB must override a sphere false positive; unknown bounds fall back",
    );

    let beta_blend = run(0.5, 0.5, 11);
    close(beta_blend[42][0], 4.0);
    close(beta_blend[42][3], 0.3);
    close(beta_blend[43][0], 1.0);
    close(beta_blend[43][1], 2.0);
    close(beta_blend[44][0], 0.0);
    close(beta_blend[44][1], 4.0);
    close(beta_blend[44][2], 2.0);
    assert!(beta_blend[44][3] >= 2.0);
    assert_eq!(beta_blend[45][0], 0.0);
    close(beta_blend[45][3], 1.0);
    close(beta_blend[46][0], 0.0);
    close(beta_blend[46][1], 0.0);
    assert_eq!(beta_blend[47][0], 0.0);
}

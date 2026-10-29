# Completing Split Radiance Cascades

* **Author:** Codex, for user review
* **Date:** 2026-10-25
* **Status:** Working hypothesis; clarification in progress. No Goldfish review has run.

## Overview

Complete the Bevy implementation using the supplied Split Radiance Cascades references, while addressing missing-sample dark rims, moving-light response, and rendering cost. Completion needs measurable visual, timing, and lighting-response requirements; passing static captures alone is insufficient.

The requested working arrangement is GPT-6.1 Sol High for architecture and GPT-6 Luna Max for implementation. EGM reviewers will need fresh contexts after the user confirms the review hypothesis.

### User direction

The user wants the best attainable visual quality and performance, with hardware-agnostic behavior and the PDF paper as the reference. No particular GPU or numeric frame-rate target is specified. The working interpretation is portable correctness with device-appropriate quality and work limits; identical quality and speed on every GPU is not an achievable acceptance condition. The rule for resolving quality versus frame-time conflicts remains open.

## Analysis

### Evidence already collected

* A GPU regression reproduced black reconstruction when the four bilinear anchors were incompatible even though a nearby compatible observation existed. A bounded fallback now retains the existing surface checks and preserves supported black. This does not establish that every dark narrow face or textured border has been eliminated.
* The implementation now preintegrates the 32 directional samples into the reference 6×6 octahedral irradiance table. Cooperative packing matches the CPU table reference, including coverage, dependency bounds, and zero-ray history.
* On the RTX 4070 Laptop GPU, the matched 1920×1080 benchmark measured median GI time decreasing from 30.54 ms to 15.37 ms. Observed frame interval decreased from 59.12 ms to 41.32 ms. Both runs used 512 updates, a 16,384-query cap, 120 warmup samples, and 240 measured samples. The frame timer is a queue envelope, not a sum of busy GPU passes.
* Continuous motion checks passed through frame 127. The explicit full history reset at frame 128 still produced about 93.9% eligible-pixel support, below the unchanged 95% requirement.
* A temporary matched-frame diagnostic classified the remaining cold holes: 724 had neighboring probes without populated irradiance; 151 had populated probes without target-direction support; none lacked a neighboring descriptor. The temporary diagnostic was removed.

### Work not yet accepted

Cold-start selection is being revised to spread the existing ray budget across visible probes before filling remaining capacity. Review found an admission race; the fix and its regression are not yet fully validated. Final cold-frame acceptance and the timed extinction/relight checks remain open.

The parsed Markdown confirms the core preintegration and sampling structure. Cold-start prioritization is an implementation policy, not a paper guarantee of 95% first-frame coverage at 512 rays.

### Trade-offs to challenge

* Coarser probes can improve coverage but lose spatial detail. Additional scheduling must justify its complexity and cost.
* More filtering can hide missing data or spread light across boundaries. Visual improvement needs surface-rejection and valid-black checks.
* Relaxing acceptance thresholds would make a test pass without establishing the requested lighting behavior.
* The measured speedup does not establish a 60 FPS result or performance on other hardware.

## Goals vs. Non-Goals

* **In scope:** Reference-based transport, dark-rim diagnosis, moving-light correctness, bounded work, measurable performance, and reproducible validation.
* **Preserve:** Existing work, surface identity and visibility checks, valid-black meaning, and the distinction between unknown and measured radiance.
* **Not currently proposed:** A new rendering backend, new public quality controls, extra dependencies, or weakened validation gates.
* **Still to define with the user:** How quality should scale when a device cannot sustain the requested work, acceptable lighting-response delay, supported rendering capabilities, visual tolerances, and the exact completion boundary.

## Key Decisions & Staging for Next Steps

1. Clarify the measurable completion criteria through open-ended discussion.
2. Update the anchored hypothesis and obtain explicit readiness for Goldfish review.
3. Run fresh-context reviews against the same brief, then record objections and decisions here.
4. Resume implementation and acceptance validation against the agreed requirements.

## Evidence

* [Parsed reference paper](<C:/Dev/Bevy/crates/bevy_penumbra/ref/Split Radiance Cascades.md>)
* [1080p baseline](C:/Dev/Bevy/crates/bevy_penumbra/tmp/perf-before-1080.json)
* [1080p cooperative preintegration](C:/Dev/Bevy/crates/bevy_penumbra/tmp/perf-cooperative-1080.json)
* [Cooperative table GPU check](C:/Dev/Bevy/crates/bevy_penumbra/tmp/cooperative-packer-verified.log)
* [Cold-frame classification](C:/Dev/Bevy/crates/bevy_penumbra/tmp/cold-classifier-sponza.log)

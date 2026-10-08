#pragma once

// Guardian-APC+ extensions:
//   1. Rewrite-aware targeting: aim boundary probes at the activations a
//      rewrite actually touches (and everything downstream of them).
//   2. Near-miss search: treat sub-tolerance output divergence as a signal and
//      climb it until the fault becomes observable.
//   3. Divergence localisation: report the first IR value where the original
//      and rewritten graphs disagree on a witness input.

#include "analysis.h"
#include "ir.h"
#include "verifier.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace modelforge {

struct RewriteImpact {
    std::vector<std::string> changedNodes;   // instructions of `before` the rewrite altered or removed
    std::vector<ActivationSite> impactedSites;  // live ReLU sites at or downstream of a change
    bool anyChange = false;
};

RewriteImpact analyzeRewriteImpact(const IRGraph& before, const IRGraph& after,
                                   const std::vector<ActivationSite>& sites);

// Static rewrite-aware suite: boundary probes for impacted sites first, ordered
// by coverage of the impacted sites, then the regular Guardian-APC pool.
std::vector<ValidationProbe> generateRewriteAwareProbes(const IRGraph& before,
                                                        const IRGraph& after,
                                                        std::uint32_t seed = 20260917);

struct GuardianPlusOptions {
    std::size_t staticBudget = 0;    // 0 = use the whole static suite
    std::size_t adaptiveBudget = 32;  // candidate evaluations for near-miss search
    bool rewriteAware = true;
    std::uint32_t seed = 20260917;
    float tolerance = 1.0e-5f;
    float radius = kDefaultInputRadius;
};

struct GuardianPlusResult {
    std::size_t firstDetection = 0;  // 1-based evaluation index, 0 = not detected
    std::size_t evaluations = 0;
    std::size_t staticProbes = 0;
    bool foundByNearMiss = false;
    float maximumAbsoluteError = 0.0f;
    float bestNearMissScore = 0.0f;
    std::optional<ValidationProbe> witness;
    std::vector<float> originalOutput;
    std::vector<float> candidateOutput;
};

// Runs the static suite (rewrite-aware or global Guardian-APC), then spends the
// adaptive budget on near-miss search seeded from the most divergent probes.
GuardianPlusResult runGuardianPlus(const IRGraph& before, const IRGraph& candidate,
                                   const GuardianPlusOptions& options = {});

// Divergence score of one probe: max output difference plus a small multiple of
// the largest difference on any live intermediate value shared by both graphs.
float divergenceScore(const IRGraph& before, const IRGraph& candidate,
                      const std::vector<float>& input, float tolerance, bool& detected,
                      std::vector<float>* originalOutput = nullptr,
                      std::vector<float>* candidateOutput = nullptr);

struct DivergenceLocation {
    bool found = false;
    std::size_t instruction = 0;  // index in `before`
    std::string node;
    std::string value;
    float maximumDifference = 0.0f;
};

// First value, in `before`'s execution order, that differs beyond tolerance.
DivergenceLocation localizeDivergence(const IRGraph& before, const IRGraph& candidate,
                                      const std::vector<float>& input,
                                      float tolerance = 1.0e-5f);

}  // namespace modelforge

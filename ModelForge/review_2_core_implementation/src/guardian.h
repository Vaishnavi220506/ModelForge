#pragma once

#include "adaptive.h"
#include "optimizer.h"
#include "verifier.h"

#include <string>
#include <vector>

namespace modelforge {

struct GuardianDecision {
    std::string passName;
    bool accepted = false;
    OptimizationReport optimization;
    TranslationValidationReport validation;
    bool foundByNearMiss = false;
    DivergenceLocation divergence;  // where the minimised witness first diverges
};

struct GuardianReport {
    OptimizationReport acceptedOptimizations;
    std::vector<GuardianDecision> decisions;
};

// Checks a single proposed rewrite against the graph immediately before it.
// On failure, the first counterexample is simplified while preserving failure.
GuardianDecision checkCandidate(const IRGraph& before,
                                const IRGraph& candidate,
                                const std::string& passName,
                                DiagnosticEngine& diagnostics,
                                float tolerance = 1.0e-5f);

// Every failed pass is rolled back. Accepted passes are checked independently.
GuardianReport optimizeWithGuardian(IRGraph& graph, DiagnosticEngine& diagnostics);

}  // namespace modelforge

#include "guardian.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace modelforge {
namespace {

void simplifyFailure(const IRGraph& before,
                     const IRGraph& candidate,
                     TranslationValidationReport& validation,
                     DiagnosticEngine& diagnostics,
                     float tolerance) {
    if (!validation.firstFailure) return;
    ValidationProbe best = *validation.firstFailure;
    // Shrink toward zero. Each accepted edit still has to reproduce the mismatch.
    for (std::size_t index = 0; index < best.values.size(); ++index) {
        for (int attempt = 0; attempt < 4; ++attempt) {
            ValidationProbe trial = best;
            trial.values[index] = attempt == 0 ? 0.0f : best.values[index] * 0.5f;
            DiagnosticEngine trialDiagnostics;
            auto result = validateOptimization(before, candidate, {trial}, trialDiagnostics, tolerance);
            if (!result.passed && result.firstFailure && !trialDiagnostics.hasErrors()) {
                best = std::move(trial);
                validation.originalOutput = std::move(result.originalOutput);
                validation.candidateOutput = std::move(result.candidateOutput);
            }
        }
    }
    (void)diagnostics;
    best.name = "minimized_from_" + validation.firstFailure->name;
    validation.firstFailure = std::move(best);
}

}  // namespace

GuardianDecision checkCandidate(const IRGraph& before,
                                const IRGraph& candidate,
                                const std::string& passName,
                                DiagnosticEngine& diagnostics,
                                float tolerance) {
    GuardianDecision decision;
    decision.passName = passName;
    const auto probes = generateValidationProbes(before);
    decision.validation = validateOptimization(before, candidate, probes, diagnostics, tolerance);
    decision.accepted = !probes.empty() && decision.validation.passed && !diagnostics.hasErrors();
    if (!decision.accepted) {
        simplifyFailure(before, candidate, decision.validation, diagnostics, tolerance);
    }
    return decision;
}

GuardianReport optimizeWithGuardian(IRGraph& graph, DiagnosticEngine& diagnostics) {
    GuardianReport result;
    result.acceptedOptimizations.instructionsBefore = graph.instructions.size();
    for (const auto pass : {OptimizationPass::ConstantFolding,
                            OptimizationPass::DenseReluFusion,
                            OptimizationPass::DeadNodeRemoval}) {
        IRGraph candidate = graph;
        auto passReport = runOptimizationPass(candidate, pass);
        GuardianDecision decision;
        if (passReport.events.empty()) {
            decision.passName = optimizationPassName(pass);
            decision.accepted = true;
            decision.validation.passed = true;
        } else {
            decision = checkCandidate(graph, candidate, optimizationPassName(pass), diagnostics);
        }
        decision.optimization = std::move(passReport);
        if (decision.accepted) {
            graph = std::move(candidate);
            auto& accepted = result.acceptedOptimizations;
            accepted.removedInstructions += decision.optimization.removedInstructions;
            accepted.fusedOperations += decision.optimization.fusedOperations;
            accepted.constantFolds += decision.optimization.constantFolds;
            accepted.events.insert(accepted.events.end(), decision.optimization.events.begin(),
                                   decision.optimization.events.end());
        }
        result.decisions.push_back(std::move(decision));
    }
    result.acceptedOptimizations.instructionsAfter = graph.instructions.size();
    return result;
}

}  // namespace modelforge

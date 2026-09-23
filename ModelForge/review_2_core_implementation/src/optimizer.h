#pragma once

#include "diagnostics.h"
#include "ir.h"

#include <string>
#include <vector>

namespace modelforge {

struct OptimizationReport {
    std::size_t instructionsBefore = 0;
    std::size_t instructionsAfter = 0;
    std::size_t removedInstructions = 0;
    std::size_t fusedOperations = 0;
    std::size_t constantFolds = 0;
    std::vector<std::string> events;
};

enum class OptimizationPass { ConstantFolding, DenseReluFusion, DeadNodeRemoval };
const char* optimizationPassName(OptimizationPass pass);
OptimizationReport runOptimizationPass(IRGraph& graph, OptimizationPass pass);
OptimizationReport optimize(IRGraph& graph, DiagnosticEngine& diagnostics);

}  // namespace modelforge

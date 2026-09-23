#pragma once

#include "diagnostics.h"
#include "ir.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace modelforge {

struct VerificationReport {
    bool passed = false;
    float maximumAbsoluteError = 0.0f;
    std::size_t mismatches = 0;
};

struct ValidationProbe {
    std::string name;
    std::vector<float> values;
};

struct TranslationValidationReport {
    bool passed = false;
    std::size_t probeCount = 0;
    std::size_t outputMismatches = 0;
    std::size_t predictionMismatches = 0;
    float maximumAbsoluteError = 0.0f;
    std::optional<ValidationProbe> firstFailure;
    std::vector<float> originalOutput;
    std::vector<float> candidateOutput;
};

std::optional<std::vector<float>> executeIR(const IRGraph& graph,
                                            const std::vector<float>& input,
                                            DiagnosticEngine& diagnostics);

VerificationReport compareOutputs(const std::vector<float>& expected,
                                   const std::vector<float>& actual,
                                   float tolerance = 1.0e-5f);

// Creates deterministic, graph-aware inputs that target activation boundaries,
// saturation, numerical range, and ordinary model behavior.
std::vector<ValidationProbe> generateValidationProbes(const IRGraph& graph,
                                                       std::uint32_t seed = 20260917);

// Empirically validates that an optimized graph preserves the original graph's
// outputs and predictions for the generated probes.
TranslationValidationReport validateOptimization(const IRGraph& original,
                                                 const IRGraph& optimized,
                                                 const std::vector<ValidationProbe>& probes,
                                                 DiagnosticEngine& diagnostics,
                                                 float tolerance = 1.0e-5f);

VerificationReport verifyWithOnnxRuntime(const std::string& onnxPath,
                                          const IRGraph& graph,
                                          const std::vector<float>& input,
                                          DiagnosticEngine& diagnostics,
                                          float tolerance = 1.0e-5f);

}  // namespace modelforge

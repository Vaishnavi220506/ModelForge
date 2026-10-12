#pragma once

// ForgeCert, part 2: differential abstract interpretation of the float model
// and its integer program, run together over a box of inputs.
//
// Every neuron carries two affine forms over shared noise symbols:
//   R  - the float model's value (exact real arithmetic, DeepZ ReLU relaxation)
//   E  - the error of the integer program, i.e. (scale * integer) - R
// Because R and E share the input symbols, the analysis keeps the correlation
// between "where the input is" and "how wrong the integer code is there", which
// interval reasoning loses. E is further split by *provenance*: one component
// per error source (input quantisation, each layer's weights incl. bias and
// requantisation multiplier, each layer's output rounding). The split is exact
// for affine steps and proportional for ReLU relaxations; the sum is what the
// proofs use, the parts tell the precision tuner which layer to blame.
//
// The analysis is sound for the bit-exact integer semantics of the emitted C
// (rounding, fixed-point multipliers, ReLU before requantisation) with respect
// to the float model evaluated in exact real arithmetic.

#include "qnet.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace modelforge {

struct AnalysisOptions {
    bool affine = true;             // false: interval domain (ablation)
    bool perChannelScales = true;   // lowering: per-channel input/activation scales
                                    // folded into the next layer's weights
    bool modularAccumulation = true;  // lowering: wrap-around accumulators sized
                                      // by the proved final sum
    bool provenance = true;         // split the error by source (needed for blame)
    // Lowering: choose activation scales from these float inputs (the maximum
    // activation seen, times `headroom`) instead of from proved domain-wide
    // ranges. Values beyond the range saturate; the analysis models the clamp.
    std::shared_ptr<const std::vector<std::vector<double>>> calibration;
    double headroom = 1.0;
};

// Error sources: 0 = input quantisation, 1 + l = weights of layer l,
// 1 + L + l = output rounding of layer l.
std::size_t errorSourceCount(std::size_t layers);
std::string errorSourceName(std::size_t source, std::size_t layers);

struct RegionAnalysis {
    int refClass = -1;
    bool refCertified = false;      // float model is constant on the box
    bool quantCertified = false;    // integer program makes the same decision on the box
    double refMarginLowerBound = 0;   // min over rivals of lb(R_c - R_j)
    double quantMarginLowerBound = 0; // min over rivals of lb(Q_c - Q_j)
    int worstRival = -1;
    std::vector<double> blame;           // per error source, at the worst rival
    std::vector<double> splitScore;      // per input dimension
    std::vector<double> descentDirection;  // sign pattern that lowers the quant margin
    std::vector<double> logitErrorBound;   // per logit: proved max |scale*q - ref|
    std::vector<double> logitLower;        // proved bounds of the integer program's
    std::vector<double> logitUpper;        // (dequantised) logits on the box
};

// Analyse a fixed integer program on a box. refClass < 0: the float model's
// class at the box centre.
RegionAnalysis analyzeRegion(const DenseNetwork& network, const QProgram& program, const Box& box,
                             const AnalysisOptions& options = {}, int refClass = -1);

// Lowering with range inference: scales, requantisation constants and
// accumulator widths are chosen from bounds proved by the same analysis over
// the whole input domain, so saturation and overflow are impossible on it.
std::optional<QProgram> lowerToInteger(const DenseNetwork& network, const Box& domain,
                                       const PrecisionConfig& config, std::string& error,
                                       const AnalysisOptions& options = {});

// Accumulator width a range-agnostic compiler must use: every input at the
// extreme of its type (what you get without range analysis).
int worstCaseAccumulatorBits(const QLayer& layer, std::int64_t inputMagnitude);

}  // namespace modelforge

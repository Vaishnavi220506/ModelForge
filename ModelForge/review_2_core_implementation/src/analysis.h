#pragma once

#include "ir.h"
#include "verifier.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace modelforge {

// Default input domain D = [-radius, radius]^n. Random baselines, coordinate
// probes and boundary solving all stay inside the same box.
constexpr float kDefaultInputRadius = 10.0f;

enum class NeuronStability { StableInactive, StableActive, Unstable };

// One ReLU-like activation (Relu, fused Gemm+ReLU, or the test-only dead zone)
// with interval bounds on every unit's pre-activation over the input domain.
struct ActivationSite {
    std::string nodeName;
    std::string preactivation;  // value name in the analysis (unfused) graph
    std::size_t layer = 0;      // ordinal among activation sites, 0-based
    std::size_t width = 0;
    std::vector<float> lower;
    std::vector<float> upper;
    std::vector<NeuronStability> stability;

    std::size_t count(NeuronStability kind) const;
};

// Splits FusedGemmRelu into Gemm + Relu so pre-activations become named values.
// Inputs and outputs are unchanged, so probes transfer between both graphs.
IRGraph unfuseForAnalysis(const IRGraph& graph);

// Interval bound propagation (IBP) over the box domain. Sound for the IR's
// semantics in exact arithmetic; float rounding is not modelled.
std::vector<ActivationSite> analyzeActivationSites(const IRGraph& graph,
                                                   float radius = kDefaultInputRadius);

// Activation-pattern coverage (APC). Every unit's pre-activation is placed in
// one of kCoverageBins bins: deep inactive, near-inactive, three positive
// bands that shrink geometrically toward zero, and deep active. A bin counts
// as feasible only if the IBP interval intersects it, so stable neurons do not
// inflate the denominator with states no input can reach.
constexpr std::size_t kCoverageBins = 6;
const char* coverageBinName(std::size_t bin);
std::size_t coverageBin(float preactivation);

struct LayerCoverage {
    std::string nodeName;
    std::size_t width = 0;
    std::size_t feasible = 0;
    std::size_t covered = 0;
    std::size_t boundaryFeasible = 0;
    std::size_t boundaryCovered = 0;
    std::size_t stableActive = 0;
    std::size_t stableInactive = 0;
    std::size_t unstable = 0;
};

struct CoverageReport {
    std::size_t feasible = 0;
    std::size_t covered = 0;
    std::size_t boundaryFeasible = 0;  // bins 1..4, the neighbourhood of ReLU's kink
    std::size_t boundaryCovered = 0;
    std::vector<LayerCoverage> layers;

    double ratio() const;
    double boundaryRatio() const;
};

CoverageReport measureActivationCoverage(const IRGraph& graph,
                                         const std::vector<ActivationSite>& sites,
                                         const std::vector<ValidationProbe>& probes);

struct DeepBoundaryOptions {
    std::size_t maxNeurons = 64;
    std::size_t maxIterations = 16;
    float radius = kDefaultInputRadius;
    std::uint32_t seed = 20260917;
};

// Model-conditioned boundary probes for ReLU units at *every* depth.
// A ReLU network is affine inside each activation region, so the exact input
// gradient of a unit's pre-activation (forward-mode Jacobian) gives a Newton
// step that lands on that unit's kink z = 0 in a few iterations. From the
// solved point we step to fixed pre-activation targets on both sides.
std::vector<ValidationProbe> generateDeepBoundaryProbes(const IRGraph& graph,
                                                        const std::vector<ActivationSite>& sites,
                                                        const DeepBoundaryOptions& options = {});

// Greedy set-cover ordering: each next probe is the one that covers the most
// not-yet-covered (unit, bin) states. Probes adding nothing keep their order.
std::vector<ValidationProbe> prioritizeByCoverage(const IRGraph& graph,
                                                  const std::vector<ActivationSite>& sites,
                                                  const std::vector<ValidationProbe>& probes);

// Guardian's full probe suite: the generic/first-layer suite plus deep
// boundary probes, ordered by coverage gain.
std::vector<ValidationProbe> generateGuardianProbes(const IRGraph& graph,
                                                    std::uint32_t seed = 20260917);

}  // namespace modelforge

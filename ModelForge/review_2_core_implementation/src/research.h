#pragma once

#include "analysis.h"
#include "diagnostics.h"
#include "ir.h"
#include "verifier.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace modelforge {

// Project author, shown in the tools, reports, dashboard and paper.
constexpr const char* kAuthor = "Vaishnavi";

// ------------------------------------------------------------ model zoo

struct ZooModelSpec {
    std::string name;
    std::uint32_t seed = 0;
    std::size_t inputs = 4;
    std::size_t classes = 3;
    std::vector<std::size_t> hidden;  // one entry per Gemm+ReLU layer
};

// Deterministic corpus of feed-forward classifiers with varied depth and width.
std::vector<ZooModelSpec> makeZooSpecs(std::size_t count, std::uint32_t seed);

// Writes the model as .mforge source: He-initialised Gemm+ReLU layers, a
// classifier Gemm, Softmax, and one output-unreachable ReLU (negative control).
std::string synthesizeManifest(const ZooModelSpec& spec);

// Parses, validates and lowers a manifest held in memory.
bool compileManifestText(const std::string& text, IRGraph& graph, DiagnosticEngine& diagnostics);

// ------------------------------------------------------------ faults

struct FaultCase {
    std::string name;      // unique within a model, e.g. dead_zone_1e-2@L1
    std::string family;    // e.g. dead_zone_1e-2
    std::string locality;  // boundary_local | numeric_local | global | control
    int layer = -1;        // activation layer, or -1 when not layer specific
    bool negativeControl = false;
    bool heldOut = false;  // family not used when designing the probe targets
    bool provablyUnobservable = false;  // IBP shows the fault cannot change outputs on D
    IRGraph candidate;
};

std::vector<FaultCase> makeFaults(const IRGraph& original,
                                  const std::vector<ActivationSite>& sites,
                                  std::uint32_t seed);

// ------------------------------------------------------------ strategies

struct StrategySuite {
    std::string name;
    std::string description;
    std::vector<ValidationProbe> probes;  // already padded / truncated to the budget
    std::size_t nativeProbes = 0;         // probes produced before random padding
    double generationMilliseconds = 0.0;
};

std::vector<ValidationProbe> uniformRandomProbes(std::size_t inputCount, std::size_t count,
                                                 std::uint32_t seed,
                                                 float radius = kDefaultInputRadius);

// Builds every strategy at exactly `budget` probes. Suites shorter than the
// budget are padded with uniform random probes so all comparisons are equal-budget.
std::vector<StrategySuite> buildStrategies(const IRGraph& original, std::size_t budget,
                                           std::uint32_t seed, std::size_t randomSeeds);

// ------------------------------------------------------------ evaluation

struct BenchmarkConfig {
    std::size_t models = 96;
    std::size_t budget = 128;
    std::vector<std::size_t> budgetCurve = {1, 2, 4, 8, 16, 32, 64, 128};
    std::size_t randomSeeds = 5;
    std::size_t oracleProbes = 4000;
    std::uint32_t seed = 20260917;
    float tolerance = 1.0e-5f;
};

using ProgressCallback = std::function<void(std::size_t done, std::size_t total,
                                            const std::string& label)>;

struct StrategySummary {
    std::string name;
    std::string description;
    std::size_t detected = 0;
    std::size_t trials = 0;
    double rate = 0.0, low = 0.0, high = 0.0;
    double meanFirst = 0.0, censoredFirst = 0.0, auc = 0.0;
    double apc = 0.0, boundaryApc = 0.0, generationMs = 0.0;
    std::size_t falsePositives = 0, controls = 0;
    std::vector<std::pair<std::size_t, double>> curve;
    std::map<std::string, double> families;
    std::map<std::string, double> localities;
    std::map<int, double> layers;
    std::map<std::string, double> subsets;  // held_out, design_aligned, first_layer, deep_layers
};

struct ComparisonSummary {
    std::string other;
    std::size_t onlyOurs = 0, onlyOther = 0;
    double pValue = 1.0;
};

struct BenchmarkSummary {
    std::vector<StrategySummary> strategies;
    std::vector<ComparisonSummary> comparisons;
    std::size_t models = 0, faults = 0, observable = 0, unknown = 0, provable = 0;
    std::size_t ibpViolations = 0, controls = 0;
    std::size_t stableActive = 0, stableInactive = 0, unstable = 0;
    double seconds = 0.0;
    std::string json;      // full results document (also embedded in the dashboard)
    std::string casesCsv;  // one row per (model, fault, strategy, seed)
};

BenchmarkSummary runBenchmark(const BenchmarkConfig& config,
                              const ProgressCallback& progress = {});

// Single-model inspection used by the dashboard and the terminal studio.
std::string inspectModelJson(const std::string& manifestPath, const IRGraph& graph,
                             std::size_t budget, std::uint32_t seed);

// ------------------------------------------------------------ statistics

struct Proportion {
    std::size_t successes = 0;
    std::size_t trials = 0;
    double rate() const;
    double wilsonLow() const;
    double wilsonHigh() const;
};

// Exact two-sided McNemar test from the two discordant counts.
double mcnemarExactP(std::size_t onlyFirst, std::size_t onlySecond);

std::string jsonEscapeText(const std::string& value);

}  // namespace modelforge

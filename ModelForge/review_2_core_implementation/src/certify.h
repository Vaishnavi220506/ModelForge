#pragma once

// ForgeCert, part 3: certified precision tuning.
//
// Specification: the integer program must make the float model's decision on
// every input in each certification region (an L-infinity ball of radius eps
// around a real sample, inside the input domain), must never overflow an
// accumulator and never saturate a requantised value anywhere in the domain.
//
// The tuner is a counterexample-guided refinement loop over bit widths:
//   lower (range inference) -> prove each region (differential abstract
//   interpretation + input splitting) -> on failure, falsify (search for a
//   concrete input where the integer code disagrees) -> blame the error
//   source with the largest share of the failing margin per byte of cost
//   (error provenance) -> raise that source's precision -> repeat; finally
//   lower any width that is not needed.
// The result is the integer program, its C code and a certificate that the
// independent checker (tools/certcheck.cpp) re-validates from the emitted C.

#include "diffai.h"
#include "qnet.h"
#include "shrink.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace modelforge {

struct CertRegion {
    std::string name;
    Box box;
    int refClass = -1;
};

// Per-feature range of the data, widened by `widen` times its spread.
Box domainFromData(const std::vector<std::vector<float>>& rows, double widen = 0.1);
// L-infinity balls of radius eps around (up to `limit`, evenly spaced) rows,
// clipped to the domain.
std::vector<CertRegion> regionsAroundRows(const DenseNetwork& network,
                                          const std::vector<std::vector<float>>& rows,
                                          double eps, const Box& domain, std::size_t limit);

// Pre-order split tree: dim < 0 is a proved leaf, otherwise the box is cut at
// `at` along `dim` and the lower child comes first.
struct SplitNode {
    int dim = -1;
    double at = 0.0;
};

struct ProofOptions {
    AnalysisOptions analysis;
    int maxDepth = 10;
    std::size_t maxAnalyses = 96;  // per region
    bool referenceOnly = false;    // prove only that the float model is constant
};

struct RegionProof {
    bool proved = false;
    std::vector<SplitNode> tree;
    std::size_t analyses = 0;
    std::size_t leaves = 0;
    std::vector<double> blame;  // at the first leaf that could not be proved
    std::vector<double> descent;
    Box failingLeaf;
};

RegionProof proveRegion(const DenseNetwork& network, const QProgram& program,
                        const CertRegion& region, const ProofOptions& options);

struct Counterexample {
    std::string region;
    std::vector<double> input;
    int refClass = -1;
    int quantClass = -1;
    std::size_t evaluations = 0;
};

// Searches the region for an input where the integer program's decision
// differs from the float model's. Seeds: the region centre, the vertex the
// abstract margin points to (`descent`), and the segment between them; then
// an adaptive random local search on the integer margin.
std::optional<Counterexample> falsifyRegion(const DenseNetwork& network, const QProgram& program,
                                            const CertRegion& region, const std::vector<double>& descent,
                                            std::size_t budget, std::uint32_t seed);

enum class CostModel { Bytes, Bits, Latency };
enum class RefineStrategy { Blame, Uniform, RandomSource };

struct TunerOptions {
    CostModel cost = CostModel::Bytes;
    RefineStrategy strategy = RefineStrategy::Blame;
    ProofOptions proof;
    bool falsifyFirst = true;          // cheap search before an expensive proof
    std::size_t falsifyBudget = 48;
    bool minimize = true;
    std::size_t maxIterations = 64;
    std::uint32_t seed = 20261012;
};

struct RegionStatus {
    std::string name;
    bool excluded = false;  // float model not provably constant on the region
    bool proved = false;
    bool counterexample = false;
    RegionProof proof;
};

struct TunerResult {
    bool ok = false;
    std::string error;
    PrecisionConfig config;
    QProgram program;
    std::vector<RegionStatus> regions;
    std::size_t proved = 0;
    std::size_t excluded = 0;
    std::size_t unproved = 0;
    std::size_t iterations = 0;
    std::size_t lowerings = 0;
    std::size_t analyses = 0;
    std::size_t falsifierCalls = 0;
    std::size_t falsifierHits = 0;
    double seconds = 0.0;
    std::vector<std::string> log;
    std::vector<Counterexample> counterexamples;  // seen during refinement
};

std::vector<int> weightLadder(CostModel cost);
std::vector<int> activationLadder(CostModel cost);
double configCost(const QProgram& program, CostModel cost);

// Reference robustness of every region (phase 0), shared by all strategies.
std::vector<bool> referenceRobustRegions(const DenseNetwork& network, const Box& domain,
                                         const std::vector<CertRegion>& regions,
                                         const ProofOptions& options);

TunerResult tunePrecision(const DenseNetwork& network, const Box& domain,
                          const std::vector<CertRegion>& regions, const TunerOptions& options,
                          const std::vector<bool>* robust = nullptr);

// Checks a fixed configuration against the same specification.
TunerResult checkConfiguration(const DenseNetwork& network, const Box& domain,
                               const std::vector<CertRegion>& regions, const PrecisionConfig& config,
                               const ProofOptions& proof, const std::vector<bool>* robust = nullptr,
                               std::size_t falsifyBudget = 0, std::uint32_t seed = 1);

// Writes certificate.json next to the emitted C (see tools/certcheck.cpp).
bool writeCertificate(const std::string& path, const std::string& modelPath,
                      const DenseNetwork& network, const std::vector<CertRegion>& regions,
                      const TunerResult& result, double eps, std::string& error);

}  // namespace modelforge

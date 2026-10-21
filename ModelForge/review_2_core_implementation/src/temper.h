#pragma once

// Temper: falsify-or-certify assurance for every step between a trained
// network and the binary that ships.
//
// ModelForge compiles; Temper decides, for every transformation on the way to
// the device, whether it is safe:
//
//   graph rewrites      (constant folding, fusion, dead-code removal, edits)
//   domain specialisation (neurons proven dead on the deployment domain removed)
//   compression         (float16, int8, int4, pruning)
//   code generation     (the compiled binary under several compiler flags)
//
// Each verdict is REFUTED (a replayable input), CERTIFIED (a proof: bit-exact
// or epsilon-bounded on the input domain), or UNDECIDED (with the certified
// share of the domain). Guardian-APC+ (adaptive.h) falsifies, the certifier
// (certify.h) proves, and each guides the other.

#include "adaptive.h"
#include "certify.h"
#include "ir.h"
#include "shrink.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace modelforge {

// ------------------------------------------------------------ transformations

struct TemperOptions {
    float tolerance = 1.0e-5f;
    std::size_t staticBudget = 96;    // Guardian-APC+ targeted inputs
    std::size_t adaptiveBudget = 32;  // Guardian-APC+ near-miss search
    std::size_t maxBoxes = 512;       // certifier branch-and-bound
    std::size_t maxEvaluations = 1024;
    bool useGuardian = true;
    bool useCertifier = true;
    bool guidedWitnesses = true;
    std::uint32_t seed = 20260917;
};

struct TemperDecision {
    Verdict verdict = Verdict::Undecided;
    CertificateKind kind = CertificateKind::None;
    std::string decidedBy;  // static-proof | guardian | branch-and-bound | <refutation kind>
    double bound = 0.0;
    double certifiedFraction = 0.0;
    std::size_t evaluations = 0;  // concrete model-pair evaluations
    std::size_t boxes = 0;
    std::size_t sources = 0;
    std::optional<std::vector<float>> witness;
    std::vector<float> originalOutput;
    std::vector<float> candidateOutput;
    DivergenceLocation divergence;
    double milliseconds = 0.0;
};

// Static proof (one box) -> Guardian-APC+ -> certificate-guided branch and bound.
TemperDecision decideTransformation(const IRGraph& before, const IRGraph& after, const Box& domain,
                                    const TemperOptions& options = {});

// ------------------------------------------------------------ domain specialisation

struct SpecializationReport {
    std::size_t hiddenUnits = 0;
    std::size_t deadUnits = 0;          // proven inactive on the domain: removed
    std::size_t alwaysActiveUnits = 0;  // proven active on the domain
    std::size_t linearizedLayers = 0;   // ReLU removed: every unit always active
    std::size_t parametersBefore = 0, parametersAfter = 0;
    std::size_t multiplyAddsBefore = 0, multiplyAddsAfter = 0;
    CertifyResult certificate;          // original vs masked model on the domain
    IRGraph masked;                     // dead units' outgoing weights set to 0
    IRGraph specialized;                // dead units removed, shapes shrunk
};

// Removes hidden ReLU units that are provably inactive for every input in
// `domain` (sound bounds, float32 rounding included) and drops ReLUs whose
// units are all provably active. The result is bit-identical to the original
// on the domain; the certifier re-checks it independently.
SpecializationReport specializeToDomain(const IRGraph& graph, const Box& domain);

std::size_t parameterCount(const IRGraph& graph);
std::size_t multiplyAddCount(const IRGraph& graph);

// ------------------------------------------------------------ compression audit

// Falsification side of the severity bracket: inside the L-infinity balls of
// `radius` around `centres`, projected sign-gradient attacks on both models'
// logit margins (and on their gap) look for inputs where the two disagree,
// then a short hill-climb raises the original's confidence at the witness.
struct SeveritySearch {
    double severity = 0.0;       // original's top-two margin at the worst disagreement found
    std::size_t refutedBalls = 0;
    std::size_t evaluations = 0;
    std::vector<float> witness;
    int originalClass = -1, compressedClass = -1;
};
SeveritySearch falsifySeverity(const IRGraph& original, const IRGraph& candidate,
                               const std::vector<std::vector<float>>& centres, float radius,
                               std::size_t steps = 16);

// Original's top-two probability margin (|2p - 1| for one sigmoid output).
double confidenceMargin(const std::vector<float>& scores);

struct CompressionAudit {
    std::string compression;
    double weightSize = 1.0;  // share of float32 weight storage kept
    double accuracyBefore = 0.0, accuracyAfter = 0.0;
    std::size_t testFlips = 0;
    SeverityCertificate certificate;  // proven upper bound
    SeveritySearch search;            // found lower bound
    std::string certifiedRisk;        // risk level of the proven bound
    std::string foundRisk;            // risk level of the worst disagreement found
    Verdict verdict = Verdict::Undecided;  // CERTIFIED low risk / REFUTED (moderate+) / open
    double milliseconds = 0.0;
};

// Severity bracket  found <= true worst <= certified  on the realistic region
// (balls of `radius` around up to `maxBalls` test rows; 0 = all rows).
CompressionAudit auditCompression(const IRGraph& original, const Compression& compression,
                                  const Dataset& test, float radius = 0.5f,
                                  std::size_t maxBalls = 0, std::size_t maxBoxesPerBall = 16);

// ------------------------------------------------------------ binary audit

struct BinaryFlagResult {
    std::string flags;
    bool built = false;
    std::size_t rows = 0;
    std::size_t predictionMismatches = 0;
    double maximumAbsoluteError = 0.0;
    double buildSeconds = 0.0;
    std::string error;
};

struct BinaryAuditReport {
    std::string compiler;
    bool available = false;
    std::vector<BinaryFlagResult> results;
};

// Compiles the generated model.cpp with the system C++ compiler under each
// flag set, replays `inputs` through the binary (--csv) and compares every
// score with the IR interpreter: translation validation down to machine code.
BinaryAuditReport auditGeneratedBinary(const IRGraph& graph, const std::string& generatedDirectory,
                                       const std::vector<std::vector<float>>& inputs,
                                       const std::vector<std::string>& flagSets);

std::vector<std::string> defaultAuditFlags();

}  // namespace modelforge

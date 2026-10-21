#pragma once

// Temper certifier: the "certify" half of falsify-or-certify.
//
// Guardian (adaptive.h) can only ever say "no counterexample found". The
// certifier proves the opposite direction: that a transformation (an
// optimisation pass, a model edit, a compression, a domain specialisation)
// cannot change the model's outputs by more than epsilon anywhere in an input
// box, and, for structural rewrites, that the outputs are bit-for-bit
// identical under the interpreter's float32 semantics.
//
//   1. Forward symbolic bounds (DeepPoly-style linear relaxations intersected
//      with interval bounds), padded outward for float32 rounding, so a proof
//      holds for the executed float program, not only in real arithmetic.
//   2. Lockstep differential propagation: both graphs are canonicalised
//      (unfused, constant-folded, dead code removed) and aligned instruction
//      by instruction; the difference Delta = candidate - original is bounded
//      per value with operator-specific rules, and an exactness bit records
//      values that are provably identical in float32.
//   3. Branch and bound over the input box, with a concrete check at every
//      box centre, so certification and falsification run in one loop.

#include "ir.h"
#include "verifier.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace modelforge {

struct Box {
    std::vector<float> lower;
    std::vector<float> upper;

    std::size_t size() const { return lower.size(); }
    std::vector<float> centre() const;
    double logVolume() const;  // sum of log widths (0-width dims ignored)
};

Box uniformBox(std::size_t inputs, float radius);
Box ballAround(const std::vector<float>& point, float radius);
// Per-feature [min - margin, max + margin] over the rows.
Box dataEnvelope(const std::vector<std::vector<float>>& rows, float margin);

// ------------------------------------------------------------ single-graph bounds

// Bounds of one value: concrete per-element interval plus, when available,
// symbolic linear bounds  L(x) <= v <= U(x)  over the box inputs.
struct ValueBounds {
    std::vector<double> lower, upper;
    // Row-major [elements x (inputs + 1)]; the last column is the constant.
    std::vector<double> symLower, symUpper;
    bool symbolic = false;
};

struct GraphBounds {
    std::unordered_map<std::string, ValueBounds> values;
    std::size_t inputs = 0;
    bool complete = false;  // every live instruction was bounded

    const ValueBounds* find(const std::string& name) const;
};

// Sound for float32 execution of the IR interpreter: affine outputs are padded
// by the standard dot-product rounding bound gamma_n * sum |w||x|.
GraphBounds computeBounds(const IRGraph& graph, const Box& box, bool symbolic = true);

// ------------------------------------------------------------ canonical form

// Unfuses Gemm+ReLU, folds constant Add/ReLU/Sigmoid/Softmax with the
// interpreter's own float formulas, and removes instructions that cannot reach
// the output. Bit-exact: the canonical graph computes the same float32 values.
IRGraph canonicalize(const IRGraph& graph);

// 64-bit FNV-1a fingerprint of the canonical graph (ops, attributes, weights).
std::uint64_t graphFingerprint(const IRGraph& graph);
std::string fingerprintHex(std::uint64_t value);

// ------------------------------------------------------------ differential bounds

// Where a divergence can start: a candidate pre-activation unit whose operator
// differs from the original's on a reachable sub-interval (e.g. a dead zone
// (0, p) that the unit can enter). Falsification aims straight at it.
struct DivergenceSource {
    std::string value;  // pre-activation value in the canonical candidate graph
    std::size_t unit = 0;
    double from = 0.0;
    double to = 0.0;
};

struct DifferenceBounds {
    bool aligned = false;          // lockstep alignment succeeded
    bool complete = false;         // both graphs fully bounded
    std::vector<double> lower;     // bounds on candidate - original, per output
    std::vector<double> upper;
    std::vector<bool> exact;       // output element provably identical in float32
    std::size_t exactValues = 0;   // aligned values proven identical
    std::size_t alignedValues = 0;
    std::vector<DivergenceSource> sources;

    bool allExact() const;
    double maxAbs() const;
};

DifferenceBounds differentialBounds(const IRGraph& original, const IRGraph& candidate,
                                    const Box& box);

// Same, but stops at the value feeding a final Softmax (the logits) and also
// returns the original's symbolic logit bounds, for prediction-agreement proofs.
struct LogitDifference {
    DifferenceBounds delta;
    ValueBounds originalLogits;
    ValueBounds candidateLogits;
    bool sigmoidOutput = false;  // single-output binary classifier
};
LogitDifference logitDifference(const IRGraph& original, const IRGraph& candidate,
                                const Box& box);

// ------------------------------------------------------------ verdicts

enum class Verdict { Certified, Refuted, Undecided };
const char* verdictName(Verdict verdict);

enum class CertificateKind {
    None,
    Exact,    // bit-identical float32 outputs on the whole box
    Bounded,  // |candidate - original| <= epsilon on the whole box
};
const char* certificateKindName(CertificateKind kind);

struct CertifyOptions {
    float tolerance = 1.0e-5f;
    std::size_t maxBoxes = 512;  // branch-and-bound budget (analysed boxes)
    std::size_t maxDepth = 24;
    bool centreChecks = true;    // concrete check at every box centre
    // Certificate-guided falsification in every undecided box: the gradient
    // vertex that maximises the linearised difference, and Newton-solved inputs
    // that place each divergence source inside its divergence interval.
    bool guidedWitnesses = true;
    std::size_t maxEvaluations = 4096;
};

struct CertifyResult {
    Verdict verdict = Verdict::Undecided;
    CertificateKind kind = CertificateKind::None;
    double bound = 0.0;            // proven max |Delta| (certified boxes)
    double certifiedFraction = 0;  // share of the box volume certified
    std::size_t boxes = 0;         // boxes analysed
    std::size_t evaluations = 0;   // concrete centre checks
    std::size_t certifiedBoxes = 0;
    std::size_t openBoxes = 0;
    bool aligned = false;
    std::optional<std::vector<float>> witness;  // refutation input
    std::string refutedBy;                     // centre | vertex | divergence-source
    std::size_t sources = 0;                   // divergence sources found statically
    double milliseconds = 0.0;
};

// Epsilon-equivalence on a box: certify every leaf, or refute at a centre.
CertifyResult certifyEquivalence(const IRGraph& original, const IRGraph& candidate,
                                 const Box& box, const CertifyOptions& options = {});

// Prediction agreement on a box: both models predict the same single class
// everywhere in the box (returns the class, or -1).
int certifyAgreement(const IRGraph& original, const IRGraph& candidate, const Box& box,
                     std::size_t maxBoxes = 1);

// Certified agreement radius around each point: the largest r (binary search
// with `iterations` steps on [0, maxRadius]) such that both models provably
// predict the same class on the whole L-infinity ball of radius r. With
// candidate == original this is the original's certified robustness radius.
std::vector<float> certifiedAgreementRadii(const IRGraph& original, const IRGraph& candidate,
                                           const std::vector<std::vector<float>>& points,
                                           float maxRadius, std::size_t iterations = 10,
                                           std::size_t maxBoxes = 4);

// Certified severity bound for a compressed (or otherwise perturbed) model on
// the union of L-infinity balls of `radius` around `centres`: at every input
// of that region where the two models predict different classes, the
// original's top-two probability margin is at most `bound`. Per ball,
// branch and bound splits the box with the largest local bound.
struct SeverityCertificate {
    double bound = 1.0;
    double maxDeviation = 0.0;       // largest certified |p' - p| on a leaf
    std::size_t balls = 0;
    std::size_t agreeingBalls = 0;   // prediction agreement proven on the whole ball
    std::size_t boxes = 0;
    std::vector<double> perBall;     // local bounds (0 = agreement proven)
    double milliseconds = 0.0;
};
SeverityCertificate certifySeverity(const IRGraph& original, const IRGraph& candidate,
                                    const std::vector<std::vector<float>>& centres, float radius,
                                    std::size_t maxBoxesPerBall = 16);

// Name of the logits in a canonical graph: the input of a final Softmax or
// Sigmoid, otherwise the output itself.
std::string logitsValue(const IRGraph& canonical, bool* sigmoidOutput = nullptr);

// Predicted class: argmax, or (single output) 1 when the score exceeds 0.5.
std::size_t predictedClass(const std::vector<float>& scores);

// Concrete check: output difference above tolerance or a different argmax.
bool exposes(const IRGraph& original, const IRGraph& candidate, const std::vector<float>& input,
             float tolerance, float* difference = nullptr);

}  // namespace modelforge

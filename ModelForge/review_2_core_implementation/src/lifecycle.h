#pragma once

// Temper across the model lifecycle: what to do after a verdict.
//
//   1. Certified mixed-precision repair. A REFUTED or UNDECIDED compression is
//      not a dead end: a greedy per-layer search assigns int4/int8/fp16/fp32 to
//      each weight matrix and keeps only assignments whose severity bound is
//      CERTIFIED low risk, returning the smallest provably safe model
//      (cf. HAWQ-style mixed precision, but with a proof instead of a proxy).
//   2. Certified culprit localisation. For a REFUTED edit with unchanged
//      structure, delta debugging (Zeller and Hildebrandt, ddmin) over the
//      changed parameters finds a 1-minimal set that alone breaks the model;
//      reverting it is then re-decided, so the fix itself is CERTIFIED.
//   3. Incremental re-certification. A certificate stores both fingerprints,
//      the domain and its branch-and-bound partition; the next model version
//      starts from that partition (cf. IVAN, PLDI 2023), re-verifying every
//      leaf, so reuse saves work but can never make a proof unsound.
//   4. Bug-class benchmark: faults modelled on the root-cause classes of
//      deep-learning compiler bug studies (layout, fusion, quantisation
//      rounding and saturation, activation lowering, loop bounds, broadcast),
//      injected into real trained models.

#include "ir.h"
#include "research.h"
#include "shrink.h"
#include "temper.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace modelforge {

// ------------------------------------------------------------ mixed precision

struct LayerPrecision {
    std::string weight;  // weight tensor name
    std::size_t parameters = 0;
    int bits = 32;       // 4, 8, 16 or 32
};

// Weight matrices quantised per tensor (int4/int8, symmetric) or rounded to
// float16; biases stay float32. bits 32 leaves a tensor unchanged.
IRGraph compressLayers(const IRGraph& graph, const std::map<std::string, int>& bits);

struct MixedPrecisionPlan {
    std::vector<LayerPrecision> layers;
    double weightSize = 1.0;          // share of float32 weight storage kept
    CompressionAudit audit;           // severity bracket of the final plan
    std::size_t trials = 0;           // certified-severity checks run
    std::string bestUniform = "fp32"; // smallest uniform compression CERTIFIED
    double bestUniformSize = 1.0;
    double milliseconds = 0.0;
};

// Largest layers first, each lowered to the fewest bits whose whole-model
// severity bound stays certified low risk.
MixedPrecisionPlan searchMixedPrecision(const IRGraph& original, const Dataset& test,
                                        float radius = 0.5f, std::size_t maxBalls = 0,
                                        std::size_t maxBoxesPerBall = 16);

// ------------------------------------------------------------ culprit localisation

struct ParameterEdit {
    std::string tensor;
    std::size_t index = 0;
    float before = 0.0f, after = 0.0f;
};

struct CulpritReport {
    bool applicable = false;           // same operators and tensor shapes
    std::size_t edits = 0;             // changed parameters
    std::vector<ParameterEdit> culprits;  // 1-minimal failure-inducing subset
    Verdict culpritsAlone = Verdict::Undecided;  // original + culprits only
    Verdict afterRevert = Verdict::Undecided;    // candidate with culprits reverted
    CertificateKind revertKind = CertificateKind::None;
    std::size_t decisions = 0;         // Temper decisions spent
    double milliseconds = 0.0;
};

CulpritReport localizeCulprits(const IRGraph& original, const IRGraph& candidate, const Box& domain,
                               const TemperOptions& options = {});

// ------------------------------------------------------------ incremental certificates

struct ProofCertificate {
    std::string original, candidate;  // canonical fingerprints (hex)
    Box domain;
    Verdict verdict = Verdict::Undecided;
    CertificateKind kind = CertificateKind::None;
    double tolerance = 1.0e-5;
    double bound = 0.0;
    std::vector<Box> partition;   // covers the domain, whatever the verdict
    std::vector<float> witness;   // refuting input, replayed first on the next version
};

bool saveCertificate(const std::string& path, const ProofCertificate& certificate);
bool loadCertificate(const std::string& path, ProofCertificate& certificate, std::string& error);

struct IncrementalResult {
    CertifyResult result;
    ProofCertificate certificate;
    bool reused = false;        // previous partition used as the seed
    bool identical = false;     // same fingerprints and domain as the stored proof
    bool carriedWitness = false;  // refuted by replaying the stored witness
    std::size_t seedLeaves = 0;
};

// Certifies `candidate` against `original` on `domain`, seeding branch and
// bound with `previous` when its domain matches (any earlier model version).
IncrementalResult certifyIncrementally(const IRGraph& original, const IRGraph& candidate,
                                       const Box& domain, const CertifyOptions& options,
                                       const ProofCertificate* previous = nullptr);

// ------------------------------------------------------------ studies

struct LifecycleStudySummary {
    // Mixed precision (real models).
    struct PrecisionRow {
        std::string model;
        MixedPrecisionPlan plan;
    };
    std::vector<PrecisionRow> precision;
    // Culprit localisation (real models, one harmful edit hidden among benign ones).
    struct CulpritRow {
        std::string model;
        std::size_t edits = 0, decisions = 0;
        bool exact = false;   // the culprit set is exactly the harmful edit
        bool fixed = false;   // reverting it is CERTIFIED
        double milliseconds = 0.0;
    };
    std::vector<CulpritRow> culprits;
    // Incremental re-certification (version chains on real models).
    struct IncrementalRow {
        std::string model, step;
        Verdict verdict = Verdict::Undecided;
        std::size_t scratchBoxes = 0, reusedBoxes = 0;
        double scratchShare = 0.0, reusedShare = 0.0;  // certified share of the domain
        double scratchMs = 0.0, reusedMs = 0.0;
        bool agree = false;   // same verdict from scratch and with reuse
    };
    std::vector<IncrementalRow> incremental;
    // Bug classes (real models).
    struct BugRow {
        std::string model, bugClass, source;
        bool expected = false;     // observable on the domain by the random oracle
        bool testSet = false;      // standard practice: test-set outputs differ
        bool random = false;       // 1,000 uniform inputs in the domain
        Verdict temper = Verdict::Undecided;
        CertificateKind kind = CertificateKind::None;
        std::size_t evaluations = 0;
    };
    std::vector<BugRow> bugs;
    std::size_t violations = 0;    // CERTIFIED yet exposed by the oracle: must be 0
    double seconds = 0.0;
    std::string json;
};

LifecycleStudySummary runLifecycleStudy(const std::string& realDirectory, std::size_t maxBalls,
                                        const ProgressCallback& progress = {});

}  // namespace modelforge

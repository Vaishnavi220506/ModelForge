#pragma once

// Temper evaluation studies (the numbers in the paper).
//
//   1. Decisiveness: on the synthetic zoo, every injected fault, negative
//      control and real optimisation pass is decided three ways (Guardian
//      alone, certifier alone, Temper). Soundness gate: no CERTIFIED case may
//      ever be exposed by any falsifier or the random oracle.
//   2. Compression: severity brackets for real trained models.
//   3. Binary: generated C++ compiled under several flag sets and replayed.

#include "research.h"
#include "temper.h"

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace modelforge {

struct VerdictCounts {
    std::size_t refuted = 0, certifiedExact = 0, certifiedBounded = 0, undecided = 0;
    std::size_t total() const { return refuted + certifiedExact + certifiedBounded + undecided; }
    std::size_t decided() const { return refuted + certifiedExact + certifiedBounded; }
};

struct StudyStrategy {
    std::string name;
    VerdictCounts observable;    // faults some method or the oracle exposes
    VerdictCounts hidden;        // faults nothing exposes (oracle + every strategy)
    VerdictCounts benign;        // negative controls and real optimisation passes
    std::size_t evaluations = 0; // total concrete evaluations
    double milliseconds = 0.0;
    std::size_t violations = 0;  // CERTIFIED but exposed: must be 0
    std::map<std::string, VerdictCounts> families;
};

struct TemperStudySummary {
    std::size_t models = 0, cases = 0, faults = 0, controls = 0, passes = 0;
    std::size_t observable = 0, hidden = 0;
    std::size_t guardianMissed = 0;         // observable faults Guardian-APC+ misses
    std::size_t guardianMissedRefuted = 0;  // ... that Temper refutes
    std::size_t oracleMissedRefuted = 0;    // observable only through a Temper witness
    std::vector<StudyStrategy> strategies;  // guardian_apc_plus, certifier, temper
    double seconds = 0.0;
    std::string json;
    std::string casesCsv;
};

TemperStudySummary runTemperStudy(const BenchmarkConfig& config,
                                  const ProgressCallback& progress = {});

struct CompressionStudySummary {
    std::size_t cases = 0;
    VerdictCounts verdicts;
    std::size_t bracketViolations = 0;  // found severity above the proven bound: must be 0
    std::size_t certifiedRecommendations = 0, agreeingRecommendations = 0, models = 0;
    struct Row {
        std::string model, compression;
        CompressionAudit audit;
    };
    std::vector<Row> rows;
    std::map<std::string, std::string> certifiedChoice, empiricalChoice;
    double seconds = 0.0;
    std::string json;
};

CompressionStudySummary runCompressionStudy(const std::string& realDirectory, std::size_t maxBalls,
                                            const ProgressCallback& progress = {});

struct BinaryStudySummary {
    bool available = false;
    std::string compiler;
    struct Row {
        std::string model;
        BinaryAuditReport report;
    };
    std::vector<Row> rows;
    std::string json;
};

BinaryStudySummary runBinaryStudy(const std::string& realDirectory, const std::string& workDirectory,
                                  const ProgressCallback& progress = {});

}  // namespace modelforge

#pragma once

// The Temper deployment dossier: one run that takes a trained model (and,
// optionally, its data) to a shippable C++ program, deciding every
// transformation on the way with falsify-or-certify, and reporting what is
// proven, what is refuted, and what is still open.

#include "certify.h"
#include "codegen.h"
#include "diagnostics.h"
#include "ir.h"
#include "shrink.h"
#include "temper.h"

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace modelforge {

struct DossierOptions {
    std::string modelPath;
    std::string dataPath;     // labelled test rows: compression audit + realistic region
    std::string domainPath;   // rows that define the deployment domain (default: dataPath)
    std::string outputDirectory = "temper_out";
    float domainMargin = 0.5f;  // envelope = per-feature [min, max] of the rows +- margin
    float realism = 0.5f;       // L-inf radius of the realistic balls around test rows
    std::size_t maxBalls = 0;   // 0 = every test row
    bool compression = true;
    bool specialize = true;
    bool binaryAudit = true;
    bool strict = false;        // keep only CERTIFIED passes
};

struct PassVerdict {
    std::string pass;
    bool changed = false;
    bool kept = false;
    TemperDecision decision;
    std::vector<std::string> events;
};

struct Dossier {
    std::string model;
    std::string fingerprint;
    std::size_t inputs = 0, outputs = 0;
    Box domain;
    std::string domainSource;
    std::vector<PassVerdict> passes;
    bool specialized = false;
    SpecializationReport specialization;
    std::vector<CompressionAudit> compressions;
    std::string recommendedCertified;  // smallest compression with a certified low risk
    std::string recommendedEmpirical;  // smallest whose worst found flip is low risk
    BinaryAuditReport binary;
    std::size_t parameters = 0, multiplyAdds = 0;
    bool ship = false;
    std::vector<std::string> blockers;
    std::vector<std::string> notes;
    std::string generatedDirectory;
    std::string json;
    double seconds = 0.0;
};

using DossierProgress = std::function<void(const std::string& stage)>;

// Returns false (with diagnostics) when the model cannot be compiled at all.
bool buildDossier(const DossierOptions& options, Dossier& dossier, DiagnosticEngine& diagnostics,
                  const DossierProgress& progress = {});

}  // namespace modelforge

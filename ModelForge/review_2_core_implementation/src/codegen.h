#pragma once

#include "diagnostics.h"
#include "ir.h"

#include <string>
#include <vector>

namespace modelforge {

// Certificate-carrying code: the input box on which Temper's certificates
// hold. The generated program can then tell, per input, whether it is inside
// the region where its guarantees were proven.
struct DeploymentEnvelope {
    std::vector<float> lower;
    std::vector<float> upper;
    std::string fingerprint;  // canonical-graph fingerprint of the certified model
    std::string summary;      // one line, e.g. "3 passes certified bit-exact"
};

bool generateCpp(const IRGraph& graph,
                 const std::string& outputDirectory,
                 DiagnosticEngine& diagnostics,
                 const DeploymentEnvelope* envelope = nullptr);

}  // namespace modelforge

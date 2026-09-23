#pragma once

#include "diagnostics.h"
#include "ir.h"

#include <string>

namespace modelforge {

bool generateCpp(const IRGraph& graph,
                 const std::string& outputDirectory,
                 DiagnosticEngine& diagnostics);

}  // namespace modelforge

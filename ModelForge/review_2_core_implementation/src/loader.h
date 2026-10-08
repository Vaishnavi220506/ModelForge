#pragma once

#include "diagnostics.h"
#include "model.h"

#include <istream>
#include <string>

namespace modelforge {

bool loadModel(const std::string& path, ModelGraph& model, DiagnosticEngine& diagnostics);
bool loadManifest(const std::string& path, ModelGraph& model, DiagnosticEngine& diagnostics);
// Parses manifest source held in memory (used by the synthetic model zoo).
bool loadManifestText(const std::string& text, ModelGraph& model, DiagnosticEngine& diagnostics);
bool parseManifest(std::istream& source, ModelGraph& model, DiagnosticEngine& diagnostics);
bool loadOnnx(const std::string& path, ModelGraph& model, DiagnosticEngine& diagnostics);

}  // namespace modelforge

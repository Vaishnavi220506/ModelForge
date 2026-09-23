#pragma once

#include "diagnostics.h"
#include "model.h"

#include <string>

namespace modelforge {

bool loadModel(const std::string& path, ModelGraph& model, DiagnosticEngine& diagnostics);
bool loadManifest(const std::string& path, ModelGraph& model, DiagnosticEngine& diagnostics);
bool loadOnnx(const std::string& path, ModelGraph& model, DiagnosticEngine& diagnostics);

}  // namespace modelforge

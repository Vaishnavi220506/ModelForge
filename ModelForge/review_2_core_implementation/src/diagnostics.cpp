#include "diagnostics.h"

#include <iostream>

namespace modelforge {

void DiagnosticEngine::error(const std::string& phase,
                             const std::string& message,
                             int line) {
    diagnostics_.push_back({phase, message, line, true});
}

void DiagnosticEngine::warning(const std::string& phase,
                               const std::string& message,
                               int line) {
    diagnostics_.push_back({phase, message, line, false});
}

bool DiagnosticEngine::hasErrors() const {
    for (const Diagnostic& diagnostic : diagnostics_) {
        if (diagnostic.isError) {
            return true;
        }
    }
    return false;
}

const std::vector<Diagnostic>& DiagnosticEngine::all() const {
    return diagnostics_;
}

void DiagnosticEngine::print(std::ostream& output) const {
    for (const Diagnostic& diagnostic : diagnostics_) {
        output << (diagnostic.isError ? "ERROR" : "WARNING") << " ["
               << diagnostic.phase << "]";
        if (diagnostic.line > 0) {
            output << " line " << diagnostic.line;
        }
        output << ": " << diagnostic.message << '\n';
    }
}

}  // namespace modelforge

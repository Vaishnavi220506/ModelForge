#pragma once

#include <ostream>
#include <string>
#include <vector>

namespace modelforge {

struct Diagnostic {
    std::string phase;
    std::string message;
    int line = 0;
    bool isError = true;
};

class DiagnosticEngine {
public:
    void error(const std::string& phase, const std::string& message, int line = 0);
    void warning(const std::string& phase, const std::string& message, int line = 0);

    bool hasErrors() const;
    const std::vector<Diagnostic>& all() const;
    void print(std::ostream& output) const;

private:
    std::vector<Diagnostic> diagnostics_;
};

}  // namespace modelforge

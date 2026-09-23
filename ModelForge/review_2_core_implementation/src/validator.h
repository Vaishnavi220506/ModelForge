#pragma once

#include "diagnostics.h"
#include "model.h"

#include <string>
#include <iosfwd>
#include <unordered_map>

namespace modelforge {

class SymbolTable {
public:
    bool insert(const TensorInfo& symbol);
    bool contains(const std::string& name) const;
    TensorInfo* lookup(const std::string& name);
    const TensorInfo* lookup(const std::string& name) const;
    std::size_t size() const;

private:
    friend void printSymbolTable(const SymbolTable& symbols, std::ostream& output);
    std::unordered_map<std::string, TensorInfo> symbols_;
};

bool validate(ModelGraph& model, DiagnosticEngine& diagnostics, SymbolTable& symbols);

// Prints the semantic symbol table used during validation.
void printSymbolTable(const SymbolTable& symbols, std::ostream& output);

}  // namespace modelforge

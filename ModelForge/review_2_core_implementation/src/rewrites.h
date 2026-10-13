#pragma once

// Graph rewrites of the kind ML compilers apply before code generation, each
// in a correct form and in faulty forms that reproduce documented compiler
// bugs. The rewrite-validation study (tools/rewritebench.cpp) asks how quickly
// a validation method notices that a rewritten graph no longer computes the
// original model.
//
// Correct rewrites are negative controls: they change floating-point rounding
// at most, and a sound validator must accept them.

#include "ir.h"

#include <string>
#include <vector>

namespace modelforge {

struct RewriteCase {
    std::string pass;    // e.g. "conv_bn_fold"
    std::string fault;   // "none" for the correct rewrite
    std::string source;  // documented bug the fault reproduces, or "synthetic"
    bool control = false;  // correct rewrite: must be accepted
    IRGraph candidate;
    std::vector<std::string> changedNodes;  // nodes of the original graph the rewrite altered
};

// Every applicable rewrite of `graph`, correct and faulty.
std::vector<RewriteCase> makeRewriteCases(const IRGraph& graph);

// The IR graph as .mforge text (constants included), e.g. for an independent
// runtime. Edge padding is written as `padmode=edge`.
std::string irToManifest(const IRGraph& graph);

}  // namespace modelforge

#include "adaptive.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace modelforge {
namespace {

using TensorMap = std::unordered_map<std::string, std::vector<float>>;

bool sameConstant(const IRGraph& before, const IRGraph& after, const std::string& name) {
    const auto left = before.constants.find(name);
    const auto right = after.constants.find(name);
    if (left == before.constants.end() || right == after.constants.end()) {
        return (left == before.constants.end()) == (right == after.constants.end());
    }
    return left->second.data == right->second.data && left->second.shape == right->second.shape;
}

bool sameInstruction(const IRInstruction& a, const IRInstruction& b) {
    return a.operation == b.operation && a.inputs == b.inputs && a.output == b.output &&
           a.axis == b.axis && a.transA == b.transA && a.transB == b.transB &&
           a.testFaultParameter == b.testFaultParameter;
}

std::size_t argmax(const std::vector<float>& values) {
    return static_cast<std::size_t>(std::max_element(values.begin(), values.end()) -
                                    values.begin());
}

float maxDifference(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return std::numeric_limits<float>::infinity();
    float result = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float d = std::fabs(a[i] - b[i]);
        if (!std::isfinite(d)) return std::numeric_limits<float>::infinity();
        result = std::max(result, d);
    }
    return result;
}

}  // namespace

RewriteImpact analyzeRewriteImpact(const IRGraph& before, const IRGraph& after,
                                   const std::vector<ActivationSite>& sites) {
    RewriteImpact impact;
    std::unordered_map<std::string, const IRInstruction*> afterByOutput;
    for (const auto& instruction : after.instructions) {
        if (instruction.operation != IROp::Return) afterByOutput[instruction.output] = &instruction;
    }

    // Forward reachability from every changed instruction in `before`.
    std::unordered_set<std::string> touched;
    for (const auto& instruction : before.instructions) {
        if (instruction.operation == IROp::Input || instruction.operation == IROp::Return) continue;
        const auto match = afterByOutput.find(instruction.output);
        bool changed = match == afterByOutput.end() || !sameInstruction(instruction, *match->second);
        for (const auto& input : instruction.inputs) {
            if (before.constants.count(input) != 0 && !sameConstant(before, after, input)) {
                changed = true;
            }
        }
        bool downstream = false;
        for (const auto& input : instruction.inputs) downstream = downstream || touched.count(input);
        if (changed && instruction.operation != IROp::Return) {
            impact.changedNodes.push_back(instruction.nodeName);
        }
        if (changed || downstream) touched.insert(instruction.output);
    }
    impact.anyChange = !impact.changedNodes.empty();

    for (const auto& site : sites) {
        for (const auto& instruction : before.instructions) {
            if (instruction.nodeName == site.nodeName && touched.count(instruction.output)) {
                impact.impactedSites.push_back(site);
                break;
            }
        }
    }
    return impact;
}

std::vector<ValidationProbe> generateRewriteAwareProbes(const IRGraph& before,
                                                        const IRGraph& after,
                                                        std::uint32_t seed) {
    const auto sites = analyzeActivationSites(before);
    const auto impact = analyzeRewriteImpact(before, after, sites);
    if (impact.impactedSites.empty()) return generateGuardianProbes(before, seed);

    DeepBoundaryOptions options;
    options.seed = seed;
    auto pool = generateDeepBoundaryProbes(before, impact.impactedSites, options);
    for (auto& probe : pool) probe.name = "impact_" + probe.name;
    auto legacy = generateValidationProbes(before, seed);
    auto global = generateDeepBoundaryProbes(before, sites, options);
    pool.insert(pool.end(), std::make_move_iterator(legacy.begin()),
                std::make_move_iterator(legacy.end()));
    pool.insert(pool.end(), std::make_move_iterator(global.begin()),
                std::make_move_iterator(global.end()));
    // Greedy coverage of the impacted sites first; the rest keeps pool order
    // (impacted boundary probes, generic probes, then global boundary probes).
    return prioritizeByCoverage(before, impact.impactedSites, pool);
}

float divergenceScore(const IRGraph& before, const IRGraph& candidate,
                      const std::vector<float>& input, float tolerance, bool& detected,
                      std::vector<float>* originalOutput, std::vector<float>* candidateOutput) {
    TensorMap left, right;
    DiagnosticEngine leftDiagnostics, rightDiagnostics;
    const auto a = executeIRTrace(before, input, leftDiagnostics, &left);
    const auto b = executeIRTrace(candidate, input, rightDiagnostics, &right);
    if (originalOutput && a) *originalOutput = *a;
    if (candidateOutput && b) *candidateOutput = *b;
    if (!a || !b || a->empty() || b->empty()) {
        detected = true;
        return std::numeric_limits<float>::infinity();
    }
    const float output = maxDifference(*a, *b);
    detected = !std::isfinite(output) || output > tolerance || argmax(*a) != argmax(*b);
    float internal = 0.0f;
    for (const auto& name : liveValues(before)) {
        const auto x = left.find(name);
        const auto y = right.find(name);
        if (x == left.end() || y == right.end() || x->second.size() != y->second.size()) continue;
        internal = std::max(internal, maxDifference(x->second, y->second));
    }
    return output + 1.0e-3f * (std::isfinite(internal) ? internal : 0.0f);
}

GuardianPlusResult runGuardianPlus(const IRGraph& before, const IRGraph& candidate,
                                   const GuardianPlusOptions& options) {
    GuardianPlusResult result;
    auto probes = options.rewriteAware
                      ? generateRewriteAwareProbes(before, candidate, options.seed)
                      : generateGuardianProbes(before, options.seed);
    std::size_t adaptiveBudget = options.adaptiveBudget;
    if (options.staticBudget > 0) {
        if (probes.size() > options.staticBudget) probes.resize(options.staticBudget);
        adaptiveBudget += options.staticBudget - probes.size();  // unused static budget
    }
    if (options.adaptiveBudget == 0 && options.staticBudget > probes.size()) {
        // Pure static mode at a fixed budget: pad with uniform random probes,
        // exactly like the equal-budget baselines.
        const auto& shape = before.values.at(before.inputName).shape;
        std::mt19937 padGenerator(options.seed ^ 0x5bd1e995u);
        std::uniform_real_distribution<float> uniform(-options.radius, options.radius);
        while (probes.size() < options.staticBudget) {
            std::vector<float> values(elementCount(shape));
            for (float& v : values) v = uniform(padGenerator);
            probes.push_back({"pad_uniform_" + std::to_string(probes.size()), std::move(values)});
        }
        adaptiveBudget = 0;
    }
    result.staticProbes = probes.size();

    std::vector<std::pair<float, std::size_t>> scored;
    const auto record = [&](const ValidationProbe& probe, bool detected,
                            const std::vector<float>& a, const std::vector<float>& b) {
        ++result.evaluations;
        if (a.size() == b.size()) {
            result.maximumAbsoluteError = std::max(result.maximumAbsoluteError, maxDifference(a, b));
        }
        if (detected && result.firstDetection == 0) {
            result.firstDetection = result.evaluations;
            result.witness = probe;
            result.originalOutput = a;
            result.candidateOutput = b;
        }
        return detected;
    };

    for (std::size_t index = 0; index < probes.size(); ++index) {
        bool detected = false;
        std::vector<float> a, b;
        const float score =
            divergenceScore(before, candidate, probes[index].values, options.tolerance, detected, &a, &b);
        if (record(probes[index], detected, a, b)) return result;
        scored.emplace_back(score, index);
    }
    if (adaptiveBudget == 0 || probes.empty()) return result;

    // Near-miss search: hill-climb the divergence score from the most divergent
    // static probes. Sub-tolerance differences point toward real failures.
    std::stable_sort(scored.begin(), scored.end(),
                     [](const auto& x, const auto& y) { return x.first > y.first; });
    const std::size_t seeds = std::min<std::size_t>(4, scored.size());
    std::mt19937 generator(options.seed ^ 0x2545f491u);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    const float initialStep = 0.1f * options.radius;
    std::size_t used = 0;
    for (std::size_t s = 0; s < seeds && used < adaptiveBudget; ++s) {
        const std::size_t share = (adaptiveBudget - used + (seeds - s) - 1) / (seeds - s);
        std::vector<float> current = probes[scored[s].second].values;
        float best = scored[s].first;
        float step = initialStep;
        for (std::size_t iteration = 0; iteration < share && used < adaptiveBudget; ++iteration) {
            std::vector<float> proposal = current;
            if (iteration % 3 == 2) {
                // Amplify the whole input: grows faults that scale with |x|.
                const float factor = 1.0f + step / initialStep * 0.5f;
                for (float& v : proposal) v *= factor;
            } else {
                for (float& v : proposal) v += step * normal(generator);
            }
            for (float& v : proposal) v = std::clamp(v, -options.radius, options.radius);
            bool detected = false;
            std::vector<float> a, b;
            const float score =
                divergenceScore(before, candidate, proposal, options.tolerance, detected, &a, &b);
            ++used;
            ValidationProbe probe{"near_miss_" + std::to_string(used), proposal};
            if (record(probe, detected, a, b)) {
                result.foundByNearMiss = true;
                result.bestNearMissScore = score;
                return result;
            }
            if (score > best) {
                best = score;
                current = std::move(proposal);
                step = std::min(step * 1.5f, options.radius);
            } else {
                step *= 0.7f;
            }
            if (step < 1.0e-4f) {
                current = probes[scored[s].second].values;
                step = initialStep * 0.5f;
            }
            result.bestNearMissScore = std::max(result.bestNearMissScore, best);
        }
    }
    return result;
}

DivergenceLocation localizeDivergence(const IRGraph& before, const IRGraph& candidate,
                                      const std::vector<float>& input, float tolerance) {
    DivergenceLocation location;
    TensorMap left, right;
    DiagnosticEngine leftDiagnostics, rightDiagnostics;
    const auto a = executeIRTrace(before, input, leftDiagnostics, &left);
    const auto b = executeIRTrace(candidate, input, rightDiagnostics, &right);
    if (!a || !b) return location;
    for (std::size_t index = 0; index < before.instructions.size(); ++index) {
        const auto& instruction = before.instructions[index];
        if (instruction.operation == IROp::Input || instruction.operation == IROp::Return) continue;
        const auto x = left.find(instruction.output);
        const auto y = right.find(instruction.output);
        if (x == left.end() || y == right.end()) continue;
        const float difference = maxDifference(x->second, y->second);
        if (difference > tolerance) {
            location.found = true;
            location.instruction = index;
            location.node = instruction.nodeName;
            location.value = instruction.output;
            location.maximumDifference = difference;
            return location;
        }
    }
    return location;
}

}  // namespace modelforge

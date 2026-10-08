#include "analysis.h"

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

struct Interval {
    std::vector<float> lower;
    std::vector<float> upper;
};

struct GemmShape {
    std::size_t rows = 0;
    std::size_t inner = 0;
    std::size_t columns = 0;
};

bool isActivation(IROp operation) {
    return operation == IROp::Relu || operation == IROp::TestReluDeadZone ||
           operation == IROp::TestReluClamp || operation == IROp::TestLeakyRelu;
}

bool gemmShape(const IRGraph& graph, const IRInstruction& instruction, GemmShape& shape) {
    if (instruction.inputs.size() < 2) return false;
    const auto left = graph.values.find(instruction.inputs[0]);
    const auto right = graph.values.find(instruction.inputs[1]);
    if (left == graph.values.end() || right == graph.values.end() ||
        left->second.shape.size() != 2 || right->second.shape.size() != 2) {
        return false;
    }
    const bool transA = instruction.operation != IROp::MatMul && instruction.transA;
    const bool transB = instruction.operation != IROp::MatMul && instruction.transB;
    shape.rows = static_cast<std::size_t>(transA ? left->second.shape[1] : left->second.shape[0]);
    shape.inner = static_cast<std::size_t>(transA ? left->second.shape[0] : left->second.shape[1]);
    shape.columns = static_cast<std::size_t>(transB ? right->second.shape[0] : right->second.shape[1]);
    return true;
}

std::size_t leftIndex(const IRInstruction& instruction, const GemmShape& shape,
                      std::size_t row, std::size_t k) {
    return instruction.operation != IROp::MatMul && instruction.transA ? k * shape.rows + row
                                                                       : row * shape.inner + k;
}

std::size_t rightIndex(const IRInstruction& instruction, const GemmShape& shape,
                       std::size_t k, std::size_t column) {
    return instruction.operation != IROp::MatMul && instruction.transB
               ? column * shape.inner + k
               : k * shape.columns + column;
}

std::size_t biasIndex(std::size_t biasSize, const GemmShape& shape,
                      std::size_t row, std::size_t column) {
    return biasSize == shape.columns ? column : row * shape.columns + column;
}

const std::vector<float>* lookup(const IRGraph& graph, const TensorMap& values,
                                 const std::string& name) {
    const auto value = values.find(name);
    if (value != values.end()) return &value->second;
    const auto constant = graph.constants.find(name);
    return constant == graph.constants.end() ? nullptr : &constant->second.data;
}

// ---------------------------------------------------------------- intervals

void multiplyInterval(float aLow, float aHigh, float bLow, float bHigh,
                      float& low, float& high) {
    const float p1 = aLow * bLow;
    const float p2 = aLow * bHigh;
    const float p3 = aHigh * bLow;
    const float p4 = aHigh * bHigh;
    low = std::min(std::min(p1, p2), std::min(p3, p4));
    high = std::max(std::max(p1, p2), std::max(p3, p4));
}

const Interval* intervalOf(const IRGraph& graph,
                           std::unordered_map<std::string, Interval>& intervals,
                           const std::string& name) {
    const auto found = intervals.find(name);
    if (found != intervals.end()) return &found->second;
    const auto constant = graph.constants.find(name);
    if (constant == graph.constants.end()) return nullptr;
    Interval exact{constant->second.data, constant->second.data};
    return &intervals.emplace(name, std::move(exact)).first->second;
}

std::unordered_map<std::string, Interval> propagateIntervals(const IRGraph& graph, float radius) {
    std::unordered_map<std::string, Interval> intervals;
    for (const IRInstruction& instruction : graph.instructions) {
        if (instruction.operation == IROp::Input) {
            const auto info = graph.values.find(instruction.output);
            if (info == graph.values.end()) break;
            const std::size_t count = elementCount(info->second.shape);
            intervals[instruction.output] = {std::vector<float>(count, -radius),
                                             std::vector<float>(count, radius)};
            continue;
        }
        if (instruction.operation == IROp::Return || instruction.inputs.empty()) continue;
        const Interval* first = intervalOf(graph, intervals, instruction.inputs[0]);
        if (first == nullptr) break;
        Interval result;
        switch (instruction.operation) {
            case IROp::Gemm:
            case IROp::FusedGemmRelu:
            case IROp::MatMul: {
                GemmShape shape;
                if (!gemmShape(graph, instruction, shape)) return intervals;
                const Interval a = *first;
                const Interval* bPointer = intervalOf(graph, intervals, instruction.inputs[1]);
                if (bPointer == nullptr) return intervals;
                const Interval b = *bPointer;
                const Interval* bias = instruction.operation != IROp::MatMul &&
                                               instruction.inputs.size() == 3
                                           ? intervalOf(graph, intervals, instruction.inputs[2])
                                           : nullptr;
                result.lower.assign(shape.rows * shape.columns, 0.0f);
                result.upper.assign(shape.rows * shape.columns, 0.0f);
                for (std::size_t row = 0; row < shape.rows; ++row) {
                    for (std::size_t column = 0; column < shape.columns; ++column) {
                        float low = 0.0f;
                        float high = 0.0f;
                        if (bias != nullptr) {
                            const std::size_t index =
                                biasIndex(bias->lower.size(), shape, row, column);
                            low = bias->lower[index];
                            high = bias->upper[index];
                        }
                        for (std::size_t k = 0; k < shape.inner; ++k) {
                            const std::size_t l = leftIndex(instruction, shape, row, k);
                            const std::size_t r = rightIndex(instruction, shape, k, column);
                            float productLow = 0.0f;
                            float productHigh = 0.0f;
                            multiplyInterval(a.lower[l], a.upper[l], b.lower[r], b.upper[r],
                                             productLow, productHigh);
                            low += productLow;
                            high += productHigh;
                        }
                        if (instruction.operation == IROp::FusedGemmRelu) {
                            low = std::max(0.0f, low);
                            high = std::max(0.0f, high);
                        }
                        result.lower[row * shape.columns + column] = low;
                        result.upper[row * shape.columns + column] = high;
                    }
                }
                break;
            }
            case IROp::Add: {
                const Interval* second = intervalOf(graph, intervals, instruction.inputs.at(1));
                if (second == nullptr || second->lower.size() != first->lower.size()) {
                    return intervals;
                }
                result = *first;
                for (std::size_t index = 0; index < result.lower.size(); ++index) {
                    result.lower[index] += second->lower[index];
                    result.upper[index] += second->upper[index];
                }
                break;
            }
            case IROp::Relu:
            case IROp::TestReluDeadZone:
            case IROp::TestReluClamp:
            case IROp::TestLeakyRelu: {
                const float p = instruction.testFaultParameter;
                result = *first;
                for (std::size_t index = 0; index < result.lower.size(); ++index) {
                    for (float* bound : {&result.lower[index], &result.upper[index]}) {
                        if (instruction.operation == IROp::TestLeakyRelu) {
                            *bound = *bound < 0.0f ? p * *bound : *bound;
                        } else {
                            *bound = std::max(0.0f, *bound);
                            if (instruction.operation == IROp::TestReluClamp) {
                                *bound = std::min(*bound, p);
                            }
                        }
                    }
                }
                if (instruction.operation == IROp::TestReluDeadZone) {
                    // Not monotone: values in (0, p) drop to 0.
                    for (float& low : result.lower) low = 0.0f;
                }
                break;
            }
            case IROp::Sigmoid:
                result = *first;
                for (std::size_t index = 0; index < result.lower.size(); ++index) {
                    result.lower[index] = 1.0f / (1.0f + std::exp(-result.lower[index]));
                    result.upper[index] = 1.0f / (1.0f + std::exp(-result.upper[index]));
                }
                break;
            case IROp::Softmax: {
                // s_i = 1 / (1 + sum_{j != i} exp(z_j - z_i)); monotone in each term.
                const std::size_t count = first->lower.size();
                result.lower.assign(count, 0.0f);
                result.upper.assign(count, 1.0f);
                for (std::size_t i = 0; i < count; ++i) {
                    double low = 0.0;
                    double high = 0.0;
                    for (std::size_t j = 0; j < count; ++j) {
                        if (j == i) continue;
                        low += std::exp(std::min(80.0, static_cast<double>(first->upper[j]) -
                                                           first->lower[i]));
                        high += std::exp(std::min(80.0, static_cast<double>(first->lower[j]) -
                                                            first->upper[i]));
                    }
                    result.lower[i] = static_cast<float>(1.0 / (1.0 + low));
                    result.upper[i] = static_cast<float>(1.0 / (1.0 + high));
                }
                break;
            }
            case IROp::Input:
            case IROp::Return:
                break;
        }
        intervals[instruction.output] = std::move(result);
    }
    return intervals;
}

// ----------------------------------------------------- forward-mode Jacobian

// Returns d(value[unit]) / d(input) for the named value, using the activation
// pattern of the traced point. Exact inside a linear region of a ReLU network.
bool inputGradient(const IRGraph& graph,
                   const TensorMap& trace,
                   const std::string& target,
                   std::size_t unit,
                   std::size_t inputCount,
                   std::vector<float>& gradient) {
    std::unordered_map<std::string, std::vector<float>> jacobians;  // rows x inputCount
    const auto jacobianOf = [&](const std::string& name) -> const std::vector<float>* {
        const auto found = jacobians.find(name);
        return found == jacobians.end() ? nullptr : &found->second;  // nullptr == zero
    };
    for (const IRInstruction& instruction : graph.instructions) {
        if (instruction.operation == IROp::Return) break;
        std::vector<float> jacobian;
        if (instruction.operation == IROp::Input) {
            jacobian.assign(inputCount * inputCount, 0.0f);
            for (std::size_t index = 0; index < inputCount; ++index) {
                jacobian[index * inputCount + index] = 1.0f;
            }
        } else {
            const auto* outputValue = lookup(graph, trace, instruction.output);
            const auto* firstValue = lookup(graph, trace, instruction.inputs.at(0));
            if (outputValue == nullptr || firstValue == nullptr) return false;
            const auto* firstJacobian = jacobianOf(instruction.inputs[0]);
            const std::size_t outputCount = outputValue->size();
            switch (instruction.operation) {
                case IROp::Gemm:
                case IROp::FusedGemmRelu:
                case IROp::MatMul: {
                    GemmShape shape;
                    if (!gemmShape(graph, instruction, shape)) return false;
                    const auto* secondValue = lookup(graph, trace, instruction.inputs[1]);
                    if (secondValue == nullptr) return false;
                    const auto* secondJacobian = jacobianOf(instruction.inputs[1]);
                    const auto* biasJacobian =
                        instruction.operation != IROp::MatMul && instruction.inputs.size() == 3
                            ? jacobianOf(instruction.inputs[2])
                            : nullptr;
                    if (firstJacobian == nullptr && secondJacobian == nullptr &&
                        biasJacobian == nullptr) {
                        break;  // constant result
                    }
                    jacobian.assign(outputCount * inputCount, 0.0f);
                    for (std::size_t row = 0; row < shape.rows; ++row) {
                        for (std::size_t column = 0; column < shape.columns; ++column) {
                            const std::size_t out = row * shape.columns + column;
                            float* destination = &jacobian[out * inputCount];
                            if (instruction.operation == IROp::FusedGemmRelu &&
                                (*outputValue)[out] <= 0.0f) {
                                continue;
                            }
                            for (std::size_t k = 0; k < shape.inner; ++k) {
                                const std::size_t l = leftIndex(instruction, shape, row, k);
                                const std::size_t r = rightIndex(instruction, shape, k, column);
                                if (firstJacobian != nullptr) {
                                    const float weight = (*secondValue)[r];
                                    const float* source = &(*firstJacobian)[l * inputCount];
                                    for (std::size_t x = 0; x < inputCount; ++x) {
                                        destination[x] += weight * source[x];
                                    }
                                }
                                if (secondJacobian != nullptr) {
                                    const float weight = (*firstValue)[l];
                                    const float* source = &(*secondJacobian)[r * inputCount];
                                    for (std::size_t x = 0; x < inputCount; ++x) {
                                        destination[x] += weight * source[x];
                                    }
                                }
                            }
                            if (biasJacobian != nullptr) {
                                const std::size_t b = biasIndex(
                                    biasJacobian->size() / inputCount, shape, row, column);
                                for (std::size_t x = 0; x < inputCount; ++x) {
                                    destination[x] += (*biasJacobian)[b * inputCount + x];
                                }
                            }
                        }
                    }
                    break;
                }
                case IROp::Add: {
                    const auto* secondJacobian = jacobianOf(instruction.inputs.at(1));
                    if (firstJacobian == nullptr && secondJacobian == nullptr) break;
                    jacobian.assign(outputCount * inputCount, 0.0f);
                    for (const auto* source : {firstJacobian, secondJacobian}) {
                        if (source == nullptr) continue;
                        for (std::size_t index = 0; index < jacobian.size(); ++index) {
                            jacobian[index] += (*source)[index];
                        }
                    }
                    break;
                }
                case IROp::Relu:
                case IROp::TestReluDeadZone:
                case IROp::TestReluClamp:
                case IROp::TestLeakyRelu:
                    if (firstJacobian == nullptr) break;
                    jacobian = *firstJacobian;
                    for (std::size_t unitIndex = 0; unitIndex < outputCount; ++unitIndex) {
                        float slope = (*outputValue)[unitIndex] > 0.0f ? 1.0f : 0.0f;
                        if (instruction.operation == IROp::TestReluClamp &&
                            (*outputValue)[unitIndex] >= instruction.testFaultParameter) {
                            slope = 0.0f;
                        }
                        if (instruction.operation == IROp::TestLeakyRelu &&
                            (*firstValue)[unitIndex] < 0.0f) {
                            slope = instruction.testFaultParameter;
                        }
                        if (slope == 1.0f) continue;
                        for (std::size_t x = 0; x < inputCount; ++x) {
                            jacobian[unitIndex * inputCount + x] *= slope;
                        }
                    }
                    break;
                case IROp::Sigmoid:
                    if (firstJacobian == nullptr) break;
                    jacobian = *firstJacobian;
                    for (std::size_t unitIndex = 0; unitIndex < outputCount; ++unitIndex) {
                        const float s = (*outputValue)[unitIndex];
                        for (std::size_t x = 0; x < inputCount; ++x) {
                            jacobian[unitIndex * inputCount + x] *= s * (1.0f - s);
                        }
                    }
                    break;
                case IROp::Softmax: {
                    if (firstJacobian == nullptr) break;
                    std::vector<float> weighted(inputCount, 0.0f);
                    for (std::size_t j = 0; j < outputCount; ++j) {
                        for (std::size_t x = 0; x < inputCount; ++x) {
                            weighted[x] += (*outputValue)[j] * (*firstJacobian)[j * inputCount + x];
                        }
                    }
                    jacobian.assign(outputCount * inputCount, 0.0f);
                    for (std::size_t i = 0; i < outputCount; ++i) {
                        for (std::size_t x = 0; x < inputCount; ++x) {
                            jacobian[i * inputCount + x] =
                                (*outputValue)[i] *
                                ((*firstJacobian)[i * inputCount + x] - weighted[x]);
                        }
                    }
                    break;
                }
                case IROp::Input:
                case IROp::Return:
                    break;
            }
        }
        if (instruction.output == target) {
            gradient.assign(inputCount, 0.0f);
            if (!jacobian.empty()) {
                std::copy_n(&jacobian[unit * inputCount], inputCount, gradient.begin());
            }
            return true;
        }
        if (!jacobian.empty()) jacobians[instruction.output] = std::move(jacobian);
    }
    return false;
}

bool preactivationAt(const IRGraph& graph, const std::string& name, std::size_t unit,
                     const std::vector<float>& input, float& value, TensorMap& trace) {
    DiagnosticEngine diagnostics;
    trace.clear();
    if (!executeIRTrace(graph, input, diagnostics, &trace)) return false;
    const auto* values = lookup(graph, trace, name);
    if (values == nullptr || unit >= values->size()) return false;
    value = (*values)[unit];
    return std::isfinite(value);
}

float squaredNorm(const std::vector<float>& values) {
    float total = 0.0f;
    for (const float value : values) total += value * value;
    return total;
}

void stepToward(std::vector<float>& point, const std::vector<float>& gradient,
                float amount, float radius) {
    for (std::size_t index = 0; index < point.size(); ++index) {
        point[index] = std::clamp(point[index] + amount * gradient[index], -radius, radius);
    }
}

}  // namespace

std::size_t ActivationSite::count(NeuronStability kind) const {
    return static_cast<std::size_t>(std::count(stability.begin(), stability.end(), kind));
}

IRGraph unfuseForAnalysis(const IRGraph& graph) {
    IRGraph result = graph;
    result.instructions.clear();
    for (const IRInstruction& instruction : graph.instructions) {
        if (instruction.operation != IROp::FusedGemmRelu) {
            result.instructions.push_back(instruction);
            continue;
        }
        IRInstruction dense = instruction;
        dense.operation = IROp::Gemm;
        dense.output = instruction.output + "__mf_preact";
        TensorInfo info = graph.values.count(instruction.output) != 0
                              ? graph.values.at(instruction.output)
                              : TensorInfo{};
        info.name = dense.output;
        info.isConstant = false;
        info.data.clear();
        result.values[dense.output] = info;
        IRInstruction relu = instruction;
        relu.operation = IROp::Relu;
        relu.inputs = {dense.output};
        result.instructions.push_back(std::move(dense));
        result.instructions.push_back(std::move(relu));
    }
    return result;
}

std::vector<ActivationSite> analyzeActivationSites(const IRGraph& original, float radius) {
    const IRGraph graph = unfuseForAnalysis(original);
    auto intervals = propagateIntervals(graph, radius);
    // Only activations that can reach the returned output are analysed; a
    // fault or probe aimed at a dead node is wasted budget.
    std::unordered_set<std::string> live = {graph.outputName};
    for (auto it = graph.instructions.rbegin(); it != graph.instructions.rend(); ++it) {
        if (it->operation == IROp::Return || live.count(it->output) != 0) {
            live.insert(it->inputs.begin(), it->inputs.end());
        }
    }
    std::vector<ActivationSite> sites;
    for (const IRInstruction& instruction : graph.instructions) {
        if (!isActivation(instruction.operation) || instruction.inputs.empty() ||
            live.count(instruction.output) == 0) {
            continue;
        }
        const Interval* bounds = intervalOf(graph, intervals, instruction.inputs[0]);
        if (bounds == nullptr) continue;
        ActivationSite site;
        site.nodeName = instruction.nodeName;
        site.preactivation = instruction.inputs[0];
        site.layer = sites.size();
        site.width = bounds->lower.size();
        site.lower = bounds->lower;
        site.upper = bounds->upper;
        for (std::size_t unit = 0; unit < site.width; ++unit) {
            site.stability.push_back(site.lower[unit] > 0.0f    ? NeuronStability::StableActive
                                     : site.upper[unit] <= 0.0f ? NeuronStability::StableInactive
                                                                : NeuronStability::Unstable);
        }
        sites.push_back(std::move(site));
    }
    return sites;
}

namespace {

// Lower / upper edges of each bin; bins are (low, high] except bin 0.
constexpr float kBinEdges[kCoverageBins + 1] = {
    -std::numeric_limits<float>::infinity(), -0.1f, 0.0f, 1.0e-3f, 1.0e-2f, 0.1f,
    std::numeric_limits<float>::infinity()};

bool binFeasible(std::size_t bin, float lower, float upper) {
    return upper >= kBinEdges[bin] && lower <= kBinEdges[bin + 1] &&
           !(bin >= 2 && upper <= kBinEdges[bin]);
}

std::vector<std::vector<std::size_t>> coveredStates(const IRGraph& graph,
                                                    const std::vector<ActivationSite>& sites,
                                                    const std::vector<ValidationProbe>& probes) {
    std::vector<std::vector<std::size_t>> result(probes.size());
    for (std::size_t probe = 0; probe < probes.size(); ++probe) {
        TensorMap trace;
        DiagnosticEngine diagnostics;
        if (!executeIRTrace(graph, probes[probe].values, diagnostics, &trace)) continue;
        std::size_t offset = 0;
        for (const ActivationSite& site : sites) {
            const auto* values = lookup(graph, trace, site.preactivation);
            if (values != nullptr && values->size() == site.width) {
                for (std::size_t unit = 0; unit < site.width; ++unit) {
                    if (!std::isfinite((*values)[unit])) continue;
                    result[probe].push_back(offset + unit * kCoverageBins +
                                            coverageBin((*values)[unit]));
                }
            }
            offset += site.width * kCoverageBins;
        }
    }
    return result;
}

}  // namespace

const char* coverageBinName(std::size_t bin) {
    static const char* names[kCoverageBins] = {"inactive", "near_inactive", "kink_1e-3",
                                               "kink_1e-2", "kink_1e-1", "active"};
    return bin < kCoverageBins ? names[bin] : "unknown";
}

std::size_t coverageBin(float z) {
    for (std::size_t bin = 0; bin + 1 < kCoverageBins; ++bin) {
        if (bin == 0 ? z < kBinEdges[1] : z <= kBinEdges[bin + 1]) return bin;
    }
    return kCoverageBins - 1;
}

double CoverageReport::ratio() const {
    return feasible == 0 ? 0.0 : static_cast<double>(covered) / static_cast<double>(feasible);
}

double CoverageReport::boundaryRatio() const {
    return boundaryFeasible == 0
               ? 0.0
               : static_cast<double>(boundaryCovered) / static_cast<double>(boundaryFeasible);
}

CoverageReport measureActivationCoverage(const IRGraph& original,
                                         const std::vector<ActivationSite>& sites,
                                         const std::vector<ValidationProbe>& probes) {
    const IRGraph graph = unfuseForAnalysis(original);
    std::unordered_set<std::size_t> covered;
    for (const auto& states : coveredStates(graph, sites, probes)) {
        covered.insert(states.begin(), states.end());
    }
    CoverageReport report;
    std::size_t offset = 0;
    for (const ActivationSite& site : sites) {
        LayerCoverage layer;
        layer.nodeName = site.nodeName;
        layer.width = site.width;
        layer.stableActive = site.count(NeuronStability::StableActive);
        layer.stableInactive = site.count(NeuronStability::StableInactive);
        layer.unstable = site.count(NeuronStability::Unstable);
        for (std::size_t unit = 0; unit < site.width; ++unit) {
            for (std::size_t bin = 0; bin < kCoverageBins; ++bin) {
                const bool hit = covered.count(offset + unit * kCoverageBins + bin) != 0;
                // A probe can only observe reachable states, but float rounding
                // at an interval edge must never make coverage exceed 100%.
                if (!binFeasible(bin, site.lower[unit], site.upper[unit]) && !hit) continue;
                ++layer.feasible;
                layer.covered += hit ? 1 : 0;
                if (bin >= 1 && bin <= 4) {
                    ++layer.boundaryFeasible;
                    layer.boundaryCovered += hit ? 1 : 0;
                }
            }
        }
        report.feasible += layer.feasible;
        report.covered += layer.covered;
        report.boundaryFeasible += layer.boundaryFeasible;
        report.boundaryCovered += layer.boundaryCovered;
        report.layers.push_back(std::move(layer));
        offset += site.width * kCoverageBins;
    }
    return report;
}

std::vector<ValidationProbe> generateDeepBoundaryProbes(const IRGraph& original,
                                                        const std::vector<ActivationSite>& sites,
                                                        const DeepBoundaryOptions& options) {
    const IRGraph graph = unfuseForAnalysis(original);
    const auto inputInfo = graph.values.find(graph.inputName);
    if (inputInfo == graph.values.end()) return {};
    const std::size_t inputCount = elementCount(inputInfo->second.shape);
    if (inputCount == 0) return {};

    // Round-robin over layers so deep layers are not starved by wide early ones.
    std::vector<std::pair<std::size_t, std::size_t>> targets;
    std::vector<std::vector<std::size_t>> unstable(sites.size());
    for (std::size_t s = 0; s < sites.size(); ++s) {
        for (std::size_t unit = 0; unit < sites[s].width; ++unit) {
            if (sites[s].stability[unit] == NeuronStability::Unstable) {
                unstable[s].push_back(unit);
            }
        }
    }
    for (std::size_t round = 0; targets.size() < options.maxNeurons; ++round) {
        bool any = false;
        for (std::size_t s = 0; s < sites.size() && targets.size() < options.maxNeurons; ++s) {
            if (round < unstable[s].size()) {
                targets.emplace_back(s, unstable[s][round]);
                any = true;
            }
        }
        if (!any) break;
    }

    std::mt19937 generator(options.seed);
    std::uniform_real_distribution<float> startDistribution(-0.5f * options.radius,
                                                            0.5f * options.radius);
    std::vector<std::vector<float>> starts = {std::vector<float>(inputCount, 0.0f)};
    for (int extra = 0; extra < 3; ++extra) {
        std::vector<float> start(inputCount);
        for (float& value : start) value = startDistribution(generator);
        starts.push_back(std::move(start));
    }

    struct Target {
        const char* tag;
        float preactivation;
    };
    // Both sides of the kink and three geometrically shrinking positive offsets,
    // so faults confined to bands of unknown width around zero are reachable.
    const Target levels[] = {{"neg", -0.05f}, {"p5e-4", 5.0e-4f}, {"p5e-3", 5.0e-3f},
                             {"p5e-2", 5.0e-2f}};

    std::vector<ValidationProbe> probes;
    TensorMap trace;
    for (const auto& [siteIndex, unit] : targets) {
        const ActivationSite& site = sites[siteIndex];
        for (const auto& start : starts) {
            std::vector<float> point = start;
            std::vector<float> gradient;
            float z = 0.0f;
            bool converged = false;
            for (std::size_t iteration = 0; iteration < options.maxIterations; ++iteration) {
                if (!preactivationAt(graph, site.preactivation, unit, point, z, trace) ||
                    !inputGradient(graph, trace, site.preactivation, unit, inputCount, gradient)) {
                    break;
                }
                const float norm = squaredNorm(gradient);
                if (norm < 1.0e-12f) break;  // unit is locally dead; try another start
                if (std::fabs(z) <= 1.0e-5f) {
                    converged = true;
                    break;
                }
                stepToward(point, gradient, -z / norm, options.radius);
            }
            if (!converged) continue;
            for (const Target& level : levels) {
                std::vector<float> probe = point;
                float current = z;
                std::vector<float> localGradient = gradient;
                // One Newton step plus one correction; regions may change en route.
                for (int correction = 0; correction < 2; ++correction) {
                    const float norm = squaredNorm(localGradient);
                    if (norm < 1.0e-12f) break;
                    stepToward(probe, localGradient, (level.preactivation - current) / norm,
                               options.radius);
                    if (!preactivationAt(graph, site.preactivation, unit, probe, current, trace) ||
                        !inputGradient(graph, trace, site.preactivation, unit, inputCount,
                                       localGradient)) {
                        break;
                    }
                }
                probes.push_back({"deep_boundary_L" + std::to_string(site.layer) + "_n" +
                                      std::to_string(unit) + "_" + level.tag,
                                  std::move(probe)});
            }
            break;
        }
    }
    return probes;
}

std::vector<ValidationProbe> prioritizeByCoverage(const IRGraph& original,
                                                  const std::vector<ActivationSite>& sites,
                                                  const std::vector<ValidationProbe>& probes) {
    const IRGraph graph = unfuseForAnalysis(original);
    const auto states = coveredStates(graph, sites, probes);
    std::unordered_set<std::size_t> covered;
    std::vector<bool> used(probes.size(), false);
    std::vector<ValidationProbe> ordered;
    ordered.reserve(probes.size());
    while (ordered.size() < probes.size()) {
        std::size_t best = probes.size();
        std::size_t bestGain = 0;
        for (std::size_t index = 0; index < probes.size(); ++index) {
            if (used[index]) continue;
            std::size_t gain = 0;
            for (const std::size_t state : states[index]) gain += covered.count(state) == 0;
            if (gain > bestGain) {
                bestGain = gain;
                best = index;
            }
        }
        if (best == probes.size()) break;
        used[best] = true;
        covered.insert(states[best].begin(), states[best].end());
        ordered.push_back(probes[best]);
    }
    for (std::size_t index = 0; index < probes.size(); ++index) {
        if (!used[index]) ordered.push_back(probes[index]);
    }
    return ordered;
}

std::vector<ValidationProbe> generateGuardianProbes(const IRGraph& graph, std::uint32_t seed) {
    auto probes = generateValidationProbes(graph, seed);
    const auto sites = analyzeActivationSites(graph);
    if (sites.empty()) return probes;
    DeepBoundaryOptions options;
    options.seed = seed;
    auto deep = generateDeepBoundaryProbes(graph, sites, options);
    probes.insert(probes.end(), std::make_move_iterator(deep.begin()),
                  std::make_move_iterator(deep.end()));
    return prioritizeByCoverage(graph, sites, probes);
}

}  // namespace modelforge

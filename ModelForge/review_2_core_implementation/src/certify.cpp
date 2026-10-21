#include "certify.h"

#include "analysis.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unordered_set>
#include <utility>

namespace modelforge {
namespace {

// Unit roundoff of float32 and the classic dot-product rounding factor.
constexpr double kUnit = 5.9604644775390625e-8;  // 2^-24
double gammaFactor(std::size_t terms) {
    const double t = static_cast<double>(terms) * kUnit;
    return t / (1.0 - t);
}
// Relative slack for exp-based operators (std::exp is within a few ulp).
constexpr double kTranscendental = 8.0 * kUnit;

double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }

bool isUnary(IROp op) {
    return op == IROp::Relu || op == IROp::TestReluDeadZone || op == IROp::TestReluClamp ||
           op == IROp::TestLeakyRelu || op == IROp::Sigmoid;
}
bool isDense(IROp op) { return op == IROp::Gemm || op == IROp::MatMul; }

// ------------------------------------------------------------ dense view

struct Dense {
    std::size_t rows = 0, inner = 0, columns = 0;
    bool transA = false, transB = false;
    bool hasBias = false;

    std::size_t left(std::size_t row, std::size_t k) const {
        return transA ? k * rows + row : row * inner + k;
    }
    std::size_t right(std::size_t k, std::size_t column) const {
        return transB ? column * inner + k : k * columns + column;
    }
    std::size_t bias(std::size_t biasSize, std::size_t row, std::size_t column) const {
        return biasSize == columns ? column : row * columns + column;
    }
};

const TensorInfo* infoOf(const IRGraph& graph, const std::string& name) {
    const auto value = graph.values.find(name);
    if (value != graph.values.end()) return &value->second;
    const auto constant = graph.constants.find(name);
    return constant == graph.constants.end() ? nullptr : &constant->second;
}

bool denseView(const IRGraph& graph, const IRInstruction& instruction, Dense& dense) {
    if (instruction.inputs.size() < 2) return false;
    const TensorInfo* left = infoOf(graph, instruction.inputs[0]);
    const TensorInfo* right = infoOf(graph, instruction.inputs[1]);
    if (left == nullptr || right == nullptr || left->shape.size() != 2 ||
        right->shape.size() != 2) {
        return false;
    }
    dense.transA = instruction.operation == IROp::Gemm && instruction.transA;
    dense.transB = instruction.operation == IROp::Gemm && instruction.transB;
    dense.rows = static_cast<std::size_t>(dense.transA ? left->shape[1] : left->shape[0]);
    dense.inner = static_cast<std::size_t>(dense.transA ? left->shape[0] : left->shape[1]);
    dense.columns = static_cast<std::size_t>(dense.transB ? right->shape[0] : right->shape[1]);
    dense.hasBias = instruction.operation == IROp::Gemm && instruction.inputs.size() == 3;
    return true;
}

const std::vector<float>* constantData(const IRGraph& graph, const std::string& name) {
    const auto constant = graph.constants.find(name);
    return constant == graph.constants.end() ? nullptr : &constant->second.data;
}

// ------------------------------------------------------------ symbolic rows

struct Rows {
    std::size_t width = 0;  // inputs + 1
    const Box* box = nullptr;

    double minimum(const double* row) const {
        double total = row[width - 1];
        double magnitude = std::fabs(total);
        for (std::size_t i = 0; i + 1 < width; ++i) {
            const double a = row[i];
            if (a == 0.0) continue;
            const double term = a >= 0.0 ? a * box->lower[i] : a * box->upper[i];
            total += term;
            magnitude += std::fabs(term);
        }
        return total - 1.0e-12 * magnitude;
    }
    double maximum(const double* row) const {
        double total = row[width - 1];
        double magnitude = std::fabs(total);
        for (std::size_t i = 0; i + 1 < width; ++i) {
            const double a = row[i];
            if (a == 0.0) continue;
            const double term = a >= 0.0 ? a * box->upper[i] : a * box->lower[i];
            total += term;
            magnitude += std::fabs(term);
        }
        return total + 1.0e-12 * magnitude;
    }
};

// Adds scale * (symbolic lower or upper row of element e of `source`) to dest.
void addScaled(double* dest, const ValueBounds& source, std::size_t e, double scale,
               bool useLower, std::size_t width) {
    if (scale == 0.0) return;
    if (!source.symbolic) {
        dest[width - 1] += scale * (useLower ? source.lower[e] : source.upper[e]);
        return;
    }
    const double* row = &(useLower ? source.symLower : source.symUpper)[e * width];
    for (std::size_t i = 0; i < width; ++i) dest[i] += scale * row[i];
}

void makeConstantRows(ValueBounds& bounds, std::size_t width) {
    const std::size_t count = bounds.lower.size();
    bounds.symLower.assign(count * width, 0.0);
    bounds.symUpper.assign(count * width, 0.0);
    for (std::size_t e = 0; e < count; ++e) {
        bounds.symLower[e * width + width - 1] = bounds.lower[e];
        bounds.symUpper[e * width + width - 1] = bounds.upper[e];
    }
    bounds.symbolic = true;
}

ValueBounds constantBounds(const std::vector<float>& data) {
    ValueBounds bounds;
    bounds.lower.assign(data.begin(), data.end());
    bounds.upper.assign(data.begin(), data.end());
    return bounds;
}

double absMax(double lower, double upper) { return std::max(std::fabs(lower), std::fabs(upper)); }

// ReLU on one element: interval plus DeepPoly triangle relaxation.
void reluElement(double& lower, double& upper, double* symLower, double* symUpper,
                 std::size_t width) {
    if (upper <= 0.0) {
        lower = upper = 0.0;
        if (symLower != nullptr) {
            std::fill(symLower, symLower + width, 0.0);
            std::fill(symUpper, symUpper + width, 0.0);
        }
        return;
    }
    if (lower >= 0.0) return;
    if (symLower != nullptr) {
        const double slope = upper / (upper - lower);
        for (std::size_t i = 0; i < width; ++i) symUpper[i] *= slope;
        symUpper[width - 1] -= slope * lower;
        if (upper < -lower) std::fill(symLower, symLower + width, 0.0);
    }
    lower = 0.0;
}

// Softmax bounds from logit bounds: s_i = 1 / (1 + sum_{j!=i} exp(z_j - z_i)).
void softmaxInterval(const std::vector<double>& lower, const std::vector<double>& upper,
                     std::vector<double>& outLower, std::vector<double>& outUpper) {
    const std::size_t count = lower.size();
    outLower.assign(count, 0.0);
    outUpper.assign(count, 1.0);
    for (std::size_t i = 0; i < count; ++i) {
        double low = 0.0, high = 0.0;
        for (std::size_t j = 0; j < count; ++j) {
            if (j == i) continue;
            low += std::exp(std::min(700.0, upper[j] - lower[i]));
            high += std::exp(std::min(700.0, lower[j] - upper[i]));
        }
        const double slack = kTranscendental * static_cast<double>(count + 4);
        outLower[i] = std::max(0.0, 1.0 / (1.0 + low) - slack);
        outUpper[i] = std::min(1.0, 1.0 / (1.0 + high) + slack);
    }
}

}  // namespace

// ------------------------------------------------------------ boxes

std::vector<float> Box::centre() const {
    std::vector<float> point(lower.size());
    for (std::size_t i = 0; i < point.size(); ++i) point[i] = 0.5f * (lower[i] + upper[i]);
    return point;
}

double Box::logVolume() const {
    double total = 0.0;
    for (std::size_t i = 0; i < lower.size(); ++i) {
        const double width = static_cast<double>(upper[i]) - lower[i];
        if (width > 0.0) total += std::log(width);
    }
    return total;
}

Box uniformBox(std::size_t inputs, float radius) {
    return {std::vector<float>(inputs, -radius), std::vector<float>(inputs, radius)};
}

Box ballAround(const std::vector<float>& point, float radius) {
    Box box{point, point};
    for (std::size_t i = 0; i < point.size(); ++i) {
        box.lower[i] -= radius;
        box.upper[i] += radius;
    }
    return box;
}

Box dataEnvelope(const std::vector<std::vector<float>>& rows, float margin) {
    if (rows.empty()) return {};
    Box box{rows.front(), rows.front()};
    for (const auto& row : rows) {
        for (std::size_t i = 0; i < row.size() && i < box.size(); ++i) {
            box.lower[i] = std::min(box.lower[i], row[i]);
            box.upper[i] = std::max(box.upper[i], row[i]);
        }
    }
    for (std::size_t i = 0; i < box.size(); ++i) {
        box.lower[i] -= margin;
        box.upper[i] += margin;
    }
    return box;
}

const ValueBounds* GraphBounds::find(const std::string& name) const {
    const auto found = values.find(name);
    return found == values.end() ? nullptr : &found->second;
}

// ------------------------------------------------------------ single-graph bounds

GraphBounds computeBounds(const IRGraph& graph, const Box& box, bool symbolic) {
    GraphBounds result;
    const std::size_t n = box.size();
    const std::size_t width = n + 1;
    result.inputs = n;
    Rows rows{width, &box};

    std::unordered_map<std::string, ValueBounds> constants;
    const auto operand = [&](const std::string& name) -> const ValueBounds* {
        if (const ValueBounds* value = result.find(name)) return value;
        const auto cached = constants.find(name);
        if (cached != constants.end()) return &cached->second;
        const std::vector<float>* data = constantData(graph, name);
        if (data == nullptr) return nullptr;
        return &constants.emplace(name, constantBounds(*data)).first->second;
    };

    for (const IRInstruction& instruction : graph.instructions) {
        if (instruction.operation == IROp::Input) {
            ValueBounds input;
            input.lower.assign(box.lower.begin(), box.lower.end());
            input.upper.assign(box.upper.begin(), box.upper.end());
            if (symbolic) {
                input.symLower.assign(n * width, 0.0);
                for (std::size_t i = 0; i < n; ++i) input.symLower[i * width + i] = 1.0;
                input.symUpper = input.symLower;
                input.symbolic = true;
            }
            result.values[instruction.output] = std::move(input);
            continue;
        }
        if (instruction.operation == IROp::Return) {
            result.complete = result.find(graph.outputName) != nullptr ||
                              operand(graph.outputName) != nullptr;
            break;
        }
        if (instruction.inputs.empty()) return result;
        const ValueBounds* first = operand(instruction.inputs[0]);
        if (first == nullptr) return result;
        ValueBounds out;
        const bool fused = instruction.operation == IROp::FusedGemmRelu;
        switch (instruction.operation) {
            case IROp::Gemm:
            case IROp::MatMul:
            case IROp::FusedGemmRelu: {
                Dense dense;
                IRInstruction view = instruction;
                if (fused) view.operation = IROp::Gemm;
                if (!denseView(graph, view, dense)) return result;
                const ValueBounds* second = operand(instruction.inputs[1]);
                const ValueBounds* bias =
                    dense.hasBias ? operand(instruction.inputs[2]) : nullptr;
                if (second == nullptr || (dense.hasBias && bias == nullptr)) return result;
                const std::vector<float>* weights = constantData(graph, instruction.inputs[1]);
                const bool useSymbolic = symbolic && weights != nullptr && first->symbolic;
                const std::size_t count = dense.rows * dense.columns;
                out.lower.assign(count, 0.0);
                out.upper.assign(count, 0.0);
                if (useSymbolic) {
                    out.symLower.assign(count * width, 0.0);
                    out.symUpper.assign(count * width, 0.0);
                    out.symbolic = true;
                }
                for (std::size_t row = 0; row < dense.rows; ++row) {
                    for (std::size_t column = 0; column < dense.columns; ++column) {
                        const std::size_t e = row * dense.columns + column;
                        double low = 0.0, high = 0.0, magnitude = 0.0;
                        if (bias != nullptr) {
                            const std::size_t b = dense.bias(bias->lower.size(), row, column);
                            low = bias->lower[b];
                            high = bias->upper[b];
                            magnitude = absMax(low, high);
                        }
                        double* symL = useSymbolic ? &out.symLower[e * width] : nullptr;
                        double* symU = useSymbolic ? &out.symUpper[e * width] : nullptr;
                        if (useSymbolic) {
                            symL[width - 1] = low;
                            symU[width - 1] = high;
                        }
                        for (std::size_t k = 0; k < dense.inner; ++k) {
                            const std::size_t l = dense.left(row, k);
                            const std::size_t r = dense.right(k, column);
                            const double aLow = first->lower[l], aHigh = first->upper[l];
                            const double bLow = second->lower[r], bHigh = second->upper[r];
                            const double p1 = aLow * bLow, p2 = aLow * bHigh;
                            const double p3 = aHigh * bLow, p4 = aHigh * bHigh;
                            low += std::min(std::min(p1, p2), std::min(p3, p4));
                            high += std::max(std::max(p1, p2), std::max(p3, p4));
                            magnitude += absMax(aLow, aHigh) * absMax(bLow, bHigh);
                            if (useSymbolic) {
                                const double w = (*weights)[r];
                                addScaled(symL, *first, l, w, w >= 0.0, width);
                                addScaled(symU, *first, l, w, w < 0.0, width);
                            }
                        }
                        // Float32 accumulation error of the interpreter's loop.
                        const double pad = gammaFactor(dense.inner + 1) * magnitude;
                        low -= pad;
                        high += pad;
                        if (useSymbolic) {
                            symL[width - 1] -= pad;
                            symU[width - 1] += pad;
                            low = std::max(low, rows.minimum(symL));
                            high = std::min(high, rows.maximum(symU));
                        }
                        if (fused) reluElement(low, high, symL, symU, width);
                        out.lower[e] = low;
                        out.upper[e] = high;
                    }
                }
                break;
            }
            case IROp::Add: {
                const ValueBounds* second = operand(instruction.inputs.at(1));
                if (second == nullptr || second->lower.size() != first->lower.size()) {
                    return result;
                }
                const std::size_t count = first->lower.size();
                const bool useSymbolic = symbolic && (first->symbolic || second->symbolic);
                out.lower.resize(count);
                out.upper.resize(count);
                if (useSymbolic) {
                    out.symLower.assign(count * width, 0.0);
                    out.symUpper.assign(count * width, 0.0);
                    out.symbolic = true;
                }
                for (std::size_t e = 0; e < count; ++e) {
                    double low = first->lower[e] + second->lower[e];
                    double high = first->upper[e] + second->upper[e];
                    const double pad = kUnit * std::max(absMax(low, high), 0.0);
                    low -= pad;
                    high += pad;
                    if (useSymbolic) {
                        double* symL = &out.symLower[e * width];
                        double* symU = &out.symUpper[e * width];
                        addScaled(symL, *first, e, 1.0, true, width);
                        addScaled(symL, *second, e, 1.0, true, width);
                        addScaled(symU, *first, e, 1.0, false, width);
                        addScaled(symU, *second, e, 1.0, false, width);
                        symL[width - 1] -= pad;
                        symU[width - 1] += pad;
                        low = std::max(low, rows.minimum(symL));
                        high = std::min(high, rows.maximum(symU));
                    }
                    out.lower[e] = low;
                    out.upper[e] = high;
                }
                break;
            }
            case IROp::Relu: {
                out = *first;
                if (symbolic && !out.symbolic) makeConstantRows(out, width);
                for (std::size_t e = 0; e < out.lower.size(); ++e) {
                    double* symL = out.symbolic ? &out.symLower[e * width] : nullptr;
                    double* symU = out.symbolic ? &out.symUpper[e * width] : nullptr;
                    reluElement(out.lower[e], out.upper[e], symL, symU, width);
                }
                break;
            }
            case IROp::TestReluDeadZone:
            case IROp::TestReluClamp:
            case IROp::TestLeakyRelu: {
                const double p = instruction.testFaultParameter;
                out.lower = first->lower;
                out.upper = first->upper;
                for (std::size_t e = 0; e < out.lower.size(); ++e) {
                    double& low = out.lower[e];
                    double& high = out.upper[e];
                    if (instruction.operation == IROp::TestLeakyRelu) {
                        low = low < 0.0 ? p * low : low;
                        high = high < 0.0 ? p * high : high;
                        if (low > high) std::swap(low, high);
                    } else if (instruction.operation == IROp::TestReluClamp) {
                        low = std::min(std::max(0.0, low), p);
                        high = std::min(std::max(0.0, high), p);
                    } else {
                        low = 0.0;  // not monotone: (0, p) drops to 0
                        high = std::max(0.0, high);
                    }
                    const double pad = kUnit * absMax(low, high);
                    low -= pad;
                    high += pad;
                }
                break;
            }
            case IROp::Sigmoid: {
                out.lower.resize(first->lower.size());
                out.upper.resize(first->lower.size());
                for (std::size_t e = 0; e < out.lower.size(); ++e) {
                    out.lower[e] = std::max(0.0, sigmoid(first->lower[e]) - kTranscendental);
                    out.upper[e] = std::min(1.0, sigmoid(first->upper[e]) + kTranscendental);
                }
                break;
            }
            case IROp::Softmax:
                softmaxInterval(first->lower, first->upper, out.lower, out.upper);
                break;
            case IROp::Input:
            case IROp::Return:
                break;
        }
        if (symbolic && !out.symbolic) makeConstantRows(out, width);
        result.values[instruction.output] = std::move(out);
    }
    return result;
}

// ------------------------------------------------------------ canonical form

IRGraph canonicalize(const IRGraph& graph) {
    IRGraph result = unfuseForAnalysis(graph);
    std::vector<IRInstruction> kept;
    const auto saveConstant = [&](const IRInstruction& instruction, std::vector<float> data) {
        TensorInfo info = result.values.count(instruction.output) != 0
                              ? result.values.at(instruction.output)
                              : TensorInfo{};
        info.name = instruction.output;
        info.isConstant = true;
        info.data = std::move(data);
        result.values[instruction.output] = info;
        result.constants[instruction.output] = info;
    };
    for (const IRInstruction& instruction : result.instructions) {
        const auto* a = instruction.inputs.empty() ? nullptr
                                                   : constantData(result, instruction.inputs[0]);
        bool folded = false;
        if (a != nullptr && instruction.operation == IROp::Add && instruction.inputs.size() == 2) {
            const auto* b = constantData(result, instruction.inputs[1]);
            if (b != nullptr && b->size() == a->size()) {
                std::vector<float> data(a->size());
                for (std::size_t i = 0; i < data.size(); ++i) data[i] = (*a)[i] + (*b)[i];
                saveConstant(instruction, std::move(data));
                folded = true;
            }
        } else if (a != nullptr && instruction.inputs.size() == 1 && !a->empty()) {
            std::vector<float> data(a->size());
            folded = true;
            if (instruction.operation == IROp::Relu) {
                for (std::size_t i = 0; i < data.size(); ++i) data[i] = std::max(0.0f, (*a)[i]);
            } else if (instruction.operation == IROp::Sigmoid) {
                for (std::size_t i = 0; i < data.size(); ++i) {
                    data[i] = 1.0f / (1.0f + std::exp(-(*a)[i]));
                }
            } else if (instruction.operation == IROp::Softmax) {
                const float maximum = *std::max_element(a->begin(), a->end());
                float total = 0.0f;
                for (std::size_t i = 0; i < data.size(); ++i) {
                    data[i] = std::exp((*a)[i] - maximum);
                    total += data[i];
                }
                for (float& value : data) value /= total;
            } else {
                folded = false;
            }
            if (folded) saveConstant(instruction, std::move(data));
        }
        if (!folded) kept.push_back(instruction);
    }
    result.instructions = std::move(kept);
    const auto live = liveValues(result);
    kept.clear();
    for (const IRInstruction& instruction : result.instructions) {
        if (instruction.operation == IROp::Input || instruction.operation == IROp::Return ||
            live.count(instruction.output) != 0) {
            kept.push_back(instruction);
        }
    }
    result.instructions = std::move(kept);
    return result;
}

std::uint64_t graphFingerprint(const IRGraph& graph) {
    const IRGraph canonical = canonicalize(graph);
    std::uint64_t hash = 1469598103934665603ull;
    const auto mix = [&](const void* data, std::size_t size) {
        const auto* bytes = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < size; ++i) {
            hash ^= bytes[i];
            hash *= 1099511628211ull;
        }
    };
    std::unordered_map<std::string, std::uint32_t> ids;
    for (const IRInstruction& instruction : canonical.instructions) {
        const auto op = static_cast<std::uint32_t>(instruction.operation);
        mix(&op, sizeof op);
        for (const std::string& input : instruction.inputs) {
            if (const auto* data = constantData(canonical, input)) {
                const std::uint32_t tag = 0xC0u;
                mix(&tag, sizeof tag);
                if (!data->empty()) mix(data->data(), data->size() * sizeof(float));
            } else {
                const std::uint32_t id = ids.count(input) ? ids[input] : 0xFFFFFFFFu;
                mix(&id, sizeof id);
            }
        }
        const std::uint32_t attributes = (instruction.transA ? 1u : 0u) |
                                         (instruction.transB ? 2u : 0u) |
                                         (static_cast<std::uint32_t>(instruction.axis + 8) << 2);
        mix(&attributes, sizeof attributes);
        mix(&instruction.testFaultParameter, sizeof(float));
        const std::uint32_t id = static_cast<std::uint32_t>(ids.size());
        ids[instruction.output] = id;
    }
    return hash;
}

std::string fingerprintHex(std::uint64_t value) {
    std::ostringstream output;
    output << std::hex << std::setw(16) << std::setfill('0') << value;
    return output.str();
}

// ------------------------------------------------------------ differential bounds

bool DifferenceBounds::allExact() const {
    return !exact.empty() && std::all_of(exact.begin(), exact.end(), [](bool e) { return e; });
}

double DifferenceBounds::maxAbs() const {
    double result = 0.0;
    for (std::size_t i = 0; i < lower.size(); ++i) {
        if (exact[i]) continue;
        result = std::max(result, absMax(lower[i], upper[i]));
    }
    return result;
}

namespace {

struct Delta {
    std::vector<double> lower, upper;
    std::vector<bool> exact;
    // Symbolic bounds on Delta in terms of the box inputs (ReluDiff-style):
    // row-major [elements x (inputs + 1)], last column constant.
    std::vector<double> symLower, symUpper;
    bool symbolic = false;
};

void constantDeltaRows(Delta& delta, std::size_t width) {
    const std::size_t count = delta.lower.size();
    delta.symLower.assign(count * width, 0.0);
    delta.symUpper.assign(count * width, 0.0);
    for (std::size_t e = 0; e < count; ++e) {
        delta.symLower[e * width + width - 1] = delta.lower[e];
        delta.symUpper[e * width + width - 1] = delta.upper[e];
    }
    delta.symbolic = true;
}

// Name of the logits: the input of a final Softmax/Sigmoid, else the output.
std::string logitsName(const IRGraph& canonical, bool& sigmoidOutput) {
    sigmoidOutput = false;
    for (const IRInstruction& instruction : canonical.instructions) {
        if (instruction.output != canonical.outputName || instruction.inputs.empty()) continue;
        if (instruction.operation == IROp::Softmax) return instruction.inputs[0];
        if (instruction.operation == IROp::Sigmoid) {
            const TensorInfo* info = infoOf(canonical, instruction.output);
            sigmoidOutput = info != nullptr && elementCount(info->shape) == 1;
            return instruction.inputs[0];
        }
    }
    return canonical.outputName;
}

// Pre-canonicalised, aligned pair of graphs; analyses any box.
class Differ {
public:
    Differ(const IRGraph& original, const IRGraph& candidate)
        : a_(canonicalize(original)), b_(canonicalize(candidate)) {
        aligned_ = align();
    }

    bool aligned() const { return aligned_; }
    const IRGraph& original() const { return a_; }
    const IRGraph& candidate() const { return b_; }

    // Bounds Delta for every aligned value; `stopAt` (a value of the original)
    // ends the walk early. Returns the Delta of `target` (original's name).
    DifferenceBounds run(const Box& box, const std::string& target, GraphBounds* boundsA = nullptr,
                         GraphBounds* boundsB = nullptr, bool collectSources = false) const {
        DifferenceBounds result;
        GraphBounds ta = computeBounds(a_, box);
        GraphBounds tb = computeBounds(b_, box);
        result.complete = ta.complete && tb.complete;
        result.aligned = aligned_;
        bool unusedSigmoid = false;
        const std::string targetB = aligned_ && mapping_.count(target) ? mapping_.at(target)
                                    : target == a_.outputName      ? b_.outputName
                                                                   : logitsName(b_, unusedSigmoid);
        if (!aligned_) {
            independentOnly(ta, tb, target, targetB, box, result);
        } else {
            std::unordered_map<std::string, Delta> deltas;
            const Rows rows{box.size() + 1, &box};
            for (std::size_t index = 0; index < a_.instructions.size(); ++index) {
                const IRInstruction& x = a_.instructions[index];
                const IRInstruction& y = b_.instructions[index];
                if (x.operation == IROp::Return) break;
                Delta delta;
                if (x.operation == IROp::Input) {
                    const std::size_t count = box.size();
                    delta.lower.assign(count, 0.0);
                    delta.upper.assign(count, 0.0);
                    delta.exact.assign(count, true);
                    constantDeltaRows(delta, box.size() + 1);
                } else if (!step(x, y, ta, tb, rows, deltas, delta,
                                 collectSources ? &result.sources : nullptr)) {
                    result.aligned = false;
                    independentOnly(ta, tb, target, targetB, box, result);
                    break;
                }
                result.alignedValues += delta.exact.size();
                result.exactValues += static_cast<std::size_t>(
                    std::count(delta.exact.begin(), delta.exact.end(), true));
                const bool done = x.output == target;
                deltas[x.output] = std::move(delta);
                if (done) break;
            }
            const auto found = deltas.find(target);
            if (result.aligned && found != deltas.end()) {
                result.lower = found->second.lower;
                result.upper = found->second.upper;
                result.exact = found->second.exact;
            } else if (result.aligned) {
                result.aligned = false;
                independentOnly(ta, tb, target, targetB, box, result);
            }
        }
        if (boundsA != nullptr) *boundsA = std::move(ta);
        if (boundsB != nullptr) *boundsB = std::move(tb);
        return result;
    }

    std::string mapped(const std::string& name) const {
        const auto found = mapping_.find(name);
        return found == mapping_.end() ? std::string() : found->second;
    }

    // Input dimensions weighted by first-layer weight magnitude (split heuristic).
    std::vector<double> influence() const {
        std::vector<double> weight;
        for (const IRGraph* graph : {&a_, &b_}) {
            for (const IRInstruction& instruction : graph->instructions) {
                Dense dense;
                if (!isDense(instruction.operation) || instruction.inputs[0] != graph->inputName ||
                    !denseView(*graph, instruction, dense) || dense.rows != 1) {
                    continue;
                }
                const auto* w = constantData(*graph, instruction.inputs[1]);
                if (w == nullptr) continue;
                weight.resize(dense.inner, 0.0);
                for (std::size_t k = 0; k < dense.inner; ++k) {
                    for (std::size_t c = 0; c < dense.columns; ++c) {
                        weight[k] += std::fabs((*w)[dense.right(k, c)]);
                    }
                }
                break;
            }
        }
        return weight;
    }

private:
    static void independentOnly(const GraphBounds& ta, const GraphBounds& tb,
                                const std::string& targetA, const std::string& targetB,
                                const Box& box, DifferenceBounds& result) {
        const ValueBounds* a = ta.find(targetA);
        const ValueBounds* b = tb.find(targetB);
        result.lower.clear();
        result.upper.clear();
        result.exact.clear();
        if (a == nullptr || b == nullptr || a->lower.size() != b->lower.size()) {
            result.complete = false;
            return;
        }
        const std::size_t width = box.size() + 1;
        const Rows rows{width, &box};
        std::vector<double> row(width);
        for (std::size_t e = 0; e < a->lower.size(); ++e) {
            double low = b->lower[e] - a->upper[e];
            double high = b->upper[e] - a->lower[e];
            // Shared-input correlation: L_b(x) - U_a(x) <= Delta <= U_b(x) - L_a(x).
            if (a->symbolic && b->symbolic) {
                for (std::size_t i = 0; i < width; ++i) {
                    row[i] = b->symLower[e * width + i] - a->symUpper[e * width + i];
                }
                low = std::max(low, rows.minimum(row.data()));
                for (std::size_t i = 0; i < width; ++i) {
                    row[i] = b->symUpper[e * width + i] - a->symLower[e * width + i];
                }
                high = std::min(high, rows.maximum(row.data()));
            }
            result.lower.push_back(low);
            result.upper.push_back(high);
            result.exact.push_back(false);
        }
    }

    bool correspond(const std::string& x, const std::string& y) const {
        const auto* cx = constantData(a_, x);
        const auto* cy = constantData(b_, y);
        if (cx != nullptr || cy != nullptr) {
            return cx != nullptr && cy != nullptr && cx->size() == cy->size();
        }
        const auto found = mapping_.find(x);
        return found != mapping_.end() && found->second == y;
    }

    bool align() {
        if (a_.instructions.size() != b_.instructions.size()) return false;
        mapping_.clear();
        for (std::size_t index = 0; index < a_.instructions.size(); ++index) {
            const IRInstruction& x = a_.instructions[index];
            const IRInstruction& y = b_.instructions[index];
            if (x.operation == IROp::Input || y.operation == IROp::Input) {
                if (x.operation != y.operation) return false;
                mapping_[x.output] = y.output;
                continue;
            }
            if (x.operation == IROp::Return || y.operation == IROp::Return) {
                if (x.operation != y.operation || x.inputs.empty() || y.inputs.empty() ||
                    !correspond(x.inputs[0], y.inputs[0])) {
                    return false;
                }
                continue;
            }
            const bool bothDense = isDense(x.operation) && isDense(y.operation);
            const bool elementwise = (isUnary(x.operation) || x.operation == IROp::Softmax) &&
                                     (isUnary(y.operation) || y.operation == IROp::Softmax);
            const bool bothAdd = x.operation == IROp::Add && y.operation == IROp::Add;
            if (!bothDense && !elementwise && !bothAdd) return false;
            if (x.inputs.empty() || y.inputs.empty() || !correspond(x.inputs[0], y.inputs[0])) {
                return false;
            }
            if (bothDense) {
                Dense dx, dy;
                if (!denseView(a_, x, dx) || !denseView(b_, y, dy) || dx.rows != dy.rows ||
                    dx.inner != dy.inner || dx.columns != dy.columns || dx.transA != dy.transA ||
                    !correspond(x.inputs[1], y.inputs[1])) {
                    return false;
                }
                if (dx.hasBias && dy.hasBias && !correspond(x.inputs[2], y.inputs[2])) return false;
                if ((dx.hasBias && constantData(a_, x.inputs[2]) == nullptr) ||
                    (dy.hasBias && constantData(b_, y.inputs[2]) == nullptr)) {
                    if (dx.hasBias != dy.hasBias) return false;
                }
            }
            if (bothAdd && (x.inputs.size() != 2 || y.inputs.size() != 2 ||
                            !correspond(x.inputs[1], y.inputs[1]))) {
                return false;
            }
            mapping_[x.output] = y.output;
        }
        return true;
    }

    // Delta of an operand: aligned value, or a pair of constants.
    bool operandDelta(const std::string& x, const std::string& y,
                      const std::unordered_map<std::string, Delta>& deltas, Delta& out,
                      std::size_t width) const {
        const auto* cx = constantData(a_, x);
        const auto* cy = constantData(b_, y);
        if (cx != nullptr && cy != nullptr) {
            out.lower.resize(cx->size());
            out.upper.resize(cx->size());
            out.exact.resize(cx->size());
            for (std::size_t i = 0; i < cx->size(); ++i) {
                const double d = static_cast<double>((*cy)[i]) - (*cx)[i];
                out.lower[i] = out.upper[i] = d;
                out.exact[i] = (*cy)[i] == (*cx)[i];
            }
            constantDeltaRows(out, width);
            return true;
        }
        const auto found = deltas.find(x);
        if (found == deltas.end()) return false;
        out = found->second;
        return true;
    }

    static const ValueBounds* boundsOf(const GraphBounds& table, const IRGraph& graph,
                                       const std::string& name, ValueBounds& scratch) {
        if (const ValueBounds* value = table.find(name)) return value;
        const auto* data = constantData(graph, name);
        if (data == nullptr) return nullptr;
        scratch = constantBounds(*data);
        return &scratch;
    }

    bool step(const IRInstruction& x, const IRInstruction& y, const GraphBounds& ta,
              const GraphBounds& tb, const Rows& rows,
              const std::unordered_map<std::string, Delta>& deltas, Delta& delta,
              std::vector<DivergenceSource>* sources) const {
        const std::size_t width = rows.width;
        const ValueBounds* outA = ta.find(x.output);
        const ValueBounds* outB = tb.find(y.output);
        if (outA == nullptr || outB == nullptr || outA->lower.size() != outB->lower.size()) {
            return false;
        }
        const std::size_t count = outA->lower.size();
        delta.lower.assign(count, 0.0);
        delta.upper.assign(count, 0.0);
        delta.exact.assign(count, false);

        if (isDense(x.operation)) {
            if (!denseStep(x, y, ta, tb, width, deltas, delta)) return false;
        } else if (x.operation == IROp::Add) {
            Delta d0, d1;
            if (!operandDelta(x.inputs[0], y.inputs[0], deltas, d0, width) ||
                !operandDelta(x.inputs[1], y.inputs[1], deltas, d1, width) ||
                d0.lower.size() != count || d1.lower.size() != count) {
                return false;
            }
            delta.symLower.assign(count * width, 0.0);
            delta.symUpper.assign(count * width, 0.0);
            delta.symbolic = d0.symbolic && d1.symbolic;
            for (std::size_t e = 0; e < count; ++e) {
                delta.exact[e] = d0.exact[e] && d1.exact[e];
                const double pad =
                    kUnit * (absMax(outA->lower[e], outA->upper[e]) +
                             absMax(outB->lower[e], outB->upper[e]));
                delta.lower[e] = d0.lower[e] + d1.lower[e] - pad;
                delta.upper[e] = d0.upper[e] + d1.upper[e] + pad;
                if (delta.symbolic) {
                    for (std::size_t i = 0; i < width; ++i) {
                        delta.symLower[e * width + i] =
                            d0.symLower[e * width + i] + d1.symLower[e * width + i];
                        delta.symUpper[e * width + i] =
                            d0.symUpper[e * width + i] + d1.symUpper[e * width + i];
                    }
                    delta.symLower[e * width + width - 1] -= pad;
                    delta.symUpper[e * width + width - 1] += pad;
                }
            }
        } else {
            Delta dz;
            if (!operandDelta(x.inputs[0], y.inputs[0], deltas, dz, width) ||
                dz.lower.size() != count) {
                return false;
            }
            ValueBounds scratchA, scratchB;
            const ValueBounds* za = boundsOf(ta, a_, x.inputs[0], scratchA);
            const ValueBounds* zb = boundsOf(tb, b_, y.inputs[0], scratchB);
            if (za == nullptr || zb == nullptr) return false;
            if (x.operation == IROp::Softmax || y.operation == IROp::Softmax) {
                softmaxStep(x, y, *za, *zb, dz, delta);
            } else {
                double from = 0.0, to = 0.0;
                const bool regional = sources != nullptr && x.operation == IROp::Relu &&
                                      y.operation != IROp::Relu && differenceRegion(y, from, to);
                for (std::size_t e = 0; e < count; ++e) {
                    unaryStep(x, y, za->lower[e], za->upper[e], zb->lower[e], zb->upper[e],
                              dz.lower[e], dz.upper[e], dz.exact[e], delta, e);
                    if (regional && !delta.exact[e]) {
                        const double low = std::max(from, zb->lower[e]);
                        const double high = std::min(to, zb->upper[e]);
                        if (low < high) sources->push_back({y.inputs[0], e, low, high});
                    }
                }
                // Symbolic Delta survives a ReLU pair whose units are active in
                // both models: relu(z') - relu(z) = z' - z exactly.
                if (dz.symbolic && x.operation == IROp::Relu && y.operation == IROp::Relu) {
                    constantDeltaRows(delta, width);
                    for (std::size_t e = 0; e < count; ++e) {
                        if (delta.exact[e] || za->lower[e] < 0.0 || zb->lower[e] < 0.0) continue;
                        std::copy_n(&dz.symLower[e * width], width, &delta.symLower[e * width]);
                        std::copy_n(&dz.symUpper[e * width], width, &delta.symUpper[e * width]);
                    }
                }
            }
        }
        // Always sound: independent bounds of the two values, and the
        // concretised symbolic Delta.
        for (std::size_t e = 0; e < count; ++e) {
            if (delta.exact[e]) {
                delta.lower[e] = delta.upper[e] = 0.0;
                if (delta.symbolic) {
                    std::fill_n(&delta.symLower[e * width], width, 0.0);
                    std::fill_n(&delta.symUpper[e * width], width, 0.0);
                }
                continue;
            }
            delta.lower[e] = std::max(delta.lower[e], outB->lower[e] - outA->upper[e]);
            delta.upper[e] = std::min(delta.upper[e], outB->upper[e] - outA->lower[e]);
            if (delta.symbolic) {
                delta.lower[e] = std::max(delta.lower[e], rows.minimum(&delta.symLower[e * width]));
                delta.upper[e] = std::min(delta.upper[e], rows.maximum(&delta.symUpper[e * width]));
            }
            if (delta.lower[e] > delta.upper[e]) {  // rounding at a degenerate point
                const double mid = 0.5 * (delta.lower[e] + delta.upper[e]);
                delta.lower[e] = delta.upper[e] = mid;
            }
        }
        return true;
    }

    bool denseStep(const IRInstruction& x, const IRInstruction& y, const GraphBounds& ta,
                   const GraphBounds& tb, std::size_t width,
                   const std::unordered_map<std::string, Delta>& deltas, Delta& delta) const {
        Dense dx, dy;
        if (!denseView(a_, x, dx) || !denseView(b_, y, dy)) return false;
        Delta dIn;
        if (!operandDelta(x.inputs[0], y.inputs[0], deltas, dIn, width)) return false;
        ValueBounds scratchA, scratchB;
        const ValueBounds* inA = boundsOf(ta, a_, x.inputs[0], scratchA);
        const ValueBounds* inB = boundsOf(tb, b_, y.inputs[0], scratchB);
        const auto* wA = constantData(a_, x.inputs[1]);
        const auto* wB = constantData(b_, y.inputs[1]);
        if (inA == nullptr || inB == nullptr) return false;
        if (wA == nullptr || wB == nullptr) {
            // Variable weights: only the independent bound (applied by caller).
            for (std::size_t e = 0; e < delta.lower.size(); ++e) {
                delta.lower[e] = -std::numeric_limits<double>::infinity();
                delta.upper[e] = std::numeric_limits<double>::infinity();
            }
            return true;
        }
        const std::vector<float>* biasA = dx.hasBias ? constantData(a_, x.inputs[2]) : nullptr;
        const std::vector<float>* biasB = dy.hasBias ? constantData(b_, y.inputs[2]) : nullptr;
        Delta dBias;
        const bool variableBias = (dx.hasBias && biasA == nullptr) || (dy.hasBias && biasB == nullptr);
        if (variableBias && !operandDelta(x.inputs[2], y.inputs[2], deltas, dBias, width)) {
            return false;
        }
        // Symbolic rows need symbolic Delta of the input and symbolic bounds of x.
        const bool symbolic = dIn.symbolic && inA->symbolic && !variableBias;
        if (symbolic) {
            delta.symLower.assign(delta.lower.size() * width, 0.0);
            delta.symUpper.assign(delta.lower.size() * width, 0.0);
            delta.symbolic = true;
        }
        for (std::size_t row = 0; row < dx.rows; ++row) {
            for (std::size_t column = 0; column < dx.columns; ++column) {
                const std::size_t e = row * dx.columns + column;
                double low = 0.0, high = 0.0, magA = 0.0, magB = 0.0;
                bool exact = true;
                if (variableBias) {
                    const std::size_t b = dx.bias(dBias.lower.size(), row, column);
                    low = dBias.lower[b];
                    high = dBias.upper[b];
                    exact = dBias.exact[b];
                } else {
                    const double ba = biasA ? (*biasA)[dx.bias(biasA->size(), row, column)] : 0.0;
                    const double bb = biasB ? (*biasB)[dy.bias(biasB->size(), row, column)] : 0.0;
                    low = high = bb - ba;
                    exact = ba == bb;
                    magA = std::fabs(ba);
                    magB = std::fabs(bb);
                }
                double* symL = symbolic ? &delta.symLower[e * width] : nullptr;
                double* symU = symbolic ? &delta.symUpper[e * width] : nullptr;
                if (symbolic) {
                    symL[width - 1] = low;
                    symU[width - 1] = high;
                }
                for (std::size_t k = 0; k < dx.inner; ++k) {
                    const std::size_t l = dx.left(row, k);
                    const double wa = (*wA)[dx.right(k, column)];
                    const double wb = (*wB)[dy.right(k, column)];
                    const double xLow = inA->lower[l], xHigh = inA->upper[l];
                    const bool zeroInput = xLow == 0.0 && xHigh == 0.0;
                    exact = exact && dIn.exact[l] && (wa == wb || zeroInput);
                    // wb * dx + (wb - wa) * x
                    const double t1 = wb * dIn.lower[l], t2 = wb * dIn.upper[l];
                    const double dw = wb - wa;
                    const double t3 = dw * xLow, t4 = dw * xHigh;
                    low += std::min(t1, t2) + std::min(t3, t4);
                    high += std::max(t1, t2) + std::max(t3, t4);
                    magA += std::fabs(wa) * absMax(xLow, xHigh);
                    magB += std::fabs(wb) * absMax(inB->lower[l], inB->upper[l]);
                    if (symbolic) {
                        const double* dL = &dIn.symLower[l * width];
                        const double* dU = &dIn.symUpper[l * width];
                        const double* xL = &inA->symLower[l * width];
                        const double* xU = &inA->symUpper[l * width];
                        const double* lowD = wb >= 0.0 ? dL : dU;
                        const double* highD = wb >= 0.0 ? dU : dL;
                        const double* lowX = dw >= 0.0 ? xL : xU;
                        const double* highX = dw >= 0.0 ? xU : xL;
                        for (std::size_t i = 0; i < width; ++i) {
                            symL[i] += wb * lowD[i] + dw * lowX[i];
                            symU[i] += wb * highD[i] + dw * highX[i];
                        }
                    }
                }
                delta.exact[e] = exact;
                const double pad = gammaFactor(dx.inner + 1) * (magA + magB);
                delta.lower[e] = low - pad;
                delta.upper[e] = high + pad;
                if (symbolic) {
                    symL[width - 1] -= pad;
                    symU[width - 1] += pad;
                }
            }
        }
        return true;
    }

    static void reluDifference(double l, double u, double l2, double u2, double dl, double du,
                               double& low, double& high, bool& exact) {
        if (u <= 0.0 && u2 <= 0.0) {
            low = high = 0.0;
            exact = true;  // both outputs are exactly zero
            return;
        }
        if (exact) {
            low = high = 0.0;
            return;
        }
        if (l >= 0.0 && l2 >= 0.0) {
            low = dl;
            high = du;
        } else {
            low = std::min(0.0, dl);
            high = std::max(0.0, du);
        }
        if (u <= 0.0) {  // Delta = relu(z')
            low = std::max(low, std::max(0.0, l2));
            high = std::min(high, std::max(0.0, u2));
        }
        if (u2 <= 0.0) {  // Delta = -relu(z)
            low = std::max(low, -std::max(0.0, u));
            high = std::min(high, -std::max(0.0, l));
        }
    }

    // Pre-activation interval where g differs from ReLU (false: everywhere).
    static bool differenceRegion(const IRInstruction& g, double& from, double& to) {
        const double p = g.testFaultParameter;
        const double infinity = std::numeric_limits<double>::infinity();
        switch (g.operation) {
            case IROp::TestReluDeadZone: from = 0.0; to = p; return true;
            case IROp::TestReluClamp: from = p; to = infinity; return true;
            case IROp::TestLeakyRelu: from = -infinity; to = 0.0; return true;
            default: return false;
        }
    }

    // Range of g(z) - relu(z) over z in [l, u], and whether it is identically 0.
    static void pointwiseRange(const IRInstruction& g, double l, double u, double& low,
                               double& high, bool& zero) {
        const double p = g.testFaultParameter;
        zero = false;
        low = high = 0.0;
        switch (g.operation) {
            case IROp::TestReluDeadZone:
                if (u <= 0.0 || l >= p) {
                    zero = true;
                } else {
                    low = -std::min(u, p);
                }
                return;
            case IROp::TestReluClamp:
                if (u <= p) {
                    zero = true;
                } else {
                    low = -(u - p);
                    high = -std::max(0.0, l - p);
                }
                return;
            case IROp::TestLeakyRelu:
                if (l >= 0.0) {
                    zero = true;
                } else {
                    const double a = p * l, b = p * std::min(u, 0.0);
                    low = std::min({a, b, 0.0});
                    high = std::max({a, b, 0.0});
                }
                return;
            case IROp::Sigmoid: {
                const auto h = [](double z) { return sigmoid(z) - std::max(0.0, z); };
                low = std::min(h(l), h(u)) - kTranscendental;
                high = h(std::clamp(0.0, l, u)) + kTranscendental;
                return;
            }
            default:
                low = -std::numeric_limits<double>::infinity();
                high = std::numeric_limits<double>::infinity();
                return;
        }
    }

    void unaryStep(const IRInstruction& x, const IRInstruction& y, double l, double u, double l2,
                   double u2, double dl, double du, bool dzExact, Delta& delta,
                   std::size_t e) const {
        const bool sameOp = x.operation == y.operation &&
                            (x.operation == IROp::Relu || x.operation == IROp::Sigmoid ||
                             x.testFaultParameter == y.testFaultParameter);
        double low = -std::numeric_limits<double>::infinity();
        double high = std::numeric_limits<double>::infinity();
        bool exact = false;
        if (sameOp && dzExact) {
            exact = true;
        } else if (sameOp && x.operation == IROp::Relu) {
            exact = false;
            reluDifference(l, u, l2, u2, dl, du, low, high, exact);
        } else if (sameOp && x.operation == IROp::Sigmoid) {
            // sigma(z') - sigma(z) = sigma'(xi) * dz with 0 < sigma'(xi) <= s.
            const double hullLow = std::min(l, l2), hullHigh = std::max(u, u2);
            const double nearest = std::clamp(0.0, hullLow, hullHigh);
            const double s = sigmoid(nearest) * (1.0 - sigmoid(nearest));
            low = std::min(0.0, s * dl) - 2.0 * kTranscendental;
            high = std::max(0.0, s * du) + 2.0 * kTranscendental;
        } else if (x.operation == IROp::Relu) {
            // g(z') - relu(z) = [relu(z') - relu(z)] + [g(z') - relu(z')].
            bool partExact = dzExact;
            double rLow = 0.0, rHigh = 0.0;
            reluDifference(l, u, l2, u2, dl, du, rLow, rHigh, partExact);
            double pLow = 0.0, pHigh = 0.0;
            bool zero = false;
            pointwiseRange(y, l2, u2, pLow, pHigh, zero);
            exact = partExact && zero;
            low = rLow + pLow;
            high = rHigh + pHigh;
        }
        delta.exact[e] = exact;
        delta.lower[e] = low;
        delta.upper[e] = high;
    }

    static void softmaxStep(const IRInstruction& x, const IRInstruction& y, const ValueBounds& za,
                            const ValueBounds& zb, const Delta& dz, Delta& delta) {
        const std::size_t count = delta.lower.size();
        const bool allExact = std::all_of(dz.exact.begin(), dz.exact.end(), [](bool v) { return v; });
        if (x.operation == IROp::Softmax && y.operation == IROp::Softmax) {
            if (allExact) {
                std::fill(delta.exact.begin(), delta.exact.end(), true);
                return;
            }
            // Softmax is shift invariant and sum_j |J_ij| = 2 p_i (1 - p_i).
            std::vector<double> hullLow(count), hullHigh(count), pLow, pHigh;
            double spanLow = std::numeric_limits<double>::infinity();
            double spanHigh = -std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i < count; ++i) {
                hullLow[i] = std::min(za.lower[i], zb.lower[i]);
                hullHigh[i] = std::max(za.upper[i], zb.upper[i]);
                spanLow = std::min(spanLow, dz.lower[i]);
                spanHigh = std::max(spanHigh, dz.upper[i]);
            }
            softmaxInterval(hullLow, hullHigh, pLow, pHigh);
            const double span = spanHigh - spanLow;
            for (std::size_t i = 0; i < count; ++i) {
                const double g = pLow[i] <= 0.5 && pHigh[i] >= 0.5
                                     ? 0.25
                                     : std::max(pLow[i] * (1.0 - pLow[i]), pHigh[i] * (1.0 - pHigh[i]));
                const double pad = 2.0 * kTranscendental * static_cast<double>(count + 4);
                delta.lower[i] = -g * span - pad;
                delta.upper[i] = g * span + pad;
            }
            return;
        }
        for (std::size_t i = 0; i < count; ++i) {
            delta.lower[i] = -std::numeric_limits<double>::infinity();
            delta.upper[i] = std::numeric_limits<double>::infinity();
        }
    }

    IRGraph a_;
    IRGraph b_;
    bool aligned_ = false;
    std::unordered_map<std::string, std::string> mapping_;
};

std::size_t argmaxOf(const std::vector<float>& values) {
    if (values.size() == 1) return values[0] > 0.5f ? 1 : 0;  // binary sigmoid
    return static_cast<std::size_t>(std::max_element(values.begin(), values.end()) -
                                    values.begin());
}

}  // namespace

DifferenceBounds differentialBounds(const IRGraph& original, const IRGraph& candidate,
                                    const Box& box) {
    const Differ differ(original, candidate);
    return differ.run(box, differ.original().outputName);
}

LogitDifference logitDifference(const IRGraph& original, const IRGraph& candidate,
                                const Box& box) {
    LogitDifference result;
    const Differ differ(original, candidate);
    const std::string logits = logitsName(differ.original(), result.sigmoidOutput);
    GraphBounds ta, tb;
    result.delta = differ.run(box, logits, &ta, &tb);
    if (const ValueBounds* a = ta.find(logits)) result.originalLogits = *a;
    const std::string mapped = differ.mapped(logits);
    if (const ValueBounds* b = tb.find(mapped.empty() ? logits : mapped)) {
        result.candidateLogits = *b;
    }
    return result;
}

// ------------------------------------------------------------ verdicts

const char* verdictName(Verdict verdict) {
    switch (verdict) {
        case Verdict::Certified: return "CERTIFIED";
        case Verdict::Refuted: return "REFUTED";
        case Verdict::Undecided: return "UNDECIDED";
    }
    return "UNDECIDED";
}

const char* certificateKindName(CertificateKind kind) {
    switch (kind) {
        case CertificateKind::Exact: return "bit-exact";
        case CertificateKind::Bounded: return "epsilon-bounded";
        case CertificateKind::None: return "none";
    }
    return "none";
}

bool exposes(const IRGraph& original, const IRGraph& candidate, const std::vector<float>& input,
             float tolerance, float* difference) {
    DiagnosticEngine diagnostics;
    const auto a = executeIR(original, input, diagnostics);
    const auto b = executeIR(candidate, input, diagnostics);
    if (!a || !b || a->size() != b->size()) return true;
    float worst = 0.0f;
    for (std::size_t i = 0; i < a->size(); ++i) {
        const float d = std::fabs((*a)[i] - (*b)[i]);
        if (!std::isfinite(d)) {
            worst = std::numeric_limits<float>::infinity();
            break;
        }
        worst = std::max(worst, d);
    }
    if (difference != nullptr) *difference = worst;
    return worst > tolerance || argmaxOf(*a) != argmaxOf(*b);
}

namespace {

struct Leaf {
    Box box;
    std::size_t depth = 0;
    double logVolume = 0.0;
};

std::size_t splitDimension(const Box& box, const std::vector<double>& influence) {
    std::size_t best = 0;
    double bestScore = -1.0;
    for (std::size_t i = 0; i < box.size(); ++i) {
        const double width = static_cast<double>(box.upper[i]) - box.lower[i];
        const double score = width * (i < influence.size() ? influence[i] + 1.0e-3 : 1.0);
        if (score > bestScore) {
            bestScore = score;
            best = i;
        }
    }
    return best;
}

// Certificate-guided falsification inside one undecided box.
std::optional<std::vector<float>> guidedWitness(const Differ& differ, const IRGraph& original,
                                                const IRGraph& candidate, const Box& box,
                                                const CertifyOptions& options,
                                                std::size_t& evaluations, std::string& how) {
    const IRGraph& a = differ.original();
    const IRGraph& b = differ.candidate();
    const std::vector<float> centre = box.centre();
    const auto check = [&](const std::vector<float>& x) {
        ++evaluations;
        return exposes(original, candidate, x, options.tolerance);
    };
    // 1. Divergence sources: put the unit inside the interval where the
    //    candidate's operator differs from the original's.
    const DifferenceBounds bounds = differ.run(box, a.outputName, nullptr, nullptr, true);
    std::size_t tried = 0;
    for (const DivergenceSource& source : bounds.sources) {
        if (tried == 4 || evaluations >= options.maxEvaluations) break;
        ++tried;
        const double low = std::isfinite(source.from) ? source.from : source.to - 1.0;
        const double high = std::isfinite(source.to) ? source.to : source.from + 1.0;
        for (const double fraction : {0.5, 0.9, 0.1}) {
            std::vector<float> point = centre;
            const float target = static_cast<float>(low + fraction * (high - low));
            solveForValue(b, source.value, source.unit, target, point, box.lower, box.upper, 10);
            if (check(point)) {
                how = "divergence-source";
                return point;
            }
            if (evaluations >= options.maxEvaluations) break;
        }
    }
    // 2. Gradient vertex: the box corner that maximises (and minimises) the
    //    linearised difference of the output with the widest bound.
    if (bounds.lower.empty() || evaluations + 2 > options.maxEvaluations) return std::nullopt;
    std::size_t k = 0;
    double widest = -1.0;
    for (std::size_t i = 0; i < bounds.lower.size(); ++i) {
        const double span = bounds.exact[i] ? 0.0 : absMax(bounds.lower[i], bounds.upper[i]);
        if (span > widest) {
            widest = span;
            k = i;
        }
    }
    float valueA = 0.0f, valueB = 0.0f;
    std::vector<float> gradientA, gradientB;
    if (!valueGradient(a, centre, a.outputName, k, valueA, gradientA) ||
        !valueGradient(b, centre, b.outputName, k, valueB, gradientB) ||
        gradientA.size() != gradientB.size()) {
        return std::nullopt;
    }
    for (const float sign : {1.0f, -1.0f}) {
        std::vector<float> vertex(centre.size());
        for (std::size_t i = 0; i < vertex.size(); ++i) {
            const float g = sign * (gradientB[i] - gradientA[i]);
            vertex[i] = g > 0.0f ? box.upper[i] : g < 0.0f ? box.lower[i] : centre[i];
        }
        if (check(vertex)) {
            how = "vertex";
            return vertex;
        }
    }
    return std::nullopt;
}

}  // namespace

CertifyResult certifyEquivalence(const IRGraph& original, const IRGraph& candidate,
                                 const Box& root, const CertifyOptions& options) {
    const auto start = std::chrono::steady_clock::now();
    CertifyResult result;
    const Differ differ(original, candidate);
    result.aligned = differ.aligned();
    const std::vector<double> influence = differ.influence();
    const double rootVolume = root.logVolume();
    bool allExact = true;
    // Largest boxes first: certified volume grows fastest.
    std::vector<Leaf> queue = {{root, 0, rootVolume}};
    const auto larger = [](const Leaf& a, const Leaf& b) { return a.logVolume < b.logVolume; };
    while (!queue.empty()) {
        std::pop_heap(queue.begin(), queue.end(), larger);
        Leaf leaf = std::move(queue.back());
        queue.pop_back();
        ++result.boxes;
        const DifferenceBounds bounds = differ.run(leaf.box, differ.original().outputName);
        if (result.boxes == 1) {
            result.sources = differ.run(leaf.box, differ.original().outputName, nullptr, nullptr, true)
                                 .sources.size();
        }
        const bool exact = bounds.complete && bounds.allExact();
        const double bound = bounds.complete ? bounds.maxAbs() : std::numeric_limits<double>::infinity();
        if (exact || bound <= options.tolerance) {
            ++result.certifiedBoxes;
            result.bound = std::max(result.bound, exact ? 0.0 : bound);
            allExact = allExact && exact;
            result.certifiedFraction += std::exp(leaf.logVolume - rootVolume);
            continue;
        }
        if (options.centreChecks && result.evaluations < options.maxEvaluations) {
            const auto centre = leaf.box.centre();
            ++result.evaluations;
            if (exposes(original, candidate, centre, options.tolerance)) {
                result.verdict = Verdict::Refuted;
                result.witness = centre;
                result.refutedBy = "centre";
                break;
            }
        }
        if (options.guidedWitnesses && result.evaluations < options.maxEvaluations) {
            std::string how;
            if (auto witness = guidedWitness(differ, original, candidate, leaf.box, options,
                                             result.evaluations, how)) {
                result.verdict = Verdict::Refuted;
                result.witness = std::move(witness);
                result.refutedBy = how;
                break;
            }
        }
        if (result.boxes + queue.size() + 2 > options.maxBoxes || leaf.depth >= options.maxDepth) {
            ++result.openBoxes;
            continue;
        }
        const std::size_t d = splitDimension(leaf.box, influence);
        const float middle = 0.5f * (leaf.box.lower[d] + leaf.box.upper[d]);
        if (!(middle > leaf.box.lower[d] && middle < leaf.box.upper[d])) {
            ++result.openBoxes;
            continue;
        }
        Leaf low = leaf, high = std::move(leaf);
        low.box.upper[d] = middle;
        high.box.lower[d] = middle;
        low.depth = high.depth = low.depth + 1;
        low.logVolume = low.box.logVolume();
        high.logVolume = high.box.logVolume();
        queue.push_back(std::move(low));
        std::push_heap(queue.begin(), queue.end(), larger);
        queue.push_back(std::move(high));
        std::push_heap(queue.begin(), queue.end(), larger);
    }
    result.openBoxes += queue.size();
    if (result.verdict != Verdict::Refuted && result.openBoxes == 0) {
        result.verdict = Verdict::Certified;
        result.kind = allExact ? CertificateKind::Exact : CertificateKind::Bounded;
        result.certifiedFraction = 1.0;
    }
    result.certifiedFraction = std::min(1.0, result.certifiedFraction);
    result.milliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return result;
}

namespace {

// Lower bound of z_c - z_j over the box from symbolic rows (or intervals).
double marginLowerBound(const ValueBounds& logits, std::size_t c, std::size_t j, const Box& box) {
    double bound = logits.lower[c] - logits.upper[j];
    if (logits.symbolic) {
        const std::size_t width = box.size() + 1;
        std::vector<double> row(width);
        for (std::size_t i = 0; i < width; ++i) {
            row[i] = logits.symLower[c * width + i] - logits.symUpper[j * width + i];
        }
        Rows rows{width, &box};
        bound = std::max(bound, rows.minimum(row.data()));
    }
    return bound;
}

int agreeOnBox(const IRGraph& original, const IRGraph& candidate, const Differ& differ,
               const std::string& logits, bool sigmoidOutput, const Box& box) {
    DiagnosticEngine diagnostics;
    const auto centreOutput = executeIR(original, box.centre(), diagnostics);
    if (!centreOutput) return -1;
    const std::size_t c = argmaxOf(*centreOutput);
    GraphBounds ta, tb;
    const DifferenceBounds delta = differ.run(box, logits, &ta, &tb);
    const ValueBounds* za = ta.find(logits);
    const std::string mapped = differ.mapped(logits);
    const ValueBounds* zb = tb.find(mapped.empty() ? logits : mapped);
    if (za == nullptr || zb == nullptr || !delta.complete || delta.lower.size() != za->lower.size()) {
        return -1;
    }
    (void)candidate;
    (void)original;
    if (sigmoidOutput) {
        const bool positive = c == 1;
        const double a = positive ? za->lower[0] : -za->upper[0];
        const double shift = positive ? delta.lower[0] : -delta.upper[0];
        const double b = positive ? zb->lower[0] : -zb->upper[0];
        return a > 0.0 && std::max(b, a + shift) > 0.0 ? static_cast<int>(c) : -1;
    }
    for (std::size_t j = 0; j < za->lower.size(); ++j) {
        if (j == c) continue;
        const double original = marginLowerBound(*za, c, j, box);
        if (!(original > 0.0)) return -1;
        const double viaDelta = original + (delta.lower[c] - delta.upper[j]);
        const double direct = marginLowerBound(*zb, c, j, box);
        if (!(std::max(viaDelta, direct) > 0.0)) return -1;
    }
    return static_cast<int>(c);
}

}  // namespace

namespace {

int agreeWithSplits(const IRGraph& original, const IRGraph& candidate, const Differ& differ,
                    const std::string& logits, bool sigmoidOutput,
                    const std::vector<double>& influence, const Box& box, std::size_t maxBoxes) {
    std::vector<Box> pending = {box};
    std::size_t analysed = 0;
    int agreed = -2;
    while (!pending.empty()) {
        Box current = std::move(pending.back());
        pending.pop_back();
        ++analysed;
        const int c = agreeOnBox(original, candidate, differ, logits, sigmoidOutput, current);
        if (c >= 0 && (agreed == -2 || agreed == c)) {
            agreed = c;
            continue;
        }
        if (c >= 0 || analysed + pending.size() + 2 > maxBoxes) return -1;
        const std::size_t d = splitDimension(current, influence);
        const float middle = 0.5f * (current.lower[d] + current.upper[d]);
        if (!(middle > current.lower[d] && middle < current.upper[d])) return -1;
        Box high = current;
        current.upper[d] = middle;
        high.lower[d] = middle;
        pending.push_back(std::move(current));
        pending.push_back(std::move(high));
    }
    return agreed < 0 ? -1 : agreed;
}

}  // namespace

int certifyAgreement(const IRGraph& original, const IRGraph& candidate, const Box& box,
                     std::size_t maxBoxes) {
    const Differ differ(original, candidate);
    bool sigmoidOutput = false;
    const std::string logits = logitsName(differ.original(), sigmoidOutput);
    return agreeWithSplits(original, candidate, differ, logits, sigmoidOutput, differ.influence(),
                           box, maxBoxes);
}

std::vector<float> certifiedAgreementRadii(const IRGraph& original, const IRGraph& candidate,
                                           const std::vector<std::vector<float>>& points,
                                           float maxRadius, std::size_t iterations,
                                           std::size_t maxBoxes) {
    const Differ differ(original, candidate);
    bool sigmoidOutput = false;
    const std::string logits = logitsName(differ.original(), sigmoidOutput);
    const std::vector<double> influence = differ.influence();
    std::vector<float> radii;
    radii.reserve(points.size());
    for (const auto& point : points) {
        const auto agrees = [&](float radius) {
            return agreeWithSplits(original, candidate, differ, logits, sigmoidOutput, influence,
                                   ballAround(point, radius), maxBoxes) >= 0;
        };
        float low = 0.0f, high = maxRadius;
        if (agrees(high)) {
            radii.push_back(high);
            continue;
        }
        for (std::size_t step = 0; step < iterations; ++step) {
            const float middle = 0.5f * (low + high);
            (agrees(middle) ? low : high) = middle;
        }
        radii.push_back(low);
    }
    return radii;
}

SeverityCertificate certifySeverity(const IRGraph& original, const IRGraph& candidate,
                                    const std::vector<std::vector<float>>& centres, float radius,
                                    std::size_t maxBoxesPerBall) {
    const auto start = std::chrono::steady_clock::now();
    SeverityCertificate result;
    result.bound = 0.0;
    const Differ differ(original, candidate);
    bool sigmoidOutput = false;
    const std::string logits = logitsName(differ.original(), sigmoidOutput);
    const std::string output = differ.original().outputName;
    const std::vector<double> influence = differ.influence();
    struct Piece {
        Box box;
        double bound = 1.0;
        double deviation = 0.0;
    };
    const auto analyse = [&](Piece& piece) {
        ++result.boxes;
        if (agreeOnBox(original, candidate, differ, logits, sigmoidOutput, piece.box) >= 0) {
            piece.bound = 0.0;
            return;
        }
        const DifferenceBounds delta = differ.run(piece.box, output);
        double epsilon = delta.complete ? delta.maxAbs() : 1.0;
        if (sigmoidOutput) epsilon *= 2.0;  // two-class margin from one probability
        piece.deviation = std::min(1.0, epsilon);
        piece.bound = std::min(1.0, 2.0 * epsilon);
    };
    for (const auto& centre : centres) {
        std::vector<Piece> pieces(1);
        pieces[0].box = ballAround(centre, radius);
        analyse(pieces[0]);
        std::size_t used = 1;
        while (used + 2 <= maxBoxesPerBall) {
            const auto worst = std::max_element(
                pieces.begin(), pieces.end(),
                [](const Piece& a, const Piece& b) { return a.bound < b.bound; });
            if (worst->bound <= 0.0) break;
            Piece low = *worst;
            const std::size_t d = splitDimension(low.box, influence);
            const float middle = 0.5f * (low.box.lower[d] + low.box.upper[d]);
            if (!(middle > low.box.lower[d] && middle < low.box.upper[d])) break;
            Piece high = low;
            low.box.upper[d] = middle;
            high.box.lower[d] = middle;
            analyse(low);
            analyse(high);
            *worst = std::move(low);
            pieces.push_back(std::move(high));
            used += 2;
        }
        double local = 0.0;
        for (const Piece& piece : pieces) {
            local = std::max(local, piece.bound);
            result.maxDeviation = std::max(result.maxDeviation, piece.deviation);
        }
        ++result.balls;
        result.agreeingBalls += local == 0.0 ? 1 : 0;
        result.perBall.push_back(local);
        result.bound = std::max(result.bound, local);
    }
    result.milliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return result;
}

std::string logitsValue(const IRGraph& canonical, bool* sigmoidOutput) {
    bool sigmoid = false;
    std::string name = logitsName(canonical, sigmoid);
    if (sigmoidOutput != nullptr) *sigmoidOutput = sigmoid;
    return name;
}

std::size_t predictedClass(const std::vector<float>& scores) { return argmaxOf(scores); }

}  // namespace modelforge

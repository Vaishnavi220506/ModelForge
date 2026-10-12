#include "qnet.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unordered_map>

namespace modelforge {
namespace {

const IRInstruction* producerOf(const IRGraph& graph, const std::string& value) {
    for (const auto& instruction : graph.instructions) {
        if (instruction.output == value && instruction.operation != IROp::Return) {
            return &instruction;
        }
    }
    return nullptr;
}

const TensorInfo* constantOf(const IRGraph& graph, const std::string& name) {
    const auto found = graph.constants.find(name);
    return found == graph.constants.end() ? nullptr : &found->second;
}

// Weight constant as out x in, row-major.
bool readWeight(const TensorInfo& tensor, bool transposed, DenseLayer& layer, std::string& error) {
    if (tensor.shape.size() != 2) {
        error = "weight '" + tensor.name + "' must be two-dimensional";
        return false;
    }
    const auto rows = static_cast<std::size_t>(tensor.shape[0]);
    const auto columns = static_cast<std::size_t>(tensor.shape[1]);
    layer.in = transposed ? columns : rows;
    layer.out = transposed ? rows : columns;
    layer.weight.assign(layer.in * layer.out, 0.0);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < columns; ++c) {
            const double value = tensor.data[r * columns + c];
            const std::size_t o = transposed ? r : c;
            const std::size_t i = transposed ? c : r;
            layer.weight[o * layer.in + i] = value;
        }
    }
    layer.bias.assign(layer.out, 0.0);
    return true;
}

bool readBias(const TensorInfo& tensor, DenseLayer& layer, std::string& error) {
    if (elementCount(tensor.shape) != layer.out) {
        error = "bias '" + tensor.name + "' does not match the layer width";
        return false;
    }
    for (std::size_t o = 0; o < layer.out; ++o) layer.bias[o] += tensor.data[o];
    return true;
}

std::int64_t signedMax(int bits) { return (std::int64_t{1} << (bits - 1)) - 1; }

const char* cType(int bits, bool isUnsigned) {
    if (bits <= 8) return isUnsigned ? "uint8_t" : "int8_t";
    if (bits <= 16) return isUnsigned ? "uint16_t" : "int16_t";
    if (bits <= 32) return isUnsigned ? "uint32_t" : "int32_t";
    return isUnsigned ? "uint64_t" : "int64_t";
}

int storageBits(std::int64_t lo, std::int64_t hi) {
    for (int bits : {8, 16, 32}) {
        if (lo >= -(std::int64_t{1} << (bits - 1)) && hi <= signedMax(bits)) return bits;
    }
    return 64;
}

// Decimal literals that round-trip exactly (17 digits for double, 9 for float);
// hex-float literals are not accepted by every C compiler.
std::string exactDouble(double value) {
    std::ostringstream text;
    text << std::setprecision(17) << value;
    return text.str();
}

std::string exactFloat(float value) {
    std::ostringstream text;
    text << std::setprecision(9) << value;
    std::string literal = text.str();
    if (literal.find_first_of(".e") == std::string::npos) literal += ".0";
    return literal;
}

template <typename T>
void writeArray(std::ostream& out, const std::string& type, const std::string& name,
                const std::vector<T>& values) {
    out << "static const " << type << " " << name << "[" << values.size() << "] = {";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i % 16 == 0) out << "\n   ";
        out << " " << static_cast<long long>(values[i]) << (i + 1 < values.size() ? "," : "");
    }
    out << "\n};\n";
}

}  // namespace

std::optional<DenseNetwork> extractDenseNetwork(const IRGraph& graph, std::string& error) {
    DenseNetwork network;
    network.name = graph.name;
    const IRInstruction* returned = nullptr;
    for (const auto& instruction : graph.instructions) {
        if (instruction.operation == IROp::Return) returned = &instruction;
    }
    if (returned == nullptr || returned->inputs.empty()) {
        error = "the graph has no return value";
        return std::nullopt;
    }
    std::string current = returned->inputs.front();
    network.outputActivation = "none";
    std::vector<DenseLayer> reversed;
    bool pendingRelu = false;
    bool sawLayer = false;
    while (current != graph.inputName) {
        const IRInstruction* instruction = producerOf(graph, current);
        if (instruction == nullptr || instruction->inputs.empty()) {
            error = "value '" + current + "' has no producer";
            return std::nullopt;
        }
        switch (instruction->operation) {
            case IROp::Softmax:
            case IROp::Sigmoid:
                if (sawLayer || pendingRelu || network.outputActivation != "none") {
                    error = "Softmax/Sigmoid is supported only as the output activation";
                    return std::nullopt;
                }
                network.outputActivation =
                    instruction->operation == IROp::Softmax ? "softmax" : "sigmoid";
                current = instruction->inputs[0];
                break;
            case IROp::Relu:
                if (!sawLayer) {
                    error = "a ReLU on the logits is not supported by the integer target";
                    return std::nullopt;
                }
                if (pendingRelu) {
                    error = "two consecutive ReLUs";
                    return std::nullopt;
                }
                pendingRelu = true;
                current = instruction->inputs[0];
                break;
            case IROp::Gemm:
            case IROp::FusedGemmRelu: {
                if (instruction->transA || instruction->inputs.size() < 2) {
                    error = "unsupported Gemm form in '" + instruction->nodeName + "'";
                    return std::nullopt;
                }
                const TensorInfo* weight = constantOf(graph, instruction->inputs[1]);
                if (weight == nullptr) {
                    error = "Gemm '" + instruction->nodeName + "' needs a constant weight";
                    return std::nullopt;
                }
                DenseLayer layer;
                layer.name = instruction->nodeName;
                if (!readWeight(*weight, instruction->transB, layer, error)) return std::nullopt;
                if (instruction->inputs.size() == 3) {
                    const TensorInfo* bias = constantOf(graph, instruction->inputs[2]);
                    if (bias == nullptr || !readBias(*bias, layer, error)) {
                        if (error.empty()) error = "Gemm bias must be constant";
                        return std::nullopt;
                    }
                }
                const bool fused = instruction->operation == IROp::FusedGemmRelu;
                if (fused && pendingRelu) {
                    error = "two consecutive ReLUs";
                    return std::nullopt;
                }
                layer.relu = pendingRelu || fused;
                if (!sawLayer && layer.relu) {
                    error = "a ReLU on the logits is not supported by the integer target";
                    return std::nullopt;
                }
                pendingRelu = false;
                sawLayer = true;
                reversed.push_back(std::move(layer));
                current = instruction->inputs[0];
                break;
            }
            case IROp::Add: {
                // MatMul followed by a constant Add is a dense layer with bias.
                std::string other;
                const TensorInfo* bias = nullptr;
                for (const auto& name : instruction->inputs) {
                    if (const auto* constant = constantOf(graph, name)) {
                        bias = constant;
                    } else {
                        other = name;
                    }
                }
                const IRInstruction* matmul = other.empty() ? nullptr : producerOf(graph, other);
                if (bias == nullptr || matmul == nullptr || matmul->operation != IROp::MatMul ||
                    matmul->inputs.size() != 2) {
                    error = "Add is supported only as the bias of a MatMul";
                    return std::nullopt;
                }
                const TensorInfo* weight = constantOf(graph, matmul->inputs[1]);
                if (weight == nullptr) {
                    error = "MatMul '" + matmul->nodeName + "' needs a constant weight";
                    return std::nullopt;
                }
                DenseLayer layer;
                layer.name = matmul->nodeName;
                if (!readWeight(*weight, false, layer, error) || !readBias(*bias, layer, error)) {
                    return std::nullopt;
                }
                layer.relu = pendingRelu;
                pendingRelu = false;
                sawLayer = true;
                reversed.push_back(std::move(layer));
                current = matmul->inputs[0];
                break;
            }
            case IROp::MatMul: {
                const TensorInfo* weight = constantOf(graph, instruction->inputs[1]);
                if (weight == nullptr || instruction->inputs.size() != 2) {
                    error = "MatMul '" + instruction->nodeName + "' needs a constant weight";
                    return std::nullopt;
                }
                DenseLayer layer;
                layer.name = instruction->nodeName;
                if (!readWeight(*weight, false, layer, error)) return std::nullopt;
                layer.relu = pendingRelu;
                pendingRelu = false;
                sawLayer = true;
                reversed.push_back(std::move(layer));
                current = instruction->inputs[0];
                break;
            }
            default:
                error = "operator " + irOpName(instruction->operation) +
                        " is not supported by the integer target";
                return std::nullopt;
        }
    }
    if (reversed.empty() || pendingRelu) {
        error = reversed.empty() ? "the model has no dense layer" : "a ReLU directly on the input";
        return std::nullopt;
    }
    network.layers.assign(reversed.rbegin(), reversed.rend());
    for (std::size_t l = 1; l < network.layers.size(); ++l) {
        if (network.layers[l].in != network.layers[l - 1].out) {
            error = "layer widths do not chain at '" + network.layers[l].name + "'";
            return std::nullopt;
        }
    }
    network.inputs = network.layers.front().in;
    auto& head = network.layers.back();
    if (head.out == 1) {
        // Decision of a single logit z is z > 0: compare against a constant 0 logit.
        std::vector<double> weight(2 * head.in, 0.0);
        std::copy(head.weight.begin(), head.weight.end(), weight.begin() + head.in);
        head.weight = std::move(weight);
        head.bias = {0.0, head.bias[0]};
        head.out = 2;
        network.singleLogitPadded = true;
    }
    network.classes = head.out;
    return network;
}

std::vector<double> referenceLogits(const DenseNetwork& network, const std::vector<double>& x) {
    std::vector<double> current = x;
    for (const auto& layer : network.layers) {
        std::vector<double> next(layer.out);
        for (std::size_t o = 0; o < layer.out; ++o) {
            double sum = layer.bias[o];
            for (std::size_t i = 0; i < layer.in; ++i) sum += layer.weight[o * layer.in + i] * current[i];
            next[o] = layer.relu ? std::max(0.0, sum) : sum;
        }
        current = std::move(next);
    }
    return current;
}

int argmaxFirst(const std::vector<double>& values) {
    return values.empty() ? -1
                          : static_cast<int>(std::max_element(values.begin(), values.end()) - values.begin());
}

int argmaxFirst(const std::vector<std::int64_t>& values) {
    return values.empty() ? -1
                          : static_cast<int>(std::max_element(values.begin(), values.end()) - values.begin());
}

bool Box::contains(const std::vector<double>& x) const {
    if (x.size() != lo.size()) return false;
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (x[i] < lo[i] || x[i] > hi[i]) return false;
    }
    return true;
}

std::vector<double> Box::center() const {
    std::vector<double> c(lo.size());
    for (std::size_t i = 0; i < lo.size(); ++i) c[i] = 0.5 * (lo[i] + hi[i]);
    return c;
}

std::string PrecisionConfig::describe() const {
    std::ostringstream text;
    text << "w[";
    for (std::size_t i = 0; i < weightBits.size(); ++i) text << (i ? "," : "") << weightBits[i];
    text << "] a[";
    for (std::size_t i = 0; i < activationBits.size(); ++i) text << (i ? "," : "") << activationBits[i];
    text << "]";
    return text.str();
}

PrecisionConfig uniformConfig(std::size_t layers, int weightBits, int activationBits) {
    PrecisionConfig config;
    config.weightBits.assign(layers, weightBits);
    config.activationBits.assign(layers, activationBits);
    return config;
}

double inputQuantisationSlack(int inputBits) {
    // |x * inv_s - x / s| and the rounding of t +/- 0.5 in float32, in LSBs.
    const double qmax = static_cast<double>(signedMax(inputBits));
    return 0.5 + 4.0 * (qmax + 1.0) * std::ldexp(1.0, -24);
}

std::vector<std::int64_t> quantizeInput(const QProgram& program, const std::vector<double>& x) {
    std::vector<std::int64_t> q(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float inverse = static_cast<float>(1.0 / program.inputScale[i]);
        const float t = static_cast<float>(x[i]) * inverse;
        const float r = t >= 0.0f ? t + 0.5f : t - 0.5f;
        long long v = static_cast<long long>(std::max(-1.0e9f, std::min(1.0e9f, r)));
        v = std::max<long long>(program.inputMin, std::min<long long>(program.inputMax, v));
        q[i] = v;
    }
    return q;
}

std::int64_t roundShift(std::int64_t value, int shift) {
    const std::int64_t half = std::int64_t{1} << (shift - 1);
    const std::int64_t u = value + half;
    if (u >= 0) return u >> shift;
    return -((-u + ((std::int64_t{1} << shift) - 1)) >> shift);
}

IntegerRun runInteger(const QProgram& program, const std::vector<std::int64_t>& qInput) {
    IntegerRun run;
    std::vector<std::int64_t> current = qInput;
    for (const auto& layer : program.layers) {
        const std::int64_t accLimit = signedMax(layer.accBits);
        std::vector<std::int64_t> next(layer.out);
        for (std::size_t o = 0; o < layer.out; ++o) {
            std::int64_t acc = layer.qBias[o];
            if (std::llabs(acc) > accLimit) run.accumulatorOverflow = true;
            for (std::size_t i = 0; i < layer.in; ++i) {
                acc += static_cast<std::int64_t>(layer.qWeight[o * layer.in + i]) * current[i];
                if (acc > accLimit || acc < -accLimit - 1) run.accumulatorOverflow = true;
            }
            if (layer.relu && acc < 0) acc = 0;
            std::int64_t q = roundShift(acc * layer.multiplier[o], layer.shift[o]);
            if (q > layer.outMax || q < layer.outMin) {
                run.saturated = true;
                q = std::max(layer.outMin, std::min(layer.outMax, q));
            }
            next[o] = q;
        }
        current = std::move(next);
    }
    run.logits = current;
    run.decision = argmaxFirst(current);
    return run;
}

IntegerRun runInteger(const QProgram& program, const std::vector<double>& x) {
    return runInteger(program, quantizeInput(program, x));
}

std::vector<double> dequantizedLogits(const QProgram& program, const IntegerRun& run) {
    std::vector<double> logits(run.logits.size());
    const auto& scale = program.layers.back().outScale;
    for (std::size_t i = 0; i < logits.size(); ++i) logits[i] = scale[i] * static_cast<double>(run.logits[i]);
    return logits;
}

ProgramCost programCost(const QProgram& program) {
    ProgramCost cost;
    std::size_t largestPair = program.inputs * static_cast<std::size_t>(program.inputBits <= 8 ? 1 : 2);
    std::size_t previous = largestPair;
    for (const auto& layer : program.layers) {
        const double count = static_cast<double>(layer.qWeight.size());
        const double perWeight = layer.weightBits <= 4 ? 0.5 : layer.weightBits <= 8 ? 1.0 : 2.0;
        cost.weightBytes += std::ceil(count * perWeight);
        cost.idealBits += count * layer.weightBits;
        std::int64_t lo = 0, hi = 0;
        for (auto b : layer.qBias) {
            lo = std::min(lo, b);
            hi = std::max(hi, b);
        }
        cost.flashBytes += static_cast<double>(layer.out) * (storageBits(lo, hi) / 8.0 + 4.0 + 1.0);
        if (&layer == &program.layers.front()) cost.flashBytes += 4.0 * layer.in;  // input scales
        const std::size_t outBytes = layer.out * static_cast<std::size_t>(layer.outBits <= 8 ? 1 : 2);
        largestPair = std::max(largestPair, previous + outBytes);
        previous = outBytes;
    }
    cost.flashBytes += cost.weightBytes;
    cost.ramBytes = static_cast<double>(largestPair);
    return cost;
}

double float32Bytes(const DenseNetwork& network) {
    double bytes = 0;
    for (const auto& layer : network.layers) bytes += 4.0 * (layer.weight.size() + layer.bias.size());
    return bytes;
}

bool emitIntegerC(const QProgram& program, const std::string& directory, std::string& error) {
    namespace fs = std::filesystem;
    std::error_code code;
    fs::create_directories(directory, code);
    if (code) {
        error = "cannot create " + directory;
        return false;
    }
    const fs::path out = directory;
    const char* inType = cType(program.inputBits, false);
    const auto& head = program.layers.back();
    const char* logitType = cType(head.outBits, false);

    std::ofstream header(out / "model_q.h");
    header << "/* Integer-only inference code generated by ModelForge / ForgeCert.\n"
           << " * Model: " << program.name << "   precision: " << program.config.describe() << "\n"
           << " * No floating point after mf_quantize_input(). See certificate.json. */\n"
           << "#ifndef MF_MODEL_Q_H\n#define MF_MODEL_Q_H\n\n#include <stdint.h>\n\n"
           << "#define MF_INPUTS " << program.inputs << "\n#define MF_CLASSES " << program.classes
           << "\n\ntypedef " << inType << " mf_input_t;\ntypedef " << logitType
           << " mf_logit_t;\n\n"
           << "/* Quantises float features (the certified input domain is in certificate.json). */\n"
           << "void mf_quantize_input(const float *x, mf_input_t *q);\n"
           << "/* Integer-only inference; returns the predicted class. logits may be NULL. */\n"
           << "int mf_infer_q(const mf_input_t *q, mf_logit_t *logits);\n"
           << "/* Convenience: quantise, infer, and optionally dequantise the logits. */\n"
           << "int mf_predict(const float *x, float *logits);\n\n#endif\n";

    std::ofstream source(out / "model_q.c");
    source << "/* Integer-only inference code generated by ModelForge / ForgeCert.\n"
           << " * MF-ARTIFACT v1. The arrays below are read verbatim by modelforge_certcheck;\n"
           << " * do not edit them by hand (the certificate would no longer apply). */\n"
           << "#include \"model_q.h\"\n\n";
    // MF_META: version, inputs, classes, input bits, layer count, then per layer
    // in, out, relu, weight bits, accumulator bits, output bits, output unsigned.
    std::vector<std::int64_t> meta{1, static_cast<std::int64_t>(program.inputs),
                                   static_cast<std::int64_t>(program.classes), program.inputBits,
                                   static_cast<std::int64_t>(program.layers.size())};
    for (const auto& layer : program.layers) {
        meta.insert(meta.end(), {static_cast<std::int64_t>(layer.in), static_cast<std::int64_t>(layer.out),
                                 layer.relu ? 1 : 0, layer.weightBits, layer.accBits, layer.outBits,
                                 layer.outUnsigned ? 1 : 0});
    }
    source << "#ifdef MF_CHECKER_METADATA\n";
    writeArray(source, "int32_t", "MF_META", meta);
    // Tensor scales: the real value of integer element q of channel k is scale[k] * q.
    // Only the checker and the optional dequantisation of the logits use them.
    const auto writeScales = [&](const std::string& name, const std::vector<double>& values) {
        source << "static const double " << name << "[" << values.size() << "] = {";
        for (std::size_t i = 0; i < values.size(); ++i) {
            source << (i % 4 == 0 ? "\n   " : "") << " " << exactDouble(values[i]) << (i + 1 < values.size() ? "," : "");
        }
        source << "\n};\n";
    };
    writeScales("MF_SCALE_IN", program.inputScale);
    for (std::size_t l = 0; l + 1 < program.layers.size(); ++l) {
        writeScales("MF_SCALE_" + std::to_string(l), program.layers[l].outScale);
    }
    source << "#endif\n";
    writeScales("MF_SCALE_" + std::to_string(program.layers.size() - 1), program.layers.back().outScale);
    source << "static const float MF_INPUT_INV_SCALE[" << program.inputs << "] = {";
    for (std::size_t i = 0; i < program.inputs; ++i) {
        source << (i % 4 == 0 ? "\n   " : "") << " " << exactFloat(static_cast<float>(1.0 / program.inputScale[i])) << "f"
               << (i + 1 < program.inputs ? "," : "");
    }
    source << "\n};\n\n";
    source << "static int64_t mf_round_shift(int64_t t, int s) {\n"
           << "    int64_t u = t + ((int64_t)1 << (s - 1));\n"
           << "    return u >= 0 ? (u >> s) : -((-u + (((int64_t)1 << s) - 1)) >> s);\n}\n\n";
    const bool anyPacked = std::any_of(program.layers.begin(), program.layers.end(),
                                       [](const QLayer& layer) { return layer.weightBits <= 4; });
    if (anyPacked) {
        source << "static int8_t mf_nibble(const uint8_t *packed, int k) {\n"
               << "    int v = (k & 1) ? (packed[k >> 1] >> 4) : (packed[k >> 1] & 15);\n"
               << "    return (int8_t)(v >= 8 ? v - 16 : v);\n}\n\n";
    }

    for (std::size_t l = 0; l < program.layers.size(); ++l) {
        const auto& layer = program.layers[l];
        const std::string id = std::to_string(l);
        const bool packed = layer.weightBits <= 4;
        if (packed) {
            std::vector<std::int64_t> bytes((layer.qWeight.size() + 1) / 2, 0);
            for (std::size_t k = 0; k < layer.qWeight.size(); ++k) {
                const std::int64_t nibble = layer.qWeight[k] & 15;
                bytes[k / 2] |= (k % 2) ? (nibble << 4) : nibble;
            }
            writeArray(source, "uint8_t", "MF_W" + id, bytes);
        } else {
            writeArray(source, cType(layer.weightBits, false), "MF_W" + id, layer.qWeight);
        }
        std::int64_t lo = 0, hi = 0;
        for (auto b : layer.qBias) {
            lo = std::min(lo, b);
            hi = std::max(hi, b);
        }
        const char* biasType = cType(std::max(storageBits(lo, hi), std::min(layer.accBits, 64)), false);
        writeArray(source, biasType, "MF_B" + id, layer.qBias);
        writeArray(source, "int32_t", "MF_M" + id, layer.multiplier);
        writeArray(source, "uint8_t", "MF_S" + id, layer.shift);

        const char* inT = l == 0 ? inType : cType(program.layers[l - 1].outBits, program.layers[l - 1].outUnsigned);
        const char* outT = cType(layer.outBits, layer.outUnsigned);
        const char* accT = cType(layer.accBits, false);
        source << "\n/* layer " << l << " (" << layer.in << " -> " << layer.out
               << (layer.relu ? ", ReLU" : "") << "): weights " << layer.weightBits << "-bit, "
               << layer.accBits << "-bit accumulator (no overflow on the domain: proved) */\n"
               << "static void mf_layer" << l << "(const " << inT << " *x, " << outT << " *y) {\n"
               << "    for (int o = 0; o < " << layer.out << "; ++o) {\n"
               << "        " << accT << " acc = (" << accT << ")MF_B" << l << "[o];\n"
               << "        for (int i = 0; i < " << layer.in << "; ++i) {\n";
        if (packed) {
            source << "            acc = (" << accT << ")(acc + (" << accT << ")mf_nibble(MF_W" << l
                   << ", o * " << layer.in << " + i) * (" << accT << ")x[i]);\n";
        } else {
            source << "            acc = (" << accT << ")(acc + (" << accT << ")MF_W" << l << "[o * "
                   << layer.in << " + i] * (" << accT << ")x[i]);\n";
        }
        source << "        }\n";
        if (layer.relu) source << "        if (acc < 0) acc = 0;\n";
        source << "        int64_t q = mf_round_shift((int64_t)acc * MF_M" << l << "[o], MF_S" << l
               << "[o]);\n"
               << "        if (q > " << layer.outMax << ") q = " << layer.outMax
               << ";  /* saturation: proved unreachable on the domain */\n"
               << "        if (q < " << layer.outMin << ") q = " << layer.outMin << ";\n"
               << "        y[o] = (" << outT << ")q;\n    }\n}\n";
    }

    source << "\nvoid mf_quantize_input(const float *x, mf_input_t *q) {\n"
           << "    for (int i = 0; i < MF_INPUTS; ++i) {\n"
           << "        float t = x[i] * MF_INPUT_INV_SCALE[i];\n"
           << "        float r = t >= 0.0f ? t + 0.5f : t - 0.5f;\n"
           << "        if (r > 1.0e9f) r = 1.0e9f;\n        if (r < -1.0e9f) r = -1.0e9f;\n"
           << "        long long v = (long long)r;\n"
           << "        if (v > " << program.inputMax << ") v = " << program.inputMax << ";\n"
           << "        if (v < " << program.inputMin << ") v = " << program.inputMin << ";\n"
           << "        q[i] = (mf_input_t)v;\n    }\n}\n\n"
           << "int mf_infer_q(const mf_input_t *q, mf_logit_t *logits) {\n";
    for (std::size_t l = 0; l + 1 < program.layers.size(); ++l) {
        const auto& layer = program.layers[l];
        source << "    " << cType(layer.outBits, layer.outUnsigned) << " t" << l << "[" << layer.out << "];\n";
    }
    source << "    mf_logit_t out[MF_CLASSES];\n";
    for (std::size_t l = 0; l < program.layers.size(); ++l) {
        const std::string in = l == 0 ? "q" : "t" + std::to_string(l - 1);
        const std::string dst = l + 1 == program.layers.size() ? "out" : "t" + std::to_string(l);
        source << "    mf_layer" << l << "(" << in << ", " << dst << ");\n";
    }
    source << "    int best = 0;\n"
           << "    for (int c = 0; c < MF_CLASSES; ++c) {\n"
           << "        if (logits) logits[c] = out[c];\n"
           << "        if (out[c] > out[best]) best = c;\n    }\n    return best;\n}\n\n"
           << "int mf_predict(const float *x, float *logits) {\n"
           << "    mf_input_t q[MF_INPUTS];\n    mf_logit_t raw[MF_CLASSES];\n"
           << "    mf_quantize_input(x, q);\n    int best = mf_infer_q(q, raw);\n"
           << "    if (logits) {\n        for (int c = 0; c < MF_CLASSES; ++c) logits[c] = (float)(MF_SCALE_"
           << program.layers.size() - 1 << "[c] * raw[c]);\n    }\n    return best;\n}\n";

    std::ofstream mainFile(out / "main_q.c");
    mainFile << "/* Command-line driver for the integer-only model (ModelForge / ForgeCert).\n"
             << " *   generated_model_q f1 f2 ... fN     one input, prints class and logits\n"
             << " *   generated_model_q --csv rows.csv   one prediction per row (extra columns ignored)\n"
             << " *   generated_model_q --raw rows.csv   'row,class,int logits...' (bit-exact check) */\n"
             << "#include \"model_q.h\"\n\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n\n"
             << "static int run_csv(const char *path, int raw) {\n"
             << "    FILE *file = fopen(path, \"r\");\n    char line[65536];\n    int row = 0;\n"
             << "    if (!file) { fprintf(stderr, \"cannot open %s\\n\", path); return 1; }\n"
             << "    while (fgets(line, sizeof line, file)) {\n"
             << "        float x[MF_INPUTS];\n        mf_input_t q[MF_INPUTS];\n        mf_logit_t logits[MF_CLASSES];\n"
             << "        char *cursor = line;\n        int n = 0, best;\n"
             << "        while (n < MF_INPUTS) {\n            char *end;\n"
             << "            x[n] = strtof(cursor, &end);\n            if (end == cursor) break;\n"
             << "            ++n;\n            cursor = end;\n            while (*cursor == ',' || *cursor == ' ') ++cursor;\n        }\n"
             << "        if (n == 0) continue;\n"
             << "        if (n < MF_INPUTS) { fprintf(stderr, \"row %d: expected %d features\\n\", row + 1, MF_INPUTS); fclose(file); return 1; }\n"
             << "        mf_quantize_input(x, q);\n        best = mf_infer_q(q, logits);\n"
             << "        printf(\"%d,%d\", row, best);\n"
             << "        if (raw) { for (int c = 0; c < MF_CLASSES; ++c) printf(\",%ld\", (long)logits[c]); }\n"
             << "        printf(\"\\n\");\n        ++row;\n    }\n    fclose(file);\n    return 0;\n}\n\n"
             << "int main(int argc, char **argv) {\n"
             << "    if (argc == 3 && (strcmp(argv[1], \"--csv\") == 0 || strcmp(argv[1], \"--raw\") == 0)) {\n"
             << "        return run_csv(argv[2], strcmp(argv[1], \"--raw\") == 0);\n    }\n"
             << "    float x[MF_INPUTS] = {0};\n    float logits[MF_CLASSES];\n"
             << "    if (argc != 1 && argc != MF_INPUTS + 1) { fprintf(stderr, \"expected %d features\\n\", MF_INPUTS); return 1; }\n"
             << "    for (int i = 1; i < argc; ++i) x[i - 1] = strtof(argv[i], NULL);\n"
             << "    int best = mf_predict(x, logits);\n    printf(\"Logits:\");\n"
             << "    for (int c = 0; c < MF_CLASSES; ++c) printf(\" %.6f\", logits[c]);\n"
             << "    printf(\"\\nPrediction: %d\\n\", best);\n    return 0;\n}\n";

    std::ofstream cmake(out / "CMakeLists.txt");
    cmake << "cmake_minimum_required(VERSION 3.16)\nproject(ModelForgeIntegerModel C)\n"
          << "set(CMAKE_C_STANDARD 99)\nset(CMAKE_C_STANDARD_REQUIRED ON)\n"
          << "add_executable(generated_model_q main_q.c model_q.c)\n";
    if (!header || !source || !mainFile || !cmake) {
        error = "failed to write the integer C sources";
        return false;
    }
    return true;
}

}  // namespace modelforge

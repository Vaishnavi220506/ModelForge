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
    if (elementCount(tensor.shape) != layer.channels()) {
        error = "bias '" + tensor.name + "' does not match the layer width";
        return false;
    }
    for (std::size_t o = 0; o < layer.channels(); ++o) layer.bias[o] += tensor.data[o];
    return true;
}

std::int64_t signedMax(int bits) { return (std::int64_t{1} << (bits - 1)) - 1; }

const char* cType(int bits, bool isUnsigned) {
    if (bits <= 8) return isUnsigned ? "uint8_t" : "int8_t";
    if (bits <= 16) return isUnsigned ? "uint16_t" : "int16_t";
    if (bits <= 32) return isUnsigned ? "uint32_t" : "int32_t";
    return isUnsigned ? "uint64_t" : "int64_t";
}

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
    return literal + "f";
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

void writeUnsignedArray(std::ostream& out, const std::string& type, const std::string& name,
                        const std::vector<unsigned long long>& values) {
    out << "static const " << type << " " << name << "[" << values.size() << "] = {";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i % 16 == 0) out << "\n   ";
        out << " " << values[i] << "u" << (i + 1 < values.size() ? "," : "");
    }
    out << "\n};\n";
}

// The code of one layer. tools/certcheck.cpp holds an independent copy of
// this template and requires the emitted file to contain exactly this text.
std::string layerFunction(const QProgram& program, std::size_t l) {
    const QLayer& q = program.layers[l];
    const char* inT = l == 0 ? cType(program.inputBits, program.inputUnsigned)
                             : cType(program.layers[l - 1].outBits, program.layers[l - 1].outUnsigned);
    const char* outT = cType(q.outBits, q.outUnsigned);
    const char* accT = cType(q.accBits, false);
    const char* uaccT = cType(q.accBits, true);
    const char* mulT = q.mulBits == 16 ? "int16_t" : "int32_t";
    const std::string id = std::to_string(l);
    const bool conv = q.kind == LayerKind::Conv;
    const std::string bias = conv ? "MF_B" + id + "[c]" : "MF_B" + id + "[o]";
    const std::string indent = conv ? "                " : "        ";
    const auto weightAt = [&](const std::string& index) {
        return q.weightBits <= 4 ? "mf_nibble(MF_W" + id + ", " + index + ")" : "MF_W" + id + "[" + index + "]";
    };
    const auto step = [&](const std::string& in, const std::string& weightIndex, const std::string& inputIndex) {
        const std::string product = "(" + std::string(mulT) + ")" + weightAt(weightIndex) + " * (" + mulT + ")x[" +
                                    inputIndex + "]";
        if (q.modular) {
            return in + "u = (" + uaccT + ")(u + (" + uaccT + ")(" + product + "));\n";
        }
        return in + "acc = (" + accT + ")(acc + (" + accT + ")(" + product + "));\n";
    };
    std::ostringstream t;
    t << "static void mf_layer" << l << "(const " << inT << " *x, " << outT << " *y) {\n";
    if (conv) {
        const auto& g = q.conv;
        t << "    for (int c = 0; c < " << g.outC << "; ++c) {\n"
          << "        for (int oy = 0; oy < " << g.outH << "; ++oy) {\n"
          << "            for (int ox = 0; ox < " << g.outW << "; ++ox) {\n";
    } else {
        t << "    for (int o = 0; o < " << q.out << "; ++o) {\n";
    }
    if (q.modular) {
        t << indent << uaccT << " u = " << bias << ";\n";
    } else {
        t << indent << accT << " acc = (" << accT << ")" << bias << ";\n";
    }
    if (conv) {
        const auto& g = q.conv;
        t << indent << "for (int ci = 0; ci < " << g.inC << "; ++ci) {\n"
          << indent << "    for (int ky = 0; ky < " << g.kH << "; ++ky) {\n"
          << indent << "        const int iy = oy * " << g.stride << " + ky - " << g.pad << ";\n"
          << indent << "        if (iy < 0 || iy >= " << g.inH << ") continue;\n"
          << indent << "        for (int kx = 0; kx < " << g.kW << "; ++kx) {\n"
          << indent << "            const int ix = ox * " << g.stride << " + kx - " << g.pad << ";\n"
          << indent << "            if (ix < 0 || ix >= " << g.inW << ") continue;\n"
          << step(indent + "            ",
                  "((c * " + std::to_string(g.inC) + " + ci) * " + std::to_string(g.kH) + " + ky) * " +
                      std::to_string(g.kW) + " + kx",
                  "(ci * " + std::to_string(g.inH) + " + iy) * " + std::to_string(g.inW) + " + ix")
          << indent << "        }\n" << indent << "    }\n" << indent << "}\n";
    } else {
        t << indent << "for (int i = 0; i < " << q.in << "; ++i) {\n"
          << step(indent + "    ", "o * " + std::to_string(q.in) + " + i", "i") << indent << "}\n";
    }
    if (q.modular) {
        t << indent << "int64_t acc = u >= " << (q.accBits == 16 ? "0x8000u" : "0x80000000u") << " ? (int64_t)u - "
          << (q.accBits == 16 ? "65536" : "4294967296LL") << " : (int64_t)u;\n";
    }
    if (q.relu) t << indent << "if (acc < 0) acc = 0;\n";
    const std::string channel = conv ? "c" : "o";
    t << indent << "int64_t q = mf_round_shift((int64_t)acc * MF_M" << l << "[" << channel << "], MF_S" << l << "["
      << channel << "]);\n"
      << indent << "if (q > " << q.outMax << ") q = " << q.outMax << ";  /* saturate (modelled by the certificate) */\n"
      << indent << "if (q < " << q.outMin << ") q = " << q.outMin << ";\n";
    if (conv) {
        const auto& g = q.conv;
        t << indent << "y[(c * " << g.outH << " + oy) * " << g.outW << " + ox] = (" << outT << ")q;\n"
          << "            }\n        }\n    }\n}\n";
    } else {
        t << indent << "y[o] = (" << outT << ")q;\n    }\n}\n";
    }
    return t.str();
}

}  // namespace

void buildTaps(DenseLayer& layer) {
    layer.taps.assign(layer.out, {});
    if (layer.kind == LayerKind::Dense) {
        for (std::size_t o = 0; o < layer.out; ++o) {
            layer.taps[o].reserve(layer.in);
            for (std::size_t i = 0; i < layer.in; ++i) {
                layer.taps[o].push_back({static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(o * layer.in + i)});
            }
        }
        return;
    }
    const auto& g = layer.conv;
    for (std::size_t c = 0; c < g.outC; ++c) {
        for (std::size_t oy = 0; oy < g.outH; ++oy) {
            for (std::size_t ox = 0; ox < g.outW; ++ox) {
                auto& taps = layer.taps[(c * g.outH + oy) * g.outW + ox];
                for (std::size_t ci = 0; ci < g.inC; ++ci) {
                    for (std::size_t ky = 0; ky < g.kH; ++ky) {
                        const long iy = static_cast<long>(oy * g.stride + ky) - static_cast<long>(g.pad);
                        if (iy < 0 || iy >= static_cast<long>(g.inH)) continue;
                        for (std::size_t kx = 0; kx < g.kW; ++kx) {
                            const long ix = static_cast<long>(ox * g.stride + kx) - static_cast<long>(g.pad);
                            if (ix < 0 || ix >= static_cast<long>(g.inW)) continue;
                            taps.push_back({static_cast<std::uint32_t>((ci * g.inH + static_cast<std::size_t>(iy)) * g.inW +
                                                                       static_cast<std::size_t>(ix)),
                                            static_cast<std::uint32_t>(((c * g.inC + ci) * g.kH + ky) * g.kW + kx)});
                        }
                    }
                }
            }
        }
    }
}

std::size_t DenseNetwork::inputGroupOf(std::size_t i) const {
    if (!layers.empty() && layers.front().kind == LayerKind::Conv) {
        const auto& g = layers.front().conv;
        return i / (g.inH * g.inW);
    }
    return i;
}

std::size_t DenseNetwork::parameterCount() const {
    std::size_t count = 0;
    for (const auto& layer : layers) count += layer.weight.size() + layer.bias.size();
    return count;
}

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
    const auto pushLayer = [&](DenseLayer layer, bool fusedRelu) -> bool {
        if (fusedRelu && pendingRelu) {
            error = "two consecutive ReLUs";
            return false;
        }
        layer.relu = pendingRelu || fusedRelu;
        if (!sawLayer && layer.relu) {
            error = "a ReLU on the logits is not supported by the integer target";
            return false;
        }
        pendingRelu = false;
        sawLayer = true;
        reversed.push_back(std::move(layer));
        return true;
    };
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
                network.outputActivation = instruction->operation == IROp::Softmax ? "softmax" : "sigmoid";
                current = instruction->inputs[0];
                break;
            case IROp::Flatten:
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
            case IROp::Conv: {
                const TensorInfo* weight = constantOf(graph, instruction->inputs[1]);
                const auto input = graph.values.find(instruction->inputs[0]);
                if (weight == nullptr || weight->shape.size() != 4 || input == graph.values.end() ||
                    input->second.shape.size() != 4) {
                    error = "Conv '" + instruction->nodeName + "' needs NCHW input and a constant weight";
                    return std::nullopt;
                }
                DenseLayer layer;
                layer.name = instruction->nodeName;
                layer.kind = LayerKind::Conv;
                auto& g = layer.conv;
                g.inC = static_cast<std::size_t>(input->second.shape[1]);
                g.inH = static_cast<std::size_t>(input->second.shape[2]);
                g.inW = static_cast<std::size_t>(input->second.shape[3]);
                g.outC = static_cast<std::size_t>(weight->shape[0]);
                g.kH = static_cast<std::size_t>(weight->shape[2]);
                g.kW = static_cast<std::size_t>(weight->shape[3]);
                g.stride = static_cast<std::size_t>(instruction->stride);
                g.pad = static_cast<std::size_t>(instruction->pad);
                g.outH = static_cast<std::size_t>(instruction->shape[2]);
                g.outW = static_cast<std::size_t>(instruction->shape[3]);
                layer.in = g.inC * g.inH * g.inW;
                layer.out = g.outC * g.outH * g.outW;
                layer.weight.assign(weight->data.begin(), weight->data.end());
                layer.bias.assign(g.outC, 0.0);
                if (instruction->inputs.size() == 3) {
                    const TensorInfo* bias = constantOf(graph, instruction->inputs[2]);
                    if (bias == nullptr || !readBias(*bias, layer, error)) {
                        if (error.empty()) error = "Conv bias must be constant";
                        return std::nullopt;
                    }
                }
                if (!pushLayer(std::move(layer), false)) return std::nullopt;
                current = instruction->inputs[0];
                break;
            }
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
                if (!pushLayer(std::move(layer), instruction->operation == IROp::FusedGemmRelu)) return std::nullopt;
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
                if (!pushLayer(std::move(layer), false)) return std::nullopt;
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
                if (!pushLayer(std::move(layer), false)) return std::nullopt;
                current = instruction->inputs[0];
                break;
            }
            default:
                error = "operator " + irOpName(instruction->operation) + " is not supported by the integer target";
                return std::nullopt;
        }
    }
    if (reversed.empty() || pendingRelu) {
        error = reversed.empty() ? "the model has no dense layer" : "a ReLU directly on the input";
        return std::nullopt;
    }
    network.layers.assign(reversed.rbegin(), reversed.rend());
    for (std::size_t l = 1; l < network.layers.size(); ++l) {
        const auto& previous = network.layers[l - 1];
        const auto& layer = network.layers[l];
        bool chains = layer.in == previous.out;
        if (layer.kind == LayerKind::Conv) {
            chains = chains && previous.kind == LayerKind::Conv && layer.conv.inC == previous.conv.outC &&
                     layer.conv.inH == previous.conv.outH && layer.conv.inW == previous.conv.outW;
        }
        if (!chains) {
            error = "layer shapes do not chain at '" + layer.name + "'";
            return std::nullopt;
        }
    }
    network.inputs = network.layers.front().in;
    if (network.layers.front().kind == LayerKind::Conv) {
        const auto& g = network.layers.front().conv;
        network.inputShape = {g.inC, g.inH, g.inW};
    } else {
        network.inputShape = {network.inputs};
    }
    auto& head = network.layers.back();
    if (head.kind != LayerKind::Dense) {
        error = "the last layer must be dense (Flatten + Gemm)";
        return std::nullopt;
    }
    if (head.out == 1) {
        // Decision of a single logit z is z > 0: compare against a constant 0 logit.
        std::vector<double> weight(2 * head.in, 0.0);
        std::copy(head.weight.begin(), head.weight.end(), weight.begin() + static_cast<long>(head.in));
        head.weight = std::move(weight);
        head.bias = {0.0, head.bias[0]};
        head.out = 2;
        network.singleLogitPadded = true;
    }
    network.classes = head.out;
    for (auto& layer : network.layers) buildTaps(layer);
    return network;
}

std::vector<double> referenceLogits(const DenseNetwork& network, const std::vector<double>& x) {
    std::vector<double> current = x;
    for (const auto& layer : network.layers) {
        std::vector<double> next(layer.out);
        for (std::size_t o = 0; o < layer.out; ++o) {
            double sum = layer.bias[layer.channelOf(o)];
            for (const auto& tap : layer.taps[o]) sum += layer.weight[tap.weight] * current[tap.input];
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

double inputQuantisationSlack(std::int64_t inputMax) {
    // |x * inv_s - x / s| and the rounding of t +/- 0.5 in float32, in LSBs.
    const double qmax = static_cast<double>(inputMax);
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
        const std::int64_t mulLimit = signedMax(layer.mulBits);
        std::vector<std::int64_t> next(layer.out);
        for (std::size_t o = 0; o < layer.out; ++o) {
            const std::size_t c = layer.channelOf(o);
            std::int64_t acc = layer.qBias[c];
            if (!layer.modular && std::llabs(acc) > accLimit) run.accumulatorOverflow = true;
            for (const auto& tap : layer.taps[o]) {
                const std::int64_t product = static_cast<std::int64_t>(layer.qWeight[tap.weight]) * current[tap.input];
                if (product > mulLimit || product < -mulLimit - 1) run.productOverflow = true;
                acc += product;
                if (!layer.modular && (acc > accLimit || acc < -accLimit - 1)) run.accumulatorOverflow = true;
            }
            // Modular accumulation reconstructs the exact sum iff it fits.
            if (layer.modular && (acc > accLimit || acc < -accLimit - 1)) run.accumulatorOverflow = true;
            if (layer.relu && acc < 0) acc = 0;
            std::int64_t q = roundShift(acc * layer.multiplier[c], layer.shift[c]);
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
        // bias (accumulator type) + multiplier (4) + shift (1) per channel
        cost.flashBytes += static_cast<double>(layer.channels) * (layer.accBits / 8.0 + 4.0 + 1.0);
        const std::size_t outBytes = layer.out * static_cast<std::size_t>(layer.outBits <= 8 ? 1 : 2);
        largestPair = std::max(largestPair, previous + outBytes);
        previous = outBytes;
    }
    cost.flashBytes += cost.weightBytes + 4.0 * static_cast<double>(program.inputs);  // + float input scales
    cost.ramBytes = static_cast<double>(largestPair);
    return cost;
}

double float32Bytes(const DenseNetwork& network) { return 4.0 * static_cast<double>(network.parameterCount()); }

std::size_t multiplyAccumulates(const DenseNetwork& network) {
    std::size_t total = 0;
    for (const auto& layer : network.layers) {
        for (const auto& taps : layer.taps) total += taps.size();
    }
    return total;
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
    const char* inType = cType(program.inputBits, program.inputUnsigned);
    const auto& head = program.layers.back();
    const char* logitType = cType(head.outBits, false);

    std::ofstream header(out / "model_q.h");
    header << "/* Integer-only inference code generated by ModelForge / ForgeCert.\n"
           << " * Model: " << program.name << "   precision: " << program.config.describe() << "\n"
           << " * No floating point after mf_quantize_input(). See certificate.json. */\n"
           << "#ifndef MF_MODEL_Q_H\n#define MF_MODEL_Q_H\n\n#include <stdint.h>\n\n"
           << "#define MF_INPUTS " << program.inputs << "\n#define MF_CLASSES " << program.classes
           << "\n\ntypedef " << inType << " mf_input_t;\ntypedef " << logitType << " mf_logit_t;\n\n"
           << "/* Quantises float features (the certified input domain is in certificate.json). */\n"
           << "void mf_quantize_input(const float *x, mf_input_t *q);\n"
           << "/* Integer-only inference; returns the predicted class. logits may be NULL. */\n"
           << "int mf_infer_q(const mf_input_t *q, mf_logit_t *logits);\n"
           << "/* Convenience: quantise, infer, and optionally dequantise the logits. */\n"
           << "int mf_predict(const float *x, float *logits);\n\n#endif\n";

    std::ofstream source(out / "model_q.c");
    source << "/* Integer-only inference code generated by ModelForge / ForgeCert.\n"
           << " * MF-ARTIFACT v3. The arrays below are read verbatim by modelforge_certcheck;\n"
           << " * do not edit them by hand (the certificate would no longer apply). */\n"
           << "#include \"model_q.h\"\n\n";
    // MF_META: version, inputs, classes, input bits, input unsigned, layer count, then per layer
    // kind, in, out, relu, weight bits, acc bits, modular, mul bits, out bits,
    // out unsigned, inC, inH, inW, outC, outH, outW, kH, kW, stride, pad.
    std::vector<std::int64_t> meta{3, static_cast<std::int64_t>(program.inputs),
                                   static_cast<std::int64_t>(program.classes), program.inputBits,
                                   program.inputUnsigned ? 1 : 0, static_cast<std::int64_t>(program.layers.size())};
    for (const auto& layer : program.layers) {
        const auto& g = layer.conv;
        meta.insert(meta.end(),
                    {layer.kind == LayerKind::Conv ? 1 : 0, static_cast<std::int64_t>(layer.in),
                     static_cast<std::int64_t>(layer.out), layer.relu ? 1 : 0, layer.weightBits, layer.accBits,
                     layer.modular ? 1 : 0, layer.mulBits, layer.outBits, layer.outUnsigned ? 1 : 0,
                     static_cast<std::int64_t>(g.inC), static_cast<std::int64_t>(g.inH),
                     static_cast<std::int64_t>(g.inW), static_cast<std::int64_t>(g.outC),
                     static_cast<std::int64_t>(g.outH), static_cast<std::int64_t>(g.outW),
                     static_cast<std::int64_t>(g.kH), static_cast<std::int64_t>(g.kW),
                     static_cast<std::int64_t>(g.stride), static_cast<std::int64_t>(g.pad)});
    }
    const auto writeScales = [&](const std::string& name, const std::vector<double>& values) {
        source << "static const double " << name << "[" << values.size() << "] = {";
        for (std::size_t i = 0; i < values.size(); ++i) {
            source << (i % 4 == 0 ? "\n   " : "") << " " << exactDouble(values[i]) << (i + 1 < values.size() ? "," : "");
        }
        source << "\n};\n";
    };
    // Metadata and tensor scales (the real value of element q is scale * q)
    // are read by the checker only; the logit scale is used by mf_predict.
    source << "#ifdef MF_CHECKER_METADATA\n";
    writeArray(source, "int32_t", "MF_META", meta);
    writeScales("MF_SCALE_IN", program.inputScale);
    for (std::size_t l = 0; l + 1 < program.layers.size(); ++l) {
        writeScales("MF_SCALE_" + std::to_string(l), program.layers[l].outScale);
    }
    source << "#endif\n";
    writeScales("MF_SCALE_" + std::to_string(program.layers.size() - 1), program.layers.back().outScale);
    source << "static const float MF_INPUT_INV_SCALE[" << program.inputs << "] = {";
    for (std::size_t i = 0; i < program.inputs; ++i) {
        source << (i % 4 == 0 ? "\n   " : "") << " " << exactFloat(static_cast<float>(1.0 / program.inputScale[i]))
               << (i + 1 < program.inputs ? "," : "");
    }
    source << "\n};\n\n";
    source << "static int64_t mf_round_shift(int64_t t, int s) {\n"
           << "    int64_t u = t + ((int64_t)1 << (s - 1));\n"
           << "    return u >= 0 ? (u >> s) : -((-u + (((int64_t)1 << s) - 1)) >> s);\n}\n\n";
    if (std::any_of(program.layers.begin(), program.layers.end(), [](const QLayer& q) { return q.weightBits <= 4; })) {
        source << "static int8_t mf_nibble(const uint8_t *packed, int k) {\n"
               << "    int v = (k & 1) ? (packed[k >> 1] >> 4) : (packed[k >> 1] & 15);\n"
               << "    return (int8_t)(v >= 8 ? v - 16 : v);\n}\n\n";
    }

    for (std::size_t l = 0; l < program.layers.size(); ++l) {
        const auto& layer = program.layers[l];
        const std::string id = std::to_string(l);
        if (layer.weightBits <= 4) {
            std::vector<std::int64_t> bytes((layer.qWeight.size() + 1) / 2, 0);
            for (std::size_t k = 0; k < layer.qWeight.size(); ++k) {
                const std::int64_t nibble = layer.qWeight[k] & 15;
                bytes[k / 2] |= (k % 2) ? (nibble << 4) : nibble;
            }
            writeArray(source, "uint8_t", "MF_W" + id, bytes);
        } else {
            writeArray(source, cType(layer.weightBits, false), "MF_W" + id, layer.qWeight);
        }
        if (layer.modular) {
            // Stored modulo 2^accBits: only the final sum must fit.
            const unsigned long long mask = layer.accBits == 16 ? 0xFFFFull : 0xFFFFFFFFull;
            std::vector<unsigned long long> wrapped;
            for (auto b : layer.qBias) wrapped.push_back(static_cast<unsigned long long>(b) & mask);
            writeUnsignedArray(source, cType(layer.accBits, true), "MF_B" + id, wrapped);
        } else {
            writeArray(source, cType(layer.accBits, false), "MF_B" + id, layer.qBias);
        }
        writeArray(source, "int32_t", "MF_M" + id, layer.multiplier);
        writeArray(source, "uint8_t", "MF_S" + id, layer.shift);
        source << "\n/* layer " << l << ": " << (layer.kind == LayerKind::Conv ? "conv " : "dense ") << layer.in
               << " -> " << layer.out << (layer.relu ? ", ReLU" : "") << "; weights " << layer.weightBits << "-bit; "
               << layer.accBits << "-bit " << (layer.modular ? "modular " : "") << "accumulator; " << layer.mulBits
               << "-bit products (widths proved on the domain) */\n"
               << layerFunction(program, l);
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
        source << "    static " << cType(layer.outBits, layer.outUnsigned) << " t" << l << "[" << layer.out << "];\n";
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
           << "    static mf_input_t q[MF_INPUTS];\n    mf_logit_t raw[MF_CLASSES];\n"
           << "    mf_quantize_input(x, q);\n    int best = mf_infer_q(q, raw);\n"
           << "    if (logits) {\n        for (int c = 0; c < MF_CLASSES; ++c) logits[c] = (float)(MF_SCALE_"
           << program.layers.size() - 1 << "[c] * raw[c]);\n    }\n    return best;\n}\n";

    std::ofstream mainFile(out / "main_q.c");
    mainFile << "/* Command-line driver for the integer-only model (ModelForge / ForgeCert).\n"
             << " *   generated_model_q f1 f2 ... fN     one input, prints class and logits\n"
             << " *   generated_model_q --csv rows.csv   one prediction per row (extra columns ignored)\n"
             << " *   generated_model_q --raw rows.csv   'row,class,int logits...' (bit-exact check) */\n"
             << "#include \"model_q.h\"\n\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n\n"
             << "static char line[1 << 20];\n\n"
             << "static int run_csv(const char *path, int raw) {\n"
             << "    FILE *file = fopen(path, \"r\");\n    int row = 0;\n"
             << "    if (!file) { fprintf(stderr, \"cannot open %s\\n\", path); return 1; }\n"
             << "    while (fgets(line, sizeof line, file)) {\n"
             << "        static float x[MF_INPUTS];\n        static mf_input_t q[MF_INPUTS];\n        mf_logit_t logits[MF_CLASSES];\n"
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
             << "    static float x[MF_INPUTS];\n    float logits[MF_CLASSES];\n"
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

bool emitFloatC(const DenseNetwork& network, const std::string& directory, std::string& error) {
    namespace fs = std::filesystem;
    std::error_code code;
    fs::create_directories(directory, code);
    const fs::path out = directory;
    std::ofstream header(out / "model_f.h");
    header << "/* float32 reference generated by ModelForge (baseline for integer code). */\n"
           << "#ifndef MF_MODEL_F_H\n#define MF_MODEL_F_H\n#define MF_F_INPUTS " << network.inputs
           << "\n#define MF_F_CLASSES " << network.classes << "\nint mf_predict_f(const float *x, float *logits);\n#endif\n";
    std::ofstream source(out / "model_f.c");
    source << "#include \"model_f.h\"\n\n";
    const auto floats = [&](const std::string& name, const std::vector<double>& values) {
        source << "static const float " << name << "[" << values.size() << "] = {";
        for (std::size_t i = 0; i < values.size(); ++i) {
            source << (i % 8 == 0 ? "\n   " : "") << " " << exactFloat(static_cast<float>(values[i]))
                   << (i + 1 < values.size() ? "," : "");
        }
        source << "\n};\n";
    };
    for (std::size_t l = 0; l < network.layers.size(); ++l) {
        const auto& layer = network.layers[l];
        floats("F_W" + std::to_string(l), layer.weight);
        floats("F_B" + std::to_string(l), layer.bias);
        source << "static void f_layer" << l << "(const float *x, float *y) {\n";
        if (layer.kind == LayerKind::Conv) {
            const auto& g = layer.conv;
            source << "    for (int c = 0; c < " << g.outC << "; ++c)\n"
                   << "    for (int oy = 0; oy < " << g.outH << "; ++oy)\n"
                   << "    for (int ox = 0; ox < " << g.outW << "; ++ox) {\n"
                   << "        float s = F_B" << l << "[c];\n"
                   << "        for (int ci = 0; ci < " << g.inC << "; ++ci)\n"
                   << "        for (int ky = 0; ky < " << g.kH << "; ++ky) {\n"
                   << "            const int iy = oy * " << g.stride << " + ky - " << g.pad << ";\n"
                   << "            if (iy < 0 || iy >= " << g.inH << ") continue;\n"
                   << "            for (int kx = 0; kx < " << g.kW << "; ++kx) {\n"
                   << "                const int ix = ox * " << g.stride << " + kx - " << g.pad << ";\n"
                   << "                if (ix < 0 || ix >= " << g.inW << ") continue;\n"
                   << "                s += F_W" << l << "[((c * " << g.inC << " + ci) * " << g.kH << " + ky) * " << g.kW
                   << " + kx] * x[(ci * " << g.inH << " + iy) * " << g.inW << " + ix];\n"
                   << "            }\n        }\n"
                   << "        y[(c * " << g.outH << " + oy) * " << g.outW << " + ox] = "
                   << (layer.relu ? "s > 0.0f ? s : 0.0f" : "s") << ";\n    }\n}\n";
        } else {
            source << "    for (int o = 0; o < " << layer.out << "; ++o) {\n        float s = F_B" << l << "[o];\n"
                   << "        for (int i = 0; i < " << layer.in << "; ++i) s += F_W" << l << "[o * " << layer.in
                   << " + i] * x[i];\n        y[o] = " << (layer.relu ? "s > 0.0f ? s : 0.0f" : "s") << ";\n    }\n}\n";
        }
    }
    source << "\nint mf_predict_f(const float *x, float *logits) {\n";
    for (std::size_t l = 0; l + 1 < network.layers.size(); ++l) {
        source << "    static float t" << l << "[" << network.layers[l].out << "];\n";
    }
    source << "    float out[MF_F_CLASSES];\n";
    for (std::size_t l = 0; l < network.layers.size(); ++l) {
        source << "    f_layer" << l << "(" << (l == 0 ? std::string("x") : "t" + std::to_string(l - 1)) << ", "
               << (l + 1 == network.layers.size() ? std::string("out") : "t" + std::to_string(l)) << ");\n";
    }
    source << "    int best = 0;\n    for (int c = 0; c < MF_F_CLASSES; ++c) {\n"
           << "        if (logits) logits[c] = out[c];\n        if (out[c] > out[best]) best = c;\n    }\n"
           << "    return best;\n}\n";
    if (!header || !source) {
        error = "failed to write the float C sources";
        return false;
    }
    return true;
}

}  // namespace modelforge

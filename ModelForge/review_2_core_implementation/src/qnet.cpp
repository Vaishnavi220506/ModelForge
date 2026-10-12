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
    // Constant tables live in flash (MF_ROM = __flash on AVR; empty elsewhere).
    out << "static const " << (name == "MF_META" ? "" : "MF_ROM ") << type << " " << name << "["
        << values.size() << "] = {";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i % 16 == 0) out << "\n   ";
        out << " " << static_cast<long long>(values[i]) << (i + 1 < values.size() ? "," : "");
    }
    out << "\n};\n";
}

void writeUnsignedArray(std::ostream& out, const std::string& type, const std::string& name,
                        const std::vector<unsigned long long>& values) {
    out << "static const MF_ROM " << type << " " << name << "[" << values.size() << "] = {";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i % 16 == 0) out << "\n   ";
        out << " " << values[i] << "u" << (i + 1 < values.size() ? "," : "");
    }
    out << "\n};\n";
}

std::string tensorType(const QProgram& program, std::size_t tensor) {
    if (tensor == 0) return cType(program.inputBits, program.inputUnsigned);
    const QLayer& producer = program.layers[tensor - 1];
    return cType(producer.outBits, producer.outUnsigned);
}

// The code of one layer. tools/certcheck.cpp holds an independent copy of
// these templates and requires the emitted file to contain exactly this text.
std::string layerFunction(const QProgram& program, std::size_t l) {
    const QLayer& q = program.layers[l];
    const std::string inT = tensorType(program, q.sources[0]);
    const char* outT = cType(q.outBits, q.outUnsigned);
    const std::string id = std::to_string(l);
    std::ostringstream t;
    if (q.kind == LayerKind::MaxPool) {
        const auto& g = q.conv;
        const std::string s = std::to_string(g.stride);
        t << "static void mf_layer" << l << "(const " << inT << " *x, " << outT << " *y) {\n"
          << "    for (int c = 0; c < " << g.outC << "; ++c) {\n"
          << "        for (int oy = 0; oy < " << g.outH << "; ++oy) {\n"
          << "            for (int ox = 0; ox < " << g.outW << "; ++ox) {\n"
          << "                " << outT << " m = x[(c * " << g.inH << " + oy * " << s << ") * " << g.inW << " + ox * "
          << s << "];\n"
          << "                for (int ky = 0; ky < " << g.kH << "; ++ky) {\n"
          << "                    for (int kx = 0; kx < " << g.kW << "; ++kx) {\n"
          << "                        const " << outT << " v = x[(c * " << g.inH << " + oy * " << s << " + ky) * "
          << g.inW << " + ox * " << s << " + kx];\n"
          << "                        if (v > m) m = v;\n"
          << "                    }\n                }\n"
          << "                y[(c * " << g.outH << " + oy) * " << g.outW << " + ox] = m;\n"
          << "            }\n        }\n    }\n}\n";
        return t.str();
    }
    const char* accT = cType(q.accBits, false);
    const char* uaccT = cType(q.accBits, true);
    const char* mulT = q.mulBits == 16 ? "int16_t" : "int32_t";
    const auto weightAt = [&](const std::string& index) {
        return q.weightBits <= 4 ? "mf_nibble(MF_W" + id + ", " + index + ")" : "MF_W" + id + "[" + index + "]";
    };
    const auto step = [&](const std::string& in, const std::string& weightIndex, const std::string& input) {
        const std::string product = "(" + std::string(mulT) + ")" + weightAt(weightIndex) + " * (" + mulT + ")" + input;
        if (q.modular) {
            return in + "u = (" + uaccT + ")(u + (" + uaccT + ")(" + product + "));\n";
        }
        return in + "acc = (" + accT + ")(acc + (" + accT + ")(" + product + "));\n";
    };
    const auto init = [&](const std::string& in, const std::string& channel) {
        if (q.modular) return in + uaccT + " u = MF_B" + id + "[" + channel + "];\n";
        return in + accT + " acc = (" + accT + ")MF_B" + id + "[" + channel + "];\n";
    };
    const auto tail = [&](const std::string& in, const std::string& channel) {
        std::ostringstream r;
        if (q.modular) {
            r << in << "int64_t acc = u >= " << (q.accBits == 16 ? "0x8000u" : "0x80000000u") << " ? (int64_t)u - "
              << (q.accBits == 16 ? "65536" : "4294967296LL") << " : (int64_t)u;\n";
        }
        if (q.relu) r << in << "if (acc < 0) acc = 0;\n";
        r << in << "int64_t q = mf_round_shift((int64_t)acc * MF_M" << l << "[" << channel << "], MF_S" << l << "["
          << channel << "]);\n"
          << in << "if (q > " << q.outMax << ") q = " << q.outMax << ";  /* saturate (modelled by the certificate) */\n"
          << in << "if (q < " << q.outMin << ") q = " << q.outMin << ";\n";
        return r.str();
    };
    if (q.kind == LayerKind::Add) {
        const std::string bT = tensorType(program, q.sources[1]);
        const std::size_t hw = q.out / q.channels;
        const std::string in = "            ";
        const std::string at = "[c * " + std::to_string(hw) + " + k]";
        t << "static void mf_layer" << l << "(const " << inT << " *a, const " << bT << " *b, " << outT << " *y) {\n"
          << "    for (int c = 0; c < " << q.channels << "; ++c) {\n"
          << "        for (int k = 0; k < " << hw << "; ++k) {\n"
          << init(in, "c") << step(in, "c * 2", "a" + at) << step(in, "c * 2 + 1", "b" + at) << tail(in, "c")
          << in << "y" << at << " = (" << outT << ")q;\n"
          << "        }\n    }\n}\n";
        return t.str();
    }
    t << "static void mf_layer" << l << "(const " << inT << " *x, " << outT << " *y) {\n";
    if (q.kind == LayerKind::Conv) {
        const auto& g = q.conv;
        const std::size_t cpg = g.inC / g.groups, opg = g.outC / g.groups;
        const std::string in = "                ";
        const std::string channel =
            g.groups == 1 ? "ci" : "(c / " + std::to_string(opg) + " * " + std::to_string(cpg) + " + ci)";
        t << "    for (int c = 0; c < " << g.outC << "; ++c) {\n"
          << "        for (int oy = 0; oy < " << g.outH << "; ++oy) {\n"
          << "            for (int ox = 0; ox < " << g.outW << "; ++ox) {\n"
          << init(in, "c")
          << in << "for (int ci = 0; ci < " << cpg << "; ++ci) {\n"
          << in << "    for (int ky = 0; ky < " << g.kH << "; ++ky) {\n"
          << in << "        const int iy = oy * " << g.stride << " + ky - " << g.pad << ";\n"
          << in << "        if (iy < 0 || iy >= " << g.inH << ") continue;\n"
          << in << "        for (int kx = 0; kx < " << g.kW << "; ++kx) {\n"
          << in << "            const int ix = ox * " << g.stride << " + kx - " << g.pad << ";\n"
          << in << "            if (ix < 0 || ix >= " << g.inW << ") continue;\n"
          << step(in + "            ",
                  "((c * " + std::to_string(cpg) + " + ci) * " + std::to_string(g.kH) + " + ky) * " +
                      std::to_string(g.kW) + " + kx",
                  "x[(" + channel + " * " + std::to_string(g.inH) + " + iy) * " + std::to_string(g.inW) + " + ix]")
          << in << "        }\n" << in << "    }\n" << in << "}\n"
          << tail(in, "c")
          << in << "y[(c * " << g.outH << " + oy) * " << g.outW << " + ox] = (" << outT << ")q;\n"
          << "            }\n        }\n    }\n}\n";
        return t.str();
    }
    const std::string in = "        ";
    t << "    for (int o = 0; o < " << q.out << "; ++o) {\n"
      << init(in, "o")
      << in << "for (int i = 0; i < " << q.in << "; ++i) {\n"
      << step(in + "    ", "o * " + std::to_string(q.in) + " + i", "x[i]") << in << "}\n"
      << tail(in, "o")
      << in << "y[o] = (" << outT << ")q;\n    }\n}\n";
    return t.str();
}

TensorShape shapeOf(const Shape& shape) {
    TensorShape t;
    if (shape.size() == 4) {
        t.c = static_cast<std::size_t>(shape[1]);
        t.h = static_cast<std::size_t>(shape[2]);
        t.w = static_cast<std::size_t>(shape[3]);
        t.spatial = true;
    } else {
        t.c = elementCount(shape);
    }
    return t;
}

// A tensor read as a feature map: vectors are C x 1 x 1.
TensorShape asFeatureMap(TensorShape t) {
    t.spatial = true;
    return t;
}

}  // namespace

void buildTaps(DenseLayer& layer, const std::vector<TensorShape>& tensors) {
    layer.taps.assign(layer.out, {});
    const auto u32 = [](std::size_t v) { return static_cast<std::uint32_t>(v); };
    switch (layer.kind) {
        case LayerKind::Dense:
            for (std::size_t o = 0; o < layer.out; ++o) {
                layer.taps[o].reserve(layer.in);
                for (std::size_t i = 0; i < layer.in; ++i) layer.taps[o].push_back({u32(i), u32(o * layer.in + i)});
            }
            return;
        case LayerKind::Add: {
            const std::size_t size = tensors[layer.sources[0]].size();
            for (std::size_t o = 0; o < layer.out; ++o) {
                const std::size_t c = layer.channelOf(o);
                layer.taps[o] = {{u32(o), u32(2 * c)}, {u32(size + o), u32(2 * c + 1)}};
            }
            return;
        }
        case LayerKind::Conv:
        case LayerKind::MaxPool: {
            const auto& g = layer.conv;
            const bool pool = layer.kind == LayerKind::MaxPool;
            const std::size_t cpg = pool ? 1 : g.inC / g.groups, opg = pool ? 1 : g.outC / g.groups;
            for (std::size_t c = 0; c < g.outC; ++c) {
                for (std::size_t oy = 0; oy < g.outH; ++oy) {
                    for (std::size_t ox = 0; ox < g.outW; ++ox) {
                        auto& taps = layer.taps[(c * g.outH + oy) * g.outW + ox];
                        for (std::size_t ci = 0; ci < cpg; ++ci) {
                            const std::size_t channel = c / opg * cpg + ci;
                            for (std::size_t ky = 0; ky < g.kH; ++ky) {
                                const long iy = static_cast<long>(oy * g.stride + ky) - static_cast<long>(g.pad);
                                if (iy < 0 || iy >= static_cast<long>(g.inH)) continue;
                                for (std::size_t kx = 0; kx < g.kW; ++kx) {
                                    const long ix = static_cast<long>(ox * g.stride + kx) - static_cast<long>(g.pad);
                                    if (ix < 0 || ix >= static_cast<long>(g.inW)) continue;
                                    const std::size_t input = (channel * g.inH + static_cast<std::size_t>(iy)) * g.inW +
                                                              static_cast<std::size_t>(ix);
                                    taps.push_back({u32(input), pool ? 0u : u32(((c * cpg + ci) * g.kH + ky) * g.kW + kx)});
                                }
                            }
                        }
                    }
                }
            }
            return;
        }
    }
}

std::size_t DenseNetwork::scaleGroupOf(std::size_t tensor, std::size_t i) const {
    const TensorShape& shape = tensors[tensor];
    if (!shape.spatial) return i;
    for (const auto& layer : layers) {
        if (layer.kind == LayerKind::Dense) continue;
        if (std::find(layer.sources.begin(), layer.sources.end(), tensor) != layer.sources.end()) {
            return shape.channelOf(i);
        }
    }
    return i;
}

std::size_t DenseNetwork::parameterCount() const {
    std::size_t count = 0;
    for (const auto& layer : layers) count += layer.weight.size() + layer.bias.size();
    return count;
}

std::vector<std::size_t> DenseNetwork::lastUse() const {
    std::vector<std::size_t> last(tensors.size(), 0);
    for (std::size_t t = 1; t < tensors.size(); ++t) last[t] = t - 1;  // dead right after its producer
    for (std::size_t l = 0; l < layers.size(); ++l) {
        for (std::size_t s : layers[l].sources) last[s] = std::max(last[s], l);
    }
    if (!layers.empty()) last.back() = layers.size();
    return last;
}

bool fixedSlot(const DenseNetwork& network, int kind, std::size_t index) {
    if (kind == 1) return index < network.layers.size() && !network.layers[index].hasWeights();
    return index > 0 && index <= network.layers.size() && network.layers[index - 1].kind == LayerKind::MaxPool;
}

std::optional<DenseNetwork> extractDenseNetwork(const IRGraph& graph, std::string& error) {
    DenseNetwork network;
    network.name = graph.name;
    network.outputActivation = "none";
    const auto input = graph.values.find(graph.inputName);
    if (input == graph.values.end()) {
        error = "the graph has no input";
        return std::nullopt;
    }
    network.tensors.push_back(shapeOf(input->second.shape));
    // Every IR value is an alias of a tensor. A layer may be changed in place
    // (ReLU, bias, batch normalisation folded in) only while no other reader
    // of its tensor remains and no layer has read it yet.
    std::unordered_map<std::string, std::size_t> uses;
    for (const auto& instruction : graph.instructions) {
        for (const auto& name : instruction.inputs) ++uses[name];
    }
    std::unordered_map<std::string, std::size_t> tensorOf{{graph.inputName, 0}};
    std::vector<std::size_t> pending{uses[graph.inputName]};
    std::vector<bool> read{false};
    const auto alias = [&](const std::string& value, std::size_t tensor) {
        tensorOf[value] = tensor;
        pending[tensor] += uses[value];
    };
    std::string activationOutput;
    const auto lookup = [&](const std::string& value, std::size_t& tensor) {
        if (value == activationOutput) {
            error = "Softmax/Sigmoid is supported only as the output activation";
            return false;
        }
        const auto found = tensorOf.find(value);
        if (found == tensorOf.end()) {
            error = "value '" + value + "' is used before it is defined";
            return false;
        }
        tensor = found->second;
        --pending[tensor];
        return true;
    };
    // The layer that may be changed in place for the value, or nullptr.
    const auto modifiable = [&](std::size_t tensor) -> DenseLayer* {
        if (tensor == 0 || pending[tensor] != 0 || read[tensor]) return nullptr;
        return &network.layers[tensor - 1];
    };
    const auto addLayer = [&](DenseLayer layer, const std::string& value) {
        layer.in = 0;
        for (std::size_t s : layer.sources) {
            layer.in += network.tensors[s].size();
            read[s] = true;
        }
        layer.out = layer.output.size();
        network.tensors.push_back(layer.output);
        pending.push_back(0);
        read.push_back(false);
        network.layers.push_back(std::move(layer));
        alias(value, network.tensors.size() - 1);
    };
    // A depthwise convolution with fixed per-channel weights (pooling, or a
    // batch normalisation that cannot be folded).
    const auto depthwise = [&](const std::string& name, std::size_t source, std::size_t k, std::size_t stride,
                               const std::vector<double>& weightPerChannel, const std::vector<double>& bias) {
        const TensorShape x = asFeatureMap(network.tensors[source]);
        DenseLayer layer;
        layer.name = name;
        layer.kind = LayerKind::Conv;
        layer.sources = {source};
        auto& g = layer.conv;
        g.inC = g.outC = g.groups = x.c;
        g.inH = x.h;
        g.inW = x.w;
        g.kH = g.kW = k;
        g.stride = stride;
        g.outH = (x.h - k) / stride + 1;
        g.outW = (x.w - k) / stride + 1;
        layer.output = {g.outC, g.outH, g.outW, true};
        layer.weightsPerChannel = k * k;
        for (std::size_t c = 0; c < x.c; ++c) layer.weight.insert(layer.weight.end(), k * k, weightPerChannel[c]);
        layer.bias = bias;
        return layer;
    };
    const auto constantData = [&](const std::string& name) -> const TensorInfo* { return constantOf(graph, name); };

    std::string returned;
    for (const auto& instruction : graph.instructions) {
        const std::string& where = instruction.nodeName;
        std::size_t x = 0;
        switch (instruction.operation) {
            case IROp::Input:
                break;
            case IROp::Return:
                returned = instruction.inputs.empty() ? "" : instruction.inputs[0];
                break;
            case IROp::Softmax:
            case IROp::Sigmoid:
                if (!lookup(instruction.inputs[0], x)) return std::nullopt;
                activationOutput = instruction.output;  // may only be returned
                network.outputActivation = instruction.operation == IROp::Softmax ? "softmax" : "sigmoid";
                alias(instruction.output, x);
                break;
            case IROp::Flatten:
                if (!lookup(instruction.inputs[0], x)) return std::nullopt;
                alias(instruction.output, x);
                break;
            case IROp::Relu: {
                if (!lookup(instruction.inputs[0], x)) return std::nullopt;
                DenseLayer* producer = modifiable(x);
                if (producer == nullptr || producer->relu || producer->kind == LayerKind::MaxPool) {
                    error = "ReLU '" + where + "' must directly follow a linear layer whose output has no other reader";
                    return std::nullopt;
                }
                producer->relu = true;
                alias(instruction.output, x);
                break;
            }
            case IROp::BatchNorm: {
                if (!lookup(instruction.inputs[0], x)) return std::nullopt;
                const TensorInfo* p[4];
                for (int k = 0; k < 4; ++k) p[k] = constantData(instruction.inputs[1 + k]);
                const std::size_t channels = network.tensors[x].c;
                std::vector<double> a(channels), b(channels);
                for (std::size_t c = 0; c < channels; ++c) {
                    a[c] = static_cast<double>(p[0]->data[c]) /
                           std::sqrt(static_cast<double>(p[3]->data[c]) + static_cast<double>(instruction.epsilon));
                    b[c] = static_cast<double>(p[1]->data[c]) - static_cast<double>(p[2]->data[c]) * a[c];
                }
                DenseLayer* producer = modifiable(x);
                if (producer != nullptr && producer->hasWeights() && !producer->relu && producer->channels() == channels) {
                    // Fold: W' = a W, b' = a b + (beta - mean a), per output channel.
                    for (std::size_t w = 0; w < producer->weight.size(); ++w) {
                        producer->weight[w] *= a[producer->weightChannel(w)];
                    }
                    for (std::size_t c = 0; c < channels; ++c) producer->bias[c] = a[c] * producer->bias[c] + b[c];
                    ++network.foldedBatchNorms;
                    alias(instruction.output, x);
                } else {
                    addLayer(depthwise(where, x, 1, 1, a, b), instruction.output);
                }
                break;
            }
            case IROp::AveragePool:
            case IROp::GlobalAveragePool: {
                if (!lookup(instruction.inputs[0], x)) return std::nullopt;
                const TensorShape shape = network.tensors[x];
                if (!shape.spatial) {
                    error = "pooling '" + where + "' needs a feature map";
                    return std::nullopt;
                }
                const bool global = instruction.operation == IROp::GlobalAveragePool;
                if (global && shape.h != shape.w) {
                    error = "GlobalAveragePool '" + where + "' needs a square feature map";
                    return std::nullopt;
                }
                const std::size_t k = global ? shape.h : static_cast<std::size_t>(instruction.kernel);
                const std::size_t stride = global ? 1 : static_cast<std::size_t>(instruction.stride);
                addLayer(depthwise(where, x, k, stride, std::vector<double>(shape.c, 1.0 / static_cast<double>(k * k)),
                                   std::vector<double>(shape.c, 0.0)),
                         instruction.output);
                ++network.loweredPools;
                break;
            }
            case IROp::MaxPool: {
                if (!lookup(instruction.inputs[0], x)) return std::nullopt;
                const TensorShape shape = network.tensors[x];
                if (!shape.spatial) {
                    error = "MaxPool '" + where + "' needs a feature map";
                    return std::nullopt;
                }
                DenseLayer layer;
                layer.name = where;
                layer.kind = LayerKind::MaxPool;
                layer.sources = {x};
                auto& g = layer.conv;
                g.inC = g.outC = g.groups = shape.c;
                g.inH = shape.h;
                g.inW = shape.w;
                g.kH = g.kW = static_cast<std::size_t>(instruction.kernel);
                g.stride = static_cast<std::size_t>(instruction.stride);
                g.outH = (shape.h - g.kH) / g.stride + 1;
                g.outW = (shape.w - g.kW) / g.stride + 1;
                layer.output = {g.outC, g.outH, g.outW, true};
                layer.weightsPerChannel = 1;
                addLayer(std::move(layer), instruction.output);
                break;
            }
            case IROp::Conv: {
                const TensorInfo* weight = constantData(instruction.inputs[1]);
                if (!lookup(instruction.inputs[0], x)) return std::nullopt;
                const TensorShape shape = network.tensors[x];
                if (weight == nullptr || weight->shape.size() != 4 || !shape.spatial || instruction.shape.size() != 4) {
                    error = "Conv '" + where + "' needs NCHW input and a constant weight";
                    return std::nullopt;
                }
                DenseLayer layer;
                layer.name = where;
                layer.kind = LayerKind::Conv;
                layer.sources = {x};
                auto& g = layer.conv;
                g.inC = shape.c;
                g.inH = shape.h;
                g.inW = shape.w;
                g.outC = static_cast<std::size_t>(weight->shape[0]);
                g.kH = static_cast<std::size_t>(weight->shape[2]);
                g.kW = static_cast<std::size_t>(weight->shape[3]);
                g.stride = static_cast<std::size_t>(instruction.stride);
                g.pad = static_cast<std::size_t>(instruction.pad);
                g.groups = static_cast<std::size_t>(instruction.group);
                g.outH = static_cast<std::size_t>(instruction.shape[2]);
                g.outW = static_cast<std::size_t>(instruction.shape[3]);
                if (g.groups == 0 || g.inC % g.groups || g.outC % g.groups ||
                    static_cast<std::size_t>(weight->shape[1]) * g.groups != g.inC) {
                    error = "Conv '" + where + "' has inconsistent groups";
                    return std::nullopt;
                }
                layer.output = {g.outC, g.outH, g.outW, true};
                layer.weightsPerChannel = g.inC / g.groups * g.kH * g.kW;
                layer.weight.assign(weight->data.begin(), weight->data.end());
                layer.bias.assign(g.outC, 0.0);
                if (instruction.inputs.size() == 3) {
                    const TensorInfo* bias = constantData(instruction.inputs[2]);
                    if (bias == nullptr || !readBias(*bias, layer, error)) {
                        if (error.empty()) error = "Conv bias must be constant";
                        return std::nullopt;
                    }
                }
                addLayer(std::move(layer), instruction.output);
                break;
            }
            case IROp::Gemm:
            case IROp::FusedGemmRelu:
            case IROp::MatMul: {
                const bool gemm = instruction.operation != IROp::MatMul;
                if ((gemm && instruction.transA) || instruction.inputs.size() < 2) {
                    error = "unsupported Gemm form in '" + where + "'";
                    return std::nullopt;
                }
                const TensorInfo* weight = constantData(instruction.inputs[1]);
                if (weight == nullptr) {
                    error = "'" + where + "' needs a constant weight";
                    return std::nullopt;
                }
                if (!lookup(instruction.inputs[0], x)) return std::nullopt;
                DenseLayer layer;
                layer.name = where;
                layer.sources = {x};
                if (!readWeight(*weight, gemm && instruction.transB, layer, error)) return std::nullopt;
                if (layer.in != network.tensors[x].size()) {
                    error = "layer shapes do not chain at '" + where + "'";
                    return std::nullopt;
                }
                layer.output = {layer.out, 1, 1, false};
                layer.weightsPerChannel = layer.in;
                if (gemm && instruction.inputs.size() == 3) {
                    const TensorInfo* bias = constantData(instruction.inputs[2]);
                    if (bias == nullptr || !readBias(*bias, layer, error)) {
                        if (error.empty()) error = "Gemm bias must be constant";
                        return std::nullopt;
                    }
                }
                layer.relu = instruction.operation == IROp::FusedGemmRelu;
                addLayer(std::move(layer), instruction.output);
                break;
            }
            case IROp::Add: {
                if (instruction.inputs.size() != 2) {
                    error = "Add '" + where + "' needs two operands";
                    return std::nullopt;
                }
                const TensorInfo* constant = nullptr;
                std::vector<std::size_t> operands;
                for (const auto& name : instruction.inputs) {
                    if (const auto* c = constantData(name)) {
                        constant = c;
                    } else {
                        std::size_t t = 0;
                        if (!lookup(name, t)) return std::nullopt;
                        operands.push_back(t);
                    }
                }
                if (constant != nullptr && operands.size() == 1) {
                    // A constant added to a linear layer is its bias.
                    DenseLayer* producer = modifiable(operands[0]);
                    if (producer == nullptr || !producer->hasWeights() || producer->relu ||
                        !readBias(*constant, *producer, error)) {
                        if (error.empty()) error = "a constant Add must follow a linear layer ('" + where + "')";
                        return std::nullopt;
                    }
                    alias(instruction.output, operands[0]);
                    break;
                }
                if (operands.size() != 2 || network.tensors[operands[0]].size() != network.tensors[operands[1]].size()) {
                    error = "Add '" + where + "' needs two tensors of the same shape";
                    return std::nullopt;
                }
                // Residual connection: y = a + b, one weight per source and channel.
                DenseLayer layer;
                layer.name = where;
                layer.kind = LayerKind::Add;
                layer.sources = operands;
                layer.output = network.tensors[operands[0]];
                const std::size_t channels = layer.output.spatial ? layer.output.c : layer.output.size();
                auto& g = layer.conv;
                g.inC = g.outC = g.groups = channels;
                g.inH = g.outH = layer.output.h;
                g.inW = g.outW = layer.output.w;
                g.kH = g.kW = 1;
                layer.weightsPerChannel = 2;
                layer.weight.assign(2 * channels, 1.0);
                layer.bias.assign(channels, 0.0);
                addLayer(std::move(layer), instruction.output);
                break;
            }
            default:
                error = "operator " + irOpName(instruction.operation) + " is not supported by the integer target";
                return std::nullopt;
        }
    }
    if (network.layers.empty()) {
        error = "the model has no dense layer";
        return std::nullopt;
    }
    const auto found = tensorOf.find(returned);
    if (found == tensorOf.end() || found->second != network.tensors.size() - 1) {
        error = "the returned value must be the output of the last layer";
        return std::nullopt;
    }
    auto& head = network.layers.back();
    if (head.kind != LayerKind::Dense) {
        error = "the last layer must be dense (Flatten + Gemm)";
        return std::nullopt;
    }
    if (head.relu) {
        error = "a ReLU on the logits is not supported by the integer target";
        return std::nullopt;
    }
    if (head.out == 1) {
        // Decision of a single logit z is z > 0: compare against a constant 0 logit.
        std::vector<double> weight(2 * head.in, 0.0);
        std::copy(head.weight.begin(), head.weight.end(), weight.begin() + static_cast<long>(head.in));
        head.weight = std::move(weight);
        head.bias = {0.0, head.bias[0]};
        head.out = 2;
        head.output = {2, 1, 1, false};
        network.tensors.back() = head.output;
        network.singleLogitPadded = true;
    }
    network.classes = head.out;
    network.inputs = network.tensors[0].size();
    const auto& in = network.tensors[0];
    network.inputShape = in.spatial ? std::vector<std::size_t>{in.c, in.h, in.w} : std::vector<std::size_t>{in.c};
    for (auto& layer : network.layers) buildTaps(layer, network.tensors);
    return network;
}

namespace {

// Concatenation of a layer's source tensors (most layers have one source).
template <typename T>
const std::vector<T>& gather(const std::vector<std::vector<T>>& tensors, const std::vector<std::size_t>& sources,
                             std::vector<T>& scratch) {
    if (sources.size() == 1) return tensors[sources[0]];
    scratch.clear();
    for (std::size_t s : sources) scratch.insert(scratch.end(), tensors[s].begin(), tensors[s].end());
    return scratch;
}

}  // namespace

std::vector<std::vector<double>> referenceActivations(const DenseNetwork& network, const std::vector<double>& x) {
    std::vector<std::vector<double>> tensors(network.layers.size() + 1);
    tensors[0] = x;
    std::vector<double> scratch;
    for (std::size_t l = 0; l < network.layers.size(); ++l) {
        const auto& layer = network.layers[l];
        const auto& in = gather(tensors, layer.sources, scratch);
        auto& next = tensors[l + 1];
        next.assign(layer.out, 0.0);
        for (std::size_t o = 0; o < layer.out; ++o) {
            if (layer.kind == LayerKind::MaxPool) {
                double m = -std::numeric_limits<double>::infinity();
                for (const auto& tap : layer.taps[o]) m = std::max(m, in[tap.input]);
                next[o] = m;
                continue;
            }
            double sum = layer.bias[layer.channelOf(o)];
            for (const auto& tap : layer.taps[o]) sum += layer.weight[tap.weight] * in[tap.input];
            next[o] = layer.relu ? std::max(0.0, sum) : sum;
        }
    }
    return tensors;
}

std::vector<double> referenceLogits(const DenseNetwork& network, const std::vector<double>& x) {
    return referenceActivations(network, x).back();
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
    std::vector<std::vector<std::int64_t>> tensors(program.layers.size() + 1);
    tensors[0] = qInput;
    std::vector<std::int64_t> scratch;
    for (std::size_t l = 0; l < program.layers.size(); ++l) {
        const auto& layer = program.layers[l];
        const auto& current = gather(tensors, layer.sources, scratch);
        auto& next = tensors[l + 1];
        next.assign(layer.out, 0);
        if (layer.kind == LayerKind::MaxPool) {
            for (std::size_t o = 0; o < layer.out; ++o) {
                std::int64_t m = current[layer.taps[o].front().input];
                for (const auto& tap : layer.taps[o]) m = std::max(m, current[tap.input]);
                next[o] = m;
            }
            continue;
        }
        const std::int64_t accLimit = signedMax(layer.accBits);
        const std::int64_t mulLimit = signedMax(layer.mulBits);
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
    }
    run.logits = tensors.back();
    run.decision = argmaxFirst(run.logits);
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
    const auto bytes = [](int bits) { return static_cast<double>(bits <= 8 ? 1 : 2); };
    // The caller's input array, the shared static buffers and the logits.
    cost.ramBytes = static_cast<double>(program.inputs) * bytes(program.inputBits) +
                    static_cast<double>(program.classes) * bytes(program.layers.back().outBits);
    for (std::size_t b = 0; b < program.bufferSize.size(); ++b) {
        for (const auto& layer : program.layers) {
            if (layer.buffer == static_cast<int>(b)) {
                cost.ramBytes += static_cast<double>(program.bufferSize[b]) * bytes(layer.outBits);
                break;
            }
        }
    }
    for (const auto& layer : program.layers) {
        if (layer.kind == LayerKind::MaxPool) continue;
        const double count = static_cast<double>(layer.qWeight.size());
        const double perWeight = layer.weightBits <= 4 ? 0.5 : layer.weightBits <= 8 ? 1.0 : 2.0;
        cost.weightBytes += std::ceil(count * perWeight);
        cost.idealBits += count * layer.weightBits;
        // bias (accumulator type) + multiplier (4) + shift (1) per channel
        cost.flashBytes += static_cast<double>(layer.channels) * (layer.accBits / 8.0 + 4.0 + 1.0);
    }
    cost.flashBytes += cost.weightBytes + 4.0 * static_cast<double>(program.inputs);  // + float input scales
    return cost;
}

void planBuffers(QProgram& program) {
    const std::size_t L = program.layers.size();
    std::vector<std::size_t> last(L + 1, 0);
    for (std::size_t t = 1; t <= L; ++t) last[t] = t - 1;
    for (std::size_t l = 0; l < L; ++l) {
        for (std::size_t s : program.layers[l].sources) last[s] = std::max(last[s], l);
    }
    struct Buffer {
        std::string type;
        std::size_t size = 0;
        std::size_t occupant = 0;  // tensor currently held
    };
    std::vector<Buffer> buffers;
    for (std::size_t l = 0; l < L; ++l) {
        auto& layer = program.layers[l];
        if (l + 1 == L) {
            layer.buffer = -1;  // the logits go to a local array
            break;
        }
        const std::string type = cType(layer.outBits, layer.outUnsigned);
        // A buffer is free once no layer from l on reads its tensor.
        int best = -1;
        for (std::size_t b = 0; b < buffers.size(); ++b) {
            if (buffers[b].type != type || last[buffers[b].occupant] >= l) continue;
            const auto fits = [&](std::size_t k) { return buffers[k].size >= layer.out; };
            if (best < 0 || (fits(b) && (!fits(static_cast<std::size_t>(best)) ||
                                         buffers[b].size < buffers[static_cast<std::size_t>(best)].size)) ||
                (!fits(b) && !fits(static_cast<std::size_t>(best)) &&
                 buffers[b].size > buffers[static_cast<std::size_t>(best)].size)) {
                best = static_cast<int>(b);
            }
        }
        if (best < 0) {
            buffers.push_back({type, 0, 0});
            best = static_cast<int>(buffers.size() - 1);
        }
        auto& buffer = buffers[static_cast<std::size_t>(best)];
        buffer.size = std::max(buffer.size, layer.out);
        buffer.occupant = l + 1;
        layer.buffer = best;
    }
    program.bufferSize.clear();
    for (const auto& buffer : buffers) program.bufferSize.push_back(buffer.size);
}

double estimatedAvrCycles(const QProgram& program) {
    // cycles per MAC [weight kind][input 8/16][mul 16/32][acc 16/32/64]; weight
    // kind 0 = packed 4-bit, 1 = int8, 2 = int16 (simavr, avr-gcc 7.3 -Os).
    static const double mac[3][2][2][3] = {
        {{{34, 40, 62}, {34, 74, 114}}, {{39, 45, 68}, {39, 75, 115}}},
        {{{23, 29, 54}, {23, 64, 105}}, {{28, 34, 59}, {28, 65, 106}}},
        {{{25, 30, 54}, {25, 59, 99}}, {{34, 40, 65}, {34, 65, 106}}},
    };
    constexpr double kRequantise = 605.0;  // 64-bit multiply and rounding shift, per neuron
    double cycles = 0.0;
    for (const auto& layer : program.layers) {
        const int inputBits = layer.sources[0] == 0 ? program.inputBits : program.layers[layer.sources[0] - 1].outBits;
        double taps = 0.0;
        for (const auto& t : layer.taps) taps += static_cast<double>(t.size());
        if (layer.kind == LayerKind::MaxPool) {
            cycles += taps * (inputBits <= 8 ? 14.0 : 20.0);  // load, compare, branch
            continue;
        }
        const int w = layer.weightBits <= 4 ? 0 : layer.weightBits <= 8 ? 1 : 2;
        const int x = inputBits <= 8 ? 0 : 1;
        const int m = layer.mulBits == 16 ? 0 : 1;
        const int a = layer.accBits == 16 ? 0 : layer.accBits == 32 ? 1 : 2;
        // Convolutions pay for the bounds checks of the sliding window.
        const double perMac = mac[w][x][m][a] * (layer.kind == LayerKind::Conv ? 1.3 : 1.0);
        cycles += taps * perMac + static_cast<double>(layer.out) * kRequantise;
    }
    return cycles;
}

double float32Bytes(const DenseNetwork& network) { return 4.0 * static_cast<double>(network.parameterCount()); }

std::size_t multiplyAccumulates(const DenseNetwork& network) {
    std::size_t total = 0;
    for (const auto& layer : network.layers) {
        if (!layer.hasWeights()) continue;
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
           << " * MF-ARTIFACT v4. The arrays below are read verbatim by modelforge_certcheck;\n"
           << " * do not edit them by hand (the certificate would no longer apply). */\n"
           << "#include \"model_q.h\"\n\n"
           << "#ifndef MF_ROM\n#define MF_ROM /* e.g. -DMF_ROM=__flash on AVR keeps tables in flash */\n#endif\n\n";
    // MF_META: version, inputs, classes, input bits, input unsigned, layer count, then per layer
    // kind (0 dense, 1 conv, 2 add, 3 max-pool), in, out, relu, weight bits, acc bits, modular,
    // mul bits, out bits, out unsigned, inC, inH, inW, outC, outH, outW, kH, kW, stride, pad,
    // groups, first source tensor, second source tensor (-1: none), output buffer (-1: logits).
    std::vector<std::int64_t> meta{4, static_cast<std::int64_t>(program.inputs),
                                   static_cast<std::int64_t>(program.classes), program.inputBits,
                                   program.inputUnsigned ? 1 : 0, static_cast<std::int64_t>(program.layers.size())};
    for (const auto& layer : program.layers) {
        const auto& g = layer.conv;
        const auto i64 = [](std::size_t v) { return static_cast<std::int64_t>(v); };
        meta.insert(meta.end(),
                    {static_cast<std::int64_t>(layer.kind), i64(layer.in), i64(layer.out), layer.relu ? 1 : 0,
                     layer.weightBits, layer.accBits, layer.modular ? 1 : 0, layer.mulBits, layer.outBits,
                     layer.outUnsigned ? 1 : 0, i64(g.inC), i64(g.inH), i64(g.inW), i64(g.outC), i64(g.outH),
                     i64(g.outW), i64(g.kH), i64(g.kW), i64(g.stride), i64(g.pad), i64(g.groups),
                     i64(layer.sources[0]), layer.sources.size() > 1 ? i64(layer.sources[1]) : -1, layer.buffer});
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
    source << "static const MF_ROM double MF_SCALE_" << program.layers.size() - 1 << "["
           << program.layers.back().outScale.size() << "] = {";
    for (std::size_t i = 0; i < program.layers.back().outScale.size(); ++i) {
        source << (i ? ", " : " ") << exactDouble(program.layers.back().outScale[i]);
    }
    source << " };\n";
    source << "static const MF_ROM float MF_INPUT_INV_SCALE[" << program.inputs << "] = {";
    for (std::size_t i = 0; i < program.inputs; ++i) {
        source << (i % 4 == 0 ? "\n   " : "") << " " << exactFloat(static_cast<float>(1.0 / program.inputScale[i]))
               << (i + 1 < program.inputs ? "," : "");
    }
    source << "\n};\n\n";
    source << "static int64_t mf_round_shift(int64_t t, int s) {\n"
           << "    int64_t u = t + ((int64_t)1 << (s - 1));\n"
           << "    return u >= 0 ? (u >> s) : -((-u + (((int64_t)1 << s) - 1)) >> s);\n}\n\n";
    if (std::any_of(program.layers.begin(), program.layers.end(),
                    [](const QLayer& q) { return q.kind != LayerKind::MaxPool && q.weightBits <= 4; })) {
        source << "static int8_t mf_nibble(const MF_ROM uint8_t *packed, int k) {\n"
               << "    int v = (k & 1) ? (packed[k >> 1] >> 4) : (packed[k >> 1] & 15);\n"
               << "    return (int8_t)(v >= 8 ? v - 16 : v);\n}\n\n";
    }

    for (std::size_t l = 0; l < program.layers.size(); ++l) {
        const auto& layer = program.layers[l];
        const std::string id = std::to_string(l);
        static const char* const kKindName[] = {"dense", "conv", "add", "max-pool"};
        if (layer.kind == LayerKind::MaxPool) {
            source << "\n/* layer " << l << ": max-pool " << layer.conv.kH << "x" << layer.conv.kW << " stride "
                   << layer.conv.stride << " on integer codes (exact) */\n" << layerFunction(program, l);
            continue;
        }
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
        source << "\n/* layer " << l << ": " << kKindName[static_cast<int>(layer.kind)]
               << (layer.kind == LayerKind::Conv && layer.conv.groups > 1 ? " (groups " + std::to_string(layer.conv.groups) + ")" : "")
               << " " << layer.in
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
    // Static buffers shared by tensors whose live ranges do not overlap.
    for (std::size_t k = 0; k < program.bufferSize.size(); ++k) {
        for (const auto& layer : program.layers) {
            if (layer.buffer != static_cast<int>(k)) continue;
            source << "    static " << cType(layer.outBits, layer.outUnsigned) << " mf_buf" << k << "["
                   << program.bufferSize[k] << "];\n";
            break;
        }
    }
    source << "    mf_logit_t out[MF_CLASSES];\n";
    const auto tensorName = [&](std::size_t t) {
        if (t == 0) return std::string("q");
        const int buffer = program.layers[t - 1].buffer;
        return buffer < 0 ? std::string("out") : "mf_buf" + std::to_string(buffer);
    };
    for (std::size_t l = 0; l < program.layers.size(); ++l) {
        source << "    mf_layer" << l << "(";
        for (std::size_t s : program.layers[l].sources) source << tensorName(s) << ", ";
        source << tensorName(l + 1) << ");\n";
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
    source << "#include \"model_f.h\"\n\n#ifndef MF_ROM\n#define MF_ROM\n#endif\n\n";
    const auto floats = [&](const std::string& name, const std::vector<double>& values) {
        source << "static const MF_ROM float " << name << "[" << values.size() << "] = {";
        for (std::size_t i = 0; i < values.size(); ++i) {
            source << (i % 8 == 0 ? "\n   " : "") << " " << exactFloat(static_cast<float>(values[i]))
                   << (i + 1 < values.size() ? "," : "");
        }
        source << "\n};\n";
    };
    for (std::size_t l = 0; l < network.layers.size(); ++l) {
        const auto& layer = network.layers[l];
        const std::string id = std::to_string(l);
        const std::string activation = layer.relu ? "s > 0.0f ? s : 0.0f" : "s";
        if (layer.hasWeights()) {
            floats("F_W" + id, layer.weight);
            floats("F_B" + id, layer.bias);
        }
        const auto& g = layer.conv;
        if (layer.kind == LayerKind::Add) {
            const std::size_t hw = layer.out / layer.channels();
            source << "static void f_layer" << l << "(const float *a, const float *b, float *y) {\n"
                   << "    for (int c = 0; c < " << layer.channels() << "; ++c)\n"
                   << "    for (int k = 0; k < " << hw << "; ++k) {\n"
                   << "        float s = F_B" << l << "[c] + F_W" << l << "[c * 2] * a[c * " << hw << " + k] + F_W" << l
                   << "[c * 2 + 1] * b[c * " << hw << " + k];\n"
                   << "        y[c * " << hw << " + k] = " << activation << ";\n    }\n}\n";
        } else if (layer.kind == LayerKind::MaxPool) {
            source << "static void f_layer" << l << "(const float *x, float *y) {\n"
                   << "    for (int c = 0; c < " << g.outC << "; ++c)\n"
                   << "    for (int oy = 0; oy < " << g.outH << "; ++oy)\n"
                   << "    for (int ox = 0; ox < " << g.outW << "; ++ox) {\n"
                   << "        float m = x[(c * " << g.inH << " + oy * " << g.stride << ") * " << g.inW << " + ox * "
                   << g.stride << "];\n"
                   << "        for (int ky = 0; ky < " << g.kH << "; ++ky)\n"
                   << "        for (int kx = 0; kx < " << g.kW << "; ++kx) {\n"
                   << "            const float v = x[(c * " << g.inH << " + oy * " << g.stride << " + ky) * " << g.inW
                   << " + ox * " << g.stride << " + kx];\n"
                   << "            if (v > m) m = v;\n        }\n"
                   << "        y[(c * " << g.outH << " + oy) * " << g.outW << " + ox] = m;\n    }\n}\n";
        } else if (layer.kind == LayerKind::Conv) {
            const std::size_t cpg = g.inC / g.groups, opg = g.outC / g.groups;
            source << "static void f_layer" << l << "(const float *x, float *y) {\n"
                   << "    for (int c = 0; c < " << g.outC << "; ++c)\n"
                   << "    for (int oy = 0; oy < " << g.outH << "; ++oy)\n"
                   << "    for (int ox = 0; ox < " << g.outW << "; ++ox) {\n"
                   << "        float s = F_B" << l << "[c];\n"
                   << "        for (int ci = 0; ci < " << cpg << "; ++ci)\n"
                   << "        for (int ky = 0; ky < " << g.kH << "; ++ky) {\n"
                   << "            const int iy = oy * " << g.stride << " + ky - " << g.pad << ";\n"
                   << "            if (iy < 0 || iy >= " << g.inH << ") continue;\n"
                   << "            for (int kx = 0; kx < " << g.kW << "; ++kx) {\n"
                   << "                const int ix = ox * " << g.stride << " + kx - " << g.pad << ";\n"
                   << "                if (ix < 0 || ix >= " << g.inW << ") continue;\n"
                   << "                s += F_W" << l << "[((c * " << cpg << " + ci) * " << g.kH << " + ky) * " << g.kW
                   << " + kx] * x[((c / " << opg << " * " << cpg << " + ci) * " << g.inH << " + iy) * " << g.inW
                   << " + ix];\n"
                   << "            }\n        }\n"
                   << "        y[(c * " << g.outH << " + oy) * " << g.outW << " + ox] = " << activation << ";\n    }\n}\n";
        } else {
            source << "static void f_layer" << l << "(const float *x, float *y) {\n"
                   << "    for (int o = 0; o < " << layer.out << "; ++o) {\n        float s = F_B" << l << "[o];\n"
                   << "        for (int i = 0; i < " << layer.in << "; ++i) s += F_W" << l << "[o * " << layer.in
                   << " + i] * x[i];\n        y[o] = " << activation << ";\n    }\n}\n";
        }
    }
    source << "\nint mf_predict_f(const float *x, float *logits) {\n";
    for (std::size_t l = 0; l + 1 < network.layers.size(); ++l) {
        source << "    static float t" << l << "[" << network.layers[l].out << "];\n";
    }
    source << "    float out[MF_F_CLASSES];\n";
    const auto name = [&](std::size_t t) {
        return t == 0 ? std::string("x") : t == network.layers.size() ? std::string("out") : "t" + std::to_string(t - 1);
    };
    for (std::size_t l = 0; l < network.layers.size(); ++l) {
        source << "    f_layer" << l << "(";
        for (std::size_t s : network.layers[l].sources) source << name(s) << ", ";
        source << name(l + 1) << ");\n";
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

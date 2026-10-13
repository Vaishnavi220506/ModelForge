#include "rewrites.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>

namespace modelforge {
namespace {

int producerOf(const IRGraph& g, const std::string& value) {
    for (std::size_t i = 0; i < g.instructions.size(); ++i) {
        const auto& in = g.instructions[i];
        if (in.output == value && in.operation != IROp::Return) return static_cast<int>(i);
    }
    return -1;
}

std::vector<std::size_t> usesOf(const IRGraph& g, const std::string& value) {
    std::vector<std::size_t> uses;
    for (std::size_t i = 0; i < g.instructions.size(); ++i) {
        for (const auto& name : g.instructions[i].inputs) {
            if (name == value) uses.push_back(i);
        }
    }
    return uses;
}

void replaceUses(IRGraph& g, const std::string& from, const std::string& to) {
    for (auto& in : g.instructions) {
        for (auto& name : in.inputs) {
            if (name == from) name = to;
        }
    }
}

std::string fresh(const IRGraph& g, const std::string& base) {
    std::string name = base;
    for (int k = 1; g.values.count(name) || g.constants.count(name); ++k) name = base + "_" + std::to_string(k);
    return name;
}

void addConstant(IRGraph& g, const std::string& name, const Shape& shape, std::vector<float> data) {
    TensorInfo t;
    t.name = name;
    t.dataType = DataType::Float32;
    t.shape = shape;
    t.isConstant = true;
    t.data = std::move(data);
    g.values[name] = t;
    g.constants[name] = t;
}

void addValue(IRGraph& g, const std::string& name, const Shape& shape) {
    TensorInfo t;
    t.name = name;
    t.dataType = DataType::Float32;
    t.shape = shape;
    g.values[name] = t;
}

// Round to the nearest half-precision value (11 significant bits).
float toHalf(float x) {
    if (x == 0.0f || !std::isfinite(x)) return x;
    int e = 0;
    std::frexp(x, &e);
    const double scale = std::ldexp(1.0, 11 - e);
    return static_cast<float>(std::nearbyint(static_cast<double>(x) * scale) / scale);
}

bool isTensorAdd(const IRGraph& g, const IRInstruction& in) {
    return in.operation == IROp::Add && in.inputs.size() == 2 && !g.constants.count(in.inputs[0]) &&
           !g.constants.count(in.inputs[1]);
}

// ------------------------------------------------------------ Conv + BatchNorm folding
// fault: "none", "epsilon" (wrong attribute read as epsilon), "axis" (scale
// applied along the input-channel axis of the weight), "fp16" (folded
// constants rounded to half precision).
bool foldBatchNorms(IRGraph& g, const std::string& fault, std::vector<std::string>& changed) {
    bool any = false;
    for (std::size_t i = 0; i < g.instructions.size(); ++i) {
        const auto bn = g.instructions[i];
        if (bn.operation != IROp::BatchNorm) continue;
        const int p = producerOf(g, bn.inputs[0]);
        if (p < 0 || g.instructions[static_cast<std::size_t>(p)].operation != IROp::Conv ||
            usesOf(g, bn.inputs[0]).size() != 1) {
            continue;
        }
        auto& conv = g.instructions[static_cast<std::size_t>(p)];
        const auto& w = g.constants.at(conv.inputs[1]);
        const std::size_t M = static_cast<std::size_t>(w.shape[0]), Cg = static_cast<std::size_t>(w.shape[1]);
        const std::size_t K = static_cast<std::size_t>(w.shape[2] * w.shape[3]);
        const auto& gamma = g.constants.at(bn.inputs[1]).data;
        const auto& beta = g.constants.at(bn.inputs[2]).data;
        const auto& mean = g.constants.at(bn.inputs[3]).data;
        const auto& var = g.constants.at(bn.inputs[4]).data;
        // PyTorch #102224: the fold read another BatchNorm argument (here the
        // default momentum, 0.1) as epsilon.
        const float eps = fault == "epsilon" ? 0.1f : bn.epsilon;
        std::vector<float> a(M), weight(w.data.size()), bias(M);
        for (std::size_t c = 0; c < M; ++c) a[c] = gamma[c] / std::sqrt(var[c] + eps);
        for (std::size_t k = 0; k < weight.size(); ++k) {
            // PyTorch #70022 class: the per-channel scale indexed along the
            // wrong weight axis (input channel within the group).
            const std::size_t channel = fault == "axis" ? (k / K) % Cg % M : k / (Cg * K);
            weight[k] = w.data[k] * a[channel];
        }
        const std::vector<float>* oldBias = conv.inputs.size() == 3 ? &g.constants.at(conv.inputs[2]).data : nullptr;
        for (std::size_t c = 0; c < M; ++c) bias[c] = (oldBias ? (*oldBias)[c] : 0.0f) * a[c] + beta[c] - mean[c] * a[c];
        if (fault == "fp16") {
            for (float& v : weight) v = toHalf(v);
            for (float& v : bias) v = toHalf(v);
        }
        const std::string wName = fresh(g, conv.inputs[1] + "_bnfold");
        const std::string bName = fresh(g, conv.nodeName + "_bias_bnfold");
        addConstant(g, wName, w.shape, std::move(weight));
        addConstant(g, bName, {static_cast<std::int64_t>(M)}, std::move(bias));
        changed.push_back(conv.nodeName);
        changed.push_back(bn.nodeName);
        conv.inputs = {conv.inputs[0], wName, bName};
        conv.output = bn.output;
        g.instructions.erase(g.instructions.begin() + static_cast<long>(i));
        --i;
        any = true;
    }
    return any;
}

}  // namespace

std::vector<RewriteCase> makeRewriteCases(const IRGraph& graph) {
    std::vector<RewriteCase> cases;
    const auto add = [&](const std::string& pass, const std::string& fault, const std::string& source, bool control,
                         IRGraph candidate, std::vector<std::string> changed) {
        RewriteCase c;
        c.pass = pass;
        c.fault = fault;
        c.source = source;
        c.control = control;
        c.candidate = std::move(candidate);
        c.changedNodes = std::move(changed);
        cases.push_back(std::move(c));
    };

    // Conv + BatchNorm folding.
    for (const auto& [fault, source] : std::vector<std::pair<std::string, std::string>>{
             {"none", ""},
             {"epsilon", "PyTorch #102224 (wrong epsilon argument in Conv+BN folding)"},
             {"axis", "PyTorch #70022 (BN scale reshaped along the wrong weight axis)"},
             {"fp16", "half-precision constant folding"}}) {
        IRGraph g = graph;
        std::vector<std::string> changed;
        if (foldBatchNorms(g, fault, changed)) add("conv_bn_fold", fault, source, fault == "none", g, changed);
    }

    // Activation fusion into a dense layer: correct (FusedGemmRelu) and a
    // fusion that drops the activation of one layer.
    {
        IRGraph fused = graph;
        std::vector<std::string> changedAll;
        for (std::size_t i = 0; i < graph.instructions.size(); ++i) {
            const auto& relu = graph.instructions[i];
            if (relu.operation != IROp::Relu) continue;
            const int p = producerOf(graph, relu.inputs[0]);
            if (p < 0 || graph.instructions[static_cast<std::size_t>(p)].operation != IROp::Gemm ||
                usesOf(graph, relu.inputs[0]).size() != 1) {
                continue;
            }
            const auto& gemm = graph.instructions[static_cast<std::size_t>(p)];
            // Correct: fuse every such pair.
            for (auto& in : fused.instructions) {
                if (in.nodeName == gemm.nodeName) {
                    in.operation = IROp::FusedGemmRelu;
                    in.output = relu.output;
                }
            }
            fused.instructions.erase(std::remove_if(fused.instructions.begin(), fused.instructions.end(),
                                                    [&](const IRInstruction& in) { return in.nodeName == relu.nodeName; }),
                                     fused.instructions.end());
            changedAll.push_back(gemm.nodeName);
            changedAll.push_back(relu.nodeName);
            // Faulty: this layer's activation is dropped.
            IRGraph dropped = graph;
            replaceUses(dropped, relu.output, relu.inputs[0]);
            dropped.instructions.erase(dropped.instructions.begin() + static_cast<long>(i));
            add("gemm_relu_fusion", "activation_dropped@" + gemm.nodeName,
                "TFLite #57818 (activation term lost when fusing into the next op)", false, dropped,
                {gemm.nodeName, relu.nodeName});
        }
        if (!changedAll.empty()) add("gemm_relu_fusion", "none", "", true, fused, changedAll);
    }

    // Fused-activation mistakes that only show on part of the input space:
    // a ReLU lowered as ReLU6 (clamped at 6; the Relu+Clip fusion bugs of
    // ONNX Runtime #9753 / #10936) and a quantised ReLU whose threshold is
    // off by a small amount (a dead zone just above 0).
    for (std::size_t i = 0; i < graph.instructions.size(); ++i) {
        const auto& relu = graph.instructions[i];
        if (relu.operation != IROp::Relu) continue;
        const int p = producerOf(graph, relu.inputs[0]);
        if (p < 0) continue;
        const auto producer = graph.instructions[static_cast<std::size_t>(p)].operation;
        if (producer != IROp::Conv && producer != IROp::Gemm && producer != IROp::Add && producer != IROp::BatchNorm) {
            continue;
        }
        for (const auto& [op, name, parameter, source] :
             std::vector<std::tuple<IROp, std::string, float, std::string>>{
                 {IROp::TestReluClamp, "relu_as_relu6@", 6.0f,
                  "ONNX Runtime #9753 / #10936 (Relu+Clip fusion; ReLU lowered as ReLU6)"},
                 {IROp::TestReluDeadZone, "relu_threshold@", 0.05f, "quantised ReLU threshold off by a small amount"}}) {
            IRGraph g = graph;
            g.instructions[i].operation = op;
            g.instructions[i].testFaultParameter = parameter;
            add("activation_lowering", name + relu.nodeName, source, false, g, {relu.nodeName});
        }
    }

    // Commuting ReLU with a following pooling layer: exact for max-pooling,
    // wrong for average pooling.
    for (std::size_t i = 0; i < graph.instructions.size(); ++i) {
        const auto& relu = graph.instructions[i];
        if (relu.operation != IROp::Relu) continue;
        const auto uses = usesOf(graph, relu.output);
        if (uses.size() != 1) continue;
        const auto& pool = graph.instructions[uses[0]];
        if (pool.operation != IROp::MaxPool && pool.operation != IROp::AveragePool) continue;
        IRGraph g = graph;
        const std::string mid = fresh(g, pool.output + "_prepool");
        addValue(g, mid, pool.shape);
        IRInstruction newPool = pool, newRelu = relu;
        newPool.inputs = {relu.inputs[0]};
        newPool.output = mid;
        newRelu.inputs = {mid};
        newRelu.output = pool.output;
        newRelu.shape = pool.shape;
        g.instructions[i] = newPool;
        g.instructions[uses[0]] = newRelu;
        const bool exact = pool.operation == IROp::MaxPool;
        add("relu_pool_commute", (exact ? "none@" : "relu_through_avgpool@") + pool.nodeName,
            exact ? "" : "synthetic (commuting rule valid only for max-pooling)", exact, g,
            {relu.nodeName, pool.nodeName});
    }

    // Residual additions: dead-code elimination that drops the skip branch,
    // and an Add+ReLU fusion that applies the ReLU to one operand.
    for (std::size_t i = 0; i < graph.instructions.size(); ++i) {
        const auto& sum = graph.instructions[i];
        if (!isTensorAdd(graph, sum)) continue;
        const int p0 = producerOf(graph, sum.inputs[0]), p1 = producerOf(graph, sum.inputs[1]);
        const std::size_t skip = p0 < p1 ? 0 : 1, main = 1 - skip;
        {
            IRGraph g = graph;
            replaceUses(g, sum.output, sum.inputs[main]);
            g.instructions.erase(g.instructions.begin() + static_cast<long>(i));
            add("dead_code_elimination", "live_branch_removed@" + sum.nodeName,
                "ONNX Runtime #13425 (a needed value optimized out)", false, g, {sum.nodeName});
        }
        const auto uses = usesOf(graph, sum.output);
        if (uses.size() == 1 && graph.instructions[uses[0]].operation == IROp::Relu) {
            const auto& relu = graph.instructions[uses[0]];
            IRGraph g = graph;
            const std::string act = fresh(g, sum.inputs[main] + "_relu");
            addValue(g, act, graph.values.at(sum.inputs[main]).shape);
            IRInstruction early = relu;
            early.nodeName = relu.nodeName + "_moved";
            early.inputs = {sum.inputs[main]};
            early.output = act;
            early.shape = graph.values.at(sum.inputs[main]).shape;
            IRInstruction newSum = sum;
            newSum.inputs[main] = act;
            newSum.output = relu.output;
            g.instructions[uses[0]] = newSum;                        // Add takes the ReLU's place
            g.instructions[i] = early;                               // ReLU on one operand, before the Add
            add("add_relu_fusion", "relu_on_operand@" + sum.nodeName,
                "TFLite #57818 (activation applied to the wrong term around an Add)", false, g,
                {sum.nodeName, relu.nodeName});
        }
    }

    // Convolution padding lowered with non-zero padded cells.
    {
        IRGraph g = graph;
        std::vector<std::string> changed;
        for (auto& in : g.instructions) {
            if (in.operation == IROp::Conv && in.pad > 0) {
                in.padMode = 1;
                changed.push_back(in.nodeName);
            }
        }
        if (!changed.empty()) {
            add("conv_padding", "edge_padding",
                "ONNX Runtime #11828 / PyTorch #86556 (wrong values in padded cells)", false, g, changed);
        }
    }

    // Layout change to NHWC before Flatten without permuting the next dense
    // layer's weights.
    for (std::size_t i = 0; i < graph.instructions.size(); ++i) {
        const auto& flat = graph.instructions[i];
        if (flat.operation != IROp::Flatten) continue;
        const auto& in = graph.values.at(flat.inputs[0]).shape;
        if (in.size() != 4 || in[2] * in[3] <= 1) continue;
        const auto uses = usesOf(graph, flat.output);
        if (uses.size() != 1 || graph.instructions[uses[0]].operation != IROp::Gemm) continue;
        const auto& gemm = graph.instructions[uses[0]];
        const auto& w = graph.constants.at(gemm.inputs[1]);
        const std::size_t C = static_cast<std::size_t>(in[1]), HW = static_cast<std::size_t>(in[2] * in[3]);
        const std::size_t n = C * HW;
        const std::size_t outs = w.data.size() / n;
        std::vector<float> permuted(w.data.size());
        for (std::size_t c = 0; c < C; ++c) {
            for (std::size_t s = 0; s < HW; ++s) {
                const std::size_t nchw = c * HW + s, nhwc = s * C + c;
                for (std::size_t o = 0; o < outs; ++o) {
                    // weights are [in, out] unless transB ([out, in])
                    if (gemm.transB) {
                        permuted[o * n + nchw] = w.data[o * n + nhwc];
                    } else {
                        permuted[nchw * outs + o] = w.data[nhwc * outs + o];
                    }
                }
            }
        }
        IRGraph g = graph;
        const std::string name = fresh(g, gemm.inputs[1] + "_nhwc");
        addConstant(g, name, w.shape, std::move(permuted));
        g.instructions[uses[0]].inputs[1] = name;
        add("layout_nhwc", "flatten_order@" + gemm.nodeName,
            "TVM PR #10593 / #10156 (Flatten and layout conversion)", false, g, {flat.nodeName, gemm.nodeName});
    }

    // Constant folding in half precision.
    {
        IRGraph g = graph;
        std::vector<std::string> changed;
        for (const auto& in : graph.instructions) {
            if (in.operation != IROp::Gemm && in.operation != IROp::Conv && in.operation != IROp::MatMul) continue;
            for (std::size_t k = 1; k < in.inputs.size(); ++k) {
                auto it = g.constants.find(in.inputs[k]);
                if (it == g.constants.end()) continue;
                for (float& v : it->second.data) v = toHalf(v);
                g.values[in.inputs[k]].data = it->second.data;
            }
            changed.push_back(in.nodeName);
        }
        if (!changed.empty()) add("constant_folding", "fp16_constants", "half-precision constant folding", false, g, changed);
    }
    return cases;
}

std::string irToManifest(const IRGraph& graph) {
    std::ostringstream out;
    out << std::setprecision(9);
    const auto dims = [](const Shape& shape) {
        std::string text;
        for (std::size_t k = 0; k < shape.size(); ++k) text += (k ? "," : "") + std::to_string(shape[k]);
        return text.empty() ? std::string("1") : text;
    };
    out << "model " << graph.name << "\n";
    out << "input " << graph.inputName << " float32 " << dims(graph.values.at(graph.inputName).shape) << "\n";
    out << "output " << graph.outputName << " float32 " << dims(graph.values.at(graph.outputName).shape) << "\n";
    std::set<std::string> used;
    for (const auto& in : graph.instructions) {
        for (const auto& name : in.inputs) {
            if (graph.constants.count(name)) used.insert(name);
        }
    }
    for (const auto& name : used) {
        const auto& t = graph.constants.at(name);
        out << "tensor " << name << " float32 " << dims(t.shape) << " values=";
        for (std::size_t k = 0; k < t.data.size(); ++k) out << (k ? "," : "") << t.data[k];
        out << "\n";
    }
    std::map<IROp, std::string> names{{IROp::Gemm, "Gemm"},       {IROp::MatMul, "MatMul"},
                                      {IROp::Add, "Add"},         {IROp::Relu, "Relu"},
                                      {IROp::Sigmoid, "Sigmoid"}, {IROp::Softmax, "Softmax"},
                                      {IROp::Conv, "Conv"},       {IROp::Flatten, "Flatten"},
                                      {IROp::BatchNorm, "BatchNormalization"},
                                      {IROp::AveragePool, "AveragePool"},
                                      {IROp::GlobalAveragePool, "GlobalAveragePool"},
                                      {IROp::MaxPool, "MaxPool"}};
    for (const auto& in : graph.instructions) {
        if (in.operation == IROp::Input || in.operation == IROp::Return) continue;
        std::string op = in.operation == IROp::FusedGemmRelu ? "Gemm" : names.count(in.operation) ? names[in.operation] : "";
        // Study-only activations (understood by research/rewrite_oracle.py).
        if (in.operation == IROp::TestReluClamp) op = "ReluClamp";
        if (in.operation == IROp::TestReluDeadZone) op = "ReluDeadZone";
        if (op.empty()) continue;
        const std::string output = in.operation == IROp::FusedGemmRelu ? in.output + "_prerelu" : in.output;
        out << "node " << in.nodeName << " " << op;
        for (const auto& name : in.inputs) out << " " << name;
        out << " -> " << output;
        if (in.transB) out << " transB=1";
        if (in.operation == IROp::Conv) {
            out << " strides=" << in.stride << " pads=" << in.pad << " group=" << in.group;
            if (in.padMode == 1) out << " padmode=edge";
        }
        if (in.operation == IROp::AveragePool || in.operation == IROp::MaxPool) {
            out << " kernel=" << in.kernel << " strides=" << in.stride;
        }
        if (in.operation == IROp::BatchNorm) out << " epsilon=" << in.epsilon;
        if (in.operation == IROp::Softmax) out << " axis=" << (in.axis < 0 ? 1 : in.axis);
        if (in.operation == IROp::TestReluClamp || in.operation == IROp::TestReluDeadZone) {
            out << " parameter=" << in.testFaultParameter;
        }
        out << "\n";
        if (in.operation == IROp::FusedGemmRelu) out << "node " << in.nodeName << "_relu Relu " << output << " -> " << in.output << "\n";
    }
    return out.str();
}

}  // namespace modelforge

#pragma once

#include "diagnostics.h"
#include "model.h"

#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace modelforge {

enum class IROp {
    Input,
    Gemm,
    MatMul,
    Add,
    Relu,
    // Fault injection only; never accepted from a model or emitted as C++.
    TestReluDeadZone,  // 0 on (0, p): a narrow band at the kink
    TestReluClamp,     // min(relu(z), p): a wrong ReLU6-style fusion
    TestLeakyRelu,     // z < 0 ? p * z : z: a wrong activation lowering
    Sigmoid,
    Softmax,
    FusedGemmRelu,
    Conv,     // NCHW, batch 1, zero padding `pad`, stride `stride`, `group` groups
    Flatten,  // [1, ...] -> [1, n]; the data order is unchanged
    BatchNorm,          // inputs x, scale, bias, mean, var (per channel)
    AveragePool,        // `kernel` x `kernel` window, stride, no padding
    GlobalAveragePool,  // [1,C,H,W] -> [1,C,1,1]
    MaxPool,            // `kernel` x `kernel` window, stride, no padding
    Return
};

std::string irOpName(IROp operation);

struct IRInstruction {
    IROp operation;
    std::string nodeName;
    std::vector<std::string> inputs;
    std::string output;
    Shape shape;
    int axis = -1;
    bool transA = false;
    bool transB = false;
    // Parameter p of the test-only fault operators above.
    float testFaultParameter = 0.1f;
    int stride = 1;  // Conv, pooling
    int pad = 0;     // Conv, pooling
    int group = 1;   // Conv
    int kernel = 0;  // pooling window
    float epsilon = 1e-5f;  // BatchNormalization
};

struct IRGraph {
    std::string name;
    std::string inputName;
    std::string outputName;
    std::unordered_map<std::string, TensorInfo> values;
    std::unordered_map<std::string, TensorInfo> constants;
    std::vector<IRInstruction> instructions;
};

std::optional<IRGraph> buildIR(const ModelGraph& model, DiagnosticEngine& diagnostics);
void printIR(const IRGraph& graph, std::ostream& output);

}  // namespace modelforge

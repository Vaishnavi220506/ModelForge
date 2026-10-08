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

#include "optimizer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <utility>

namespace modelforge {
namespace {

bool isConstant(const IRGraph& graph, const std::string& name) {
    return graph.constants.find(name) != graph.constants.end();
}

bool foldConstantInstruction(IRGraph& graph,
                             const IRInstruction& instruction,
                             std::vector<std::string>& events) {
    auto saveConstant = [&](std::vector<float> values) {
        TensorInfo value = graph.values[instruction.output];
        value.name = instruction.output;
        value.isConstant = true;
        value.hasProducer = false;
        value.producer = "constant-folding";
        value.data = std::move(values);
        graph.values[instruction.output] = value;
        graph.constants[instruction.output] = value;
    };

    if (instruction.operation == IROp::Add && instruction.inputs.size() == 2 &&
        isConstant(graph, instruction.inputs[0]) && isConstant(graph, instruction.inputs[1])) {
        const auto& left = graph.constants.at(instruction.inputs[0]).data;
        const auto& right = graph.constants.at(instruction.inputs[1]).data;
        if (left.size() != right.size()) {
            return false;
        }
        std::vector<float> result(left.size());
        for (std::size_t index = 0; index < result.size(); ++index) {
            result[index] = left[index] + right[index];
        }
        saveConstant(std::move(result));
        events.push_back("Constant-folded ADD '" + instruction.nodeName +
                         "' into '" + instruction.output + "'");
        return true;
    }

    if (instruction.inputs.size() != 1 || !isConstant(graph, instruction.inputs[0])) {
        return false;
    }
    const auto& input = graph.constants.at(instruction.inputs[0]).data;
    std::vector<float> result(input.size());
    if (instruction.operation == IROp::Relu) {
        for (std::size_t index = 0; index < input.size(); ++index) {
            result[index] = std::max(0.0f, input[index]);
        }
    } else if (instruction.operation == IROp::Sigmoid) {
        for (std::size_t index = 0; index < input.size(); ++index) {
            result[index] = 1.0f / (1.0f + std::exp(-input[index]));
        }
    } else if (instruction.operation == IROp::Softmax) {
        if (input.empty()) {
            return false;
        }
        const float maximum = *std::max_element(input.begin(), input.end());
        float total = 0.0f;
        for (std::size_t index = 0; index < input.size(); ++index) {
            result[index] = std::exp(input[index] - maximum);
            total += result[index];
        }
        if (total == 0.0f) {
            return false;
        }
        for (float& value : result) {
            value /= total;
        }
    } else {
        return false;
    }
    saveConstant(std::move(result));
    events.push_back("Constant-folded " + irOpName(instruction.operation) +
                     " '" + instruction.nodeName + "' into '" + instruction.output + "'");
    return true;
}

std::size_t countUses(const std::vector<IRInstruction>& instructions,
                      const std::string& value) {
    std::size_t count = 0;
    for (const IRInstruction& instruction : instructions) {
        count += static_cast<std::size_t>(std::count(instruction.inputs.begin(),
                                                      instruction.inputs.end(), value));
    }
    return count;
}

std::size_t fuseDenseRelu(IRGraph& graph, std::vector<std::string>& events) {
    std::vector<IRInstruction> result;
    std::size_t fused = 0;
    for (std::size_t index = 0; index < graph.instructions.size(); ++index) {
        if (index + 1 < graph.instructions.size() &&
            graph.instructions[index].operation == IROp::Gemm &&
            graph.instructions[index + 1].operation == IROp::Relu &&
            graph.instructions[index + 1].inputs.size() == 1 &&
            graph.instructions[index].output == graph.instructions[index + 1].inputs[0] &&
            countUses(graph.instructions, graph.instructions[index].output) == 1) {
            IRInstruction instruction = graph.instructions[index];
            const IRInstruction& relu = graph.instructions[index + 1];
            instruction.operation = IROp::FusedGemmRelu;
            instruction.nodeName += "+" + relu.nodeName;
            instruction.output = relu.output;
            instruction.shape = relu.shape;
            result.push_back(std::move(instruction));
            ++fused;
            events.push_back("Fused GEMM '" + graph.instructions[index].nodeName +
                             "' with ReLU '" + relu.nodeName + "'");
            ++index;
        } else {
            result.push_back(graph.instructions[index]);
        }
    }
    graph.instructions = std::move(result);
    return fused;
}

std::size_t removeDeadNodes(IRGraph& graph, std::vector<std::string>& events) {
    std::unordered_set<std::string> required;
    required.insert(graph.outputName);
    std::vector<IRInstruction> kept;

    for (auto iterator = graph.instructions.rbegin(); iterator != graph.instructions.rend(); ++iterator) {
        const IRInstruction& instruction = *iterator;
        const bool isReturn = instruction.operation == IROp::Return;
        const bool isInput = instruction.operation == IROp::Input;
        const bool needed = isReturn || required.count(instruction.output) != 0;
        if (!needed) {
            events.push_back("Removed dead instruction '" + instruction.nodeName +
                             "' producing '" + instruction.output + "'");
            continue;
        }
        kept.push_back(instruction);
        for (const std::string& input : instruction.inputs) {
            if (!isConstant(graph, input)) {
                required.insert(input);
            }
        }
        if (isInput) {
            required.insert(instruction.output);
        }
    }

    std::reverse(kept.begin(), kept.end());
    const std::size_t removed = graph.instructions.size() - kept.size();
    graph.instructions = std::move(kept);
    return removed;
}

}  // namespace

const char* optimizationPassName(OptimizationPass pass) {
    switch (pass) {
        case OptimizationPass::ConstantFolding: return "constant_folding";
        case OptimizationPass::DenseReluFusion: return "dense_relu_fusion";
        case OptimizationPass::DeadNodeRemoval: return "dead_node_removal";
    }
    return "unknown";
}

OptimizationReport runOptimizationPass(IRGraph& graph, OptimizationPass pass) {
    OptimizationReport report;
    report.instructionsBefore = graph.instructions.size();
    if (pass == OptimizationPass::ConstantFolding) {
        std::vector<IRInstruction> foldedInstructions;
        for (const IRInstruction& instruction : graph.instructions) {
            if (instruction.operation == IROp::Input || instruction.operation == IROp::Return ||
                !foldConstantInstruction(graph, instruction, report.events)) {
                foldedInstructions.push_back(instruction);
            } else {
                ++report.constantFolds;
            }
        }
        graph.instructions = std::move(foldedInstructions);
    } else if (pass == OptimizationPass::DenseReluFusion) {
        report.fusedOperations = fuseDenseRelu(graph, report.events);
    } else if (pass == OptimizationPass::DeadNodeRemoval) {
        report.removedInstructions = removeDeadNodes(graph, report.events);
    }
    report.instructionsAfter = graph.instructions.size();
    return report;
}

OptimizationReport optimize(IRGraph& graph, DiagnosticEngine& diagnostics) {
    (void)diagnostics;
    OptimizationReport total;
    total.instructionsBefore = graph.instructions.size();
    for (const auto pass : {OptimizationPass::ConstantFolding,
                            OptimizationPass::DenseReluFusion,
                            OptimizationPass::DeadNodeRemoval}) {
        const auto report = runOptimizationPass(graph, pass);
        total.removedInstructions += report.removedInstructions;
        total.fusedOperations += report.fusedOperations;
        total.constantFolds += report.constantFolds;
        total.events.insert(total.events.end(), report.events.begin(), report.events.end());
    }
    total.instructionsAfter = graph.instructions.size();
    return total;
}

}  // namespace modelforge

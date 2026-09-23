#include "ir.h"

#include <iostream>

namespace modelforge {

std::string irOpName(IROp operation) {
    switch (operation) {
        case IROp::Input:
            return "INPUT";
        case IROp::Gemm:
            return "GEMM";
        case IROp::MatMul:
            return "MATMUL";
        case IROp::Add:
            return "ADD";
        case IROp::Relu:
            return "RELU";
        case IROp::Sigmoid:
            return "SIGMOID";
        case IROp::Softmax:
            return "SOFTMAX";
        case IROp::FusedGemmRelu:
            return "FUSED_GEMM_RELU";
        case IROp::Return:
            return "RETURN";
    }
    return "UNKNOWN";
}

namespace {

bool convertOp(const std::string& operation, IROp& result) {
    if (operation == "Gemm") {
        result = IROp::Gemm;
    } else if (operation == "MatMul") {
        result = IROp::MatMul;
    } else if (operation == "Add") {
        result = IROp::Add;
    } else if (operation == "Relu") {
        result = IROp::Relu;
    } else if (operation == "Sigmoid") {
        result = IROp::Sigmoid;
    } else if (operation == "Softmax") {
        result = IROp::Softmax;
    } else {
        return false;
    }
    return true;
}

}  // namespace

std::optional<IRGraph> buildIR(const ModelGraph& model,
                               DiagnosticEngine& diagnostics) {
    IRGraph graph;
    graph.name = model.name;
    graph.inputName = model.inputName;
    graph.outputName = model.outputName;
    graph.values = model.tensors;

    for (const auto& entry : model.tensors) {
        if (entry.second.isConstant) {
            graph.constants.emplace(entry.first, entry.second);
        }
    }

    const auto inputIterator = model.tensors.find(model.inputName);
    if (inputIterator == model.tensors.end()) {
        diagnostics.error("ir", "Cannot create IR input because the model input is missing");
        return std::nullopt;
    }
    graph.instructions.push_back({IROp::Input,
                                  "input",
                                  {},
                                  model.inputName,
                                  inputIterator->second.shape,
                                  -1,
                                  false,
                                  false});

    for (const Node& node : model.nodes) {
        IROp operation;
        if (!convertOp(node.op, operation)) {
            diagnostics.error("ir", "Cannot lower unsupported operator '" + node.op + "'");
            return std::nullopt;
        }
        const auto valueIterator = model.tensors.find(node.output);
        if (valueIterator == model.tensors.end()) {
            diagnostics.error("ir", "Missing shape information for output '" + node.output + "'");
            return std::nullopt;
        }
        graph.instructions.push_back({operation,
                                      node.name,
                                      node.inputs,
                                      node.output,
                                      valueIterator->second.shape,
                                      node.axis,
                                      node.transA,
                                      node.transB});
    }

    const auto outputIterator = model.tensors.find(model.outputName);
    if (outputIterator == model.tensors.end()) {
        diagnostics.error("ir", "Cannot create IR return because the model output is missing");
        return std::nullopt;
    }
    graph.instructions.push_back({IROp::Return,
                                  "return",
                                  {model.outputName},
                                  model.outputName,
                                  outputIterator->second.shape,
                                  -1,
                                  false,
                                  false});

    return graph;
}

void printIR(const IRGraph& graph, std::ostream& output) {
    output << "MODEL FORGE IR\n";
    output << "===============\n";
    output << "Model: " << graph.name << "\n";
    output << "Input: " << graph.inputName << "\n";
    output << "Output: " << graph.outputName << "\n\n";
    output << "Constants: " << graph.constants.size() << "\n";
    for (const auto& entry : graph.constants) {
        output << "  " << entry.first << " shape=" << entry.second.shape.size() << "D\n";
    }
    output << "\nInstructions:\n";
    for (std::size_t index = 0; index < graph.instructions.size(); ++index) {
        const IRInstruction& instruction = graph.instructions[index];
        output << "  " << index << ": " << irOpName(instruction.operation);
        if (!instruction.nodeName.empty()) {
            output << " [" << instruction.nodeName << "]";
        }
        if (!instruction.inputs.empty()) {
            output << " inputs=";
            for (std::size_t inputIndex = 0; inputIndex < instruction.inputs.size(); ++inputIndex) {
                if (inputIndex != 0) {
                    output << ",";
                }
                output << instruction.inputs[inputIndex];
            }
        }
        if (!instruction.output.empty()) {
            output << " -> " << instruction.output;
        }
        if (instruction.axis >= 0) {
            output << " axis=" << instruction.axis;
        }
        if (instruction.transA) {
            output << " transA=1";
        }
        if (instruction.transB) {
            output << " transB=1";
        }
        output << "\n";
    }
    output << '\n';
}

}  // namespace modelforge

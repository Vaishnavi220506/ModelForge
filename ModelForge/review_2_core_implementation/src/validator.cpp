#include "validator.h"

#include <algorithm>
#include <cmath>
#include <ostream>
#include <sstream>
#include <unordered_set>

namespace modelforge {
namespace {

const std::unordered_set<std::string> kSupportedOperators = {
    "Gemm", "MatMul", "Add", "Relu", "Sigmoid", "Softmax"};

bool sameShape(const Shape& left, const Shape& right) {
    return left == right;
}

std::string shapeText(const Shape& shape) {
    std::ostringstream output;
    output << "[";
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) {
            output << " x ";
        }
        output << shape[index];
    }
    output << "]";
    return output.str();
}

}  // namespace

bool SymbolTable::insert(const TensorInfo& symbol) {
    return symbols_.emplace(symbol.name, symbol).second;
}

bool SymbolTable::contains(const std::string& name) const {
    return symbols_.find(name) != symbols_.end();
}

TensorInfo* SymbolTable::lookup(const std::string& name) {
    const auto iterator = symbols_.find(name);
    return iterator == symbols_.end() ? nullptr : &iterator->second;
}

const TensorInfo* SymbolTable::lookup(const std::string& name) const {
    const auto iterator = symbols_.find(name);
    return iterator == symbols_.end() ? nullptr : &iterator->second;
}

std::size_t SymbolTable::size() const {
    return symbols_.size();
}

void printSymbolTable(const SymbolTable& symbols, std::ostream& output) {
    std::vector<const TensorInfo*> entries;
    entries.reserve(symbols.symbols_.size());
    for (const auto& entry : symbols.symbols_) {
        entries.push_back(&entry.second);
    }
    std::sort(entries.begin(), entries.end(), [](const TensorInfo* left, const TensorInfo* right) {
        return left->name < right->name;
    });

    output << "\nSYMBOL TABLE\n"
           << "============\n"
           << "Name                 Type       Shape          Kind       Producer\n";
    for (const TensorInfo* symbol : entries) {
        const std::string kind = symbol->isConstant ? "constant" :
                                 symbol->isInput ? "input" : "value";
        output << symbol->name << "\t"
               << dataTypeName(symbol->dataType) << "\t"
               << shapeText(symbol->shape) << "\t"
               << kind << "\t"
               << (symbol->producer.empty() ? "-" : symbol->producer) << "\n";
    }
    output << "\n";
}

bool validate(ModelGraph& model,
              DiagnosticEngine& diagnostics,
              SymbolTable& symbols) {
    symbols = {};
    for (const auto& entry : model.tensors) {
        symbols.insert(entry.second);
        const TensorInfo& tensor = entry.second;
        if (tensor.dataType == DataType::Unknown) {
            diagnostics.error("semantic", "Unknown datatype for tensor '" + tensor.name + "'");
        }
        if (tensor.shape.empty() || elementCount(tensor.shape) == 0) {
            diagnostics.error("semantic", "Invalid shape for tensor '" + tensor.name + "'");
        }
        if (tensor.isConstant && tensor.data.size() != elementCount(tensor.shape)) {
            diagnostics.error("semantic",
                              "Constant tensor '" + tensor.name + "' has " +
                                  std::to_string(tensor.data.size()) +
                                  " values but needs " +
                                  std::to_string(elementCount(tensor.shape)));
        }
    }

    if (model.inputName.empty() || model.tensors.find(model.inputName) == model.tensors.end()) {
        diagnostics.error("semantic", "Declared model input is missing");
    }
    if (model.outputName.empty() || model.tensors.find(model.outputName) == model.tensors.end()) {
        diagnostics.error("semantic", "Declared model output is missing");
    }

    std::unordered_set<std::string> available;
    for (const auto& entry : model.tensors) {
        if (entry.second.isConstant) {
            available.insert(entry.first);
        }
    }
    if (!model.inputName.empty()) {
        available.insert(model.inputName);
    }

    for (Node& node : model.nodes) {
        if (!kSupportedOperators.count(node.op)) {
            diagnostics.error("semantic",
                              "Unsupported operator '" + node.op + "' in node '" + node.name + "'",
                              node.sourceLine);
            continue;
        }

        for (const std::string& inputName : node.inputs) {
            if (!available.count(inputName)) {
                diagnostics.error("semantic",
                                  "Input tensor '" + inputName + "' is not available before node '" +
                                      node.name + "'",
                                  node.sourceLine);
            }
        }

        const auto outputIterator = model.tensors.find(node.output);
        if (outputIterator != model.tensors.end() &&
            (outputIterator->second.isInput || outputIterator->second.isConstant ||
             outputIterator->second.hasProducer)) {
            diagnostics.error("semantic",
                              "Duplicate or read-only output tensor '" + node.output + "'",
                              node.sourceLine);
            continue;
        }

        auto getInput = [&](std::size_t index) -> const TensorInfo* {
            if (index >= node.inputs.size()) {
                return nullptr;
            }
            const auto iterator = model.tensors.find(node.inputs[index]);
            return iterator == model.tensors.end() ? nullptr : &iterator->second;
        };

        bool validNode = true;
        Shape outputShape;
        const TensorInfo* first = getInput(0);

        if (node.op == "Gemm" || node.op == "MatMul") {
            const bool inputCountValid = node.op == "Gemm"
                                              ? (node.inputs.size() == 2 || node.inputs.size() == 3)
                                              : node.inputs.size() == 2;
            if (!inputCountValid) {
                diagnostics.error("semantic",
                                  node.op == "Gemm"
                                      ? "Gemm expects two inputs and an optional bias"
                                      : "MatMul expects exactly two inputs",
                                  node.sourceLine);
                validNode = false;
            }
            if (node.op == "MatMul" && (node.transA || node.transB)) {
                diagnostics.error("semantic", "MatMul transpose attributes are not supported", node.sourceLine);
                validNode = false;
            }
            if (node.op == "Gemm" &&
                (std::fabs(node.alpha - 1.0f) > 1.0e-6f || std::fabs(node.beta - 1.0f) > 1.0e-6f)) {
                diagnostics.error("semantic", "Gemm alpha and beta must both be 1.0", node.sourceLine);
                validNode = false;
            }
            const TensorInfo* second = getInput(1);
            if (first == nullptr || second == nullptr || first->shape.size() != 2 || second->shape.size() != 2) {
                diagnostics.error("semantic", node.op + " requires two rank-2 tensors", node.sourceLine);
                validNode = false;
            } else {
                const std::int64_t firstRows = node.transA ? first->shape[1] : first->shape[0];
                const std::int64_t firstColumns = node.transA ? first->shape[0] : first->shape[1];
                const std::int64_t secondRows = node.transB ? second->shape[1] : second->shape[0];
                const std::int64_t secondColumns = node.transB ? second->shape[0] : second->shape[1];
                if (firstColumns != secondRows) {
                    diagnostics.error("semantic",
                                      node.op + " dimensions are incompatible: " +
                                          shapeText(first->shape) + " cannot multiply " +
                                          shapeText(second->shape),
                                      node.sourceLine);
                    validNode = false;
                } else {
                    outputShape = {firstRows, secondColumns};
                }
            }
            if (node.op == "Gemm" && node.inputs.size() == 3) {
                const TensorInfo* bias = getInput(2);
                const std::size_t biasElements = bias == nullptr ? 0 : elementCount(bias->shape);
                const std::size_t outputColumns = outputShape.size() == 2
                                                      ? static_cast<std::size_t>(outputShape[1])
                                                      : 0;
                if (bias == nullptr || (biasElements != outputColumns &&
                                        biasElements != elementCount(outputShape))) {
                    diagnostics.error("semantic", "Gemm bias must contain one value per output element", node.sourceLine);
                    validNode = false;
                }
            }
        } else if (node.op == "Add") {
            if (node.inputs.size() != 2) {
                diagnostics.error("semantic", "Add expects exactly two inputs", node.sourceLine);
                validNode = false;
            }
            const TensorInfo* second = getInput(1);
            if (first == nullptr || second == nullptr || !sameShape(first->shape, second->shape)) {
                diagnostics.error("semantic", "Add inputs must have identical shapes", node.sourceLine);
                validNode = false;
            } else {
                outputShape = first->shape;
            }
        } else {
            if (node.inputs.size() != 1 || first == nullptr) {
                diagnostics.error("semantic", node.op + " expects exactly one input", node.sourceLine);
                validNode = false;
            } else {
                outputShape = first->shape;
            }
            if (node.op == "Softmax" && first != nullptr) {
                const int rank = static_cast<int>(first->shape.size());
                const int resolvedAxis = node.axis < 0 ? rank - 1 : node.axis;
                if (resolvedAxis < 0 || resolvedAxis >= rank) {
                    diagnostics.error("semantic", "Softmax axis is outside the tensor rank", node.sourceLine);
                    validNode = false;
                } else if (resolvedAxis != rank - 1) {
                    diagnostics.error("semantic",
                                      "Only the final Softmax axis is supported by the first code generator",
                                      node.sourceLine);
                    validNode = false;
                } else {
                    node.axis = resolvedAxis;
                }
            }
        }

        for (const std::string& inputName : node.inputs) {
            const auto iterator = model.tensors.find(inputName);
            if (iterator != model.tensors.end() && iterator->second.dataType != DataType::Float32) {
                diagnostics.error("semantic", "Only float32 tensors are supported", node.sourceLine);
                validNode = false;
            }
        }

        if (!validNode) {
            continue;
        }

        TensorInfo output;
        if (outputIterator != model.tensors.end()) {
            output = outputIterator->second;
            if (!output.shape.empty() && output.shape != outputShape) {
                diagnostics.error("semantic",
                                  "Declared shape for '" + node.output + "' is " +
                                      shapeText(output.shape) + ", but the operation produces " +
                                      shapeText(outputShape),
                                  node.sourceLine);
                continue;
            }
        }
        output.name = node.output;
        output.dataType = DataType::Float32;
        output.shape = outputShape;
        output.isConstant = false;
        output.isInput = false;
        output.hasProducer = true;
        output.producer = node.name;
        model.tensors[node.output] = output;
        if (TensorInfo* symbol = symbols.lookup(node.output)) {
            *symbol = output;
        } else {
            symbols.insert(output);
        }
        available.insert(node.output);
    }

    if (!model.outputName.empty() && !available.count(model.outputName)) {
        diagnostics.error("semantic", "Declared output tensor '" + model.outputName + "' is never produced");
    }

    return !diagnostics.hasErrors();
}

}  // namespace modelforge

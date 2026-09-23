#include "model.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <vector>

namespace modelforge {
namespace {

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

std::string tensorText(const TensorInfo& tensor) {
    std::string kind = tensor.isConstant ? "constant" :
                       tensor.isInput ? "input" : "value";
    return tensor.name + " : " + dataTypeName(tensor.dataType) + " " +
           shapeText(tensor.shape) + " (" + kind + ")";
}

}  // namespace

std::string dataTypeName(DataType type) {
    switch (type) {
        case DataType::Float32:
            return "float32";
        case DataType::Unknown:
            return "unknown";
    }
    return "unknown";
}

std::size_t elementCount(const Shape& shape) {
    if (shape.empty()) {
        return 0;
    }

    std::size_t result = 1;
    for (std::int64_t dimension : shape) {
        if (dimension <= 0 ||
            result > std::numeric_limits<std::size_t>::max() /
                         static_cast<std::size_t>(dimension)) {
            return 0;
        }
        result *= static_cast<std::size_t>(dimension);
    }
    return result;
}

void printSyntaxTree(const ModelGraph& model, std::ostream& output) {
    std::vector<const TensorInfo*> tensors;
    tensors.reserve(model.tensors.size());
    for (const auto& entry : model.tensors) {
        tensors.push_back(&entry.second);
    }
    std::sort(tensors.begin(), tensors.end(), [](const TensorInfo* left, const TensorInfo* right) {
        return left->name < right->name;
    });

    output << "\nMODEL SYNTAX TREE (parsed ModelGraph)\n"
           << "=====================================\n"
           << "MODEL " << model.name << "\n"
           << "|-- INPUT: " << model.inputName << "\n"
           << "|-- TENSORS (" << tensors.size() << ")\n";
    for (std::size_t index = 0; index < tensors.size(); ++index) {
        output << "|   " << (index + 1 == tensors.size() ? "`-- " : "|-- ")
               << tensorText(*tensors[index]) << "\n";
    }

    output << "|-- NODES (" << model.nodes.size() << ")\n";
    for (std::size_t index = 0; index < model.nodes.size(); ++index) {
        const Node& node = model.nodes[index];
        const bool lastNode = index + 1 == model.nodes.size();
        const std::string nodeBranch = lastNode ? "`-- " : "|-- ";
        const std::string childPrefix = std::string("|   ") +
                                        (lastNode ? "    " : "|   ");
        output << "|   " << nodeBranch
               << "NODE " << node.name << " : " << node.op;
        if (node.axis >= 0) {
            output << " axis=" << node.axis;
        }
        if (node.transA) {
            output << " transA=1";
        }
        if (node.transB) {
            output << " transB=1";
        }
        output << "\n" << childPrefix << "|-- INPUTS: ";
        for (std::size_t inputIndex = 0; inputIndex < node.inputs.size(); ++inputIndex) {
            if (inputIndex != 0) {
                output << ", ";
            }
            output << node.inputs[inputIndex];
        }
        output << "\n" << childPrefix << "`-- OUTPUT: " << node.output << "\n";
    }
    output << "`-- OUTPUT: " << model.outputName << "\n\n";
}

}  // namespace modelforge

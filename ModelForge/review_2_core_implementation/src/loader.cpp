#include "loader.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef MODELFORGE_WITH_ONNX
#include <cstring>
#include <onnx/onnx_pb.h>
#endif

namespace modelforge {
namespace {

std::string trim(const std::string& value) {
    const std::size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const std::size_t last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::vector<std::string> tokensOf(const std::string& line) {
    std::istringstream stream(line);
    std::vector<std::string> tokens;
    std::string token;
    while (stream >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

bool parseDataType(const std::string& text, DataType& type) {
    if (text == "float32" || text == "float") {
        type = DataType::Float32;
        return true;
    }
    type = DataType::Unknown;
    return false;
}

bool parseShape(const std::string& text, Shape& shape) {
    shape.clear();
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        try {
            const long long dimension = std::stoll(item);
            if (dimension <= 0) {
                return false;
            }
            shape.push_back(static_cast<std::int64_t>(dimension));
        } catch (...) {
            return false;
        }
    }
    return !shape.empty();
}

bool parseValues(const std::string& text, std::vector<float>& values) {
    values.clear();
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        try {
            values.push_back(std::stof(item));
        } catch (...) {
            return false;
        }
    }
    return !values.empty();
}

bool insertTensor(ModelGraph& model,
                  const TensorInfo& tensor,
                  DiagnosticEngine& diagnostics,
                  int line) {
    if (model.tensors.find(tensor.name) != model.tensors.end()) {
        diagnostics.error("loader", "Duplicate tensor declaration '" + tensor.name + "'", line);
        return false;
    }
    model.tensors.emplace(tensor.name, tensor);
    return true;
}

}  // namespace

bool loadModel(const std::string& path,
               ModelGraph& model,
               DiagnosticEngine& diagnostics) {
    const std::string extension = [&]() {
        std::string result = std::filesystem::path(path).extension().string();
        std::transform(result.begin(), result.end(), result.begin(),
                       [](unsigned char character) {
                           return static_cast<char>(std::tolower(character));
                       });
        return result;
    }();

    if (extension == ".onnx") {
        return loadOnnx(path, model, diagnostics);
    }
    return loadManifest(path, model, diagnostics);
}

bool loadManifest(const std::string& path,
                  ModelGraph& model,
                  DiagnosticEngine& diagnostics) {
    std::ifstream file(path);
    if (!file) {
        diagnostics.error("loader", "Cannot open model file '" + path + "'");
        return false;
    }

    model = {};
    std::string rawLine;
    int lineNumber = 0;
    while (std::getline(file, rawLine)) {
        ++lineNumber;
        const std::size_t comment = rawLine.find('#');
        const std::string line = trim(rawLine.substr(0, comment));
        if (line.empty()) {
            continue;
        }

        const std::vector<std::string> tokens = tokensOf(line);
        const std::string& keyword = tokens.front();

        if (keyword == "model") {
            if (tokens.size() != 2) {
                diagnostics.error("loader", "Expected: model <name>", lineNumber);
                continue;
            }
            model.name = tokens[1];
            continue;
        }

        if (keyword == "input" || keyword == "output") {
            if (tokens.size() != 4) {
                diagnostics.error("loader",
                                  "Expected: " + keyword +
                                      " <name> <datatype> <shape>",
                                  lineNumber);
                continue;
            }
            DataType dataType = DataType::Unknown;
            Shape shape;
            if (!parseDataType(tokens[2], dataType)) {
                diagnostics.error("loader", "Unsupported datatype '" + tokens[2] + "'", lineNumber);
                continue;
            }
            if (!parseShape(tokens[3], shape)) {
                diagnostics.error("loader", "Invalid tensor shape '" + tokens[3] + "'", lineNumber);
                continue;
            }

            TensorInfo tensor;
            tensor.name = tokens[1];
            tensor.dataType = dataType;
            tensor.shape = shape;
            tensor.isInput = keyword == "input";
            if (!insertTensor(model, tensor, diagnostics, lineNumber)) {
                continue;
            }
            if (keyword == "input") {
                if (!model.inputName.empty()) {
                    diagnostics.error("loader", "Only one input is supported in the first implementation", lineNumber);
                }
                model.inputName = tensor.name;
            } else {
                if (!model.outputName.empty()) {
                    diagnostics.error("loader", "Only one output is supported in the first implementation", lineNumber);
                }
                model.outputName = tensor.name;
            }
            continue;
        }

        if (keyword == "tensor") {
            if (tokens.size() < 4) {
                diagnostics.error("loader",
                                  "Expected: tensor <name> <datatype> <shape> values=<comma-separated values>",
                                  lineNumber);
                continue;
            }
            DataType dataType = DataType::Unknown;
            Shape shape;
            if (!parseDataType(tokens[2], dataType) || !parseShape(tokens[3], shape)) {
                diagnostics.error("loader", "Invalid constant tensor declaration", lineNumber);
                continue;
            }

            TensorInfo tensor;
            tensor.name = tokens[1];
            tensor.dataType = dataType;
            tensor.shape = shape;
            tensor.isConstant = true;
            for (std::size_t index = 4; index < tokens.size(); ++index) {
                const std::string prefix = "values=";
                if (tokens[index].rfind(prefix, 0) == 0) {
                    if (!parseValues(tokens[index].substr(prefix.size()), tensor.data)) {
                        diagnostics.error("loader", "Invalid constant values for '" + tensor.name + "'", lineNumber);
                    }
                }
            }
            if (tensor.data.empty()) {
                diagnostics.error("loader", "Constant tensor '" + tensor.name + "' needs values=...", lineNumber);
            }
            insertTensor(model, tensor, diagnostics, lineNumber);
            continue;
        }

        if (keyword == "node") {
            const auto arrow = std::find(tokens.begin(), tokens.end(), "->");
            if (tokens.size() < 5 || arrow == tokens.end() || arrow - tokens.begin() < 4 ||
                arrow + 1 == tokens.end()) {
                diagnostics.error("loader",
                                  "Expected: node <name> <op> <inputs...> -> <output> [axis=N]",
                                  lineNumber);
                continue;
            }

            Node node;
            node.name = tokens[1];
            node.op = tokens[2];
            node.sourceLine = lineNumber;
            for (auto iterator = tokens.begin() + 3; iterator != arrow; ++iterator) {
                node.inputs.push_back(*iterator);
            }
            node.output = *(arrow + 1);
            for (auto iterator = arrow + 2; iterator != tokens.end(); ++iterator) {
                const std::string prefix = "axis=";
                const std::string transAPrefix = "transA=";
                const std::string transBPrefix = "transB=";
                const std::string alphaPrefix = "alpha=";
                const std::string betaPrefix = "beta=";
                if (iterator->rfind(prefix, 0) == 0) {
                    try {
                        node.axis = std::stoi(iterator->substr(prefix.size()));
                    } catch (...) {
                        diagnostics.error("loader", "Invalid axis value '" + *iterator + "'", lineNumber);
                    }
                } else if (iterator->rfind(transAPrefix, 0) == 0 ||
                           iterator->rfind(transBPrefix, 0) == 0) {
                    try {
                        const bool value = std::stoi(iterator->substr(7)) != 0;
                        if (iterator->rfind(transAPrefix, 0) == 0) {
                            node.transA = value;
                        } else {
                            node.transB = value;
                        }
                    } catch (...) {
                        diagnostics.error("loader", "Invalid transpose value '" + *iterator + "'", lineNumber);
                    }
                } else if (iterator->rfind(alphaPrefix, 0) == 0 ||
                           iterator->rfind(betaPrefix, 0) == 0) {
                    try {
                        const float value = std::stof(iterator->substr(6));
                        if (iterator->rfind(alphaPrefix, 0) == 0) {
                            node.alpha = value;
                        } else {
                            node.beta = value;
                        }
                    } catch (...) {
                        diagnostics.error("loader", "Invalid Gemm scale value '" + *iterator + "'", lineNumber);
                    }
                } else {
                    diagnostics.warning("loader", "Ignoring unknown node attribute '" + *iterator + "'", lineNumber);
                }
            }
            model.nodes.push_back(node);
            continue;
        }

        diagnostics.error("loader", "Unknown declaration '" + keyword + "'", lineNumber);
    }

    if (model.name.empty()) {
        diagnostics.error("loader", "Model declaration is missing");
    }
    if (model.inputName.empty()) {
        diagnostics.error("loader", "Input declaration is missing");
    }
    if (model.outputName.empty()) {
        diagnostics.error("loader", "Output declaration is missing");
    }
    if (model.nodes.empty()) {
        diagnostics.error("loader", "At least one node is required");
    }
    return !diagnostics.hasErrors();
}

bool loadOnnx(const std::string& path,
              ModelGraph& model,
              DiagnosticEngine& diagnostics) {
#ifndef MODELFORGE_WITH_ONNX
    (void)model;
    diagnostics.error(
        "loader",
        "ONNX support is not enabled for this build. Configure with "
        "-DMODELFORGE_WITH_ONNX=ON after installing ONNX and Protobuf.");
    (void)path;
    return false;
#else
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        diagnostics.error("loader", "Cannot open ONNX model '" + path + "'");
        return false;
    }

    onnx::ModelProto onnxModel;
    if (!onnxModel.ParseFromIstream(&file)) {
        diagnostics.error("loader", "The file is not a valid ONNX protobuf model");
        return false;
    }

    model = {};
    model.name = onnxModel.graph().name().empty() ? "onnx_model" : onnxModel.graph().name();
    const onnx::GraphProto& graph = onnxModel.graph();

    auto addValueInfo = [&](const onnx::ValueInfoProto& valueInfo, bool isInput, bool isOutput) {
        TensorInfo tensor;
        tensor.name = valueInfo.name();
        tensor.dataType = DataType::Float32;
        tensor.isInput = isInput;
        if (valueInfo.has_type() && valueInfo.type().has_tensor_type()) {
            const auto& tensorType = valueInfo.type().tensor_type();
            if (tensorType.elem_type() != onnx::TensorProto::FLOAT) {
                diagnostics.error("loader", "Only float32 ONNX tensors are supported: '" + tensor.name + "'");
                return;
            }
            for (const auto& dimension : tensorType.shape().dim()) {
                if (!dimension.has_dim_value() || dimension.dim_value() <= 0) {
                    diagnostics.error("loader", "Dynamic dimensions are not supported: '" + tensor.name + "'");
                    return;
                }
                tensor.shape.push_back(dimension.dim_value());
            }
        }
        if (tensor.shape.empty()) {
            diagnostics.error("loader", "Missing shape for ONNX tensor '" + tensor.name + "'");
            return;
        }
        if (!insertTensor(model, tensor, diagnostics, 0)) {
            return;
        }
        if (isInput) {
            model.inputName = tensor.name;
        }
        if (isOutput) {
            model.outputName = tensor.name;
        }
    };

    for (const auto& initializer : graph.initializer()) {
        TensorInfo tensor;
        tensor.name = initializer.name();
        tensor.dataType = DataType::Float32;
        tensor.isConstant = true;
        for (const auto dimension : initializer.dims()) {
            tensor.shape.push_back(dimension);
        }
        if (initializer.data_type() != onnx::TensorProto::FLOAT) {
            diagnostics.error("loader", "Only float32 initializers are supported: '" + tensor.name + "'");
            continue;
        }
        if (initializer.raw_data().empty()) {
            for (const float value : initializer.float_data()) {
                tensor.data.push_back(value);
            }
        } else {
            const std::string& raw = initializer.raw_data();
            if (raw.size() % sizeof(float) != 0) {
                diagnostics.error("loader", "Invalid raw_data size for initializer '" + tensor.name + "'");
                continue;
            }
            tensor.data.resize(raw.size() / sizeof(float));
            std::memcpy(tensor.data.data(), raw.data(), raw.size());
        }
        insertTensor(model, tensor, diagnostics, 0);
    }

    for (const auto& input : graph.input()) {
        if (model.tensors.find(input.name()) == model.tensors.end()) {
            addValueInfo(input, true, false);
        } else if (!model.tensors.at(input.name()).isConstant) {
            model.inputName = input.name();
        }
    }
    for (const auto& output : graph.output()) {
        if (model.tensors.find(output.name()) == model.tensors.end()) {
            addValueInfo(output, false, true);
        } else {
            model.outputName = output.name();
        }
    }

    for (const auto& onnxNode : graph.node()) {
        if (onnxNode.output_size() != 1) {
            diagnostics.error("loader", "Only single-output ONNX nodes are supported");
            continue;
        }
        Node node;
        node.name = onnxNode.name().empty() ? onnxNode.output(0) : onnxNode.name();
        node.op = onnxNode.op_type();
        node.output = onnxNode.output(0);
        for (const auto& input : onnxNode.input()) {
            if (!input.empty()) {
                node.inputs.push_back(input);
            }
        }
        node.axis = -1;
        for (const auto& attribute : onnxNode.attribute()) {
            if (attribute.name() == "axis" && attribute.has_i()) {
                node.axis = static_cast<int>(attribute.i());
            } else if (attribute.name() == "transA" && attribute.has_i()) {
                node.transA = attribute.i() != 0;
            } else if (attribute.name() == "transB" && attribute.has_i()) {
                node.transB = attribute.i() != 0;
            } else if (attribute.name() == "alpha" && attribute.has_f()) {
                node.alpha = attribute.f();
            } else if (attribute.name() == "beta" && attribute.has_f()) {
                node.beta = attribute.f();
            }
        }
        model.nodes.push_back(node);
    }

    return !diagnostics.hasErrors();
#endif
}

}  // namespace modelforge

#include "codegen.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace modelforge {
namespace {

std::string sanitizeIdentifier(const std::string& value) {
    std::string result;
    for (const unsigned char character : value) {
        if (std::isalnum(character) || character == '_') {
            result.push_back(static_cast<char>(character));
        } else {
            result.push_back('_');
        }
    }
    if (result.empty() || std::isdigit(static_cast<unsigned char>(result.front()))) {
        result.insert(result.begin(), '_');
    }
    return result;
}

std::string variableFor(const std::string& name) {
    return "v_" + sanitizeIdentifier(name);
}

std::string constantFor(const std::string& name) {
    return "c_" + sanitizeIdentifier(name);
}

std::string floatLiteral(float value) {
    std::ostringstream output;
    output << std::setprecision(9) << std::scientific << value << "f";
    return output.str();
}

void emitFloatVector(std::ostream& output,
                     const std::string& variable,
                     const std::vector<float>& values) {
    output << "static const Tensor " << variable << " = {";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ", ";
        }
        output << floatLiteral(values[index]);
    }
    output << "};\n";
}

std::string referenceFor(const IRGraph& graph, const std::string& name) {
    return graph.constants.find(name) != graph.constants.end() ? constantFor(name) : variableFor(name);
}

bool writeFile(const std::filesystem::path& path,
               const std::string& content,
               DiagnosticEngine& diagnostics) {
    std::ofstream file(path);
    if (!file) {
        diagnostics.error("codegen", "Cannot write generated file '" + path.string() + "'");
        return false;
    }
    file << content;
    return true;
}

}  // namespace

bool generateCpp(const IRGraph& graph,
                 const std::string& outputDirectory,
                 DiagnosticEngine& diagnostics) {
    const auto inputIterator = graph.values.find(graph.inputName);
    if (inputIterator == graph.values.end()) {
        diagnostics.error("codegen", "Input tensor information is missing");
        return false;
    }

    std::error_code filesystemError;
    const std::filesystem::path directory(outputDirectory);
    std::filesystem::create_directories(directory, filesystemError);
    if (filesystemError) {
        diagnostics.error("codegen", "Cannot create output directory '" + outputDirectory + "'");
        return false;
    }

    std::ostringstream header;
    header << "#pragma once\n\n#include <vector>\n\n";
    header << "std::vector<float> infer(const std::vector<float>& input);\n";

    std::ostringstream source;
    source << "#include \"model.h\"\n"
           << "#include <algorithm>\n"
           << "#include <cmath>\n"
           << "#include <iostream>\n"
           << "#include <stdexcept>\n"
           << "#include <vector>\n\n"
           << "using namespace std;\n"
           << "using Tensor = std::vector<float>;\n\n";

    source << R"cpp(
static Tensor gemm(const Tensor& a,
                   const Tensor& b,
                   const Tensor* bias,
                   int rows,
                   int inner,
                   int columns,
                   bool transA,
                   bool transB) {
    Tensor result(static_cast<std::size_t>(rows * columns), 0.0f);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            const std::size_t biasIndex = bias == nullptr
                                              ? 0
                                              : (bias->size() == static_cast<std::size_t>(columns)
                                                     ? static_cast<std::size_t>(column)
                                                     : static_cast<std::size_t>(row * columns + column));
            float value = bias == nullptr ? 0.0f : (*bias)[biasIndex];
            for (int k = 0; k < inner; ++k) {
                const std::size_t leftIndex = transA
                                                  ? static_cast<std::size_t>(k * rows + row)
                                                  : static_cast<std::size_t>(row * inner + k);
                const std::size_t rightIndex = transB
                                                   ? static_cast<std::size_t>(column * inner + k)
                                                   : static_cast<std::size_t>(k * columns + column);
                value += a[leftIndex] * b[rightIndex];
            }
            result[static_cast<std::size_t>(row * columns + column)] = value;
        }
    }
    return result;
}

static Tensor matmul(const Tensor& a,
                     const Tensor& b,
                     int rows,
                     int inner,
                     int columns) {
    return gemm(a, b, nullptr, rows, inner, columns, false, false);
}

static Tensor add(const Tensor& left, const Tensor& right) {
    if (left.size() != right.size()) {
        throw std::runtime_error("Add operands have different sizes");
    }
    Tensor result(left.size());
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = left[index] + right[index];
    }
    return result;
}

static Tensor relu(const Tensor& input) {
    Tensor result = input;
    for (float& value : result) {
        value = std::max(0.0f, value);
    }
    return result;
}

static Tensor sigmoid(const Tensor& input) {
    Tensor result = input;
    for (float& value : result) {
        value = 1.0f / (1.0f + std::exp(-value));
    }
    return result;
}

static Tensor softmax(const Tensor& input) {
    if (input.empty()) {
        return {};
    }
    const float maximum = *std::max_element(input.begin(), input.end());
    Tensor result(input.size());
    float total = 0.0f;
    for (std::size_t index = 0; index < input.size(); ++index) {
        result[index] = std::exp(input[index] - maximum);
        total += result[index];
    }
    for (float& value : result) {
        value /= total;
    }
    return result;
}

static Tensor fused_gemm_relu(const Tensor& a,
                              const Tensor& b,
                              const Tensor* bias,
                              int rows,
                              int inner,
                              int columns,
                              bool transA,
                              bool transB) {
    return relu(gemm(a, b, bias, rows, inner, columns, transA, transB));
}

)cpp";

    for (const auto& entry : graph.constants) {
        emitFloatVector(source, constantFor(entry.first), entry.second.data);
    }
    source << "\nstd::vector<float> infer(const std::vector<float>& input) {\n";
    source << "    if (input.size() != " << elementCount(inputIterator->second.shape)
           << ") {\n        throw std::runtime_error(\"Unexpected input size\");\n    }\n";

    for (const IRInstruction& instruction : graph.instructions) {
        switch (instruction.operation) {
            case IROp::Input:
                source << "    Tensor " << variableFor(instruction.output) << " = input;\n";
                break;
            case IROp::Return: {
                const std::string value = referenceFor(graph, instruction.inputs.front());
                source << "    return " << value << ";\n";
                break;
            }
            case IROp::Gemm:
            case IROp::FusedGemmRelu: {
                if (instruction.inputs.size() < 2) {
                    diagnostics.error("codegen", "Gemm instruction has too few inputs");
                    return false;
                }
                const TensorInfo& left = graph.values.at(instruction.inputs[0]);
                const TensorInfo& right = graph.values.at(instruction.inputs[1]);
                if (left.shape.size() != 2 || right.shape.size() != 2) {
                    diagnostics.error("codegen", "Only rank-2 Gemm tensors can be generated");
                    return false;
                }
                const std::string leftRef = referenceFor(graph, instruction.inputs[0]);
                const std::string rightRef = referenceFor(graph, instruction.inputs[1]);
                std::string biasArgument = "nullptr";
                if (instruction.inputs.size() == 3) {
                    biasArgument = "&" + referenceFor(graph, instruction.inputs[2]);
                }
                const std::string function = instruction.operation == IROp::Gemm
                                                 ? "gemm"
                                                 : "fused_gemm_relu";
                source << "    Tensor " << variableFor(instruction.output) << " = " << function
                       << "(" << leftRef << ", " << rightRef << ", " << biasArgument << ", "
                       << (instruction.transA ? left.shape[1] : left.shape[0]) << ", "
                       << (instruction.transA ? left.shape[0] : left.shape[1]) << ", "
                       << (instruction.transB ? right.shape[0] : right.shape[1]) << ", "
                       << (instruction.transA ? "true" : "false") << ", "
                       << (instruction.transB ? "true" : "false") << ");\n";
                break;
            }
            case IROp::MatMul: {
                const TensorInfo& left = graph.values.at(instruction.inputs.at(0));
                const TensorInfo& right = graph.values.at(instruction.inputs.at(1));
                source << "    Tensor " << variableFor(instruction.output) << " = matmul("
                       << referenceFor(graph, instruction.inputs[0]) << ", "
                       << referenceFor(graph, instruction.inputs[1]) << ", " << left.shape[0]
                       << ", " << left.shape[1] << ", " << right.shape[1] << ");\n";
                break;
            }
            case IROp::Add:
                source << "    Tensor " << variableFor(instruction.output) << " = add("
                       << referenceFor(graph, instruction.inputs[0]) << ", "
                       << referenceFor(graph, instruction.inputs[1]) << ");\n";
                break;
            case IROp::Relu:
                source << "    Tensor " << variableFor(instruction.output) << " = relu("
                       << referenceFor(graph, instruction.inputs[0]) << ");\n";
                break;
            case IROp::TestReluDeadZone:
                diagnostics.error("codegen", "Test-only fault injection operator cannot be generated");
                return false;
            case IROp::Sigmoid:
                source << "    Tensor " << variableFor(instruction.output) << " = sigmoid("
                       << referenceFor(graph, instruction.inputs[0]) << ");\n";
                break;
            case IROp::Softmax:
                source << "    Tensor " << variableFor(instruction.output) << " = softmax("
                       << referenceFor(graph, instruction.inputs[0]) << ");\n";
                break;
        }
    }

    source << "}\n\nint main() {\n"
           << "    const Tensor sampleInput(" << elementCount(inputIterator->second.shape)
           << ", 0.0f);\n"
           << "    const Tensor output = infer(sampleInput);\n"
           << "    const auto best = std::max_element(output.begin(), output.end());\n"
           << "    std::cout << \"Prediction: \" << (best - output.begin()) << \"\\n\";\n"
           << "    std::cout << \"Confidence: \" << *best << \"\\n\";\n"
           << "    return 0;\n}\n";

    const std::string generatedCMake =
        "cmake_minimum_required(VERSION 3.16)\n"
        "project(GeneratedModel LANGUAGES CXX)\n"
        "set(CMAKE_CXX_STANDARD 17)\n"
        "set(CMAKE_CXX_STANDARD_REQUIRED ON)\n"
        "add_executable(generated_model model.cpp)\n";

    return writeFile(directory / "model.h", header.str(), diagnostics) &&
           writeFile(directory / "model.cpp", source.str(), diagnostics) &&
           writeFile(directory / "CMakeLists.txt", generatedCMake, diagnostics);
}

}  // namespace modelforge

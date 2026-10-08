#include "verifier.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <unordered_map>
#include <utility>

#ifdef MODELFORGE_WITH_ONNX_RUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace modelforge {
namespace {

using TensorMap = std::unordered_map<std::string, std::vector<float>>;

const std::vector<float>* findValue(const IRGraph& graph,
                                    const TensorMap& values,
                                    const std::string& name) {
    const auto value = values.find(name);
    if (value != values.end()) {
        return &value->second;
    }
    const auto constant = graph.constants.find(name);
    return constant == graph.constants.end() ? nullptr : &constant->second.data;
}

std::vector<float> gemm(const std::vector<float>& a,
                        const std::vector<float>& b,
                        const std::vector<float>* bias,
                        int rows,
                        int inner,
                        int columns,
                        bool applyRelu,
                        bool transA,
                        bool transB) {
    std::vector<float> result(static_cast<std::size_t>(rows * columns), 0.0f);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            float value = 0.0f;
            if (bias != nullptr) {
                const std::size_t biasIndex = bias->size() == static_cast<std::size_t>(columns)
                                                  ? static_cast<std::size_t>(column)
                                                  : static_cast<std::size_t>(row * columns + column);
                value = (*bias)[biasIndex];
            }
            for (int k = 0; k < inner; ++k) {
                const std::size_t leftIndex = transA
                                                  ? static_cast<std::size_t>(k * rows + row)
                                                  : static_cast<std::size_t>(row * inner + k);
                const std::size_t rightIndex = transB
                                                   ? static_cast<std::size_t>(column * inner + k)
                                                   : static_cast<std::size_t>(k * columns + column);
                value += a[leftIndex] * b[rightIndex];
            }
            result[static_cast<std::size_t>(row * columns + column)] =
                applyRelu ? std::max(0.0f, value) : value;
        }
    }
    return result;
}

std::vector<float> softmax(const std::vector<float>& input) {
    if (input.empty()) {
        return {};
    }
    const float maximum = *std::max_element(input.begin(), input.end());
    std::vector<float> result(input.size());
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

}  // namespace

std::optional<std::vector<float>> executeIR(const IRGraph& graph,
                                            const std::vector<float>& input,
                                            DiagnosticEngine& diagnostics) {
    return executeIRTrace(graph, input, diagnostics, nullptr);
}

std::optional<std::vector<float>> executeIRTrace(
    const IRGraph& graph,
    const std::vector<float>& input,
    DiagnosticEngine& diagnostics,
    std::unordered_map<std::string, std::vector<float>>* trace) {
    const auto inputInfo = graph.values.find(graph.inputName);
    if (inputInfo == graph.values.end() || input.size() != elementCount(inputInfo->second.shape)) {
        diagnostics.error("runtime", "Input size does not match the IR input shape");
        return std::nullopt;
    }

    TensorMap values;
    for (const auto& instruction : graph.instructions) {
        if (instruction.operation == IROp::Input) {
            values[instruction.output] = input;
            continue;
        }
        if (instruction.operation == IROp::Return) {
            const auto* result = findValue(graph, values, instruction.inputs.front());
            if (result == nullptr) {
                diagnostics.error("runtime", "Return value is unavailable");
                return std::nullopt;
            }
            if (trace != nullptr) *trace = values;
            return *result;
        }

        const std::vector<float>* first = instruction.inputs.empty()
                                              ? nullptr
                                              : findValue(graph, values, instruction.inputs[0]);
        if (first == nullptr) {
            diagnostics.error("runtime", "Runtime input is unavailable for '" + instruction.nodeName + "'");
            return std::nullopt;
        }

        std::vector<float> result;
        switch (instruction.operation) {
            case IROp::Gemm:
            case IROp::FusedGemmRelu: {
                if (instruction.inputs.size() < 2) {
                    diagnostics.error("runtime", "Gemm runtime instruction has too few inputs");
                    return std::nullopt;
                }
                const auto* second = findValue(graph, values, instruction.inputs[1]);
                const auto* bias = instruction.inputs.size() == 3
                                       ? findValue(graph, values, instruction.inputs[2])
                                       : nullptr;
                const auto leftInfo = graph.values.find(instruction.inputs[0]);
                const auto rightInfo = graph.values.find(instruction.inputs[1]);
                if (second == nullptr || leftInfo == graph.values.end() || rightInfo == graph.values.end()) {
                    diagnostics.error("runtime", "Gemm runtime tensor information is unavailable");
                    return std::nullopt;
                }
                result = gemm(*first,
                              *second,
                              bias,
                              static_cast<int>(instruction.transA ? leftInfo->second.shape[1]
                                                                    : leftInfo->second.shape[0]),
                              static_cast<int>(instruction.transA ? leftInfo->second.shape[0]
                                                                    : leftInfo->second.shape[1]),
                              static_cast<int>(instruction.transB ? rightInfo->second.shape[0]
                                                                    : rightInfo->second.shape[1]),
                              instruction.operation == IROp::FusedGemmRelu,
                              instruction.transA,
                              instruction.transB);
                break;
            }
            case IROp::MatMul: {
                const auto* second = findValue(graph, values, instruction.inputs.at(1));
                const auto leftInfo = graph.values.find(instruction.inputs.at(0));
                const auto rightInfo = graph.values.find(instruction.inputs.at(1));
                if (second == nullptr || leftInfo == graph.values.end() || rightInfo == graph.values.end()) {
                    diagnostics.error("runtime", "MatMul runtime tensor information is unavailable");
                    return std::nullopt;
                }
                result = gemm(*first,
                              *second,
                              nullptr,
                              static_cast<int>(leftInfo->second.shape[0]),
                              static_cast<int>(leftInfo->second.shape[1]),
                              static_cast<int>(rightInfo->second.shape[1]),
                              false,
                              false,
                              false);
                break;
            }
            case IROp::Add: {
                const auto* second = findValue(graph, values, instruction.inputs.at(1));
                if (second == nullptr || second->size() != first->size()) {
                    diagnostics.error("runtime", "Add runtime operands have different sizes");
                    return std::nullopt;
                }
                result.resize(first->size());
                for (std::size_t index = 0; index < result.size(); ++index) {
                    result[index] = (*first)[index] + (*second)[index];
                }
                break;
            }
            case IROp::Relu:
                result = *first;
                for (float& value : result) {
                    value = std::max(0.0f, value);
                }
                break;
            case IROp::TestReluDeadZone:
                result = *first;
                for (float& value : result) {
                    value = value > 0.0f && value < instruction.testFaultParameter
                                ? 0.0f
                                : std::max(0.0f, value);
                }
                break;
            case IROp::TestReluClamp:
                result = *first;
                for (float& value : result) {
                    value = std::min(std::max(0.0f, value), instruction.testFaultParameter);
                }
                break;
            case IROp::TestLeakyRelu:
                result = *first;
                for (float& value : result) {
                    value = value < 0.0f ? instruction.testFaultParameter * value : value;
                }
                break;
            case IROp::Sigmoid:
                result = *first;
                for (float& value : result) {
                    value = 1.0f / (1.0f + std::exp(-value));
                }
                break;
            case IROp::Softmax:
                result = softmax(*first);
                break;
            case IROp::Input:
            case IROp::Return:
                break;
        }
        values[instruction.output] = std::move(result);
    }

    diagnostics.error("runtime", "IR does not contain a return instruction");
    return std::nullopt;
}

VerificationReport compareOutputs(const std::vector<float>& expected,
                                   const std::vector<float>& actual,
                                   float tolerance) {
    VerificationReport report;
    if (expected.size() != actual.size()) {
        report.mismatches = std::max(expected.size(), actual.size());
        return report;
    }
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (!std::isfinite(expected[index]) || !std::isfinite(actual[index])) {
            ++report.mismatches;
            report.maximumAbsoluteError = std::numeric_limits<float>::infinity();
            continue;
        }
        const float difference = std::fabs(expected[index] - actual[index]);
        report.maximumAbsoluteError = std::max(report.maximumAbsoluteError, difference);
        if (difference > tolerance) {
            ++report.mismatches;
        }
    }
    report.passed = report.mismatches == 0;
    return report;
}

std::vector<ValidationProbe> generateValidationProbes(const IRGraph& graph,
                                                       std::uint32_t seed) {
    const auto input = graph.values.find(graph.inputName);
    if (input == graph.values.end()) {
        return {};
    }
    const std::size_t count = elementCount(input->second.shape);
    if (count == 0) {
        return {};
    }

    bool hasRelu = false;
    bool hasSigmoid = false;
    bool hasSoftmax = false;
    bool hasGemm = false;
    for (const IRInstruction& instruction : graph.instructions) {
        hasRelu = hasRelu || instruction.operation == IROp::Relu ||
                  instruction.operation == IROp::FusedGemmRelu;
        hasSigmoid = hasSigmoid || instruction.operation == IROp::Sigmoid;
        hasSoftmax = hasSoftmax || instruction.operation == IROp::Softmax;
        hasGemm = hasGemm || instruction.operation == IROp::Gemm ||
                  instruction.operation == IROp::FusedGemmRelu;
    }

    std::vector<ValidationProbe> probes;
    probes.push_back({"zero", std::vector<float>(count, 0.0f)});
    probes.push_back({"ones", std::vector<float>(count, 1.0f)});

    if (hasRelu) {
        std::vector<float> boundary(count);
        for (std::size_t index = 0; index < count; ++index) {
            boundary[index] = index % 2 == 0 ? -1.0f : 1.0f;
        }
        probes.push_back({"relu_boundary", std::move(boundary)});
        probes.push_back({"relu_near_zero_positive", std::vector<float>(count, 1.0e-4f)});
        probes.push_back({"relu_near_zero_negative", std::vector<float>(count, -1.0e-4f)});
    }

    if (hasSigmoid) {
        probes.push_back({"sigmoid_saturation", std::vector<float>(count, 6.0f)});
        std::vector<float> negative(count, -6.0f);
        probes.push_back({"sigmoid_negative_saturation", std::move(negative)});
    }

    if (hasSoftmax) {
        std::vector<float> range(count);
        for (std::size_t index = 0; index < count; ++index) {
            range[index] = -3.0f + 6.0f * static_cast<float>(index) /
                                      static_cast<float>(std::max<std::size_t>(1, count - 1));
        }
        probes.push_back({"softmax_range", std::move(range)});
    }

    if (hasGemm) {
        std::vector<float> ramp(count);
        for (std::size_t index = 0; index < count; ++index) {
            ramp[index] = -1.0f + 2.0f * static_cast<float>(index) /
                                      static_cast<float>(std::max<std::size_t>(1, count - 1));
        }
        probes.push_back({"gemm_ramp", std::move(ramp)});
    }

    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<float> randomValues(count);
    for (float& value : randomValues) {
        value = distribution(generator);
    }
    probes.push_back({"seeded_random", std::move(randomValues)});
    for (float magnitude : {0.1f, 1.0f, 10.0f}) {
        probes.push_back({"all_positive_" + std::to_string(magnitude),
                          std::vector<float>(count, magnitude)});
        probes.push_back({"all_negative_" + std::to_string(magnitude),
                          std::vector<float>(count, -magnitude)});
    }
    const std::size_t coordinateCount = std::min<std::size_t>(count, 16);
    for (std::size_t index = 0; index < coordinateCount; ++index) {
        for (float value : {-10.0f, -1.0f, 1.0f, 10.0f}) {
            std::vector<float> sparse(count, 0.0f);
            sparse[index] = value;
            probes.push_back({"coordinate_" + std::to_string(index) + "_" +
                                  std::to_string(value), std::move(sparse)});
        }
    }
    for (int sample = 0; sample < 32; ++sample) {
        std::vector<float> values(count);
        for (float& value : values) {
            value = distribution(generator) * (sample % 2 == 0 ? 1.0f : 10.0f);
        }
        probes.push_back({"random_" + std::to_string(sample), std::move(values)});
    }

    // Solve the first dense layer's affine equation for each ReLU neuron:
    // x[k] * W[k,j] + bias[j] = 0. Probe both sides of the actual activation
    // boundary, rather than assuming zero at the model input reaches it.
    for (std::size_t node = 0; node + 1 < graph.instructions.size(); ++node) {
        const auto& dense = graph.instructions[node];
        const auto& activation = graph.instructions[node + 1];
        if (dense.operation != IROp::Gemm || activation.operation != IROp::Relu ||
            activation.inputs.size() != 1 || activation.inputs[0] != dense.output ||
            dense.inputs.size() < 2 || dense.inputs[0] != graph.inputName || dense.transA ||
            input->second.shape.size() != 2 || input->second.shape[0] != 1) {
            continue;
        }
        const auto weights = graph.constants.find(dense.inputs[1]);
        if (weights == graph.constants.end() || weights->second.shape.size() != 2) continue;
        const std::size_t columns = static_cast<std::size_t>(
            dense.transB ? weights->second.shape[0] : weights->second.shape[1]);
        const std::size_t inner = static_cast<std::size_t>(
            dense.transB ? weights->second.shape[1] : weights->second.shape[0]);
        if (inner != count || columns == 0) continue;
        const std::vector<float>* bias = nullptr;
        if (dense.inputs.size() == 3) {
            const auto found = graph.constants.find(dense.inputs[2]);
            if (found == graph.constants.end()) continue;
            bias = &found->second.data;
        }
        for (std::size_t column = 0; column < std::min<std::size_t>(columns, 32); ++column) {
            std::size_t coordinate = 0;
            float strongest = 0.0f;
            float coefficient = 0.0f;
            for (std::size_t k = 0; k < inner; ++k) {
                const std::size_t offset = dense.transB ? column * inner + k : k * columns + column;
                const float value = weights->second.data[offset];
                if (std::fabs(value) > strongest) {
                    strongest = std::fabs(value);
                    coefficient = value;
                    coordinate = k;
                }
            }
            if (strongest < 1.0e-8f) continue;
            const float intercept = bias == nullptr ? 0.0f : (*bias)[column];
            const float boundary = -intercept / coefficient;
            if (!std::isfinite(boundary) || std::fabs(boundary) > 10.0f) continue;
            // Aim for +/-0.05 in preactivation space, within a narrow fault's
            // dead zone, regardless of the weight's scale.
            const float delta = 0.05f / strongest;
            if (std::fabs(boundary) + delta > 10.0f) continue;
            for (int side = -1; side <= 1; ++side) {
                std::vector<float> values(count, 0.0f);
                values[coordinate] = boundary + side * delta;
                probes.push_back({"affine_relu_" + std::to_string(column) + "_" +
                                      std::to_string(side), std::move(values)});
            }
        }
        break;
    }

    // If sampled inputs give different classes, bisect a segment between them.
    // This targets prediction boundaries where small numerical errors matter most.
    if (count <= 64) {
        std::vector<float> left;
        std::size_t leftClass = 0;
        bool foundLeft = false;
        const std::size_t existingProbeCount = probes.size();
        for (std::size_t probeIndex = 0; probeIndex < existingProbeCount; ++probeIndex) {
            const auto& probe = probes[probeIndex];
            DiagnosticEngine localDiagnostics;
            const auto output = executeIR(graph, probe.values, localDiagnostics);
            if (!output || output->size() < 2 || localDiagnostics.hasErrors()) continue;
            const std::size_t predicted = static_cast<std::size_t>(
                std::max_element(output->begin(), output->end()) - output->begin());
            if (!foundLeft) {
                left = probe.values;
                leftClass = predicted;
                foundLeft = true;
                continue;
            }
            if (predicted == leftClass) continue;
            std::vector<float> right = probe.values;
            for (int step = 0; step < 16; ++step) {
                std::vector<float> midpoint(count);
                for (std::size_t k = 0; k < count; ++k) {
                    midpoint[k] = (left[k] + right[k]) * 0.5f;
                }
                DiagnosticEngine midpointDiagnostics;
                const auto midpointOutput = executeIR(graph, midpoint, midpointDiagnostics);
                if (!midpointOutput || midpointDiagnostics.hasErrors()) break;
                const std::size_t midpointClass = static_cast<std::size_t>(
                    std::max_element(midpointOutput->begin(), midpointOutput->end()) -
                    midpointOutput->begin());
                if (midpointClass == leftClass) left = std::move(midpoint);
                else right = std::move(midpoint);
            }
            probes.push_back({"decision_boundary_left", std::move(left)});
            probes.push_back({"decision_boundary_right", std::move(right)});
            break;
        }
    }
    return probes;
}

TranslationValidationReport validateOptimization(const IRGraph& original,
                                                 const IRGraph& optimized,
                                                 const std::vector<ValidationProbe>& probes,
                                                 DiagnosticEngine& diagnostics,
                                                 float tolerance) {
    TranslationValidationReport report;
    report.probeCount = probes.size();
    for (const ValidationProbe& probe : probes) {
        const auto originalOutput = executeIR(original, probe.values, diagnostics);
        const auto optimizedOutput = executeIR(optimized, probe.values, diagnostics);
        if (!originalOutput.has_value() || !optimizedOutput.has_value()) {
            ++report.outputMismatches;
            if (!report.firstFailure) {
                report.firstFailure = probe;
                if (originalOutput) report.originalOutput = *originalOutput;
                if (optimizedOutput) report.candidateOutput = *optimizedOutput;
            }
            continue;
        }

        const VerificationReport outputReport =
            compareOutputs(*originalOutput, *optimizedOutput, tolerance);
        report.maximumAbsoluteError =
            std::max(report.maximumAbsoluteError, outputReport.maximumAbsoluteError);
        report.outputMismatches += outputReport.mismatches;

        const auto originalClass = std::max_element(originalOutput->begin(), originalOutput->end());
        const auto optimizedClass = std::max_element(optimizedOutput->begin(), optimizedOutput->end());
        const bool predictionMismatch = originalClass == originalOutput->end() ||
            optimizedClass == optimizedOutput->end() ||
            (originalClass - originalOutput->begin()) !=
                (optimizedClass - optimizedOutput->begin());
        if (predictionMismatch) {
            ++report.predictionMismatches;
        }
        if ((outputReport.mismatches != 0 || predictionMismatch) && !report.firstFailure) {
            report.firstFailure = probe;
            report.originalOutput = *originalOutput;
            report.candidateOutput = *optimizedOutput;
        }
    }
    report.passed = report.outputMismatches == 0 && report.predictionMismatches == 0;
    return report;
}

VerificationReport verifyWithOnnxRuntime(const std::string& onnxPath,
                                          const IRGraph& graph,
                                          const std::vector<float>& input,
                                          DiagnosticEngine& diagnostics,
                                          float tolerance) {
#ifndef MODELFORGE_WITH_ONNX_RUNTIME
    (void)onnxPath;
    (void)graph;
    (void)input;
    (void)tolerance;
    diagnostics.error(
        "verification",
        "ONNX Runtime support is not enabled. Configure with "
        "-DMODELFORGE_WITH_ONNX_RUNTIME=ON after installing ONNX Runtime.");
    return {};
#else
    const auto generatedOutput = executeIR(graph, input, diagnostics);
    if (!generatedOutput.has_value()) {
        return {};
    }

    try {
        Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "ModelForge");
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
        Ort::Session session(environment, onnxPath.c_str(), options);
        Ort::AllocatorWithDefaultOptions allocator;

        if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) {
            diagnostics.error("verification", "ONNX Runtime verification requires one input and one output");
            return {};
        }

        auto inputName = session.GetInputNameAllocated(0, allocator);
        auto outputName = session.GetOutputNameAllocated(0, allocator);
        const auto inputInfo = graph.values.find(graph.inputName);
        if (inputInfo == graph.values.end()) {
            diagnostics.error("verification", "IR input information is missing");
            return {};
        }
        std::vector<std::int64_t> dimensions = inputInfo->second.shape;
        Ort::MemoryInfo memoryInfo("Cpu", OrtArenaAllocator, 0, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            const_cast<float*>(input.data()),
            input.size(),
            dimensions.data(),
            dimensions.size());

        const char* inputNames[] = {inputName.get()};
        const char* outputNames[] = {outputName.get()};
        auto outputs = session.Run(Ort::RunOptions{nullptr},
                                   inputNames,
                                   &inputTensor,
                                   1,
                                   outputNames,
                                   1);
        if (outputs.empty() || !outputs[0].IsTensor()) {
            diagnostics.error("verification", "ONNX Runtime returned no tensor output");
            return {};
        }
        const auto outputInfo = outputs[0].GetTensorTypeAndShapeInfo();
        const std::size_t outputCount = outputInfo.GetElementCount();
        const float* outputData = outputs[0].GetTensorData<float>();
        std::vector<float> onnxOutput(outputData, outputData + outputCount);
        return compareOutputs(onnxOutput, *generatedOutput, tolerance);
    } catch (const Ort::Exception& exception) {
        diagnostics.error("verification", "ONNX Runtime failed: " + std::string(exception.what()));
        return {};
    }
#endif
}

}  // namespace modelforge

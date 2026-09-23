#include "codegen.h"
#include "guardian.h"
#include "ir.h"
#include "loader.h"
#include "optimizer.h"
#include "validator.h"
#include "verifier.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

bool loadAndValidate(const std::filesystem::path& path,
                     modelforge::ModelGraph& model,
                     modelforge::DiagnosticEngine& diagnostics) {
    modelforge::SymbolTable symbols;
    if (!modelforge::loadModel(path.string(), model, diagnostics)) {
        return false;
    }
    return modelforge::validate(model, diagnostics, symbols);
}

void testValidPipeline(const std::filesystem::path& fixtureDirectory) {
    modelforge::ModelGraph model;
    modelforge::DiagnosticEngine diagnostics;
    const auto path = fixtureDirectory.parent_path().parent_path() / "models" / "iris_demo.mforge";
    check(loadAndValidate(path, model, diagnostics), "valid Iris manifest should pass loading and validation");

    const auto ir = modelforge::buildIR(model, diagnostics);
    check(ir.has_value(), "valid model should lower to IR");
    if (!ir.has_value()) {
        return;
    }

    modelforge::IRGraph optimized = *ir;
    const modelforge::OptimizationReport report = modelforge::optimize(optimized, diagnostics);
    check(report.fusedOperations == 1, "Gemm followed by Relu should be fused");
    check(report.removedInstructions >= 1, "unreachable node should be removed");
    check(!report.events.empty(), "optimization should emit auditable rewrite events");

    const auto probes = modelforge::generateValidationProbes(*ir);
    check(std::any_of(probes.begin(), probes.end(), [](const auto& probe) {
              return probe.name.rfind("affine_relu_", 0) == 0;
          }), "first dense layer should produce actual affine ReLU boundary probes");
    const auto validation =
        modelforge::validateOptimization(*ir, optimized, probes, diagnostics);
    check(validation.passed, "optimized IR should preserve original IR behavior");
    check(validation.probeCount >= 5, "graph-aware validation should generate multiple probes");
    check(validation.predictionMismatches == 0,
          "optimization should preserve predicted classes across probes");

    modelforge::IRGraph guarded = *ir;
    const auto guardian = modelforge::optimizeWithGuardian(guarded, diagnostics);
    check(guardian.decisions.size() == 3, "Guardian should inspect all three passes");
    check(guardian.acceptedOptimizations.fusedOperations == 1,
          "Guardian should accept the valid fusion");
    check(guardian.acceptedOptimizations.removedInstructions == 1,
          "Guardian should accept dead-node removal");
    check(modelforge::validateOptimization(*ir, guarded, probes, diagnostics).passed,
          "Guardian result should preserve model outputs");

    modelforge::IRGraph buggy = *ir;
    for (auto& instruction : buggy.instructions) {
        if (instruction.operation == modelforge::IROp::Relu) {
            instruction.operation = modelforge::IROp::Sigmoid;
            break;
        }
    }
    const auto rejection = modelforge::checkCandidate(*ir, buggy, "test_wrong_relu", diagnostics);
    check(!rejection.accepted, "Guardian should reject a wrong ReLU rewrite");
    check(rejection.validation.firstFailure.has_value(),
          "Guardian should save a reproducible counterexample");
    if (rejection.validation.firstFailure) {
        const auto replay = modelforge::validateOptimization(
            *ir, buggy, {*rejection.validation.firstFailure}, diagnostics);
        check(!replay.passed, "minimized counterexample should still reproduce the failure");
    }

    const std::vector<float> input(4, 0.0f);
    const auto result = modelforge::executeIR(optimized, input, diagnostics);
    check(result.has_value(), "optimized IR should execute");
    if (result.has_value()) {
        check(result->size() == 3, "Iris result should contain three class scores");
        check(std::distance(result->begin(), std::max_element(result->begin(), result->end())) == 2,
              "Iris demo should predict class 2 for the zero input");
        float total = 0.0f;
        for (const float value : *result) {
            total += value;
        }
        check(std::fabs(total - 1.0f) < 1.0e-5f, "Softmax output should sum to one");
    }
    const auto userInputResult = modelforge::executeIR(optimized, {1.0f, 2.0f, 3.0f, 4.0f},
                                                       diagnostics);
    check(userInputResult.has_value() && userInputResult->size() == 3 &&
              std::fabs((*userInputResult)[1] - 0.55681473f) < 1.0e-5f,
          "non-zero user input should produce the expected class-1 score");

    const auto outputDirectory = std::filesystem::temp_directory_path() / "modelforge_test_generated";
    check(modelforge::generateCpp(optimized, outputDirectory.string(), diagnostics),
          "C++ code should be generated");
    check(std::filesystem::exists(outputDirectory / "model.h"), "generated header should exist");
    check(std::filesystem::exists(outputDirectory / "model.cpp"), "generated source should exist");
    {
        std::ifstream generated(outputDirectory / "model.cpp");
        const std::string source((std::istreambuf_iterator<char>(generated)),
                                 std::istreambuf_iterator<char>());
        check(source.find("std::stof(text, &parsed)") != std::string::npos,
              "generated executable should accept numeric inputs");
        check(source.find("Scores:") != std::string::npos,
              "generated executable should print all output scores");
        check(source.find("row,prediction") != std::string::npos,
              "generated executable should support batch CSV predictions");
    }
}

void testConstantFolding(const std::filesystem::path& fixtureDirectory) {
    modelforge::ModelGraph model;
    modelforge::DiagnosticEngine diagnostics;
    check(loadAndValidate(fixtureDirectory / "constant_fold.mforge", model, diagnostics),
          "constant-fold fixture should pass validation");
    const auto ir = modelforge::buildIR(model, diagnostics);
    check(ir.has_value(), "constant-fold fixture should lower to IR");
    if (!ir.has_value()) {
        return;
    }
    modelforge::IRGraph optimized = *ir;
    const auto report = modelforge::optimize(optimized, diagnostics);
    check(report.constantFolds == 1, "constant Add should be folded");
    check(optimized.constants.find("y") != optimized.constants.end(),
          "folded output should become a constant");
    const auto result = modelforge::executeIR(optimized, {0.0f}, diagnostics);
    check(result.has_value() && result->size() == 2 && (*result)[0] == 4.0f && (*result)[1] == 3.0f,
          "folded constant should execute to [4, 3]");
}

void testTransposedGemm(const std::filesystem::path& fixtureDirectory) {
    modelforge::ModelGraph model;
    modelforge::DiagnosticEngine diagnostics;
    check(loadAndValidate(fixtureDirectory / "transposed_gemm.mforge", model, diagnostics),
          "transposed Gemm fixture should pass validation");
    const auto ir = modelforge::buildIR(model, diagnostics);
    check(ir.has_value(), "transposed Gemm fixture should lower to IR");
    if (!ir.has_value()) {
        return;
    }
    const auto result = modelforge::executeIR(*ir, {1.0f, 1.0f}, diagnostics);
    check(result.has_value() && result->size() == 2 && (*result)[0] == 3.0f && (*result)[1] == 7.0f,
          "transB Gemm should use the transposed weight layout");
}

void testOperatorCoverage(const std::filesystem::path& fixtureDirectory) {
    modelforge::ModelGraph model;
    modelforge::DiagnosticEngine diagnostics;
    check(loadAndValidate(fixtureDirectory / "operator_coverage.mforge", model, diagnostics),
          "operator coverage fixture should pass validation");
    const auto ir = modelforge::buildIR(model, diagnostics);
    check(ir.has_value(), "operator coverage fixture should lower to IR");
    if (!ir.has_value()) {
        return;
    }
    const auto result = modelforge::executeIR(*ir, {0.0f, 1.0f}, diagnostics);
    check(result.has_value() && result->size() == 2,
          "MatMul, Add, Sigmoid, and Softmax should execute");
    if (result.has_value()) {
        check(std::distance(result->begin(), std::max_element(result->begin(), result->end())) == 1,
              "operator coverage model should predict its second class");
        check(std::fabs((*result)[0] + (*result)[1] - 1.0f) < 1.0e-5f,
              "operator coverage Softmax output should sum to one");
    }
}

void testInvalidInputs(const std::filesystem::path& fixtureDirectory) {
    const std::vector<std::string> names = {
        "unsupported.mforge",
        "missing_tensor.mforge",
        "duplicate_output.mforge",
        "shape_mismatch.mforge",
        "invalid_softmax_axis.mforge"};
    for (const std::string& name : names) {
        modelforge::ModelGraph model;
        modelforge::DiagnosticEngine diagnostics;
        check(!loadAndValidate(fixtureDirectory / name, model, diagnostics),
              name + " should be rejected");
        check(diagnostics.hasErrors(), name + " should produce a diagnostic");
    }
}

void testOutputComparison() {
    const auto report = modelforge::compareOutputs({1.0f, 2.0f}, {1.0f, 2.000001f});
    check(report.passed, "close outputs should pass tolerance comparison");
    check(!modelforge::compareOutputs({1.0f}, {2.0f}).passed,
          "different outputs should fail tolerance comparison");
    check(!modelforge::compareOutputs({1.0f}, {std::nanf("")}).passed,
          "non-finite output should be rejected");
}

}  // namespace

int main(int argc, char* argv[]) {
    const std::filesystem::path fixtureDirectory =
        argc > 1 ? argv[1] : std::filesystem::path("tests") / "fixtures";
    testValidPipeline(fixtureDirectory);
    testConstantFolding(fixtureDirectory);
    testTransposedGemm(fixtureDirectory);
    testOperatorCoverage(fixtureDirectory);
    testInvalidInputs(fixtureDirectory);
    testOutputComparison();

    if (failures == 0) {
        std::cout << "All ModelForge tests passed.\n";
        return 0;
    }
    std::cerr << failures << " ModelForge test(s) failed.\n";
    return 1;
}

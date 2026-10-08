#include "adaptive.h"
#include "analysis.h"
#include "codegen.h"
#include "guardian.h"
#include "ir.h"
#include "loader.h"
#include "optimizer.h"
#include "research.h"
#include "validator.h"
#include "verifier.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
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

bool coverageBinCheck() {
    return modelforge::coverageBin(-1.0f) == 0 && modelforge::coverageBin(-0.05f) == 1 &&
           modelforge::coverageBin(0.0f) == 1 && modelforge::coverageBin(5.0e-4f) == 2 &&
           modelforge::coverageBin(5.0e-3f) == 3 && modelforge::coverageBin(0.05f) == 4 &&
           modelforge::coverageBin(2.0f) == 5;
}

void testGuardianApc() {
    // A depth-3 synthetic model exercises boundary solving beyond the first layer.
    const auto specs = modelforge::makeZooSpecs(4, 20260917);
    modelforge::IRGraph graph;
    modelforge::DiagnosticEngine diagnostics;
    check(modelforge::compileManifestText(modelforge::synthesizeManifest(specs[2]), graph, diagnostics),
          "synthetic zoo model should compile");
    const auto sites = modelforge::analyzeActivationSites(graph);
    check(sites.size() == specs[2].hidden.size(),
          "IBP should analyse every live ReLU layer and skip the unreachable one");

    // IBP soundness: every sampled pre-activation lies inside its interval.
    std::mt19937 generator(7);
    std::uniform_real_distribution<float> uniform(-10.0f, 10.0f);
    bool sound = true;
    for (int sample = 0; sample < 200; ++sample) {
        std::vector<float> input(specs[2].inputs);
        for (float& value : input) value = uniform(generator);
        std::unordered_map<std::string, std::vector<float>> trace;
        modelforge::DiagnosticEngine local;
        modelforge::executeIRTrace(graph, input, local, &trace);
        for (const auto& site : sites) {
            const auto& values = trace.at(site.preactivation);
            for (std::size_t unit = 0; unit < site.width; ++unit) {
                const float slack = 1.0e-3f * (1.0f + std::fabs(values[unit]));
                sound = sound && values[unit] >= site.lower[unit] - slack &&
                        values[unit] <= site.upper[unit] + slack;
            }
        }
    }
    check(sound, "interval bound propagation should contain every sampled pre-activation");

    const auto deep = modelforge::generateDeepBoundaryProbes(graph, sites);
    check(std::any_of(deep.begin(), deep.end(), [](const auto& probe) {
              return probe.name.rfind("deep_boundary_L2_", 0) == 0;
          }), "Newton boundary solving should reach the third ReLU layer");
    for (const auto& probe : deep) {
        check(std::all_of(probe.values.begin(), probe.values.end(),
                          [](float v) { return v >= -10.0f && v <= 10.0f; }),
              "boundary probes must stay inside the input domain");
    }

    const auto suite = modelforge::generateGuardianProbes(graph);
    const auto coverage = modelforge::measureActivationCoverage(graph, sites, suite);
    check(coverage.covered <= coverage.feasible && coverage.ratio() > 0.9,
          "Guardian-APC suite should cover most feasible activation states");
    const auto randomSuite = modelforge::uniformRandomProbes(specs[2].inputs, suite.size(), 1);
    check(modelforge::measureActivationCoverage(graph, sites, randomSuite).ratio() <
              coverage.ratio(),
          "coverage-guided probes should beat uniform random coverage at equal budget");

    // A 1e-3 dead zone in the deepest layer is caught by Guardian-APC.
    modelforge::IRGraph faulty = graph;
    for (auto& instruction : faulty.instructions) {
        if (instruction.nodeName == sites.back().nodeName) {
            instruction.operation = modelforge::IROp::TestReluDeadZone;
            instruction.testFaultParameter = 1.0e-3f;
        }
    }
    check(!modelforge::validateOptimization(graph, faulty, suite, diagnostics).passed,
          "Guardian-APC should expose a narrow dead zone in a deep layer");

    check(coverageBinCheck(), "coverage bins should partition the real line");
    check(std::fabs(modelforge::mcnemarExactP(10, 0) - 0.001953125) < 1e-9,
          "exact McNemar p-value for 10 vs 0 discordant pairs");
    modelforge::Proportion half{50, 100};
    check(half.wilsonLow() < 0.5 && half.wilsonHigh() > 0.5, "Wilson interval should contain p");
}

void testGuardianPlus(const std::filesystem::path& fixtureDirectory) {
    const auto models = fixtureDirectory.parent_path().parent_path() / "models";
    const auto compile = [&](const char* name) {
        modelforge::ModelGraph model;
        modelforge::DiagnosticEngine diagnostics;
        loadAndValidate(models / name, model, diagnostics);
        return *modelforge::buildIR(model, diagnostics);
    };
    const auto reference = compile("iris_demo.mforge");
    modelforge::IRGraph fused = reference;
    modelforge::runOptimizationPass(fused, modelforge::OptimizationPass::DenseReluFusion);
    const auto sites = modelforge::analyzeActivationSites(reference);
    const auto impact = modelforge::analyzeRewriteImpact(reference, fused, sites);
    check(std::count(impact.changedNodes.begin(), impact.changedNodes.end(), "dense_1") == 1 &&
              std::count(impact.changedNodes.begin(), impact.changedNodes.end(), "activation_1") == 1,
          "rewrite impact should list the fused Gemm and ReLU");
    check(impact.impactedSites.size() == 1, "fusion should impact the single ReLU site");
    check(modelforge::runGuardianPlus(reference, fused).firstDetection == 0,
          "Guardian-APC+ must accept a correct fusion");

    modelforge::IRGraph wrong = reference;
    for (auto& instruction : wrong.instructions) {
        if (instruction.nodeName == "activation_1") instruction.operation = modelforge::IROp::Sigmoid;
    }
    const auto result = modelforge::runGuardianPlus(reference, wrong);
    check(result.firstDetection != 0 && result.witness.has_value(),
          "Guardian-APC+ should reject a wrong ReLU rewrite");
    if (result.witness) {
        const auto where = modelforge::localizeDivergence(reference, wrong, result.witness->values);
        check(where.found && where.node == "activation_1",
              "divergence should be localised to the rewritten activation");
    }

    const auto handOk = compile("iris_handopt_ok.mforge");
    const auto handBug = compile("iris_handopt_bug.mforge");
    check(modelforge::runGuardianPlus(reference, handOk).firstDetection == 0,
          "--compare should accept an equivalent hand-optimised model");
    const auto bug = modelforge::runGuardianPlus(reference, handBug);
    check(bug.firstDetection != 0, "--compare should catch a mistyped weight");
    if (bug.witness) {
        check(modelforge::localizeDivergence(reference, handBug, bug.witness->values).node == "dense_2",
              "the mistyped weight should be localised to dense_2");
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
    testGuardianApc();
    testGuardianPlus(fixtureDirectory);

    if (failures == 0) {
        std::cout << "All ModelForge tests passed.\n";
        return 0;
    }
    std::cerr << failures << " ModelForge test(s) failed.\n";
    return 1;
}

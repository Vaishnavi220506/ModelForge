#include "analysis.h"
#include "codegen.h"
#include "guardian.h"
#include "ir.h"
#include "loader.h"
#include "optimizer.h"
#include "shrink.h"
#include "validator.h"
#include "verifier.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct CommandLineOptions {
    std::string inputPath;
    std::string outputDirectory = "generated";
    bool dumpIR = true;
    bool dumpCpp = false;
    bool optimize = true;
    bool verifyOnnx = false;
    bool guardianDemoBug = false;
    std::string comparePath;
    std::string shrink;
    std::string dataPath;
};

void printUsage() {
    std::cout << "ModelForge - ONNX/manifest to standalone C++ compiler (by Vaishnavi)\n\n"
              << "Usage: modelforge <input.mforge|input.onnx> [options]\n\n"
              << "Options:\n"
              << "  --out <directory>  Generated C++ output directory\n"
              << "  --dump-cpp         Print the generated model.cpp source\n"
              << "  --no-opt           Skip optimization passes\n"
              << "  --no-ir            Do not print IR before and after optimization\n"
              << "  --verify-onnx      Compare zero-input output with ONNX Runtime\n"
              << "  --guardian-demo-bug  Demonstrate detection of a deliberately wrong ReLU rewrite\n"
              << "  --shrink <kind>    Compress (fp16, int8, int4, prune30, prune50 or all),\n"
              << "                     hunt for realistic prediction flips, recommend a level\n"
              << "  --data <csv>       Test data for --shrink (features..., label per row)\n"
              << "  --compare <other>  Check that <other> behaves like the input model\n"
              << "                     (e.g. a hand-optimised, converted or edited copy)\n"
              << "  --help             Show this help\n";
}

std::optional<CommandLineOptions> parseArguments(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage();
        return std::nullopt;
    }

    CommandLineOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") {
            printUsage();
            return std::nullopt;
        }
        if (argument == "--no-opt") {
            options.optimize = false;
            continue;
        }
        if (argument == "--dump-cpp") {
            options.dumpCpp = true;
            continue;
        }
        if (argument == "--no-ir") {
            options.dumpIR = false;
            continue;
        }
        if (argument == "--verify-onnx") {
            options.verifyOnnx = true;
            continue;
        }
        if (argument == "--shrink" || argument == "--data") {
            if (index + 1 >= argc) {
                std::cerr << argument << " requires a value\n";
                return std::nullopt;
            }
            (argument == "--shrink" ? options.shrink : options.dataPath) = argv[++index];
            continue;
        }
        if (argument == "--compare") {
            if (index + 1 >= argc) {
                std::cerr << "--compare requires a second model\n";
                return std::nullopt;
            }
            options.comparePath = argv[++index];
            continue;
        }
        if (argument == "--guardian-demo-bug") {
            options.guardianDemoBug = true;
            continue;
        }
        if (argument == "--out") {
            if (index + 1 >= argc) {
                std::cerr << "--out requires a directory\n";
                return std::nullopt;
            }
            options.outputDirectory = argv[++index];
            continue;
        }
        if (!argument.empty() && argument.front() == '-') {
            std::cerr << "Unknown option: " << argument << "\n";
            return std::nullopt;
        }
        if (!options.inputPath.empty()) {
            std::cerr << "Only one input model may be provided\n";
            return std::nullopt;
        }
        options.inputPath = argument;
    }

    if (options.inputPath.empty()) {
        std::cerr << "An input model is required\n";
        return std::nullopt;
    }
    return options;
}

std::string jsonEscape(const std::string& value) {
    std::string escaped;
    for (const char character : value) {
        switch (character) {
            case '\\':
                escaped += "\\\\";
                break;
            case '"':
                escaped += "\\\"";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                escaped += character;
                break;
        }
    }
    return escaped;
}

bool writeOptimizationCertificate(
    const std::filesystem::path& outputDirectory,
    const modelforge::ModelGraph& model,
    const modelforge::OptimizationReport& optimization,
    const modelforge::GuardianReport& guardian,
    const modelforge::TranslationValidationReport& validation,
    const std::vector<modelforge::ValidationProbe>& probes,
    const modelforge::CoverageReport& coverage,
    modelforge::DiagnosticEngine& diagnostics) {
    const std::filesystem::path certificatePath =
        outputDirectory / "optimization_certificate.json";
    std::ofstream certificate(certificatePath);
    if (!certificate) {
        diagnostics.error("report", "Could not write optimization certificate: " +
                                     certificatePath.string());
        return false;
    }

    certificate << std::setprecision(9)
                << "{\n"
                << "  \"model\": \"" << jsonEscape(model.name) << "\",\n"
                << "  \"author\": \"Vaishnavi\",\n"
                << "  \"method\": \"bounded empirical translation validation (Guardian-APC)\",\n"
                << "  \"activation_coverage\": {\"apc\": " << coverage.ratio()
                << ", \"boundary_apc\": " << coverage.boundaryRatio()
                << ", \"covered_states\": " << coverage.covered
                << ", \"feasible_states\": " << coverage.feasible << "},\n"
                << "  \"optimization\": {\n"
                << "    \"instructions_before\": " << optimization.instructionsBefore << ",\n"
                << "    \"instructions_after\": " << optimization.instructionsAfter << ",\n"
                << "    \"removed_instructions\": " << optimization.removedInstructions << ",\n"
                << "    \"fused_operations\": " << optimization.fusedOperations << ",\n"
                << "    \"constant_folds\": " << optimization.constantFolds << "\n"
                << "  },\n"
                << "  \"translation_validation\": {\n"
                << "    \"status\": \"" << (validation.passed ? "PASS" : "FAIL") << "\",\n"
                << "    \"probe_count\": " << validation.probeCount << ",\n"
                << "    \"maximum_absolute_error\": " << validation.maximumAbsoluteError << ",\n"
                << "    \"output_mismatches\": " << validation.outputMismatches << ",\n"
                << "    \"prediction_mismatches\": " << validation.predictionMismatches << "\n"
                << "  },\n"
                << "  \"optimization_events\": [\n";
    for (std::size_t index = 0; index < optimization.events.size(); ++index) {
        certificate << "    \"" << jsonEscape(optimization.events[index]) << "\""
                    << (index + 1 == optimization.events.size() ? "\n" : ",\n");
    }
    certificate << "  ],\n"
                << "  \"guardian_decisions\": [\n";
    for (std::size_t index = 0; index < guardian.decisions.size(); ++index) {
        const auto& decision = guardian.decisions[index];
        certificate << "    {\"pass\": \"" << jsonEscape(decision.passName)
                    << "\", \"accepted\": " << (decision.accepted ? "true" : "false")
                    << ", \"probes\": " << decision.validation.probeCount
                    << ", \"maximum_absolute_error\": ";
        if (std::isfinite(decision.validation.maximumAbsoluteError)) {
            certificate << decision.validation.maximumAbsoluteError;
        } else {
            certificate << "null";
        }
        certificate << ", \"counterexample_file\": "
                    << (decision.accepted ? "null" : "\"counterexample_" +
                        jsonEscape(decision.passName) + ".json\"") << "}"
                    << (index + 1 == guardian.decisions.size() ? "\n" : ",\n");
    }
    certificate << "  ],\n"
                << "  \"probes\": [\n";
    for (std::size_t probeIndex = 0; probeIndex < probes.size(); ++probeIndex) {
        const auto& probe = probes[probeIndex];
        certificate << "    {\"name\": \"" << jsonEscape(probe.name)
                    << "\", \"values\": [";
        for (std::size_t valueIndex = 0; valueIndex < probe.values.size(); ++valueIndex) {
            certificate << probe.values[valueIndex]
                        << (valueIndex + 1 == probe.values.size() ? "" : ", ");
        }
        certificate << "]}"
                    << (probeIndex + 1 == probes.size() ? "\n" : ",\n");
    }
    certificate << "  ]\n}\n";
    return true;
}

bool writeCounterexample(const std::filesystem::path& outputDirectory,
                         const modelforge::GuardianDecision& decision,
                         const std::string& filename,
                         modelforge::DiagnosticEngine& diagnostics) {
    if (!decision.validation.firstFailure) return false;
    const auto path = outputDirectory / filename;
    std::ofstream output(path);
    if (!output) {
        diagnostics.error("report", "Could not write counterexample: " + path.string());
        return false;
    }
    const auto writeVector = [&](const std::vector<float>& values) {
        output << "[";
        for (std::size_t index = 0; index < values.size(); ++index) {
            if (index) output << ", ";
            output << values[index];
        }
        output << "]";
    };
    output << std::setprecision(9) << "{\n  \"rejected_rewrite\": \""
           << jsonEscape(decision.passName) << "\",\n  \"probe\": \""
           << jsonEscape(decision.validation.firstFailure->name) << "\",\n  \"input\": ";
    writeVector(decision.validation.firstFailure->values);
    if (decision.divergence.found) {
        output << ",\n  \"first_divergent_node\": \"" << jsonEscape(decision.divergence.node)
               << "\",\n  \"first_divergent_value\": \"" << jsonEscape(decision.divergence.value)
               << "\",\n  \"first_divergence\": " << decision.divergence.maximumDifference;
    }
    output << ",\n  \"found_by_near_miss_search\": "
           << (decision.foundByNearMiss ? "true" : "false");
    output << ",\n  \"original_output\": ";
    writeVector(decision.validation.originalOutput);
    output << ",\n  \"candidate_output\": ";
    writeVector(decision.validation.candidateOutput);
    output << "\n}\n";
    return static_cast<bool>(output);
}

}  // namespace

std::optional<modelforge::IRGraph> compileForComparison(const std::string& path,
                                                         modelforge::DiagnosticEngine& diagnostics) {
    modelforge::ModelGraph model;
    modelforge::SymbolTable symbols;
    if (!modelforge::loadModel(path, model, diagnostics) ||
        !modelforge::validate(model, diagnostics, symbols)) {
        return std::nullopt;
    }
    return modelforge::buildIR(model, diagnostics);
}

// Model equivalence check: Guardian-APC+ applied to two independently written
// models instead of one rewrite. Exit code 0 = no difference found, 3 = differ.
int compareModels(const CommandLineOptions& options) {
    modelforge::DiagnosticEngine diagnostics;
    std::cout << "Comparing models with Guardian-APC+\n"
              << "  reference: " << options.inputPath << "\n"
              << "  candidate: " << options.comparePath << "\n";
    const auto reference = compileForComparison(options.inputPath, diagnostics);
    const auto candidate = compileForComparison(options.comparePath, diagnostics);
    if (!reference || !candidate) {
        diagnostics.print(std::cerr);
        return 1;
    }
    const auto referenceInput = reference->values.find(reference->inputName);
    const auto candidateInput = candidate->values.find(candidate->inputName);
    if (referenceInput == reference->values.end() || candidateInput == candidate->values.end() ||
        referenceInput->second.shape != candidateInput->second.shape) {
        std::cerr << "The two models must take inputs of the same shape\n";
        return 1;
    }
    modelforge::GuardianPlusOptions guardianOptions;
    guardianOptions.adaptiveBudget = 64;
    const auto result = modelforge::runGuardianPlus(*reference, *candidate, guardianOptions);
    const auto impact = modelforge::analyzeRewriteImpact(
        *reference, *candidate, modelforge::analyzeActivationSites(*reference));

    std::filesystem::create_directories(options.outputDirectory);
    const auto reportPath =
        std::filesystem::path(options.outputDirectory) / "equivalence_report.json";
    std::ofstream report(reportPath);
    const auto writeVector = [&](const std::vector<float>& values) {
        report << "[";
        for (std::size_t i = 0; i < values.size(); ++i) report << (i ? ", " : "") << values[i];
        report << "]";
    };
    report << std::setprecision(9) << "{\n  \"author\": \"Vaishnavi\",\n  \"reference\": \""
           << jsonEscape(options.inputPath) << "\",\n  \"candidate\": \""
           << jsonEscape(options.comparePath) << "\",\n  \"verdict\": \""
           << (result.firstDetection ? "DIFFERENT" : "NO_DIFFERENCE_FOUND")
           << "\",\n  \"evaluations\": " << result.evaluations
           << ",\n  \"changed_nodes\": [";
    for (std::size_t i = 0; i < impact.changedNodes.size(); ++i) {
        report << (i ? ", " : "") << "\"" << jsonEscape(impact.changedNodes[i]) << "\"";
    }
    report << "]";

    std::cout << "\n  Nodes that differ structurally: ";
    if (impact.changedNodes.empty()) std::cout << "none";
    for (std::size_t i = 0; i < impact.changedNodes.size(); ++i) {
        std::cout << (i ? ", " : "") << impact.changedNodes[i];
    }
    std::cout << "\n  Test inputs evaluated: " << result.evaluations << " ("
              << result.staticProbes << " targeted + near-miss search)\n";
    if (result.firstDetection == 0) {
        std::cout << "\n  RESULT: no behavioural difference found (max output difference "
                  << result.maximumAbsoluteError << ").\n"
                  << "  This is strong evidence, not a proof of equivalence.\n";
        report << ",\n  \"maximum_absolute_error\": " << result.maximumAbsoluteError << "\n}\n";
        std::cout << "  Report: " << reportPath.string() << "\n";
        return 0;
    }
    const auto where = modelforge::localizeDivergence(*reference, *candidate,
                                                      result.witness->values);
    std::cout << "\n  RESULT: the models DIFFER (found at test input #" << result.firstDetection
              << (result.foundByNearMiss ? ", by near-miss search" : "") << ").\n  input:     ";
    for (std::size_t i = 0; i < result.witness->values.size(); ++i) {
        std::cout << (i ? ", " : "") << result.witness->values[i];
    }
    std::cout << "\n  reference: ";
    for (std::size_t i = 0; i < result.originalOutput.size(); ++i) {
        std::cout << (i ? ", " : "") << result.originalOutput[i];
    }
    std::cout << "\n  candidate: ";
    for (std::size_t i = 0; i < result.candidateOutput.size(); ++i) {
        std::cout << (i ? ", " : "") << result.candidateOutput[i];
    }
    std::cout << "\n";
    if (where.found) {
        std::cout << "  First difference inside the model: node '" << where.node << "' (value '"
                  << where.value << "'), max difference " << where.maximumDifference << "\n";
    }
    report << ",\n  \"found_by_near_miss_search\": " << (result.foundByNearMiss ? "true" : "false")
           << ",\n  \"input\": ";
    writeVector(result.witness->values);
    report << ",\n  \"reference_output\": ";
    writeVector(result.originalOutput);
    report << ",\n  \"candidate_output\": ";
    writeVector(result.candidateOutput);
    if (where.found) {
        report << ",\n  \"first_divergent_node\": \"" << jsonEscape(where.node) << "\"";
    }
    report << "\n}\n";
    std::cout << "  Report: " << reportPath.string() << "\n";
    return 3;
}

// Shrink-and-check: compress, search for realistic disagreements, recommend.
int shrinkModel(const CommandLineOptions& options) {
    modelforge::DiagnosticEngine diagnostics;
    const auto graph = compileForComparison(options.inputPath, diagnostics);
    if (!graph) {
        diagnostics.print(std::cerr);
        return 1;
    }
    if (options.dataPath.empty()) {
        std::cerr << "--shrink needs --data <test.csv> with rows of features (and labels)\n";
        return 1;
    }
    modelforge::Dataset test;
    std::string error;
    const std::size_t inputs = modelforge::elementCount(graph->values.at(graph->inputName).shape);
    if (!modelforge::loadDataset(options.dataPath, inputs, test, error)) {
        std::cerr << error << "\n";
        return 1;
    }
    std::vector<modelforge::Compression> kinds;
    if (options.shrink == "all") {
        kinds = modelforge::standardCompressions();
    } else {
        modelforge::Compression one;
        if (!modelforge::parseCompression(options.shrink, one)) {
            std::cerr << "Unknown compression '" << options.shrink
                      << "' (use fp16, int8, int4, intN, prune30, pruneNN or all)\n";
            return 1;
        }
        kinds.push_back(one);
    }
    std::filesystem::create_directories(options.outputDirectory);
    const std::filesystem::path out = options.outputDirectory;
    std::ofstream report(out / "shrink_report.json");
    report << std::setprecision(6) << "{\n  \"author\": \"Vaishnavi\",\n  \"model\": \""
           << jsonEscape(options.inputPath) << "\",\n  \"data\": \"" << jsonEscape(options.dataPath)
           << "\",\n  \"rows\": " << test.rows.size() << ",\n  \"results\": [\n";

    std::cout << "Shrink-and-check with boundary-shift probing (by Vaishnavi)\n"
              << "  model: " << options.inputPath << "\n  data:  " << options.dataPath << " ("
              << test.rows.size() << " rows)\n\n"
              << "  compression   size   accuracy        test-set flips   worst realistic flip        risk\n";
    const double baseAccuracy = modelforge::accuracy(*graph, test);
    std::string recommended;
    float recommendedSize = 1.0f;
    for (std::size_t k = 0; k < kinds.size(); ++k) {
        const auto& kind = kinds[k];
        const auto small = modelforge::compressModel(*graph, kind);
        const auto testSet = modelforge::searchTestSet(*graph, small, test);
        const auto search = modelforge::searchBoundaryShift(*graph, small, test, 200);
        const double smallAccuracy = modelforge::accuracy(small, test);
        const std::string risk = modelforge::riskLevel(search.worstSeverity, 0);
        // Fraction of the float32 weight storage that remains (ignoring sparse indexing).
        const float size = kind.kind == modelforge::CompressionKind::Pruning
                               ? 1.0f - kind.pruneFraction
                               : static_cast<float>(kind.bits) / 32.0f;
        std::ostringstream accuracyText;
        accuracyText << std::fixed << std::setprecision(1) << baseAccuracy * 100 << "% -> "
                     << smallAccuracy * 100 << "%";
        std::ostringstream flipText;
        if (search.worstInput.empty()) {
            flipText << "none found";
        } else {
            flipText << std::fixed << std::setprecision(0) << "class " << search.originalClass << "->"
                     << search.compressedClass << ", margin " << std::setprecision(2)
                     << search.worstSeverity;
        }
        std::cout << "  " << std::left << std::setw(12) << kind.name << std::right << std::setw(5)
                  << static_cast<int>(std::round(size * 100)) << "%   " << std::left
                  << std::setw(16) << accuracyText.str() << std::right << std::setw(5)
                  << testSet.disagreements << "/" << std::left << std::setw(10)
                  << test.rows.size() << std::setw(26) << flipText.str() << "  " << risk << "\n";
        if ((risk == "negligible" || risk == "low") && size < recommendedSize) {
            recommended = kind.name;
            recommendedSize = size;
        }
        const auto modelPath = out / (graph->name + "_" + kind.name + ".mforge");
        std::ofstream(modelPath) << modelforge::writeManifestText(small);
        report << "    {\"compression\": \"" << kind.name << "\", \"weight_size\": " << size
               << ", \"accuracy_before\": " << baseAccuracy << ", \"accuracy_after\": "
               << smallAccuracy << ", \"test_set_flips\": " << testSet.disagreements
               << ", \"worst_margin\": " << search.worstSeverity << ", \"risk\": \"" << risk
               << "\", \"boundary_shift\": " << search.maxBoundaryShift
               << ", \"witness_distance\": " << search.worstDistance
               << ", \"compressed_model\": \"" << jsonEscape(modelPath.string()) << "\", \"witness\": [";
        for (std::size_t i = 0; i < search.worstInput.size(); ++i) {
            report << (i ? ", " : "") << search.worstInput[i];
        }
        report << "]}" << (k + 1 == kinds.size() ? "\n" : ",\n");
        if (kinds.size() == 1 && !modelforge::generateCpp(small, (out / "cpp").string(), diagnostics)) {
            diagnostics.print(std::cerr);
            return 1;
        }
    }
    report << "  ],\n  \"recommended\": \"" << recommended << "\"\n}\n";
    std::cout << "\n  'worst realistic flip': an input within 0.5 std of real data where the original\n"
              << "  model and the compressed model disagree; margin = how sure the original was.\n";
    if (!recommended.empty()) {
        std::cout << "\n  RECOMMENDED: " << recommended << " (weights at "
                  << static_cast<int>(std::round(recommendedSize * 100))
                  << "% of float32 size, risk negligible/low)\n";
    } else {
        std::cout << "\n  No tested compression is low-risk for this model.\n";
    }
    std::cout << "  Compressed models and shrink_report.json written to " << out.string() << "\n";
    if (kinds.size() == 1) std::cout << "  Standalone C++ for the compressed model: " << (out / "cpp").string() << "\n";
    return 0;
}

int main(int argc, char* argv[]) {
    const auto options = parseArguments(argc, argv);
    if (!options.has_value()) {
        return argc < 2 ? 1 : 0;
    }

    modelforge::DiagnosticEngine diagnostics;
    modelforge::ModelGraph model;

    if (!options->shrink.empty()) {
        return shrinkModel(*options);
    }
    if (!options->comparePath.empty()) {
        return compareModels(*options);
    }

    std::cout << "[1/5] Loading model...\n";
    if (!modelforge::loadModel(options->inputPath, model, diagnostics)) {
        diagnostics.print(std::cerr);
        return 1;
    }
    std::cout << "      Loaded '" << model.name << "' with " << model.nodes.size() << " nodes.\n";
    modelforge::printSyntaxTree(model, std::cout);

    std::cout << "[2/5] Validating model and building symbol table...\n";
    modelforge::SymbolTable symbols;
    if (!modelforge::validate(model, diagnostics, symbols)) {
        diagnostics.print(std::cerr);
        return 1;
    }
    std::cout << "      Validation passed; symbols: " << symbols.size() << ".\n";
    modelforge::printSymbolTable(symbols, std::cout);

    std::cout << "[3/5] Lowering to ModelForge IR...\n";
    const auto ir = modelforge::buildIR(model, diagnostics);
    if (!ir.has_value()) {
        diagnostics.print(std::cerr);
        return 1;
    }
    if (options->dumpIR) {
        modelforge::printIR(*ir, std::cout);
    }

    modelforge::IRGraph optimizedIR = *ir;
    modelforge::OptimizationReport optimizationReport;
    modelforge::GuardianReport guardianReport;
    std::optional<modelforge::GuardianDecision> demoFailure;
    if (options->guardianDemoBug) {
        modelforge::IRGraph buggyIR = *ir;
        const auto relu = std::find_if(buggyIR.instructions.begin(), buggyIR.instructions.end(),
                                       [](const modelforge::IRInstruction& instruction) {
                                           return instruction.operation == modelforge::IROp::Relu;
                                       });
        if (relu == buggyIR.instructions.end()) {
            std::cerr << "--guardian-demo-bug requires a model with ReLU (try iris_demo.mforge)\n";
            return 1;
        }
        relu->operation = modelforge::IROp::Sigmoid;
        demoFailure = modelforge::checkCandidate(*ir, buggyIR, "deliberately_wrong_relu_rewrite",
                                                diagnostics);
        std::cout << "      Guardian demo: " << (demoFailure->accepted ? "MISSED" : "REJECTED")
                  << " incorrect ReLU rewrite after " << demoFailure->validation.probeCount
                  << " targeted probes.\n";
        if (demoFailure->divergence.found) {
            std::cout << "      Bug localised: outputs first diverge at '"
                      << demoFailure->divergence.node << "' (value '"
                      << demoFailure->divergence.value << "', max difference "
                      << demoFailure->divergence.maximumDifference << ").\n";
        }
        if (demoFailure->accepted || !demoFailure->validation.firstFailure) {
            std::cerr << "Guardian failed to detect the demonstration bug\n";
            return 1;
        }
    }
    // Guardian-APC suite: generic + first-layer probes, IBP-guided deep boundary
    // probes, ordered by activation-pattern coverage gain.
    std::vector<modelforge::ValidationProbe> validationProbes =
        modelforge::generateGuardianProbes(*ir);
    const auto activationSites = modelforge::analyzeActivationSites(*ir);
    const auto coverage =
        modelforge::measureActivationCoverage(*ir, activationSites, validationProbes);
    std::size_t unstableNeurons = 0;
    for (const auto& site : activationSites) {
        unstableNeurons += site.count(modelforge::NeuronStability::Unstable);
    }
    std::cout << "      Guardian-APC: " << validationProbes.size() << " probes, "
              << activationSites.size() << " ReLU layer(s), " << unstableNeurons
              << " IBP-unstable neuron(s), activation-pattern coverage "
              << std::fixed << std::setprecision(1) << coverage.ratio() * 100.0
              << "% (boundary " << coverage.boundaryRatio() * 100.0 << "%).\n"
              << std::defaultfloat << std::setprecision(6);
    modelforge::TranslationValidationReport translationValidation;
    if (options->optimize) {
        std::cout << "[4/5] Optimizing IR...\n";
        guardianReport = modelforge::optimizeWithGuardian(optimizedIR, diagnostics);
        optimizationReport = guardianReport.acceptedOptimizations;
        std::cout << "      Instructions: " << optimizationReport.instructionsBefore << " -> "
                  << optimizationReport.instructionsAfter << "\n"
                  << "      Removed: " << optimizationReport.removedInstructions
                  << ", fused: " << optimizationReport.fusedOperations
                  << ", constant-folded: " << optimizationReport.constantFolds << "\n";
        for (const std::string& event : optimizationReport.events) {
            std::cout << "      Rewrite: " << event << "\n";
        }
        for (const auto& decision : guardianReport.decisions) {
            std::cout << "      Guardian " << decision.passName << ": "
                      << (decision.accepted ? "ACCEPT" : "REJECT")
                      << " (" << decision.validation.probeCount << " probes)\n";
        }
        if (options->dumpIR) {
            modelforge::printIR(optimizedIR, std::cout);
        }
    } else {
        std::cout << "[4/5] Optimization skipped.\n";
    }

    translationValidation = modelforge::validateOptimization(
        *ir, optimizedIR, validationProbes, diagnostics);
    std::cout << "      Translation validation: "
              << (translationValidation.passed ? "PASS" : "FAIL")
              << " across " << translationValidation.probeCount << " graph-aware probes"
              << ", max error " << translationValidation.maximumAbsoluteError << ".\n";
    if (!translationValidation.passed) {
        diagnostics.error("verification", "Optimization changed model behavior");
        diagnostics.print(std::cerr);
        return 1;
    }

    std::cout << "[5/5] Generating standalone C++ code and running IR execution check...\n";
    if (!modelforge::generateCpp(optimizedIR, options->outputDirectory, diagnostics)) {
        diagnostics.print(std::cerr);
        return 1;
    }
    std::cout << "      Generated " << options->outputDirectory << "/model.h and model.cpp.\n";

    if (demoFailure && !writeCounterexample(options->outputDirectory, *demoFailure,
                                             "counterexample.json",
                                             diagnostics)) {
        diagnostics.print(std::cerr);
        return 1;
    }
    if (demoFailure) {
        std::cout << "      Wrote counterexample.json for the rejected demo rewrite.\n";
    }
    for (const auto& decision : guardianReport.decisions) {
        if (!decision.accepted && decision.validation.firstFailure &&
            !writeCounterexample(options->outputDirectory, decision,
                                 "counterexample_" + decision.passName + ".json",
                                 diagnostics)) {
            diagnostics.print(std::cerr);
            return 1;
        }
    }

    if (!writeOptimizationCertificate(options->outputDirectory,
                                      model,
                                      optimizationReport,
                                      guardianReport,
                                      translationValidation,
                                      validationProbes,
                                      coverage,
                                      diagnostics)) {
        diagnostics.print(std::cerr);
        return 1;
    }
    std::cout << "      Wrote optimization_certificate.json.\n";

    if (options->dumpCpp) {
        const std::filesystem::path generatedCpp =
            std::filesystem::path(options->outputDirectory) / "model.cpp";
        std::ifstream source(generatedCpp);
        if (!source) {
            std::cerr << "Could not read generated C++ file: " << generatedCpp << "\n";
            return 1;
        }
        std::cout << "\n===== GENERATED model.cpp =====\n";
        std::cout << source.rdbuf();
        std::cout << "\n===== END GENERATED model.cpp =====\n";
    }

    const auto inputInfo = optimizedIR.values.find(optimizedIR.inputName);
    if (inputInfo == optimizedIR.values.end()) {
        std::cerr << "Runtime input information is missing\n";
        return 1;
    }
    const std::vector<float> sampleInput(modelforge::elementCount(inputInfo->second.shape), 0.0f);
    const auto executionResult = modelforge::executeIR(optimizedIR, sampleInput, diagnostics);
    if (!executionResult.has_value()) {
        diagnostics.print(std::cerr);
        return 1;
    }
    const auto best = std::max_element(executionResult->begin(), executionResult->end());
    std::cout << "      IR execution passed; predicted class "
              << (best - executionResult->begin()) << " with confidence " << *best << ".\n";

    if (options->verifyOnnx) {
        if (std::filesystem::path(options->inputPath).extension() != ".onnx") {
            std::cerr << "--verify-onnx requires an .onnx input model\n";
            return 1;
        }
        const modelforge::VerificationReport report = modelforge::verifyWithOnnxRuntime(
            options->inputPath, optimizedIR, sampleInput, diagnostics);
        if (diagnostics.hasErrors()) {
            diagnostics.print(std::cerr);
            return 1;
        }
        std::cout << "      ONNX comparison: " << (report.passed ? "PASSED" : "FAILED")
                  << ", max absolute error: " << report.maximumAbsoluteError
                  << ", mismatches: " << report.mismatches << ".\n";
        if (!report.passed) {
            return 1;
        }
    }

    if (!diagnostics.all().empty()) {
        diagnostics.print(std::cerr);
    }
    std::cout << "\nCOMPILATION PIPELINE COMPLETED\n";
    std::cout << "Compile the generated program with the Visual Studio C++ toolchain.\n";
    return 0;
}

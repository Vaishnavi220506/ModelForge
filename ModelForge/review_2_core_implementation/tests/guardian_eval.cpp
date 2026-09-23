#include "ir.h"
#include "loader.h"
#include "validator.h"
#include "verifier.h"

#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using modelforge::IRGraph;
using modelforge::IROp;
using modelforge::ValidationProbe;

struct Case {
    std::string name;
    IRGraph candidate;
    bool shouldChangeOutput;
};

bool detected(const IRGraph& original,
              const IRGraph& candidate,
              const std::vector<ValidationProbe>& probes) {
    modelforge::DiagnosticEngine diagnostics;
    const auto result = modelforge::validateOptimization(original, candidate, probes, diagnostics);
    return !result.passed;
}

std::size_t firstDetection(const IRGraph& original,
                           const IRGraph& candidate,
                           const std::vector<ValidationProbe>& probes) {
    for (std::size_t index = 0; index < probes.size(); ++index) {
        if (detected(original, candidate, {probes[index]})) return index + 1;
    }
    return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: modelforge_guardian_eval <iris_demo.mforge>\n";
        return 2;
    }
    modelforge::DiagnosticEngine diagnostics;
    modelforge::ModelGraph model;
    modelforge::SymbolTable symbols;
    if (!modelforge::loadModel(argv[1], model, diagnostics) ||
        !modelforge::validate(model, diagnostics, symbols)) {
        diagnostics.print(std::cerr);
        return 1;
    }
    const auto original = modelforge::buildIR(model, diagnostics);
    if (!original) {
        diagnostics.print(std::cerr);
        return 1;
    }

    std::vector<Case> cases;
    {
        IRGraph candidate = *original;
        for (auto& instruction : candidate.instructions) {
            if (instruction.operation == IROp::Relu && instruction.output == "hidden2") {
                instruction.operation = IROp::Sigmoid;
                break;
            }
        }
        cases.push_back({"relu_to_sigmoid", std::move(candidate), true});
    }
    {
        IRGraph candidate = *original;
        for (auto& instruction : candidate.instructions) {
            if (instruction.operation == IROp::Softmax) {
                instruction.operation = IROp::Sigmoid;
                break;
            }
        }
        cases.push_back({"softmax_to_sigmoid", std::move(candidate), true});
    }
    {
        IRGraph candidate = *original;
        for (auto& instruction : candidate.instructions) {
            if (instruction.operation == IROp::Gemm && instruction.inputs.size() == 3) {
                instruction.inputs.pop_back();
                break;
            }
        }
        cases.push_back({"missing_bias", std::move(candidate), true});
    }
    {
        IRGraph candidate = *original;
        candidate.constants.at("w1").data[0] += 5.0f;
        cases.push_back({"weight_index_zero", std::move(candidate), true});
    }
    {
        IRGraph candidate = *original;
        for (auto& instruction : candidate.instructions) {
            if (instruction.operation == IROp::Relu && instruction.output == "hidden2") {
                instruction.operation = IROp::TestReluDeadZone;
                break;
            }
        }
        cases.push_back({"relu_near_boundary_dead_zone", std::move(candidate), true});
    }
    {
        IRGraph candidate = *original;
        for (auto& instruction : candidate.instructions) {
            if (instruction.nodeName == "unused_activation") {
                instruction.operation = IROp::Sigmoid;
                break;
            }
        }
        cases.push_back({"unreachable_change", std::move(candidate), false});
    }

    const auto directed = modelforge::generateValidationProbes(*original);
    std::vector<ValidationProbe> generic;
    for (const auto& probe : directed) {
        if (probe.name.rfind("affine_relu_", 0) != 0 &&
            probe.name.rfind("decision_boundary_", 0) != 0) {
            generic.push_back(probe);
        }
    }
    const auto inputCount = modelforge::elementCount(original->values.at(original->inputName).shape);
    const std::vector<ValidationProbe> zero = {{"zero", std::vector<float>(inputCount, 0.0f)}};
    std::vector<ValidationProbe> random;
    std::mt19937 engine(20260917);
    std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);
    for (std::size_t index = 0; index < directed.size(); ++index) {
        std::vector<float> values(inputCount);
        for (float& value : values) value = distribution(engine);
        random.push_back({"random_" + std::to_string(index), std::move(values)});
    }

    std::cout << "fault,expected_change,zero_detected,random_detected,guardian_detected,"
                 "random_probes,guardian_probes,random_first_detection,guardian_first_detection,"
                 "generic_detected,generic_first_detection,generic_probes\n";
    bool valid = true;
    for (const Case& test : cases) {
        const bool z = detected(*original, test.candidate, zero);
        const bool r = detected(*original, test.candidate, random);
        const bool g = detected(*original, test.candidate, directed);
        const std::size_t randomFirst = firstDetection(*original, test.candidate, random);
        const std::size_t guardianFirst = firstDetection(*original, test.candidate, directed);
        const std::size_t genericFirst = firstDetection(*original, test.candidate, generic);
        std::cout << test.name << ',' << test.shouldChangeOutput << ',' << z << ',' << r << ','
                  << g << ',' << random.size() << ',' << directed.size() << ','
                  << randomFirst << ',' << guardianFirst << ',' << (genericFirst != 0) << ','
                  << genericFirst << ',' << generic.size() << '\n';
        if (g != test.shouldChangeOutput) valid = false;
        if (test.name == "weight_index_zero" && z) valid = false;
    }
    return valid ? 0 : 1;
}

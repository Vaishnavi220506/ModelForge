// modelforge_certbench: evaluation of ForgeCert (certified integer compilation).
// Author: Vaishnavi
//
// For the 8 real classifiers (Iris, Wine, Breast Cancer, Digits; MLP 16 and
// 32x16) and several region radii it compares
//   * common practice: uniform int8, uniform int4 weights, uniform int16, and
//     accuracy-guided precision selection (lower every width while no test
//     prediction changes), each then checked by ForgeCert's prover + falsifier;
//   * ForgeCert's counterexample-guided tuner and its ablations (uniform or
//     random refinement instead of provenance, interval domain, per-tensor
//     scales, no falsifier, no minimisation);
//   * the exhaustive optimum over the byte-level precision lattice;
// and audits soundness by sampling, and times the independent checker.
//
// Usage: modelforge_certbench [--quick] [--out dir] [--regions N]

#include "certify.h"
#include "research.h"
#include "shrink.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifndef MODELFORGE_MODELS_DIR
#define MODELFORGE_MODELS_DIR "models"
#endif
#ifndef MODELFORGE_CERTCHECK
#define MODELFORGE_CERTCHECK ""
#endif

namespace fs = std::filesystem;
using namespace modelforge;

namespace {

struct Outcome {
    std::string name;
    bool ok = false;
    std::string config;
    double flash = 0, ram = 0, weightBytes = 0, cost = 0, idealBytes = 0;
    std::size_t proved = 0, unproved = 0, counterexamples = 0, excluded = 0;
    std::size_t iterations = 0, analyses = 0, lowerings = 0;
    double seconds = 0;
    double cycles = 0;          // estimated AVR cycles per inference
    bool calibrated = false;
    std::size_t testFlips = 0;  // integer vs float disagreements on the test set
    std::vector<int> accBits, worstAccBits;
};

std::size_t testFlips(const DenseNetwork& network, const QProgram& program, const Dataset& test) {
    std::size_t flips = 0;
    for (const auto& row : test.rows) {
        const std::vector<double> x(row.begin(), row.end());
        flips += runInteger(program, x).decision != argmaxFirst(referenceLogits(network, x));
    }
    return flips;
}

Outcome summarise(const std::string& name, const DenseNetwork& network, const TunerResult& r, const Dataset& test) {
    Outcome o;
    o.name = name;
    o.ok = r.ok;
    if (!r.ok) return o;
    o.config = r.program.config.describe();
    const auto c = programCost(r.program);
    o.flash = c.flashBytes;
    o.ram = c.ramBytes;
    o.weightBytes = c.weightBytes;
    o.cost = configCost(r.program, CostModel::Bytes);
    o.idealBytes = configCost(r.program, CostModel::Bits);
    o.proved = r.proved;
    o.unproved = r.unproved;
    o.excluded = r.excluded;
    for (const auto& s : r.regions) o.counterexamples += !s.excluded && !s.proved && s.counterexample;
    o.iterations = r.iterations;
    o.analyses = r.analyses;
    o.lowerings = r.lowerings;
    o.seconds = r.seconds;
    o.cycles = estimatedAvrCycles(r.program);
    o.calibrated = r.program.calibrated;
    o.testFlips = testFlips(network, r.program, test);
    for (const auto& layer : r.program.layers) {
        o.accBits.push_back(layer.accBits);
        // Without range analysis a compiler must assume every input at the
        // extreme of its storage type.
        std::int64_t extreme = 0;
        for (std::size_t source : layer.sources) {
            extreme = std::max(extreme, source == 0 ? r.program.inputMax : r.program.layers[source - 1].outMax);
        }
        o.worstAccBits.push_back(worstCaseAccumulatorBits(layer, extreme));
    }
    return o;
}

// Common practice: start at 16 bits everywhere and greedily lower the width
// that saves the most bytes while no test prediction changes.
PrecisionConfig accuracyGuided(const DenseNetwork& network, const Box& domain, const Dataset& test) {
    const std::size_t L = network.layers.size();
    PrecisionConfig config = uniformConfig(L, 16, 16);
    const auto wl = weightLadder(CostModel::Bytes);
    const auto al = activationLadder(CostModel::Bytes);
    std::string error;
    bool changed = true;
    while (changed) {
        changed = false;
        auto current = lowerToInteger(network, domain, config, error);
        const double cost = configCost(*current, CostModel::Bytes);
        std::vector<std::pair<double, PrecisionConfig>> options;
        for (std::size_t k = 0; k < 2 * L; ++k) {
            if (k < L ? fixedSlot(network, 1, k) : fixedSlot(network, 0, k - L)) continue;
            PrecisionConfig trial = config;
            int& value = k < L ? trial.weightBits[k] : trial.activationBits[k - L];
            const auto& ladder = k < L ? wl : al;
            int lower = 0;
            for (int rung : ladder) {
                if (rung < value) lower = rung;
            }
            if (lower == 0) continue;
            value = lower;
            const auto program = lowerToInteger(network, domain, trial, error);
            if (!program) continue;
            options.push_back({cost - configCost(*program, CostModel::Bytes), trial});
        }
        std::sort(options.begin(), options.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        for (const auto& option : options) {
            const auto program = lowerToInteger(network, domain, option.second, error);
            if (program && testFlips(network, *program, test) == 0) {
                config = option.second;
                changed = true;
                break;
            }
        }
    }
    return config;
}

// Exhaustive search of the byte lattice: cheapest configuration that proves
// every robust region (regions checked with early exit).
std::pair<PrecisionConfig, std::size_t> exhaustive(const DenseNetwork& network, const Box& domain,
                                                   const std::vector<CertRegion>& regions,
                                                   const std::vector<bool>& robust, const ProofOptions& proof,
                                                   double& bestCost) {
    const std::size_t L = network.layers.size();
    const auto wl = weightLadder(CostModel::Bytes);
    const auto al = activationLadder(CostModel::Bytes);
    std::size_t total = 1;
    for (std::size_t k = 0; k < L; ++k) total *= wl.size() * al.size();
    std::vector<std::pair<double, PrecisionConfig>> all;
    std::string error;
    for (std::size_t code = 0; code < total; ++code) {
        PrecisionConfig config = uniformConfig(L, 8, 8);
        std::size_t rest = code;
        for (std::size_t k = 0; k < L; ++k) {
            config.weightBits[k] = wl[rest % wl.size()];
            rest /= wl.size();
            config.activationBits[k] = al[rest % al.size()];
            rest /= al.size();
        }
        const auto program = lowerToInteger(network, domain, config, error);
        if (program) all.push_back({configCost(*program, CostModel::Bytes), config});
    }
    std::sort(all.begin(), all.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    for (const auto& candidate : all) {
        const auto program = lowerToInteger(network, domain, candidate.second, error);
        bool ok = true;
        for (std::size_t r = 0; r < regions.size() && ok; ++r) {
            if (!robust[r]) continue;
            ok = proveRegion(network, *program, regions[r], proof).proved;
        }
        if (ok) {
            bestCost = candidate.first;
            return {candidate.second, all.size()};
        }
    }
    bestCost = -1;
    return {uniformConfig(L, 16, 16), all.size()};
}

struct Audit {
    std::size_t samples = 0, violations = 0;
    double maxRatio = 0;  // observed / proved logit error, over proved regions
};

Audit audit(const DenseNetwork& network, const TunerResult& r, const std::vector<CertRegion>& regions,
            std::size_t perRegion) {
    // With calibrated scales saturation is allowed (and modelled), not a violation.
    const bool saturationAllowed = r.program.calibrated;
    Audit a;
    std::mt19937 rng(11);
    for (std::size_t k = 0; k < regions.size(); ++k) {
        if (!r.regions[k].proved) continue;
        const auto bounds = analyzeRegion(network, r.program, regions[k].box);
        const auto& box = regions[k].box;
        for (std::size_t s = 0; s < perRegion; ++s) {
            std::vector<double> x(box.size());
            for (std::size_t i = 0; i < x.size(); ++i) {
                double u = std::uniform_real_distribution<double>(0, 1)(rng);
                if (s % 3 == 0) u = u < 0.5 ? 0.0 : 1.0;
                x[i] = box.lo[i] + u * (box.hi[i] - box.lo[i]);
            }
            const auto run = runInteger(r.program, x);
            const auto ref = referenceLogits(network, x);
            const auto logits = dequantizedLogits(r.program, run);
            ++a.samples;
            bool bad = run.decision != regions[k].refClass || argmaxFirst(ref) != regions[k].refClass ||
                       run.accumulatorOverflow || run.productOverflow || (run.saturated && !saturationAllowed);
            for (std::size_t c = 0; c < logits.size(); ++c) {
                const double err = std::fabs(logits[c] - ref[c]);
                if (err > bounds.logitErrorBound[c]) bad = true;
                if (bounds.logitErrorBound[c] > 0) a.maxRatio = std::max(a.maxRatio, err / bounds.logitErrorBound[c]);
            }
            a.violations += bad;
        }
    }
    const auto& d = r.program.domain;
    for (int s = 0; s < 5000; ++s) {
        std::vector<double> x(d.size());
        for (std::size_t i = 0; i < x.size(); ++i) {
            double u = std::uniform_real_distribution<double>(0, 1)(rng);
            if (s % 2) u = u < 0.5 ? 0.0 : 1.0;
            x[i] = d.lo[i] + u * (d.hi[i] - d.lo[i]);
        }
        const auto run = runInteger(r.program, x);
        ++a.samples;
        a.violations += run.accumulatorOverflow || run.productOverflow || (run.saturated && !saturationAllowed);
    }
    return a;
}

std::string json(const Outcome& o) {
    std::ostringstream t;
    t << std::setprecision(6) << "{\"name\": \"" << o.name << "\", \"ok\": " << (o.ok ? "true" : "false")
      << ", \"config\": \"" << o.config << "\", \"flash\": " << o.flash << ", \"ram\": " << o.ram
      << ", \"weight_bytes\": " << o.weightBytes << ", \"cost\": " << o.cost << ", \"ideal_bytes\": " << o.idealBytes
      << ", \"proved\": " << o.proved << ", \"unproved\": " << o.unproved
      << ", \"counterexamples\": " << o.counterexamples << ", \"excluded\": " << o.excluded
      << ", \"iterations\": " << o.iterations << ", \"analyses\": " << o.analyses << ", \"lowerings\": " << o.lowerings
      << ", \"seconds\": " << o.seconds << ", \"cycles\": " << o.cycles
      << ", \"calibrated\": " << (o.calibrated ? "true" : "false") << ", \"test_flips\": " << o.testFlips << ", \"acc_bits\": [";
    for (std::size_t i = 0; i < o.accBits.size(); ++i) t << (i ? ", " : "") << o.accBits[i];
    t << "], \"worst_case_acc_bits\": [";
    for (std::size_t i = 0; i < o.worstAccBits.size(); ++i) t << (i ? ", " : "") << o.worstAccBits[i];
    t << "]}";
    return t.str();
}

}  // namespace

int main(int argc, char** argv) {
    bool quick = false;
    std::string outDir = "certbench";
    std::size_t maxRegions = 100;
    std::string suite = "all";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--quick") quick = true;
        if (a == "--suite" && i + 1 < argc) suite = argv[++i];
        if (a == "--out" && i + 1 < argc) outDir = argv[++i];
        if (a == "--regions" && i + 1 < argc) maxRegions = static_cast<std::size_t>(std::atoi(argv[++i]));
    }
    fs::create_directories(outDir);
    // Tabular models: standardised features, domain = training range + 10%.
    // Image models: pixels in [0, 1], domain = [0, 1]^784, calibration rows
    // from the training split.
    struct Case {
        std::string folder, model, dataset;
        bool image;
        std::vector<double> radii;
        bool lite = false;  // CIFAR-10: baselines and the three tuner objectives only
    };
    std::vector<Case> cases;
    if (suite == "all" || suite == "tabular") {
        for (const char* m : {"iris_mlp16", "iris_mlp32x16", "wine_mlp16", "wine_mlp32x16", "breast_cancer_mlp16",
                              "breast_cancer_mlp32x16", "digits_mlp16", "digits_mlp32x16"}) {
            const std::string name = m;
            cases.push_back({"real", name, name.substr(0, name.find("_mlp")), false, {0.05, 0.1, 0.2}});
        }
    }
    if (suite == "all" || suite == "image") {
        for (const char* m : {"mnist_mlp", "mnist_cnn", "fashion_mlp", "fashion_cnn"}) {
            const std::string name = m;
            cases.push_back({"image", name, name.substr(0, name.find('_')), true, {0.01, 0.02}});
        }
        cases.push_back({"image", "cifar_cnn", "cifar", true, {1.0 / 255.0}, true});
    }
    if (suite == "all" || suite == "zoo") {
        // Architectures beyond plain conv stacks: max/avg pooling, depthwise-
        // separable blocks with batch norm, residual connections.
        for (const char* d : {"mnist", "fashion"}) {
            for (const char* a : {"lenet", "lenet_avg", "dscnn", "resnet"}) {
                cases.push_back({"image", std::string(d) + "_" + a, d, true, {1.0 / 255.0}, true});
            }
        }
    }
    if (quick) {
        cases = {{"real", "iris_mlp16", "iris", false, {0.1}}, {"real", "wine_mlp16", "wine", false, {0.1}}};
        maxRegions = std::min<std::size_t>(maxRegions, 30);
    }
    std::ofstream results(fs::path(outDir) / "certbench.json");
    results << "{\n  \"author\": \"Vaishnavi\",\n  \"max_regions\": " << maxRegions << ",\n  \"cases\": [\n";
    bool firstCase = true;
    std::cout << std::fixed;
    const auto start = std::chrono::steady_clock::now();
    for (const auto& entry : cases) {
        const std::string& model = entry.model;
        const std::string& dataset = entry.dataset;
        const fs::path real = fs::path(MODELFORGE_MODELS_DIR) / entry.folder;
        const std::vector<double>& radii = entry.radii;
        std::ifstream file(real / (model + ".mforge"));
        const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        IRGraph graph;
        DiagnosticEngine diagnostics;
        std::string error;
        if (!compileManifestText(text, graph, diagnostics)) {
            std::cerr << "cannot compile " << model << "\n";
            return 1;
        }
        const auto network = extractDenseNetwork(graph, error);
        Dataset train, test;
        const std::string calibrationFile = entry.image ? dataset + "_calib.csv" : dataset + "_train.csv";
        if (!network || !loadDataset((real / calibrationFile).string(), network->inputs, train, error) ||
            !loadDataset((real / (dataset + "_test.csv")).string(), network->inputs, test, error)) {
            std::cerr << "cannot load " << model << ": " << error << "\n";
            return 1;
        }
        Box domain = domainFromData(train.rows);
        if (entry.image) {
            domain.lo.assign(network->inputs, 0.0);
            domain.hi.assign(network->inputs, 1.0);
        }
        auto calibration = std::make_shared<std::vector<std::vector<double>>>();
        for (const auto& row : train.rows) calibration->emplace_back(row.begin(), row.end());
        const PrecisionConfig guided = entry.lite ? uniformConfig(network->layers.size(), 8, 8)
                                                  : accuracyGuided(*network, domain, test);
        for (double eps : radii) {
            // Lite cases (large images, deep graphs) use at most 50 regions.
            const auto regions =
                regionsAroundRows(*network, test.rows, eps, domain, entry.lite ? std::min<std::size_t>(maxRegions, 50) : maxRegions);
            TunerOptions base;
            const auto robust = referenceRobustRegions(*network, domain, regions, base.proof);
            std::vector<Outcome> outcomes;
            const auto fixed = [&](const std::string& name, const PrecisionConfig& config) {
                outcomes.push_back(summarise(
                    name, *network, checkConfiguration(*network, domain, regions, config, base.proof, &robust, 400, 3),
                    test));
            };
            const std::size_t L = network->layers.size();
            fixed("uniform_int8", uniformConfig(L, 8, 8));
            {
                // Industry default (TFLite-style): int8 with scales calibrated on data.
                ProofOptions calibrated = base.proof;
                calibrated.analysis.calibration = calibration;
                outcomes.push_back(summarise("uniform_int8_calibrated", *network,
                                             checkConfiguration(*network, domain, regions, uniformConfig(L, 8, 8),
                                                                calibrated, &robust, 400, 3),
                                             test));
            }
            fixed("uniform_int4", uniformConfig(L, 4, 8));
            fixed("uniform_int16", uniformConfig(L, 16, 16));
            if (!entry.lite) fixed("accuracy_guided", guided);

            const auto tuned = tunePrecision(*network, domain, regions, base, &robust);
            outcomes.push_back(summarise("forgecert", *network, tuned, test));
            const auto variant = [&](const std::string& name, const std::function<void(TunerOptions&)>& edit) {
                TunerOptions options = base;
                edit(options);
                const auto r = tunePrecision(*network, domain, regions, options, &robust);
                outcomes.push_back(summarise(name, *network, r, test));
            };
            variant("forgecert_calibrated", [&](TunerOptions& o) { o.proof.analysis.calibration = calibration; });
            variant("forgecert_latency", [](TunerOptions& o) { o.cost = CostModel::Latency; });
            if (!entry.lite) {
                variant("ablation_interval_domain", [](TunerOptions& o) { o.proof.analysis.affine = false; });
                variant("ablation_per_tensor_scales",
                        [](TunerOptions& o) { o.proof.analysis.perChannelScales = false; });
                variant("ablation_no_modular_accumulation",
                        [](TunerOptions& o) { o.proof.analysis.modularAccumulation = false; });
                variant("ablation_latency_greedy_search", [](TunerOptions& o) {
                    o.cost = CostModel::Latency;
                    o.compoundMoves = false;
                    o.multiStart = false;
                });
            }
            if (!entry.image) {  // the full set of tuner ablations on the tabular suite
                variant("ablation_uniform_refine", [](TunerOptions& o) { o.strategy = RefineStrategy::Uniform; });
                for (std::uint32_t seed : {1u, 2u, 3u}) {
                    variant("ablation_random_refine_s" + std::to_string(seed), [seed](TunerOptions& o) {
                        o.strategy = RefineStrategy::RandomSource;
                        o.seed = seed;
                    });
                }
                variant("ablation_no_falsifier", [](TunerOptions& o) { o.falsifyFirst = false; });
                variant("ablation_no_minimise", [](TunerOptions& o) { o.minimize = false; });
                variant("ablation_no_splitting", [](TunerOptions& o) { o.proof.maxDepth = 0; });
                variant("bits_cost_model", [](TunerOptions& o) { o.cost = CostModel::Bits; });
                variant("bits_uniform_refine", [](TunerOptions& o) {
                    o.cost = CostModel::Bits;
                    o.strategy = RefineStrategy::Uniform;
                });
                for (std::uint32_t seed : {1u, 2u, 3u}) {
                    variant("bits_random_refine_s" + std::to_string(seed), [seed](TunerOptions& o) {
                        o.cost = CostModel::Bits;
                        o.strategy = RefineStrategy::RandomSource;
                        o.seed = seed;
                    });
                }
            }

            double optimum = -1;
            std::size_t lattice = 0;
            PrecisionConfig best;
            if (!quick && !entry.image && eps == 0.1) {
                const auto e = exhaustive(*network, domain, regions, robust, base.proof, optimum);
                best = e.first;
                lattice = e.second;
            }
            const auto soundness = audit(*network, tuned, regions, quick || entry.lite ? 200 : 2000);

            // Independent checker on the emitted artifact.
            double checkSeconds = -1;
            int checkExit = -1;
            if (!quick && std::string(MODELFORGE_CERTCHECK).size() > 0 && tuned.ok) {
                const fs::path artifact = fs::path(outDir) / "artifacts" / (model + "_eps" + std::to_string(eps).substr(0, 6));
                emitIntegerC(tuned.program, artifact.string(), error);
                writeCertificate((artifact / "certificate.json").string(), (real / (model + ".mforge")).string(),
                                 *network, regions, tuned, eps, error);
                const std::string command = std::string("\"") + MODELFORGE_CERTCHECK + "\" \"" + artifact.string() +
                                            "\" \"" + (real / (model + ".mforge")).string() + "\" > \"" +
                                            (artifact / "check.txt").string() + "\"";
                const auto t0 = std::chrono::steady_clock::now();
                checkExit = std::system(command.c_str());
                checkSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            }

            std::size_t robustCount = 0;
            for (bool b : robust) robustCount += b;
            std::cout << "\n== " << model << "  eps " << std::setprecision(2) << eps << "  regions " << regions.size()
                      << " (float model provably constant on " << robustCount << ")  float32 "
                      << std::setprecision(0) << float32Bytes(*network) << " B\n";
            std::cout << "  " << std::left << std::setw(30) << "method" << std::setw(26) << "precision" << std::right
                      << std::setw(8) << "flash" << std::setw(8) << "proved" << std::setw(9) << "unproved"
                      << std::setw(6) << "cex" << std::setw(6) << "flips" << std::setw(10) << "cycles"
                      << std::setw(8) << "time\n";
            for (const auto& o : outcomes) {
                std::cout << "  " << std::left << std::setw(30) << o.name << std::setw(26) << o.config << std::right
                          << std::setw(8) << std::setprecision(0) << o.flash << std::setw(8) << o.proved
                          << std::setw(9) << o.unproved << std::setw(6) << o.counterexamples << std::setw(6)
                          << o.testFlips << std::setw(10) << std::setprecision(0) << o.cycles << std::setw(8)
                          << std::setprecision(2) << o.seconds << "s\n";
            }
            if (optimum >= 0) {
                std::cout << "  exhaustive optimum over " << lattice << " configurations: " << best.describe()
                          << " cost " << std::setprecision(0) << optimum << " B (ForgeCert "
                          << std::find_if(outcomes.begin(), outcomes.end(),
                                          [](const Outcome& o) { return o.name == "forgecert"; })->cost
                          << " B)\n";
            }
            std::cout << "  soundness audit: " << soundness.samples << " samples, " << soundness.violations
                      << " violations, max observed/proved error " << std::setprecision(3) << soundness.maxRatio
                      << "\n";
            if (checkSeconds >= 0) {
                std::cout << "  independent checker: " << (checkExit == 0 ? "VALID" : "REJECTED") << " in "
                          << std::setprecision(3) << checkSeconds << " s\n";
            }

            results << (firstCase ? "" : ",\n") << "    {\"model\": \"" << model << "\", \"eps\": " << eps
                    << ", \"regions\": " << regions.size() << ", \"robust\": " << robustCount
                    << ", \"float32_bytes\": " << float32Bytes(*network) << ", \"test_rows\": " << test.rows.size()
                    << ",\n     \"outcomes\": [\n";
            firstCase = false;
            for (std::size_t i = 0; i < outcomes.size(); ++i) {
                results << "       " << json(outcomes[i]) << (i + 1 < outcomes.size() ? ",\n" : "\n");
            }
            results << "     ],\n     \"exhaustive\": {\"lattice\": " << lattice << ", \"optimum_cost\": " << optimum
                    << ", \"config\": \"" << (optimum >= 0 ? best.describe() : "") << "\"},\n"
                    << "     \"audit\": {\"samples\": " << soundness.samples << ", \"violations\": "
                    << soundness.violations << ", \"max_observed_over_proved\": " << soundness.maxRatio << "},\n"
                    << "     \"checker\": {\"seconds\": " << checkSeconds << ", \"valid\": "
                    << (checkExit == 0 ? "true" : "false") << "}}";
        }
    }
    const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    results << "\n  ],\n  \"seconds\": " << total << "\n}\n";
    std::cout << "\nWrote " << (fs::path(outDir) / "certbench.json").string() << " in " << std::setprecision(1) << total
              << " s\n";
    return 0;
}

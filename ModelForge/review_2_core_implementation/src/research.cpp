#include "research.h"

#include "adaptive.h"
#include "guardian.h"
#include "loader.h"
#include "optimizer.h"
#include "validator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace modelforge {
namespace {

// ------------------------------------------------------------ tiny JSON writer

std::string number(double value, int precision = 6) {
    if (!std::isfinite(value)) return "null";
    std::ostringstream output;
    output << std::setprecision(precision) << value;
    return output.str();
}

std::string quoted(const std::string& value) { return "\"" + jsonEscapeText(value) + "\""; }

template <typename T, typename F>
std::string array(const std::vector<T>& values, F format) {
    std::string result = "[";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) result += ",";
        result += format(values[index]);
    }
    return result + "]";
}

std::string floats(const std::vector<float>& values) {
    return array(values, [](float value) { return number(value, 7); });
}

std::string platformName() {
#if defined(_WIN32)
    return "Windows";
#elif defined(__APPLE__)
    return "macOS";
#elif defined(__linux__)
    return "Linux";
#else
    return "unknown";
#endif
}

std::string compilerName() {
    std::ostringstream output;
#if defined(__clang__)
    output << "Clang " << __clang_major__ << "." << __clang_minor__;
#elif defined(__GNUC__)
    output << "GCC " << __GNUC__ << "." << __GNUC_MINOR__;
#elif defined(_MSC_VER)
    output << "MSVC " << _MSC_VER;
#else
    output << "unknown";
#endif
    return output.str();
}

std::string utcTimestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return output.str();
}

std::size_t argmax(const std::vector<float>& values) {
    return static_cast<std::size_t>(std::max_element(values.begin(), values.end()) -
                                    values.begin());
}

// Same acceptance rule as validateOptimization: every output within tolerance
// and the same predicted class.
bool mismatch(const std::optional<std::vector<float>>& expected,
              const std::optional<std::vector<float>>& actual, float tolerance) {
    if (!expected || !actual || expected->empty() || actual->empty()) return true;
    if (!compareOutputs(*expected, *actual, tolerance).passed) return true;
    return argmax(*expected) != argmax(*actual);
}

std::vector<std::optional<std::vector<float>>> outputsFor(
    const IRGraph& graph, const std::vector<ValidationProbe>& probes) {
    std::vector<std::optional<std::vector<float>>> outputs;
    outputs.reserve(probes.size());
    for (const auto& probe : probes) {
        DiagnosticEngine diagnostics;
        outputs.push_back(executeIR(graph, probe.values, diagnostics));
    }
    return outputs;
}

// 1-based index of the first probe exposing the fault, or 0.
std::size_t firstDetection(const IRGraph& candidate, const std::vector<ValidationProbe>& probes,
                           const std::vector<std::optional<std::vector<float>>>& expected,
                           float tolerance) {
    for (std::size_t index = 0; index < probes.size(); ++index) {
        DiagnosticEngine diagnostics;
        if (mismatch(expected[index], executeIR(candidate, probes[index].values, diagnostics),
                     tolerance)) {
            return index + 1;
        }
    }
    return 0;
}

IRInstruction* findInstruction(IRGraph& graph, const std::string& nodeName) {
    for (auto& instruction : graph.instructions) {
        if (instruction.nodeName == nodeName) return &instruction;
    }
    return nullptr;
}

std::string layerTag(int layer) { return "@L" + std::to_string(layer); }

std::vector<ValidationProbe> padded(std::vector<ValidationProbe> probes, std::size_t budget,
                                    std::size_t inputCount, std::uint32_t seed) {
    if (probes.size() > budget) probes.resize(budget);
    if (probes.size() < budget) {
        auto fill = uniformRandomProbes(inputCount, budget - probes.size(), seed);
        for (auto& probe : fill) probe.name = "pad_" + probe.name;
        probes.insert(probes.end(), std::make_move_iterator(fill.begin()),
                      std::make_move_iterator(fill.end()));
    }
    return probes;
}

bool isBoundaryProbe(const ValidationProbe& probe) {
    return probe.name.rfind("affine_relu_", 0) == 0 ||
           probe.name.rfind("decision_boundary_", 0) == 0;
}

}  // namespace

std::string jsonEscapeText(const std::string& value) {
    std::string escaped;
    for (const char character : value) {
        switch (character) {
            case '\\': escaped += "\\\\"; break;
            case '"': escaped += "\\\""; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped += character; break;
        }
    }
    return escaped;
}

// ------------------------------------------------------------ statistics

double Proportion::rate() const {
    return trials == 0 ? 0.0 : static_cast<double>(successes) / static_cast<double>(trials);
}

double Proportion::wilsonLow() const {
    if (trials == 0) return 0.0;
    const double z = 1.959963984540054;
    const double n = static_cast<double>(trials);
    const double p = rate();
    const double centre = p + z * z / (2 * n);
    const double spread = z * std::sqrt(p * (1 - p) / n + z * z / (4 * n * n));
    return std::max(0.0, (centre - spread) / (1 + z * z / n));
}

double Proportion::wilsonHigh() const {
    if (trials == 0) return 0.0;
    const double z = 1.959963984540054;
    const double n = static_cast<double>(trials);
    const double p = rate();
    const double centre = p + z * z / (2 * n);
    const double spread = z * std::sqrt(p * (1 - p) / n + z * z / (4 * n * n));
    return std::min(1.0, (centre + spread) / (1 + z * z / n));
}

double mcnemarExactP(std::size_t onlyFirst, std::size_t onlySecond) {
    const std::size_t n = onlyFirst + onlySecond;
    if (n == 0) return 1.0;
    const std::size_t k = std::min(onlyFirst, onlySecond);
    // Two-sided exact binomial test with p = 0.5, accumulated in log space.
    double tail = 0.0;
    for (std::size_t i = 0; i <= k; ++i) {
        const double logTerm = std::lgamma(static_cast<double>(n) + 1) -
                               std::lgamma(static_cast<double>(i) + 1) -
                               std::lgamma(static_cast<double>(n - i) + 1) -
                               static_cast<double>(n) * std::log(2.0);
        tail += std::exp(logTerm);
    }
    return std::min(1.0, 2.0 * tail);
}

// ------------------------------------------------------------ model zoo

std::vector<ZooModelSpec> makeZooSpecs(std::size_t count, std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> inputs(4, 16);
    std::uniform_int_distribution<int> classes(2, 5);
    std::uniform_int_distribution<int> width(4, 16);
    std::vector<ZooModelSpec> specs;
    for (std::size_t index = 0; index < count; ++index) {
        ZooModelSpec spec;
        std::ostringstream name;
        name << "zoo_" << std::setw(3) << std::setfill('0') << index;
        spec.name = name.str();
        spec.seed = seed + static_cast<std::uint32_t>(index) * 7919u;
        spec.inputs = static_cast<std::size_t>(inputs(generator));
        spec.classes = static_cast<std::size_t>(classes(generator));
        // Depths cycle 1..4 so every depth is equally represented.
        const std::size_t depth = 1 + index % 4;
        for (std::size_t layer = 0; layer < depth; ++layer) {
            spec.hidden.push_back(static_cast<std::size_t>(width(generator)));
        }
        specs.push_back(std::move(spec));
    }
    return specs;
}

std::string synthesizeManifest(const ZooModelSpec& spec) {
    std::mt19937 generator(spec.seed);
    std::normal_distribution<float> unit(0.0f, 1.0f);
    std::ostringstream text;
    text << std::setprecision(9);
    text << "# Synthetic ModelForge zoo model (seed " << spec.seed << ").\n"
         << "model " << spec.name << "\n"
         << "input x float32 1," << spec.inputs << "\n"
         << "output y float32 1," << spec.classes << "\n";
    std::vector<std::size_t> widths = {spec.inputs};
    widths.insert(widths.end(), spec.hidden.begin(), spec.hidden.end());
    widths.push_back(spec.classes);
    for (std::size_t layer = 0; layer + 1 < widths.size(); ++layer) {
        const std::size_t fanIn = widths[layer];
        const std::size_t fanOut = widths[layer + 1];
        const float scale = std::sqrt(2.0f / static_cast<float>(fanIn));
        text << "tensor w" << layer << " float32 " << fanIn << "," << fanOut << " values=";
        for (std::size_t index = 0; index < fanIn * fanOut; ++index) {
            text << (index ? "," : "") << unit(generator) * scale;
        }
        text << "\ntensor b" << layer << " float32 1," << fanOut << " values=";
        for (std::size_t index = 0; index < fanOut; ++index) {
            text << (index ? "," : "") << unit(generator) * 0.5f;
        }
        text << "\n";
    }
    std::string previous = "x";
    for (std::size_t layer = 0; layer < spec.hidden.size(); ++layer) {
        text << "node dense_" << layer << " Gemm " << previous << " w" << layer << " b" << layer
             << " -> z" << layer << "\n"
             << "node relu_" << layer << " Relu z" << layer << " -> h" << layer << "\n";
        previous = "h" + std::to_string(layer);
    }
    const std::size_t last = spec.hidden.size();
    text << "node classifier Gemm " << previous << " w" << last << " b" << last << " -> logits\n"
         << "node probabilities Softmax logits -> y axis=1\n"
         << "node unused_activation Relu " << previous << " -> unused\n";
    return text.str();
}

bool compileManifestText(const std::string& text, IRGraph& graph, DiagnosticEngine& diagnostics) {
    ModelGraph model;
    SymbolTable symbols;
    if (!loadManifestText(text, model, diagnostics) || !validate(model, diagnostics, symbols)) {
        return false;
    }
    auto ir = buildIR(model, diagnostics);
    if (!ir) return false;
    graph = std::move(*ir);
    return true;
}

// ------------------------------------------------------------ faults

std::vector<FaultCase> makeFaults(const IRGraph& original,
                                  const std::vector<ActivationSite>& sites,
                                  std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::vector<FaultCase> faults;
    const auto add = [&](FaultCase fault) { faults.push_back(std::move(fault)); };

    const auto activationFault = [&](const ActivationSite& site, const std::string& family,
                                     IROp operation, float parameter, bool heldOut,
                                     bool provablyUnobservable) {
        FaultCase fault;
        fault.family = family;
        fault.name = family + layerTag(static_cast<int>(site.layer));
        fault.locality = operation == IROp::Sigmoid ? "global" : "boundary_local";
        fault.layer = static_cast<int>(site.layer);
        fault.heldOut = heldOut;
        fault.provablyUnobservable = provablyUnobservable;
        fault.candidate = original;
        IRInstruction* relu = findInstruction(fault.candidate, site.nodeName);
        if (relu == nullptr) return;
        relu->operation = operation;
        relu->testFaultParameter = parameter;
        add(std::move(fault));
    };
    // IBP decides whether any unit's pre-activation interval meets the region
    // where the faulty activation differs from ReLU.
    const auto anyUnit = [](const ActivationSite& site, auto predicate) {
        for (std::size_t unit = 0; unit < site.width; ++unit) {
            if (predicate(site.lower[unit], site.upper[unit])) return true;
        }
        return false;
    };
    std::uniform_real_distribution<float> logWidth(std::log(1.0e-4f), std::log(1.0e-1f));
    std::uniform_real_distribution<float> clampLevel(1.0f, 6.0f);
    for (const ActivationSite& site : sites) {
        // Dead zones whose widths match the boundary-probe targets (design aligned).
        for (const auto& [tag, width] : std::vector<std::pair<std::string, float>>{
                 {"1e-1", 0.1f}, {"1e-2", 0.01f}, {"1e-3", 0.001f}}) {
            const float w = width;
            activationFault(site, "dead_zone_" + tag, IROp::TestReluDeadZone, w, false,
                            !anyUnit(site, [w](float lo, float hi) { return hi > 0 && lo < w; }));
        }
        // Held-out activation faults: widths/levels the probe design never saw.
        const float randomWidth = std::exp(logWidth(generator));
        activationFault(site, "dead_zone_random", IROp::TestReluDeadZone, randomWidth, true,
                        !anyUnit(site, [randomWidth](float lo, float hi) {
                            return hi > 0 && lo < randomWidth;
                        }));
        const float level = clampLevel(generator);
        activationFault(site, "relu_clamp", IROp::TestReluClamp, level, true,
                        !anyUnit(site, [level](float, float hi) { return hi > level; }));
        activationFault(site, "leaky_relu_1e-2", IROp::TestLeakyRelu, 0.01f, true,
                        !anyUnit(site, [](float lo, float) { return lo < 0; }));
        activationFault(site, "relu_to_sigmoid", IROp::Sigmoid, 0.0f, false, false);
    }

    // Small numeric corruptions of one weight and one bias entry in a random layer.
    std::vector<std::string> denseNodes;
    for (const auto& instruction : original.instructions) {
        if (instruction.operation == IROp::Gemm && instruction.inputs.size() == 3) {
            denseNodes.push_back(instruction.nodeName);
        }
    }
    if (!denseNodes.empty()) {
        std::uniform_int_distribution<std::size_t> pick(0, denseNodes.size() - 1);
        {
            FaultCase fault;
            fault.family = "weight_eps_1e-3";
            fault.locality = "numeric_local";
            fault.heldOut = true;
            fault.candidate = original;
            IRInstruction* dense = findInstruction(fault.candidate, denseNodes[pick(generator)]);
            auto& weights = fault.candidate.constants.at(dense->inputs[1]).data;
            std::uniform_int_distribution<std::size_t> element(0, weights.size() - 1);
            weights[element(generator)] += 1.0e-3f;
            fault.name = fault.family + "@" + dense->nodeName;
            add(std::move(fault));
        }
        {
            FaultCase fault;
            fault.family = "bias_eps_1e-2";
            fault.locality = "numeric_local";
            fault.heldOut = true;
            fault.candidate = original;
            IRInstruction* dense = findInstruction(fault.candidate, denseNodes[pick(generator)]);
            auto& bias = fault.candidate.constants.at(dense->inputs[2]).data;
            std::uniform_int_distribution<std::size_t> element(0, bias.size() - 1);
            bias[element(generator)] += 1.0e-2f;
            fault.name = fault.family + "@" + dense->nodeName;
            add(std::move(fault));
        }
        {
            FaultCase fault;
            fault.family = "missing_bias";
            fault.locality = "global";
            fault.candidate = original;
            IRInstruction* dense = findInstruction(fault.candidate, denseNodes.front());
            dense->inputs.pop_back();
            fault.name = fault.family + "@" + dense->nodeName;
            add(std::move(fault));
        }
    }
    for (auto& instruction : original.instructions) {
        if (instruction.operation != IROp::Softmax) continue;
        FaultCase fault;
        fault.family = "softmax_to_sigmoid";
        fault.name = fault.family;
        fault.locality = "global";
        fault.candidate = original;
        findInstruction(fault.candidate, instruction.nodeName)->operation = IROp::Sigmoid;
        add(std::move(fault));
        break;
    }
    // Negative control: rewriting an output-unreachable node must never be flagged.
    {
        FaultCase fault;
        fault.family = "unreachable_change";
        fault.name = fault.family;
        fault.locality = "control";
        fault.negativeControl = true;
        fault.candidate = original;
        if (IRInstruction* unused = findInstruction(fault.candidate, "unused_activation")) {
            unused->operation = IROp::Sigmoid;
            add(std::move(fault));
        }
    }
    return faults;
}

// ------------------------------------------------------------ strategies

std::vector<ValidationProbe> uniformRandomProbes(std::size_t inputCount, std::size_t count,
                                                 std::uint32_t seed, float radius) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-radius, radius);
    std::vector<ValidationProbe> probes;
    probes.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        std::vector<float> values(inputCount);
        for (float& value : values) value = distribution(generator);
        probes.push_back({"uniform_" + std::to_string(index), std::move(values)});
    }
    return probes;
}

std::vector<StrategySuite> buildStrategies(const IRGraph& original, std::size_t budget,
                                           std::uint32_t seed, std::size_t randomSeeds) {
    using Clock = std::chrono::steady_clock;
    const std::size_t inputCount =
        elementCount(original.values.at(original.inputName).shape);
    const std::uint32_t padSeed = seed ^ 0x5bd1e995u;
    std::vector<StrategySuite> suites;
    const auto finish = [&](StrategySuite suite, Clock::time_point start) {
        suite.nativeProbes = std::min(suite.probes.size(), budget);
        suite.probes = padded(std::move(suite.probes), budget, inputCount, padSeed);
        suite.generationMilliseconds =
            std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        suites.push_back(std::move(suite));
    };

    for (std::size_t r = 0; r < randomSeeds; ++r) {
        const auto start = Clock::now();
        StrategySuite suite;
        suite.name = "random#" + std::to_string(r);
        suite.description = "Uniform random inputs on [-10, 10]^n";
        suite.probes = uniformRandomProbes(inputCount, budget, seed + 1000003u * (r + 1));
        finish(std::move(suite), start);
    }

    auto start = Clock::now();
    auto legacy = generateValidationProbes(original, seed);
    const double legacyMs = std::chrono::duration<double, std::milli>(Clock::now() - start).count();

    {
        start = Clock::now();
        StrategySuite suite;
        suite.name = "generic";
        suite.description = "Operator-aware generic probes (zero, ramps, extremes, coordinates)";
        for (const auto& probe : legacy) {
            if (!isBoundaryProbe(probe)) suite.probes.push_back(probe);
        }
        finish(std::move(suite), start);
    }
    {
        start = Clock::now() - std::chrono::duration_cast<Clock::duration>(
                                   std::chrono::duration<double, std::milli>(legacyMs));
        StrategySuite suite;
        suite.name = "guardian_v1";
        suite.description = "Previous Guardian: generic + first-layer affine + decision bisection";
        suite.probes = legacy;
        finish(std::move(suite), start);
    }

    start = Clock::now();
    const auto sites = analyzeActivationSites(original);
    DeepBoundaryOptions options;
    options.seed = seed;
    const auto deep = generateDeepBoundaryProbes(original, sites, options);
    const double deepMs = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    const auto shifted = [](double ms) {
        return Clock::now() - std::chrono::duration_cast<Clock::duration>(
                                  std::chrono::duration<double, std::milli>(ms));
    };
    {
        start = shifted(deepMs);
        StrategySuite suite;
        suite.name = "deep_boundary";
        suite.description = "Ablation: IBP + Newton boundary probes only";
        suite.probes = deep;
        finish(std::move(suite), start);
    }
    std::vector<ValidationProbe> unionSuite = legacy;
    unionSuite.insert(unionSuite.end(), deep.begin(), deep.end());
    {
        start = shifted(legacyMs + deepMs);
        StrategySuite suite;
        suite.name = "union_unordered";
        suite.description = "Ablation: v1 + deep probes, no coverage ordering";
        suite.probes = unionSuite;
        finish(std::move(suite), start);
    }
    {
        start = shifted(legacyMs);
        StrategySuite suite;
        suite.name = "apc_without_deep";
        suite.description = "Ablation: v1 probes ordered by activation-pattern coverage";
        suite.probes = prioritizeByCoverage(original, sites, legacy);
        finish(std::move(suite), start);
    }
    {
        start = shifted(legacyMs + deepMs);
        StrategySuite suite;
        suite.name = "guardian_apc";
        suite.description = "Guardian-APC (ours): v1 + deep boundary probes, coverage-ordered";
        suite.probes = prioritizeByCoverage(original, sites, unionSuite);
        finish(std::move(suite), start);
    }
    return suites;
}

// ------------------------------------------------------------ benchmark

namespace {

struct StrategyStats {
    std::string name;
    std::string description;
    Proportion overall;
    std::vector<Proportion> curve;
    std::map<std::string, Proportion> families;
    std::map<std::string, Proportion> localities;
    std::map<int, Proportion> layers;
    std::map<std::string, Proportion> subsets;
    std::vector<double> firsts;      // detected cases only
    double censoredFirstSum = 0.0;   // misses counted as budget + 1
    std::size_t falsePositives = 0;
    std::size_t controls = 0;
    double apcSum = 0.0;
    double boundaryApcSum = 0.0;
    double generationMsSum = 0.0;
    double nativeProbeSum = 0.0;
    std::size_t modelCount = 0;
    bool perFault = false;
    std::vector<int> outcomes;       // per (case, seed) detection at budget, for McNemar
    std::vector<std::size_t> firstIndex;  // per (case, seed) first detection (0 = none)
};

std::string baseName(const std::string& suiteName) {
    const auto hash = suiteName.find('#');
    return hash == std::string::npos ? suiteName : suiteName.substr(0, hash);
}

std::string proportionJson(const Proportion& p) {
    return "{\"detected\":" + std::to_string(p.successes) + ",\"trials\":" +
           std::to_string(p.trials) + ",\"rate\":" + number(p.rate()) + ",\"ci\":[" +
           number(p.wilsonLow()) + "," + number(p.wilsonHigh()) + "]}";
}

std::string instructionsJson(const IRGraph& graph) {
    std::string result = "[";
    for (std::size_t index = 0; index < graph.instructions.size(); ++index) {
        const auto& instruction = graph.instructions[index];
        if (index) result += ",";
        result += "{\"index\":" + std::to_string(index) + ",\"op\":" +
                  quoted(irOpName(instruction.operation)) + ",\"node\":" +
                  quoted(instruction.nodeName) + ",\"inputs\":" +
                  array(instruction.inputs, quoted) + ",\"output\":" + quoted(instruction.output) +
                  ",\"shape\":" +
                  array(instruction.shape, [](std::int64_t d) { return std::to_string(d); }) + "}";
    }
    return result + "]";
}

std::string sitesJson(const std::vector<ActivationSite>& sites) {
    return array(sites, [](const ActivationSite& site) {
        return "{\"node\":" + quoted(site.nodeName) + ",\"layer\":" + std::to_string(site.layer) +
               ",\"width\":" + std::to_string(site.width) + ",\"lower\":" + floats(site.lower) +
               ",\"upper\":" + floats(site.upper) + ",\"stable_active\":" +
               std::to_string(site.count(NeuronStability::StableActive)) +
               ",\"stable_inactive\":" +
               std::to_string(site.count(NeuronStability::StableInactive)) + ",\"unstable\":" +
               std::to_string(site.count(NeuronStability::Unstable)) + "}";
    });
}

std::string coverageJson(const CoverageReport& coverage) {
    return "{\"apc\":" + number(coverage.ratio()) + ",\"boundary_apc\":" +
           number(coverage.boundaryRatio()) + ",\"covered\":" + std::to_string(coverage.covered) +
           ",\"feasible\":" + std::to_string(coverage.feasible) + ",\"layers\":" +
           array(coverage.layers, [](const LayerCoverage& layer) {
               return "{\"node\":" + quoted(layer.nodeName) + ",\"width\":" +
                      std::to_string(layer.width) + ",\"covered\":" +
                      std::to_string(layer.covered) + ",\"feasible\":" +
                      std::to_string(layer.feasible) + ",\"boundary_covered\":" +
                      std::to_string(layer.boundaryCovered) + ",\"boundary_feasible\":" +
                      std::to_string(layer.boundaryFeasible) + "}";
           }) +
           "}";
}

}  // namespace

BenchmarkSummary runBenchmark(const BenchmarkConfig& config,
                              const ProgressCallback& progress) {
    BenchmarkSummary summary;
    const auto wallStart = std::chrono::steady_clock::now();
    const auto specs = makeZooSpecs(config.models, config.seed);
    std::vector<std::size_t> curve = config.budgetCurve;
    curve.erase(std::remove_if(curve.begin(), curve.end(),
                               [&](std::size_t b) { return b == 0 || b > config.budget; }),
                curve.end());
    if (curve.empty() || curve.back() != config.budget) curve.push_back(config.budget);

    std::vector<StrategyStats> stats;
    const auto statsFor = [&](const StrategySuite& suite) -> StrategyStats& {
        const std::string name = baseName(suite.name);
        for (auto& entry : stats) {
            if (entry.name == name) return entry;
        }
        StrategyStats entry;
        entry.name = name;
        entry.description = suite.description;
        entry.perFault = suite.perFault;
        entry.curve.resize(curve.size());
        stats.push_back(std::move(entry));
        return stats.back();
    };

    std::ostringstream csv;
    csv << "model,depth,fault,family,locality,held_out,layer,observable,provably_unobservable,strategy,"
           "seed,first_detection,detected_at_budget\n";
    std::size_t faultsTotal = 0, observable = 0, unknown = 0, provable = 0, controls = 0,
                ibpViolations = 0;
    std::size_t stableActive = 0, stableInactive = 0, unstable = 0;
    std::map<std::size_t, std::size_t> depthHistogram;
    std::string modelsJson = "[";
    std::string missesJson = "[";
    std::size_t missCount = 0;

    for (std::size_t m = 0; m < specs.size(); ++m) {
        const ZooModelSpec& spec = specs[m];
        if (progress) progress(m, specs.size(), spec.name);
        IRGraph original;
        DiagnosticEngine diagnostics;
        if (!compileManifestText(synthesizeManifest(spec), original, diagnostics)) continue;
        const auto sites = analyzeActivationSites(original);
        auto faults = makeFaults(original, sites, spec.seed ^ 0x9e3779b9u);
        auto suites = buildStrategies(original, config.budget, spec.seed, config.randomSeeds);
        {
            // Guardian-APC+ variants, evaluated per fault at the same total budget.
            const std::size_t reserve = config.budget / 4;
            const auto perFault = [&](const std::string& name, const std::string& description,
                                      bool rewriteAware, std::size_t adaptive) {
                StrategySuite suite;
                suite.name = name;
                suite.description = description;
                suite.perFault = true;
                suite.rewriteAware = rewriteAware;
                suite.adaptiveBudget = adaptive;
                suites.push_back(std::move(suite));
            };
            perFault("guardian_delta", "Ablation: rewrite-aware static suite only", true, 0);
            perFault("apc_near_miss", "Ablation: Guardian-APC static + near-miss search", false,
                     reserve);
            perFault("guardian_apc_plus",
                     "Guardian-APC+ (ours): rewrite-aware probes + near-miss search", true, reserve);
        }
        const std::size_t inputCount = spec.inputs;
        const auto oracle = uniformRandomProbes(inputCount, config.oracleProbes,
                                                spec.seed ^ 0x85ebca6bu);
        const auto oracleExpected = outputsFor(original, oracle);
        std::vector<std::vector<std::optional<std::vector<float>>>> expected;
        for (const auto& suite : suites) expected.push_back(outputsFor(original, suite.probes));

        depthHistogram[spec.hidden.size()]++;
        std::size_t modelUnstable = 0, modelStable = 0;
        for (const auto& site : sites) {
            stableActive += site.count(NeuronStability::StableActive);
            stableInactive += site.count(NeuronStability::StableInactive);
            unstable += site.count(NeuronStability::Unstable);
            modelUnstable += site.count(NeuronStability::Unstable);
            modelStable += site.width - site.count(NeuronStability::Unstable);
        }
        std::vector<double> perFaultMs(suites.size(), 0.0);
        for (std::size_t s = 0; s < suites.size(); ++s) {
            StrategyStats& entry = statsFor(suites[s]);
            if (suites[s].perFault) continue;
            if (suites[s].name.find('#') != std::string::npos &&
                suites[s].name != baseName(suites[s].name) + "#0") {
                entry.generationMsSum += suites[s].generationMilliseconds / config.randomSeeds;
                continue;
            }
            const auto coverage = measureActivationCoverage(original, sites, suites[s].probes);
            entry.apcSum += coverage.ratio();
            entry.boundaryApcSum += coverage.boundaryRatio();
            entry.generationMsSum += suites[s].name.find('#') != std::string::npos
                                         ? suites[s].generationMilliseconds / config.randomSeeds
                                         : suites[s].generationMilliseconds;
            entry.nativeProbeSum += static_cast<double>(suites[s].nativeProbes);
            entry.modelCount += 1;
        }

        std::size_t modelObservable = 0;
        for (const FaultCase& fault : faults) {
            ++faultsTotal;
            std::vector<std::size_t> firsts(suites.size(), 0);
            bool anyDetected = false;
            for (std::size_t s = 0; s < suites.size(); ++s) {
                if (suites[s].perFault) {
                    GuardianPlusOptions options;
                    options.rewriteAware = suites[s].rewriteAware;
                    options.adaptiveBudget = suites[s].adaptiveBudget;
                    options.staticBudget = config.budget - suites[s].adaptiveBudget;
                    options.seed = spec.seed;
                    options.tolerance = config.tolerance;
                    const auto start = std::chrono::steady_clock::now();
                    firsts[s] = runGuardianPlus(original, fault.candidate, options).firstDetection;
                    perFaultMs[s] += std::chrono::duration<double, std::milli>(
                                         std::chrono::steady_clock::now() - start)
                                         .count();
                } else {
                    firsts[s] = firstDetection(fault.candidate, suites[s].probes, expected[s],
                                               config.tolerance);
                }
                anyDetected = anyDetected || firsts[s] != 0;
            }
            const bool oracleDetected =
                anyDetected ||
                firstDetection(fault.candidate, oracle, oracleExpected, config.tolerance) != 0;
            if (fault.provablyUnobservable) ++provable;
            if (fault.provablyUnobservable && oracleDetected) ++ibpViolations;

            for (std::size_t s = 0; s < suites.size(); ++s) {
                csv << spec.name << ',' << spec.hidden.size() << ',' << fault.name << ','
                    << fault.family << ',' << fault.locality << ',' << fault.heldOut << ','
                    << fault.layer << ','
                    << oracleDetected << ',' << fault.provablyUnobservable << ','
                    << baseName(suites[s].name) << ','
                    << (suites[s].name.find('#') == std::string::npos
                            ? std::string("0")
                            : suites[s].name.substr(suites[s].name.find('#') + 1))
                    << ',' << firsts[s] << ',' << (firsts[s] != 0) << '\n';
            }

            if (fault.negativeControl) {
                ++controls;
                for (std::size_t s = 0; s < suites.size(); ++s) {
                    StrategyStats& entry = statsFor(suites[s]);
                    entry.controls += 1;
                    entry.falsePositives += firsts[s] != 0;
                }
                continue;
            }
            if (!oracleDetected) {
                ++unknown;
                continue;
            }
            ++observable;
            ++modelObservable;
            for (std::size_t s = 0; s < suites.size(); ++s) {
                StrategyStats& entry = statsFor(suites[s]);
                const bool hit = firsts[s] != 0;
                entry.overall.trials++;
                entry.overall.successes += hit;
                entry.families[fault.family].trials++;
                entry.families[fault.family].successes += hit;
                entry.localities[fault.locality].trials++;
                entry.localities[fault.locality].successes += hit;
                if (fault.layer >= 0) {
                    entry.layers[fault.layer].trials++;
                    entry.layers[fault.layer].successes += hit;
                    auto& depthSubset = entry.subsets[fault.layer == 0 ? "first_layer" : "deep_layers"];
                    depthSubset.trials++;
                    depthSubset.successes += hit;
                }
                auto& designSubset = entry.subsets[fault.heldOut ? "held_out" : "design_aligned"];
                designSubset.trials++;
                designSubset.successes += hit;
                for (std::size_t b = 0; b < curve.size(); ++b) {
                    entry.curve[b].trials++;
                    entry.curve[b].successes += hit && firsts[s] <= curve[b];
                }
                if (hit) entry.firsts.push_back(static_cast<double>(firsts[s]));
                entry.censoredFirstSum +=
                    hit ? static_cast<double>(firsts[s]) : static_cast<double>(config.budget + 1);
                entry.outcomes.push_back(hit ? 1 : 0);
                entry.firstIndex.push_back(firsts[s]);
                if (suites[s].name == "guardian_apc_plus" && !hit && missCount < 60) {
                    missesJson += (missCount++ ? "," : "") + std::string("{\"model\":") +
                                  quoted(spec.name) + ",\"fault\":" + quoted(fault.name) + "}";
                }
            }
        }
        for (std::size_t s = 0; s < suites.size(); ++s) {
            if (!suites[s].perFault || faults.empty()) continue;
            StrategyStats& entry = statsFor(suites[s]);
            entry.generationMsSum += perFaultMs[s] / static_cast<double>(faults.size());
            entry.modelCount += 1;
        }
        if (m) modelsJson += ",";
        modelsJson += "{\"name\":" + quoted(spec.name) + ",\"inputs\":" +
                      std::to_string(spec.inputs) + ",\"classes\":" +
                      std::to_string(spec.classes) + ",\"hidden\":" +
                      array(spec.hidden, [](std::size_t w) { return std::to_string(w); }) +
                      ",\"unstable\":" + std::to_string(modelUnstable) + ",\"stable\":" +
                      std::to_string(modelStable) + ",\"faults\":" +
                      std::to_string(faults.size()) + ",\"observable\":" +
                      std::to_string(modelObservable) + "}";
    }
    modelsJson += "]";
    missesJson += "]";
    if (progress) progress(specs.size(), specs.size(), "done");
    summary.casesCsv = csv.str();

    // Paired comparisons: Guardian-APC against every other strategy.
    // Random has `randomSeeds` outcomes per case; Guardian's outcome is repeated.
    const StrategyStats* ours = nullptr;
    for (const auto& entry : stats) {
        if (entry.name == "guardian_apc_plus") ours = &entry;
    }
    std::string comparisons = "[";
    bool firstComparison = true;
    for (const auto& entry : stats) {
        if (ours == nullptr || &entry == ours) continue;
        const std::size_t repeat = entry.outcomes.size() / std::max<std::size_t>(1, ours->outcomes.size());
        std::size_t onlyOurs = 0, onlyOther = 0, both = 0, neither = 0;
        // Outcomes were appended per case in suite order, so for random the
        // `repeat` seeds of one case are adjacent.
        for (std::size_t c = 0; c < ours->outcomes.size(); ++c) {
            for (std::size_t r = 0; r < repeat; ++r) {
                const int a = ours->outcomes[c];
                const int b = entry.outcomes[c * repeat + r];
                onlyOurs += a && !b;
                onlyOther += !a && b;
                both += a && b;
                neither += !a && !b;
            }
        }
        summary.comparisons.push_back({entry.name, onlyOurs, onlyOther,
                                       mcnemarExactP(onlyOurs, onlyOther)});
        comparisons += std::string(firstComparison ? "" : ",") + "{\"a\":\"guardian_apc_plus\",\"b\":" +
                       quoted(entry.name) + ",\"only_a\":" + std::to_string(onlyOurs) +
                       ",\"only_b\":" + std::to_string(onlyOther) + ",\"both\":" +
                       std::to_string(both) + ",\"neither\":" + std::to_string(neither) +
                       ",\"p_value\":" + number(mcnemarExactP(onlyOurs, onlyOther)) + "}";
        firstComparison = false;
    }
    comparisons += "]";

    // Paired tests at every budget on the curve: does Guardian-APC+ find faults
    // with fewer test inputs than Guardian-APC and than random testing?
    std::string budgetTests = "[";
    bool firstBudgetTest = true;
    for (const auto& entry : stats) {
        if (ours == nullptr || (entry.name != "guardian_apc" && entry.name != "random")) continue;
        const std::size_t repeat =
            entry.firstIndex.size() / std::max<std::size_t>(1, ours->firstIndex.size());
        for (const std::size_t budget : curve) {
            std::size_t onlyOurs = 0, onlyOther = 0;
            for (std::size_t c = 0; c < ours->firstIndex.size(); ++c) {
                for (std::size_t r = 0; r < repeat; ++r) {
                    const std::size_t a = ours->firstIndex[c];
                    const std::size_t b = entry.firstIndex[c * repeat + r];
                    const bool x = a != 0 && a <= budget;
                    const bool y = b != 0 && b <= budget;
                    onlyOurs += x && !y;
                    onlyOther += y && !x;
                }
            }
            budgetTests += std::string(firstBudgetTest ? "" : ",") + "{\"b\":" +
                           quoted(entry.name) + ",\"budget\":" + std::to_string(budget) +
                           ",\"only_a\":" + std::to_string(onlyOurs) + ",\"only_b\":" +
                           std::to_string(onlyOther) + ",\"p_value\":" +
                           number(mcnemarExactP(onlyOurs, onlyOther)) + "}";
            firstBudgetTest = false;
        }
    }
    budgetTests += "]";

    std::string strategiesJson = "[";
    for (std::size_t index = 0; index < stats.size(); ++index) {
        const auto& entry = stats[index];
        auto sorted = entry.firsts;
        std::sort(sorted.begin(), sorted.end());
        const double meanFirst =
            sorted.empty() ? NAN
                           : std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
        const double medianFirst = sorted.empty() ? NAN : sorted[sorted.size() / 2];
        double auc = 0.0;
        for (const auto& point : entry.curve) auc += point.rate();
        auc /= static_cast<double>(std::max<std::size_t>(1, entry.curve.size()));
        const double models = static_cast<double>(std::max<std::size_t>(1, entry.modelCount));
        std::string curveJson = "[";
        for (std::size_t b = 0; b < curve.size(); ++b) {
            curveJson += (b ? "," : "") + std::string("{\"budget\":") + std::to_string(curve[b]) +
                         ",\"rate\":" + number(entry.curve[b].rate()) + ",\"ci\":[" +
                         number(entry.curve[b].wilsonLow()) + "," +
                         number(entry.curve[b].wilsonHigh()) + "]}";
        }
        curveJson += "]";
        const auto mapJson = [](const auto& values) {
            std::string result = "{";
            bool first = true;
            for (const auto& [key, value] : values) {
                std::ostringstream name;
                name << key;
                result += (first ? "" : ",") + quoted(name.str()) + ":" + proportionJson(value);
                first = false;
            }
            return result + "}";
        };
        StrategySummary brief;
        brief.name = entry.name;
        brief.description = entry.description;
        brief.detected = entry.overall.successes;
        brief.trials = entry.overall.trials;
        brief.rate = entry.overall.rate();
        brief.low = entry.overall.wilsonLow();
        brief.high = entry.overall.wilsonHigh();
        brief.meanFirst = meanFirst;
        brief.censoredFirst =
            entry.censoredFirstSum / std::max<double>(1.0, entry.outcomes.size());
        brief.auc = auc;
        brief.apc = entry.apcSum / models;
        brief.boundaryApc = entry.boundaryApcSum / models;
        brief.generationMs = entry.generationMsSum / models;
        brief.falsePositives = entry.falsePositives;
        brief.controls = entry.controls;
        for (std::size_t b = 0; b < curve.size(); ++b) {
            brief.curve.emplace_back(curve[b], entry.curve[b].rate());
        }
        for (const auto& [key, value] : entry.families) brief.families[key] = value.rate();
        for (const auto& [key, value] : entry.localities) brief.localities[key] = value.rate();
        for (const auto& [key, value] : entry.layers) brief.layers[key] = value.rate();
        for (const auto& [key, value] : entry.subsets) brief.subsets[key] = value.rate();
        summary.strategies.push_back(std::move(brief));
        strategiesJson += std::string(index ? "," : "") + "{\"name\":" + quoted(entry.name) +
                          ",\"description\":" + quoted(entry.description) +
                          ",\"overall\":" + proportionJson(entry.overall) +
                          ",\"mean_first\":" + number(meanFirst) +
                          ",\"median_first\":" + number(medianFirst) +
                          ",\"mean_first_censored\":" +
                          number(entry.censoredFirstSum /
                                 std::max<double>(1.0, entry.outcomes.size())) +
                          ",\"auc\":" + number(auc) + ",\"curve\":" + curveJson +
                          ",\"families\":" + mapJson(entry.families) +
                          ",\"localities\":" + mapJson(entry.localities) +
                          ",\"layers\":" + mapJson(entry.layers) +
                          ",\"subsets\":" + mapJson(entry.subsets) +
                          ",\"false_positives\":" + std::to_string(entry.falsePositives) +
                          ",\"controls\":" + std::to_string(entry.controls) +
                          ",\"per_fault\":" + (entry.perFault ? "true" : "false") +
                          ",\"apc\":" + (entry.perFault ? "null" : number(entry.apcSum / models)) +
                          ",\"boundary_apc\":" +
                          (entry.perFault ? "null" : number(entry.boundaryApcSum / models)) +
                          ",\"generation_ms\":" + number(entry.generationMsSum / models) +
                          ",\"native_probes\":" + number(entry.nativeProbeSum / models) + "}";
    }
    strategiesJson += "]";

    std::string depthJson = "{";
    bool firstDepth = true;
    for (const auto& [depth, count] : depthHistogram) {
        depthJson += std::string(firstDepth ? "" : ",") + "\"" + std::to_string(depth) +
                     "\":" + std::to_string(count);
        firstDepth = false;
    }
    depthJson += "}";

    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - wallStart).count();
    std::ostringstream json;
    json << "{\"tool\":\"modelforge_bench\",\"author\":" << quoted(kAuthor) << ",\"generated_at\":" << quoted(utcTimestamp())
         << ",\"platform\":" << quoted(platformName()) << ",\"compiler\":"
         << quoted(compilerName()) << ",\"runtime_seconds\":" << number(seconds)
         << ",\"config\":{\"models\":" << config.models << ",\"budget\":" << config.budget
         << ",\"random_seeds\":" << config.randomSeeds << ",\"oracle_probes\":"
         << config.oracleProbes << ",\"seed\":" << config.seed << ",\"tolerance\":"
         << number(config.tolerance) << ",\"input_domain\":[" << -kDefaultInputRadius << ","
         << kDefaultInputRadius << "]}"
         << ",\"corpus\":{\"models\":" << specs.size() << ",\"faults_total\":" << faultsTotal
         << ",\"observable\":" << observable << ",\"unknown\":" << unknown
         << ",\"provably_unobservable\":" << provable << ",\"ibp_violations\":" << ibpViolations
         << ",\"controls\":" << controls << ",\"depths\":" << depthJson
         << ",\"neurons\":{\"stable_active\":" << stableActive << ",\"stable_inactive\":"
         << stableInactive << ",\"unstable\":" << unstable << "}}"
         << ",\"strategies\":" << strategiesJson << ",\"comparisons\":" << comparisons
         << ",\"budget_tests\":" << budgetTests
         << ",\"models\":" << modelsJson << ",\"misses\":" << missesJson << "}";
    summary.json = json.str();
    summary.models = specs.size();
    summary.faults = faultsTotal;
    summary.observable = observable;
    summary.unknown = unknown;
    summary.provable = provable;
    summary.ibpViolations = ibpViolations;
    summary.controls = controls;
    summary.stableActive = stableActive;
    summary.stableInactive = stableInactive;
    summary.unstable = unstable;
    summary.seconds = seconds;
    return summary;
}

std::string inspectModelJson(const std::string& manifestPath, const IRGraph& graph,
                             std::size_t budget, std::uint32_t seed) {
    const auto sites = analyzeActivationSites(graph);
    DiagnosticEngine diagnostics;
    IRGraph optimized = graph;
    const auto guardian = optimizeWithGuardian(optimized, diagnostics);

    // The same deliberately wrong ReLU -> Sigmoid rewrite as --guardian-demo-bug.
    std::string demo = "null";
    {
        IRGraph buggy = graph;
        for (auto& instruction : buggy.instructions) {
            if (instruction.operation == IROp::Relu) {
                instruction.operation = IROp::Sigmoid;
                DiagnosticEngine local;
                const auto decision =
                    checkCandidate(graph, buggy, "deliberately_wrong_relu_rewrite", local);
                demo = "{\"accepted\":" + std::string(decision.accepted ? "true" : "false") +
                       ",\"probes\":" + std::to_string(decision.validation.probeCount);
                if (decision.divergence.found) {
                    demo += ",\"divergent_node\":" + quoted(decision.divergence.node) +
                            ",\"divergent_value\":" + quoted(decision.divergence.value);
                }
                if (decision.validation.firstFailure) {
                    demo += ",\"probe\":" + quoted(decision.validation.firstFailure->name) +
                            ",\"input\":" + floats(decision.validation.firstFailure->values) +
                            ",\"original_output\":" + floats(decision.validation.originalOutput) +
                            ",\"candidate_output\":" + floats(decision.validation.candidateOutput);
                }
                demo += "}";
                break;
            }
        }
    }

    std::string decisions = array(guardian.decisions, [](const GuardianDecision& decision) {
        return "{\"pass\":" + quoted(decision.passName) + ",\"accepted\":" +
               (decision.accepted ? "true" : "false") + ",\"probes\":" +
               std::to_string(decision.validation.probeCount) + ",\"max_error\":" +
               number(decision.validation.maximumAbsoluteError) + ",\"events\":" +
               array(decision.optimization.events, quoted) + "}";
    });

    auto suites = buildStrategies(graph, budget, seed, 1);
    std::string coverage = "[";
    std::string faultTable = "[";
    const auto faults = makeFaults(graph, sites, seed);
    std::vector<std::vector<std::optional<std::vector<float>>>> expected;
    for (const auto& suite : suites) expected.push_back(outputsFor(graph, suite.probes));
    for (std::size_t s = 0; s < suites.size(); ++s) {
        coverage += std::string(s ? "," : "") + "{\"strategy\":" +
                    quoted(baseName(suites[s].name)) + ",\"coverage\":" +
                    coverageJson(measureActivationCoverage(graph, sites, suites[s].probes)) + "}";
    }
    coverage += "]";
    for (std::size_t f = 0; f < faults.size(); ++f) {
        faultTable += std::string(f ? "," : "") + "{\"fault\":" + quoted(faults[f].name) +
                      ",\"family\":" + quoted(faults[f].family) + ",\"locality\":" +
                      quoted(faults[f].locality) + ",\"held_out\":" +
                      (faults[f].heldOut ? "true" : "false") + ",\"provably_unobservable\":" +
                      (faults[f].provablyUnobservable ? "true" : "false") + ",\"first\":{";
        GuardianPlusOptions plus;
        plus.staticBudget = budget - budget / 4;
        plus.adaptiveBudget = budget / 4;
        plus.seed = seed;
        faultTable += "\"guardian_apc_plus\":" +
                      std::to_string(runGuardianPlus(graph, faults[f].candidate, plus).firstDetection);
        for (std::size_t s = 0; s < suites.size(); ++s) {
            faultTable += "," + quoted(baseName(suites[s].name)) + ":" +
                          std::to_string(firstDetection(faults[f].candidate, suites[s].probes,
                                                        expected[s], 1.0e-5f));
        }
        faultTable += "}}";
    }
    faultTable += "]";

    const auto ordered = generateGuardianProbes(graph, seed);
    std::string schedule = "[";
    for (std::size_t index = 0; index < std::min<std::size_t>(ordered.size(), 24); ++index) {
        schedule += std::string(index ? "," : "") + quoted(ordered[index].name);
    }
    schedule += "]";

    const auto& inputShape = graph.values.at(graph.inputName).shape;
    const auto& outputShape = graph.values.at(graph.outputName).shape;
    return "{\"model\":" + quoted(graph.name) + ",\"author\":" + quoted(kAuthor) + ",\"path\":" + quoted(manifestPath) +
           ",\"input_shape\":" +
           array(inputShape, [](std::int64_t d) { return std::to_string(d); }) +
           ",\"output_shape\":" +
           array(outputShape, [](std::int64_t d) { return std::to_string(d); }) +
           ",\"ir_before\":" + instructionsJson(graph) + ",\"ir_after\":" +
           instructionsJson(optimized) + ",\"guardian\":" + decisions + ",\"demo_bug\":" + demo +
           ",\"sites\":" + sitesJson(sites) + ",\"coverage\":" + coverage +
           ",\"faults\":" + faultTable + ",\"schedule\":" + schedule +
           ",\"guardian_probe_count\":" + std::to_string(ordered.size()) + "}";
}

}  // namespace modelforge

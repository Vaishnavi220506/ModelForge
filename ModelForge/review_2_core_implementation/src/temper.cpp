#include "temper.h"

#include "analysis.h"
#include "verifier.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <set>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace modelforge {
namespace {

double elapsedMs(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
}

bool insideBox(const std::vector<float>& x, const Box& box) {
    if (x.size() != box.size()) return false;
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (!(x[i] >= box.lower[i] && x[i] <= box.upper[i])) return false;
    }
    return true;
}

float boxRadius(const Box& box) {
    float radius = 0.0f;
    for (std::size_t i = 0; i < box.size(); ++i) {
        radius = std::max({radius, std::fabs(box.lower[i]), std::fabs(box.upper[i])});
    }
    return radius;
}

std::vector<float> run(const IRGraph& graph, const std::vector<float>& x) {
    DiagnosticEngine diagnostics;
    auto result = executeIR(graph, x, diagnostics);
    return result ? *result : std::vector<float>{};
}

void fillOutputs(const IRGraph& before, const IRGraph& after, TemperDecision& decision,
                 float tolerance) {
    if (!decision.witness) return;
    decision.originalOutput = run(before, *decision.witness);
    decision.candidateOutput = run(after, *decision.witness);
    decision.divergence = localizeDivergence(before, after, *decision.witness, tolerance);
}

}  // namespace

// ------------------------------------------------------------ transformations

TemperDecision decideTransformation(const IRGraph& before, const IRGraph& after, const Box& domain,
                                    const TemperOptions& options) {
    const auto start = std::chrono::steady_clock::now();
    TemperDecision decision;
    const auto certified = [&](const CertifyResult& result, const char* how) {
        decision.verdict = Verdict::Certified;
        decision.kind = result.kind;
        decision.bound = result.bound;
        decision.certifiedFraction = 1.0;
        decision.decidedBy = how;
    };
    // 1. Static proof on the whole domain: no model evaluation at all.
    if (options.useCertifier) {
        CertifyOptions quick;
        quick.tolerance = options.tolerance;
        quick.maxBoxes = 1;
        quick.centreChecks = false;
        quick.guidedWitnesses = false;
        const CertifyResult result = certifyEquivalence(before, after, domain, quick);
        decision.boxes += result.boxes;
        decision.sources = result.sources;
        if (result.verdict == Verdict::Certified) {
            certified(result, "static-proof");
            decision.milliseconds = elapsedMs(start);
            return decision;
        }
    }
    // 2. Guardian-APC+: rewrite-aware boundary inputs and near-miss search.
    if (options.useGuardian) {
        GuardianPlusOptions guardian;
        guardian.staticBudget = options.staticBudget;
        guardian.adaptiveBudget = options.adaptiveBudget;
        guardian.seed = options.seed;
        guardian.tolerance = options.tolerance;
        guardian.radius = boxRadius(domain);
        const GuardianPlusResult result = runGuardianPlus(before, after, guardian);
        decision.evaluations += result.evaluations;
        if (result.firstDetection != 0 && result.witness && insideBox(result.witness->values, domain)) {
            decision.verdict = Verdict::Refuted;
            decision.decidedBy = "guardian";
            decision.witness = result.witness->values;
            fillOutputs(before, after, decision, options.tolerance);
            decision.milliseconds = elapsedMs(start);
            return decision;
        }
    }
    // 3. Certificate-guided branch and bound.
    if (options.useCertifier) {
        CertifyOptions full;
        full.tolerance = options.tolerance;
        full.maxBoxes = options.maxBoxes;
        full.maxEvaluations = options.maxEvaluations;
        full.guidedWitnesses = options.guidedWitnesses;
        const CertifyResult result = certifyEquivalence(before, after, domain, full);
        decision.evaluations += result.evaluations;
        decision.boxes += result.boxes;
        decision.certifiedFraction = result.certifiedFraction;
        if (result.verdict == Verdict::Certified) {
            certified(result, "branch-and-bound");
        } else if (result.verdict == Verdict::Refuted) {
            decision.verdict = Verdict::Refuted;
            decision.decidedBy = result.refutedBy;
            decision.witness = result.witness;
            fillOutputs(before, after, decision, options.tolerance);
        }
    }
    decision.milliseconds = elapsedMs(start);
    return decision;
}

// ------------------------------------------------------------ domain specialisation

namespace {

struct DenseLayout {
    std::size_t inner = 0, columns = 0;
    bool transB = false;
};

bool layoutOf(const IRGraph& graph, const IRInstruction& instruction, DenseLayout& layout) {
    if (instruction.operation != IROp::Gemm && instruction.operation != IROp::MatMul &&
        instruction.operation != IROp::FusedGemmRelu) {
        return false;
    }
    if (instruction.inputs.size() < 2 || graph.constants.count(instruction.inputs[1]) == 0) {
        return false;
    }
    if (instruction.operation != IROp::MatMul && instruction.transA) return false;
    const auto left = graph.values.find(instruction.inputs[0]);
    const auto& weights = graph.constants.at(instruction.inputs[1]);
    if (left == graph.values.end() || left->second.shape.size() != 2 ||
        left->second.shape[0] != 1 || weights.shape.size() != 2) {
        return false;
    }
    layout.transB = instruction.operation != IROp::MatMul && instruction.transB;
    layout.inner = static_cast<std::size_t>(layout.transB ? weights.shape[1] : weights.shape[0]);
    layout.columns = static_cast<std::size_t>(layout.transB ? weights.shape[0] : weights.shape[1]);
    return true;
}

std::size_t uses(const IRGraph& graph, const std::string& name) {
    std::size_t count = 0;
    for (const auto& instruction : graph.instructions) {
        count += static_cast<std::size_t>(
            std::count(instruction.inputs.begin(), instruction.inputs.end(), name));
    }
    return count;
}

void setShape(IRGraph& graph, const std::string& name, const Shape& shape) {
    if (graph.values.count(name)) graph.values[name].shape = shape;
    if (graph.constants.count(name)) graph.constants[name].shape = shape;
}

// Keeps only the listed input rows (k) / output columns (c) of a weight matrix.
void keepWeights(IRGraph& graph, const std::string& name, const DenseLayout& layout,
                 const std::vector<std::size_t>& rows, const std::vector<std::size_t>& columns) {
    auto& data = graph.constants.at(name).data;
    std::vector<float> kept;
    kept.reserve(rows.size() * columns.size());
    if (!layout.transB) {
        for (const std::size_t k : rows) {
            for (const std::size_t c : columns) kept.push_back(data[k * layout.columns + c]);
        }
        setShape(graph, name, {static_cast<std::int64_t>(rows.size()),
                               static_cast<std::int64_t>(columns.size())});
    } else {
        for (const std::size_t c : columns) {
            for (const std::size_t k : rows) kept.push_back(data[c * layout.inner + k]);
        }
        setShape(graph, name, {static_cast<std::int64_t>(columns.size()),
                               static_cast<std::int64_t>(rows.size())});
    }
    data = std::move(kept);
    graph.values[name].data = data;
}

std::vector<std::size_t> allIndices(std::size_t count) {
    std::vector<std::size_t> result(count);
    for (std::size_t i = 0; i < count; ++i) result[i] = i;
    return result;
}

IRInstruction* producerOf(IRGraph& graph, const std::string& value) {
    for (auto& instruction : graph.instructions) {
        if (instruction.output == value) return &instruction;
    }
    return nullptr;
}

}  // namespace

std::size_t parameterCount(const IRGraph& graph) {
    std::set<std::string> used;
    for (const auto& instruction : graph.instructions) {
        for (const auto& input : instruction.inputs) {
            if (graph.constants.count(input)) used.insert(input);
        }
    }
    std::size_t total = 0;
    for (const auto& name : used) total += graph.constants.at(name).data.size();
    return total;
}

std::size_t multiplyAddCount(const IRGraph& graph) {
    std::size_t total = 0;
    for (const auto& instruction : graph.instructions) {
        DenseLayout layout;
        if (layoutOf(graph, instruction, layout)) total += layout.inner * layout.columns;
    }
    return total;
}

SpecializationReport specializeToDomain(const IRGraph& graph, const Box& domain) {
    SpecializationReport report;
    report.masked = graph;
    report.specialized = graph;
    report.parametersBefore = parameterCount(graph);
    report.multiplyAddsBefore = multiplyAddCount(graph);
    const GraphBounds bounds = computeBounds(graph, domain);
    const auto live = liveValues(graph);

    for (const IRInstruction& relu : graph.instructions) {
        if (relu.operation != IROp::Relu || relu.inputs.size() != 1 ||
            live.count(relu.output) == 0 || relu.output == graph.outputName) {
            continue;
        }
        const std::string& pre = relu.inputs[0];
        const ValueBounds* preBounds = bounds.find(pre);
        if (preBounds == nullptr || uses(graph, pre) != 1) continue;
        const IRInstruction* producer = nullptr;
        for (const auto& instruction : graph.instructions) {
            if (instruction.output == pre) producer = &instruction;
        }
        DenseLayout producerLayout;
        if (producer == nullptr || !layoutOf(graph, *producer, producerLayout)) continue;
        if (producer->operation == IROp::Gemm && producer->inputs.size() == 3 &&
            graph.constants.count(producer->inputs[2]) == 0) {
            continue;
        }
        // Every consumer must be a dense layer reading the activation as its
        // left operand with private constant weights.
        std::vector<const IRInstruction*> consumers;
        bool ok = true;
        for (const auto& instruction : graph.instructions) {
            const auto found = std::find(instruction.inputs.begin(), instruction.inputs.end(),
                                         relu.output);
            if (found == instruction.inputs.end()) continue;
            DenseLayout layout;
            if (found != instruction.inputs.begin() || !layoutOf(graph, instruction, layout) ||
                uses(graph, instruction.inputs[1]) != 1) {
                ok = false;
                break;
            }
            consumers.push_back(&instruction);
        }
        if (!ok || consumers.empty() || uses(graph, producer->inputs[1]) != 1) continue;

        const std::size_t width = preBounds->lower.size();
        report.hiddenUnits += width;
        std::vector<std::size_t> keep, dead;
        std::size_t active = 0;
        for (std::size_t unit = 0; unit < width; ++unit) {
            if (preBounds->upper[unit] <= 0.0) {
                dead.push_back(unit);
            } else {
                keep.push_back(unit);
                active += preBounds->lower[unit] > 0.0 ? 1 : 0;
            }
        }
        if (keep.empty()) {  // keep one unit so every tensor stays non-empty
            keep.push_back(dead.back());
            dead.pop_back();
        }
        report.alwaysActiveUnits += active;
        report.deadUnits += dead.size();

        // Masked model: same shapes, outgoing weights of dead units set to 0.
        for (const IRInstruction* consumer : consumers) {
            DenseLayout layout;
            layoutOf(report.masked, *consumer, layout);
            auto& data = report.masked.constants.at(consumer->inputs[1]).data;
            for (const std::size_t k : dead) {
                for (std::size_t c = 0; c < layout.columns; ++c) {
                    data[layout.transB ? c * layout.inner + k : k * layout.columns + c] = 0.0f;
                }
            }
            report.masked.values[consumer->inputs[1]].data = data;
        }

        // Specialised model: dead units removed from both weight matrices.
        if (!dead.empty()) {
            IRGraph& s = report.specialized;
            DenseLayout layout;
            IRInstruction* p = producerOf(s, pre);
            layoutOf(s, *p, layout);
            keepWeights(s, p->inputs[1], layout, allIndices(layout.inner), keep);
            if (p->operation == IROp::Gemm && p->inputs.size() == 3) {
                auto& bias = s.constants.at(p->inputs[2]);
                if (bias.data.size() == width) {
                    std::vector<float> kept;
                    for (const std::size_t unit : keep) kept.push_back(bias.data[unit]);
                    bias.data = kept;
                    Shape shape = bias.shape;
                    if (!shape.empty()) shape.back() = static_cast<std::int64_t>(keep.size());
                    setShape(s, p->inputs[2], shape);
                    s.values[p->inputs[2]].data = kept;
                }
            }
            const Shape narrow = {1, static_cast<std::int64_t>(keep.size())};
            setShape(s, pre, narrow);
            setShape(s, relu.output, narrow);
            for (const IRInstruction* consumer : consumers) {
                IRInstruction* c = producerOf(s, consumer->output);
                DenseLayout consumerLayout;
                layoutOf(report.masked, *consumer, consumerLayout);
                keepWeights(s, c->inputs[1], consumerLayout, keep,
                            allIndices(consumerLayout.columns));
            }
        }
        // Every remaining unit always active: the ReLU is the identity.
        if (active == keep.size()) {
            IRGraph& s = report.specialized;
            for (auto& instruction : s.instructions) {
                for (auto& input : instruction.inputs) {
                    if (input == relu.output) input = pre;
                }
            }
            s.instructions.erase(std::remove_if(s.instructions.begin(), s.instructions.end(),
                                                [&](const IRInstruction& instruction) {
                                                    return instruction.output == relu.output;
                                                }),
                                 s.instructions.end());
            ++report.linearizedLayers;
        }
    }
    report.parametersAfter = parameterCount(report.specialized);
    report.multiplyAddsAfter = multiplyAddCount(report.specialized);
    CertifyOptions options;
    options.maxBoxes = 1;
    options.centreChecks = false;
    options.guidedWitnesses = false;
    report.certificate = certifyEquivalence(graph, report.masked, domain, options);
    return report;
}

// ------------------------------------------------------------ compression audit

namespace {

// Margin of class c (logit c minus the best other logit) and its gradient.
bool marginGradient(const IRGraph& canonical, const std::string& logits, bool sigmoidOutput,
                    const std::vector<float>& x, std::size_t c, float& margin,
                    std::vector<float>& gradient) {
    if (sigmoidOutput) {
        float z = 0.0f;
        if (!valueGradient(canonical, x, logits, 0, z, gradient)) return false;
        const float sign = c == 1 ? 1.0f : -1.0f;
        margin = sign * z;
        for (float& g : gradient) g *= sign;
        return true;
    }
    const std::vector<float> scores = run(canonical, x);
    if (scores.size() < 2 || c >= scores.size()) return false;
    std::size_t other = c == 0 ? 1 : 0;
    for (std::size_t j = 0; j < scores.size(); ++j) {
        if (j != c && scores[j] > scores[other]) other = j;
    }
    float zc = 0.0f, zj = 0.0f;
    std::vector<float> gj;
    if (!valueGradient(canonical, x, logits, c, zc, gradient) ||
        !valueGradient(canonical, x, logits, other, zj, gj)) {
        return false;
    }
    margin = zc - zj;
    for (std::size_t i = 0; i < gradient.size(); ++i) gradient[i] -= gj[i];
    return true;
}

}  // namespace

double confidenceMargin(const std::vector<float>& scores) {
    if (scores.empty()) return 0.0;
    if (scores.size() == 1) return std::fabs(2.0 * scores[0] - 1.0);
    float first = -std::numeric_limits<float>::infinity(), second = first;
    for (const float value : scores) {
        if (value > first) {
            second = first;
            first = value;
        } else if (value > second) {
            second = value;
        }
    }
    return static_cast<double>(first) - second;
}

SeveritySearch falsifySeverity(const IRGraph& original, const IRGraph& candidate,
                               const std::vector<std::vector<float>>& centres, float radius,
                               std::size_t steps) {
    SeveritySearch result;
    const IRGraph a = canonicalize(original);
    const IRGraph b = canonicalize(candidate);
    bool sigmoidA = false, sigmoidB = false;
    const std::string logitsA = logitsValue(a, &sigmoidA);
    const std::string logitsB = logitsValue(b, &sigmoidB);
    // Returns the original's margin at a disagreement, or -1.
    const auto severityAt = [&](const std::vector<float>& x, int* ca = nullptr, int* cb = nullptr) {
        ++result.evaluations;
        const auto p = run(a, x);
        const auto q = run(b, x);
        if (p.empty() || q.empty()) return -1.0;
        const std::size_t pc = predictedClass(p), qc = predictedClass(q);
        if (pc == qc) return -1.0;
        if (ca != nullptr) *ca = static_cast<int>(pc);
        if (cb != nullptr) *cb = static_cast<int>(qc);
        return confidenceMargin(p);
    };
    std::mt19937 generator(20260917u);
    for (const auto& centre : centres) {
        const std::size_t c = predictedClass(run(a, centre));
        double ballBest = severityAt(centre);
        std::vector<float> ballPoint = centre;
        const float step = 2.0f * radius / static_cast<float>(std::max<std::size_t>(1, steps));
        for (int objective = 0; objective < 4; ++objective) {
            std::vector<float> x = centre;
            for (std::size_t iteration = 0; iteration < steps; ++iteration) {
                float ma = 0.0f, mb = 0.0f;
                std::vector<float> ga, gb, g(x.size(), 0.0f);
                if ((objective != 0 && !marginGradient(a, logitsA, sigmoidA, x, c, ma, ga)) ||
                    (objective != 1 && !marginGradient(b, logitsB, sigmoidB, x, c, mb, gb))) {
                    break;
                }
                for (std::size_t i = 0; i < g.size(); ++i) {
                    g[i] = objective == 0   ? gb[i]
                           : objective == 1 ? ga[i]
                           : objective == 2 ? gb[i] - ga[i]
                                            : ga[i] - gb[i];
                }
                for (std::size_t i = 0; i < x.size(); ++i) {
                    const float move = g[i] > 0.0f ? -step : g[i] < 0.0f ? step : 0.0f;
                    x[i] = std::clamp(x[i] + move, centre[i] - radius, centre[i] + radius);
                }
                const double severity = severityAt(x);
                if (severity > ballBest) {
                    ballBest = severity;
                    ballPoint = x;
                }
            }
        }
        if (ballBest < 0.0) continue;
        // Raise the original's confidence while the models still disagree.
        float scale = 0.25f * radius;
        std::normal_distribution<float> noise(0.0f, 1.0f);
        for (int trial = 0; trial < 24; ++trial) {
            std::vector<float> x = ballPoint;
            for (std::size_t i = 0; i < x.size(); ++i) {
                x[i] = std::clamp(x[i] + scale * noise(generator), centre[i] - radius,
                                  centre[i] + radius);
            }
            const double severity = severityAt(x);
            if (severity > ballBest) {
                ballBest = severity;
                ballPoint = std::move(x);
                scale *= 1.5f;
            } else {
                scale *= 0.7f;
            }
        }
        ++result.refutedBalls;
        if (ballBest > result.severity || result.witness.empty()) {
            result.severity = ballBest;
            result.witness = ballPoint;
            severityAt(ballPoint, &result.originalClass, &result.compressedClass);
        }
    }
    return result;
}

CompressionAudit auditCompression(const IRGraph& original, const Compression& compression,
                                  const Dataset& test, float radius, std::size_t maxBalls,
                                  std::size_t maxBoxesPerBall) {
    const double weightSize = compression.kind == CompressionKind::Pruning
                                  ? 1.0 - compression.pruneFraction
                                  : static_cast<double>(compression.bits) / 32.0;
    return auditCompressedModel(original, compressModel(original, compression), compression.name,
                                weightSize, test, radius, maxBalls, maxBoxesPerBall);
}

CompressionAudit auditCompressedModel(const IRGraph& original, const IRGraph& compressed,
                                      const std::string& name, double weightSize,
                                      const Dataset& test, float radius, std::size_t maxBalls,
                                      std::size_t maxBoxesPerBall) {
    const auto start = std::chrono::steady_clock::now();
    CompressionAudit audit;
    audit.compression = name;
    audit.weightSize = weightSize;
    audit.accuracyBefore = accuracy(original, test);
    audit.accuracyAfter = accuracy(compressed, test);
    audit.testFlips = searchTestSet(original, compressed, test).disagreements;
    std::vector<std::vector<float>> centres = test.rows;
    if (maxBalls != 0 && centres.size() > maxBalls) centres.resize(maxBalls);
    audit.certificate = certifySeverity(original, compressed, centres, radius, maxBoxesPerBall);
    audit.search = falsifySeverity(original, compressed, centres, radius);
    audit.certifiedRisk = riskLevel(static_cast<float>(audit.certificate.bound), 0);
    audit.foundRisk = audit.search.witness.empty()
                          ? "none found"
                          : riskLevel(static_cast<float>(audit.search.severity), 0);
    const bool certifiedLow = audit.certificate.bound < 0.2;
    const bool refutedHigh = !audit.search.witness.empty() && audit.search.severity >= 0.2;
    audit.verdict = certifiedLow ? Verdict::Certified
                    : refutedHigh ? Verdict::Refuted
                                  : Verdict::Undecided;
    audit.milliseconds = elapsedMs(start);
    return audit;
}

// ------------------------------------------------------------ binary audit

std::vector<std::string> defaultAuditFlags() {
    return {"-O0", "-O2", "-O3 -march=native -ffp-contract=fast", "-O3 -ffast-math"};
}

namespace {

std::string quotePath(const std::filesystem::path& path) { return "\"" + path.string() + "\""; }

std::string compilerCommand() {
    if (const char* cxx = std::getenv("CXX")) {
        if (*cxx != '\0') return cxx;
    }
    return "c++";
}

#if defined(_WIN32)
constexpr const char* kNull = "NUL";
#else
constexpr const char* kNull = "/dev/null";
#endif

}  // namespace

BinaryAuditReport auditGeneratedBinary(const IRGraph& graph, const std::string& generatedDirectory,
                                       const std::vector<std::vector<float>>& inputs,
                                       const std::vector<std::string>& flagSets) {
    BinaryAuditReport report;
    report.compiler = compilerCommand();
    const std::string probe = report.compiler + " --version > " + kNull + " 2>&1";
    report.available = std::system(probe.c_str()) == 0;
    if (!report.available || inputs.empty()) return report;

    const std::filesystem::path directory = std::filesystem::path(generatedDirectory) / "audit";
    std::filesystem::create_directories(directory);
    const auto inputPath = directory / "inputs.csv";
    {
        std::ofstream file(inputPath);
        file << std::setprecision(9);
        for (const auto& row : inputs) {
            for (std::size_t i = 0; i < row.size(); ++i) file << (i ? "," : "") << row[i];
            file << "\n";
        }
    }
    std::vector<std::vector<float>> expected;
    for (const auto& row : inputs) expected.push_back(run(graph, row));

    for (std::size_t index = 0; index < flagSets.size(); ++index) {
        BinaryFlagResult result;
        result.flags = flagSets[index];
#if defined(_WIN32)
        const auto binary = directory / ("model_" + std::to_string(index) + ".exe");
#else
        const auto binary = directory / ("model_" + std::to_string(index));
#endif
        const auto output = directory / ("scores_" + std::to_string(index) + ".csv");
        const auto log = directory / ("build_" + std::to_string(index) + ".log");
        const std::string build = report.compiler + " -std=c++17 " + result.flags + " " +
                                  quotePath(std::filesystem::path(generatedDirectory) / "model.cpp") +
                                  " -o " + quotePath(binary) + " > " + quotePath(log) + " 2>&1";
        const auto start = std::chrono::steady_clock::now();
        result.built = std::system(build.c_str()) == 0;
        result.buildSeconds = elapsedMs(start) / 1000.0;
        if (!result.built) {
            result.error = "build failed (see " + log.string() + ")";
            report.results.push_back(result);
            continue;
        }
        const auto runLog = directory / ("run_" + std::to_string(index) + ".log");
        const std::string execute = quotePath(binary) + " --csv " + quotePath(inputPath) + " > " +
                                    quotePath(output) + " 2> " + quotePath(runLog);
#if defined(_WIN32)
        // cmd.exe strips the outer quotes of a command that starts with one.
        const std::string shell = "\"" + execute + "\"";
#else
        const std::string& shell = execute;
#endif
        if (std::system(shell.c_str()) != 0) {
            result.error = "binary failed (see " + runLog.string() + ")";
            report.results.push_back(result);
            continue;
        }
        std::ifstream scores(output);
        std::string line;
        std::getline(scores, line);  // header
        std::size_t row = 0;
        while (std::getline(scores, line) && row < expected.size()) {
            std::stringstream cells(line);
            std::string cell;
            std::vector<double> values;
            while (std::getline(cells, cell, ',')) values.push_back(std::strtod(cell.c_str(), nullptr));
            const auto& want = expected[row];
            if (values.size() < 2 + want.size()) break;
            std::vector<float> got;
            for (std::size_t i = 0; i < want.size(); ++i) {
                got.push_back(static_cast<float>(values[2 + i]));
                result.maximumAbsoluteError =
                    std::max(result.maximumAbsoluteError,
                             std::fabs(static_cast<double>(got.back()) - want[i]));
            }
            // The binary reports argmax; compare it with the interpreter's.
            const std::size_t binaryClass = static_cast<std::size_t>(values[1]);
            const std::size_t interpreterClass = static_cast<std::size_t>(
                std::max_element(want.begin(), want.end()) - want.begin());
            result.predictionMismatches += binaryClass != interpreterClass ? 1 : 0;
            ++row;
        }
        result.rows = row;
        if (row != expected.size()) result.error = "read " + std::to_string(row) + " rows";
        report.results.push_back(result);
    }
    return report;
}

}  // namespace modelforge

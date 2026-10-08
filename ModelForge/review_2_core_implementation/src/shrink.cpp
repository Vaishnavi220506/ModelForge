#include "shrink.h"

#include "verifier.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>

namespace modelforge {
namespace {

constexpr float kBox = 4.0f;  // z-score box for standardised inputs
constexpr float kRealistic = 0.5f;  // max RMS z distance to a real test row

std::vector<float> probabilities(const IRGraph& graph, const std::vector<float>& x) {
    DiagnosticEngine diagnostics;
    auto result = executeIR(graph, x, diagnostics);
    return result ? *result : std::vector<float>{};
}

int argmax(const std::vector<float>& values) {
    if (values.empty()) return -1;
    return static_cast<int>(std::max_element(values.begin(), values.end()) - values.begin());
}

float rmsDistance(const std::vector<float>& a, const std::vector<float>& b) {
    float total = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) total += (a[i] - b[i]) * (a[i] - b[i]);
    return std::sqrt(total / static_cast<float>(std::max<std::size_t>(1, a.size())));
}

// Evaluates both models, counts evaluations and keeps the worst disagreement.
class Tracker {
public:
    Tracker(const IRGraph& original, const IRGraph& compressed, const Dataset& test,
            std::string strategy, std::size_t budget)
        : original_(original), compressed_(compressed), test_(test) {
        result_.strategy = std::move(strategy);
        result_.budget = budget;
    }

    bool exhausted() const { return result_.candidateEvaluations >= base_ + result_.budget; }

    // Every search first runs the standard test-set check (not charged to the
    // extra budget) and returns the compressed model's margin per test row.
    std::vector<float> testPass() {
        std::vector<float> margins;
        for (const auto& row : test_.rows) margins.push_back(check(row));
        base_ = result_.candidateEvaluations;
        return margins;
    }

    int originalClass(const std::vector<float>& x) {
        ++result_.originalEvaluations;
        return argmax(probabilities(original_, x));
    }

    int compressedClass(const std::vector<float>& x) {
        ++result_.candidateEvaluations;
        return argmax(probabilities(compressed_, x));
    }

    // Full check of one input; returns the candidate's margin for the
    // original's class (negative means disagreement), used as GA fitness.
    float check(const std::vector<float>& x, float* originalMargin = nullptr,
                bool* disagree = nullptr) {
        const auto p = probabilities(original_, x);
        const auto q = probabilities(compressed_, x);
        ++result_.originalEvaluations;
        ++result_.candidateEvaluations;
        const int c = argmax(p);
        const int d = argmax(q);
        if (c < 0 || d < 0) return 0.0f;
        float other = 0.0f;
        for (std::size_t j = 0; j < q.size(); ++j) {
            if (static_cast<int>(j) != c) other = std::max(other, q[j]);
        }
        std::vector<float> sorted = p;
        std::sort(sorted.begin(), sorted.end(), std::greater<float>());
        const float margin = sorted.size() > 1 ? sorted[0] - sorted[1] : 1.0f;
        if (originalMargin) *originalMargin = margin;
        if (disagree) *disagree = c != d;
        // Severity: the original's margin between its top two classes.
        if (c != d) record(x, margin, c, d);
        return q[static_cast<std::size_t>(c)] - other;
    }

    void shift(float value) { result_.maxBoundaryShift = std::max(result_.maxBoundaryShift, value); }
    DisagreementSearch finish() { return result_; }

private:
    void record(const std::vector<float>& x, float severity, int c, int d) {
        // Only realistic inputs count: within kRealistic z-units (RMS) of a test row.
        float nearest = std::numeric_limits<float>::infinity();
        for (const auto& row : test_.rows) nearest = std::min(nearest, rmsDistance(x, row));
        if (nearest > kRealistic) return;
        ++result_.disagreements;
        if (result_.firstDisagreement == 0) result_.firstDisagreement = result_.candidateEvaluations;
        if (severity <= result_.worstSeverity && !result_.worstInput.empty()) return;
        result_.worstSeverity = severity;
        result_.worstInput = x;
        result_.originalClass = c;
        result_.compressedClass = d;
        result_.worstDistance = nearest;
    }

    const IRGraph& original_;
    const IRGraph& compressed_;
    const Dataset& test_;
    DisagreementSearch result_;
    std::size_t base_ = 0;
};

std::vector<float> lerp(const std::vector<float>& a, const std::vector<float>& b, float t) {
    std::vector<float> x(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) x[i] = a[i] + t * (b[i] - a[i]);
    return x;
}

float roundToHalf(float value) {
    if (value == 0.0f || !std::isfinite(value)) return value;
    if (std::fabs(value) > 65504.0f) return std::copysign(65504.0f, value);
    if (std::fabs(value) < std::ldexp(1.0f, -14)) {  // half subnormal: multiples of 2^-24
        return std::nearbyint(value * std::ldexp(1.0f, 24)) * std::ldexp(1.0f, -24);
    }
    int exponent = 0;
    const float mantissa = std::frexp(value, &exponent);  // [0.5, 1)
    return std::ldexp(std::nearbyint(mantissa * 2048.0f) / 2048.0f, exponent);
}

}  // namespace

bool loadDataset(const std::string& path, std::size_t expectedFeatures, Dataset& dataset,
                 std::string& error) {
    std::ifstream file(path);
    if (!file) {
        error = "Cannot open " + path;
        return false;
    }
    dataset = {};
    std::string line;
    std::size_t number = 0;
    while (std::getline(file, line)) {
        ++number;
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        std::vector<float> values;
        std::stringstream stream(line);
        std::string item;
        while (std::getline(stream, item, ',')) {
            char* end = nullptr;
            const float value = std::strtof(item.c_str(), &end);
            if (end == item.c_str()) {
                error = path + ":" + std::to_string(number) + ": not a number '" + item + "'";
                return false;
            }
            values.push_back(value);
        }
        if (values.size() == expectedFeatures + 1) {
            dataset.labels.push_back(static_cast<int>(values.back()));
            values.pop_back();
        } else if (values.size() == expectedFeatures) {
            dataset.labels.push_back(-1);
        } else {
            error = path + ":" + std::to_string(number) + ": expected " +
                    std::to_string(expectedFeatures) + " features";
            return false;
        }
        dataset.rows.push_back(std::move(values));
    }
    if (dataset.rows.empty()) error = path + " has no rows";
    return !dataset.rows.empty();
}

std::vector<Compression> standardCompressions() {
    return {{"fp16", CompressionKind::Float16, 16, 0.0f},
            {"int8", CompressionKind::IntQuantization, 8, 0.0f},
            {"int4", CompressionKind::IntQuantization, 4, 0.0f},
            {"prune30", CompressionKind::Pruning, 32, 0.3f},
            {"prune50", CompressionKind::Pruning, 32, 0.5f}};
}

bool parseCompression(const std::string& name, Compression& compression) {
    for (const auto& candidate : standardCompressions()) {
        if (candidate.name == name) {
            compression = candidate;
            return true;
        }
    }
    if (name.rfind("int", 0) == 0 || name.rfind("prune", 0) == 0) {
        try {
            if (name.rfind("int", 0) == 0) {
                const int bits = std::stoi(name.substr(3));
                if (bits < 2 || bits > 16) return false;
                compression = {name, CompressionKind::IntQuantization, bits, 0.0f};
            } else {
                const int percent = std::stoi(name.substr(5));
                if (percent <= 0 || percent >= 100) return false;
                compression = {name, CompressionKind::Pruning, 32, percent / 100.0f};
            }
            return true;
        } catch (...) {
            return false;
        }
    }
    return false;
}

IRGraph compressModel(const IRGraph& graph, const Compression& compression) {
    IRGraph result = graph;
    std::set<std::string> weights;
    for (const auto& instruction : graph.instructions) {
        if ((instruction.operation == IROp::Gemm || instruction.operation == IROp::MatMul ||
             instruction.operation == IROp::FusedGemmRelu) &&
            instruction.inputs.size() >= 2 && graph.constants.count(instruction.inputs[1])) {
            weights.insert(instruction.inputs[1]);
        }
    }
    for (auto& [name, tensor] : result.constants) {
        auto& data = tensor.data;
        if (compression.kind == CompressionKind::Float16) {
            for (float& v : data) v = roundToHalf(v);
        } else if (weights.count(name) && !data.empty()) {
            if (compression.kind == CompressionKind::IntQuantization) {
                float largest = 0.0f;
                for (const float v : data) largest = std::max(largest, std::fabs(v));
                const float levels = static_cast<float>((1 << (compression.bits - 1)) - 1);
                const float scale = largest > 0.0f ? largest / levels : 1.0f;
                for (float& v : data) v = std::nearbyint(v / scale) * scale;
            } else {
                std::vector<float> magnitudes;
                for (const float v : data) magnitudes.push_back(std::fabs(v));
                const std::size_t cut = static_cast<std::size_t>(
                    compression.pruneFraction * static_cast<float>(magnitudes.size()));
                if (cut > 0) {
                    std::nth_element(magnitudes.begin(), magnitudes.begin() + (cut - 1), magnitudes.end());
                    const float threshold = magnitudes[cut - 1];
                    std::size_t removed = 0;
                    for (float& v : data) {
                        if (std::fabs(v) <= threshold && removed < cut) {
                            v = 0.0f;
                            ++removed;
                        }
                    }
                }
            }
        }
        if (result.values.count(name)) result.values[name].data = data;
    }
    result.name = graph.name + "_" + compression.name;
    return result;
}

std::string writeManifestText(const IRGraph& graph) {
    const auto shape = [](const Shape& s) {
        std::string text;
        for (std::size_t i = 0; i < s.size(); ++i) text += (i ? "," : "") + std::to_string(s[i]);
        return text;
    };
    std::ostringstream out;
    out << std::setprecision(9);
    out << "# Generated by ModelForge shrink (author: Vaishnavi).\n"
        << "model " << graph.name << "\n"
        << "input " << graph.inputName << " float32 " << shape(graph.values.at(graph.inputName).shape) << "\n"
        << "output " << graph.outputName << " float32 " << shape(graph.values.at(graph.outputName).shape) << "\n";
    std::set<std::string> written;
    for (const auto& instruction : graph.instructions) {
        for (const auto& input : instruction.inputs) {
            const auto constant = graph.constants.find(input);
            if (constant == graph.constants.end() || !written.insert(input).second) continue;
            out << "tensor " << input << " float32 " << shape(constant->second.shape) << " values=";
            for (std::size_t i = 0; i < constant->second.data.size(); ++i) {
                out << (i ? "," : "") << constant->second.data[i];
            }
            out << "\n";
        }
    }
    static const std::map<IROp, std::string> names = {
        {IROp::Gemm, "Gemm"}, {IROp::MatMul, "MatMul"}, {IROp::Add, "Add"},
        {IROp::Relu, "Relu"}, {IROp::Sigmoid, "Sigmoid"}, {IROp::Softmax, "Softmax"}};
    for (const auto& instruction : graph.instructions) {
        const auto name = names.find(instruction.operation);
        if (name == names.end()) continue;
        out << "node " << instruction.nodeName << " " << name->second;
        for (const auto& input : instruction.inputs) out << " " << input;
        out << " -> " << instruction.output;
        if (instruction.axis >= 0) out << " axis=" << instruction.axis;
        if (instruction.transA) out << " transA=1";
        if (instruction.transB) out << " transB=1";
        out << "\n";
    }
    return out.str();
}

DisagreementSearch searchTestSet(const IRGraph& original, const IRGraph& compressed,
                                 const Dataset& test) {
    Tracker tracker(original, compressed, test, "test_set", test.rows.size());
    for (const auto& row : test.rows) tracker.check(row);
    return tracker.finish();
}

DisagreementSearch searchNoise(const IRGraph& original, const IRGraph& compressed,
                               const Dataset& test, std::size_t budget, std::uint32_t seed,
                               float sigma) {
    Tracker tracker(original, compressed, test, "noise", budget);
    tracker.testPass();
    std::mt19937 generator(seed);
    std::uniform_int_distribution<std::size_t> pick(0, test.rows.size() - 1);
    std::normal_distribution<float> noise(0.0f, sigma);
    while (!tracker.exhausted()) {
        auto x = test.rows[pick(generator)];
        for (float& v : x) v = std::clamp(v + noise(generator), -kBox, kBox);
        tracker.check(x);
    }
    return tracker.finish();
}

DisagreementSearch searchGenetic(const IRGraph& original, const IRGraph& compressed,
                                 const Dataset& test, std::size_t budget, std::uint32_t seed) {
    Tracker tracker(original, compressed, test, "genetic", budget);
    std::mt19937 generator(seed);
    std::uniform_int_distribution<std::size_t> pick(0, test.rows.size() - 1);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    struct Individual {
        std::vector<float> x;
        float fitness;
    };
    // Fitness: minus the compressed model's margin for the original's class;
    // a positive value means the models disagree.
    std::vector<Individual> population;
    const auto margins = tracker.testPass();
    std::vector<std::size_t> rank(margins.size());
    std::iota(rank.begin(), rank.end(), 0);
    std::sort(rank.begin(), rank.end(), [&](auto a, auto b) { return margins[a] < margins[b]; });
    for (std::size_t i = 0; i < std::min<std::size_t>(12, rank.size()); ++i) {
        population.push_back({test.rows[rank[i]], -margins[rank[i]]});
    }
    (void)pick;
    const auto tournament = [&]() -> const Individual& {
        const auto& a = population[generator() % population.size()];
        const auto& b = population[generator() % population.size()];
        return a.fitness > b.fitness ? a : b;
    };
    while (!tracker.exhausted() && !population.empty()) {
        const auto& p1 = tournament();
        const auto& p2 = tournament();
        std::vector<float> child(p1.x.size());
        for (std::size_t i = 0; i < child.size(); ++i) {
            child[i] = unit(generator) < 0.5f ? p1.x[i] : p2.x[i];
            if (unit(generator) < 0.3f) child[i] += noise(generator);
            child[i] = std::clamp(child[i], -kBox, kBox);
        }
        const float fitness = -tracker.check(child);
        auto worst = std::min_element(population.begin(), population.end(),
                                      [](const auto& a, const auto& b) { return a.fitness < b.fitness; });
        if (fitness > worst->fitness) *worst = {std::move(child), fitness};
    }
    return tracker.finish();
}

DisagreementSearch searchBoundaryShift(const IRGraph& original, const IRGraph& compressed,
                                       const Dataset& test, std::size_t budget) {
    Tracker tracker(original, compressed, test, "boundary_shift", budget);
    tracker.testPass();
    const std::size_t n = test.rows.size();
    std::vector<int> classes(n);
    std::vector<float> margins(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto p = probabilities(original, test.rows[i]);
        classes[i] = argmax(p);
        std::sort(p.begin(), p.end(), std::greater<float>());
        margins[i] = p.size() > 1 ? p[0] - p[1] : 1.0f;
    }
    // Segments: each test row to its nearest rows of another class, most
    // uncertain rows first (boundaries are closest to them).
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](auto a, auto b) { return margins[a] < margins[b]; });
    std::vector<std::pair<std::size_t, std::size_t>> segments;
    std::set<std::pair<std::size_t, std::size_t>> seen;
    for (int neighbour = 0; neighbour < 3; ++neighbour) {
        for (const std::size_t i : order) {
            std::vector<std::pair<float, std::size_t>> others;
            for (std::size_t j = 0; j < n; ++j) {
                if (classes[j] != classes[i]) others.emplace_back(rmsDistance(test.rows[i], test.rows[j]), j);
            }
            if (static_cast<std::size_t>(neighbour) >= others.size()) continue;
            std::nth_element(others.begin(), others.begin() + neighbour, others.end());
            const std::size_t j = others[static_cast<std::size_t>(neighbour)].second;
            if (seen.insert({std::min(i, j), std::max(i, j)}).second) segments.emplace_back(i, j);
        }
    }

    // Phase 1 (screen, at most half the budget): locate the original boundary
    // on each segment and spend two compressed evaluations to see whether the
    // compressed boundary moved, and in which direction.
    struct Shifted {
        std::size_t i, j;
        float lo, hi;
        bool towardA;
    };
    std::vector<Shifted> shifted;
    const std::size_t screenEnd = tracker.finish().candidateEvaluations + budget * 3 / 10;
    for (const auto& [i, j] : segments) {
        if (tracker.finish().candidateEvaluations + 2 > screenEnd) break;
        const auto& a = test.rows[i];
        const auto& b = test.rows[j];
        const int classA = classes[i];
        float lo = 0.0f, hi = 1.0f;
        for (int step = 0; step < 30; ++step) {
            const float mid = 0.5f * (lo + hi);
            (tracker.originalClass(lerp(a, b, mid)) == classA ? lo : hi) = mid;
        }
        const bool movedTowardA = tracker.compressedClass(lerp(a, b, lo)) != classA;
        const bool movedTowardB = tracker.compressedClass(lerp(a, b, hi)) == classA;
        if (movedTowardA || movedTowardB) shifted.push_back({i, j, lo, hi, movedTowardA});
    }

    // Phase 2: bisect the compressed boundary on the shifted segments; the far
    // edge of each gap is the most severe disagreement on that segment.
    std::vector<std::vector<float>> witnesses;
    const std::size_t bisectEnd = screenEnd + budget * 3 / 10;
    for (const auto& segment : shifted) {
        if (tracker.exhausted() || tracker.finish().candidateEvaluations >= bisectEnd) break;
        const auto& a = test.rows[segment.i];
        const auto& b = test.rows[segment.j];
        const int classA = classes[segment.i];
        const float length = rmsDistance(a, b);
        float left = segment.towardA ? 0.0f : segment.hi;
        float right = segment.towardA ? segment.lo : 1.0f;
        for (int step = 0; step < 12 && !tracker.exhausted(); ++step) {
            const float mid = 0.5f * (left + right);
            (tracker.compressedClass(lerp(a, b, mid)) == classA ? left : right) = mid;
        }
        const float edge = segment.towardA ? right : left;
        tracker.shift(std::fabs(edge - (segment.towardA ? segment.lo : segment.hi)) * length);
        if (tracker.exhausted()) break;
        const auto witness = lerp(a, b, edge);
        tracker.check(witness);
        witnesses.push_back(witness);
    }

    // Phase 3: severity-aware evolutionary refinement seeded with the
    // boundary witnesses and the test rows the compressed model is least sure
    // about. Fitness rewards realistic disagreements where the original model
    // was confident; before a disagreement it rewards closing the gap.
    struct Individual {
        std::vector<float> x;
        float fitness;
    };
    std::vector<Individual> population;
    const auto score = [&](const std::vector<float>& x) {
        float margin = 0.0f;
        bool disagree = false;
        const float candidateMargin = tracker.check(x, &margin, &disagree);
        return disagree ? 1.0f + margin : -candidateMargin;
    };
    for (const auto& witness : witnesses) {
        if (tracker.exhausted() || population.size() >= 12) break;
        population.push_back({witness, score(witness)});
    }
    for (std::size_t k = 0; k < order.size() && population.size() < 12 && !tracker.exhausted(); ++k) {
        population.push_back({test.rows[order[k]], score(test.rows[order[k]])});
    }
    std::mt19937 generator(7);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    while (!tracker.exhausted() && !population.empty()) {
        const auto pickOne = [&]() -> const Individual& {
            const auto& x = population[generator() % population.size()];
            const auto& y = population[generator() % population.size()];
            return x.fitness > y.fitness ? x : y;
        };
        const auto& p1 = pickOne();
        const auto& p2 = pickOne();
        std::vector<float> child(p1.x.size());
        for (std::size_t k = 0; k < child.size(); ++k) {
            child[k] = unit(generator) < 0.5f ? p1.x[k] : p2.x[k];
            if (unit(generator) < 0.3f) child[k] += noise(generator);
            child[k] = std::clamp(child[k], -kBox, kBox);
        }
        const float fitness = score(child);
        auto weakest = std::min_element(population.begin(), population.end(),
                                        [](const auto& x, const auto& y) { return x.fitness < y.fitness; });
        if (fitness > weakest->fitness) *weakest = {std::move(child), fitness};
    }
    return tracker.finish();
}

double accuracy(const IRGraph& graph, const Dataset& data) {
    std::size_t correct = 0, labelled = 0;
    for (std::size_t i = 0; i < data.rows.size(); ++i) {
        if (data.labels[i] < 0) continue;
        ++labelled;
        correct += argmax(probabilities(graph, data.rows[i])) == data.labels[i];
    }
    return labelled == 0 ? std::nan("") : static_cast<double>(correct) / labelled;
}

std::string riskLevel(float worstSeverity, std::size_t) {
    // worstSeverity is the original's top-two margin at the worst realistic flip.
    if (worstSeverity < 0.05f) return "negligible";
    if (worstSeverity < 0.2f) return "low";
    if (worstSeverity < 0.5f) return "moderate";
    return "high";
}

}  // namespace modelforge

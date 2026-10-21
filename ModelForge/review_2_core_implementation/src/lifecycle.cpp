#include "lifecycle.h"

#include "optimizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <optional>
#include <random>
#include <set>
#include <sstream>

namespace modelforge {
namespace {

namespace fs = std::filesystem;

double elapsedMs(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

std::string number(double value) {
    if (!std::isfinite(value)) return "null";
    std::ostringstream output;
    output << std::setprecision(6) << value;
    return output.str();
}

std::string quoted(const std::string& value) { return "\"" + jsonEscapeText(value) + "\""; }

bool isDense(IROp op) { return op == IROp::Gemm || op == IROp::MatMul || op == IROp::FusedGemmRelu; }

// Weight tensors of dense layers, in instruction order.
std::vector<std::string> weightTensors(const IRGraph& graph) {
    std::vector<std::string> names;
    for (const auto& instruction : graph.instructions) {
        if (isDense(instruction.operation) && instruction.inputs.size() >= 2 &&
            graph.constants.count(instruction.inputs[1]) &&
            std::find(names.begin(), names.end(), instruction.inputs[1]) == names.end()) {
            names.push_back(instruction.inputs[1]);
        }
    }
    return names;
}

void setConstant(IRGraph& graph, const std::string& name, std::vector<float> data) {
    graph.constants[name].data = data;
    if (graph.values.count(name)) graph.values[name].data = std::move(data);
}

std::vector<std::vector<float>> ballCentres(const Dataset& test, std::size_t maxBalls) {
    std::vector<std::vector<float>> centres = test.rows;
    if (maxBalls != 0 && centres.size() > maxBalls) centres.resize(maxBalls);
    return centres;
}

IRInstruction* firstHiddenRelu(IRGraph& graph) {
    for (auto& instruction : graph.instructions) {
        if (instruction.operation == IROp::Relu) return &instruction;
    }
    return nullptr;
}

const IRInstruction* firstDense(const IRGraph& graph) {
    for (const auto& instruction : graph.instructions) {
        if (isDense(instruction.operation) && instruction.inputs.size() >= 3) return &instruction;
    }
    return nullptr;
}

}  // namespace

// ------------------------------------------------------------ mixed precision

IRGraph compressLayers(const IRGraph& graph, const std::map<std::string, int>& bits) {
    std::map<int, IRGraph> levels;
    IRGraph result = graph;
    for (const auto& [name, b] : bits) {
        if (b >= 32 || !graph.constants.count(name)) continue;
        if (!levels.count(b)) {
            Compression compression;
            if (b == 16) {
                compression.kind = CompressionKind::Float16;
            } else {
                compression.kind = CompressionKind::IntQuantization;
                compression.bits = b;
            }
            compression.name = "b" + std::to_string(b);
            levels.emplace(b, compressModel(graph, compression));
        }
        setConstant(result, name, levels.at(b).constants.at(name).data);
    }
    std::string suffix;
    for (const auto& [name, b] : bits) suffix += (suffix.empty() ? "_" : "-") + std::to_string(b);
    result.name = graph.name + "_mixed" + suffix;
    return result;
}

MixedPrecisionPlan searchMixedPrecision(const IRGraph& original, const Dataset& test, float radius,
                                        std::size_t maxBalls, std::size_t maxBoxesPerBall) {
    const auto start = std::chrono::steady_clock::now();
    MixedPrecisionPlan plan;
    const auto centres = ballCentres(test, maxBalls);
    std::size_t total = 0;
    std::map<std::string, int> bits;
    for (const auto& name : weightTensors(original)) {
        plan.layers.push_back({name, original.constants.at(name).data.size(), 32});
        bits[name] = 32;
        total += original.constants.at(name).data.size();
    }
    const auto sizeOf = [&](const std::map<std::string, int>& assignment) {
        double kept = 0.0;
        for (const auto& layer : plan.layers) {
            kept += static_cast<double>(layer.parameters) * assignment.at(layer.weight);
        }
        return total == 0 ? 1.0 : kept / (32.0 * static_cast<double>(total));
    };
    const auto safe = [&](const std::map<std::string, int>& assignment) {
        ++plan.trials;
        const IRGraph candidate = compressLayers(original, assignment);
        return certifySeverity(original, candidate, centres, radius, maxBoxesPerBall).bound < 0.2;
    };
    // Smallest uniform precision that is certified, for comparison.
    for (const int b : {4, 8, 16}) {
        auto uniform = bits;
        for (auto& [name, value] : uniform) value = b;
        if (safe(uniform)) {
            plan.bestUniform = b == 16 ? "fp16" : "int" + std::to_string(b);
            plan.bestUniformSize = b / 32.0;
            break;
        }
    }
    // Greedy: largest layers first, fewest bits that keep the proof.
    std::vector<std::size_t> order(plan.layers.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return plan.layers[a].parameters > plan.layers[b].parameters;
    });
    for (const std::size_t i : order) {
        for (const int b : {4, 8, 16}) {
            auto trial = bits;
            trial[plan.layers[i].weight] = b;
            if (sizeOf(trial) >= sizeOf(bits)) break;
            if (safe(trial)) {
                bits = std::move(trial);
                break;
            }
        }
    }
    for (auto& layer : plan.layers) layer.bits = bits.at(layer.weight);
    plan.weightSize = sizeOf(bits);
    std::string name = "mixed(";
    for (std::size_t i = 0; i < plan.layers.size(); ++i) {
        name += (i ? "," : "") + std::to_string(plan.layers[i].bits);
    }
    plan.audit = auditCompressedModel(original, compressLayers(original, bits), name + ")",
                                      plan.weightSize, test, radius, maxBalls, maxBoxesPerBall);
    plan.milliseconds = elapsedMs(start);
    return plan;
}

// ------------------------------------------------------------ culprit localisation

CulpritReport localizeCulprits(const IRGraph& originalInput, const IRGraph& candidateInput,
                               const Box& domain, const TemperOptions& options) {
    const auto start = std::chrono::steady_clock::now();
    CulpritReport report;
    // Canonical forms (unfused, folded, dead code removed) are bit-identical to
    // the inputs, so a hand-fused copy can be compared parameter by parameter.
    const IRGraph original = canonicalize(originalInput);
    const IRGraph candidate = canonicalize(candidateInput);
    // Same operators, wiring and tensor shapes: only parameter values differ.
    bool same = original.instructions.size() == candidate.instructions.size() &&
                original.constants.size() == candidate.constants.size();
    for (std::size_t i = 0; same && i < original.instructions.size(); ++i) {
        const auto& a = original.instructions[i];
        const auto& b = candidate.instructions[i];
        same = a.operation == b.operation && a.inputs == b.inputs && a.output == b.output &&
               a.transA == b.transA && a.transB == b.transB;
    }
    std::vector<ParameterEdit> edits;
    for (const auto& [name, tensor] : original.constants) {
        if (!same) break;
        const auto other = candidate.constants.find(name);
        same = other != candidate.constants.end() && other->second.data.size() == tensor.data.size();
        if (!same) break;
        for (std::size_t i = 0; i < tensor.data.size(); ++i) {
            const float a = tensor.data[i], b = other->second.data[i];
            if (std::memcmp(&a, &b, sizeof a) != 0) edits.push_back({name, i, a, b});
        }
    }
    report.applicable = same && !edits.empty();
    report.edits = edits.size();
    if (!report.applicable) {
        report.milliseconds = elapsedMs(start);
        return report;
    }
    std::sort(edits.begin(), edits.end(), [](const ParameterEdit& a, const ParameterEdit& b) {
        return a.tensor != b.tensor ? a.tensor < b.tensor : a.index < b.index;
    });
    const auto apply = [&](const IRGraph& base, const std::vector<std::size_t>& subset, bool revert) {
        IRGraph result = base;
        for (const std::size_t e : subset) {
            const auto& edit = edits[e];
            auto data = result.constants.at(edit.tensor).data;
            data[edit.index] = revert ? edit.before : edit.after;
            setConstant(result, edit.tensor, std::move(data));
        }
        return result;
    };
    const auto refuted = [&](const std::vector<std::size_t>& subset) {
        ++report.decisions;
        return decideTransformation(original, apply(original, subset, false), domain, options).verdict ==
               Verdict::Refuted;
    };
    std::vector<std::size_t> current(edits.size());
    for (std::size_t i = 0; i < current.size(); ++i) current[i] = i;
    if (!refuted(current)) {
        report.milliseconds = elapsedMs(start);
        return report;  // nothing to localise: the edit set is not refuted
    }
    // ddmin (Zeller and Hildebrandt 2002): a 1-minimal failure-inducing subset.
    std::size_t granularity = 2;
    while (current.size() >= 2) {
        const std::size_t chunk = (current.size() + granularity - 1) / granularity;
        bool reduced = false;
        for (std::size_t begin = 0; begin < current.size() && !reduced; begin += chunk) {
            const std::size_t end = std::min(current.size(), begin + chunk);
            const std::vector<std::size_t> part(current.begin() + begin, current.begin() + end);
            if (refuted(part)) {
                current = part;
                granularity = 2;
                reduced = true;
                break;
            }
            if (granularity > 2) {
                std::vector<std::size_t> complement(current.begin(), current.begin() + begin);
                complement.insert(complement.end(), current.begin() + end, current.end());
                if (refuted(complement)) {
                    current = std::move(complement);
                    granularity = std::max<std::size_t>(granularity - 1, 2);
                    reduced = true;
                }
            }
        }
        if (reduced) continue;
        if (granularity >= current.size()) break;
        granularity = std::min(current.size(), granularity * 2);
    }
    for (const std::size_t e : current) report.culprits.push_back(edits[e]);
    report.culpritsAlone = Verdict::Refuted;
    ++report.decisions;
    const auto fixed = decideTransformation(original, apply(candidate, current, true), domain, options);
    report.afterRevert = fixed.verdict;
    report.revertKind = fixed.kind;
    report.milliseconds = elapsedMs(start);
    return report;
}

// ------------------------------------------------------------ incremental certificates

namespace {

void writeFloats(std::ostream& out, const std::vector<float>& values) {
    out << values.size();
    for (const float v : values) out << ' ' << v;
}

bool readFloats(std::istream& in, std::vector<float>& values) {
    std::size_t n = 0;
    if (!(in >> n) || n > (1u << 20)) return false;
    values.resize(n);
    for (float& v : values) {
        std::string token;
        if (!(in >> token)) return false;
        v = std::strtof(token.c_str(), nullptr);
    }
    return true;
}

Verdict verdictFromName(const std::string& name) {
    return name == "CERTIFIED" ? Verdict::Certified
           : name == "REFUTED" ? Verdict::Refuted
                               : Verdict::Undecided;
}

CertificateKind kindFromName(const std::string& name) {
    for (const auto kind : {CertificateKind::Exact, CertificateKind::Bounded}) {
        if (name == certificateKindName(kind)) return kind;
    }
    return CertificateKind::None;
}

}  // namespace

bool saveCertificate(const std::string& path, const ProofCertificate& c) {
    std::ofstream out(path);
    if (!out) return false;
    out << std::setprecision(9);
    out << "temper-certificate 1\n"
        << "original " << c.original << "\ncandidate " << c.candidate << "\n"
        << "verdict " << verdictName(c.verdict) << "\nkind " << certificateKindName(c.kind) << "\n"
        << "tolerance " << c.tolerance << "\nbound " << c.bound << "\n";
    out << "domain ";
    writeFloats(out, c.domain.lower);
    out << ' ';
    writeFloats(out, c.domain.upper);
    out << "\nwitness ";
    writeFloats(out, c.witness);
    out << "\nleaves " << c.partition.size() << "\n";
    for (const auto& box : c.partition) {
        writeFloats(out, box.lower);
        out << ' ';
        writeFloats(out, box.upper);
        out << "\n";
    }
    return static_cast<bool>(out);
}

bool loadCertificate(const std::string& path, ProofCertificate& c, std::string& error) {
    std::ifstream in(path);
    std::string key, value;
    int version = 0;
    if (!in || !(in >> key >> version) || key != "temper-certificate" || version != 1) {
        error = path + ": not a Temper certificate";
        return false;
    }
    bool ok = true;
    const auto expect = [&](const char* name) {
        ok = ok && (in >> key) && key == name;
        return ok;
    };
    if (expect("original")) in >> c.original;
    if (expect("candidate")) in >> c.candidate;
    if (expect("verdict") && (in >> value)) c.verdict = verdictFromName(value);
    if (expect("kind") && (in >> value)) c.kind = kindFromName(value);
    if (expect("tolerance")) ok = ok && static_cast<bool>(in >> c.tolerance);
    if (expect("bound")) ok = ok && static_cast<bool>(in >> c.bound);
    if (expect("domain")) ok = ok && readFloats(in, c.domain.lower) && readFloats(in, c.domain.upper);
    if (expect("witness")) ok = ok && readFloats(in, c.witness);
    std::size_t leaves = 0;
    if (expect("leaves")) ok = ok && static_cast<bool>(in >> leaves);
    for (std::size_t i = 0; ok && i < leaves; ++i) {
        Box box;
        ok = readFloats(in, box.lower) && readFloats(in, box.upper) &&
             box.lower.size() == c.domain.size() && box.upper.size() == c.domain.size();
        c.partition.push_back(std::move(box));
    }
    ok = ok && c.domain.lower.size() == c.domain.upper.size();
    if (!ok) error = path + ": malformed Temper certificate";
    return ok;
}

IncrementalResult certifyIncrementally(const IRGraph& original, const IRGraph& candidate,
                                       const Box& domain, const CertifyOptions& options,
                                       const ProofCertificate* previous) {
    IncrementalResult out;
    out.certificate.original = fingerprintHex(graphFingerprint(canonicalize(original)));
    out.certificate.candidate = fingerprintHex(graphFingerprint(canonicalize(candidate)));
    out.certificate.domain = domain;
    out.certificate.tolerance = options.tolerance;
    const bool sameDomain = previous != nullptr && previous->domain.lower == domain.lower &&
                            previous->domain.upper == domain.upper;
    CertifyOptions run = options;
    if (sameDomain && previous->original == out.certificate.original &&
        previous->candidate == out.certificate.candidate && previous->verdict == Verdict::Certified &&
        previous->tolerance <= options.tolerance) {
        // Same pair, same domain: the stored proof still applies. Its leaves are
        // re-checked anyway (cheap: no splitting), so the file is not trusted.
        out.identical = true;
    }
    if (sameDomain && previous->witness.size() == domain.size() &&
        exposes(original, candidate, previous->witness, options.tolerance)) {
        // The last version's counterexample still breaks this one: one evaluation.
        out.carriedWitness = true;
        out.result.verdict = Verdict::Refuted;
        out.result.witness = previous->witness;
        out.result.refutedBy = "carried-witness";
        out.result.evaluations = 1;
        out.result.partition = previous->partition;
        out.certificate.verdict = Verdict::Refuted;
        out.certificate.partition = previous->partition;
        out.certificate.witness = previous->witness;
        return out;
    }
    if (sameDomain && !previous->partition.empty()) {
        run.seedPartition = &previous->partition;
        out.reused = true;
        out.seedLeaves = previous->partition.size();
    }
    out.result = certifyEquivalence(original, candidate, domain, run);
    out.certificate.verdict = out.result.verdict;
    out.certificate.kind = out.result.kind;
    out.certificate.bound = out.result.bound;
    out.certificate.partition = out.result.partition;
    if (out.result.witness) out.certificate.witness = *out.result.witness;
    return out;
}

// ------------------------------------------------------------ studies

namespace {

std::vector<fs::path> realModels(const std::string& directory) {
    std::vector<fs::path> models;
    std::error_code error;
    for (const auto& entry : fs::directory_iterator(directory, error)) {
        if (entry.path().extension() == ".mforge") models.push_back(entry.path());
    }
    std::sort(models.begin(), models.end());
    return models;
}

bool loadReal(const fs::path& path, const std::string& directory, IRGraph& graph, Dataset& test) {
    std::ifstream file(path);
    const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    DiagnosticEngine diagnostics;
    if (!compileManifestText(text, graph, diagnostics)) return false;
    const std::string stem = path.stem().string();
    const std::string dataset = stem.substr(0, stem.rfind('_'));
    std::string message;
    return loadDataset((fs::path(directory) / (dataset + "_test.csv")).string(),
                       elementCount(graph.values.at(graph.inputName).shape), test, message);
}

bool anyExposed(const IRGraph& a, const IRGraph& b, const std::vector<std::vector<float>>& inputs,
                float tolerance) {
    for (const auto& x : inputs) {
        if (exposes(a, b, x, tolerance)) return true;
    }
    return false;
}

std::vector<std::vector<float>> uniformInputs(const Box& box, std::size_t count, std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::vector<std::vector<float>> inputs(count, std::vector<float>(box.size()));
    for (auto& x : inputs) {
        for (std::size_t i = 0; i < box.size(); ++i) {
            x[i] = std::uniform_real_distribution<float>(box.lower[i], box.upper[i])(generator);
        }
    }
    return inputs;
}

struct BugCase {
    std::string bugClass, source;
    IRGraph reference, candidate;
};

// Faults modelled on root-cause classes reported by studies of real deep-learning
// compiler and converter bugs (Shen et al., FSE 2021; Du et al., 2021) and on
// recurring issue themes in TVM, TFLite and ONNX Runtime trackers.
std::vector<BugCase> makeBugCases(const IRGraph& graph) {
    std::vector<BugCase> cases;
    const IRInstruction* dense = firstDense(graph);
    if (dense == nullptr) return cases;
    const std::string w = dense->inputs[1], b = dense->inputs[2];
    const auto& wShape = graph.constants.at(w).shape;
    const std::size_t in = wShape[0], out = wShape[1];
    const auto withConstant = [&](const IRGraph& base, const std::string& name, auto edit) {
        IRGraph g = base;
        auto data = g.constants.at(name).data;
        edit(data);
        setConstant(g, name, std::move(data));
        return g;
    };
    const auto withRelu = [&](IROp op, float p) {
        IRGraph g = graph;
        if (IRInstruction* relu = firstHiddenRelu(g)) {
            relu->operation = op;
            relu->testFaultParameter = p;
        }
        return g;
    };
    // Layout / stride: two output channels of the weights swapped, bias not.
    cases.push_back({"layout_channel_swap", "layout transform", graph,
                     withConstant(graph, w, [&](std::vector<float>& d) {
                         for (std::size_t k = 0; k < in; ++k) std::swap(d[k * out], d[k * out + 1]);
                     })});
    // Fusion drops the bias of the fused dense layer.
    cases.push_back({"fusion_drops_bias", "operator fusion", graph,
                     withConstant(graph, b, [](std::vector<float>& d) { std::fill(d.begin(), d.end(), 0.0f); })});
    // Broadcast: bias[0] broadcast to every unit.
    cases.push_back({"bias_broadcast", "shape/broadcast", graph,
                     withConstant(graph, b, [](std::vector<float>& d) { std::fill(d.begin(), d.end(), d[0]); })});
    // Loop bound off by one: the last input feature is never read.
    cases.push_back({"loop_off_by_one", "code generation", graph,
                     withConstant(graph, w, [&](std::vector<float>& d) {
                         for (std::size_t j = 0; j < out; ++j) d[(in - 1) * out + j] = 0.0f;
                     })});
    // Activation lowering: ReLU lowered as ReLU6, LeakyReLU with a wrong slope,
    // and a comparison with an epsilon (x > 1e-6) instead of x > 0.
    cases.push_back({"relu6_lowering", "activation lowering", graph, withRelu(IROp::TestReluClamp, 6.0f)});
    cases.push_back({"leaky_default_slope", "activation lowering", graph, withRelu(IROp::TestLeakyRelu, 0.01f)});
    cases.push_back({"relu_epsilon_compare", "activation lowering", graph,
                     withRelu(IROp::TestReluDeadZone, 1.0e-3f)});
    // Quantisation, against the correct int8 model as reference.
    Compression int8;
    parseCompression("int8", int8);
    const IRGraph q = compressModel(graph, int8);
    const auto& original = graph.constants.at(w).data;
    float largest = 0.0f;
    for (const float v : original) largest = std::max(largest, std::fabs(v));
    const float scale = largest > 0.0f ? largest / 127.0f : 1.0f;
    cases.push_back({"requant_truncation", "quantisation rounding", q,
                     withConstant(q, w, [&](std::vector<float>& d) {
                         for (std::size_t i = 0; i < d.size(); ++i) d[i] = std::trunc(original[i] / scale) * scale;
                     })});
    cases.push_back({"requant_no_saturation", "quantisation overflow", q,
                     withConstant(q, w, [&](std::vector<float>& d) {
                         // Scale from a clipped range: the largest weight wraps to -128.
                         const auto it = std::max_element(d.begin(), d.end(), [](float a, float c) {
                             return std::fabs(a) < std::fabs(c);
                         });
                         *it = -std::copysign(128.0f * scale, *it);
                     })});
    // Rounding ties away from zero instead of to even: a real converter
    // discrepancy that only matters when a weight lands exactly on a tie.
    cases.push_back({"requant_tie_rounding", "quantisation rounding", q,
                     withConstant(q, w, [&](std::vector<float>& d) {
                         for (std::size_t i = 0; i < d.size(); ++i) d[i] = std::round(original[i] / scale) * scale;
                     })});
    // Controls: correct rewrites that must be CERTIFIED.
    IRGraph optimized = graph;
    DiagnosticEngine diagnostics;
    optimize(optimized, diagnostics);
    cases.push_back({"control_optimizer_pipeline", "control", graph, optimized});
    return cases;
}

}  // namespace

LifecycleStudySummary runLifecycleStudy(const std::string& realDirectory, std::size_t maxBalls,
                                        const ProgressCallback& progress) {
    const auto start = std::chrono::steady_clock::now();
    LifecycleStudySummary summary;
    const auto models = realModels(realDirectory);
    std::size_t index = 0;
    for (const auto& path : models) {
        const std::string name = path.stem().string();
        if (progress) progress(index++, models.size(), name);
        IRGraph graph;
        Dataset test;
        if (!loadReal(path, realDirectory, graph, test)) continue;
        const Box domain = dataEnvelope(test.rows, 0.5f);

        // 1. Mixed precision.
        summary.precision.push_back({name, searchMixedPrecision(graph, test, 0.5f, maxBalls)});

        // 2. Culprit localisation: one typo (a sign flip of the largest first-layer
        //    weight) hidden among 24 one-ulp edits that are numerically harmless.
        {
            IRGraph edited = graph;
            const IRInstruction* dense = firstDense(graph);
            const std::string w = dense->inputs[1];
            auto data = edited.constants.at(w).data;
            std::mt19937 generator(20261021);
            std::uniform_int_distribution<std::size_t> pick(0, data.size() - 1);
            for (int i = 0; i < 24; ++i) {
                float& v = data[pick(generator)];
                v = std::nextafter(v, 10.0f);
            }
            const auto largest = static_cast<std::size_t>(
                std::max_element(data.begin(), data.end(),
                                 [](float a, float b) { return std::fabs(a) < std::fabs(b); }) -
                data.begin());
            data[largest] = -data[largest];
            setConstant(edited, w, std::move(data));
            // Tolerance 1e-3: one-ulp edits move the probabilities by ~1e-7, but a
            // whole-domain proof much below that is out of reach of the float32 padding.
            TemperOptions options;
            options.tolerance = 1.0e-3f;
            const auto report = localizeCulprits(graph, edited, domain, options);
            LifecycleStudySummary::CulpritRow row;
            row.model = name;
            row.edits = report.edits;
            row.decisions = report.decisions;
            row.exact = report.culprits.size() == 1 && report.culprits[0].tensor == w &&
                        report.culprits[0].index == largest;
            row.fixed = report.afterRevert == Verdict::Certified;
            row.milliseconds = report.milliseconds;
            summary.culprits.push_back(row);
        }

        // 3. Incremental re-certification along a version chain: fp16 export,
        //    then two small bias patches of the exported model.
        {
            CertifyOptions options;
            options.tolerance = 2.0e-3f;
            options.maxBoxes = 4096;
            options.guidedWitnesses = false;
            Compression fp16;
            parseCompression("fp16", fp16);
            IRGraph version = compressModel(graph, fp16);
            const IRInstruction* dense = firstDense(graph);
            std::optional<ProofCertificate> previous;
            std::size_t patch = 0;
            for (const char* step : {"v1 fp16", "v2 bias patch", "v3 bias patch"}) {
                if (previous) {
                    auto bias = version.constants.at(dense->inputs[2]).data;
                    bias[patch++ % bias.size()] += 1.0e-5f;
                    setConstant(version, dense->inputs[2], std::move(bias));
                }
                const auto scratch = certifyIncrementally(graph, version, domain, options);
                const auto reused = certifyIncrementally(graph, version, domain, options,
                                                         previous ? &*previous : nullptr);
                LifecycleStudySummary::IncrementalRow row;
                row.model = name;
                row.step = step;
                row.verdict = reused.result.verdict;
                row.scratchBoxes = scratch.result.boxes;
                row.reusedBoxes = reused.result.boxes;
                row.scratchShare = scratch.result.certifiedFraction;
                row.reusedShare = reused.result.certifiedFraction;
                row.scratchMs = scratch.result.milliseconds;
                row.reusedMs = reused.result.milliseconds;
                row.agree = scratch.result.verdict == reused.result.verdict;
                summary.incremental.push_back(row);
                previous = reused.certificate;
            }
        }

        // 4. Bug classes.
        const auto oracle = uniformInputs(domain, 20000, 7);
        const auto random = uniformInputs(domain, 1000, 11);
        for (const auto& bug : makeBugCases(graph)) {
            LifecycleStudySummary::BugRow row;
            row.model = name;
            row.bugClass = bug.bugClass;
            row.source = bug.source;
            row.testSet = anyExposed(bug.reference, bug.candidate, test.rows, 1.0e-5f);
            row.random = anyExposed(bug.reference, bug.candidate, random, 1.0e-5f);
            const auto decision = decideTransformation(bug.reference, bug.candidate, domain);
            row.temper = decision.verdict;
            row.kind = decision.kind;
            row.evaluations = decision.evaluations;
            const bool oracleHit = anyExposed(bug.reference, bug.candidate, oracle, 1.0e-5f);
            row.expected = oracleHit || decision.verdict == Verdict::Refuted;
            summary.violations += decision.verdict == Verdict::Certified && oracleHit;
            summary.bugs.push_back(row);
        }
    }
    summary.seconds = elapsedMs(start) / 1000.0;

    std::ostringstream json;
    json << "{\n  \"study\": \"temper_lifecycle\",\n  \"author\": " << quoted(kAuthor)
         << ",\n  \"violations\": " << summary.violations << ",\n  \"mixed_precision\": [";
    for (std::size_t i = 0; i < summary.precision.size(); ++i) {
        const auto& p = summary.precision[i].plan;
        json << (i ? "," : "") << "\n    {\"model\": " << quoted(summary.precision[i].model)
             << ", \"plan\": " << quoted(p.audit.compression) << ", \"weight_size\": " << number(p.weightSize)
             << ", \"verdict\": " << quoted(verdictName(p.audit.verdict))
             << ", \"certified_bound\": " << number(p.audit.certificate.bound)
             << ", \"accuracy_after\": " << number(p.audit.accuracyAfter)
             << ", \"best_uniform\": " << quoted(p.bestUniform)
             << ", \"best_uniform_size\": " << number(p.bestUniformSize) << ", \"trials\": " << p.trials
             << ", \"ms\": " << number(p.milliseconds) << "}";
    }
    json << "\n  ],\n  \"culprits\": [";
    for (std::size_t i = 0; i < summary.culprits.size(); ++i) {
        const auto& c = summary.culprits[i];
        json << (i ? "," : "") << "\n    {\"model\": " << quoted(c.model) << ", \"edits\": " << c.edits
             << ", \"decisions\": " << c.decisions << ", \"exact\": " << (c.exact ? "true" : "false")
             << ", \"fix_certified\": " << (c.fixed ? "true" : "false") << ", \"ms\": " << number(c.milliseconds)
             << "}";
    }
    json << "\n  ],\n  \"incremental\": [";
    for (std::size_t i = 0; i < summary.incremental.size(); ++i) {
        const auto& r = summary.incremental[i];
        json << (i ? "," : "") << "\n    {\"model\": " << quoted(r.model) << ", \"step\": " << quoted(r.step)
             << ", \"verdict\": " << quoted(verdictName(r.verdict)) << ", \"scratch_boxes\": " << r.scratchBoxes
             << ", \"reused_boxes\": " << r.reusedBoxes << ", \"scratch_share\": " << number(r.scratchShare)
             << ", \"reused_share\": " << number(r.reusedShare) << ", \"scratch_ms\": " << number(r.scratchMs)
             << ", \"reused_ms\": " << number(r.reusedMs) << ", \"agree\": " << (r.agree ? "true" : "false")
             << "}";
    }
    json << "\n  ],\n  \"bug_classes\": [";
    for (std::size_t i = 0; i < summary.bugs.size(); ++i) {
        const auto& r = summary.bugs[i];
        json << (i ? "," : "") << "\n    {\"model\": " << quoted(r.model) << ", \"class\": " << quoted(r.bugClass)
             << ", \"source\": " << quoted(r.source) << ", \"observable\": " << (r.expected ? "true" : "false")
             << ", \"test_set\": " << (r.testSet ? "true" : "false")
             << ", \"random_1000\": " << (r.random ? "true" : "false")
             << ", \"temper\": " << quoted(verdictName(r.temper))
             << ", \"certificate\": " << quoted(certificateKindName(r.kind))
             << ", \"evaluations\": " << r.evaluations << "}";
    }
    json << "\n  ],\n  \"seconds\": " << number(summary.seconds) << "\n}\n";
    summary.json = json.str();
    return summary;
}

}  // namespace modelforge

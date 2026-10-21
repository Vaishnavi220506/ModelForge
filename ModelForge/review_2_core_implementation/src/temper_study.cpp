#include "temper_study.h"

#include "analysis.h"
#include "codegen.h"
#include "optimizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <sstream>

namespace modelforge {
namespace {

namespace fs = std::filesystem;

std::string number(double value) {
    if (!std::isfinite(value)) return "null";
    std::ostringstream output;
    output << std::setprecision(6) << value;
    return output.str();
}

std::string quoted(const std::string& value) { return "\"" + jsonEscapeText(value) + "\""; }

std::string countsJson(const VerdictCounts& c) {
    return "{\"refuted\":" + std::to_string(c.refuted) +
           ",\"certified_exact\":" + std::to_string(c.certifiedExact) +
           ",\"certified_bounded\":" + std::to_string(c.certifiedBounded) +
           ",\"undecided\":" + std::to_string(c.undecided) + "}";
}

void count(VerdictCounts& counts, Verdict verdict, CertificateKind kind) {
    if (verdict == Verdict::Refuted) ++counts.refuted;
    else if (verdict == Verdict::Certified && kind == CertificateKind::Exact) ++counts.certifiedExact;
    else if (verdict == Verdict::Certified) ++counts.certifiedBounded;
    else ++counts.undecided;
}

double since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

struct Outcome {
    Verdict verdict = Verdict::Undecided;
    CertificateKind kind = CertificateKind::None;
    std::size_t evaluations = 0;
    double milliseconds = 0.0;
    std::optional<std::vector<float>> witness;
};

struct Case {
    std::string name, family, kind;  // kind: fault | control | pass
    IRGraph candidate;
};

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

}  // namespace

// ------------------------------------------------------------ decisiveness

TemperStudySummary runTemperStudy(const BenchmarkConfig& config, const ProgressCallback& progress) {
    const auto start = std::chrono::steady_clock::now();
    TemperStudySummary summary;
    for (const char* name : {"guardian_apc_plus", "certifier", "temper"}) {
        StudyStrategy strategy;
        strategy.name = name;
        summary.strategies.push_back(std::move(strategy));
    }
    std::ostringstream csv;
    csv << "model,case,family,kind,observable,strategy,verdict,certificate,evaluations,ms\n";
    const auto specs = makeZooSpecs(config.models, config.seed);
    for (std::size_t m = 0; m < specs.size(); ++m) {
        const auto& spec = specs[m];
        if (progress) progress(m, specs.size(), spec.name);
        IRGraph graph;
        DiagnosticEngine diagnostics;
        if (!compileManifestText(synthesizeManifest(spec), graph, diagnostics)) continue;
        ++summary.models;
        const auto sites = analyzeActivationSites(graph);
        std::vector<Case> cases;
        for (auto& fault : makeFaults(graph, sites, spec.seed)) {
            cases.push_back({fault.name, fault.family, fault.negativeControl ? "control" : "fault",
                             std::move(fault.candidate)});
        }
        for (const auto pass : {OptimizationPass::ConstantFolding, OptimizationPass::DenseReluFusion,
                                OptimizationPass::DeadNodeRemoval}) {
            IRGraph candidate = graph;
            if (!runOptimizationPass(candidate, pass).events.empty()) {
                cases.push_back({optimizationPassName(pass), optimizationPassName(pass), "pass",
                                 std::move(candidate)});
            }
        }
        const std::size_t inputs = elementCount(graph.values.at(graph.inputName).shape);
        const Box domain = uniformBox(inputs, kDefaultInputRadius);
        const auto oracle = uniformRandomProbes(inputs, config.oracleProbes, spec.seed ^ 0xA5A5u);

        for (const Case& c : cases) {
            ++summary.cases;
            summary.faults += c.kind == "fault";
            summary.controls += c.kind == "control";
            summary.passes += c.kind == "pass";
            Outcome outcomes[3];
            {  // Guardian-APC+ alone: refutes or says nothing.
                const auto t = std::chrono::steady_clock::now();
                GuardianPlusOptions options;
                options.staticBudget = 96;
                options.adaptiveBudget = 32;
                options.seed = config.seed;
                options.tolerance = config.tolerance;
                const auto result = runGuardianPlus(graph, c.candidate, options);
                outcomes[0].evaluations = result.evaluations;
                if (result.firstDetection != 0 && result.witness) {
                    outcomes[0].verdict = Verdict::Refuted;
                    outcomes[0].witness = result.witness->values;
                }
                outcomes[0].milliseconds = since(t);
            }
            {  // Certifier alone: branch and bound with centre checks, no guidance.
                CertifyOptions options;
                options.tolerance = config.tolerance;
                options.maxBoxes = 512;
                options.maxEvaluations = 1024;
                options.guidedWitnesses = false;
                const auto result = certifyEquivalence(graph, c.candidate, domain, options);
                outcomes[1] = {result.verdict, result.kind, result.evaluations, result.milliseconds,
                               result.witness};
            }
            {  // Temper: static proof, Guardian, guided branch and bound.
                TemperOptions options;
                options.tolerance = config.tolerance;
                options.seed = config.seed;
                const auto result = decideTransformation(graph, c.candidate, domain, options);
                outcomes[2] = {result.verdict, result.kind, result.evaluations, result.milliseconds,
                               result.witness};
            }
            // Observability: the oracle, or any verified witness of any method.
            bool observable = false;
            for (const auto& probe : oracle) {
                if (exposes(graph, c.candidate, probe.values, config.tolerance)) {
                    observable = true;
                    break;
                }
            }
            const bool oracleSaw = observable;
            bool temperOnly = false;
            for (int s = 0; s < 3; ++s) {
                if (outcomes[s].witness &&
                    exposes(graph, c.candidate, *outcomes[s].witness, config.tolerance)) {
                    if (!observable && s == 2) temperOnly = true;
                    observable = true;
                }
            }
            // Soundness gate: a certificate may never meet an input whose
            // output difference exceeds the tolerance.
            bool exceeds = false;
            const bool anyCertified = std::any_of(std::begin(outcomes), std::end(outcomes),
                                                  [](const Outcome& o) {
                                                      return o.verdict == Verdict::Certified;
                                                  });
            if (anyCertified) {
                for (const auto& probe : oracle) {
                    float difference = 0.0f;
                    exposes(graph, c.candidate, probe.values, config.tolerance, &difference);
                    if (difference > config.tolerance) {
                        exceeds = true;
                        break;
                    }
                }
                for (const Outcome& o : outcomes) {
                    float difference = 0.0f;
                    if (!exceeds && o.witness) {
                        exposes(graph, c.candidate, *o.witness, config.tolerance, &difference);
                        exceeds = difference > config.tolerance;
                    }
                }
            }
            if (c.kind == "fault") {
                summary.observable += observable;
                summary.hidden += !observable;
                if (observable && outcomes[0].verdict != Verdict::Refuted) {
                    ++summary.guardianMissed;
                    summary.guardianMissedRefuted += outcomes[2].verdict == Verdict::Refuted;
                }
                summary.oracleMissedRefuted += !oracleSaw && temperOnly;
            }
            for (int s = 0; s < 3; ++s) {
                StudyStrategy& strategy = summary.strategies[s];
                const Outcome& o = outcomes[s];
                VerdictCounts& bucket = c.kind != "fault" ? strategy.benign
                                        : observable      ? strategy.observable
                                                          : strategy.hidden;
                count(bucket, o.verdict, o.kind);
                count(strategy.families[c.family], o.verdict, o.kind);
                strategy.evaluations += o.evaluations;
                strategy.milliseconds += o.milliseconds;
                if (o.verdict == Verdict::Certified) strategy.violations += exceeds;
                csv << spec.name << ',' << c.name << ',' << c.family << ',' << c.kind << ','
                    << observable << ',' << strategy.name << ',' << verdictName(o.verdict) << ','
                    << certificateKindName(o.kind) << ',' << o.evaluations << ','
                    << std::setprecision(4) << o.milliseconds << '\n';
            }
        }
    }
    if (progress) progress(specs.size(), specs.size(), "done");
    summary.seconds = since(start) / 1000.0;
    summary.casesCsv = csv.str();

    std::ostringstream json;
    json << "{\n  \"study\": \"temper_decisiveness\",\n  \"author\": " << quoted(kAuthor)
         << ",\n  \"seed\": " << config.seed << ",\n  \"models\": " << summary.models
         << ",\n  \"cases\": " << summary.cases << ",\n  \"faults\": " << summary.faults
         << ",\n  \"controls\": " << summary.controls << ",\n  \"passes\": " << summary.passes
         << ",\n  \"observable\": " << summary.observable << ",\n  \"hidden\": " << summary.hidden
         << ",\n  \"guardian_missed\": " << summary.guardianMissed
         << ",\n  \"guardian_missed_refuted_by_temper\": " << summary.guardianMissedRefuted
         << ",\n  \"found_only_by_temper\": " << summary.oracleMissedRefuted
         << ",\n  \"oracle_probes\": " << config.oracleProbes
         << ",\n  \"seconds\": " << number(summary.seconds) << ",\n  \"strategies\": [";
    for (std::size_t s = 0; s < summary.strategies.size(); ++s) {
        const auto& st = summary.strategies[s];
        json << (s ? "," : "") << "\n    {\"name\": " << quoted(st.name)
             << ", \"observable\": " << countsJson(st.observable)
             << ", \"hidden\": " << countsJson(st.hidden) << ", \"benign\": " << countsJson(st.benign)
             << ", \"evaluations\": " << st.evaluations
             << ", \"milliseconds\": " << number(st.milliseconds)
             << ", \"violations\": " << st.violations << ", \"families\": {";
        std::size_t f = 0;
        for (const auto& [family, counts] : st.families) {
            json << (f++ ? ", " : "") << quoted(family) << ": " << countsJson(counts);
        }
        json << "}}";
    }
    json << "\n  ]\n}\n";
    summary.json = json.str();
    return summary;
}

// ------------------------------------------------------------ compression

CompressionStudySummary runCompressionStudy(const std::string& realDirectory, std::size_t maxBalls,
                                            const ProgressCallback& progress) {
    const auto start = std::chrono::steady_clock::now();
    CompressionStudySummary summary;
    const auto models = realModels(realDirectory);
    std::ostringstream json;
    json << "{\n  \"study\": \"temper_compression\",\n  \"author\": " << quoted(kAuthor)
         << ",\n  \"realism_radius\": 0.5,\n  \"max_balls\": " << maxBalls << ",\n  \"cases\": [";
    std::size_t index = 0;
    for (const auto& path : models) {
        if (progress) progress(index++, models.size(), path.stem().string());
        IRGraph graph;
        Dataset test;
        if (!loadReal(path, realDirectory, graph, test)) continue;
        ++summary.models;
        const std::string name = path.stem().string();
        double bestCertified = 2.0, bestEmpirical = 2.0;
        for (const Compression& compression : standardCompressions()) {
            CompressionAudit audit = auditCompression(graph, compression, test, 0.5f, maxBalls);
            ++summary.cases;
            count(summary.verdicts, audit.verdict, CertificateKind::Bounded);
            const double found = audit.search.witness.empty() ? 0.0 : audit.search.severity;
            summary.bracketViolations += found > audit.certificate.bound + 1e-9;
            if (audit.verdict == Verdict::Certified && audit.weightSize < bestCertified) {
                bestCertified = audit.weightSize;
                summary.certifiedChoice[name] = compression.name;
            }
            if (found < 0.2 && audit.weightSize < bestEmpirical) {
                bestEmpirical = audit.weightSize;
                summary.empiricalChoice[name] = compression.name;
            }
            json << (summary.cases > 1 ? "," : "") << "\n    {\"model\": " << quoted(name)
                 << ", \"compression\": " << quoted(compression.name)
                 << ", \"accuracy_before\": " << number(audit.accuracyBefore)
                 << ", \"accuracy_after\": " << number(audit.accuracyAfter)
                 << ", \"test_flips\": " << audit.testFlips
                 << ", \"found_severity\": " << number(found)
                 << ", \"certified_severity\": " << number(audit.certificate.bound)
                 << ", \"max_deviation\": " << number(audit.certificate.maxDeviation)
                 << ", \"balls\": " << audit.certificate.balls
                 << ", \"agreeing_balls\": " << audit.certificate.agreeingBalls
                 << ", \"refuted_balls\": " << audit.search.refutedBalls
                 << ", \"verdict\": " << quoted(verdictName(audit.verdict))
                 << ", \"milliseconds\": " << number(audit.milliseconds) << "}";
            summary.rows.push_back({name, compression.name, std::move(audit)});
        }
        summary.certifiedRecommendations += summary.certifiedChoice.count(name);
        summary.agreeingRecommendations +=
            summary.certifiedChoice.count(name) && summary.empiricalChoice.count(name) &&
            summary.certifiedChoice[name] == summary.empiricalChoice[name];
    }
    if (progress) progress(models.size(), models.size(), "done");
    summary.seconds = since(start) / 1000.0;
    json << "\n  ],\n  \"refuted\": " << summary.verdicts.refuted
         << ",\n  \"certified\": " << summary.verdicts.certifiedBounded
         << ",\n  \"undecided\": " << summary.verdicts.undecided
         << ",\n  \"bracket_violations\": " << summary.bracketViolations
         << ",\n  \"recommendations\": {";
    std::size_t r = 0;
    for (const auto& [model, choice] : summary.empiricalChoice) {
        const auto certified = summary.certifiedChoice.find(model);
        json << (r++ ? ", " : "") << quoted(model) << ": {\"certified\": "
             << quoted(certified == summary.certifiedChoice.end() ? "" : certified->second)
             << ", \"empirical\": " << quoted(choice) << "}";
    }
    json << "},\n  \"seconds\": " << number(summary.seconds) << "\n}\n";
    summary.json = json.str();
    return summary;
}

// ------------------------------------------------------------ binary

BinaryStudySummary runBinaryStudy(const std::string& realDirectory, const std::string& workDirectory,
                                  const ProgressCallback& progress) {
    BinaryStudySummary summary;
    const auto models = realModels(realDirectory);
    std::ostringstream json;
    json << "{\n  \"study\": \"temper_binary\",\n  \"author\": " << quoted(kAuthor)
         << ",\n  \"models\": [";
    std::size_t index = 0;
    for (const auto& path : models) {
        if (progress) progress(index++, models.size(), path.stem().string());
        IRGraph graph;
        Dataset test;
        if (!loadReal(path, realDirectory, graph, test)) continue;
        DiagnosticEngine diagnostics;
        optimize(graph, diagnostics);
        const std::string directory = (fs::path(workDirectory) / path.stem()).string();
        if (!generateCpp(graph, directory, diagnostics)) continue;
        std::vector<std::vector<float>> inputs = test.rows;
        for (const auto& probe : generateGuardianProbes(graph)) inputs.push_back(probe.values);
        BinaryAuditReport report = auditGeneratedBinary(graph, directory, inputs, defaultAuditFlags());
        summary.available = report.available;
        summary.compiler = report.compiler;
        if (!report.available) break;
        json << (summary.rows.empty() ? "" : ",") << "\n    {\"model\": "
             << quoted(path.stem().string()) << ", \"results\": [";
        for (std::size_t i = 0; i < report.results.size(); ++i) {
            const auto& r = report.results[i];
            json << (i ? ", " : "") << "{\"flags\": " << quoted(r.flags)
                 << ", \"built\": " << (r.built ? "true" : "false") << ", \"rows\": " << r.rows
                 << ", \"prediction_mismatches\": " << r.predictionMismatches
                 << ", \"max_abs_error\": " << number(r.maximumAbsoluteError)
                 << ", \"error\": " << quoted(r.error) << "}";
        }
        json << "]}";
        summary.rows.push_back({path.stem().string(), std::move(report)});
    }
    if (progress) progress(models.size(), models.size(), "done");
    json << "\n  ],\n  \"compiler\": " << quoted(summary.compiler)
         << ",\n  \"available\": " << (summary.available ? "true" : "false") << "\n}\n";
    summary.json = json.str();
    return summary;
}

}  // namespace modelforge

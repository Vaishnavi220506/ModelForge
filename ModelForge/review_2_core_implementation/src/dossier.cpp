#include "dossier.h"

#include "analysis.h"
#include "loader.h"
#include "optimizer.h"
#include "research.h"
#include "validator.h"
#include "verifier.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace modelforge {
namespace {

std::string number(double value) {
    if (!std::isfinite(value)) return "null";
    std::ostringstream output;
    output << std::setprecision(7) << value;
    return output.str();
}

std::string quoted(const std::string& value) { return "\"" + jsonEscapeText(value) + "\""; }

std::string floats(const std::vector<float>& values) {
    std::string result = "[";
    for (std::size_t i = 0; i < values.size(); ++i) result += (i ? "," : "") + number(values[i]);
    return result + "]";
}

std::string decisionJson(const TemperDecision& d) {
    std::string json = "{\"verdict\":" + quoted(verdictName(d.verdict)) +
                       ",\"certificate\":" + quoted(certificateKindName(d.kind)) +
                       ",\"decided_by\":" + quoted(d.decidedBy) + ",\"bound\":" + number(d.bound) +
                       ",\"certified_fraction\":" + number(d.certifiedFraction) +
                       ",\"evaluations\":" + std::to_string(d.evaluations) +
                       ",\"boxes\":" + std::to_string(d.boxes) +
                       ",\"milliseconds\":" + number(d.milliseconds);
    if (d.witness) {
        json += ",\"witness\":" + floats(*d.witness) + ",\"original_output\":" +
                floats(d.originalOutput) + ",\"candidate_output\":" + floats(d.candidateOutput);
        if (d.divergence.found) json += ",\"first_divergent_node\":" + quoted(d.divergence.node);
    }
    return json + "}";
}

}  // namespace

bool buildDossier(const DossierOptions& options, Dossier& dossier, DiagnosticEngine& diagnostics,
                  const DossierProgress& progress) {
    const auto start = std::chrono::steady_clock::now();
    const auto say = [&](const std::string& stage) {
        if (progress) progress(stage);
    };

    // ---------------------------------------------------------------- compile
    say("compile");
    ModelGraph model;
    SymbolTable symbols;
    if (!loadModel(options.modelPath, model, diagnostics) || !validate(model, diagnostics, symbols)) {
        return false;
    }
    auto ir = buildIR(model, diagnostics);
    if (!ir) return false;
    IRGraph graph = std::move(*ir);
    dossier.model = graph.name;
    dossier.inputs = elementCount(graph.values.at(graph.inputName).shape);
    dossier.outputs = elementCount(graph.values.at(graph.outputName).shape);

    Dataset test;
    bool haveData = false;
    if (!options.dataPath.empty()) {
        std::string error;
        if (!loadDataset(options.dataPath, dossier.inputs, test, error)) {
            diagnostics.error("data", error);
            return false;
        }
        haveData = !test.rows.empty();
    }
    Dataset domainRows = test;
    if (!options.domainPath.empty()) {
        std::string error;
        domainRows = {};
        if (!loadDataset(options.domainPath, dossier.inputs, domainRows, error)) {
            diagnostics.error("domain", error);
            return false;
        }
    }
    if (!domainRows.rows.empty()) {
        dossier.domain = dataEnvelope(domainRows.rows, options.domainMargin);
        std::ostringstream source;
        source << "data envelope of " << domainRows.rows.size() << " rows +- "
               << options.domainMargin;
        dossier.domainSource = source.str();
    } else {
        dossier.domain = uniformBox(dossier.inputs, kDefaultInputRadius);
        dossier.domainSource = "default box [-10, 10] per input";
    }

    // ------------------------------------------------- domain specialisation
    if (options.specialize) {
        say("specialise");
        dossier.specialization = specializeToDomain(graph, dossier.domain);
        if (dossier.specialization.deadUnits > 0 &&
            dossier.specialization.certificate.verdict == Verdict::Certified) {
            graph = dossier.specialization.specialized;
            dossier.specialized = true;
        }
    }

    // Compression is applied to the unfused model (what gets written back as
    // .mforge); every analysis below treats fused and unfused graphs alike.
    const IRGraph unfused = graph;

    // ------------------------------------------------- optimisation passes
    TemperOptions temper;
    for (const auto pass : {OptimizationPass::ConstantFolding, OptimizationPass::DenseReluFusion,
                            OptimizationPass::DeadNodeRemoval}) {
        say(std::string("pass ") + optimizationPassName(pass));
        PassVerdict verdict;
        verdict.pass = optimizationPassName(pass);
        IRGraph candidate = graph;
        const OptimizationReport report = runOptimizationPass(candidate, pass);
        verdict.events = report.events;
        verdict.changed = !report.events.empty();
        if (!verdict.changed) {
            verdict.kept = true;
            verdict.decision.verdict = Verdict::Certified;
            verdict.decision.kind = CertificateKind::Exact;
            verdict.decision.decidedBy = "unchanged";
        } else {
            verdict.decision = decideTransformation(graph, candidate, dossier.domain, temper);
            verdict.kept = verdict.decision.verdict == Verdict::Certified ||
                           (!options.strict && verdict.decision.verdict == Verdict::Undecided);
        }
        if (verdict.kept) graph = std::move(candidate);
        if (verdict.decision.verdict == Verdict::Refuted) {
            dossier.notes.push_back("pass " + verdict.pass + " was REFUTED and rolled back");
        } else if (verdict.decision.verdict == Verdict::Undecided) {
            dossier.notes.push_back("pass " + verdict.pass + " is UNDECIDED (" +
                                    (verdict.kept ? "kept on Guardian evidence" : "dropped: strict") +
                                    ")");
        }
        dossier.passes.push_back(std::move(verdict));
    }
    dossier.fingerprint = fingerprintHex(graphFingerprint(graph));
    dossier.parameters = parameterCount(graph);
    dossier.multiplyAdds = multiplyAddCount(graph);

    // ------------------------------------------------- compression audit
    if (options.compression && haveData) {
        std::vector<CompressionAudit> audits;
        double bestCertified = 2.0, bestEmpirical = 2.0;
        for (const Compression& compression : standardCompressions()) {
            say("compress " + compression.name);
            CompressionAudit audit =
                auditCompression(unfused, compression, test, options.realism, options.maxBalls);
            if (audit.verdict == Verdict::Certified && audit.weightSize < bestCertified) {
                bestCertified = audit.weightSize;
                dossier.recommendedCertified = audit.compression;
            }
            const bool empiricallyLow = audit.search.witness.empty() || audit.search.severity < 0.2;
            if (empiricallyLow && audit.weightSize < bestEmpirical) {
                bestEmpirical = audit.weightSize;
                dossier.recommendedEmpirical = audit.compression;
            }
            audits.push_back(std::move(audit));
        }
        dossier.compressions = std::move(audits);
        if (!dossier.recommendedCertified.empty()) {
            const std::filesystem::path out = options.outputDirectory;
            std::filesystem::create_directories(out);
            Compression chosen;
            parseCompression(dossier.recommendedCertified, chosen);
            std::ofstream(out / (graph.name + "_" + chosen.name + ".mforge"))
                << writeManifestText(compressModel(unfused, chosen));
        }
    }

    // ------------------------------------------------- code generation
    say("generate C++");
    DeploymentEnvelope envelope;
    envelope.lower = dossier.domain.lower;
    envelope.upper = dossier.domain.upper;
    envelope.fingerprint = dossier.fingerprint;
    std::size_t certifiedPasses = 0;
    for (const auto& pass : dossier.passes) {
        certifiedPasses += pass.changed && pass.decision.verdict == Verdict::Certified ? 1 : 0;
    }
    envelope.summary = std::to_string(certifiedPasses) + " rewrites certified on " + dossier.domainSource;
    const std::filesystem::path generated = std::filesystem::path(options.outputDirectory) / "cpp";
    dossier.generatedDirectory = generated.string();
    if (!generateCpp(graph, generated.string(), diagnostics, &envelope)) return false;

    // ------------------------------------------------- binary audit
    if (options.binaryAudit) {
        say("audit binary");
        std::vector<std::vector<float>> inputs;
        for (const auto& row : test.rows) inputs.push_back(row);
        for (const auto& probe : generateGuardianProbes(graph)) {
            if (inputs.size() >= test.rows.size() + 128) break;
            std::vector<float> clipped = probe.values;
            for (std::size_t i = 0; i < clipped.size(); ++i) {
                clipped[i] = std::clamp(clipped[i], dossier.domain.lower[i], dossier.domain.upper[i]);
            }
            inputs.push_back(std::move(clipped));
        }
        dossier.binary = auditGeneratedBinary(graph, generated.string(), inputs, defaultAuditFlags());
    }

    // ------------------------------------------------- ship decision
    for (const auto& pass : dossier.passes) {
        if (pass.kept && pass.decision.verdict != Verdict::Certified) {
            dossier.blockers.push_back("pass " + pass.pass + " shipped without a certificate");
        }
    }
    if (options.binaryAudit && dossier.binary.available) {
        for (const auto& result : dossier.binary.results) {
            const bool standard = result.flags.find("fast-math") == std::string::npos &&
                                  result.flags.find("fp-contract") == std::string::npos;
            if (!result.built || !result.error.empty()) {
                dossier.blockers.push_back("binary " + result.flags + ": " + result.error);
            } else if (standard && (result.predictionMismatches > 0 || result.maximumAbsoluteError > 0)) {
                dossier.blockers.push_back("binary " + result.flags +
                                           " does not reproduce the certified model bit-for-bit");
            } else if (!standard && result.predictionMismatches > 0) {
                dossier.notes.push_back("'" + result.flags + "' changes " +
                                        std::to_string(result.predictionMismatches) +
                                        " predictions: do not build with it");
            } else if (!standard && result.maximumAbsoluteError > 0) {
                std::ostringstream text;
                text << "'" << result.flags << "': scores move by up to " << std::setprecision(2)
                     << result.maximumAbsoluteError << ", predictions unchanged";
                dossier.notes.push_back(text.str());
            }
        }
    } else if (options.binaryAudit) {
        dossier.notes.push_back("no C++ compiler found (set CXX): binary audit skipped");
    }
    dossier.ship = dossier.blockers.empty();
    dossier.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    // ------------------------------------------------- JSON
    std::ostringstream json;
    json << "{\n  \"tool\": \"Temper (ModelForge)\",\n  \"author\": " << quoted(kAuthor)
         << ",\n  \"model\": " << quoted(dossier.model) << ",\n  \"source\": "
         << quoted(options.modelPath) << ",\n  \"fingerprint\": " << quoted(dossier.fingerprint)
         << ",\n  \"domain\": {\"source\": " << quoted(dossier.domainSource)
         << ", \"lower\": " << floats(dossier.domain.lower)
         << ", \"upper\": " << floats(dossier.domain.upper) << "},\n  \"passes\": [";
    for (std::size_t i = 0; i < dossier.passes.size(); ++i) {
        const auto& pass = dossier.passes[i];
        json << (i ? ",\n    " : "\n    ") << "{\"pass\": " << quoted(pass.pass)
             << ", \"changed\": " << (pass.changed ? "true" : "false")
             << ", \"kept\": " << (pass.kept ? "true" : "false")
             << ", \"decision\": " << decisionJson(pass.decision) << "}";
    }
    const auto& s = dossier.specialization;
    json << "\n  ],\n  \"specialization\": {\"applied\": " << (dossier.specialized ? "true" : "false")
         << ", \"hidden_units\": " << s.hiddenUnits << ", \"dead_units_removed\": " << s.deadUnits
         << ", \"always_active_units\": " << s.alwaysActiveUnits
         << ", \"parameters_before\": " << s.parametersBefore
         << ", \"parameters_after\": " << s.parametersAfter
         << ", \"certificate\": " << quoted(verdictName(s.certificate.verdict)) << "},\n"
         << "  \"compressions\": [";
    for (std::size_t i = 0; i < dossier.compressions.size(); ++i) {
        const auto& a = dossier.compressions[i];
        json << (i ? ",\n    " : "\n    ") << "{\"compression\": " << quoted(a.compression)
             << ", \"weight_size\": " << number(a.weightSize)
             << ", \"accuracy_before\": " << number(a.accuracyBefore)
             << ", \"accuracy_after\": " << number(a.accuracyAfter)
             << ", \"test_flips\": " << a.testFlips
             << ", \"found_severity\": " << number(a.search.witness.empty() ? 0.0 : a.search.severity)
             << ", \"certified_severity\": " << number(a.certificate.bound)
             << ", \"agreeing_balls\": " << a.certificate.agreeingBalls
             << ", \"balls\": " << a.certificate.balls
             << ", \"verdict\": " << quoted(verdictName(a.verdict))
             << ", \"witness\": " << floats(a.search.witness) << "}";
    }
    json << "\n  ],\n  \"recommended_certified\": " << quoted(dossier.recommendedCertified)
         << ",\n  \"recommended_empirical\": " << quoted(dossier.recommendedEmpirical)
         << ",\n  \"binary_audit\": {\"compiler\": " << quoted(dossier.binary.compiler)
         << ", \"available\": " << (dossier.binary.available ? "true" : "false") << ", \"results\": [";
    for (std::size_t i = 0; i < dossier.binary.results.size(); ++i) {
        const auto& r = dossier.binary.results[i];
        json << (i ? ", " : "") << "{\"flags\": " << quoted(r.flags)
             << ", \"built\": " << (r.built ? "true" : "false") << ", \"rows\": " << r.rows
             << ", \"prediction_mismatches\": " << r.predictionMismatches
             << ", \"max_abs_error\": " << number(r.maximumAbsoluteError)
             << ", \"error\": " << quoted(r.error) << "}";
    }
    json << "]},\n  \"parameters\": " << dossier.parameters
         << ",\n  \"multiply_adds\": " << dossier.multiplyAdds
         << ",\n  \"ship\": " << (dossier.ship ? "true" : "false") << ",\n  \"blockers\": [";
    for (std::size_t i = 0; i < dossier.blockers.size(); ++i) {
        json << (i ? ", " : "") << quoted(dossier.blockers[i]);
    }
    json << "],\n  \"notes\": [";
    for (std::size_t i = 0; i < dossier.notes.size(); ++i) {
        json << (i ? ", " : "") << quoted(dossier.notes[i]);
    }
    json << "],\n  \"seconds\": " << number(dossier.seconds) << "\n}\n";
    dossier.json = json.str();
    std::filesystem::create_directories(options.outputDirectory);
    std::ofstream(std::filesystem::path(options.outputDirectory) / "dossier.json") << dossier.json;
    return true;
}

}  // namespace modelforge

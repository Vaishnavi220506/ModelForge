// temper: falsify-or-certify assurance from trained model to shipped binary.
//
//   temper model.mforge                         compile, decide every rewrite, emit C++, audit binary
//   temper model.mforge --data test.csv         + compression audit and data-derived domain
//   temper model.mforge --compare other.mforge  is `other` equivalent? CERTIFIED / REFUTED / UNDECIDED
//                                               (REFUTED edits: the culprit weights, by delta debugging)
//   temper model.mforge --data test.csv --repair  smallest mixed-precision model CERTIFIED low risk
//
// Exit codes: 0 ship / certified, 3 blocked / refuted, 4 undecided, 1 error.
// ModelForge is the compiler underneath; Temper decides whether its output can
// be trusted.

#include "dossier.h"
#include "dossier_view.h"
#include "lifecycle.h"
#include "loader.h"
#include "term.h"
#include "temper.h"
#include "validator.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <iostream>
#include <string>
#include <vector>

using namespace term;
namespace mf = modelforge;

namespace {

constexpr std::size_t kWidth = view::kWidth;
using view::vec;

void usage() {
    std::cout
        << "temper - falsify-or-certify assurance for on-device neural networks (by Vaishnavi)\n\n"
        << "Usage:\n"
        << "  temper <model.mforge> [--data test.csv] [--domain rows.csv] [--out dir]\n"
        << "  temper <model.mforge> --compare <other.mforge> [--domain rows.csv]\n"
        << "         [--save-cert proof.cert] [--reuse-cert previous.cert]\n"
        << "  temper <model.mforge> --data test.csv --repair [--out dir]\n\n"
        << "Options:\n"
        << "  --data <csv>     labelled test rows: compression audit, realistic region, domain\n"
        << "  --domain <csv>   rows whose envelope (+- --margin) is the deployment domain\n"
        << "  --margin <z>     envelope margin (default 0.5)\n"
        << "  --out <dir>      output directory (default temper_out)\n"
        << "  --strict         keep only CERTIFIED rewrites\n"
        << "  --repair         per-layer int4/int8/fp16/fp32 search: smallest CERTIFIED model\n"
        << "  --save-cert <f>  --compare: write a reusable proof certificate\n"
        << "  --reuse-cert <f> --compare: start from an earlier version's certificate\n"
        << "  --balls <n>      audit at most n test rows per compression (default all)\n"
        << "  --no-compress    skip the compression audit\n"
        << "  --no-binary      skip compiling and auditing the generated program\n"
        << "  --ascii          ASCII-only output;  --no-color  plain text\n\n"
        << "Exit codes: 0 ship/certified, 3 blocked/refuted, 4 undecided, 1 error.\n";
}

std::optional<mf::IRGraph> compile(const std::string& path, mf::DiagnosticEngine& diagnostics) {
    mf::ModelGraph model;
    mf::SymbolTable symbols;
    if (!mf::loadModel(path, model, diagnostics) || !mf::validate(model, diagnostics, symbols)) {
        return std::nullopt;
    }
    return mf::buildIR(model, diagnostics);
}

void certificateBox(const mf::IRGraph& a, const mf::IRGraph& b, const mf::Box& domain,
                    const std::string& savePath, const std::string& reusePath) {
    mf::ProofCertificate previous;
    bool havePrevious = false;
    if (!reusePath.empty()) {
        std::string error;
        havePrevious = mf::loadCertificate(reusePath, previous, error);
        if (!havePrevious) std::cerr << error << "\n";
    }
    mf::CertifyOptions options;
    const auto run = mf::certifyIncrementally(a, b, domain, options, havePrevious ? &previous : nullptr);
    std::vector<std::string> lines = {
        verdictBadge(mf::verdictName(run.result.verdict)) + "  " + std::to_string(run.result.boxes) +
            muted() + " boxes, " + reset() + std::to_string(run.result.evaluations) + muted() +
            " evaluations, " + reset() + fixed(run.result.milliseconds, 1) + " ms"};
    if (run.carriedWitness) {
        lines.push_back("the previous version's counterexample still breaks this one");
    } else if (run.reused) {
        lines.push_back("started from " + std::to_string(run.seedLeaves) + " leaves of " + reusePath +
                        muted() + " (every leaf re-verified)" + reset());
    } else if (havePrevious) {
        lines.push_back(muted() + std::string("certificate not reusable: different domain") + reset());
    }
    if (!savePath.empty()) {
        lines.push_back((mf::saveCertificate(savePath, run.certificate) ? "saved " : "could not save ") +
                        savePath + muted() + " (" + std::to_string(run.certificate.partition.size()) +
                        " leaves)" + reset());
    }
    box("Proof certificate", lines, kWidth);
}

void culpritBox(const mf::IRGraph& a, const mf::IRGraph& b, const mf::Box& domain) {
    const auto report = mf::localizeCulprits(a, b, domain);
    if (!report.applicable || report.culprits.empty()) return;
    std::vector<std::string> lines = {
        std::to_string(report.edits) + muted() + " weights differ; delta debugging isolates " + reset() +
        bold() + std::to_string(report.culprits.size()) + reset() + muted() + " that alone break the model:" +
        reset()};
    for (std::size_t i = 0; i < report.culprits.size() && i < 8; ++i) {
        const auto& c = report.culprits[i];
        lines.push_back("  " + c.tensor + "[" + std::to_string(c.index) + "]  " + fixed(c.before, 6) + "  ->  " +
                        bold() + fixed(c.after, 6) + reset());
    }
    if (report.culprits.size() > 8) lines.push_back("  ...");
    lines.push_back("revert them: " + verdictBadge(mf::verdictName(report.afterRevert)) + muted() + "  (" +
                    std::to_string(report.decisions) + " decisions, " + fixed(report.milliseconds, 0) + " ms)" +
                    reset());
    box("Culprit", lines, kWidth, bad());
}

int repairCommand(const mf::DossierOptions& options) {
    mf::DiagnosticEngine diagnostics;
    const auto graph = compile(options.modelPath, diagnostics);
    if (!graph) {
        diagnostics.print(std::cerr);
        return 1;
    }
    mf::Dataset test;
    std::string error;
    if (options.dataPath.empty() ||
        !mf::loadDataset(options.dataPath, mf::elementCount(graph->values.at(graph->inputName).shape), test,
                         error)) {
        std::cerr << (options.dataPath.empty() ? "--repair needs --data test.csv" : error) << "\n";
        return 1;
    }
    rule("Smallest model that is provably low risk", kWidth);
    std::cout << "  " << muted() << "searching per-layer precision (int4 / int8 / fp16 / fp32)..." << reset()
              << std::flush;
    const auto plan = mf::searchMixedPrecision(*graph, test, 0.5f, options.maxBalls);
    std::cout << "\r" << std::string(70, ' ') << "\r";
    std::vector<std::string> lines;
    std::map<std::string, int> bits;
    for (const auto& layer : plan.layers) {
        bits[layer.weight] = layer.bits;
        lines.push_back(padRight(layer.weight, 12) + padRight(std::to_string(layer.parameters) + " weights", 16) +
                        bold() + (layer.bits == 32 ? std::string("fp32") : layer.bits == 16 ? "fp16"
                                                                         : "int" + std::to_string(layer.bits)) +
                        reset());
    }
    lines.push_back("");
    lines.push_back(verdictBadge(mf::verdictName(plan.audit.verdict)) + "  weights kept " + bold() +
                    percent(plan.weightSize) + reset() + muted() + "  (best uniform: " + plan.bestUniform + ", " +
                    percent(plan.bestUniformSize) + ")" + reset());
    lines.push_back("worst flip margin found " + fixed(plan.audit.search.witness.empty() ? 0.0 : plan.audit.search.severity, 2) +
                    "  <=  proven " + fixed(plan.audit.certificate.bound, 2) + muted() + "   accuracy " + reset() +
                    percent(plan.audit.accuracyBefore) + " -> " + percent(plan.audit.accuracyAfter));
    const std::filesystem::path out = options.outputDirectory;
    std::filesystem::create_directories(out);
    const auto path = out / (graph->name + "_mixed.mforge");
    std::ofstream(path) << mf::writeManifestText(mf::compressLayers(*graph, bits));
    lines.push_back(muted() + std::string("written ") + reset() + path.string());
    box("Certified repair", lines, kWidth, plan.audit.verdict == mf::Verdict::Certified ? good() : warn());
    return plan.audit.verdict == mf::Verdict::Certified ? 0 : 4;
}

int compareCommand(const std::string& reference, const std::string& other,
                   const std::string& domainPath, float margin, const std::string& savePath,
                   const std::string& reusePath) {
    mf::DiagnosticEngine diagnostics;
    const auto a = compile(reference, diagnostics);
    const auto b = compile(other, diagnostics);
    if (!a || !b) {
        diagnostics.print(std::cerr);
        return 1;
    }
    const std::size_t inputs = mf::elementCount(a->values.at(a->inputName).shape);
    if (inputs != mf::elementCount(b->values.at(b->inputName).shape)) {
        std::cerr << "The two models take inputs of different sizes\n";
        return 1;
    }
    mf::Box domain = mf::uniformBox(inputs, mf::kDefaultInputRadius);
    std::string source = "default box [-10, 10]";
    if (!domainPath.empty()) {
        mf::Dataset rows;
        std::string error;
        if (!mf::loadDataset(domainPath, inputs, rows, error)) {
            std::cerr << error << "\n";
            return 1;
        }
        domain = mf::dataEnvelope(rows.rows, margin);
        source = "envelope of " + domainPath;
    }
    rule("Is the candidate equivalent to the reference?", kWidth);
    std::cout << "  reference  " << reference << "\n  candidate  " << other << "\n  domain     "
              << source << "\n\n";
    const auto d = mf::decideTransformation(*a, *b, domain);
    std::vector<std::string> lines = {
        verdictBadge(mf::verdictName(d.verdict)) + "  " +
            (d.verdict == mf::Verdict::Certified
                 ? std::string(d.kind == mf::CertificateKind::Exact
                                   ? "bit-identical outputs for every input in the domain"
                                   : "outputs within " + fixed(d.bound * 1e6, 3) + "e-6 everywhere in the domain")
             : d.verdict == mf::Verdict::Refuted ? std::string("the models behave differently")
                                                 : "no difference found, no proof either (" +
                                                       percent(d.certifiedFraction) + " of the domain proven)"),
        muted() + "decided by " + reset() + d.decidedBy + muted() + " after " + reset() +
            std::to_string(d.evaluations) + muted() + " evaluations, " + reset() +
            std::to_string(d.boxes) + muted() + " analysed boxes, " + reset() +
            fixed(d.milliseconds, 1) + " ms"};
    if (d.witness) {
        lines.push_back("input      " + vec(*d.witness, 8));
        lines.push_back("reference  " + vec(d.originalOutput, 8, 6));
        lines.push_back("candidate  " + vec(d.candidateOutput, 8, 6));
        if (d.divergence.found) lines.push_back("first difference at node " + bold() + d.divergence.node + reset());
    }
    box("Verdict", lines, kWidth,
        d.verdict == mf::Verdict::Certified ? good() : d.verdict == mf::Verdict::Refuted ? bad() : warn());
    if (d.verdict == mf::Verdict::Refuted) culpritBox(*a, *b, domain);
    if (!savePath.empty() || !reusePath.empty()) certificateBox(*a, *b, domain, savePath, reusePath);
    return d.verdict == mf::Verdict::Certified ? 0 : d.verdict == mf::Verdict::Refuted ? 3 : 4;
}

}  // namespace

int main(int argc, char* argv[]) {
    mf::DossierOptions options;
    std::string compare, saveCert, reuseCert;
    bool ascii = false, noColor = false, repair = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << a << " needs a value\n";
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--help" || a == "-h") {
            usage();
            return 0;
        } else if (a == "--data") {
            options.dataPath = value();
        } else if (a == "--domain") {
            options.domainPath = value();
        } else if (a == "--margin") {
            options.domainMargin = std::stof(value());
        } else if (a == "--out") {
            options.outputDirectory = value();
        } else if (a == "--compare") {
            compare = value();
        } else if (a == "--balls") {
            options.maxBalls = static_cast<std::size_t>(std::stoul(value()));
        } else if (a == "--save-cert") {
            saveCert = value();
        } else if (a == "--reuse-cert") {
            reuseCert = value();
        } else if (a == "--repair") {
            repair = true;
        } else if (a == "--strict") {
            options.strict = true;
        } else if (a == "--no-compress") {
            options.compression = false;
        } else if (a == "--no-binary") {
            options.binaryAudit = false;
        } else if (a == "--ascii") {
            ascii = true;
        } else if (a == "--no-color") {
            noColor = true;
        } else if (!a.empty() && a[0] == '-') {
            std::cerr << "Unknown option " << a << "\n";
            return 1;
        } else {
            options.modelPath = a;
        }
    }
    init(ascii, noColor);
    if (options.modelPath.empty()) {
        usage();
        return 1;
    }
    temperBanner("falsify-or-certify, from trained model to shipped binary  " +
                 std::string(glyphs().dot) + "  compiler: ModelForge");
    if (!compare.empty()) {
        return compareCommand(options.modelPath, compare, options.domainPath, options.domainMargin, saveCert,
                              reuseCert);
    }
    if (repair) return repairCommand(options);
    mf::Dossier dossier;
    mf::DiagnosticEngine diagnostics;
    const bool ok = mf::buildDossier(options, dossier, diagnostics, [](const std::string& stage) {
        std::cout << "\r  " << muted() << "working: " << padRight(stage, 40) << reset() << std::flush;
    });
    std::cout << "\r" << std::string(60, ' ') << "\r";
    if (!ok) {
        diagnostics.print(std::cerr);
        return 1;
    }
    view::renderDossier(dossier, options);
    return dossier.ship ? 0 : 3;
}

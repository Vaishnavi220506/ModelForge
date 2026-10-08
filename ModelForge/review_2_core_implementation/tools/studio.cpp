// modelforge_studio: interactive terminal dashboard for ModelForge.
//
//   modelforge_studio                 interactive menu (Iris demo model)
//   modelforge_studio model.mforge    open another model
//   modelforge_studio --demo          run every screen once, non-interactively
//
// Designed for the VS Code integrated terminal; add --ascii on terminals
// without Unicode, --no-color to disable colours.

#include "analysis.h"
#include "codegen.h"
#include "guardian.h"
#include "loader.h"
#include "render.h"
#include "research.h"
#include "term.h"
#include "validator.h"
#include "verifier.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#ifndef MODELFORGE_DASHBOARD_DIR
#define MODELFORGE_DASHBOARD_DIR "dashboard"
#endif
#ifndef MODELFORGE_DEFAULT_MODEL
#define MODELFORGE_DEFAULT_MODEL "models/iris_demo.mforge"
#endif
#ifndef MODELFORGE_MODELS_DIR
#define MODELFORGE_MODELS_DIR "models"
#endif

namespace fs = std::filesystem;
using namespace term;

namespace {

constexpr std::size_t kWidth = 84;
constexpr std::size_t kBudget = 128;
fs::path gDashboard = MODELFORGE_DASHBOARD_DIR;

struct Stage {
    std::string name;
    double milliseconds = 0.0;
    bool ok = false;
    std::string detail;
};

struct Session {
    std::string path;
    std::string source;
    bool loaded = false;
    modelforge::IRGraph original;
    modelforge::IRGraph optimized;
    modelforge::GuardianReport guardian;
    std::vector<modelforge::ActivationSite> sites;
    std::vector<modelforge::ValidationProbe> probes;
    modelforge::CoverageReport coverage;
    std::vector<Stage> stages;
    fs::path generated;
    std::string errors;
    bool interactive = true;
};

std::string opColour(modelforge::IROp op) {
    using modelforge::IROp;
    switch (op) {
        case IROp::Input: return rgb(148, 163, 184);
        case IROp::Return: return rgb(148, 163, 184);
        case IROp::Gemm:
        case IROp::MatMul: return rgb(129, 140, 248);
        case IROp::FusedGemmRelu: return rgb(45, 212, 191);
        case IROp::Add: return rgb(96, 165, 250);
        case IROp::Relu: return rgb(251, 191, 36);
        case IROp::Sigmoid: return rgb(244, 114, 182);
        case IROp::Softmax: return rgb(192, 132, 252);
        default: return bad();
    }
}

std::string shapeText(const modelforge::Shape& shape) {
    std::string result;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        result += (i ? "x" : "") + std::to_string(shape[i]);
    }
    return result.empty() ? "-" : result;
}

std::string readFile(const std::string& path) {
    std::ifstream file(path);
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

template <typename F>
Stage timed(const std::string& name, F body) {
    Stage stage;
    stage.name = name;
    const auto start = std::chrono::steady_clock::now();
    stage.ok = body(stage.detail);
    stage.milliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return stage;
}

bool compileSession(Session& session) {
    session.stages.clear();
    session.loaded = false;
    modelforge::DiagnosticEngine diagnostics;
    modelforge::ModelGraph model;
    modelforge::SymbolTable symbols;
    const auto fail = [&]() {
        std::ostringstream text;
        diagnostics.print(text);
        session.errors = text.str();
        return false;
    };

    session.stages.push_back(timed("Load manifest", [&](std::string& detail) {
        if (!modelforge::loadManifestText(session.source, model, diagnostics)) return false;
        detail = "'" + model.name + "', " + std::to_string(model.nodes.size()) + " nodes, " +
                 std::to_string(model.tensors.size()) + " tensors";
        return true;
    }));
    if (!session.stages.back().ok) return fail();
    session.stages.push_back(timed("Semantic validation", [&](std::string& detail) {
        if (!modelforge::validate(model, diagnostics, symbols)) return false;
        detail = std::to_string(symbols.size()) + " symbols, shapes and operators checked";
        return true;
    }));
    if (!session.stages.back().ok) return fail();
    session.stages.push_back(timed("Lower to ModelForge IR", [&](std::string& detail) {
        auto ir = modelforge::buildIR(model, diagnostics);
        if (!ir) return false;
        session.original = *ir;
        detail = std::to_string(ir->instructions.size()) + " instructions, " +
                 std::to_string(ir->constants.size()) + " constants";
        return true;
    }));
    if (!session.stages.back().ok) return fail();
    session.stages.push_back(timed("Interval bound propagation", [&](std::string& detail) {
        session.sites = modelforge::analyzeActivationSites(session.original);
        std::size_t unstable = 0, total = 0;
        for (const auto& site : session.sites) {
            unstable += site.count(modelforge::NeuronStability::Unstable);
            total += site.width;
        }
        detail = std::to_string(session.sites.size()) + " ReLU layer(s), " +
                 std::to_string(unstable) + "/" + std::to_string(total) + " neurons unstable on D";
        return true;
    }));
    session.stages.push_back(timed("Guardian-APC probe synthesis", [&](std::string& detail) {
        session.probes = modelforge::generateGuardianProbes(session.original);
        session.coverage =
            modelforge::measureActivationCoverage(session.original, session.sites, session.probes);
        detail = std::to_string(session.probes.size()) + " probes, APC " +
                 percent(session.coverage.ratio()) + ", boundary APC " +
                 percent(session.coverage.boundaryRatio());
        return !session.probes.empty();
    }));
    if (!session.stages.back().ok) return fail();
    session.stages.push_back(timed("Guarded optimisation", [&](std::string& detail) {
        session.optimized = session.original;
        session.guardian = modelforge::optimizeWithGuardian(session.optimized, diagnostics);
        std::size_t accepted = 0;
        for (const auto& d : session.guardian.decisions) accepted += d.accepted;
        detail = std::to_string(accepted) + "/" + std::to_string(session.guardian.decisions.size()) +
                 " passes accepted, " +
                 std::to_string(session.guardian.acceptedOptimizations.instructionsBefore) + " " +
                 glyphs().arrow + " " +
                 std::to_string(session.guardian.acceptedOptimizations.instructionsAfter) +
                 " instructions";
        return !diagnostics.hasErrors();
    }));
    if (!session.stages.back().ok) return fail();
    session.stages.push_back(timed("Emit standalone C++", [&](std::string& detail) {
        session.generated = fs::path("build") / "studio_generated";
        if (!modelforge::generateCpp(session.optimized, session.generated.string(), diagnostics)) {
            return false;
        }
        detail = (session.generated / "model.cpp").string();
        return true;
    }));
    if (!session.stages.back().ok) return fail();
    session.loaded = true;
    return true;
}

bool openModel(Session& session, const std::string& path) {
    session.path = path;
    session.source = readFile(path);
    if (session.source.empty()) {
        session.errors = "Cannot read " + path + "\n";
        session.loaded = false;
        return false;
    }
    return compileSession(session);
}

void openZooModel(Session& session, std::size_t index) {
    const auto specs = modelforge::makeZooSpecs(index + 1, 20260917);
    session.path = "synthetic:" + specs.back().name;
    session.source = modelforge::synthesizeManifest(specs.back());
    compileSession(session);
}

void pause(const Session& session) {
    if (!session.interactive) return;
    std::cout << "\n  " << muted() << "Press Enter to return to the menu" << reset() << std::flush;
    std::string ignored;
    std::getline(std::cin, ignored);
}

void header(const Session& session) {
    if (session.interactive) std::cout << code("\x1b[2J\x1b[H");
    banner();
    std::cout << "\n  " << muted() << "Guardian-APC studio   model " << reset() << bold()
              << session.original.name << reset() << "\n";
}

// ------------------------------------------------------------------ screens

void screenPipeline(const Session& session) {
    rule("1  Compilation pipeline", kWidth);
    std::size_t index = 0;
    double total = 0.0;
    for (const Stage& stage : session.stages) {
        total += stage.milliseconds;
        std::cout << "  " << (stage.ok ? good() + glyphs().check : bad() + glyphs().cross)
                  << reset() << " " << muted() << "[" << ++index << "/" << session.stages.size()
                  << "]" << reset() << " " << padRight(bold() + stage.name + reset(), 30) << " "
                  << padLeft(fixed(stage.milliseconds, 2) + " ms", 10) << "  " << muted()
                  << stage.detail << reset() << "\n";
    }
    if (!session.loaded) {
        std::cout << "\n" << bad() << session.errors << reset();
        return;
    }
    const auto& opt = session.guardian.acceptedOptimizations;
    std::size_t unstable = 0;
    for (const auto& site : session.sites) {
        unstable += site.count(modelforge::NeuronStability::Unstable);
    }
    const auto card = [](const std::string& label, const std::string& value,
                         const std::string& colour) {
        return muted() + label + reset() + "  " + bold() + colour + value + reset();
    };
    std::cout << "\n";
    box("Summary",
        {card("instructions", std::to_string(opt.instructionsBefore) + " " + glyphs().arrow + " " +
                                  std::to_string(opt.instructionsAfter),
              teal()) +
             "     " + card("fused", std::to_string(opt.fusedOperations), accent()) + "     " +
             card("removed", std::to_string(opt.removedInstructions), accent()) + "     " +
             card("folded", std::to_string(opt.constantFolds), accent()),
         card("probes", std::to_string(session.probes.size()), teal()) + "     " +
             card("unstable neurons", std::to_string(unstable), warn()) + "     " +
             card("APC", percent(session.coverage.ratio()), good()) + "  " +
             bar(session.coverage.ratio(), 16, good()),
         card("total", fixed(total, 2) + " ms", text()) + "     " + muted() + "generated " +
             reset() + (session.generated / "model.cpp").string()},
        kWidth);
}

std::vector<std::string> graphColumn(const modelforge::IRGraph& graph, const std::string& title) {
    std::vector<std::string> lines = {bold() + title + reset(), ""};
    for (std::size_t i = 0; i < graph.instructions.size(); ++i) {
        const auto& instruction = graph.instructions[i];
        const std::string op = modelforge::irOpName(instruction.operation);
        std::string inputs;
        for (std::size_t k = 0; k < instruction.inputs.size(); ++k) {
            inputs += (k ? "," : "") + instruction.inputs[k];
        }
        lines.push_back(opColour(instruction.operation) + bold() + " " + op + " " + reset() + " " +
                        muted() + instruction.nodeName + reset());
        lines.push_back("   " + dim() + (inputs.empty() ? std::string("") : inputs + " ") +
                        glyphs().arrow + " " + reset() + instruction.output + muted() + " [" +
                        shapeText(instruction.shape) + "]" + reset());
        if (i + 1 < graph.instructions.size()) {
            lines.push_back("   " + rgb(71, 85, 105) + glyphs().down + reset());
        }
    }
    return lines;
}

void screenGraph(const Session& session) {
    rule("2  IR graph: before and after guarded optimisation", kWidth);
    const auto left = graphColumn(session.original, "Before (lowered IR)");
    const auto right = graphColumn(session.optimized, "After (Guardian-accepted)");
    const std::size_t rows = std::max(left.size(), right.size());
    for (std::size_t i = 0; i < rows; ++i) {
        std::cout << "  " << padRight(i < left.size() ? left[i] : "", 46) << rgb(51, 65, 85)
                  << glyphs().v << reset() << "  " << (i < right.size() ? right[i] : "") << "\n";
    }
    std::cout << "\n";
    for (const auto& decision : session.guardian.decisions) {
        for (const auto& event : decision.optimization.events) {
            std::cout << "  " << (decision.accepted ? good() : bad()) << glyphs().dot << reset()
                      << " " << event << muted() << "  (" << decision.passName << ")" << reset()
                      << "\n";
        }
    }
}

void screenStability(const Session& session) {
    rule("3  Neuron stability map (interval bound propagation on D = [-10, 10]^n)", kWidth);
    std::cout << "  " << good() << "■" << reset() << " stable active   " << muted() << "■"
              << reset() << " stable inactive   " << warn() << "■" << reset()
              << " unstable (kink reachable; Guardian targets these)\n\n";
    if (session.sites.empty()) {
        std::cout << "  No ReLU layers in this model.\n";
        return;
    }
    for (const auto& site : session.sites) {
        std::string cells;
        for (const auto kind : site.stability) {
            cells += (kind == modelforge::NeuronStability::Unstable       ? warn()
                      : kind == modelforge::NeuronStability::StableActive ? good()
                                                                          : muted()) +
                     (settings().unicode ? "■ " : "# ") + reset();
        }
        std::cout << "  " << bold() << padRight("L" + std::to_string(site.layer), 4) << reset()
                  << muted() << padRight(site.nodeName, 16) << reset() << cells << "\n";
    }
    // Interval plot for the first layer's units on a shared axis.
    const auto& site = session.sites.front();
    float extent = 1.0f;
    for (std::size_t u = 0; u < site.width; ++u) {
        extent = std::max({extent, std::fabs(site.lower[u]), std::fabs(site.upper[u])});
    }
    const std::size_t axis = 50;
    std::cout << "\n  " << muted() << "Pre-activation intervals, layer L0 (axis ±"
              << fixed(extent, 1) << ", " << reset() << pink() << "|" << reset() << muted()
              << " = ReLU kink at 0)" << reset() << "\n";
    for (std::size_t u = 0; u < std::min<std::size_t>(site.width, 16); ++u) {
        const auto column = [&](float v) {
            return static_cast<std::size_t>(std::round((v / extent + 1.0f) * 0.5f * (axis - 1)));
        };
        const std::size_t lo = column(site.lower[u]);
        const std::size_t hi = column(site.upper[u]);
        const std::size_t zero = column(0.0f);
        std::string line;
        for (std::size_t c = 0; c < axis; ++c) {
            if (c == zero) line += pink() + "|" + reset();
            else if (c >= lo && c <= hi) line += teal() + (settings().unicode ? "━" : "=") + reset();
            else line += rgb(51, 65, 85) + (settings().unicode ? "·" : ".") + reset();
        }
        std::cout << "  " << muted() << padRight("n" + std::to_string(u), 5) << reset() << line
                  << "  " << muted() << "[" << fixed(site.lower[u], 2) << ", "
                  << fixed(site.upper[u], 2) << "]" << reset() << "\n";
    }
}

void screenGuardian(const Session& session) {
    rule("4  Guardian decisions and activation-pattern coverage", kWidth);
    Table decisions;
    decisions.headers = {"pass", "verdict", "probes", "max |err|", "rewrites"};
    decisions.rightAlign = {false, false, true, true, true};
    for (const auto& d : session.guardian.decisions) {
        decisions.rows.push_back(
            {d.passName,
             d.accepted ? good() + glyphs().check + " ACCEPT" + reset()
                        : bad() + glyphs().cross + " REJECT" + reset(),
             std::to_string(d.validation.probeCount), fixed(d.validation.maximumAbsoluteError, 6),
             std::to_string(d.optimization.events.size())});
    }
    decisions.print();

    std::cout << "\n  " << muted()
              << "Coverage of (neuron, pre-activation bin) states at equal budget B = " << kBudget
              << reset() << "\n";
    const auto suites = modelforge::buildStrategies(session.original, kBudget, 20260917, 1);
    Table coverage;
    coverage.headers = {"strategy", "APC", "", "boundary APC", ""};
    coverage.rightAlign = {false, true, false, true, false};
    for (const auto& suite : suites) {
        const auto report =
            modelforge::measureActivationCoverage(session.original, session.sites, suite.probes);
        const bool ours = suite.name == "guardian_apc";
        coverage.rows.push_back({render::strategyLabel(suite.name.substr(0, suite.name.find('#'))),
                                 percent(report.ratio()), bar(report.ratio(), 16, ours ? teal() : accent()),
                                 percent(report.boundaryRatio()),
                                 bar(report.boundaryRatio(), 16, ours ? teal() : accent())});
    }
    coverage.print();

    std::cout << "\n  " << muted() << "Guardian-APC schedule (greedy coverage order), first 12:"
              << reset() << "\n  ";
    for (std::size_t i = 0; i < std::min<std::size_t>(12, session.probes.size()); ++i) {
        const bool deep = session.probes[i].name.rfind("deep_boundary", 0) == 0;
        std::cout << (deep ? teal() : accent()) << session.probes[i].name << reset()
                  << (i % 3 == 2 ? "\n  " : muted() + "  " + glyphs().arrow + "  " + reset());
    }
    std::cout << "\n";
}

void screenFaults(const Session& session) {
    rule("5  Fault-injection arena: which probe suite catches each faulty rewrite?", kWidth);
    const auto faults = modelforge::makeFaults(session.original, session.sites, 20260917);
    const auto suites = modelforge::buildStrategies(session.original, kBudget, 20260917, 1);
    std::vector<std::string> shown = {"random#0", "guardian_v1", "guardian_apc"};
    Table table;
    table.headers = {"fault", "kind"};
    table.rightAlign = {false, false};
    for (const auto& name : shown) {
        table.headers.push_back(name.substr(0, name.find('#')));
        table.rightAlign.push_back(true);
    }
    std::vector<std::vector<std::optional<std::vector<float>>>> expected;
    for (const auto& suite : suites) {
        std::vector<std::optional<std::vector<float>>> outputs;
        for (const auto& probe : suite.probes) {
            modelforge::DiagnosticEngine d;
            outputs.push_back(modelforge::executeIR(session.original, probe.values, d));
        }
        expected.push_back(std::move(outputs));
    }
    for (const auto& fault : faults) {
        std::vector<std::string> row = {
            fault.name, muted() + fault.locality.substr(0, fault.locality.find('_')) + (fault.heldOut ? " *" : "") + reset()};
        for (const auto& name : shown) {
            for (std::size_t s = 0; s < suites.size(); ++s) {
                if (suites[s].name != name) continue;
                std::size_t first = 0;
                for (std::size_t p = 0; p < suites[s].probes.size() && first == 0; ++p) {
                    modelforge::DiagnosticEngine d;
                    const auto out = modelforge::executeIR(fault.candidate, suites[s].probes[p].values, d);
                    const auto& exp = expected[s][p];
                    if (!out || !exp || !modelforge::compareOutputs(*exp, *out).passed ||
                        std::max_element(exp->begin(), exp->end()) - exp->begin() !=
                            std::max_element(out->begin(), out->end()) - out->begin()) {
                        first = p + 1;
                    }
                }
                if (fault.negativeControl) {
                    row.push_back(first ? bad() + "false alarm" + reset()
                                        : good() + "silent " + glyphs().check + reset());
                } else {
                    row.push_back(first ? good() + "#" + std::to_string(first) + reset()
                                        : bad() + "missed" + reset());
                }
            }
        }
        table.rows.push_back(std::move(row));
    }
    table.print();
    std::cout << "  " << muted() << "#k = first probe that caught the fault (lower is better).  * = held-out fault type." << reset() << "\n";

    // Replayable counterexample for the classic wrong ReLU rewrite.
    modelforge::IRGraph buggy = session.original;
    for (auto& instruction : buggy.instructions) {
        if (instruction.operation == modelforge::IROp::Relu) {
            instruction.operation = modelforge::IROp::Sigmoid;
            modelforge::DiagnosticEngine d;
            const auto decision = modelforge::checkCandidate(session.original, buggy,
                                                             "wrong_relu_rewrite", d);
            if (!decision.validation.firstFailure) break;
            const auto& f = *decision.validation.firstFailure;
            std::vector<std::string> lines = {
                muted() + "rewrite  " + reset() + "ReLU " + glyphs().arrow + " Sigmoid   " +
                    (decision.accepted ? bad() + "MISSED" : good() + "REJECTED + rolled back") +
                    reset(),
                muted() + "witness  " + reset() + f.name};
            std::string input = muted() + "input    " + reset() + "[";
            for (std::size_t k = 0; k < std::min<std::size_t>(f.values.size(), 8); ++k) {
                input += (k ? ", " : "") + fixed(f.values[k], 3);
            }
            lines.push_back(input + (f.values.size() > 8 ? ", ...]" : "]"));
            const auto& a = decision.validation.originalOutput;
            const auto& b = decision.validation.candidateOutput;
            for (std::size_t k = 0; k < std::min(a.size(), b.size()); ++k) {
                lines.push_back(muted() + "class " + std::to_string(k) + "  " + reset() +
                                bar(a[k], 18, accent()) + " " + padLeft(fixed(a[k], 4), 7) +
                                muted() + "  vs  " + reset() + bar(b[k], 18, pink()) + " " +
                                padLeft(fixed(b[k], 4), 7));
            }
            lines.push_back(accent() + "■ original" + reset() + "   " + pink() + "■ faulty candidate" +
                            reset() + muted() + "   (minimised, replayable counterexample)" + reset());
            std::cout << "\n";
            box("Counterexample", lines, kWidth, pink());
            break;
        }
    }
}

void screenBenchmark(Session& session, bool quick) {
    rule(std::string("6  Research benchmark (") + (quick ? "quick" : "full") + ")", kWidth);
    modelforge::BenchmarkConfig config;
    if (quick) {
        config.models = 12;
        config.randomSeeds = 2;
        config.oracleProbes = 1000;
    }
    const auto summary = modelforge::runBenchmark(
        config, [](std::size_t done, std::size_t total, const std::string& label) {
            progress(done, total, label);
        });
    render::benchmark(summary, config.budget);
    const std::string inspect =
        modelforge::inspectModelJson(session.path, session.original, config.budget, config.seed);
    const fs::path dashboard = gDashboard;
    if (render::writeDashboardData(dashboard, summary.json, inspect)) {
        std::cout << "\n  " << good() << glyphs().check << reset() << " Dashboard data refreshed: "
                  << (dashboard / "index.html").string() << "\n";
    }
}

void screenCode(const Session& session) {
    rule("7  Generated C++ (model.cpp)", kWidth);
    std::ifstream file(session.generated / "model.cpp");
    std::string line;
    std::size_t number = 0;
    std::size_t shown = 0;
    bool started = false;
    const std::vector<std::string> keywords = {"#include", "namespace", "static", "const", "float",
                                               "std::", "return", "for", "Tensor", "int ", "void"};
    // Skip the shared kernel helpers; start at the baked-in weights and infer().
    while (std::getline(file, line) && shown < 48) {
        ++number;
        if (!started && line.rfind("static const Tensor c_", 0) != 0) continue;
        started = true;
        ++shown;
        if (line.size() > 84) line = line.substr(0, 81) + "...";
        std::string coloured = line;
        if (line.find("//") != std::string::npos) {
            coloured = muted() + line + reset();
        } else {
            for (const auto& keyword : keywords) {
                if (line.find(keyword) != std::string::npos) {
                    coloured = accent() + line + reset();
                    break;
                }
            }
        }
        std::cout << "  " << rgb(71, 85, 105) << padLeft(std::to_string(number), 4) << " "
                  << reset() << coloured << "\n";
    }
    std::cout << "  " << muted() << "... full file: " << (session.generated / "model.cpp").string()
              << reset() << "\n";
}

void screenDashboard() {
    rule("8  Web dashboard", kWidth);
    const fs::path index = fs::path(MODELFORGE_DASHBOARD_DIR) / "index.html";
    box("Open the dashboard",
        {"File: " + bold() + index.string() + reset(),
         muted() + "VS Code: right-click dashboard/index.html " + glyphs().arrow +
             " Reveal in File Explorer, then double-click;" + reset(),
         muted() + "or run the task 'ModelForge: open dashboard'. No server needed." + reset(),
         muted() + "Run option 6 first to refresh the data with this machine's results." + reset()},
        kWidth);
}

void chooseModel(Session& session) {
    rule("Choose a model", kWidth);
    std::vector<fs::path> files;
    std::error_code error;
    for (const auto& entry : fs::directory_iterator(MODELFORGE_MODELS_DIR, error)) {
        if (entry.path().extension() == ".mforge") files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    for (std::size_t i = 0; i < files.size(); ++i) {
        std::cout << "  " << accent() << "[" << i + 1 << "]" << reset() << " "
                  << files[i].filename().string() << "\n";
    }
    std::cout << "  " << accent() << "[z]" << reset()
              << " synthetic zoo model (deeper networks, e.g. z7)\n  " << accent() << "[p]"
              << reset() << " path to a .mforge file\n\n  > " << std::flush;
    std::string choice;
    std::getline(std::cin, choice);
    if (!choice.empty() && choice[0] == 'z') {
        std::size_t index = 3;
        try {
            if (choice.size() > 1) index = std::stoul(choice.substr(1));
        } catch (...) {
        }
        openZooModel(session, index);
    } else if (choice == "p") {
        std::cout << "  path: " << std::flush;
        std::getline(std::cin, choice);
        openModel(session, choice);
    } else {
        try {
            const std::size_t index = std::stoul(choice);
            if (index >= 1 && index <= files.size()) openModel(session, files[index - 1].string());
        } catch (...) {
        }
    }
}

// One-line summary for the home screen; the full pipeline lives under option 1.
void statusLine(const Session& session) {
    double total = 0.0;
    for (const Stage& stage : session.stages) total += stage.milliseconds;
    const auto& opt = session.guardian.acceptedOptimizations;
    std::cout << "\n  " << good() << glyphs().check << reset() << " compiled in "
              << bold() << fixed(total, 1) << " ms" << reset() << muted() << "   "
              << opt.instructionsBefore << " " << glyphs().arrow << " " << opt.instructionsAfter
              << " instructions   " << session.probes.size() << " probes   APC "
              << percent(session.coverage.ratio(), 0) << reset() << "\n";
}

void menu(const Session& session) {
    const auto item = [](const std::string& key, const std::string& label, const std::string& hint) {
        return "  " + accent() + bold() + key + reset() + "   " + padRight(label, 24) + muted() +
               hint + reset();
    };
    const auto group = [](const std::string& title) {
        return "\n  " + dim() + title + reset();
    };
    std::cout << group("COMPILER") << "\n"
              << item("1", "Pipeline", "stages and timings") << "\n"
              << item("2", "IR graph", "before / after") << "\n"
              << item("7", "Generated C++", "model.cpp") << "\n"
              << group("GUARDIAN-APC") << "\n"
              << item("3", "Neuron stability", "IBP intervals") << "\n"
              << item("4", "Coverage", "pass decisions, APC") << "\n"
              << item("5", "Fault arena", "who catches which bug") << "\n"
              << group("RESEARCH") << "\n"
              << item("6", "Benchmark", "quick run  (6f = full)") << "\n"
              << item("8", "Web dashboard", "how to open it") << "\n"
              << group("OTHER") << "\n"
              << item("m", "Change model", "") << "\n"
              << item("q", "Quit", "") << "\n\n";
    if (!session.loaded) std::cout << bad() << session.errors << reset();
    std::cout << "  " << accent() << glyphs().arrow << reset() << " " << std::flush;
}

}  // namespace

int main(int argc, char* argv[]) {
    Session session;
    std::string modelPath = MODELFORGE_DEFAULT_MODEL;
    bool demo = false, ascii = false, noColor = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--demo") demo = true;
        else if (arg == "--ascii") ascii = true;
        else if (arg == "--no-color") noColor = true;
        else if (arg == "--dashboard" && i + 1 < argc) gDashboard = argv[++i];
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: modelforge_studio [model.mforge] [--demo] [--ascii] [--no-color] [--dashboard DIR]\n";
            return 0;
        } else modelPath = arg;
    }
    init(ascii, noColor);
    session.interactive = !demo;
    openModel(session, modelPath);

    if (demo) {
        header(session);
        if (!session.loaded) {
            std::cerr << session.errors;
            return 1;
        }
        screenPipeline(session);
        std::cout << "\n";
        screenGraph(session);
        std::cout << "\n";
        screenStability(session);
        std::cout << "\n";
        screenGuardian(session);
        std::cout << "\n";
        screenFaults(session);
        std::cout << "\n";
        screenCode(session);
        std::cout << "\n";
        screenBenchmark(session, true);
        screenDashboard();
        return 0;
    }

    while (true) {
        header(session);
        if (session.loaded) statusLine(session);
        menu(session);
        std::string choice;
        if (!std::getline(std::cin, choice)) break;
        if (choice == "q" || choice == "Q" || choice == "0") break;
        if (choice == "m") {
            chooseModel(session);
            continue;
        }
        if (!session.loaded) continue;
        header(session);
        if (choice == "1") screenPipeline(session);
        else if (choice == "2") screenGraph(session);
        else if (choice == "3") screenStability(session);
        else if (choice == "4") screenGuardian(session);
        else if (choice == "5") screenFaults(session);
        else if (choice == "6") screenBenchmark(session, true);
        else if (choice == "6f") screenBenchmark(session, false);
        else if (choice == "7") screenCode(session);
        else if (choice == "8") screenDashboard();
        else continue;
        pause(session);
    }
    std::cout << "\n  " << muted() << "Goodbye from ModelForge Studio by Vaishnavi." << reset() << "\n\n";
    return 0;
}

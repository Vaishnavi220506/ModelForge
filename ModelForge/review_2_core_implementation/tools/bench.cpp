// modelforge_bench: equal-budget fault-injection study over a synthetic model
// zoo. Writes results.json, cases.csv and the dashboard's data.js.

#include "render.h"
#include "research.h"
#include "term.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

#ifndef MODELFORGE_DASHBOARD_DIR
#define MODELFORGE_DASHBOARD_DIR "dashboard"
#endif
#ifndef MODELFORGE_MODELS_DIR
#define MODELFORGE_MODELS_DIR "models"
#endif
#ifndef MODELFORGE_DEFAULT_MODEL
#define MODELFORGE_DEFAULT_MODEL "models/iris_demo.mforge"
#endif

namespace {

void usage() {
    std::cout << "Usage: modelforge_bench [options]\n\n"
              << "  --models N       synthetic models in the zoo (default 96)\n"
              << "  --budget B       probes per strategy (default 128)\n"
              << "  --seeds R        random-baseline seeds (default 5)\n"
              << "  --oracle K       random oracle probes per fault (default 4000)\n"
              << "  --seed S         master seed (default 20260917)\n"
              << "  --quick          12 models, 2 seeds, 1000 oracle probes\n"
              << "  --out DIR        results directory (default build/research)\n"
              << "  --dashboard DIR  where to write data.js (default: repository dashboard/)\n"
              << "  --inspect FILE   model shown in the dashboard's inspector tab\n"
              << "  --ascii          plain ASCII output\n"
              << "  --no-color       disable colours\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    modelforge::BenchmarkConfig config;
    std::filesystem::path out = "build/research";
    std::filesystem::path dashboard = MODELFORGE_DASHBOARD_DIR;
    std::string inspect = MODELFORGE_DEFAULT_MODEL;
    bool ascii = false, noColor = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << arg << " needs a value\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--models") config.models = std::stoul(next());
        else if (arg == "--budget") config.budget = std::stoul(next());
        else if (arg == "--seeds") config.randomSeeds = std::stoul(next());
        else if (arg == "--oracle") config.oracleProbes = std::stoul(next());
        else if (arg == "--seed") config.seed = static_cast<std::uint32_t>(std::stoul(next()));
        else if (arg == "--out") out = next();
        else if (arg == "--dashboard") dashboard = next();
        else if (arg == "--inspect") inspect = next();
        else if (arg == "--ascii") ascii = true;
        else if (arg == "--no-color") noColor = true;
        else if (arg == "--quick") {
            config.models = 12;
            config.randomSeeds = 2;
            config.oracleProbes = 1000;
        } else if (arg == "--help" || arg == "-h") {
            usage();
            return 0;
        } else {
            std::cerr << "Unknown option " << arg << "\n";
            usage();
            return 2;
        }
    }
    if (config.budget == 0 || config.models == 0 || config.randomSeeds == 0) {
        std::cerr << "--models, --budget and --seeds must be positive\n";
        return 2;
    }
    term::init(ascii, noColor);
    term::banner();
    std::cout << "  " << term::muted() << "Guardian-APC research benchmark by Vaishnavi" << term::reset()
              << "  " << term::dim() << config.models << " models, budget " << config.budget
              << ", " << config.randomSeeds << " random seeds" << term::reset() << "\n\n";

    const auto summary = modelforge::runBenchmark(
        config, [](std::size_t done, std::size_t total, const std::string& label) {
            term::progress(done, total, label);
        });
    render::benchmark(summary, config.budget);

    // Real trained models under compression (fp16, int8, int4, pruning).
    const auto real = modelforge::runRealStudy(
        (std::filesystem::path(MODELFORGE_MODELS_DIR) / "real").string(), 200,
        config.randomSeeds, [](std::size_t done, std::size_t total, const std::string& label) {
            term::progress(done, total, label);
        });
    render::realStudy(real);

    std::string inspectJson;
    {
        modelforge::DiagnosticEngine diagnostics;
        modelforge::IRGraph graph;
        std::ifstream file(inspect);
        const std::string text((std::istreambuf_iterator<char>(file)),
                               std::istreambuf_iterator<char>());
        if (file && modelforge::compileManifestText(text, graph, diagnostics)) {
            inspectJson = modelforge::inspectModelJson(
                std::filesystem::path(inspect).filename().string(), graph, config.budget,
                config.seed);
        } else {
            std::cerr << "warning: could not load inspector model " << inspect << "\n";
        }
    }

    const bool ok = render::writeText(out / "results.json", summary.json) &&
                    render::writeText(out / "cases.csv", summary.casesCsv) &&
                    (inspectJson.empty() || render::writeText(out / "inspect.json", inspectJson)) &&
                    render::writeText(out / "real_results.json", real.json) &&
                    render::writeDashboardData(dashboard, summary.json, inspectJson, real.json);
    term::rule("Outputs");
    std::cout << "  " << term::muted() << "results   " << term::reset()
              << (out / "results.json").string() << "\n  " << term::muted() << "per-case  "
              << term::reset() << (out / "cases.csv").string() << "\n  " << term::muted()
              << "dashboard " << term::reset() << (dashboard / "index.html").string()
              << term::muted() << "  (data.js refreshed)" << term::reset() << "\n\n";
    if (!ok) {
        std::cerr << "Could not write benchmark outputs\n";
        return 1;
    }
    // Sanity gates for CI: IBP must be sound and the negative control must stay silent.
    bool sane = summary.ibpViolations == 0;
    for (const auto& s : summary.strategies) sane = sane && s.falsePositives == 0;
    if (!sane) std::cerr << "Sanity check failed: IBP violation or negative-control false positive\n";
    return sane ? 0 : 1;
}

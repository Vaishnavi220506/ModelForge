// modelforge_bench: every number in the paper, in the terminal.
//   1. Guardian-APC+ equal-budget fault-injection study (synthetic zoo)
//   2. Boundary-shift probing on compressed real models
//   3. Temper decisiveness: falsify-or-certify over faults, controls and passes
//   4. Temper compression severity brackets
//   5. Temper binary audit of the generated C++
// Writes JSON/CSV snapshots to --out.

#include "render.h"
#include "research.h"
#include "term.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

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
              << "  --inspect FILE   model summarised in inspect.json\n"
              << "  --balls N        test rows per compression audit (default all; quick 30)\n"
              << "  --guardian-only  skip the Temper studies\n"
              << "  --no-binary      skip compiling generated binaries\n"
              << "  --ascii          plain ASCII output\n"
              << "  --no-color       disable colours\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    modelforge::BenchmarkConfig config;
    std::filesystem::path out = "build/research";
    std::size_t balls = 0;
    bool temperStudies = true, binary = true;
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
        else if (arg == "--balls") balls = std::stoul(next());
        else if (arg == "--guardian-only") temperStudies = false;
        else if (arg == "--no-binary") binary = false;
        else if (arg == "--inspect") inspect = next();
        else if (arg == "--ascii") ascii = true;
        else if (arg == "--no-color") noColor = true;
        else if (arg == "--quick") {
            config.models = 12;
            config.randomSeeds = 2;
            config.oracleProbes = 1000;
            balls = 30;
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
    term::temperBanner("research benchmark: Guardian-APC+ and Temper (compiler: ModelForge)");
    std::cout << "  " << term::muted() << "every number in the paper" << term::reset()
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

    const auto bar = [](std::size_t done, std::size_t total, const std::string& label) {
        term::progress(done, total, label);
    };
    bool sane = summary.ibpViolations == 0;
    for (const auto& s : summary.strategies) sane = sane && s.falsePositives == 0;
    bool ok = render::writeText(out / "results.json", summary.json) &&
              render::writeText(out / "cases.csv", summary.casesCsv) &&
              (inspectJson.empty() || render::writeText(out / "inspect.json", inspectJson)) &&
              render::writeText(out / "real_results.json", real.json);
    if (temperStudies) {
        const auto temper = modelforge::runTemperStudy(config, bar);
        render::temperStudy(temper);
        const auto realDirectory = (std::filesystem::path(MODELFORGE_MODELS_DIR) / "real").string();
        const auto compression = modelforge::runCompressionStudy(realDirectory, balls, bar);
        render::compressionStudy(compression);
        ok = ok && render::writeText(out / "temper_results.json", temper.json) &&
             render::writeText(out / "temper_cases.csv", temper.casesCsv) &&
             render::writeText(out / "compression_results.json", compression.json);
        for (const auto& s : temper.strategies) sane = sane && s.violations == 0;
        sane = sane && compression.bracketViolations == 0;
        if (binary) {
            const auto binaries =
                modelforge::runBinaryStudy(realDirectory, (out / "binaries").string(), bar);
            render::binaryStudy(binaries);
            ok = ok && render::writeText(out / "binary_results.json", binaries.json);
        }
    }
    term::rule("Outputs");
    std::cout << "  " << term::muted() << "JSON and CSV snapshots in " << term::reset()
              << out.string() << "\n\n";
    if (!ok) {
        std::cerr << "Could not write benchmark outputs\n";
        return 1;
    }
    // Sanity gates for CI: IBP and every certificate must be sound, the
    // negative control must stay silent, and no bracket may invert.
    if (!sane) std::cerr << "Sanity check failed: unsound bound, certificate or false alarm\n";
    return sane ? 0 : 1;
}

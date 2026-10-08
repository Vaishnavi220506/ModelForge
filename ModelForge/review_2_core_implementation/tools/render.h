#pragma once

// Terminal renderings shared by modelforge_bench and modelforge_studio.

#include "research.h"
#include "term.h"

#include <filesystem>
#include <fstream>
#include <string>

namespace render {

inline std::string strategyLabel(const std::string& name) {
    if (name == "guardian_apc") return term::bold() + term::teal() + "guardian_apc ★" + term::reset();
    return name;
}

inline std::string rateColour(double rate) {
    if (rate >= 0.95) return term::good();
    if (rate >= 0.75) return term::warn();
    return term::bad();
}

inline void benchmark(const modelforge::BenchmarkSummary& summary, std::size_t budget) {
    using namespace term;
    box("Corpus",
        {muted() + "models " + reset() + bold() + std::to_string(summary.models) + reset() +
             muted() + "   faults " + reset() + bold() + std::to_string(summary.faults) + reset() +
             muted() + "   observable " + reset() + bold() + std::to_string(summary.observable) +
             reset() + muted() + "   unknown " + reset() + std::to_string(summary.unknown) +
             muted() + "   controls " + reset() + std::to_string(summary.controls),
         muted() + "IBP neurons: " + reset() + good() + std::to_string(summary.stableActive) +
             " stable-active" + reset() + ", " + muted() + std::to_string(summary.stableInactive) +
             " stable-inactive" + reset() + ", " + warn() + std::to_string(summary.unstable) +
             " unstable" + reset(),
         muted() + "IBP-proved unobservable " + reset() + std::to_string(summary.provable) +
             muted() + "   IBP soundness violations " + reset() +
             (summary.ibpViolations == 0 ? good() : bad()) +
             std::to_string(summary.ibpViolations) + reset() + muted() + "   runtime " + reset() +
             fixed(summary.seconds, 1) + "s"});

    rule("Fault detection at equal budget B = " + std::to_string(budget));
    Table table;
    table.headers = {"strategy", "rate", "", "95% CI", "1st hit", "false alarms"};
    table.rightAlign = {false, true, false, true, true, true};
    for (const auto& s : summary.strategies) {
        table.rows.push_back({strategyLabel(s.name), rateColour(s.rate) + percent(s.rate) + reset(),
                              bar(s.rate, 16, s.name == "guardian_apc" ? teal() : accent()),
                              muted() + percent(s.low, 0) + "-" + percent(s.high, 0) + reset(),
                              fixed(s.meanFirst, 1),
                              (s.falsePositives ? bad() : good()) +
                                  std::to_string(s.falsePositives) + reset()});
    }
    table.print();

    rule("Detection rate vs probe budget");
    Table curve;
    curve.headers = {"strategy"};
    curve.rightAlign = {false};
    if (!summary.strategies.empty()) {
        for (const auto& point : summary.strategies.front().curve) {
            curve.headers.push_back("B=" + std::to_string(point.first));
            curve.rightAlign.push_back(true);
        }
        curve.headers.push_back("trend");
        curve.rightAlign.push_back(false);
    }
    for (const auto& s : summary.strategies) {
        if (s.name != "random" && s.name != "guardian_v1" && s.name != "deep_boundary" &&
            s.name != "guardian_apc") {
            continue;  // ablations stay in the leaderboard and the dashboard
        }
        std::vector<std::string> row = {strategyLabel(s.name)};
        std::vector<double> values;
        for (const auto& point : s.curve) {
            row.push_back(rateColour(point.second) + percent(point.second, 0) + reset());
            values.push_back(point.second);
        }
        row.push_back(sparkline(values, s.name == "guardian_apc" ? teal() : accent()));
        curve.rows.push_back(std::move(row));
    }
    curve.print();

    rule("Detection rate by fault family (full budget)");
    Table families;
    families.headers = {"family"};
    families.rightAlign = {false};
    std::vector<std::string> shown = {"random", "generic", "guardian_v1", "guardian_apc"};
    for (const auto& name : shown) {
        families.headers.push_back(name);
        families.rightAlign.push_back(true);
    }
    const modelforge::StrategySummary* reference = nullptr;
    for (const auto& s : summary.strategies) {
        if (s.name == "guardian_apc") reference = &s;
    }
    if (reference != nullptr) {
        for (const auto& [family, unused] : reference->families) {
            std::vector<std::string> row = {family};
            for (const auto& name : shown) {
                for (const auto& s : summary.strategies) {
                    if (s.name != name) continue;
                    const auto found = s.families.find(family);
                    const double rate = found == s.families.end() ? 0.0 : found->second;
                    row.push_back(rateColour(rate) + percent(rate, 0) + reset());
                }
            }
            families.rows.push_back(std::move(row));
        }
    }
    families.print();

    rule("Paired McNemar tests (guardian_apc vs ...)");
    Table tests;
    tests.headers = {"baseline", "only ours", "only baseline", "exact p", "verdict"};
    tests.rightAlign = {false, true, true, true, false};
    for (const auto& c : summary.comparisons) {
        const bool better = c.onlyOurs > c.onlyOther && c.pValue < 0.05;
        const bool worse = c.onlyOther > c.onlyOurs && c.pValue < 0.05;
        tests.rows.push_back({c.other, std::to_string(c.onlyOurs), std::to_string(c.onlyOther),
                              c.pValue < 1e-4 ? "<1e-4" : fixed(c.pValue, 4),
                              better  ? good() + "significantly better" + reset()
                              : worse ? bad() + "significantly worse" + reset()
                                      : muted() + "no significant difference" + reset()});
    }
    tests.print();
}

inline bool writeText(const std::filesystem::path& path, const std::string& value) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    output << value;
    return static_cast<bool>(output);
}

// The dashboard reads data.js through a <script> tag, so it works from file://
// without a web server.
inline bool writeDashboardData(const std::filesystem::path& directory,
                               const std::string& benchJson, const std::string& inspectJson) {
    return writeText(directory / "data.js",
                     "// Generated by ModelForge (author: Vaishnavi). Re-run modelforge_bench to refresh.\n"
                     "window.MODELFORGE_DATA = {\"bench\": " +
                         (benchJson.empty() ? std::string("null") : benchJson) +
                         ", \"inspect\": " +
                         (inspectJson.empty() ? std::string("null") : inspectJson) + "};\n");
}

}  // namespace render

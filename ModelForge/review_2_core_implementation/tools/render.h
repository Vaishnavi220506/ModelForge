#pragma once

// Terminal renderings shared by modelforge_bench and modelforge_studio.

#include "research.h"
#include "temper_study.h"

#include <array>
#include <map>
#include "term.h"

#include <filesystem>
#include <fstream>
#include <string>

namespace render {

inline std::string strategyLabel(const std::string& name) {
    if (name == "guardian_apc_plus") return term::bold() + term::teal() + "guardian_apc_plus ★" + term::reset();
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
                              bar(s.rate, 16, s.name == "guardian_apc_plus" ? teal() : accent()),
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
        if (s.name != "random" && s.name != "guardian_v1" &&
            s.name != "guardian_apc" && s.name != "guardian_apc_plus") {
            continue;  // ablations stay in the leaderboard and results.json
        }
        std::vector<std::string> row = {strategyLabel(s.name)};
        std::vector<double> values;
        for (const auto& point : s.curve) {
            row.push_back(rateColour(point.second) + percent(point.second, 0) + reset());
            values.push_back(point.second);
        }
        row.push_back(sparkline(values, s.name == "guardian_apc_plus" ? teal() : accent()));
        curve.rows.push_back(std::move(row));
    }
    curve.print();

    rule("Detection rate by fault family (full budget)");
    Table families;
    families.headers = {"family"};
    families.rightAlign = {false};
    std::vector<std::string> shown = {"random", "guardian_v1", "guardian_apc", "guardian_apc_plus"};
    for (const auto& name : shown) {
        families.headers.push_back(name);
        families.rightAlign.push_back(true);
    }
    const modelforge::StrategySummary* reference = nullptr;
    for (const auto& s : summary.strategies) {
        if (s.name == "guardian_apc_plus") reference = &s;
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

    rule("Paired McNemar tests (guardian_apc_plus vs ...)");
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

inline void realStudy(const modelforge::RealStudySummary& real) {
    using namespace term;
    rule("Real models: is the compressed model safe to ship?");
    std::cout << "  " << muted() << real.cases << " model x compression cases, " << real.risky
              << " risky (a realistic input exists where the original was sure but the "
                 "compressed model disagrees)" << reset() << "\n\n";
    Table table;
    table.headers = {"method", "risky caught", "", "missed by test set", "accuracy unchanged"};
    table.rightAlign = {false, true, false, true, true};
    const std::vector<std::pair<std::string, std::string>> names = {
        {"test_set", "test set only"}, {"noise", "test set + noise"},
        {"genetic", "test set + genetic"}, {"boundary_shift", "test set + boundary shift"}};
    for (const auto& entry : names) {
        const std::string& key = entry.first;
        const std::string& label = entry.second;
        const auto get = [&](const std::map<std::string, double>& values) {
            const auto found = values.find(key);
            return found == values.end() ? 0.0 : found->second;
        };
        const bool ours = key == "boundary_shift";
        table.rows.push_back(
            {ours ? bold() + teal() + label + " ★" + reset() : label,
             fixed(get(real.detected), 1) + "/" + std::to_string(real.risky),
             bar(real.risky ? get(real.detected) / real.risky : 0.0, 14, ours ? teal() : accent()),
             fixed(get(real.detectedHidden), 1) + "/" + std::to_string(real.hiddenFromTestSet),
             fixed(get(real.detectedUnchanged), 1) + "/" + std::to_string(real.accuracyUnchanged)});
    }
    table.print();
}

// ------------------------------------------------------------ Temper studies

inline std::string share(std::size_t part, std::size_t whole) {
    return whole == 0 ? "-" : term::percent(static_cast<double>(part) / static_cast<double>(whole));
}

inline void temperStudy(const modelforge::TemperStudySummary& t) {
    using namespace term;
    rule("Temper: falsify-or-certify every transformation");
    std::cout << "  " << muted() << t.models << " models, " << t.cases << " transformations: "
              << t.faults << " injected faults (" << t.observable << " observable, " << t.hidden
              << " exposed by nothing), " << t.controls << " negative controls, " << t.passes
              << " real optimisation passes" << reset() << "\n\n";
    Table table;
    table.headers = {"strategy", "observable refuted", "benign certified", "hidden faults R/C/U",
                     "all decided", "evals/case", "ms/case", "unsound"};
    table.rightAlign = {false, true, true, true, true, true, true, true};
    for (const auto& s : t.strategies) {
        const std::size_t benignCertified = s.benign.certifiedExact + s.benign.certifiedBounded;
        const std::size_t decided = s.observable.decided() + s.hidden.decided() + s.benign.decided();
        const std::size_t total = s.observable.total() + s.hidden.total() + s.benign.total();
        const bool ours = s.name == "temper";
        table.rows.push_back(
            {ours ? bold() + teal() + "temper " + glyphs().dot + reset() : s.name,
             share(s.observable.refuted, s.observable.total()),
             share(benignCertified, s.benign.total()),
             std::to_string(s.hidden.refuted) + "/" +
                 std::to_string(s.hidden.certifiedExact + s.hidden.certifiedBounded) + "/" +
                 std::to_string(s.hidden.undecided),
             (ours ? bold() : std::string()) + share(decided, total) + reset(),
             fixed(total ? static_cast<double>(s.evaluations) / total : 0.0, 1),
             fixed(total ? s.milliseconds / total : 0.0, 2),
             (s.violations ? bad() : good()) + std::to_string(s.violations) + reset()});
    }
    table.print();
    std::cout << "\n  " << muted() << "Guardian-APC+ alone misses " << reset() << bold()
              << t.guardianMissed << reset() << muted() << " observable faults; Temper refutes "
              << reset() << bold() << t.guardianMissedRefuted << reset() << muted() << " of them. "
              << reset() << bold() << t.oracleMissedRefuted << reset() << muted()
              << " faults were exposed by no oracle input, only by a certificate-guided witness."
              << reset() << "\n  " << muted()
              << "unsound = CERTIFIED cases later exposed by any input (must be 0)." << reset()
              << "\n";
}

inline void compressionStudy(const modelforge::CompressionStudySummary& c) {
    using namespace term;
    rule("Temper: compression severity brackets on real models");
    std::map<std::string, std::vector<const modelforge::CompressionStudySummary::Row*>> byModel;
    for (const auto& row : c.rows) byModel[row.model].push_back(&row);
    Table table;
    table.headers = {"model"};
    table.rightAlign = {false};
    for (const auto& compression : modelforge::standardCompressions()) {
        table.headers.push_back(compression.name);
        table.rightAlign.push_back(false);
    }
    table.headers.push_back("certified pick");
    table.headers.push_back("search-only pick");
    table.rightAlign.push_back(false);
    table.rightAlign.push_back(false);
    for (const auto& [model, rows] : byModel) {
        std::vector<std::string> line = {model};
        for (const auto* row : rows) {
            const auto& a = row->audit;
            const double found = a.search.witness.empty() ? 0.0 : a.search.severity;
            const std::string colour = a.verdict == modelforge::Verdict::Certified ? good()
                                       : a.verdict == modelforge::Verdict::Refuted ? bad()
                                                                                    : warn();
            line.push_back(colour + fixed(found, 2) + ".." + fixed(a.certificate.bound, 2) + reset());
        }
        const auto pick = [&](const std::map<std::string, std::string>& choices) {
            const auto found = choices.find(model);
            return found == choices.end() ? std::string("float32") : found->second;
        };
        line.push_back(bold() + pick(c.certifiedChoice) + reset());
        line.push_back(pick(c.empiricalChoice));
        table.rows.push_back(std::move(line));
    }
    table.print();
    std::cout << "\n  " << muted() << "cell = found..proven worst-flip margin; " << reset() << good()
              << "certified" << reset() << " " << c.verdicts.certifiedBounded << ", " << bad()
              << "refuted" << reset() << " " << c.verdicts.refuted << ", " << warn() << "undecided"
              << reset() << " " << c.verdicts.undecided << muted() << " of " << c.cases
              << "; bracket violations " << reset()
              << (c.bracketViolations ? bad() : good()) << c.bracketViolations << reset() << "\n";
}

inline void binaryStudy(const modelforge::BinaryStudySummary& b) {
    using namespace term;
    rule("Temper: generated binaries vs the certified model");
    if (!b.available) {
        std::cout << "  " << warn() << "no C++ compiler found; binary study skipped" << reset() << "\n";
        return;
    }
    std::map<std::string, std::array<double, 4>> perFlags;  // models, bit-exact, max err, flips
    std::vector<std::string> order;
    for (const auto& row : b.rows) {
        for (const auto& r : row.report.results) {
            if (!perFlags.count(r.flags)) {
                perFlags[r.flags] = {0, 0, 0, 0};
                order.push_back(r.flags);
            }
            auto& v = perFlags[r.flags];
            v[0] += r.built ? 1 : 0;
            v[1] += r.built && r.maximumAbsoluteError == 0 && r.predictionMismatches == 0 ? 1 : 0;
            v[2] = std::max(v[2], r.maximumAbsoluteError);
            v[3] += static_cast<double>(r.predictionMismatches);
        }
    }
    Table table;
    table.headers = {"flags (" + b.compiler + ")", "models built", "bit-exact", "max |diff|",
                     "prediction changes"};
    table.rightAlign = {false, true, true, true, true};
    for (const auto& flags : order) {
        const auto& v = perFlags[flags];
        table.rows.push_back({flags, fixed(v[0], 0), (v[1] == v[0] ? good() : warn()) + fixed(v[1], 0) + reset(),
                              v[2] == 0 ? std::string("0") : fixed(v[2] * 1e6, 3) + "e-6",
                              (v[3] > 0 ? bad() : good()) + fixed(v[3], 0) + reset()});
    }
    table.print();
}

}  // namespace render

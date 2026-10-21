#pragma once

// Terminal rendering of a Temper deployment dossier, shared by `temper` and
// the studio.

#include "dossier.h"
#include "term.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace view {

using namespace term;
namespace mf = modelforge;

constexpr std::size_t kWidth = 92;

inline std::string vec(const std::vector<float>& values, std::size_t limit = 6, int digits = 3) {
    std::string text;
    for (std::size_t i = 0; i < values.size() && i < limit; ++i) {
        text += (i ? ", " : "") + fixed(values[i], digits);
    }
    if (values.size() > limit) text += ", ...";
    return "[" + text + "]";
}

// [found ... certified] bracket on the 0..1 margin scale.
inline std::string bracketBar(double found, double certified, bool foundAny, std::size_t width = 24) {
    const auto at = [&](double v) {
        return static_cast<std::size_t>(std::round(std::clamp(v, 0.0, 1.0) * (width - 1)));
    };
    const std::size_t f = foundAny ? at(found) : 0;
    const std::size_t c = at(certified);
    const Glyphs& g = glyphs();
    std::string bar;
    for (std::size_t i = 0; i < width; ++i) {
        if (foundAny && i == f) {
            bar += bad() + (settings().unicode ? "●" : "o") + reset();
        } else if (i == c) {
            bar += good() + (settings().unicode ? "┃" : "|") + reset();
        } else if (i > f && i < c) {
            bar += warn() + (settings().unicode ? "━" : "=") + reset();
        } else {
            bar += rgb(51, 65, 85) + g.h + reset();
        }
    }
    return bar;
}

inline std::string riskColour(const std::string& risk) {
    if (risk == "negligible" || risk == "none found") return good();
    if (risk == "low") return teal();
    if (risk == "moderate") return warn();
    return bad();
}

inline void renderDossier(const mf::Dossier& d, const mf::DossierOptions& options) {
    box("Model",
        {muted() + "model        " + reset() + bold() + d.model + reset() + muted() + "  (" +
             options.modelPath + ")" + reset(),
         muted() + "shape        " + reset() + std::to_string(d.inputs) + " inputs " + glyphs().arrow +
             " " + std::to_string(d.outputs) + " outputs" + muted() + "   parameters " + reset() +
             std::to_string(d.parameters) + muted() + "   multiply-adds " + reset() +
             std::to_string(d.multiplyAdds),
         muted() + "domain       " + reset() + d.domainSource,
         muted() + "fingerprint  " + reset() + d.fingerprint +
             muted() + "  (canonical graph that ships)" + reset()},
        kWidth);

    rule("1  Graph rewrites: falsify or certify each pass", kWidth);
    Table passes;
    passes.headers = {"pass", "changed", "verdict", "certificate", "decided by", "evals", "ms"};
    passes.rightAlign = {false, false, false, false, false, true, true};
    for (const auto& p : d.passes) {
        passes.rows.push_back(
            {p.pass, p.changed ? std::to_string(p.events.size()) + " edits" : muted() + "no" + reset(),
             verdictBadge(mf::verdictName(p.decision.verdict)),
             mf::certificateKindName(p.decision.kind), p.decision.decidedBy,
             std::to_string(p.decision.evaluations), fixed(p.decision.milliseconds, 1)});
    }
    passes.print();
    for (const auto& p : d.passes) {
        if (p.decision.witness) {
            std::cout << "\n  " << bad() << glyphs().cross << reset() << " " << p.pass
                      << " counterexample " << vec(*p.decision.witness) << "\n    original  "
                      << vec(p.decision.originalOutput, 6, 6) << "\n    rewritten "
                      << vec(p.decision.candidateOutput, 6, 6) << "\n";
            if (p.decision.divergence.found) {
                std::cout << "    first diverges at " << bold() << p.decision.divergence.node
                          << reset() << "\n";
            }
        }
    }

    rule("2  Domain specialisation: neurons proven dead on the domain", kWidth);
    const auto& s = d.specialization;
    std::cout << "  " << s.hiddenUnits << " hidden units analysed: " << bold() << s.deadUnits
              << reset() << " provably dead (removed), " << s.alwaysActiveUnits
              << " provably always active.  Parameters " << s.parametersBefore << " "
              << glyphs().arrow << " " << s.parametersAfter << "; masked model "
              << verdictBadge(mf::verdictName(s.certificate.verdict)) << " "
              << mf::certificateKindName(s.certificate.kind) << ".\n";

    if (!d.compressions.empty()) {
        rule("3  Compression audit: worst realistic flip, found vs proven", kWidth);
        Table table;
        table.headers = {"compression", "size", "accuracy", "test flips", "found",
                         "proven <=", "0 . . . margin . . . 1", "verdict"};
        table.rightAlign = {false, true, true, true, true, true, false, false};
        for (const auto& a : d.compressions) {
            const bool foundAny = !a.search.witness.empty();
            table.rows.push_back(
                {a.compression, percent(a.weightSize, 0), percent(a.accuracyAfter, 1),
                 std::to_string(a.testFlips),
                 foundAny ? riskColour(a.foundRisk) + fixed(a.search.severity, 3) + reset()
                          : good() + "none" + reset(),
                 riskColour(a.certifiedRisk) + fixed(a.certificate.bound, 3) + reset(),
                 bracketBar(a.search.severity, a.certificate.bound, foundAny),
                 verdictBadge(mf::verdictName(a.verdict))});
        }
        table.print();
        std::cout << "\n  " << muted()
                  << "found: the original's top-two margin at the worst disagreement found near real"
                  << "\n  data; proven: no disagreement anywhere in that region when the original is"
                  << "\n  more confident than this. CERTIFIED = proven low risk (< 0.2)." << reset()
                  << "\n\n  Certified recommendation: " << bold()
                  << (d.recommendedCertified.empty() ? std::string("none (keep float32)")
                                                     : d.recommendedCertified)
                  << reset() << muted() << "    (test-and-search alone would pick: "
                  << (d.recommendedEmpirical.empty() ? "none" : d.recommendedEmpirical) << ")"
                  << reset() << "\n";
    }

    rule("4  Binary audit: the compiled program vs the certified model", kWidth);
    if (!d.binary.available) {
        std::cout << "  " << warn() << "no C++ compiler found" << reset()
                  << " (set CXX); skipped. Generated code: " << d.generatedDirectory << "\n";
    } else {
        Table table;
        table.headers = {"compiler flags", "built", "inputs", "max |score diff|", "prediction changes"};
        table.rightAlign = {false, false, true, true, true};
        for (const auto& r : d.binary.results) {
            table.rows.push_back(
                {r.flags, r.built ? good() + glyphs().check + reset() : bad() + glyphs().cross + reset(),
                 std::to_string(r.rows),
                 (r.maximumAbsoluteError == 0 ? good() + std::string("0 (bit-exact)")
                                              : warn() + fixed(r.maximumAbsoluteError * 1e6, 3) +
                                                    "e-6") + reset(),
                 (r.predictionMismatches ? bad() : good()) + std::to_string(r.predictionMismatches) +
                     reset()});
        }
        table.print();
    }

    std::vector<std::string> lines;
    lines.push_back(d.ship ? bold() + good() + glyphs().check + "  SHIP: every kept rewrite is certified and the binary reproduces it" + reset()
                           : bold() + bad() + glyphs().cross + "  DO NOT SHIP YET" + reset());
    for (const auto& b : d.blockers) lines.push_back(bad() + "  blocker: " + reset() + b);
    for (const auto& n : d.notes) lines.push_back(warn() + "  note:    " + reset() + n);
    box("Verdict", lines, kWidth, d.ship ? good() : bad());
    std::cout << "  " << muted() << "C++ with certified-domain guard: " << reset()
              << d.generatedDirectory << "\n  " << muted() << "dossier:  " << reset()
              << options.outputDirectory << "/dossier.json" << muted() << "   ("
              << fixed(d.seconds, 1) << " s)" << reset() << "\n";
}

}  // namespace view

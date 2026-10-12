#include "certify.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>

namespace modelforge {
namespace {

std::size_t layerCount(const DenseNetwork& network) { return network.layers.size(); }

// Which configuration slot an error source maps to (-1: not tunable).
// kind 0 = activation bits, 1 = weight bits.
struct Slot {
    int kind = -1;
    std::size_t index = 0;
};

Slot slotOf(std::size_t source, std::size_t layers) {
    if (source == 0) return {0, 0};
    if (source <= layers) return {1, source - 1};
    const std::size_t l = source - 1 - layers;
    if (l + 1 < layers) return {0, l + 1};
    return {};
}

int& slotValue(PrecisionConfig& config, const Slot& slot) {
    return slot.kind == 0 ? config.activationBits[slot.index] : config.weightBits[slot.index];
}

int nextRung(const std::vector<int>& ladder, int value, int direction) {
    if (direction > 0) {
        for (int rung : ladder) {
            if (rung > value) return rung;
        }
    } else {
        for (auto it = ladder.rbegin(); it != ladder.rend(); ++it) {
            if (*it < value) return *it;
        }
    }
    return 0;
}

double marginOf(const std::vector<std::int64_t>& logits, int cls) {
    double rival = -std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < logits.size(); ++j) {
        if (static_cast<int>(j) == cls) continue;
        // A lower-index tie wins in the integer program: count it as a loss.
        rival = std::max(rival, static_cast<double>(logits[j]) + (static_cast<int>(j) < cls ? 0.5 : 0.0));
    }
    return static_cast<double>(logits[static_cast<std::size_t>(cls)]) - rival;
}

std::string jsonNumber(double value) {
    if (!std::isfinite(value)) return "null";
    std::ostringstream text;
    text << std::setprecision(17) << value;
    return text.str();
}

void writeDoubles(std::ostream& out, const std::vector<double>& values) {
    out << "[";
    for (std::size_t i = 0; i < values.size(); ++i) out << (i ? ", " : "") << jsonNumber(values[i]);
    out << "]";
}

std::string escape(const std::string& text) {
    std::string result;
    for (char c : text) {
        if (c == '"' || c == '\\') result += '\\';
        result += c;
    }
    return result;
}

}  // namespace

Box domainFromData(const std::vector<std::vector<float>>& rows, double widen) {
    Box box;
    if (rows.empty()) return box;
    const std::size_t n = rows.front().size();
    box.lo.assign(n, std::numeric_limits<double>::infinity());
    box.hi.assign(n, -std::numeric_limits<double>::infinity());
    for (const auto& row : rows) {
        for (std::size_t i = 0; i < n && i < row.size(); ++i) {
            box.lo[i] = std::min(box.lo[i], static_cast<double>(row[i]));
            box.hi[i] = std::max(box.hi[i], static_cast<double>(row[i]));
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        const double spread = std::max(box.hi[i] - box.lo[i], 1e-3);
        box.lo[i] -= widen * spread;
        box.hi[i] += widen * spread;
    }
    return box;
}

std::vector<CertRegion> regionsAroundRows(const DenseNetwork& network,
                                          const std::vector<std::vector<float>>& rows,
                                          double eps, const Box& domain, std::size_t limit) {
    std::vector<CertRegion> regions;
    const std::size_t count = std::min(limit, rows.size());
    for (std::size_t k = 0; k < count; ++k) {
        const std::size_t index = rows.size() <= limit ? k : k * rows.size() / count;
        CertRegion region;
        region.name = "row" + std::to_string(index);
        const auto& row = rows[index];
        region.box.lo.resize(row.size());
        region.box.hi.resize(row.size());
        for (std::size_t i = 0; i < row.size(); ++i) {
            region.box.lo[i] = std::max(domain.lo[i], std::min(domain.hi[i], row[i] - eps));
            region.box.hi[i] = std::min(domain.hi[i], std::max(domain.lo[i], row[i] + eps));
        }
        region.refClass = argmaxFirst(referenceLogits(network, region.box.center()));
        regions.push_back(std::move(region));
    }
    return regions;
}

RegionProof proveRegion(const DenseNetwork& network, const QProgram& program,
                        const CertRegion& region, const ProofOptions& options) {
    RegionProof proof;
    bool failed = false;
    std::function<bool(const Box&, int)> prove = [&](const Box& box, int depth) -> bool {
        ++proof.analyses;
        const auto analysis = analyzeRegion(network, program, box, options.analysis, region.refClass);
        const bool ok = options.referenceOnly ? analysis.refCertified : analysis.quantCertified;
        if (ok) {
            proof.tree.push_back({-1, 0.0});
            ++proof.leaves;
            return true;
        }
        if (depth >= options.maxDepth || proof.analyses + 2 > options.maxAnalyses) {
            failed = true;
            proof.blame = analysis.blame;
            proof.descent = analysis.descentDirection;
            proof.failingLeaf = box;
            return false;
        }
        std::size_t dim = 0;
        double best = -1.0;
        for (std::size_t i = 0; i < box.size(); ++i) {
            const double width = box.hi[i] - box.lo[i];
            if (width <= 0.0) continue;
            const double score = analysis.splitScore[i] > 0 ? analysis.splitScore[i] : 1e-12 * width;
            if (score > best) {
                best = score;
                dim = i;
            }
        }
        if (best < 0.0) {  // a point box that cannot be proved
            failed = true;
            proof.blame = analysis.blame;
            proof.descent = analysis.descentDirection;
            proof.failingLeaf = box;
            return false;
        }
        const double at = 0.5 * (box.lo[dim] + box.hi[dim]);
        proof.tree.push_back({static_cast<int>(dim), at});
        Box low = box, high = box;
        low.hi[dim] = at;
        high.lo[dim] = at;
        return prove(low, depth + 1) && prove(high, depth + 1);
    };
    proof.proved = prove(region.box, 0) && !failed;
    if (!proof.proved) proof.tree.clear();
    return proof;
}

std::optional<Counterexample> falsifyRegion(const DenseNetwork& network, const QProgram& program,
                                            const CertRegion& region, const std::vector<double>& descent,
                                            std::size_t budget, std::uint32_t seed) {
    const Box& box = region.box;
    const std::size_t n = box.size();
    const int cls = region.refClass;
    std::mt19937 rng(seed);
    std::size_t evaluations = 0;
    std::optional<Counterexample> found;
    std::vector<double> best;
    double bestMargin = std::numeric_limits<double>::infinity();
    const auto evaluate = [&](std::vector<double> x) {
        if (found || evaluations >= budget) return;
        for (std::size_t i = 0; i < n; ++i) x[i] = std::max(box.lo[i], std::min(box.hi[i], x[i]));
        ++evaluations;
        const auto run = runInteger(program, x);
        const double margin = marginOf(run.logits, cls);
        if (run.decision != cls && argmaxFirst(referenceLogits(network, x)) == cls) {
            found = Counterexample{region.name, x, cls, run.decision, evaluations};
            return;
        }
        if (margin < bestMargin) {
            bestMargin = margin;
            best = x;
        }
    };
    const auto center = box.center();
    evaluate(center);
    if (descent.size() == n) {
        std::vector<double> vertex(n);
        for (std::size_t i = 0; i < n; ++i) {
            vertex[i] = center[i] + descent[i] * 0.5 * (box.hi[i] - box.lo[i]);
        }
        for (double t : {1.0, 0.75, 0.5, 0.25}) {
            std::vector<double> x(n);
            for (std::size_t i = 0; i < n; ++i) x[i] = center[i] + t * (vertex[i] - center[i]);
            evaluate(x);
        }
    }
    std::normal_distribution<double> gaussian(0.0, 1.0);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    double step = 0.5;
    while (!found && evaluations < budget) {
        std::vector<double> x(n);
        if (unit(rng) < 0.15) {  // restart: a random vertex-biased point
            for (std::size_t i = 0; i < n; ++i) {
                const double u = unit(rng);
                x[i] = box.lo[i] + (u < 0.3 ? 0.0 : u > 0.7 ? 1.0 : unit(rng)) * (box.hi[i] - box.lo[i]);
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                x[i] = best[i] + step * gaussian(rng) * 0.5 * (box.hi[i] - box.lo[i]);
            }
        }
        const double before = bestMargin;
        evaluate(x);
        step = bestMargin < before ? std::min(1.0, step * 1.5) : std::max(0.02, step * 0.85);
    }
    if (found) found->evaluations = evaluations;
    return found;
}

std::vector<int> weightLadder(CostModel cost) {
    return cost == CostModel::Bytes ? std::vector<int>{4, 8, 16}
                                    : std::vector<int>{2, 3, 4, 5, 6, 7, 8, 10, 12, 16};
}

std::vector<int> activationLadder(CostModel cost) {
    return cost == CostModel::Bytes ? std::vector<int>{8, 16}
                                    : std::vector<int>{4, 5, 6, 7, 8, 10, 12, 16};
}

double configCost(const QProgram& program, CostModel cost) {
    const auto c = programCost(program);
    if (cost == CostModel::Bytes) return c.flashBytes + c.ramBytes;
    double activationBits = program.inputs * program.inputBits;
    for (std::size_t l = 0; l + 1 < program.layers.size(); ++l) {
        activationBits += program.layers[l].out * program.layers[l].outBits;
    }
    return (c.idealBits + activationBits) / 8.0;
}

std::vector<bool> referenceRobustRegions(const DenseNetwork& network, const Box& domain,
                                         const std::vector<CertRegion>& regions,
                                         const ProofOptions& options) {
    std::string error;
    const auto program = lowerToInteger(network, domain, uniformConfig(network.layers.size(), 16, 16), error);
    std::vector<bool> robust(regions.size(), false);
    if (!program) return robust;
    ProofOptions reference = options;
    reference.referenceOnly = true;
    for (std::size_t r = 0; r < regions.size(); ++r) {
        robust[r] = proveRegion(network, *program, regions[r], reference).proved;
    }
    return robust;
}

namespace {

struct Evaluation {
    std::vector<RegionStatus> status;
    std::vector<double> blame;  // aggregated, normalised per failing region
    std::size_t failing = 0;
    std::size_t analyses = 0;
    std::size_t falsifierCalls = 0;
    std::size_t falsifierHits = 0;
    std::vector<Counterexample> counterexamples;
};

Evaluation evaluateProgram(const DenseNetwork& network, const QProgram& program,
                           const std::vector<CertRegion>& regions, const std::vector<bool>& robust,
                           const ProofOptions& proofOptions, std::size_t falsifyBudget,
                           const std::vector<std::size_t>& order, bool stopAtFirstFailure,
                           std::uint32_t seed) {
    Evaluation evaluation;
    const std::size_t sources = errorSourceCount(network.layers.size());
    evaluation.blame.assign(sources, 0.0);
    evaluation.status.resize(regions.size());
    for (std::size_t r = 0; r < regions.size(); ++r) {
        evaluation.status[r].name = regions[r].name;
        evaluation.status[r].excluded = !robust[r];
    }
    for (std::size_t r : order) {
        if (!robust[r]) continue;
        auto& status = evaluation.status[r];
        std::vector<double> blame;
        bool failed = false;
        if (falsifyBudget > 0) {
            // Cheap concrete search first: a real counterexample settles the
            // region without an expensive proof attempt.
            const auto root = analyzeRegion(network, program, regions[r].box, proofOptions.analysis, regions[r].refClass);
            ++evaluation.analyses;
            if (root.quantCertified) {
                status.proved = true;
                status.proof.proved = true;
                status.proof.tree = {{-1, 0.0}};
                status.proof.analyses = status.proof.leaves = 1;
                continue;
            }
            ++evaluation.falsifierCalls;
            const auto cex = falsifyRegion(network, program, regions[r], root.descentDirection,
                                           falsifyBudget, seed + static_cast<std::uint32_t>(r));
            if (cex) {
                ++evaluation.falsifierHits;
                status.counterexample = true;
                evaluation.counterexamples.push_back(*cex);
                blame = root.blame;
                failed = true;
            }
        }
        if (!failed) {
            status.proof = proveRegion(network, program, regions[r], proofOptions);
            evaluation.analyses += status.proof.analyses;
            status.proved = status.proof.proved;
            if (!status.proved) {
                failed = true;
                blame = status.proof.blame;
            }
        }
        if (failed) {
            ++evaluation.failing;
            const double sum = std::accumulate(blame.begin(), blame.end(), 0.0);
            for (std::size_t s = 0; s < sources && s < blame.size(); ++s) {
                evaluation.blame[s] += sum > 0 ? blame[s] / sum : 0.0;
            }
            if (stopAtFirstFailure) break;
        }
    }
    return evaluation;
}

void summarise(TunerResult& result) {
    result.proved = result.excluded = result.unproved = 0;
    for (const auto& status : result.regions) {
        if (status.excluded) {
            ++result.excluded;
        } else if (status.proved) {
            ++result.proved;
        } else {
            ++result.unproved;
        }
    }
}

}  // namespace

TunerResult checkConfiguration(const DenseNetwork& network, const Box& domain,
                               const std::vector<CertRegion>& regions, const PrecisionConfig& config,
                               const ProofOptions& proof, const std::vector<bool>* robustIn,
                               std::size_t falsifyBudget, std::uint32_t seed) {
    const auto start = std::chrono::steady_clock::now();
    TunerResult result;
    const auto robust = robustIn ? *robustIn : referenceRobustRegions(network, domain, regions, proof);
    auto program = lowerToInteger(network, domain, config, result.error, proof.analysis);
    if (!program) return result;
    result.lowerings = 1;
    std::vector<std::size_t> order(regions.size());
    std::iota(order.begin(), order.end(), 0);
    auto evaluation = evaluateProgram(network, *program, regions, robust, proof, 0, order, false, seed);
    result.analyses = evaluation.analyses;
    if (falsifyBudget > 0) {
        for (std::size_t r = 0; r < regions.size(); ++r) {
            auto& status = evaluation.status[r];
            if (status.excluded || status.proved) continue;
            ++result.falsifierCalls;
            const auto cex = falsifyRegion(network, *program, regions[r], status.proof.descent, falsifyBudget,
                                           seed + static_cast<std::uint32_t>(r));
            if (cex) {
                ++result.falsifierHits;
                status.counterexample = true;
                result.counterexamples.push_back(*cex);
            }
        }
    }
    result.regions = std::move(evaluation.status);
    result.config = config;
    result.program = std::move(*program);
    result.ok = true;
    summarise(result);
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
}

TunerResult tunePrecision(const DenseNetwork& network, const Box& domain,
                          const std::vector<CertRegion>& regions, const TunerOptions& options,
                          const std::vector<bool>* robustIn) {
    const auto start = std::chrono::steady_clock::now();
    TunerResult result;
    const std::size_t L = layerCount(network);
    const std::size_t sources = errorSourceCount(L);
    const auto wLadder = weightLadder(options.cost);
    const auto aLadder = activationLadder(options.cost);
    const auto robust = robustIn ? *robustIn : referenceRobustRegions(network, domain, regions, options.proof);
    std::mt19937 rng(options.seed);

    const auto lower = [&](const PrecisionConfig& config) {
        ++result.lowerings;
        std::string error;
        auto program = lowerToInteger(network, domain, config, error, options.proof.analysis);
        if (!program) result.error = error;
        return program;
    };
    const auto ladderFor = [&](const Slot& slot) -> const std::vector<int>& {
        return slot.kind == 0 ? aLadder : wLadder;
    };

    PrecisionConfig config = uniformConfig(L, wLadder.front(), aLadder.front());
    std::vector<std::size_t> order(regions.size());
    std::iota(order.begin(), order.end(), 0);
    std::optional<QProgram> program;
    bool allProved = false;
    for (std::size_t iteration = 0; iteration < options.maxIterations; ++iteration) {
        program = lower(config);
        if (!program) return result;
        ++result.iterations;
        const auto evaluation = evaluateProgram(network, *program, regions, robust, options.proof,
                                                options.falsifyFirst ? options.falsifyBudget : 0, order,
                                                false, options.seed + static_cast<std::uint32_t>(iteration * 7919));
        result.analyses += evaluation.analyses;
        result.falsifierCalls += evaluation.falsifierCalls;
        result.falsifierHits += evaluation.falsifierHits;
        for (const auto& cex : evaluation.counterexamples) {
            if (result.counterexamples.size() < 32) result.counterexamples.push_back(cex);
        }
        const double cost = configCost(*program, options.cost);
        std::ostringstream line;
        line << "iteration " << iteration << ": " << config.describe() << "  cost " << std::fixed
             << std::setprecision(1) << cost << " B  failing " << evaluation.failing << "  (counterexamples "
             << evaluation.falsifierHits << ")";
        if (evaluation.failing == 0) {
            result.log.push_back(line.str() + "  -> all regions proved");
            allProved = true;
            break;
        }
        // Failing regions first next time: they decide the next step.
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            const auto& sa = evaluation.status[a];
            const auto& sb = evaluation.status[b];
            return (!sa.proved && !sa.excluded) > (!sb.proved && !sb.excluded);
        });

        std::vector<std::size_t> bumpable;
        std::vector<double> delta(sources, 0.0);
        for (std::size_t s = 0; s < sources; ++s) {
            const Slot slot = slotOf(s, L);
            if (slot.kind < 0) continue;
            PrecisionConfig trial = config;
            const int next = nextRung(ladderFor(slot), slotValue(trial, slot), +1);
            if (next == 0) continue;
            slotValue(trial, slot) = next;
            if (options.strategy == RefineStrategy::Blame) {
                const auto trialProgram = lower(trial);
                if (!trialProgram) continue;
                delta[s] = std::max(1.0, configCost(*trialProgram, options.cost) - cost);
            }
            bumpable.push_back(s);
        }
        if (bumpable.empty()) {
            result.log.push_back(line.str() + "  -> no precision left to add");
            break;
        }
        std::vector<std::size_t> chosen;
        if (options.strategy == RefineStrategy::Uniform) {
            chosen = bumpable;
        } else if (options.strategy == RefineStrategy::RandomSource) {
            chosen = {bumpable[std::uniform_int_distribution<std::size_t>(0, bumpable.size() - 1)(rng)]};
        } else {
            std::size_t best = bumpable.front();
            double bestScore = -1.0;
            for (std::size_t s : bumpable) {
                const double score = evaluation.blame[s] / delta[s];
                if (score > bestScore) {
                    bestScore = score;
                    best = s;
                }
            }
            if (bestScore <= 0.0) {
                // No provenance signal (e.g. only the logit rounding is blamed):
                // fall back to the cheapest step.
                best = *std::min_element(bumpable.begin(), bumpable.end(),
                                         [&](std::size_t a, std::size_t b) { return delta[a] < delta[b]; });
            }
            chosen = {best};
        }
        std::ostringstream raised;
        for (std::size_t s : chosen) {
            const Slot slot = slotOf(s, L);
            slotValue(config, slot) = nextRung(ladderFor(slot), slotValue(config, slot), +1);
            raised << (raised.tellp() > 0 ? ", " : "") << errorSourceName(s, L);
        }
        result.log.push_back(line.str() + "  -> raise " + raised.str());
    }

    if (allProved && options.minimize) {
        // Undo any refinement that turned out to be unnecessary.
        bool changed = true;
        while (changed) {
            changed = false;
            const double cost = configCost(*program, options.cost);
            std::vector<std::pair<double, std::size_t>> candidates;
            for (std::size_t s = 0; s < sources; ++s) {
                const Slot slot = slotOf(s, L);
                if (slot.kind < 0) continue;
                PrecisionConfig trial = config;
                const int previous = nextRung(ladderFor(slot), slotValue(trial, slot), -1);
                if (previous == 0) continue;
                slotValue(trial, slot) = previous;
                const auto trialProgram = lower(trial);
                if (!trialProgram) continue;
                candidates.push_back({cost - configCost(*trialProgram, options.cost), s});
            }
            std::sort(candidates.rbegin(), candidates.rend());
            for (const auto& candidate : candidates) {
                if (candidate.first <= 0.0) continue;
                const Slot slot = slotOf(candidate.second, L);
                PrecisionConfig trial = config;
                slotValue(trial, slot) = nextRung(ladderFor(slot), slotValue(trial, slot), -1);
                auto trialProgram = lower(trial);
                if (!trialProgram) continue;
                const auto evaluation = evaluateProgram(network, *trialProgram, regions, robust, options.proof,
                                                        0, order, true, options.seed);
                result.analyses += evaluation.analyses;
                if (evaluation.failing == 0) {
                    result.log.push_back("minimise: lower " + errorSourceName(candidate.second, L) + " -> " +
                                         trial.describe());
                    config = trial;
                    program = std::move(trialProgram);
                    changed = true;
                    break;
                }
            }
        }
    }

    // Final proofs (the certificate) and a deeper search on what is left.
    std::iota(order.begin(), order.end(), 0);
    auto final = evaluateProgram(network, *program, regions, robust, options.proof, 0, order, false, options.seed);
    result.analyses += final.analyses;
    for (std::size_t r = 0; r < regions.size(); ++r) {
        auto& status = final.status[r];
        if (status.excluded || status.proved) continue;
        ++result.falsifierCalls;
        const auto cex = falsifyRegion(network, *program, regions[r], status.proof.descent, 400,
                                       options.seed + static_cast<std::uint32_t>(r));
        if (cex) {
            ++result.falsifierHits;
            status.counterexample = true;
            result.counterexamples.push_back(*cex);
        }
    }
    result.regions = std::move(final.status);
    result.config = config;
    result.program = std::move(*program);
    result.ok = true;
    summarise(result);
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
}

bool writeCertificate(const std::string& path, const std::string& modelPath, const DenseNetwork& network,
                      const std::vector<CertRegion>& regions, const TunerResult& result, double eps,
                      std::string& error) {
    std::ofstream out(path);
    if (!out) {
        error = "cannot write " + path;
        return false;
    }
    const auto& program = result.program;
    const auto global = analyzeRegion(network, program, program.domain);
    out << "{\n  \"format\": \"forgecert-1\",\n  \"author\": \"Vaishnavi\",\n"
        << "  \"model\": \"" << escape(modelPath) << "\",\n  \"artifact\": \"model_q.c\",\n"
        << "  \"precision\": \"" << program.config.describe() << "\",\n"
        << "  \"semantics\": \"on every proved region the emitted integer C program returns the argmax of "
           "the float model evaluated in exact real arithmetic\",\n"
        << "  \"domain\": {\"lo\": ";
    writeDoubles(out, program.domain.lo);
    out << ", \"hi\": ";
    writeDoubles(out, program.domain.hi);
    out << "},\n  \"claims\": {\n    \"no_accumulator_overflow_on_domain\": true,\n"
        << "    \"no_saturation_on_domain\": true,\n    \"accumulator_bits\": [";
    for (std::size_t l = 0; l < program.layers.size(); ++l) out << (l ? ", " : "") << program.layers[l].accBits;
    out << "],\n    \"max_abs_logit_error_on_domain\": ";
    std::vector<double> claimed;
    for (double v : global.logitErrorBound) claimed.push_back(v * (1.0 + 1e-6) + 1e-12);
    writeDoubles(out, claimed);
    out << "\n  },\n  \"summary\": {\"regions\": " << result.regions.size() << ", \"proved\": " << result.proved
        << ", \"excluded_reference_not_robust\": " << result.excluded << ", \"unproved\": " << result.unproved
        << ", \"unproved_with_counterexample\": "
        << std::count_if(result.regions.begin(), result.regions.end(),
                         [](const RegionStatus& s) { return !s.excluded && !s.proved && s.counterexample; })
        << "},\n  \"region_radius\": " << jsonNumber(eps) << ",\n  \"regions\": [";
    bool first = true;
    for (std::size_t r = 0; r < result.regions.size() && r < regions.size(); ++r) {
        const auto& status = result.regions[r];
        if (!status.proved) continue;
        out << (first ? "\n" : ",\n") << "    {\"name\": \"" << escape(status.name)
            << "\", \"class\": " << regions[r].refClass << ", \"lo\": ";
        first = false;
        writeDoubles(out, regions[r].box.lo);
        out << ", \"hi\": ";
        writeDoubles(out, regions[r].box.hi);
        out << ", \"tree\": \"";
        for (std::size_t k = 0; k < status.proof.tree.size(); ++k) {
            const auto& node = status.proof.tree[k];
            if (k) out << ' ';
            if (node.dim < 0) {
                out << 'L';
            } else {
                out << 'S' << node.dim << ':' << jsonNumber(node.at);
            }
        }
        out << "\"}";
    }
    out << "\n  ]\n}\n";
    return static_cast<bool>(out);
}

}  // namespace modelforge

#include "certify.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <atomic>
#include <thread>

namespace modelforge {
namespace {

std::size_t layerCount(const DenseNetwork& network) { return network.layers.size(); }

// Which configuration slot an error source maps to (-1: not tunable).
// kind 0 = activation bits, 1 = weight bits.
struct Slot {
    int kind = -1;
    std::size_t index = 0;
};

Slot slotOf(std::size_t source, const DenseNetwork& network) {
    const std::size_t layers = network.layers.size();
    Slot slot;
    if (source == 0) {
        slot = {0, 0};
    } else if (source <= layers) {
        slot = {1, source - 1};
    } else if (source - layers < layers) {
        slot = {0, source - layers};  // output rounding of layer l: activation slot l + 1
    }
    if (slot.kind >= 0 && fixedSlot(network, slot.kind, slot.index)) return {};
    return slot;
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

// Smallest rung whose C storage type is wider (weights: <=4 packed, <=8 int8,
// int16; activations: <=8 one byte, two bytes).
int nextStorageClass(const std::vector<int>& ladder, int value, int kind) {
    const auto storage = [kind](int bits) { return kind == 1 ? (bits <= 4 ? 0 : bits <= 8 ? 1 : 2) : (bits <= 8 ? 0 : 1); };
    for (int rung : ladder) {
        if (rung > value && storage(rung) > storage(value)) return rung;
    }
    return 0;
}

// Largest rung whose C storage type is narrower.
int previousStorageClass(const std::vector<int>& ladder, int value, int kind) {
    const auto storage = [kind](int bits) { return kind == 1 ? (bits <= 4 ? 0 : bits <= 8 ? 1 : 2) : (bits <= 8 ? 0 : 1); };
    int best = 0;
    for (int rung : ladder) {
        if (rung < value && storage(rung) < storage(value)) best = rung;
    }
    return best;
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
    AnalysisOptions fast = options.analysis;
    fast.provenance = false;  // blame is only needed where the proof fails
    const auto blameAt = [&](const Box& box) {
        AnalysisOptions withBlame = options.analysis;
        withBlame.provenance = true;
        return analyzeRegion(network, program, box, withBlame, region.refClass).blame;
    };
    std::function<bool(const Box&, int)> prove = [&](const Box& box, int depth) -> bool {
        ++proof.analyses;
        const auto analysis = analyzeRegion(network, program, box, fast, region.refClass);
        const bool ok = options.referenceOnly ? analysis.refCertified : analysis.quantCertified;
        if (ok) {
            proof.tree.push_back({-1, 0.0});
            ++proof.leaves;
            return true;
        }
        if (depth >= options.maxDepth || proof.analyses + 2 > options.maxAnalyses) {
            failed = true;
            if (!options.referenceOnly) proof.blame = blameAt(box);
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
            if (!options.referenceOnly) proof.blame = blameAt(box);
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
    // Latency uses the fine lattice: e.g. 6-bit weights x 10-bit activations
    // still give products that provably fit a 16-bit multiply.
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
    // Latency: AVR cycles, with bytes as a small tie-breaker.
    if (cost == CostModel::Latency) return estimatedAvrCycles(program) + 1e-3 * (c.flashBytes + c.ramBytes);
    double activationBits = program.inputs * program.inputBits;
    for (std::size_t l = 0; l + 1 < program.layers.size(); ++l) {
        activationBits += program.layers[l].out * program.layers[l].outBits;
    }
    return (c.idealBits + activationBits) / 8.0;
}

namespace {

// Runs body(r) for every index in `order` on all cores. Each region is
// independent; results go to per-region slots, so the outcome does not depend
// on the schedule (except which failure stops an early-exit run first).
void parallelFor(const std::vector<std::size_t>& order, const std::function<bool(std::size_t)>& body) {
    std::atomic<std::size_t> next{0};
    std::atomic<bool> stop{false};
    const std::size_t workers =
        std::max<std::size_t>(1, std::min<std::size_t>(std::thread::hardware_concurrency(), 8));
    const auto work = [&]() {
        for (;;) {
            const std::size_t k = next.fetch_add(1);
            if (k >= order.size() || stop.load()) return;
            if (!body(order[k])) stop.store(true);
        }
    };
    std::vector<std::thread> threads;
    for (std::size_t t = 1; t < workers; ++t) threads.emplace_back(work);
    work();
    for (auto& thread : threads) thread.join();
}

}  // namespace

std::vector<bool> referenceRobustRegions(const DenseNetwork& network, const Box& domain,
                                         const std::vector<CertRegion>& regions,
                                         const ProofOptions& options) {
    std::string error;
    const auto program = lowerToInteger(network, domain, uniformConfig(network.layers.size(), 16, 16), error);
    std::vector<bool> robust(regions.size(), false);
    if (!program) return robust;
    ProofOptions reference = options;
    reference.referenceOnly = true;
    std::vector<char> proved(regions.size(), 0);
    std::vector<std::size_t> order(regions.size());
    std::iota(order.begin(), order.end(), 0);
    parallelFor(order, [&](std::size_t r) {
        proved[r] = proveRegion(network, *program, regions[r], reference).proved;
        return true;
    });
    for (std::size_t r = 0; r < regions.size(); ++r) robust[r] = proved[r] != 0;
    return robust;
}

namespace {

struct Evaluation {
    std::vector<std::size_t> failedRegions;
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
    struct Slot {
        bool failed = false;
        std::vector<double> blame;
        std::size_t analyses = 0, falsifierCalls = 0, falsifierHits = 0;
        std::optional<Counterexample> counterexample;
    };
    std::vector<Slot> slots(regions.size());
    AnalysisOptions fast = proofOptions.analysis;
    fast.provenance = false;
    parallelFor(order, [&](std::size_t r) {
        if (!robust[r]) return true;
        auto& status = evaluation.status[r];
        auto& slot = slots[r];
        if (falsifyBudget > 0) {
            // Cheap concrete search first: a real counterexample settles the
            // region without an expensive proof attempt.
            const auto root = analyzeRegion(network, program, regions[r].box, fast, regions[r].refClass);
            ++slot.analyses;
            if (root.quantCertified) {
                status.proved = true;
                status.proof.proved = true;
                status.proof.tree = {{-1, 0.0}};
                status.proof.analyses = status.proof.leaves = 1;
                return true;
            }
            ++slot.falsifierCalls;
            const auto cex = falsifyRegion(network, program, regions[r], root.descentDirection, falsifyBudget,
                                           seed + static_cast<std::uint32_t>(r));
            if (cex) {
                ++slot.falsifierHits;
                status.counterexample = true;
                slot.counterexample = *cex;
                AnalysisOptions withBlame = proofOptions.analysis;
                withBlame.provenance = true;
                slot.blame = analyzeRegion(network, program, regions[r].box, withBlame, regions[r].refClass).blame;
                ++slot.analyses;
                slot.failed = true;
            }
        }
        if (!slot.failed) {
            status.proof = proveRegion(network, program, regions[r], proofOptions);
            slot.analyses += status.proof.analyses;
            status.proved = status.proof.proved;
            if (!status.proved) {
                slot.failed = true;
                slot.blame = status.proof.blame;
            }
        }
        return !(slot.failed && stopAtFirstFailure);
    });
    for (std::size_t r : order) {
        const auto& slot = slots[r];
        evaluation.analyses += slot.analyses;
        evaluation.falsifierCalls += slot.falsifierCalls;
        evaluation.falsifierHits += slot.falsifierHits;
        if (slot.counterexample) evaluation.counterexamples.push_back(*slot.counterexample);
        if (!slot.failed) continue;
        evaluation.failedRegions.push_back(r);
        ++evaluation.failing;
        const double sum = std::accumulate(slot.blame.begin(), slot.blame.end(), 0.0);
        for (std::size_t s = 0; s < sources && s < slot.blame.size(); ++s) {
            evaluation.blame[s] += sum > 0 ? slot.blame[s] / sum : 0.0;
        }
    }
    return evaluation;
}

// Deeper search on every region left unproved (in parallel).
void falsifyUnproved(const DenseNetwork& network, const QProgram& program, const std::vector<CertRegion>& regions,
                     std::vector<RegionStatus>& status, std::size_t budget, std::uint32_t seed, TunerResult& result) {
    std::vector<std::size_t> open;
    for (std::size_t r = 0; r < regions.size(); ++r) {
        if (!status[r].excluded && !status[r].proved) open.push_back(r);
    }
    std::vector<std::optional<Counterexample>> found(regions.size());
    parallelFor(open, [&](std::size_t r) {
        found[r] = falsifyRegion(network, program, regions[r], status[r].proof.descent, budget,
                                 seed + static_cast<std::uint32_t>(r));
        return true;
    });
    for (std::size_t r : open) {
        ++result.falsifierCalls;
        if (!found[r]) continue;
        ++result.falsifierHits;
        status[r].counterexample = true;
        result.counterexamples.push_back(*found[r]);
    }
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
        falsifyUnproved(network, *program, regions, evaluation.status, falsifyBudget, seed, result);
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
    // Proof budget while searching. Bisecting a box one input at a time pays
    // off in a few dimensions; for images it almost never does, so the search
    // rejects candidates after a few analyses. A region proved with a small
    // budget is also proved with the full one, which the final proofs use.
    const bool large = network.inputs > 100;
    ProofOptions search = options.proof;
    const std::size_t searchBudget = options.searchAnalyses > 0 ? options.searchAnalyses
                                     : large                      ? 12
                                                                  : options.proof.maxAnalyses;
    search.maxAnalyses = std::min(search.maxAnalyses, searchBudget);

    // Every log line is also streamed to stderr with --trace.
    const auto note = [&](const std::string& line) {
        result.log.push_back(line);
        if (options.trace) std::cerr << "      [tuner " << std::fixed << std::setprecision(0)
                                     << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                                     << " s] " << line << std::endl;
    };
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

    std::vector<std::size_t> order(regions.size());
    std::iota(order.begin(), order.end(), 0);
    // The search targets the regions that the most precise program on the
    // lattice proves: no configuration can do better than it, so the others
    // would only drive every width to the top. They are still attempted for
    // the certificate at the end.
    std::vector<bool> target = robust;
    // Only on large inputs: provability is not monotone in precision (a
    // mixed configuration can prove a region that uniform 16 bits cannot),
    // so small models, where climbing is cheap, keep every robust region.
    if (options.reachableTargets && large) {
        if (const auto top = lower(uniformConfig(L, wLadder.back(), aLadder.back()))) {
            const auto evaluation = evaluateProgram(network, *top, regions, robust, options.proof,
                                                    options.falsifyFirst ? options.falsifyBudget : 0, order, false,
                                                    options.seed);
            result.analyses += evaluation.analyses;
            for (std::size_t r : evaluation.failedRegions) target[r] = false;
            result.unreachable = evaluation.failedRegions.size();
            if (result.unreachable > 0) {
                note(std::to_string(result.unreachable) + " robust region(s) not provable even at " +
                                     uniformConfig(L, wLadder.back(), aLadder.back()).describe() +
                                     ": not targeted by the search");
            }
        }
    }

    PrecisionConfig config = uniformConfig(L, wLadder.front(), aLadder.front());
    std::optional<QProgram> program;
    bool allProved = false;
    for (std::size_t iteration = 0; iteration < options.maxIterations; ++iteration) {
        program = lower(config);
        if (!program) return result;
        ++result.iterations;
        const auto evaluation = evaluateProgram(network, *program, regions, target, search,
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
             << std::setprecision(1) << cost << (options.cost == CostModel::Latency ? " cycles" : " B")
             << "  failing " << evaluation.failing << "  (counterexamples "
             << evaluation.falsifierHits << ")";
        if (evaluation.failing == 0) {
            note(line.str() + "  -> all regions proved");
            allProved = true;
            break;
        }
        // Failing regions first next time: they decide the next step.
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            const auto& sa = evaluation.status[a];
            const auto& sb = evaluation.status[b];
            return (!sa.proved && !sa.excluded) > (!sb.proved && !sb.excluded);
        });

        std::vector<std::size_t> candidates;
        std::vector<PrecisionConfig> trials(sources);
        std::vector<double> delta(sources, 0.0);
        std::vector<char> lowered(sources, 1);
        for (std::size_t s = 0; s < sources; ++s) {
            const Slot slot = slotOf(s, network);
            if (slot.kind < 0) continue;
            trials[s] = config;
            const int next = nextRung(ladderFor(slot), slotValue(trials[s], slot), +1);
            if (next == 0) continue;
            slotValue(trials[s], slot) = next;
            candidates.push_back(s);
        }
        if (options.strategy == RefineStrategy::Blame) {
            // Cost of each one-step raise, lowered on all cores.
            parallelFor(candidates, [&](std::size_t s) {
                std::string error;
                const auto trialProgram = lowerToInteger(network, domain, trials[s], error, options.proof.analysis);
                lowered[s] = trialProgram.has_value();
                if (trialProgram) delta[s] = std::max(1.0, configCost(*trialProgram, options.cost) - cost);
                return true;
            });
            result.lowerings += candidates.size();
        }
        std::vector<std::size_t> bumpable;
        for (std::size_t s : candidates) {
            if (lowered[s]) bumpable.push_back(s);
        }
        if (bumpable.empty()) {
            note(line.str() + "  -> no precision left to add");
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
            const Slot slot = slotOf(s, network);
            slotValue(config, slot) = nextRung(ladderFor(slot), slotValue(config, slot), +1);
            raised << (raised.tellp() > 0 ? ", " : "") << errorSourceName(s, L);
        }
        note(line.str() + "  -> raise " + raised.str());
    }

    if (allProved && options.minimize) {
        // Local search around a certified configuration. Moves: one rung down or
        // up, or a jump to the next / previous storage class (packed 4-bit, int8,
        // int16 weights; 1- or 2-byte activations); with compound moves, also
        // every pair of such moves on two different sources (e.g. fewer weight
        // bits and more activation bits, which keeps 16-bit multiplies). Any
        // certified move that lowers the cost is taken, until none does.
        std::map<std::string, double> costCache;
        const auto costOf = [&](const PrecisionConfig& c) -> double {
            const auto key = c.describe();
            const auto hit = costCache.find(key);
            if (hit != costCache.end()) return hit->second;
            const auto p = lower(c);
            const double value = p ? configCost(*p, options.cost) : std::numeric_limits<double>::infinity();
            costCache[key] = value;
            return value;
        };
        // Costs of many candidates at once, lowered on all cores.
        const auto precompute = [&](const std::vector<PrecisionConfig>& candidates) {
            std::vector<PrecisionConfig> todo;
            std::map<std::string, bool> queued;
            for (const auto& c : candidates) {
                const auto key = c.describe();
                if (costCache.count(key) || queued.count(key)) continue;
                queued[key] = true;
                todo.push_back(c);
            }
            std::vector<double> values(todo.size(), std::numeric_limits<double>::infinity());
            std::vector<std::size_t> indices(todo.size());
            std::iota(indices.begin(), indices.end(), 0);
            parallelFor(indices, [&](std::size_t k) {
                std::string error;
                const auto p = lowerToInteger(network, domain, todo[k], error, options.proof.analysis);
                if (p) values[k] = configCost(*p, options.cost);
                return true;
            });
            result.lowerings += todo.size();
            for (std::size_t k = 0; k < todo.size(); ++k) costCache[todo[k].describe()] = values[k];
        };
        // Compound moves pair the slots of one layer: its weights with the
        // activations it reads or writes (e.g. fewer weight bits and more
        // activation bits keep a 16-bit multiply).
        const auto related = [&](std::size_t s1, std::size_t s2) {
            const Slot a = slotOf(s1, network), b = slotOf(s2, network);
            if (a.kind < 0 || b.kind < 0 || a.kind == b.kind) return false;
            const Slot& w = a.kind == 1 ? a : b;
            const Slot& x = a.kind == 1 ? b : a;
            const auto& sources = network.layers[w.index].sources;
            return x.index == w.index + 1 || std::find(sources.begin(), sources.end(), x.index) != sources.end();
        };
        const auto apply = [&](PrecisionConfig& c, std::size_t source, int direction) {
            const Slot slot = slotOf(source, network);
            if (slot.kind < 0) return false;
            const int value = slotValue(c, slot);
            const int next = direction == 2    ? nextStorageClass(ladderFor(slot), value, slot.kind)
                             : direction == -2 ? previousStorageClass(ladderFor(slot), value, slot.kind)
                                               : nextRung(ladderFor(slot), value, direction);
            if (next == 0 || next == value) return false;
            slotValue(c, slot) = next;
            return true;
        };
        // A region that defeated one candidate probably defeats the next:
        // it moves to the front, and a short falsifier run precedes proofs.
        // Large inputs: the number of candidate proofs in the local search is
        // bounded (deterministically), so a long plateau of near-misses
        // cannot dominate the compile time.
        const std::size_t proofLimit = options.maxCandidateProofs > 0 ? options.maxCandidateProofs
                                       : large                        ? 120
                                                                      : 0;
        std::size_t candidateProofs = 0;
        const auto certifies = [&](const QProgram& p) {
            if (proofLimit > 0 && ++candidateProofs > proofLimit) return false;
            const auto evaluation = evaluateProgram(network, p, regions, target, search,
                                                    options.falsifyFirst ? 8 : 0, order, true, options.seed);
            result.analyses += evaluation.analyses;
            result.falsifierCalls += evaluation.falsifierCalls;
            result.falsifierHits += evaluation.falsifierHits;
            for (std::size_t r : evaluation.failedRegions) {
                const auto at = std::find(order.begin(), order.end(), r);
                if (at != order.end()) std::rotate(order.begin(), at, at + 1);
            }
            return evaluation.failing == 0;
        };
        // On large inputs (expensive lowerings and proofs) the neighbourhood
        // escalates: compound moves are tried only when no single move
        // certifies. Small models search the full neighbourhood at once.
        const auto localSearch = [&](PrecisionConfig current, const std::string& tag) {
            for (;;) {
                const double cost = costOf(current);
                const int directions[] = {-1, +1, -2, +2};
                const auto neighbours = [&](bool singles, bool pairs) {
                    std::vector<PrecisionConfig> candidates;
                    for (std::size_t s1 = 0; s1 < sources; ++s1) {
                        for (int d1 : directions) {
                            PrecisionConfig one = current;
                            if (!apply(one, s1, d1)) continue;
                            if (singles) candidates.push_back(one);
                            if (!pairs) continue;
                            for (std::size_t s2 = s1 + 1; s2 < sources; ++s2) {
                                if (!related(s1, s2)) continue;
                                for (int d2 : directions) {
                                    PrecisionConfig two = one;
                                    if (apply(two, s2, d2)) candidates.push_back(two);
                                }
                            }
                        }
                    }
                    return candidates;
                };
                const auto tryMoves = [&](const std::vector<PrecisionConfig>& candidates) {
                    precompute(candidates);
                    std::vector<std::pair<double, PrecisionConfig>> moves;
                    std::map<std::string, bool> seen;
                    for (const auto& c : candidates) {
                        const double value = costOf(c);
                        if (value < cost - 1e-9 && !seen[c.describe()]) {
                            seen[c.describe()] = true;
                            moves.push_back({cost - value, c});
                        }
                    }
                    std::sort(moves.begin(), moves.end(),
                              [](const auto& x, const auto& y) { return x.first > y.first; });
                    for (const auto& move : moves) {
                        const auto p = lower(move.second);
                        if (!p || !certifies(*p)) continue;
                        note(tag + ": " + current.describe() + " -> " + move.second.describe());
                        current = move.second;
                        return true;
                    }
                    return false;
                };
                bool improved = false;
                if (large) {
                    improved = tryMoves(neighbours(true, false)) ||
                               (options.compoundMoves && tryMoves(neighbours(false, true)));
                } else {
                    improved = tryMoves(neighbours(true, options.compoundMoves));
                }
                if (!improved) return current;
            }
        };
        PrecisionConfig best = localSearch(config, "improve");
        if (options.multiStart) {
            // Second start: the cheapest certified uniform configuration.
            std::vector<std::pair<double, PrecisionConfig>> uniform;
            std::vector<PrecisionConfig> all;
            for (int w : wLadder) {
                for (int a : aLadder) all.push_back(uniformConfig(L, w, a));
            }
            precompute(all);
            for (const auto& c : all) uniform.push_back({costOf(c), c});
            std::sort(uniform.begin(), uniform.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
            for (const auto& candidate : uniform) {
                if (candidate.first >= costOf(best)) break;  // cannot beat what we have
                const auto p = lower(candidate.second);
                if (!p || !certifies(*p)) continue;
                const auto other = localSearch(candidate.second, "restart");
                if (costOf(other) < costOf(best)) best = other;
                break;
            }
        }
        config = best;
        program = lower(config);
    }

    // Final proofs (the certificate) and a deeper search on what is left.
    std::iota(order.begin(), order.end(), 0);
    auto final = evaluateProgram(network, *program, regions, robust, options.proof, 0, order, false, options.seed);
    result.analyses += final.analyses;
    falsifyUnproved(network, *program, regions, final.status, 400, options.seed, result);
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
        << "    \"no_saturation_on_domain\": " << (program.saturationPossible ? "false" : "true")
        << ",\n    \"scales\": \"" << (program.calibrated ? "calibrated" : "proved") << "\",\n    \"accumulator_bits\": [";
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

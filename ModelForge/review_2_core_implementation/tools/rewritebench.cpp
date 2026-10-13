// modelforge_rewritebench: how fast does a validation method notice that a
// compiler rewrite broke a model?
// Author: Vaishnavi
//
// For every trained model (tabular MLPs, MNIST/Fashion/CIFAR MLPs and CNNs,
// LeNet-5, MobileNet-style and ResNet-style CNNs) and every applicable rewrite
// from src/rewrites.h (correct rewrites as negative controls, faulty ones
// reproducing documented compiler bugs), each method searches for an input on
// which the rewritten graph disagrees with the original model.
//
// Fair budgets: every method gets the same number of *model evaluations*.
// Checking a probe costs 2 (the original model, as a float64 oracle, and the
// rewritten graph); a gradient of the original costs 2 (forward + backward);
// an extra forward pass of the original costs 1. Wall time is recorded too.
//
// Oracle: the original model is evaluated by an independent float64
// implementation (the layer DAG of src/qnet.h); the rewritten graph by the IR
// interpreter in float32. A probe detects the fault when some logit differs by
// more than tau * max(1, max|logit|). Witnesses are written out for an
// independent re-check with ONNX Runtime (research/rewrite_oracle.py).
//
// Methods:
//   random            uniform in the input domain
//   data              held-out data rows (validation-set testing)
//   data_noise        data rows plus Gaussian noise
//   tensorfuzz        coverage-guided fuzzing (TensorFuzz / DeepHunter style):
//                     mutate a corpus, keep inputs that reach new ReLU states
//   deepxplore        gradient ascent on uncovered neurons (DeepXplore style)
//   diffchaser        genetic search maximising the output difference between
//                     the two models (DiffChaser style, black box)
//   guardian_apc      boundary probes for ReLU units at every depth
//   guardian_rw       ... only for units the rewrite can affect (rewrite-aware)
//   guardian_nm       ... global, plus near-miss search on the divergence
//   guardian_apc_plus rewrite-aware boundary probes + near-miss search
//
// Usage: modelforge_rewritebench [--quick] [--budget N] [--seeds K] [--out DIR] [--only model,...]

#include "certify.h"
#include "qnet.h"
#include "research.h"
#include "rewrites.h"
#include "shrink.h"
#include "verifier.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifndef MODELFORGE_MODELS
#define MODELFORGE_MODELS "models"
#endif

using namespace modelforge;
namespace fs = std::filesystem;

namespace {

constexpr double kTau = 1e-3;  // relative logit tolerance

// ------------------------------------------------------------ float64 model with gradients

struct Forward {
    std::vector<std::vector<double>> post;  // per tensor (0 = input)
    std::vector<std::vector<double>> pre;   // per layer, before ReLU (weighted layers)
    std::vector<std::vector<std::uint32_t>> argmax;  // max-pool: winning tap per output
};

Forward forward(const DenseNetwork& net, const std::vector<double>& x) {
    Forward f;
    const std::size_t L = net.layers.size();
    f.post.resize(L + 1);
    f.pre.resize(L);
    f.argmax.resize(L);
    f.post[0] = x;
    std::vector<double> scratch;
    for (std::size_t l = 0; l < L; ++l) {
        const auto& layer = net.layers[l];
        const std::vector<double>* in = &f.post[layer.sources[0]];
        if (layer.sources.size() > 1) {
            scratch.clear();
            for (std::size_t s : layer.sources) scratch.insert(scratch.end(), f.post[s].begin(), f.post[s].end());
            in = &scratch;
        }
        auto& out = f.post[l + 1];
        out.assign(layer.out, 0.0);
        if (layer.kind == LayerKind::MaxPool) {
            f.argmax[l].assign(layer.out, 0);
            for (std::size_t o = 0; o < layer.out; ++o) {
                double m = -INFINITY;
                for (const auto& t : layer.taps[o]) {
                    if ((*in)[t.input] > m) {
                        m = (*in)[t.input];
                        f.argmax[l][o] = t.input;
                    }
                }
                out[o] = m;
            }
            continue;
        }
        auto& pre = f.pre[l];
        pre.assign(layer.out, 0.0);
        for (std::size_t o = 0; o < layer.out; ++o) {
            double sum = layer.bias[layer.channelOf(o)];
            for (const auto& t : layer.taps[o]) sum += layer.weight[t.weight] * (*in)[t.input];
            pre[o] = sum;
            out[o] = layer.relu ? std::max(0.0, sum) : sum;
        }
    }
    return f;
}

// Gradient of the pre-activation of unit `unit` of layer `target` with
// respect to the input (exact: the network is piecewise linear).
std::vector<double> gradient(const DenseNetwork& net, const Forward& f, std::size_t target, std::size_t unit) {
    const std::size_t L = net.layers.size();
    std::vector<std::vector<double>> g(L + 1);
    for (std::size_t t = 0; t <= L; ++t) g[t].assign(f.post[t].size(), 0.0);
    const auto spread = [&](std::size_t l, std::size_t o, double gpre) {
        const auto& layer = net.layers[l];
        std::vector<std::size_t> offset;
        std::size_t total = 0;
        for (std::size_t s : layer.sources) {
            offset.push_back(total);
            total += f.post[s].size();
        }
        const auto at = [&](std::size_t input) -> double& {
            std::size_t k = layer.sources.size() - 1;
            while (k > 0 && input < offset[k]) --k;
            return g[layer.sources[k]][input - offset[k]];
        };
        if (layer.kind == LayerKind::MaxPool) {
            at(f.argmax[l][o]) += gpre;
            return;
        }
        for (const auto& t : layer.taps[o]) at(t.input) += layer.weight[t.weight] * gpre;
    };
    spread(target, unit, 1.0);
    for (std::size_t l = target; l-- > 0;) {
        const auto& layer = net.layers[l];
        const auto& gout = g[l + 1];
        for (std::size_t o = 0; o < layer.out; ++o) {
            double v = gout[o];
            if (v == 0.0) continue;
            if (layer.kind != LayerKind::MaxPool && layer.relu && f.pre[l][o] <= 0.0) continue;
            spread(l, o, v);
        }
    }
    return g[0];
}

// ------------------------------------------------------------ one validation run

struct Context {
    std::string model;
    const DenseNetwork* net = nullptr;
    const IRGraph* original = nullptr;
    Box domain;
    std::vector<std::vector<double>> pool;  // data rows the data-driven methods may use
    std::vector<std::size_t> reluLayers;    // weighted layers with ReLU
    std::string logitValue;                 // IR value holding the logits (before Softmax/Sigmoid)
};

struct CaseInfo {
    const RewriteCase* rewrite = nullptr;
    std::vector<std::size_t> targetLayers;  // rewrite-aware ReLU layers
};

struct Outcome {
    std::size_t first = 0;      // model evaluations used up to and including the detecting probe (0 = none)
    std::size_t firstFlip = 0;  // ... for a probe on which the predicted class changes
    std::size_t probes = 0;
    double maxDivergence = 0.0;
    double seconds = 0.0;
    std::vector<double> witness;
};

class Run {
public:
    Run(const Context& ctx, const IRGraph& candidate, std::size_t budget)
        : ctx_(ctx), candidate_(candidate), budget_(budget) {}

    bool done() const { return outcome_.first > 0 || used_ >= budget_; }
    bool affordable(std::size_t units) const { return used_ + units <= budget_; }
    void charge(std::size_t units) { used_ += units; }

    // Checks one input (2 units). Returns the relative divergence.
    double check(const std::vector<double>& xIn, Forward* keep = nullptr) {
        if (!affordable(2) || outcome_.first > 0) return 0.0;
        used_ += 2;
        ++outcome_.probes;
        std::vector<double> x = xIn;
        clip(x);
        Forward f = forward(*ctx_.net, x);
        const auto& ref = f.post.back();
        std::vector<float> xf(x.begin(), x.end());
        DiagnosticEngine diagnostics;
        std::unordered_map<std::string, std::vector<float>> trace;
        executeIRTrace(candidate_, xf, diagnostics, &trace);
        const auto it = trace.find(ctx_.logitValue);
        double divergence = 1e9;  // a graph that fails to run diverges
        bool flip = true;
        if (it != trace.end()) {
            std::vector<double> got(it->second.begin(), it->second.end());
            if (ctx_.net->singleLogitPadded) got.insert(got.begin(), 0.0);
            if (got.size() == ref.size()) {
                double scale = 1.0, diff = 0.0;
                for (std::size_t k = 0; k < ref.size(); ++k) {
                    scale = std::max(scale, std::fabs(ref[k]));
                    diff = std::max(diff, std::fabs(got[k] - ref[k]));
                }
                divergence = diff / scale;
                flip = argmaxFirst(got) != argmaxFirst(ref);
            }
        }
        outcome_.maxDivergence = std::max(outcome_.maxDivergence, divergence);
        if (divergence > kTau && outcome_.first == 0) {
            outcome_.first = used_;
            outcome_.witness = x;
        }
        if (flip && outcome_.firstFlip == 0) outcome_.firstFlip = used_;
        if (keep) *keep = std::move(f);
        return divergence;
    }

    // Forward pass of the original only (1 unit).
    std::optional<Forward> probeOriginal(const std::vector<double>& x) {
        if (!affordable(1)) return std::nullopt;
        used_ += 1;
        return forward(*ctx_.net, x);
    }
    // Gradient of one unit at a known forward pass (1 more unit: backward).
    std::optional<std::vector<double>> grad(const Forward& f, std::size_t layer, std::size_t unit) {
        if (!affordable(1)) return std::nullopt;
        used_ += 1;
        return gradient(*ctx_.net, f, layer, unit);
    }

    void clip(std::vector<double>& x) const {
        for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::max(ctx_.domain.lo[i], std::min(ctx_.domain.hi[i], x[i]));
    }
    Outcome& outcome() { return outcome_; }

private:
    const Context& ctx_;
    const IRGraph& candidate_;
    std::size_t budget_;
    std::size_t used_ = 0;
    Outcome outcome_;
};

std::vector<double> uniformPoint(const Box& box, std::mt19937& rng) {
    std::vector<double> x(box.size());
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::uniform_real_distribution<double>(box.lo[i], box.hi[i])(rng);
    return x;
}

std::vector<double> noisy(const std::vector<double>& x, const Box& box, double sigma, std::mt19937& rng) {
    std::vector<double> y = x;
    std::normal_distribution<double> n(0.0, 1.0);
    for (std::size_t i = 0; i < y.size(); ++i) y[i] += sigma * (box.hi[i] - box.lo[i]) * n(rng);
    return y;
}

// ------------------------------------------------------------ methods

void methodRandom(Run& run, const Context& ctx, std::mt19937& rng) {
    while (!run.done()) run.check(uniformPoint(ctx.domain, rng));
}

void methodData(Run& run, const Context& ctx, std::mt19937& rng, double sigma) {
    std::vector<std::size_t> order(ctx.pool.size());
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    for (std::size_t k = 0; !run.done(); ++k) {
        const auto& row = ctx.pool[order[k % order.size()]];
        run.check(sigma > 0 || k >= order.size() ? noisy(row, ctx.domain, sigma > 0 ? sigma : 0.05, rng) : row);
    }
}

// Coverage: (layer, unit, active?) states reached by the original model.
using Coverage = std::set<std::tuple<std::size_t, std::size_t, bool>>;
std::size_t addCoverage(const Context& ctx, const Forward& f, Coverage& cov) {
    std::size_t added = 0;
    for (std::size_t l : ctx.reluLayers) {
        for (std::size_t o = 0; o < f.pre[l].size(); ++o) added += cov.insert({l, o, f.pre[l][o] > 0.0}).second;
    }
    return added;
}

void methodTensorFuzz(Run& run, const Context& ctx, std::mt19937& rng) {
    std::vector<std::vector<double>> corpus;
    Coverage cov;
    for (std::size_t k = 0; k < 4 && !run.done(); ++k) {
        const auto& row = ctx.pool[std::uniform_int_distribution<std::size_t>(0, ctx.pool.size() - 1)(rng)];
        Forward f;
        run.check(row, &f);
        if (!f.post.empty()) addCoverage(ctx, f, cov);
        corpus.push_back(row);
    }
    while (!run.done()) {
        const auto& parent = corpus[std::uniform_int_distribution<std::size_t>(0, corpus.size() - 1)(rng)];
        auto child = noisy(parent, ctx.domain, 0.05, rng);
        run.clip(child);
        Forward f;
        run.check(child, &f);
        if (!f.post.empty() && addCoverage(ctx, f, cov) > 0) corpus.push_back(child);
    }
}

void methodDeepXplore(Run& run, const Context& ctx, std::mt19937& rng) {
    Coverage cov;
    while (!run.done()) {
        auto x = ctx.pool[std::uniform_int_distribution<std::size_t>(0, ctx.pool.size() - 1)(rng)];
        // The seed itself is a test input first (as in DeepXplore).
        Forward seedForward;
        run.check(x, &seedForward);
        if (run.done() || seedForward.post.empty()) break;
        auto f = std::optional<Forward>(std::move(seedForward));
        addCoverage(ctx, *f, cov);
        // A ReLU unit that has not been active yet: raise its pre-activation.
        std::size_t layer = ctx.reluLayers[std::uniform_int_distribution<std::size_t>(0, ctx.reluLayers.size() - 1)(rng)];
        std::size_t unit = std::uniform_int_distribution<std::size_t>(0, f->pre[layer].size() - 1)(rng);
        for (int tries = 0; tries < 16 && cov.count({layer, unit, true}); ++tries) {
            unit = std::uniform_int_distribution<std::size_t>(0, f->pre[layer].size() - 1)(rng);
        }
        for (int step = 0; step < 3 && !run.done(); ++step) {
            const auto g = run.grad(*f, layer, unit);
            if (!g) break;
            double norm = 0;
            for (double v : *g) norm = std::max(norm, std::fabs(v));
            if (norm == 0) break;
            for (std::size_t i = 0; i < x.size(); ++i) {
                x[i] += 0.05 * (ctx.domain.hi[i] - ctx.domain.lo[i]) * ((*g)[i] > 0 ? 1.0 : (*g)[i] < 0 ? -1.0 : 0.0);
            }
            run.clip(x);
            f = run.probeOriginal(x);
            if (!f) break;
        }
        Forward checked;
        run.check(x, &checked);
        if (!checked.post.empty()) addCoverage(ctx, checked, cov);
    }
}

void methodDiffChaser(Run& run, const Context& ctx, std::mt19937& rng) {
    struct Member {
        std::vector<double> x;
        double fitness;
    };
    std::vector<Member> population;
    for (std::size_t k = 0; k < 8 && !run.done(); ++k) {
        auto x = ctx.pool[std::uniform_int_distribution<std::size_t>(0, ctx.pool.size() - 1)(rng)];
        population.push_back({x, run.check(x)});
    }
    while (!run.done()) {
        const auto pick = [&]() -> const Member& {
            const auto& a = population[std::uniform_int_distribution<std::size_t>(0, population.size() - 1)(rng)];
            const auto& b = population[std::uniform_int_distribution<std::size_t>(0, population.size() - 1)(rng)];
            return a.fitness >= b.fitness ? a : b;
        };
        const auto& p1 = pick();
        const auto& p2 = pick();
        std::vector<double> child(p1.x.size());
        for (std::size_t i = 0; i < child.size(); ++i) child[i] = std::bernoulli_distribution(0.5)(rng) ? p1.x[i] : p2.x[i];
        child = noisy(child, ctx.domain, 0.02, rng);
        run.clip(child);
        const double fitness = run.check(child);
        auto worst = std::min_element(population.begin(), population.end(),
                                      [](const Member& a, const Member& b) { return a.fitness < b.fitness; });
        if (fitness > worst->fitness) *worst = {child, fitness};
    }
}

// Guardian-APC for CNN graphs: boundary-value analysis of ReLU units.
// A data seed is moved, along the exact gradient of one unit's
// pre-activation (the network is piecewise linear), so that the unit's input
// takes a ladder of values on both sides of its boundary, from tiny to large
// (scaled by the layer's typical pre-activation). This exercises the
// activation at the kink and in its saturating range, where lowering and
// fusion mistakes show.
// rewriteAware: units only from the layers the rewrite can affect.
// nearMiss: every other round, a small move from the most divergent input so
// far (climbing sub-tolerance divergence).
void methodGuardian(Run& run, const Context& ctx, const CaseInfo& info, std::mt19937& rng, bool rewriteAware,
                    bool nearMiss) {
    const auto& layers = rewriteAware && !info.targetLayers.empty() ? info.targetLayers : ctx.reluLayers;
    // The most divergent probes so far: the near-miss population.
    std::vector<std::pair<double, std::vector<double>>> elite;
    const auto record = [&](const std::vector<double>& x, double d) {
        if (elite.size() < 8) {
            elite.push_back({d, x});
        } else {
            auto worst = std::min_element(elite.begin(), elite.end(),
                                          [](const auto& a, const auto& b) { return a.first < b.first; });
            if (d > worst->first) *worst = {d, x};
        }
    };
    // Smoke probes: two data rows and a random point find loud faults at once.
    for (int k = 0; k < 2; ++k) {
        const auto& row = ctx.pool[std::uniform_int_distribution<std::size_t>(0, ctx.pool.size() - 1)(rng)];
        record(row, run.check(row));
    }
    {
        const auto r = uniformPoint(ctx.domain, rng);
        record(r, run.check(r));
    }
    // Ladder of target pre-activations, in units of the layer's scale:
    // small and large values interleaved so both ranges are reached early.
    static const double kLadder[] = {0.03, -0.03, 3.0, -3.0, 0.3, -0.3, 10.0, 0.01, -0.01, 1.0, -1.0, 0.1, -0.1, 30.0};
    std::size_t round = 0;
    while (!run.done()) {
        ++round;
        const auto x0 = ctx.pool[std::uniform_int_distribution<std::size_t>(0, ctx.pool.size() - 1)(rng)];
        auto f = run.probeOriginal(x0);
        if (!f) break;
        const std::size_t layer = layers[round % layers.size()];
        const auto& pre = f->pre[layer];
        double scale = 0.0;
        for (double v : pre) scale += v * v;
        scale = std::max(1e-6, std::sqrt(scale / static_cast<double>(pre.size())));
        const std::size_t unit = std::uniform_int_distribution<std::size_t>(0, pre.size() - 1)(rng);
        const auto g = run.grad(*f, layer, unit);
        if (!g) break;
        double norm2 = 0;
        for (double v : *g) norm2 += v * v;
        if (norm2 == 0) continue;
        for (double level : kLadder) {
            if (run.done()) break;
            const double t = (level * scale - pre[unit]) / norm2;
            std::vector<double> x(x0.size());
            for (std::size_t i = 0; i < x.size(); ++i) x[i] = x0[i] + t * (*g)[i];
            run.clip(x);
            record(x, run.check(x));
            if (nearMiss && elite.size() >= 2 && !run.done()) {
                // Near-miss: recombine two of the most divergent probes and
                // perturb slightly; sub-tolerance divergence is climbed.
                const auto& a = elite[std::uniform_int_distribution<std::size_t>(0, elite.size() - 1)(rng)].second;
                const auto& b = elite[std::uniform_int_distribution<std::size_t>(0, elite.size() - 1)(rng)].second;
                std::vector<double> y(a.size());
                for (std::size_t i = 0; i < y.size(); ++i) y[i] = std::bernoulli_distribution(0.5)(rng) ? a[i] : b[i];
                y = noisy(y, ctx.domain, 0.02, rng);
                run.clip(y);
                record(y, run.check(y));
            }
        }
    }
}

// ------------------------------------------------------------ cases

std::vector<std::size_t> targetLayers(const DenseNetwork& net, const RewriteCase& rewrite) {
    const std::size_t L = net.layers.size();
    std::vector<bool> impacted(L, false);
    std::set<std::string> changed(rewrite.changedNodes.begin(), rewrite.changedNodes.end());
    for (std::size_t l = 0; l < L; ++l) {
        for (const auto& n : net.layers[l].irNodes) impacted[l] = impacted[l] || changed.count(n);
    }
    // The rewritten operators themselves: changed layers with a ReLU.
    std::vector<std::size_t> targets;
    for (std::size_t l = 0; l < L; ++l) {
        if (impacted[l] && net.layers[l].relu) targets.push_back(l);
    }
    // Otherwise the nearest ReLU layers downstream of a change.
    for (int depth = 0; targets.empty() && depth < static_cast<int>(L); ++depth) {
        std::vector<bool> next = impacted;
        for (std::size_t l = 0; l < L; ++l) {
            for (std::size_t s : net.layers[l].sources) {
                if (s > 0 && impacted[s - 1]) next[l] = true;
            }
        }
        if (next == impacted) break;
        for (std::size_t l = 0; l < L; ++l) {
            if (next[l] && !impacted[l] && net.layers[l].relu) targets.push_back(l);
        }
        impacted = next;
    }
    if (targets.empty()) {
        // Only the head changed: exercise the ReLU layers that feed it.
        for (std::size_t l = 0; l < L; ++l) {
            if (!impacted[l]) continue;
            for (std::size_t s : net.layers[l].sources) {
                if (s > 0 && net.layers[s - 1].relu) targets.push_back(s - 1);
            }
        }
    }
    std::sort(targets.begin(), targets.end());
    targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
    return targets;
}

struct ModelEntry {
    std::string name, folder, dataset, poolFile;
    bool image = false;
};

std::string logitValueOf(const IRGraph& g) {
    // The value returned, or the input of a final Softmax / Sigmoid.
    std::string value;
    for (const auto& in : g.instructions) {
        if (in.operation == IROp::Return) value = in.inputs[0];
    }
    for (const auto& in : g.instructions) {
        if (in.output == value && (in.operation == IROp::Softmax || in.operation == IROp::Sigmoid)) return in.inputs[0];
    }
    return value;
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t budget = 512, seeds = 5;
    bool quick = false;
    std::string outDir = "rewritebench";
    std::set<std::string> only;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--quick") quick = true;
        if (a == "--budget" && i + 1 < argc) budget = static_cast<std::size_t>(std::atoi(argv[++i]));
        if (a == "--seeds" && i + 1 < argc) seeds = static_cast<std::size_t>(std::atoi(argv[++i]));
        if (a == "--out" && i + 1 < argc) outDir = argv[++i];
        if (a == "--only" && i + 1 < argc) {
            std::stringstream list(argv[++i]);
            std::string m;
            while (std::getline(list, m, ',')) only.insert(m);
        }
    }
    if (quick) {
        budget = std::min<std::size_t>(budget, 128);
        seeds = std::min<std::size_t>(seeds, 2);
    }
    fs::create_directories(fs::path(outDir) / "witnesses");
    const fs::path models = MODELFORGE_MODELS;
    std::vector<ModelEntry> entries;
    for (const char* m : {"iris_mlp16", "iris_mlp32x16", "wine_mlp16", "wine_mlp32x16", "breast_cancer_mlp16",
                          "breast_cancer_mlp32x16", "digits_mlp16", "digits_mlp32x16"}) {
        const std::string n = m;
        const std::string ds = n.substr(0, n.find("_mlp"));
        entries.push_back({n, "real", ds, ds + "_train.csv", false});
    }
    for (const char* m : {"mnist_mlp", "mnist_cnn", "fashion_mlp", "fashion_cnn", "cifar_cnn", "mnist_lenet",
                          "mnist_lenet_avg", "mnist_dscnn", "mnist_resnet", "fashion_lenet", "fashion_lenet_avg",
                          "fashion_dscnn", "fashion_resnet"}) {
        const std::string n = m;
        const std::string ds = n.substr(0, n.find('_'));
        entries.push_back({n, "image", ds, ds + "_calib.csv", true});
    }
    if (quick) entries = {entries[2], entries[9], entries[16]};  // wine_mlp16, mnist_cnn, mnist_resnet
    const std::vector<std::string> methods{"random",      "data",         "data_noise",  "tensorfuzz",
                                           "deepxplore",  "diffchaser",   "guardian_apc", "guardian_rw",
                                           "guardian_nm", "guardian_apc_plus"};

    std::ofstream csv(fs::path(outDir) / "runs.csv");
    csv << "model,pass,fault,source,control,method,seed,budget,first,first_flip,probes,max_divergence,seconds\n";
    std::ofstream casesFile(fs::path(outDir) / "cases.csv");
    casesFile << "model,pass,fault,control,changed_nodes,target_layers,witness\n";
    std::mutex lock;
    const auto start = std::chrono::steady_clock::now();
    for (const auto& entry : entries) {
        if (!only.empty() && !only.count(entry.name)) continue;
        const fs::path dir = models / entry.folder;
        std::ifstream file(dir / (entry.name + ".mforge"));
        const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        IRGraph original;
        DiagnosticEngine diagnostics;
        std::string error;
        if (!compileManifestText(text, original, diagnostics)) {
            std::cerr << "cannot compile " << entry.name << "\n";
            continue;
        }
        auto net = extractDenseNetwork(original, error);
        if (!net) {
            std::cerr << entry.name << ": " << error << "\n";
            continue;
        }
        Dataset pool;
        if (!loadDataset((dir / entry.poolFile).string(), net->inputs, pool, error)) {
            std::cerr << entry.name << ": " << error << "\n";
            continue;
        }
        Context ctx;
        ctx.model = entry.name;
        ctx.net = &*net;
        ctx.original = &original;
        if (entry.image) {
            ctx.domain.lo.assign(net->inputs, 0.0);
            ctx.domain.hi.assign(net->inputs, 1.0);
        } else {
            ctx.domain = domainFromData(pool.rows, 0.1);
        }
        for (const auto& row : pool.rows) ctx.pool.emplace_back(row.begin(), row.end());
        for (std::size_t l = 0; l < net->layers.size(); ++l) {
            if (net->layers[l].relu) ctx.reluLayers.push_back(l);
        }
        ctx.logitValue = logitValueOf(original);
        const auto rewrites = makeRewriteCases(original);
        std::cout << "== " << entry.name << ": " << rewrites.size() << " rewrites, " << net->layers.size()
                  << " layers" << std::endl;

        // Every (rewrite, method, seed) is independent: run them on all cores.
        struct Job {
            std::size_t rewrite, method, seed;
        };
        std::vector<Job> jobs;
        std::vector<CaseInfo> infos(rewrites.size());
        for (std::size_t r = 0; r < rewrites.size(); ++r) {
            infos[r].rewrite = &rewrites[r];
            infos[r].targetLayers = targetLayers(*net, rewrites[r]);
            for (std::size_t m = 0; m < methods.size(); ++m) {
                for (std::size_t s = 0; s < seeds; ++s) jobs.push_back({r, m, s});
            }
        }
        std::vector<Outcome> outcomes(jobs.size());
        std::atomic<std::size_t> next{0};
        const auto worker = [&]() {
            for (;;) {
                const std::size_t k = next.fetch_add(1);
                if (k >= jobs.size()) return;
                const auto& job = jobs[k];
                const auto t0 = std::chrono::steady_clock::now();
                Run run(ctx, rewrites[job.rewrite].candidate, budget);
                std::mt19937 rng(static_cast<std::uint32_t>(1000003 * job.seed + 7919 * job.rewrite + 17));
                const std::string& m = methods[job.method];
                const auto& info = infos[job.rewrite];
                if (m == "random") methodRandom(run, ctx, rng);
                if (m == "data") methodData(run, ctx, rng, 0.0);
                if (m == "data_noise") methodData(run, ctx, rng, 0.05);
                if (m == "tensorfuzz") methodTensorFuzz(run, ctx, rng);
                if (m == "deepxplore") methodDeepXplore(run, ctx, rng);
                if (m == "diffchaser") methodDiffChaser(run, ctx, rng);
                if (m == "guardian_apc") methodGuardian(run, ctx, info, rng, false, false);
                if (m == "guardian_rw") methodGuardian(run, ctx, info, rng, true, false);
                if (m == "guardian_nm") methodGuardian(run, ctx, info, rng, false, true);
                if (m == "guardian_apc_plus") methodGuardian(run, ctx, info, rng, true, true);
                outcomes[k] = run.outcome();
                outcomes[k].seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            }
        };
        std::vector<std::thread> threads;
        const std::size_t workers = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
        for (std::size_t t = 0; t < workers; ++t) threads.emplace_back(worker);
        for (auto& t : threads) t.join();

        std::map<std::size_t, std::string> witnessFile;
        for (std::size_t k = 0; k < jobs.size(); ++k) {
            const auto& job = jobs[k];
            const auto& rw = rewrites[job.rewrite];
            const auto& o = outcomes[k];
            std::string source = rw.source;
            std::replace(source.begin(), source.end(), ',', ';');
            csv << entry.name << "," << rw.pass << "," << rw.fault << "," << source << "," << (rw.control ? 1 : 0)
                << "," << methods[job.method] << "," << job.seed << "," << budget << "," << o.first << ","
                << o.firstFlip << "," << o.probes << "," << std::setprecision(6) << o.maxDivergence << ","
                << o.seconds << "\n";
            // One witness per detected rewrite, for the independent ONNX Runtime check.
            if (o.first > 0 && !witnessFile.count(job.rewrite)) {
                const std::string base = entry.name + "_" + std::to_string(job.rewrite);
                std::ofstream(fs::path(outDir) / "witnesses" / (base + "_candidate.mforge")) << irToManifest(rw.candidate);
                std::ofstream w(fs::path(outDir) / "witnesses" / (base + "_input.csv"));
                w << std::setprecision(17);
                for (std::size_t i = 0; i < o.witness.size(); ++i) w << (i ? "," : "") << o.witness[i];
                w << "\n";
                witnessFile[job.rewrite] = base;
            }
        }
        for (std::size_t r = 0; r < rewrites.size(); ++r) {
            const auto& rw = rewrites[r];
            casesFile << entry.name << "," << rw.pass << "," << rw.fault << "," << (rw.control ? 1 : 0) << ",";
            for (std::size_t k = 0; k < rw.changedNodes.size(); ++k) casesFile << (k ? " " : "") << rw.changedNodes[k];
            casesFile << ",";
            for (std::size_t k = 0; k < infos[r].targetLayers.size(); ++k) casesFile << (k ? " " : "") << infos[r].targetLayers[k];
            casesFile << "," << (witnessFile.count(r) ? witnessFile[r] : "") << "\n";
        }
        std::ofstream(fs::path(outDir) / "witnesses" / (entry.name + "_original.mforge")) << irToManifest(original);
        csv.flush();
        casesFile.flush();
        // Short console summary: detection rate per method at the full budget.
        std::map<std::string, std::pair<std::size_t, std::size_t>> rate;
        for (std::size_t k = 0; k < jobs.size(); ++k) {
            if (rewrites[jobs[k].rewrite].control) continue;
            auto& r = rate[methods[jobs[k].method]];
            ++r.second;
            r.first += outcomes[k].first > 0;
        }
        for (const auto& m : methods) {
            std::cout << "   " << std::left << std::setw(20) << m << rate[m].first << "/" << rate[m].second << "\n";
        }
        std::cout << std::flush;
    }
    std::cout << "done in " << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
              << " s; results in " << outDir << "/runs.csv\n";
    return 0;
}

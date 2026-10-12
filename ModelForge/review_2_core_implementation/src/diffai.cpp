#include "diffai.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace modelforge {
namespace {

// Affine form c + sum_k a_k * eta_k with eta_k in [-1, 1], stored sparsely as
// (symbol, coefficient) pairs sorted by symbol. A first-layer convolution
// neuron touches a few dozen symbols out of thousands, so sparse forms keep
// image models (thousands of inputs and neurons) cheap.
struct Term {
    std::uint32_t symbol;
    double value;
};

struct Form {
    double c = 0.0;
    std::vector<Term> a;
};

double radius(const Form& f) {
    double r = 0.0;
    for (const auto& t : f.a) r += std::fabs(t.value);
    return r;
}

// Outward slack for the analysis' own double rounding.
double slack(const Form& f, double r) { return 1e-12 * (std::fabs(f.c) + r) + 1e-300; }
double lower(const Form& f) {
    const double r = radius(f);
    return f.c - r - slack(f, r);
}
double upper(const Form& f) {
    const double r = radius(f);
    return f.c + r + slack(f, r);
}

// dst += alpha * src (sorted merge).
void axpy(Form& dst, double alpha, const Form& src) {
    if (alpha == 0.0) return;
    dst.c += alpha * src.c;
    if (src.a.empty()) return;
    if (dst.a.empty()) {
        dst.a.reserve(src.a.size());
        for (const auto& t : src.a) dst.a.push_back({t.symbol, alpha * t.value});
        return;
    }
    std::vector<Term> merged;
    merged.reserve(dst.a.size() + src.a.size());
    std::size_t i = 0, j = 0;
    while (i < dst.a.size() || j < src.a.size()) {
        if (j == src.a.size() || (i < dst.a.size() && dst.a[i].symbol < src.a[j].symbol)) {
            merged.push_back(dst.a[i++]);
        } else if (i == dst.a.size() || src.a[j].symbol < dst.a[i].symbol) {
            merged.push_back({src.a[j].symbol, alpha * src.a[j].value});
            ++j;
        } else {
            merged.push_back({dst.a[i].symbol, dst.a[i].value + alpha * src.a[j].value});
            ++i;
            ++j;
        }
    }
    dst.a = std::move(merged);
}

double coefficientOf(const Form& f, std::size_t symbol) {
    const auto s = static_cast<std::uint32_t>(symbol);
    auto it = std::lower_bound(f.a.begin(), f.a.end(), s, [](const Term& t, std::uint32_t k) { return t.symbol < k; });
    return it != f.a.end() && it->symbol == s ? it->value : 0.0;
}

bool isZero(const Form& f) { return f.c == 0.0 && f.a.empty(); }

void addSymbol(Form& f, std::size_t symbol, double coefficient) {
    const auto s = static_cast<std::uint32_t>(symbol);
    if (f.a.empty() || f.a.back().symbol < s) {
        f.a.push_back({s, coefficient});  // fresh symbols are always the largest
        return;
    }
    auto it = std::lower_bound(f.a.begin(), f.a.end(), s, [](const Term& t, std::uint32_t k) { return t.symbol < k; });
    if (it != f.a.end() && it->symbol == s) {
        it->value += coefficient;
    } else {
        f.a.insert(it, {s, coefficient});
    }
}

// Sparse accumulator for sums of many forms (the affine step): O(total terms).
class Accumulator {
public:
    void reserve(std::size_t symbols) {
        if (value_.size() < symbols) {
            value_.resize(symbols, 0.0);
            used_.resize(symbols, 0);
        }
    }
    void add(double alpha, const Form& f) {
        if (alpha == 0.0) return;
        constant_ += alpha * f.c;
        for (const auto& t : f.a) {
            if (!used_[t.symbol]) {
                used_[t.symbol] = 1;
                touched_.push_back(t.symbol);
            }
            value_[t.symbol] += alpha * t.value;
        }
    }
    void addConstant(double v) { constant_ += v; }
    Form take() {
        std::sort(touched_.begin(), touched_.end());
        Form f;
        f.c = constant_;
        f.a.reserve(touched_.size());
        for (auto k : touched_) {
            if (value_[k] != 0.0) f.a.push_back({k, value_[k]});
            value_[k] = 0.0;
            used_[k] = 0;
        }
        touched_.clear();
        constant_ = 0.0;
        return f;
    }

private:
    std::vector<double> value_;
    std::vector<char> used_;
    std::vector<std::uint32_t> touched_;
    double constant_ = 0.0;
};

struct Neuron {
    Form r;               // float model
    std::vector<Form> e;  // integer-program error, one component per source
};

Form totalError(const Neuron& n) {
    Form t;
    for (const auto& part : n.e) axpy(t, 1.0, part);
    return t;
}

Form quantValue(const Neuron& n) {
    Form q = n.r;
    for (const auto& part : n.e) axpy(q, 1.0, part);
    return q;
}

// Share of each source in the current error, used to attribute a relaxation.
std::vector<double> errorShares(const Neuron& n) {
    std::vector<double> share(n.e.size(), 0.0);
    double sum = 0.0;
    for (std::size_t s = 0; s < n.e.size(); ++s) {
        share[s] = radius(n.e[s]) + std::fabs(n.e[s].c);
        sum += share[s];
    }
    if (sum <= 0.0) {
        std::fill(share.begin(), share.end(), 1.0 / static_cast<double>(share.size()));
    } else {
        for (double& v : share) v /= sum;
    }
    return share;
}

// Without provenance every source maps to component 0.
class Engine {
public:
    Engine(std::size_t sources, bool affine, bool provenance)
        : sources_(provenance ? sources : 1), affine_(affine), provenance_(provenance) {}
    std::size_t fresh() { return symbols_++; }
    std::size_t sources() const { return sources_; }
    std::size_t part(std::size_t source) const { return provenance_ ? source : 0; }

    std::vector<Neuron> input(const Box& box, const std::vector<double>& inputScale, double slackLsb) {
        std::vector<Neuron> layer(box.size());
        for (std::size_t i = 0; i < box.size(); ++i) {
            const std::size_t symbol = fresh();  // input symbols are 0..n-1
            layer[i].r.c = 0.5 * (box.lo[i] + box.hi[i]);
            if (box.hi[i] > box.lo[i]) addSymbol(layer[i].r, symbol, 0.5 * (box.hi[i] - box.lo[i]));
            layer[i].e.assign(sources_, Form{});
        }
        for (std::size_t i = 0; i < box.size(); ++i) {
            addSymbol(layer[i].e[0], fresh(), inputScale[i] * slackLsb);
        }
        return layer;
    }

    // y = W x + b for the float model, y^ = W^ x^ + b^ for the integer program
    // (W^, b^: effective real weights of the emitted integers, per tap).
    std::vector<Neuron> affine(const std::vector<Neuron>& in, const DenseLayer& layer,
                               const std::vector<std::vector<double>>& tapWeight, const std::vector<double>& bhat,
                               std::size_t weightSource) const {
        std::vector<Neuron> out(layer.out);
        const std::size_t ws = part(weightSource);
        Accumulator sum;
        sum.reserve(symbols_);
        for (std::size_t o = 0; o < layer.out; ++o) {
            Neuron& n = out[o];
            n.e.assign(sources_, Form{});
            const std::size_t c = layer.channelOf(o);
            const auto& taps = layer.taps[o];
            for (std::size_t k = 0; k < taps.size(); ++k) sum.add(layer.weight[taps[k].weight], in[taps[k].input].r);
            sum.addConstant(layer.bias[c]);
            n.r = sum.take();
            for (std::size_t s = 0; s < sources_; ++s) {
                for (std::size_t k = 0; k < taps.size(); ++k) {
                    const Neuron& x = in[taps[k].input];
                    const double wq = tapWeight[o][k];
                    if (s == ws) sum.add(wq - layer.weight[taps[k].weight], x.r);
                    if (!isZero(x.e[s])) sum.add(wq, x.e[s]);
                }
                if (s == ws) sum.addConstant(bhat[o] - layer.bias[c]);
                n.e[s] = sum.take();
            }
        }
        return out;
    }

    struct Activation {
        std::vector<double> preLo;  // integer program's value after ReLU and clamp, before rounding
        std::vector<double> preHi;
        bool saturates = false;     // some clamp can trigger
    };

    // ReLU (if any), then rounding to each neuron's output scale.
    // The emitted code clamps every requantised value to its type range:
    // [0, outMax * t] after ReLU, [-outMax * t, outMax * t] for signed logits.
    // `outMax` = 0 disables the clamp model (lowering, before scales exist).
    Activation activate(std::vector<Neuron>& layer, bool relu, const std::vector<double>& outScale,
                        std::size_t roundSource, std::int64_t outMax = 0) {
        Activation bounds;
        const std::size_t rs = part(roundSource);
        for (std::size_t index = 0; index < layer.size(); ++index) {
            auto& n = layer[index];
            const Form q = quantValue(n);
            double ql = lower(q), qu = upper(q);
            if (relu) {
                const double l = lower(n.r), u = upper(n.r);
                Form refOut;   // DeepZ relaxation of ReLU(r)
                Form quantOut; // DeepZ relaxation of ReLU(q)
                const auto relax = [&](const Form& x, double lo, double hi, Form& result) {
                    if (hi <= 0.0) return;  // stays zero
                    if (lo >= 0.0) {
                        result = x;
                        return;
                    }
                    const double lambda = hi / (hi - lo);
                    const double mu = -lambda * lo / 2.0;
                    axpy(result, lambda, x);
                    result.c += mu;
                    addSymbol(result, fresh(), mu);
                };
                relax(n.r, l, u, refOut);
                relax(q, ql, qu, quantOut);
                const double lambdaQ = qu <= 0.0 ? 0.0 : ql >= 0.0 ? 1.0 : qu / (qu - ql);
                const auto share = errorShares(n);

                // Candidate 1 (affine): E' = ReLU~(q) - ReLU~(r)
                //   = lambdaQ * E + [ReLU~(q) - lambdaQ * q - ReLU~(r) + lambdaQ * r].
                Form mismatch = quantOut;
                axpy(mismatch, -lambdaQ, q);
                axpy(mismatch, -1.0, refOut);
                axpy(mismatch, lambdaQ, n.r);
                Form affineTotal = mismatch;
                for (const auto& part : n.e) axpy(affineTotal, lambdaQ, part);
                const double affineWidth = upper(affineTotal) - lower(affineTotal);

                // Candidate 2 (interval, ReluDiff-style): ReLU(q) - ReLU(r) lies
                // between 0 and E, intersected with what the stable cases imply.
                const Form e = totalError(n);
                const double el = lower(e), eu = upper(e);
                double lo = std::min(0.0, el), hi = std::max(0.0, eu);
                if (l >= 0.0) {  // = max(E, -r)
                    lo = std::max(lo, std::max(el, -u));
                    hi = std::min(hi, std::max(eu, -l));
                }
                if (u <= 0.0) {  // = ReLU(q)
                    lo = std::max(lo, std::max(0.0, ql));
                    hi = std::min(hi, std::max(0.0, qu));
                }
                if (qu <= 0.0) {  // = -ReLU(r)
                    lo = std::max(lo, -std::max(0.0, u));
                    hi = std::min(hi, -std::max(0.0, l));
                }
                if (ql >= 0.0) {  // = E + min(r, 0)
                    lo = std::max(lo, el + std::min(l, 0.0));
                    hi = std::min(hi, eu + std::min(u, 0.0));
                }
                if (hi < lo) hi = lo = 0.5 * (lo + hi);  // only through slack; keep sound and tiny

                std::vector<Form> next(n.e.size());
                const bool bothInactive = u <= 0.0 && qu <= 0.0;
                const bool bothActive = l >= 0.0 && ql >= 0.0;
                if (bothInactive) {
                    // ReLU(q) - ReLU(r) = 0
                } else if (bothActive) {
                    next = n.e;  // ReLU(q) - ReLU(r) = E exactly
                } else if (affine_ && affineWidth <= (hi - lo) * (1.0 + 1e-9)) {
                    for (std::size_t s = 0; s < n.e.size(); ++s) {
                        axpy(next[s], lambdaQ, n.e[s]);
                        axpy(next[s], share[s], mismatch);
                    }
                } else {
                    const std::size_t symbol = fresh();
                    for (std::size_t s = 0; s < n.e.size(); ++s) {
                        if (share[s] == 0.0) continue;
                        next[s].c = share[s] * 0.5 * (lo + hi);
                        addSymbol(next[s], symbol, share[s] * 0.5 * (hi - lo));
                    }
                }
                n.r = std::move(refOut);
                n.e = std::move(next);
                ql = std::max(0.0, ql);
                qu = std::max(0.0, qu);
            }
            bool exact = relu && qu <= 0.0;  // provably 0: rounds exactly
            if (outMax > 0) {
                // Saturation s = clamp(Y) - Y with Y the value after ReLU:
                // min(Y, U) - Y = -ReLU(Y - U) and max(Y, -U) - Y = ReLU(-U - Y),
                // both relaxed like a ReLU (DeepZ), so correlations survive.
                const double U = static_cast<double>(outMax) * outScale[index];
                const auto saturate = [&](double sign, double lo, double hi) {
                    // adds sign * ReLU(sign * (Y - U')) where [lo, hi] bounds sign * (Y - U')
                    if (hi <= 0.0) return;
                    bounds.saturates = true;
                    const Form y = quantValue(n);
                    if (lo >= 0.0) {
                        // always saturated: quant value is the constant limit
                        axpy(n.e[rs], -1.0, y);
                        n.e[rs].c += sign * U;
                        exact = true;
                        return;
                    }
                    const double lambda = hi / (hi - lo);
                    const double mu = -lambda * lo / 2.0;
                    // -sign * (lambda * sign * (Y - sign*U) + mu + mu * eta)
                    axpy(n.e[rs], -lambda, y);
                    n.e[rs].c += lambda * sign * U - sign * mu;
                    addSymbol(n.e[rs], fresh(), mu);
                };
                saturate(1.0, ql - U, qu - U);
                ql = std::min(ql, U);
                qu = std::min(qu, U);
                if (!relu) {
                    saturate(-1.0, -U - qu, -U - ql);
                    ql = std::max(ql, -U);
                    qu = std::max(qu, -U);
                }
            }
            bounds.preLo.push_back(ql);
            bounds.preHi.push_back(qu);
            // Rounding to the nearest multiple of the output scale.
            if (!exact) addSymbol(n.e[rs], fresh(), 0.5 * outScale[index] * (1.0 + 1e-12));
        }
        if (!affine_) collapse(layer);
        return bounds;
    }

private:
    // Interval ablation: forget all correlations between neurons.
    void collapse(std::vector<Neuron>& layer) {
        for (auto& n : layer) {
            const double l = lower(n.r), u = upper(n.r);
            const Form e = totalError(n);
            const double el = lower(e), eu = upper(e);
            const auto share = errorShares(n);
            n.r = Form{};
            n.r.c = 0.5 * (l + u);
            addSymbol(n.r, fresh(), 0.5 * (u - l));
            const std::size_t symbol = fresh();
            for (std::size_t s = 0; s < n.e.size(); ++s) {
                n.e[s] = Form{};
                if (share[s] == 0.0) continue;
                n.e[s].c = share[s] * 0.5 * (el + eu);
                addSymbol(n.e[s], symbol, share[s] * 0.5 * (eu - el));
            }
        }
    }

    std::size_t sources_;
    bool affine_;
    bool provenance_;
    std::size_t symbols_ = 0;
};

// Effective real weights of the emitted integers: the value represented by the
// requantised output, before rounding, is  sum_k W^_k x^_k + b^  with
//   W^_k = outScale[o] * m_c * 2^-s_c * qW[w_k] / inScale[i_k],
//   b^   = outScale[o] * m_c * 2^-s_c * qB[c]          (c = channel of o).
void effectiveWeights(const QLayer& layer, std::vector<std::vector<double>>& weight, std::vector<double>& bias) {
    weight.assign(layer.out, {});
    bias.assign(layer.out, 0.0);
    for (std::size_t o = 0; o < layer.out; ++o) {
        const std::size_t c = layer.channelOf(o);
        const double factor = layer.outScale[o] * std::ldexp(static_cast<double>(layer.multiplier[c]), -layer.shift[c]);
        weight[o].reserve(layer.taps[o].size());
        for (const auto& tap : layer.taps[o]) {
            weight[o].push_back(factor * layer.qWeight[tap.weight] / layer.inScale[tap.input]);
        }
        bias[o] = factor * static_cast<double>(layer.qBias[c]);
    }
}

Box clipToDomain(const Box& box, const Box& domain) {
    Box clipped = box;
    for (std::size_t i = 0; i < box.size(); ++i) {
        clipped.lo[i] = std::max(box.lo[i], domain.lo[i]);
        clipped.hi[i] = std::min(box.hi[i], domain.hi[i]);
        if (clipped.hi[i] < clipped.lo[i]) clipped.hi[i] = clipped.lo[i];
    }
    return clipped;
}

int accumulatorBitsFor(std::int64_t bound) {
    if (bound <= 32767) return 16;
    if (bound <= 2147483647LL) return 32;
    return 64;
}

std::int64_t signedMaxBits(int bits) { return (std::int64_t{1} << (bits - 1)) - 1; }

// Scale group of each output neuron: one per channel for convolutions (the
// next layer shares weights across positions), one per neuron for hidden dense
// layers, one for the logits (so the argmax compares integers).
std::size_t outputGroup(const DenseLayer& layer, std::size_t o, bool last, bool perChannel) {
    if (last || !perChannel) return 0;
    return layer.channelOf(o);
}

// Largest activation of every neuron over the calibration inputs (after
// ReLU for hidden layers, absolute value for the logits).
std::vector<std::vector<double>> calibrationRanges(const DenseNetwork& network,
                                                   const std::vector<std::vector<double>>& rows) {
    std::vector<std::vector<double>> range;
    for (const auto& layer : network.layers) range.emplace_back(layer.out, 0.0);
    for (const auto& row : rows) {
        std::vector<double> current = row;
        for (std::size_t l = 0; l < network.layers.size(); ++l) {
            const auto& layer = network.layers[l];
            std::vector<double> next(layer.out);
            for (std::size_t o = 0; o < layer.out; ++o) {
                double sum = layer.bias[layer.channelOf(o)];
                for (const auto& tap : layer.taps[o]) sum += layer.weight[tap.weight] * current[tap.input];
                next[o] = layer.relu ? std::max(0.0, sum) : sum;
                range[l][o] = std::max(range[l][o], std::fabs(next[o]));
            }
            current = std::move(next);
        }
    }
    return range;
}

}  // namespace

std::size_t errorSourceCount(std::size_t layers) { return 1 + 2 * layers; }

std::string errorSourceName(std::size_t source, std::size_t layers) {
    if (source == 0) return "input quantisation";
    if (source <= layers) return "layer " + std::to_string(source - 1) + " weights";
    return "layer " + std::to_string(source - 1 - layers) + " rounding";
}

RegionAnalysis analyzeRegion(const DenseNetwork& network, const QProgram& program, const Box& region,
                             const AnalysisOptions& options, int refClass) {
    const std::size_t L = program.layers.size();
    const Box box = clipToDomain(region, program.domain);
    Engine engine(errorSourceCount(L), options.affine, options.provenance);
    auto values = engine.input(box, program.inputScale, inputQuantisationSlack(program.inputMax));
    for (std::size_t l = 0; l < L; ++l) {
        std::vector<std::vector<double>> weight;
        std::vector<double> bias;
        effectiveWeights(program.layers[l], weight, bias);
        values = engine.affine(values, network.layers[l], weight, bias, 1 + l);
        engine.activate(values, network.layers[l].relu, program.layers[l].outScale, 1 + L + l,
                        program.layers[l].outMax);
    }

    RegionAnalysis result;
    result.refClass = refClass >= 0 ? refClass : argmaxFirst(referenceLogits(network, box.center()));
    const std::size_t classes = values.size();
    for (const auto& n : values) {
        const Form e = totalError(n);
        result.logitErrorBound.push_back(std::max(std::fabs(lower(e)), std::fabs(upper(e))));
        const Form q = quantValue(n);
        result.logitLower.push_back(lower(q));
        result.logitUpper.push_back(upper(q));
    }
    const std::size_t c = static_cast<std::size_t>(result.refClass);
    result.refMarginLowerBound = std::numeric_limits<double>::infinity();
    result.quantMarginLowerBound = std::numeric_limits<double>::infinity();
    Form worstQuant;
    Form worstRef;
    result.blame.assign(errorSourceCount(L), 0.0);
    for (std::size_t j = 0; j < classes; ++j) {
        if (j == c) continue;
        Form refMargin = values[c].r;
        axpy(refMargin, -1.0, values[j].r);
        Form quantMargin = refMargin;
        for (std::size_t s = 0; s < engine.sources(); ++s) {
            axpy(quantMargin, 1.0, values[c].e[s]);
            axpy(quantMargin, -1.0, values[j].e[s]);
        }
        const double refLb = lower(refMargin);
        const double quantLb = lower(quantMargin);
        // Ties in the integer program go to the lower index, so a rival with a
        // smaller index must be beaten strictly by at least one LSB.
        const double needed = j < c ? program.layers.back().outScale[0] * 0.5 : 0.0;
        if (refLb < result.refMarginLowerBound) {
            result.refMarginLowerBound = refLb;
            worstRef = refMargin;
        }
        if (quantLb - needed < result.quantMarginLowerBound) {
            result.quantMarginLowerBound = quantLb - needed;
            result.worstRival = static_cast<int>(j);
            worstQuant = quantMargin;
            std::fill(result.blame.begin(), result.blame.end(), 0.0);
            if (options.provenance) {
                for (std::size_t s = 0; s < engine.sources(); ++s) {
                    Form part = values[c].e[s];
                    axpy(part, -1.0, values[j].e[s]);
                    result.blame[s] = radius(part) + std::max(0.0, -part.c);
                }
            }
        }
    }
    if (classes < 2) {
        result.refMarginLowerBound = result.quantMarginLowerBound = 1.0;
    }
    // A small safety margin keeps the compiler strictly more conservative than
    // the independent checker, which re-derives these bounds in its own code.
    constexpr double kSafety = 1e-9;
    result.refCertified = result.refMarginLowerBound > kSafety;
    result.quantCertified = result.refCertified && result.quantMarginLowerBound > kSafety;
    const std::size_t n = box.size();
    result.splitScore.assign(n, 0.0);
    result.descentDirection.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double qc = coefficientOf(worstQuant, i);
        const double rc = coefficientOf(worstRef, i);
        result.splitScore[i] = result.refCertified ? std::fabs(qc) : std::fabs(rc);
        result.descentDirection[i] = qc > 0 ? -1.0 : qc < 0 ? 1.0 : 0.0;
    }
    return result;
}

int worstCaseAccumulatorBits(const QLayer& layer, std::int64_t inputMagnitude) {
    std::int64_t worst = 0;
    for (std::size_t o = 0; o < layer.out; ++o) {
        std::int64_t bound = std::llabs(layer.qBias[layer.channelOf(o)]);
        for (const auto& tap : layer.taps[o]) {
            bound += std::llabs(static_cast<std::int64_t>(layer.qWeight[tap.weight])) * inputMagnitude;
        }
        worst = std::max(worst, bound);
    }
    return accumulatorBitsFor(worst);
}

std::optional<QProgram> lowerToInteger(const DenseNetwork& network, const Box& domain,
                                       const PrecisionConfig& config, std::string& error,
                                       const AnalysisOptions& options) {
    const std::size_t L = network.layers.size();
    if (config.weightBits.size() != L || config.activationBits.size() != L) {
        error = "precision configuration does not match the layer count";
        return std::nullopt;
    }
    QProgram program;
    program.name = network.name;
    program.inputs = network.inputs;
    program.classes = network.classes;
    program.inputShape = network.inputShape;
    program.domain = domain;
    program.config = config;
    program.inputBits = config.activationBits[0];
    program.inputUnsigned = std::all_of(domain.lo.begin(), domain.lo.end(), [](double v) { return v >= 0.0; });
    const std::int64_t inputQmax = program.inputUnsigned ? (std::int64_t{1} << program.inputBits) - 1
                                                         : signedMaxBits(program.inputBits);
    program.inputMax = inputQmax;
    program.inputMin = program.inputUnsigned ? 0 : -inputQmax;
    // Input scales: per input group (feature, or channel for images) or one
    // shared. One LSB of headroom keeps the float32 rounding off the clamp.
    program.inputScale.assign(network.inputs, 1.0);
    std::vector<double> groupMagnitude(network.inputs, 0.0);
    double shared = 0.0;
    for (std::size_t i = 0; i < domain.size(); ++i) {
        const double magnitude = std::max(std::fabs(domain.lo[i]), std::fabs(domain.hi[i]));
        groupMagnitude[network.inputGroupOf(i)] = std::max(groupMagnitude[network.inputGroupOf(i)], magnitude);
        shared = std::max(shared, magnitude);
    }
    for (std::size_t i = 0; i < network.inputs; ++i) {
        const double magnitude = options.perChannelScales ? groupMagnitude[network.inputGroupOf(i)] : shared;
        program.inputScale[i] = magnitude > 0.0 ? magnitude / static_cast<double>(inputQmax - 1) : 1.0;
    }
    const double inputSlack = inputQuantisationSlack(program.inputMax);
    std::vector<std::int64_t> qLo(network.inputs), qHi(network.inputs);
    for (std::size_t i = 0; i < network.inputs; ++i) {
        const double s = program.inputScale[i];
        qLo[i] = std::max<std::int64_t>(program.inputMin, static_cast<std::int64_t>(std::ceil(domain.lo[i] / s - inputSlack)));
        qHi[i] = std::min<std::int64_t>(program.inputMax, static_cast<std::int64_t>(std::floor(domain.hi[i] / s + inputSlack)));
    }

    Engine engine(errorSourceCount(L), options.affine, false);
    auto values = engine.input(domain, program.inputScale, inputSlack);
    std::vector<double> inScale = program.inputScale;
    const bool calibrated = options.calibration && !options.calibration->empty();
    program.calibrated = calibrated;
    const auto calibration = calibrated ? calibrationRanges(network, *options.calibration)
                                        : std::vector<std::vector<double>>{};
    // Inputs proved to be exactly 0 on the whole domain, in the float model
    // and in the integer program (dead ReLUs of the previous layer).
    std::vector<bool> inputDead(network.inputs, false);
    for (std::size_t l = 0; l < L; ++l) {
        const DenseLayer& dense = network.layers[l];
        QLayer layer;
        layer.kind = dense.kind;
        layer.conv = dense.conv;
        layer.in = dense.in;
        layer.out = dense.out;
        layer.channels = dense.channels();
        layer.relu = dense.relu;
        layer.taps = dense.taps;
        layer.inScale = inScale;
        layer.weightBits = config.weightBits[l];
        const bool last = l + 1 == L;
        layer.outBits = last ? config.logitBits : config.activationBits[l + 1];
        layer.outUnsigned = !last && dense.relu;
        if (layer.weightBits < 2 || layer.weightBits > 16 || layer.outBits < 2 || layer.outBits > 16) {
            error = "bit widths must be between 2 and 16";
            return std::nullopt;
        }
        layer.outMax = layer.outUnsigned ? (std::int64_t{1} << layer.outBits) - 1 : signedMaxBits(layer.outBits);
        layer.outMin = layer.outUnsigned ? 0 : -layer.outMax;

        // Symmetric per-output-channel quantisation of W * inScale(input). For
        // a convolution, every parameter of input channel ci meets inputs of
        // that channel only, which share one scale.
        const auto paramInScale = [&](std::size_t w) {
            if (dense.kind == LayerKind::Conv) {
                const auto& g = dense.conv;
                const std::size_t ci = (w / (g.kH * g.kW)) % g.inC;
                return inScale[ci * g.inH * g.inW];
            }
            return inScale[w % dense.in];
        };
        // A parameter that only ever multiplies proved-zero inputs cannot
        // matter: it is left out of the scale and emitted as 0.
        const auto paramDead = [&](std::size_t w) {
            if (dense.kind == LayerKind::Conv) {
                const auto& g = dense.conv;
                const std::size_t ci = (w / (g.kH * g.kW)) % g.inC;
                for (std::size_t k = ci * g.inH * g.inW; k < (ci + 1) * g.inH * g.inW; ++k) {
                    if (!inputDead[k]) return false;
                }
                return true;
            }
            return static_cast<bool>(inputDead[w % dense.in]);
        };
        const std::int64_t wmax = signedMaxBits(layer.weightBits);
        layer.qWeight.assign(dense.weight.size(), 0);
        layer.weightScale.assign(layer.channels, 0.0);
        std::vector<bool> dead(dense.weight.size(), false);
        for (std::size_t w = 0; w < dense.weight.size(); ++w) {
            dead[w] = paramDead(w);
            if (dead[w]) continue;
            auto& scale = layer.weightScale[dense.weightChannel(w)];
            scale = std::max(scale, std::fabs(dense.weight[w] * paramInScale(w)));
        }
        for (double& scale : layer.weightScale) scale = scale > 0.0 ? scale / static_cast<double>(wmax) : 1.0;
        for (std::size_t w = 0; w < dense.weight.size(); ++w) {
            if (dead[w]) continue;
            const double q = std::round(dense.weight[w] * paramInScale(w) / layer.weightScale[dense.weightChannel(w)]);
            layer.qWeight[w] = static_cast<std::int32_t>(std::max<double>(-wmax, std::min<double>(wmax, q)));
        }
        layer.qBias.assign(layer.channels, 0);
        for (std::size_t c = 0; c < layer.channels; ++c) {
            layer.qBias[c] = static_cast<std::int64_t>(std::llround(dense.bias[c] / layer.weightScale[c]));
        }

        // Classical bound on every partial sum, and the multiply width.
        layer.prefixBound.assign(dense.out, 0);
        std::int64_t prefixMax = 1, productMax = 0;
        for (std::size_t o = 0; o < dense.out; ++o) {
            std::int64_t bound = std::llabs(layer.qBias[layer.channelOf(o)]);
            for (const auto& tap : dense.taps[o]) {
                const std::int64_t product = std::llabs(static_cast<std::int64_t>(layer.qWeight[tap.weight])) *
                                             std::max(std::llabs(qLo[tap.input]), std::llabs(qHi[tap.input]));
                bound += product;
                productMax = std::max(productMax, product);
            }
            layer.prefixBound[o] = bound;
            prefixMax = std::max(prefixMax, bound);
        }
        if (prefixMax > (std::int64_t{1} << 40)) {
            error = "accumulator range too large at layer " + std::to_string(l);
            return std::nullopt;
        }
        layer.mulBits = productMax <= 32767 ? 16 : 32;

        // Pass 1: the exact accumulator value (sum before requantisation) is
        // weightScale[c] * acc, so its proved range bounds the final sum.
        std::vector<std::vector<double>> ideal(dense.out);
        std::vector<double> idealBias(dense.out);
        for (std::size_t o = 0; o < dense.out; ++o) {
            const std::size_t c = layer.channelOf(o);
            for (const auto& tap : dense.taps[o]) {
                ideal[o].push_back(layer.weightScale[c] * layer.qWeight[tap.weight] / inScale[tap.input]);
            }
            idealBias[o] = layer.weightScale[c] * static_cast<double>(layer.qBias[c]);
        }
        const auto pre = engine.affine(values, dense, ideal, idealBias, 1 + l);
        std::vector<double> range(dense.out, 0.0);
        layer.finalBound.assign(dense.out, 0);
        std::int64_t finalMax = 1;
        for (std::size_t o = 0; o < dense.out; ++o) {
            const Form q = quantValue(pre[o]);
            const double lo = lower(q), hi = upper(q);
            range[o] = dense.relu ? std::max(0.0, hi) : std::max(std::fabs(lo), std::fabs(hi));
            const double sigma = layer.weightScale[layer.channelOf(o)];
            const double magnitude = std::max(std::fabs(lo), std::fabs(hi)) / sigma;
            layer.finalBound[o] = std::min<std::int64_t>(
                layer.prefixBound[o], static_cast<std::int64_t>(std::ceil(magnitude * (1.0 + 1e-9) + 1e-6)));
            finalMax = std::max(finalMax, layer.finalBound[o]);
        }
        const int classicBits = accumulatorBitsFor(prefixMax);
        const int modularBits = accumulatorBitsFor(finalMax);
        layer.modular = options.modularAccumulation && modularBits < classicBits;
        layer.accBits = layer.modular ? modularBits : classicBits;
        const std::int64_t accMax = layer.modular ? finalMax : prefixMax;

        // Output scales: one per group, from the proved range or from data.
        const bool perChannel = options.perChannelScales;
        std::vector<double> groupRange(dense.out, 0.0);
        for (std::size_t o = 0; o < dense.out; ++o) {
            auto& g = groupRange[outputGroup(dense, o, last, perChannel)];
            const double r = calibrated ? std::min(range[o], calibration[l][o] * options.headroom) : range[o];
            g = std::max(g, r);
        }
        layer.outScale.assign(dense.out, 0.0);
        double smallestScale = 0.0;
        for (std::size_t o = 0; o < dense.out; ++o) {
            const double r = groupRange[outputGroup(dense, o, last, perChannel)];
            if (r <= 0.0) continue;
            layer.outScale[o] = r / static_cast<double>(layer.outMax) * (1.0 + 1e-6);
            smallestScale = smallestScale > 0.0 ? std::min(smallestScale, layer.outScale[o]) : layer.outScale[o];
        }
        // A group that is always 0 takes the layer's smallest scale (any scale is exact for 0).
        for (double& scale : layer.outScale) {
            if (scale <= 0.0) scale = smallestScale > 0.0 ? smallestScale : 1.0;
        }

        int bitLength = 0;
        while ((std::int64_t{1} << bitLength) <= accMax) ++bitLength;
        const int multiplierBits = std::min(31, 62 - bitLength);
        bool saturationFree = false;
        std::vector<Neuron> next;
        Engine::Activation bounds;
        for (int attempt = 0; attempt < 40 && !saturationFree; ++attempt) {
            layer.multiplier.assign(layer.channels, 0);
            layer.shift.assign(layer.channels, 1);
            for (std::size_t c = 0; c < layer.channels; ++c) {
                // A channel's neurons share one output scale: use its first neuron.
                const std::size_t representative =
                    dense.kind == LayerKind::Conv ? c * dense.conv.outH * dense.conv.outW : c;
                const double M = layer.weightScale[c] / layer.outScale[representative];
                int exponent = 0;
                std::frexp(M, &exponent);  // M = f * 2^exponent, f in [0.5, 1)
                int shift = multiplierBits - exponent;
                if (shift < 1) {
                    error = "requantisation ratio too large at layer " + std::to_string(l);
                    return std::nullopt;
                }
                shift = std::min(shift, 62);
                std::int64_t m = std::llround(std::ldexp(M, shift));
                if (m >= (std::int64_t{1} << multiplierBits)) {
                    m >>= 1;
                    --shift;
                }
                layer.multiplier[c] = static_cast<std::int32_t>(m);
                layer.shift[c] = shift;
            }
            // Pass 2: the actual integer semantics.
            std::vector<std::vector<double>> weight;
            std::vector<double> bias;
            effectiveWeights(layer, weight, bias);
            Engine trial = engine;
            next = trial.affine(values, dense, weight, bias, 1 + l);
            // Proved scales: no clamp model, so any saturation shows up as an
            // out-of-range bound and the scale grows. Calibrated: clamps modelled.
            bounds = trial.activate(next, dense.relu, layer.outScale, 1 + L + l, calibrated ? layer.outMax : 0);
            layer.qOutLo.assign(dense.out, 0);
            layer.qOutHi.assign(dense.out, 0);
            saturationFree = true;
            std::vector<bool> grow(dense.out, false);
            for (std::size_t o = 0; o < dense.out; ++o) {
                const double tLo = bounds.preLo[o] / layer.outScale[o];
                const double tHi = bounds.preHi[o] / layer.outScale[o];
                layer.qOutLo[o] = static_cast<std::int64_t>(std::floor(tLo - std::fabs(tLo) * 1e-9 - 1e-9 + 0.5));
                layer.qOutHi[o] = static_cast<std::int64_t>(std::floor(tHi + std::fabs(tHi) * 1e-9 + 1e-9 + 0.5));
                if (!calibrated && (layer.qOutHi[o] > layer.outMax || layer.qOutLo[o] < layer.outMin)) {
                    saturationFree = false;
                    grow[outputGroup(dense, o, last, perChannel)] = true;
                }
            }
            if (calibrated) program.saturationPossible = program.saturationPossible || bounds.saturates;
            if (saturationFree) {
                engine = trial;
            } else {
                for (std::size_t o = 0; o < dense.out; ++o) {
                    if (grow[outputGroup(dense, o, last, perChannel)]) layer.outScale[o] *= 1.002;
                }
            }
        }
        if (!saturationFree) {
            error = "could not place layer " + std::to_string(l) + " output range without saturation";
            return std::nullopt;
        }
        values = std::move(next);
        inputDead.assign(dense.out, false);
        for (std::size_t o = 0; o < dense.out; ++o) {
            bool zero = dense.relu && isZero(values[o].r);
            for (const auto& part : values[o].e) zero = zero && isZero(part);
            inputDead[o] = zero;
        }
        qLo = layer.qOutLo;
        qHi = layer.qOutHi;
        inScale = layer.outScale;
        program.layers.push_back(std::move(layer));
    }
    return program;
}

}  // namespace modelforge

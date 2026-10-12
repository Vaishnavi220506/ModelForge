#include "diffai.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace modelforge {
namespace {

// Affine form c + sum_k a[k] * eta_k with eta_k in [-1, 1]. Missing
// coefficients are zero, so forms grow lazily as fresh symbols appear.
struct Form {
    double c = 0.0;
    std::vector<double> a;
};

double radius(const Form& f) {
    double r = 0.0;
    for (double v : f.a) r += std::fabs(v);
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

void axpy(Form& dst, double alpha, const Form& src) {
    if (alpha == 0.0) return;
    dst.c += alpha * src.c;
    if (dst.a.size() < src.a.size()) dst.a.resize(src.a.size(), 0.0);
    for (std::size_t k = 0; k < src.a.size(); ++k) dst.a[k] += alpha * src.a[k];
}

bool isZero(const Form& f) { return f.c == 0.0 && f.a.empty(); }

void addSymbol(Form& f, std::size_t symbol, double coefficient) {
    if (f.a.size() <= symbol) f.a.resize(symbol + 1, 0.0);
    f.a[symbol] += coefficient;
}

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

class Engine {
public:
    explicit Engine(std::size_t sources, bool affine) : sources_(sources), affine_(affine) {}
    std::size_t fresh() { return symbols_++; }
    std::size_t sources() const { return sources_; }

    std::vector<Neuron> input(const Box& box, const std::vector<double>& inputScale, double slackLsb) {
        std::vector<Neuron> layer(box.size());
        for (std::size_t i = 0; i < box.size(); ++i) {
            const std::size_t symbol = fresh();  // input symbols are 0..n-1
            layer[i].r.c = 0.5 * (box.lo[i] + box.hi[i]);
            addSymbol(layer[i].r, symbol, 0.5 * (box.hi[i] - box.lo[i]));
            layer[i].e.assign(sources_, Form{});
        }
        for (std::size_t i = 0; i < box.size(); ++i) {
            addSymbol(layer[i].e[0], fresh(), inputScale[i] * slackLsb);
        }
        return layer;
    }

    // y = W x + b for the float model, y^ = W^ x^ + b^ for the integer program
    // (W^, b^: effective real weights of the emitted integers).
    std::vector<Neuron> affine(const std::vector<Neuron>& in, const DenseLayer& layer,
                               const std::vector<double>& What, const std::vector<double>& bhat,
                               std::size_t weightSource) const {
        std::vector<Neuron> out(layer.out);
        for (std::size_t o = 0; o < layer.out; ++o) {
            Neuron& n = out[o];
            n.e.assign(sources_, Form{});
            n.r.c = layer.bias[o];
            for (std::size_t i = 0; i < layer.in; ++i) {
                const double w = layer.weight[o * layer.in + i];
                const double wq = What[o * layer.in + i];
                axpy(n.r, w, in[i].r);
                axpy(n.e[weightSource], wq - w, in[i].r);
                for (std::size_t s = 0; s < sources_; ++s) {
                    if (!isZero(in[i].e[s])) axpy(n.e[s], wq, in[i].e[s]);
                }
            }
            n.e[weightSource].c += bhat[o] - layer.bias[o];
        }
        return out;
    }

    struct Activation {
        std::vector<double> preLo;  // bounds of the integer program's value before rounding
        std::vector<double> preHi;
    };

    // ReLU (if any), then rounding to the output scale.
    Activation activate(std::vector<Neuron>& layer, bool relu, const std::vector<double>& outScale,
                        std::size_t roundSource) {
        Activation bounds;
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
            bounds.preLo.push_back(ql);
            bounds.preHi.push_back(qu);
            // Rounding to the nearest multiple of the output scale.
            addSymbol(n.e[roundSource], fresh(), 0.5 * outScale[index] * (1.0 + 1e-12));
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
    std::size_t symbols_ = 0;
};

// Effective real weights of the emitted integers: the value represented by the
// requantised output, before rounding, is  W^ x^ + b^  with
//   W^[o,i] = outScale[o] * m_o * 2^-s_o * qW[o,i] / inScale[i],
//   b^[o]   = outScale[o] * m_o * 2^-s_o * qB[o].
void effectiveWeights(const QLayer& layer, std::vector<double>& weight, std::vector<double>& bias) {
    weight.assign(layer.out * layer.in, 0.0);
    bias.assign(layer.out, 0.0);
    for (std::size_t o = 0; o < layer.out; ++o) {
        const double factor = layer.outScale[o] * std::ldexp(static_cast<double>(layer.multiplier[o]), -layer.shift[o]);
        for (std::size_t i = 0; i < layer.in; ++i) {
            weight[o * layer.in + i] = factor * layer.qWeight[o * layer.in + i] / layer.inScale[i];
        }
        bias[o] = factor * static_cast<double>(layer.qBias[o]);
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
    Engine engine(errorSourceCount(L), options.affine);
    auto values = engine.input(box, program.inputScale, inputQuantisationSlack(program.inputBits));
    for (std::size_t l = 0; l < L; ++l) {
        std::vector<double> weight, bias;
        effectiveWeights(program.layers[l], weight, bias);
        values = engine.affine(values, network.layers[l], weight, bias, 1 + l);
        engine.activate(values, network.layers[l].relu, program.layers[l].outScale, 1 + L + l);
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
            result.blame.assign(engine.sources(), 0.0);
            for (std::size_t s = 0; s < engine.sources(); ++s) {
                Form part = values[c].e[s];
                axpy(part, -1.0, values[j].e[s]);
                result.blame[s] = radius(part) + std::max(0.0, -part.c);
            }
        }
    }
    if (classes < 2) {
        result.refMarginLowerBound = result.quantMarginLowerBound = 1.0;
        result.blame.assign(engine.sources(), 0.0);
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
        const double qc = i < worstQuant.a.size() ? worstQuant.a[i] : 0.0;
        const double rc = i < worstRef.a.size() ? worstRef.a[i] : 0.0;
        result.splitScore[i] = result.refCertified ? std::fabs(qc) : std::fabs(rc);
        result.descentDirection[i] = qc > 0 ? -1.0 : qc < 0 ? 1.0 : 0.0;
    }
    return result;
}

int worstCaseAccumulatorBits(const QLayer& layer, std::int64_t inputMagnitude) {
    std::int64_t worst = 0;
    for (std::size_t o = 0; o < layer.out; ++o) {
        std::int64_t bound = std::llabs(layer.qBias[o]);
        for (std::size_t i = 0; i < layer.in; ++i) {
            bound += std::llabs(static_cast<std::int64_t>(layer.qWeight[o * layer.in + i])) * inputMagnitude;
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
    program.domain = domain;
    program.config = config;
    program.inputBits = config.activationBits[0];
    const std::int64_t inputQmax = signedMaxBits(program.inputBits);
    program.inputMax = inputQmax;
    program.inputMin = -inputQmax;
    // Input scales: per feature (folded into layer 0's weights) or one shared.
    // One LSB of headroom keeps the float32 rounding away from the clamp.
    program.inputScale.assign(network.inputs, 1.0);
    double shared = 0.0;
    for (std::size_t i = 0; i < domain.size(); ++i) {
        const double magnitude = std::max(std::fabs(domain.lo[i]), std::fabs(domain.hi[i]));
        shared = std::max(shared, magnitude);
        program.inputScale[i] = magnitude > 0.0 ? magnitude / static_cast<double>(inputQmax - 1) : 1.0;
    }
    if (!options.perChannelScales) {
        std::fill(program.inputScale.begin(), program.inputScale.end(),
                  shared > 0.0 ? shared / static_cast<double>(inputQmax - 1) : 1.0);
    }
    const double inputSlack = inputQuantisationSlack(program.inputBits);
    std::vector<std::int64_t> qLo(network.inputs), qHi(network.inputs);
    for (std::size_t i = 0; i < network.inputs; ++i) {
        const double s = program.inputScale[i];
        qLo[i] = std::max<std::int64_t>(program.inputMin, static_cast<std::int64_t>(std::ceil(domain.lo[i] / s - inputSlack)));
        qHi[i] = std::min<std::int64_t>(program.inputMax, static_cast<std::int64_t>(std::floor(domain.hi[i] / s + inputSlack)));
    }

    Engine engine(errorSourceCount(L), options.affine);
    auto values = engine.input(domain, program.inputScale, inputSlack);
    std::vector<double> inScale = program.inputScale;
    for (std::size_t l = 0; l < L; ++l) {
        const DenseLayer& dense = network.layers[l];
        QLayer layer;
        layer.in = dense.in;
        layer.out = dense.out;
        layer.relu = dense.relu;
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

        // Symmetric per-output-channel quantisation of W[o,i] * inScale[i].
        const std::int64_t wmax = signedMaxBits(layer.weightBits);
        layer.qWeight.assign(dense.weight.size(), 0);
        layer.weightScale.assign(dense.out, 1.0);
        layer.qBias.assign(dense.out, 0);
        for (std::size_t o = 0; o < dense.out; ++o) {
            double largest = 0.0;
            for (std::size_t i = 0; i < dense.in; ++i) {
                largest = std::max(largest, std::fabs(dense.weight[o * dense.in + i] * inScale[i]));
            }
            const double scale = largest > 0.0 ? largest / static_cast<double>(wmax) : 1.0;
            layer.weightScale[o] = scale;
            for (std::size_t i = 0; i < dense.in; ++i) {
                const double q = std::round(dense.weight[o * dense.in + i] * inScale[i] / scale);
                layer.qWeight[o * dense.in + i] = static_cast<std::int32_t>(std::max<double>(-wmax, std::min<double>(wmax, q)));
            }
            layer.qBias[o] = static_cast<std::int64_t>(std::llround(dense.bias[o] / scale));
        }
        // Proved bound on every partial sum, from the proved input ranges.
        layer.accBound.assign(dense.out, 0);
        std::int64_t accMax = 1;
        for (std::size_t o = 0; o < dense.out; ++o) {
            std::int64_t bound = std::llabs(layer.qBias[o]);
            for (std::size_t i = 0; i < dense.in; ++i) {
                bound += std::llabs(static_cast<std::int64_t>(layer.qWeight[o * dense.in + i])) *
                         std::max(std::llabs(qLo[i]), std::llabs(qHi[i]));
            }
            layer.accBound[o] = bound;
            accMax = std::max(accMax, bound);
        }
        if (accMax > (std::int64_t{1} << 40)) {
            error = "accumulator range too large at layer " + std::to_string(l);
            return std::nullopt;
        }
        layer.accBits = accumulatorBitsFor(accMax);

        // Pass 1: range of the pre-rounding value with an exact multiplier.
        std::vector<double> ideal(dense.weight.size()), idealBias(dense.out);
        for (std::size_t o = 0; o < dense.out; ++o) {
            for (std::size_t i = 0; i < dense.in; ++i) {
                ideal[o * dense.in + i] = layer.weightScale[o] * layer.qWeight[o * dense.in + i] / inScale[i];
            }
            idealBias[o] = layer.weightScale[o] * static_cast<double>(layer.qBias[o]);
        }
        const auto pre = engine.affine(values, dense, ideal, idealBias, 1 + l);
        std::vector<double> range(dense.out, 0.0);
        double sharedRange = 0.0;
        for (std::size_t o = 0; o < dense.out; ++o) {
            const Form q = quantValue(pre[o]);
            const double lo = lower(q), hi = upper(q);
            range[o] = dense.relu ? std::max(0.0, hi) : std::max(std::fabs(lo), std::fabs(hi));
            sharedRange = std::max(sharedRange, range[o]);
        }
        // Logits share one scale so the argmax can compare integers directly.
        const bool perChannel = options.perChannelScales && !last;
        layer.outScale.assign(dense.out, 1.0);
        for (std::size_t o = 0; o < dense.out; ++o) {
            const double r = perChannel ? range[o] : sharedRange;
            layer.outScale[o] = r > 0.0 ? r / static_cast<double>(layer.outMax) * (1.0 + 1e-6) : 1.0;
        }

        int bitLength = 0;
        while ((std::int64_t{1} << bitLength) <= accMax) ++bitLength;
        const int multiplierBits = std::min(31, 62 - bitLength);
        bool saturationFree = false;
        std::vector<Neuron> next;
        Engine::Activation bounds;
        for (int attempt = 0; attempt < 40 && !saturationFree; ++attempt) {
            layer.multiplier.assign(dense.out, 0);
            layer.shift.assign(dense.out, 1);
            for (std::size_t o = 0; o < dense.out; ++o) {
                const double M = layer.weightScale[o] / layer.outScale[o];
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
                layer.multiplier[o] = static_cast<std::int32_t>(m);
                layer.shift[o] = shift;
            }
            // Pass 2: the actual integer semantics.
            std::vector<double> weight, bias;
            effectiveWeights(layer, weight, bias);
            Engine trial = engine;
            next = trial.affine(values, dense, weight, bias, 1 + l);
            bounds = trial.activate(next, dense.relu, layer.outScale, 1 + L + l);
            layer.qOutLo.assign(dense.out, 0);
            layer.qOutHi.assign(dense.out, 0);
            saturationFree = true;
            bool grow = false;
            for (std::size_t o = 0; o < dense.out; ++o) {
                const double tLo = bounds.preLo[o] / layer.outScale[o];
                const double tHi = bounds.preHi[o] / layer.outScale[o];
                layer.qOutLo[o] = static_cast<std::int64_t>(std::floor(tLo - std::fabs(tLo) * 1e-9 - 1e-9 + 0.5));
                layer.qOutHi[o] = static_cast<std::int64_t>(std::floor(tHi + std::fabs(tHi) * 1e-9 + 1e-9 + 0.5));
                if (layer.qOutHi[o] > layer.outMax || layer.qOutLo[o] < layer.outMin) {
                    saturationFree = false;
                    grow = true;
                    if (perChannel) layer.outScale[o] *= 1.002;
                }
            }
            if (saturationFree) {
                engine = trial;
            } else if (grow && !perChannel) {
                for (double& scale : layer.outScale) scale *= 1.002;
            }
        }
        if (!saturationFree) {
            error = "could not place layer " + std::to_string(l) + " output range without saturation";
            return std::nullopt;
        }
        values = std::move(next);
        qLo = layer.qOutLo;
        qHi = layer.qOutHi;
        inScale = layer.outScale;
        program.layers.push_back(std::move(layer));
    }
    return program;
}

}  // namespace modelforge

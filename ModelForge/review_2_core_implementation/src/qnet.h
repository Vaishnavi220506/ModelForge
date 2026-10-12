#pragma once

// ForgeCert, part 1: the integer-only target of ModelForge.
//
// A float model (IR) is lowered to a *quantised integer program* (QProgram):
// every tensor is an integer array with a scale, every dense layer is an
// integer dot product into an accumulator of an inferred width, followed by
// fixed-point requantisation (an integer multiplier and a right shift). The
// program uses no floating point after the input has been quantised, so the
// emitted C99 runs on microcontrollers without an FPU.
//
// The precision of every tensor is a compiler decision (PrecisionConfig).
// ForgeCert chooses it so that the *emitted* program provably makes the same
// decision as the float model on user-specified input regions (diffai.h,
// certify.h).

#include "ir.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace modelforge {

// A chain of dense layers recovered from the IR (Gemm / MatMul+Add, optional
// ReLU after every hidden layer). The output activation (Softmax, Sigmoid or
// none) is order-preserving, so the decision is the argmax of the logits.
struct DenseLayer {
    std::string name;
    std::size_t in = 0;
    std::size_t out = 0;
    std::vector<double> weight;  // out x in, row-major: weight[o * in + i]
    std::vector<double> bias;    // out
    bool relu = false;
};

struct DenseNetwork {
    std::string name;
    std::size_t inputs = 0;
    std::size_t classes = 0;
    std::vector<DenseLayer> layers;
    std::string outputActivation;  // "softmax", "sigmoid" or "none"
    bool singleLogitPadded = false; // a 1-logit head got an implicit 0 logit
};

std::optional<DenseNetwork> extractDenseNetwork(const IRGraph& graph, std::string& error);
std::vector<double> referenceLogits(const DenseNetwork& network, const std::vector<double>& x);
int argmaxFirst(const std::vector<double>& values);
int argmaxFirst(const std::vector<std::int64_t>& values);

// An axis-aligned box of inputs.
struct Box {
    std::vector<double> lo;
    std::vector<double> hi;
    std::size_t size() const { return lo.size(); }
    bool contains(const std::vector<double>& x) const;
    std::vector<double> center() const;
};

// Bit widths chosen by the compiler. activationBits[0] is the input tensor,
// activationBits[l] (l = 1..L-1) the output of hidden layer l-1. The logits
// always use logitBits (signed). Hidden ReLU outputs are unsigned.
struct PrecisionConfig {
    std::vector<int> weightBits;
    std::vector<int> activationBits;
    int logitBits = 16;
    bool operator==(const PrecisionConfig& other) const {
        return weightBits == other.weightBits && activationBits == other.activationBits &&
               logitBits == other.logitBits;
    }
    std::string describe() const;  // e.g. "w[4,8] a[8,8]"
};

PrecisionConfig uniformConfig(std::size_t layers, int weightBits, int activationBits);

struct QLayer {
    std::size_t in = 0;
    std::size_t out = 0;
    bool relu = false;
    int weightBits = 8;
    std::vector<std::int32_t> qWeight;  // out x in
    // Weights are quantised after folding the input scales in: the integer
    // weight approximates W[o,i] * inScale[i] / weightScale[o], so per-channel
    // input scales cost nothing at run time.
    std::vector<double> weightScale;    // per output channel
    std::vector<std::int64_t> qBias;    // at scale weightScale[o]
    std::vector<double> inScale;        // per input channel
    std::vector<double> outScale;       // per output channel (one shared value for logits)
    int outBits = 8;
    bool outUnsigned = true;
    std::int64_t outMin = 0;
    std::int64_t outMax = 255;
    std::vector<std::int32_t> multiplier;  // requantisation: q = round(acc * m / 2^shift)
    std::vector<int> shift;
    int accBits = 32;                      // 16, 32 or 64; proved by range analysis
    std::vector<std::int64_t> accBound;    // proved bound on |every partial sum|
    std::vector<std::int64_t> qOutLo;      // proved range of the quantised output
    std::vector<std::int64_t> qOutHi;
};

struct QProgram {
    std::string name;
    std::size_t inputs = 0;
    std::size_t classes = 0;
    int inputBits = 8;
    std::vector<double> inputScale;  // per input feature
    std::int64_t inputMin = -127;
    std::int64_t inputMax = 127;
    Box domain;  // the input domain every global claim is about
    std::vector<QLayer> layers;
    PrecisionConfig config;
};

// Error radius of input quantisation in units of the input scale. The C code
// computes x * (1/s) in float32; this covers that rounding as well.
double inputQuantisationSlack(int inputBits);

// Exactly what the emitted C does (float32 input scaling, round half away
// from zero, saturation).
std::vector<std::int64_t> quantizeInput(const QProgram& program, const std::vector<double>& x);

struct IntegerRun {
    std::vector<std::int64_t> logits;
    int decision = -1;
    bool accumulatorOverflow = false;  // a partial sum left the accumulator type
    bool saturated = false;            // a requantised value hit the clamp
};

// Bit-exact integer semantics of the emitted C program.
IntegerRun runInteger(const QProgram& program, const std::vector<std::int64_t>& qInput);
IntegerRun runInteger(const QProgram& program, const std::vector<double>& x);
std::vector<double> dequantizedLogits(const QProgram& program, const IntegerRun& run);

// Rounding right shift used everywhere: floor((t + 2^(s-1)) / 2^s).
std::int64_t roundShift(std::int64_t value, int shift);

// Storage of the emitted arrays: weights (packed 4-bit, int8 or int16),
// biases, requantisation constants, and the activation buffers (RAM).
struct ProgramCost {
    double weightBytes = 0;  // flash for weights only
    double flashBytes = 0;   // weights + biases + multipliers + shifts
    double ramBytes = 0;     // two largest consecutive activation buffers
    double idealBits = 0;    // sum of weightBits * |W| (sub-byte packing)
};
ProgramCost programCost(const QProgram& program);
double float32Bytes(const DenseNetwork& network);  // weights + biases in float32

// Emits model_q.h, model_q.c (integer-only C99), main_q.c and CMakeLists.txt.
// The data arrays in model_q.c are what the independent checker reads.
bool emitIntegerC(const QProgram& program, const std::string& directory, std::string& error);

}  // namespace modelforge

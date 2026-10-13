#pragma once

// ForgeCert, part 1: the integer-only target of ModelForge.
//
// A float model (IR) is lowered to a *quantised integer program* (QProgram):
// every tensor is an integer array with per-channel scales, every linear
// layer (dense, grouped 2-D convolution, residual add) is an integer dot
// product into an
// accumulator of a proved width, followed by fixed-point requantisation (an
// integer multiplier and a right shift). The program uses no floating point
// after the input has been quantised, so the emitted C99 runs on
// microcontrollers without an FPU (AVR, Cortex-M0).
//
// Two width decisions are made from *proved* ranges (diffai.h):
//   * the accumulator type: classical (every partial sum fits) or modular
//     (partial sums may wrap around in unsigned arithmetic; only the final sum
//     must fit, which the abstract interpretation bounds much more tightly);
//   * the multiply type: 16-bit when |weight| * |activation| provably fits.
// The precision of every tensor is chosen by the tuner (certify.h).

#include "ir.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace modelforge {

// Dense: fully connected. Conv: 2-D convolution (grouped; depthwise when
// groups = channels); average pooling, global average pooling and a batch
// normalisation that cannot be folded are lowered to depthwise convolutions
// with fixed weights. Add: element-wise sum of two tensors (residual
// connections), a linear layer with one weight per source and channel.
// MaxPool: exact maximum of integer codes, no requantisation.
enum class LayerKind { Dense, Conv, Add, MaxPool };

struct ConvShape {
    std::size_t inC = 0, inH = 0, inW = 0;
    std::size_t outC = 0, outH = 0, outW = 0;
    std::size_t kH = 0, kW = 0;
    std::size_t stride = 1, pad = 0;
    std::size_t groups = 1;
};

// Shape of a tensor: C x H x W for feature maps, n x 1 x 1 for vectors.
struct TensorShape {
    std::size_t c = 0, h = 1, w = 1;
    bool spatial = false;
    std::size_t size() const { return c * h * w; }
    std::size_t channelOf(std::size_t i) const { return spatial ? i / (h * w) : i; }
};

// One term of an output neuron's dot product: input neuron x parameter. The
// input index runs over the concatenation of the layer's source tensors.
struct Tap {
    std::uint32_t input = 0;
    std::uint32_t weight = 0;
};

// A layer recovered from the IR, with optional ReLU. Layers form a DAG over
// tensors: tensor 0 is the network input, tensor k + 1 the output of layer
// k; `sources` lists the tensors a layer reads (two for Add). Flatten is the
// identity on data and disappears; batch normalisation is folded into the
// preceding linear layer when it has no other consumer.
struct DenseLayer {
    std::string name;
    LayerKind kind = LayerKind::Dense;
    std::vector<std::size_t> sources{0};
    TensorShape output;
    std::size_t in = 0;   // total size of the sources
    std::size_t out = 0;
    std::vector<double> weight;  // dense: out x in; conv: outC x inC/groups x kH x kW; add: C x 2
    std::vector<double> bias;    // one per output channel
    bool relu = false;
    ConvShape conv;              // conv and pooling geometry
    std::size_t weightsPerChannel = 0;
    std::vector<std::vector<Tap>> taps;  // per output neuron
    std::vector<std::string> irNodes;    // IR nodes lowered into this layer (incl. folded ones)

    bool hasWeights() const { return kind != LayerKind::MaxPool; }
    std::size_t channels() const { return output.spatial ? output.c : out; }
    std::size_t channelOf(std::size_t o) const { return output.channelOf(o); }
    std::size_t weightChannel(std::size_t w) const { return w / weightsPerChannel; }
};

// Fills `taps` from the layer kind and the shapes of its sources.
void buildTaps(DenseLayer& layer, const std::vector<TensorShape>& tensors);

struct DenseNetwork {
    std::string name;
    std::size_t inputs = 0;
    std::size_t classes = 0;
    std::vector<std::size_t> inputShape;  // {C, H, W} for images, {n} otherwise
    std::vector<TensorShape> tensors;     // 0: input; k + 1: output of layer k
    std::vector<DenseLayer> layers;       // in topological order
    std::string outputActivation;   // "softmax", "sigmoid" or "none"
    bool singleLogitPadded = false; // a 1-logit head got an implicit 0 logit
    std::size_t foldedBatchNorms = 0;
    std::size_t loweredPools = 0;   // average pools lowered to depthwise convolutions

    // Neurons of a tensor that share one scale: all positions of a channel
    // when a convolution-like layer reads the tensor (it shares weights across
    // positions), otherwise each neuron on its own.
    std::size_t scaleGroupOf(std::size_t tensor, std::size_t i) const;
    std::size_t inputGroupOf(std::size_t i) const { return scaleGroupOf(0, i); }
    std::size_t parameterCount() const;
    // Last layer that reads each tensor (layers.size() for the logits).
    std::vector<std::size_t> lastUse() const;
};

std::optional<DenseNetwork> extractDenseNetwork(const IRGraph& graph, std::string& error);
std::vector<double> referenceLogits(const DenseNetwork& network, const std::vector<double>& x);
// Every tensor of the float model (index as DenseNetwork::tensors).
std::vector<std::vector<double>> referenceActivations(const DenseNetwork& network, const std::vector<double>& x);
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
// activationBits[l] (l = 1..L-1) the output of hidden layer l-1. A max-pool
// keeps the codes of its input: its slots are ignored (see fixedSlot). The logits
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

// True if a configuration slot has no effect (kind 0: activation bits of
// tensor `index`, kind 1: weight bits of layer `index`).
bool fixedSlot(const DenseNetwork& network, int kind, std::size_t index);


struct QLayer {
    LayerKind kind = LayerKind::Dense;
    ConvShape conv;
    std::vector<std::size_t> sources{0};
    TensorShape output;
    std::size_t weightsPerChannel = 0;
    std::size_t in = 0;
    std::size_t out = 0;
    std::size_t channels = 0;
    bool relu = false;
    int weightBits = 8;
    // Weights are quantised after folding the input scales in: the integer
    // weight approximates W[w] * inScale(input) / weightScale[channel], so
    // per-channel input scales cost nothing at run time.
    std::vector<std::int32_t> qWeight;  // same layout as DenseLayer::weight
    std::vector<double> weightScale;    // per output channel
    std::vector<std::int64_t> qBias;    // per output channel, at scale weightScale
    std::vector<double> inScale;        // per input neuron
    std::vector<double> outScale;       // per output neuron (shared within a channel group)
    int outBits = 8;
    bool outUnsigned = true;
    std::int64_t outMin = 0;
    std::int64_t outMax = 255;
    std::vector<std::int32_t> multiplier;  // per channel: q = round(acc * m / 2^shift)
    std::vector<int> shift;
    int accBits = 32;                      // 16, 32 or 64
    bool modular = false;                  // wrap-around accumulation, final sum proved to fit
    int mulBits = 32;                      // 16: every product provably fits int16
    std::vector<std::int64_t> prefixBound; // classical bound on every partial sum
    std::vector<std::int64_t> finalBound;  // proved bound on the final sum
    std::vector<std::int64_t> qOutLo;      // proved range of the quantised output
    std::vector<std::int64_t> qOutHi;
    std::vector<std::vector<Tap>> taps;
    int buffer = -1;                       // static buffer of the output (-1: the logits)

    std::size_t channelOf(std::size_t o) const { return output.channelOf(o); }
};

struct QProgram {
    std::string name;
    std::size_t inputs = 0;
    std::size_t classes = 0;
    std::vector<std::size_t> inputShape;
    int inputBits = 8;
    bool inputUnsigned = false;      // non-negative domain: codes 0 .. 2^bits - 1
    std::vector<double> inputScale;  // per input feature
    std::int64_t inputMin = -127;
    std::int64_t inputMax = 127;
    Box domain;  // the input domain every global claim is about
    std::vector<QLayer> layers;
    std::vector<TensorShape> tensors;
    std::vector<std::size_t> bufferSize;  // elements of each static buffer
    PrecisionConfig config;
    bool calibrated = false;          // scales from data, not from proved ranges
    bool saturationPossible = false;  // some clamp can trigger on the domain (modelled)
};

// Assigns every hidden tensor to a static buffer: tensors of the same C type
// whose live ranges do not overlap share a buffer, and no layer writes a
// buffer it reads. Fills QLayer::buffer and QProgram::bufferSize.
void planBuffers(QProgram& program);

// Error radius of input quantisation in units of the input scale. The C code
// computes x * (1/s) in float32; this covers that rounding as well.
double inputQuantisationSlack(std::int64_t inputMax);

// Exactly what the emitted C does (float32 input scaling, round half away
// from zero, saturation).
std::vector<std::int64_t> quantizeInput(const QProgram& program, const std::vector<double>& x);

struct IntegerRun {
    std::vector<std::int64_t> logits;
    int decision = -1;
    bool accumulatorOverflow = false;  // the C accumulator would not hold the sum
    bool productOverflow = false;      // a product left the multiply type
    bool saturated = false;            // a requantised value hit the clamp
};

// Bit-exact integer semantics of the emitted C program (in the absence of
// overflow, which is flagged; modular accumulation is exact iff the final
// sum fits).
IntegerRun runInteger(const QProgram& program, const std::vector<std::int64_t>& qInput);
IntegerRun runInteger(const QProgram& program, const std::vector<double>& x);
std::vector<double> dequantizedLogits(const QProgram& program, const IntegerRun& run);

// Rounding right shift used everywhere: floor((t + 2^(s-1)) / 2^s).
std::int64_t roundShift(std::int64_t value, int shift);

// Storage of the emitted arrays (flash) and activation buffers (RAM).
struct ProgramCost {
    double weightBytes = 0;  // flash for weights only
    double flashBytes = 0;   // weights + biases + multipliers + shifts + input scales
    double ramBytes = 0;     // static activation buffers (after buffer sharing)
    double idealBits = 0;    // sum of weightBits * |W| (sub-byte packing)
};
ProgramCost programCost(const QProgram& program);

// Estimated cycles per inference on an 8-bit AVR (ATmega2560), from per-MAC
// costs measured with simavr for every combination of weight storage
// (4-bit packed / 8 / 16), activation width (8 / 16), multiply width (16 /
// 32) and accumulator width (16 / 32 / 64), plus requantisation per neuron.
double estimatedAvrCycles(const QProgram& program);
double float32Bytes(const DenseNetwork& network);  // weights + biases in float32
std::size_t multiplyAccumulates(const DenseNetwork& network);

// Emits model_q.h, model_q.c (integer-only C99), main_q.c and CMakeLists.txt.
// The data arrays in model_q.c are what the independent checker reads.
bool emitIntegerC(const QProgram& program, const std::string& directory, std::string& error);

// Emits model_f.h / model_f.c: the float32 reference as plain C99 (the
// baseline a microcontroller without FPU runs with soft-float).
bool emitFloatC(const DenseNetwork& network, const std::string& directory, std::string& error);

}  // namespace modelforge

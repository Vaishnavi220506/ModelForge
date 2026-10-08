#pragma once

// Shrink-and-check: compress a model (float16, int8, int4, pruning) and hunt
// for realistic inputs where the compressed model disagrees with the original.
//
// Our method, decision-boundary shift probing, walks the segment between two
// real test points that the original classifies differently, locates the
// original's decision boundary by bisection, then checks whether the
// compressed model's boundary moved. If it moved, the gap between the two
// boundaries is a disagreement region; the point at the far edge of the gap is
// where the original is most confident while the compressed model disagrees.

#include "ir.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace modelforge {

struct Dataset {
    std::vector<std::vector<float>> rows;
    std::vector<int> labels;  // -1 when the file has no label column
};

// CSV with one row per sample: features..., label (label column optional
// when `expectedFeatures` equals the column count).
bool loadDataset(const std::string& path, std::size_t expectedFeatures, Dataset& dataset,
                 std::string& error);

enum class CompressionKind { Float16, IntQuantization, Pruning };

struct Compression {
    std::string name;  // fp16, int8, int4, prune30, prune50
    CompressionKind kind = CompressionKind::Float16;
    int bits = 8;
    float pruneFraction = 0.0f;
};

std::vector<Compression> standardCompressions();
bool parseCompression(const std::string& name, Compression& compression);

// Weight matrices are quantised per tensor (symmetric) or pruned by magnitude;
// float16 rounds every constant. Biases stay float32 for int/prune, as is usual.
IRGraph compressModel(const IRGraph& graph, const Compression& compression);

// Serialises an IR graph (unoptimised operators only) back to .mforge text.
std::string writeManifestText(const IRGraph& graph);

struct DisagreementSearch {
    std::string strategy;
    std::size_t budget = 0;            // compressed-model evaluations allowed
    std::size_t candidateEvaluations = 0;
    std::size_t originalEvaluations = 0;
    std::size_t firstDisagreement = 0;  // 1-based candidate evaluation, 0 = none
    std::size_t disagreements = 0;       // realistic disagreements only
    // Original model's margin between its top two classes at the worst
    // realistic disagreement: 0 = it was a near-tie, 1 = it was certain.
    float worstSeverity = 0.0f;
    std::vector<float> worstInput;
    int originalClass = -1;
    int compressedClass = -1;
    float worstDistance = 0.0f;         // RMS z-score distance to the nearest test row
    float maxBoundaryShift = 0.0f;      // boundary-shift probing only (RMS z units)
};

// Standard practice: run the compressed model on the test set.
DisagreementSearch searchTestSet(const IRGraph& original, const IRGraph& compressed,
                                 const Dataset& test);
// Gaussian noise (sigma in z units) around test rows.
DisagreementSearch searchNoise(const IRGraph& original, const IRGraph& compressed,
                               const Dataset& test, std::size_t budget, std::uint32_t seed,
                               float sigma = 0.25f);
// DiffChaser-style genetic search seeded from test rows.
DisagreementSearch searchGenetic(const IRGraph& original, const IRGraph& compressed,
                                 const Dataset& test, std::size_t budget, std::uint32_t seed);
// Ours: decision-boundary shift probing between real test points.
DisagreementSearch searchBoundaryShift(const IRGraph& original, const IRGraph& compressed,
                                       const Dataset& test, std::size_t budget);

// Fraction of labelled rows classified correctly (NaN without labels).
double accuracy(const IRGraph& graph, const Dataset& data);

// A disagreement counts only within 0.5 RMS z-units of a real test row.
// Plain-language risk level from the worst severity (margin).
std::string riskLevel(float worstSeverity, std::size_t classes);

}  // namespace modelforge

#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <vector>

namespace modelforge {

using Shape = std::vector<std::int64_t>;

enum class DataType {
    Float32,
    Unknown
};

std::string dataTypeName(DataType type);
std::size_t elementCount(const Shape& shape);

struct TensorInfo {
    std::string name;
    DataType dataType = DataType::Unknown;
    Shape shape;
    bool isConstant = false;
    bool isInput = false;
    bool hasProducer = false;
    std::string producer;
    std::vector<float> data;
};

struct Node {
    std::string name;
    std::string op;
    std::vector<std::string> inputs;
    std::string output;
    int axis = -1;
    bool transA = false;
    bool transB = false;
    float alpha = 1.0f;
    float beta = 1.0f;
    int stride = 1;  // Conv, pooling
    int pad = 0;     // Conv, pooling: zero padding on every side
    int group = 1;   // Conv: 1 = dense convolution, C = depthwise
    int kernel = 0;  // AveragePool / MaxPool window (square)
    float epsilon = 1e-5f;  // BatchNormalization
    int sourceLine = 0;
};

struct ModelGraph {
    std::string name;
    std::string inputName;
    std::string outputName;
    std::unordered_map<std::string, TensorInfo> tensors;
    std::vector<Node> nodes;
};

// Prints the parsed manifest as an AST-like model syntax tree for demonstrations.
void printSyntaxTree(const ModelGraph& model, std::ostream& output);

}  // namespace modelforge

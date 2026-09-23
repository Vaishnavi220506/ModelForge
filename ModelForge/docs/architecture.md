# ModelForge Architecture

```text
ONNX Model
    |
    v
ONNX Loader
    |
    v
Operator and Tensor Extraction
    |
    v
Semantic Validator
    |
    v
ModelForge IR
    |
    v
Optimization Passes
    |
    v
C++ Code Generator
    |
    v
Generated C++ Source
    |
    v
Native Compilation and Inference
    |
    v
Output Equivalence Validation
```

The optional visualization module reads the IR and optimization report. It supports explanation but is not part of the core compilation path.

## Module responsibilities

1. **ONNX Loader** - reads model metadata, graph nodes, initializers, attributes, tensor types, and shapes.
2. **Semantic Validator** - rejects unsupported operators, incompatible dimensions, invalid types, missing tensors, duplicate names, and invalid attributes.
3. **ModelForge IR** - represents inputs, constants, dense operations, matrix multiplication, element-wise operations, activations, Softmax, and return values independent of ONNX.
4. **Optimizer** - applies constant folding, redundant-operation elimination, identity removal, and optional Dense plus ReLU fusion.
5. **C++ Code Generator** - emits headers, weights, buffers, activation helpers, and an inference function.
6. **Validation Engine** - compares original ONNX Runtime output and generated native output within a documented tolerance.

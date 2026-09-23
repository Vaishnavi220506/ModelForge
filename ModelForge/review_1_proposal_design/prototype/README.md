# Phase 1 Prototype

The prototype is a dependency-free C++17 executable that demonstrates the first compiler stages:

1. Read a small model manifest.
2. Identify model input, output, and nodes.
3. Check the supported-operator set.
4. Check tensor production order and duplicate names.
5. Print a simplified ModelForge IR.

The manifest is a temporary Phase 1 de-risking format. The Phase 2 loader will replace it with ONNX protobuf loading while preserving the same validator and IR boundary.

## Build and run

```text
cmake -S . -B build
cmake --build build
build\\modelforge_prototype.exe sample_model.mforge
```

The expected result is a successful operator check followed by the IR sequence `INPUT`, `Gemm`, `Relu`, `Gemm`, `Softmax`, and `RETURN`.

# Review 2 Core Implementation

## Research enhancement

Guardian checks every optimization pass with deterministic probes, rejects changes that alter outputs, and writes a simplified counterexample for rejected passes. The verifier also compares the original and final IR. Each compilation writes optimization_certificate.json with pass decisions, probe values, instruction reduction, maximum error, and prediction agreement. Use `--guardian-demo-bug` with iris_demo.mforge to show a deliberately wrong ReLU rewrite being rejected.

This directory contains the five-module C++17 implementation of ModelForge.

## Five modules

1. **Model loader/front end** - reads the dependency-free `.mforge` manifest and exposes an optional ONNX loader behind the same `ModelGraph` interface.
2. **Semantic analysis and symbol table** - tracks tensor names, shapes, types, producers, and constants, then rejects invalid graphs.
3. **ModelForge IR** - lowers validated nodes into a source-format-independent intermediate representation and prints it.
4. **Optimizer** - performs constant folding, dense-plus-ReLU fusion, and dead-node removal.
5. **Code generation, execution, and verification** - emits standalone `model.h`/`model.cpp`, executes the IR for a local check, and provides output comparison helpers.

Error diagnostics are shared by every module. The input graph is intentionally limited to float32, rank-2 feed-forward models using `Gemm`, `MatMul`, `Add`, `Relu`, `Sigmoid`, and `Softmax`; Softmax must operate along the final axis.

## Build with Visual Studio CMake

Open either this directory or the parent `Compiler` directory as the workspace root in VS Code with the **CMake Tools**, **C/C++**, and **CMake language support** extensions. The parent workspace includes a CMake wrapper and tasks that point to this implementation automatically. Use the Command Palette to run `Tasks: Run Task`, then choose `ModelForge: test` or `ModelForge: run demo`.

The same tasks can be run from a VS Code terminal:

```text
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Run the demonstration from this directory:

```text
build\Release\modelforge.exe models\iris_demo.mforge --out generated
```

The command prints the five compilation stages, IR before and after optimization, optimization statistics, and an IR execution prediction. It writes the compilable generated files to `generated/model.h`, `generated/model.cpp`, and `generated/CMakeLists.txt`.

To build the generated program separately:

```text
cmake -S generated -B generated\build
cmake --build generated\build --config Release
generated\build\Release\generated_model.exe
```

## Optional ONNX support

The default build has no external dependency and is intended for learning and fast demonstrations. After installing ONNX C++ headers/library and Protobuf, configure with:

```text
cmake -S . -B build-onnx -DMODELFORGE_WITH_ONNX=ON
```

Then pass a `.onnx` file instead of the manifest. The validator, IR, optimizer, and code generator are unchanged.

To compare the ModelForge execution result with ONNX Runtime for the same zero-valued input, also provide the ONNX Runtime headers and library:

```text
cmake -S . -B build-onnx -DMODELFORGE_WITH_ONNX=ON -DMODELFORGE_WITH_ONNX_RUNTIME=ON
build\Release\modelforge.exe path\to\your_model.onnx --verify-onnx
```

In Visual Studio, install the **Desktop development with C++** workload, CMake tools, ONNX/Protobuf development files, and ONNX Runtime. Set the include/library paths used by the two `find_path`/`find_library` checks if they are not installed in a standard location.

## Review 2 evidence

- Working loader and model graph
- Symbol table and semantic diagnostics
- IR before and after optimization
- Optimization statistics
- Generated standalone C++ source
- Built-in execution check and output comparison helper
- Valid, invalid, and boundary test fixtures
- Module-wise explanation in `src/README.md`

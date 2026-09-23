# Development Log

## 11 September 2026

- Reviewed the ModelForge project brief.
- Reviewed the Compiler Design Laboratory instruction manual.
- Froze the initial scope around small feed-forward ONNX models.
- Created separate folders for Reviews 1, 2, and 3.
- Added a dependency-free C++17 prototype for operator validation and IR reporting.
- Prepared the Review 1 project proposal and design document.

## 17 September 2026

- Implemented the Review 2 five-module C++17 core pipeline.
- Added manifest loading, a shared model graph, semantic validation, and a tensor symbol table.
- Added ModelForge IR construction and readable IR printing.
- Added constant folding, dense-plus-ReLU fusion, and dead-node removal.
- Added standalone C++ code generation and a dependency-free IR execution/checking engine.
- Added valid and invalid fixtures plus a CTest-compatible test runner.
- Added an optional ONNX/Protobuf loader path guarded by `MODELFORGE_WITH_ONNX`.

## Next planned work

- Install/configure the Visual Studio C++ and CMake toolchain in the development environment.
- Build and run the Review 2 pipeline and generated `model.cpp`.
- Connect the optional ONNX loader to installed ONNX/Protobuf libraries.
- Add ONNX Runtime equivalence tests and final integration evidence.

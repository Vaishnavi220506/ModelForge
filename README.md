# ModelForge — VS Code workspace

This workspace contains the ModelForge C++17 compiler under
`ModelForge/review_2_core_implementation`. It loads a `.mforge` model, validates
tensor symbols, prints the syntax tree and IR, optimizes the graph, and emits
standalone C++ inference code. The optional ONNX loader needs separate ONNX and
Protobuf development libraries; the default demo does not.

## Build from the folder opened in VS Code

Open the integrated PowerShell terminal at:

```text
C:\Users\Sri vaishnavi\Compiler
```

Run these commands:

```powershell
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
.\build\Release\modelforge.exe .\ModelForge\review_2_core_implementation\models\iris_demo.mforge --out .\build\generated
```

Add `--dump-cpp` to print the generated `model.cpp` source in the terminal as
proof during the demonstration:

```powershell
.\build\Release\modelforge.exe .\ModelForge\review_2_core_implementation\models\iris_demo.mforge --out .\build\generated --dump-cpp --guardian-demo-bug
```

PowerShell needs `.\` before a local executable. The root `CMakeLists.txt`
forwards the build to the Review 2 implementation.

The Guardian demo deliberately changes ReLU to Sigmoid in a temporary graph.
Guardian rejects that candidate and writes `build/generated/counterexample.json`
with the input and both outputs. Real optimizations are checked separately;
`build/generated/optimization_certificate.json` lists pass decisions and probes.
The deliberately wrong graph is never emitted as C++.

Compile and run the generated C++ program:

```powershell
cmake -S .\build\generated -B .\build\generated_build -G "Visual Studio 17 2022"
cmake --build .\build\generated_build --config Release
.\build\generated_build\Release\generated_model.exe
```

For another input, replace `iris_demo.mforge` with `binary_sigmoid_demo.mforge`.
See [the research note](ModelForge/docs/research_contribution.md) for the
Guardian experiment and its limits.

The small Guardian fault-injection comparison prints a CSV table:

```powershell
.\build\Release\modelforge_guardian_eval.exe .\ModelForge\review_2_core_implementation\models\iris_demo.mforge
```

## VS Code tasks

Open **Terminal → Run Task** and choose:

- `ModelForge: Configure`
- `ModelForge: Build Release`
- `ModelForge: Run Tests`
- `ModelForge: Run Iris Demo`

VS Code is the editor. A C++ compiler/toolchain must also be installed on the
laptop. CMake alone cannot compile the project.

If Windows Application Control blocks a built `.exe`, the build succeeded but
that laptop needs administrator approval to run it. GitHub Actions builds and
runs the project independently on Windows and Linux. Each successful run also
attaches the generated C++ source, certificate, and counterexample as a
downloadable demo artifact.

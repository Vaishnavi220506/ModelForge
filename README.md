# ModelForge — VS Code workspace

ModelForge is a C++17 compiler for small neural-network models with a built-in
per-pass validator, **Guardian-APC**. Before any graph rewrite is kept, Guardian
tests it with probes placed on the model's own ReLU boundaries at every depth,
ordered by activation-pattern coverage. A wrong rewrite is rolled back and
saved as a replayable counterexample.

On 96 synthetic models with 2,160 injected faults, Guardian-APC detects 99.1%
of observable faults with 128 probes. Uniform random testing detects 75.1% and
the previous Guardian 85.7% (paired McNemar p < 1e-4). See
[Guardian-APC](ModelForge/docs/guardian_apc.md) and the
[paper draft](ModelForge/paper/guardian_apc.tex).

## Quick demo (terminal + dashboard)

After building (below), run in the VS Code terminal:

```powershell
.\build\Release\modelforge_studio.exe          # interactive terminal dashboard
.\build\Release\modelforge_studio.exe --demo   # every screen, non-interactive
.\build\Release\modelforge_bench.exe           # full study, refreshes dashboard/data.js
```

On Linux or macOS, use `./build/Release/modelforge_studio` and so on.

The studio menu has the compilation pipeline, IR graph before and after
optimisation, IBP neuron stability map, Guardian decisions with coverage, a
fault-injection arena with a counterexample, the benchmark, and the generated
C++. Add `--ascii` if the terminal font lacks box-drawing characters.

Open `dashboard/index.html` in a browser (double-click it, or run the task
**ModelForge: open web dashboard**). No server is needed. It has five tabs:
Overview, Fault families, Statistics, Model inspector, and Corpus. The data
shipped in the repository comes from a Linux run; re-run `modelforge_bench` to
regenerate it on your machine.

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
.\build\generated_build\Release\generated_model.exe 1 2 3 4
.\build\generated_build\Release\generated_model.exe --csv .\ModelForge\review_2_core_implementation\models\iris_inputs.csv
```

The first run uses four zeros for the original demo. The second runs a real
four-feature input, prints all output scores, and predicts class 1. The
generated program rejects a wrong number of features or non-numeric input;
its `infer()` function in `model.h` can also be called from another C++ app.
The `--csv` form runs every numeric row in a file and prints a machine-readable
CSV with row number, predicted class, and each output score. The sample file
has two four-feature rows. Blank lines are ignored; a malformed row stops
processing with its line number. You can save the output with PowerShell:

```powershell
.\build\generated_build\Release\generated_model.exe --csv .\ModelForge\review_2_core_implementation\models\iris_inputs.csv > .\build\predictions.csv
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
- `ModelForge: studio (terminal dashboard)`
- `ModelForge: studio demo (all screens)`
- `ModelForge: research benchmark (96 models)`
- `ModelForge: open web dashboard`

VS Code is the editor. A C++ compiler/toolchain must also be installed on the
laptop. CMake alone cannot compile the project.

If Windows Application Control blocks a built `.exe`, the build succeeded but
that laptop needs administrator approval to run it. GitHub Actions builds and
runs the project independently on Windows and Linux. Each successful run also
attaches the generated C++ source, certificate, and counterexample as a
downloadable demo artifact.

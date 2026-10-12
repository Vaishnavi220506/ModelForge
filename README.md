# ModelForge — VS Code workspace

**Author: Vaishnavi**

ModelForge is a C++17 compiler for small neural networks. Its main research
contribution is **ForgeCert: proof-carrying integer code**. ModelForge compiles
a trained float model to **integer-only C99** for microcontrollers without an
FPU, and ships a **certificate** that the emitted C program

1. never overflows an accumulator or saturates a value anywhere in the input domain, and
2. returns exactly the float model's decision for *every* input in each certified
   region (an L∞ ball around a real sample), not just on the test set.

A counterexample-guided tuner chooses the bit width of every layer, using error
provenance carried by the abstract domain to decide which layer to make more
precise. A separate checker, `modelforge_certcheck`, shares no code with the
compiler and re-validates the certificate from the source model and the emitted
`.c` file alone.

**Results** (8 trained classifiers, 24 model/radius cases, 1,441 regions on
which the float model is provably constant; `modelforge_certbench`):

| Method | Cases fully certified | Regions proved | Regions with a real counterexample | Size vs float32 |
| --- | --- | --- | --- | --- |
| Uniform int8 | 4 / 24 | 1,136 | 1 | 36.5% |
| Uniform int4 | 2 / 24 | 964 | 149 | 24.3% |
| Uniform int16 | 23 / 24 | 1,440 | 0 | 63.0% |
| Accuracy-guided (common practice) | 9 / 24 | 1,345 | 4 (all missed by the test set) | 35.6% |
| **ForgeCert** | **23 / 24** | **1,440** | **0** | **39.7%** |

* ForgeCert matches the exhaustive optimum in 7 of 8 models (mean gap 0.3%).
* 3,000,000 sampled executions: 0 wrong decisions, 0 overflows, 0 error-bound
  violations.
* The independent checker accepts all 24 certificates in under 0.1 s each.
* The compiled C binaries match the checker's semantics bit for bit on every
  test row.
* The one unproved region cannot be proved even at 16 bits, and no
  counterexample exists for it among the inputs we searched.

```powershell
.\build\Release\modelforge.exe .\ModelForge\review_2_core_implementation\models\real\wine_mlp32x16.mforge --certify --data .\ModelForge\review_2_core_implementation\models\real\wine_test.csv --domain .\ModelForge\review_2_core_implementation\models\real\wine_train.csv --eps 0.1 --out .\build\cert
.\build\Release\modelforge_certcheck.exe .\build\cert .\ModelForge\review_2_core_implementation\models\real\wine_mlp32x16.mforge
.\build\Release\modelforge_certbench.exe --out .\build\certbench    # all paper tables
```

See [ForgeCert](ModelForge/docs/forgecert.md) and the
[paper draft](ModelForge/paper/forgecert.tex).

### Earlier contributions (now stages of the pipeline)

Every graph rewrite is validated by **Guardian-APC+** before it is kept. Guardian
tests the rewrite with inputs placed on the model's own ReLU boundaries at
every depth, focused on the layers the rewrite changed, ordered by
activation-pattern coverage, and followed by a near-miss search. A wrong
rewrite is rolled back with a replayable counterexample and the layer where
the bug first shows.

On 96 synthetic models with 2,160 injected faults, Guardian-APC+ detects 99.2%
of observable faults with 128 inputs. Uniform random testing detects 75.0% and
the previous Guardian 85.6% (paired McNemar p < 1e-4). With only 8 inputs it
already catches 88.2%.

**Shrink and check:** is a compressed (float16, int8, int4, pruned) model
safe to ship? It finds realistic inputs where the original model was sure but
the compressed one disagrees, and recommends the smallest safe option:

```powershell
.\build\Release\modelforge.exe .\ModelForge\review_2_core_implementation\models\real\wine_mlp16.mforge --shrink all --data .\ModelForge\review_2_core_implementation\models\real\wine_test.csv --out .\build\shrink
```

On 8 models trained on real data, it caught 22 of 24 risky compressions; the
usual test-set check caught 14.

**Model diff checker:** check that an edited or hand-optimised model still
behaves like the original:

```powershell
.\build\Release\modelforge.exe .\ModelForge\review_2_core_implementation\models\iris_demo.mforge --compare .\ModelForge\review_2_core_implementation\models\iris_handopt_bug.mforge
``` See
[Guardian-APC](ModelForge/docs/guardian_apc.md) and the
[paper draft](ModelForge/paper/guardian_apc.tex).

## Demo for faculty (one command)

`samples/` has a ready-made model and a script that compiles it without and
with optimisation, shows the generated C++ side by side, builds and runs both
programs on the same inputs (identical answers), shows Guardian rejecting a
broken rewrite, and checks compressed versions of a real model. See
[samples/README.md](samples/README.md).

```powershell
.\samples\run_demo.ps1
```

## Interactive website

Open `website/index.html` in a browser (no server or install needed). It runs
the real trained models in the page:

- **Compiler**: paste, upload or pick a `.mforge` model and get the same C++
  the real compiler writes, with a before/after view of the optimised code, an
  inspector (internal graph, Guardian decisions, neuron analysis, symbol
  table), a run panel and an effectiveness table. Plant a bug in the optimiser
  to watch Guardian reject it, or switch Guardian off to see the bug reach the C++.
- **Shrink lab**: pick a model, compress it (float16, int8, int4, pruning),
  run the Guardian check live and compare all levels with a recommendation.
- **Boundary explorer**: a 2D slice through the input space showing where the
  original and the smaller model disagree; hover to read both models.
- **Try an input**: draw a digit or move feature sliders, then ask Guardian to
  find the closest input where the smaller model changes its answer.

To publish it, enable GitHub Pages (Settings, Pages, Source: GitHub Actions);
the `Publish ModelForge website` workflow deploys it on every push to `main`
at `https://vaishnavi220506.github.io/ModelForge/`.

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

# Core Source Modules

The implementation is deliberately split so that a student can explain one compiler phase at a time.

## Data flow

```text
loadModel -> validate -> buildIR -> optimize -> generateCpp -> executeIR
```

`ModelGraph` is the source-level graph. `SymbolTable` is the semantic-analysis table. `IRGraph` is the source-independent middle-end representation. The optimizer changes only `IRGraph`; the code generator consumes only `IRGraph`, which keeps ONNX-specific details out of the back end.

## How to read the code

- Start with `main.cpp` to see the pipeline order.
- Read `model.h` to learn the shared data structures.
- Read `loader.cpp` to see line tokenization and manifest parsing.
- Read `validator.cpp` to see symbol insertion, availability checks, and shape inference.
- Read `ir.cpp` to see lowering and IR printing.
- Read `optimizer.cpp` to see constant folding, fusion, and liveness-based dead-node removal.
- Read `codegen.cpp` to see C++ source emission.
- Read `verifier.cpp` to see the small execution engine and numerical comparison helper.

The manifest format is:

```text
model <name>
input <name> float32 <shape>
output <name> float32 <shape>
tensor <name> float32 <shape> values=<comma-separated-values>
node <name> <operator> <inputs...> -> <output> [axis=<number>] [transB=1]
```

Example:

```text
node dense Gemm x weights bias -> hidden
node activation Relu hidden -> activated
```

For a common ONNX-exported `Gemm`, `transB=1` preserves the original weight layout.

ONNX-specific code is isolated in `loadOnnx`; it is compiled only when `MODELFORGE_WITH_ONNX` is enabled.

## Guardian-APC research modules

- `analysis.h/.cpp` - interval bound propagation, deep boundary probes (Newton on the exact input gradient), activation-pattern coverage and greedy coverage ordering.
- `research.h/.cpp` - synthetic model zoo, fault catalogue, equal-budget strategies, benchmark statistics (Wilson intervals, exact McNemar) and JSON export for the dashboard.
- `../tools/studio.cpp` - `modelforge_studio`, the interactive terminal dashboard.
- `../tools/bench.cpp` - `modelforge_bench`, the research benchmark.

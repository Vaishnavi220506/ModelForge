# ModelForge Guardian: a research direction

## Problem and idea

Model compilers change computation graphs to make inference cheaper. A faulty graph rewrite can silently change a prediction. Guardian checks each rewrite immediately after it is applied, tries inputs selected for the graph's operators, and rolls back a rewrite when it finds a mismatch. It saves the input and both outputs so the failure can be replayed.

This is an **empirical testing method**. A pass means the generated probes did not find a mismatch; it is not a proof for all floating-point inputs. The certificate is an audit record, not a formal correctness certificate.

## Implemented method

1. Start from the validated ModelForge IR.
2. Apply one of constant folding, GEMM plus ReLU fusion, or dead-node removal to a copy.
3. Test the original and candidate graphs on deterministic probes: zero, ones, activation boundary values, positive and negative extremes, sparse coordinate values, and seeded random values.
4. Compare every output value within tolerance and compare the predicted class.
5. Accept the pass or restore the preceding graph. On failure, simplify the input one coordinate at a time while preserving the mismatch.
6. Record pass decisions in `optimization_certificate.json` and failures in `counterexample_<pass>.json`. `--guardian-demo-bug` demonstrates rejection with a deliberately incorrect ReLU rewrite in a temporary graph and writes `counterexample.json`.

The normal generated C++ always comes from the accepted graph. The demo bug is never emitted as model code.

## Research question

Can operator-directed probes detect incorrect tensor-graph rewrites with fewer executions than uniform random probes, while keeping compilation time practical for small feed-forward models?

## Evaluation needed for a paper

- Run a set of small feed-forward graphs with different shapes, weights, and operator sequences.
- Inject realistic faults: missing ReLU, wrong GEMM transpose, omitted bias, wrong constant fold, and broken Softmax.
- Compare equal-budget zero-only, random-only, and Guardian probe suites. Report fault detection rate and first-detection probe count.
- Measure compilation time, generated C++ runtime, instruction reduction, and code size on the same machine.
- Report failures that none of the probe suites detect. Include all model inputs, seeds, compiler version, and machine details.

The repository has the compiler, rejection demo, tests, and CI workflow. It does **not** yet contain a full multi-model experimental data set or results that justify a publication claim.

## Limits and prior work

The verifier and code generator share some operator definitions; a common semantic bug could escape this comparison. The optional real ONNX path has not been validated in the default build. Testing a finite input set cannot establish universal equivalence. For stronger confidence, compare against ONNX Runtime and use symbolic or SMT checking for a restricted subset.

Graph optimization is well established in [TVM](https://arxiv.org/abs/1802.04799) and [Glow](https://arxiv.org/abs/1805.00907). [MLIR translation validation](https://link.springer.com/chapter/10.1007/978-3-031-13188-2_19) studies formal correctness, and [Propilot](https://arxiv.org/abs/2606.06747) studies property-based AI compiler tests. Guardian is a compact, inspectable experimental system for testing per-pass failures in this project's supported operator subset. It should not be described as the first compiler validation technique.

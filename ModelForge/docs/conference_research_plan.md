# From ModelForge demo to a conference submission

## The claim to test

**Hypothesis:** for a fixed testing budget, probes constructed from *the model's actual activation and decision boundaries* expose localized graph-rewrite errors earlier than probes drawn uniformly at random. A failed experiment is informative too: if random testing performs equally well, do not claim superiority.

The mechanism is simple. For a dense neuron, the ReLU changes behavior when `sum(x[k] * W[k,j]) + b[j] = 0`. Holding other coordinates at zero lets us solve for one input coordinate. ModelForge now probes both sides and the boundary itself. For classification, it also bisects between inputs whose predicted classes differ. The optimization Guardian runs these probes before accepting each pass, rolls back a detected bad pass, and saves a replayable witness.

## What is implemented now

- Deterministic probes derived from first-layer `Gemm -> Relu` when weights are constant and the model has one rank-2 input row. The implementation handles transposed B weights and limits generation to 32 neurons.
- Decision-boundary bisection for small inputs when existing probes find different classes.
- A controlled, test-only near-boundary ReLU fault, plus five earlier fault/negative-control cases. It is excluded from the input loader and C++ emitter.
- A CSV evaluation with equal probe counts for random and Guardian, first detection indices, and a generic-probe ablation. CI preserves the CSV with generated code and counterexamples.

These are *finite tests*, not proof of equivalence. Boundary probes are currently limited to the first dense layer. They are generated after generic probes, so first-detection latency may be worse than random even when a boundary probe eventually finds a fault. That ordering is a testable design choice, not a result to hide.

## Experiment required before submitting

1. Freeze the implementation, compiler version, tolerance, and hypotheses before examining aggregate results.
2. Build an open corpus of at least several dozen feed-forward models with varied widths, depths, weight scales, biases, and seeds; keep held-out models for final evaluation. Add real ONNX models only after the optional loader/runtime path is operational and independently checked.
3. Inject faults at each supported operator and depth. Include localized numerical faults, broad semantic faults, and output-unreachable negative controls. Verify each fault is observable on at least one independent high-budget search, otherwise label it *unknown*, not a Guardian miss.
4. Compare equal-budget suites: zero/coordinate, uniform random over explicitly specified ranges, generic operator probes without model conditioning, and the full boundary-aware suite. Run many random seeds and report paired results.
5. Measure fault detection rate, probes to first detection, false positives on known-equivalent rewrites, probe-generation time, total compilation overhead, and saved-counterexample replay success. Include 95% confidence intervals and effect sizes; disclose every excluded case.
6. Validate generated C++ against an independent reference such as ONNX Runtime on supported ONNX models. The current verifier and emitter share operator assumptions, so agreement between them alone cannot rule out common-mode errors.
7. Publish model manifests, fault definitions, seeds, CSVs, analysis script, and hardware/toolchain details. Run both Windows and Linux builds.

## Novelty boundary and related work

Do **not** claim to be the first to validate or fuzz ML compilers. [TensorRight](https://doi.org/10.1145/3704865) formally verifies tensor graph rewrites; [SMT-based translation validation for ML compilers](https://link.springer.com/chapter/10.1007/978-3-031-13188-2_19) studies correctness with solver methods; [NNSmith](https://arxiv.org/abs/2207.13066) searches for compiler bugs using generated networks. The possible contribution here is a lightweight *model-specific boundary targeting plus per-pass rollback and replay* system, if a rigorous equal-budget study establishes an advantage. It may be best positioned as a systems/tool demonstration or a focused empirical paper until that evidence exists.

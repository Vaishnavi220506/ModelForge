# Temper: the pitch, the idea and how to show it

Author: **Vaishnavi**

## The pitch (30 seconds)

> Every AI model that runs on a phone or a microcontroller has been *transformed*
> on the way there: compiled, optimised, quantised, pruned, turned into C++ and
> compiled again. Each step can silently change what the model predicts, and
> today nobody checks the whole chain. Tests can only say "no bug found";
> verifiers can only say "unknown" when there is one.
>
> **Temper** gives every step a definite answer: *here is an input that breaks
> it*, or *here is a proof that it cannot break, bit for bit, on your data's
> domain*. It finds bugs that random testing and our own strong falsifier both
> miss, it tells you which compression level is provably safe, and it checks
> the final binary. One terminal command, a JSON report, and a CI exit code.

ModelForge, the compiler, is one component: forging makes the program,
tempering makes it trustworthy.

## What is new (the research contributions)

1. **Falsify-or-certify for neural-network transformations.** One decision
   procedure (static proof, then Guardian-APC+, then certificate-guided branch
   and bound) gives REFUTED / CERTIFIED / UNDECIDED for compiler passes, model
   edits, compression, domain specialisation and the binary.
2. **A float32-sound, exactness-tracking differential certifier.** Most network
   verifiers reason in real numbers. Temper pads every affine bound with the
   dot-product rounding bound, so "this neuron is always off" holds for the
   executed float program. An *exactness bit* per value proves outputs are
   **bit-identical** after structural rewrites (fusion, dead-code removal,
   constant folding), with zero model evaluations.
3. **Divergence sources.** The certifier reports *which units, in which value
   ranges*, can make two models differ. The falsifier Newton-solves an input
   into that range. This exposes thin-slab faults (e.g. a ReLU that zeroes
   values in (0, 1e-4)) that a 4,000-input random oracle and Guardian-APC+ both
   miss: 54 such faults in the benchmark.
4. **Compression risk as a bracket.** For each compression: the worst
   disagreement *found* near real data and a *proven* upper bound on how
   confident the original can be at any disagreement. Proposition: if
   |p' − p| ≤ ε then a disagreement needs the original's top-two margin ≤ 2ε.
5. **Certificate-carrying code and a binary audit.** The generated C++ embeds
   its certified domain and model fingerprint; the binary is compiled under
   several flag sets and replayed against the certified model.

6. **Acting on the verdict** (`src/lifecycle.{h,cpp}`).
   *Certified mixed-precision repair*: per-layer int4/int8/fp16/fp32, accepted only
   when the whole model's severity bound stays proven low (cf. HAWQ, but with a proof).
   *Certified culprit localisation*: ddmin over changed weights with a Temper decision
   as the oracle; the revert is re-decided. *Incremental re-certification*: certificates
   store fingerprints, domain, partition and witness; the next version replays the
   witness, then starts branch and bound from the partition (cf. IVAN, PLDI 2023).
7. **Real-world bug classes**: faults modelled on the root causes in Shen et al.
   (FSE 2021); Temper refutes 71/71 observable ones (test set 62) and proves 9 harmless.

## Where the code is

| Piece | File |
|---|---|
| Certifier (bounds, canonical form, differential analysis, branch and bound, severity) | `review_2_core_implementation/src/certify.{h,cpp}` |
| Decisions, specialisation, compression audit, binary audit | `src/temper.{h,cpp}` |
| Deployment dossier (`temper` command) | `src/dossier.{h,cpp}`, `tools/temper.cpp`, `tools/dossier_view.h` |
| Evaluation studies | `src/temper_study.{h,cpp}`, `tools/bench.cpp` |
| Falsifier (Guardian-APC+) | `src/analysis.cpp`, `src/adaptive.cpp`, `src/guardian.cpp` |
| Studio screens t, c, k, d | `tools/studio.cpp` |
| Repair, culprit localisation, incremental certificates, bug-class study | `src/lifecycle.{h,cpp}` |

## Results

See the main [README](../../README.md) and the paper
[`../paper/temper.tex`](../paper/temper.tex). Snapshots of every number are in
[`../research/results/`](../research/results/).

## Honest limits

- Dense ReLU networks (up to 64 inputs, 32 hidden units per layer). Convolutions,
  attention and ONNX models are future work.
- Bounds loosen with depth: int8 on the deeper Digits / Breast Cancer models
  stays UNDECIDED (found risk is low, proof is not yet tight enough).
- Divergence sources need an abstract semantics for each candidate operator.
- The canonicaliser, bound rules and interpreter are trusted; the binary audit
  is testing, not proof.
- Domain specialisation is implemented and certified, but these compact trained
  models have almost no provably dead neurons (1 of 48 in Iris (32,16)).

## Five-minute demo

```bash
./build/Release/temper ModelForge/review_2_core_implementation/models/real/wine_mlp16.mforge \
    --data ModelForge/review_2_core_implementation/models/real/wine_test.csv
./build/Release/temper ModelForge/review_2_core_implementation/models/iris_demo.mforge \
    --compare ModelForge/review_2_core_implementation/models/iris_handopt_bug.mforge
./build/Release/temper ModelForge/review_2_core_implementation/models/real/wine_mlp32x16.mforge \
    --data ModelForge/review_2_core_implementation/models/real/wine_test.csv --repair
./build/Release/modelforge_studio      # then t, c, k, d
```

1. **Dossier**: passes certified bit-exact with 0 evaluations; int8 certified
   low-risk, int4 refuted although no test prediction changes; binary bit-exact
   at -O2, fast-math flagged.
2. **Compare**: a single mistyped weight is refuted at the second input,
   localised to `dense_2`, and delta debugging names it (`w2[8]: 0.5 -> 0.505`) and
   certifies the revert; the correct hand-optimised copy is certified.
3. **Repair**: Wine (32,16) gets fp16/int8/int8, 35.7% of the weights, CERTIFIED
   low risk, where the best certified uniform format is fp16 (50%).
4. **Studio c**: the certified-region map shows, on a 2-D slice of input space,
   exactly where a subtle bug can act (refuted cells) and where it provably
   cannot (certified cells).

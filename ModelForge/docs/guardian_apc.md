# Guardian-APC: what is new and how to show it

Author: **Vaishnavi**

Guardian-APC extends the original Guardian per-pass validator with three ideas,
and Guardian-APC+ adds three more (sections 4 to 6).
Together they form the project's research contribution. The conference draft is
in [`../paper/guardian_apc.tex`](../paper/guardian_apc.tex).

## 1. Interval bound propagation (IBP) finds the neurons worth testing

`analyzeActivationSites()` (`src/analysis.cpp`) pushes the input box
D = [-10, 10]^n through the IR and computes a lower and upper bound for every
ReLU pre-activation. A neuron is

- **stable inactive** if its upper bound is at most 0,
- **stable active** if its lower bound is above 0,
- **unstable** otherwise: some input reaches its kink at 0.

Only activations that can reach the model output are analysed (dead nodes are
skipped). The benchmark checks soundness: no fault IBP calls "unobservable" may
ever be detected (`ibp_violations` must be 0). A unit test also checks 200
random inputs against the bounds.

## 2. Deep boundary probes reach every layer

The first Guardian only solved boundary equations for the *first* dense layer.
`generateDeepBoundaryProbes()` handles any depth. A ReLU network is affine inside
each activation region, so the exact input gradient of a neuron's pre-activation
(forward-mode Jacobian) gives a Newton step that lands on that neuron's kink in a
few iterations. From the solved point it then steps to pre-activations
-0.05, 5e-4, 5e-3 and 5e-2, so faults confined to a narrow band near zero are
exposed. Neurons are taken round-robin across layers, up to 64.

## 3. Activation-pattern coverage (APC) schedules the probes

Every neuron's pre-activation falls into one of six bins (deep inactive,
near inactive, three shrinking bands just above zero, deep active). A
(neuron, bin) state counts as *feasible* only if its IBP interval meets the bin.
APC = covered feasible states / feasible states.

`prioritizeByCoverage()` orders the probe pool greedily by new states covered.
Coverage is monotone submodular, so every greedy prefix of length B covers at
least (1 - 1/e) of what the best B probes from the pool could cover. Order
matters because Guardian stops at the first mismatch and budgets are finite.

## 4. Guardian-APC+: rewrite-aware targeting

`analyzeRewriteImpact()` (`src/adaptive.cpp`) diffs the IR before and after a
pass (operator, operands, attributes, constant data). The changed instructions
and everything downstream of them are "impacted". Guardian-APC+ generates
boundary probes for the impacted ReLU layers and covers those first, so the
inputs most likely to expose this particular rewrite run first.

## 5. Near-miss search

Most inputs give a faulty candidate an output difference below the tolerance.
Guardian-APC+ scores each input by that difference (plus a small weight on
internal differences) and, after the targeted suite, hill-climbs the score from
the four most divergent inputs. It mainly catches small numeric errors.

## 6. Bug localisation and the model diff checker

When a rewrite is rejected, Guardian traces both graphs on the minimised witness
and reports the first value that differs, for example
`first diverges at activation_1 (value 'hidden2')`. It is also in
`counterexample.json`.

The same engine compares two model files:

```text
modelforge models/iris_demo.mforge --compare models/iris_handopt_bug.mforge
```

It reports which nodes differ, whether the behaviour differs, the input that
proves it, and where the difference starts. Exit code 0 means no difference
found, 3 means the models differ. Studio option 9 shows the same thing.

## 7. Shrink and check: is a compressed model safe to ship?

People shrink models (float16, int8, int4, pruning) to run them on small
devices, and usually only check test accuracy. `searchBoundaryShift()`
(`src/shrink.cpp`) finds realistic inputs (within 0.5 std of real data) where
the original model was sure but the compressed one disagrees:

1. pick two real test rows the original classifies differently,
2. bisect the original's decision boundary on the line between them,
3. check with two evaluations whether the compressed model's boundary moved,
4. where it moved, bisect the compressed boundary: the edge of the gap is the
   worst disagreement on that line,
5. refine the worst ones with a small evolutionary search.

```text
modelforge models/real/wine_mlp16.mforge --shrink all --data models/real/wine_test.csv --out build/shrink
```

Result on 8 models trained on real data (Iris, Wine, Breast Cancer, Digits) and
5 compressions: of 24 risky compressions, the test set alone flags 14, a
genetic search 18.2, boundary-shift probing 22. Of the 7 risky compressions
whose accuracy did not drop at all, it flags 6 (test set: 1). Example: Wine
with int4 weights keeps 98.1% accuracy and 0 test-set flips, but has a
realistic input where the original is clearly sure and the compressed model
disagrees. The command recommends int8 instead (for 7 of 8 models; for one
Breast Cancer model its 200-try search underestimated int4 and recommended it,
a known miss). Studio option `s` shows it.

The models are trained by `../research/train_real_models.py` (scikit-learn) and
committed, so running the tools needs no Python.

## Measured results (96 models, 2,160 faults, budget 128)

| Strategy | Detection | 95% CI | Caught within 8 inputs |
|---|---|---|---|
| Uniform random | 75.0% | 74.2 to 75.8 | 57.9% |
| Guardian v1 | 85.6% | 84.0 to 87.1 | 62.0% |
| Guardian-APC | 99.0% | 98.5 to 99.4 | 79.9% |
| **Guardian-APC+** | **99.2%** | 98.7 to 99.5 | **88.2%** |

- At 128 inputs, APC+ and APC are statistically tied (p = 0.34). The gain is
  speed: with 4, 8 or 16 inputs APC+ is significantly better (p < 1e-11), and
  the B = 8 result replicates with seeds 1, 2 and 3.
- Near-miss search lifts the 1e-3 weight perturbation from 94.5% to 97.8%
  (random testing 97.1%).
- Held-out fault families: 98.2% (random 85.2%, v1 90.7%).
- No false alarms on 96 negative controls.

Snapshot: [`../research/results/results.json`](../research/results/results.json).

## Limits

- Results are on small synthetic dense networks (width ≤ 16), not real ONNX models.
- IBP on [-10, 10]^n is loose: 2,346 of 2,357 neurons are "unstable", so IBP
  mostly guides probe targeting rather than pruning.
- Passing Guardian is strong empirical evidence, not a proof of equivalence.
- The IR interpreter and the C++ generator share operator definitions.

## Suggested demo order (about 5 minutes)

1. `modelforge_studio` in the VS Code terminal: screen 1 (pipeline), 2 (IR
   graph), 3 (neuron stability map), 5 (fault arena and counterexample).
2. Option `s` checks compressed versions of the real Wine model and
   recommends int8 (int4 looks fine on accuracy but is risky).
3. Option 9 compares the Iris model with a hand-optimised copy that has one
   mistyped weight: the difference is found at the second input and traced to `dense_2`.
4. Option 6 runs the quick benchmark live and refreshes the dashboard.
5. Open `dashboard/index.html`: Overview (headline and budget curve), Fault
   families (where the methods differ), Statistics (McNemar), Model inspector.

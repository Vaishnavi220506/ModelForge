# Guardian-APC: what is new and how to show it

Author: **Vaishnavi**

Guardian-APC extends the original Guardian per-pass validator with three ideas.
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

## Measured results (96 models, 2,160 faults, budget 128)

| Strategy | Detection | 95% CI | Mean probes to 1st detection |
|---|---|---|---|
| Uniform random | 75.1% | 74.2–75.9 | 11.0 |
| Generic probes | 83.9% | 82.2–85.4 | 9.4 |
| Guardian v1 | 85.7% | 84.1–87.1 | 11.2 |
| **Guardian-APC** | **99.1%** | 98.6–99.4 | 5.6 |

- Held-out fault families: 97.9% (random 85.4%, v1 90.9%).
- Faults beyond the first ReLU layer: 99.0% (v1 82.1%).
- Exact McNemar: 2,451 vs 26 discordant cases against random, 273 vs 2 against
  v1 (p < 1e-4). No false alarms on 96 negative controls.
- With 16 probes Guardian-APC already detects 91.6% (random 62.4%).
- Master seeds 1, 2, 3: 99.4%, 99.5%, 99.2%.
- Weakness: on a 1e-3 weight perturbation, random (97.1%) beats Guardian-APC (94.5%).

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
2. Option 6 runs the quick benchmark live and refreshes the dashboard.
3. Open `dashboard/index.html`: Overview (headline and budget curve), Fault
   families (where the methods differ), Statistics (McNemar), Model inspector.
